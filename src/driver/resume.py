"""Check a previous run's outputs and merge retried entries into them on resume.

Three rules govern resume:

- A manifest row is an entry's completion marker. An entry without one, or
  with an incomplete one, is retried, and rows it left in other outputs are
  discarded by the merge.
- In the manifest's stage counts, blank means the stage never ran and zero
  means it ran and found nothing. A blank bond count is why a density-only
  entry is revisited when bonds are enabled.
- Retries write to a staging directory, and only entries whose manifest row
  was written there are merged back, data files first and the manifest last,
  so an interruption at any point leaves the entry retryable.
"""

import contextlib
import csv
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set

from analysis_config import MAX_ANALYZED_METAL_SITES
from codes import EntryStatus, ReasonCode
from coordination.schema import BOND_COLUMNS, CANDIDATE_COLUMNS
from driver.layout import OutputLayout
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS, OutputTargets
from scratch import RESUME_SCRATCH, create_owned_scratch_directory
from worker_contracts import EntryResult

# DictReader uses None for missing cells and stores surplus cells under a None key.
_CsvRow = dict[str | None, str | list[str] | None]

#: Accepted spellings of a manifest boolean, as written now and by older builds.
_TRUE_TEXT = ("true", "1", "yes")
_FALSE_TEXT = ("false", "0", "no")

#: The statistics columns that identify one selected metal site.
_SITE_KEY_COLUMNS = (
    "metal_model_index",
    "metal_chain_index",
    "metal_residue_index",
    "metal_atom_index",
)


def _complete_csv_row(row: _CsvRow) -> bool:
    """Whether DictReader received every declared field exactly once."""
    return None not in row and all(isinstance(value, str) for value in row.values())


def _csv_text(row: _CsvRow, column: str) -> str:
    """Return one textual CSV cell, safely blanking absent older columns."""
    value = row.get(column, "")
    return value if isinstance(value, str) else ""


class _ManifestRow:
    """One complete manifest row, exposing the fields resume decides by.

    There are two tiers of accessor. ``text`` and the properties built on it
    return a cell's text and never raise; a missing column reads as blank. They
    are safe on any row, including one from an older manifest, and they are all
    that is needed to decide whether a row is terminal.

    ``count``, ``flag``, and ``metal_site_limit_exceeded`` parse that text and
    raise ``ValueError`` naming the entry when it is malformed. They are called
    only on terminal rows, because those are the rows resume will keep, so
    their fields must be trustworthy.
    """

    def __init__(self, row: _CsvRow) -> None:
        self._row = row
        self.pdb_id = _csv_text(row, "pdbID").strip().lower()
        self.status = _csv_text(row, "status").strip().lower()

    def text(self, column: str) -> str:
        """The text of one cell, blank when the column is absent."""
        return _csv_text(self._row, column)

    @property
    def reason_codes(self) -> frozenset[str]:
        """The machine-readable codes explaining a status other than ok."""
        return frozenset(code for code in self.text("reason_codes").split("|") if code)

    @property
    def is_terminal(self) -> bool:
        """Whether the row protects its entry: ok, or a partial no retry can improve.

        A blank or malformed ``retryable`` leaves a partial unprotected, so a
        row an older or interrupted build could not vouch for is retried.
        """
        return self.status == EntryStatus.OK or (
            self.status == EntryStatus.PARTIAL
            and self.text("retryable").strip().lower() in _FALSE_TEXT
        )

    @property
    def has_bond_counts(self) -> bool:
        """Whether the bond stage ran: both of its counts are non-blank."""
        return bool(self.text("n_bonds").strip()) and bool(
            self.text("n_candidates").strip()
        )

    @property
    def awaits_bond_stage(self) -> bool:
        """A successful density-only row that a bond-enabled run must revisit."""
        return (
            self.status == EntryStatus.OK
            and not self.text("n_bonds").strip()
            and not self.text("n_candidates").strip()
        )

    def count(self, column: str, *, blank_is_zero: bool = False) -> int:
        """A non-negative integer field, blank meaning zero only where asked."""
        value = self.text(column).strip()
        if blank_is_zero and not value:
            return 0
        try:
            count = int(value)
        except ValueError:
            count = -1
        if count < 0:
            raise ValueError(
                f"Existing manifest has invalid {column}={value!r} for {self.pdb_id}; "
                "resume requires a non-negative integer."
            )
        return count

    def flag(self, column: str) -> bool:
        """A boolean field, in any spelling a manifest has ever used."""
        value = self.text(column).strip().lower()
        if value in _TRUE_TEXT:
            return True
        if value in _FALSE_TEXT:
            return False
        raise ValueError(
            f"Existing manifest has invalid {column}={value!r} for {self.pdb_id}; "
            "resume requires a boolean value."
        )

    @property
    def metal_site_limit_exceeded(self) -> bool:
        """The policy-exclusion flag, which must agree with its reason code."""
        reason_present = ReasonCode.METAL_SITE_LIMIT_EXCEEDED in self.reason_codes
        flag = self.flag("metal_site_limit_exceeded")
        if flag != reason_present:
            raise ValueError(
                f"Existing manifest has inconsistent metal-site exclusion "
                f"fields for {self.pdb_id}; "
                f"metal_site_limit_exceeded={str(flag).lower()} "
                f"and reason_codes={self.text('reason_codes')!r}."
            )
        return flag


def _manifest_rows(path: str) -> Iterator[_ManifestRow]:
    """Yield the complete rows of a manifest, if there is one.

    An interrupted writer can leave the final row shorter than the header.
    DictReader fills those trailing cells with None; that row was never
    committed, so it is skipped and its entry stays eligible for a retry.
    """
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _complete_csv_row(row):
                yield _ManifestRow(row)


def load_done(
    manifest_path: str,
    *,
    bonds_required: bool = False,
    bond_output_present: bool = True,
    candidate_output_present: bool = True,
    retry_partial_ids: Iterable[str] = (),
) -> set[str]:
    """Return IDs whose requested stages are complete or terminal.

    Blank bond or candidate counts require reprocessing when bonds are requested,
    unless no site was analyzable because metal presence was indeterminate.
    Missing bond files also require reprocessing. retry_partial_ids releases
    terminal partial rows, but never ok rows. Errors remain eligible for resume
    because inputs or software may have been repaired.
    """
    retry_partial_ids = {
        normalized
        for pdb_id in retry_partial_ids
        if (normalized := pdb_id.strip().lower())
    }
    bond_outputs_present = bond_output_present and candidate_output_present
    done: set[str] = set()
    for row in _manifest_rows(manifest_path):
        if not row.pdb_id or not row.is_terminal:
            continue
        if row.status != EntryStatus.OK and row.pdb_id in retry_partial_ids:
            continue
        bonds_inapplicable = ReasonCode.METAL_PRESENCE_INDETERMINATE in row.reason_codes
        bonds_complete = not bonds_required or (
            bond_outputs_present and (bonds_inapplicable or row.has_bond_counts)
        )
        if bonds_complete:
            done.add(row.pdb_id)
    return done


def resume_replacement_succeeded(result: EntryResult) -> bool:
    """Whether a retry produced a terminal result suitable for replacement."""
    return result.status == EntryStatus.OK or (
        result.status == EntryStatus.PARTIAL and not result.retryable
    )


def manifest_values_by_id(path: str, column: str) -> dict[str, str]:
    """Return one manifest column keyed by normalized PDB ID."""
    return {row.pdb_id: row.text(column) for row in _manifest_rows(path) if row.pdb_id}


def _csv_header(path: str) -> list[str] | None:
    """The header row of a CSV; ``None`` when the file is absent or empty.

    Callers treat no header as "the stage has not written yet" and a wrong
    header as an incompatible schema, so the two must stay distinguishable.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), None)


def _terminal_manifest_rows(path: str) -> dict[str, _ManifestRow]:
    """Return protected rows, rejecting ambiguous duplicate manifest IDs."""
    terminal_rows: dict[str, _ManifestRow] = {}
    complete_ids: set[str] = set()
    for row in _manifest_rows(path):
        if not row.pdb_id:
            continue
        if row.pdb_id in complete_ids:
            raise ValueError(
                f"Existing {os.path.basename(path)} contains duplicate rows "
                f"for {row.pdb_id}. Resume cannot determine which result owns "
                "that entry; choose a new --output-dir."
            )
        complete_ids.add(row.pdb_id)
        if row.is_terminal:
            terminal_rows[row.pdb_id] = row
    return terminal_rows


def _rows_for_ids(path: str, terminal_ids: Set[str]) -> Iterator[tuple[str, _CsvRow]]:
    """Yield complete output rows owned by protected manifest entries.

    Rows for other IDs may be remnants of a write interrupted before its
    manifest row. The replacement merge intentionally removes them if that ID
    is retried, so they must not make an otherwise safe resume fail.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pdb_id = _csv_text(row, "pdbID").strip().lower()
            if pdb_id not in terminal_ids:
                continue
            if not _complete_csv_row(row):
                raise ValueError(
                    f"Existing {os.path.basename(path)} contains an incomplete "
                    f"row for terminal entry {pdb_id}; choose a new --output-dir."
                )
            yield pdb_id, row


def _count_selected_sites(
    terminal_rows: Mapping[str, _ManifestRow], stats_path: str
) -> Counter[str]:
    """Count each terminal entry's selected sites, refusing ambiguous site keys."""
    terminal_ids = set(terminal_rows)
    selected_stats: Counter[str] = Counter()
    selected_sites: set[tuple[str, tuple[str, ...]]] = set()
    for pdb_id, row in _rows_for_ids(stats_path, terminal_ids):
        if _csv_text(row, "selected_metal_site_status").strip() != "selected":
            continue
        site = tuple(_csv_text(row, column).strip() for column in _SITE_KEY_COLUMNS)
        if not all(site) or (pdb_id, site) in selected_sites:
            detail = "an incomplete" if not all(site) else "a duplicate"
            raise ValueError(
                f"Existing {os.path.basename(stats_path)} contains {detail} "
                f"selected-site key for terminal entry {pdb_id}; choose a new "
                "--output-dir."
            )
        selected_sites.add((pdb_id, site))
        selected_stats[pdb_id] += 1
    return selected_stats


def _validate_manifest_site_fields(
    terminal_rows: Mapping[str, _ManifestRow],
    stats_path: str,
    selected_stats: Mapping[str, int],
) -> None:
    """Check each terminal row's site fields against each other and the stats rows."""
    for pdb_id, row in terminal_rows.items():
        expected = row.count("n_metals")
        actual = selected_stats[pdb_id]
        is_ok = row.status == EntryStatus.OK
        excluded = row.metal_site_limit_exceeded
        no_metals = row.flag("no_metals")
        if excluded and (
            no_metals or expected <= MAX_ANALYZED_METAL_SITES or not is_ok
        ):
            raise ValueError(
                f"Existing manifest has invalid policy-exclusion fields for "
                f"{pdb_id}; excluded entries must be ok, not metal-free, and "
                f"have more than {MAX_ANALYZED_METAL_SITES} detected sites."
            )
        if no_metals and (expected != 0 or not is_ok):
            raise ValueError(
                f"Existing manifest has invalid no_metals fields for {pdb_id}; "
                "metal-free entries must be ok with n_metals=0."
            )
        if excluded:
            stats_match = actual == 0
            relation = "exactly 0 policy-excluded rows"
        elif is_ok:
            stats_match = actual == expected
            relation = f"exactly {expected}"
        else:
            stats_match = actual <= expected
            relation = f"no more than {expected}"
        if not stats_match:
            raise ValueError(
                f"Resume artifact mismatch for {pdb_id}: manifest n_metals="
                f"{expected}, but {os.path.basename(stats_path)} has {actual} "
                f"selected row(s) (expected {relation})."
            )


def _validate_stage_row_counts(
    terminal_rows: Mapping[str, _ManifestRow],
    layout: OutputLayout,
    *,
    bonds_enabled: bool,
    confidence_path: str | None,
) -> None:
    """Check that every enabled stage output has each terminal row's row count."""
    checks: list[tuple[str, str]] = []
    if bonds_enabled:
        checks.extend(
            (
                (layout.bonds, "n_bonds"),
                (layout.candidates, "n_candidates"),
            )
        )
    if confidence_path is not None:
        checks.append((confidence_path, "n_metals"))

    terminal_ids = set(terminal_rows)
    for path, manifest_column in checks:
        counts = (
            Counter(pdb_id for pdb_id, _row in _rows_for_ids(path, terminal_ids))
            if os.path.isfile(path)
            else Counter[str]()
        )
        for pdb_id, row in terminal_rows.items():
            # Confidence is disabled with --no-bonds. These entries must run
            # again before their confidence rows can be required or retained.
            if path == confidence_path and row.awaits_bond_stage:
                continue
            expected = row.count(
                manifest_column, blank_is_zero=manifest_column != "n_metals"
            )
            if manifest_column == "n_metals" and row.metal_site_limit_exceeded:
                expected = 0
            actual = counts[pdb_id]
            if actual != expected:
                raise ValueError(
                    f"Resume artifact mismatch for {pdb_id}: manifest "
                    f"{manifest_column}={expected}, but {os.path.basename(path)} "
                    f"has {actual} row(s)."
                )


def _validate_terminal_artifacts(
    terminal_rows: Mapping[str, _ManifestRow],
    layout: OutputLayout,
    *,
    bonds_enabled: bool,
    confidence_path: str | None,
) -> None:
    """Check that the outputs back every terminal manifest row, in reading order."""
    selected_stats = _count_selected_sites(terminal_rows, layout.stats)
    _validate_manifest_site_fields(terminal_rows, layout.stats, selected_stats)
    _validate_stage_row_counts(
        terminal_rows,
        layout,
        bonds_enabled=bonds_enabled,
        confidence_path=confidence_path,
    )


def validate_resume_schemas(
    layout: OutputLayout,
    *,
    bonds_enabled: bool = True,
    confidence_path: str | None = None,
    confidence_columns: Sequence[str] | None = None,
    additional_outputs: Sequence[tuple[str, Sequence[str]]] = (),
) -> None:
    """Reject incompatible schemas or inconsistent completed results before resume.

    Validate full headers, required stage outputs, and row counts. Density-only
    entries may lack bond and confidence output when those stages will be retried.
    Ignore orphan rows without a complete manifest record; retry removes them.
    """
    checks = [(layout.manifest, MANIFEST_COLUMNS), (layout.stats, STATS_COLUMNS)]
    if bonds_enabled:
        checks.extend(
            ((layout.bonds, BOND_COLUMNS), (layout.candidates, CANDIDATE_COLUMNS))
        )
    if confidence_path is not None:
        if confidence_columns is None:
            raise ValueError("confidence columns are required with a confidence output")
        checks.append((confidence_path, list(confidence_columns)))
    checks.extend((path, list(columns)) for path, columns in additional_outputs)
    for path, expected in checks:
        expected = list(expected)
        header = _csv_header(path)
        if header is None or header == expected:
            continue
        missing = [column for column in expected if column not in header]
        unexpected = [column for column in header if column not in expected]
        parts: list[str] = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if unexpected:
            parts.append("unexpected " + ", ".join(unexpected))
        difference = "; ".join(parts) or "the same columns in a different order"
        raise ValueError(
            f"Existing {os.path.basename(path)} uses an incompatible schema "
            f"({difference}). Its rows were written by a different Alchemy "
            "build, and resume can only append beneath a matching header; "
            "choose a new --output-dir."
        )

    terminal_rows = _terminal_manifest_rows(layout.manifest)
    if not terminal_rows:
        return

    pending_bonds_only = all(row.awaits_bond_stage for row in terminal_rows.values())
    bond_stage_paths: set[str | None] = {
        layout.bonds,
        layout.candidates,
        confidence_path,
    }
    for path, _expected in checks:
        if pending_bonds_only and path in bond_stage_paths:
            continue
        if _csv_header(path) is None:
            raise ValueError(
                f"Existing {os.path.basename(path)} is missing or empty, but "
                "the manifest contains terminal rows. Resume cannot verify the "
                "completed entries; choose a new --output-dir."
            )

    _validate_terminal_artifacts(
        terminal_rows,
        layout,
        bonds_enabled=bonds_enabled,
        confidence_path=confidence_path,
    )


def remove_stale_disabled_bond_outputs(
    paths: Iterable[str], *, resume: bool, bonds_enabled: bool
) -> list[str]:
    """Remove previous bond-stage CSVs before a fresh run with bonds disabled.

    A non-resume run replaces the manifest and statistics outputs, so retaining
    older bond-stage files would associate them with the new run.
    """
    if resume or bonds_enabled:
        return []
    removed: list[str] = []
    for path in paths:
        if os.path.lexists(path):
            os.unlink(path)
            removed.append(path)
    return removed


def _require_id_column(header: Sequence[str], path: str) -> None:
    """Refuse to merge a CSV whose first column is not the entry ID."""
    if not header or header[0] != "pdbID":
        raise ValueError(
            f"{os.path.basename(path)} does not lead with pdbID, so its rows "
            "cannot be matched to entries for replacement"
        )


def _merge_csv_replacements(
    path: str, staged_path: str, pdb_ids: Iterable[str]
) -> None:
    """Atomically replace selected IDs with rows from a completed staging file.

    Rows for IDs absent from ``pdb_ids`` are copied verbatim. Rows are matched
    on their first cell, so every merged output must lead with ``pdbID``.
    """
    replacement_ids = {pdb_id.lower() for pdb_id in pdb_ids}
    if not replacement_ids:
        return

    directory = os.path.dirname(os.path.abspath(path))
    original_mode = os.stat(path).st_mode if os.path.exists(path) else None
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory, text=True
    )
    try:
        # os.fdopen takes ownership of the descriptor, so it is closed here
        # only if the wrapping itself failed: a second close could land on a
        # descriptor the runtime has since reused.
        try:
            replacement = os.fdopen(fd, "w", newline="", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        destination_header: list[str] | None = None
        with replacement as dst:
            writer = csv.writer(dst)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, newline="", encoding="utf-8") as src:
                    reader = csv.reader(src)
                    destination_header = next(reader, None)
                    if destination_header is not None:
                        _require_id_column(destination_header, path)
                        writer.writerow(destination_header)
                    for row in reader:
                        if row and row[0].strip().lower() in replacement_ids:
                            continue
                        writer.writerow(row)

            if os.path.exists(staged_path) and os.path.getsize(staged_path) > 0:
                with open(staged_path, newline="", encoding="utf-8") as staged:
                    reader = csv.reader(staged)
                    staged_header = next(reader, None)
                    if destination_header is None and staged_header is not None:
                        _require_id_column(staged_header, staged_path)
                        destination_header = staged_header
                        writer.writerow(staged_header)
                    elif (
                        staged_header is not None
                        and staged_header != destination_header
                    ):
                        raise ValueError(f"staged CSV schema does not match {path}")
                    for row in reader:
                        if row and row[0].strip().lower() in replacement_ids:
                            writer.writerow(row)
            dst.flush()
            os.fsync(dst.fileno())

        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
    except BaseException:
        # Also on an interrupt: the temp file is not scratch, so nothing else
        # would ever remove it.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class ResumeStaging:
    """Stage retry outputs and merge entries with completed manifest rows."""

    def __init__(self, output_dir: str, targets: OutputTargets) -> None:
        """Create staging paths for all outputs participating in resume."""
        self.targets = targets
        self.dir = create_owned_scratch_directory(
            output_dir,
            prefix=".alchemy-resume-",
            kind=RESUME_SCRATCH,
            # Staging may hold the only completed copy after merge failure; delete it
            # only after success.
            preserve=True,
        )
        self.staged = targets.staged_in(self.dir)
        self.replacement_ids: set[str] = set()

    def commit(self, bonds_enabled: bool, confidence_enabled: bool = False) -> None:
        """Replace the retried entries' rows in the real output files."""
        if not self.replacement_ids:
            return
        names = ["stats"]
        if bonds_enabled:
            names += OutputTargets.BOND_STAGE_OUTPUTS
        names += OutputTargets.ALWAYS_WRITTEN_EXTRAS
        if confidence_enabled:
            names += [
                name
                for name in OutputTargets.CONFIDENCE_OUTPUTS
                if getattr(self.targets, name) is not None
            ]
        # Data files are committed before the manifest completion marker, so an
        # interruption between replacements leaves the entry safely retryable.
        names.append("manifest")
        for name in names:
            _merge_csv_replacements(
                getattr(self.targets, name),
                getattr(self.staged, name),
                self.replacement_ids,
            )

    def discard(self) -> None:
        """Discard every staged output without changing published files."""
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
