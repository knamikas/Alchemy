"""Resuming a run over output a previous run left behind.

Three rules carry the weight. An entry counts as finished only when its
manifest row says so, and that row is written after its data rows, so an
interruption between the two costs a repeated entry, never a lost one. A blank
``n_bonds`` means the bond stage never ran while ``0`` means it ran and found
nothing, so reading blank as zero would mark an unanalysed entry permanently
done. And a retry batch is staged to a temporary directory so a re-run replaces
an entry's rows instead of appending beside them; the merge keeps every entry
that reached its manifest row, whether or not the batch as a whole finished.
"""

import csv
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set

from codes import EntryStatus, ReasonCode
from coordination.schema import BOND_COLUMNS, CANDIDATE_COLUMNS
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS
from driver.output_lock import create_owned_scratch_directory
from worker_contracts import EntryResult


# A row read back is not ``dict[str, str]``: DictReader fills a short row's
# missing fields with ``None`` and collects a long row's surplus cells in a
# list under the ``None`` key, which is exactly what the completeness checks
# below look for.
_CsvRow = dict[str | None, str | list[str] | None]


def _complete_csv_row(row: _CsvRow) -> bool:
    """Whether DictReader received every declared field exactly once."""
    return None not in row and all(isinstance(value, str) for value in row.values())


def _csv_text(row: _CsvRow, column: str) -> str:
    """Return one textual CSV cell, safely blanking absent older columns."""
    value = row.get(column, "")
    return value if isinstance(value, str) else ""


def load_done(
    manifest_path: str,
    bonds_required: bool = False,
    bond_output_present: bool = True,
    candidate_output_present: bool = True,
    retry_partial_ids: Iterable[str] = (),
) -> set[str]:
    """PDB IDs whose requested result is terminal in an existing manifest.

    Blank ``n_bonds`` or ``n_candidates`` mean the bond stage never ran, so
    under ``bonds_required`` such a row is not done unless the manifest says
    metal presence was indeterminate and therefore supplied no analyzable site.
    A missing bond or candidate CSV otherwise has the same effect.
    ``retry_partial_ids`` releases only non-retryable ``partial`` rows, never
    ``ok`` ones.

    An ``error`` row is never done, whatever its reason code claims about
    recurring: a resumed run may have been given a repaired input file, and
    skipping the entry would silently drop work the operator had just fixed.
    """
    retry_partial_ids = {
        str(pdb_id).strip().lower()
        for pdb_id in retry_partial_ids
        if str(pdb_id).strip()
    }
    done: set[str] = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and {"pdbID", "status"}.issubset(reader.fieldnames):
                for row in reader:
                    # An interrupted writer can leave the final row shorter
                    # than the header. DictReader fills those trailing cells
                    # with None; skip the row so resume retries that entry.
                    if not _complete_csv_row(row):
                        continue
                    status = _csv_text(row, "status").strip().lower()
                    retryable = _csv_text(row, "retryable").strip().lower()
                    terminal_partial = status == "partial" and retryable in (
                        "false",
                        "0",
                        "no",
                    )
                    pdb_id = _csv_text(row, "pdbID").strip().lower()
                    reason_codes = {
                        code
                        for code in _csv_text(row, "reason_codes").split("|")
                        if code
                    }
                    bonds_inapplicable = (
                        ReasonCode.METAL_PRESENCE_INDETERMINATE in reason_codes
                    )
                    bonds_complete = not bonds_required or (
                        bond_output_present
                        and candidate_output_present
                        and (
                            bonds_inapplicable
                            or (
                                _csv_text(row, "n_bonds").strip() != ""
                                and _csv_text(row, "n_candidates").strip() != ""
                            )
                        )
                    )
                    protected_terminal = status == "ok" or (
                        terminal_partial and pdb_id not in retry_partial_ids
                    )
                    if protected_terminal and bonds_complete and pdb_id:
                        done.add(pdb_id)
    return done


def resume_replacement_succeeded(result: EntryResult) -> bool:
    """Whether a retry produced a terminal result suitable for replacement."""
    return result.status == EntryStatus.OK or (
        result.status == EntryStatus.PARTIAL and not result.retryable
    )


def manifest_values_by_id(path: str, column: str) -> dict[str, str]:
    """Return one manifest column keyed by normalized PDB ID."""
    values: dict[str, str] = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return values
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if not _complete_csv_row(row):
                continue
            pdb_id = _csv_text(row, "pdbID").strip().lower()
            if pdb_id:
                values[pdb_id] = _csv_text(row, column)
    return values


def _csv_header(path: str) -> list[str] | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="") as handle:
        return next(csv.reader(handle), None)


def _is_terminal_manifest_row(row: _CsvRow) -> bool:
    status = _csv_text(row, "status").strip().lower()
    retryable = _csv_text(row, "retryable").strip().lower()
    return status == "ok" or (status == "partial" and retryable in ("false", "0", "no"))


def _terminal_manifest_rows(path: str) -> dict[str, _CsvRow]:
    """Return protected rows, rejecting ambiguous duplicate manifest IDs."""
    terminal_rows: dict[str, _CsvRow] = {}
    complete_ids: set[str] = set()
    if _csv_header(path) is None:
        return terminal_rows
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            # A crash may leave the last manifest row incomplete. It was never
            # committed and remains eligible for replacement on resume.
            if not _complete_csv_row(row):
                continue
            pdb_id = _csv_text(row, "pdbID").strip().lower()
            if not pdb_id:
                continue
            if pdb_id in complete_ids:
                raise ValueError(
                    f"Existing {os.path.basename(path)} contains duplicate rows "
                    f"for {pdb_id}. Resume cannot determine which result owns "
                    "that entry; choose a new --output-dir."
                )
            complete_ids.add(pdb_id)
            if _is_terminal_manifest_row(row):
                terminal_rows[pdb_id] = row
    return terminal_rows


def _manifest_count(
    row: _CsvRow, column: str, pdb_id: str, *, blank_is_zero: bool = False
) -> int:
    value = _csv_text(row, column).strip()
    if blank_is_zero and not value:
        return 0
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Existing manifest has invalid {column}={value!r} for {pdb_id}; "
            "resume requires a non-negative integer."
        ) from exc
    if count < 0:
        raise ValueError(
            f"Existing manifest has invalid {column}={value!r} for {pdb_id}; "
            "resume requires a non-negative integer."
        )
    return count


def _rows_for_ids(path: str, terminal_ids: Set[str]) -> Iterator[tuple[str, _CsvRow]]:
    """Yield complete output rows owned by protected manifest entries.

    Rows for other IDs may be remnants of a write interrupted before its
    manifest row. The replacement merge intentionally removes them if that ID
    is retried, so they must not make an otherwise safe resume fail.
    """
    with open(path, newline="") as handle:
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


def _validate_terminal_artifacts(
    terminal_rows: Mapping[str, _CsvRow],
    stats_path: str,
    bonds_path: str,
    candidates_path: str,
    *,
    bonds_enabled: bool,
    confidence_path: str | None,
) -> None:
    terminal_ids = set(terminal_rows)

    def metal_site_limit_exceeded(row: _CsvRow) -> bool:
        return ReasonCode.METAL_SITE_LIMIT_EXCEEDED in {
            code for code in _csv_text(row, "reason_codes").split("|") if code
        }

    selected_stats: Counter[str] = Counter()
    selected_sites: set[tuple[str, tuple[str, ...]]] = set()
    site_columns = (
        "metal_model_index",
        "metal_chain_index",
        "metal_residue_index",
        "metal_atom_index",
    )
    for pdb_id, row in _rows_for_ids(stats_path, terminal_ids):
        if _csv_text(row, "selected_metal_site_status").strip() != "selected":
            continue
        site = tuple(_csv_text(row, column).strip() for column in site_columns)
        if not all(site) or (pdb_id, site) in selected_sites:
            detail = "an incomplete" if not all(site) else "a duplicate"
            raise ValueError(
                f"Existing {os.path.basename(stats_path)} contains {detail} "
                f"selected-site key for terminal entry {pdb_id}; choose a new "
                "--output-dir."
            )
        selected_sites.add((pdb_id, site))
        selected_stats[pdb_id] += 1

    for pdb_id, row in terminal_rows.items():
        expected = _manifest_count(row, "n_metals", pdb_id)
        actual = selected_stats[pdb_id]
        status = _csv_text(row, "status").strip().lower()
        excluded = metal_site_limit_exceeded(row)
        if excluded:
            stats_match = actual == 0
            relation = "exactly 0 policy-excluded rows"
        elif status == "ok":
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

    checks: list[tuple[str, str]] = []
    if bonds_enabled:
        checks.extend(
            (
                (bonds_path, "n_bonds"),
                (candidates_path, "n_candidates"),
            )
        )
    if confidence_path is not None:
        checks.append((confidence_path, "n_metals"))

    for path, manifest_column in checks:
        counts = Counter(pdb_id for pdb_id, _row in _rows_for_ids(path, terminal_ids))
        for pdb_id, row in terminal_rows.items():
            expected = _manifest_count(
                row,
                manifest_column,
                pdb_id,
                blank_is_zero=manifest_column != "n_metals",
            )
            if manifest_column == "n_metals" and metal_site_limit_exceeded(row):
                expected = 0
            actual = counts[pdb_id]
            if actual != expected:
                raise ValueError(
                    f"Resume artifact mismatch for {pdb_id}: manifest "
                    f"{manifest_column}={expected}, but {os.path.basename(path)} "
                    f"has {actual} row(s)."
                )


def validate_resume_schemas(
    manifest_path: str,
    stats_path: str,
    bonds_path: str,
    candidates_path: str,
    bonds_enabled: bool = True,
    confidence_path: str | None = None,
    confidence_columns: Sequence[str] | None = None,
) -> None:
    """Refuse to resume into incompatible or internally inconsistent output.

    Whole headers are compared, including the EDSTATS block of
    metal_stats_all.csv: a header from a different EDSTATS build would misalign
    every density column with no other symptom. Once the manifest contains a
    terminal result, every enabled output is required and its per-entry rows
    must agree with the manifest. Orphan rows without a complete manifest row
    remain recoverable and are ignored here because retry replacement removes
    them.
    """
    checks = [(manifest_path, MANIFEST_COLUMNS), (stats_path, STATS_COLUMNS)]
    if bonds_enabled:
        checks.extend(
            ((bonds_path, BOND_COLUMNS), (candidates_path, CANDIDATE_COLUMNS))
        )
    if confidence_path is not None:
        if confidence_columns is None:
            raise ValueError("confidence columns are required with a confidence output")
        checks.append((confidence_path, list(confidence_columns)))
    for path, expected in checks:
        expected = list(expected)
        header = _csv_header(path)
        if header is None or header == expected:
            continue
        missing = [column for column in expected if column not in header]
        unexpected = [column for column in header if column not in expected]
        difference = (
            "; ".join(
                part
                for part in (
                    "missing " + ", ".join(missing) if missing else "",
                    "unexpected " + ", ".join(unexpected) if unexpected else "",
                )
                if part
            )
            or "the same columns in a different order"
        )
        raise ValueError(
            f"Existing {os.path.basename(path)} uses an incompatible schema "
            f"({difference}). Its rows were written by a different Alchemy "
            "build, and resume can only append beneath a matching header; "
            "choose a new --output-dir."
        )

    terminal_rows = _terminal_manifest_rows(manifest_path)
    if not terminal_rows:
        return

    for path, _expected in checks:
        if _csv_header(path) is None:
            raise ValueError(
                f"Existing {os.path.basename(path)} is missing or empty, but "
                "the manifest contains terminal rows. Resume cannot verify the "
                "completed entries; choose a new --output-dir."
            )

    _validate_terminal_artifacts(
        terminal_rows,
        stats_path,
        bonds_path,
        candidates_path,
        bonds_enabled=bonds_enabled,
        confidence_path=confidence_path,
    )


def remove_stale_disabled_bond_outputs(
    paths: Iterable[str], resume: bool, bonds_enabled: bool
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


def _merge_csv_replacements(
    path: str, staged_path: str, pdb_ids: Iterable[str]
) -> None:
    """Atomically replace selected IDs with rows from a completed staging file.

    Rows for IDs absent from ``pdb_ids`` are copied verbatim.
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
            replacement = os.fdopen(fd, "w", newline="")
        except BaseException:
            os.close(fd)
            raise
        destination_header: list[str] | None = None
        with replacement as dst:
            writer = csv.writer(dst)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, newline="") as src:
                    reader = csv.reader(src)
                    destination_header = next(reader, None)
                    if destination_header is not None:
                        writer.writerow(destination_header)
                    for row in reader:
                        if row and row[0].strip().lower() in replacement_ids:
                            continue
                        writer.writerow(row)

            if os.path.exists(staged_path) and os.path.getsize(staged_path) > 0:
                with open(staged_path, newline="") as staged:
                    reader = csv.reader(staged)
                    staged_header = next(reader, None)
                    if destination_header is None and staged_header is not None:
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
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class ResumeStaging:
    """Hold a resumed run's rows so completed entries replace their old ones.

    Only ids in ``replacement_ids`` are merged, and an id is added there after
    the entry's manifest row is written, so a commit promotes finished entries
    and silently leaves behind any entry that was still being written.
    """

    def __init__(self, output_dir: str, targets: Sequence[str]) -> None:
        self.targets = targets
        self.dir = create_owned_scratch_directory(
            output_dir,
            prefix=".alchemy-resume-",
            kind="resume",
        )
        self.staged = tuple(
            os.path.join(self.dir, os.path.basename(path)) for path in targets
        )
        self.replacement_ids: set[str] = set()

    def commit(self, bonds_enabled: bool, confidence_enabled: bool = False) -> None:
        """Replace the retried entries' rows in the real output files."""
        if not self.replacement_ids:
            return
        manifest_path, stats_path, bonds_path, candidates_path = self.targets[:4]
        (staged_manifest, staged_stats, staged_bonds, staged_candidates) = self.staged[
            :4
        ]
        # Data files are committed before the manifest completion marker, so an
        # interruption between replacements leaves the entry safely retryable.
        _merge_csv_replacements(stats_path, staged_stats, self.replacement_ids)
        if bonds_enabled:
            _merge_csv_replacements(bonds_path, staged_bonds, self.replacement_ids)
            _merge_csv_replacements(
                candidates_path, staged_candidates, self.replacement_ids
            )
        if confidence_enabled:
            for target, staged in zip(self.targets[4:], self.staged[4:]):
                _merge_csv_replacements(target, staged, self.replacement_ids)
        _merge_csv_replacements(manifest_path, staged_manifest, self.replacement_ids)

    def discard(self) -> None:
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
