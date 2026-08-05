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
from typing import Optional, Sequence

from codes import ReasonCode
from coordination.schema import BOND_COLUMNS, CANDIDATE_COLUMNS
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS
from driver.output_lock import create_owned_scratch_directory


def _complete_csv_row(row):
    """Whether DictReader received every declared field exactly once."""
    return None not in row and all(isinstance(value, str) for value in row.values())


def _csv_text(row, column):
    """Return one textual CSV cell, safely blanking absent older columns."""
    value = row.get(column, "")
    return value if isinstance(value, str) else ""


def load_done(
    manifest_path,
    bonds_required=False,
    bond_output_present=True,
    candidate_output_present=True,
    retry_partial_ids=(),
):
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
    done = set()
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


def _resume_replacement_succeeded(result):
    """Whether a retry produced a terminal result suitable for replacement."""
    status = str(result.status).strip().lower()
    return status == "ok" or (status == "partial" and not bool(result.retryable))


def _manifest_values_by_id(path, column):
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


def _csv_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="") as handle:
        return next(csv.reader(handle), None)


def validate_resume_schemas(
    manifest_path: str,
    stats_path: str,
    bonds_path: str,
    candidates_path: str,
    bonds_enabled: bool = True,
    confidence_path: Optional[str] = None,
    confidence_columns: Optional[Sequence[str]] = None,
) -> None:
    """Refuse to append rows beneath an incompatible old header.

    Whole headers are compared, including the EDSTATS block of
    metal_stats_all.csv: a header from a different EDSTATS build would misalign
    every density column with no other symptom.
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


def remove_stale_disabled_bond_outputs(paths, resume, bonds_enabled):
    """Remove previous bond-stage CSVs before a fresh run with bonds disabled.

    A non-resume run replaces the manifest and statistics outputs, so retaining
    older bond-stage files would associate them with the new run.
    """
    if resume or bonds_enabled:
        return []
    removed = []
    for path in paths:
        if os.path.lexists(path):
            os.unlink(path)
            removed.append(path)
    return removed


def _merge_csv_replacements(path, staged_path, pdb_ids):
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
            staged = os.fdopen(fd, "w", newline="")
        except BaseException:
            os.close(fd)
            raise
        destination_header = None
        with staged as dst:
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


class _ResumeStaging:
    """Hold a resumed run's rows so completed entries replace their old ones.

    Only ids in ``replacement_ids`` are merged, and an id is added there after
    the entry's manifest row is written, so a commit promotes finished entries
    and silently leaves behind any entry that was still being written.
    """

    def __init__(self, output_dir, targets):
        self.targets = targets
        self.dir = create_owned_scratch_directory(
            output_dir,
            prefix=".alchemy-resume-",
            kind="resume",
        )
        self.staged = tuple(
            os.path.join(self.dir, os.path.basename(path)) for path in targets
        )
        self.replacement_ids = set()

    def commit(self, bonds_enabled, confidence_enabled=False):
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

    def discard(self):
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
