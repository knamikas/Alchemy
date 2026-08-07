"""The four streamed CSV outputs and the schemas they are written against.

Every entry's rows are appended and flushed as soon as the worker returns, so
an interrupted batch keeps the results it already has. The column lists are an
external contract twice over: users read the files, and ``--resume`` reads them
back to decide what still needs running.
"""

import csv
import math
from typing import Any, TextIO
from collections.abc import Mapping, Sequence

from coordination.schema import (
    BOND_COLUMNS,
    CANDIDATE_COLUMNS,
    STATS_EXTRA_COLUMNS,
    BondRow,
    CandidateRow,
)
from confidence_score import CONFIDENCE_INPUT_COLUMNS
from metal_identification import EDSTATS_COLUMNS
from worker_contracts import EntryResult, blank_if_unmeasured
from output_rows import MetalStatsRow


# CSV column names keep the deposited-data spelling ``pdbID`` even though every
# Python identifier is ``pdb_id``: downstream scripts and joins address the
# columns by name, so renaming one breaks them silently.
MANIFEST_COLUMNS = [
    "pdbID",
    "status",
    "retryable",
    "no_metals",
    "metal_site_limit_exceeded",
    "n_metals",
    "n_bonds",
    "n_candidates",
    "runtime_s",
    "reason_codes",
    "warning_codes",
    "status_detail",
    "alchemy_version",
    "alchemy_commit",
    "gemmi_version",
    "ccp4_version",
    "reference_data_id",
    "refinement_state",
    "pdb_redo_version",
    "pdb_redo_date",
    "source_coordinate_format",
    "analysis_coordinate_format",
    "coordinate_conversion_performed",
    "source_coordinate_path",
    "model_policy",
    "input_model_count",
    "model_analyzed",
    "multi_model_structure",
    "altloc_policy",
    "symmetry_contact_policy",
]

# The middle block is the EDSTATS residue table, whose column set and order
# `extract_metal_statistics` validates against EDSTATS_COLUMNS before emitting
# any row, so the full header is fixed.
STATS_COLUMNS = (
    ["pdbID", "category"]
    + list(EDSTATS_COLUMNS)
    + ["aa_geometry_coverage"]
    + list(STATS_EXTRA_COLUMNS)
)


# ``status_detail`` avoids classifying expected partial-result explanations as
# errors in the public CSV while preserving the worker's internal exception
# field used by logging and recovery paths.
MANIFEST_FIELDS = {column: column for column in MANIFEST_COLUMNS} | {
    "pdbID": "pdb_id",
    "status_detail": "error",
}


def _manifest_value(value: object) -> object:
    value = blank_if_unmeasured(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _scientific_csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def manifest_row(
    result: EntryResult,
    resume: bool,
    bonds_enabled: bool,
    prior_bond_counts: Mapping[str, str],
    prior_candidate_counts: Mapping[str, str],
) -> dict[str, Any]:
    """Project one worker result onto the manifest schema."""
    row = {
        column: _manifest_value(getattr(result, field))
        for column, field in MANIFEST_FIELDS.items()
    }
    n_bonds = blank_if_unmeasured(result.n_bonds)
    n_candidates = blank_if_unmeasured(result.n_candidates)
    if not bonds_enabled:
        n_bonds = prior_bond_counts.get(result.pdb_id.lower(), "") if resume else ""
        n_candidates = (
            prior_candidate_counts.get(result.pdb_id.lower(), "") if resume else ""
        )
    row.update(
        n_bonds=n_bonds,
        n_candidates=n_candidates,
        runtime_s=f"{result.runtime_s:.3f}",
        reason_codes="|".join(result.reason_codes),
        warning_codes="|".join(result.warning_codes),
    )
    return row


class OutputWriters:
    """The streamed CSV outputs, with running row counts."""

    def __init__(
        self,
        manifest_fh: TextIO,
        stats_fh: TextIO,
        bonds_fh: TextIO | None,
        candidates_fh: TextIO | None,
        confidence_fh: TextIO | None = None,
        confidence_columns: Sequence[str] | None = None,
        confidence_inputs_fh: TextIO | None = None,
    ) -> None:
        self._manifest_fh = manifest_fh
        self._stats_fh = stats_fh
        self._bonds_fh = bonds_fh
        self._candidates_fh = candidates_fh
        self._confidence_fh = confidence_fh
        self._confidence_inputs_fh = confidence_inputs_fh
        self._manifest = csv.DictWriter(manifest_fh, fieldnames=MANIFEST_COLUMNS)
        self._stats = csv.writer(stats_fh)
        self._bonds = csv.writer(bonds_fh) if bonds_fh is not None else None
        self._candidates = (
            csv.writer(candidates_fh) if candidates_fh is not None else None
        )
        if confidence_fh is not None and confidence_columns is None:
            raise ValueError("confidence columns are required with a confidence output")
        self._confidence: csv.DictWriter[str] | None = None
        self._confidence_inputs: csv.DictWriter[str] | None = None
        if confidence_fh is not None and confidence_columns is not None:
            self._confidence = csv.DictWriter(
                confidence_fh, fieldnames=confidence_columns
            )
        if confidence_inputs_fh is not None:
            if confidence_fh is None:
                raise ValueError(
                    "confidence inputs synchronization requires scored output"
                )
            self._confidence_inputs = csv.DictWriter(
                confidence_inputs_fh, fieldnames=CONFIDENCE_INPUT_COLUMNS
            )
        self._confidence_columns = confidence_columns
        self.n_rows = 0
        self.n_bonds = 0
        self.n_candidates = 0
        self.n_confidence = 0
        self._manifest.writeheader()
        self._stats.writerow(STATS_COLUMNS)
        if self._bonds is not None:
            self._bonds.writerow(BOND_COLUMNS)
        if self._candidates is not None:
            self._candidates.writerow(CANDIDATE_COLUMNS)
        if self._confidence is not None:
            self._confidence.writeheader()
        if self._confidence_inputs is not None:
            self._confidence_inputs.writeheader()

    def write_stats_rows(self, rows: Sequence[MetalStatsRow]) -> None:
        if not rows:
            return
        output_rows = [row.as_output_dict(STATS_COLUMNS) for row in rows]
        for row in output_rows:
            self._stats.writerow(
                [_scientific_csv_value(row[column]) for column in STATS_COLUMNS]
            )
            self.n_rows += 1
        self._stats_fh.flush()

    def write_bond_rows(self, bond_rows: Sequence[BondRow]) -> None:
        if self._bonds is None or self._bonds_fh is None or not bond_rows:
            return
        for bond in bond_rows:
            values = bond.as_dict()
            self._bonds.writerow(
                [_scientific_csv_value(values[column]) for column in BOND_COLUMNS]
            )
            self.n_bonds += 1
        self._bonds_fh.flush()

    def write_candidate_rows(self, candidate_rows: Sequence[CandidateRow]) -> None:
        if (
            self._candidates is None
            or self._candidates_fh is None
            or not candidate_rows
        ):
            return
        for candidate in candidate_rows:
            values = candidate.as_dict()
            self._candidates.writerow(
                [_scientific_csv_value(values[column]) for column in CANDIDATE_COLUMNS]
            )
            self.n_candidates += 1
        self._candidates_fh.flush()

    def write_manifest_row(self, row: Mapping[str, Any]) -> None:
        self._manifest.writerow(row)
        self._manifest_fh.flush()

    def write_confidence_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self._confidence is None or not rows:
            return
        if self._confidence_columns is None or self._confidence_fh is None:
            raise RuntimeError("confidence output is not fully configured")
        expected = set(self._confidence_columns)
        for row in rows:
            if set(row) != expected:
                raise RuntimeError("confidence row does not match its output schema")
        input_rows = (
            [
                {column: row[column] for column in CONFIDENCE_INPUT_COLUMNS}
                for row in rows
            ]
            if self._confidence_inputs is not None
            else None
        )
        self._confidence.writerows(rows)
        if self._confidence_inputs is not None and input_rows is not None:
            self._confidence_inputs.writerows(input_rows)
        self.n_confidence += len(rows)
        self._confidence_fh.flush()
        if self._confidence_inputs_fh is not None:
            self._confidence_inputs_fh.flush()
