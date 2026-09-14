"""Write and flush per-entry CSV results using shared output schemas.

Flush each result so interrupted runs retain completed work. Resume reads
these same schemas to determine what remains.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from operator import attrgetter
from typing import Any, ClassVar, TextIO

from analysis_config import ALTLOC_POLICY, MODEL_POLICY, SYMMETRY_POLICY
from confidence_score import CONFIDENCE_INPUT_COLUMNS
from coordination.schema import (
    BOND_COLUMNS,
    CANDIDATE_COLUMNS,
    STATS_EXTRA_COLUMNS,
    BondRow,
    CandidateRow,
)
from crystallization_conditions import (
    CONDITION_COLUMNS,
    SUMMARY_COLUMNS,
    unavailable_summary,
)
from metal_identification import DENSITY_CONTEXT_COLUMNS, EDSTATS_COLUMNS
from output_rows import MetalStatsRow, blank_if_unmeasured, scientific_csv_value
from worker_contracts import EntryResult

# Keep the published pdbID spelling for downstream readers and joins.
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
    "analysis_config_id",
    "refinement_state",
    "pdb_redo_is_twin",
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

# EDSTATS column order is validated before rows reach this writer.
STATS_COLUMNS = (
    ["pdbID", "category"]
    + list(EDSTATS_COLUMNS)
    + ["aa_geometry_coverage"]
    + list(STATS_EXTRA_COLUMNS)
)


#: Manifest columns that describe the run's analysis policy, not the entry.
RUN_POLICY_COLUMNS: Mapping[str, str] = {
    "model_policy": MODEL_POLICY,
    "altloc_policy": ALTLOC_POLICY,
    "symmetry_contact_policy": SYMMETRY_POLICY,
}

#: Per-entry manifest column -> attribute path on ``EntryResult``.
MANIFEST_FIELDS: Mapping[str, str] = {
    "pdbID": "pdb_id",
    "status": "status",
    "retryable": "retryable",
    "no_metals": "no_metals",
    "metal_site_limit_exceeded": "metal_site_limit_exceeded",
    "n_metals": "n_metals",
    "n_bonds": "n_bonds",
    "n_candidates": "n_candidates",
    "runtime_s": "runtime_s",
    "reason_codes": "reason_codes",
    "warning_codes": "warning_codes",
    "status_detail": "status_detail",
    "alchemy_version": "software.alchemy_version",
    "alchemy_commit": "software.alchemy_commit",
    "gemmi_version": "software.gemmi_version",
    "ccp4_version": "software.ccp4_version",
    "reference_data_id": "software.reference_data_id",
    "analysis_config_id": "software.analysis_config_id",
    "refinement_state": "pdb_redo.refinement_state",
    "pdb_redo_is_twin": "pdb_redo.pdb_redo_is_twin",
    "pdb_redo_version": "pdb_redo.pdb_redo_version",
    "pdb_redo_date": "pdb_redo.pdb_redo_date",
    "source_coordinate_format": "coordinates.source_coordinate_format",
    "analysis_coordinate_format": "coordinates.analysis_coordinate_format",
    "coordinate_conversion_performed": "coordinates.coordinate_conversion_performed",
    "source_coordinate_path": "coordinates.source_coordinate_path",
    "input_model_count": "coordinates.input_model_count",
    "model_analyzed": "coordinates.model_analyzed",
    "multi_model_structure": "coordinates.multi_model_structure",
}


def _manifest_value(value: object) -> object:
    value = blank_if_unmeasured(value)
    if isinstance(value, bool):
        return "true" if value else "false"
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
        column: _manifest_value(attrgetter(path)(result))
        for column, path in MANIFEST_FIELDS.items()
    } | dict(RUN_POLICY_COLUMNS)
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


@dataclass(frozen=True)
class OutputTargets:
    """The files one run writes, each by name.

    Resume staging mirrors these under its scratch directory and merges each
    one back by name, so nothing depends on the field order.
    """

    manifest: str
    stats: str
    bonds: str
    candidates: str
    crystallization_conditions: str
    crystallization_summary: str
    density_context: str
    confidence: str | None = None
    confidence_inputs: str | None = None

    #: Written whether or not bonds are enabled; every resume commit merges them.
    ALWAYS_WRITTEN_EXTRAS: ClassVar[tuple[str, ...]] = (
        "crystallization_conditions",
        "crystallization_summary",
        "density_context",
    )
    #: Present only when confidence analysis is on; merged only when enabled.
    CONFIDENCE_OUTPUTS: ClassVar[tuple[str, ...]] = ("confidence", "confidence_inputs")

    def present(self) -> dict[str, str]:
        """Every path this run writes, keyed by field name."""
        return {
            field.name: path
            for field in fields(self)
            if (path := getattr(self, field.name)) is not None
        }

    def staged_in(self, directory: str) -> OutputTargets:
        """The same targets relocated by basename under ``directory``."""
        return replace(
            self,
            **{
                name: os.path.join(directory, os.path.basename(path))
                for name, path in self.present().items()
            },
        )


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
        crystallization_conditions_fh: TextIO | None = None,
        crystallization_summary_fh: TextIO | None = None,
        density_context_fh: TextIO | None = None,
    ) -> None:
        """Initialize CSV writers and emit all enabled output headers."""
        self._manifest_fh = manifest_fh
        self._stats_fh = stats_fh
        self._bonds_fh = bonds_fh
        self._candidates_fh = candidates_fh
        self._confidence_fh = confidence_fh
        self._confidence_inputs_fh = confidence_inputs_fh
        self._crystallization_conditions_fh = crystallization_conditions_fh
        self._crystallization_summary_fh = crystallization_summary_fh
        self._density_context_fh = density_context_fh
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
        self._crystallization_conditions = (
            csv.DictWriter(crystallization_conditions_fh, fieldnames=CONDITION_COLUMNS)
            if crystallization_conditions_fh is not None
            else None
        )
        self._crystallization_summary = (
            csv.DictWriter(crystallization_summary_fh, fieldnames=SUMMARY_COLUMNS)
            if crystallization_summary_fh is not None
            else None
        )
        self._density_context = (
            csv.DictWriter(density_context_fh, fieldnames=DENSITY_CONTEXT_COLUMNS)
            if density_context_fh is not None
            else None
        )
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
        self.n_crystallization_conditions = 0
        self.n_crystallization_summaries = 0
        self.n_density_contexts = 0
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
        if self._crystallization_conditions is not None:
            self._crystallization_conditions.writeheader()
        if self._crystallization_summary is not None:
            self._crystallization_summary.writeheader()
        if self._density_context is not None:
            self._density_context.writeheader()

    def write_stats_rows(self, rows: Sequence[MetalStatsRow]) -> None:
        """Write and flush metal-site statistics rows."""
        if not rows:
            return
        output_rows = [row.as_output_dict(STATS_COLUMNS) for row in rows]
        for row in output_rows:
            self._stats.writerow(
                [scientific_csv_value(row[column]) for column in STATS_COLUMNS]
            )
            self.n_rows += 1
        self._stats_fh.flush()

    def write_bond_rows(self, bond_rows: Sequence[BondRow]) -> None:
        """Write and flush analyzed bond rows when that output is enabled."""
        if self._bonds is None or self._bonds_fh is None or not bond_rows:
            return
        for bond in bond_rows:
            values = bond.as_dict()
            self._bonds.writerow(
                [scientific_csv_value(values[column]) for column in BOND_COLUMNS]
            )
            self.n_bonds += 1
        self._bonds_fh.flush()

    def write_candidate_rows(self, candidate_rows: Sequence[CandidateRow]) -> None:
        """Write and flush contact-candidate rows when enabled."""
        if (
            self._candidates is None
            or self._candidates_fh is None
            or not candidate_rows
        ):
            return
        for candidate in candidate_rows:
            values = candidate.as_dict()
            self._candidates.writerow(
                [scientific_csv_value(values[column]) for column in CANDIDATE_COLUMNS]
            )
            self.n_candidates += 1
        self._candidates_fh.flush()

    def write_manifest_row(self, row: Mapping[str, Any]) -> None:
        """Write and flush one entry manifest row."""
        self._manifest.writerow(row)
        self._manifest_fh.flush()

    def write_crystallization_rows(self, result: EntryResult) -> None:
        """Write condition and summary rows for one entry when enabled."""
        if (
            self._crystallization_conditions is None
            or self._crystallization_summary is None
            or self._crystallization_conditions_fh is None
            or self._crystallization_summary_fh is None
        ):
            return
        for row in result.crystallization_condition_rows:
            self._crystallization_conditions.writerow(
                {
                    column: scientific_csv_value(row[column])
                    for column in CONDITION_COLUMNS
                }
            )
            self.n_crystallization_conditions += 1
        summary = result.crystallization_summary_row or unavailable_summary(
            result.pdb_id
        )
        self._crystallization_summary.writerow(
            {
                column: scientific_csv_value(summary[column])
                for column in SUMMARY_COLUMNS
            }
        )
        self.n_crystallization_summaries += 1
        self._crystallization_conditions_fh.flush()
        self._crystallization_summary_fh.flush()

    def write_density_context_row(self, result: EntryResult) -> None:
        """Write exactly one available or explicitly uncomputed entry row."""
        if self._density_context is None or self._density_context_fh is None:
            return
        row = dict.fromkeys(DENSITY_CONTEXT_COLUMNS, "")
        row.update(
            {
                "pdbID": result.pdb_id,
                "density_context_status": "not_computed",
            }
        )
        if result.density_context_row:
            if set(result.density_context_row) != set(DENSITY_CONTEXT_COLUMNS):
                raise RuntimeError(
                    "density context row does not match its output schema"
                )
            row.update(result.density_context_row)
        self._density_context.writerow(
            {
                column: scientific_csv_value(row[column])
                for column in DENSITY_CONTEXT_COLUMNS
            }
        )
        self.n_density_contexts += 1
        self._density_context_fh.flush()

    def write_confidence_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write scored confidence rows and synchronized inputs when enabled."""
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
        self._confidence.writerows(
            {
                column: scientific_csv_value(row[column])
                for column in self._confidence_columns
            }
            for row in rows
        )
        if self._confidence_inputs is not None and input_rows is not None:
            self._confidence_inputs.writerows(
                {
                    column: scientific_csv_value(row[column])
                    for column in CONFIDENCE_INPUT_COLUMNS
                }
                for row in input_rows
            )
        self.n_confidence += len(rows)
        self._confidence_fh.flush()
        if self._confidence_inputs_fh is not None:
            self._confidence_inputs_fh.flush()
