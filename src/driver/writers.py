"""Write and flush per-entry CSV results using shared output schemas.

Flush each result so interrupted runs retain completed work. Resume reads
these same schemas to determine what remains.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

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
from driver.layout import OutputLayout
from edstats_statistics import DENSITY_CONTEXT_COLUMNS, EDSTATS_COLUMNS
from output_rows import MetalStatsRow, blank_if_unmeasured, scientific_csv_value
from worker.contracts import EntryResult

if TYPE_CHECKING:
    # The plan module reads this module's schemas; only the annotation crosses back.
    from driver.confidence import ConfidencePlan

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

    #: Written only when bond analysis runs; ``--no-bonds`` leaves them closed.
    BOND_STAGE_OUTPUTS: ClassVar[tuple[str, ...]] = ("bonds", "candidates")
    #: Written whether or not bonds are enabled; every resume commit merges them.
    ALWAYS_WRITTEN_EXTRAS: ClassVar[tuple[str, ...]] = (
        "crystallization_conditions",
        "crystallization_summary",
        "density_context",
    )
    #: Present only when confidence analysis is on; merged only when enabled.
    CONFIDENCE_OUTPUTS: ClassVar[tuple[str, ...]] = ("confidence", "confidence_inputs")

    @classmethod
    def from_layout(cls, layout: OutputLayout, plan: ConfidencePlan) -> OutputTargets:
        """The files a run with this layout and confidence plan writes."""
        return cls(
            manifest=layout.manifest,
            stats=layout.stats,
            bonds=layout.bonds,
            candidates=layout.candidates,
            crystallization_conditions=layout.crystallization_conditions,
            crystallization_summary=layout.crystallization_summary,
            density_context=layout.density_context,
            confidence=plan.stream_path,
            confidence_inputs=(
                layout.confidence_inputs if plan.synchronize_inputs else None
            ),
        )

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


class _CsvStream:
    """One open CSV output: its handle, its schema, and a running row count.

    The header is written on creation, so a run that finds nothing still
    leaves a readable file behind. Every write flushes, so an interrupted
    run retains the rows of the entries that completed.
    """

    def __init__(self, handle: TextIO, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        self.n_rows = 0
        self._handle = handle
        self._writer = csv.DictWriter(handle, fieldnames=self.columns)
        self._writer.writeheader()

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Project rows onto the schema as scientific CSV text, then flush."""
        for row in rows:
            self._writer.writerow(
                {column: scientific_csv_value(row[column]) for column in self.columns}
            )
            self.n_rows += 1
        self._handle.flush()

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Write one row whose values are already CSV text, then flush."""
        self._writer.writerow(row)
        self.n_rows += 1
        self._handle.flush()


def _stream_if_open(
    handles: Mapping[str, TextIO], name: str, columns: Sequence[str]
) -> _CsvStream | None:
    handle = handles.get(name)
    return _CsvStream(handle, columns) if handle is not None else None


def _rows_written(stream: _CsvStream | None) -> int:
    return stream.n_rows if stream is not None else 0


class OutputWriters:
    """The streamed CSV outputs, with running row counts."""

    def __init__(
        self,
        handles: Mapping[str, TextIO],
        *,
        confidence_columns: Sequence[str] | None = None,
    ) -> None:
        """Start one stream per open output and emit every header.

        ``handles`` is keyed by ``OutputTargets`` field name; the manifest and
        statistics outputs are always present, every other one is optional.
        """
        confidence_handle = handles.get("confidence")
        if confidence_handle is not None and confidence_columns is None:
            raise ValueError("confidence columns are required with a confidence output")
        if "confidence_inputs" in handles and confidence_handle is None:
            raise ValueError("confidence inputs synchronization requires scored output")
        self._manifest = _CsvStream(handles["manifest"], MANIFEST_COLUMNS)
        self._stats = _CsvStream(handles["stats"], STATS_COLUMNS)
        self._bonds = _stream_if_open(handles, "bonds", BOND_COLUMNS)
        self._candidates = _stream_if_open(handles, "candidates", CANDIDATE_COLUMNS)
        self._confidence = (
            _CsvStream(confidence_handle, confidence_columns)
            if confidence_handle is not None and confidence_columns is not None
            else None
        )
        self._confidence_inputs = _stream_if_open(
            handles, "confidence_inputs", CONFIDENCE_INPUT_COLUMNS
        )
        self._crystallization_conditions = _stream_if_open(
            handles, "crystallization_conditions", CONDITION_COLUMNS
        )
        self._crystallization_summary = _stream_if_open(
            handles, "crystallization_summary", SUMMARY_COLUMNS
        )
        self._density_context = _stream_if_open(
            handles, "density_context", DENSITY_CONTEXT_COLUMNS
        )

    @property
    def n_sites(self) -> int:
        """Metal and cofactor site rows written to the statistics output."""
        return self._stats.n_rows

    @property
    def n_bonds(self) -> int:
        """Bond rows written; zero while that output is disabled."""
        return _rows_written(self._bonds)

    @property
    def n_candidates(self) -> int:
        """Contact-candidate rows written; zero while disabled."""
        return _rows_written(self._candidates)

    @property
    def n_confidence(self) -> int:
        """Scored confidence rows written; zero while disabled."""
        return _rows_written(self._confidence)

    @property
    def n_crystallization_conditions(self) -> int:
        """Crystallization condition rows written."""
        return _rows_written(self._crystallization_conditions)

    @property
    def n_crystallization_summaries(self) -> int:
        """Crystallization summary rows written."""
        return _rows_written(self._crystallization_summary)

    @property
    def n_density_contexts(self) -> int:
        """Density context rows written, one per entry."""
        return _rows_written(self._density_context)

    def write_stats_rows(self, rows: Sequence[MetalStatsRow]) -> None:
        """Write and flush metal-site statistics rows."""
        if not rows:
            return
        # Project every row before writing any, so a malformed batch writes nothing.
        self._stats.write_rows([row.as_output_dict(STATS_COLUMNS) for row in rows])

    def write_bond_rows(self, bond_rows: Sequence[BondRow]) -> None:
        """Write and flush analyzed bond rows when that output is enabled."""
        if self._bonds is None or not bond_rows:
            return
        self._bonds.write_rows(bond.as_dict() for bond in bond_rows)

    def write_candidate_rows(self, candidate_rows: Sequence[CandidateRow]) -> None:
        """Write and flush contact-candidate rows when enabled."""
        if self._candidates is None or not candidate_rows:
            return
        self._candidates.write_rows(candidate.as_dict() for candidate in candidate_rows)

    def write_manifest_row(self, row: Mapping[str, Any]) -> None:
        """Write and flush one entry manifest row."""
        self._manifest.write_row(row)

    def write_crystallization_rows(self, result: EntryResult) -> None:
        """Write condition and summary rows for one entry when enabled."""
        if (
            self._crystallization_conditions is None
            or self._crystallization_summary is None
        ):
            return
        self._crystallization_conditions.write_rows(
            result.crystallization_condition_rows
        )
        summary = result.crystallization_summary_row or unavailable_summary(
            result.pdb_id
        )
        self._crystallization_summary.write_rows([summary])

    def write_density_context_row(self, result: EntryResult) -> None:
        """Write exactly one available or explicitly uncomputed entry row."""
        if self._density_context is None:
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
        self._density_context.write_rows([row])

    def write_confidence_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write scored confidence rows and synchronized inputs when enabled."""
        if self._confidence is None or not rows:
            return
        expected = set(self._confidence.columns)
        for row in rows:
            if set(row) != expected:
                raise RuntimeError("confidence row does not match its output schema")
        self._confidence.write_rows(rows)
        if self._confidence_inputs is not None:
            self._confidence_inputs.write_rows(rows)
