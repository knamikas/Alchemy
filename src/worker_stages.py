"""The per-entry analysis stages and the outcomes they hand back.

Each stage returns an outcome instead of editing the entry result itself;
``worker`` folds the outcomes in, in stage order. The early exit, the density
stage, the bond stage, and the checks that join their rows all live here.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from analysis_config import MAX_ANALYZED_METAL_SITES
from codes import (
    CoordinateMappingStatus,
    EntryStatus,
    ReasonCode,
    SelectedSiteStatus,
    WarningCode,
)
from coordination.analysis import (
    BondAnalysisMetadata,
    BondAnalysisResult,
    run_bond_analysis,
)
from coordination.dpi import DpiInputs
from coordination.schema import (
    STATS_EXTRA_COLUMNS,
    check_row_schema,
    stats_extra_values,
)
from density_analysis import (
    Ccp4ToolTimeoutError,
    MtzfixValidationError,
    elapsed_s,
    run_density_analysis,
)
from edstats_statistics import extract_metal_statistics
from metal_elements import METAL_ELEMENTS
from output_rows import MetalStatsRow, csv_value
from run_logging import truncate
from structure_analysis import NAN, AtomSite, StructureContext
from worker_contracts import DensityProvenance, WorkerConfig
from worker_inputs import EntryInputs

METALS_SET = set(METAL_ELEMENTS)

# Reported alongside the machine-readable code in ``status_detail``.
IDENTIFICATION_REASON_MESSAGES = {
    ReasonCode.COFACTOR_COORDINATE_JOIN_FAILED: (
        "cofactor EDSTATS row did not match a coordinate residue"
    ),
    ReasonCode.COFACTOR_WITHOUT_SELECTED_METAL: (
        "matched cofactor has no selected configured metal site"
    ),
    ReasonCode.METAL_SITE_WITHOUT_DENSITY: (
        "selected coordinate metal site is absent from the statistics table"
    ),
}

# The manifest keeps a bounded summary; the full message is logged.
MAX_MANIFEST_STATUS_DETAIL_CHARS = 300

TIMEOUT_LOG_DIRNAME = "ccp4_timeout_logs"

CodeT = TypeVar("CodeT", bound=str)


def merged_codes(existing: Iterable[CodeT], new: Iterable[CodeT]) -> list[CodeT]:
    """Append ``new`` codes to ``existing``, keeping first occurrences in order.

    Reason and warning codes are serialized in the order they were recorded,
    so a stage never reorders codes an earlier stage already set.
    """
    return list(dict.fromkeys([*existing, *new]))


@dataclass(slots=True)
class DensityOutcome:
    """What the density stage produced, or why it could not.

    Stages return an outcome instead of editing the entry result themselves;
    the worker folds each one in, in stage order.
    """

    rows: list[MetalStatsRow] = field(default_factory=list[MetalStatsRow])
    header: list[str] = field(default_factory=list[str])
    #: Set only when density could not be produced.
    reason_code: str = ""
    status_detail: str = ""
    timings: dict[str, float] = field(default_factory=dict[str, float])
    warning_codes: list[str] = field(default_factory=list[str])
    density_context_row: dict[str, Any] = field(default_factory=dict[str, Any])
    provenance: DensityProvenance = field(default_factory=DensityProvenance)

    @property
    def failed(self) -> bool:
        """Whether density is unavailable for this entry."""
        return bool(self.reason_code)


@dataclass(slots=True)
class BondOutcome:
    """The bond-stage analysis, or the failure that stood in for it."""

    analysis: BondAnalysisResult
    #: Set only when the geometry stage raised; the analysis is then empty.
    status_detail: str = ""
    timings: dict[str, float] = field(default_factory=dict[str, float])

    @property
    def failed(self) -> bool:
        """Whether the geometry stage raised."""
        return bool(self.status_detail)


@dataclass(frozen=True, slots=True)
class EarlyOutcome:
    """An entry that finishes before any map is calculated."""

    status: EntryStatus
    n_metals: int
    n_bonds: int | None
    n_candidates: int | None
    reason_codes: tuple[str, ...] = ()
    status_detail: str = ""
    no_metals: bool = False
    metal_site_limit_exceeded: bool = False
    confidence_inputs_missing_reason: str = ""


def excluded_zero_occupancy_metals(
    structure: StructureContext, selected_metals: Sequence[AtomSite]
) -> int:
    """Return metal records excluded for valid zero occupancy.

    Report these separately so an entry with modeled-absent metals is
    distinguishable from an entry with no metal records.
    """
    with_absent = structure.metal_atoms(
        METALS_SET, canonical=True, include_zero_occupancy=True
    )
    return len(with_absent) - len(selected_metals)


def early_outcome(
    structure: StructureContext, selected_metals: Sequence[AtomSite]
) -> EarlyOutcome | None:
    """Finish early when neither density nor contact stages can produce output.

    Entries above the site limit are excluded before any map is calculated;
    entries without a selected metal need no analysis, unless unknown element
    symbols make metal absence indeterminate.
    """
    if len(selected_metals) > MAX_ANALYZED_METAL_SITES:
        return EarlyOutcome(
            status=EntryStatus.OK,
            n_metals=len(selected_metals),
            n_bonds=0,
            n_candidates=0,
            reason_codes=(ReasonCode.METAL_SITE_LIMIT_EXCEEDED,),
            metal_site_limit_exceeded=True,
        )
    if selected_metals:
        return None
    if structure.records.unknown_element_atom_count:
        unknown_count = structure.records.unknown_element_atom_count
        return EarlyOutcome(
            status=EntryStatus.PARTIAL,
            n_metals=0,
            n_bonds=None,
            n_candidates=None,
            reason_codes=(ReasonCode.METAL_PRESENCE_INDETERMINATE,),
            status_detail=truncate(
                "cannot establish metal absence: "
                f"{unknown_count} atom(s) have missing or invalid element symbols",
                MAX_MANIFEST_STATUS_DETAIL_CHARS,
            ),
            confidence_inputs_missing_reason=ReasonCode.METAL_PRESENCE_INDETERMINATE,
        )
    return EarlyOutcome(
        status=EntryStatus.OK, n_metals=0, n_bonds=0, n_candidates=0, no_metals=True
    )


def _preserve_timeout_log(
    timeout: Ccp4ToolTimeoutError, pdb_id: str, output_dir: str
) -> str:
    """Preserve a timed-out program's log outside its scratch directory.

    Return an empty string if no log can be copied. Diagnostic failures must not
    change the entry outcome.
    """
    source = getattr(timeout, "log_path", "")
    if not source or not os.path.isfile(source):
        return ""
    try:
        destination_dir = os.path.join(output_dir, TIMEOUT_LOG_DIRNAME)
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(
            destination_dir, f"{pdb_id}_{timeout.tool}_timeout.log"
        )
        shutil.copyfile(source, destination)
        return destination
    except OSError:
        return ""


def run_density_stage(
    pdb_id: str,
    cfg: WorkerConfig,
    inputs: EntryInputs,
    structure: StructureContext,
) -> DensityOutcome:
    """Calculate the maps, run EDSTATS, and extract this entry's statistics.

    A handled failure leaves ``rows`` and ``header`` empty and records the
    reason on the outcome.
    """
    outcome = DensityOutcome()
    density_started = time.monotonic()
    try:
        res = run_density_analysis(
            pdb_id,
            mtz_path=inputs.mtz,
            pdb_path=inputs.pdb,
            out_dir=inputs.work_dir,
            reslo=inputs.map_reslo,
            reshi=inputs.map_reshi,
            env=cfg.env,
            map_scope=cfg.density_map_scope,
            keep_full_maps=cfg.keep_intermediates,
            pdb_redo_is_twin=inputs.pdb_redo_is_twin,
            tool_timeout_s=cfg.ccp4_timeout,
        )
    except Ccp4ToolTimeoutError as exc:
        # A timeout does not establish an input defect; see ``retryable_for``.
        outcome.provenance = DensityProvenance(
            ccp4_timeout_log_path=_preserve_timeout_log(exc, pdb_id, cfg.output_dir)
        )
        outcome.reason_code = ReasonCode.CCP4_TOOL_TIMEOUT
        outcome.status_detail = truncate(
            f"density unavailable: {exc}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
        outcome.timings.update(exc.timings)
    except MtzfixValidationError as exc:
        # Coefficient validation failed; geometry can still be assessed.
        outcome.reason_code = ReasonCode.MTZFIX_VALIDATION_FAILURE
        outcome.status_detail = truncate(
            f"density unavailable: {exc}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
        outcome.timings.update(exc.timings)
    else:
        outcome.timings.update(res.timings)
        outcome.provenance = DensityProvenance(
            density_map_scope_used=res.density_map_scope_used,
            density_full_map_bytes=res.full_map_bytes,
            density_edstats_map_bytes=res.edstats_map_bytes,
        )
        if res.twin_coefficient_normalization_applied:
            outcome.warning_codes.append(
                WarningCode.TWIN_REFMAC_COEFFICIENTS_NORMALIZED
            )
        statistics_started = time.monotonic()
        extraction = extract_metal_statistics(
            pdb_id, res.stats_out, METALS_SET, cfg.cofactors, structure=structure
        )
        outcome.rows = extraction.rows
        outcome.header = extraction.header
        outcome.density_context_row = extraction.density_context_row
        outcome.warning_codes = merged_codes(
            outcome.warning_codes, extraction.warning_codes
        )
        outcome.timings["statistics_extraction_s"] = elapsed_s(statistics_started)
    finally:
        outcome.timings["density_total_s"] = elapsed_s(density_started)
    return outcome


def identification_reason_codes(rows: list[MetalStatsRow]) -> list[ReasonCode]:
    """Deduplicated reason codes for EDSTATS rows that could not be joined."""
    codes: list[ReasonCode] = []
    for row in rows:
        mapping_status = row.coordinate_mapping_status
        site_status = row.selected_metal_site_status
        if mapping_status == CoordinateMappingStatus.RESIDUE_NOT_FOUND:
            codes.append(ReasonCode.COFACTOR_COORDINATE_JOIN_FAILED)
        elif site_status == SelectedSiteStatus.NO_SELECTED_METAL:
            codes.append(ReasonCode.COFACTOR_WITHOUT_SELECTED_METAL)
    return list(dict.fromkeys(codes))


def sites_without_density_rows(
    rows: list[MetalStatsRow], selected_metals: Sequence[AtomSite]
) -> list[tuple[int, int, int, int]]:
    """Find selected metal sites missing from the statistics rows.

    Derive expected keys from coordinates so this check also works when bond
    analysis is disabled or fails.
    """
    reported = {row.site_key for row in rows if row.site_key is not None}
    return [
        metal.source_key
        for metal in selected_metals
        if metal.source_key not in reported
    ]


def run_bond_stage(
    pdb_id: str,
    cfg: WorkerConfig,
    inputs: EntryInputs,
    structure: StructureContext,
    selected_metals: Sequence[AtomSite],
    rows: list[MetalStatsRow],
    header: list[str],
) -> BondOutcome:
    """Evaluate the contacts around each site, unless ``--no-bonds`` cleared it.

    ``selected_metals`` is the structure's canonical metal selection, computed
    once by the caller. A bond-stage failure must not lose the EDSTATS rows
    already computed, so it is recorded on the outcome and the empty analysis
    is returned instead.
    """
    analysis = BondAnalysisResult(
        bond_rows=[],
        candidate_rows=[],
        site_summaries={},
        metadata=BondAnalysisMetadata(
            partial_reason_codes=[],
            warning_codes=list(structure.warning_codes),
            messages=[],
        ),
    )
    outcome = BondOutcome(analysis)
    if not cfg.bonds:
        non_finite_metals = [
            metal for metal in selected_metals if not metal.coordinates_valid
        ]
        if non_finite_metals:
            analysis.metadata.partial_reason_codes.append(
                ReasonCode.NON_FINITE_METAL_COORDINATES
            )
            analysis.metadata.messages.append(
                "geometry unavailable for selected metal site(s) with "
                "non-finite coordinates"
            )
        return outcome

    bond_started = time.monotonic()
    try:
        outcome.analysis = run_bond_analysis(
            pdb_id,
            inputs.pdb,
            rows,
            header,
            DpiInputs(
                resolution=inputs.data_reshi,
                data_json=inputs.data_json,
                pdb_path=inputs.pdb,
                mtz_path=inputs.mtz,
            ),
            structure=structure,
            connection_path=inputs.source_coordinate_path,
        )
    except Exception as e:
        outcome.status_detail = truncate(
            f"bond: {type(e).__name__}: {e}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
    finally:
        outcome.timings["bond_analysis_s"] = elapsed_s(bond_started)
    return outcome


def append_site_fields(
    rows: list[MetalStatsRow],
    # Keyed by ``AtomSite.source_key``, but the lookup key below is a row's own
    # ``site_key``, which is ``None`` for a cofactor that joined no metal site.
    site_summaries: Mapping[Any, Mapping[str, Any]],
    structure: StructureContext,
) -> None:
    """Extend each EDSTATS row with its per-site contact and provenance values."""
    for index, row in enumerate(rows):
        summary = dict(site_summaries.get(row.site_key, {}))
        summary["density_observation_id"] = row.density_observation_id
        summary["density_scope"] = row.density_scope
        summary["density_shared_site_count"] = row.density_shared_site_count
        summary["density_is_shared"] = row.density_is_shared
        summary["coordinate_mapping_status"] = row.coordinate_mapping_status
        summary["selected_metal_site_status"] = row.selected_metal_site_status
        coverage = summary.get("geometry_coverage_image_inclusive", NAN)
        if isinstance(coverage, float) and not math.isfinite(coverage):
            coverage = summary.get("geometry_coverage_explicit", NAN)
        extra = stats_extra_values(row.pdb_id, structure, row.site, summary)
        if index == 0:
            check_row_schema(extra, STATS_EXTRA_COLUMNS, "metal_sites_all.csv")
        rows[index] = row.with_fields(
            (
                *row.fields,
                csv_value(coverage),
                *(csv_value(extra[column]) for column in STATS_EXTRA_COLUMNS),
            )
        )
