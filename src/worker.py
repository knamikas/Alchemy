"""Process one PDB entry and record its results and failures.

Configuration is initialized once per worker to avoid serializing it per entry.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import signal
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from multiprocessing.queues import Queue, SimpleQueue
from typing import Any, Literal

from analysis_config import MAX_ANALYZED_METAL_SITES
from codes import (
    CoordinateMappingStatus,
    EntryStatus,
    ReasonCode,
    SelectedSiteStatus,
    WarningCode,
)
from coordinate_conversion import first_model_pdb
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
from crystallization_conditions import extract_crystallization_context
from density_analysis import (
    Ccp4EntryLimitationError,
    Ccp4ToolTimeoutError,
    MtzfixValidationError,
    run_density_analysis,
)
from inputs import (
    PdbRedoMetadata,
    ensure_entry_available,
    entry_dir_for,
    first_existing,
    prepare_inputs,
    read_map_column_resolution,
    read_pdb_redo_metadata,
    read_resolution,
    resolve_manual_inputs,
)
from metal_elements import METAL_ELEMENTS
from metal_identification import extract_metal_statistics
from output_rows import MetalStatsRow, csv_value
from run_logging import configure_worker_logging, logger_for, truncate
from scratch import create_owned_scratch_directory
from structure_analysis import NAN, StructureContext, load_structure
from worker_contracts import (
    CoordinateProvenance,
    DensityProvenance,
    EntryResult,
    InflightEvent,
    PdbRedoProvenance,
    SoftwareProvenance,
    WorkerConfig,
)
from worker_memory import release_idle_memory

METALS_SET = set(METAL_ELEMENTS)

# Reported alongside the machine-readable code in ``status_detail``.
IDENTIFICATION_REASON_MESSAGES = {
    ReasonCode.COFACTOR_COORDINATE_JOIN_FAILED: (
        "cofactor EDSTATS row did not match a coordinate residue"
    ),
    ReasonCode.AMBIGUOUS_COORDINATE_RESIDUE_JOIN: (
        "EDSTATS row matched multiple coordinate residues"
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

logger = logger_for(__name__)

# Classify failures likely to recur on identical inputs. Resume still retries
# errors because the inputs or software may have changed.
DETERMINISTIC_PROCESSING_ERRORS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    Ccp4EntryLimitationError,
    LookupError,
    NotImplementedError,
    TypeError,
    ValueError,
)


worker_config: WorkerConfig | None = None
_inflight_queue: SimpleQueue[InflightEvent] | None = None


def initialize_worker(
    cfg: WorkerConfig,
    inflight: SimpleQueue[InflightEvent] | None = None,
    log_queue: Queue[logging.LogRecord] | None = None,
) -> None:
    """Initialize process-local configuration, logging, and signal state."""
    global worker_config, _inflight_queue
    worker_config = cfg
    _inflight_queue = inflight
    # Keep each worker and its CCP4 children in one group for driver cleanup.
    # Windows uses Process.kill because setpgrp is unavailable.
    with contextlib.suppress(AttributeError, OSError):
        os.setpgrp()
    # Reset the inherited SIGTERM handler so pool termination cannot interrupt
    # worker finalizers with KeyboardInterrupt.
    with contextlib.suppress(AttributeError, OSError, ValueError):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    configure_worker_logging(log_queue, level=cfg.log_level)


def announce_inflight(state: Literal["start", "end"], pdb_id: str) -> None:
    """Notify the driver which entry this worker holds.

    Synchronous queue writes let the driver identify the entry if the worker dies
    without returning a result.
    """
    if _inflight_queue is None:
        return
    with contextlib.suppress(Exception):  # bookkeeping must never fail an entry
        _inflight_queue.put(InflightEvent(state, os.getpid(), pdb_id))


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


def initial_result(
    pdb_id: str,
    cfg: WorkerConfig,
    manual_inputs: dict[str, str | None] | None,
) -> EntryResult:
    """Return the per-entry result skeleton, pre-filled with run provenance."""
    return EntryResult(
        pdb_id=pdb_id,
        software=SoftwareProvenance(
            alchemy_commit=cfg.alchemy_commit,
            gemmi_version=cfg.gemmi_version,
            ccp4_version=cfg.ccp4_version,
            reference_data_id=cfg.reference_data_id,
            analysis_config_id=cfg.analysis_config_id,
        ),
        pdb_redo=PdbRedoProvenance(
            refinement_state="manual" if manual_inputs else "final"
        ),
    )


# Partial outcomes worth retrying: the stage that failed reported nothing about
# the entry itself, so a rerun may still complete it.
RETRYABLE_PARTIAL_REASON_CODES: frozenset[str] = frozenset(
    {ReasonCode.CCP4_TOOL_TIMEOUT, ReasonCode.BOND_STAGE_FAILURE}
)


def retryable_for(status: EntryStatus, reason_codes: Iterable[str]) -> bool:
    """Whether an ordinary resume should retry an entry with this outcome.

    A completed entry is terminal. Skips and errors are always retried, since a
    resume may read repaired inputs or tools. A partial entry is retried only
    when the failed stage said nothing about the entry: a CCP4 timeout or a
    bond-stage exception. Every other partial reason describes the entry itself
    and would recur on the same inputs.
    """
    if status == EntryStatus.OK:
        return False
    if status in (EntryStatus.SKIP, EntryStatus.ERROR):
        return True
    return not RETRYABLE_PARTIAL_REASON_CODES.isdisjoint(reason_codes)


def worker_death_result(pdb_id: str, cfg: WorkerConfig, pid: int) -> EntryResult:
    """Synthesize the retryable result a killed worker could not return."""
    result = initial_result(pdb_id, cfg, cfg.manual_inputs)
    result.status = EntryStatus.ERROR
    result.reason_codes = [ReasonCode.WORKER_PROCESS_DIED]
    result.retryable = retryable_for(result.status, result.reason_codes)
    result.status_detail = (
        f"worker process {pid} terminated without returning a result "
        f"(out-of-memory kill or crash); {pdb_id} was not analyzed"
    )
    return result


def is_worker_death_result(result: EntryResult) -> bool:
    """Whether ``result`` was synthesized by ``worker_death_result``."""
    return result.reason_codes == [ReasonCode.WORKER_PROCESS_DIED]


def _source_coordinate_format(cfg: WorkerConfig, source_path: str) -> tuple[str, bool]:
    """The deposited coordinate format and whether analysis converted it to PDB."""
    manual = cfg.manual_inputs
    if manual:
        converted = bool(manual.get("cif_file"))
    else:
        converted = source_path.lower().endswith((".cif", ".cif.gz"))
    return ("mmcif" if converted else "pdb", converted)


def _source_coordinate_path(
    cfg: WorkerConfig, pdb_id: str, entry: str, analysis_path: str
) -> str:
    manual = cfg.manual_inputs
    if manual:
        return manual.get("cif_file") or manual.get("pdb_file") or ""
    return (
        first_existing(
            os.path.join(entry, f"{pdb_id}_final.cif"),
            os.path.join(entry, f"{pdb_id}_final.cif.gz"),
            os.path.join(entry, f"{pdb_id}_final.pdb"),
            os.path.join(entry, f"{pdb_id}_final.pdb.gz"),
        )
        or analysis_path
    )


def source_coordinate_provenance_path(
    cfg: WorkerConfig, pdb_id: str, source_path: str
) -> str:
    """Keep mirror provenance portable while preserving a manual input path."""
    if cfg.manual_inputs:
        return source_path
    return f"{pdb_id[1:3]}/{pdb_id}/{os.path.basename(source_path)}"


def _resolve_entry_dir(pdb_id: str, cfg: WorkerConfig) -> str:
    """Locate an entry's PDB-REDO directory, downloading it when permitted."""
    if cfg.allow_download:
        used_root = ensure_entry_available(
            pdb_id, cfg.pdb_redo_root, cfg.pdb_redo_cache
        )
        return entry_dir_for(used_root, pdb_id)
    return entry_dir_for(cfg.input_root, pdb_id)


@dataclass(frozen=True)
class EntryInputs:
    """Resolved input paths, metadata, and resolution limits for one entry."""

    work_dir: str
    mtz: str
    # First-model analysis coordinates; source_coordinate_path retains the input.
    pdb: str
    # PDB-REDO metadata or manual --data-json; None if none was supplied.
    data_json: str | None
    # The diffraction data's own high-resolution limit, as distinct from the
    # map columns' range below.
    data_reshi: float
    map_reslo: float
    map_reshi: float
    pdb_redo_is_twin: bool
    source_coordinate_path: str


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Coordinate and PDB-REDO provenance the input stage establishes."""

    coordinates: CoordinateProvenance
    pdb_redo_metadata: PdbRedoMetadata


@dataclass(slots=True)
class DensityOutcome:
    """What the density stage produced, or why it could not.

    Stages return an outcome instead of editing the entry result themselves;
    ``_process_entry`` folds each one in, in stage order.
    """

    rows: list[MetalStatsRow] = field(default_factory=list)
    header: list[str] = field(default_factory=list)
    #: Set only when density could not be produced.
    reason_code: str = ""
    status_detail: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    warning_codes: list[str] = field(default_factory=list)
    density_context_row: dict[str, Any] = field(default_factory=dict)
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
    timings: dict[str, float] = field(default_factory=dict)

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


def _run_density_stage(
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
        outcome.warning_codes = list(
            dict.fromkeys([*outcome.warning_codes, *extraction.warning_codes])
        )
        outcome.timings["statistics_extraction_s"] = round(
            time.monotonic() - statistics_started, 3
        )
    finally:
        outcome.timings["density_total_s"] = round(
            time.monotonic() - density_started, 3
        )
    return outcome


def _apply_density_outcome(result: EntryResult, outcome: DensityOutcome) -> None:
    """Fold the density stage into the entry result."""
    result.timings.update(outcome.timings)
    result.warning_codes = list(
        dict.fromkeys(result.warning_codes + outcome.warning_codes)
    )
    result.density_context_row = outcome.density_context_row
    result.density = outcome.provenance
    if outcome.failed:
        result.reason_codes = list(
            dict.fromkeys(result.reason_codes + [outcome.reason_code])
        )
        result.status_detail = outcome.status_detail
        result.confidence_inputs_missing_reason = outcome.reason_code


def _identification_reason_codes(rows: list[MetalStatsRow]) -> list[ReasonCode]:
    """Deduplicated reason codes for EDSTATS rows that could not be joined."""
    codes: list[ReasonCode] = []
    for row in rows:
        mapping_status = row.coordinate_mapping_status
        site_status = row.selected_metal_site_status
        if mapping_status == CoordinateMappingStatus.RESIDUE_NOT_FOUND:
            codes.append(ReasonCode.COFACTOR_COORDINATE_JOIN_FAILED)
        elif mapping_status == CoordinateMappingStatus.MULTIPLE_RESIDUES:
            codes.append(ReasonCode.AMBIGUOUS_COORDINATE_RESIDUE_JOIN)
        elif site_status == SelectedSiteStatus.NO_SELECTED_METAL:
            codes.append(ReasonCode.COFACTOR_WITHOUT_SELECTED_METAL)
    return list(dict.fromkeys(codes))


def _excluded_zero_occupancy_metals(structure: StructureContext) -> int:
    """Return metal records excluded for valid zero occupancy.

    Report these separately so an entry with modeled-absent metals is
    distinguishable from an entry with no metal records.
    """
    selected = structure.metal_atoms(METALS_SET, canonical=True)
    with_absent = structure.metal_atoms(
        METALS_SET, canonical=True, include_zero_occupancy=True
    )
    return len(with_absent) - len(selected)


def _sites_without_density_rows(
    rows: list[MetalStatsRow], structure: StructureContext
) -> list[tuple[int, int, int, int]]:
    """Find selected metal sites missing from the statistics rows.

    Derive expected keys from coordinates so this check also works when bond
    analysis is disabled or fails.
    """
    reported = {row.site_key for row in rows if row.site_key is not None}
    selected = structure.metal_atoms(METALS_SET, canonical=True)
    return [metal.source_key for metal in selected if metal.source_key not in reported]


def append_site_fields(
    rows: list[MetalStatsRow],
    # Keyed by ``AtomSite.source_key``, but the lookup key below is a row's own
    # ``site_key``, which is ``None`` for a cofactor that joined no metal site.
    site_summaries: Mapping[Any, dict[str, Any]],
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


def run_bond_stage(
    pdb_id: str,
    cfg: WorkerConfig,
    inputs: EntryInputs,
    structure: StructureContext,
    rows: list[MetalStatsRow],
    header: list[str],
) -> BondOutcome:
    """Evaluate the contacts around each site, unless ``--no-bonds`` cleared it.

    A bond-stage failure must not lose the EDSTATS rows already computed, so
    it is recorded on the outcome and the empty analysis is returned instead.
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
            metal
            for metal in structure.metal_atoms(METALS_SET, canonical=True)
            if not metal.coordinates_valid
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
        outcome.timings["bond_analysis_s"] = round(time.monotonic() - bond_started, 3)
    return outcome


def _apply_bond_outcome(result: EntryResult, outcome: BondOutcome) -> None:
    """Fold the bond stage into the entry result."""
    if outcome.failed:
        result.status_detail = outcome.status_detail
        result.reason_codes = list(
            dict.fromkeys(result.reason_codes + [ReasonCode.BOND_STAGE_FAILURE])
        )
        result.confidence_inputs_missing_reason = ReasonCode.BOND_STAGE_FAILURE
    result.timings.update(outcome.timings)


def _finalize_result(
    result: EntryResult,
    identification_codes: list[ReasonCode],
    bond_meta: BondAnalysisMetadata,
    structure: StructureContext,
) -> None:
    """Combine stage outcomes into the entry status, reason codes, and counts."""
    result.reason_codes = list(
        dict.fromkeys(
            result.reason_codes + identification_codes + bond_meta.partial_reason_codes
        )
    )
    messages = [IDENTIFICATION_REASON_MESSAGES[code] for code in identification_codes]
    messages.extend(bond_meta.messages)
    if messages:
        existing_detail = result.status_detail
        result.status_detail = truncate(
            "; ".join(([existing_detail] if existing_detail else []) + messages),
            MAX_MANIFEST_STATUS_DETAIL_CHARS,
        )
    result.warning_codes = list(
        dict.fromkeys(result.warning_codes + bond_meta.warning_codes)
    )
    result.status = EntryStatus.PARTIAL if result.reason_codes else EntryStatus.OK
    # Count coordinate sites even when their EDSTATS joins failed.
    result.n_metals = len(structure.metal_atoms(METALS_SET, canonical=True))
    result.n_bonds = len(result.bond_rows)
    result.n_candidates = len(result.candidate_rows)


def _prepare_analysis_inputs(
    pdb_id: str,
    cfg: WorkerConfig,
    entry: str,
    work_dir: str,
) -> tuple[EntryInputs, StructureContext, InputProvenance]:
    """Prepare the same first-model PDB for EDSTATS and Gemmi."""
    manual_inputs = cfg.manual_inputs
    data_json: str | None = None
    if manual_inputs:
        mtz, pdb = resolve_manual_inputs(
            pdb_id,
            pdb_file=manual_inputs.get("pdb_file"),
            mtz_file=manual_inputs.get("mtz_file"),
            cif_file=manual_inputs.get("cif_file"),
            work_dir=work_dir,
        )
        entry = os.path.dirname(pdb) or work_dir
        data_json = manual_inputs.get("data_json")
        data_reshi = read_resolution(entry, mtz, data_json_path=data_json)
    else:
        mtz, pdb = prepare_inputs(pdb_id, entry, work_dir)
        data_reshi = read_resolution(entry, mtz)

    density_data_json = data_json if manual_inputs else os.path.join(entry, "data.json")
    pdb_redo_metadata = read_pdb_redo_metadata(
        density_data_json,
        required=bool(manual_inputs and data_json),
    )
    map_reslo, map_reshi = read_map_column_resolution(mtz)
    source_pdb = pdb
    source_coordinate_path = _source_coordinate_path(cfg, pdb_id, entry, source_pdb)
    source_format, converted = _source_coordinate_format(cfg, source_coordinate_path)
    model1_pdb = os.path.join(work_dir, f"{pdb_id}_model1.pdb")
    if os.path.realpath(model1_pdb) == os.path.realpath(source_pdb):
        model1_pdb = os.path.join(work_dir, f"{pdb_id}_analysis_model1.pdb")
    pdb, input_model_count = first_model_pdb(source_pdb, model1_pdb)
    inputs = EntryInputs(
        work_dir=work_dir,
        mtz=mtz,
        pdb=pdb,
        data_json=density_data_json,
        data_reshi=data_reshi,
        map_reslo=map_reslo,
        map_reshi=map_reshi,
        pdb_redo_is_twin=pdb_redo_metadata.is_twin,
        source_coordinate_path=source_coordinate_path,
    )
    structure = load_structure(pdb_id, pdb, source_model_count=input_model_count)
    provenance = InputProvenance(
        coordinates=CoordinateProvenance(
            source_coordinate_format=source_format,
            analysis_coordinate_format=structure.analysis_coordinate_format,
            coordinate_conversion_performed=converted,
            source_coordinate_path=source_coordinate_provenance_path(
                cfg, pdb_id, source_coordinate_path
            ),
            input_model_count=structure.input_model_count,
            model_analyzed=structure.model_analyzed,
            multi_model_structure=structure.multi_model_structure,
        ),
        pdb_redo_metadata=pdb_redo_metadata,
    )
    return inputs, structure, provenance


def _apply_input_provenance(result: EntryResult, provenance: InputProvenance) -> None:
    """Record where the analyzed coordinates and metadata came from."""
    metadata = provenance.pdb_redo_metadata
    result.coordinates = provenance.coordinates
    result.pdb_redo = replace(
        result.pdb_redo,
        pdb_redo_is_twin=metadata.is_twin,
        pdb_redo_version=metadata.version,
        pdb_redo_date=metadata.date,
    )


def _early_outcome(structure: StructureContext) -> EarlyOutcome | None:
    """Finish early when neither density nor contact stages can produce output.

    Entries above the site limit are excluded before any map is calculated;
    entries without a selected metal need no analysis, unless unknown element
    symbols make metal absence indeterminate.
    """
    selected = structure.metal_atoms(METALS_SET, canonical=True)
    if len(selected) > MAX_ANALYZED_METAL_SITES:
        return EarlyOutcome(
            status=EntryStatus.OK,
            n_metals=len(selected),
            n_bonds=0,
            n_candidates=0,
            reason_codes=(ReasonCode.METAL_SITE_LIMIT_EXCEEDED,),
            metal_site_limit_exceeded=True,
        )
    if selected:
        return None
    if structure.unknown_element_atom_count:
        unknown_count = structure.unknown_element_atom_count
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


def _apply_early_outcome(result: EntryResult, outcome: EarlyOutcome) -> None:
    """Record an entry that finished before density analysis."""
    result.status = outcome.status
    result.n_metals = outcome.n_metals
    result.n_bonds = outcome.n_bonds
    result.n_candidates = outcome.n_candidates
    result.reason_codes = list(outcome.reason_codes)
    result.status_detail = outcome.status_detail
    result.no_metals = outcome.no_metals
    result.metal_site_limit_exceeded = outcome.metal_site_limit_exceeded
    result.confidence_inputs_missing_reason = outcome.confidence_inputs_missing_reason


def process(pdb_id: str) -> EntryResult:
    """Run one entry in an initialized worker and return its result."""
    try:
        return _process_entry(pdb_id)
    finally:
        # Release allocations after the analysis frame exits; ignore housekeeping errors.
        with contextlib.suppress(Exception):
            release_idle_memory()


def _process_entry(pdb_id: str) -> EntryResult:
    """Run the analysis in a separate frame so its allocations can be released."""
    cfg = worker_config
    if cfg is None:
        raise RuntimeError("worker configuration has not been initialized")
    t0 = time.monotonic()
    # Remove only scratch created here; predictable entry paths may contain user data.
    work_dir: str | None = None
    manual_inputs = cfg.manual_inputs
    result = initial_result(pdb_id, cfg, manual_inputs)
    announce_inflight("start", pdb_id)
    try:
        if manual_inputs:
            work_dir = create_owned_scratch_directory(
                cfg.output_dir,
                prefix=f".alchemy-{pdb_id}-",
                kind="entry",
                preserve=cfg.keep_intermediates,
            )
            entry = work_dir
        else:
            entry = _resolve_entry_dir(pdb_id, cfg)
            if not os.path.isdir(entry):
                result.status = EntryStatus.SKIP
                result.status_detail = "entry dir missing"
                return result
            work_dir = create_owned_scratch_directory(
                cfg.output_dir,
                prefix=f".alchemy-{pdb_id}-",
                kind="entry",
                preserve=cfg.keep_intermediates,
            )
        inputs, structure, provenance = _prepare_analysis_inputs(
            pdb_id, cfg, entry, work_dir
        )
        result.timings["input_structure_s"] = round(time.monotonic() - t0, 3)
        _apply_input_provenance(result, provenance)
        result.warning_codes = list(structure.warning_codes)
        crystallization = extract_crystallization_context(
            pdb_id,
            inputs.source_coordinate_path,
            cfg.pdb_metadata_cache,
            prefer_coordinate_file=manual_inputs is not None,
        )
        result.crystallization_condition_rows = list(crystallization.conditions)
        result.crystallization_summary_row = crystallization.summary
        if _excluded_zero_occupancy_metals(structure):
            result.warning_codes = list(
                dict.fromkeys(
                    result.warning_codes + [WarningCode.ZERO_OCCUPANCY_METAL_EXCLUDED]
                )
            )
        early = _early_outcome(structure)
        if early is not None:
            _apply_early_outcome(result, early)
            return result
        density = _run_density_stage(pdb_id, cfg, inputs, structure)
        _apply_density_outcome(result, density)
        identification_reason_codes = _identification_reason_codes(density.rows)
        bond = run_bond_stage(
            pdb_id, cfg, inputs, structure, density.rows, density.header
        )
        _apply_bond_outcome(result, bond)
        # An empty header already has a density-failure code; check completeness
        # only for a successfully produced table.
        sites_without_density = (
            _sites_without_density_rows(density.rows, structure)
            if density.header
            else []
        )
        if sites_without_density:
            identification_reason_codes = list(
                dict.fromkeys(
                    identification_reason_codes
                    + [ReasonCode.METAL_SITE_WITHOUT_DENSITY]
                )
            )
            logger.debug(
                "%s: %d selected coordinate metal site(s) are absent from the "
                "statistics table: %s",
                pdb_id,
                len(sites_without_density),
                sites_without_density,
            )
        append_site_fields(density.rows, bond.analysis.site_summaries, structure)
        result.rows = density.rows
        result.bond_rows = bond.analysis.bond_rows
        result.candidate_rows = bond.analysis.candidate_rows
        _finalize_result(
            result, identification_reason_codes, bond.analysis.metadata, structure
        )
    except FileNotFoundError as e:
        result.status = EntryStatus.SKIP
        result.reason_codes = [ReasonCode.MISSING_INPUT]
        result.status_detail = truncate(
            f"missing input: {e}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
    except Exception as e:  # one bad entry must not kill the batch
        deterministic = isinstance(e, DETERMINISTIC_PROCESSING_ERRORS)
        result.status = EntryStatus.ERROR
        result.reason_codes = [
            ReasonCode.DETERMINISTIC_PROCESSING_ERROR
            if deterministic
            else ReasonCode.UNEXPECTED_PROCESSING_ERROR
        ]
        result.status_detail = truncate(
            f"{type(e).__name__}: {e}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
        # Keep the traceback in the debug log; the manifest holds only a short summary.
        logger.debug(
            "%s: %s ended the entry (%s)",
            pdb_id,
            type(e).__name__,
            "terminal" if deterministic else "retryable",
            exc_info=True,
        )
    finally:
        if (
            not cfg.keep_intermediates
            and work_dir is not None
            and os.path.isdir(work_dir)
        ):
            cleanup_started = time.monotonic()
            shutil.rmtree(work_dir, ignore_errors=True)
            result.timings["cleanup_s"] = round(time.monotonic() - cleanup_started, 3)
        result.retryable = retryable_for(result.status, result.reason_codes)
        result.runtime_s = round(time.monotonic() - t0, 3)
        announce_inflight("end", pdb_id)
    return result
