"""Running one PDB-REDO entry, from the worker process that owns it.

``process`` never raises: one bad entry must not take the batch down, so every
failure is recorded in the result as a status, a reason code and a bounded
message. Configuration reaches a worker once through ``initialize_worker`` and is
held in ``worker_config``, because a pool initializer cannot return a value and passing
the config with every task would pickle it per entry.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import signal
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue, SimpleQueue
from typing import Any
from collections.abc import Mapping

from coordination.analysis import (
    BondAnalysisMetadata,
    BondAnalysisResult,
    run_bond_analysis,
)
from codes import EntryStatus, ReasonCode, WarningCode
from coordination.schema import (
    STATS_EXTRA_COLUMNS,
    check_row_schema,
    stats_extra_values,
)
from crystallization_conditions import extract_crystallization_context
from coordinate_conversion import first_model_pdb
from density_analysis import (
    Ccp4EntryLimitationError,
    Ccp4ToolTimeoutError,
    MtzfixValidationError,
    run_density_analysis,
)
from driver.output_lock import create_owned_scratch_directory
from inputs import (
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
from structure_analysis import NAN, StructureContext, load_structure
from worker_contracts import MAX_ANALYZED_METAL_SITES, EntryResult, WorkerConfig


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

# Unanticipated exceptions that will recur on the same inputs: a parse, lookup,
# or type error is a property of the entry's data or of Alchemy's code, not of
# the machine. The distinction is advisory -- it separates "this will keep
# failing until something changes" from "this may have been the machine", which
# is what triage of a database-scale run needs. It deliberately does not change
# what --resume does: an entry is retried either way, because nothing here can
# establish that a later run reads the same bytes.
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
_inflight_queue: SimpleQueue[tuple[str, int, str]] | None = None


def initialize_worker(
    cfg: WorkerConfig,
    inflight: SimpleQueue[tuple[str, int, str]] | None = None,
    log_queue: Queue[logging.LogRecord] | None = None,
) -> None:
    global worker_config, _inflight_queue
    worker_config = cfg
    _inflight_queue = inflight
    # Give this worker and every CCP4 descendant a group the driver can kill as
    # one unit. If the worker itself is SIGKILLed, its active external program
    # otherwise survives as an orphan and keeps consuming resources/writing
    # scratch. Windows has no ``setpgrp``; Process.kill remains the fallback.
    with contextlib.suppress(AttributeError, OSError):
        os.setpgrp()
    # A forked worker inherits the driver's SIGTERM-to-KeyboardInterrupt
    # handler, and raising there during ``pool.terminate`` interrupts whatever
    # the worker is doing, including the log queue's feeder-thread finalizer.
    with contextlib.suppress(AttributeError, OSError, ValueError):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    configure_worker_logging(log_queue, level=cfg.log_level)


def announce_inflight(state: str, pdb_id: str) -> None:
    """Tell the driver which entry this worker process currently holds.

    A worker killed by the OOM killer or a segfault runs no further Python and
    delivers no result, so these notifications are the driver's only record of
    which entry to fail retryably. ``SimpleQueue.put`` writes to the pipe
    synchronously, so a notification sent before the work begins is already
    readable when the process dies mid-entry.
    """
    if _inflight_queue is None:
        return
    try:
        _inflight_queue.put((state, os.getpid(), pdb_id))
    except Exception:  # noqa: BLE001 - bookkeeping must never fail an entry
        pass


def _preserve_timeout_log(
    timeout: Ccp4ToolTimeoutError, pdb_id: str, output_dir: str
) -> str:
    """Copy a timed-out program's partial log out of the scratch directory.

    Only the log: the maps beside it run to hundreds of megabytes per entry.
    Returns ``""`` when there was nothing to copy, and never raises, because
    failing to keep a diagnostic must not change the entry's outcome.
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
        alchemy_commit=cfg.alchemy_commit,
        gemmi_version=cfg.gemmi_version,
        ccp4_version=cfg.ccp4_version,
        reference_data_id=cfg.reference_data_id,
        analysis_config_id=cfg.analysis_config_id,
        refinement_state="manual" if manual_inputs else "final",
    )


def worker_death_result(pdb_id: str, cfg: WorkerConfig, pid: int) -> EntryResult:
    """Synthesize the retryable result a killed worker could not return."""
    result = initial_result(pdb_id, cfg, cfg.manual_inputs)
    result.status = EntryStatus.ERROR
    result.retryable = True
    result.reason_codes = [ReasonCode.WORKER_PROCESS_DIED]
    result.error = (
        f"worker process {pid} terminated without returning a result "
        f"(out-of-memory kill or crash); {pdb_id} was not analyzed"
    )
    return result


def _coordinate_provenance(
    cfg: WorkerConfig, source_path: str
) -> tuple[str, str, bool]:
    manual = cfg.manual_inputs
    if manual:
        converted = bool(manual.get("cif_file"))
        return ("mmcif" if converted else "pdb", "pdb", converted)
    coordinate_name = source_path.lower()
    converted = coordinate_name.endswith((".cif", ".cif.gz"))
    return ("mmcif" if converted else "pdb", "pdb", converted)


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
        used_root = ensure_entry_available(pdb_id, cfg.mirror_root, cfg.cache_root)
        return entry_dir_for(used_root, pdb_id)
    return entry_dir_for(cfg.root, pdb_id)


@dataclass(frozen=True)
class EntryInputs:
    """What the input stage resolved, and the later stages then read.

    Held together rather than threaded through each stage separately, so that
    the density and bond stages take one entry-shaped argument instead of a
    handful of loose paths and resolution limits.
    """

    work_dir: str
    mtz: str
    # First-model-only coordinates: what the analysis reads, which is not
    # necessarily what was deposited (see ``source_coordinate_path``).
    pdb: str
    # PDB-REDO's per-entry metadata, or a manual run's ``--data-json``, and
    # ``None`` when a manual run supplied none.
    data_json: str | None
    # The diffraction data's own high-resolution limit, as distinct from the
    # map columns' range below.
    data_reshi: float
    map_reslo: float
    map_reshi: float
    pdb_redo_is_twin: bool
    source_coordinate_path: str


def _run_density_stage(
    result: EntryResult,
    cfg: WorkerConfig,
    inputs: EntryInputs,
    structure: StructureContext,
) -> tuple[list[MetalStatsRow], list[str]]:
    """Calculate the maps, run EDSTATS, and extract this entry's statistics.

    Returns ``(rows, header)``, both empty when density could not be produced;
    the reason is recorded on ``result`` in that case, along with whether it is
    worth retrying.
    """
    rows: list[MetalStatsRow] = []
    header: list[str] = []
    density_started = time.monotonic()
    try:
        res = run_density_analysis(
            result.pdb_id,
            inputs.mtz,
            inputs.pdb,
            inputs.work_dir,
            inputs.map_reslo,
            inputs.map_reshi,
            env=cfg.env,
            map_scope=cfg.density_map_scope,
            keep_full_maps=cfg.keep,
            pdb_redo_is_twin=inputs.pdb_redo_is_twin,
            tool_timeout_s=cfg.ccp4_timeout_s,
        )
    except Ccp4ToolTimeoutError as exc:
        # A killed program says nothing about the entry, so unlike a
        # failure exit this is retryable, under its own reason code so a
        # stalled CCP4 install is visible in the manifest.
        rows, header = [], []
        kept_log = _preserve_timeout_log(exc, result.pdb_id, cfg.output_dir)
        result.retryable = True
        result.reason_codes = [ReasonCode.CCP4_TOOL_TIMEOUT]
        result.error = truncate(
            f"density unavailable: {exc}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
        result.confidence_inputs_missing_reason = ReasonCode.CCP4_TOOL_TIMEOUT
        result.ccp4_timeout_log_path = kept_log
        result.timings.update(exc.timings)
    except MtzfixValidationError as exc:
        # MTZFIX could not make the Fourier coefficients internally
        # consistent, which retrying will not change; geometry stays
        # independently assessable.
        rows, header = [], []
        result.retryable = False
        result.reason_codes = [ReasonCode.MTZFIX_VALIDATION_FAILURE]
        result.error = truncate(
            f"density unavailable: {exc}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
        result.confidence_inputs_missing_reason = ReasonCode.MTZFIX_VALIDATION_FAILURE
        result.timings.update(exc.timings)
    else:
        result.timings.update(res.timings)
        result.density_map_scope_used = res.density_map_scope_used
        result.density_full_map_bytes = res.full_map_bytes
        result.density_edstats_map_bytes = res.edstats_map_bytes
        if res.twin_coefficient_normalization_applied:
            result.warning_codes = list(
                dict.fromkeys(
                    result.warning_codes
                    + [WarningCode.TWIN_REFMAC_COEFFICIENTS_NORMALIZED]
                )
            )
        statistics_started = time.monotonic()
        rows, header = extract_metal_statistics(
            result.pdb_id,
            res.stats_out,
            METALS_SET,
            cfg.cofactors,
            structure=structure,
            density_context_out=result.density_context_row,
            warning_codes_out=result.warning_codes,
        )
        result.timings["statistics_extraction_s"] = round(
            time.monotonic() - statistics_started, 3
        )
        # Inputs and density succeeded; later limitations are terminal.
        result.retryable = False
    finally:
        result.timings["density_total_s"] = round(time.monotonic() - density_started, 3)
    return rows, header


def _identification_reason_codes(rows: list[MetalStatsRow]) -> list[ReasonCode]:
    """Deduplicated reason codes for EDSTATS rows that could not be joined."""
    codes: list[ReasonCode] = []
    for row in rows:
        mapping_status = row.coordinate_mapping_status
        site_status = row.selected_metal_site_status
        if mapping_status == "coordinate_residue_not_found":
            codes.append(ReasonCode.COFACTOR_COORDINATE_JOIN_FAILED)
        elif mapping_status == "multiple_coordinate_residues":
            codes.append(ReasonCode.AMBIGUOUS_COORDINATE_RESIDUE_JOIN)
        elif site_status == "no_selected_metal":
            codes.append(ReasonCode.COFACTOR_WITHOUT_SELECTED_METAL)
    return list(dict.fromkeys(codes))


def _excluded_zero_occupancy_metals(structure: StructureContext) -> int:
    """Metal records dropped from site selection for a valid zero occupancy.

    Site selection excludes them deliberately -- a metal modeled as absent is
    not evidence for a site -- but the exclusion must not be silent: a
    structure whose only metal is at zero occupancy would otherwise report
    ``no_metals``, an authoritative negative about a file that plainly contains
    a metal record.
    """
    selected = structure.metal_atoms(METALS_SET, canonical=True)
    with_absent = structure.metal_atoms(
        METALS_SET, canonical=True, include_zero_occupancy=True
    )
    return len(with_absent) - len(selected)


def _sites_without_density_rows(
    rows: list[MetalStatsRow], structure: StructureContext
) -> list[tuple[int, int, int, int]]:
    """Selected coordinate metal sites that the statistics table omits.

    Coordinate analysis selects every canonical metal atom by element, while
    statistics reports a residue only when it is a catalog cofactor or a
    single-atom ion. A metal inside a multi-atom residue outside the bundled
    catalog therefore has no density row. Deriving the expected keys directly
    from the structure keeps this completeness check active when bond analysis
    is disabled or fails, and covers any later divergence between selections.
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
    result: EntryResult,
    cfg: WorkerConfig,
    inputs: EntryInputs,
    structure: StructureContext,
    rows: list[MetalStatsRow],
    header: list[str],
) -> BondAnalysisResult:
    """Evaluate the contacts around each site, unless ``--no-bonds`` cleared it.

    Returns ``(bond_rows, candidate_rows, site_summaries, bond_meta)``. A
    bond-stage failure must not lose the EDSTATS rows already computed, so it
    is recorded on ``result`` and the empty defaults are returned instead.
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
        return analysis

    bond_started = time.monotonic()
    try:
        analysis = run_bond_analysis(
            result.pdb_id,
            inputs.pdb,
            rows,
            header,
            {
                "data_json": inputs.data_json,
                "pdb_path": inputs.pdb,
                "mtz_path": inputs.mtz,
                "resolution": inputs.data_reshi,
            },
            structure=structure,
            connection_path=inputs.source_coordinate_path,
        )
    except Exception as e:  # noqa: BLE001
        result.error = truncate(
            f"bond: {type(e).__name__}: {e}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
        result.reason_codes = list(
            dict.fromkeys(result.reason_codes + [ReasonCode.BOND_STAGE_FAILURE])
        )
        result.confidence_inputs_missing_reason = ReasonCode.BOND_STAGE_FAILURE
        result.retryable = True
    finally:
        result.timings["bond_analysis_s"] = round(time.monotonic() - bond_started, 3)
    return analysis


def _finalize_result(
    result: EntryResult,
    identification_codes: list[ReasonCode],
    bond_meta: BondAnalysisMetadata,
    structure: StructureContext,
) -> None:
    """Merge the stage outcomes into the final status, codes, and counts.

    The rows each stage produced are already on ``result``; only what no stage
    could record there is passed in.
    """
    result.reason_codes = list(
        dict.fromkeys(
            result.reason_codes + identification_codes + bond_meta.partial_reason_codes
        )
    )
    messages = [IDENTIFICATION_REASON_MESSAGES[code] for code in identification_codes]
    messages.extend(bond_meta.messages)
    if messages:
        existing_error = result.error
        result.error = "; ".join(
            ([existing_error] if existing_error else []) + messages
        )
        result.error = truncate(result.error, MAX_MANIFEST_STATUS_DETAIL_CHARS)
    if bond_meta.retryable:
        result.retryable = True
    result.warning_codes = list(
        dict.fromkeys(result.warning_codes + bond_meta.warning_codes)
    )
    status = EntryStatus.PARTIAL if result.reason_codes else EntryStatus.OK
    if status == EntryStatus.OK:
        result.retryable = False
    result.status = status
    # Coordinate-model metal sites, not emitted statistics rows: a failed
    # EDSTATS join leaves a diagnostic row without a site even where bond
    # analysis found and evaluated the deposited metal.
    result.n_metals = len(structure.metal_atoms(METALS_SET, canonical=True))
    result.n_bonds = len(result.bond_rows)
    result.n_candidates = len(result.candidate_rows)


def _prepare_analysis_inputs(
    pdb_id: str,
    cfg: WorkerConfig,
    result: EntryResult,
    entry: str,
    work_dir: str,
) -> tuple[EntryInputs, StructureContext]:
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
    source_format, analysis_format, converted = _coordinate_provenance(
        cfg, source_coordinate_path
    )
    model1_pdb = os.path.join(work_dir, f"{pdb_id}_model1.pdb")
    if os.path.realpath(model1_pdb) == os.path.realpath(source_pdb):
        model1_pdb = os.path.join(work_dir, f"{pdb_id}_analysis_model1.pdb")
    pdb, input_model_count = first_model_pdb(source_pdb, model1_pdb)
    result.source_coordinate_format = source_format
    result.analysis_coordinate_format = analysis_format
    result.coordinate_conversion_performed = converted
    result.pdb_redo_is_twin = pdb_redo_metadata.is_twin
    result.pdb_redo_version = pdb_redo_metadata.version
    result.pdb_redo_date = pdb_redo_metadata.date
    result.source_coordinate_path = source_coordinate_provenance_path(
        cfg, pdb_id, source_coordinate_path
    )
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
    return inputs, structure


def _finish_if_no_analyzable_metals(
    result: EntryResult, structure: StructureContext
) -> bool:
    """Finish early only when density and contact stages cannot produce output."""
    if structure.metal_atoms(METALS_SET, canonical=True):
        return False
    if structure.unknown_element_atom_count:
        # A missing or invalid deposited element could belong to a configured
        # metal. Under the no-inference policy, zero recognized sites is
        # therefore not proof of metal absence.
        unknown_count = structure.unknown_element_atom_count
        result.status = EntryStatus.PARTIAL
        result.retryable = False
        result.reason_codes = [ReasonCode.METAL_PRESENCE_INDETERMINATE]
        result.error = truncate(
            "cannot establish metal absence: "
            f"{unknown_count} atom(s) have missing or invalid element symbols",
            MAX_MANIFEST_STATUS_DETAIL_CHARS,
        )
        result.n_metals = 0
        result.confidence_inputs_missing_reason = (
            ReasonCode.METAL_PRESENCE_INDETERMINATE
        )
        return True

    # Neither density nor contact analysis can produce output without a
    # canonical metal site, so avoid running two FFTs and EDSTATS.
    result.status = EntryStatus.OK
    result.retryable = False
    result.n_metals = 0
    result.rows = []
    result.bond_rows = []
    result.candidate_rows = []
    result.n_bonds = 0
    result.n_candidates = 0
    result.no_metals = True
    return True


def _finish_if_metal_site_limit_exceeded(
    result: EntryResult, structure: StructureContext
) -> bool:
    selected_count = len(structure.metal_atoms(METALS_SET, canonical=True))
    if selected_count <= MAX_ANALYZED_METAL_SITES:
        return False

    result.status = EntryStatus.OK
    result.retryable = False
    result.n_metals = selected_count
    result.rows = []
    result.bond_rows = []
    result.candidate_rows = []
    result.n_bonds = 0
    result.n_candidates = 0
    result.reason_codes = [ReasonCode.METAL_SITE_LIMIT_EXCEEDED]
    result.metal_site_limit_exceeded = True
    return True


def process(pdb_id: str) -> EntryResult:
    """Run one entry in an initialized worker and return its result."""
    cfg = worker_config
    if cfg is None:
        raise RuntimeError("worker configuration has not been initialized")
    t0 = time.monotonic()
    # Only a directory created by this invocation may be removed in ``finally``:
    # a predictable <output-dir>/<pdbID> path could already hold user data.
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
                preserve=cfg.keep,
            )
            entry = work_dir
        else:
            # Resolved before any scratch space exists, so a missing entry
            # leaves no temporary directory behind.
            entry = _resolve_entry_dir(pdb_id, cfg)
            if not os.path.isdir(entry):
                result.status = EntryStatus.SKIP
                result.error = "entry dir missing"
                return result
            work_dir = create_owned_scratch_directory(
                cfg.output_dir,
                prefix=f".alchemy-{pdb_id}-",
                kind="entry",
                preserve=cfg.keep,
            )
        inputs, structure = _prepare_analysis_inputs(
            pdb_id, cfg, result, entry, work_dir
        )
        result.timings["input_structure_s"] = round(time.monotonic() - t0, 3)
        result.analysis_coordinate_format = structure.analysis_coordinate_format
        result.input_model_count = structure.input_model_count
        result.model_analyzed = structure.model_analyzed
        result.multi_model_structure = structure.multi_model_structure
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
        if _finish_if_metal_site_limit_exceeded(result, structure):
            return result
        if _finish_if_no_analyzable_metals(result, structure):
            return result
        rows, header = _run_density_stage(result, cfg, inputs, structure)
        identification_reason_codes = _identification_reason_codes(rows)
        bond_analysis = run_bond_stage(result, cfg, inputs, structure, rows, header)
        # An empty header means density production itself failed and already
        # supplied the precise reason code. This comparison diagnoses a
        # successful statistics table whose site selection is incomplete.
        sites_without_density = (
            _sites_without_density_rows(rows, structure) if header else []
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
        append_site_fields(rows, bond_analysis.site_summaries, structure)
        result.rows = rows
        result.bond_rows = bond_analysis.bond_rows
        result.candidate_rows = bond_analysis.candidate_rows
        _finalize_result(
            result, identification_reason_codes, bond_analysis.metadata, structure
        )
    except FileNotFoundError as e:
        result.status = EntryStatus.SKIP
        result.retryable = True
        result.reason_codes = [ReasonCode.MISSING_INPUT]
        result.error = truncate(f"missing input: {e}", MAX_MANIFEST_STATUS_DETAIL_CHARS)
    except Exception as e:  # noqa: BLE001 - one bad entry must not kill the batch
        deterministic = isinstance(e, DETERMINISTIC_PROCESSING_ERRORS)
        result.status = EntryStatus.ERROR
        # Retryable regardless: the reason code says the failure will recur on
        # the same inputs, but a resumed run cannot know the inputs are the
        # same. A manual run may name a repaired file, and a mirror entry may
        # have been re-downloaded, so skipping the entry could silently drop
        # work the operator has already fixed.
        result.retryable = True
        result.reason_codes = [
            ReasonCode.DETERMINISTIC_PROCESSING_ERROR
            if deterministic
            else ReasonCode.UNEXPECTED_PROCESSING_ERROR
        ]
        result.error = truncate(
            f"{type(e).__name__}: {e}", MAX_MANIFEST_STATUS_DETAIL_CHARS
        )
        # The manifest keeps one truncated line, which names the exception but
        # not where it came from. Without this the only way to locate an
        # unanticipated failure is to rerun the entry by hand with
        # --keep-intermediates, so the traceback goes to the debug log that
        # --log-file and -v already collect.
        logger.debug(
            "%s: %s ended the entry (%s)",
            pdb_id,
            type(e).__name__,
            "terminal" if deterministic else "retryable",
            exc_info=True,
        )
    finally:
        if not cfg.keep and work_dir is not None and os.path.isdir(work_dir):
            cleanup_started = time.monotonic()
            shutil.rmtree(work_dir, ignore_errors=True)
            result.timings["cleanup_s"] = round(time.monotonic() - cleanup_started, 3)
        result.runtime_s = round(time.monotonic() - t0, 3)
        announce_inflight("end", pdb_id)
    return result
