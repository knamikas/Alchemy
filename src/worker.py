"""Process one PDB entry and record its results and failures.

Configuration is initialized once per worker to avoid serializing it per entry.
Input resolution lives in ``worker_inputs`` and the analysis stages in
``worker_stages``; this module owns the process lifecycle, the entry's scratch
directory, and the folds that combine stage outcomes into an ``EntryResult``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from multiprocessing.queues import Queue, SimpleQueue
from typing import Literal

from codes import EntryStatus, ReasonCode, WarningCode
from coordination.analysis import BondAnalysisMetadata
from crystallization_conditions import extract_crystallization_context
from density_analysis import Ccp4EntryLimitationError, elapsed_s
from inputs import MissingInputError
from run_logging import configure_worker_logging, logger_for, truncate
from scratch import create_owned_scratch_directory
from structure_analysis import AtomSite
from worker_contracts import (
    EntryResult,
    InflightEvent,
    PdbRedoProvenance,
    SoftwareProvenance,
    WorkerConfig,
)
from worker_inputs import (
    InputProvenance,
    prepare_analysis_inputs,
    resolve_entry_dir,
)
from worker_memory import release_idle_memory
from worker_stages import (
    IDENTIFICATION_REASON_MESSAGES,
    MAX_MANIFEST_STATUS_DETAIL_CHARS,
    METALS_SET,
    BondOutcome,
    DensityOutcome,
    EarlyOutcome,
    append_site_fields,
    early_outcome,
    excluded_zero_occupancy_metals,
    identification_reason_codes,
    merged_codes,
    run_bond_stage,
    run_density_stage,
    sites_without_density_rows,
)

# The per-entry seams callers reach through ``worker`` besides the lifecycle
# functions defined below.
__all__ = [
    "DETERMINISTIC_PROCESSING_ERRORS",
    "RETRYABLE_PARTIAL_REASON_CODES",
    "announce_inflight",
    "initial_result",
    "initialize_worker",
    "is_worker_death_result",
    "process",
    "retryable_for",
    "worker_config",
    "worker_death_result",
]

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


def _apply_density_outcome(result: EntryResult, outcome: DensityOutcome) -> None:
    """Fold the density stage into the entry result."""
    result.timings.update(outcome.timings)
    result.warning_codes = merged_codes(result.warning_codes, outcome.warning_codes)
    result.density_context_row = outcome.density_context_row
    result.density = outcome.provenance
    if outcome.failed:
        result.reason_codes = merged_codes(result.reason_codes, [outcome.reason_code])
        result.status_detail = outcome.status_detail
        result.confidence_inputs_missing_reason = outcome.reason_code


def _apply_bond_outcome(result: EntryResult, outcome: BondOutcome) -> None:
    """Fold the bond stage into the entry result."""
    if outcome.failed:
        result.status_detail = outcome.status_detail
        result.reason_codes = merged_codes(
            result.reason_codes, [ReasonCode.BOND_STAGE_FAILURE]
        )
        result.confidence_inputs_missing_reason = ReasonCode.BOND_STAGE_FAILURE
    result.timings.update(outcome.timings)


def _finalize_result(
    result: EntryResult,
    identification_codes: list[ReasonCode],
    bond_meta: BondAnalysisMetadata,
    selected_metals: Sequence[AtomSite],
) -> None:
    """Combine stage outcomes into the entry status, reason codes, and counts."""
    result.reason_codes = merged_codes(
        result.reason_codes,
        [*identification_codes, *bond_meta.partial_reason_codes],
    )
    messages = [IDENTIFICATION_REASON_MESSAGES[code] for code in identification_codes]
    messages.extend(bond_meta.messages)
    if messages:
        existing_detail = result.status_detail
        result.status_detail = truncate(
            "; ".join(([existing_detail] if existing_detail else []) + messages),
            MAX_MANIFEST_STATUS_DETAIL_CHARS,
        )
    result.warning_codes = merged_codes(result.warning_codes, bond_meta.warning_codes)
    result.status = EntryStatus.PARTIAL if result.reason_codes else EntryStatus.OK
    # Count coordinate sites even when their EDSTATS joins failed.
    result.n_metals = len(selected_metals)
    result.n_bonds = len(result.bond_rows)
    result.n_candidates = len(result.candidate_rows)


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
        # Manual inputs have no entry directory; the scratch directory stands in.
        entry_dir = None if manual_inputs else resolve_entry_dir(pdb_id, cfg)
        if entry_dir is not None and not os.path.isdir(entry_dir):
            result.status = EntryStatus.SKIP
            result.status_detail = "entry dir missing"
            return result
        work_dir = create_owned_scratch_directory(
            cfg.output_dir,
            prefix=f".alchemy-{pdb_id}-",
            kind="entry",
            preserve=cfg.keep_intermediates,
        )
        inputs, structure, provenance = prepare_analysis_inputs(
            pdb_id, cfg, work_dir if entry_dir is None else entry_dir, work_dir
        )
        result.timings["input_structure_s"] = elapsed_s(t0)
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
        selected_metals = structure.metal_atoms(METALS_SET, canonical=True)
        if excluded_zero_occupancy_metals(structure, selected_metals):
            result.warning_codes = merged_codes(
                result.warning_codes, [WarningCode.ZERO_OCCUPANCY_METAL_EXCLUDED]
            )
        early = early_outcome(structure, selected_metals)
        if early is not None:
            _apply_early_outcome(result, early)
            return result
        density = run_density_stage(pdb_id, cfg, inputs, structure)
        _apply_density_outcome(result, density)
        identification_codes = identification_reason_codes(density.rows)
        bond = run_bond_stage(
            pdb_id,
            cfg,
            inputs,
            structure,
            selected_metals,
            density.rows,
            density.header,
        )
        _apply_bond_outcome(result, bond)
        # An empty header already has a density-failure code; check completeness
        # only for a successfully produced table.
        sites_without_density = (
            sites_without_density_rows(density.rows, selected_metals)
            if density.header
            else []
        )
        if sites_without_density:
            identification_codes = merged_codes(
                identification_codes, [ReasonCode.METAL_SITE_WITHOUT_DENSITY]
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
            result, identification_codes, bond.analysis.metadata, selected_metals
        )
    except MissingInputError as e:
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
            result.timings["cleanup_s"] = elapsed_s(cleanup_started)
        result.retryable = retryable_for(result.status, result.reason_codes)
        result.runtime_s = elapsed_s(t0)
        announce_inflight("end", pdb_id)
    return result
