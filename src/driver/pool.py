"""Manage a batch of entry analyses from setup to final results.

Prepare the environment, select entries, and check whether a previous run
can be resumed. Open the outputs, hand the batch to the dispatcher, and
write results, confidence scores, and the final report.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Collection, Sequence
from typing import NamedTuple

from analysis_config import analysis_config_id, analysis_configs_are_compatible
from confidence_score import CONFIDENCE_INPUT_COLUMNS
from crystallization_conditions import (
    CONDITION_COLUMNS,
    SUMMARY_COLUMNS,
    CrystallizationMetadataError,
    prefetch_rcsb_crystallization_metadata,
)
from driver import confidence, dispatch, environment
from driver.entries import schedule_entries
from driver.errors import DriverError
from driver.layout import OutputLayout, prepare_output_directory
from driver.output_lock import (
    OutputDirectoryBusyError,
    OutputDirectoryLock,
    OutputDirectoryLockError,
)
from driver.report import report_batch
from driver.resources import (
    MemoryPlan,
    available_memory_bytes,
    estimate_entry_memory,
    scheduling_memory_budget,
    worker_limits_for_budget,
)
from driver.resume import (
    ResumeStaging,
    manifest_values_by_id,
    remove_stale_disabled_bond_outputs,
    resume_replacement_succeeded,
    validate_resume_schemas,
)
from driver.runlog import RunLog
from driver.writers import OutputTargets, OutputWriters, manifest_row
from metal_identification import DENSITY_CONTEXT_COLUMNS
from reference_data import (
    cofactor_ids,
    reference_data_checksums,
    reference_data_id,
)
from run_config import RunConfig
from run_logging import logger_for, worker_level
from scratch import sweep_owned_scratch_directories
from worker_contracts import EntryResult, WorkerConfig

logger = logger_for(__name__)


def plan_entry_memory(
    args: RunConfig, ids: Sequence[str], cfg: WorkerConfig, run_log: RunLog
) -> MemoryPlan:
    """Estimate entry memory and record the resulting admission budget."""
    if not ids:
        raise ValueError("memory planning needs at least one entry")
    estimates = tuple(
        estimate_entry_memory(pdb_id, cfg.input_root, cfg.manual_inputs)
        for pdb_id in ids
    )
    available = available_memory_bytes()
    budget, reserve = scheduling_memory_budget(
        available,
        memory_limit_bytes=args.memory_limit,
        utilization=args.memory_utilization,
    )
    source_counts: dict[str, int] = {}
    for estimate in estimates:
        source_counts[estimate.source] = source_counts.get(estimate.source, 0) + 1
    largest = max(estimates, key=lambda estimate: estimate.bytes)
    high_memory_entries = sum(estimate.is_high_memory for estimate in estimates)

    run_log.details.update(
        memory_scheduler="adaptive_weighted_entry_estimates",
        memory_scheduler_detected_available_bytes=(
            available if available is not None else "unavailable"
        ),
        memory_scheduler_configured_limit_bytes=(
            args.memory_limit if args.memory_limit is not None else "automatic"
        ),
        memory_scheduler_utilization=args.memory_utilization,
        memory_scheduler_budget_bytes=budget if budget is not None else "unavailable",
        memory_scheduler_reserve_bytes=(
            reserve if reserve is not None else "unavailable"
        ),
        memory_estimate_sources=source_counts,
        memory_estimate_max_bytes=largest.bytes,
        memory_estimate_max_entry=largest.pdb_id,
        memory_high_memory_entries=high_memory_entries,
    )
    if budget is None or reserve is None:
        logger.warning(
            "available memory could not be measured; per-entry estimates were "
            "computed, but weighted admission has no byte budget"
        )
    else:
        logger.info(
            "memory-aware scheduling: %.2f GiB worker budget, %.2f GiB "
            "protected reserve; largest estimate %.2f GiB (%s); all entries "
            "share the byte budget",
            budget / (1024**3),
            reserve / (1024**3),
            largest.bytes / (1024**3),
            largest.pdb_id,
        )
    logger.info(
        "memory estimates: %s",
        ", ".join(
            f"{count} {source}" for source, count in sorted(source_counts.items())
        ),
    )
    return MemoryPlan(
        estimates,
        budget,
        reserve,
        initial_available_bytes=available,
        configured_limit_bytes=args.memory_limit,
    )


def _load_cofactor_catalog() -> frozenset[str]:
    """Read the bundled metallocofactor catalog, or fail the run naming it."""
    try:
        return cofactor_ids()
    except (OSError, UnicodeError, ValueError) as exc:
        raise DriverError(f"Invalid bundled metallocofactor catalog: {exc}") from None


def _check_resume_is_compatible(
    args: RunConfig,
    layout: OutputLayout,
    plan: confidence.ConfidencePlan,
    current_analysis_config_id: str,
) -> None:
    """Refuse to resume onto output this run cannot safely extend."""
    if not args.resume:
        return
    try:
        validate_resume_schemas(
            layout,
            bonds_enabled=args.bonds,
            confidence_path=plan.stream_path,
            confidence_columns=plan.columns,
            additional_outputs=(
                (layout.density_context, DENSITY_CONTEXT_COLUMNS),
                (layout.crystallization_conditions, CONDITION_COLUMNS),
                (layout.crystallization_summary, SUMMARY_COLUMNS),
            ),
        )
        if plan.synchronize_inputs:
            validate_resume_schemas(
                layout,
                bonds_enabled=args.bonds,
                confidence_path=layout.confidence_inputs,
                confidence_columns=CONFIDENCE_INPUT_COLUMNS,
            )
    except ValueError as exc:
        raise DriverError(str(exc)) from None
    prior_config_values = manifest_values_by_id(layout.manifest, "analysis_config_id")
    if not analysis_configs_are_compatible(
        prior_config_values.values(), current_analysis_config_id
    ):
        raise DriverError(
            "Cannot resume output produced with a different analysis "
            "configuration identity; use a fresh output directory."
        )
    plan.validate_resumed_output()


def _finish_without_entries(
    args: RunConfig,
    layout: OutputLayout,
    plan: confidence.ConfidencePlan,
    run_log: RunLog,
) -> int:
    """Exit code for a run whose work list came back empty."""
    finalized = plan.finalize_resumed_inputs(layout, run_log, resume=args.resume)
    if finalized is not None:
        logger.info(
            "no entries required retry; finalized %d confidence rows "
            "(%d scored; database cohort %d) -> %s",
            finalized.rows,
            finalized.scored_rows,
            finalized.cohort,
            layout.confidence_scores,
        )
        logger.info("confidence reference -> %s", layout.reference_dir)
    if plan.enabled:
        confidence.finalize_review_queue(layout, run_log)
    if finalized is None:
        logger.info("no entries to process")
    return 0


def _clear_stale_outputs(
    args: RunConfig, layout: OutputLayout, plan: confidence.ConfidencePlan
) -> None:
    """Remove stale outputs that conflict with the current run mode."""
    try:
        removed = remove_stale_disabled_bond_outputs(
            (layout.bonds, layout.candidates),
            resume=args.resume,
            bonds_enabled=args.bonds,
        )
    except OSError as exc:
        raise DriverError(f"Could not remove stale bond-stage output: {exc}") from None
    for removed_path in removed:
        logger.info("removed stale bond-stage output: %s", removed_path)
    try:
        if os.path.isfile(layout.review_queue):
            os.unlink(layout.review_queue)
    except OSError as exc:
        raise DriverError(f"Could not clear stale review queue: {exc}") from None
    if args.resume:
        return
    try:
        for path in (*plan.stale_outputs(layout), *layout.legacy_scientific_outputs):
            if os.path.isfile(path):
                os.unlink(path)
    except OSError as exc:
        raise DriverError(f"Could not clear stale confidence output: {exc}") from None


def choose_worker_count(
    args: RunConfig,
    entry_count: int,
    run_log: RunLog,
    *,
    memory_budget_bytes: int | None,
) -> int:
    """Size the pool, never above the number of entries there are to run.

    A Pool creates every worker up front, and under the spawn start method each
    one re-imports gemmi into its own interpreter. The memory budget is the one
    the entry plan measured, so the run sizes itself from a single reading.
    """
    cpu_limit, memory_limit = worker_limits_for_budget(memory_budget_bytes)
    requested = cpu_limit if args.workers is None else args.workers
    workers = min(requested, entry_count)
    if memory_limit is not None:
        workers = min(workers, memory_limit)
    memory_limit_detail = memory_limit if memory_limit is not None else "unavailable"
    if args.workers is None:
        run_log.details["worker_selection"] = "automatic"
        run_log.details["cpu_worker_limit"] = cpu_limit
        run_log.details["memory_worker_limit"] = memory_limit_detail
        run_log.details["selected_workers"] = workers
        logger.info("automatic worker selection:")
        logger.info("  CPU worker limit: %s", cpu_limit)
        logger.info("  memory worker limit: %s", memory_limit_detail)
        logger.info("  selected workers: %s", workers)
    else:
        run_log.details["worker_selection"] = "explicit"
        run_log.details["requested_workers"] = args.workers
        run_log.details["memory_worker_limit"] = memory_limit_detail
        run_log.details["selected_workers"] = workers
        if workers < min(args.workers, entry_count):
            logger.info(
                "capped the requested %d workers at %d to protect process "
                "overhead; weighted admission may activate fewer",
                args.workers,
                workers,
            )
    logger.info(
        "processing %d entr%s with %d worker(s)",
        entry_count,
        "y" if entry_count == 1 else "ies",
        workers,
    )
    return workers


class AnalysisIdentity(NamedTuple):
    """The reference-data and analysis-policy identities of this checkout."""

    reference_data_id: str
    analysis_config_id: str

    @classmethod
    def current(cls) -> AnalysisIdentity:
        """Hash the bundled reference data once and derive both identities."""
        current_reference_data_id = reference_data_id()
        return cls(
            current_reference_data_id,
            analysis_config_id(reference_data_id=current_reference_data_id),
        )


def worker_config_from_args(
    args: RunConfig,
    env: dict[str, str],
    input_root: str,
    pdb_redo_cache: str,
    cofactors: Collection[str],
    manual_inputs: dict[str, str | None] | None,
    *,
    identity: AnalysisIdentity,
) -> WorkerConfig:
    """Build the config every worker is initialized with, once per run."""
    return WorkerConfig(
        input_root=input_root,
        pdb_redo_root=args.pdb_redo_root,
        pdb_redo_cache=pdb_redo_cache,
        env=env,
        output_dir=args.output_dir,
        cofactors=cofactors,
        keep_intermediates=args.keep_intermediates,
        bonds=args.bonds,
        density_map_scope=args.density_map_scope,
        ccp4_timeout=args.ccp4_timeout,
        log_level=worker_level(args.log_level, args.log_file),
        allow_download=bool(args.id or args.id_file),
        manual_inputs=manual_inputs,
        alchemy_commit=environment.alchemy_commit(),
        gemmi_version=environment.gemmi_version(),
        ccp4_version=environment.ccp4_version(env),
        reference_data_id=identity.reference_data_id,
        analysis_config_id=identity.analysis_config_id,
        pdb_metadata_cache=args.pdb_metadata_cache,
    )


def record_run_provenance(
    run_log: RunLog, cfg: WorkerConfig, plan: confidence.ConfidencePlan
) -> None:
    """Record the software, reference data, and confidence mode the run used."""
    run_log.details.update(
        alchemy_version=environment.ALCHEMY_VERSION,
        alchemy_commit=cfg.alchemy_commit,
        gemmi_version=cfg.gemmi_version,
        ccp4_version=cfg.ccp4_version,
        confidence_mode=plan.mode or "disabled",
        reference_data_id=cfg.reference_data_id,
        analysis_config_id=cfg.analysis_config_id,
        # Retain per-file hashes to explain changes in the combined reference ID.
        **{
            f"{name.split('.')[0]}_sha256": digest
            for name, digest in reference_data_checksums().items()
        },
    )


def prepare_crystallization_metadata(
    args: RunConfig,
    ids: Sequence[str],
    run_log: RunLog,
    *,
    allow_download: bool,
) -> None:
    """Warm the original-PDB metadata cache before costly worker execution."""
    try:
        stats = prefetch_rcsb_crystallization_metadata(
            ids, args.pdb_metadata_cache, allow_download=allow_download
        )
    except CrystallizationMetadataError as exc:
        raise DriverError(
            "Could not prepare original-PDB crystallization metadata: "
            f"{exc}. Re-run with --no-crystallization-download to use only "
            "cached and coordinate-file metadata."
        ) from None
    run_log.details.update(
        crystallization_metadata_cache=args.pdb_metadata_cache,
        crystallization_metadata_requested=stats.requested,
        crystallization_metadata_cache_hits=stats.cache_hits,
        crystallization_metadata_fetched=stats.fetched,
        crystallization_metadata_available=stats.available,
        crystallization_metadata_not_reported=stats.not_reported,
        crystallization_metadata_entry_unavailable=stats.entry_unavailable,
    )
    logger.info(
        "crystallization metadata: %d cached, %d fetched, %d with conditions, "
        "%d without reported conditions, %d unavailable",
        stats.cache_hits,
        stats.fetched,
        stats.available,
        stats.not_reported,
        stats.entry_unavailable,
    )


def _open_writers(
    handles: contextlib.ExitStack,
    targets: OutputTargets,
    *,
    bonds: bool,
    confidence_columns: Sequence[str] | None,
) -> OutputWriters:
    """Open every output stream this run writes and give them their headers."""
    paths = targets.present()
    if not bonds:
        for name in OutputTargets.BOND_STAGE_OUTPUTS:
            del paths[name]
    return OutputWriters(
        {
            name: handles.enter_context(open(path, "w", newline="", encoding="utf-8"))
            for name, path in paths.items()
        },
        confidence_columns=confidence_columns,
    )


def should_write_entry(
    resuming: bool, result: EntryResult, prior_ids: set[str]
) -> bool:
    """Return whether this result should replace or add output rows.

    Protect existing results from unsuccessful retries. Always record entries
    not yet represented in the manifest.
    """
    if not resuming:
        return True
    if str(result.pdb_id).strip().lower() not in prior_ids:
        return True
    return resume_replacement_succeeded(result)


def write_entry(
    result: EntryResult,
    plan: confidence.ConfidencePlan,
    writers: OutputWriters,
    staging: ResumeStaging | None,
    prior_counts: tuple[dict[str, str], dict[str, str]],
    *,
    resume: bool,
    bonds: bool,
) -> None:
    """Write one entry's rows, manifest row last.

    The manifest row is the entry's completion marker, so an interruption
    between it and the data rows costs a repeat on the next resume, not a loss.
    """
    prior_bond_counts, prior_candidate_counts = prior_counts
    writers.write_stats_rows(result.rows)
    writers.write_bond_rows(result.bond_rows)
    writers.write_candidate_rows(result.candidate_rows)
    writers.write_crystallization_rows(result)
    writers.write_density_context_row(result)
    if plan.enabled:
        writers.write_confidence_rows(confidence.confidence_rows_for(result, plan))
    writers.write_manifest_row(
        manifest_row(result, resume, bonds, prior_bond_counts, prior_candidate_counts)
    )
    if staging is not None:
        staging.replacement_ids.add(result.pdb_id.lower())


def commit_staged_entries(
    staging: ResumeStaging,
    args: RunConfig,
    plan: confidence.ConfidencePlan,
    run_log: RunLog,
    *,
    interrupted: bool,
) -> None:
    """Merge the staged rows of a resumed batch into the outputs.

    Only IDs with a written manifest row enter replacement_ids, so an entry
    interrupted mid-write is left for the next resume. A failed merge keeps
    the staging directory, which may hold the only copy of completed work,
    and records where it is. After an interrupt the failure is logged and
    swallowed so the interrupt stays the run's outcome; otherwise it is the
    outcome, and propagates.
    """
    kept = len(staging.replacement_ids)
    if interrupted and not kept:
        staging.discard()
        return
    try:
        staging.commit(args.bonds, confidence_enabled=plan.enabled)
    except BaseException as exc:
        if interrupted and not isinstance(exc, Exception):
            raise  # a second interrupt, during the merge itself
        run_log.summary.resume_staging_recovery_dir = staging.dir
        run_log.summary.resume_staging_commit_error = f"{type(exc).__name__}: {exc}"
        if not interrupted:
            logger.error(
                "resume merge failed; completed rows retained in %s", staging.dir
            )
            raise
        logger.error(
            "could not merge %d completed entries into the output after an "
            "interrupted resume; their rows are left in %s",
            kept,
            staging.dir,
        )
        return
    staging.discard()
    if interrupted:
        run_log.summary.resume_entries_committed_after_interrupt = kept
        logger.warning(
            "interrupted after %d completed entries; their rows were merged and "
            "--resume will skip them",
            kept,
        )


def _prior_manifest_counts(
    args: RunConfig, layout: OutputLayout
) -> tuple[dict[str, str], dict[str, str]]:
    """Bond and candidate counts a bond-less resume carries over per entry."""
    if not (args.resume and not args.bonds):
        return {}, {}
    return (
        manifest_values_by_id(layout.manifest, "n_bonds"),
        manifest_values_by_id(layout.manifest, "n_candidates"),
    )


def _record_written_outputs(
    run_log: RunLog, layout: OutputLayout, writers: OutputWriters, *, bonds: bool
) -> None:
    """Record every closed output stream and its row count in the run report."""
    summary = run_log.summary
    summary.metal_rows_written = writers.n_sites
    summary.bond_rows_written = writers.n_bonds
    summary.candidate_rows_written = writers.n_candidates
    summary.confidence_rows_written = writers.n_confidence
    summary.crystallization_condition_rows_written = (
        writers.n_crystallization_conditions
    )
    summary.crystallization_summary_rows_written = writers.n_crystallization_summaries
    summary.density_context_rows_written = writers.n_density_contexts
    summary.manifest_path = layout.manifest
    summary.metal_sites_path = layout.stats
    summary.metal_bonds_path = layout.bonds if bonds else "disabled"
    summary.metal_contact_candidates_path = layout.candidates if bonds else "disabled"
    summary.crystallization_conditions_path = layout.crystallization_conditions
    summary.crystallization_summary_path = layout.crystallization_summary
    summary.density_context_path = layout.density_context


def process_entries(
    args: RunConfig,
    ids: Sequence[str],
    cfg: WorkerConfig,
    workers: int,
    layout: OutputLayout,
    plan: confidence.ConfidencePlan,
    run_log: RunLog,
    memory_plan: MemoryPlan,
) -> tuple[dispatch.BatchTally, OutputWriters]:
    """Open the outputs, run the batch, and commit any staged retry rows.

    An interrupted batch still commits the entries that completed, so the work
    already done survives; a retried entry that did not finish leaves the
    previous run's rows exactly as they were.
    """
    prior_counts = _prior_manifest_counts(args, layout)
    # Which ids the manifest already describes; see ``should_write_entry``,
    # which uses it to decide whether a retry may overwrite an existing row.
    prior_ids: set[str] = (
        set(manifest_values_by_id(layout.manifest, "status")) if args.resume else set()
    )
    targets = OutputTargets.from_layout(layout, plan)
    staging = ResumeStaging(args.output_dir, targets) if args.resume else None
    write_targets = staging.staged if staging is not None else targets

    processing_completed = False
    try:
        with contextlib.ExitStack() as handles:
            writers = _open_writers(
                handles,
                write_targets,
                bonds=args.bonds,
                confidence_columns=plan.columns,
            )

            def deliver(result: EntryResult) -> None:
                if should_write_entry(args.resume, result, prior_ids):
                    write_entry(
                        result,
                        plan,
                        writers,
                        staging,
                        prior_counts,
                        resume=args.resume,
                        bonds=args.bonds,
                    )

            tally = dispatch.dispatch_entries(
                ids, cfg, workers, memory_plan, run_log, deliver
            )
            processing_completed = True
    finally:
        if staging is not None and not processing_completed:
            commit_staged_entries(staging, args, plan, run_log, interrupted=True)

    _record_written_outputs(run_log, layout, writers, bonds=args.bonds)
    if staging is not None:
        commit_staged_entries(staging, args, plan, run_log, interrupted=False)
    return tally, writers


def run(args: RunConfig, run_log: RunLog) -> int:
    """Execute one batch, returning its exit code."""
    try:
        return _execute(args, run_log)
    except DriverError as exc:
        run_log.driver_error = str(exc)
        logger.error("%s", exc)
        return 1


def _execute_with_output_lock(
    args: RunConfig,
    run_log: RunLog,
    cofactors: Collection[str],
    env: dict[str, str],
) -> int:
    """Run every output-reading and output-writing phase under one lease."""
    sweep_owned_scratch_directories(args.output_dir)
    layout = OutputLayout(args.output_dir)
    run_mode = confidence.classify_run(args)
    run_log.details["run_mode"] = run_mode

    plan = confidence.plan_confidence(args, layout, run_mode, run_log)
    identity = AnalysisIdentity.current()
    run_log.details["analysis_config_id"] = identity.analysis_config_id
    _check_resume_is_compatible(args, layout, plan, identity.analysis_config_id)

    ids, input_root, manual_inputs = schedule_entries(
        args, layout, args.pdb_redo_cache, run_log
    )
    if not ids:
        return _finish_without_entries(args, layout, plan, run_log)

    # Manual labels may not be deposited PDB IDs; use cached metadata without
    # requiring a network lookup.
    prepare_crystallization_metadata(
        args,
        ids,
        run_log,
        allow_download=args.crystallization_download and manual_inputs is None,
    )
    _clear_stale_outputs(args, layout, plan)
    cfg = worker_config_from_args(
        args,
        env,
        input_root,
        args.pdb_redo_cache,
        cofactors,
        manual_inputs,
        identity=identity,
    )
    record_run_provenance(run_log, cfg, plan)
    memory_plan = plan_entry_memory(args, ids, cfg, run_log)
    workers = choose_worker_count(
        args, len(ids), run_log, memory_budget_bytes=memory_plan.budget_bytes
    )

    tally, writers = process_entries(
        args, ids, cfg, workers, layout, plan, run_log, memory_plan
    )
    return report_batch(args, layout, plan, tally, writers, run_log)


def _execute(args: RunConfig, run_log: RunLog) -> int:
    """Resolve prerequisites, then exclusively own the output for the run."""
    cofactors = _load_cofactor_catalog()
    if environment.configure_ccp4(args):
        return 0  # --configure-ccp4 saved a setup path and runs nothing else.
    env = environment.resolve_ccp4_environment(args)
    prepare_output_directory(args.output_dir)
    try:
        with OutputDirectoryLock(args.output_dir, run_log.command):
            return _execute_with_output_lock(args, run_log, cofactors, env)
    except (OutputDirectoryBusyError, OutputDirectoryLockError) as exc:
        raise DriverError(str(exc)) from None
