"""Manage a batch of entry analyses from setup to final results.

Prepare the environment, select entries, and check whether a previous run
can be resumed. Open the outputs, hand the batch to the dispatcher, and
write results, confidence scores, and the final report.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar, Literal, NamedTuple, TextIO

from analysis_config import analysis_config_id, analysis_configs_are_compatible
from ccp4_setup import REPO_DIR
from confidence_score import (
    ANALYSIS_COLUMNS as CONFIDENCE_ANALYSIS_COLUMNS,
)
from confidence_score import (
    CONFIDENCE_INPUT_COLUMNS,
    REFERENCE_METADATA_FILE,
    ConfidenceReference,
    classify_without_reference,
    complete_confidence_site_count,
    finalize_database_confidence,
    prepare_result_confidence_inputs,
    score_against_reference,
    validate_scored_reference,
)
from confidence_score import (
    load_reference as load_confidence_reference,
)
from crystallization_conditions import (
    CONDITION_COLUMNS,
    SUMMARY_COLUMNS,
    CrystallizationMetadataError,
    prefetch_rcsb_crystallization_metadata,
    write_review_queue,
)
from driver import dispatch, environment
from driver.dispatch import BatchTally
from driver.errors import DriverError
from driver.output_lock import (
    OutputDirectoryBusyError,
    OutputDirectoryLock,
    OutputDirectoryLockError,
    sweep_owned_scratch_directories,
)
from driver.resources import (
    MemoryPlan,
    automatic_worker_limits,
    available_memory_bytes,
    estimate_entry_memory,
    scheduling_memory_budget,
)
from driver.resume import (
    ResumeStaging,
    load_done,
    manifest_values_by_id,
    remove_stale_disabled_bond_outputs,
    resume_replacement_succeeded,
    validate_resume_schemas,
)
from driver.runlog import RunLog
from driver.writers import STATS_COLUMNS, OutputWriters, manifest_row
from inputs import (
    ensure_entry_available,
    enumerate_entries,
    infer_pdb_id_from_path,
    read_data_json_properties,
)
from metal_identification import DENSITY_CONTEXT_COLUMNS
from reference_data import (
    cofactor_ids,
    reference_data_checksums,
    reference_data_id,
)
from run_config import RunConfig
from run_logging import level_for_verbosity, logger_for, worker_level
from worker_contracts import EntryResult, WorkerConfig

DEFAULT_CONFIDENCE_REFERENCE_DIR = os.path.join(
    REPO_DIR, "src", "data", "confidence_reference"
)

logger = logger_for(__name__)


def resolve_confidence_reference_dir(
    output_dir: str, configured_dir: str | None = None
) -> tuple[str | None, tuple[str, ...]]:
    """Find a frozen confidence reference, honoring an explicit override."""
    candidates: tuple[str, ...]
    if configured_dir is not None:
        candidates = (configured_dir,)
    else:
        candidates = (
            os.path.join(output_dir, "confidence_reference"),
            DEFAULT_CONFIDENCE_REFERENCE_DIR,
        )
    for candidate in candidates:
        metadata_path = os.path.join(candidate, REFERENCE_METADATA_FILE)
        if os.path.isfile(metadata_path):
            return candidate, candidates
    return None, candidates


def load_ids_from_file(path: str) -> list[str]:
    """Read PDB IDs from comma- or newline-separated text.

    Use utf-8-sig to accept byte-order marks from Windows editors and exports.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"id file not found: {path}")
    ids: list[str] = []
    with open(path, encoding="utf-8-sig") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in re.split(r"[,\s]+", line):
                if not token:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9]{4}", token):
                    raise ValueError(f"invalid PDB id {token!r} at {path}:{lineno}")
                ids.append(token.lower())
    return list(dict.fromkeys(ids))


def select_entry_ids(
    args: RunConfig, cache_root: str
) -> tuple[list[str], str, dict[str, str | None] | None]:
    """Resolve the run's work list, returning ``(ids, root, manual_inputs)``."""
    root = args.pdb_redo_root or cache_root
    if args.pdb_file or args.mtz_file or args.cif_file:
        pdb_id = (
            args.id
            or infer_pdb_id_from_path(args.cif_file)
            or infer_pdb_id_from_path(args.pdb_file)
            or infer_pdb_id_from_path(args.mtz_file)
        )
        if not pdb_id:
            raise DriverError(
                "Manual input mode requires --id or a file name that contains "
                "a 4-character PDB id."
            )
        if args.data_json:
            try:
                read_data_json_properties(args.data_json)
            except ValueError as exc:
                raise DriverError(f"Invalid --data-json: {exc}") from None
        return (
            [pdb_id],
            root,
            {
                "pdb_file": args.pdb_file,
                "mtz_file": args.mtz_file,
                "cif_file": args.cif_file,
                "data_json": args.data_json,
            },
        )

    if args.id:
        try:
            used_root = ensure_entry_available(args.id, args.pdb_redo_root, cache_root)
        except FileNotFoundError:
            raise DriverError(
                f"Entry {args.id} not found locally and download failed."
            ) from None
        except OSError as exc:
            # Report cache and filesystem failures as usage errors without mislabeling
            # them as missing entries.
            raise DriverError(
                f"Entry {args.id} could not be prepared: {type(exc).__name__}: {exc}"
            ) from None
        if used_root != args.pdb_redo_root:
            logger.info("auto-downloaded %s into cache at %s", args.id, cache_root)
        return [args.id], used_root, None

    if args.id_file:
        try:
            ids = load_ids_from_file(args.id_file)
        except (FileNotFoundError, ValueError) as exc:
            raise DriverError(str(exc)) from None
        logger.info("loaded %d IDs from %s", len(ids), args.id_file)
        return ids, root, None

    if not args.pdb_redo_root:
        raise DriverError(
            "Supply --pdb-redo-root to process a local PDB-REDO mirror, "
            "or choose entries with --id, --id-file, or manual input files."
        )
    logger.info("enumerating final PDB-REDO entries under %s", root)
    # Resume subtracts finished entries from the full set, so enumeration can
    # stop early only when not resuming.
    limit = args.max_pdbs if (args.max_pdbs and not args.resume) else None
    return enumerate_entries(root, limit=limit), root, None


class OutputLayout:
    """Paths to run artifacts derived from the output directory."""

    def __init__(self, output_dir: str) -> None:
        """Derive every run artifact path from an output directory."""
        self.output_dir = output_dir
        self.manifest = os.path.join(output_dir, "manifest.csv")
        self.stats = os.path.join(output_dir, "metal_sites_all.csv")
        self.density_context = os.path.join(output_dir, "density_context_all.csv")
        self.bonds = os.path.join(output_dir, "metal_bonds_all.csv")
        self.candidates = os.path.join(output_dir, "metal_contact_candidates_all.csv")
        self.confidence_inputs = os.path.join(output_dir, "confidence_inputs_all.csv")
        self.confidence_scores = os.path.join(output_dir, "confidence_scores_all.csv")
        self.crystallization_conditions = os.path.join(
            output_dir, "crystallization_conditions_all.csv"
        )
        self.crystallization_summary = os.path.join(
            output_dir, "crystallization_summary_all.csv"
        )
        self.review_queue = os.path.join(output_dir, "review_queue_all.csv")
        self.reference_dir = os.path.join(output_dir, "confidence_reference")
        self.legacy_scientific_outputs = (
            os.path.join(output_dir, "metal_stats_all.csv"),
            os.path.join(output_dir, "metal_candidates_all.csv"),
        )

    @property
    def core(self) -> tuple[str, str, str, str]:
        """The four always-written outputs, in resume-validation order."""
        return (self.manifest, self.stats, self.bonds, self.candidates)


ConfidenceMode = Literal["database", "reference", "classification"]


class ConfidencePlan:
    """Whether this run scores confidence, and against what.

    ``database`` streams the inputs an uncapped full-database run finalizes
    into a new reference; ``reference`` scores against one that already exists;
    ``classification`` emits raw-threshold verdicts without empirical ranking;
    ``None`` means confidence analysis is off because bonds are disabled.
    """

    def __init__(self) -> None:
        """Initialize a disabled confidence-analysis plan."""
        self.mode: ConfidenceMode | None = None
        self.reference: ConfidenceReference | None = None
        self.stream_path: str | None = None
        self.columns: Sequence[str] | None = None
        # A resumed ``reference`` run also extends the inputs stream a prior
        # database run left behind, so the two stay row-for-row aligned.
        self.synchronize_inputs: bool = False

    @property
    def enabled(self) -> bool:
        """Return whether confidence analysis is enabled."""
        return self.mode is not None

    @property
    def output_path(self) -> str:
        """The confidence stream this run writes; set whenever a mode is."""
        if self.stream_path is None:
            raise RuntimeError("confidence output path is not configured")
        return self.stream_path

    @property
    def scored_reference(self) -> ConfidenceReference:
        """The frozen reference a ``reference`` run scores against."""
        if self.reference is None:
            raise RuntimeError("confidence reference is not configured")
        return self.reference


@dataclass(frozen=True)
class OutputTargets:
    """The files one run writes, each by name.

    Resume staging sees the same files as one flat sequence: the four core
    outputs, the always-written extras, then whichever confidence streams
    this run keeps. Field order here is that sequence, so ``ordered`` and
    ``rebound`` are the only places it is spelled out.
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

    ALWAYS_WRITTEN_EXTRAS: ClassVar[tuple[str, ...]] = (
        "crystallization_conditions",
        "crystallization_summary",
        "density_context",
    )

    @classmethod
    def for_run(cls, layout: OutputLayout, plan: ConfidencePlan) -> OutputTargets:
        """The targets a run with this layout and confidence plan writes."""
        return cls(
            manifest=layout.manifest,
            stats=layout.stats,
            bonds=layout.bonds,
            candidates=layout.candidates,
            crystallization_conditions=layout.crystallization_conditions,
            crystallization_summary=layout.crystallization_summary,
            density_context=layout.density_context,
            confidence=plan.output_path if plan.enabled else None,
            confidence_inputs=(
                layout.confidence_inputs if plan.synchronize_inputs else None
            ),
        )

    def _present(self) -> list[str]:
        return [
            field.name
            for field in fields(self)
            if getattr(self, field.name) is not None
        ]

    def ordered(self) -> tuple[str, ...]:
        """Every path this run writes, in the order resume staging expects."""
        return tuple(getattr(self, name) for name in self._present())

    def rebound(self, paths: Sequence[str]) -> OutputTargets:
        """The same targets pointed at a parallel sequence of paths."""
        return replace(self, **dict(zip(self._present(), paths, strict=True)))


def plan_entry_memory(
    args: RunConfig, ids: Sequence[str], cfg: WorkerConfig, run_log: RunLog
) -> MemoryPlan:
    """Estimate entry memory and record the resulting admission budget."""
    if not ids:
        raise ValueError("memory planning needs at least one entry")
    estimates = tuple(
        estimate_entry_memory(pdb_id, cfg.root, cfg.manual_inputs) for pdb_id in ids
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


def _prepare_output_directory(output_dir: str) -> None:
    """Create ``--output-dir`` before its stable lock file is opened."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise DriverError(
            f"Cannot use --output-dir {output_dir}: {exc.strerror or exc}"
        ) from None


def _classify_run(args: RunConfig) -> tuple[str, bool]:
    """Return the run mode and whether this is an uncapped full-database run.

    Only the full-database mode can build a new confidence reference.
    """
    manual_requested = bool(args.pdb_file or args.mtz_file or args.cif_file)
    database_run = (
        not args.id
        and not args.id_file
        and not manual_requested
        and args.max_pdbs is None
    )
    run_mode = (
        "manual"
        if manual_requested
        else "single"
        if args.id
        else "id_file"
        if args.id_file
        else "database"
        if database_run
        else "capped_database"
    )
    return run_mode, database_run


def plan_confidence(
    args: RunConfig,
    layout: OutputLayout,
    database_run: bool,
    run_log: RunLog,
) -> ConfidencePlan:
    """Decide this run's confidence mode before any entry is processed."""
    plan = ConfidencePlan()
    if not args.bonds:
        return plan
    if database_run:
        # A full-database run builds its own reference. Warn about an existing-reference
        # option without rejecting commands shared with smaller runs.
        if args.confidence_reference_dir:
            logger.warning(
                "--confidence-reference-dir is ignored on an uncapped "
                "full-database run: that run builds the reference later runs "
                "are scored against, so it cannot be scored against an "
                "existing one. Cap the run with --max-pdbs to use %s.",
                args.confidence_reference_dir,
            )
        plan.mode = "database"
        plan.stream_path = layout.confidence_inputs
        plan.columns = CONFIDENCE_INPUT_COLUMNS
        return plan

    reference_dir, searched_dirs = resolve_confidence_reference_dir(
        args.output_dir, args.confidence_reference_dir
    )
    if reference_dir is None:
        logger.info(
            "no frozen confidence reference is installed, so Alchemy will "
            "emit authoritative PASS/REVIEW/SUSPECT classifications without "
            "empirical ranking scores. Complete an uncapped full-database run "
            "or pass --confidence-reference-dir to add rankings. (searched: %s)",
            ", ".join(searched_dirs),
        )
        plan.mode = "classification"
        plan.stream_path = layout.confidence_scores
        plan.columns = (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS)
        return plan

    try:
        plan.reference = load_confidence_reference(reference_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError(f"Invalid confidence reference: {exc}") from None
    run_log.details["confidence_reference_dir"] = reference_dir
    plan.mode = "reference"
    plan.stream_path = layout.confidence_scores
    plan.columns = (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS)
    plan.synchronize_inputs = bool(args.resume) and os.path.isfile(
        layout.confidence_inputs
    )
    return plan


def _check_resume_is_compatible(
    args: RunConfig,
    layout: OutputLayout,
    plan: ConfidencePlan,
    current_analysis_config_id: str,
) -> None:
    """Refuse to resume onto output this run cannot safely extend."""
    if not args.resume:
        return
    try:
        validate_resume_schemas(
            *layout.core,
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
                *layout.core,
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
    if plan.mode == "reference" and os.path.isfile(plan.output_path):
        try:
            validate_scored_reference(plan.output_path, plan.scored_reference)
        except (OSError, ValueError) as exc:
            raise DriverError(f"Cannot resume confidence output: {exc}") from None


def schedule_entries(
    args: RunConfig,
    layout: OutputLayout,
    cache_root: str,
    run_log: RunLog,
) -> tuple[list[str], str, dict[str, str | None] | None]:
    """Return ``(ids, root, manual_inputs)`` for the entries this run will do.

    ``--resume`` removes finished entries before ``--max-pdbs`` caps what is
    left; capping first would re-offer the same finished prefix forever.
    """
    ids, root, manual_inputs = select_entry_ids(args, cache_root)
    run_log.details["entries_selected_before_resume"] = len(ids)
    run_log.details["resolved_input_root"] = root

    if args.resume:
        bonds_required = bool(args.bonds)
        bond_output_present = os.path.isfile(layout.bonds)
        candidate_output_present = os.path.isfile(layout.candidates)
        normally_done = load_done(
            layout.manifest,
            bonds_required=bonds_required,
            bond_output_present=bond_output_present,
            candidate_output_present=candidate_output_present,
        )
        if args.retry_partials:
            done = load_done(
                layout.manifest,
                bonds_required=bonds_required,
                bond_output_present=bond_output_present,
                candidate_output_present=candidate_output_present,
                retry_partial_ids=ids,
            )
            reselected = normally_done - done
            run_log.details["terminal_partials_reselected"] = len(reselected)
            logger.info(
                "selected %d terminal partial entr%s for retry",
                len(reselected),
                "y" if len(reselected) == 1 else "ies",
            )
        else:
            done = normally_done
        # Normalize IDs for manifest comparison while preserving on-disk path spelling.
        ids = [i for i in ids if i.lower() not in done]
    if args.max_pdbs is not None:
        ids = ids[: args.max_pdbs]
    run_log.details["entries_scheduled"] = len(ids)
    return ids, root, manual_inputs


def _finalize_confidence_reference(layout: OutputLayout) -> tuple[int, int, int]:
    """Score the streamed inputs and freeze the database reference."""
    try:
        return finalize_database_confidence(
            layout.confidence_inputs,
            layout.confidence_scores,
            layout.reference_dir,
            manifest_path=layout.manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError(f"Confidence finalization failed: {exc}") from None


def _finish_without_entries(
    args: RunConfig, layout: OutputLayout, plan: ConfidencePlan
) -> int:
    """Exit code for a run whose work list came back empty."""
    if (
        args.resume
        and plan.mode == "database"
        and os.path.isfile(layout.confidence_inputs)
    ):
        total, scored, cohort = _finalize_confidence_reference(layout)
        logger.info(
            "no entries required retry; finalized %d confidence rows "
            "(%d scored; database cohort %d) -> %s",
            total,
            scored,
            cohort,
            layout.confidence_scores,
        )
        logger.info("confidence reference -> %s", layout.reference_dir)
        _finalize_review_queue(layout)
        return 0
    if plan.enabled:
        _finalize_review_queue(layout)
    logger.info("no entries to process")
    return 0


def _clear_stale_outputs(
    args: RunConfig, layout: OutputLayout, plan: ConfidencePlan
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
    # The reference metadata file is the reference's completion marker.
    reference_marker = os.path.join(layout.reference_dir, REFERENCE_METADATA_FILE)
    stale: tuple[str, ...]
    if plan.mode == "database":
        stale = (layout.confidence_scores, reference_marker)
    elif plan.mode == "reference":
        stale = (layout.confidence_inputs,)
    else:
        stale = (layout.confidence_inputs, layout.confidence_scores, reference_marker)
    try:
        for path in (*stale, *layout.legacy_scientific_outputs):
            if os.path.isfile(path):
                os.unlink(path)
    except OSError as exc:
        raise DriverError(f"Could not clear stale confidence output: {exc}") from None


def choose_worker_count(args: RunConfig, entry_count: int, run_log: RunLog) -> int:
    """Size the pool, never above the number of entries there are to run.

    A Pool creates every worker up front, and under the spawn start method each
    one re-imports gemmi into its own interpreter.
    """
    cpu_limit, memory_limit = automatic_worker_limits(
        memory_limit_bytes=args.memory_limit,
        utilization=args.memory_utilization,
    )
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
    root: str,
    cache_root: str,
    cofactors: Collection[str],
    manual_inputs: dict[str, str | None] | None,
    plan: ConfidencePlan,
    run_log: RunLog,
    *,
    identity: AnalysisIdentity,
) -> WorkerConfig:
    """Build the config every worker is initialized with, once per run."""
    cfg = WorkerConfig(
        root=root,
        mirror_root=args.pdb_redo_root,
        cache_root=cache_root,
        env=env,
        output_dir=args.output_dir,
        cofactors=cofactors,
        keep=args.keep_intermediates,
        bonds=args.bonds,
        density_map_scope=args.density_map_scope,
        ccp4_timeout_s=args.ccp4_timeout,
        log_level=worker_level(
            level_for_verbosity(args.verbose, args.quiet), args.log_file
        ),
        allow_download=bool(args.id or args.id_file),
        manual_inputs=manual_inputs,
        alchemy_commit=environment.alchemy_commit(),
        gemmi_version=environment.gemmi_version(),
        ccp4_version=environment.ccp4_version(env),
        reference_data_id=identity.reference_data_id,
        analysis_config_id=identity.analysis_config_id,
        pdb_metadata_cache=args.pdb_metadata_cache,
    )
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
    return cfg


def prepare_crystallization_metadata(
    args: RunConfig,
    ids: Sequence[str],
    run_log: RunLog,
    *,
    allow_download: bool | None = None,
) -> None:
    """Warm the original-PDB metadata cache before costly worker execution."""
    try:
        stats = prefetch_rcsb_crystallization_metadata(
            ids,
            args.pdb_metadata_cache,
            allow_download=(
                args.crystallization_download
                if allow_download is None
                else allow_download
            ),
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

    def opened(path: str) -> TextIO:
        return handles.enter_context(open(path, "w", newline=""))

    def opened_if(path: str | None) -> TextIO | None:
        return opened(path) if path is not None else None

    return OutputWriters(
        opened(targets.manifest),
        opened(targets.stats),
        opened(targets.bonds) if bonds else None,
        opened(targets.candidates) if bonds else None,
        opened_if(targets.confidence),
        confidence_columns,
        opened_if(targets.confidence_inputs),
        opened(targets.crystallization_conditions),
        opened(targets.crystallization_summary),
        opened(targets.density_context),
    )


def confidence_rows_for(
    result: EntryResult, plan: ConfidencePlan
) -> list[dict[str, Any]]:
    """Prepare one entry's confidence rows, scoring against a reference if available."""
    if result.metal_site_limit_exceeded:
        return []
    rows = prepare_result_confidence_inputs(
        result.rows, result.bond_rows, STATS_COLUMNS
    )
    rows = complete_confidence_site_count(
        rows,
        result.pdb_id,
        result.n_metals,
        result.confidence_inputs_missing_reason,
    )
    if plan.reference is not None:
        rows = score_against_reference(rows, plan.reference)
    elif plan.mode == "classification":
        rows = classify_without_reference(rows)
    return rows


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
    plan: ConfidencePlan,
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
        writers.write_confidence_rows(confidence_rows_for(result, plan))
    writers.write_manifest_row(
        manifest_row(result, resume, bonds, prior_bond_counts, prior_candidate_counts)
    )
    if staging is not None:
        staging.replacement_ids.add(result.pdb_id.lower())


def keep_completed_staging(
    staging: ResumeStaging,
    args: RunConfig,
    plan: ConfidencePlan,
    run_log: RunLog,
) -> None:
    """Commit entries completed before a resumed batch was interrupted.

    Only IDs with a written manifest row enter replacement_ids. Keep unfinished
    rows in staging and preserve completed work if committing fails.
    """
    kept = len(staging.replacement_ids)
    if not kept:
        staging.discard()
        return
    try:
        staging.commit(args.bonds, confidence_enabled=plan.enabled)
    except Exception as exc:
        # Preserve staging for manual recovery if it is the only copy of completed work.
        run_log.summary["resume_staging_recovery_dir"] = staging.dir
        run_log.summary["resume_staging_commit_error"] = f"{type(exc).__name__}: {exc}"
        logger.error(
            "could not merge %d completed entries into the output after an "
            "interrupted resume; their rows are left in %s",
            kept,
            staging.dir,
        )
        return
    staging.discard()
    run_log.summary["resume_entries_committed_after_interrupt"] = kept
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


def process_entries(
    args: RunConfig,
    ids: Sequence[str],
    cfg: WorkerConfig,
    workers: int,
    layout: OutputLayout,
    plan: ConfidencePlan,
    run_log: RunLog,
    memory_plan: MemoryPlan | None = None,
) -> tuple[BatchTally, OutputWriters]:
    """Open the outputs, run the batch, and commit any staged retry rows.

    An interrupted batch still commits the entries that completed, so the work
    already done survives; a retried entry that did not finish leaves the
    previous run's rows exactly as they were.
    """
    prior_counts = _prior_manifest_counts(args, layout)
    if memory_plan is None:
        memory_plan = plan_entry_memory(args, ids, cfg, run_log)
    # Which ids the manifest already describes; see ``should_write_entry``,
    # which uses it to decide whether a retry may overwrite an existing row.
    prior_ids: set[str] = (
        set(manifest_values_by_id(layout.manifest, "status")) if args.resume else set()
    )
    targets = OutputTargets.for_run(layout, plan)
    staging = (
        ResumeStaging(
            args.output_dir,
            targets.ordered(),
            always_extra_count=len(OutputTargets.ALWAYS_WRITTEN_EXTRAS),
        )
        if args.resume
        else None
    )
    write_targets = targets.rebound(staging.staged) if staging is not None else targets

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
            keep_completed_staging(staging, args, plan, run_log)

    run_log.summary.update(
        metal_rows_written=writers.n_rows,
        bond_rows_written=writers.n_bonds,
        candidate_rows_written=writers.n_candidates,
        confidence_rows_written=writers.n_confidence,
        crystallization_condition_rows_written=writers.n_crystallization_conditions,
        crystallization_summary_rows_written=writers.n_crystallization_summaries,
        density_context_rows_written=writers.n_density_contexts,
        manifest_path=layout.manifest,
        metal_sites_path=layout.stats,
        metal_bonds_path=layout.bonds if args.bonds else "disabled",
        metal_contact_candidates_path=(layout.candidates if args.bonds else "disabled"),
        crystallization_conditions_path=layout.crystallization_conditions,
        crystallization_summary_path=layout.crystallization_summary,
        density_context_path=layout.density_context,
    )
    if staging is not None:
        try:
            staging.commit(args.bonds, confidence_enabled=plan.enabled)
        except BaseException as exc:
            run_log.summary["resume_staging_recovery_dir"] = staging.dir
            run_log.summary["resume_staging_commit_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            logger.error(
                "resume merge failed; completed rows retained in %s", staging.dir
            )
            raise
        staging.discard()
    return tally, writers


def _finalize_review_queue(layout: OutputLayout) -> int:
    """Regenerate the derived triage view from canonical completed outputs."""
    try:
        return write_review_queue(
            layout.confidence_scores,
            layout.crystallization_summary,
            layout.review_queue,
            (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS),
        )
    except (OSError, ValueError) as exc:
        raise DriverError(f"Review queue finalization failed: {exc}") from None


def _report_batch(
    args: RunConfig,
    layout: OutputLayout,
    plan: ConfidencePlan,
    tally: BatchTally,
    writers: OutputWriters,
    run_log: RunLog,
) -> int:
    """Report batch results, finalize eligible confidence outputs, and return the exit code.

    Build a database reference only when no recoverable entries remain.
    """
    print(
        f"Done. ok={tally.counts['ok']} partial={tally.counts['partial']} "
        f"skip={tally.counts['skip']} error={tally.counts['error']} "
        f"no_metals={tally.no_metals} "
        f"metal_site_limit_exceeded={tally.metal_site_limit_exceeded}; "
        f"{writers.n_rows} metal/cofactor rows -> {layout.stats}",
        flush=True,
    )
    if args.bonds:
        print(f"      {writers.n_bonds} bond rows -> {layout.bonds}", flush=True)
        print(
            f"      {writers.n_candidates} candidate rows -> {layout.candidates}",
            flush=True,
        )
    print(
        f"      {writers.n_crystallization_conditions} crystallization condition "
        f"rows -> {layout.crystallization_conditions}",
        flush=True,
    )
    print(
        f"      {writers.n_crystallization_summaries} crystallization summary "
        f"rows -> {layout.crystallization_summary}",
        flush=True,
    )
    print(
        f"      {writers.n_density_contexts} density context rows -> "
        f"{layout.density_context}",
        flush=True,
    )
    database_run = plan.mode == "database"
    exit_code = tally.exit_code(database_run=database_run)
    if plan.mode == "database":
        # Finalize when no recoverable entries remain. Known deterministic exclusions
        # are recorded in reference metadata and do not prevent completion.
        unfinished = tally.recoverable_incompleteness()
        if unfinished == 0:
            permanent = tally.terminal_errors
            if permanent:
                logger.warning(
                    "finalizing the confidence reference with %d permanently "
                    "failed entr%s: no retry can add them, and their absence "
                    "is recorded in the reference metadata",
                    permanent,
                    "y" if permanent == 1 else "ies",
                )
            total, scored, cohort = _finalize_confidence_reference(layout)
            run_log.summary.update(
                confidence_status="finalized",
                confidence_rows=total,
                confidence_scored_rows=scored,
                confidence_reference_cohort=cohort,
                confidence_scores_path=layout.confidence_scores,
                confidence_reference_path=layout.reference_dir,
            )
            print(
                f"      {total} confidence rows ({scored} scored; "
                f"reference cohort {cohort}) -> {layout.confidence_scores}",
                flush=True,
            )
            print(f"      confidence reference -> {layout.reference_dir}", flush=True)
        else:
            run_log.summary["confidence_status"] = "not_finalized_incomplete_run"
            run_log.summary["confidence_recoverable_entries"] = unfinished
            print(
                f"      confidence inputs were retained, but the database "
                f"reference was not finalized: {unfinished} entr"
                f"{'y' if unfinished == 1 else 'ies'} could still be added by "
                f"--resume (missing inputs, lost workers, or retryable "
                f"processing failures).",
                flush=True,
            )
    elif plan.mode == "reference":
        cohort_size = plan.scored_reference.cohort_size
        print(
            f"      {writers.n_confidence} confidence rows compared with "
            f"database cohort {cohort_size} -> "
            f"{layout.confidence_scores}",
            flush=True,
        )
        run_log.summary.update(
            confidence_status="scored_against_reference",
            confidence_reference_cohort=cohort_size,
            confidence_scores_path=layout.confidence_scores,
        )
    elif plan.mode == "classification":
        print(
            f"      {writers.n_confidence} confidence classifications "
            f"(empirical ranking unavailable) -> {layout.confidence_scores}",
            flush=True,
        )
        run_log.summary.update(
            confidence_status="classified_without_reference",
            confidence_rows=writers.n_confidence,
            confidence_scores_path=layout.confidence_scores,
        )
    if plan.enabled and os.path.isfile(layout.confidence_scores):
        review_rows = _finalize_review_queue(layout)
        run_log.summary.update(
            review_queue_rows=review_rows,
            review_queue_path=layout.review_queue,
        )
        print(
            f"      {review_rows} REVIEW/SUSPECT rows -> {layout.review_queue}",
            flush=True,
        )
    if exit_code:
        logger.warning(
            "completed with incomplete entries: errors=%d "
            "(recoverable=%d, terminal=%d), skips=%d, retryable_partials=%d",
            tally.counts["error"],
            tally.recoverable_errors,
            tally.terminal_errors,
            tally.counts["skip"],
            tally.retryable_partials,
        )
    return exit_code


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
    run_mode, database_run = _classify_run(args)
    run_log.details["run_mode"] = run_mode

    plan = plan_confidence(args, layout, database_run, run_log)
    identity = AnalysisIdentity.current()
    run_log.details["analysis_config_id"] = identity.analysis_config_id
    _check_resume_is_compatible(args, layout, plan, identity.analysis_config_id)

    ids, root, manual_inputs = schedule_entries(
        args, layout, args.pdb_redo_cache, run_log
    )
    if not ids:
        return _finish_without_entries(args, layout, plan)

    # Manual labels may not be deposited PDB IDs; use cached metadata without
    # requiring a network lookup.
    prepare_crystallization_metadata(
        args, ids, run_log, allow_download=False if manual_inputs else None
    )
    _clear_stale_outputs(args, layout, plan)
    cfg = worker_config_from_args(
        args,
        env,
        root,
        args.pdb_redo_cache,
        cofactors,
        manual_inputs,
        plan,
        run_log,
        identity=identity,
    )
    memory_plan = plan_entry_memory(args, ids, cfg, run_log)
    workers = choose_worker_count(args, len(ids), run_log)

    tally, writers = process_entries(
        args, ids, cfg, workers, layout, plan, run_log, memory_plan
    )
    return _report_batch(args, layout, plan, tally, writers, run_log)


def _execute(args: RunConfig, run_log: RunLog) -> int:
    """Resolve prerequisites, then exclusively own the output for the run."""
    cofactors = _load_cofactor_catalog()
    env, _ = environment.resolve_ccp4_environment(args)
    if env is None:
        return 0  # --configure-ccp4 saved a setup path and ran nothing else.
    _prepare_output_directory(args.output_dir)
    try:
        with OutputDirectoryLock(args.output_dir, run_log.command):
            return _execute_with_output_lock(args, run_log, cofactors, env)
    except (OutputDirectoryBusyError, OutputDirectoryLockError) as exc:
        raise DriverError(str(exc)) from None
