"""The batch driver: everything between the parsed arguments and the workers.

Two constraints shape the dispatch loop. ``multiprocessing.Pool`` silently
replaces a worker that died and never delivers a result for the task it held,
so a lost entry is recovered from the pool roster rather than waited on. And a
worker killed while blocked on the task queue never releases that queue's lock,
which wedges ``Pool.terminate``, so shutdown is bounded explicitly.
"""

import contextlib
import json
import os
import re
import subprocess
import threading
import time
from multiprocessing import (
    Pool,
    SimpleQueue,
    TimeoutError as MultiprocessingTimeoutError,
)
from typing import Any, Dict, Optional, Sequence, Tuple

from _version import __version__
from reference_data import (
    cofactor_ids,
    reference_data_checksums,
    reference_data_id,
)
from run_logging import (
    logger_for,
    worker_level,
    level_for_verbosity,
    create_worker_log_queue,
    start_worker_log_listener,
)
from ccp4_setup import (
    REPO_DIR,
    REQUIRED_CCP4_TOOLS,
    Ccp4SetupError,
    ccp4_tools_available,
    find_ccp4_setup,
    load_ccp4_setup_config,
    resolve_env,
    save_ccp4_setup,
    verify_ccp4,
)
from inputs import (
    ensure_entry_available,
    enumerate_entries,
    infer_pdb_id_from_path,
)
from worker import (
    WorkerConfig,
    _init_worker,
    _worker_death_result,
    process,
)
from driver.progress import _ProgressReporter
from driver.output_lock import (
    OutputDirectoryBusyError,
    OutputDirectoryLock,
    OutputDirectoryLockError,
    sweep_owned_scratch_directories,
)
from driver.resources import automatic_worker_limits
from driver.resume import (
    _ResumeStaging,
    _manifest_values_by_id,
    _resume_replacement_succeeded,
    load_done,
    remove_stale_disabled_bond_outputs,
    validate_resume_schemas,
)
from driver.writers import STATS_COLUMNS, _manifest_row, _OutputWriters
from confidence_score import (
    ANALYSIS_COLUMNS as CONFIDENCE_ANALYSIS_COLUMNS,
    CONFIDENCE_INPUT_COLUMNS,
    REFERENCE_METADATA_FILE,
    ConfidenceReference,
    finalize_database_confidence,
    complete_confidence_site_count,
    load_reference as load_confidence_reference,
    prepare_result_confidence_inputs,
    score_against_reference,
    validate_scored_reference,
)


DEFAULT_ROOT = "/datasets/bioinfo/pdb-redo"
DEFAULT_CONFIDENCE_REFERENCE_DIR = os.path.join(REPO_DIR, "confidence_reference")

ALCHEMY_VERSION = __version__
# Seconds of no completed entry, after a worker died without naming the entry
# it held, before the remaining outstanding entries are failed retryably.
WORKER_STALL_GRACE_S = 600.0
# Seconds to let a worker pool shut down cleanly before its children are killed
# outright. Every result has been collected by then, so only a wedged pool ever
# reaches the deadline.
WORKER_SHUTDOWN_GRACE_S = 5.0

# Budget for the `git` probes that stamp run provenance. Exceeding it costs the
# commit hash, which degrades to "unknown", rather than the run.
PROVENANCE_COMMAND_TIMEOUT_S = 1


logger = logger_for(__name__)


def _verify_resolved_ccp4(env, setup_path):
    """Verify ``env``, naming the script that was run when it comes up short."""
    try:
        verify_ccp4(env)
    except Ccp4SetupError as exc:
        raise Ccp4SetupError(
            f"Ran {setup_path}, but CCP4 tools are still not available. {exc}"
        ) from None


def _resolve_ccp4_environment(args):
    """Resolve the CCP4 environment, raising ``Ccp4SetupError`` on any failure."""
    config = load_ccp4_setup_config()
    if args.configure_ccp4:
        setup_path = os.path.abspath(os.path.expanduser(args.configure_ccp4))
        if not os.path.exists(setup_path):
            raise Ccp4SetupError(f"CCP4 setup file not found: {setup_path}")

        env = resolve_env(setup_path)
        _verify_resolved_ccp4(env, setup_path)

        saved = save_ccp4_setup(setup_path)
        logger.info(
            "verified %s are available; saved CCP4 setup path to %s",
            ", ".join(REQUIRED_CCP4_TOOLS),
            ", ".join(saved),
        )
        return None, None

    environment = os.environ.copy()

    # An explicit --ccp4-setup is checked before the ambient environment, and a
    # missing path is an error rather than a fallthrough: honouring PATH first
    # would run against the very installation the user is replacing, and stamp
    # that installation's version as the run's provenance.
    if args.ccp4_setup:
        setup_path = os.path.abspath(os.path.expanduser(args.ccp4_setup))
        if not os.path.exists(setup_path):
            # The user's own spelling, not the expanded path, so the message
            # shows the typo as they typed it.
            raise Ccp4SetupError(f"CCP4 setup file not found: {args.ccp4_setup}")
        env = resolve_env(setup_path)
        _verify_resolved_ccp4(env, setup_path)
        return env, setup_path

    if ccp4_tools_available(environment):
        return environment, None

    ccp4_setup = find_ccp4_setup(env=environment, config=config)
    if ccp4_setup is None:
        raise Ccp4SetupError(
            f"Required CCP4 tools ({', '.join(REQUIRED_CCP4_TOOLS)}) were not "
            "found on PATH and no setup file could be auto-detected. "
            "Set them up once with --configure-ccp4 /path/to/ccp4.setup-sh, "
            "export CCP4_SETUP=/path/to/ccp4.setup-sh, or source CCP4 in\n"
            "your shell before running."
        )
    env = resolve_env(ccp4_setup)
    verify_ccp4(env)
    return env, ccp4_setup


def resolve_ccp4_environment(args):
    """Return ``(env, setup_path)`` for this run, or raise ``DriverError``."""
    try:
        return _resolve_ccp4_environment(args)
    except Ccp4SetupError as exc:
        raise DriverError(str(exc)) from None


def _alchemy_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=PROVENANCE_COMMAND_TIMEOUT_S,
        )
        commit = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=PROVENANCE_COMMAND_TIMEOUT_S,
        )
        return commit + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _gemmi_version():
    try:
        import gemmi

        return str(getattr(gemmi, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _ccp4_version(env):
    for key in ("CCP4_VERSION", "CCP4_VERSION_CODE", "CCP4VER"):
        if env.get(key):
            return env[key]
    ccp4_root = env.get("CCP4", "")
    return os.path.basename(ccp4_root.rstrip(os.sep)) if ccp4_root else "unknown"


def _drain_inflight(inflight, assignments):
    """Apply pending worker notifications to the pid -> entry assignment map."""
    while True:
        try:
            if inflight.empty():
                return
            state, pid, pdb_id = inflight.get()
        except (OSError, EOFError):  # pragma: no cover - pipe torn down
            return
        if state == "start":
            assignments[pid] = pdb_id
        else:
            assignments.pop(pid, None)


def _dead_worker_pids(pool, known_pids):
    """Return worker pids that have disappeared since the last check."""
    current = {
        child.pid for child in getattr(pool, "_pool", ()) or () if child.pid is not None
    }
    if not current:
        return set()
    dead = known_pids - current
    known_pids.clear()
    known_pids.update(current)
    return dead


def _sweep_leaked_work_dirs(output_dir):
    """Remove marked disposable scratch after output ownership is acquired."""
    return sweep_owned_scratch_directories(output_dir)


def _shutdown_pool(pool):
    """Close a worker pool, killing its children if a clean shutdown hangs.

    A worker killed while blocked in the task queue's ``get()`` never releases
    that queue's lock, and ``Pool.terminate`` then blocks acquiring it, so the
    clean shutdown runs on a deadline. Returns ``True`` when it had to be
    forced.
    """
    children = [
        child
        for child in getattr(pool, "_pool", ()) or ()
        if getattr(child, "pid", None)
    ]
    closer = threading.Thread(target=pool.terminate, daemon=True)
    closer.start()
    closer.join(WORKER_SHUTDOWN_GRACE_S)
    if not closer.is_alive():
        return False

    # Cancel the at-exit finalizer, which would repeat the same blocked wait.
    finalizer = getattr(pool, "_terminate", None)
    if finalizer is not None:
        try:
            finalizer.cancel()
        except Exception:  # noqa: BLE001 - best effort, shutdown must proceed
            pass
    for child in children:
        try:
            # Process.kill is SIGKILL on POSIX and TerminateProcess on Windows,
            # where signal.SIGKILL does not exist.
            child.kill()
        except (OSError, ValueError, AttributeError):  # already reaped
            pass
    # The closer thread is abandoned: the lock is held by a process that is
    # already gone, and a daemon thread does not keep the interpreter alive.
    return True


def resolve_confidence_reference_dir(output_dir, configured_dir=None):
    """Find a frozen confidence reference, honoring an explicit override."""
    candidates: Tuple[str, ...]
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


def _batch_exit_code(counts, retryable_partial_count):
    """Return failure when one or more entries remain operationally incomplete."""
    incomplete = (
        counts.get("error", 0) + counts.get("skip", 0) + retryable_partial_count
    )
    return 1 if incomplete else 0


def load_ids_from_file(path):
    """Return a list of PDB ids from a comma/newline-separated text file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"id file not found: {path}")
    ids = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in re.split(r"[,\s]+", line):
                token = token.strip()
                if not token:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9]{4}", token):
                    raise ValueError(f"invalid PDB id {token!r} at {path}:{lineno}")
                ids.append(token.lower())
    return list(dict.fromkeys(ids))


class DriverError(Exception):
    """A user-facing driver failure: the message is reported and the run exits 1."""


def _select_entry_ids(args, cache_root):
    """Resolve the run's work list, returning ``(ids, root, manual_inputs)``."""
    root = args.pdb_redo_root
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

    logger.info("enumerating final PDB-REDO entries under %s", root)
    # Resume subtracts finished entries from the full set, so enumeration can
    # stop early only when not resuming.
    limit = args.max_pdbs if (args.max_pdbs and not args.resume) else None
    return enumerate_entries(root, limit=limit), root, None


class _OutputLayout:
    """Every path a run reads or writes, all derived from ``--output-dir``."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.manifest = os.path.join(output_dir, "manifest.csv")
        self.stats = os.path.join(output_dir, "metal_stats_all.csv")
        self.bonds = os.path.join(output_dir, "metal_bonds_all.csv")
        self.candidates = os.path.join(output_dir, "metal_candidates_all.csv")
        self.confidence_inputs = os.path.join(output_dir, "confidence_inputs_all.csv")
        self.confidence_scores = os.path.join(output_dir, "confidence_scores_all.csv")
        self.reference_dir = os.path.join(output_dir, "confidence_reference")

    @property
    def core(self) -> Tuple[str, str, str, str]:
        """The four always-written outputs, in resume-validation order."""
        return (self.manifest, self.stats, self.bonds, self.candidates)


class _ConfidencePlan:
    """Whether this run scores confidence, and against what.

    ``database`` streams the inputs an uncapped full-database run finalizes
    into a new reference; ``reference`` scores against one that already exists;
    ``None`` means scoring is off.
    """

    def __init__(self):
        self.mode: Optional[str] = None
        self.reference: Optional[ConfidenceReference] = None
        self.stream_path: Optional[str] = None
        self.columns: Optional[Sequence[str]] = None
        self.synchronize_inputs: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode is not None


class _BatchTally:
    """Running totals for the batch and the exit code they imply."""

    def __init__(self):
        self.counts = {"ok": 0, "partial": 0, "skip": 0, "error": 0}
        self.no_metals = 0
        self.retryable_partials = 0

    def record(self, result):
        status = result.status
        self.counts[status] = self.counts.get(status, 0) + 1
        if result.no_metals:
            self.no_metals += 1
        if status == "partial" and result.retryable:
            self.retryable_partials += 1

    def exit_code(self):
        return _batch_exit_code(self.counts, self.retryable_partials)


def _load_cofactor_catalog():
    """Read the bundled metallocofactor catalog, or fail the run naming it."""
    try:
        return cofactor_ids()
    except (OSError, UnicodeError, ValueError) as exc:
        raise DriverError(f"Invalid bundled metallocofactor catalog: {exc}") from None


def _prepare_output_directory(output_dir):
    """Create ``--output-dir`` before its stable lock file is opened."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise DriverError(
            f"Cannot use --output-dir {output_dir}: {exc.strerror or exc}"
        ) from None


def _classify_run(args):
    """Return ``(run_mode, database_run)`` for this invocation.

    ``database_run`` is the uncapped full-mirror case, the only one that may
    finalize a confidence reference: a capped or hand-picked run is no cohort.
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


def _plan_confidence(args, layout, database_run, run_log):
    """Decide this run's confidence mode before any entry is processed."""
    plan = _ConfidencePlan()
    if not args.bonds:
        return plan
    if database_run:
        plan.mode = "database"
        plan.stream_path = layout.confidence_inputs
        plan.columns = CONFIDENCE_INPUT_COLUMNS
        return plan

    reference_dir, searched_dirs = resolve_confidence_reference_dir(
        args.output_dir, args.confidence_reference_dir
    )
    if reference_dir is None:
        logger.info(
            "confidence scoring is not enabled: no frozen reference is "
            "distributed with Alchemy, because the score is not yet "
            "finalized. All other outputs are unaffected. To enable it, "
            "complete an uncapped full-database run or pass "
            "--confidence-reference-dir. (searched: %s)",
            ", ".join(searched_dirs),
        )
        return plan

    try:
        plan.reference = load_confidence_reference(reference_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError(f"Invalid confidence reference: {exc}") from None
    run_log.details["confidence_reference_dir"] = reference_dir
    plan.mode = "reference"
    plan.stream_path = layout.confidence_scores
    plan.columns = (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS)
    return plan


def _check_resume_is_compatible(args, layout: _OutputLayout, plan) -> None:
    """Refuse to resume onto output this run cannot safely extend.

    Sets ``plan.synchronize_inputs`` as a side effect.
    """
    if not args.resume:
        return
    if plan.enabled and (
        plan.stream_path is None or not os.path.isfile(plan.stream_path)
    ):
        raise DriverError(
            "Cannot resume confidence-aware output because "
            f"{plan.stream_path} is missing; use a fresh output directory."
        )
    try:
        validate_resume_schemas(
            *layout.core,
            bonds_enabled=args.bonds,
            confidence_path=plan.stream_path,
            confidence_columns=plan.columns,
        )
        plan.synchronize_inputs = plan.mode == "reference" and os.path.isfile(
            layout.confidence_inputs
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
    if plan.mode == "reference":
        try:
            validate_scored_reference(plan.stream_path, plan.reference)
        except (OSError, ValueError) as exc:
            raise DriverError(f"Cannot resume confidence output: {exc}") from None


def _schedule_entries(args, layout, cache_root, run_log):
    """Return ``(ids, root, manual_inputs)`` for the entries this run will do.

    ``--resume`` removes finished entries before ``--max-pdbs`` caps what is
    left; capping first would re-offer the same finished prefix forever.
    """
    ids, root, manual_inputs = _select_entry_ids(args, cache_root)
    run_log.details["entries_selected_before_resume"] = len(ids)
    run_log.details["resolved_input_root"] = root

    if args.resume:
        done_kwargs = {
            "bonds_required": args.bonds,
            "bond_output_present": os.path.isfile(layout.bonds),
            "candidate_output_present": os.path.isfile(layout.candidates),
        }
        normally_done = load_done(layout.manifest, **done_kwargs)
        if args.retry_partials:
            done = load_done(layout.manifest, retry_partial_ids=ids, **done_kwargs)
            reselected = normally_done - done
            run_log.details["terminal_partials_reselected"] = len(reselected)
            logger.info(
                "selected %d terminal partial entr%s for retry",
                len(reselected),
                "y" if len(reselected) == 1 else "ies",
            )
        else:
            done = normally_done
        ids = [i for i in ids if i not in done]
    if args.max_pdbs is not None:
        ids = ids[: args.max_pdbs]
    run_log.details["entries_scheduled"] = len(ids)
    return ids, root, manual_inputs


def _finalize_confidence_reference(layout):
    """Score the streamed inputs and freeze the database reference."""
    try:
        return finalize_database_confidence(
            layout.confidence_inputs, layout.confidence_scores, layout.reference_dir
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError(f"Confidence finalization failed: {exc}") from None


def _finish_without_entries(args, layout, plan):
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
        return 0
    logger.info("no entries to process")
    return 0


def _clear_stale_outputs(args, layout, plan):
    """Remove output from a previous run that this one is about to contradict."""
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
    if args.resume:
        return
    # The reference metadata file is the reference's completion marker.
    reference_marker = os.path.join(layout.reference_dir, REFERENCE_METADATA_FILE)
    stale: Tuple[str, ...]
    if plan.mode == "database":
        stale = (layout.confidence_scores, reference_marker)
    elif plan.mode == "reference":
        stale = (layout.confidence_inputs,)
    else:
        stale = (layout.confidence_inputs, layout.confidence_scores, reference_marker)
    try:
        for path in stale:
            if os.path.isfile(path):
                os.unlink(path)
    except OSError as exc:
        raise DriverError(f"Could not clear stale confidence output: {exc}") from None


def _choose_worker_count(args, entry_count, run_log):
    """Size the pool, never above the number of entries there are to run.

    A Pool creates every worker up front, and under the spawn start method each
    one re-imports gemmi into its own interpreter.
    """
    if args.workers is None:
        cpu_limit, memory_limit = automatic_worker_limits()
        automatic_limit = cpu_limit
        if memory_limit is not None:
            automatic_limit = min(automatic_limit, memory_limit)
        workers = min(automatic_limit, entry_count)
        run_log.details["worker_selection"] = "automatic"
        run_log.details["CPU worker limit"] = cpu_limit
        run_log.details["Memory worker limit"] = (
            memory_limit if memory_limit is not None else "unavailable"
        )
        run_log.details["Selected workers"] = workers
        logger.info("automatic worker selection:")
        logger.info("  CPU worker limit: %s", cpu_limit)
        logger.info(
            "  memory worker limit: %s",
            memory_limit if memory_limit is not None else "unavailable",
        )
        logger.info("  selected workers: %s", workers)
    else:
        workers = min(args.workers, entry_count)
        run_log.details["worker_selection"] = "explicit"
        run_log.details["Selected workers"] = workers
    logger.info(
        "processing %d entr%s with %d worker(s)",
        entry_count,
        "y" if entry_count == 1 else "ies",
        workers,
    )
    return workers


def _worker_config(
    args, env, root, cache_root, cofactors, manual_inputs, plan, run_log
):
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
        alchemy_commit=_alchemy_commit(),
        gemmi_version=_gemmi_version(),
        ccp4_version=_ccp4_version(env),
        reference_data_id=reference_data_id(),
    )
    run_log.details.update(
        alchemy_version=ALCHEMY_VERSION,
        gemmi_version=cfg.gemmi_version,
        ccp4_version=cfg.ccp4_version,
        confidence_mode=plan.mode or "disabled",
        reference_data_id=cfg.reference_data_id,
        # The manifest carries one combined id; the per-file digests here are
        # what attributes a change in it to a file.
        **{
            f"{name.split('.')[0]}_sha256": digest
            for name, digest in reference_data_checksums().items()
        },
    )
    return cfg


def _output_targets(args, layout, plan):
    """The output files this run writes, in the order staging expects them."""
    paths = [*layout.core]
    if plan.enabled:
        if plan.stream_path is None:
            raise RuntimeError("confidence output path is not configured")
        paths.append(plan.stream_path)
    if plan.synchronize_inputs:
        paths.append(layout.confidence_inputs)
    return tuple(paths)


def _open_writers(handles, args, plan, write_paths) -> _OutputWriters:
    """Open every output stream this run writes and give them their headers."""
    manifest_path, stats_path, bonds_path, candidates_path = write_paths[:4]
    confidence_path = write_paths[4] if plan.enabled else None
    confidence_inputs_path = write_paths[5] if plan.synchronize_inputs else None

    def opened(path):
        return handles.enter_context(open(path, "w", newline=""))

    return _OutputWriters(
        opened(manifest_path),
        opened(stats_path),
        opened(bonds_path) if args.bonds else None,
        opened(candidates_path) if args.bonds else None,
        opened(confidence_path) if confidence_path is not None else None,
        plan.columns,
        opened(confidence_inputs_path) if confidence_inputs_path is not None else None,
    )


def _confidence_rows_for(result, plan):
    """The confidence rows one entry contributes, scored where a reference is."""
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
    return rows


def _write_entry(result, args, plan, writers, staging, prior_counts):
    """Write one entry's rows, manifest row last.

    The manifest row is the entry's completion marker, so an interruption
    between it and the data rows costs a repeat on the next resume, not a loss.
    """
    prior_bond_counts, prior_candidate_counts = prior_counts
    writers.write_stats_rows(result.rows)
    writers.write_bond_rows(result.bond_rows)
    writers.write_candidate_rows(result.candidate_rows)
    if plan.enabled:
        writers.write_confidence_rows(_confidence_rows_for(result, plan))
    writers.write_manifest_row(
        _manifest_row(
            result, args.resume, args.bonds, prior_bond_counts, prior_candidate_counts
        )
    )
    if staging is not None:
        staging.replacement_ids.add(result.pdb_id.lower())


class _WorkerDeathWatch:
    """Stands in for the pool on the results a dead worker will never deliver.

    Workers announce the entry they are holding, so a pid that leaves the pool
    roster is usually attributable and its entry can be failed retryably at
    once. A worker killed before it announced anything leaves no such trace:
    that death is only counted, and is settled by ``stalled_losses`` once the
    run has gone quiet for long enough to rule out a live worker.
    """

    def __init__(self, pool, inflight, ids, cfg):
        self._pool = pool
        self._inflight = inflight
        self._ids = ids
        self._cfg = cfg
        self._assignments: Dict[int, str] = {}
        self._worker_pids: set = set()
        self._lost_ids: set = set()
        self._unattributed_deaths = 0
        # ``Pool`` has started its workers before its constructor returns, and
        # no tasks are submitted until after this watch is created. Snapshot
        # that original roster now: if the first task kills its worker before
        # the result loop's first poll, its vanished pid must still be known.
        _dead_worker_pids(self._pool, self._worker_pids)

    def poll(self):
        """Results for the entries whose worker has died since the last call."""
        _drain_inflight(self._inflight, self._assignments)
        losses = []
        for dead_pid in _dead_worker_pids(self._pool, self._worker_pids):
            dead_id = self._assignments.pop(dead_pid, None)
            if dead_id is None:
                self._unattributed_deaths += 1
            elif dead_id not in self._lost_ids:
                losses.append(self._lose(dead_id, dead_pid))
        return losses

    def stalled_losses(self, completed_ids, stalled_for, remaining):
        """Results for the outstanding entries an unattributed death held.

        ``None`` until every outstanding entry can be accounted for by a death
        that named no entry and the run has been quiet for the grace period,
        which leaves the caller waiting rather than failing live work.
        """
        if not (
            self._unattributed_deaths
            and stalled_for > WORKER_STALL_GRACE_S
            and remaining <= self._unattributed_deaths
        ):
            return None
        # Only entries that never returned a result can still be held by a
        # process that died before naming its entry; blaming a completed one
        # would write a second, failed row for an entry that succeeded.
        losses = []
        for stuck_id in self._ids:
            if stuck_id in self._lost_ids or stuck_id in completed_ids:
                continue
            losses.append(self._lose(stuck_id, 0))
            if len(losses) >= remaining:
                break
        return losses

    def superseded(self, result):
        """True when a synthesized loss row already stands for this entry."""
        died = result.reason_codes == ["worker_process_died"]
        return result.pdb_id in self._lost_ids and not died

    def _lose(self, pdb_id, pid):
        self._lost_ids.add(pdb_id)
        return _worker_death_result(pdb_id, self._cfg, pid)


def _dispatch_entries(
    args, ids, cfg, workers, writers, plan, staging, prior_counts, run_log
):
    """Run every entry across a worker pool and write the results as they land.

    The loop polls rather than iterating the pool's results, because waiting on
    the pool alone would hang forever on an entry a killed worker was holding.
    """
    tally = _BatchTally()
    progress = _ProgressReporter(len(ids))
    inflight: SimpleQueue[Any] = SimpleQueue()
    completed_ids: set = set()
    last_progress = time.monotonic()
    # Not a ``with`` block: ``Pool.__exit__`` calls the same ``terminate`` a
    # killed idle worker can wedge, so the ``finally`` below bounds shutdown.
    # The log listener thread starts only after the pool: forking a process
    # that already has running threads risks the child deadlocking on a lock no
    # thread there holds.
    log_queue = create_worker_log_queue()
    pool = Pool(workers, initializer=_init_worker, initargs=(cfg, inflight, log_queue))
    log_listener = start_worker_log_listener(log_queue)
    deaths = _WorkerDeathWatch(pool, inflight, ids, cfg)
    try:
        results = pool.imap_unordered(process, ids, chunksize=1)
        completed = 0
        progress.render(completed, tally.counts, tally.no_metals, force=True)
        while completed < len(ids):
            batch = []
            try:
                batch.append(results.next(timeout=1.0))
            except MultiprocessingTimeoutError:
                pass
            batch.extend(deaths.poll())
            if not batch:
                losses = deaths.stalled_losses(
                    completed_ids,
                    time.monotonic() - last_progress,
                    len(ids) - completed,
                )
                if losses is None:
                    progress.render(completed, tally.counts, tally.no_metals)
                    continue
                batch = losses
            last_progress = time.monotonic()
            for r in batch:
                if deaths.superseded(r):
                    continue
                completed += 1
                completed_ids.add(r.pdb_id)
                run_log.record_entry(r)
                if not args.resume or _resume_replacement_succeeded(r):
                    _write_entry(r, args, plan, writers, staging, prior_counts)
                tally.record(r)
                finished = completed == len(ids)
                progress.render(
                    completed,
                    tally.counts,
                    tally.no_metals,
                    force=progress.terminal or finished,
                    final=finished,
                )
    finally:
        forced = _shutdown_pool(pool)
        # Stopped after the pool is gone, so records emitted during shutdown
        # are still forwarded.
        log_listener.stop()
        log_queue.close()
        if forced:
            run_log.summary["worker_pool_forced_shutdown"] = True
            logger.warning(
                "a worker pool shutdown had to be forced after a worker died "
                "holding the task-queue lock; results above are complete"
            )
        progress.close()
    return tally


def _process_entries(args, ids, cfg, workers, layout, plan, run_log):
    """Open the outputs, run the batch, and commit any staged retry rows.

    Staging is committed only once the batch has run to completion, so an
    interrupted retry leaves the previous run's rows exactly as they were.
    """
    prior_counts = (
        _manifest_values_by_id(layout.manifest, "n_bonds")
        if args.resume and not args.bonds
        else {},
        _manifest_values_by_id(layout.manifest, "n_candidates")
        if args.resume and not args.bonds
        else {},
    )
    output_paths = _output_targets(args, layout, plan)
    staging = _ResumeStaging(args.output_dir, output_paths) if args.resume else None
    write_paths = staging.staged if staging is not None else output_paths

    writers: Optional[_OutputWriters] = None
    processing_completed = False
    try:
        with contextlib.ExitStack() as handles:
            writers = _open_writers(handles, args, plan, write_paths)
            tally = _dispatch_entries(
                args,
                ids,
                cfg,
                workers,
                writers,
                plan,
                staging,
                prior_counts,
                run_log,
            )
            processing_completed = True
            # Bound here so the summary below needs no ``Optional`` narrowing.
            opened_writers = writers
    finally:
        if staging is not None and not processing_completed:
            staging.discard()

    run_log.summary.update(
        metal_rows_written=opened_writers.n_rows,
        bond_rows_written=opened_writers.n_bonds,
        candidate_rows_written=opened_writers.n_candidates,
        confidence_rows_written=opened_writers.n_confidence,
        manifest_path=layout.manifest,
        metal_stats_path=layout.stats,
        metal_bonds_path=layout.bonds if args.bonds else "disabled",
        metal_candidates_path=layout.candidates if args.bonds else "disabled",
    )
    if staging is not None:
        try:
            staging.commit(args.bonds, confidence_enabled=plan.enabled)
        finally:
            staging.discard()
    return tally, opened_writers


def _report_batch(args, layout, plan, tally, writers, run_log):
    """Print the human summary, finalize confidence, and return the exit code.

    Confidence is finalized only on a clean batch: a reference built from
    incomplete entries becomes the cohort every later run is scored against.
    """
    print(
        f"Done. ok={tally.counts['ok']} partial={tally.counts['partial']} "
        f"skip={tally.counts['skip']} error={tally.counts['error']} "
        f"no_metals={tally.no_metals}; "
        f"{writers.n_rows} metal/cofactor rows -> {layout.stats}",
        flush=True,
    )
    if args.bonds:
        print(f"      {writers.n_bonds} bond rows -> {layout.bonds}", flush=True)
        print(
            f"      {writers.n_candidates} candidate rows -> {layout.candidates}",
            flush=True,
        )
    exit_code = tally.exit_code()
    if plan.mode == "database":
        if exit_code == 0:
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
            print(
                "      confidence inputs were retained, but the database "
                "reference was not finalized because the run is incomplete.",
                flush=True,
            )
    elif plan.mode == "reference":
        if plan.reference is None:
            raise RuntimeError("confidence reference is not configured")
        print(
            f"      {writers.n_confidence} confidence rows compared with "
            f"database cohort {plan.reference.cohort_size} -> "
            f"{layout.confidence_scores}",
            flush=True,
        )
        run_log.summary.update(
            confidence_status="scored_against_reference",
            confidence_reference_cohort=plan.reference.cohort_size,
            confidence_scores_path=layout.confidence_scores,
        )
    if exit_code:
        logger.warning(
            "completed with incomplete entries: errors=%d, skips=%d, "
            "retryable_partials=%d",
            tally.counts["error"],
            tally.counts["skip"],
            tally.retryable_partials,
        )
    return exit_code


def _run(args, run_log):
    """Execute one batch, returning its exit code."""
    try:
        return _execute(args, run_log)
    except DriverError as exc:
        run_log.driver_error = str(exc)
        logger.error("%s", exc)
        return 1


def _execute_with_output_lock(args, run_log, cofactors, env):
    """Run every output-reading and output-writing phase under one lease."""
    _sweep_leaked_work_dirs(args.output_dir)
    layout = _OutputLayout(args.output_dir)
    run_mode, database_run = _classify_run(args)
    run_log.details["run_mode"] = run_mode

    plan = _plan_confidence(args, layout, database_run, run_log)
    _check_resume_is_compatible(args, layout, plan)

    ids, root, manual_inputs = _schedule_entries(
        args, layout, args.pdb_redo_cache, run_log
    )
    if not ids:
        return _finish_without_entries(args, layout, plan)

    _clear_stale_outputs(args, layout, plan)
    workers = _choose_worker_count(args, len(ids), run_log)
    cfg = _worker_config(
        args, env, root, args.pdb_redo_cache, cofactors, manual_inputs, plan, run_log
    )

    tally, writers = _process_entries(args, ids, cfg, workers, layout, plan, run_log)
    return _report_batch(args, layout, plan, tally, writers, run_log)


def _execute(args, run_log):
    """Resolve prerequisites, then exclusively own the output for the run."""
    cofactors = _load_cofactor_catalog()
    env, _ = resolve_ccp4_environment(args)
    if env is None:
        return 0  # --configure-ccp4 saved a setup path and ran nothing else.
    _prepare_output_directory(args.output_dir)
    try:
        with OutputDirectoryLock(args.output_dir, run_log.command):
            return _execute_with_output_lock(args, run_log, cofactors, env)
    except (OutputDirectoryBusyError, OutputDirectoryLockError) as exc:
        raise DriverError(str(exc)) from None
