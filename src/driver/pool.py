"""The batch driver: everything between the parsed arguments and the workers.

One entrypoint, ``_run``, which is a handler around ``_execute`` -- an
orchestrator that names the phases of a run in order and does nothing else.
Each phase below either produces the value the next one needs or raises
``_DriverError`` carrying the message the user should see. That is what makes
the shape of a batch readable: it is the sequence of calls in ``_execute``, not
a 600-line function whose phases were visible only as blank lines.

Also here: the driver's half of the worker protocol. ``multiprocessing.Pool``
silently replaces a worker that died and never delivers a result for the task
it was holding, so the dispatch loop cannot wait on the pool alone. It drains
the notifications workers send (``_drain_inflight``), watches the pool roster
(``_dead_worker_pids``), and bounds shutdown explicitly (``_shutdown_pool``),
because a worker killed while holding the task-queue lock never releases it.
"""

import contextlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from multiprocessing import (
    Pool,
    SimpleQueue,
    TimeoutError as MultiprocessingTimeoutError,
)
from typing import Dict, Optional, Sequence

from _version import __version__
from reference_data import cofactor_ids
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


# ``REPO_DIR`` is imported rather than recomputed. The two-``dirname`` walk it
# is built from is relative to the file that runs it, so repeating it here --
# one directory deeper than ``src/`` -- would silently name ``src/`` as the
# checkout root. One definition also settles the duplicate ``ccp4_setup`` and
# the driver used to keep side by side.
DEFAULT_ROOT = "/datasets/bioinfo/pdb-redo"
DEFAULT_CONFIDENCE_REFERENCE_DIR = os.path.join(REPO_DIR, "confidence_reference")

ALCHEMY_VERSION = __version__
# Seconds of no completed entry, after a worker died without naming the entry
# it held, before the remaining outstanding entries are failed retryably.
WORKER_STALL_GRACE_S = 600.0
# Seconds to let a worker pool shut down cleanly before its children are killed
# outright. Every result has already been collected by then and the workers are
# idle, so a healthy pool finishes this in milliseconds and never approaches the
# deadline; it exists only to bound the hang described in ``_shutdown_pool``.
WORKER_SHUTDOWN_GRACE_S = 5.0

# Budget for the `git` probes that stamp run provenance. These read a local
# repository and are expected to finish in milliseconds; a second is generous.
# Exceeding it is not an entry failure -- `_alchemy_commit` already degrades to
# "unknown" for any subprocess error -- so a stuck index lock costs the commit
# hash rather than the run.
PROVENANCE_COMMAND_TIMEOUT_S = 1


logger = logger_for(__name__)


# --------------------------------------------------------------------------- #
# CCP4 environment -- the CLI boundary over ccp4_setup
# --------------------------------------------------------------------------- #
def _verify_resolved_ccp4(env, setup_path):
    """Verify ``env``, naming the script that was run when it comes up short.

    Sourcing a setup script successfully and still not finding the programs is
    a different diagnosis from never having found a script at all: the path is
    real but points at an incomplete or wrong installation.
    """
    try:
        verify_ccp4(env)
    except Ccp4SetupError as exc:
        raise Ccp4SetupError(
            f"Ran {setup_path}, but CCP4 tools are still not available. {exc}"
        ) from None


def _resolve_ccp4_environment(args):
    """Resolve the CCP4 environment, raising ``Ccp4SetupError`` on any failure.

    Split from the ``SystemExit`` wrapper below so the whole decision -- the
    configure branch, the explicit override, the ambient environment and
    auto-detection -- reports failure the one way ``ccp4_setup`` does.

    Which files hold the saved setup path is deliberately not named here.
    ``ccp4_setup.DEFAULT_CONFIG_FILES`` is the one definition, and a second
    list in the driver is exactly the drift that made ``--configure-ccp4``
    report success and then be ignored.
    """
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

    # An explicit --ccp4-setup is checked before the ambient environment, not
    # after. A user passing it is overriding whatever the shell already has --
    # typically because the wrong CCP4 version is sourced -- so honouring PATH
    # first would silently run against the very installation they were
    # replacing, and record that installation's version as the run's
    # provenance. A path that does not exist is an error for the same reason:
    # falling through would make a typo indistinguishable from success.
    if args.ccp4_setup:
        setup_path = os.path.abspath(os.path.expanduser(args.ccp4_setup))
        if not os.path.exists(setup_path):
            # The user's own spelling, not the expanded path, so the message
            # shows the typo they can see in their command line.
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
    """Return ``(env, setup_path)`` for this run, or exit with a diagnostic.

    The CLI boundary. ``ccp4_setup`` raises ``Ccp4SetupError`` so that CCP4
    resolution stays usable outside a command-line process; this is the single
    place that turns one into the ``SystemExit`` a run expects.
    """
    try:
        return _resolve_ccp4_environment(args)
    except Ccp4SetupError as exc:
        raise SystemExit(str(exc)) from None


# --------------------------------------------------------------------------- #
# Run provenance -- stamped once by the driver and copied into every result
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Worker-pool supervision -- the driver's half of the protocol in ``worker``
# --------------------------------------------------------------------------- #
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
    """Return worker pids that have disappeared since the last check.

    ``Pool`` silently replaces a worker that died, so a pid leaving the pool's
    roster is the only signal that its task will never produce a result.
    """
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
    """Remove per-entry scratch directories a previous run left behind.

    Each entry works inside ``<output-dir>/.alchemy-<id>-XXXX`` and deletes it
    once its rows are extracted, but a run that is interrupted -- Ctrl-C,
    SIGTERM, an out-of-memory kill -- never reaches that cleanup, and the
    directory survives holding that entry's maps. Nothing removed them, so they
    accumulated over every interrupted attempt.

    Sweeping at startup is safe because the directories belong to a run that is
    no longer executing: a second Alchemy process sharing one ``--output-dir``
    would already be overwriting the first one's manifest and CSVs.

    Returns the number of directories removed.
    """
    removed = 0
    try:
        names = os.listdir(output_dir)
    except OSError:
        return 0
    for name in names:
        if not name.startswith((".alchemy-", ".alchemy-resume-")):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path) or os.path.islink(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += not os.path.exists(path)
    return removed


def _shutdown_pool(pool):
    """Close a worker pool without letting a dead worker hang the driver.

    A pool worker waiting for its next task blocks inside the task queue's
    ``get()`` while holding that queue's lock. A process killed there -- by the
    out-of-memory killer, or by a crash in a compiled extension -- never
    releases it, and ``Pool.terminate`` in turn blocks acquiring the same lock.
    Shutdown then hangs *after* every result has already been collected and
    written, so the batch is complete but the run log, confidence finalization
    and exit code never happen.

    Run the clean shutdown on a side thread and give it a deadline. If it does
    not return, cancel the finalizer that would repeat the same wait at
    interpreter exit and kill the remaining children directly. The thread is a
    daemon, so an abandoned one cannot keep the process alive.

    Returns ``True`` when shutdown had to be forced.
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
    # The lock belongs to a process that is already gone, so killing the
    # remaining children cannot release it and the thread will not return.
    # Abandon it: a daemon thread does not keep the interpreter alive.
    return True


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def resolve_confidence_reference_dir(output_dir, configured_dir=None):
    """Find a frozen confidence reference, honoring an explicit override."""
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


class _DriverError(Exception):
    """A user-facing driver failure: the message is printed and the run exits 1."""


def _select_entry_ids(args, cache_root):
    """Resolve the run's work list, returning ``(ids, root, manual_inputs)``.

    Raises ``_DriverError`` when the request cannot be satisfied at all.
    """
    root = args.pdb_redo_root
    if args.pdb_file or args.mtz_file or args.cif_file:
        pdb_id = (
            args.id
            or infer_pdb_id_from_path(args.cif_file)
            or infer_pdb_id_from_path(args.pdb_file)
            or infer_pdb_id_from_path(args.mtz_file)
        )
        if not pdb_id:
            raise _DriverError(
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
        # Ensure requested single entry is available locally (mirror or cache).
        try:
            used_root = ensure_entry_available(args.id, args.pdb_redo_root, cache_root)
        except FileNotFoundError:
            raise _DriverError(
                f"Entry {args.id} not found locally and download failed."
            ) from None
        if used_root != args.pdb_redo_root:
            logger.info("auto-downloaded %s into cache at %s", args.id, cache_root)
        return [args.id], used_root, None

    if args.id_file:
        try:
            ids = load_ids_from_file(args.id_file)
        except (FileNotFoundError, ValueError) as exc:
            raise _DriverError(str(exc)) from None
        logger.info("loaded %d IDs from %s", len(ids), args.id_file)
        return ids, root, None

    logger.info("enumerating final PDB-REDO entries under %s", root)
    # Early-stop only when capping and not resuming (resume needs the full set).
    limit = args.max_pdbs if (args.max_pdbs and not args.resume) else None
    return enumerate_entries(root, limit=limit), root, None


# --------------------------------------------------------------------------- #
# Run phases
# --------------------------------------------------------------------------- #
# ``_run`` below is an orchestrator: it names the phases in order and does
# nothing else. Each phase is a function here, and each one either produces the
# value the next phase needs or raises ``_DriverError`` with the message the
# user should see. That is what makes the order of a run readable -- the shape
# of a batch is the sequence of calls in ``_run``, not a 600-line function whose
# phases are only visible as blank lines.


class _OutputLayout:
    """Every path a run reads or writes, all derived from ``--output-dir``.

    Grouped because they travel together through every later phase, and because
    a run that resumes has to name the *same* files a previous run wrote.
    """

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
    def core(self):
        """The four always-written outputs, in resume-validation order."""
        return (self.manifest, self.stats, self.bonds, self.candidates)


class _ConfidencePlan:
    """Whether this run scores confidence, and against what.

    Three states, decided once before any entry runs. ``database`` streams the
    inputs an uncapped full-database run finalizes into a new reference at the
    end. ``reference`` scores each entry against a reference that already
    exists. ``None`` means the bond stage is off, or no reference was found --
    the expected state on a fresh checkout, since none is distributed.
    """

    #: The two names ``mode`` takes when scoring is on. Annotated rather than
    #: left to inference: every field starts at ``None`` for the disabled case,
    #: so an unannotated ``self.mode = None`` types the attribute as ``None``
    #: and makes every later assignment an error.
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
        status = result["status"]
        self.counts[status] = self.counts.get(status, 0) + 1
        if result.get("no_metals", False):
            self.no_metals += 1
        if status == "partial" and result.get("retryable", False):
            self.retryable_partials += 1

    def exit_code(self):
        return _batch_exit_code(self.counts, self.retryable_partials)


def _load_cofactor_catalog():
    """Read the bundled metallocofactor catalog, or fail the run naming it."""
    try:
        return cofactor_ids()
    except (OSError, UnicodeError, ValueError) as exc:
        raise _DriverError(f"Invalid bundled metallocofactor catalog: {exc}") from None


def _prepare_output_directory(output_dir):
    """Create ``--output-dir`` and remove scratch a previous run abandoned."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        # A read-only mount or someone else's directory is a fixable user
        # mistake, so it exits the way every other unusable input does rather
        # than as a traceback that reads like an Alchemy bug.
        raise SystemExit(
            f"Cannot use --output-dir {output_dir}: {exc.strerror or exc}"
        ) from None
    _sweep_leaked_work_dirs(output_dir)


def _classify_run(args):
    """Return ``(run_mode, database_run)`` for this invocation.

    ``database_run`` is the uncapped full-mirror case, and it is the only one
    that may finalize a confidence reference: a capped or hand-picked run is
    not a cohort.
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
        # Expected on a fresh checkout: no reference is distributed with
        # Alchemy because the confidence score is not finalized. Say so
        # plainly -- naming the searched directories alone read as a
        # misconfiguration the user was supposed to fix.
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
        raise _DriverError(f"Invalid confidence reference: {exc}") from None
    run_log.details["confidence_reference_dir"] = reference_dir
    plan.mode = "reference"
    plan.stream_path = layout.confidence_scores
    plan.columns = (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS)
    return plan


def _check_resume_is_compatible(args, layout, plan):
    """Refuse to resume onto output this run cannot safely extend.

    Every check here is about appending rows beneath a header that no longer
    means what it did. Sets ``plan.synchronize_inputs`` as a side effect, since
    whether the streamed inputs are also being extended is part of the answer.
    """
    if not args.resume:
        return
    if plan.enabled and (
        plan.stream_path is None or not os.path.isfile(plan.stream_path)
    ):
        raise _DriverError(
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
        raise _DriverError(str(exc)) from None
    if plan.mode == "reference":
        try:
            validate_scored_reference(plan.stream_path, plan.reference)
        except (OSError, ValueError) as exc:
            raise _DriverError(f"Cannot resume confidence output: {exc}") from None


def _schedule_entries(args, layout, cache_root, run_log):
    """Return ``(ids, root, manual_inputs)`` for the entries this run will do.

    The work list is selected first and then reduced: ``--resume`` removes what
    an existing manifest already reports as finished, and ``--max-pdbs`` caps
    what is left. That order matters -- capping first would let a resumed run
    keep re-offering the same finished prefix and never reach the tail.
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
        raise _DriverError(f"Confidence finalization failed: {exc}") from None


def _finish_without_entries(args, layout, plan):
    """Exit code for a run whose work list came back empty.

    A resumed database run with nothing left is the normal way a long batch
    ends: its last chunk finished, and finalizing the reference is the only
    remaining step.
    """
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
        raise _DriverError(f"Could not remove stale bond-stage output: {exc}") from None
    for removed_path in removed:
        logger.info("removed stale bond-stage output: %s", removed_path)
    if args.resume:
        return
    # Reference metadata is its completion marker. Fresh runs must not leave
    # incompatible confidence artifacts beside newly replaced core output.
    reference_marker = os.path.join(layout.reference_dir, REFERENCE_METADATA_FILE)
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
        raise _DriverError(f"Could not clear stale confidence output: {exc}") from None


def _choose_worker_count(args, entry_count, run_log):
    """Size the pool, never above the number of entries there are to run.

    A Pool creates every worker up front, so asking for more than there are
    entries just pays each one's startup for no work. Costly under the spawn
    start method, where each worker re-imports gemmi into its own interpreter.
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
    )
    run_log.details.update(
        alchemy_version=ALCHEMY_VERSION,
        gemmi_version=cfg.gemmi_version,
        ccp4_version=cfg.ccp4_version,
        confidence_mode=plan.mode or "disabled",
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


def _open_writers(handles, args, plan, write_paths):
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
        result["rows"], result["bond_rows"], STATS_COLUMNS
    )
    rows = complete_confidence_site_count(
        rows,
        result["pdbID"],
        result["n"],
        result.get("confidence_inputs_missing_reason", ""),
    )
    # Equivalent to `plan.mode == "reference"`: the mode is set only where a
    # reference has just been loaded. Testing the reference itself says what
    # the call actually needs.
    if plan.reference is not None:
        rows = score_against_reference(rows, plan.reference)
    return rows


def _write_entry(result, args, plan, writers, staging, prior_counts):
    """Write one entry's rows, manifest row last.

    The manifest is the completion marker, so it is written only after this
    entry's data rows have flushed: an interruption between the two costs a
    repeated entry on the next resume, never a lost one.
    """
    prior_bond_counts, prior_candidate_counts = prior_counts
    writers.write_stats_rows(result["rows"])
    writers.write_bond_rows(result["bond_rows"])
    writers.write_candidate_rows(result["candidate_rows"])
    if plan.enabled:
        writers.write_confidence_rows(_confidence_rows_for(result, plan))
    writers.write_manifest_row(
        _manifest_row(
            result, args.resume, args.bonds, prior_bond_counts, prior_candidate_counts
        )
    )
    if staging is not None:
        staging.replacement_ids.add(result["pdbID"].lower())


def _dispatch_entries(
    args, ids, cfg, workers, writers, plan, staging, prior_counts, run_log
):
    """Run every entry across a worker pool and write the results as they land.

    Returns the batch tally. The loop does three things at once, which is why
    it does not simply iterate the pool's results: it collects finished entries,
    it watches the pool roster for workers that died without returning anything,
    and it renders progress. A killed worker never delivers a result, so waiting
    on the pool alone would hang the batch forever on the entry it was holding.
    """
    tally = _BatchTally()
    progress = _ProgressReporter(len(ids))
    inflight = SimpleQueue()
    assignments: Dict[int, str] = {}
    worker_pids: set = set()
    lost_ids: set = set()
    completed_ids: set = set()
    unattributed_deaths = 0
    last_progress = time.monotonic()
    # Not a ``with`` block: ``Pool.__exit__`` calls the same ``terminate`` that
    # a killed idle worker can wedge forever, so shutdown is bounded explicitly
    # in the ``finally`` below.
    # Worker records travel over this queue and are re-emitted by the driver, so
    # only one process ever writes to a handler. The queue is created before the
    # pool but the listener thread is started after it: forking a process that
    # already has running threads risks the child deadlocking on a lock no
    # thread there holds.
    log_queue = create_worker_log_queue()
    pool = Pool(workers, initializer=_init_worker, initargs=(cfg, inflight, log_queue))
    log_listener = start_worker_log_listener(log_queue)
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
            # A killed worker never delivers a result, so its entry is
            # recovered from the pool roster rather than waited on.
            _drain_inflight(inflight, assignments)
            for dead_pid in _dead_worker_pids(pool, worker_pids):
                dead_id = assignments.pop(dead_pid, None)
                if dead_id is None:
                    unattributed_deaths += 1
                elif dead_id not in lost_ids:
                    lost_ids.add(dead_id)
                    batch.append(_worker_death_result(dead_id, cfg, dead_pid))
            if not batch:
                stalled = time.monotonic() - last_progress
                remaining = len(ids) - completed
                if (
                    unattributed_deaths
                    and stalled > WORKER_STALL_GRACE_S
                    and remaining <= unattributed_deaths
                ):
                    # Every entry still outstanding can only be held by a
                    # process that died before naming its entry. Entries that
                    # already returned a result are not outstanding: blaming
                    # one of those would write a second, failed row for an
                    # entry that succeeded and leave the entry that actually
                    # died unreported.
                    for stuck_id in ids:
                        if stuck_id in lost_ids or stuck_id in completed_ids:
                            continue
                        lost_ids.add(stuck_id)
                        batch.append(_worker_death_result(stuck_id, cfg, 0))
                        if len(batch) >= remaining:
                            break
                else:
                    progress.render(completed, tally.counts, tally.no_metals)
                    continue
            last_progress = time.monotonic()
            for r in batch:
                if r["pdbID"] in lost_ids and r.get("reason_codes") != [
                    "worker_process_died"
                ]:
                    # A real result arrived for an entry already declared lost.
                    # The synthesized row stands, so this one is dropped rather
                    # than written twice.
                    continue
                completed += 1
                completed_ids.add(r["pdbID"])
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
        # Stopped only after the pool is gone, so records emitted during
        # shutdown are still forwarded.
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

    Returns ``(tally, writers)``. Staging is discarded rather than committed
    unless the batch ran to completion, so an interrupted retry leaves the
    previous run's rows exactly as they were.
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

    writers = None
    processing_completed = False
    try:
        # ExitStack closes whichever handles were opened, so a failure partway
        # through opening them cannot leak the earlier ones.
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
    finally:
        if staging is not None and not processing_completed:
            staging.discard()

    run_log.summary.update(
        metal_rows_written=writers.n_rows,
        bond_rows_written=writers.n_bonds,
        candidate_rows_written=writers.n_candidates,
        confidence_rows_written=writers.n_confidence,
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
    return tally, writers


def _report_batch(args, layout, plan, tally, writers, run_log):
    """Print the human summary, finalize confidence, and return the exit code.

    Confidence is finalized only on a clean batch: a reference built from a run
    with incomplete entries would silently become the cohort every later run is
    scored against.
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
    except _DriverError as exc:
        run_log.driver_error = str(exc)
        logger.error("%s", exc)
        return 1


def _execute(args, run_log):
    """The order of a run, phase by phase."""
    cofactors = _load_cofactor_catalog()
    env, _ = resolve_ccp4_environment(args)
    if env is None:
        return 0  # --configure-ccp4 saved a setup path and ran nothing else.
    _prepare_output_directory(args.output_dir)

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
