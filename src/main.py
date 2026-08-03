#!/usr/bin/env python
"""Batch-run the Alchemy core pipeline over PDB-REDO entries.

For each PDB entry this validates/corrects its Fourier map coefficients with
CCP4 `mtzfix`, computes 2mFo-DFc and mFo-DFc maps with `fft`, and runs
`edstats`, then extracts per-atom real-space statistics for metal ions and
metal-containing cofactors. Core results are streamed to four CSVs under
--output-dir:

  metal_stats_all.csv  -- one row per selected metal site
  metal_bonds_all.csv  -- one row per inferred or declared contact
  metal_candidates_all.csv -- one row per discovered or declared candidate
  manifest.csv         -- one row per entry with status and provenance

An uncapped database run additionally streams compact confidence inputs and,
after successful completion, writes final confidence scores plus a reusable
database reference. Smaller runs use an installed frozen reference when one is
available.

Requirements
------------
* CCP4 `mtzfix`, `fft`, `mapmask`, and `edstats` on PATH -- either already
  sourced, or via --ccp4-setup pointing at a CCP4 setup script
  (e.g. <CCP4>/bin/ccp4.setup-sh).
* Run under Python 3.11+ with gemmi>=0.7.0 and numpy>=1.17. Both are
  required; gemmi does not install numpy.

Examples
--------
  python src/main.py --id 109m \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
  python src/main.py --max-pdbs 20 --workers 4 \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
"""

import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing import (
    Pool,
    SimpleQueue,
    TimeoutError as MultiprocessingTimeoutError,
    cpu_count,
)
from typing import Dict

from _version import __version__
from density_analysis import (
    CCP4_TOOL_TIMEOUT_S,
    DENSITY_MAP_SCOPES,
    MODEL_ENVELOPE_BORDER_ANGSTROM,
)
from metal_identification import (
    EDSTATS_COLUMNS,
    load_cofactor_ids,
)
from bond_analysis import (
    BOND_COLUMNS,
    CANDIDATE_COLUMNS,
    STATS_EXTRA_COLUMNS,
)
from run_logging import (
    configure_driver_logging,
    level_for_verbosity,
    worker_level,
    logger_for,
    create_worker_log_queue,
    start_worker_log_listener,
)
from ccp4_setup import (
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
    _check_row_schema,
    _init_worker,
    _worker_death_result,
    process,
)
from confidence_score import (
    ANALYSIS_COLUMNS as CONFIDENCE_ANALYSIS_COLUMNS,
    CONFIDENCE_INPUT_COLUMNS,
    REFERENCE_METADATA_FILE,
    finalize_database_confidence,
    complete_confidence_site_count,
    load_reference as load_confidence_reference,
    prepare_result_confidence_inputs,
    score_against_reference,
    validate_scored_reference,
)


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = "/datasets/bioinfo/pdb-redo"
DEFAULT_CONFIDENCE_REFERENCE_DIR = os.path.join(REPO_DIR, "confidence_reference")

ALCHEMY_VERSION = __version__
AUTO_WORKER_MEMORY_BYTES = 1280 * 1024 * 1024
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


# CSV column names keep the deposited-data spelling ``pdbID`` even though every
# Python identifier is ``pdb_id``. The columns are an external contract: users'
# scripts, notebooks and downstream joins address them by name, and renaming
# one would break those silently while gaining nothing. The two spellings meet
# only where a row dict is built, e.g. ``{"pdbID": pdb_id}``.
MANIFEST_COLUMNS = [
    "pdbID",
    "status",
    "retryable",
    "n_metals",
    "n_bonds",
    "n_candidates",
    "runtime_s",
    "reason_codes",
    "warning_codes",
    "error",
    "alchemy_version",
    "alchemy_commit",
    "gemmi_version",
    "ccp4_version",
    "refinement_state",
    "source_coordinate_format",
    "analysis_coordinate_format",
    "coordinate_conversion_performed",
    "source_coordinate_path",
    "analysis_coordinate_path",
    "model_policy",
    "input_model_count",
    "model_analyzed",
    "multi_model_structure",
    "altloc_policy",
    "symmetry_contact_policy",
]

# metal_stats_all.csv schema. The middle block is the EDSTATS residue table,
# whose column set and order `extract_metal_statistics` validates against
# EDSTATS_COLUMNS before emitting any row, so the full header is fixed. Defining
# it once keeps the written header and the --resume compatibility check from
# disagreeing about the columns between them.
STATS_COLUMNS = (
    ["pdbID", "category"]
    + list(EDSTATS_COLUMNS)
    + ["aa_geometry_coverage"]
    + list(STATS_EXTRA_COLUMNS)
)

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


def _install_termination_handler():
    """Route SIGTERM through the same unwind path as Ctrl-C.

    SIGTERM's default disposition kills the process immediately: no ``finally``
    runs, so the worker pool is never shut down and its children -- which are
    daemonic but not yet signalled -- are reparented to init and keep working,
    driving CCP4 subprocesses and holding their scratch directories open.
    Raising ``KeyboardInterrupt`` instead makes a scheduler stop or a plain
    ``kill`` unwind exactly like an interactive interrupt, so the pool is
    terminated and the run log is still written.

    Returns the previous handler, or ``None`` where SIGTERM cannot be trapped
    (a non-main thread, or a platform without it).
    """

    def _raise_interrupt(signum, frame):  # noqa: ARG001 - signal API
        raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGTERM, _raise_interrupt)
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        return None


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
def load_done(
    manifest_path,
    bonds_required=False,
    bond_output_present=True,
    candidate_output_present=True,
    retry_partial_ids=(),
):
    """PDB IDs whose requested result is terminal in an existing manifest.

    Blank ``n_bonds`` and ``n_candidates`` values mean bond analysis was
    disabled, while ``0`` means it ran successfully and found no rows of that
    type. When bonds are requested, a terminal density result with either blank
    count still needs processing. Absent bond or candidate CSVs also make
    bond-stage results incomplete. IDs in ``retry_partial_ids`` are removed
    from the done set only when their row is a non-retryable ``partial``;
    successful ``ok`` rows remain protected from accidental reprocessing.
    """
    retry_partial_ids = {
        str(pdb_id).strip().lower()
        for pdb_id in retry_partial_ids
        if str(pdb_id).strip()
    }
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and {"pdbID", "status"}.issubset(reader.fieldnames):
                for row in reader:
                    status = row.get("status", "").strip().lower()
                    retryable = row.get("retryable", "").strip().lower()
                    terminal_partial = status == "partial" and retryable in (
                        "false",
                        "0",
                        "no",
                    )
                    pdb_id = row.get("pdbID", "").strip().lower()
                    bonds_complete = not bonds_required or (
                        bond_output_present
                        and candidate_output_present
                        and row.get("n_bonds", "").strip() != ""
                        and row.get("n_candidates", "").strip() != ""
                    )
                    protected_terminal = status == "ok" or (
                        terminal_partial and pdb_id not in retry_partial_ids
                    )
                    if protected_terminal and bonds_complete and pdb_id:
                        done.add(pdb_id)
    return done


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


def _resume_replacement_succeeded(result):
    """Whether a retry produced a terminal result suitable for replacement."""
    status = str(result.get("status", "")).strip().lower()
    return status == "ok" or (
        status == "partial" and not bool(result.get("retryable", True))
    )


def _manifest_values_by_id(path, column):
    """Return one manifest column keyed by normalized PDB ID."""
    values = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return values
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = row.get("pdbID", "").strip().lower()
            if pdb_id:
                values[pdb_id] = row.get(column, "")
    return values


def _merge_csv_replacements(path, staged_path, pdb_ids):
    """Atomically replace selected IDs with rows from a completed staging file.

    The existing destination is never changed until the full replacement file
    has been written successfully. Rows for IDs absent from ``pdb_ids`` are
    copied verbatim.
    """
    replacement_ids = {pdb_id.lower() for pdb_id in pdb_ids}
    if not replacement_ids:
        return

    directory = os.path.dirname(os.path.abspath(path))
    original_mode = os.stat(path).st_mode if os.path.exists(path) else None
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory, text=True
    )
    try:
        # os.fdopen takes ownership of the descriptor and closes it on exit, so
        # it is closed here only if the wrapping itself failed. Closing it again
        # afterwards could target a descriptor the runtime has since reused.
        try:
            staged = os.fdopen(fd, "w", newline="")
        except BaseException:
            os.close(fd)
            raise
        destination_header = None
        with staged as dst:
            writer = csv.writer(dst)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, newline="") as src:
                    reader = csv.reader(src)
                    destination_header = next(reader, None)
                    if destination_header is not None:
                        writer.writerow(destination_header)
                    for row in reader:
                        if row and row[0].strip().lower() in replacement_ids:
                            continue
                        writer.writerow(row)

            if os.path.exists(staged_path) and os.path.getsize(staged_path) > 0:
                with open(staged_path, newline="") as staged:
                    reader = csv.reader(staged)
                    staged_header = next(reader, None)
                    if destination_header is None and staged_header is not None:
                        destination_header = staged_header
                        writer.writerow(staged_header)
                    elif (
                        staged_header is not None
                        and staged_header != destination_header
                    ):
                        raise ValueError(f"staged CSV schema does not match {path}")
                    for row in reader:
                        if row and row[0].strip().lower() in replacement_ids:
                            writer.writerow(row)
            dst.flush()
            os.fsync(dst.fileno())

        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _csv_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="") as handle:
        return next(csv.reader(handle), None)


def _batch_exit_code(counts, retryable_partial_count):
    """Return failure when one or more entries remain operationally incomplete."""
    incomplete = (
        counts.get("error", 0) + counts.get("skip", 0) + retryable_partial_count
    )
    return 1 if incomplete else 0


def validate_resume_schemas(
    manifest_path,
    stats_path,
    bonds_path,
    candidates_path,
    bonds_enabled=True,
    confidence_path=None,
    confidence_columns=None,
):
    """Refuse to append migration rows beneath an incompatible old header.

    Whole headers are compared, including the EDSTATS block of
    metal_stats_all.csv. Appending rows beneath a header from a different
    EDSTATS build would misalign every density column without any other
    symptom.
    """
    checks = [(manifest_path, MANIFEST_COLUMNS), (stats_path, STATS_COLUMNS)]
    if bonds_enabled:
        checks.extend(
            ((bonds_path, BOND_COLUMNS), (candidates_path, CANDIDATE_COLUMNS))
        )
    if confidence_path is not None:
        if confidence_columns is None:
            raise ValueError("confidence columns are required with a confidence output")
        checks.append((confidence_path, list(confidence_columns)))
    for path, expected in checks:
        header = _csv_header(path)
        if header is not None and header != expected:
            raise ValueError(
                f"Existing {os.path.basename(path)} uses an incompatible "
                "schema; choose a new --output-dir for this Gemmi migration "
                "run."
            )


def remove_stale_disabled_bond_outputs(paths, resume, bonds_enabled):
    """Remove previous bond-stage CSVs before a fresh disabled run.

    A non-resume run replaces the manifest and statistics outputs, so retaining
    older bond-stage files would falsely associate them with the new run.
    Resume mode is different: completed entries and their existing rows are
    retained.
    """
    if resume or bonds_enabled:
        return []
    removed = []
    for path in paths:
        if os.path.lexists(path):
            os.unlink(path)
            removed.append(path)
    return removed


def available_cpu_count():
    """Return the number of CPUs this process is actually permitted to use.

    ``multiprocessing.cpu_count()`` reports every logical CPU on the machine
    and ignores CPU affinity, so inside a container or under a scheduler
    allocation it can report far more than the job was granted -- defaulting a
    batch run to dozens of workers on a handful of cores. The affinity-aware
    interfaces are preferred where the platform provides them.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    if hasattr(os, "sched_getaffinity"):  # Linux
        try:
            count = len(os.sched_getaffinity(0))
        except OSError:
            count = 0
        if count:
            return count
    try:
        return cpu_count()
    except NotImplementedError:
        return 1


def available_memory_bytes():
    """Return currently available physical memory, or ``None`` if unknown.

    Available memory is used instead of total RAM so an automatic batch run
    does not compete with memory already committed to other applications. Keep
    this dependency-free because worker selection happens before the analysis
    environment is initialized.
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, ValueError):
            pass

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            if page_size > 0 and available_pages > 0:
                return int(page_size * available_pages)
        except (OSError, TypeError, ValueError):
            pass
    return None


def automatic_worker_limits():
    """Return the CPU and optional memory limits for automatic parallelism."""
    cpu_limit = max(1, available_cpu_count() - 2)
    available_memory = available_memory_bytes()
    memory_limit = None
    if available_memory is not None:
        memory_limit = max(1, available_memory // AUTO_WORKER_MEMORY_BYTES)
    return cpu_limit, memory_limit


def parse_pdb_id(value):
    if not re.fullmatch(r"[A-Za-z0-9]{4}", value):
        raise argparse.ArgumentTypeError(
            "PDB ID must contain exactly four alphanumeric characters"
        )
    return value.lower()


def positive_int(value):
    """Argparse type for integer options that must be at least one."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Batch Alchemy core pipeline over PDB-REDO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--id", type=parse_pdb_id, help="process a single PDB id (else batch the root)"
    )
    ap.add_argument(
        "--id-file", help="path to a file of PDB ids (comma- and/or newline-separated)"
    )
    ap.add_argument("--pdb-file", help="path to a local PDB file for manual input mode")
    ap.add_argument("--mtz-file", help="path to a local MTZ file for manual input mode")
    ap.add_argument(
        "--cif-file", help="path to a local mmCIF file for manual input mode"
    )
    ap.add_argument(
        "--data-json", help="optional path to a local data.json for manual input mode"
    )
    ap.add_argument(
        "--pdb-redo-root", default=DEFAULT_ROOT, help="root of the PDB-REDO mirror"
    )
    ap.add_argument(
        "--pdb-redo-cache",
        default=os.path.join(REPO_DIR, "pdb-redo-cache"),
        help="root of local cache for auto-downloaded PDB-REDO entries",
    )
    ap.add_argument(
        "--max-pdbs",
        type=positive_int,
        default=None,
        help="process only the first N entries (minimum: 1)",
    )
    ap.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help=(
            "number of worker processes (minimum: 1); by default Alchemy "
            "uses the lower CPU or available-memory limit"
        ),
    )
    ap.add_argument("--output-dir", default=os.path.join(REPO_DIR, "output"))
    ap.add_argument(
        "--density-map-scope",
        choices=DENSITY_MAP_SCOPES,
        default="model-envelope",
        help=(
            "map extent supplied to EDSTATS; model-envelope retains every "
            f"coordinate plus a {MODEL_ENVELOPE_BORDER_ANGSTROM} Angstrom "
            "border and falls back to full when cropping would be unsafe or "
            "larger"
        ),
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "increase diagnostic detail; -v adds per-entry and per-CCP4-program "
            "records from inside the workers"
        ),
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="report warnings and errors only, suppressing the run narrative",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help=(
            "also write full debug-level diagnostics to this file, whatever "
            "the console verbosity"
        ),
    )
    ap.add_argument(
        "--ccp4-timeout",
        type=positive_int,
        default=CCP4_TOOL_TIMEOUT_S,
        help=(
            "per-program wall-clock budget in seconds for each CCP4 step "
            "(mtzfix, fft, mapmask, edstats); raise it for exceptionally "
            "large structures"
        ),
    )
    ap.add_argument(
        "--confidence-reference-dir",
        default=None,
        help=(
            "explicit frozen full-database confidence reference for single, "
            "ID-file, manual, and capped runs; otherwise Alchemy searches "
            "the output directory and repository default"
        ),
    )
    ap.add_argument(
        "--ccp4-setup",
        default=None,
        help="optional CCP4 setup script override (e.g. .../bin/ccp4.setup-sh)",
    )
    ap.add_argument(
        "--configure-ccp4",
        default=None,
        help="save a CCP4 setup script path for future runs",
    )
    ap.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="keep per-entry maps/logs (default: delete after extract)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip terminal ok/partial results; retry retryable incomplete ids",
    )
    ap.add_argument(
        "--retry-partials",
        action="store_true",
        help=(
            "with --resume, reprocess non-retryable partial entries from "
            "the manifest while still skipping successful entries; --id "
            "or --id-file may restrict the retry set"
        ),
    )
    # ArgumentDefaultsHelpFormatter appends the default of ``bonds``, not of
    # the flag, so an unqualified help string renders "(default: True)" -- the
    # negation of what --no-bonds does. Naming %(default)s explicitly suppresses
    # that append and lets the value be labelled with the setting it belongs to.
    ap.add_argument(
        "--no-bonds",
        dest="bonds",
        action="store_false",
        help="skip the metal-ligand bond-distance stage (edstats "
        "stats only); bond analysis is enabled by default "
        "(bonds=%(default)s)",
    )
    ap.set_defaults(bonds=True)

    args = ap.parse_args(argv)
    if args.id and args.id_file:
        raise SystemExit("use either --id or --id-file, not both")
    if args.retry_partials and not args.resume:
        ap.error("--retry-partials requires --resume")
    if args.retry_partials and (args.pdb_file or args.mtz_file or args.cif_file):
        ap.error("--retry-partials cannot be used with manual structure inputs")
    return args


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


def _manifest_row(
    result, resume, bonds_enabled, prior_bond_counts, prior_candidate_counts
):
    """Project one worker result onto the manifest schema."""
    row = {column: result.get(column, "") for column in MANIFEST_COLUMNS}
    n_bonds = result["n_bonds"]
    n_candidates = result["n_candidates"]
    if not bonds_enabled:
        n_bonds = prior_bond_counts.get(result["pdbID"].lower(), "") if resume else ""
        n_candidates = (
            prior_candidate_counts.get(result["pdbID"].lower(), "") if resume else ""
        )
    row.update(
        n_metals=result["n"],
        n_bonds=n_bonds,
        n_candidates=n_candidates,
        runtime_s=result["runtime"],
        reason_codes="|".join(result.get("reason_codes", [])),
        warning_codes="|".join(result.get("warning_codes", [])),
    )
    return row


class _ResumeStaging:
    """Holds retried entries' rows until the whole retry batch has succeeded.

    Rows go to a temporary directory and are merged into the real outputs only
    once the batch completes, so a failed or interrupted retry leaves the
    previous rows untouched.
    """

    def __init__(self, output_dir, targets):
        self.targets = targets
        self.dir = tempfile.mkdtemp(prefix=".alchemy-resume-", dir=output_dir)
        self.staged = tuple(
            os.path.join(self.dir, os.path.basename(path)) for path in targets
        )
        self.replacement_ids = set()

    def commit(self, bonds_enabled, confidence_enabled=False):
        """Replace the retried entries' rows in the real output files."""
        if not self.replacement_ids:
            return
        manifest_path, stats_path, bonds_path, candidates_path = self.targets[:4]
        (staged_manifest, staged_stats, staged_bonds, staged_candidates) = self.staged[
            :4
        ]
        # Data files are committed before the manifest completion marker. If an
        # interruption occurs between replacements, the old manifest causes the
        # entry to be retried safely.
        _merge_csv_replacements(stats_path, staged_stats, self.replacement_ids)
        if bonds_enabled:
            _merge_csv_replacements(bonds_path, staged_bonds, self.replacement_ids)
            _merge_csv_replacements(
                candidates_path, staged_candidates, self.replacement_ids
            )
        if confidence_enabled:
            for target, staged in zip(self.targets[4:], self.staged[4:]):
                _merge_csv_replacements(target, staged, self.replacement_ids)
        _merge_csv_replacements(manifest_path, staged_manifest, self.replacement_ids)

    def discard(self):
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)


class _OutputWriters:
    """The streamed CSV outputs, with running row counts.

    Each stream is flushed after every entry so an interrupted batch run
    retains the results it already completed. Headers are written on creation.
    """

    def __init__(
        self,
        manifest_fh,
        stats_fh,
        bonds_fh,
        candidates_fh,
        confidence_fh=None,
        confidence_columns=None,
        confidence_inputs_fh=None,
    ):
        self._manifest_fh = manifest_fh
        self._stats_fh = stats_fh
        self._bonds_fh = bonds_fh
        self._candidates_fh = candidates_fh
        self._confidence_fh = confidence_fh
        self._confidence_inputs_fh = confidence_inputs_fh
        self._manifest = csv.DictWriter(manifest_fh, fieldnames=MANIFEST_COLUMNS)
        self._stats = csv.writer(stats_fh)
        self._bonds = csv.writer(bonds_fh) if bonds_fh is not None else None
        self._candidates = (
            csv.writer(candidates_fh) if candidates_fh is not None else None
        )
        if confidence_fh is not None and confidence_columns is None:
            raise ValueError("confidence columns are required with a confidence output")
        self._confidence = None
        self._confidence_inputs = None
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

    def write_stats_rows(self, rows):
        if not rows:
            return
        for row in rows:
            self._stats.writerow([row["pdbID"], row["category"]] + row["fields"])
            self.n_rows += 1
        self._stats_fh.flush()

    def write_bond_rows(self, bond_rows):
        if self._bonds is None or not bond_rows:
            return
        _check_row_schema(bond_rows[0], BOND_COLUMNS, "metal_bonds_all.csv")
        for bond in bond_rows:
            self._bonds.writerow([bond[column] for column in BOND_COLUMNS])
            self.n_bonds += 1
        self._bonds_fh.flush()

    def write_candidate_rows(self, candidate_rows):
        if self._candidates is None or not candidate_rows:
            return
        _check_row_schema(
            candidate_rows[0], CANDIDATE_COLUMNS, "metal_candidates_all.csv"
        )
        for candidate in candidate_rows:
            self._candidates.writerow(
                [candidate[column] for column in CANDIDATE_COLUMNS]
            )
            self.n_candidates += 1
        self._candidates_fh.flush()

    def write_manifest_row(self, row):
        self._manifest.writerow(row)
        self._manifest_fh.flush()

    def write_confidence_rows(self, rows):
        if self._confidence is None or not rows:
            return
        if self._confidence_columns is None or self._confidence_fh is None:
            raise RuntimeError("confidence output is not fully configured")
        expected = set(self._confidence_columns)
        if rows[0].keys() != expected:
            raise RuntimeError("confidence row does not match its output schema")
        self._confidence.writerows(rows)
        if self._confidence_inputs is not None:
            self._confidence_inputs.writerows(
                {column: row[column] for column in CONFIDENCE_INPUT_COLUMNS}
                for row in rows
            )
        self.n_confidence += len(rows)
        self._confidence_fh.flush()
        if self._confidence_inputs_fh is not None:
            self._confidence_inputs_fh.flush()


class _ProgressReporter:
    """Render a throttled parent-process heartbeat without worker overhead."""

    TERMINAL_INTERVAL_S = 1.0
    REDIRECTED_INTERVAL_S = 30.0

    def __init__(self, total, stream=None, clock=None):
        self.total = total
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.monotonic
        self.started = self.clock()
        self.last_rendered = float("-inf")
        self.last_width = 0
        self.terminal = bool(self.stream.isatty())
        self.line_open = False

    @staticmethod
    def _elapsed_text(elapsed_s):
        elapsed = max(0, int(elapsed_s))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def render(self, completed, counts, no_metal_count, force=False, final=False):
        now = self.clock()
        interval = (
            self.TERMINAL_INTERVAL_S if self.terminal else self.REDIRECTED_INTERVAL_S
        )
        if not force and now - self.last_rendered < interval:
            return
        percent = 100.0 * completed / self.total if self.total else 100.0
        line = (
            f"[{completed}/{self.total} {percent:5.1f}%] "
            f"elapsed={self._elapsed_text(now - self.started)} | "
            f"ok={counts['ok']} partial={counts['partial']} "
            f"skip={counts['skip']} error={counts['error']} | "
            f"no_metals={no_metal_count}"
        )
        if self.terminal:
            padded = line.ljust(self.last_width)
            print(
                f"\r{padded}", end="\n" if final else "", file=self.stream, flush=True
            )
            self.last_width = len(line)
            self.line_open = not final
        else:
            print(line, file=self.stream, flush=True)
        self.last_rendered = now

    def close(self):
        """Finish an in-place terminal line after success or an exception."""
        if self.terminal and self.line_open:
            print(file=self.stream, flush=True)
            self.line_open = False


class _RunLog:
    """Collect compact run diagnostics and write one human-readable log."""

    def __init__(self, args, command):
        self.args = args
        self.command = command
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.details = {
            "initial_available_memory_bytes": available_memory_bytes(),
        }
        self.summary = {}
        self.entries = []
        self.driver_error = ""

    def record_entry(self, result):
        """Retain diagnostic fields without keeping large result-row payloads."""
        self.entries.append(
            {
                "pdbID": result.get("pdbID", ""),
                "status": result.get("status", "unknown"),
                "retryable": bool(result.get("retryable", False)),
                "no_metals": bool(result.get("no_metals", False)),
                "n_metals": result.get("n", 0),
                "n_bonds": result.get("n_bonds", 0),
                "n_candidates": result.get("n_candidates", 0),
                "runtime_s": float(result.get("runtime", 0.0)),
                "timings": dict(result.get("timings", {})),
                "reason_codes": list(result.get("reason_codes", [])),
                "warning_codes": list(result.get("warning_codes", [])),
                "error": str(result.get("error", "")),
                "density_map_scope_used": result.get("density_map_scope_used", ""),
                "density_full_map_bytes": result.get("density_full_map_bytes", 0),
                "density_edstats_map_bytes": result.get("density_edstats_map_bytes", 0),
            }
        )

    @staticmethod
    def _clean(value):
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _counter_text(counter):
        if not counter:
            return "none"
        return ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        )

    def _render(self, exit_code, finished_at, elapsed_s):
        lines = [
            "Alchemy detailed run log",
            "========================",
            f"Started (UTC): {self.started_at.isoformat()}",
            f"Finished (UTC): {finished_at.isoformat()}",
            f"Wall time: {elapsed_s:.3f} s",
            f"Exit code: {exit_code}",
            f"Command: {self.command}",
            "",
            "System",
            "------",
            f"Platform: {platform.platform()}",
            f"Python: {platform.python_version()}",
            f"Available CPUs at startup: {available_cpu_count()}",
        ]
        initial_memory = self.details.get("initial_available_memory_bytes")
        if initial_memory is None:
            lines.append("Available memory at startup: unknown")
        else:
            lines.append(
                f"Available memory at startup: {initial_memory / (1024**3):.2f} GiB"
            )
        final_memory = available_memory_bytes()
        lines.append(
            "Available memory at finish: "
            + (
                f"{final_memory / (1024**3):.2f} GiB"
                if final_memory is not None
                else "unknown"
            )
        )

        lines.extend(["", "Configuration", "-------------"])
        for name, value in sorted(vars(self.args).items()):
            lines.append(f"{name}: {self._clean(value)}")
        for name, value in sorted(self.details.items()):
            if name == "initial_available_memory_bytes":
                continue
            lines.append(f"{name}: {self._clean(value)}")

        status_counts = Counter(entry["status"] for entry in self.entries)
        reason_counts = Counter(
            reason for entry in self.entries for reason in entry["reason_codes"]
        )
        warning_counts = Counter(
            warning for entry in self.entries for warning in entry["warning_codes"]
        )
        retryable_count = sum(entry["retryable"] for entry in self.entries)
        no_metal_count = sum(entry["no_metals"] for entry in self.entries)
        map_scope_counts = Counter(
            entry["density_map_scope_used"]
            for entry in self.entries
            if entry["density_map_scope_used"]
        )
        total_entry_s = sum(entry["runtime_s"] for entry in self.entries)
        throughput = len(self.entries) * 60.0 / elapsed_s if elapsed_s > 0 else 0.0

        lines.extend(
            [
                "",
                "Summary",
                "-------",
                f"Entries returned: {len(self.entries)}",
                f"Status counts: {self._counter_text(status_counts)}",
                f"Retryable entries: {retryable_count}",
                f"Metal-free entries: {no_metal_count}",
                f"Summed entry runtime: {total_entry_s:.3f} s",
                f"Throughput: {throughput:.2f} entries/minute",
                f"Reason codes: {self._counter_text(reason_counts)}",
                f"Warning codes: {self._counter_text(warning_counts)}",
                f"Density map scopes used: {self._counter_text(map_scope_counts)}",
            ]
        )
        for name, value in sorted(self.summary.items()):
            lines.append(f"{name}: {self._clean(value)}")
        if self.driver_error:
            lines.append(f"Driver error: {self._clean(self.driver_error)}")

        stage_values = {}
        for entry in self.entries:
            for name, value in entry["timings"].items():
                try:
                    stage_values.setdefault(name, []).append(float(value))
                except (TypeError, ValueError):
                    continue
        lines.extend(["", "Stage timing", "------------"])
        if not stage_values:
            lines.append("No completed stage timings were recorded.")
        else:
            lines.append(
                "Totals sum per-entry measurements; density_total_s contains "
                "its subprocess stages, and parallel totals are not wall time."
            )
            lines.append("stage | entries | total_s | mean_s | max_s | max_entry")
            for name in sorted(stage_values):
                values = stage_values[name]
                stage_entries = [
                    entry for entry in self.entries if name in entry["timings"]
                ]
                max_entry = max(
                    stage_entries, key=lambda entry: float(entry["timings"][name])
                )
                lines.append(
                    f"{name} | {len(values)} | {sum(values):.3f} | "
                    f"{sum(values) / len(values):.3f} | {max(values):.3f} | "
                    f"{max_entry['pdbID']}"
                )

        incomplete_entries = [
            entry for entry in self.entries if entry["status"] != "ok"
        ]
        lines.extend(["", "Incomplete entries", "------------------"])
        if not incomplete_entries:
            lines.append("None.")
        else:
            lines.append("pdbID | status | retryable | reasons | error")
            for entry in incomplete_entries:
                lines.append(
                    f"{entry['pdbID']} | {entry['status']} | "
                    f"{self._clean(entry['retryable'])} | "
                    f"{'|'.join(entry['reason_codes']) or '-'} | "
                    f"{self._clean(entry['error']) or '-'}"
                )

        lines.extend(["", "Slowest entries", "---------------"])
        if not self.entries:
            lines.append("No entries were processed.")
        else:
            lines.append(
                "pdbID | status | runtime_s | metals | bonds | candidates | reasons"
            )
            for entry in sorted(
                self.entries, key=lambda item: item["runtime_s"], reverse=True
            )[:20]:
                lines.append(
                    f"{entry['pdbID']} | {entry['status']} | "
                    f"{entry['runtime_s']:.2f} | {entry['n_metals']} | "
                    f"{entry['n_bonds']} | {entry['n_candidates']} | "
                    f"{'|'.join(entry['reason_codes']) or '-'}"
                )

        lines.extend(["", "Per-entry results", "-----------------"])
        if not self.entries:
            lines.append("No entries were processed.")
        for entry in self.entries:
            timing_text = (
                ",".join(
                    f"{name}={float(value):.3f}"
                    for name, value in sorted(entry["timings"].items())
                )
                or "-"
            )
            lines.append(
                f"{entry['pdbID']} | status={entry['status']} | "
                f"retryable={self._clean(entry['retryable'])} | "
                f"no_metals={self._clean(entry['no_metals'])} | "
                f"runtime_s={entry['runtime_s']:.2f} | "
                f"metals={entry['n_metals']} | bonds={entry['n_bonds']} | "
                f"candidates={entry['n_candidates']} | timings={timing_text} | "
                f"density_map_scope={entry['density_map_scope_used'] or '-'} | "
                f"full_map_bytes={entry['density_full_map_bytes']} | "
                f"edstats_map_bytes={entry['density_edstats_map_bytes']} | "
                f"reasons={'|'.join(entry['reason_codes']) or '-'} | "
                f"warnings={'|'.join(entry['warning_codes']) or '-'} | "
                f"error={self._clean(entry['error']) or '-'}"
            )
        lines.append("")
        return "\n".join(lines)

    def write(self, exit_code):
        """Atomically write the final timestamped log and return its path."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        finished_at = datetime.now(timezone.utc)
        elapsed_s = time.monotonic() - self.started_monotonic
        run_date = self.started_at.strftime("%Y%m%d")
        log_stem = f"alchemy_run_{run_date}"
        path = os.path.join(self.args.output_dir, f"{log_stem}.log")
        suffix = 2
        while os.path.lexists(path):
            path = os.path.join(self.args.output_dir, f"{log_stem}_{suffix}.log")
            suffix += 1
        handle, temporary_path = tempfile.mkstemp(
            prefix=".alchemy-run-log-", dir=self.args.output_dir, text=True
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as log:
                log.write(self._render(exit_code, finished_at, elapsed_s))
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
        return path


def _run(args, run_log):

    try:
        cofactors = load_cofactor_ids()
    except (OSError, UnicodeError, ValueError) as exc:
        message = f"Invalid bundled metallocofactor catalog: {exc}"
        run_log.driver_error = message
        logger.error("%s", message)
        return 1

    env, _ = resolve_ccp4_environment(args)
    if env is None:
        return 0
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as exc:
        # A read-only mount or someone else's directory is a fixable user
        # mistake, so it exits the way every other unusable input does rather
        # than as a traceback that reads like an Alchemy bug.
        raise SystemExit(
            f"Cannot use --output-dir {args.output_dir}: {exc.strerror or exc}"
        ) from None
    _sweep_leaked_work_dirs(args.output_dir)

    cache_root = args.pdb_redo_cache
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    stats_path = os.path.join(args.output_dir, "metal_stats_all.csv")
    bonds_path = os.path.join(args.output_dir, "metal_bonds_all.csv")
    candidates_path = os.path.join(args.output_dir, "metal_candidates_all.csv")
    confidence_inputs_path = os.path.join(args.output_dir, "confidence_inputs_all.csv")
    confidence_scores_path = os.path.join(args.output_dir, "confidence_scores_all.csv")
    database_reference_dir = os.path.join(args.output_dir, "confidence_reference")

    manual_requested = bool(args.pdb_file or args.mtz_file or args.cif_file)
    database_run = (
        not args.id
        and not args.id_file
        and not manual_requested
        and args.max_pdbs is None
    )
    run_log.details["run_mode"] = (
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
    confidence_mode = None
    confidence_reference = None
    confidence_stream_path = None
    confidence_columns = None
    synchronize_confidence_inputs = False
    if args.bonds and database_run:
        confidence_mode = "database"
        confidence_stream_path = confidence_inputs_path
        confidence_columns = CONFIDENCE_INPUT_COLUMNS
    elif args.bonds:
        confidence_reference_dir, searched_reference_dirs = (
            resolve_confidence_reference_dir(
                args.output_dir, args.confidence_reference_dir
            )
        )
        if confidence_reference_dir is not None:
            try:
                confidence_reference = load_confidence_reference(
                    confidence_reference_dir
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                message = f"Invalid confidence reference: {exc}"
                run_log.driver_error = message
                logger.error("%s", message)
                return 1
            run_log.details["confidence_reference_dir"] = confidence_reference_dir
            confidence_mode = "reference"
            confidence_stream_path = confidence_scores_path
            confidence_columns = (
                *CONFIDENCE_INPUT_COLUMNS,
                *CONFIDENCE_ANALYSIS_COLUMNS,
            )
        else:
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
                ", ".join(searched_reference_dirs),
            )
    if args.resume:
        if confidence_mode is not None and (
            confidence_stream_path is None or not os.path.isfile(confidence_stream_path)
        ):
            message = (
                "Cannot resume confidence-aware output because "
                f"{confidence_stream_path} is missing; use a fresh output "
                "directory."
            )
            run_log.driver_error = message
            logger.error("%s", message)
            return 1
        try:
            validate_resume_schemas(
                manifest_path,
                stats_path,
                bonds_path,
                candidates_path,
                bonds_enabled=args.bonds,
                confidence_path=confidence_stream_path,
                confidence_columns=confidence_columns,
            )
            synchronize_confidence_inputs = (
                confidence_mode == "reference"
                and os.path.isfile(confidence_inputs_path)
            )
            if synchronize_confidence_inputs:
                validate_resume_schemas(
                    manifest_path,
                    stats_path,
                    bonds_path,
                    candidates_path,
                    bonds_enabled=args.bonds,
                    confidence_path=confidence_inputs_path,
                    confidence_columns=CONFIDENCE_INPUT_COLUMNS,
                )
        except ValueError as exc:
            run_log.driver_error = str(exc)
            logger.error("%s", exc)
            return 1
        if confidence_mode == "reference":
            try:
                validate_scored_reference(confidence_stream_path, confidence_reference)
            except (OSError, ValueError) as exc:
                message = f"Cannot resume confidence output: {exc}"
                run_log.driver_error = message
                logger.error("%s", message)
                return 1

    try:
        ids, root, manual_inputs = _select_entry_ids(args, cache_root)
    except _DriverError as exc:
        run_log.driver_error = str(exc)
        logger.error("%s", exc)
        return 1

    run_log.details["entries_selected_before_resume"] = len(ids)
    run_log.details["resolved_input_root"] = root

    if args.resume:
        done_kwargs = {
            "bonds_required": args.bonds,
            "bond_output_present": os.path.isfile(bonds_path),
            "candidate_output_present": os.path.isfile(candidates_path),
        }
        normally_done = load_done(manifest_path, **done_kwargs)
        if args.retry_partials:
            done = load_done(manifest_path, retry_partial_ids=ids, **done_kwargs)
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

    if not ids:
        if (
            args.resume
            and confidence_mode == "database"
            and os.path.isfile(confidence_inputs_path)
        ):
            try:
                total, scored, cohort = finalize_database_confidence(
                    confidence_inputs_path,
                    confidence_scores_path,
                    database_reference_dir,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                message = f"Confidence finalization failed: {exc}"
                run_log.driver_error = message
                logger.error("%s", message)
                return 1
            logger.info(
                "no entries required retry; finalized %d confidence rows "
                "(%d scored; database cohort %d) -> %s",
                total,
                scored,
                cohort,
                confidence_scores_path,
            )
            logger.info("confidence reference -> %s", database_reference_dir)
            return 0
        logger.info("no entries to process")
        return 0

    try:
        removed_stale_bond_outputs = remove_stale_disabled_bond_outputs(
            (bonds_path, candidates_path), resume=args.resume, bonds_enabled=args.bonds
        )
    except OSError as exc:
        message = f"Could not remove stale bond-stage output: {exc}"
        run_log.driver_error = message
        logger.error("%s", message)
        return 1
    for removed_path in removed_stale_bond_outputs:
        logger.info("removed stale bond-stage output: %s", removed_path)
    if not args.resume:
        # Reference metadata is its completion marker. Fresh runs must not leave
        # incompatible confidence artifacts beside newly replaced core output.
        if confidence_mode == "database":
            stale_confidence_paths = (
                confidence_scores_path,
                os.path.join(database_reference_dir, REFERENCE_METADATA_FILE),
            )
        elif confidence_mode == "reference":
            stale_confidence_paths = (confidence_inputs_path,)
        else:
            stale_confidence_paths = (
                confidence_inputs_path,
                confidence_scores_path,
                os.path.join(database_reference_dir, REFERENCE_METADATA_FILE),
            )
        try:
            for path in stale_confidence_paths:
                if os.path.isfile(path):
                    os.unlink(path)
        except OSError as exc:
            message = f"Could not clear stale confidence output: {exc}"
            run_log.driver_error = message
            logger.error("%s", message)
            return 1

    # A Pool creates every worker up front, so asking for more than there are
    # entries just pays each one's startup for no work. Costly under the spawn
    # start method, where each worker re-imports gemmi into its own interpreter.
    if args.workers is None:
        cpu_limit, memory_limit = automatic_worker_limits()
        automatic_limit = cpu_limit
        if memory_limit is not None:
            automatic_limit = min(automatic_limit, memory_limit)
        workers = min(automatic_limit, len(ids))
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
        workers = min(args.workers, len(ids))
        run_log.details["worker_selection"] = "explicit"
        run_log.details["Selected workers"] = workers
    logger.info(
        "processing %d entr%s with %d worker(s)",
        len(ids),
        "y" if len(ids) == 1 else "ies",
        workers,
    )

    cfg = {
        "root": root,
        "mirror_root": args.pdb_redo_root,
        "cache_root": cache_root,
        "env": env,
        "output_dir": args.output_dir,
        "cofactors": cofactors,
        "keep": args.keep_intermediates,
        "bonds": args.bonds,
        "density_map_scope": args.density_map_scope,
        "ccp4_timeout_s": args.ccp4_timeout,
        "log_level": worker_level(
            level_for_verbosity(args.verbose, args.quiet), args.log_file
        ),
        "allow_download": bool(args.id or args.id_file),
        "manual_inputs": manual_inputs,
        "alchemy_commit": _alchemy_commit(),
        "gemmi_version": _gemmi_version(),
        "ccp4_version": _ccp4_version(env),
    }
    run_log.details.update(
        alchemy_version=ALCHEMY_VERSION,
        gemmi_version=cfg["gemmi_version"],
        ccp4_version=cfg["ccp4_version"],
        confidence_mode=confidence_mode or "disabled",
    )

    prior_bond_counts = (
        _manifest_values_by_id(manifest_path, "n_bonds")
        if args.resume and not args.bonds
        else {}
    )
    prior_candidate_counts = (
        _manifest_values_by_id(manifest_path, "n_candidates")
        if args.resume and not args.bonds
        else {}
    )
    output_paths = [manifest_path, stats_path, bonds_path, candidates_path]
    if confidence_mode is not None:
        if confidence_stream_path is None:
            raise RuntimeError("confidence output path is not configured")
        output_paths.append(confidence_stream_path)
    if synchronize_confidence_inputs:
        output_paths.append(confidence_inputs_path)
    output_paths = tuple(output_paths)
    staging = _ResumeStaging(args.output_dir, output_paths) if args.resume else None
    write_paths = staging.staged if staging is not None else output_paths
    (write_manifest_path, write_stats_path, write_bonds_path, write_candidates_path) = (
        write_paths[:4]
    )
    write_confidence_path = write_paths[4] if confidence_mode is not None else None
    write_confidence_inputs_path = (
        write_paths[5] if synchronize_confidence_inputs else None
    )

    counts = {"ok": 0, "partial": 0, "skip": 0, "error": 0}
    no_metal_count = 0
    retryable_partial_count = 0
    processing_completed = False
    writers = None
    progress = _ProgressReporter(len(ids))
    try:
        # ExitStack closes whichever handles were opened, so a failure partway
        # through opening them cannot leak the earlier ones.
        with contextlib.ExitStack() as handles:
            writers = _OutputWriters(
                handles.enter_context(open(write_manifest_path, "w", newline="")),
                handles.enter_context(open(write_stats_path, "w", newline="")),
                handles.enter_context(open(write_bonds_path, "w", newline=""))
                if args.bonds
                else None,
                handles.enter_context(open(write_candidates_path, "w", newline=""))
                if args.bonds
                else None,
                handles.enter_context(open(write_confidence_path, "w", newline=""))
                if write_confidence_path is not None
                else None,
                confidence_columns,
                handles.enter_context(
                    open(write_confidence_inputs_path, "w", newline="")
                )
                if write_confidence_inputs_path is not None
                else None,
            )
            inflight = SimpleQueue()
            assignments: Dict[int, str] = {}
            worker_pids: set = set()
            lost_ids: set = set()
            completed_ids: set = set()
            unattributed_deaths = 0
            last_progress = time.monotonic()
            # Not a ``with`` block: ``Pool.__exit__`` calls the same
            # ``terminate`` that a killed idle worker can wedge forever, so
            # shutdown is bounded explicitly in the ``finally`` below.
            # Worker records travel over this queue and are re-emitted by the
            # driver, so only one process ever writes to a handler. The queue
            # is created before the pool but the listener thread is started
            # after it: forking a process that already has running threads
            # risks the child deadlocking on a lock no thread there holds.
            log_queue = create_worker_log_queue()
            pool = Pool(
                workers,
                initializer=_init_worker,
                initargs=(cfg, inflight, log_queue),
            )
            log_listener = start_worker_log_listener(log_queue)
            try:
                results = pool.imap_unordered(process, ids, chunksize=1)
                completed = 0
                progress.render(completed, counts, no_metal_count, force=True)
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
                            # Every entry still outstanding can only be held by
                            # a process that died before naming its entry.
                            # Entries that already returned a result are not
                            # outstanding: blaming one of those would write a
                            # second, failed row for an entry that succeeded and
                            # leave the entry that actually died unreported.
                            for stuck_id in ids:
                                if stuck_id in lost_ids or stuck_id in completed_ids:
                                    continue
                                lost_ids.add(stuck_id)
                                batch.append(_worker_death_result(stuck_id, cfg, 0))
                                if len(batch) >= remaining:
                                    break
                        else:
                            progress.render(completed, counts, no_metal_count)
                            continue
                    last_progress = time.monotonic()
                    for r in batch:
                        if r["pdbID"] in lost_ids and r.get("reason_codes") != [
                            "worker_process_died"
                        ]:
                            # A real result arrived for an entry already
                            # declared lost. The synthesized row stands, so
                            # this one is dropped rather than written twice.
                            continue
                        completed += 1
                        completed_ids.add(r["pdbID"])
                        run_log.record_entry(r)
                        if not args.resume or _resume_replacement_succeeded(r):
                            writers.write_stats_rows(r["rows"])
                            writers.write_bond_rows(r["bond_rows"])
                            writers.write_candidate_rows(r["candidate_rows"])
                            confidence_rows = []
                            if confidence_mode is not None:
                                confidence_rows = prepare_result_confidence_inputs(
                                    r["rows"], r["bond_rows"], STATS_COLUMNS
                                )
                                confidence_rows = complete_confidence_site_count(
                                    confidence_rows,
                                    r["pdbID"],
                                    r["n"],
                                    r.get("confidence_inputs_missing_reason", ""),
                                )
                                # Equivalent to `confidence_mode == "reference"`:
                                # the mode is set only where a reference has
                                # just been loaded. Testing the reference
                                # itself says what the call actually needs.
                                if confidence_reference is not None:
                                    confidence_rows = score_against_reference(
                                        confidence_rows, confidence_reference
                                    )
                                writers.write_confidence_rows(confidence_rows)
                            # The manifest is the completion marker, so write
                            # it only after this entry's rows have flushed.
                            writers.write_manifest_row(
                                _manifest_row(
                                    r,
                                    args.resume,
                                    args.bonds,
                                    prior_bond_counts,
                                    prior_candidate_counts,
                                )
                            )
                            if staging is not None:
                                staging.replacement_ids.add(r["pdbID"].lower())
                        counts[r["status"]] = counts.get(r["status"], 0) + 1
                        if r.get("no_metals", False):
                            no_metal_count += 1
                        if r["status"] == "partial" and r.get("retryable", False):
                            retryable_partial_count += 1
                        finished = completed == len(ids)
                        progress.render(
                            completed,
                            counts,
                            no_metal_count,
                            force=progress.terminal or finished,
                            final=finished,
                        )
            finally:
                forced = _shutdown_pool(pool)
                # Stopped only after the pool is gone, so records emitted
                # during shutdown are still forwarded.
                log_listener.stop()
                log_queue.close()
                if forced:
                    run_log.summary["worker_pool_forced_shutdown"] = True
                    logger.warning(
                        "a worker pool shutdown had to be forced after a "
                        "worker died holding the task-queue lock; results "
                        "above are complete"
                    )
            processing_completed = True
    finally:
        progress.close()
        if staging is not None and not processing_completed:
            staging.discard()

    n_rows = writers.n_rows if writers is not None else 0
    n_bonds = writers.n_bonds if writers is not None else 0
    n_candidates = writers.n_candidates if writers is not None else 0
    run_log.summary.update(
        metal_rows_written=n_rows,
        bond_rows_written=n_bonds,
        candidate_rows_written=n_candidates,
        confidence_rows_written=(writers.n_confidence if writers is not None else 0),
        manifest_path=manifest_path,
        metal_stats_path=stats_path,
        metal_bonds_path=bonds_path if args.bonds else "disabled",
        metal_candidates_path=(candidates_path if args.bonds else "disabled"),
    )

    if staging is not None:
        try:
            staging.commit(args.bonds, confidence_enabled=confidence_mode is not None)
        finally:
            staging.discard()

    print(
        f"Done. ok={counts['ok']} partial={counts['partial']} "
        f"skip={counts['skip']} error={counts['error']} "
        f"no_metals={no_metal_count}; "
        f"{n_rows} metal/cofactor rows -> {stats_path}",
        flush=True,
    )
    if args.bonds:
        print(f"      {n_bonds} bond rows -> {bonds_path}", flush=True)
        print(f"      {n_candidates} candidate rows -> {candidates_path}", flush=True)
    exit_code = _batch_exit_code(counts, retryable_partial_count)
    if confidence_mode == "database":
        if exit_code == 0:
            try:
                total, scored, cohort = finalize_database_confidence(
                    confidence_inputs_path,
                    confidence_scores_path,
                    database_reference_dir,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                message = f"Confidence finalization failed: {exc}"
                run_log.driver_error = message
                logger.error("%s", message)
                return 1
            run_log.summary.update(
                confidence_status="finalized",
                confidence_rows=total,
                confidence_scored_rows=scored,
                confidence_reference_cohort=cohort,
                confidence_scores_path=confidence_scores_path,
                confidence_reference_path=database_reference_dir,
            )
            print(
                f"      {total} confidence rows ({scored} scored; "
                f"reference cohort {cohort}) -> {confidence_scores_path}",
                flush=True,
            )
            print(f"      confidence reference -> {database_reference_dir}", flush=True)
        else:
            run_log.summary["confidence_status"] = "not_finalized_incomplete_run"
            print(
                "      confidence inputs were retained, but the database "
                "reference was not finalized because the run is incomplete.",
                flush=True,
            )
    elif confidence_mode == "reference":
        if confidence_reference is None:
            raise RuntimeError("confidence reference is not configured")
        print(
            f"      {writers.n_confidence} confidence rows compared with "
            f"database cohort {confidence_reference.cohort_size} -> "
            f"{confidence_scores_path}",
            flush=True,
        )
        run_log.summary.update(
            confidence_status="scored_against_reference",
            confidence_reference_cohort=confidence_reference.cohort_size,
            confidence_scores_path=confidence_scores_path,
        )
    if exit_code:
        logger.warning(
            "completed with incomplete entries: errors=%d, skips=%d, "
            "retryable_partials=%d",
            counts["error"],
            counts["skip"],
            retryable_partial_count,
        )
    return exit_code


def main(argv=None):
    """Parse arguments, execute the driver, and always emit a run log."""
    raw_args = None if argv is None else list(argv)
    args = parse_args(raw_args)
    # Handlers are attached once, here, and nowhere else: the driver is the
    # only process that writes them, and workers reach them over a queue.
    configure_driver_logging(
        level=level_for_verbosity(args.verbose, args.quiet),
        log_file=args.log_file,
    )
    command_parts = list(sys.argv) if raw_args is None else [sys.argv[0], *raw_args]
    run_log = _RunLog(args, shlex.join(command_parts))
    exit_code = 1
    previous_term = _install_termination_handler()
    try:
        exit_code = _run(args, run_log)
        return exit_code
    except KeyboardInterrupt:
        # Ctrl-C or SIGTERM. The pool has already been shut down by _run's
        # own finally, so this only decides how the driver reports it: a
        # conventional interrupt status and one line, rather than a traceback
        # that looks like a crash.
        run_log.driver_error = "interrupted before completion"
        print(
            "\nInterrupted: workers stopped; rows already flushed are kept "
            "and --resume will continue.",
            file=sys.stderr,
            flush=True,
        )
        exit_code = 130
        return exit_code
    except BaseException as exc:
        run_log.driver_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if previous_term is not None:
            with contextlib.suppress(AttributeError, OSError, ValueError):
                signal.signal(signal.SIGTERM, previous_term)
        try:
            log_path = run_log.write(exit_code)
        except OSError as exc:
            logger.error("could not write detailed run log: %s", exc)
        else:
            logger.info("detailed run log -> %s", log_path)


if __name__ == "__main__":
    sys.exit(main())
