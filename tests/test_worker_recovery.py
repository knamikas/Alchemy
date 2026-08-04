"""Worker-death recovery in the batch driver (``src/main.py``).

A worker felled by the OOM killer or by a segfault in a compiled extension runs
no further Python, and ``multiprocessing.Pool`` never delivers a result for the
task it was holding: it silently replaces the process. The driver therefore
tracks which entry each pid holds (``_drain_inflight`` over a ``SimpleQueue``
the workers write to), diffs the pool roster (``_dead_worker_pids``) and
synthesizes a retryable result for the orphaned entry
(``_worker_death_result``).

The end-to-end tests drive the real ``driver_pool._run``, with only the
per-entry pipeline replaced by a scripted stub, and need POSIX (``fork`` and
``SIGKILL``); the ``spawn`` coverage runs anywhere.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import multiprocessing
import os
import pickle
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

import pytest

import density_analysis as density
import main
import worker
from driver import runlog, writers
from driver.writers import MANIFEST_COLUMNS
import cli
from driver import pool as driver_pool
import reference_data


_POSIX_KILL = pytest.mark.skipif(
    sys.platform == "win32"
    or not hasattr(signal, "SIGKILL")
    or "fork" not in multiprocessing.get_all_start_methods(),
    reason="needs the fork start method and SIGKILL (POSIX only)",
)


class _FakeWorker:
    """Stand-in for a ``multiprocessing.Process`` in ``pool._pool``."""

    def __init__(self, pid):
        self.pid = pid


class _FakePool:
    """Minimal object exposing the ``_pool`` roster ``_dead_worker_pids`` reads."""

    def __init__(self, pids=()):
        self.set_roster(pids)

    def set_roster(self, pids):
        self._pool = [_FakeWorker(pid) for pid in pids]


class _BrokenQueue:
    """A notification queue whose pipe has been torn down."""

    def __init__(self, error):
        self.error = error
        self.get_calls = 0

    def empty(self):
        if isinstance(self.error, OSError):
            raise self.error
        return False

    def get(self):
        self.get_calls += 1
        raise self.error


def _reference_cfg(output_dir, manual_inputs=None):
    """Build the worker config exactly as ``driver_pool._run`` assembles it."""
    env = dict(os.environ)
    return worker.WorkerConfig(
        root=os.path.join(output_dir, "root"),
        mirror_root=os.path.join(output_dir, "mirror"),
        cache_root=os.path.join(output_dir, "cache"),
        env=env,
        output_dir=output_dir,
        cofactors=reference_data.cofactor_ids(),
        keep=False,
        bonds=True,
        density_map_scope="model-envelope",
        ccp4_timeout_s=density.CCP4_TOOL_TIMEOUT_S,
        log_level=logging.INFO,
        allow_download=False,
        manual_inputs=manual_inputs,
        alchemy_commit=driver_pool._alchemy_commit(),
        gemmi_version=driver_pool._gemmi_version(),
        ccp4_version=driver_pool._ccp4_version(env),
        reference_data_id=reference_data.reference_data_id(),
    )


@pytest.mark.parametrize(
    "notifications, expected, label",
    [
        ([("start", 11, "1abc")], {11: "1abc"}, "start records the holder"),
        ([("start", 11, "1abc"), ("end", 11, "1abc")], {}, "end releases the holder"),
        ([("end", 11, "1abc")], {}, "end for an unknown pid is ignored"),
        (
            [("start", 11, "1abc"), ("end", 12, "2xyz")],
            {11: "1abc"},
            "end for another pid leaves the holder intact",
        ),
        (
            [("start", 11, "1abc"), ("end", 11, "1abc"), ("start", 11, "2xyz")],
            {11: "2xyz"},
            "a reused worker holds only its current entry",
        ),
        (
            [("start", 11, "1abc"), ("start", 12, "2xyz")],
            {11: "1abc", 12: "2xyz"},
            "each worker is tracked separately",
        ),
        ([], {}, "an empty queue leaves the map untouched"),
    ],
)
def test_drain_inflight_applies_notifications_in_order(notifications, expected, label):
    """Draining applies every queued start/end to the pid -> entry map.

    The map is the only record of which entry a killed process held, so a stale
    or missing assignment either loses an entry or blames the wrong one.
    """
    inflight: multiprocessing.SimpleQueue = multiprocessing.SimpleQueue()
    try:
        for notification in notifications:
            inflight.put(notification)
        assignments: dict[int, str] = {}
        driver_pool._drain_inflight(inflight, assignments)
        assert assignments == expected, label
    finally:
        inflight.close()


def test_drain_inflight_preserves_unrelated_existing_assignments():
    """Draining only touches the pids it has notifications for."""
    inflight: multiprocessing.SimpleQueue = multiprocessing.SimpleQueue()
    try:
        inflight.put(("start", 22, "2xyz"))
        assignments = {11: "1abc"}
        driver_pool._drain_inflight(inflight, assignments)
        assert assignments == {11: "1abc", 22: "2xyz"}
    finally:
        inflight.close()


def test_drain_inflight_returns_promptly_on_an_empty_queue():
    """An empty queue must not block the driver's polling loop.

    ``SimpleQueue.get`` on an empty queue blocks forever, so the drain runs in a
    daemon thread: losing this property must be a red test, not a hung CI job.
    """
    inflight: multiprocessing.SimpleQueue = multiprocessing.SimpleQueue()
    returned = threading.Event()
    assignments: dict[int, str] = {}

    def drain():
        driver_pool._drain_inflight(inflight, assignments)
        returned.set()

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    drainer.join(10.0)
    try:
        assert returned.is_set(), (
            "_drain_inflight blocked on an empty queue; the dispatch loop "
            "would never reach the roster check that recovers lost entries"
        )
        assert assignments == {}
    finally:
        if returned.is_set():
            # Closing the pipe under a still-blocked reader only adds noise to
            # an already-failed test.
            inflight.close()


@pytest.mark.parametrize("error", [OSError("pipe gone"), EOFError()])
def test_drain_inflight_survives_a_torn_down_queue(error):
    """A dead notification pipe must not propagate out of bookkeeping.

    The queue's other end lives in worker processes that may all have exited; a
    raised error here would abort a batch the driver could still finish.
    """
    assignments = {11: "1abc"}
    driver_pool._drain_inflight(_BrokenQueue(error), assignments)
    assert assignments == {11: "1abc"}


def test_dead_worker_pids_reports_only_newly_missing_workers():
    """A pid leaving the roster is reported exactly once, then forgotten.

    The pool replaces a dead worker with a new pid, so the driver has to diff
    successive rosters. Reporting the same death twice would fabricate a second
    lost entry; never reporting it hangs the batch.
    """
    known: set[int] = set()
    pool = _FakePool([101, 102])

    assert driver_pool._dead_worker_pids(pool, known) == set()
    assert known == {101, 102}

    pool.set_roster([101, 103])
    assert driver_pool._dead_worker_pids(pool, known) == {102}
    assert known == {101, 103}

    assert driver_pool._dead_worker_pids(pool, known) == set()
    assert known == {101, 103}


def test_dead_worker_pids_reports_several_simultaneous_deaths():
    """Two workers lost between polls are both reported.

    An out-of-memory event typically kills more than one worker at a time.
    """
    known = {1, 2, 3}
    pool = _FakePool([1, 4, 5])
    assert driver_pool._dead_worker_pids(pool, known) == {2, 3}
    assert known == {1, 4, 5}


@pytest.mark.parametrize(
    "pool, label",
    [
        (_FakePool([]), "roster emptied during shutdown"),
        (type("_NoRoster", (), {"_pool": None})(), "roster attribute is None"),
        (type("_Unrostered", (), {})(), "pool exposes no roster at all"),
        (_FakePool([None, None]), "processes have not been assigned pids yet"),
    ],
)
def test_dead_worker_pids_treats_an_unavailable_roster_as_no_news(pool, label):
    """An empty or absent roster must not declare every worker dead.

    The pool tears its roster down at shutdown and has not filled it in yet at
    startup; diffing against either would synthesize a lost-entry row for every
    worker that ever ran.
    """
    known = {101, 102}
    assert driver_pool._dead_worker_pids(pool, known) == set(), label
    assert known == {101, 102}, "the known set must be left alone"


def test_worker_death_result_is_a_complete_retryable_manifest_row(tmp_path):
    """The synthesized result is a full, retryable, error-status entry row.

    The driver writes it straight to the manifest, and the retryable
    ``worker_process_died`` marking is what makes ``--resume`` pick the entry up
    again instead of treating it as terminally finished.
    """
    cfg = _reference_cfg(str(tmp_path))
    result = worker._worker_death_result("1abc", cfg, 4321)

    assert result.pdb_id == "1abc"
    assert result.status == "error"
    assert result.retryable is True
    assert result.reason_codes == ["worker_process_died"]
    assert result.n_metals == 0
    assert result.rows == []
    assert result.bond_rows == []
    assert result.candidate_rows == []
    assert "4321" in result.error and "1abc" in result.error
    assert result.alchemy_commit == cfg.alchemy_commit
    assert result.gemmi_version == cfg.gemmi_version
    assert result.ccp4_version == cfg.ccp4_version

    row = writers._manifest_row(
        result,
        resume=False,
        bonds_enabled=True,
        prior_bond_counts={},
        prior_candidate_counts={},
    )
    assert set(row) == set(MANIFEST_COLUMNS)
    assert row["status"] == "error"
    assert row["retryable"] is True
    assert row["reason_codes"] == "worker_process_died"
    assert row["n_metals"] == 0


def test_worker_death_result_leaves_the_bond_counts_blank(tmp_path):
    """A killed worker reports no bond counts, not a measured zero.

    Blank means "the bond stage did not run", ``0`` means "it ran and found
    nothing", and ``--resume`` reads the difference.
    """
    cfg = _reference_cfg(str(tmp_path))
    result = worker._worker_death_result("1abc", cfg, 7)
    assert result.n_bonds is None
    assert result.n_candidates is None

    row = writers._manifest_row(
        result,
        resume=False,
        bonds_enabled=True,
        prior_bond_counts={},
        prior_candidate_counts={},
    )
    assert row["n_bonds"] == ""
    assert row["n_candidates"] == ""


@pytest.mark.parametrize(
    "manual_inputs, expected_state",
    [(None, "final"), ({"pdb_file": "x.pdb"}, "manual")],
)
def test_worker_death_result_reports_the_run_refinement_state(
    tmp_path, manual_inputs, expected_state
):
    """Even a synthesized failure records which refinement the run targeted."""
    cfg = _reference_cfg(str(tmp_path), manual_inputs=manual_inputs)
    result = worker._worker_death_result("1abc", cfg, 7)
    assert result.refinement_state == expected_state


def test_worker_death_reason_codes_discriminate_synthesized_from_real(tmp_path):
    """``["worker_process_died"]`` alone identifies a synthesized result.

    The dispatch loop uses that exact list to tell its own synthesized row apart
    from a genuine worker result for the same entry.
    """
    cfg = _reference_cfg(str(tmp_path))
    synthesized = worker._worker_death_result("1abc", cfg, 7)
    genuine = worker._initial_result("1abc", cfg, None)

    assert synthesized.reason_codes == ["worker_process_died"]
    assert genuine.reason_codes == []
    assert genuine.reason_codes != synthesized.reason_codes


# ``_STUB_SCRIPT`` maps a pdbID to what its worker should do. A module global
# because the driver child sets it before creating the pool and the (forked)
# workers inherit it. Recognised keys:
#
#   runtime    seconds the entry "takes" before its result is returned
#   die        "announced": name the entry, then SIGKILL mid-entry (an OOM kill)
#              "silent":    SIGKILL mid-entry without ever naming the entry
#   claim      after finishing cleanly, announce a start for THIS other entry,
#              leaving the driver with a stale attribution for it
#   die_after  seconds after finishing cleanly to SIGKILL this now-idle worker
_STUB_SCRIPT: dict = {}
_STUB_MARKER_DIR = ""
_DRIVER_HARD_TIMEOUT_S = 60.0


def _announce(state, pdb_id):
    """Announce like a real worker, on builds that have the mechanism.

    Without ``_announce_inflight`` the stub has to behave like an ordinary
    worker, so that these tests fail on the driver's observable behaviour rather
    than on an AttributeError raised inside the test's own fake.
    """
    getattr(worker, "_announce_inflight", lambda *args: None)(state, pdb_id)


def _first_visit(pdb_id):
    """True the first time this entry is seen, across every worker process."""
    marker = os.path.join(_STUB_MARKER_DIR, f"visited-{pdb_id}")
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        return False
    return True


def _kill_self_after(delay):
    """SIGKILL this worker ``delay`` seconds from now, from a helper thread.

    The delay has to outlast the entry's own return so the death lands after the
    result was delivered, which no in-band code path can arrange.
    """

    def kill():
        time.sleep(delay)
        os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=kill, daemon=True).start()


def _stub_process(pdb_id):
    """Stand in for ``worker.process``: no CCP4, no downloads, scripted deaths.

    Every entry announces itself exactly as the real worker does, so the driver
    sees the notification stream it would see in production.
    """
    cfg = worker._CFG
    if cfg is None:  # pragma: no cover - would mean the initializer never ran
        raise RuntimeError("worker configuration has not been initialized")
    step = _STUB_SCRIPT.get(pdb_id, {})
    runtime = float(step.get("runtime", 0.02))
    result = worker._initial_result(pdb_id, cfg, cfg.manual_inputs)

    die = step.get("die")
    if die and _first_visit(pdb_id):
        if die == "announced":
            _announce("start", pdb_id)
        # The pause has to outlast the driver's first roster snapshot, so that
        # the snapshot still contains this pid when the kill lands.
        time.sleep(runtime)
        os.kill(os.getpid(), signal.SIGKILL)

    _announce("start", pdb_id)
    time.sleep(runtime)
    _announce("end", pdb_id)

    if step.get("claim"):
        _announce("start", step["claim"])
    if step.get("die_after") is not None:
        _kill_self_after(float(step["die_after"]))

    result.status = "ok"
    result.n_metals = 0
    result.no_metals = True
    result.retryable = False
    result.runtime_s = runtime
    result.n_bonds = 0
    result.n_candidates = 0
    return result


def _driver_child(argv, script, marker_dir, stall_grace, channel, session_ready):
    """Run the real ``driver_pool._run`` with the per-entry pipeline stubbed out.

    Executed in its own process so the parent can impose a hard timeout: a
    driver that waits forever for a dead worker's result must fail the test
    quickly rather than hang the suite.
    """
    # Its own process group, so a hung driver can be reaped whole rather than
    # leaking its Pool children into the test host.
    if hasattr(os, "setsid"):
        os.setsid()
    session_ready.set()

    global _STUB_SCRIPT, _STUB_MARKER_DIR
    _STUB_SCRIPT = script
    _STUB_MARKER_DIR = marker_dir
    try:
        driver_pool.resolve_ccp4_environment = lambda args: (dict(os.environ), None)
        driver_pool.process = _stub_process
        if stall_grace is not None:
            driver_pool.WORKER_STALL_GRACE_S = stall_grace
        args = cli.parse_args(argv)
        run_log = runlog._RunLog(args, "pytest")
        channel.put(("exit_code", driver_pool._run(args, run_log)))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        channel.put(("crash", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def _terminate_driver_tree(child, session_ready):
    """Force-stop a timed-out driver and every process in its test session."""
    killed_group = False
    if (
        child.pid is not None
        and session_ready.is_set()
        and hasattr(os, "killpg")
        and hasattr(signal, "SIGKILL")
    ):
        try:
            # _driver_child calls setsid() before setting the event, so its pid
            # is the process-group id, and stays valid even once the driver
            # itself has died and left only Pool grandchildren behind.
            os.killpg(child.pid, signal.SIGKILL)
            killed_group = True
        except ProcessLookupError:
            killed_group = True
    if child.is_alive() and not killed_group:
        child.kill()
    child.join(10)


def _run_driver(tmp_path, script, stall_grace=None, workers=None):
    """Drive ``driver_pool._run`` over ``script`` in a child process, with a timeout."""
    ids = list(script)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    id_file = tmp_path / "ids.txt"
    id_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    argv = [
        "--id-file",
        str(id_file),
        "--output-dir",
        str(output_dir),
        "--pdb-redo-root",
        str(tmp_path / "mirror"),
        "--pdb-redo-cache",
        str(tmp_path / "cache"),
        "--workers",
        str(workers if workers is not None else len(ids)),
    ]

    ctx = multiprocessing.get_context("fork")
    channel = ctx.Queue()
    session_ready = ctx.Event()
    child = ctx.Process(
        target=_driver_child,
        args=(argv, dict(script), str(marker_dir), stall_grace, channel, session_ready),
    )
    started = time.monotonic()
    child.start()
    try:
        child.join(_DRIVER_HARD_TIMEOUT_S)
        if child.is_alive():
            _terminate_driver_tree(child, session_ready)
            pytest.fail(
                "the driver did not finish within "
                f"{_DRIVER_HARD_TIMEOUT_S:.0f}s after a worker was killed: "
                "a dead worker's entry is being waited on forever"
            )
        elapsed = time.monotonic() - started
        try:
            kind, payload = channel.get(timeout=10)
        except queue.Empty:
            _terminate_driver_tree(child, session_ready)
            pytest.fail(
                f"the driver process exited (code {child.exitcode}) without "
                "reporting an outcome"
            )
        assert kind == "exit_code", payload
        return payload, output_dir, elapsed
    finally:
        if child.is_alive():  # pragma: no cover - only on an unexpected hang
            _terminate_driver_tree(child, session_ready)
        channel.close()


def _read_manifest(output_dir):
    with open(output_dir / "manifest.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == MANIFEST_COLUMNS
        return list(reader)


@_POSIX_KILL
@pytest.mark.parametrize(
    "script, stall_grace",
    [
        (
            {
                "aaaa": {"die": "announced", "runtime": 1.2},
                "bbbb": {},
                "cccc": {},
                "dddd": {},
            },
            None,
        ),
        ({"aaaa": {"die": "silent", "runtime": 2.0}}, 1.0),
    ],
    ids=["worker_named_its_entry", "worker_died_before_naming_its_entry"],
)
def test_driver_recovers_from_a_sigkilled_worker(tmp_path, script, stall_grace):
    """A worker killed mid-entry must not hang the batch.

    ``multiprocessing.Pool`` never yields a result for a task whose worker died
    abnormally, so a driver that waits on the pool alone spins on TimeoutError
    forever and writes neither manifest nor run log.
    """
    ids = list(script)
    victim = ids[0]
    exit_code, output_dir, elapsed = _run_driver(
        tmp_path, script, stall_grace=stall_grace, workers=3 if len(ids) > 1 else 1
    )

    assert elapsed < _DRIVER_HARD_TIMEOUT_S
    rows = _read_manifest(output_dir)
    assert {row["pdbID"] for row in rows} == set(ids)
    assert len(rows) == len(ids), "each entry is written exactly once"

    by_id = {row["pdbID"]: row for row in rows}
    lost = by_id[victim]
    assert lost["status"] == "error"
    assert lost["retryable"] == "True"
    assert lost["reason_codes"] == "worker_process_died"
    assert victim in lost["error"]
    # Blank, not "0": the bond stage never ran for this entry.
    assert lost["n_bonds"] == ""
    assert lost["n_candidates"] == ""

    for pdb_id in ids[1:]:
        assert by_id[pdb_id]["status"] == "ok"

    assert exit_code == 1


@_POSIX_KILL
def test_idle_worker_death_does_not_wedge_pool_shutdown(tmp_path):
    """The driver still exits when a worker is killed while awaiting a task.

    A worker waiting for its next task blocks inside the pool task queue's
    ``get()`` holding that queue's lock, so a process killed there never
    releases it and ``Pool.terminate`` blocks acquiring the same lock. The
    failure is invisible in the results: every row is written and only teardown
    hangs, leaving no run log and no exit code.

    ``aaaa`` finishes quickly and is killed 0.3 s later, by which time it is
    idle and holding the lock; ``bbbb`` is still working, so the driver has not
    reached teardown yet. Reaching the assertions at all is the signal.
    """
    script = {
        "aaaa": {"runtime": 0.05, "die_after": 0.3},
        "bbbb": {"runtime": 1.5},
    }
    exit_code, output_dir, elapsed = _run_driver(tmp_path, script, workers=2)

    assert elapsed < _DRIVER_HARD_TIMEOUT_S
    by_id = {row["pdbID"]: row for row in _read_manifest(output_dir)}
    assert set(by_id) == set(script)
    # The kill lands after both entries are done, so the work itself completed.
    assert by_id["aaaa"]["status"] == "ok"
    assert by_id["bbbb"]["status"] == "ok"
    assert exit_code == 0


@_POSIX_KILL
def test_silent_death_fallback_attributes_only_the_outstanding_entry(tmp_path):
    """A silent fourth-task death must not be blamed on a completed first task.

    One worker completes ``aaaa``, ``bbbb`` and ``cccc`` in order, then dies
    during ``dddd`` before announcing it. Exactly one id is outstanding at
    fallback time, so a fallback that ignores the ids already written blames
    ``aaaa`` -- a second, failed row for an entry that succeeded.
    """
    script = {
        "aaaa": {"runtime": 0.02},
        "bbbb": {"runtime": 0.02},
        "cccc": {"runtime": 0.02},
        "dddd": {"die": "silent", "runtime": 0.5},
    }
    exit_code, output_dir, _ = _run_driver(tmp_path, script, stall_grace=0.5, workers=1)
    rows = _read_manifest(output_dir)

    assert len(rows) == len(script)
    assert [row["pdbID"] for row in rows].count("dddd") == 1
    assert {row["pdbID"] for row in rows} == set(script)
    by_id = {row["pdbID"]: row for row in rows}
    for pdb_id in ("aaaa", "bbbb", "cccc"):
        assert by_id[pdb_id]["status"] == "ok"
    assert by_id["dddd"]["status"] == "error"
    assert by_id["dddd"]["reason_codes"] == "worker_process_died"
    assert exit_code == 1


# One run staging every composition rule the dispatch loop has to obey: two
# workers finish their own entry and then, while idle, leave the driver holding
# a start notification for "aaaa" before being killed a second apart; a third
# dies idle holding nothing; "aaaa" is really processed elsewhere and returns a
# genuine result long after the driver has given up on it, and "bbbb" later
# still, so the loop is still running when it does.
#
# "ffff" is scaffolding, not a case: a worker SIGKILLed while blocked in
# ``inqueue.get()`` holds the pool's task-queue read lock forever and
# ``Pool.terminate()`` then deadlocks in ``_help_stuff_finish``. The lock is
# held by the first worker to run out of work, so "ffff" (far the quickest
# entry, and never killed) owns it and the workers that do get killed queue
# harmlessly behind it.
_STALE_ATTRIBUTION_SCRIPT = {
    "aaaa": {"runtime": 3.0},  # late, real
    "bbbb": {"runtime": 3.8},  # keeps it open
    "cccc": {"runtime": 0.5, "claim": "aaaa", "die_after": 0.6},
    "dddd": {"runtime": 0.5, "claim": "aaaa", "die_after": 1.2},
    "eeee": {"runtime": 0.5, "die_after": 0.9},  # unattributed
    "ffff": {"runtime": 0.02},  # lock holder
}


@pytest.fixture(scope="module")
def stale_attribution_batch(tmp_path_factory):
    """One real ``driver_pool._run`` over ``_STALE_ATTRIBUTION_SCRIPT``.

    Module-scoped: the run takes a few seconds and the three tests below read
    different invariants out of the one manifest.
    """
    tmp_path = tmp_path_factory.mktemp("stale_attribution")
    exit_code, output_dir, elapsed = _run_driver(tmp_path, _STALE_ATTRIBUTION_SCRIPT)
    rows = _read_manifest(output_dir)
    return {
        "exit_code": exit_code,
        "elapsed": elapsed,
        "rows": rows,
        "by_id": {row["pdbID"]: row for row in rows},
    }


@_POSIX_KILL
def test_lost_entry_is_written_once_even_if_a_real_result_arrives(
    stale_attribution_batch,
):
    """A genuine result for an already-declared-lost entry is dropped.

    Two manifest rows for one entry would make ``--resume`` see a completed
    entry whose data rows were never produced, and the extra completion would
    end the batch with another entry still unwritten. ``aaaa`` runs to
    completion in its own worker, yet is declared lost first because the worker
    that died held a start notification for it, so dropping the guard in
    ``driver_pool._run`` writes ``aaaa`` twice and never writes ``bbbb``.
    """
    batch = stale_attribution_batch
    ids = list(_STALE_ATTRIBUTION_SCRIPT)

    assert [row["pdbID"] for row in batch["rows"]].count("aaaa") == 1, (
        "the entry declared lost was written twice: the synthesized row and "
        "the real result that arrived afterwards"
    )
    assert {row["pdbID"] for row in batch["rows"]} == set(ids), (
        "an extra completion ended the batch before every entry was written"
    )
    assert len(batch["rows"]) == len(ids)

    lost = batch["by_id"]["aaaa"]
    assert lost["status"] == "error"
    assert lost["retryable"] == "True"
    assert lost["reason_codes"] == "worker_process_died", (
        "the real result overwrote the synthesized lost-entry row"
    )
    # Blank counts: the synthesized row stands, the real result's do not.
    assert lost["n_bonds"] == ""
    assert lost["n_candidates"] == ""
    assert batch["exit_code"] == 1


@_POSIX_KILL
def test_a_repeatedly_missing_pid_does_not_duplicate_its_entry(stale_attribution_batch):
    """An entry is declared lost once, however the roster churns.

    Two workers die a second apart, both holding a start notification for
    ``aaaa``. A second lost-entry row would both duplicate the entry and inflate
    the completion count, ending the batch early.
    """
    batch = stale_attribution_batch
    lost_rows = [
        row for row in batch["rows"] if row["reason_codes"] == "worker_process_died"
    ]
    assert [row["pdbID"] for row in lost_rows] == ["aaaa"], (
        "a single lost entry produced more than one synthesized row"
    )


@_POSIX_KILL
def test_an_entry_released_before_the_death_is_not_declared_lost(
    stale_attribution_batch,
):
    """A worker that finished its entry before dying loses nothing.

    Three workers are killed after announcing the end of their own entry and
    returning a real result, ``eeee``'s death being unattributed altogether.
    None of them may fabricate a failure, for their own entry or for any other
    that happens to be outstanding when the death is noticed.
    """
    batch = stale_attribution_batch
    for pdb_id in ("bbbb", "cccc", "dddd", "eeee", "ffff"):
        row = batch["by_id"][pdb_id]
        assert row["status"] == "ok", (
            f"{pdb_id} returned a real result but was recorded as "
            f"{row['status']} ({row['reason_codes']})"
        )
        assert row["reason_codes"] == ""


def _spawn_task(payload):
    """Announce, then either finish or die abnormally, inside a spawn worker."""
    action, pdb_id = payload
    worker._announce_inflight("start", pdb_id)
    if action == "die":
        # SIGKILL is not portable, os._exit is; the pause keeps the death after
        # the driver's first roster snapshot.
        time.sleep(0.5)
        os._exit(9)
    worker._announce_inflight("end", pdb_id)
    cfg = worker._CFG
    assert cfg is not None
    return (pdb_id, cfg.output_dir, sorted(cfg.cofactors)[:1])


def test_spawn_workers_report_inflight_entries_and_their_deaths(tmp_path):
    """The notification queue and config survive the spawn start method.

    Windows has no fork: ``SimpleQueue`` and the config reach the worker only by
    being pickled through ``Pool(initializer=..., initargs=...)``, and without
    that the workers stop announcing and every death is unattributable.
    """
    ctx = multiprocessing.get_context("spawn")
    cfg = _reference_cfg(str(tmp_path))
    inflight = ctx.SimpleQueue()
    assignments: dict[int, str] = {}
    worker_pids: set[int] = set()
    dead_pids: set[int] = set()
    finished: list = []

    with ctx.Pool(2, initializer=worker._init_worker, initargs=(cfg, inflight)) as pool:
        # Prime the roster, as the driver's first loop iteration does.
        driver_pool._dead_worker_pids(pool, worker_pids)
        results = pool.imap_unordered(
            _spawn_task, [("ok", "1abc"), ("die", "2xyz")], chunksize=1
        )
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline and not (finished and dead_pids):
            try:
                finished.append(results.next(timeout=0.5))
            except driver_pool.MultiprocessingTimeoutError:
                pass
            driver_pool._drain_inflight(inflight, assignments)
            dead_pids |= driver_pool._dead_worker_pids(pool, worker_pids)
        pool.terminate()

    assert finished, "the healthy spawn worker returned no result"
    pdb_id, output_dir, cofactor_sample = finished[0]
    assert pdb_id == "1abc"
    assert output_dir == str(tmp_path)
    assert cofactor_sample == sorted(cfg.cofactors)[:1]

    assert "1abc" not in assignments.values()
    assert dead_pids, "the death of a spawn worker went unnoticed"
    attributed = {assignments.get(pid) for pid in dead_pids}
    assert "2xyz" in attributed, (
        "the killed spawn worker's entry was not recoverable from its "
        f"announcement (assignments={assignments}, dead={dead_pids})"
    )


@pytest.mark.parametrize(
    "manual_inputs",
    [
        None,
        {"pdb_file": "a.pdb", "mtz_file": "a.mtz", "cif_file": None, "data_json": None},
    ],
)
def test_worker_config_is_picklable(tmp_path, manual_inputs):
    """The worker config must survive pickling, as spawn requires.

    Under fork the config is inherited and an unpicklable member goes unnoticed;
    under spawn it is pickled into ``Pool`` initargs, where the same member
    breaks every worker before it starts.
    """
    cfg = _reference_cfg(str(tmp_path), manual_inputs=manual_inputs)
    restored = pickle.loads(pickle.dumps(cfg))

    assert restored == cfg
    assert restored.cofactors == cfg.cofactors
    assert restored.cofactors, "the bundled cofactor catalog must be loaded"
    assert worker._initial_result(
        "1abc", restored, restored.manual_inputs
    ) == worker._initial_result("1abc", cfg, cfg.manual_inputs)


def test_worker_config_cannot_be_edited_by_a_worker(tmp_path):
    """One run decides the config; no worker may change it for itself.

    Each worker holds its own unpickled copy, so an edit here is invisible to
    the driver and the other workers: one process would behave differently for
    one entry with nothing in the manifest to say so.
    """
    cfg = _reference_cfg(str(tmp_path))

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.keep = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.cofactors = frozenset()  # type: ignore[misc]

    assert cfg.keep is False


def test_the_driver_maps_its_options_onto_the_worker_config(tmp_path):
    """The one hand-written mapping from CLI options to config fields.

    ``WorkerConfig`` makes a missing or misspelled *field* impossible, but not a
    field wired to the wrong option -- ``keep=args.bonds`` type-checks perfectly.
    """
    args = cli.parse_args(
        [
            "--id",
            "109m",
            "--output-dir",
            str(tmp_path),
            "--keep-intermediates",
            "--no-bonds",
            "--density-map-scope",
            "full",
            "--ccp4-timeout",
            "42",
        ]
    )
    env = {"PATH": "/nonexistent"}
    run_log = runlog._RunLog(args, "pytest")

    cfg = driver_pool._worker_config(
        args,
        env,
        str(tmp_path / "root"),
        str(tmp_path / "cache"),
        frozenset({"HEM"}),
        None,
        driver_pool._ConfidencePlan(),
        run_log,
    )

    assert isinstance(cfg, worker.WorkerConfig)
    # The log keeps the two hashes the id was composed from, so a changed id can
    # be attributed to a file.
    assert cfg.reference_data_id == reference_data.reference_data_id()
    assert run_log.details["reference_data_id"] == cfg.reference_data_id
    assert (
        run_log.details["metallocofactors_id_sha256"]
        == reference_data.reference_data_checksums()["metallocofactors_id.txt"]
    )
    assert (
        run_log.details["metal_distances_info_sha256"]
        == reference_data.reference_data_checksums()["metal_distances_info.txt"]
    )
    assert cfg.root == str(tmp_path / "root")
    assert cfg.cache_root == str(tmp_path / "cache")
    assert cfg.output_dir == str(tmp_path)
    assert cfg.env == env
    assert cfg.cofactors == frozenset({"HEM"})
    assert cfg.keep is True
    assert cfg.bonds is False
    assert cfg.density_map_scope == "full"
    assert cfg.ccp4_timeout_s == 42
    assert cfg.manual_inputs is None
    # ``--id`` means an entry may be fetched; a manual run may not.
    assert cfg.allow_download is True
    manual = cli.parse_args(
        [
            "--pdb-file",
            str(tmp_path / "a.pdb"),
            "--mtz-file",
            str(tmp_path / "a.mtz"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    manual_cfg = driver_pool._worker_config(
        manual,
        env,
        str(tmp_path / "root"),
        None,
        frozenset(),
        {"pdb_file": "a.pdb", "mtz_file": "a.mtz", "cif_file": None, "data_json": None},
        driver_pool._ConfidencePlan(),
        runlog._RunLog(manual, "pytest"),
    )
    assert manual_cfg.allow_download is False
    assert manual_cfg.manual_inputs is not None
    assert run_log.details["gemmi_version"] == cfg.gemmi_version
    assert run_log.details["ccp4_version"] == cfg.ccp4_version


def _never_finishing_process(pdb_id):
    """A per-entry stub that outlives the signal, so workers are busy.

    Module level, not a closure: ``Pool`` pickles the task callable even under
    ``fork``, and a locally defined function fails with "Can't pickle local
    object" before any worker starts.
    """
    time.sleep(30)
    return worker._initial_result(pdb_id, worker._CFG, None)


def _sigterm_driver_child(argv, ready, channel):
    """Run the real ``main.main`` with a slow stub so SIGTERM lands mid-run."""
    if hasattr(os, "setsid"):
        os.setsid()

    driver_pool.resolve_ccp4_environment = lambda args: (dict(os.environ), None)
    driver_pool.process = _never_finishing_process
    ready.set()
    try:
        channel.put(("exit_code", main.main(list(argv))))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        channel.put(("crash", f"{type(exc).__name__}: {exc}"))


@_POSIX_KILL
def test_sigterm_to_the_driver_stops_its_workers_and_writes_a_log(tmp_path):
    """SIGTERM must unwind through cleanup instead of killing the driver dead.

    Under SIGTERM's default disposition no ``finally`` runs: the pool is never
    shut down, its children are reparented to init and keep driving CCP4
    subprocesses, and nothing is written to explain the stop. The driver is
    signalled directly rather than through its process group, so the workers
    only stop if the driver itself stops them.
    """
    output_dir = tmp_path / "out"
    id_file = tmp_path / "ids.txt"
    id_file.write_text(" ".join(f"e{i:03d}" for i in range(8)), encoding="utf-8")
    argv = [
        "--id-file",
        str(id_file),
        "--workers",
        "4",
        "--output-dir",
        str(output_dir),
        "--pdb-redo-root",
        str(tmp_path / "absent-mirror"),
        "--pdb-redo-cache",
        str(tmp_path / "cache"),
    ]

    ctx = multiprocessing.get_context("fork")
    ready, channel = ctx.Event(), ctx.Queue()
    child = ctx.Process(target=_sigterm_driver_child, args=(argv, ready, channel))
    child.start()
    try:
        assert ready.wait(30), "the driver process never started"
        driver_pid = child.pid
        assert driver_pid is not None
        workers: list[int] = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(workers) < 4:
            time.sleep(0.2)
            workers = _child_pids(driver_pid)
        assert workers, "no pool workers appeared to be signalled"

        os.kill(driver_pid, signal.SIGTERM)
        child.join(60)
        assert not child.is_alive(), "the driver did not exit within 60s of SIGTERM"

        kind, payload = channel.get(timeout=10)
        assert kind == "exit_code", payload
        assert payload != 0, "an interrupted run must not report success"

        time.sleep(1.0)
        alive = [pid for pid in workers if _process_exists(pid)]
        assert alive == [], f"workers outlived the driver: {alive}"
        log_dir = os.path.join(output_dir, runlog.DEFAULT_LOG_DIRNAME)
        logs = [name for name in os.listdir(log_dir) if name.endswith(".log")]
        assert logs, "no run log was written for the interrupted run"
    finally:
        _terminate_driver_tree(child, ready)
        channel.close()


def _child_pids(pid):
    """Direct children of ``pid`` (POSIX only, used to find pool workers)."""
    result = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    return [int(line) for line in result.stdout.split()]


def _process_exists(pid):
    return os.path.exists(f"/proc/{pid}")
