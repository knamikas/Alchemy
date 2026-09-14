"""Regression tests for sustained pressure and recovery between large entries."""

from __future__ import annotations

import csv
import gc
import threading
import time
import weakref
from multiprocessing.pool import ThreadPool
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import cli
import worker
import worker_memory
from codes import EntryStatus
from driver import dispatch, pool, resources
from driver.memory_admission import MemoryAdmission
from driver.runlog import RunLog
from worker_contracts import EntryResult, WorkerConfig

GIB = 1024**3


def controller() -> MemoryAdmission:
    return MemoryAdmission(16 * GIB, GIB // 2, 4 * GIB)


def test_reserve_flapping_does_not_ratchet_budget_to_one_worker() -> None:
    admission = controller()
    for second in range(1800):
        available = 3.9 if second % 2 else 4.1
        admission.observe(int(available * GIB), 12 * GIB, second, active=True)
    assert admission.backoffs == 1
    assert admission.budget == int(16 * GIB * 0.8)


def test_recovery_requires_sustained_headroom_and_restores_original_budget() -> None:
    admission = controller()
    admission.observe(3 * GIB, 14 * GIB, 0, active=True)
    reduced = admission.budget
    admission.observe(12 * GIB, 6 * GIB, 1, active=True)
    admission.observe(12 * GIB, 6 * GIB, 30, active=True)
    assert admission.budget == reduced
    admission.observe(12 * GIB, 6 * GIB, 31, active=True)
    assert reduced is not None and admission.budget is not None
    assert admission.maximum is not None
    assert reduced < admission.budget < admission.maximum
    for second in range(32, 300):
        admission.observe(20 * GIB, 0, second, active=False)
    # The margin is left unused even when nothing is active.
    assert admission.budget == 15 * GIB
    admission.observe(24 * GIB, 0, 330, active=False)
    admission.observe(24 * GIB, 0, 360, active=False)
    assert admission.budget == 16 * GIB


def test_brief_recovery_or_unknown_memory_does_not_reset_pressure_episode() -> None:
    admission = controller()
    observations: list[tuple[int, int | None]] = [
        (0, 3),
        (1, 10),
        (29, 3),
        (30, 10),
        (59, None),
        (60, 3),
    ]
    for second, available in observations:
        admission.observe(
            None if available is None else available * GIB,
            10 * GIB,
            second,
            active=True,
        )
    assert admission.backoffs == 1
    assert admission.recoveries == 0


def test_oversized_singleton_pauses_without_penalizing_ordinary_entries() -> None:
    admission = controller()
    assert admission.observe(2 * GIB, 30 * GIB, 0, active=True, oversized=True)
    assert admission.budget == admission.maximum
    assert not admission.observe(2 * GIB, 0, 1, active=False)
    assert admission.backoffs == 0


def test_new_pressure_after_sustained_recovery_can_back_off_again() -> None:
    admission = controller()
    admission.observe(3 * GIB, 12 * GIB, 0, active=True)
    admission.observe(10 * GIB, 10 * GIB, 1, active=True)
    admission.observe(10 * GIB, 10 * GIB, 31, active=True)
    admission.observe(3 * GIB, 12 * GIB, 32, active=True)
    assert admission.backoffs == 2


def test_worker_death_resets_recovery_timer_and_floor_is_respected() -> None:
    admission = controller()
    for second in range(100):
        admission.back_off(second)
    assert admission.budget == GIB // 2
    admission.observe(20 * GIB, 0, 100, active=False)
    admission.back_off(129)
    admission.observe(20 * GIB, 0, 130, active=False)
    admission.observe(20 * GIB, 0, 159, active=False)
    assert admission.budget == GIB // 2
    admission.observe(20 * GIB, 0, 160, active=False)
    assert admission.budget > GIB // 2


def test_unknown_budget_or_reserve_never_invents_a_limit() -> None:
    admission = MemoryAdmission(None, GIB, None)
    admission.back_off(0)
    assert not admission.observe(None, 0, 100, active=True)
    assert admission.budget is None


def test_cleanup_runs_after_analysis_locals_are_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Payload:
        pass

    refs: list[weakref.ReferenceType[Payload]] = []
    result = entry_result()
    released: list[bool] = []

    def analyze(_: str) -> EntryResult:
        payload = Payload()
        refs.append(weakref.ref(payload))
        return result

    def clean() -> None:
        gc.collect()
        released.append(refs[0]() is None)

    monkeypatch.setattr(worker, "_process_entry", analyze)
    monkeypatch.setattr(worker, "release_idle_memory", clean)
    assert worker.process("test") is result
    assert released == [True]


def test_unsupported_allocator_still_collects_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collected: list[bool] = []
    monkeypatch.setattr(worker_memory, "_allocator_trim", lambda: None)
    monkeypatch.setattr(gc, "collect", lambda: collected.append(True))
    worker_memory.release_idle_memory()
    assert collected == [True]


def test_housekeeping_failure_cannot_change_an_entry_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = entry_result()

    def failed_cleanup() -> None:
        raise OSError("optional allocator operation unavailable")

    def analyze(_: str) -> EntryResult:
        return result

    monkeypatch.setattr(worker, "_process_entry", analyze)
    monkeypatch.setattr(worker, "release_idle_memory", failed_cleanup)
    assert worker.process("test") is result


def entry_result() -> EntryResult:
    return EntryResult(
        pdb_id="test",
        alchemy_commit="test",
        gemmi_version="test",
        ccp4_version="test",
        reference_data_id="test",
        analysis_config_id="test",
        refinement_state="final",
    )


@pytest.mark.parametrize(
    ("counters", "expected"),
    [
        ("file 150\ninactive_file 140\nanon 50\n", 140),
        ("file 150\ninactive_file 140\nfile_dirty 20\nfile_writeback 10\n", 110),
        ("file 150\ninactive_file 140\nfile_mapped 30\nshmem 20\n", 90),
        ("file 150\ninactive_file 999\n", 150),
        ("file 999\ninactive_file 999\n", 200),
        ("file 150\ninactive_file 50\nfile_dirty 100\n", 0),
        ("file malformed\n", 0),
    ],
)
def test_only_reclaimable_file_cache_is_discounted(
    tmp_path: Path, counters: str, expected: int
) -> None:
    (tmp_path / "memory.stat").write_text(counters)
    assert resources.reclaimable_file_bytes(str(tmp_path), 200) == expected


def test_cgroup_cache_discount_preserves_ancestor_and_host_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the Linux memory probes against these fixtures on every host.
    monkeypatch.setattr("driver.resources.sys.platform", "linux")
    child = tmp_path / "job"
    child.mkdir()
    for directory, limit, usage in [(tmp_path, 100, 80), (child, 80, 70)]:
        (directory / "memory.max").write_text(str(limit))
        (directory / "memory.current").write_text(str(usage))
    (child / "memory.stat").write_text("file 60\ninactive_file 60\n")
    membership = tmp_path / "membership"
    membership.write_text("0::/job\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 1000 kB\n")
    monkeypatch.setattr(resources, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(membership))
    monkeypatch.setattr(resources, "PROC_MEMINFO_PATH", str(meminfo))
    assert resources.available_memory_bytes() == 20
    meminfo.write_text("MemAvailable: 0 kB\n")
    assert resources.available_memory_bytes() == 0


@pytest.mark.parametrize("entry_gib", [2, 7])
def test_dispatcher_recovers_parallelism_after_pressure_without_losing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_gib: int
) -> None:
    """Exercise admission in the actual dispatch loop with a cheap test pipeline."""
    lock = threading.Lock()
    started = 0
    finished = 0

    def analyze(pdb_id: str) -> EntryResult:
        nonlocal started, finished
        with lock:
            started += 1
            first_wave = started <= 4
        time.sleep(0.15 if first_wave else 0.03)
        with lock:
            finished += 1
        result = entry_result()
        result.pdb_id = pdb_id
        result.status = EntryStatus.OK
        result.no_metals = True
        result.n_metals = result.n_bonds = result.n_candidates = 0
        return result

    def available() -> int:
        with lock:
            return (3 if started and finished < 4 else 24) * GIB

    threads = ThreadPool(4)

    def thread_pool(*_args: object, **_kwargs: object) -> ThreadPool:
        return threads

    def no_op(*_args: object) -> None:
        pass

    def always_false(*_args: object) -> bool:
        return False

    def death_watch(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(
            track_submitted_entry=no_op,
            poll=list,
            stalled_losses=no_op,
            superseded=always_false,
        )

    monkeypatch.setattr(dispatch, "Pool", thread_pool)
    monkeypatch.setattr(dispatch, "SimpleQueue", lambda: None)
    monkeypatch.setattr(dispatch, "create_worker_log_queue", lambda: None)
    monkeypatch.setattr(dispatch, "start_worker_log_listener", no_op)
    monkeypatch.setattr(dispatch, "stop_log_listener", always_false)
    monkeypatch.setattr(dispatch, "_shutdown_pool", always_false)
    monkeypatch.setattr(dispatch, "process", analyze)
    monkeypatch.setattr(dispatch, "available_memory_bytes", available)
    monkeypatch.setattr(MemoryAdmission, "HEALTHY_SECONDS", 0.01)
    monkeypatch.setattr(dispatch, "_WorkerDeathWatch", death_watch)
    args = cli.parse_args(["--output-dir", str(tmp_path), "--workers", "4"])
    ids = [f"x{i:03}" for i in range(40)]
    log = RunLog(args, "pytest")
    try:
        pool.process_entries(
            args,
            ids,
            cast(WorkerConfig, None),
            4,
            pool.OutputLayout(str(tmp_path)),
            pool.ConfidencePlan(),
            log,
            resources.MemoryPlan(
                [
                    resources.EntryMemoryEstimate(p, entry_gib * GIB, "test")
                    for p in ids
                ],
                8 * GIB,
                4 * GIB,
            ),
        )
    finally:
        threads.close()
        threads.join()
    if entry_gib == 2:
        assert log.summary["memory_scheduler_budget_backoffs"] == 1
        assert log.summary["memory_scheduler_budget_recoveries"] >= 1
        assert log.summary["memory_scheduler_max_active_entries"] == 4
    else:
        # A 7 GiB entry plus three idle workers exceeds the 8 GiB budget.
        # Its pressure must not penalize the budget for ordinary entries.
        assert log.summary["memory_scheduler_budget_backoffs"] == 0
        assert log.summary["memory_scheduler_budget_recoveries"] == 0
        assert log.summary["memory_scheduler_max_active_entries"] == 1
        assert log.summary["memory_scheduler_oversized_entries"] == len(ids)
    assert log.summary["memory_scheduler_final_budget_bytes"] == 8 * GIB
    with (tmp_path / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(ids)
    assert {r["pdbID"] for r in rows} == set(ids)
    assert {r["status"] for r in rows} == {"ok"}
