"""Automatic sizing across CPU topologies, memory allowances, and entry mixes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from helpers import automatic_limits

from driver import dispatch, resources

if TYPE_CHECKING:
    from multiprocessing.pool import AsyncResult

    from worker.contracts import EntryResult

GIB = 1024**3


@pytest.mark.parametrize(
    ("logical", "physical", "available_gib", "expected"),
    [
        (2, 2, 4, 2),
        (8, 4, 8, 4),
        (16, 8, 12, 8),
        (32, 16, 20, 16),
        (32, 16, 30, 16),
        (128, 64, 240, 64),
        (128, 64, 8, 4),
        (1, 1, 1, 1),
    ],
)
def test_cpu_and_memory_profiles(
    monkeypatch: pytest.MonkeyPatch,
    logical: int,
    physical: int,
    available_gib: int,
    expected: int,
) -> None:
    monkeypatch.setattr(resources, "available_cpu_count", lambda: logical)
    monkeypatch.setattr(resources, "available_physical_cpu_count", lambda: physical)
    monkeypatch.setattr(resources, "available_cpu_quota", lambda: None)
    cpu, memory = automatic_limits(available_gib * GIB)
    assert memory is not None
    assert min(cpu, memory) == expected


def test_explicit_limit_cannot_override_less_available_memory() -> None:
    assert automatic_limits(8 * GIB, memory_limit_bytes=100 * GIB)[1] == 4
    assert automatic_limits(None)[1] == 1
    assert automatic_limits(None, memory_limit_bytes=12 * GIB)[1] == 8


def test_topology_counts_sockets_and_partial_affinity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("driver.resources.sys.platform", "linux")
    monkeypatch.setattr(resources, "CPU_TOPOLOGY_ROOT", str(tmp_path))

    def affinity(_pid: int) -> set[int]:
        return {0, 1, 2, 3}

    monkeypatch.setattr(
        "driver.resources.os.sched_getaffinity", affinity, raising=False
    )
    # Two siblings on socket 0, and distinct cores on socket 1. Core IDs may
    # repeat between sockets; an affinity set need not contain every sibling.
    for cpu, package, core in [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 1, 1)]:
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(str(package))
        (topology / "core_id").write_text(str(core))
    assert resources.available_physical_cpu_count() == 3
    (tmp_path / "cpu3/topology/core_id").write_text("-1")
    assert resources.available_physical_cpu_count() is None


def test_unknown_topology_does_not_assume_smt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 12)
    monkeypatch.setattr(resources, "available_physical_cpu_count", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_quota", lambda: None)
    assert resources.worker_limits_for_budget(None)[0] == 10
    monkeypatch.setattr(resources, "available_cpu_quota", lambda: 3)
    assert resources.worker_limits_for_budget(None)[0] == 3


def test_cpu_quota_respects_ancestors_and_fractional_allowances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("driver.resources.sys.platform", "linux")
    root = tmp_path / "cgroup"
    child = root / "job"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/job\n")
    monkeypatch.setattr(resources, "CGROUP_ROOT", str(root))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(membership))
    (root / "cpu.max").write_text("250000 100000")
    (child / "cpu.max").write_text("800000 100000")
    assert resources.available_cpu_quota() == 2
    (root / "cpu.max").write_text("max 100000")
    (child / "cpu.max").write_text("50000 100000")
    assert resources.available_cpu_quota() == 1
    (child / "cpu.max").write_text("invalid")
    assert resources.available_cpu_quota() is None
    membership.write_text("0::/../../escape\n")
    assert resources.available_cpu_quota() is None


def test_idle_workers_prevent_over_admission_but_allow_small_entries() -> None:
    ledger = dispatch._AdmissionLedger(workers=8)  # pyright: ignore[reportPrivateUsage]
    ledger.admit(
        cast("AsyncResult[EntryResult]", None),
        resources.EntryMemoryEstimate("active", 2 * GIB, "test"),
    )
    pending = [
        resources.EntryMemoryEstimate("large", 3 * GIB, "test"),
        resources.EntryMemoryEstimate("small", GIB, "test"),
    ]
    # Six other idle workers consume 3 GiB after the candidate starts. Only
    # the small candidate fits, although both fit if idle overhead is ignored.
    admitted = ledger.pop_admissible(pending, 6 * GIB)
    assert admitted is not None and admitted.pdb_id == "small"
    assert [entry.pdb_id for entry in pending] == ["large"]
    # An oversized singleton must still make progress after the pool drains.
    ledger.release("active")
    assert ledger.pop_admissible(pending, GIB) is not None


def test_missing_inputs_keep_conservative_fallback(tmp_path: Path) -> None:
    estimate = resources.estimate_entry_memory("none", str(tmp_path))
    assert estimate.source == "default"
    assert estimate.bytes == 2 * GIB
