"""Test worker limits, memory probes, and per-entry memory estimates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

import cli
from driver import pool, resources, runlog


def test_worker_limits_leave_headroom_and_respect_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both limits are floored at one, so a small machine still runs.

    The CPU limit leaves two cores for the driver and the OS; the memory limit
    bounds resident process overhead; entry admission accounts for the maps.
    """
    monkeypatch.setattr(resources, "available_physical_cpu_count", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_quota", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 16)
    monkeypatch.setattr(
        resources,
        "available_memory_bytes",
        lambda: 8 * resources.AUTO_WORKER_MEMORY_BYTES,
    )
    assert resources.automatic_worker_limits() == (14, 12)

    monkeypatch.setattr(resources, "available_physical_cpu_count", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_quota", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 1)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 0)
    assert resources.automatic_worker_limits() == (1, 1)


def test_unknown_memory_limits_concurrency_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown memory uses one worker until an explicit allowance is supplied."""
    monkeypatch.setattr(resources, "available_physical_cpu_count", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_quota", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 8)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: None)

    cpu_limit, memory_limit = resources.automatic_worker_limits()
    assert cpu_limit == 6
    assert memory_limit == 1


def test_explicit_memory_controls_bound_automatic_worker_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    monkeypatch.setattr(resources, "available_physical_cpu_count", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_quota", lambda: None)
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 64)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 100 * gib)

    # The explicit 20 GiB capacity is tighter than detection. At 80%, with the
    # 4 GiB minimum reserve, it supplies a 16 GiB budget: 16 resident workers.
    assert resources.automatic_worker_limits(memory_limit_bytes=20 * gib) == (62, 16)

    # On a large allocation the requested utilization controls the reserve.
    assert resources.scheduling_memory_budget(100 * gib, utilization=0.9) == (
        90 * gib,
        10 * gib,
    )


def test_explicit_workers_are_still_capped_for_process_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(["--workers", "50", "--output-dir", str(tmp_path)])
    run_log = runlog.RunLog(args, "pytest")

    def worker_limits_for_budget(_budget: object) -> tuple[int, int]:
        return 8, 3

    monkeypatch.setattr(pool, "worker_limits_for_budget", worker_limits_for_budget)

    workers = pool.choose_worker_count(
        args, entry_count=20, run_log=run_log, memory_budget_bytes=None
    )

    assert workers == 3
    assert run_log.details["requested_workers"] == 50
    assert run_log.details["selected_workers"] == 3


def _host_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_bytes: int
) -> Path:
    """Point the memory probe at a synthetic ``/proc/meminfo``."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {host_bytes // 1024} kB\n", encoding="ascii")
    monkeypatch.setattr(resources, "PROC_MEMINFO_PATH", str(meminfo))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(tmp_path / "no-cgroup"))
    monkeypatch.setattr("driver.resources.sys.platform", "linux")
    return meminfo


def test_available_memory_is_read_from_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker sizing follows what the machine reports as free."""
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=7 * budget)

    assert resources.available_memory_bytes() == 7 * budget
    assert resources.automatic_worker_limits()[1] == 10


def test_an_unreadable_meminfo_uses_one_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memory probe that fails must not be read as zero available memory."""
    monkeypatch.setattr(
        resources, "PROC_MEMINFO_PATH", str(tmp_path / "definitely-absent")
    )
    monkeypatch.setattr("driver.resources.sys.platform", "linux")

    def unavailable_sysconf(_name: str) -> int:
        return 0

    monkeypatch.setattr(
        "driver.resources.os.sysconf", unavailable_sysconf, raising=False
    )

    assert resources.available_memory_bytes() is None
    assert resources.automatic_worker_limits()[1] == 1


def _cgroup_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    membership: str,
    groups: Mapping[str, Mapping[str, str]],
) -> None:
    """Build a synthetic cgroup tree and point the memory probe at it.

    ``groups`` maps each group path under the root (``""`` for the root
    itself) to the control files it holds; ``membership`` is the content of
    this process's ``/proc/self/cgroup``.
    """
    cgroup_root = tmp_path / "cgroup"
    for group, files in groups.items():
        directory = cgroup_root / group
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (directory / name).write_text(content, encoding="ascii")
    cgroup = tmp_path / "self-cgroup"
    cgroup.write_text(membership, encoding="ascii")
    monkeypatch.setattr(resources, "CGROUP_ROOT", str(cgroup_root))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(cgroup))


def test_a_cgroup_v2_memory_limit_caps_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containers must not schedule against host RAM they cannot access."""
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=64 * budget)
    # A limit far below host memory, of the kind SLURM or a container imposes.
    _cgroup_tree(
        tmp_path,
        monkeypatch,
        "0::/batch\n",
        {"batch": {"memory.max": str(4 * budget), "memory.current": str(budget)}},
    )

    assert resources.available_memory_bytes() == 3 * budget


def test_unlimited_cgroup_leaves_host_memory_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=8 * budget)
    _cgroup_tree(
        tmp_path,
        monkeypatch,
        "0::/\n",
        {"": {"memory.max": "max", "memory.current": "123"}},
    )

    assert resources.available_memory_bytes() == 8 * budget


def test_parent_cgroup_limit_applies_when_child_is_unlimited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=32 * budget)
    _cgroup_tree(
        tmp_path,
        monkeypatch,
        "0::/batch/task\n",
        {
            "": {"memory.max": "max", "memory.current": "0"},
            "batch": {
                "memory.max": str(6 * budget),
                "memory.current": str(2 * budget),
            },
            "batch/task": {"memory.max": "max", "memory.current": str(budget)},
        },
    )

    assert resources.available_memory_bytes() == 4 * budget


def test_a_cgroup_v1_memory_limit_caps_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=32 * budget)
    _cgroup_tree(
        tmp_path,
        monkeypatch,
        "7:cpu,cpuacct:/batch\n8:memory:/batch\n",
        {
            "memory/batch": {
                "memory.limit_in_bytes": str(5 * budget),
                "memory.usage_in_bytes": str(budget),
            }
        },
    )

    assert resources.available_memory_bytes() == 4 * budget


def test_density_memory_estimate_grows_with_cell_volume() -> None:
    small = resources.estimate_from_properties(
        "tiny",
        {"AAXIS": 30, "BAXIS": 40, "CAXIS": 50, "RESOLUTION": 2.0},
    )
    large = resources.estimate_from_properties(
        "huge",
        {"AAXIS": 210, "BAXIS": 450, "CAXIS": 620, "RESOLUTION": 2.5},
    )

    assert small is not None and large is not None
    assert resources.WORKER_FIXED_OVERHEAD_BYTES < small.bytes < 2 * 1024**3
    assert large.bytes > 8 * 1024**3
    assert large.combined_map_bytes is not None


def test_entry_estimate_reads_only_leading_properties(
    tmp_path: Path,
) -> None:
    pdb_id = "1abc"
    entry = tmp_path / "ab" / pdb_id
    entry.mkdir(parents=True)
    (entry / "data.json").write_text(
        '{"properties":{"AAXIS":200,"BAXIS":400,"CAXIS":600,'
        '"RESOLUTION":2.5},"large_later_value":"' + "x" * (5 * 1024**2) + '"}',
        encoding="utf-8",
    )

    estimate = resources.estimate_entry_memory(pdb_id, str(tmp_path))

    assert estimate.source == "data_json"
    assert estimate.bytes > resources.AUTO_WORKER_MEMORY_BYTES


def test_entry_estimate_falls_back_to_mtz_size(tmp_path: Path) -> None:
    pdb_id = "1abc"
    entry = tmp_path / "ab" / pdb_id
    entry.mkdir(parents=True)
    (entry / "data.json").write_text("", encoding="utf-8")
    mtz = entry / f"{pdb_id}_final.mtz"
    mtz.write_bytes(b"x" * 1024)

    estimate = resources.estimate_entry_memory(pdb_id, str(tmp_path))

    assert estimate.source == "mtz_size"
    assert estimate.bytes == resources.AUTO_WORKER_MEMORY_BYTES
