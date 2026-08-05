"""What this machine will actually let a batch run use.

Keep these probes dependency-free: worker selection happens before the analysis
environment exists.
"""

import os
import sys
from multiprocessing import cpu_count
from typing import cast
from collections.abc import Callable


# Peak resident memory to budget for one worker, measured against the largest
# entries in the integration set, so it is a ceiling rather than an average.
AUTO_WORKER_MEMORY_BYTES = 1280 * 1024 * 1024

PROC_MEMINFO_PATH = "/proc/meminfo"


def available_cpu_count() -> int:
    """Return the number of CPUs this process is actually permitted to use.

    ``multiprocessing.cpu_count()`` ignores CPU affinity, so in a container or
    under a scheduler allocation it reports far more than the job was granted.
    """
    # ``getattr`` is the only way to reach a 3.13+ probe from a 3.11 floor, and
    # it erases the signature the standard library documents.
    process_cpu_count = cast(
        Callable[[], int | None] | None,
        getattr(os, "process_cpu_count", None),  # Python 3.13+
    )
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


def _read_linux_available_memory() -> int | None:
    try:
        with open(PROC_MEMINFO_PATH, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def available_memory_bytes() -> int | None:
    """Return memory currently available to this process, or ``None``.

    This reports what the *machine* has free. A cgroup memory limit -- which is
    what a container or an HPC scheduler such as SLURM imposes -- is not
    detected, so a run under one must be given ``--workers`` explicitly rather
    than sizing itself from memory it is not permitted to use. Detecting it
    cost roughly half this module in cgroup v1 and v2 parsing, namespace
    remapping and controller-tree walking, exercised by nobody running Alchemy
    from a clone, and worker count is capped at the entry count anyway, so a
    single-structure run never depends on this at all.
    """
    if sys.platform.startswith("linux"):
        available = _read_linux_available_memory()
        if available is not None:
            return available

    elif os.name == "nt":
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
            # ``windll`` exists only on Windows, where this branch runs.
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                ctypes.byref(status)
            ):
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


def automatic_worker_limits() -> tuple[int, int | None]:
    """Return the CPU and optional memory limits for automatic parallelism."""
    cpu_limit = max(1, available_cpu_count() - 2)
    available_memory = available_memory_bytes()
    memory_limit: int | None = None
    if available_memory is not None:
        memory_limit = max(1, available_memory // AUTO_WORKER_MEMORY_BYTES)
    return cpu_limit, memory_limit
