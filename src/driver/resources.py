"""What this machine will actually let a batch run use.

Keep these probes dependency-free: worker selection happens before the analysis
environment exists.
"""

import os
import sys
from multiprocessing import cpu_count


# Peak resident memory to budget for one worker, measured against the largest
# entries in the integration set, so it is a ceiling rather than an average.
AUTO_WORKER_MEMORY_BYTES = 1280 * 1024 * 1024

PROC_MEMINFO_PATH = "/proc/meminfo"
PROC_SELF_CGROUP_PATH = "/proc/self/cgroup"
CGROUP_ROOT = "/sys/fs/cgroup"

# Cgroup v1 represents an unlimited memory controller with a page-aligned
# value just below signed LONG_MAX. No real machine approaches an exbibyte, so
# values at this threshold are controller sentinels rather than useful limits.
CGROUP_V1_UNLIMITED_BYTES = 1 << 60


def available_cpu_count():
    """Return the number of CPUs this process is actually permitted to use.

    ``multiprocessing.cpu_count()`` ignores CPU affinity, so in a container or
    under a scheduler allocation it reports far more than the job was granted.
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


def _read_linux_available_memory():
    try:
        with open(PROC_MEMINFO_PATH, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _read_cgroup_integer(path, *, limit=False):
    try:
        with open(path, encoding="ascii") as handle:
            text = handle.read().strip()
        if limit and text == "max":
            return None
        value = int(text)
    except (OSError, ValueError):
        return None
    if value < 0 or (limit and value >= CGROUP_V1_UNLIMITED_BYTES):
        return None
    return value


def _cgroup_memberships():
    """Return ``(v2_path, v1_memory_path)`` from this process's cgroups."""
    v2_path = None
    v1_memory_path = None
    try:
        with open(PROC_SELF_CGROUP_PATH, encoding="ascii") as handle:
            for line in handle:
                try:
                    _hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
                except ValueError:
                    continue
                if not controllers:
                    v2_path = path
                elif "memory" in controllers.split(","):
                    v1_memory_path = path
    except OSError:
        pass
    return v2_path, v1_memory_path


def _cgroup_directory(root, membership):
    # Anchor normalization at / so even a malformed proc entry cannot escape
    # the controller mount through ``..`` components.
    relative = os.path.normpath("/" + membership.lstrip("/")).lstrip("/")
    return os.path.join(root, relative)


def _cgroup_headroom(directory, controller_root, limit_name, usage_name):
    """Return the tightest remaining allowance from a cgroup and its parents."""
    directory = os.path.abspath(directory)
    controller_root = os.path.abspath(controller_root)
    try:
        if os.path.commonpath((directory, controller_root)) != controller_root:
            return None
    except ValueError:
        return None

    available = None
    current = directory
    while True:
        limit = _read_cgroup_integer(os.path.join(current, limit_name), limit=True)
        if limit is not None:
            usage = _read_cgroup_integer(os.path.join(current, usage_name))
            headroom = limit if usage is None else max(0, limit - usage)
            available = headroom if available is None else min(available, headroom)
        if current == controller_root:
            break
        parent = os.path.dirname(current)
        outside_controller = (
            os.path.commonpath((parent, controller_root)) != controller_root
        )
        if parent == current or outside_controller:
            break
        current = parent
    return available


def _cgroup_available_memory():
    """Return remaining cgroup memory under v2 or v1, or ``None`` if unlimited."""
    v2_path, v1_memory_path = _cgroup_memberships()
    candidates = []

    if v2_path is not None:
        directory = _cgroup_directory(CGROUP_ROOT, v2_path)
        candidates.append(
            _cgroup_headroom(
                directory,
                CGROUP_ROOT,
                "memory.max",
                "memory.current",
            )
        )

    if v1_memory_path is not None:
        # The usual v1 mount is /sys/fs/cgroup/memory. Some systems mount a
        # single named hierarchy directly at /sys/fs/cgroup, so probe both.
        for controller_root in (
            os.path.join(CGROUP_ROOT, "memory"),
            CGROUP_ROOT,
        ):
            directory = _cgroup_directory(controller_root, v1_memory_path)
            candidates.append(
                _cgroup_headroom(
                    directory,
                    controller_root,
                    "memory.limit_in_bytes",
                    "memory.usage_in_bytes",
                )
            )

    # A cgroup namespace can expose the current group as the mount root even
    # when /proc/self/cgroup is unavailable or names a host-side path.
    candidates.extend(
        (
            _cgroup_headroom(
                CGROUP_ROOT,
                CGROUP_ROOT,
                "memory.max",
                "memory.current",
            ),
            _cgroup_headroom(
                os.path.join(CGROUP_ROOT, "memory"),
                os.path.join(CGROUP_ROOT, "memory"),
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            ),
        )
    )
    finite = [value for value in candidates if value is not None]
    return min(finite) if finite else None


def available_memory_bytes():
    """Return memory currently available to this process, or ``None``.

    On Linux this is the smaller of host-available memory and remaining cgroup
    allowance, so containers and scheduler jobs cannot size themselves from
    RAM they are forbidden to use.
    """
    if sys.platform.startswith("linux"):
        available = [
            value
            for value in (
                _read_linux_available_memory(),
                _cgroup_available_memory(),
            )
            if value is not None
        ]
        if available:
            return min(available)

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


def automatic_worker_limits():
    """Return the CPU and optional memory limits for automatic parallelism."""
    cpu_limit = max(1, available_cpu_count() - 2)
    available_memory = available_memory_bytes()
    memory_limit = None
    if available_memory is not None:
        memory_limit = max(1, available_memory // AUTO_WORKER_MEMORY_BYTES)
    return cpu_limit, memory_limit
