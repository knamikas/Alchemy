"""Keep planning independent of gemmi so it runs before worker imports and
still handles inputs incomplete enough to become ordinary worker errors.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, TextIO, cast
from collections.abc import Callable, Mapping


GIB = 1024**3

# A 2 GiB floor prevents many ordinary CCP4 processes from consuming the OS
# reserve together before their map-dependent estimates become significant.
AUTO_WORKER_MEMORY_BYTES = 2 * GIB

# Match density_analysis.py's GRID SAMP=5 and add margin because CCP4 rounds
# each requested dimension to an FFT-compatible grid.
FFT_GRID_SAMPLING = 5.0
FFT_GRID_MARGIN = 1.05
MAP_VALUE_BYTES = 4
DENSITY_MAP_COUNT = 2
MAP_PEAK_COPIES = 2.0
WORKER_FIXED_OVERHEAD_BYTES = 512 * 1024**2
ESTIMATE_SAFETY_FACTOR = 1.10
MTZ_SIZE_FALLBACK_MULTIPLIER = 64
MTZ_GZIP_SIZE_FALLBACK_MULTIPLIER = 128

# Do not turn every byte reported as available into a CCP4 allocation.  The
# driver, desktop, filesystem cache and short-lived program overlap need room.
MEMORY_RESERVE_MIN_BYTES = 4 * GIB
DEFAULT_MEMORY_UTILIZATION = 0.80

PROC_MEMINFO_PATH = "/proc/meminfo"
PROC_SELF_CGROUP_PATH = "/proc/self/cgroup"
CGROUP_ROOT = "/sys/fs/cgroup"

_PROPERTIES_PATTERN = re.compile(r'"properties"\s*:\s*')
_PROPERTY_PREFIX_LIMIT = 4 * 1024**2


@dataclass(frozen=True)
class EntryMemoryEstimate:
    pdb_id: str
    bytes: int
    source: str
    combined_map_bytes: int | None = None

    @property
    def is_high_memory(self) -> bool:
        return self.bytes > AUTO_WORKER_MEMORY_BYTES


def available_cpu_count() -> int:
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


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as handle:
            value = handle.read().strip()
        if value == "max":
            return None
        parsed = int(value)
        return parsed if parsed >= 0 else None
    except (OSError, ValueError):
        return None


def _safe_cgroup_dir(relative_path: str) -> str | None:
    relative = os.path.normpath(relative_path.lstrip("/"))
    if relative == ".":
        relative = ""
    if relative == ".." or relative.startswith(f"..{os.sep}"):
        return None
    root = os.path.abspath(CGROUP_ROOT)
    candidate = os.path.abspath(os.path.join(root, relative))
    if os.path.commonpath((root, candidate)) != root:
        return None
    return candidate


def _cgroup_tree_available(
    directory: str,
    controller_root: str,
    limit_name: str,
    usage_name: str,
    *,
    unlimited_threshold: int | None = None,
) -> int | None:
    current = os.path.abspath(directory)
    root = os.path.abspath(controller_root)
    allowances: list[int] = []
    while True:
        limit = _read_int(os.path.join(current, limit_name))
        usage = _read_int(os.path.join(current, usage_name))
        if (
            limit is not None
            and usage is not None
            and (unlimited_threshold is None or limit < unlimited_threshold)
        ):
            allowances.append(max(0, limit - usage))
        if current == root:
            break
        parent = os.path.dirname(current)
        if parent == current or os.path.commonpath((root, parent)) != root:
            break
        current = parent
    return min(allowances) if allowances else None


def _read_cgroup_available_memory() -> int | None:
    try:
        with open(PROC_SELF_CGROUP_PATH, encoding="ascii") as handle:
            lines = tuple(handle)
    except OSError:
        return None

    for line in lines:
        fields = line.rstrip("\n").split(":", 2)
        if len(fields) != 3 or fields[0] != "0" or fields[1] != "":
            continue
        directory = _safe_cgroup_dir(fields[2])
        if directory is None:
            return None
        return _cgroup_tree_available(
            directory, CGROUP_ROOT, "memory.max", "memory.current"
        )

    for line in lines:
        fields = line.rstrip("\n").split(":", 2)
        if len(fields) != 3 or "memory" not in fields[1].split(","):
            continue
        directory = _safe_cgroup_dir(os.path.join("memory", fields[2].lstrip("/")))
        if directory is None:
            return None
        return _cgroup_tree_available(
            directory,
            os.path.join(CGROUP_ROOT, "memory"),
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            # v1 uses a huge page-aligned integer for "unlimited".
            unlimited_threshold=1 << 60,
        )
    return None


def available_memory_bytes() -> int | None:
    """Use the tighter host/cgroup allowance so container and scheduler jobs
    cannot size themselves against inaccessible host RAM.
    """
    if sys.platform.startswith("linux"):
        host_available = _read_linux_available_memory()
        cgroup_available = _read_cgroup_available_memory()
        known = [v for v in (host_available, cgroup_available) if v is not None]
        if known:
            return min(known)

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


def scheduling_memory_budget(
    available: int | None,
    *,
    memory_limit_bytes: int | None = None,
    utilization: float = DEFAULT_MEMORY_UTILIZATION,
) -> tuple[int | None, int | None]:
    """Return the entry budget and protected reserve for this process.

    An explicit limit is still bounded by a tighter detected host or cgroup
    allowance. The utilization is a ceiling rather than a promise: the 4 GiB
    reserve remains in force on smaller machines unless doing so would prevent
    even one ordinary worker from running.
    """
    if not 0 < utilization <= 1:
        raise ValueError("memory utilization must be greater than 0 and at most 1")
    known = [value for value in (available, memory_limit_bytes) if value is not None]
    if not known:
        return None, None
    capacity = min(known)
    reserve = max(MEMORY_RESERVE_MIN_BYTES, math.ceil(capacity * (1.0 - utilization)))
    # Keep at least one ordinary worker possible whenever the machine has that
    # much memory.  On a smaller machine an oversized entry is still admitted
    # alone, because refusing it forever would deadlock the batch.
    reserve = min(reserve, max(0, capacity - AUTO_WORKER_MEMORY_BYTES))
    return max(1, capacity - reserve), reserve


def automatic_worker_limits(
    *,
    memory_limit_bytes: int | None = None,
    utilization: float = DEFAULT_MEMORY_UTILIZATION,
) -> tuple[int, int | None]:
    cpu_limit = max(1, available_cpu_count() - 2)
    budget, _ = scheduling_memory_budget(
        available_memory_bytes(),
        memory_limit_bytes=memory_limit_bytes,
        utilization=utilization,
    )
    memory_limit: int | None = None
    if budget is not None:
        memory_limit = max(1, budget // AUTO_WORKER_MEMORY_BYTES)
    return cpu_limit, memory_limit


def _open_text(path: str) -> TextIO:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _read_properties_prefix(path: str) -> dict[str, object] | None:
    """Avoid loading large validation arrays merely to plan memory admission."""
    decoder = json.JSONDecoder()
    data = ""
    try:
        with _open_text(path) as handle:
            while len(data) < _PROPERTY_PREFIX_LIMIT:
                chunk = handle.read(min(65536, _PROPERTY_PREFIX_LIMIT - len(data)))
                if not chunk:
                    break
                data += chunk
                match = _PROPERTIES_PATTERN.search(data)
                if match is None:
                    continue
                try:
                    value, _ = decoder.raw_decode(data, match.end())
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return cast(dict[str, object], value)
                return None
    except (OSError, UnicodeError):
        return None
    return None


def _positive_float(properties: Mapping[str, object], name: str) -> float | None:
    try:
        value = float(cast(Any, properties[name]))
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def estimate_from_properties(
    pdb_id: str, properties: Mapping[str, object]
) -> EntryMemoryEstimate | None:
    axes = tuple(
        _positive_float(properties, name) for name in ("AAXIS", "BAXIS", "CAXIS")
    )
    resolution = _positive_float(properties, "RESOLUTION")
    if any(axis is None for axis in axes) or resolution is None:
        return None
    grid = tuple(
        max(
            1,
            math.ceil(
                cast(float, axis) * FFT_GRID_SAMPLING / resolution * FFT_GRID_MARGIN
            ),
        )
        for axis in axes
    )
    one_map = 1024 + math.prod(grid) * MAP_VALUE_BYTES
    combined_maps = DENSITY_MAP_COUNT * one_map
    peak = math.ceil(
        (combined_maps * MAP_PEAK_COPIES + WORKER_FIXED_OVERHEAD_BYTES)
        * ESTIMATE_SAFETY_FACTOR
    )
    return EntryMemoryEstimate(
        pdb_id=pdb_id,
        bytes=max(AUTO_WORKER_MEMORY_BYTES, peak),
        source="data_json",
        combined_map_bytes=combined_maps,
    )


def _first_existing(paths: tuple[str | None, ...]) -> str | None:
    return next((path for path in paths if path and os.path.isfile(path)), None)


def estimate_entry_memory(
    pdb_id: str,
    root: str,
    manual_inputs: Mapping[str, str | None] | None = None,
) -> EntryMemoryEstimate:
    entry_dir = os.path.join(root, pdb_id[1:3], pdb_id)
    if manual_inputs is not None:
        data_path = _first_existing((manual_inputs.get("data_json"),))
        mtz_path = _first_existing((manual_inputs.get("mtz_file"),))
    else:
        data_path = _first_existing(
            (
                os.path.join(entry_dir, "data.json"),
                os.path.join(entry_dir, "data.json.gz"),
            )
        )
        mtz_path = _first_existing(
            (
                os.path.join(entry_dir, f"{pdb_id}_final.mtz"),
                os.path.join(entry_dir, f"{pdb_id}_final.mtz.gz"),
            )
        )

    if data_path is not None:
        properties = _read_properties_prefix(data_path)
        if properties is not None:
            estimate = estimate_from_properties(pdb_id, properties)
            if estimate is not None:
                return estimate

    if mtz_path is not None:
        try:
            multiplier = (
                MTZ_GZIP_SIZE_FALLBACK_MULTIPLIER
                if mtz_path.endswith(".gz")
                else MTZ_SIZE_FALLBACK_MULTIPLIER
            )
            estimated_bytes = os.path.getsize(mtz_path) * multiplier
        except OSError:
            estimated_bytes = 0
        if estimated_bytes:
            return EntryMemoryEstimate(
                pdb_id=pdb_id,
                bytes=max(AUTO_WORKER_MEMORY_BYTES, estimated_bytes),
                source="mtz_size",
            )

    return EntryMemoryEstimate(
        pdb_id=pdb_id,
        bytes=AUTO_WORKER_MEMORY_BYTES,
        source="default",
    )
