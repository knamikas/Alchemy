"""Return unused worker allocations between entries without retiring workers."""

from __future__ import annotations

import ctypes
import gc
from collections.abc import Callable
from functools import lru_cache


@lru_cache(maxsize=1)
def _allocator_trim() -> Callable[[], None] | None:
    """Find glibc's optional trim operation without assuming a libc filename."""
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    except (AttributeError, OSError):
        return None

    def release() -> None:
        trim(0)

    return release


def release_idle_memory() -> None:
    """Collect unreachable objects and release free allocator pages if supported.

    Call only after the entry's analysis frame has returned, so its large
    structures and arrays are no longer live. Unsupported allocators retain
    their normal behavior. Neither collection nor trimming frees live result
    objects or changes scientific calculations.
    """
    gc.collect()
    trim = _allocator_trim()
    if trim is not None:
        trim()
