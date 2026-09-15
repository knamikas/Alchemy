"""Scratch directories Alchemy owns under an output directory.

Workers and resume staging create them with a marker that records who made
them and whether they may be swept; startup cleanup removes only marked,
disposable directories left by an earlier run.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from datetime import UTC, datetime
from typing import cast

SCRATCH_MARKER_FILENAME = ".alchemy-scratch.json"
SCRATCH_MARKER_SCHEMA = 1

# The kinds of scratch a marker may declare; the sweep removes no other kind.
ENTRY_SCRATCH = "entry"
RESUME_SCRATCH = "resume"
DISPOSABLE_SCRATCH_KINDS = frozenset({ENTRY_SCRATCH, RESUME_SCRATCH})


def create_owned_scratch_directory(
    output_dir: str,
    *,
    prefix: str,
    kind: str,
    preserve: bool = False,
) -> str:
    """Create scratch carrying the marker required for automatic cleanup."""
    path = tempfile.mkdtemp(prefix=prefix, dir=output_dir)
    marker = os.path.join(path, SCRATCH_MARKER_FILENAME)
    metadata: dict[str, object] = {
        "schema": SCRATCH_MARKER_SCHEMA,
        "owner": "alchemy",
        "kind": kind,
        "preserve": preserve,
        "created_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
    }
    try:
        with open(marker, "x", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True)
            handle.write("\n")
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise
    return path


def _scratch_cleanup_allowed(path: str) -> bool:
    """Whether ``path`` has a valid disposable Alchemy ownership marker."""
    marker = os.path.join(path, SCRATCH_MARKER_FILENAME)
    try:
        marker_stat = os.lstat(marker)
        if not stat.S_ISREG(marker_stat.st_mode):
            return False
        with open(marker, encoding="utf-8") as handle:
            loaded: object = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(loaded, dict):
        return False
    metadata = cast("dict[str, object]", loaded)
    return (
        metadata.get("schema") == SCRATCH_MARKER_SCHEMA
        and metadata.get("owner") == "alchemy"
        and metadata.get("kind") in DISPOSABLE_SCRATCH_KINDS
        and metadata.get("preserve") is False
    )


def sweep_owned_scratch_directories(output_dir: str) -> int:
    """Remove only marked disposable scratch from a previously ended run."""
    removed = 0
    try:
        names = os.listdir(output_dir)
    except OSError:
        return 0
    root = os.path.realpath(output_dir)
    for name in names:
        if not name.startswith(".alchemy-"):
            continue
        path = os.path.join(output_dir, name)
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        # A Windows junction is not a link to ``islink`` but resolves elsewhere.
        if os.path.dirname(os.path.realpath(path)) != root:
            continue
        if not _scratch_cleanup_allowed(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            removed += 1
    return removed
