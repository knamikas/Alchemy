"""Exclusive output-directory ownership and conservative scratch cleanup."""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import tempfile
from datetime import datetime, timezone
from typing import IO, Optional

import fcntl


LOCK_FILENAME = ".alchemy.lock"
SCRATCH_MARKER_FILENAME = ".alchemy-scratch.json"
SCRATCH_MARKER_SCHEMA = 1


class OutputDirectoryBusyError(RuntimeError):
    """Another Alchemy driver currently owns this output directory."""


class OutputDirectoryLockError(RuntimeError):
    """The output-directory lease could not be created or recorded."""


_active_lock_handle: Optional[IO[str]] = None
_active_lock_pid: Optional[int] = None


def _open_lock_handle(path: str) -> IO[str]:
    """Open one safe lock inode without following the final path component."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OutputDirectoryLockError(
            "Cannot safely open the output lock: this platform does not support "
            "O_NOFOLLOW."
        )
    flags = os.O_RDWR | os.O_CREAT | nofollow
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise OutputDirectoryLockError(
            f"Cannot safely open output lock {path}: {exc.strerror or exc}. "
            "Symbolic links and other unsafe lock paths are refused."
        ) from None

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            problem = "it is not a regular file"
        elif metadata.st_uid != os.geteuid():
            problem = "it is not owned by the current user"
        elif metadata.st_nlink != 1:
            problem = "it has multiple hard links"
        else:
            return os.fdopen(descriptor, "r+", encoding="utf-8")
        raise OutputDirectoryLockError(
            f"Cannot safely use output lock {path}: {problem}. Remove or rename "
            "the unsafe path before running Alchemy."
        )
    except BaseException:
        os.close(descriptor)
        raise


def _close_inherited_lock_after_fork() -> None:
    """Ensure a worker cannot keep its dead parent's output lease alive."""
    global _active_lock_handle, _active_lock_pid
    if (
        _active_lock_handle is None
        or _active_lock_pid is None
        or os.getpid() == _active_lock_pid
    ):
        return
    try:
        # Close the file object, rather than only its descriptor, so a later
        # garbage collection in the child cannot close a reused descriptor.
        _active_lock_handle.close()
    except OSError:
        pass
    _active_lock_handle = None
    _active_lock_pid = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_inherited_lock_after_fork)


def _owner_description(handle) -> str:
    try:
        handle.seek(0)
        metadata = json.load(handle)
    except (OSError, ValueError, TypeError):
        return "owner details unavailable"
    fields = []
    for key in ("pid", "hostname", "started_utc", "command"):
        value = metadata.get(key)
        if value not in (None, ""):
            fields.append(f"{key}={value}")
    return ", ".join(fields) or "owner details unavailable"


class OutputDirectoryLock:
    """Hold the stable advisory lock for one output directory."""

    def __init__(self, output_dir: str, command: str = "") -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.path = os.path.join(self.output_dir, LOCK_FILENAME)
        self.command = str(command)
        self._handle = None

    def __enter__(self):
        global _active_lock_handle, _active_lock_pid
        try:
            handle = _open_lock_handle(self.path)
        except OutputDirectoryLockError:
            raise
        except OSError as exc:
            raise OutputDirectoryLockError(
                f"Cannot lock output directory {self.output_dir}: {exc.strerror or exc}"
            ) from None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owner = _owner_description(handle)
            handle.close()
            raise OutputDirectoryBusyError(
                f"Output directory {self.output_dir} is already in use by "
                f"another Alchemy run ({owner}). Choose another --output-dir "
                "or wait for that run to finish."
            ) from None
        except OSError as exc:
            handle.close()
            raise OutputDirectoryLockError(
                f"Cannot lock output directory {self.output_dir}: {exc.strerror or exc}"
            ) from None
        try:
            metadata = {
                "schema": 1,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "command": self.command,
            }
            handle.seek(0)
            handle.truncate()
            json.dump(metadata, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as exc:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            if isinstance(exc, OSError):
                raise OutputDirectoryLockError(
                    f"Cannot record the lock for output directory "
                    f"{self.output_dir}: {exc.strerror or exc}"
                ) from None
            raise
        self._handle = handle
        _active_lock_handle = handle
        _active_lock_pid = os.getpid()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        global _active_lock_handle, _active_lock_pid
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        _active_lock_handle = None
        _active_lock_pid = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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
    metadata = {
        "schema": SCRATCH_MARKER_SCHEMA,
        "owner": "alchemy",
        "kind": str(kind),
        "preserve": bool(preserve),
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
            metadata = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    return (
        metadata.get("schema") == SCRATCH_MARKER_SCHEMA
        and metadata.get("owner") == "alchemy"
        and metadata.get("kind") in ("entry", "resume")
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
        if not name.startswith((".alchemy-", ".alchemy-resume-")):
            continue
        path = os.path.join(output_dir, name)
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        if os.path.dirname(os.path.realpath(path)) != root:
            continue
        if not _scratch_cleanup_allowed(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += not os.path.exists(path)
    return removed
