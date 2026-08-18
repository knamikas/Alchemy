"""Exclusive output-directory ownership and conservative scratch cleanup."""

from __future__ import annotations

import contextlib
import errno
import importlib
import json
import os
import shutil
import socket
import stat
import tempfile
from datetime import UTC, datetime
from types import TracebackType
from typing import IO, Any, cast

_IS_WINDOWS = os.name == "nt"
_platform_lock: Any = importlib.import_module("msvcrt" if _IS_WINDOWS else "fcntl")


LOCK_FILENAME = ".alchemy.lock"
SCRATCH_MARKER_FILENAME = ".alchemy-scratch.json"
SCRATCH_MARKER_SCHEMA = 1


class OutputDirectoryBusyError(RuntimeError):
    """Another Alchemy driver currently owns this output directory."""


class OutputDirectoryLockError(RuntimeError):
    """The output-directory lease could not be created or recorded."""


_active_lock_handle: IO[str] | None = None
_active_lock_pid: int | None = None
WINDOWS_LOCK_OFFSET = 0x7FFFFFFF


def _open_lock_handle(path: str) -> IO[str]:
    """Open one safe lock inode without following the final path component."""
    flags = os.O_RDWR | os.O_CREAT
    if _IS_WINDOWS:
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOINHERIT", 0)
    else:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise OutputDirectoryLockError(
                "Cannot safely open the output lock: this platform does not "
                "support O_NOFOLLOW."
            )
        flags |= nofollow
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
        path_metadata = os.lstat(path)
        file_attributes = getattr(path_metadata, "st_file_attributes", 0)
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        current_uid = getattr(os, "geteuid", None)
        if reparse_attribute and file_attributes & reparse_attribute:
            problem = "it is a symbolic link or other reparse point"
        elif not os.path.samestat(metadata, path_metadata):
            problem = "the path changed while it was being opened"
        elif not stat.S_ISREG(metadata.st_mode):
            problem = "it is not a regular file"
        elif current_uid is not None and metadata.st_uid != current_uid():
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


def acquire_file_lock(handle: IO[str]) -> None:
    """Acquire the platform's non-blocking exclusive lease."""
    if not _IS_WINDOWS:
        _platform_lock.flock(
            handle.fileno(), _platform_lock.LOCK_EX | _platform_lock.LOCK_NB
        )
        return
    try:
        # Keep the locked byte outside the JSON metadata so a contending
        # process can still report the current owner's details. Windows permits
        # byte-range locks beyond end-of-file.
        handle.seek(WINDOWS_LOCK_OFFSET)
        _platform_lock.locking(handle.fileno(), _platform_lock.LK_NBLCK, 1)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            raise BlockingIOError(exc.errno, str(exc)) from None
        raise
    finally:
        handle.seek(0)


def release_file_lock(handle: IO[str]) -> None:
    """Release the platform lease held by ``handle``."""
    if not _IS_WINDOWS:
        _platform_lock.flock(handle.fileno(), _platform_lock.LOCK_UN)
        return
    try:
        handle.seek(WINDOWS_LOCK_OFFSET)
        _platform_lock.locking(handle.fileno(), _platform_lock.LK_UNLCK, 1)
    finally:
        handle.seek(0)


def _close_inherited_lock_after_fork() -> None:
    """Ensure a worker cannot keep its dead parent's output lease alive."""
    global _active_lock_handle, _active_lock_pid
    if (
        _active_lock_handle is None
        or _active_lock_pid is None
        or os.getpid() == _active_lock_pid
    ):
        return
    with contextlib.suppress(OSError):
        # Close the file object, rather than only its descriptor, so a later
        # garbage collection in the child cannot close a reused descriptor.
        _active_lock_handle.close()
    _active_lock_handle = None
    _active_lock_pid = None


_register_at_fork = getattr(os, "register_at_fork", None)
if _register_at_fork is not None:
    _register_at_fork(after_in_child=_close_inherited_lock_after_fork)


def _owner_description(handle: IO[str]) -> str:
    try:
        handle.seek(0)
        loaded: object = json.load(handle)
    except (OSError, ValueError, TypeError):
        return "owner details unavailable"
    if not isinstance(loaded, dict):
        return "owner details unavailable"
    metadata = cast("dict[str, object]", loaded)
    fields: list[str] = []
    for key in ("pid", "hostname", "started_utc", "command"):
        value = metadata.get(key)
        if value not in (None, ""):
            fields.append(f"{key}={value}")
    return ", ".join(fields) or "owner details unavailable"


class OutputDirectoryLock:
    """Hold the stable advisory lock for one output directory."""

    def __init__(self, output_dir: str, command: str = "") -> None:
        """Initialize a lock for the stable output-directory path."""
        self.output_dir = os.path.abspath(output_dir)
        self.path = os.path.join(self.output_dir, LOCK_FILENAME)
        self.command = str(command)
        self._handle: IO[str] | None = None

    def __enter__(self) -> OutputDirectoryLock:
        """Acquire the lock and record this process as its owner."""
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
            acquire_file_lock(handle)
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
            metadata: dict[str, object] = {
                "schema": 1,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_utc": datetime.now(UTC).isoformat(),
                "command": self.command,
            }
            handle.seek(0)
            handle.truncate()
            json.dump(metadata, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as exc:
            release_file_lock(handle)
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release and close the output-directory lock."""
        global _active_lock_handle, _active_lock_pid
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        _active_lock_handle = None
        _active_lock_pid = None
        try:
            release_file_lock(handle)
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
    metadata: dict[str, object] = {
        "schema": SCRATCH_MARKER_SCHEMA,
        "owner": "alchemy",
        "kind": str(kind),
        "preserve": bool(preserve),
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
