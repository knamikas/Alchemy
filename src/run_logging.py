"""Configure logging for the driver and workers.

Workers send records to one driver listener to avoid interleaved writes.
Progress display and the structured run report use separate output paths.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import sys
import time
from typing import IO

from typing_extensions import override

# Bound external output and traceback text in each record.
MAX_RECORD_CHARS = 2000

# Longest excerpt of an external program's output to embed in a message.
# Smaller than MAX_RECORD_CHARS so the surrounding context still fits.
MAX_TOOL_OUTPUT_CHARS = 500

# Count the truncation marker within the message limit.
_OMITTED_MARKER = "... [{} more characters]"

LOGGER_NAME = "alchemy"

_QUEUE_HANDLER_NAME = "alchemy-worker-queue"


def logger_for(module_name: str) -> logging.Logger:
    """Return the ``alchemy.*`` logger for a module.

    Every module logs through a child of one root so that a single handler
    configuration governs all of them, including the ones running in workers.
    """
    leaf = module_name.rsplit(".", 1)[-1]
    # A module run as a script is named __main__; log it under its real name.
    if leaf == "__main__":
        leaf = "cli"
    return logging.getLogger(f"{LOGGER_NAME}.{leaf}")


def truncate(text: object, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate text to the limit, including a marker when space permits."""
    rendered = str(text).strip()
    if len(rendered) <= limit:
        return rendered

    # The marker length depends on the omitted count; two passes fit both.
    keep = limit
    for _ in range(2):
        marker = _OMITTED_MARKER.format(len(rendered) - keep)
        keep = max(0, limit - len(marker))
    marker = _OMITTED_MARKER.format(len(rendered) - keep)

    if keep == 0:
        return rendered[:limit]
    return (rendered[:keep] + marker)[:limit]


def elapsed_s(started: float) -> float:
    """Return the seconds since a ``time.monotonic()`` reading, as a timing.

    Every stage timing the pipeline records is rounded this way, so the
    manifest's ``timings`` column is byte-stable across stages.
    """
    return round(time.monotonic() - started, 3)


class _BoundedMessage(logging.Filter):
    """Limit message length on handlers so every logging call is covered."""

    def __init__(self, limit: int = MAX_RECORD_CHARS) -> None:
        super().__init__()
        self.limit = limit

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if len(message) > self.limit:
            record.msg = truncate(message, self.limit)
            record.args = ()
        # ``exc_info`` is formatted by the handler after filters run, so only
        # the record's message obeys the limit.
        return True


def level_for_verbosity(verbose: int = 0, quiet: bool = False) -> int:
    """Map verbosity flags to the console level; repeated -v has no further effect."""
    if quiet:
        return logging.WARNING
    return logging.DEBUG if verbose else logging.INFO


def worker_level(console_level: int, log_file: str | None = None) -> int:
    """Return DEBUG for file logging, otherwise the console level.

    Worker filtering happens before queueing, so it must retain file-log detail.
    """
    return logging.DEBUG if log_file else console_level


def configure_driver_logging(
    level: int = logging.INFO,
    stream: IO[str] | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """Attach the process-wide handlers. Called once, in the driver.

    Diagnostics go to stderr so that stdout carries only the progress line and
    the final summary, keeping a redirected stdout usable as a result record.
    """
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stderr if stream is None else stream)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    console.addFilter(_BoundedMessage())
    root.addHandler(console)

    if log_file:
        # Let the caller report failures before a handler exists.
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        # Keep debug detail in the file regardless of console verbosity.
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(processName)-14s %(name)s: %(message)s"
            )
        )
        file_handler.addFilter(_BoundedMessage())
        root.addHandler(file_handler)
        root.setLevel(min(level, logging.DEBUG))
    return root


def create_worker_log_queue() -> multiprocessing.Queue[logging.LogRecord]:
    """Create the worker logging queue before starting the pool.

    Start the listener after forking to avoid inheriting locks held by threads
    that do not exist in the child.
    """
    return multiprocessing.Queue(-1)


def start_worker_log_listener(
    queue: multiprocessing.Queue[logging.LogRecord],
) -> logging.handlers.QueueListener:
    """Return a started listener re-emitting queued records in this process.

    The caller must ``stop()`` it after the pool is gone, so records emitted
    during shutdown are still forwarded.
    """
    root = logging.getLogger(LOGGER_NAME)
    listener = logging.handlers.QueueListener(
        queue, *root.handlers, respect_handler_level=True
    )
    listener.start()
    return listener


def configure_worker_logging(
    queue: multiprocessing.Queue[logging.LogRecord] | None,
    level: int = logging.INFO,
) -> None:
    """Point this worker's ``alchemy`` logger at the driver's queue.

    Handlers inherited through ``fork`` are removed first: a forked copy of the
    driver's stream handler writes to the same file descriptor from several
    processes at once.
    """
    root = logging.getLogger(LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    if queue is None:
        return
    handler = logging.handlers.QueueHandler(queue)
    handler.set_name(_QUEUE_HANDLER_NAME)
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
