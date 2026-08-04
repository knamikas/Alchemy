"""Logging setup for the driver and its worker processes.

``_ProgressReporter``'s terminal line and ``_RunLog``'s per-invocation document
are not routed through logging: the first would interleave with records and
lose its carriage-return redraw, and the second has its own schema.

Worker processes cannot share the driver's handlers, because several processes
writing one stream interleave mid-record. Records are forwarded over a queue
and re-emitted by a single listener thread in the driver, which is the only
place handlers are attached.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import sys
from typing import Optional


# Longest message a single record may carry. External programs and tracebacks
# produce output of unbounded size, and a record's level does not bound it.
MAX_RECORD_CHARS = 2000

# Longest excerpt of an external program's output to embed in a message.
# Smaller than MAX_RECORD_CHARS so the surrounding context still fits.
MAX_TOOL_OUTPUT_CHARS = 500

# Appended to a shortened value, counted against the budget rather than added
# to it.
_OMITTED_MARKER = "... [{} more characters]"

LOGGER_NAME = "alchemy"

_QUEUE_HANDLER_NAME = "alchemy-worker-queue"


def logger_for(module_name: str) -> logging.Logger:
    """Return the ``alchemy.*`` logger for a module.

    Every module logs through a child of one root so that a single handler
    configuration governs all of them, including the ones running in workers.
    """
    leaf = module_name.rsplit(".", 1)[-1]
    # main.py is normally executed as a script, where ``__name__`` is
    # "__main__", so records would otherwise be named differently depending on
    # whether it was run or imported.
    if leaf == "__main__":
        leaf = "main"
    return logging.getLogger(f"{LOGGER_NAME}.{leaf}")


def truncate(text: object, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Shorten ``text`` to at most ``limit`` characters, marking the cut.

    The marker is counted against the limit rather than appended to it, so a
    bound chosen to fit a fixed-width field cannot overflow it. An unmarked cut
    would leave a truncated traceback looking like one that ended early.
    """
    rendered = str(text).strip()
    if len(rendered) <= limit:
        return rendered

    # The marker's length depends on how much is omitted, which depends on the
    # marker's length. Two passes converge, and the second only shrinks what is
    # kept.
    keep = limit
    for _ in range(2):
        marker = _OMITTED_MARKER.format(len(rendered) - keep)
        keep = max(0, limit - len(marker))
    marker = _OMITTED_MARKER.format(len(rendered) - keep)

    if keep == 0:
        # The budget cannot hold even the marker; a hard cut is all that fits.
        return rendered[:limit]
    return (rendered[:keep] + marker)[:limit]


class _BoundedMessage(logging.Filter):
    """Cap the rendered message of every record.

    Applied to the handlers rather than to individual call sites, so a new
    logging call cannot forget it.
    """

    def __init__(self, limit: int = MAX_RECORD_CHARS):
        super().__init__()
        self.limit = limit

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if len(message) > self.limit:
            record.msg = truncate(message, self.limit)
            record.args = ()
        # ``exc_info`` is formatted by the handler after filters run, so only
        # the record's message obeys the limit.
        return True


def level_for_verbosity(verbose: int = 0, quiet: bool = False) -> int:
    """Map the CLI's verbosity flags onto a console level.

    A single ``-v`` already enables every debug record there is, so further
    occurrences are accepted and change nothing.
    """
    if quiet:
        return logging.WARNING
    return logging.DEBUG if verbose else logging.INFO


def worker_level(console_level: int, log_file: Optional[str] = None) -> int:
    """The level a worker filters at.

    A worker discards records below its own level before they reach the queue,
    so with a log file active it must run at DEBUG regardless of the console
    level; otherwise ``--log-file`` holds no worker detail without ``-v``.
    """
    return logging.DEBUG if log_file else console_level


def configure_driver_logging(
    level: int = logging.INFO,
    stream=None,
    log_file: Optional[str] = None,
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
        # ``OSError`` rather than a message and an exit: this runs before any
        # handler exists, so only the caller can report a failure.
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        # A file is read after the fact, when the console level may have
        # discarded the detail that explains what happened.
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


def create_worker_log_queue():
    """Return the queue workers will forward their records over.

    Separate from :func:`start_worker_log_listener` because ``fork`` copies
    only the calling thread, so a child can deadlock on a lock held by a thread
    that does not exist in it. Create the queue, fork the pool, then start the
    listener.
    """
    return multiprocessing.Queue(-1)


def start_worker_log_listener(queue):
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


def configure_worker_logging(queue, level: int = logging.INFO) -> None:
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
