"""Logging setup for the driver and its worker processes.

Alchemy produces three kinds of output, and keeping them apart is the point of
this module:

* **The progress line** -- a single rewritten terminal line owned by
  ``_ProgressReporter``. It is user interface, not diagnostics, and stays on
  ``print``: routing it through logging would interleave it with records and
  break the carriage-return redraw.
* **The run report** -- ``_RunLog`` writes one immutable, structured document
  per invocation. It is a deliberate artifact with its own schema, not a
  transcript, so it is not a logging handler and is not fed from this module.
* **Log records** -- everything else. Emitted through a per-module
  ``logging.getLogger(__name__)``, from the driver and from worker processes
  alike.

Worker processes cannot share the driver's handlers: each is a separate
interpreter, and several writing to one stream interleave mid-record. Records
are therefore forwarded over a queue and re-emitted by a single listener thread
in the driver, which is the only place handlers are attached.
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing
import os
import sys
from typing import Optional


#: Longest message a single record may carry.
#:
#: External programs and tracebacks can produce output of unbounded size, and a
#: record built from one used to be truncated at the call site with a bare
#: ``[:300]``. Levels do not bound size -- a DEBUG record is no shorter than an
#: ERROR one -- so the limit is applied here, once, to every record.
MAX_RECORD_CHARS = 2000

#: Longest excerpt of an external program's output to embed in a message.
#: Deliberately smaller than ``MAX_RECORD_CHARS``: the rest of the record still
#: needs room for the context that says which program produced it.
MAX_TOOL_OUTPUT_CHARS = 500

#: Appended to a shortened value, and counted against the budget rather than
#: added to it.
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
    # "__main__"; naming the logger after the file keeps records readable
    # whether it was run or imported.
    if leaf == "__main__":
        leaf = "main"
    return logging.getLogger(f"{LOGGER_NAME}.{leaf}")


def truncate(text: object, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Shorten ``text`` to at most ``limit`` characters, marking the cut.

    The marker is counted against the limit rather than appended to it, so the
    result never exceeds what the caller asked for -- otherwise a bound chosen
    to fit a fixed-width field would silently overflow it.

    Marking matters: a silently truncated traceback looks like a complete one
    that simply ended early, which sends a reader looking for a fault that is
    not there.
    """
    rendered = str(text).strip()
    if len(rendered) <= limit:
        return rendered

    # The marker's own length depends on how much is omitted, which depends on
    # the marker's length. Two passes converge; the second only ever shrinks
    # what is kept, so the result cannot grow past the limit.
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
        # ``exc_info`` is formatted by the handler after filters run, so a
        # traceback attached by ``logger.exception`` is not bounded here. It is
        # shortened into the message instead, which keeps the guarantee that a
        # record's *message* obeys the limit without discarding a traceback a
        # caller deliberately attached.
        return True


def level_for_verbosity(verbose: int = 0, quiet: bool = False) -> int:
    """Map the CLI's verbosity flags onto a console level.

    ``--quiet`` reports problems only; the default adds the run narrative; a
    single ``-v`` enables every debug record there is, from the driver and from
    the workers alike. Further occurrences are accepted but change nothing --
    there is no third tier of detail to unlock.
    """
    if quiet:
        return logging.WARNING
    return logging.DEBUG if verbose else logging.INFO


def worker_level(console_level: int, log_file: Optional[str] = None) -> int:
    """The level a worker filters at.

    A worker discards records below its own level before they ever reach the
    queue, so the driver's file handler can only record what the worker agreed
    to send. With a log file active the worker must therefore run at DEBUG
    regardless of the console level -- otherwise ``--log-file`` would silently
    contain no worker detail unless ``-v`` were also given, which is precisely
    what the option promises to avoid.
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
        # handler exists, and the caller is the only code that knows how a
        # failure can still be reported. ``cli.main`` does that in one place.
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        # A file is read after the fact, when the cheap console level may have
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

    Separate from :func:`start_worker_log_listener` because the queue must
    exist before the pool is created while the listener's thread must not:
    ``fork`` copies only the calling thread, and forking a process that already
    has running threads risks a child deadlocking on a lock held by a thread
    that does not exist in it. CPython warns about this from 3.12. Create the
    queue, fork the pool, then start the listener.
    """
    return multiprocessing.Queue(-1)


def start_worker_log_listener(queue):
    """Return a started listener re-emitting queued records in this process.

    Records pass through the driver's own handlers, so a worker's output is
    formatted and filtered exactly like the driver's and only one process ever
    writes to a stream.

    The caller must ``stop()`` it, after the pool is gone, so records emitted
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

    Called from the pool initializer, once per worker. Handlers inherited
    through ``fork`` are removed first: a forked copy of the driver's stream
    handler would write to the same file descriptor from several processes at
    once, which is exactly the interleaving the queue exists to prevent.
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
