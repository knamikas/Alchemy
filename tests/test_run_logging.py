"""Diagnostics reach one place, bounded, from every process.

Scope: the driver owns the handlers, workers reach them over a queue rather
than writing concurrently, and no single record grows without limit.

Out of scope here (owned elsewhere): what individual modules choose to log, and
the per-run report written by ``driver.runlog._RunLog``, which is an artifact
rather than a transcript.
"""

from __future__ import annotations

import io
import logging
import logging.handlers  # importing logging alone does not bind it
import multiprocessing
import re
from pathlib import Path
from collections.abc import Iterator

import pytest

import run_logging


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Restore the ``alchemy`` logger: handler configuration is process-wide."""
    root = logging.getLogger(run_logging.LOGGER_NAME)
    saved = (list(root.handlers), root.level, root.propagate)
    try:
        yield
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved[0]:
            root.addHandler(handler)
        root.setLevel(saved[1])
        root.propagate = saved[2]


def test_module_loggers_share_one_configurable_root() -> None:
    """Every module logs through a child of ``alchemy``, so one configuration
    governs all of them, worker-only modules included."""
    assert run_logging.logger_for("bond_analysis").name == "alchemy.bond_analysis"
    assert run_logging.logger_for("alchemy.main").name == "alchemy.main"


def test_the_script_entry_point_is_not_named_dunder_main() -> None:
    """``main.py`` runs as ``__main__``, which would name every driver record
    ``alchemy.__main__``."""
    assert run_logging.logger_for("__main__").name == "alchemy.main"


def test_records_are_bounded_regardless_of_level() -> None:
    """A level says how important a record is, not how long it may be."""
    stream = io.StringIO()
    run_logging.configure_driver_logging(level=logging.DEBUG, stream=stream)
    logging.getLogger("alchemy.test").error("x" * (run_logging.MAX_RECORD_CHARS * 3))

    written = stream.getvalue()
    assert len(written) < run_logging.MAX_RECORD_CHARS * 2
    assert "more characters" in written, "a truncated record must say that it was cut"


def test_truncation_is_marked_rather_than_silent() -> None:
    """A silently shortened traceback reads as a complete one that ended early."""
    assert run_logging.truncate("short", limit=50) == "short"
    assert "more characters" in run_logging.truncate("y" * 200, limit=50)


@pytest.mark.parametrize("limit", [8, 20, 30, 50, 300, 500])
@pytest.mark.parametrize("length", [5, 49, 300, 5000])
def test_truncation_never_exceeds_the_limit_it_was_given(
    limit: int, length: int
) -> None:
    """The marker counts against the budget rather than being added to it."""
    result = run_logging.truncate("z" * length, limit=limit)
    assert len(result) <= limit, f"{len(result)} > {limit}: {result!r}"


def test_a_bounded_record_also_respects_the_limit() -> None:
    """The handler filter uses the same accounting as ``truncate``."""
    stream = io.StringIO()
    run_logging.configure_driver_logging(level=logging.DEBUG, stream=stream)
    logging.getLogger("alchemy.test").error("q" * 50_000)

    record = stream.getvalue().rstrip("\n")
    _level_and_name, message = record.split(": ", 1)
    assert len(message) <= run_logging.MAX_RECORD_CHARS
    assert "more characters" in message


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected"),
    [
        (0, False, logging.INFO),
        (1, False, logging.DEBUG),
        (2, False, logging.DEBUG),
        (0, True, logging.WARNING),
        (2, True, logging.WARNING),
    ],
)
def test_verbosity_maps_to_levels_and_quiet_always_wins(
    verbose: int, quiet: bool, expected: int
) -> None:
    assert run_logging.level_for_verbosity(verbose, quiet) == expected


def test_diagnostics_default_to_stderr_leaving_stdout_for_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Redirecting stdout must yield results, not a mixture of results and logs."""
    run_logging.configure_driver_logging(level=logging.INFO)
    logging.getLogger("alchemy.test").info("a diagnostic")

    captured = capsys.readouterr()
    assert "a diagnostic" in captured.err
    assert captured.out == ""


def test_a_log_file_keeps_debug_detail_the_console_discarded(tmp_path: Path) -> None:
    """The file records what the console level filtered out."""
    stream = io.StringIO()
    log_file = tmp_path / "run.log"
    run_logging.configure_driver_logging(
        level=logging.WARNING, stream=stream, log_file=str(log_file)
    )
    logger = logging.getLogger("alchemy.test")
    logger.debug("quiet detail")
    logger.warning("loud problem")

    console = stream.getvalue()
    assert "quiet detail" not in console
    assert "loud problem" in console

    recorded = log_file.read_text(encoding="utf-8")
    assert "quiet detail" in recorded
    assert "loud problem" in recorded


def _worker_emits(
    queue: multiprocessing.Queue[logging.LogRecord], level: int, message: str
) -> None:
    """Log from a genuinely separate process, as a pool worker would."""
    run_logging.configure_worker_logging(queue, level=level)
    logging.getLogger("alchemy.worker").warning(message)


def test_worker_records_reach_the_drivers_handlers(tmp_path: Path) -> None:
    """A record logged in another process is emitted by the driver's handlers.

    Workers cannot share those handlers: several processes writing one stream
    interleave mid-record, so the queue is the only way across the boundary.
    """
    log_file = tmp_path / "run.log"
    run_logging.configure_driver_logging(
        level=logging.INFO, stream=io.StringIO(), log_file=str(log_file)
    )
    queue = run_logging.create_worker_log_queue()

    # The queue and the worker must share a start context, and this is the one
    # ``multiprocessing.Pool`` uses for the real workers.
    context = multiprocessing.get_context()
    process = context.Process(
        target=_worker_emits, args=(queue, logging.INFO, "from the worker")
    )
    process.start()
    listener = run_logging.start_worker_log_listener(queue)
    try:
        process.join(timeout=60)
        assert process.exitcode == 0, "the worker failed to log"
    finally:
        listener.stop()
        queue.close()
        queue.join_thread()

    recorded = log_file.read_text(encoding="utf-8")
    assert "from the worker" in recorded
    assert re.search(r"alchemy\.worker", recorded)


def test_a_worker_never_writes_through_an_inherited_handler() -> None:
    """Forked handlers are dropped before the queue handler is attached.

    A forked copy of the driver's stream handler shares the parent's file
    descriptor, which is the interleaving the queue exists to prevent.
    """
    stream = io.StringIO()
    run_logging.configure_driver_logging(level=logging.INFO, stream=stream)
    root = logging.getLogger(run_logging.LOGGER_NAME)
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    queue = run_logging.create_worker_log_queue()
    try:
        run_logging.configure_worker_logging(queue, level=logging.INFO)
        handlers = logging.getLogger(run_logging.LOGGER_NAME).handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.handlers.QueueHandler)
    finally:
        queue.close()


def test_configuring_the_driver_twice_does_not_duplicate_records() -> None:
    """Re-configuration replaces handlers rather than stacking them."""
    stream = io.StringIO()
    run_logging.configure_driver_logging(level=logging.INFO, stream=stream)
    run_logging.configure_driver_logging(level=logging.INFO, stream=stream)

    logging.getLogger("alchemy.test").info("once")
    assert stream.getvalue().count("once") == 1


def test_a_log_file_raises_the_worker_level_to_debug() -> None:
    """Workers filter before the queue, so the console level would discard the
    DEBUG records the file handler is there to keep."""
    console = run_logging.level_for_verbosity(verbose=0, quiet=True)
    assert console == logging.WARNING
    assert run_logging.worker_level(console, log_file=None) == logging.WARNING
    assert run_logging.worker_level(console, log_file="run.log") == logging.DEBUG


def _worker_emits_debug(
    queue: multiprocessing.Queue[logging.LogRecord], level: int, message: str
) -> None:
    run_logging.configure_worker_logging(queue, level=level)
    logging.getLogger("alchemy.worker").debug(message)


def test_worker_debug_records_reach_a_log_file_under_a_quiet_console(
    tmp_path: Path,
) -> None:
    """The end-to-end case the option exists for: quiet console, full file."""
    log_file = tmp_path / "run.log"
    console = io.StringIO()
    run_logging.configure_driver_logging(
        level=logging.WARNING, stream=console, log_file=str(log_file)
    )
    queue = run_logging.create_worker_log_queue()

    context = multiprocessing.get_context()
    process = context.Process(
        target=_worker_emits_debug,
        args=(
            queue,
            run_logging.worker_level(logging.WARNING, log_file=str(log_file)),
            "worker detail",
        ),
    )
    process.start()
    listener = run_logging.start_worker_log_listener(queue)
    try:
        process.join(timeout=60)
        assert process.exitcode == 0
    finally:
        listener.stop()
        queue.close()
        queue.join_thread()

    assert "worker detail" in log_file.read_text(encoding="utf-8")
    assert "worker detail" not in console.getvalue(), (
        "a WARNING console must not display DEBUG records"
    )


def test_an_unusable_log_file_is_raised_for_the_caller_to_report(
    tmp_path: Path,
) -> None:
    """An ``OSError`` rather than an exit: this runs before any handler exists,
    so only the caller can still report the failure."""
    with pytest.raises(OSError):
        run_logging.configure_driver_logging(
            level=logging.INFO, stream=io.StringIO(), log_file=str(tmp_path)
        )


def test_main_reports_an_unusable_log_file_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--log-file`` on a directory is a fixable failure: name the path and
    exit 1, not a traceback and not argparse's usage status."""
    import cli

    exit_code = cli.main(["--id", "109m", "--log-file", str(tmp_path)])

    message = capsys.readouterr().err
    assert exit_code == 1
    assert str(tmp_path) in message, message
    assert "Traceback" not in message


def test_repeated_verbose_flags_are_accepted_without_a_further_tier() -> None:
    """One ``-v`` unlocks everything; further ones change nothing."""
    assert run_logging.level_for_verbosity(1) == run_logging.level_for_verbosity(5)
