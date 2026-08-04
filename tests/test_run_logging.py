"""Diagnostics reach one place, bounded, from every process.

Scope: the contract that ``run_logging`` establishes -- that the driver owns
the handlers, that worker processes reach them over a queue rather than writing
concurrently, and that no single record can grow without limit.

Out of scope here (owned elsewhere): what individual modules choose to log, and
the structured per-run report written by ``driver.runlog._RunLog``, which is a separate
artifact rather than a transcript.
"""

from __future__ import annotations

import io
import logging
import logging.handlers  # `logging.handlers` needs its own import
import multiprocessing
import re

import pytest

import run_logging


@pytest.fixture(autouse=True)
def _restore_logging():
    """Leave the ``alchemy`` logger exactly as it was found.

    Handler configuration is process-wide state; a test that alters it and does
    not restore it changes the behaviour of every later test in the session.
    """
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


def test_module_loggers_share_one_configurable_root():
    """Every module logs through a child of ``alchemy``.

    One handler configuration must govern all of them, including modules that
    only ever execute inside a worker.
    """
    assert run_logging.logger_for("bond_analysis").name == "alchemy.bond_analysis"
    assert run_logging.logger_for("alchemy.main").name == "alchemy.main"


def test_the_script_entry_point_is_not_named_dunder_main():
    """``main.py`` is normally executed, where ``__name__`` is ``__main__``.

    Without this, every driver record would be attributed to
    ``alchemy.__main__``, which tells a reader nothing.
    """
    assert run_logging.logger_for("__main__").name == "alchemy.main"


def test_records_are_bounded_regardless_of_level():
    """A level says how important a record is, not how long it may be.

    Regression risk: the call-site ``[:300]`` slices were replaced by handler
    configuration, so the bound must be enforced here or it is enforced nowhere.
    """
    stream = io.StringIO()
    run_logging.configure_driver_logging(level=logging.DEBUG, stream=stream)
    logging.getLogger("alchemy.test").error("x" * (run_logging.MAX_RECORD_CHARS * 3))

    written = stream.getvalue()
    assert len(written) < run_logging.MAX_RECORD_CHARS * 2
    assert "more characters" in written, "a truncated record must say that it was cut"


def test_truncation_is_marked_rather_than_silent():
    """A silently shortened traceback reads as a complete one that ended early."""
    assert run_logging.truncate("short", limit=50) == "short"
    assert "more characters" in run_logging.truncate("y" * 200, limit=50)


@pytest.mark.parametrize("limit", [8, 20, 30, 50, 300, 500])
@pytest.mark.parametrize("length", [5, 49, 300, 5000])
def test_truncation_never_exceeds_the_limit_it_was_given(limit, length):
    """The marker counts against the budget rather than being added to it.

    Regression: the marker used to be appended after the cut, so
    ``truncate(text, 300)`` returned more than 300 characters -- overflowing
    any fixed-width field the bound was chosen to fit.
    """
    result = run_logging.truncate("z" * length, limit=limit)
    assert len(result) <= limit, f"{len(result)} > {limit}: {result!r}"


def test_a_bounded_record_also_respects_the_limit():
    """The handler filter uses the same accounting as ``truncate``."""
    stream = io.StringIO()
    run_logging.configure_driver_logging(level=logging.DEBUG, stream=stream)
    logging.getLogger("alchemy.test").error("q" * 50_000)

    # One record: the level/name prefix plus a message within the bound.
    written = stream.getvalue().rstrip("\n")
    message = written.split(": ", 1)[1]
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
def test_verbosity_maps_to_levels_and_quiet_always_wins(verbose, quiet, expected):
    assert run_logging.level_for_verbosity(verbose, quiet) == expected


def test_diagnostics_default_to_stderr_leaving_stdout_for_results(capsys):
    """Redirecting stdout must yield results, not a mixture of results and logs."""
    run_logging.configure_driver_logging(level=logging.INFO)
    logging.getLogger("alchemy.test").info("a diagnostic")

    captured = capsys.readouterr()
    assert "a diagnostic" in captured.err
    assert captured.out == ""


def test_a_log_file_keeps_debug_detail_the_console_discarded(tmp_path):
    """The file is read afterwards, when the console level has already filtered.

    Recording only what was displayed would make ``--log-file`` useless for the
    case it exists to serve.
    """
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


def _worker_emits(queue, level, message):
    """Log from a genuinely separate process, as a pool worker would."""
    run_logging.configure_worker_logging(queue, level=level)
    logging.getLogger("alchemy.worker").warning(message)


def test_worker_records_reach_the_drivers_handlers(tmp_path):
    """A record logged in another process is emitted by the driver.

    Workers cannot share the driver's handlers: each is a separate interpreter,
    and several writing one stream interleave mid-record. This proves the queue
    actually carries records across the boundary rather than dropping them.
    """
    log_file = tmp_path / "run.log"
    run_logging.configure_driver_logging(
        level=logging.INFO, stream=io.StringIO(), log_file=str(log_file)
    )
    queue = run_logging.create_worker_log_queue()

    # The default context, matching how ``multiprocessing.Pool`` creates the
    # real workers; the queue is created under the same one. The inherited
    # handlers a fork brings along are removed by configure_worker_logging,
    # which the neighbouring test pins, so the record can only arrive by queue.
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


def test_a_worker_never_writes_through_an_inherited_handler():
    """Forked handlers are dropped before the queue handler is attached.

    A forked copy of the driver's stream handler shares the parent's file
    descriptor, so several workers would write to one stream concurrently --
    precisely the interleaving the queue exists to prevent.
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


def test_configuring_the_driver_twice_does_not_duplicate_records():
    """Re-configuration replaces handlers rather than stacking them."""
    stream = io.StringIO()
    run_logging.configure_driver_logging(level=logging.INFO, stream=stream)
    run_logging.configure_driver_logging(level=logging.INFO, stream=stream)

    logging.getLogger("alchemy.test").info("once")
    assert stream.getvalue().count("once") == 1


def test_a_log_file_raises_the_worker_level_to_debug():
    """``--log-file`` promises full debug detail whatever the console shows.

    Regression: workers filter before the queue, so a worker configured at the
    console level discarded DEBUG records before the driver's file handler
    could ever see them. ``--log-file`` then quietly contained no worker detail
    unless ``-v`` happened to be given too.
    """
    console = run_logging.level_for_verbosity(verbose=0, quiet=True)
    assert console == logging.WARNING
    assert run_logging.worker_level(console, log_file=None) == logging.WARNING
    assert run_logging.worker_level(console, log_file="run.log") == logging.DEBUG


def _worker_emits_debug(queue, level, message):
    run_logging.configure_worker_logging(queue, level=level)
    logging.getLogger("alchemy.worker").debug(message)


def test_worker_debug_records_reach_a_log_file_under_a_quiet_console(tmp_path):
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


def test_an_unusable_log_file_is_raised_for_the_caller_to_report(tmp_path):
    """A bad path must not bypass the driver's error handling.

    The file is opened before ``main`` enters its protected block, so without
    this it surfaced as a bare traceback with no structured run report. It is
    an ``OSError`` rather than an exit: this runs before any handler exists,
    and only the caller knows how a failure can still be reported.
    """
    with pytest.raises(OSError):
        run_logging.configure_driver_logging(
            level=logging.INFO, stream=io.StringIO(), log_file=str(tmp_path)
        )


def test_main_reports_an_unusable_log_file_and_exits_one(tmp_path, capsys):
    """The other half: the caller turns that error into the run's one policy.

    ``--log-file`` pointing at a directory is a fixable mistake, so it reports
    the path and exits 1 like every other fixable failure -- not a traceback,
    and not a status that says "you called this wrong", which argparse owns.
    """
    import cli

    exit_code = cli.main(["--id", "109m", "--log-file", str(tmp_path)])

    message = capsys.readouterr().err
    assert exit_code == 1
    assert str(tmp_path) in message, message
    assert "Traceback" not in message


def test_repeated_verbose_flags_are_accepted_without_a_further_tier():
    """Documented behaviour: one -v unlocks everything; more changes nothing."""
    assert run_logging.level_for_verbosity(1) == run_logging.level_for_verbosity(5)
