"""Test the progress reporter's terminal and log-file throttling."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from driver.progress import ProgressReporter


@dataclass(frozen=True)
class _Totals:
    """What the heartbeat reads from the batch tally."""

    counts: Mapping[str, int] = field(
        default_factory=lambda: {"ok": 1, "partial": 0, "skip": 0, "error": 0}
    )
    no_metals: int = 0
    metal_site_limit_exceeded: int = 0


class TestProgressReporter:
    """The heartbeat is throttled differently for a terminal and a log file."""

    class _Stream(io.StringIO):
        def __init__(self, terminal: bool) -> None:
            super().__init__()
            self._terminal = terminal

        # typing.override requires Python 3.12; this project supports 3.11.
        def isatty(self) -> bool:  # type: ignore[explicit-override]
            return self._terminal

    def _reporter(
        self, terminal: bool, clock: Callable[[], float]
    ) -> tuple[ProgressReporter, TestProgressReporter._Stream]:
        stream = self._Stream(terminal)
        return ProgressReporter(total=10, stream=stream, clock=clock), stream

    def test_a_redirected_run_renders_far_less_often_than_a_terminal(self) -> None:
        """A run redirected to a file must not grow it by a line per second.

        The counts follow from stepping the clock by ``TERMINAL_INTERVAL_S``:
        a terminal renders every step, a log renders once.
        """
        now = [1000.0]
        for terminal, expected in ((True, 3), (False, 1)):
            reporter, stream = self._reporter(terminal, lambda: now[0])
            for _ in range(3):
                reporter.render(1, _Totals())
                now[0] += ProgressReporter.TERMINAL_INTERVAL_S
            assert stream.getvalue().count("elapsed=") == expected, (
                f"terminal={terminal} rendered the wrong number of lines"
            )

    def test_force_renders_regardless_of_the_interval(self) -> None:
        reporter, stream = self._reporter(False, lambda: 1000.0)
        reporter.render(1, _Totals())
        reporter.render(2, _Totals())
        assert stream.getvalue().count("elapsed=") == 1
        reporter.render(10, _Totals(), force=True, final=True)
        assert stream.getvalue().count("elapsed=") == 2
        assert "[10/10 100.0%]" in stream.getvalue()

    def test_reports_policy_exclusions_beside_metal_free_entries(self) -> None:
        reporter, stream = self._reporter(False, lambda: 1000.0)
        reporter.render(2, _Totals(no_metals=1, metal_site_limit_exceeded=1))

        assert "no_metals=1 metal_site_limit_exceeded=1" in stream.getvalue()
