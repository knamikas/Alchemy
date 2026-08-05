"""The batch run's one line of terminal output.

Rendered by the parent from results it has already collected, so a worker never
pays for it. The throttle differs by consumer: a terminal wants a line that
moves, a redirected log wants a file that does not grow by a megabyte an hour.
"""

import sys
import time
from typing import Callable, Mapping, Optional, TextIO


class _ProgressReporter:
    """Render a throttled one-line progress heartbeat."""

    TERMINAL_INTERVAL_S = 1.0
    REDIRECTED_INTERVAL_S = 30.0

    def __init__(
        self,
        total: int,
        stream: Optional[TextIO] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.total = total
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.monotonic
        self.started = self.clock()
        self.last_rendered = float("-inf")
        self.last_width = 0
        self.terminal = bool(self.stream.isatty())
        self.line_open = False

    @staticmethod
    def _elapsed_text(elapsed_s: float) -> str:
        elapsed = max(0, int(elapsed_s))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def render(
        self,
        completed: int,
        counts: Mapping[str, int],
        no_metal_count: int,
        force: bool = False,
        final: bool = False,
    ) -> None:
        now = self.clock()
        interval = (
            self.TERMINAL_INTERVAL_S if self.terminal else self.REDIRECTED_INTERVAL_S
        )
        if not force and now - self.last_rendered < interval:
            return
        percent = 100.0 * completed / self.total if self.total else 100.0
        line = (
            f"[{completed}/{self.total} {percent:5.1f}%] "
            f"elapsed={self._elapsed_text(now - self.started)} | "
            f"ok={counts['ok']} partial={counts['partial']} "
            f"skip={counts['skip']} error={counts['error']} | "
            f"no_metals={no_metal_count}"
        )
        if self.terminal:
            padded = line.ljust(self.last_width)
            print(
                f"\r{padded}", end="\n" if final else "", file=self.stream, flush=True
            )
            self.last_width = len(line)
            self.line_open = not final
        else:
            print(line, file=self.stream, flush=True)
        self.last_rendered = now

    def close(self) -> None:
        """Finish an in-place terminal line after success or an exception."""
        if self.terminal and self.line_open:
            print(file=self.stream, flush=True)
            self.line_open = False
