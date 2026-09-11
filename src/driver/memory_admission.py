"""Recoverable memory admission with hysteresis around the protected reserve."""

from __future__ import annotations


class MemoryAdmission:
    """Back off once per pressure episode and recover after sustained headroom.

    A brief crossing of the reserve must not repeatedly multiply the budget
    down. Recovery requires 30 continuous seconds above the reserve plus a
    margin, advances in small steps, and is bounded by both the original plan
    and measured headroom. An explicitly oversized singleton may pause other
    work without teaching the ordinary-entry scheduler a smaller budget.
    """

    HEALTHY_SECONDS = 30.0

    def __init__(self, maximum: int | None, floor: int, reserve: int | None) -> None:
        """Start with the plan's budget and retain its original upper bound."""
        self.maximum = maximum
        self.budget = maximum
        self.floor = floor
        self.reserve = reserve
        self.margin = max(512 * 1024**2, (reserve or 0) // 4)
        self.pressure_active = False
        self.healthy_since: float | None = None
        self.last_change = float("-inf")
        self.backoffs = 0
        self.recoveries = 0
        self.pauses = 0

    def back_off(self, now: float) -> None:
        """Reduce future admission and restart the healthy observation window."""
        if self.budget is not None and self.budget > self.floor:
            self.budget = max(self.floor, int(self.budget * 0.80))
            self.backoffs += 1
        self.last_change = now
        self.healthy_since = None

    def observe(
        self,
        available: int | None,
        reserved: int,
        now: float,
        *,
        active: bool,
        oversized: bool = False,
    ) -> bool:
        """Update the budget and report whether new concurrent work must pause."""
        if available is None or self.reserve is None:
            self.healthy_since = None
            return False
        if available <= self.reserve:
            self.healthy_since = None
            if active and not self.pressure_active:
                self.pauses += 1
                self.pressure_active = True
                if not oversized:
                    self.back_off(now)
            return active
        if available < self.reserve + self.margin:
            self.healthy_since = None
            return False
        if self.healthy_since is None:
            self.healthy_since = now
        if now - max(self.healthy_since, self.last_change) < self.HEALTHY_SECONDS:
            return False
        self.pressure_active = False
        if self.budget is not None and self.maximum is not None:
            step = max(self.floor, self.maximum // 20)
            safe_budget = reserved + available - self.reserve - self.margin
            recovered = min(self.maximum, self.budget + step, safe_budget)
            if recovered > self.budget:
                self.budget = recovered
                self.recoveries += 1
                self.last_change = now
        return False
