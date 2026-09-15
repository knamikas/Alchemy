"""Adjust memory admission with delayed recovery after pressure subsides."""

from __future__ import annotations

#: Headroom above the reserve that must hold before admission recovers.
PRESSURE_MARGIN_FLOOR_BYTES = 512 * 1024**2
#: Share of the budget kept after each pressure episode.
BACKOFF_FACTOR = 0.80
#: Each recovery step raises the budget by this fraction of its original
#: maximum, but never by less than the floor, which dominates below 10 GiB.
RECOVERY_STEP_DIVISOR = 20


class MemoryAdmission:
    """Reduce admission during memory pressure and recover after sustained headroom.

    Back off once per pressure episode. An episode ends only after available
    memory has stayed above the reserve plus a margin for the healthy window,
    so repeated dips within that window cost a single back-off. Once the
    window closes, recover in steps bounded by the original budget and by the
    headroom actually observed. An oversized single entry pauses admission
    without reducing the ordinary budget.
    """

    HEALTHY_SECONDS = 30.0

    def __init__(self, maximum: int | None, floor: int, reserve: int | None) -> None:
        """Start with the plan's budget and retain its original upper bound.

        ``maximum`` is the plan's byte budget for active entries and the most a
        recovery can restore; ``None`` means memory could not be measured and
        the controller never pauses. ``floor`` is the lowest budget a back-off
        may reach and the smallest recovery step; the dispatcher passes one
        worker's resident overhead. ``reserve`` is the protected headroom the
        plan set aside; available memory at or below it is pressure.
        """
        self.maximum = maximum
        self.budget = maximum
        self.floor = floor
        self.reserve = reserve
        self.margin = max(PRESSURE_MARGIN_FLOOR_BYTES, (reserve or 0) // 4)
        self.pressure_active = False
        self.healthy_since: float | None = None
        self.last_change = float("-inf")
        self.backoffs = 0
        self.recoveries = 0
        self.pauses = 0

    def back_off(self, now: float) -> None:
        """Reduce future admission and restart the healthy observation window."""
        if self.budget is not None and self.budget > self.floor:
            self.budget = max(self.floor, int(self.budget * BACKOFF_FACTOR))
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
            step = max(self.floor, self.maximum // RECOVERY_STEP_DIVISOR)
            safe_budget = reserved + available - self.reserve - self.margin
            recovered = min(self.maximum, self.budget + step, safe_budget)
            if recovered > self.budget:
                self.budget = recovered
                self.recoveries += 1
                self.last_change = now
        return False
