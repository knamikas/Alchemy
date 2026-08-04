"""Validated-but-unfixed defects, pinned so a fix cannot land unnoticed.

A pin asserts the behaviour Alchemy *should* have and is marked
``xfail(strict=True)``: while the defect stands the pin is a quiet ``XFAIL``,
and a fix turns the suite red, at which point the marker goes and the test moves
into the module that owns the behaviour. Keep each pin narrow enough that only a
fix to that defect can satisfy it, and free of CCP4 and the network so the
findings stay visible on a bare checkout. A finding reproducible only by killing
a process, filling a disk or exhausting memory is recorded as a ``skip`` with
its manual reproduction in the docstring rather than as a fragile automation.

This module currently owns no pins.
"""
