"""Test the batch tally and the exit code it derives."""

from __future__ import annotations

import pytest
from helpers import entry_result, tally_of

from codes import EntryStatus
from driver import dispatch


@pytest.mark.parametrize(
    ("incomplete_entries", "expected"),
    [
        (0, 0),
        (1, 1),
        (24, 1),
    ],
)
def test_the_exit_code_reports_operational_incompleteness(
    incomplete_entries: int, expected: int
) -> None:
    """Nonzero exactly when the caller has recoverable work outstanding."""
    assert dispatch.batch_exit_code(incomplete_entries) == expected


def test_a_permanently_failed_entry_does_not_defer_the_reference() -> None:
    """A full-database run must still publish its reference.

    An entry that defeats CCP4 or carries unusable data fails identically on
    every pass, so deferring the reference until it succeeds deferred it
    forever -- the whole point of the uncapped run was to build one.
    """
    tally = tally_of(
        entry_result(status=EntryStatus.OK),
        entry_result(
            status=EntryStatus.ERROR, reason_codes=["deterministic_processing_error"]
        ),
    )

    assert tally.recoverable_incompleteness() == 0
    assert tally.terminal_errors == 1
    assert tally.recoverable_errors == 0
    assert tally.exit_code(database_run=True) == 0
    assert tally.exit_code() == 1, "a targeted entry error remains a failure"


def test_an_unexpected_error_defers_the_reference_and_fails_the_run() -> None:
    """A new failure mode must not silently become a cohort exclusion."""
    tally = tally_of(
        entry_result(
            status=EntryStatus.ERROR, reason_codes=["unexpected_processing_error"]
        )
    )

    assert tally.recoverable_incompleteness() == 1
    assert tally.terminal_errors == 0
    assert tally.recoverable_errors == 1
    assert tally.exit_code(database_run=True) == 1


def test_a_lost_worker_defers_the_reference() -> None:
    """A killed worker says nothing about the entry it held, so a retry can add it."""
    tally = tally_of(
        entry_result(status=EntryStatus.OK),
        entry_result(
            status=EntryStatus.ERROR,
            reason_codes=["worker_process_died", "unexpected_processing_error"],
        ),
    )

    assert tally.recoverable_incompleteness() == 1
    assert tally.exit_code(database_run=True) == 1


def test_a_missing_input_defers_the_reference() -> None:
    """A skipped entry's input may arrive before the next run."""
    tally = tally_of(entry_result(status=EntryStatus.SKIP))
    assert tally.recoverable_incompleteness() == 1
    assert tally.exit_code(database_run=True) == 1


def test_a_retryable_partial_defers_the_reference_but_a_terminal_one_does_not() -> None:
    retryable = tally_of(entry_result(status=EntryStatus.PARTIAL, retryable=True))
    terminal = tally_of(entry_result(status=EntryStatus.PARTIAL, retryable=False))

    assert retryable.recoverable_incompleteness() == 1
    assert terminal.recoverable_incompleteness() == 0
    assert retryable.exit_code(database_run=True) == 1
    assert terminal.exit_code(database_run=True) == 0
