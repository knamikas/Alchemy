"""Test the end-of-batch report and confidence finalization gate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from helpers import entry_result

import cli
import worker_contracts
from codes import EntryStatus
from driver import confidence as driver_confidence
from driver import dispatch, runlog, writers
from driver import layout as driver_layout
from driver import report as driver_report


def _tally_of(*results: worker_contracts.EntryResult) -> dispatch.BatchTally:
    tally = dispatch.BatchTally()
    for result in results:
        tally.record(result)
    return tally


def _empty_writer_counts() -> writers.OutputWriters:
    return cast(
        writers.OutputWriters,
        SimpleNamespace(
            n_sites=0,
            n_bonds=0,
            n_candidates=0,
            n_crystallization_conditions=0,
            n_crystallization_summaries=0,
            n_density_contexts=0,
            n_confidence=0,
        ),
    )


def test_database_report_finalizes_and_exits_zero_for_terminal_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalization gate and process status must use the same policy."""
    args = cli.parse_args(["--output-dir", str(tmp_path)])
    layout = driver_layout.OutputLayout(str(tmp_path))
    plan = driver_confidence.DatabasePlan(layout)
    tally = _tally_of(
        entry_result(
            status=EntryStatus.ERROR, reason_codes=["deterministic_processing_error"]
        )
    )
    run_log = runlog.RunLog(args, "pytest")
    finalized: list[str] = []

    def finalize(_layout: driver_layout.OutputLayout) -> tuple[int, int, str]:
        finalized.append(_layout.output_dir)
        return 0, 0, "test-cohort"

    monkeypatch.setattr(driver_confidence, "finalize_confidence_reference", finalize)

    exit_code = driver_report.report_batch(
        args, layout, plan, tally, _empty_writer_counts(), run_log
    )

    assert exit_code == 0
    assert finalized == [str(tmp_path)]
    assert run_log.summary.confidence_status == "finalized"
    assert run_log.summary.confidence_scores_path == layout.confidence_scores
    assert run_log.summary.confidence_reference_path == layout.reference_dir


def test_database_report_defers_and_exits_nonzero_for_unexpected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retryable error must neither publish a reference nor report success."""
    args = cli.parse_args(["--output-dir", str(tmp_path)])
    layout = driver_layout.OutputLayout(str(tmp_path))
    plan = driver_confidence.DatabasePlan(layout)
    tally = _tally_of(
        entry_result(
            status=EntryStatus.ERROR, reason_codes=["unexpected_processing_error"]
        )
    )
    run_log = runlog.RunLog(args, "pytest")

    def must_not_finalize(_layout: driver_layout.OutputLayout) -> tuple[int, int, str]:
        raise AssertionError("a recoverable error must defer finalization")

    monkeypatch.setattr(
        driver_confidence, "finalize_confidence_reference", must_not_finalize
    )

    exit_code = driver_report.report_batch(
        args, layout, plan, tally, _empty_writer_counts(), run_log
    )

    assert exit_code == 1
    assert run_log.summary.confidence_status == "not_finalized_incomplete_run"
    assert run_log.summary.confidence_recoverable_entries == 1
