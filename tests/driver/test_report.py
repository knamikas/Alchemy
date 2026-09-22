"""Test the end-of-batch report and score finalization gate."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from helpers import entry_result, tally_of

import cli
from codes import EntryStatus
from driver import (
    layout as driver_layout,
    report as driver_report,
    runlog,
    scoring as driver_scoring,
    writers,
)
from score.reference import FinalizedScore


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
            n_score=0,
        ),
    )


def test_database_report_finalizes_and_exits_zero_for_terminal_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalization gate and process status must use the same policy."""
    args = cli.parse_args(["--output-dir", str(tmp_path)])
    layout = driver_layout.OutputLayout(str(tmp_path))
    plan = driver_scoring.DatabasePlan(layout)
    tally = tally_of(
        entry_result(
            status=EntryStatus.ERROR, reason_codes=["deterministic_processing_error"]
        )
    )
    run_log = runlog.RunLog(args, "pytest")
    finalized: list[str] = []

    def finalize(
        inputs_path: str, scores_path: str, reference_dir: str, *, manifest_path: str
    ) -> FinalizedScore:
        del scores_path, reference_dir, manifest_path
        finalized.append(os.path.dirname(inputs_path))
        return FinalizedScore(rows=0, scored_rows=0, cohort_size=0)

    monkeypatch.setattr(driver_scoring, "finalize_database_score", finalize)

    exit_code = driver_report.report_batch(
        args, layout, plan, tally, _empty_writer_counts(), run_log
    )

    assert exit_code == 0
    assert finalized == [str(tmp_path)]
    assert run_log.summary.score_status == "finalized"
    assert run_log.summary.scores_path == layout.scores
    assert run_log.summary.score_reference_path == layout.reference_dir


def test_database_report_defers_and_exits_nonzero_for_unexpected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retryable error must neither publish a reference nor report success."""
    args = cli.parse_args(["--output-dir", str(tmp_path)])
    layout = driver_layout.OutputLayout(str(tmp_path))
    plan = driver_scoring.DatabasePlan(layout)
    tally = tally_of(
        entry_result(
            status=EntryStatus.ERROR, reason_codes=["unexpected_processing_error"]
        )
    )
    run_log = runlog.RunLog(args, "pytest")

    def must_not_finalize(*args: object, **kwargs: object) -> tuple[int, int, str]:
        raise AssertionError("a recoverable error must defer finalization")

    monkeypatch.setattr(driver_scoring, "finalize_database_score", must_not_finalize)

    exit_code = driver_report.report_batch(
        args, layout, plan, tally, _empty_writer_counts(), run_log
    )

    assert exit_code == 1
    assert run_log.summary.score_status == "not_finalized_incomplete_run"
    assert run_log.summary.score_recoverable_entries == 1
