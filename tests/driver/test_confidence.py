"""Test confidence planning across database, classification, and reference runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pytest

import cli
import confidence_score
from codes import RunMode
from driver import (
    confidence as driver_confidence,
    errors as driver_errors,
    layout as driver_layout,
    runlog,
)


def test_an_uncapped_database_run_says_it_ignores_an_explicit_reference() -> None:
    """Verify a database run warns that it ignores an existing-reference option."""
    args = cli.parse_args(["--confidence-reference-dir", "/tmp/reference"])
    # Attach directly because Alchemy disables propagation to caplog's handler.
    messages: list[str] = []

    class _Capture(logging.Handler):
        # typing.override requires Python 3.12; this project supports 3.11.
        def emit(  # type: ignore[explicit-override]
            self, record: logging.LogRecord
        ) -> None:
            messages.append(record.getMessage())

    alchemy_logger = logging.getLogger("alchemy.confidence")
    handler = _Capture()
    previous_level = alchemy_logger.level
    alchemy_logger.addHandler(handler)
    alchemy_logger.setLevel(logging.WARNING)
    try:
        # ``None`` for the run log: the uncapped-database branch returns before
        # it is touched, and the parameter is not declared Optional.
        plan = driver_confidence.plan_confidence(
            args,
            driver_layout.OutputLayout("/tmp/out"),
            RunMode.DATABASE,
            cast("runlog.RunLog", None),
        )
    finally:
        alchemy_logger.removeHandler(handler)
        alchemy_logger.setLevel(previous_level)

    assert isinstance(plan, driver_confidence.DatabasePlan)
    assert plan.mode == "database"
    assert plan.builds_reference
    assert any("is ignored on an uncapped" in message for message in messages)
    assert any("/tmp/reference" in message for message in messages)


def test_targeted_run_without_reference_still_plans_classifications(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(
        [
            "--id",
            "1abc",
            "--output-dir",
            str(tmp_path),
            "--confidence-reference-dir",
            str(tmp_path / "absent-reference"),
        ]
    )
    plan = driver_confidence.plan_confidence(
        args,
        driver_layout.OutputLayout(str(tmp_path)),
        RunMode.SINGLE,
        cast("runlog.RunLog", None),
    )
    assert isinstance(plan, driver_confidence.ClassificationPlan)
    assert plan.mode == "classification"
    assert not plan.builds_reference
    assert plan.stream_path == str(tmp_path / "confidence_scores_all.csv")
    assert plan.columns == (
        *confidence_score.CONFIDENCE_INPUT_COLUMNS,
        *confidence_score.ANALYSIS_COLUMNS,
    )


def test_fresh_targeted_run_automatically_uses_the_manuscript_reference(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(["--id", "9myr", "--output-dir", str(tmp_path)])
    run_log = runlog.RunLog(args, "pytest")
    plan = driver_confidence.plan_confidence(
        args, driver_layout.OutputLayout(str(tmp_path)), RunMode.SINGLE, run_log
    )

    assert isinstance(plan, driver_confidence.ReferencePlan)
    assert plan.mode == "reference"
    assert plan.reference.reference_id == "alchemy-confidence-eb792a9fda5ce16dd016"
    assert plan.reference.cohort_size == 330978
    assert (
        run_log.details["confidence_reference_dir"]
        == driver_confidence.DEFAULT_CONFIDENCE_REFERENCE_DIR
    )


@pytest.mark.parametrize(
    "arguments,run_mode",
    [
        (["--pdb-file", "a.pdb", "--mtz-file", "a.mtz"], RunMode.MANUAL),
        (["--id", "1abc"], RunMode.SINGLE),
        (["--id-file", "ids.txt"], RunMode.ID_FILE),
        ([], RunMode.DATABASE),
        (["--max-pdbs", "5"], RunMode.CAPPED_DATABASE),
    ],
)
def test_classify_run_names_how_the_entries_were_chosen(
    arguments: list[str], run_mode: RunMode
) -> None:
    """Only the uncapped database mode may build a reference."""
    assert driver_confidence.classify_run(cli.parse_args(arguments)) is run_mode


def test_a_disabled_plan_scores_nothing_and_clears_every_confidence_output(
    tmp_path: Path,
) -> None:
    layout = driver_layout.OutputLayout(str(tmp_path))
    plan = driver_confidence.plan_confidence(
        cli.parse_args(["--no-bonds"]),
        layout,
        RunMode.DATABASE,
        cast("runlog.RunLog", None),
    )

    assert type(plan) is driver_confidence.ConfidencePlan
    assert not plan.enabled
    assert plan.stream_path is None
    assert plan.columns is None
    assert plan.stale_outputs(layout) == (
        layout.confidence_inputs,
        layout.confidence_scores,
        driver_confidence.reference_marker(layout),
    )
    rows = [{"pdbID": "1abc"}]
    assert plan.score_rows(rows) is rows


def _malformed_csv_text(header: str) -> str:
    """A CSV whose single field exceeds the reader's limit, so parsing fails."""
    return f'{header}\n"{"x" * 200_000}"\n'


def test_a_malformed_resumed_scores_file_is_reported_as_a_driver_error(
    tmp_path: Path,
) -> None:
    """``csv.Error`` is not a ``ValueError``, so it needs catching in its own right."""
    layout = driver_layout.OutputLayout(str(tmp_path))
    reference = confidence_score.write_reference(
        str(tmp_path / "reference"), {1.0: 1}, {0.5: 1}, 1
    )
    Path(layout.confidence_scores).write_text(
        _malformed_csv_text("confidence_reference_id,confidence_cohort_id"),
        encoding="utf-8",
    )
    plan = driver_confidence.ReferencePlan(layout, reference, synchronize_inputs=False)
    with pytest.raises(driver_errors.DriverError, match="Cannot resume confidence"):
        plan.validate_resumed_output()


def test_a_malformed_reference_distribution_is_reported_as_a_driver_error(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    confidence_score.write_reference(str(reference_dir), {1.0: 1}, {0.5: 1}, 1)
    (reference_dir / confidence_score.REFERENCE_DISTRIBUTION_FILE).write_text(
        _malformed_csv_text("component,value,count"), encoding="utf-8"
    )
    args = cli.parse_args(
        [
            "--id",
            "1abc",
            "--output-dir",
            str(tmp_path),
            "--confidence-reference-dir",
            str(reference_dir),
        ]
    )
    with pytest.raises(driver_errors.DriverError, match="Invalid confidence reference"):
        driver_confidence.plan_confidence(
            args,
            driver_layout.OutputLayout(str(tmp_path)),
            RunMode.SINGLE,
            cast("runlog.RunLog", None),
        )
