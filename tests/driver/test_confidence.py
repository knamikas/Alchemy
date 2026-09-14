"""Test confidence planning across database, classification, and reference runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cli
import confidence_score
from driver import confidence as driver_confidence
from driver import layout as driver_layout
from driver import runlog


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
            True,
            cast("runlog.RunLog", None),
        )
    finally:
        alchemy_logger.removeHandler(handler)
        alchemy_logger.setLevel(previous_level)

    assert plan.mode == "database"
    assert plan.reference is None
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
        False,
        cast("runlog.RunLog", None),
    )
    assert plan.mode == "classification"
    assert plan.reference is None
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
        args, driver_layout.OutputLayout(str(tmp_path)), False, run_log
    )

    assert plan.mode == "reference"
    assert plan.reference is not None
    assert plan.reference.reference_id == "alchemy-confidence-8ba6808c816791ffbb87"
    assert plan.reference.cohort_size == 330978
    assert (
        run_log.details["confidence_reference_dir"]
        == driver_confidence.DEFAULT_CONFIDENCE_REFERENCE_DIR
    )
