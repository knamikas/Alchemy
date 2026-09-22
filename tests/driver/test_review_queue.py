"""Tests for the review queue the driver derives after scoring."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from crystallization_conditions import SUMMARY_COLUMNS
from driver.review_queue import REVIEW_CONTEXT_COLUMNS, write_review_queue
from score import ANALYSIS_COLUMNS, SCORE_INPUT_COLUMNS


def test_review_queue_filters_levels_and_joins_site_specific_context(
    tmp_path: Path,
) -> None:
    score_columns: list[str] = [*SCORE_INPUT_COLUMNS, *ANALYSIS_COLUMNS]
    scores_path = tmp_path / "score.csv"
    with open(scores_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_columns)
        writer.writeheader()
        for pdb_id, site_id, element, level in (
            ("1abc", "site-pass", "ZN", "PASS"),
            ("1abc", "site-review", "ZN", "REVIEW"),
            ("2def", "site-suspect", "FE", "SUSPECT"),
        ):
            writer.writerow(
                dict.fromkeys(score_columns, "")
                | {
                    "pdbID": pdb_id,
                    "metal_site_id": site_id,
                    "metal_element": element,
                    "alchemy_level": level,
                }
            )

    summary_path = tmp_path / "summary.csv"
    summary_columns: list[str] = list(SUMMARY_COLUMNS)
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerow(
            dict.fromkeys(summary_columns, "")
            | {
                "pdbID": "1abc",
                "crystallization_data_status": "available",
                "crystallization_detected_metals": "NI|ZN",
                "crystallization_any_metal": "true",
                "crystallization_promiscuous_transition_metal": "true",
                "crystallization_ni_co_like_metal": "true",
            }
        )
        writer.writerow(
            dict.fromkeys(summary_columns, "")
            | {
                "pdbID": "2def",
                "crystallization_data_status": "not_reported",
            }
        )

    output = tmp_path / "review.csv"
    assert (
        write_review_queue(
            str(scores_path), str(summary_path), str(output), score_columns
        )
        == 2
    )
    with open(output, newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [*score_columns, *REVIEW_CONTEXT_COLUMNS]
        rows = list(reader)

    review, suspect = rows
    assert review["metal_site_id"] == "site-review"
    assert review["crystallization_contains_modeled_metal"] == "true"
    assert (
        review["crystallization_contains_different_promiscuous_transition_metal"]
        == "true"
    )
    assert review["crystallization_context_flags"] == (
        "modeled_metal|different_promiscuous_transition_metal|ni_co_like_metal"
    )
    assert suspect["metal_site_id"] == "site-suspect"
    assert suspect["crystallization_contains_modeled_metal"] == ""
    assert suspect["crystallization_context_flags"] == ""


def test_review_queue_without_inputs_writes_only_the_header(tmp_path: Path) -> None:
    score_columns: list[str] = [*SCORE_INPUT_COLUMNS, *ANALYSIS_COLUMNS]
    output = tmp_path / "review.csv"

    written = write_review_queue(
        str(tmp_path / "missing_scores.csv"),
        str(tmp_path / "missing_summary.csv"),
        str(output),
        score_columns,
    )

    assert written == 0
    with open(output, newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [*score_columns, *REVIEW_CONTEXT_COLUMNS]
        assert list(reader) == []


def test_review_queue_rejects_a_different_score_schema(tmp_path: Path) -> None:
    score_columns: list[str] = [*SCORE_INPUT_COLUMNS, *ANALYSIS_COLUMNS]
    scores_path = tmp_path / "score.csv"
    with open(scores_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*score_columns, "extra"])
        writer.writeheader()

    with pytest.raises(
        ValueError, match="score schema is incompatible with review queue"
    ):
        write_review_queue(
            str(scores_path),
            str(tmp_path / "missing_summary.csv"),
            str(tmp_path / "review.csv"),
            score_columns,
        )


def test_unknown_context_leaves_the_metal_columns_blank(tmp_path: Path) -> None:
    """Without conditions, or without a modeled element, nothing is asserted."""
    score_columns: list[str] = [*SCORE_INPUT_COLUMNS, *ANALYSIS_COLUMNS]
    scores_path = tmp_path / "score.csv"
    with open(scores_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_columns)
        writer.writeheader()
        for pdb_id, element in (("1abc", ""), ("9zzz", "ZN")):
            writer.writerow(
                dict.fromkeys(score_columns, "")
                | {
                    "pdbID": pdb_id,
                    "metal_element": element,
                    "alchemy_level": "REVIEW",
                }
            )
    summary_path = tmp_path / "summary.csv"
    summary_columns: list[str] = list(SUMMARY_COLUMNS)
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerow(
            dict.fromkeys(summary_columns, "")
            | {
                "pdbID": "1abc",
                "crystallization_data_status": "available",
                "crystallization_detected_metals": "ZN|NI",
                "crystallization_sulfate": "true",
            }
        )

    output = tmp_path / "review.csv"
    write_review_queue(str(scores_path), str(summary_path), str(output), score_columns)
    with open(output, newline="") as handle:
        no_element, no_summary = list(csv.DictReader(handle))

    assert no_element["crystallization_contains_modeled_metal"] == ""
    assert (
        no_element["crystallization_contains_different_promiscuous_transition_metal"]
        == "true"
    )
    assert no_element["crystallization_context_flags"] == (
        "different_promiscuous_transition_metal|sulfate"
    )
    assert no_summary["crystallization_data_status"] == ""
    assert no_summary["crystallization_contains_modeled_metal"] == ""
    assert (
        no_summary["crystallization_contains_different_promiscuous_transition_metal"]
        == ""
    )
    assert no_summary["crystallization_context_flags"] == ""
