"""An independent re-implementation of the confidence scoring policy.

Integration tests cross-check ``src/confidence_score`` against this oracle, so
nothing here may import a threshold, rank, or decision rule from it; the raw
thresholds are restated deliberately, and
``test_the_scoring_policy_under_test_is_the_shipped_one`` in
``tests/test_confidence_score.py`` proves the two copies still agree. Only the
column names, file names, and the ``finalize`` entry point come from
production, because those are the interface under test.
"""

from __future__ import annotations

import contextlib
import csv
import io
import itertools
import json
import math
import os
from collections.abc import Sequence

from helpers import approx

import confidence_score
import confidence_score.cli

_StrPath = str | os.PathLike[str]

# Independent threshold values; the shipped-policy test in
# ``tests/test_confidence_score.py`` checks them against production.
DENSITY_THRESHOLDS = (3.0, 6.0)
GEOMETRY_THRESHOLDS = (1.0, 2.0)

# Evidence grid used to build a small frozen test reference. It crosses all
# three levels in both components and includes missing-evidence rows.
COHORT_RSZD = (0.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0)
COHORT_RMS_ZBOND = (0.25, 0.75, 1.0, 1.5, 2.5)

# A second cohort: same policy, a different database snapshot.
ALTERNATE_RSZD = (1.0, 2.2, 3.3, 4.4, 5.5, 7.7)
ALTERNATE_RMS_ZBOND = (0.5, 1.2, 2.0, 3.0)


def raw_level(value: float, thresholds: tuple[float, float]) -> str:
    """The three-level verdict for one component value, or INCOMPLETE."""
    if not math.isfinite(value) or value < 0:
        return "INCOMPLETE"
    if value < thresholds[0]:
        return "PASS"
    if value < thresholds[1]:
        return "REVIEW"
    return "SUSPECT"


def frozen_cohort(
    reference_dir: _StrPath,
) -> dict[str, list[tuple[float, int]]]:
    """Read both raw component distributions from a published reference."""
    path = os.path.join(
        str(reference_dir), confidence_score.REFERENCE_DISTRIBUTION_FILE
    )
    result: dict[str, list[tuple[float, int]]] = {"density": [], "geometry": []}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["component"]].append((float(row["value"]), int(row["count"])))
    return result


def reference_metadata(reference_dir: _StrPath) -> dict[str, object]:
    """The published policy metadata of a frozen reference directory."""
    path = os.path.join(str(reference_dir), confidence_score.REFERENCE_METADATA_FILE)
    with open(path, encoding="utf-8") as handle:
        metadata: dict[str, object] = json.load(handle)
    return metadata


def reverse_average_rank_support(
    cohort: Sequence[tuple[float, int]], value: float, tolerance: float = 1e-9
) -> float:
    """Reverse average-rank ECDF, computed independently of production code."""
    value = round(value, 3)
    total = sum(count for _, count in cohort)
    below = sum(count for item, count in cohort if item < value - tolerance)
    equal = sum(count for item, count in cohort if abs(item - value) <= tolerance)
    return 100.0 * (total - below - 0.5 * equal) / total


def frozen_reference(
    directory: _StrPath,
    rszd_values: Sequence[float] = COHORT_RSZD,
    rms_zbond_values: Sequence[float] = COHORT_RMS_ZBOND,
) -> str:
    """Publish a frozen confidence reference and return its directory.

    Built the way a real one is, through the documented
    ``python -m confidence_score finalize`` entry point. Two rows carry no
    assessable component, so component sizes and total input size cannot be
    confused.
    """
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    rows: list[dict[str, str]] = []
    for index, (rszd, rms) in enumerate(
        itertools.product(rszd_values, rms_zbond_values)
    ):
        row: dict[str, str] = dict.fromkeys(
            confidence_score.CONFIDENCE_INPUT_COLUMNS, ""
        )
        row.update(
            pdbID=f"c{index:03d}",
            category="ion",
            selected_metal_site_status="selected",
            metal_element="ZN",
            rszd_abs=repr(rszd),
            geometry_rms_zbond=repr(rms),
            geometry_coverage="1",
            assigned_contact_count="4",
            reference_covered_contact_count="4",
            geometry_bond_count="4",
            confidence_inputs_status="complete",
        )
        rows.append(row)
    for index in range(2):
        row = dict.fromkeys(confidence_score.CONFIDENCE_INPUT_COLUMNS, "")
        row.update(
            pdbID=f"u{index:03d}",
            selected_metal_site_status="selected",
            confidence_inputs_status="unscorable",
            confidence_inputs_missing_reasons="rszd_unavailable",
        )
        rows.append(row)

    inputs = os.path.join(directory, "confidence_inputs.csv")
    with open(inputs, "w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reference_dir = os.path.join(directory, "confidence_reference")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = confidence_score.cli.main(
            [
                "finalize",
                "--input",
                inputs,
                "--output",
                os.path.join(directory, "confidence_scores.csv"),
                "--reference-dir",
                reference_dir,
            ]
        )
    assert code == 0, out.getvalue() + err.getvalue()
    return reference_dir


def assert_policy_was_applied(
    rows: Sequence[dict[str, str]], reference_dir: _StrPath
) -> int:
    """Every row obeys the raw decision matrix and frozen component ranks."""
    metadata = reference_metadata(reference_dir)
    cohorts = frozen_cohort(reference_dir)
    density_size = sum(count for _, count in cohorts["density"])
    geometry_size = sum(count for _, count in cohorts["geometry"])
    assert density_size == metadata["density_reference_size"]
    assert geometry_size == metadata["geometry_reference_size"]

    scored = 0
    for row in rows:
        where = (
            f"{row['pdbID']} {row['metal_resname']}"
            f"{row['metal_resnum']}/{row['metal_atom']}"
        )
        assert row["confidence_reference_id"] == metadata["reference_id"], where
        assert int(row["confidence_cohort_size"]) == metadata["input_row_count"], where
        rszd = float(row["rszd_abs"]) if row["rszd_abs"] else math.nan
        rms = (
            float(row["geometry_rms_zbond"]) if row["geometry_rms_zbond"] else math.nan
        )
        density_level = raw_level(rszd, DENSITY_THRESHOLDS)
        geometry_level = raw_level(rms, GEOMETRY_THRESHOLDS)
        available = [
            level for level in (density_level, geometry_level) if level != "INCOMPLETE"
        ]
        overall = (
            "INCOMPLETE"
            if not available
            else "SUSPECT"
            if "SUSPECT" in available or available.count("REVIEW") == 2
            else "REVIEW"
            if "REVIEW" in available
            else "PASS"
        )
        assert row["density_level"] == density_level, where
        assert row["geometry_level"] == geometry_level, where
        assert row["alchemy_level"] == overall, where

        component_scores: list[float] = []
        for component, value in (("density", rszd), ("geometry", rms)):
            score_text = row[f"{component}_score"]
            if not math.isfinite(value):
                assert score_text == "", where
                continue
            expected = (
                0.0
                if component == "density" and value >= 99.9
                else reverse_average_rank_support(cohorts[component], value)
            )
            assert float(score_text) == approx(expected, abs=1e-6), where
            component_scores.append(expected)
        if component_scores:
            assert float(row["alchemy_score"]) == approx(
                min(component_scores), abs=1e-6
            ), where
            scored += 1
        else:
            assert row["alchemy_score"] == "", where
    return scored
