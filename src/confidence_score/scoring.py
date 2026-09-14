"""Classify sites and rank component metrics against a reference cohort."""

import bisect
import math
from collections.abc import Mapping, Sequence
from typing import Any

from confidence_score.schema import (
    DENSITY_REVIEW_THRESHOLD,
    DENSITY_SUSPECT_THRESHOLD,
    EDSTATS_SATURATION_MAGNITUDE,
    GEOMETRY_REVIEW_THRESHOLD,
    GEOMETRY_SUSPECT_THRESHOLD,
)


def component_level(value: float, review: float, suspect: float) -> str:
    """Classify one non-negative measurement at the final raw thresholds."""
    if not math.isfinite(value) or value < 0:
        return "INCOMPLETE"
    if value < review:
        return "PASS"
    if value < suspect:
        return "REVIEW"
    return "SUSPECT"


def density_level(rszd_abs: float) -> str:
    """Classify an absolute RSZD value at the density thresholds."""
    return component_level(
        rszd_abs, DENSITY_REVIEW_THRESHOLD, DENSITY_SUSPECT_THRESHOLD
    )


def geometry_level(geometry_rms_zbond: float) -> str:
    """Classify an RMS bond Z score at the geometry thresholds."""
    return component_level(
        geometry_rms_zbond, GEOMETRY_REVIEW_THRESHOLD, GEOMETRY_SUSPECT_THRESHOLD
    )


def classify_site(rszd_abs: float, geometry_rms_zbond: float) -> dict[str, str]:
    """Apply the non-compensatory final decision matrix to one site."""
    density = density_level(rszd_abs)
    geometry = geometry_level(geometry_rms_zbond)
    available = [level for level in (density, geometry) if level != "INCOMPLETE"]
    evidence_basis = (
        "density_and_geometry"
        if len(available) == 2
        else "density_only"
        if density != "INCOMPLETE"
        else "geometry_only"
        if geometry != "INCOMPLETE"
        else "no_assessable_evidence"
    )

    if not available:
        overall = "INCOMPLETE"
        reason = "no_assessable_evidence"
    elif density == "SUSPECT" and geometry == "SUSPECT":
        overall = "SUSPECT"
        reason = "density_and_geometry_suspect"
    elif density == "SUSPECT":
        overall = "SUSPECT"
        reason = "density_suspect"
    elif geometry == "SUSPECT":
        overall = "SUSPECT"
        reason = "geometry_suspect"
    elif density == "REVIEW" and geometry == "REVIEW":
        overall = "SUSPECT"
        reason = "review_plus_review"
    elif density == "REVIEW":
        overall = "REVIEW"
        reason = "density_review"
    elif geometry == "REVIEW":
        overall = "REVIEW"
        reason = "geometry_review"
    else:
        overall = "PASS"
        reason = "all_available_components_pass"

    return {
        "density_level": density,
        "geometry_level": geometry,
        "alchemy_level": overall,
        "evidence_basis": evidence_basis,
        "verdict_reason": reason,
    }


class _EmpiricalDistribution:
    """A compact average-rank survival distribution for one raw metric."""

    def __init__(self, values: Sequence[float], counts: Sequence[int]) -> None:
        if len(values) != len(counts):
            raise ValueError("confidence reference values and counts differ in size")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("confidence reference contains an invalid value")
        if any(isinstance(count, bool) or count < 1 for count in counts):
            raise ValueError("confidence reference contains an invalid count")
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("confidence reference values are not increasing")
        self.values = tuple(values)
        self.counts = tuple(counts)
        cumulative: list[int] = []
        running = 0
        for count in counts:
            cumulative.append(running)
            running += count
        self.cumulative_below = tuple(cumulative)
        self.size = running

    def support_score(self, value: float) -> float:
        """Return reverse average-rank ECDF support; ordinary values rank high."""
        if not math.isfinite(value) or value < 0 or not self.values:
            return math.nan
        index = bisect.bisect_left(self.values, value)
        if index < len(self.values) and self.values[index] == value:
            below = self.cumulative_below[index]
            equal = self.counts[index]
        else:
            below = (
                self.cumulative_below[index] if index < len(self.values) else self.size
            )
            equal = 0
        return 100.0 * (self.size - below - 0.5 * equal) / self.size


class ConfidenceReference:
    """Frozen empirical density and RMS-Zbond distributions."""

    def __init__(
        self,
        density_values: Sequence[float],
        density_counts: Sequence[int],
        geometry_values: Sequence[float],
        geometry_counts: Sequence[int],
        metadata: Mapping[str, Any],
    ) -> None:
        """Initialize the frozen empirical distributions and their metadata."""
        self.density = _EmpiricalDistribution(density_values, density_counts)
        self.geometry = _EmpiricalDistribution(geometry_values, geometry_counts)
        if self.density.size == 0 and self.geometry.size == 0:
            raise ValueError("confidence reference has no assessable evidence")
        self.metadata = dict(metadata)
        self.reference_id: str = self.metadata.get("reference_id", "")
        self.cohort_id: str = self.metadata.get("cohort_id", "")
        self.cohort_size = int(self.metadata.get("input_row_count", 0))

    @property
    def density_reference_size(self) -> int:
        """Return the number of density observations in the reference."""
        return self.density.size

    @property
    def geometry_reference_size(self) -> int:
        """Return the number of geometry observations in the reference."""
        return self.geometry.size


def score_site(
    rszd_abs: float,
    geometry_rms_zbond: float,
    reference: ConfidenceReference | None = None,
) -> dict[str, str | float]:
    """Return authoritative levels plus secondary empirical ranking scores."""
    result: dict[str, str | float] = dict(
        classify_site(rszd_abs, geometry_rms_zbond).items()
    )
    density_score = (
        0.0
        if reference and rszd_abs >= EDSTATS_SATURATION_MAGNITUDE
        else reference.density.support_score(rszd_abs)
        if reference
        else math.nan
    )
    geometry_score = (
        reference.geometry.support_score(geometry_rms_zbond) if reference else math.nan
    )
    available_scores = [
        score for score in (density_score, geometry_score) if math.isfinite(score)
    ]
    result.update(
        {
            "density_score": density_score,
            "geometry_score": geometry_score,
            "alchemy_score": min(available_scores) if available_scores else math.nan,
        }
    )
    return result
