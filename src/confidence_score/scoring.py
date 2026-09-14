"""Classify sites and rank component metrics against a reference cohort."""

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Self

from codes import ConfidenceLevel, EvidenceBasis, VerdictReason
from confidence_score.schema import (
    DENSITY_REVIEW_THRESHOLD,
    DENSITY_SUSPECT_THRESHOLD,
    EDSTATS_SATURATION_MAGNITUDE,
    GEOMETRY_REVIEW_THRESHOLD,
    GEOMETRY_SUSPECT_THRESHOLD,
)


def component_level(value: float, review: float, suspect: float) -> ConfidenceLevel:
    """Classify one non-negative measurement at the final raw thresholds."""
    if not math.isfinite(value) or value < 0:
        return ConfidenceLevel.INCOMPLETE
    if value < review:
        return ConfidenceLevel.PASS
    if value < suspect:
        return ConfidenceLevel.REVIEW
    return ConfidenceLevel.SUSPECT


def density_level(rszd_abs: float) -> ConfidenceLevel:
    """Classify an absolute RSZD value at the density thresholds."""
    return component_level(
        rszd_abs, DENSITY_REVIEW_THRESHOLD, DENSITY_SUSPECT_THRESHOLD
    )


def geometry_level(geometry_rms_zbond: float) -> ConfidenceLevel:
    """Classify an RMS bond Z score at the geometry thresholds."""
    return component_level(
        geometry_rms_zbond, GEOMETRY_REVIEW_THRESHOLD, GEOMETRY_SUSPECT_THRESHOLD
    )


@dataclass(frozen=True, slots=True)
class SiteVerdict:
    """One site's authoritative levels and, when ranked, its support scores.

    The levels and the decision route always come from the raw thresholds.
    The three scores are reverse average-rank percentages against a frozen
    reference cohort; they are NaN when no reference was applied or the
    component is not assessable.
    """

    density_level: ConfidenceLevel
    geometry_level: ConfidenceLevel
    alchemy_level: ConfidenceLevel
    evidence_basis: EvidenceBasis
    verdict_reason: VerdictReason
    density_score: float = math.nan
    geometry_score: float = math.nan
    alchemy_score: float = math.nan

    def as_row(self) -> dict[str, str | float]:
        """Return the verdict keyed by the ``ANALYSIS_COLUMNS`` names it owns."""
        return {
            "density_level": self.density_level,
            "density_score": self.density_score,
            "geometry_level": self.geometry_level,
            "geometry_score": self.geometry_score,
            "alchemy_level": self.alchemy_level,
            "alchemy_score": self.alchemy_score,
            "evidence_basis": self.evidence_basis,
            "verdict_reason": self.verdict_reason,
        }


def classify_site(rszd_abs: float, geometry_rms_zbond: float) -> SiteVerdict:
    """Apply the non-compensatory final decision matrix to one site."""
    density = density_level(rszd_abs)
    geometry = geometry_level(geometry_rms_zbond)
    available = [
        level for level in (density, geometry) if level != ConfidenceLevel.INCOMPLETE
    ]
    evidence_basis = (
        EvidenceBasis.DENSITY_AND_GEOMETRY
        if len(available) == 2
        else EvidenceBasis.DENSITY_ONLY
        if density != ConfidenceLevel.INCOMPLETE
        else EvidenceBasis.GEOMETRY_ONLY
        if geometry != ConfidenceLevel.INCOMPLETE
        else EvidenceBasis.NO_ASSESSABLE_EVIDENCE
    )

    if not available:
        overall = ConfidenceLevel.INCOMPLETE
        reason = VerdictReason.NO_ASSESSABLE_EVIDENCE
    elif density == ConfidenceLevel.SUSPECT and geometry == ConfidenceLevel.SUSPECT:
        overall = ConfidenceLevel.SUSPECT
        reason = VerdictReason.DENSITY_AND_GEOMETRY_SUSPECT
    elif density == ConfidenceLevel.SUSPECT:
        overall = ConfidenceLevel.SUSPECT
        reason = VerdictReason.DENSITY_SUSPECT
    elif geometry == ConfidenceLevel.SUSPECT:
        overall = ConfidenceLevel.SUSPECT
        reason = VerdictReason.GEOMETRY_SUSPECT
    elif density == ConfidenceLevel.REVIEW and geometry == ConfidenceLevel.REVIEW:
        overall = ConfidenceLevel.SUSPECT
        reason = VerdictReason.REVIEW_PLUS_REVIEW
    elif density == ConfidenceLevel.REVIEW:
        overall = ConfidenceLevel.REVIEW
        reason = VerdictReason.DENSITY_REVIEW
    elif geometry == ConfidenceLevel.REVIEW:
        overall = ConfidenceLevel.REVIEW
        reason = VerdictReason.GEOMETRY_REVIEW
    else:
        overall = ConfidenceLevel.PASS
        reason = VerdictReason.ALL_AVAILABLE_COMPONENTS_PASS

    return SiteVerdict(
        density_level=density,
        geometry_level=geometry,
        alchemy_level=overall,
        evidence_basis=evidence_basis,
        verdict_reason=reason,
    )


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

    @classmethod
    def from_counts(
        cls,
        density_counts: Mapping[float, int],
        geometry_counts: Mapping[float, int],
        metadata: Mapping[str, Any],
    ) -> Self:
        """Build a reference from per-value count mappings, sorted by value."""
        return cls(
            sorted(density_counts),
            [density_counts[value] for value in sorted(density_counts)],
            sorted(geometry_counts),
            [geometry_counts[value] for value in sorted(geometry_counts)],
            metadata,
        )

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
) -> SiteVerdict:
    """Return authoritative levels plus secondary empirical ranking scores."""
    verdict = classify_site(rszd_abs, geometry_rms_zbond)
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
    return replace(
        verdict,
        density_score=density_score,
        geometry_score=geometry_score,
        alchemy_score=min(available_scores) if available_scores else math.nan,
    )
