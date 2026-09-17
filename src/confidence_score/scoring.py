"""Classify sites and rank component metrics against a reference cohort."""

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Self

from codes import ConfidenceLevel, EvidenceBasis, VerdictReason
from confidence_score.schema import (
    DENSITY_REVIEW_THRESHOLD,
    DENSITY_SUSPECT_THRESHOLD,
    GEOMETRY_REVIEW_THRESHOLD,
    GEOMETRY_SUSPECT_THRESHOLD,
    is_density_saturated,
)


def component_level(value: float, review: float, suspect: float) -> ConfidenceLevel:
    """Classify one measurement at the final raw thresholds.

    The input is not validated: a non-finite or negative value is read as "not
    measured" and classified ``INCOMPLETE`` rather than rejected, because every
    caller parses its metric with ``finite_float``, which already turns a blank,
    malformed, or non-finite cell into NaN. A negative magnitude is likewise
    impossible for the absolute RSZD and RMS-Zbond metrics this scores, so it
    can only mean the value did not come from a measurement.
    """
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
    reference cohort, so a higher score means the site is more ordinary
    relative to that cohort and a lower score means it stands out. They are
    NaN when no reference was applied or the component is not assessable, with
    one deliberate exception: a density at the EDSTATS saturation ceiling
    scores a finite ``0.0``, not NaN, because saturation is a measurement, not
    a missing value.
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


def _evidence_basis(density_available: bool, geometry_available: bool) -> EvidenceBasis:
    """Name the evidence a verdict rests on, given which components are assessable."""
    if density_available and geometry_available:
        return EvidenceBasis.DENSITY_AND_GEOMETRY
    if density_available:
        return EvidenceBasis.DENSITY_ONLY
    if geometry_available:
        return EvidenceBasis.GEOMETRY_ONLY
    return EvidenceBasis.NO_ASSESSABLE_EVIDENCE


def classify_site(rszd_abs: float, geometry_rms_zbond: float) -> SiteVerdict:
    """Apply the non-compensatory final decision matrix to one site."""
    density = density_level(rszd_abs)
    geometry = geometry_level(geometry_rms_zbond)
    density_available = density != ConfidenceLevel.INCOMPLETE
    geometry_available = geometry != ConfidenceLevel.INCOMPLETE
    evidence_basis = _evidence_basis(density_available, geometry_available)

    if not density_available and not geometry_available:
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


class EmpiricalDistribution:
    """A compact average-rank survival distribution for one raw metric.

    ``values`` is strictly increasing and ``counts`` holds each value's
    multiplicity, so the pair is the metric's whole empirical distribution.
    This is the canonical validator for a reference component: every rule a
    published distribution must satisfy is enforced in ``__init__``.
    """

    __slots__ = ("counts", "cumulative_below", "size", "values")

    def __init__(self, values: Sequence[float], counts: Sequence[int]) -> None:
        """Validate one component's values and counts, then freeze their totals."""
        if len(values) != len(counts):
            raise ValueError(
                "confidence reference values and counts differ in size: "
                f"{len(values)} values, {len(counts)} counts"
            )
        for index, value in enumerate(values):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "confidence reference contains an invalid value: "
                    f"index {index}, {value}"
                )
        for index, count in enumerate(counts):
            if isinstance(count, bool) or count < 1:
                raise ValueError(
                    "confidence reference contains an invalid count: "
                    f"index {index}, {count!r}"
                )
        for index, (left, right) in enumerate(
            zip(values, values[1:], strict=False), start=1
        ):
            if right <= left:
                raise ValueError(
                    "confidence reference values are not increasing: "
                    f"index {index}, {left} then {right}"
                )
        self.values = tuple(values)
        self.counts = tuple(counts)
        cumulative: list[int] = []
        running = 0
        for count in counts:
            cumulative.append(running)
            running += count
        self.cumulative_below = tuple(cumulative)
        self.size = running

    @classmethod
    def from_counts(cls, counts: Mapping[float, int]) -> Self:
        """Build one validated distribution from a per-value count mapping."""
        values = sorted(counts)
        return cls(values, [counts[value] for value in values])

    @property
    def distinct_value_count(self) -> int:
        """Return how many distinct values the distribution holds."""
        return len(self.values)

    def support_score(self, value: float) -> float:
        """Return reverse average-rank ECDF support; ordinary values rank high.

        Ties are detected by exact equality, so the caller must pass a value
        already rounded with ``schema.canonical_metric`` -- the rounding the
        reference's own values went through. Every production path does, via
        the canonicalized metrics written to the confidence input columns. An
        unrounded value simply misses its tie group and is scored as if it fell
        between two reference values.
        """
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
    """Frozen empirical density and RMS-Zbond distributions.

    "Frozen" means read-only rather than immutable by construction: every
    attribute is assigned once during ``__init__``, ``__slots__`` blocks new
    ones, the two distributions expose only tuples, and ``metadata`` is handed
    out as a ``MappingProxyType`` over a private copy so a loaded reference
    cannot be edited in place by the code that scores against it.
    """

    __slots__ = (
        "cohort_id",
        "cohort_size",
        "density",
        "geometry",
        "metadata",
        "reference_id",
    )

    def __init__(
        self,
        *,
        density_values: Sequence[float],
        density_counts: Sequence[int],
        geometry_values: Sequence[float],
        geometry_counts: Sequence[int],
        metadata: Mapping[str, Any],
    ) -> None:
        """Validate and freeze the empirical distributions and their metadata.

        Metadata is validated here rather than coerced, so a reference built
        from a hand-edited or truncated ``metadata.json`` fails with
        ``ValueError`` instead of a ``TypeError`` escaping the
        ``(OSError, ValueError)`` contract the CLI and driver catch.
        """
        self.density = EmpiricalDistribution(density_values, density_counts)
        self.geometry = EmpiricalDistribution(geometry_values, geometry_counts)
        if self.density.size == 0 and self.geometry.size == 0:
            raise ValueError("confidence reference has no assessable evidence")
        fields = dict(metadata)
        cohort_size = fields.get("input_row_count", 0)
        if (
            isinstance(cohort_size, bool)
            or not isinstance(cohort_size, int)
            or cohort_size < 0
        ):
            raise ValueError(
                f"confidence reference input row count is invalid: {cohort_size!r}"
            )
        reference_id = fields.get("reference_id", "")
        if not isinstance(reference_id, str):
            raise ValueError(
                f"confidence reference identifier is invalid: {reference_id!r}"
            )
        cohort_id = fields.get("cohort_id", "")
        if not isinstance(cohort_id, str):
            raise ValueError(
                f"confidence reference cohort identifier is invalid: {cohort_id!r}"
            )
        self.metadata: Mapping[str, Any] = MappingProxyType(fields)
        self.reference_id: str = reference_id
        self.cohort_id: str = cohort_id
        self.cohort_size: int = cohort_size

    @classmethod
    def from_counts(
        cls,
        density_counts: Mapping[float, int],
        geometry_counts: Mapping[float, int],
        metadata: Mapping[str, Any],
    ) -> Self:
        """Build a reference from per-value count mappings, sorted by value."""
        density = EmpiricalDistribution.from_counts(density_counts)
        geometry = EmpiricalDistribution.from_counts(geometry_counts)
        return cls(
            density_values=density.values,
            density_counts=density.counts,
            geometry_values=geometry.values,
            geometry_counts=geometry.counts,
            metadata=metadata,
        )

    @property
    def density_reference_size(self) -> int:
        """Return the number of density observations in the reference."""
        return self.density.size

    @property
    def geometry_reference_size(self) -> int:
        """Return the number of geometry observations in the reference."""
        return self.geometry.size

    @property
    def density_distinct_value_count(self) -> int:
        """Return how many distinct density values the reference holds."""
        return self.density.distinct_value_count

    @property
    def geometry_distinct_value_count(self) -> int:
        """Return how many distinct geometry values the reference holds."""
        return self.geometry.distinct_value_count


def _density_support(reference: ConfidenceReference | None, rszd_abs: float) -> float:
    """Rank one density metric, pinning a saturated measurement to zero support."""
    if reference is None:
        return math.nan
    if is_density_saturated(rszd_abs):
        return 0.0
    return reference.density.support_score(rszd_abs)


def score_site(
    rszd_abs: float,
    geometry_rms_zbond: float,
    reference: ConfidenceReference | None = None,
) -> SiteVerdict:
    """Return authoritative levels plus secondary empirical ranking scores."""
    verdict = classify_site(rszd_abs, geometry_rms_zbond)
    density_score = _density_support(reference, rszd_abs)
    geometry_score = (
        math.nan
        if reference is None
        else reference.geometry.support_score(geometry_rms_zbond)
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
