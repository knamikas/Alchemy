"""Typed contact records and results for each coordination-analysis stage."""

from dataclasses import dataclass, field
from typing import Literal

from codes import (
    CandidateSource,
    DonorRuleOverride,
    EligibilityReason,
    EligibilityStatus,
    InferredDonorRule,
    MultiDonorStatus,
    ReferenceKind,
    ScoreExclusionReason,
)
from structure_analysis import AtomSite, ContactImage


@dataclass(frozen=True, slots=True)
class DeclaredConnectionRecord:
    """Serialized provenance for one source-declared metal contact.

    Records are shared by reference. ``candidates.merge_candidates`` copies
    the list that holds them but not the records themselves, so one record is
    reachable from every candidate the declaration bound. Freezing the record
    is what makes that safe: rewriting one in place would silently rewrite the
    provenance of every other candidate that shares it.
    """

    source: CandidateSource
    connection_id: str
    connection_type: str
    connection_link_id: str
    connection_asu: str
    connection_reported_distance: float

    @property
    def identity(self) -> tuple[CandidateSource, str]:
        """What makes two records the same declaration when provenance merges."""
        return (self.source, self.connection_id)


@dataclass(frozen=True, slots=True)
class DonorPolicy:
    """Record whether donor chemistry permits an inferred contact."""

    inferred_allowed: bool
    #: The donor-atom rule that decided ``inferred_allowed``, as
    #: ``eligibility._inferred_donor_rule`` chose it.
    rule: InferredDonorRule
    #: Empty when the donor-atom rule was not overridden.
    override: DonorRuleOverride | Literal[""]


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    """Record first-sphere eligibility and its assignment thresholds."""

    status: EligibilityStatus
    reason: EligibilityReason
    first_sphere_eligible: bool
    inferred_contact_eligible: bool
    assignment_target: float
    assignment_tolerance: float
    first_sphere_cutoff: float
    assignment_reference_kind: ReferenceKind
    assignment_reference: str


@dataclass(frozen=True, slots=True)
class GeometryResult:
    """Record a candidate contact's reference-based geometry assessment.

    ``outlier`` compares ``zscore`` against
    ``coordination.policy.ZSCORE_OUTLIER_CUTOFF``. It is three-valued:
    ``True``, ``False``, or ``None`` when ``zscore`` is not finite -- no
    bundled reference covered the pair, or no DPI was available -- so the
    contact was never assessed. ``None`` is falsy and is published as a blank
    cell, so consumers must compare with ``is True`` (or ``is False``); a
    plain truth test silently folds the unassessed case into ``False``.

    ``consistent`` is not stored: it is exactly ``not outlier`` for an
    assessed contact and ``None`` for an unassessed one. It stays part of the
    record because the bond schema publishes both columns.

    ``score_eligible`` says whether this contact contributes geometry evidence
    to the aggregates. It is decided from this contact's own z-score, never
    from the multi-donor group it belongs to: group status is contextual and
    excludes nothing (docs/method.md).
    """

    distance: float
    literature_distance: float
    literature_stdev: float
    zscore: float
    reference_covered: bool
    outlier: bool | None
    score_eligible: bool
    #: Empty when the contact is scored.
    score_exclusion_reason: ScoreExclusionReason | Literal[""]

    @property
    def consistent(self) -> bool | None:
        """The complement of ``outlier``, and ``None`` when it is unassessed."""
        return None if self.outlier is None else not self.outlier


@dataclass(frozen=True, slots=True)
class MultiDonorResult:
    """Record the geometry verdict for one residue's donor group."""

    detected: bool
    contact_count: int
    geometry_status: MultiDonorStatus
    contains_suspect_bond: bool


@dataclass(slots=True)
class Candidate:
    """Carry one donor-like atom image through coordination analysis stages.

    ``candidate_sources`` and ``declared_connections`` stay mutable so that
    ``candidates.merge_candidates`` can fold the provenance of several
    discovery passes into one candidate, and so that ``eligibility`` can add
    the declaration provenance it resolves. Every such merge must happen
    before the annotation stages run: donor policy, eligibility, geometry and
    multi-donor status each read the provenance once, so provenance added
    afterwards is invisible to the verdicts and to the rows derived from them.
    """

    neighbor: AtomSite
    image: ContactImage
    candidate_sources: set[CandidateSource]
    #: One record per source declaration binding this image, as
    #: ``declared_candidate_for_connection`` writes them.
    declared_connections: list[DeclaredConnectionRecord] = field(
        default_factory=list[DeclaredConnectionRecord]
    )
    #: False for donor classes no bundled reference covers. Those stay
    #: candidate evidence and are never promoted to bond rows.
    donor_class_supported: bool = True
    # The four stage results below are ``init=False``, so any copy made through
    # ``__init__`` -- including ``dataclasses.replace()`` -- starts out
    # unannotated. Copy a candidate with ``with_provenance`` instead, which
    # refuses to copy one that an annotation stage has already reached.
    _donor_policy: DonorPolicy | None = field(default=None, init=False, repr=False)
    _eligibility: EligibilityResult | None = field(default=None, init=False, repr=False)
    _geometry: GeometryResult | None = field(default=None, init=False, repr=False)
    _multi_donor: MultiDonorResult | None = field(default=None, init=False, repr=False)

    def _absorb_provenance(self, other: "Candidate") -> None:
        """Add ``other``'s discovery and declaration provenance to this record.

        Declarations are deduplicated on ``(source, connection_id)``, so the
        same declaration reached by two discovery routes contributes one
        record. The records themselves are shared by reference, not copied.
        """
        self.candidate_sources.update(other.candidate_sources)
        known_connections = {record.identity for record in self.declared_connections}
        for record in other.declared_connections:
            connection_key = record.identity
            if connection_key not in known_connections:
                self.declared_connections.append(record)
                known_connections.add(connection_key)

    def add_provenance(self, other: "Candidate") -> None:
        """Fold another record's provenance in before any stage has read it.

        Donor policy, eligibility and the rows derived from them each read the
        provenance once, so a merge that arrives afterwards is invisible to
        the verdicts already recorded. This refuses such a merge outright;
        the one collapse that legitimately runs later goes through
        ``merge_provenance_after_annotation``.
        """
        if self._donor_policy is not None:
            raise RuntimeError(
                "candidate provenance cannot be added after the donor-policy stage"
            )
        self._absorb_provenance(other)

    def merge_provenance_after_annotation(self, other: "Candidate") -> None:
        """Fold provenance in after annotation, where that cannot change a verdict.

        ``candidates.deduplicate_special_position_contacts`` collapses
        near-coincident images of one deposited atom, and it can only run once
        eligibility has selected the contacts to collapse -- which is after
        ``eligibility.annotate_donor_policy`` and ``set_eligibility`` have
        read the provenance of every candidate.

        That is safe because of what the two stages actually read, which is
        ``bool(declared_connections)`` and nothing else:

        * ``annotate_donor_policy`` uses it only to grant
          ``DonorRuleOverride.DECLARED_CONNECTION``, and only when the donor
          rule refused the atom.
        * ``_eligibility_for`` uses it only on the missing-reference branch,
          and only when the donor rule refused the atom.

        Every candidate that reaches the collapse is either already declared,
        so the flag is already ``True`` and adding another record cannot
        change it, or inferred-eligible, which required the donor rule to
        allow the atom and a finite reference -- and on that branch neither
        stage consults the flag at all. ``candidate_sources`` is read by no
        stage, only by the published ``candidate_source`` column.
        """
        self._absorb_provenance(other)

    def with_provenance(
        self,
        *,
        candidate_sources: set[CandidateSource],
        declared_connections: list[DeclaredConnectionRecord],
    ) -> "Candidate":
        """Return an unannotated copy of this candidate with new provenance.

        The copy owns the collections it is given, so a caller that merges
        provenance into it cannot reach back into the record it copied.

        The four stage results are ``init=False`` and cannot be carried over,
        so copying an annotated candidate would silently drop its verdicts.
        This raises instead, which makes the "merge provenance before the
        annotation stages run" invariant an enforced one rather than a
        documented one.
        """
        annotated = (
            self._donor_policy,
            self._eligibility,
            self._geometry,
            self._multi_donor,
        )
        if any(result is not None for result in annotated):
            raise RuntimeError(
                "candidate cannot be copied after an annotation stage has run"
            )
        return Candidate(
            neighbor=self.neighbor,
            image=self.image,
            candidate_sources=candidate_sources,
            declared_connections=declared_connections,
            donor_class_supported=self.donor_class_supported,
        )

    def set_donor_policy(self, result: DonorPolicy) -> None:
        """Set the donor policy exactly once."""
        if self._donor_policy is not None:
            raise RuntimeError("candidate donor policy was already evaluated")
        self._donor_policy = result

    def donor_policy(self) -> DonorPolicy:
        """Return the evaluated donor policy."""
        if self._donor_policy is None:
            raise RuntimeError("candidate donor policy has not been evaluated")
        return self._donor_policy

    def set_eligibility(self, result: EligibilityResult) -> None:
        """Set eligibility exactly once after donor-policy evaluation."""
        self.donor_policy()
        if self._eligibility is not None:
            raise RuntimeError("candidate eligibility was already evaluated")
        self._eligibility = result

    def eligibility(self) -> EligibilityResult:
        """Return the evaluated eligibility result."""
        if self._eligibility is None:
            raise RuntimeError("candidate eligibility has not been evaluated")
        return self._eligibility

    def set_geometry(self, result: GeometryResult) -> None:
        """Set the contact geometry exactly once after eligibility assignment."""
        self.eligibility()
        if self._geometry is not None:
            raise RuntimeError("candidate geometry was already evaluated")
        self._geometry = result

    def geometry(self) -> GeometryResult:
        """Return the evaluated contact geometry."""
        if self._geometry is None:
            raise RuntimeError("candidate geometry has not been evaluated")
        return self._geometry

    def set_multi_donor(self, result: MultiDonorResult) -> None:
        """Set multi-donor status exactly once after geometry evaluation."""
        self.geometry()
        if self._multi_donor is not None:
            raise RuntimeError("candidate multi-donor status was already evaluated")
        self._multi_donor = result

    def multi_donor(self) -> MultiDonorResult:
        """Return the evaluated multi-donor result."""
        if self._multi_donor is None:
            raise RuntimeError("candidate multi-donor status has not been evaluated")
        return self._multi_donor
