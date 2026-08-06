"""One donor-like atom image near one metal, at each coordination stage."""

from dataclasses import dataclass, field
from typing import TypedDict

from codes import CandidateSource, ContactScope, MultiDonorStatus
from structure_analysis import AtomSite


class DeclaredConnectionRecord(TypedDict):
    """Serialized provenance for one source-declared metal contact."""

    source: CandidateSource
    connection_id: str
    connection_type: str
    connection_link_id: str
    connection_asu: str
    connection_reported_distance: float


def _declared_connection_records() -> list[DeclaredConnectionRecord]:
    """Return a fresh, fully typed declaration-provenance collection."""
    return []


@dataclass(frozen=True, slots=True)
class DonorPolicy:
    inferred_allowed: bool
    rule: str
    override: str


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    status: str
    reason: str
    first_sphere_eligible: bool
    inferred_contact_eligible: bool
    assignment_target: float
    assignment_tolerance: float
    first_sphere_cutoff: float
    assignment_reference_kind: str
    assignment_reference: str


@dataclass(frozen=True, slots=True)
class GeometryResult:
    distance: float
    literature_distance: float
    literature_stdev: float
    zscore: float
    reference_covered: bool
    outlier: bool | str
    consistent: bool | str


@dataclass(frozen=True, slots=True)
class MultiDonorResult:
    detected: bool
    contact_count: int
    geometry_status: MultiDonorStatus
    contains_suspect_bond: bool
    score_eligible: bool
    score_exclusion_reason: str


@dataclass(slots=True)
class Candidate:
    neighbor: AtomSite
    distance_raw: float
    transformed_position: tuple[float, float, float]
    symmetry_contact: bool
    crystallographic_contact: bool
    strict_ncs_contact: bool
    strict_ncs_operation_id: str
    contact_scope: ContactScope
    symmetry_image_index: int
    symmetry_operation: str
    translation: tuple[int, int, int]
    candidate_sources: set[CandidateSource]
    #: One record per source declaration binding this image, with the fixed
    #: keys ``declared_candidate_for_connection`` writes.
    declared_connections: list[DeclaredConnectionRecord] = field(
        default_factory=_declared_connection_records
    )
    #: Set only on declaration-derived candidates; proximity discovery leaves
    #: it unset because the metal is already the search centre.
    metal: AtomSite | None = None
    #: False for donor classes no bundled reference covers. Those stay
    #: candidate evidence and are never promoted to bond rows.
    donor_class_supported: bool = True
    _donor_policy: DonorPolicy | None = field(default=None, init=False, repr=False)
    _eligibility: EligibilityResult | None = field(default=None, init=False, repr=False)
    _geometry: GeometryResult | None = field(default=None, init=False, repr=False)
    _multi_donor: MultiDonorResult | None = field(default=None, init=False, repr=False)

    def set_donor_policy(self, result: DonorPolicy) -> None:
        if self._donor_policy is not None:
            raise RuntimeError("candidate donor policy was already evaluated")
        self._donor_policy = result

    def donor_policy(self) -> DonorPolicy:
        if self._donor_policy is None:
            raise RuntimeError("candidate donor policy has not been evaluated")
        return self._donor_policy

    def set_eligibility(self, result: EligibilityResult) -> None:
        self.donor_policy()
        if self._eligibility is not None:
            raise RuntimeError("candidate eligibility was already evaluated")
        self._eligibility = result

    def eligibility(self) -> EligibilityResult:
        if self._eligibility is None:
            raise RuntimeError("candidate eligibility has not been evaluated")
        return self._eligibility

    def set_geometry(self, result: GeometryResult) -> None:
        if self._geometry is not None:
            raise RuntimeError("candidate geometry was already evaluated")
        self._geometry = result

    def geometry(self) -> GeometryResult:
        if self._geometry is None:
            raise RuntimeError("candidate geometry has not been evaluated")
        return self._geometry

    def set_multi_donor(self, result: MultiDonorResult) -> None:
        self.geometry()
        if self._multi_donor is not None:
            raise RuntimeError("candidate multi-donor status was already evaluated")
        self._multi_donor = result

    def multi_donor(self) -> MultiDonorResult:
        if self._multi_donor is None:
            raise RuntimeError("candidate multi-donor status has not been evaluated")
        return self._multi_donor
