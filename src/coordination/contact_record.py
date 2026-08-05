"""One donor-like atom image near one metal, at each coordination stage."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from codes import CandidateSource, ContactScope, MultiDonorStatus
from structure_analysis import AtomSite


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
    #: keys ``_declared_candidate_for_connection`` writes.
    declared_connections: list[dict[str, Any]] = field(default_factory=list)
    #: Set only on declaration-derived candidates; proximity discovery leaves
    #: it unset because the metal is already the search centre.
    metal: Optional[AtomSite] = None
    #: False for donor classes no bundled reference covers. Those stay
    #: candidate evidence and are never promoted to bond rows.
    donor_class_supported: bool = True

    inferred_donor_allowed: bool = True
    inferred_donor_rule: Optional[str] = None
    #: ``declared_connection`` where a declaration overrides a donor the atom
    #: rule would have refused, blank otherwise.
    donor_rule_override: Optional[str] = None

    #: Whether the distance rule alone accepts this candidate. Distinct from
    #: ``inferred_contact_eligible``, which also requires a typical donor atom.
    first_sphere_eligible: bool = False
    inferred_contact_eligible: Optional[bool] = None
    eligibility_status: Optional[str] = None
    eligibility_reason: Optional[str] = None
    assignment_target: Optional[float] = None
    assignment_tolerance: Optional[float] = None
    first_sphere_cutoff: Optional[float] = None
    #: ``exact`` or ``element_fallback``, and the reference key it resolved to.
    assignment_reference_kind: Optional[str] = None
    assignment_reference: Optional[str] = None

    distance: Optional[float] = None
    literature_distance: Optional[float] = None
    literature_stdev: Optional[float] = None
    zscore: Optional[float] = None
    reference_covered: Optional[bool] = None
    #: ``True``/``False`` where a z-score exists, blank where one does not: the
    #: outputs keep the two answers apart.
    geometry_outlier: Union[bool, str, None] = None
    geometry_consistent: Union[bool, str, None] = None

    multi_donor_detected: Optional[bool] = None
    multi_donor_contact_count: Optional[int] = None
    #: ``MultiDonorStatus.SUSPECT`` is not the site-level
    #: ``GeometryStatus.SUSPECT``: same string, different question.
    multi_donor_geometry_status: Optional[MultiDonorStatus] = None
    multi_donor_contains_suspect_bond: bool = False
    score_eligible: Optional[bool] = None
    score_exclusion_reason: Optional[str] = None
