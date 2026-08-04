"""The record a metal-donor contact is, from discovery to bond row.

One `Candidate` is created per donor-like atom image found near a metal --
either by the 4 A proximity search or by a source ``struct_conn``/``LINK``
declaration -- and then annotated in place by four later stages. It is the
pipeline's most-travelled data structure, and while it was a dict, nothing
declared what it holds at any point along the way.

**The stages fill it in, in order, and each depends on the last.**
``_annotate_donor_policy`` decides whether geometry alone may infer this donor;
``_identify_first_sphere_candidates`` applies the distance rule and reads that
decision; ``_annotate_contacts`` measures the z-score for whatever survives;
``_annotate_multi_donor_groups`` judges each contact against the others sharing
its donor residue. Every field below whose default is ``None`` is filled by one
of those stages, and ``None`` surviving to an output row means a stage was
skipped -- it is not a value the analysis produces.

Two shapes used to flow through the same functions without either being
declared: a declared candidate carries ``metal`` and ``donor_class_supported``,
a proximity candidate carried neither, and the code compensated with
``.get(name, default)`` calls that each invented their own default. Those
defaults are now the field defaults, in one place, where a reader can compare
them.

``slots=True`` is deliberate: a database run creates one of these per contact
per metal per entry, and, more usefully, it makes ``candidate.eligibilty = x``
an ``AttributeError`` at the assignment rather than a new attribute nobody
reads.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class Candidate:
    """One donor-like atom image near one metal, at any stage of analysis."""

    # -- Discovery: known the moment the atom image is found ---------------- #
    #: The donor atom, and -- for a declared connection only -- the metal it
    #: was declared against. The proximity search already knows its metal, so
    #: it does not carry one.
    neighbor: Any
    #: Measured metal-donor distance, before any rounding for output.
    distance_raw: float
    #: Where the donor image sits, after any symmetry transform.
    transformed_position: tuple
    symmetry_contact: bool
    crystallographic_contact: bool
    strict_ncs_contact: bool
    strict_ncs_operation_id: str
    contact_scope: str
    symmetry_image_index: int
    symmetry_operation: str
    translation: tuple
    #: How this candidate was found: ``proximity_4A``, ``struct_conn``, or
    #: ``LINK`` -- and more than one once ``_merge_candidates`` has run. No
    #: default: both discovery paths know how they found the atom, and a
    #: candidate that cannot say where it came from is not one.
    candidate_sources: set
    #: The declarations naming this contact, empty for a proximity-only find.
    declared_connections: list = field(default_factory=list)
    metal: Optional[Any] = None
    #: Whether a literature reference could cover this donor's residue class at
    #: all. Declared connections to nucleic acids, modified residues and
    #: organic ligands set this false: they stay candidate evidence carrying a
    #: measured distance, but are never promoted to bond rows. A proximity
    #: candidate is only ever an amino acid or a water, so it defaults true.
    donor_class_supported: bool = True

    # -- Donor policy: _annotate_donor_policy ------------------------------- #
    #: Whether geometry alone may infer this atom as a donor. The permissive
    #: default is inherited from the ``.get(name, True)`` the dict used here,
    #: and like that one it is unreachable: ``run_bond_analysis`` runs the
    #: donor-policy pass over every candidate before the first-sphere stage
    #: reads this. Kept rather than removed because dropping it would make an
    #: out-of-order caller silently refuse every donor instead.
    inferred_donor_allowed: bool = True
    inferred_donor_rule: Optional[str] = None
    #: ``declared_connection`` where a declaration overrides a donor the atom
    #: rule would have refused, blank otherwise.
    donor_rule_override: Optional[str] = None

    # -- First-sphere eligibility: _identify_first_sphere_candidates -------- #
    #: Whether the distance rule alone accepts this candidate. Distinct from
    #: ``inferred_contact_eligible``, which also requires a typical donor atom.
    first_sphere_eligible: bool = False
    inferred_contact_eligible: Optional[bool] = None
    eligibility_status: Optional[str] = None
    eligibility_reason: Optional[str] = None
    #: The literature target this candidate was measured against, the tolerance
    #: added to it, and the resulting cutoff. ``NAN`` where no reference covers
    #: the metal-donor pair.
    assignment_target: Optional[float] = None
    assignment_tolerance: Optional[float] = None
    first_sphere_cutoff: Optional[float] = None
    #: ``exact`` or ``element_fallback``, and the reference key it resolved to.
    assignment_reference_kind: Optional[str] = None
    assignment_reference: Optional[str] = None

    # -- Bond geometry: _annotate_contacts ---------------------------------- #
    #: The measured distance as published, rounded to 3 decimals.
    distance: Optional[float] = None
    literature_distance: Optional[float] = None
    literature_stdev: Optional[float] = None
    zscore: Optional[float] = None
    reference_covered: Optional[bool] = None
    #: ``True``/``False`` where a z-score exists, and blank where one does not
    #: -- the two are different answers and the outputs keep them apart.
    geometry_outlier: Any = None
    geometry_consistent: Any = None

    # -- Donor-group context: _annotate_multi_donor_groups ------------------ #
    multi_donor_detected: Optional[bool] = None
    multi_donor_contact_count: Optional[int] = None
    #: ``single_donor``, ``consistent``, ``suspect`` or ``indeterminate`` --
    #: see ``codes.MultiDonorStatus``, whose ``suspect`` is *not* the site-level
    #: ``suspect`` of ``codes.GeometryStatus``.
    multi_donor_geometry_status: Optional[str] = None
    multi_donor_contains_suspect_bond: bool = False
    score_eligible: Optional[bool] = None
    score_exclusion_reason: Optional[str] = None
