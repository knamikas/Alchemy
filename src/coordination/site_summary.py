"""Condense one metal site's assessed contacts into its summary columns.

``SiteSummary`` is the coordination analysis's share of ``STATS_EXTRA_COLUMNS``:
``stats_extra_values`` in schema.py merges it into the EDSTATS row of the
site, and fills the remaining columns from the structure and the metal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import TypedDict

from codes import (
    ContactScope,
    ContextWarningReason,
    DonorRuleOverride,
    GeometryStatus,
    MultiDonorStatus,
    ReasonCode,
)
from coordination.contact_record import Candidate, MultiDonorResult
from coordination.dpi import DpiComponents
from coordination.geometry import residue_image_key
from coordination.schema import STATS_EXTRA_COLUMNS
from coordination.site_environment import (
    EntryModelStatistics,
    MetalProximity,
    MetalSpecialPosition,
)
from structure_analysis import NAN, AtomSite, ContactImage, StructureContext


class SiteSummary(TypedDict):
    """The per-site summary columns, in the order ``site_summary`` writes them.

    Counts are integers when the scope was assessed and NaN when it was not.
    Blank strings mark values that were not assessed; ``False`` marks values
    that were assessed and found absent.
    """

    dpi: float
    resolution: float
    r_free: float
    reflection_count: float
    asu_volume: float
    occupancy_weighted_atom_count: float
    deposited_occupancy_weighted_atom_count: float
    dpi_unavailable_reason: str
    candidate_contact_count: int | float
    reference_covered_contact_count: int | float
    geometry_outlier_contact_count: int | float
    geometry_consistent_contact_count: int | float
    score_eligible_contact_count: int | float
    score_excluded_contact_count: int | float
    scored_geometry_outlier_contact_count: int | float
    scored_geometry_consistent_contact_count: int | float
    multi_donor_residue_group_count: int | float
    multi_donor_contact_count: int | float
    suspect_multi_donor_residue_group_count: int | float
    indeterminate_multi_donor_residue_group_count: int | float
    explicit_contact_count: int | float
    symmetry_contact_count: int | float
    image_inclusive_contact_count: int | float
    crystallographic_contact_count: int | float
    strict_ncs_contact_count: int | float
    combined_ncs_crystallographic_contact_count: int | float
    geometry_outlier_count_explicit: int | float
    geometry_outlier_count_image_inclusive: int | float
    geometry_coverage_explicit: float
    geometry_coverage_image_inclusive: float
    explicit_geometry_status: GeometryStatus | str
    image_inclusive_geometry_status: GeometryStatus | str
    generated_contact_scope: str
    geometry_classification_changes_with_generated_images: bool | str
    coordination_depends_on_crystallographic_symmetry: bool | str
    coordination_depends_on_strict_ncs: bool | str
    metal_overfull_occupancy: bool | str
    geometry_not_assessed_reason: str
    context_warning: bool
    context_warning_reasons: str
    non_typical_first_sphere_candidate_count: int
    declared_donor_override_contact_count: int
    donor_b_iso_count: int
    donor_median_b_iso: float
    metal_donor_b_ratio: float
    metal_minus_donor_b_iso: float
    metal_donor_b_similarity: float
    nearest_metal_distance: float
    nearest_metal_element: str
    nearest_metal_site_id: str
    nearby_metal_count_6a: int | float
    metal_special_position: bool | str
    metal_site_symmetry_order: int | float
    metal_expected_crystallographic_occupancy: float
    metal_occupancy_matches_site_symmetry: bool | str
    entry_nonwater_median_b_iso: float


#: The per-site summary's share of ``STATS_EXTRA_COLUMNS``. ``site_summary``
#: covers exactly these; ``stats_extra_values`` in schema.py fills the rest
#: from the structure, the metal, and the density row.
SUMMARY_OWNED_STATS_EXTRA_COLUMNS: frozenset[str] = SiteSummary.__required_keys__
if not SUMMARY_OWNED_STATS_EXTRA_COLUMNS.issubset(STATS_EXTRA_COLUMNS):
    raise ValueError(
        "SiteSummary names columns outside STATS_EXTRA_COLUMNS: "
        + ", ".join(
            sorted(SUMMARY_OWNED_STATS_EXTRA_COLUMNS - frozenset(STATS_EXTRA_COLUMNS))
        )
    )


@dataclass(frozen=True, slots=True)
class _DonorBFactors:
    """One metal's B factor compared with the median of its assigned donors."""

    donor_count: int
    donor_median_b_iso: float
    ratio: float
    difference: float
    similarity: float


@dataclass(frozen=True, slots=True)
class _SiteContext:
    """Coordination-relevant context of one site that changes no score or classification."""

    warning: bool
    warning_reasons: str
    non_typical_first_sphere_candidate_count: int
    declared_donor_override_contact_count: int


@dataclass(frozen=True, slots=True)
class _ScopeSummary:
    """Contact counts and the geometry verdict for one search scope.

    Counts are integers when the scope was assessed and NaN when it was not;
    ``status`` is then blank instead of a ``GeometryStatus``. ``coverage`` is
    the one exception: it is NaN both for a scope that was never searched and
    for an assessed scope with no contact to divide by, so it cannot be read
    as a record of whether the scope was assessed.
    """

    #: Contacts assigned in this scope; published as
    #: ``candidate_contact_count``.
    contact_count: int | float
    reference_covered_count: int | float
    outlier_count: int | float
    consistent_count: int | float
    score_eligible_count: int | float
    score_excluded_count: int | float
    scored_outlier_count: int | float
    scored_consistent_count: int | float
    multi_donor_group_count: int | float
    multi_donor_contact_count: int | float
    suspect_multi_donor_group_count: int | float
    indeterminate_multi_donor_group_count: int | float
    coverage: float
    status: GeometryStatus | str

    @classmethod
    def unavailable(cls) -> _ScopeSummary:
        """The summary of a scope that could not be searched."""
        return cls(
            contact_count=NAN,
            reference_covered_count=NAN,
            outlier_count=NAN,
            consistent_count=NAN,
            score_eligible_count=NAN,
            score_excluded_count=NAN,
            scored_outlier_count=NAN,
            scored_consistent_count=NAN,
            multi_donor_group_count=NAN,
            multi_donor_contact_count=NAN,
            suspect_multi_donor_group_count=NAN,
            indeterminate_multi_donor_group_count=NAN,
            coverage=NAN,
            status="",
        )

    @property
    def scored_assessable_count(self) -> int | float:
        """Scored contacts with a definite outlier or consistent verdict."""
        return self.scored_outlier_count + self.scored_consistent_count


def _donor_b_factors(metal: AtomSite, contacts: Sequence[Candidate]) -> _DonorBFactors:
    """Compare one metal B factor with the median of its assigned donors."""
    donor_values = [
        contact.neighbor.b_iso
        for contact in contacts
        if math.isfinite(contact.neighbor.b_iso)
    ]
    donor_count = len(donor_values)
    if not donor_values:
        return _DonorBFactors(donor_count, NAN, NAN, NAN, NAN)

    donor_median = float(median(donor_values))
    metal_b = metal.b_iso
    difference = NAN
    ratio = NAN
    similarity = NAN
    if math.isfinite(metal_b):
        difference = round(metal_b - donor_median, 3)
        if metal_b > 0.0 and donor_median > 0.0:
            raw_ratio = metal_b / donor_median
            if math.isfinite(raw_ratio):
                ratio = round(raw_ratio, 4)
                similarity = round(math.exp(-abs(math.log(raw_ratio))), 4)
    return _DonorBFactors(
        donor_count=donor_count,
        donor_median_b_iso=round(donor_median, 3),
        ratio=ratio,
        difference=difference,
        similarity=similarity,
    )


def _image_count(
    contacts: Sequence[Candidate] | None,
    predicate: Callable[[ContactImage], bool],
) -> int | float:
    """Count the image contacts ``predicate`` accepts.

    ``None`` is the unavailable symmetry search, which counts nothing: every
    generated-image count is then NaN rather than zero.
    """
    if contacts is None:
        return NAN
    return sum(predicate(contact.image) for contact in contacts)


def _scope_summary(contacts: Sequence[Candidate]) -> _ScopeSummary:
    """Count the assessed contacts of one scope and classify its geometry."""
    candidate = len(contacts)
    geometries = [contact.geometry() for contact in contacts]
    multi_donor = [contact.multi_donor() for contact in contacts]
    covered = sum(geometry.reference_covered for geometry in geometries)
    outlier = sum(geometry.outlier is True for geometry in geometries)
    consistent = sum(geometry.consistent is True for geometry in geometries)
    score_eligible = sum(geometry.score_eligible for geometry in geometries)
    score_excluded = candidate - score_eligible
    scored_outlier = sum(
        geometry.score_eligible and geometry.outlier is True for geometry in geometries
    )
    scored_consistent = sum(
        geometry.score_eligible and geometry.consistent is True
        for geometry in geometries
    )
    multi_donor_groups = {
        residue_image_key(contact)
        for contact, result in zip(contacts, multi_donor, strict=True)
        if result.detected
    }
    suspect_multi_donor_groups = {
        residue_image_key(contact)
        for contact, result in zip(contacts, multi_donor, strict=True)
        if result.geometry_status == MultiDonorStatus.SUSPECT
    }
    indeterminate_multi_donor_groups = {
        residue_image_key(contact)
        for contact, result in zip(contacts, multi_donor, strict=True)
        if result.geometry_status == MultiDonorStatus.INDETERMINATE
    }
    multi_donor_contacts = sum(result.detected for result in multi_donor)
    scored_assessable = scored_outlier + scored_consistent
    if scored_assessable == 0:
        status = GeometryStatus.INSUFFICIENT_DATA
    elif scored_outlier:
        status = GeometryStatus.SUSPECT
    else:
        status = GeometryStatus.PLAUSIBLE
    coverage = round(covered / candidate, 4) if candidate else NAN
    return _ScopeSummary(
        contact_count=candidate,
        reference_covered_count=covered,
        outlier_count=outlier,
        consistent_count=consistent,
        score_eligible_count=score_eligible,
        score_excluded_count=score_excluded,
        scored_outlier_count=scored_outlier,
        scored_consistent_count=scored_consistent,
        multi_donor_group_count=len(multi_donor_groups),
        multi_donor_contact_count=multi_donor_contacts,
        suspect_multi_donor_group_count=len(suspect_multi_donor_groups),
        indeterminate_multi_donor_group_count=len(indeterminate_multi_donor_groups),
        coverage=coverage,
        status=status,
    )


def context_warning_reasons(
    candidate: Candidate,
    include_proximal: bool = False,
    multi_donor: MultiDonorResult | None = None,
) -> list[str]:
    """Return the contextual warnings that apply to one candidate contact.

    ``include_proximal`` also reports a non-typical donor that is outside the
    first sphere, which suits a candidate row but not an assigned contact.

    ``multi_donor`` is this candidate's evaluated multi-donor result and the
    only source of ``suspect_multi_donor_group``. ``None`` means the group was
    not evaluated for this row, and the reason is then silently omitted rather
    than reported as absent: a caller holding the result must pass it, or the
    column understates the site and the scoring fed from it does
    too. The candidate rows are the one correct ``None``, because the
    multi-donor stage annotates only the contacts that were assigned, so an
    unassigned candidate has no result to pass.
    """
    reasons: list[str] = []
    policy = candidate.donor_policy()
    eligibility = candidate.eligibility()
    if candidate.neighbor.occupancy_valid and candidate.neighbor.occupancy == 0.0:
        reasons.append(ContextWarningReason.ZERO_OCCUPANCY_NEIGHBOR)
    if not policy.inferred_allowed:
        if candidate.declared_connections:
            reasons.append(ContextWarningReason.DECLARED_NON_TYPICAL_DONOR)
        elif eligibility.first_sphere_eligible:
            reasons.append(ContextWarningReason.NON_TYPICAL_FIRST_SPHERE_CANDIDATE)
        elif include_proximal:
            reasons.append(ContextWarningReason.NON_TYPICAL_PROXIMAL_CANDIDATE)
    if multi_donor is not None and multi_donor.contains_suspect_bond:
        reasons.append(ContextWarningReason.SUSPECT_MULTI_DONOR_GROUP)
    # Each branch above appends at most one reason, so there is nothing to
    # deduplicate here; ``_site_context`` pools several candidates and does.
    return reasons


def _site_context(
    contacts: Sequence[Candidate], candidates: Sequence[Candidate]
) -> _SiteContext:
    """Aggregate coordination-relevant context without changing scores or classifications."""
    reasons: list[str] = []
    for contact in contacts:
        reasons.extend(
            context_warning_reasons(contact, multi_donor=contact.multi_donor())
        )
    non_typical_first_sphere = [
        candidate
        for candidate in candidates
        if (
            not candidate.donor_policy().inferred_allowed
            and candidate.eligibility().first_sphere_eligible
        )
    ]
    if non_typical_first_sphere:
        reasons.append(ContextWarningReason.NON_TYPICAL_FIRST_SPHERE_CANDIDATE)
    reasons = list(dict.fromkeys(reasons))
    return _SiteContext(
        warning=bool(reasons),
        warning_reasons="|".join(reasons),
        non_typical_first_sphere_candidate_count=len(non_typical_first_sphere),
        declared_donor_override_contact_count=sum(
            contact.donor_policy().override == DonorRuleOverride.DECLARED_CONNECTION
            for contact in contacts
        ),
    )


# Generated-contact scope and its two dependency flags, keyed by whether the
# site has any crystallographic and any strict-NCS contact. Read only once a
# site has generated contacts, and every generated contact is one or the other,
# so the (False, False) row is unreachable; it is kept so an inconsistent count
# degrades to the NCS scope instead of raising.
_GENERATED_SCOPES: dict[tuple[bool, bool], tuple[ContactScope, bool, bool]] = {
    (True, True): (ContactScope.STRICT_NCS_AND_CRYSTALLOGRAPHIC, True, True),
    (True, False): (ContactScope.CRYSTALLOGRAPHIC, True, False),
    (False, True): (ContactScope.STRICT_NCS, False, True),
    (False, False): (ContactScope.STRICT_NCS, False, True),
}


def _metal_overfull_occupancy(
    metal: AtomSite, structure: StructureContext
) -> bool | str:
    """Whether this metal's own alternate occupancies sum above one.

    The entry-wide survey can only answer that for a chemical site whose every
    deposited record carries a usable occupancy: without one the sum is
    unknown, and the site is absent from ``overfull_site_keys`` because it was
    never judged, not because it was judged and found sound. Such a site is
    reported blank; ``False`` stays a positive statement that the deposited
    occupancies were read and do not exceed one.
    """
    identity = metal.chemical_site_identity
    alternates = [
        atom
        for atom in structure.residue_for_atom(metal).source_atoms
        if atom.chemical_site_identity == identity
    ]
    if not all(atom.occupancy_valid for atom in alternates):
        return ""
    return identity in structure.occupancy.overfull_site_keys


def site_summary(
    metal: AtomSite,
    structure: StructureContext,
    explicit_contacts: Sequence[Candidate],
    image_contacts: Sequence[Candidate] | None,
    candidates: Sequence[Candidate],
    dpi_components: DpiComponents,
    model_statistics: EntryModelStatistics,
    proximity: MetalProximity,
    special_position: MetalSpecialPosition,
    geometry_not_assessed_reason: str | None = None,
) -> SiteSummary:
    """Assemble the summary of one metal site from its assessed contacts.

    ``image_contacts`` is ``None`` when the symmetry search was unavailable;
    otherwise the image-inclusive scope is primary. ``candidates`` are every
    annotated candidate of the primary scope. ``geometry_not_assessed_reason``
    replaces the reasons derived here when the caller already knows geometry
    could not be assessed at all.
    """
    explicit = _scope_summary(explicit_contacts)
    image_search_available = image_contacts is not None
    image_inclusive = (
        _scope_summary(image_contacts)
        if image_contacts is not None
        else _ScopeSummary.unavailable()
    )
    primary = image_inclusive if image_search_available else explicit
    primary_contacts = (
        image_contacts if image_contacts is not None else explicit_contacts
    )
    context = _site_context(primary_contacts, candidates)
    donor_b = _donor_b_factors(metal, primary_contacts)
    ni = model_statistics.occupancy_weighted_atom_count
    deposited_ni = model_statistics.deposited_occupancy_weighted_atom_count
    # Report overfull occupancy on the metal itself, separately from entry-wide warnings.
    metal_overfull = _metal_overfull_occupancy(metal, structure)
    symmetry_count = _image_count(image_contacts, lambda image: image.symmetry_contact)
    crystallographic_count = _image_count(
        image_contacts, lambda image: image.crystallographic_contact
    )
    strict_ncs_count = _image_count(
        image_contacts, lambda image: image.strict_ncs_contact
    )
    combined_count = _image_count(
        image_contacts,
        lambda image: image.crystallographic_contact and image.strict_ncs_contact,
    )
    # Blank means symmetry was not assessed; False means it was assessed and absent.
    changed: str | bool
    depends_crystallographic: str | bool
    depends_strict_ncs: str | bool
    if not image_search_available:
        generated_scope = ""
        changed = ""
        depends_crystallographic = ""
        depends_strict_ncs = ""
    else:
        changed = explicit.status != image_inclusive.status
        generated_scope, depends_crystallographic, depends_strict_ncs = (
            _GENERATED_SCOPES[
                bool(crystallographic_count),
                bool(strict_ncs_count),
            ]
            if symmetry_count
            else (ContactScope.NONE, False, False)
        )

    if geometry_not_assessed_reason is None:
        reasons: list[str] = []
        if dpi_components.reason_code:
            reasons.append(dpi_components.reason_code)
        if not image_search_available:
            reasons.append(ReasonCode.SYMMETRY_SEARCH_UNAVAILABLE)
        if primary.scored_assessable_count == 0:
            reasons.append("no_assessable_reference_contacts")
        geometry_not_assessed_reason = "|".join(dict.fromkeys(reasons))

    return SiteSummary(
        dpi=dpi_components.dpi,
        resolution=dpi_components.resolution,
        r_free=dpi_components.r_free,
        reflection_count=dpi_components.reflection_count,
        asu_volume=dpi_components.asu_volume,
        occupancy_weighted_atom_count=(round(ni, 6) if math.isfinite(ni) else NAN),
        deposited_occupancy_weighted_atom_count=(
            round(deposited_ni, 6) if math.isfinite(deposited_ni) else NAN
        ),
        dpi_unavailable_reason=dpi_components.reason_code,
        candidate_contact_count=primary.contact_count,
        reference_covered_contact_count=primary.reference_covered_count,
        geometry_outlier_contact_count=primary.outlier_count,
        geometry_consistent_contact_count=primary.consistent_count,
        score_eligible_contact_count=primary.score_eligible_count,
        score_excluded_contact_count=primary.score_excluded_count,
        scored_geometry_outlier_contact_count=primary.scored_outlier_count,
        scored_geometry_consistent_contact_count=primary.scored_consistent_count,
        multi_donor_residue_group_count=primary.multi_donor_group_count,
        multi_donor_contact_count=primary.multi_donor_contact_count,
        suspect_multi_donor_residue_group_count=(
            primary.suspect_multi_donor_group_count
        ),
        indeterminate_multi_donor_residue_group_count=(
            primary.indeterminate_multi_donor_group_count
        ),
        explicit_contact_count=explicit.contact_count,
        symmetry_contact_count=symmetry_count,
        image_inclusive_contact_count=image_inclusive.contact_count,
        crystallographic_contact_count=crystallographic_count,
        strict_ncs_contact_count=strict_ncs_count,
        combined_ncs_crystallographic_contact_count=combined_count,
        geometry_outlier_count_explicit=explicit.outlier_count,
        geometry_outlier_count_image_inclusive=image_inclusive.outlier_count,
        geometry_coverage_explicit=explicit.coverage,
        geometry_coverage_image_inclusive=image_inclusive.coverage,
        explicit_geometry_status=explicit.status,
        image_inclusive_geometry_status=image_inclusive.status,
        generated_contact_scope=generated_scope,
        geometry_classification_changes_with_generated_images=changed,
        coordination_depends_on_crystallographic_symmetry=depends_crystallographic,
        coordination_depends_on_strict_ncs=depends_strict_ncs,
        metal_overfull_occupancy=metal_overfull,
        geometry_not_assessed_reason=geometry_not_assessed_reason,
        context_warning=context.warning,
        context_warning_reasons=context.warning_reasons,
        non_typical_first_sphere_candidate_count=(
            context.non_typical_first_sphere_candidate_count
        ),
        declared_donor_override_contact_count=(
            context.declared_donor_override_contact_count
        ),
        donor_b_iso_count=donor_b.donor_count,
        donor_median_b_iso=donor_b.donor_median_b_iso,
        metal_donor_b_ratio=donor_b.ratio,
        metal_minus_donor_b_iso=donor_b.difference,
        metal_donor_b_similarity=donor_b.similarity,
        nearest_metal_distance=proximity.nearest_distance,
        nearest_metal_element=proximity.nearest_element,
        nearest_metal_site_id=proximity.nearest_site_id,
        nearby_metal_count_6a=proximity.count_within_6a,
        metal_special_position=special_position.special_position,
        metal_site_symmetry_order=special_position.site_symmetry_order,
        metal_expected_crystallographic_occupancy=special_position.expected_occupancy,
        metal_occupancy_matches_site_symmetry=(
            special_position.occupancy_matches_site_symmetry
        ),
        entry_nonwater_median_b_iso=model_statistics.nonwater_median_b_iso,
    )


def unassessable_site_summary(
    metal: AtomSite,
    structure: StructureContext,
    dpi_components: DpiComponents,
    model_statistics: EntryModelStatistics,
    proximity: MetalProximity,
    special_position: MetalSpecialPosition,
) -> SiteSummary:
    """The site summary for a metal whose coordinates rule out any geometry.

    Both scopes are assessed as empty rather than unavailable, so the summary
    reports zero contacts wherever the symmetry search itself was possible.
    """
    return site_summary(
        metal,
        structure,
        [],
        [] if structure.symmetry.search_available else None,
        [],
        dpi_components,
        model_statistics,
        proximity,
        special_position,
        geometry_not_assessed_reason=ReasonCode.NON_FINITE_METAL_COORDINATES,
    )
