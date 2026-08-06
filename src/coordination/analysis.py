"""Metal-ligand bond-distance analysis for one PDB entry.

Discovers explicit, crystallographic and strict-NCS candidates within 4 A of
every metal, supplements them with source ``struct_conn``/``LINK``
declarations, decides first-sphere eligibility in a separate stage, and scores
each contact against the literature reference distances in
``metal_distances_info.txt`` (Harding 2006, and Zheng et al. 2008 for Ni):

    z = (d_observed - mu) / sqrt(DPI**2 + sigma_lit**2)

Adding the DPI (Blow 2002 eq. 7) in quadrature with the literature spread makes
the same absolute deviation more significant in a high-resolution structure
than in a low-resolution one. Missing inputs produce NaN derived values without
discarding measured bond geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
from collections.abc import Iterable, Mapping, Sequence

from coordination.schema import (
    ZSCORE_OUTLIER_CUTOFF,
    bond_row,
    candidate_row,
    context_warning_values,
)
from codes import (
    CandidateSource,
    ContactScope,
    GeometryStatus,
    MultiDonorStatus,
    ReasonCode,
)
from coordination.contact_record import Candidate
from coordination.declared_connections import collect_declared_candidates
from coordination.donor_chemistry import (
    AA,
    C_TERMINAL_DONOR_ATOMS,
    DONOR_ELEMENTS,
    INFERRED_DONOR_ATOMS,
    N_TERMINAL_DONOR_ATOMS,
)
from coordination.dpi import calculate_dpi_details
from metal_elements import METAL_ELEMENTS
from metal_identification import sigma_for, sigma_index, zd_indices
from reference_data import (
    cluster_ids,
    first_sphere_targets,
    heme_ids,
    literature_distances,
)
from structure_analysis import (
    NAN,
    AtomSite,
    StructureContext,
    count_deposited_ni,
    count_ni,
    load_structure,
    pbc_translation,
    position_distance,
)

if TYPE_CHECKING:
    import gemmi


CANDIDATE_SEARCH_RADIUS = 4.0
SEARCH_EPSILON = 1e-6

# First-sphere definition: donor distance <= target distance + 0.75 A.
# Harding, M. M. (2004), Acta Cryst. D60, 849-859.
# https://doi.org/10.1107/S0907444904004081
FIRST_SPHERE_TOLERANCE = 0.75

# Gemmi's ContactSearch uses 0.8 A by default to distinguish near-coincident
# symmetry images of an atom intended to occupy a special position.
# NeighborSearch returns those images unfiltered, so apply the cutoff here.
SPECIAL_POSITION_DEDUP_CUTOFF = 0.8

#: One deposited atom record, as ``AtomSite.source_key`` reports it.
AtomKey = tuple[int, int, int, int]

#: What makes two candidate records the same atom image around one metal.
_CandidateIdentity = tuple[AtomKey, str, tuple[int, int, int], tuple[float, ...]]

#: What makes two contacts share one donor-residue image around one metal.
_ResidueImageKey = tuple[
    tuple[int, int, int], ContactScope, str, int, str, tuple[int, int, int]
]


def _bonding_key(
    neighbor: AtomSite, nb_res: str, metal_el: str
) -> tuple[str, str, str]:
    """Exact (residue, atom, metal) key matching metal_distances_info.txt columns."""
    name = neighbor.atom_name.strip()
    if neighbor.is_water:
        return ("HOH", "O", metal_el)
    if name in N_TERMINAL_DONOR_ATOMS:
        # No terminal-amine reference is bundled, so this key misses and falls
        # through to the element fallback rather than borrowing the histidine
        # side-chain nitrogen reference.
        return ("NTERM", "N", metal_el)
    if name in C_TERMINAL_DONOR_ATOMS:
        # No terminal-carboxylate reference is bundled, so this key misses and
        # falls through to the element fallback rather than borrowing a
        # side-chain reference.
        return ("CTERM", "O", metal_el)
    if name == "O":
        return ("CA", "O", metal_el)  # backbone carbonyl O is keyed "CA"
    if name.startswith("O"):
        return (nb_res, "O", metal_el)
    return (nb_res, neighbor.element, metal_el)


def _parent_type(
    structure: StructureContext, metal: AtomSite, metal_res: str, metal_el: str
) -> str:
    if metal_res in cluster_ids():
        return "cluster"
    if metal_res in heme_ids():
        return "heme"
    if metal_el not in METAL_ELEMENTS:
        return "other"  # unreachable while metal_atoms is pre-filtered
    residue = structure.residue_for_atom(metal)
    if residue.chemical_atom_site_count == 1:
        return "ion"
    return "other"


def zscore(dist: float, mu: float, stdev: float, dpi: float) -> float:
    """Bond-distance z-score, ``(dist - mu)/sqrt(stdev^2 + dpi^2)``.

    The denominator carries one DPI, not the ``sqrt(2) * DPI`` an
    independent-error treatment of two atoms would give: the metal is a heavy
    scatterer among the best-ordered atoms in the model, so the single DPI
    stands for the donor. Widening it to ``2 * dpi ** 2`` would shrink every
    z-score and change which contacts pass ZSCORE_OUTLIER_CUTOFF.
    """
    if not (math.isfinite(dpi) and math.isfinite(mu) and math.isfinite(stdev)):
        return NAN
    denom = math.sqrt(dpi**2 + stdev**2)
    return (dist - mu) / denom if denom > 0 else NAN


def _contact_sort_key(
    contact: Candidate,
) -> tuple[int, int, int, str, tuple[int, int, int], tuple[float, float, float]]:
    neighbor = contact.neighbor
    return (
        neighbor.chain_index,
        neighbor.residue_index,
        neighbor.atom_index,
        contact.symmetry_operation,
        contact.translation,
        contact.transformed_position,
    )


def _merge_candidate_provenance(target: Candidate, source: Candidate) -> None:
    """Add discovery and declaration provenance without duplicating records."""
    target.candidate_sources.update(source.candidate_sources)
    known_connections = {
        (record["source"], record["connection_id"])
        for record in target.declared_connections
    }
    for record in source.declared_connections:
        connection_key = (record["source"], record["connection_id"])
        if connection_key not in known_connections:
            target.declared_connections.append(record)
            known_connections.add(connection_key)


def _special_position_preference(
    contact: Candidate,
) -> tuple[bool, float, int, str, tuple[int, int, int], tuple[float, float, float]]:
    """Order near-coincident symmetry images so the retained one is stable.

    An explicit image sorts first, so an off-axis refinement artifact cannot
    turn an otherwise explicit contact into a symmetry-dependent one.
    """
    return (
        contact.symmetry_contact,
        contact.distance_raw,
        contact.symmetry_image_index,
        contact.symmetry_operation,
        contact.translation,
        contact.transformed_position,
    )


def deduplicate_special_position_contacts(
    candidates: Iterable[Candidate],
) -> list[Candidate]:
    """Collapse near-coincident images of each deposited source atom.

    Sorting each source-atom group before the spatial comparison makes the
    result independent of Gemmi's NeighborSearch mark order.
    """
    by_source: dict[AtomKey, list[Candidate]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate.neighbor.source_key, []).append(candidate)

    contacts: list[Candidate] = []
    for source_key in sorted(by_source):
        retained: list[Candidate] = []
        for candidate in sorted(
            by_source[source_key], key=_special_position_preference
        ):
            duplicate = next(
                (
                    current
                    for current in retained
                    if position_distance(
                        current.transformed_position,
                        candidate.transformed_position,
                    )
                    <= SPECIAL_POSITION_DEDUP_CUTOFF
                ),
                None,
            )
            if duplicate is not None:
                _merge_candidate_provenance(duplicate, candidate)
                continue
            retained.append(candidate)
        contacts.extend(retained)
    contacts.sort(key=_contact_sort_key)
    return contacts


def first_sphere_rule(
    metal: AtomSite, neighbor: AtomSite
) -> tuple[float, float, str, str]:
    """Return target, cutoff, and provenance for proximity eligibility."""
    exact_key = _bonding_key(neighbor, neighbor.residue_name, metal.element)
    literature = literature_distances().get(exact_key)
    target: float | None
    if literature is not None:
        target = literature[0]
        reference_kind = "exact"
        reference_key = exact_key
    else:
        # The element fallback decides sphere membership only. An exact
        # reference stays mandatory for the z-score.
        target = first_sphere_targets().get((metal.element, neighbor.element))
        if target is None:
            return NAN, NAN, "missing", ""
        reference_kind = "element_fallback"
        reference_key = ("*", neighbor.element, metal.element)
    cutoff = min(CANDIDATE_SEARCH_RADIUS, target + FIRST_SPHERE_TOLERANCE)
    return target, cutoff, reference_kind, ":".join(reference_key)


def _polymer_terminal_position(
    structure: StructureContext, atom: AtomSite
) -> tuple[bool, bool]:
    """Return ``(is_n_terminal, is_c_terminal)`` for a polymer residue."""
    import gemmi

    selected = structure.residue_for_atom(atom)
    if selected.source_polymer_position:
        position = selected.source_polymer_position
        return position in ("N", "NC"), position in ("C", "NC")
    try:
        chain = structure.model[atom.chain_index]
        residue = chain[atom.residue_index]
        if residue.entity_type != gemmi.EntityType.Polymer:
            return False, False
        indices = [
            index
            for index, current in enumerate(chain)
            if (
                current.entity_type == gemmi.EntityType.Polymer
                and current.subchain == residue.subchain
            )
        ]
        if not indices:
            return False, False
        full_sequence = next(
            (
                tuple(str(name) for name in entity.full_sequence)
                for entity in structure.structure.entities
                if str(residue.subchain) in {str(value) for value in entity.subchains}
                and entity.full_sequence
            ),
            (),
        )
        modeled_sequence = tuple(str(chain[index].name) for index in indices)
        if not full_sequence or modeled_sequence != full_sequence:
            return False, False
        return atom.residue_index == indices[0], atom.residue_index == indices[-1]
    except (AttributeError, IndexError, TypeError):
        return False, False


def _inferred_donor_rule(
    structure: StructureContext, atom: AtomSite
) -> tuple[bool, str]:
    """Return whether ``atom`` is a typical geometry-inferable donor and why."""
    residue = structure.residue_for_atom(atom)
    atom_name = atom.atom_name.strip().upper()
    residue_name = residue.residue_name.upper()
    if residue.is_water:
        if atom.element == "O":
            return True, "water_oxygen"
        return False, "outside_typical_donor_list"
    if residue_name not in INFERRED_DONOR_ATOMS:
        return False, "outside_typical_donor_list"
    if atom_name in INFERRED_DONOR_ATOMS[residue_name]:
        return (
            True,
            "backbone_carbonyl_oxygen"
            if atom_name == "O"
            else "typical_sidechain_donor",
        )

    is_n_terminal, is_c_terminal = _polymer_terminal_position(structure, atom)
    if is_n_terminal and atom_name in N_TERMINAL_DONOR_ATOMS:
        return True, "n_terminal_nitrogen"
    if is_c_terminal and atom_name in C_TERMINAL_DONOR_ATOMS:
        return True, "c_terminal_oxygen"
    return False, "outside_typical_donor_list"


def _annotate_donor_policy(
    structure: StructureContext, candidates: Iterable[Candidate]
) -> None:
    """Annotate candidates with inference permission and declaration override."""
    for candidate in candidates:
        allowed, rule = _inferred_donor_rule(structure, candidate.neighbor)
        declared = bool(candidate.declared_connections)
        # A declaration overrides the donor-atom rule only inside a residue
        # class Alchemy can assess: claiming the override elsewhere would label
        # a row as a declared bond that never becomes one.
        supported = candidate.donor_class_supported
        candidate.inferred_donor_allowed = allowed
        candidate.inferred_donor_rule = rule
        candidate.donor_rule_override = (
            "declared_connection" if declared and not allowed and supported else ""
        )


def _candidate_has_zero_occupancy(candidate: Candidate, metal: AtomSite) -> bool:
    """Whether either endpoint is explicitly modeled with zero occupancy."""
    return any(
        atom.occupancy_valid and atom.occupancy == 0.0
        for atom in (metal, candidate.neighbor)
    )


def _identify_first_sphere_candidates(
    candidates: Iterable[Candidate], metal: AtomSite
) -> tuple[list[Candidate], set[tuple[str, str]]]:
    """Annotate every candidate with eligibility, and return those eligible.

    Also returns the metal-donor pairs no reference covers, which the caller
    reports as incomplete assignment evidence. Passing the distance rule does
    not by itself establish a chemically assigned bond.
    """
    eligible: list[Candidate] = []
    unsupported_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        neighbor = candidate.neighbor
        donor_allowed = candidate.inferred_donor_allowed
        target, cutoff, reference_kind, reference_key = first_sphere_rule(
            metal, neighbor
        )
        if _candidate_has_zero_occupancy(candidate, metal):
            candidate.eligibility_status = "zero_occupancy"
            candidate.eligibility_reason = "zero_occupancy_atom_is_not_contact_evidence"
            candidate.first_sphere_eligible = False
            candidate.inferred_contact_eligible = False
            candidate.assignment_target = target
            candidate.assignment_tolerance = FIRST_SPHERE_TOLERANCE
            candidate.first_sphere_cutoff = cutoff
            candidate.assignment_reference_kind = reference_kind
            candidate.assignment_reference = reference_key
            continue
        if not math.isfinite(cutoff):
            declared = bool(candidate.declared_connections)
            if donor_allowed or declared:
                unsupported_pairs.add((metal.element, neighbor.element))
            candidate.eligibility_status = (
                "missing_assignment_reference"
                if donor_allowed or declared
                else "non_typical_donor"
            )
            candidate.eligibility_reason = (
                "no_metal_donor_assignment_reference"
                if donor_allowed or declared
                else "atom_not_in_typical_inferred_donor_list"
            )
            candidate.first_sphere_eligible = False
            candidate.inferred_contact_eligible = False
            candidate.assignment_target = NAN
            candidate.assignment_tolerance = FIRST_SPHERE_TOLERANCE
            candidate.first_sphere_cutoff = NAN
            candidate.assignment_reference_kind = reference_kind
            candidate.assignment_reference = reference_key
            continue
        is_eligible = candidate.distance_raw <= cutoff + SEARCH_EPSILON
        inferred_eligible = is_eligible and donor_allowed
        candidate.eligibility_status = (
            "first_sphere_eligible"
            if inferred_eligible
            else ("non_typical_donor" if is_eligible else "outside_first_sphere")
        )
        candidate.eligibility_reason = (
            "distance_within_target_plus_0.75"
            if inferred_eligible
            else (
                "atom_not_in_typical_inferred_donor_list"
                if is_eligible
                else "distance_exceeds_target_plus_0.75"
            )
        )
        candidate.first_sphere_eligible = is_eligible
        candidate.inferred_contact_eligible = inferred_eligible
        candidate.assignment_target = target
        candidate.assignment_tolerance = FIRST_SPHERE_TOLERANCE
        candidate.first_sphere_cutoff = cutoff
        candidate.assignment_reference_kind = reference_kind
        candidate.assignment_reference = reference_key
        if inferred_eligible:
            eligible.append(candidate)
    return (deduplicate_special_position_contacts(eligible), unsupported_pairs)


def collect_proximal_candidates(
    structure: StructureContext,
    search: gemmi.NeighborSearch,
    metal: AtomSite,
    include_symmetry: bool,
) -> list[Candidate]:
    """Return donor-like candidates within 4 A for one search scope.

    Discovery only: no literature target, no first-sphere tolerance, no
    assignment. That happens in ``_identify_first_sphere_candidates``.
    """
    if not metal.coordinates_valid:
        return []
    candidates: list[Candidate] = []
    marks = search.find_atoms(
        metal.pos, "\x00", min_dist=0.0, radius=CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON
    )
    for mark in marks:
        neighbor = structure.atom_for_mark(mark)
        if neighbor is None:
            continue
        if not neighbor.coordinates_valid:
            continue
        # Cheapest test first: the residue lookup below only runs for the few
        # marks that can still qualify.
        if neighbor.element not in DONOR_ELEMENTS:
            continue
        if not (neighbor.occupancy_valid and neighbor.occupancy > 0.0):
            continue
        residue = structure.residue_for_atom(neighbor)
        if not (residue.is_water or residue.residue_name in AA):
            continue

        if include_symmetry:
            nearest = structure.structure.cell.find_nearest_pbc_image(
                metal.pos, neighbor.pos, mark.image_idx
            )
            transformed = structure.structure.cell.find_nearest_pbc_position(
                metal.pos, neighbor.pos, mark.image_idx
            )
            translation = pbc_translation(nearest)
            image_index = int(nearest.sym_idx)
            (
                crystallographic_contact,
                strict_ncs_contact,
                strict_ncs_operation_id,
                contact_scope,
            ) = structure.image_provenance(image_index, translation)
            symmetry_contact = crystallographic_contact or strict_ncs_contact
            operation = nearest.symmetry_code()
            distance = float(nearest.dist())
        else:
            transformed = neighbor.pos
            translation = (0, 0, 0)
            image_index = 0
            symmetry_contact = False
            crystallographic_contact = False
            strict_ncs_contact = False
            strict_ncs_operation_id = ""
            contact_scope = ContactScope.EXPLICIT
            operation = "1_555"
            distance = position_distance(metal.xyz, neighbor.xyz)

        # A symmetry copy is a distinct residue image, so only the source
        # residue in the explicit asymmetric unit is excluded.
        if neighbor.residue_key == metal.residue_key and not symmetry_contact:
            continue
        if not (0.0 < distance <= CANDIDATE_SEARCH_RADIUS + 1e-9):
            continue

        position = (float(transformed.x), float(transformed.y), float(transformed.z))
        candidates.append(
            Candidate(
                neighbor=neighbor,
                distance_raw=distance,
                transformed_position=position,
                symmetry_contact=symmetry_contact,
                crystallographic_contact=crystallographic_contact,
                strict_ncs_contact=strict_ncs_contact,
                strict_ncs_operation_id=strict_ncs_operation_id,
                # ``image_provenance`` widens the scope it returns to str;
                # every value it can produce is a ContactScope member.
                contact_scope=cast(ContactScope, contact_scope),
                symmetry_image_index=image_index,
                symmetry_operation=operation,
                translation=translation,
                candidate_sources={CandidateSource.PROXIMITY_4A},
            )
        )

    candidates.sort(key=_contact_sort_key)
    return candidates


def _candidate_identity(candidate: Candidate) -> _CandidateIdentity:
    return (
        candidate.neighbor.source_key,
        candidate.symmetry_operation,
        candidate.translation,
        tuple(round(value, 5) for value in candidate.transformed_position),
    )


def _merge_candidates(*candidate_groups: Iterable[Candidate]) -> list[Candidate]:
    """Merge proximity and declaration provenance for the same atom image."""
    merged: dict[_CandidateIdentity, Candidate] = {}
    for candidates in candidate_groups:
        for candidate in candidates:
            key = _candidate_identity(candidate)
            existing = merged.get(key)
            if existing is None:
                # The provenance collections are copied too: merging appends to
                # them, and the caller's candidate must not gain the other
                # group's sources.
                merged[key] = replace(
                    candidate,
                    candidate_sources=set(candidate.candidate_sources),
                    declared_connections=list(candidate.declared_connections),
                )
                continue
            _merge_candidate_provenance(existing, candidate)
    result = list(merged.values())
    result.sort(key=_contact_sort_key)
    return result


def _current_contacts_from_candidates(
    candidates: Sequence[Candidate], metal: AtomSite
) -> tuple[list[Candidate], set[tuple[str, str]]]:
    """Return typical inferred and explicitly declared contacts."""
    eligible, unsupported_pairs = _identify_first_sphere_candidates(candidates, metal)
    declared_not_inferred = [
        candidate
        for candidate in candidates
        if candidate.declared_connections
        and not candidate.inferred_contact_eligible
        and candidate.donor_class_supported
        and not _candidate_has_zero_occupancy(candidate, metal)
    ]
    return (
        deduplicate_special_position_contacts(eligible + declared_not_inferred),
        unsupported_pairs,
    )


def annotate_contacts(
    contacts: Iterable[Candidate], metal_element: str, dpi: float
) -> None:
    for contact in contacts:
        neighbor = contact.neighbor
        reported_distance = round(contact.distance_raw, 3)
        literature = literature_distances().get(
            _bonding_key(neighbor, neighbor.residue_name, metal_element)
        )
        if literature is None:
            mu = stdev = zscore_raw = NAN
        else:
            mu, stdev = literature
            zscore_raw = zscore(contact.distance_raw, mu, stdev, dpi)
        rounded_zscore = round(zscore_raw, 4) if math.isfinite(zscore_raw) else NAN
        contact.distance = reported_distance
        contact.literature_distance = mu
        contact.literature_stdev = stdev
        contact.zscore = rounded_zscore
        contact.reference_covered = literature is not None
        if math.isfinite(zscore_raw):
            magnitude = abs(zscore_raw)
            outlier = magnitude >= ZSCORE_OUTLIER_CUTOFF or math.isclose(
                magnitude,
                ZSCORE_OUTLIER_CUTOFF,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            contact.geometry_outlier = outlier
            contact.geometry_consistent = not outlier
        else:
            contact.geometry_outlier = ""
            contact.geometry_consistent = ""


def _residue_image_key(contact: Candidate) -> _ResidueImageKey:
    """Identity of one donor residue image around the current metal site."""
    return (
        contact.neighbor.residue_key,
        contact.contact_scope,
        contact.strict_ncs_operation_id,
        contact.symmetry_image_index,
        contact.symmetry_operation,
        contact.translation,
    )


def _annotate_multi_donor_groups(contacts: Iterable[Candidate]) -> None:
    """Annotate assigned contacts that share one donor-residue image.

    Group status is contextual: a suspect member marks every member as
    belonging to a suspect group without weakening any individual result.
    """
    groups: dict[_ResidueImageKey, list[Candidate]] = {}
    for contact in contacts:
        groups.setdefault(_residue_image_key(contact), []).append(contact)

    for group in groups.values():
        count = len(group)
        multi_donor = count >= 2
        if not multi_donor:
            contact = group[0]
            assessable = (
                contact.geometry_outlier is True or contact.geometry_consistent is True
            )
            contact.multi_donor_detected = False
            contact.multi_donor_contact_count = 1
            contact.multi_donor_geometry_status = MultiDonorStatus.SINGLE_DONOR
            contact.multi_donor_contains_suspect_bond = False
            contact.score_eligible = assessable
            contact.score_exclusion_reason = "" if assessable else "zscore_unavailable"
            continue

        any_outlier = any(contact.geometry_outlier is True for contact in group)
        all_consistent = all(contact.geometry_consistent is True for contact in group)
        if all_consistent:
            status = MultiDonorStatus.CONSISTENT
        elif any_outlier:
            status = MultiDonorStatus.SUSPECT
        else:
            status = MultiDonorStatus.INDETERMINATE
        for contact in group:
            assessable = (
                contact.geometry_outlier is True or contact.geometry_consistent is True
            )
            contact.multi_donor_detected = True
            contact.multi_donor_contact_count = count
            contact.multi_donor_geometry_status = status
            contact.multi_donor_contains_suspect_bond = any_outlier
            contact.score_eligible = assessable
            contact.score_exclusion_reason = "" if assessable else "zscore_unavailable"


def _scope_summary(
    contacts: Sequence[Candidate], unavailable: bool = False
) -> dict[str, Any]:
    if unavailable:
        return {
            "candidate": NAN,
            "covered": NAN,
            "outlier": NAN,
            "consistent": NAN,
            "score_eligible": NAN,
            "score_excluded": NAN,
            "scored_outlier": NAN,
            "scored_consistent": NAN,
            "multi_donor_groups": NAN,
            "multi_donor_contacts": NAN,
            "suspect_multi_donor_groups": NAN,
            "indeterminate_multi_donor_groups": NAN,
            "coverage": NAN,
            "status": "",
        }
    candidate = len(contacts)
    covered = sum(bool(contact.reference_covered) for contact in contacts)
    outlier = sum(contact.geometry_outlier is True for contact in contacts)
    consistent = sum(contact.geometry_consistent is True for contact in contacts)
    score_eligible = sum(contact.score_eligible is True for contact in contacts)
    score_excluded = candidate - score_eligible
    scored_outlier = sum(
        contact.score_eligible is True and contact.geometry_outlier is True
        for contact in contacts
    )
    scored_consistent = sum(
        contact.score_eligible is True and contact.geometry_consistent is True
        for contact in contacts
    )
    multi_donor_groups = {
        _residue_image_key(contact)
        for contact in contacts
        if contact.multi_donor_detected
    }
    suspect_multi_donor_groups = {
        _residue_image_key(contact)
        for contact in contacts
        if contact.multi_donor_geometry_status == MultiDonorStatus.SUSPECT
    }
    indeterminate_multi_donor_groups = {
        _residue_image_key(contact)
        for contact in contacts
        if contact.multi_donor_geometry_status == MultiDonorStatus.INDETERMINATE
    }
    multi_donor_contacts = sum(
        contact.multi_donor_detected is True for contact in contacts
    )
    scored_assessable = scored_outlier + scored_consistent
    if scored_assessable == 0:
        status = GeometryStatus.INSUFFICIENT_DATA
    elif scored_outlier:
        status = GeometryStatus.SUSPECT
    else:
        status = GeometryStatus.PLAUSIBLE
    coverage = round(covered / candidate, 4) if candidate else NAN
    return {
        "candidate": candidate,
        "covered": covered,
        "outlier": outlier,
        "consistent": consistent,
        "score_eligible": score_eligible,
        "score_excluded": score_excluded,
        "scored_outlier": scored_outlier,
        "scored_consistent": scored_consistent,
        "multi_donor_groups": len(multi_donor_groups),
        "multi_donor_contacts": multi_donor_contacts,
        "suspect_multi_donor_groups": len(suspect_multi_donor_groups),
        "indeterminate_multi_donor_groups": len(indeterminate_multi_donor_groups),
        "coverage": coverage,
        "status": status,
    }


def _site_context_values(
    contacts: Sequence[Candidate], candidates: Sequence[Candidate]
) -> dict[str, bool | int | str]:
    """Aggregate coordination-relevant context without changing confidence."""
    reasons: list[str] = []
    for contact in contacts:
        values = context_warning_values(contact)
        if values["context_warning_reasons"]:
            reasons.extend(values["context_warning_reasons"].split("|"))
    non_typical_first_sphere = [
        candidate
        for candidate in candidates
        if (not candidate.inferred_donor_allowed and candidate.first_sphere_eligible)
    ]
    if non_typical_first_sphere:
        reasons.append("non_typical_first_sphere_candidate")
    reasons = list(dict.fromkeys(reasons))
    return {
        "context_warning": bool(reasons),
        "context_warning_reasons": "|".join(reasons),
        "non_typical_first_sphere_candidate_count": len(non_typical_first_sphere),
        "declared_donor_override_contact_count": sum(
            contact.donor_rule_override == "declared_connection" for contact in contacts
        ),
    }


# Generated-contact scope and its two dependency flags, keyed by whether the
# site has any crystallographic and any strict-NCS contact. Read only once a
# site has generated contacts, so a contact that is neither counts as NCS.
_GENERATED_SCOPES = {
    (True, True): ("strict_ncs_and_crystallographic", True, True),
    (True, False): ("crystallographic", True, False),
    (False, True): ("strict_ncs", False, True),
    (False, False): ("strict_ncs", False, True),
}


def _site_summary(
    metal: AtomSite,
    explicit_contacts: Sequence[Candidate],
    image_contacts: Sequence[Candidate] | None,
    dpi: float,
    resolution: float,
    ni: float,
    deposited_ni: float,
    dpi_reason: str,
    structure: StructureContext,
) -> dict[str, Any]:
    explicit = _scope_summary(explicit_contacts)
    image_search_available = image_contacts is not None
    image_inclusive = _scope_summary(
        image_contacts or [],
        unavailable=not image_search_available,
    )
    primary = image_inclusive if image_search_available else explicit
    # Whether the metal itself carries overfull alternate occupancy, rather than
    # a residue elsewhere that the entry-level warning reports identically. Only
    # the metal is tracked: it is the object of study, and a donor's presence is
    # already answered empirically by its real-space density statistics.
    metal_overfull = (
        metal.chemical_site_identity in structure.overfull_occupancy_site_keys
    )
    symmetry_count = (
        sum(contact.symmetry_contact for contact in image_contacts)
        if image_contacts is not None
        else NAN
    )
    crystallographic_count = (
        sum(contact.crystallographic_contact for contact in image_contacts)
        if image_contacts is not None
        else NAN
    )
    strict_ncs_count = (
        sum(contact.strict_ncs_contact for contact in image_contacts)
        if image_contacts is not None
        else NAN
    )
    combined_count = (
        sum(
            contact.crystallographic_contact and contact.strict_ncs_contact
            for contact in image_contacts
        )
        if image_contacts is not None
        else NAN
    )
    # Blank where symmetry was never searched, boolean where it was: "not
    # assessed" and "assessed false" are different answers, and the columns
    # keep them apart.
    changed: str | bool
    depends_crystallographic: str | bool
    depends_strict_ncs: str | bool
    if not image_search_available:
        generated_scope = ""
        changed = ""
        depends_crystallographic = ""
        depends_strict_ncs = ""
    else:
        changed = explicit["status"] != image_inclusive["status"]
        generated_scope, depends_crystallographic, depends_strict_ncs = (
            _GENERATED_SCOPES[
                bool(crystallographic_count),
                bool(strict_ncs_count),
            ]
            if symmetry_count
            else ("none", False, False)
        )

    reasons: list[str] = []
    if dpi_reason:
        reasons.append(dpi_reason)
    if not image_search_available:
        reasons.append(ReasonCode.SYMMETRY_SEARCH_UNAVAILABLE)
    if primary["scored_consistent"] + primary["scored_outlier"] == 0:
        reasons.append("no_assessable_reference_contacts")
    return {
        "dpi": dpi,
        "resolution": resolution,
        "occupancy_weighted_atom_count": (round(ni, 6) if math.isfinite(ni) else NAN),
        "deposited_occupancy_weighted_atom_count": (
            round(deposited_ni, 6) if math.isfinite(deposited_ni) else NAN
        ),
        "dpi_atom_count_multiplier": structure.dpi_atom_count_multiplier,
        "strict_ncs_operation_count": structure.strict_ncs_operation_count,
        "crystallographic_operation_count": (
            structure.crystallographic_operation_count
        ),
        "dpi_unavailable_reason": dpi_reason,
        "candidate_contact_count": primary["candidate"],
        "reference_covered_contact_count": primary["covered"],
        "geometry_outlier_contact_count": primary["outlier"],
        "geometry_consistent_contact_count": primary["consistent"],
        "score_eligible_contact_count": primary["score_eligible"],
        "score_excluded_contact_count": primary["score_excluded"],
        "scored_geometry_outlier_contact_count": primary["scored_outlier"],
        "scored_geometry_consistent_contact_count": (primary["scored_consistent"]),
        "multi_donor_residue_group_count": primary["multi_donor_groups"],
        "multi_donor_contact_count": primary["multi_donor_contacts"],
        "suspect_multi_donor_residue_group_count": (
            primary["suspect_multi_donor_groups"]
        ),
        "indeterminate_multi_donor_residue_group_count": (
            primary["indeterminate_multi_donor_groups"]
        ),
        "explicit_contact_count": explicit["candidate"],
        "symmetry_contact_count": symmetry_count,
        "image_inclusive_contact_count": image_inclusive["candidate"],
        "crystallographic_contact_count": crystallographic_count,
        "strict_ncs_contact_count": strict_ncs_count,
        "combined_ncs_crystallographic_contact_count": combined_count,
        "geometry_outlier_count_explicit": explicit["outlier"],
        "geometry_outlier_count_image_inclusive": image_inclusive["outlier"],
        "geometry_coverage_explicit": explicit["coverage"],
        "geometry_coverage_image_inclusive": image_inclusive["coverage"],
        "explicit_geometry_status": explicit["status"],
        "image_inclusive_geometry_status": image_inclusive["status"],
        "generated_contact_scope": generated_scope,
        "geometry_classification_changes_with_generated_images": changed,
        "coordination_depends_on_crystallographic_symmetry": (depends_crystallographic),
        "coordination_depends_on_strict_ncs": depends_strict_ncs,
        "metal_overfull_occupancy": metal_overfull,
        "geometry_not_assessed_reason": "|".join(dict.fromkeys(reasons)),
    }


@dataclass(frozen=True)
class _MetalAnalysisResult:
    bond_rows: list[dict[str, Any]]
    candidate_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    unsupported_pairs: set[tuple[str, str]]


def _analyze_metal_site(
    pdb_id: str,
    structure: StructureContext,
    metal: AtomSite,
    metal_declarations: Sequence[Candidate],
    explicit_search: gemmi.NeighborSearch,
    image_search: gemmi.NeighborSearch | None,
    dpi: float,
    resolution: float,
    ni: float,
    deposited_ni: float,
    dpi_reason: str,
    sig: Mapping[str, Mapping[tuple[Any, ...], Sequence[str]]],
    zd_idx: Sequence[int] | None,
) -> _MetalAnalysisResult:
    explicit_declarations = [
        candidate for candidate in metal_declarations if not candidate.symmetry_contact
    ]
    explicit_candidates = _merge_candidates(
        collect_proximal_candidates(structure, explicit_search, metal, False),
        explicit_declarations,
    )
    _annotate_donor_policy(structure, explicit_candidates)
    explicit_contacts, unsupported_pairs = _current_contacts_from_candidates(
        explicit_candidates, metal
    )
    annotate_contacts(explicit_contacts, metal.element, dpi)
    _annotate_multi_donor_groups(explicit_contacts)

    image_candidates: list[Candidate] | None = None
    image_contacts: list[Candidate] | None = None
    if image_search is not None:
        image_candidates = _merge_candidates(
            collect_proximal_candidates(structure, image_search, metal, True),
            metal_declarations,
        )
        _annotate_donor_policy(structure, image_candidates)
        image_contacts, image_unsupported = _current_contacts_from_candidates(
            image_candidates, metal
        )
        unsupported_pairs.update(image_unsupported)
        annotate_contacts(image_contacts, metal.element, dpi)
        _annotate_multi_donor_groups(image_contacts)

    primary_contacts = (
        image_contacts if image_contacts is not None else explicit_contacts
    )
    primary_candidates = (
        image_candidates if image_candidates is not None else explicit_candidates
    )
    summary = _site_summary(
        metal,
        explicit_contacts,
        image_contacts,
        dpi,
        resolution,
        ni,
        deposited_ni,
        dpi_reason,
        structure,
    )
    summary.update(_site_context_values(primary_contacts, primary_candidates))

    sigma = sigma_for(
        sig,
        metal.residue_name,
        metal.chain_id,
        metal.resnum,
        zd_idx,
        site_key=metal.source_key,
    )
    parent_type = _parent_type(structure, metal, metal.residue_name, metal.element)
    return _MetalAnalysisResult(
        bond_rows=[
            bond_row(
                pdb_id,
                structure,
                metal,
                contact,
                dpi,
                resolution,
                sigma,
                parent_type,
            )
            for contact in primary_contacts
        ],
        candidate_rows=[
            candidate_row(pdb_id, structure, metal, candidate)
            for candidate in primary_candidates
        ],
        summary=summary,
        unsupported_pairs=unsupported_pairs,
    )


def run_bond_analysis(
    pdb_id: str,
    pdb_path: str,
    stats_rows: Sequence[Mapping[str, Any]],
    header: Sequence[str] | None,
    dpi_inputs: Mapping[str, Any],
    structure: StructureContext | None = None,
    connection_path: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[AtomKey, dict[str, Any]],
    dict[str, Any],
]:
    """Return contact rows, candidate rows, site summaries, and metadata.

    Bond rows come from first-sphere-eligible candidates and from source
    ``struct_conn``/``LINK`` declarations, which are evaluated even outside the
    4 A discovery radius. Image-inclusive results are primary wherever symmetry
    metadata is available.
    """
    if structure is None:
        structure = load_structure(pdb_id, pdb_path)

    metals_in_model = structure.metal_atoms(METAL_ELEMENTS, canonical=True)
    spatial_metals = [metal for metal in metals_in_model if metal.coordinates_valid]
    non_finite_metals = [
        metal for metal in metals_in_model if not metal.coordinates_valid
    ]
    metadata: dict[str, Any] = {
        "partial_reason_codes": [],
        "warning_codes": list(structure.warning_codes),
        "messages": [],
        "retryable": False,
    }
    if not metals_in_model:
        return [], [], {}, metadata

    (declared_candidates, declared_issues, declared_warnings) = (
        collect_declared_candidates(
            structure, connection_path or pdb_path, spatial_metals
        )
    )
    if declared_issues:
        metadata["partial_reason_codes"].append(
            ReasonCode.DECLARED_CONNECTION_RESOLUTION_INCOMPLETE
        )
        metadata["messages"].extend(declared_issues)
    metadata["warning_codes"].extend(declared_warnings)
    if non_finite_metals:
        metadata["partial_reason_codes"].append(ReasonCode.NON_FINITE_METAL_COORDINATES)
        metadata["messages"].append(
            "geometry unavailable for selected metal site(s) with non-finite "
            "coordinates: "
            + ", ".join(
                f"{metal.residue_name}/{metal.chain_id or '_'}/{metal.resnum}/"
                f"{metal.atom_name}"
                for metal in non_finite_metals
            )
        )
    declared_by_metal: dict[AtomKey, list[Candidate]] = {}
    for candidate in declared_candidates:
        # Every declaration-derived candidate carries the metal it was resolved
        # against; only proximity discovery leaves the field unset.
        declared_by_metal.setdefault(
            cast(AtomSite, candidate.metal).source_key, []
        ).append(candidate)

    dpi, resolution, dpi_reason = calculate_dpi_details(structure, dpi_inputs)
    ni = count_ni(structure)
    deposited_ni = count_deposited_ni(structure)
    if dpi_reason:
        metadata["partial_reason_codes"].append(dpi_reason)
        metadata["messages"].append(f"DPI unavailable: {dpi_reason}")
    if not structure.symmetry_search_available:
        metadata["partial_reason_codes"].append(ReasonCode.SYMMETRY_SEARCH_UNAVAILABLE)
        metadata["messages"].append(
            "symmetry search unavailable: "
            + (structure.symmetry_search_failure_reason or "unknown reason")
        )

    explicit_search = structure.make_neighbor_search(
        CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON,
        include_symmetry=False,
        positive_occupancy_only=True,
    )
    image_search = None
    if structure.symmetry_search_available:
        image_search = structure.make_neighbor_search(
            CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON,
            include_symmetry=True,
            positive_occupancy_only=True,
        )
    sig = sigma_index(stats_rows)
    zd_idx = zd_indices(header)

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    summaries: dict[AtomKey, dict[str, Any]] = {}
    for metal in metals_in_model:
        if not metal.coordinates_valid:
            summary = _site_summary(
                metal,
                [],
                [] if structure.symmetry_search_available else None,
                dpi,
                resolution,
                ni,
                deposited_ni,
                dpi_reason,
                structure,
            )
            summary["geometry_not_assessed_reason"] = (
                ReasonCode.NON_FINITE_METAL_COORDINATES
            )
            summary.update(_site_context_values([], []))
            summaries[metal.source_key] = summary
            continue
        site_result = _analyze_metal_site(
            pdb_id,
            structure,
            metal,
            declared_by_metal.get(metal.source_key, ()),
            explicit_search,
            image_search,
            dpi,
            resolution,
            ni,
            deposited_ni,
            dpi_reason,
            sig,
            zd_idx,
        )
        if site_result.unsupported_pairs:
            metadata["partial_reason_codes"].append(
                ReasonCode.MISSING_FIRST_SPHERE_REFERENCE
            )
            pairs = ", ".join(
                f"{metal_element}-{donor_element}"
                for metal_element, donor_element in sorted(
                    site_result.unsupported_pairs
                )
            )
            metadata["messages"].append(
                f"first-sphere reference unavailable for {pairs}"
            )
        summaries[metal.source_key] = site_result.summary
        rows.extend(site_result.bond_rows)
        candidate_rows.extend(site_result.candidate_rows)

    metadata["partial_reason_codes"] = list(
        dict.fromkeys(metadata["partial_reason_codes"])
    )
    metadata["warning_codes"] = list(dict.fromkeys(metadata["warning_codes"]))
    metadata["messages"] = list(dict.fromkeys(metadata["messages"]))
    return rows, candidate_rows, summaries, metadata
