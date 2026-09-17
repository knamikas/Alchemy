"""Decide which discovered candidates count as first-sphere contacts.

The donor rule says whether an atom may be inferred as a donor from geometry
alone; the distance rule compares the observed distance with the literature
target plus ``FIRST_SPHERE_TOLERANCE``. Declared connections may join the
contact set without passing the donor rule, but never with zero occupancy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import gemmi

from codes import (
    DonorRuleOverride,
    EligibilityReason,
    EligibilityStatus,
    ReferenceKind,
)
from coordination.candidates import deduplicate_special_position_contacts
from coordination.contact_record import Candidate, DonorPolicy, EligibilityResult
from coordination.donor_chemistry import (
    C_TERMINAL_DONOR_ATOMS,
    INFERRED_DONOR_ATOMS,
    N_TERMINAL_DONOR_ATOMS,
)
from coordination.policy import (
    CANDIDATE_SEARCH_RADIUS,
    FIRST_SPHERE_TOLERANCE,
    SEARCH_EPSILON,
)
from reference_data import first_sphere_targets, literature_distances
from structure_analysis import NAN, AtomSite, StructureContext


def bonding_key(neighbor: AtomSite, metal_element: str) -> tuple[str, str, str]:
    """Exact (residue, atom, metal) key matching metal_distances_info.txt columns."""
    name = neighbor.atom_name.strip()
    if neighbor.is_water:
        return ("HOH", "O", metal_element)
    if name in N_TERMINAL_DONOR_ATOMS:
        # No terminal-amine reference is bundled; use the element fallback for eligibility.
        return ("NTERM", "N", metal_element)
    if name in C_TERMINAL_DONOR_ATOMS:
        # No terminal-carboxylate reference is bundled; use the element fallback.
        return ("CTERM", "O", metal_element)
    if name == "O":
        return ("CA", "O", metal_element)  # backbone carbonyl O is keyed "CA"
    if name.startswith("O"):
        return (neighbor.residue_name, "O", metal_element)
    return (neighbor.residue_name, neighbor.element, metal_element)


def first_sphere_rule(
    metal: AtomSite, neighbor: AtomSite
) -> tuple[float, float, ReferenceKind, str]:
    """Return target, cutoff, and provenance for proximity eligibility."""
    exact_key = bonding_key(neighbor, metal.element)
    literature = literature_distances().get(exact_key)
    target: float | None
    if literature is not None:
        target = literature[0]
        reference_kind = ReferenceKind.EXACT
        reference_key = exact_key
    else:
        # The element fallback decides sphere membership only. An exact
        # reference stays mandatory for the z-score.
        target = first_sphere_targets().get((metal.element, neighbor.element))
        if target is None:
            return NAN, NAN, ReferenceKind.MISSING, ""
        reference_kind = ReferenceKind.ELEMENT_FALLBACK
        reference_key = ("*", neighbor.element, metal.element)
    cutoff = min(CANDIDATE_SEARCH_RADIUS, target + FIRST_SPHERE_TOLERANCE)
    return target, cutoff, reference_kind, ":".join(reference_key)


def _polymer_terminal_position(
    structure: StructureContext, atom: AtomSite
) -> tuple[bool, bool]:
    """Return ``(is_n_terminal, is_c_terminal)`` for a polymer residue."""
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


def annotate_donor_policy(
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
        candidate.set_donor_policy(
            DonorPolicy(
                inferred_allowed=allowed,
                rule=rule,
                override=(
                    DonorRuleOverride.DECLARED_CONNECTION
                    if declared and not allowed and supported
                    else ""
                ),
            )
        )


def _candidate_has_zero_occupancy(candidate: Candidate, metal: AtomSite) -> bool:
    """Whether either endpoint is explicitly modeled with zero occupancy."""
    return any(
        atom.occupancy_valid and atom.occupancy == 0.0
        for atom in (metal, candidate.neighbor)
    )


def _eligibility_for(candidate: Candidate, metal: AtomSite) -> EligibilityResult:
    """Classify one candidate against the first-sphere distance rule.

    A zero-occupancy endpoint is never eligible. Without a reference the
    candidate is unassignable when a typical or declared donor would have
    needed one, and simply non-typical otherwise. With a reference, distance
    decides sphere membership and the donor rule decides inference.
    """
    target, cutoff, reference_kind, reference_key = first_sphere_rule(
        metal, candidate.neighbor
    )
    donor_allowed = candidate.donor_policy().inferred_allowed
    declared = bool(candidate.declared_connections)
    first_sphere = False
    inferred = False
    if _candidate_has_zero_occupancy(candidate, metal):
        status = EligibilityStatus.ZERO_OCCUPANCY
        reason = EligibilityReason.ZERO_OCCUPANCY_ATOM
    elif not math.isfinite(cutoff):
        # ``first_sphere_rule`` reports a missing reference as NaN target and cutoff.
        if donor_allowed or declared:
            status = EligibilityStatus.MISSING_ASSIGNMENT_REFERENCE
            reason = EligibilityReason.NO_ASSIGNMENT_REFERENCE
        else:
            status = EligibilityStatus.NON_TYPICAL_DONOR
            reason = EligibilityReason.ATOM_NOT_TYPICAL_DONOR
    else:
        first_sphere = candidate.image.distance <= cutoff + SEARCH_EPSILON
        inferred = first_sphere and donor_allowed
        if inferred:
            status = EligibilityStatus.FIRST_SPHERE_ELIGIBLE
            reason = EligibilityReason.DISTANCE_WITHIN_TOLERANCE
        elif first_sphere:
            status = EligibilityStatus.NON_TYPICAL_DONOR
            reason = EligibilityReason.ATOM_NOT_TYPICAL_DONOR
        else:
            status = EligibilityStatus.OUTSIDE_FIRST_SPHERE
            reason = EligibilityReason.DISTANCE_EXCEEDS_TOLERANCE
    return EligibilityResult(
        status=status,
        reason=reason,
        first_sphere_eligible=first_sphere,
        inferred_contact_eligible=inferred,
        assignment_target=target,
        assignment_tolerance=FIRST_SPHERE_TOLERANCE,
        first_sphere_cutoff=cutoff,
        assignment_reference_kind=reference_kind,
        assignment_reference=reference_key,
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
        eligibility = _eligibility_for(candidate, metal)
        candidate.set_eligibility(eligibility)
        if eligibility.status == EligibilityStatus.MISSING_ASSIGNMENT_REFERENCE:
            unsupported_pairs.add((metal.element, candidate.neighbor.element))
        if eligibility.inferred_contact_eligible:
            eligible.append(candidate)
    return (deduplicate_special_position_contacts(eligible), unsupported_pairs)


def current_contacts_from_candidates(
    candidates: Sequence[Candidate], metal: AtomSite
) -> tuple[list[Candidate], set[tuple[str, str]]]:
    """Return typical inferred and explicitly declared contacts.

    Every candidate receives its eligibility result as a side effect; the
    returned pairs are the metal-donor elements no reference covers.
    """
    eligible, unsupported_pairs = _identify_first_sphere_candidates(candidates, metal)
    declared_not_inferred = [
        candidate
        for candidate in candidates
        if candidate.declared_connections
        and not candidate.eligibility().inferred_contact_eligible
        and candidate.donor_class_supported
        and not _candidate_has_zero_occupancy(candidate, metal)
    ]
    return (
        deduplicate_special_position_contacts(eligible + declared_not_inferred),
        unsupported_pairs,
    )
