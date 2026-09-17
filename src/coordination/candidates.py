"""Discover donor-like atom images around a metal and merge their provenance.

Discovery only: every candidate within the 4 A search radius is kept, with
near-coincident special-position images collapsed. Eligibility and geometry
are decided later by ``coordination.eligibility`` and ``coordination.geometry``.
"""

from __future__ import annotations

from collections.abc import Iterable

import gemmi

from codes import CandidateSource
from coordination.contact_record import Candidate
from coordination.donor_chemistry import AA, DONOR_ELEMENTS
from coordination.policy import (
    CANDIDATE_ACCEPT_EPSILON,
    CANDIDATE_SEARCH_RADIUS,
    SAME_IMAGE_TOLERANCE,
    SEARCH_EPSILON,
    SPECIAL_POSITION_DEDUP_CUTOFF,
)
from structure_analysis import (
    AtomKey,
    AtomSite,
    ContactImage,
    StructureContext,
    position_distance,
)

#: What makes two candidate records the same atom image around one metal.
#: The position is compared separately, with ``SAME_IMAGE_TOLERANCE``.
_CandidateIdentity = tuple[AtomKey, str, tuple[int, int, int]]


def _contact_sort_key(
    contact: Candidate,
) -> tuple[int, int, int, str, tuple[int, int, int], tuple[float, float, float]]:
    """Deposited-order key that makes every candidate list deterministic."""
    neighbor = contact.neighbor
    return (
        neighbor.chain_index,
        neighbor.residue_index,
        neighbor.atom_index,
        contact.image.symmetry_operation,
        contact.image.translation,
        contact.image.position,
    )


def _special_position_preference(
    contact: Candidate,
) -> tuple[bool, float, int, str, tuple[int, int, int], tuple[float, float, float]]:
    """Order near-coincident symmetry images so the retained one is stable.

    An explicit image sorts first, so an off-axis refinement artifact cannot
    turn an otherwise explicit contact into a symmetry-dependent one.
    """
    return (
        contact.image.symmetry_contact,
        contact.image.distance,
        contact.image.image_index,
        contact.image.symmetry_operation,
        contact.image.translation,
        contact.image.position,
    )


def deduplicate_special_position_contacts(
    candidates: Iterable[Candidate],
) -> list[Candidate]:
    """Collapse near-coincident images of each deposited source atom.

    Merges in place: the returned records are the caller's own ``Candidate``
    objects, and a collapsed image adds its provenance to the retained one by
    mutating it. That is deliberate, and the opposite of ``merge_candidates``,
    which copies: this function runs after the eligibility stage has annotated
    each candidate, and a copy would drop those annotations. The merge goes
    through ``Candidate.merge_provenance_after_annotation``, which carries the
    argument for why a merge this late cannot change a recorded verdict.

    Sorting each source-atom group before the spatial comparison makes the
    result independent of Gemmi's NeighborSearch mark order.
    """
    by_source: dict[AtomKey, list[Candidate]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate.neighbor.source_key, []).append(candidate)

    contacts: list[Candidate] = []
    for group in by_source.values():
        retained: list[Candidate] = []
        for candidate in sorted(group, key=_special_position_preference):
            duplicate = next(
                (
                    current
                    for current in retained
                    if position_distance(
                        current.image.position,
                        candidate.image.position,
                    )
                    <= SPECIAL_POSITION_DEDUP_CUTOFF
                ),
                None,
            )
            if duplicate is not None:
                duplicate.merge_provenance_after_annotation(candidate)
                continue
            retained.append(candidate)
        contacts.extend(retained)
    contacts.sort(key=_contact_sort_key)
    return contacts


def collect_proximal_candidates(
    structure: StructureContext,
    search: gemmi.NeighborSearch,
    metal: AtomSite,
    include_symmetry: bool,
) -> list[Candidate]:
    """Return donor-like candidates within 4 A for one search scope.

    Discovery only: no literature target, no first-sphere tolerance, no
    assignment. That happens in ``coordination.eligibility``.

    ``search`` must come from ``StructureContext.make_neighbor_search``, which
    already excludes non-finite coordinates and, with
    ``positive_occupancy_only``, zero-occupancy atoms.
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
        if neighbor.element not in DONOR_ELEMENTS:
            continue
        residue = structure.residue_for_atom(neighbor)
        if not (residue.is_water or residue.residue_name in AA):
            continue

        if include_symmetry:
            cell = structure.structure.cell
            nearest = cell.find_nearest_pbc_image(
                metal.pos, neighbor.pos, mark.image_idx
            )
            transformed = cell.find_nearest_pbc_position(
                metal.pos, neighbor.pos, mark.image_idx
            )
            image = ContactImage.from_nearest_image(structure, nearest, transformed)
        else:
            image = ContactImage.explicit(metal, neighbor)

        # A symmetry copy is a distinct residue image, so only the source
        # residue in the explicit asymmetric unit is excluded.
        if neighbor.residue_key == metal.residue_key and not image.symmetry_contact:
            continue
        if not (
            0.0 < image.distance <= CANDIDATE_SEARCH_RADIUS + CANDIDATE_ACCEPT_EPSILON
        ):
            continue

        candidates.append(
            Candidate(
                neighbor=neighbor,
                image=image,
                candidate_sources={CandidateSource.PROXIMITY_4A},
            )
        )

    candidates.sort(key=_contact_sort_key)
    return candidates


def _candidate_identity(candidate: Candidate) -> _CandidateIdentity:
    """Group key for one atom image: everything but its computed position."""
    return (
        candidate.neighbor.source_key,
        candidate.image.symmetry_operation,
        candidate.image.translation,
    )


def merge_candidates(*candidate_groups: Iterable[Candidate]) -> list[Candidate]:
    """Merge proximity and declaration provenance for the same atom image.

    Copies: the returned records are new ``Candidate`` objects and the
    caller's inputs are left untouched, unlike
    ``deduplicate_special_position_contacts``, which merges in place. The copy
    goes through ``Candidate.with_provenance``, which cannot carry the
    ``init=False`` stage results over and so raises rather than dropping them:
    this must run before the donor-policy, eligibility, geometry, and
    multi-donor stages annotate anything.

    Within one group key, positions are compared with
    ``SAME_IMAGE_TOLERANCE`` rather than matched exactly, and the first
    candidate seen is the one retained.
    """
    merged: dict[_CandidateIdentity, list[Candidate]] = {}
    for candidates in candidate_groups:
        for candidate in candidates:
            group = merged.setdefault(_candidate_identity(candidate), [])
            existing = next(
                (
                    current
                    for current in group
                    if position_distance(
                        current.image.position,
                        candidate.image.position,
                    )
                    <= SAME_IMAGE_TOLERANCE
                ),
                None,
            )
            if existing is None:
                # Copy provenance collections because merging appends to them.
                group.append(
                    candidate.with_provenance(
                        candidate_sources=set(candidate.candidate_sources),
                        declared_connections=list(candidate.declared_connections),
                    )
                )
                continue
            existing.add_provenance(candidate)
    result = [candidate for group in merged.values() for candidate in group]
    result.sort(key=_contact_sort_key)
    return result
