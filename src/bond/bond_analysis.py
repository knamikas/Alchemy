"""Metal-ligand bond-distance analysis for the Alchemy pipeline.

For one PDB entry this finds every metal atom in the first model, uses Gemmi to
discover broad explicit, crystallographic, and strict-NCS candidates within 4 A,
supplements them with source ``struct_conn``/``LINK`` declarations, identifies
first-sphere-eligible candidates in a separate stage, and computes a
resolution-aware z-score against the consolidated
literature reference distances in ``metal_distances_info.txt`` (Harding 2006 and
Zheng et al. 2008 [Ni only]):

    z = (d_observed - mu) / sqrt(DPI**2 + sigma_lit**2)

The DPI (Diffraction-component Precision Index, Blow 2002 eq. 7) is the model's
per-atom coordinate uncertainty; adding it in quadrature with the literature
spread makes the same absolute deviation more significant in a high-resolution
structure than in a low-resolution one. A bond involves two atoms, but only one
DPI enters the denominator: the metal is treated as well enough ordered that its
positional uncertainty is negligible beside the donor's. See ``_zscore``.

The analysis covers every configured metal element, uses exact
``(residue, atom, metal)`` reference-distance keys, and reads DPI inputs from
PDB-REDO ``data.json``. The asymmetric-unit volume is computed from the crystal
cell and symmetry with gemmi. Missing inputs produce NaN derived values without
discarding measured bond geometry. ``main.py`` calls ``run_bond_analysis`` from
its per-entry worker and supplies edstats rows in memory for the sigma join.
"""

import math
from dataclasses import replace
from typing import Any, Union

from bond.bond_schema import (
    ZSCORE_OUTLIER_CUTOFF,
    _bond_row,
    _candidate_row,
    _context_warning_values,
)
from codes import CandidateSource, ContactScope, GeometryStatus, MultiDonorStatus
from bond.contact_record import Candidate
from bond.declared_connections import _collect_declared_candidates
from bond.donor_chemistry import (
    AA,
    C_TERMINAL_DONOR_ATOMS,
    DONOR_ELEMENTS,
    INFERRED_DONOR_ATOMS,
    N_TERMINAL_DONOR_ATOMS,
)
from bond.dpi import _calculate_dpi_details
from metal_elements import METAL_ELEMENTS
from metal_identification import _sigma_for, _sigma_index, _zd_indices
from reference_data import (
    cluster_ids,
    first_sphere_targets,
    heme_ids,
    literature_distances,
)
from structure_analysis import (
    NAN,
    count_deposited_ni,
    count_ni,
    load_structure,
    position_distance,
)


# Broad candidate-search radius. This must not be treated as a bond cutoff.
CUTOFF = 4.0
SEARCH_EPSILON = 1e-6

# First-sphere definition: donor distance <= target distance + 0.75 A.
# Harding, M. M. (2004), Acta Cryst. D60, 849-859.
# https://doi.org/10.1107/S0907444904004081
FIRST_SPHERE_TOLERANCE = 0.75

# Gemmi's ContactSearch uses 0.8 A by default to distinguish near-coincident
# symmetry images of an atom intended to occupy a special position.  NeighborSearch
# returns those images unfiltered, so Alchemy applies the same cutoff explicitly.
SPECIAL_POSITION_DEDUP_CUTOFF = 0.8


# --------------------------------------------------------------------------- #
# Bond rows
# --------------------------------------------------------------------------- #
def _bonding_key(neighbor, nb_res, metal_el):
    """Exact (residue, atom, metal) key matching metal_distances_info.txt columns."""
    name = neighbor.atom_name.strip()
    if neighbor.is_water:
        return ("HOH", "O", metal_el)
    if name in C_TERMINAL_DONOR_ATOMS:
        # No terminal-carboxylate-specific reference is bundled. Keep these
        # chemically typical donors eligible through the element fallback, but
        # do not misrepresent a residue side-chain reference as exact.
        return ("CTERM", "O", metal_el)
    if name == "O":  # backbone carbonyl O -> literal "CA" row
        return ("CA", "O", metal_el)
    if name.startswith("O"):  # side-chain O (OD1/OE1/OG/OH/...)
        return (nb_res, "O", metal_el)
    return (nb_res, neighbor.element, metal_el)  # His N, Cys S, ...


def _parent_type(structure, metal, metal_res, metal_el):
    if metal_res in cluster_ids():
        return "cluster"
    if metal_res in heme_ids():
        return "heme"
    if metal_el not in METAL_ELEMENTS:
        return "other"  # defensive: shouldn't happen, metal_atoms is pre-filtered
    residue = structure.residue_for_atom(metal)
    if residue.chemical_atom_site_count == 1:
        return "ion"
    return "other"


def _zscore(dist, mu, stdev, dpi):
    """Bond-distance z-score, ``(dist - mu)/sqrt(stdev^2 + dpi^2)``.

    The denominator carries **one** DPI, not the ``sqrt(2) * DPI`` that an
    independent-error treatment of two atoms would give. This is deliberate: the
    metal is a heavy scatterer and is normally among the best-ordered atoms in
    the model, so its positional uncertainty is taken as negligible beside the
    light donor's, and the single DPI stands for the donor. Do not "correct"
    this to ``2 * dpi ** 2`` -- it would shrink every z-score by up to a factor
    of sqrt(2) and change which contacts pass ZSCORE_OUTLIER_CUTOFF.
    """
    if not (math.isfinite(dpi) and math.isfinite(mu) and math.isfinite(stdev)):
        return NAN
    denom = math.sqrt(dpi**2 + stdev**2)
    return round((dist - mu) / denom, 4) if denom > 0 else NAN


def _contact_sort_key(contact):
    neighbor = contact.neighbor
    return (
        neighbor.chain_index,
        neighbor.residue_index,
        neighbor.atom_index,
        contact.symmetry_operation,
        contact.translation,
        contact.transformed_position,
    )


def _merge_candidate_provenance(target, source):
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


def _special_position_preference(contact):
    """Choose a stable representative for near-coincident symmetry images.

    Prefer an explicit image when one exists so an off-axis refinement artifact
    cannot turn an otherwise explicit contact into a symmetry-dependent one.
    Within the same scope, retain the shortest contact and then use stable
    symmetry provenance to break any remaining tie.
    """
    return (
        contact.symmetry_contact,
        contact.distance_raw,
        contact.symmetry_image_index,
        contact.symmetry_operation,
        contact.translation,
        contact.transformed_position,
    )


def _deduplicate_special_position_contacts(candidates):
    """Collapse near-coincident images of each deposited source atom.

    Sorting each source-atom group before the spatial comparison makes the
    result independent of Gemmi's NeighborSearch mark order.  Images farther
    apart than Gemmi's special-position cutoff remain distinct contacts.
    """
    by_source: dict[Any, list[Candidate]] = {}
    for candidate in candidates:
        by_source.setdefault(candidate.neighbor.source_key, []).append(candidate)

    contacts = []
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


def _first_sphere_rule(metal, neighbor):
    """Return target, cutoff, and provenance for proximity eligibility."""
    exact_key = _bonding_key(neighbor, neighbor.residue_name, metal.element)
    literature = literature_distances().get(exact_key)
    if literature is not None:
        target = literature[0]
        reference_kind = "exact"
        reference_key = exact_key
    else:
        # The scoring table may omit a residue-specific donor while still
        # defining the same donor element for this metal. Use the largest such
        # target only for sphere membership; exact references remain mandatory
        # for z-score calculation.
        target = first_sphere_targets().get((metal.element, neighbor.element))
        if target is None:
            return NAN, NAN, "missing", ""
        reference_kind = "element_fallback"
        reference_key = ("*", neighbor.element, metal.element)
    cutoff = min(CUTOFF, target + FIRST_SPHERE_TOLERANCE)
    return target, cutoff, reference_kind, ":".join(reference_key)


def _polymer_terminal_position(structure, atom):
    """Return ``(is_n_terminal, is_c_terminal)`` for a polymer residue."""
    import gemmi

    selected = structure.residue_for_atom(atom)
    if selected.source_polymer_position:
        position = selected.source_polymer_position
        return "N" in position, "C" in position
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
        return atom.residue_index == indices[0], atom.residue_index == indices[-1]
    except (AttributeError, IndexError, TypeError):
        return False, False


def _inferred_donor_rule(structure, atom):
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


def _annotate_donor_policy(structure, candidates):
    """Annotate candidates with inference permission and declaration override."""
    for candidate in candidates:
        allowed, rule = _inferred_donor_rule(structure, candidate.neighbor)
        declared = bool(candidate.declared_connections)
        # A declaration overrides the donor-atom rule only inside a residue
        # class Alchemy can assess. Claiming the override for a donor whose
        # class has no reference at all would label a row as a declared bond
        # that never becomes one.
        supported = candidate.donor_class_supported
        candidate.inferred_donor_allowed = allowed
        candidate.inferred_donor_rule = rule
        candidate.donor_rule_override = (
            "declared_connection" if declared and not allowed and supported else ""
        )


def _identify_first_sphere_candidates(candidates, metal):
    """Identify proximal candidates eligible for the first sphere.

    Candidate discovery deliberately performs no bond assignment. This stage
    applies the current distance-based eligibility rule after discovery and
    keeps unsupported metal-donor pairs visible to the caller as incomplete
    assignment evidence. Passing this rule does not by itself establish a
    chemically assigned bond.
    """
    eligible = []
    unsupported_pairs = set()
    for candidate in candidates:
        neighbor = candidate.neighbor
        donor_allowed = candidate.inferred_donor_allowed
        target, cutoff, reference_kind, reference_key = _first_sphere_rule(
            metal, neighbor
        )
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
    return (_deduplicate_special_position_contacts(eligible), unsupported_pairs)


def _collect_proximal_candidates(structure, search, metal, include_symmetry):
    """Return broad donor-like candidates within 4 A for one search scope.

    This function answers only the discovery question. It does not use a
    literature target, apply the first-sphere tolerance, or call any candidate
    a bond. Coordination assignment is performed later by
    ``_identify_first_sphere_candidates`` and future donor-group logic.
    """
    candidates = []
    marks = search.find_atoms(
        metal.pos, "\x00", min_dist=0.0, radius=CUTOFF + SEARCH_EPSILON
    )
    for mark in marks:
        neighbor = structure.atom_for_mark(mark)
        if neighbor is None:
            continue
        # Cheapest rejections first: the donor-element test discards most marks,
        # so the residue lookup only runs for atoms that can still qualify.
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
            # Unpacked rather than comprehended: ``pbc_shift`` is a lattice
            # triple, and spelling that out gives the field the exact
            # ``tuple[int, int, int]`` it declares.
            shift_a, shift_b, shift_c = nearest.pbc_shift
            translation = (int(shift_a), int(shift_b), int(shift_c))
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

        # A symmetry copy is a distinct residue image. Exclude only the actual
        # source residue in the explicit asymmetric unit.
        if neighbor.residue_key == metal.residue_key and not symmetry_contact:
            continue
        if not (0.0 < distance <= CUTOFF + 1e-9):
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
                contact_scope=contact_scope,
                symmetry_image_index=image_index,
                symmetry_operation=operation,
                translation=translation,
                candidate_sources={CandidateSource.PROXIMITY_4A},
            )
        )

    candidates.sort(key=_contact_sort_key)
    return candidates


def _candidate_identity(candidate):
    return (
        candidate.neighbor.source_key,
        candidate.symmetry_operation,
        candidate.translation,
        tuple(round(value, 5) for value in candidate.transformed_position),
    )


def _merge_candidates(*candidate_groups):
    """Merge proximity and declaration provenance for the same atom image."""
    merged: dict[Any, Candidate] = {}
    for candidates in candidate_groups:
        for candidate in candidates:
            key = _candidate_identity(candidate)
            existing = merged.get(key)
            if existing is None:
                # Copied, with its two provenance collections copied as well:
                # merging appends to them, and the caller's candidate must not
                # gain the other group's sources.
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


def _current_contacts_from_candidates(candidates, metal):
    """Return typical inferred and explicitly declared contacts."""
    eligible, unsupported_pairs = _identify_first_sphere_candidates(candidates, metal)
    declared_not_inferred = [
        candidate
        for candidate in candidates
        if candidate.declared_connections
        and not candidate.inferred_contact_eligible
        and
        # A donor class with no literature reference stays candidate evidence:
        # it is reported with its measured distance but never scored.
        candidate.donor_class_supported
    ]
    return (
        _deduplicate_special_position_contacts(eligible + declared_not_inferred),
        unsupported_pairs,
    )


def _annotate_contacts(contacts, metal_element, dpi):
    for contact in contacts:
        neighbor = contact.neighbor
        reported_distance = round(contact.distance_raw, 3)
        literature = literature_distances().get(
            _bonding_key(neighbor, neighbor.residue_name, metal_element)
        )
        if literature is None:
            mu = stdev = zscore = NAN
        else:
            mu, stdev = literature
            zscore = _zscore(reported_distance, mu, stdev, dpi)
        contact.distance = reported_distance
        contact.literature_distance = mu
        contact.literature_stdev = stdev
        contact.zscore = zscore
        contact.reference_covered = literature is not None
        contact.geometry_outlier = (
            abs(zscore) >= ZSCORE_OUTLIER_CUTOFF if math.isfinite(zscore) else ""
        )
        contact.geometry_consistent = (
            abs(zscore) < ZSCORE_OUTLIER_CUTOFF if math.isfinite(zscore) else ""
        )


def _residue_image_key(contact):
    """Identity of one donor residue image around the current metal site."""
    return (
        contact.neighbor.residue_key,
        contact.contact_scope,
        contact.strict_ncs_operation_id,
        contact.symmetry_image_index,
        contact.symmetry_operation,
        contact.translation,
    )


def _annotate_multi_donor_groups(contacts):
    """Annotate assigned contacts that share one donor-residue image.

    There is no upper limit on group size. All bond-level z-scores and outlier
    flags contribute normally when assessable. Group status is contextual: if
    any member is suspect, every member records that it belongs to a suspect
    multi-donor group, without weakening or excluding any individual result.
    """
    groups: dict[Any, list[Candidate]] = {}
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


def _scope_summary(contacts, metal_zero_occupancy, unavailable=False):
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
    if metal_zero_occupancy or scored_assessable == 0:
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


def _site_context_values(contacts, candidates):
    """Aggregate coordination-relevant context without changing confidence."""
    reasons = []
    for contact in contacts:
        values = _context_warning_values(contact)
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


def _site_summary(
    metal,
    explicit_contacts,
    image_contacts,
    dpi,
    resolution,
    ni,
    deposited_ni,
    dpi_reason,
    structure,
):
    metal_zero = metal.occupancy_valid and metal.occupancy == 0.0
    explicit = _scope_summary(explicit_contacts, metal_zero)
    image_search_available = image_contacts is not None
    image_inclusive = _scope_summary(
        image_contacts or [],
        metal_zero,
        unavailable=not image_search_available,
    )
    primary = image_inclusive if image_search_available else explicit
    symmetry_count = (
        sum(contact.symmetry_contact for contact in image_contacts)
        if image_search_available
        else NAN
    )
    crystallographic_count = (
        sum(contact.crystallographic_contact for contact in image_contacts)
        if image_search_available
        else NAN
    )
    strict_ncs_count = (
        sum(contact.strict_ncs_contact for contact in image_contacts)
        if image_search_available
        else NAN
    )
    combined_count = (
        sum(
            contact.crystallographic_contact and contact.strict_ncs_contact
            for contact in image_contacts
        )
        if image_search_available
        else NAN
    )
    # Blank where symmetry was never searched, boolean where it was: "not
    # assessed" and "assessed false" are different answers, and the columns
    # keep them apart. Declared because the first branch would otherwise fix
    # the inferred type at ``str``.
    changed: Union[str, bool]
    depends_crystallographic: Union[str, bool]
    depends_strict_ncs: Union[str, bool]
    if not image_search_available:
        generated_scope = ""
        changed = ""
        depends_crystallographic = ""
        depends_strict_ncs = ""
    elif not symmetry_count:
        generated_scope = "none"
        changed = explicit["status"] != image_inclusive["status"]
        depends_crystallographic = False
        depends_strict_ncs = False
    elif crystallographic_count and strict_ncs_count:
        generated_scope = "strict_ncs_and_crystallographic"
        changed = explicit["status"] != image_inclusive["status"]
        depends_crystallographic = True
        depends_strict_ncs = True
    elif crystallographic_count:
        generated_scope = "crystallographic"
        changed = explicit["status"] != image_inclusive["status"]
        depends_crystallographic = True
        depends_strict_ncs = False
    else:
        generated_scope = "strict_ncs"
        changed = explicit["status"] != image_inclusive["status"]
        depends_crystallographic = False
        depends_strict_ncs = True

    reasons = []
    if metal_zero:
        reasons.append("metal_zero_occupancy")
    if dpi_reason:
        reasons.append(dpi_reason)
    if not image_search_available:
        reasons.append("symmetry_search_unavailable")
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
        "metal_zero_occupancy": metal_zero,
        "geometry_not_assessed_reason": "|".join(dict.fromkeys(reasons)),
    }


def run_bond_analysis(
    pdb_id,
    pdb_path,
    stats_rows,
    header,
    dpi_inputs,
    structure=None,
    connection_path=None,
):
    """Return contact rows, candidate rows, site summaries, and metadata.

    Every external proximal candidate is retained in the candidate rows.
    First-sphere-eligible candidates and contacts declared by source
    ``struct_conn``/``LINK`` records are emitted as the current bond rows.
    Declared contacts are evaluated even outside the 4 A discovery radius or
    first-sphere cutoff. Atoms in the metal's own residue remain excluded.
    Assigned contacts are grouped by donor-residue image for the multi-donor
    scoring policy after every bond receives its individual z-score.
    Image-inclusive results are primary when symmetry metadata is available.
    Missing DPI does not prevent identification or distance reporting.
    """
    if structure is None:
        structure = load_structure(pdb_id, pdb_path)

    metals_in_model = structure.metal_atoms(METAL_ELEMENTS, canonical=True)
    # Annotated because the values are three lists and a bool: inference
    # settles on ``object``, and every ``append`` below then looks wrong.
    metadata: dict[str, Any] = {
        "partial_reason_codes": [],
        "warning_codes": list(structure.warning_codes),
        "messages": [],
        "retryable": False,
    }
    if not metals_in_model:
        return [], [], {}, metadata

    (declared_candidates, declared_issues, declared_warnings) = (
        _collect_declared_candidates(
            structure, connection_path or pdb_path, metals_in_model
        )
    )
    if declared_issues:
        metadata["partial_reason_codes"].append(
            "declared_connection_resolution_incomplete"
        )
        metadata["messages"].extend(declared_issues)
    metadata["warning_codes"].extend(declared_warnings)
    declared_by_metal: dict[Any, list[Candidate]] = {}
    for candidate in declared_candidates:
        declared_by_metal.setdefault(candidate.metal.source_key, []).append(candidate)

    dpi, resolution, dpi_reason = _calculate_dpi_details(structure, dpi_inputs)
    ni = count_ni(structure)
    deposited_ni = count_deposited_ni(structure)
    if dpi_reason:
        metadata["partial_reason_codes"].append(dpi_reason)
        metadata["messages"].append(f"DPI unavailable: {dpi_reason}")
    if not structure.symmetry_search_available:
        metadata["partial_reason_codes"].append("symmetry_search_unavailable")
        metadata["messages"].append(
            "symmetry search unavailable: "
            + (structure.symmetry_search_failure_reason or "unknown reason")
        )

    explicit_search = structure.make_neighbor_search(
        CUTOFF + SEARCH_EPSILON, include_symmetry=False, positive_occupancy_only=True
    )
    image_search = None
    if structure.symmetry_search_available:
        image_search = structure.make_neighbor_search(
            CUTOFF + SEARCH_EPSILON, include_symmetry=True, positive_occupancy_only=True
        )
    sig = _sigma_index(stats_rows)
    zd_idx = _zd_indices(header)

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    summaries = {}
    for metal in metals_in_model:
        metal_declarations = declared_by_metal.get(metal.source_key, ())
        explicit_declarations = [
            candidate
            for candidate in metal_declarations
            if not candidate.symmetry_contact
        ]
        explicit_candidates = _merge_candidates(
            _collect_proximal_candidates(structure, explicit_search, metal, False),
            explicit_declarations,
        )
        _annotate_donor_policy(structure, explicit_candidates)
        explicit, unsupported_pairs = _current_contacts_from_candidates(
            explicit_candidates, metal
        )
        _annotate_contacts(explicit, metal.element, dpi)
        _annotate_multi_donor_groups(explicit)
        image_candidates = None
        image_contacts = None
        if image_search is not None:
            image_candidates = _merge_candidates(
                _collect_proximal_candidates(structure, image_search, metal, True),
                metal_declarations,
            )
            _annotate_donor_policy(structure, image_candidates)
            image_contacts, image_unsupported = _current_contacts_from_candidates(
                image_candidates, metal
            )
            unsupported_pairs.update(image_unsupported)
            _annotate_contacts(image_contacts, metal.element, dpi)
            _annotate_multi_donor_groups(image_contacts)
        if unsupported_pairs:
            metadata["partial_reason_codes"].append("missing_first_sphere_reference")
            pairs = ", ".join(
                f"{metal_element}-{donor_element}"
                for metal_element, donor_element in sorted(unsupported_pairs)
            )
            metadata["messages"].append(
                f"first-sphere reference unavailable for {pairs}"
            )
        primary_contacts = image_contacts if image_contacts is not None else explicit
        primary_candidates = (
            image_candidates if image_candidates is not None else explicit_candidates
        )
        summary = _site_summary(
            metal,
            explicit,
            image_contacts,
            dpi,
            resolution,
            ni,
            deposited_ni,
            dpi_reason,
            structure,
        )
        summary.update(_site_context_values(primary_contacts, primary_candidates))
        summaries[metal.source_key] = summary
        if summary["metal_zero_occupancy"]:
            metadata["partial_reason_codes"].append("metal_zero_occupancy")
            metadata["messages"].append(
                f"zero-occupancy metal: {metal.chain_id}/{metal.resnum}/"
                f"{metal.atom_name}"
            )

        sigma = _sigma_for(
            sig, metal.residue_name, metal.chain_id, metal.resnum, zd_idx
        )
        parent_type = _parent_type(structure, metal, metal.residue_name, metal.element)
        rows.extend(
            _bond_row(
                pdb_id, structure, metal, contact, dpi, resolution, sigma, parent_type
            )
            for contact in primary_contacts
        )
        candidate_rows.extend(
            _candidate_row(pdb_id, structure, metal, candidate)
            for candidate in primary_candidates
        )

    metadata["partial_reason_codes"] = list(
        dict.fromkeys(metadata["partial_reason_codes"])
    )
    metadata["warning_codes"] = list(dict.fromkeys(metadata["warning_codes"]))
    metadata["messages"] = list(dict.fromkeys(metadata["messages"]))
    return rows, candidate_rows, summaries, metadata
