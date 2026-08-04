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
from typing import Any, Mapping, Optional

from declared_connections import _collect_declared_candidates
from donor_chemistry import (
    AA,
    C_TERMINAL_DONOR_ATOMS,
    DONOR_ELEMENTS,
    INFERRED_DONOR_ATOMS,
    N_TERMINAL_DONOR_ATOMS,
)
from dpi import _calculate_dpi_details
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

# Conservative cutoff for a reference-covered geometry outlier.
ZSCORE_OUTLIER_CUTOFF = 6.0


# Fixed output schema; main.py imports this so the module and driver never
# drift. Legacy "candidate" field names are retained for CSV compatibility,
# but now describe inferred first-sphere or source-declared contacts.
BOND_COLUMNS = [
    "pdbID",
    "metal_resname",
    "metal_chain",
    "metal_resnum",
    "metal_element",
    "neighbor_resname",
    "neighbor_atom",
    "neighbor_element",
    "distance",
    "coordination_status",
    "coordination_source",
    "declared_connection",
    "connection_id",
    "connection_type",
    "connection_link_id",
    "connection_asu",
    "connection_reported_distance",
    "inferred_donor_allowed",
    "inferred_donor_rule",
    "donor_rule_override",
    "context_warning",
    "context_warning_reasons",
    "literature_distance",
    "literature_stdev",
    "zscore",
    "dpi",
    "resolution",
    "sigma_mag",
    "sigma_neg",
    "sigma_pos",
    "parent_type",
    "bonded_to",
    "model_id",
    "metal_model_index",
    "metal_chain_index",
    "metal_residue_index",
    "metal_atom_index",
    "metal_atom",
    "metal_icode",
    "metal_altloc",
    "metal_occupancy",
    "metal_occupancy_valid",
    "metal_occupancy_status",
    "metal_conformer_mean_occupancy",
    "metal_altloc_options",
    "metal_altloc_selection_fallback",
    "neighbor_chain",
    "neighbor_resnum",
    "neighbor_icode",
    "neighbor_model_index",
    "neighbor_chain_index",
    "neighbor_residue_index",
    "neighbor_atom_index",
    "neighbor_altloc",
    "neighbor_occupancy",
    "neighbor_occupancy_valid",
    "neighbor_occupancy_status",
    "neighbor_conformer_mean_occupancy",
    "neighbor_altloc_options",
    "neighbor_altloc_selection_fallback",
    "alternative_conformers_present",
    "altloc_selection_fallback",
    "neighbor_class",
    "candidate_contact",
    "reference_covered",
    "geometry_outlier",
    "geometry_consistent",
    "multi_donor_detected",
    "multi_donor_contact_count",
    "multi_donor_geometry_status",
    "multi_donor_contains_suspect_bond",
    "score_eligible",
    "score_exclusion_reason",
    "zscore_outlier_cutoff",
    "contact_scope",
    "symmetry_contact",
    "crystallographic_contact",
    "strict_ncs_contact",
    "strict_ncs_operation_id",
    "symmetry_image_index",
    "symmetry_operation",
    "cell_translation_x",
    "cell_translation_y",
    "cell_translation_z",
    "transformed_neighbor_x",
    "transformed_neighbor_y",
    "transformed_neighbor_z",
]


# Candidate evidence is kept separately from assigned bond rows. This schema
# intentionally contains no z-score or geometry classification: failing
# first-sphere eligibility does not establish that an atom is chemically
# nonbonded, and a source declaration may supersede the proximity-only result.
CANDIDATE_COLUMNS = [
    "pdbID",
    "candidate_source",
    "eligibility_status",
    "eligibility_reason",
    "first_sphere_eligible",
    "candidate_distance",
    "assignment_target",
    "assignment_tolerance",
    "first_sphere_cutoff",
    "assignment_reference_kind",
    "assignment_reference",
    "inferred_contact_eligible",
    "inferred_donor_allowed",
    "inferred_donor_rule",
    "donor_rule_override",
    "context_warning",
    "context_warning_reasons",
    "coordination_status",
    "coordination_source",
    "declared_connection",
    "connection_id",
    "connection_type",
    "connection_link_id",
    "connection_asu",
    "connection_reported_distance",
    "metal_resname",
    "metal_chain",
    "metal_resnum",
    "metal_element",
    "metal_atom",
    "metal_icode",
    "metal_altloc",
    "metal_occupancy",
    "model_id",
    "metal_model_index",
    "metal_chain_index",
    "metal_residue_index",
    "metal_atom_index",
    "neighbor_resname",
    "neighbor_chain",
    "neighbor_resnum",
    "neighbor_atom",
    "neighbor_element",
    "neighbor_icode",
    "neighbor_altloc",
    "neighbor_occupancy",
    "neighbor_class",
    "neighbor_model_index",
    "neighbor_chain_index",
    "neighbor_residue_index",
    "neighbor_atom_index",
    "contact_scope",
    "symmetry_contact",
    "crystallographic_contact",
    "strict_ncs_contact",
    "strict_ncs_operation_id",
    "symmetry_image_index",
    "symmetry_operation",
    "cell_translation_x",
    "cell_translation_y",
    "cell_translation_z",
    "transformed_neighbor_x",
    "transformed_neighbor_y",
    "transformed_neighbor_z",
]


# Appended after the dynamic EDSTATS header in metal_stats_all.csv.
STATS_EXTRA_COLUMNS = [
    "model_policy",
    "input_model_count",
    "model_analyzed",
    "model_id",
    "multi_model_structure",
    "metal_model_index",
    "metal_chain_index",
    "metal_residue_index",
    "metal_atom_index",
    "metal_resname",
    "metal_chain",
    "metal_resnum",
    "metal_atom",
    "metal_element",
    "metal_icode",
    "metal_altloc",
    "metal_occupancy",
    "metal_occupancy_valid",
    "metal_occupancy_status",
    "metal_conformer_mean_occupancy",
    "metal_altloc_options",
    "alternative_conformers_present",
    "altloc_selection_fallback",
    "density_observation_id",
    "density_scope",
    "density_shared_site_count",
    "density_is_shared",
    "coordinate_mapping_status",
    "selected_metal_site_status",
    "dpi",
    "resolution",
    "occupancy_weighted_atom_count",
    "deposited_occupancy_weighted_atom_count",
    "dpi_atom_count_multiplier",
    "strict_ncs_operation_count",
    "crystallographic_operation_count",
    "dpi_unavailable_reason",
    "candidate_contact_count",
    "reference_covered_contact_count",
    "geometry_outlier_contact_count",
    "geometry_consistent_contact_count",
    "score_eligible_contact_count",
    "score_excluded_contact_count",
    "scored_geometry_outlier_contact_count",
    "scored_geometry_consistent_contact_count",
    "multi_donor_residue_group_count",
    "multi_donor_contact_count",
    "suspect_multi_donor_residue_group_count",
    "indeterminate_multi_donor_residue_group_count",
    "context_warning",
    "context_warning_reasons",
    "non_typical_first_sphere_candidate_count",
    "declared_donor_override_contact_count",
    "explicit_contact_count",
    "symmetry_contact_count",
    "image_inclusive_contact_count",
    "crystallographic_contact_count",
    "strict_ncs_contact_count",
    "combined_ncs_crystallographic_contact_count",
    "geometry_outlier_count_explicit",
    "geometry_outlier_count_image_inclusive",
    "geometry_coverage_explicit",
    "geometry_coverage_image_inclusive",
    "explicit_geometry_status",
    "image_inclusive_geometry_status",
    "generated_contact_scope",
    "geometry_classification_changes_with_generated_images",
    "coordination_depends_on_crystallographic_symmetry",
    "coordination_depends_on_strict_ncs",
    "symmetry_search_available",
    "symmetry_search_failure_reason",
    "occupancy_validation_failed",
    "missing_occupancy_count",
    "invalid_occupancy_count",
    "zero_occupancy_atom_count",
    "metal_zero_occupancy",
    "geometry_not_assessed_reason",
    "duplicate_atom_records_present",
    "duplicate_atom_record_count",
    "duplicate_atom_coordinate_conflict_count",
    "malformed_duplicate_atom_name_count",
    "raw_occupancy_mapping_failed",
    "raw_occupancy_mapping_failure_reason",
    "unknown_element_atom_count",
    "element_validation_warning",
    "zscore_outlier_cutoff",
]


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


def _bonded_to(is_water=False):
    return "HOH" if is_water else "P"


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
    neighbor = contact["neighbor"]
    return (
        neighbor.chain_index,
        neighbor.residue_index,
        neighbor.atom_index,
        contact["symmetry_operation"],
        contact["translation"],
        contact["transformed_position"],
    )


def _merge_candidate_provenance(target, source):
    """Add discovery and declaration provenance without duplicating records."""
    target.setdefault("candidate_sources", set()).update(
        source.get("candidate_sources", ())
    )
    target.setdefault("declared_connections", [])
    known_connections = {
        (record["source"], record["connection_id"])
        for record in target["declared_connections"]
    }
    for record in source.get("declared_connections", ()):
        connection_key = (record["source"], record["connection_id"])
        if connection_key not in known_connections:
            target["declared_connections"].append(record)
            known_connections.add(connection_key)


def _special_position_preference(contact):
    """Choose a stable representative for near-coincident symmetry images.

    Prefer an explicit image when one exists so an off-axis refinement artifact
    cannot turn an otherwise explicit contact into a symmetry-dependent one.
    Within the same scope, retain the shortest contact and then use stable
    symmetry provenance to break any remaining tie.
    """
    return (
        contact["symmetry_contact"],
        contact["distance_raw"],
        contact["symmetry_image_index"],
        contact["symmetry_operation"],
        contact["translation"],
        contact["transformed_position"],
    )


def _deduplicate_special_position_contacts(candidates):
    """Collapse near-coincident images of each deposited source atom.

    Sorting each source-atom group before the spatial comparison makes the
    result independent of Gemmi's NeighborSearch mark order.  Images farther
    apart than Gemmi's special-position cutoff remain distinct contacts.
    """
    by_source = {}
    for candidate in candidates:
        by_source.setdefault(candidate["neighbor"].source_key, []).append(candidate)

    contacts = []
    for source_key in sorted(by_source):
        retained = []
        for candidate in sorted(
            by_source[source_key], key=_special_position_preference
        ):
            duplicate = next(
                (
                    current
                    for current in retained
                    if position_distance(
                        current["transformed_position"],
                        candidate["transformed_position"],
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
        allowed, rule = _inferred_donor_rule(structure, candidate["neighbor"])
        declared = bool(candidate.get("declared_connections"))
        # A declaration overrides the donor-atom rule only inside a residue
        # class Alchemy can assess. Claiming the override for a donor whose
        # class has no reference at all would label a row as a declared bond
        # that never becomes one.
        supported = candidate.get("donor_class_supported", True)
        candidate.update(
            inferred_donor_allowed=allowed,
            inferred_donor_rule=rule,
            donor_rule_override=(
                "declared_connection" if declared and not allowed and supported else ""
            ),
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
        neighbor = candidate["neighbor"]
        donor_allowed = candidate.get("inferred_donor_allowed", True)
        target, cutoff, reference_kind, reference_key = _first_sphere_rule(
            metal, neighbor
        )
        if not math.isfinite(cutoff):
            declared = bool(candidate.get("declared_connections"))
            if donor_allowed or declared:
                unsupported_pairs.add((metal.element, neighbor.element))
            candidate.update(
                eligibility_status=(
                    "missing_assignment_reference"
                    if donor_allowed or declared
                    else "non_typical_donor"
                ),
                eligibility_reason=(
                    "no_metal_donor_assignment_reference"
                    if donor_allowed or declared
                    else "atom_not_in_typical_inferred_donor_list"
                ),
                first_sphere_eligible=False,
                inferred_contact_eligible=False,
                assignment_target=NAN,
                assignment_tolerance=FIRST_SPHERE_TOLERANCE,
                first_sphere_cutoff=NAN,
                assignment_reference_kind=reference_kind,
                assignment_reference=reference_key,
            )
            continue
        is_eligible = candidate["distance_raw"] <= cutoff + SEARCH_EPSILON
        inferred_eligible = is_eligible and donor_allowed
        candidate.update(
            eligibility_status=(
                "first_sphere_eligible"
                if inferred_eligible
                else ("non_typical_donor" if is_eligible else "outside_first_sphere")
            ),
            eligibility_reason=(
                "distance_within_target_plus_0.75"
                if inferred_eligible
                else (
                    "atom_not_in_typical_inferred_donor_list"
                    if is_eligible
                    else "distance_exceeds_target_plus_0.75"
                )
            ),
            first_sphere_eligible=is_eligible,
            inferred_contact_eligible=inferred_eligible,
            assignment_target=target,
            assignment_tolerance=FIRST_SPHERE_TOLERANCE,
            first_sphere_cutoff=cutoff,
            assignment_reference_kind=reference_kind,
            assignment_reference=reference_key,
        )
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
            translation = tuple(int(value) for value in nearest.pbc_shift)
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
            contact_scope = "explicit"
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
            {
                "neighbor": neighbor,
                "distance_raw": distance,
                "transformed_position": position,
                "symmetry_contact": symmetry_contact,
                "crystallographic_contact": crystallographic_contact,
                "strict_ncs_contact": strict_ncs_contact,
                "strict_ncs_operation_id": strict_ncs_operation_id,
                "contact_scope": contact_scope,
                "symmetry_image_index": image_index,
                "symmetry_operation": operation,
                "translation": translation,
                "candidate_sources": {"proximity_4A"},
                "declared_connections": [],
            }
        )

    candidates.sort(key=_contact_sort_key)
    return candidates


def _candidate_identity(candidate):
    return (
        candidate["neighbor"].source_key,
        candidate["symmetry_operation"],
        candidate["translation"],
        tuple(round(value, 5) for value in candidate["transformed_position"]),
    )


def _merge_candidates(*candidate_groups):
    """Merge proximity and declaration provenance for the same atom image."""
    merged = {}
    for candidates in candidate_groups:
        for candidate in candidates:
            key = _candidate_identity(candidate)
            existing = merged.get(key)
            if existing is None:
                merged[key] = {
                    **candidate,
                    "candidate_sources": set(candidate.get("candidate_sources", ())),
                    "declared_connections": list(
                        candidate.get("declared_connections", ())
                    ),
                }
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
        if candidate["declared_connections"]
        and not candidate["inferred_contact_eligible"]
        and
        # A donor class with no literature reference stays candidate evidence:
        # it is reported with its measured distance but never scored.
        candidate.get("donor_class_supported", True)
    ]
    return (
        _deduplicate_special_position_contacts(eligible + declared_not_inferred),
        unsupported_pairs,
    )


def _connection_output_values(candidate):
    records = candidate.get("declared_connections", ())
    inferred = bool(candidate.get("inferred_contact_eligible", False))

    def joined(name):
        return "|".join(
            dict.fromkeys(
                str(record[name])
                for record in records
                if str(record.get(name, "")) not in ("", "nan")
            )
        )

    return {
        "coordination_status": (
            "declared" if records else ("inferred" if inferred else "unassigned")
        ),
        "coordination_source": (
            joined("source") if records else ("proximity_rule" if inferred else "")
        ),
        "declared_connection": bool(records),
        "connection_id": joined("connection_id"),
        "connection_type": joined("connection_type"),
        "connection_link_id": joined("connection_link_id"),
        "connection_asu": joined("connection_asu"),
        "connection_reported_distance": joined("connection_reported_distance"),
    }


def _donor_output_values(candidate):
    return {
        "inferred_donor_allowed": candidate["inferred_donor_allowed"],
        "inferred_donor_rule": candidate["inferred_donor_rule"],
        "donor_rule_override": candidate["donor_rule_override"],
    }


def _context_warning_values(candidate, include_proximal=False):
    """Return the extensible binary context flag and auditable reason codes."""
    reasons = []
    if not candidate["inferred_donor_allowed"]:
        if candidate.get("declared_connections"):
            reasons.append("declared_non_typical_donor")
        elif candidate.get("first_sphere_eligible", False):
            reasons.append("non_typical_first_sphere_candidate")
        elif include_proximal:
            reasons.append("non_typical_proximal_candidate")
    if candidate.get("multi_donor_contains_suspect_bond", False):
        reasons.append("suspect_multi_donor_group")
    reasons = list(dict.fromkeys(reasons))
    return {
        "context_warning": bool(reasons),
        "context_warning_reasons": "|".join(reasons),
    }


def _annotate_contacts(contacts, metal_element, dpi):
    for contact in contacts:
        neighbor = contact["neighbor"]
        reported_distance = round(contact["distance_raw"], 3)
        literature = literature_distances().get(
            _bonding_key(neighbor, neighbor.residue_name, metal_element)
        )
        if literature is None:
            mu = stdev = zscore = NAN
        else:
            mu, stdev = literature
            zscore = _zscore(reported_distance, mu, stdev, dpi)
        contact.update(
            distance=reported_distance,
            literature_distance=mu,
            literature_stdev=stdev,
            zscore=zscore,
            reference_covered=literature is not None,
            geometry_outlier=(
                abs(zscore) >= ZSCORE_OUTLIER_CUTOFF if math.isfinite(zscore) else ""
            ),
            geometry_consistent=(
                abs(zscore) < ZSCORE_OUTLIER_CUTOFF if math.isfinite(zscore) else ""
            ),
        )


def _residue_image_key(contact):
    """Identity of one donor residue image around the current metal site."""
    return (
        contact["neighbor"].residue_key,
        contact["contact_scope"],
        contact["strict_ncs_operation_id"],
        contact["symmetry_image_index"],
        contact["symmetry_operation"],
        contact["translation"],
    )


def _annotate_multi_donor_groups(contacts):
    """Annotate assigned contacts that share one donor-residue image.

    There is no upper limit on group size. All bond-level z-scores and outlier
    flags contribute normally when assessable. Group status is contextual: if
    any member is suspect, every member records that it belongs to a suspect
    multi-donor group, without weakening or excluding any individual result.
    """
    groups = {}
    for contact in contacts:
        groups.setdefault(_residue_image_key(contact), []).append(contact)

    for group in groups.values():
        count = len(group)
        multi_donor = count >= 2
        if not multi_donor:
            contact = group[0]
            assessable = (
                contact["geometry_outlier"] is True
                or contact["geometry_consistent"] is True
            )
            contact.update(
                multi_donor_detected=False,
                multi_donor_contact_count=1,
                multi_donor_geometry_status="single_donor",
                multi_donor_contains_suspect_bond=False,
                score_eligible=assessable,
                score_exclusion_reason=("" if assessable else "zscore_unavailable"),
            )
            continue

        any_outlier = any(contact["geometry_outlier"] is True for contact in group)
        all_consistent = all(
            contact["geometry_consistent"] is True for contact in group
        )
        if all_consistent:
            status = "consistent"
        elif any_outlier:
            status = "suspect"
        else:
            status = "indeterminate"
        for contact in group:
            assessable = (
                contact["geometry_outlier"] is True
                or contact["geometry_consistent"] is True
            )
            contact.update(
                multi_donor_detected=True,
                multi_donor_contact_count=count,
                multi_donor_geometry_status=status,
                multi_donor_contains_suspect_bond=any_outlier,
                score_eligible=assessable,
                score_exclusion_reason=("" if assessable else "zscore_unavailable"),
            )


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
    covered = sum(bool(contact["reference_covered"]) for contact in contacts)
    outlier = sum(contact["geometry_outlier"] is True for contact in contacts)
    consistent = sum(contact["geometry_consistent"] is True for contact in contacts)
    score_eligible = sum(contact["score_eligible"] is True for contact in contacts)
    score_excluded = candidate - score_eligible
    scored_outlier = sum(
        contact["score_eligible"] is True and contact["geometry_outlier"] is True
        for contact in contacts
    )
    scored_consistent = sum(
        contact["score_eligible"] is True and contact["geometry_consistent"] is True
        for contact in contacts
    )
    multi_donor_groups = {
        _residue_image_key(contact)
        for contact in contacts
        if contact["multi_donor_detected"]
    }
    suspect_multi_donor_groups = {
        _residue_image_key(contact)
        for contact in contacts
        if contact["multi_donor_geometry_status"] == "suspect"
    }
    indeterminate_multi_donor_groups = {
        _residue_image_key(contact)
        for contact in contacts
        if contact["multi_donor_geometry_status"] == "indeterminate"
    }
    multi_donor_contacts = sum(
        contact["multi_donor_detected"] is True for contact in contacts
    )
    scored_assessable = scored_outlier + scored_consistent
    if metal_zero_occupancy or scored_assessable == 0:
        status = "insufficient data"
    elif scored_outlier:
        status = "suspect"
    else:
        status = "plausible"
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
        if (
            not candidate["inferred_donor_allowed"]
            and candidate.get("first_sphere_eligible", False)
        )
    ]
    if non_typical_first_sphere:
        reasons.append("non_typical_first_sphere_candidate")
    reasons = list(dict.fromkeys(reasons))
    return {
        "context_warning": bool(reasons),
        "context_warning_reasons": "|".join(reasons),
        "non_typical_first_sphere_candidate_count": len(non_typical_first_sphere),
        "declared_donor_override_contact_count": sum(
            contact["donor_rule_override"] == "declared_connection"
            for contact in contacts
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
        sum(contact["symmetry_contact"] for contact in image_contacts)
        if image_search_available
        else NAN
    )
    crystallographic_count = (
        sum(contact["crystallographic_contact"] for contact in image_contacts)
        if image_search_available
        else NAN
    )
    strict_ncs_count = (
        sum(contact["strict_ncs_contact"] for contact in image_contacts)
        if image_search_available
        else NAN
    )
    combined_count = (
        sum(
            contact["crystallographic_contact"] and contact["strict_ncs_contact"]
            for contact in image_contacts
        )
        if image_search_available
        else NAN
    )
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


def stats_extra_values(
    structure: Any,
    metal: Optional[Any] = None,
    summary: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return fixed per-site values appended to an EDSTATS row."""
    summary = summary or {}
    residue = structure.residue_for_atom(metal) if metal is not None else None
    values = {
        "model_policy": structure.model_policy,
        "input_model_count": structure.input_model_count,
        "model_analyzed": structure.model_analyzed,
        "model_id": structure.analyzed_model_id,
        "multi_model_structure": structure.multi_model_structure,
        "metal_model_index": metal.model_index if metal else "",
        "metal_chain_index": metal.output_chain_index if metal else "",
        "metal_residue_index": metal.output_residue_index if metal else "",
        "metal_atom_index": metal.atom_index if metal else "",
        "metal_resname": metal.residue_name if metal else "",
        "metal_chain": metal.chain_id if metal else "",
        "metal_resnum": metal.resnum if metal else "",
        "metal_atom": metal.atom_name if metal else "",
        "metal_element": metal.element if metal else "",
        "metal_icode": metal.insertion_code if metal else "",
        "metal_altloc": metal.altloc if metal else "",
        "metal_occupancy": metal.occupancy if metal else NAN,
        "metal_occupancy_valid": metal.occupancy_valid if metal else "",
        "metal_occupancy_status": metal.occupancy_status if metal else "",
        "metal_conformer_mean_occupancy": (
            residue.selected_conformer_mean_occupancy if residue else NAN
        ),
        "metal_altloc_options": residue.altloc_options if residue else "",
        "alternative_conformers_present": (
            residue.alternative_conformers_present if residue else ""
        ),
        "altloc_selection_fallback": (
            residue.altloc_selection_fallback if residue else ""
        ),
        "symmetry_search_available": structure.symmetry_search_available,
        "symmetry_search_failure_reason": structure.symmetry_search_failure_reason,
        "strict_ncs_operation_count": structure.strict_ncs_operation_count,
        "crystallographic_operation_count": (
            structure.crystallographic_operation_count
        ),
        "dpi_atom_count_multiplier": structure.dpi_atom_count_multiplier,
        "occupancy_validation_failed": structure.occupancy_validation_failed,
        "missing_occupancy_count": structure.missing_occupancy_count,
        "invalid_occupancy_count": structure.invalid_occupancy_count,
        "zero_occupancy_atom_count": structure.zero_occupancy_atom_count,
        "duplicate_atom_records_present": structure.duplicate_atom_records_present,
        "duplicate_atom_record_count": structure.duplicate_atom_record_count,
        "duplicate_atom_coordinate_conflict_count": (
            structure.duplicate_coordinate_conflict_count
        ),
        "malformed_duplicate_atom_name_count": (
            structure.malformed_duplicate_atom_name_count
        ),
        "raw_occupancy_mapping_failed": structure.raw_occupancy_mapping_failed,
        "raw_occupancy_mapping_failure_reason": (
            structure.raw_occupancy_mapping_failure_reason
        ),
        "unknown_element_atom_count": structure.unknown_element_atom_count,
        "element_validation_warning": structure.element_validation_warning,
        "zscore_outlier_cutoff": ZSCORE_OUTLIER_CUTOFF,
    }
    for column in STATS_EXTRA_COLUMNS:
        values.setdefault(column, summary.get(column, ""))
    values.update(
        {key: value for key, value in summary.items() if key in STATS_EXTRA_COLUMNS}
    )
    return values


def _bond_row(pdb_id, structure, metal, contact, dpi, resolution, sigma, parent_type):
    neighbor = contact["neighbor"]
    metal_residue = structure.residue_for_atom(metal)
    neighbor_residue = structure.residue_for_atom(neighbor)
    x, y, z = contact["transformed_position"]
    tx, ty, tz = contact["translation"]
    mag, neg, pos = sigma
    connection_values = _connection_output_values(contact)
    donor_values = _donor_output_values(contact)
    context_values = _context_warning_values(contact)
    return {
        "pdbID": pdb_id,
        "metal_resname": metal.residue_name,
        "metal_chain": metal.chain_id,
        "metal_resnum": metal.resnum,
        "metal_element": metal.element,
        "neighbor_resname": neighbor.residue_name,
        "neighbor_atom": neighbor.atom_name,
        "neighbor_element": neighbor.element,
        "distance": contact["distance"],
        **connection_values,
        **donor_values,
        **context_values,
        "literature_distance": contact["literature_distance"],
        "literature_stdev": contact["literature_stdev"],
        "zscore": contact["zscore"],
        "dpi": dpi,
        "resolution": resolution,
        "sigma_mag": mag,
        "sigma_neg": neg,
        "sigma_pos": pos,
        "parent_type": parent_type,
        "bonded_to": _bonded_to(neighbor.is_water),
        "model_id": structure.analyzed_model_id,
        "metal_model_index": metal.model_index,
        "metal_chain_index": metal.output_chain_index,
        "metal_residue_index": metal.output_residue_index,
        "metal_atom_index": metal.atom_index,
        "metal_atom": metal.atom_name,
        "metal_icode": metal.insertion_code,
        "metal_altloc": metal.altloc,
        "metal_occupancy": metal.occupancy,
        "metal_occupancy_valid": metal.occupancy_valid,
        "metal_occupancy_status": metal.occupancy_status,
        "metal_conformer_mean_occupancy": (
            metal_residue.selected_conformer_mean_occupancy
        ),
        "metal_altloc_options": metal_residue.altloc_options,
        "metal_altloc_selection_fallback": (metal_residue.altloc_selection_fallback),
        "neighbor_chain": neighbor.chain_id,
        "neighbor_resnum": neighbor.resnum,
        "neighbor_icode": neighbor.insertion_code,
        "neighbor_model_index": neighbor.model_index,
        "neighbor_chain_index": neighbor.output_chain_index,
        "neighbor_residue_index": neighbor.output_residue_index,
        "neighbor_atom_index": neighbor.atom_index,
        "neighbor_altloc": neighbor.altloc,
        "neighbor_occupancy": neighbor.occupancy,
        "neighbor_occupancy_valid": neighbor.occupancy_valid,
        "neighbor_occupancy_status": neighbor.occupancy_status,
        "neighbor_conformer_mean_occupancy": (
            neighbor_residue.selected_conformer_mean_occupancy
        ),
        "neighbor_altloc_options": neighbor_residue.altloc_options,
        "neighbor_altloc_selection_fallback": (
            neighbor_residue.altloc_selection_fallback
        ),
        "alternative_conformers_present": (
            metal_residue.alternative_conformers_present
            or neighbor_residue.alternative_conformers_present
        ),
        "altloc_selection_fallback": (
            metal_residue.altloc_selection_fallback
            or neighbor_residue.altloc_selection_fallback
        ),
        "neighbor_class": "water" if neighbor.is_water else "amino_acid",
        "candidate_contact": True,
        "reference_covered": contact["reference_covered"],
        "geometry_outlier": contact["geometry_outlier"],
        "geometry_consistent": contact["geometry_consistent"],
        "multi_donor_detected": contact["multi_donor_detected"],
        "multi_donor_contact_count": contact["multi_donor_contact_count"],
        "multi_donor_geometry_status": (contact["multi_donor_geometry_status"]),
        "multi_donor_contains_suspect_bond": (
            contact["multi_donor_contains_suspect_bond"]
        ),
        "score_eligible": contact["score_eligible"],
        "score_exclusion_reason": contact["score_exclusion_reason"],
        "zscore_outlier_cutoff": ZSCORE_OUTLIER_CUTOFF,
        "contact_scope": contact["contact_scope"],
        "symmetry_contact": contact["symmetry_contact"],
        "crystallographic_contact": contact["crystallographic_contact"],
        "strict_ncs_contact": contact["strict_ncs_contact"],
        "strict_ncs_operation_id": contact["strict_ncs_operation_id"],
        "symmetry_image_index": contact["symmetry_image_index"],
        "symmetry_operation": contact["symmetry_operation"],
        "cell_translation_x": tx,
        "cell_translation_y": ty,
        "cell_translation_z": tz,
        "transformed_neighbor_x": round(x, 6),
        "transformed_neighbor_y": round(y, 6),
        "transformed_neighbor_z": round(z, 6),
    }


def _candidate_row(pdb_id, structure, metal, candidate):
    """Return one discovered or declared candidate with full provenance."""
    neighbor = candidate["neighbor"]
    x, y, z = candidate["transformed_position"]
    tx, ty, tz = candidate["translation"]
    connection_values = _connection_output_values(candidate)
    donor_values = _donor_output_values(candidate)
    context_values = _context_warning_values(candidate, include_proximal=True)
    return {
        "pdbID": pdb_id,
        "candidate_source": "|".join(sorted(candidate["candidate_sources"])),
        "eligibility_status": candidate["eligibility_status"],
        "eligibility_reason": candidate["eligibility_reason"],
        "first_sphere_eligible": candidate["first_sphere_eligible"],
        "candidate_distance": round(candidate["distance_raw"], 3),
        "assignment_target": candidate["assignment_target"],
        "assignment_tolerance": candidate["assignment_tolerance"],
        "first_sphere_cutoff": candidate["first_sphere_cutoff"],
        "assignment_reference_kind": candidate["assignment_reference_kind"],
        "assignment_reference": candidate["assignment_reference"],
        "inferred_contact_eligible": candidate["inferred_contact_eligible"],
        **donor_values,
        **context_values,
        **connection_values,
        "metal_resname": metal.residue_name,
        "metal_chain": metal.chain_id,
        "metal_resnum": metal.resnum,
        "metal_element": metal.element,
        "metal_atom": metal.atom_name,
        "metal_icode": metal.insertion_code,
        "metal_altloc": metal.altloc,
        "metal_occupancy": metal.occupancy,
        "model_id": structure.analyzed_model_id,
        "metal_model_index": metal.model_index,
        "metal_chain_index": metal.output_chain_index,
        "metal_residue_index": metal.output_residue_index,
        "metal_atom_index": metal.atom_index,
        "neighbor_resname": neighbor.residue_name,
        "neighbor_chain": neighbor.chain_id,
        "neighbor_resnum": neighbor.resnum,
        "neighbor_atom": neighbor.atom_name,
        "neighbor_element": neighbor.element,
        "neighbor_icode": neighbor.insertion_code,
        "neighbor_altloc": neighbor.altloc,
        "neighbor_occupancy": neighbor.occupancy,
        "neighbor_class": "water" if neighbor.is_water else "amino_acid",
        "neighbor_model_index": neighbor.model_index,
        "neighbor_chain_index": neighbor.output_chain_index,
        "neighbor_residue_index": neighbor.output_residue_index,
        "neighbor_atom_index": neighbor.atom_index,
        "contact_scope": candidate["contact_scope"],
        "symmetry_contact": candidate["symmetry_contact"],
        "crystallographic_contact": candidate["crystallographic_contact"],
        "strict_ncs_contact": candidate["strict_ncs_contact"],
        "strict_ncs_operation_id": candidate["strict_ncs_operation_id"],
        "symmetry_image_index": candidate["symmetry_image_index"],
        "symmetry_operation": candidate["symmetry_operation"],
        "cell_translation_x": tx,
        "cell_translation_y": ty,
        "cell_translation_z": tz,
        "transformed_neighbor_x": round(x, 6),
        "transformed_neighbor_y": round(y, 6),
        "transformed_neighbor_z": round(z, 6),
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
    metadata = {
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
    declared_by_metal = {}
    for candidate in declared_candidates:
        declared_by_metal.setdefault(candidate["metal"].source_key, []).append(
            candidate
        )

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

    rows = []
    candidate_rows = []
    summaries = {}
    for metal in metals_in_model:
        metal_declarations = declared_by_metal.get(metal.source_key, ())
        explicit_declarations = [
            candidate
            for candidate in metal_declarations
            if not candidate["symmetry_contact"]
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
