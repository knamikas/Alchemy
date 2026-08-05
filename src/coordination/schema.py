"""The three CSV schemas Alchemy publishes, and the rows written against them.

This is the output contract: ``metal_bonds_all.csv`` carries assigned
metal-donor contacts, ``metal_candidates_all.csv`` the evidence behind them,
and ``metal_stats_all.csv`` the EDSTATS table with per-site columns appended.
Rows are written by projecting them onto their column list, so a key a builder
gained without a matching column would be dropped in silence and one it lost
would surface downstream as a bare ``KeyError``. ``_check_row_schema`` catches
both.

The builders are serialization only: every decision they report was made in
``coordination.analysis`` before the row is built.
"""

from typing import Any, Mapping, Optional

import gemmi

from structure_analysis import NAN


# Cutoff for a reference-covered geometry outlier. Both CSVs publish it as a
# column, so a consumer reading an old file can see which threshold produced
# its `geometry_outlier` values.
ZSCORE_OUTLIER_CUTOFF = 6.0


# The driver's writers import these column lists, so the row builders and the
# CSV header cannot drift apart. "candidate" field names are kept for CSV
# compatibility and describe inferred first-sphere or source-declared contacts.
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


# No z-score or geometry classification here: failing first-sphere eligibility
# does not establish that an atom is chemically nonbonded, and a source
# declaration may supersede the proximity-only result.
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
    "metal_coordinates_valid",
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
    "overfull_occupancy_site_count",
    "overfull_occupancy_excess",
    "defaulted_occupancy_atom_count",
    "zero_occupancy_atom_count",
    "metal_overfull_occupancy",
    "geometry_not_assessed_reason",
    "duplicate_atom_records_present",
    "duplicate_atom_record_count",
    "duplicate_atom_coordinate_conflict_count",
    "malformed_duplicate_atom_name_count",
    "raw_occupancy_mapping_failed",
    "raw_occupancy_mapping_failure_reason",
    "unknown_element_atom_count",
    "element_validation_warning",
    "non_finite_coordinate_atom_count",
    "zscore_outlier_cutoff",
]


def _check_row_schema(row, columns, name):
    """Fail loudly when a row builder and its CSV schema have drifted apart."""
    expected = set(columns)
    if row.keys() == expected:
        return
    details = []
    missing = sorted(expected - row.keys())
    unexpected = sorted(row.keys() - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise RuntimeError(
        f"{name} row does not match its column schema: " + "; ".join(details)
    )


def _bonded_to(is_water=False):
    return "HOH" if is_water else "P"


def _connection_output_values(candidate):
    records = candidate.declared_connections
    inferred = bool(candidate.inferred_contact_eligible)

    def joined(name):
        values = []
        for record in records:
            value = record.get(name, "")
            text = "" if value is None else str(value)
            values.append("" if text in ("", "nan") else text)
        return "|".join(values)

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
        "inferred_donor_allowed": candidate.inferred_donor_allowed,
        "inferred_donor_rule": candidate.inferred_donor_rule,
        "donor_rule_override": candidate.donor_rule_override,
    }


def _neighbor_class(neighbor):
    if neighbor.is_water:
        return "water"
    residue_info = gemmi.find_tabulated_residue(neighbor.residue_name)
    if residue_info.is_nucleic_acid():
        return "nucleotide"
    if residue_info.is_amino_acid():
        return "amino_acid"
    return "other"


def _context_warning_values(candidate, include_proximal=False):
    """Return the context flag and the reason codes behind it."""
    reasons = []
    if candidate.neighbor.occupancy_valid and candidate.neighbor.occupancy == 0.0:
        reasons.append("zero_occupancy_neighbor")
    if not candidate.inferred_donor_allowed:
        if candidate.declared_connections:
            reasons.append("declared_non_typical_donor")
        elif candidate.first_sphere_eligible:
            reasons.append("non_typical_first_sphere_candidate")
        elif include_proximal:
            reasons.append("non_typical_proximal_candidate")
    if candidate.multi_donor_contains_suspect_bond:
        reasons.append("suspect_multi_donor_group")
    reasons = list(dict.fromkeys(reasons))
    return {
        "context_warning": bool(reasons),
        "context_warning_reasons": "|".join(reasons),
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
        "metal_coordinates_valid": metal.coordinates_valid if metal else "",
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
        "overfull_occupancy_site_count": structure.overfull_occupancy_site_count,
        "overfull_occupancy_excess": structure.overfull_occupancy_excess,
        "defaulted_occupancy_atom_count": structure.defaulted_occupancy_atom_count,
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
        "non_finite_coordinate_atom_count": (
            structure.non_finite_coordinate_atom_count
        ),
        "zscore_outlier_cutoff": ZSCORE_OUTLIER_CUTOFF,
    }
    for column in STATS_EXTRA_COLUMNS:
        values.setdefault(column, summary.get(column, ""))
    values.update(
        {key: value for key, value in summary.items() if key in STATS_EXTRA_COLUMNS}
    )
    return values


def _bond_row(pdb_id, structure, metal, contact, dpi, resolution, sigma, parent_type):
    neighbor = contact.neighbor
    metal_residue = structure.residue_for_atom(metal)
    neighbor_residue = structure.residue_for_atom(neighbor)
    x, y, z = contact.transformed_position
    tx, ty, tz = contact.translation
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
        "distance": contact.distance,
        **connection_values,
        **donor_values,
        **context_values,
        "literature_distance": contact.literature_distance,
        "literature_stdev": contact.literature_stdev,
        "zscore": contact.zscore,
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
        "neighbor_class": _neighbor_class(neighbor),
        "candidate_contact": True,
        "reference_covered": contact.reference_covered,
        "geometry_outlier": contact.geometry_outlier,
        "geometry_consistent": contact.geometry_consistent,
        "multi_donor_detected": contact.multi_donor_detected,
        "multi_donor_contact_count": contact.multi_donor_contact_count,
        "multi_donor_geometry_status": (contact.multi_donor_geometry_status),
        "multi_donor_contains_suspect_bond": (
            contact.multi_donor_contains_suspect_bond
        ),
        "score_eligible": contact.score_eligible,
        "score_exclusion_reason": contact.score_exclusion_reason,
        "zscore_outlier_cutoff": ZSCORE_OUTLIER_CUTOFF,
        "contact_scope": contact.contact_scope,
        "symmetry_contact": contact.symmetry_contact,
        "crystallographic_contact": contact.crystallographic_contact,
        "strict_ncs_contact": contact.strict_ncs_contact,
        "strict_ncs_operation_id": contact.strict_ncs_operation_id,
        "symmetry_image_index": contact.symmetry_image_index,
        "symmetry_operation": contact.symmetry_operation,
        "cell_translation_x": tx,
        "cell_translation_y": ty,
        "cell_translation_z": tz,
        "transformed_neighbor_x": round(x, 6),
        "transformed_neighbor_y": round(y, 6),
        "transformed_neighbor_z": round(z, 6),
    }


def _candidate_row(pdb_id, structure, metal, candidate):
    """Return one discovered or declared candidate as a candidate CSV row."""
    neighbor = candidate.neighbor
    x, y, z = candidate.transformed_position
    tx, ty, tz = candidate.translation
    connection_values = _connection_output_values(candidate)
    donor_values = _donor_output_values(candidate)
    context_values = _context_warning_values(candidate, include_proximal=True)
    return {
        "pdbID": pdb_id,
        "candidate_source": "|".join(sorted(candidate.candidate_sources)),
        "eligibility_status": candidate.eligibility_status,
        "eligibility_reason": candidate.eligibility_reason,
        "first_sphere_eligible": candidate.first_sphere_eligible,
        "candidate_distance": round(candidate.distance_raw, 3),
        "assignment_target": candidate.assignment_target,
        "assignment_tolerance": candidate.assignment_tolerance,
        "first_sphere_cutoff": candidate.first_sphere_cutoff,
        "assignment_reference_kind": candidate.assignment_reference_kind,
        "assignment_reference": candidate.assignment_reference,
        "inferred_contact_eligible": candidate.inferred_contact_eligible,
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
        "neighbor_class": _neighbor_class(neighbor),
        "neighbor_model_index": neighbor.model_index,
        "neighbor_chain_index": neighbor.output_chain_index,
        "neighbor_residue_index": neighbor.output_residue_index,
        "neighbor_atom_index": neighbor.atom_index,
        "contact_scope": candidate.contact_scope,
        "symmetry_contact": candidate.symmetry_contact,
        "crystallographic_contact": candidate.crystallographic_contact,
        "strict_ncs_contact": candidate.strict_ncs_contact,
        "strict_ncs_operation_id": candidate.strict_ncs_operation_id,
        "symmetry_image_index": candidate.symmetry_image_index,
        "symmetry_operation": candidate.symmetry_operation,
        "cell_translation_x": tx,
        "cell_translation_y": ty,
        "cell_translation_z": tz,
        "transformed_neighbor_x": round(x, 6),
        "transformed_neighbor_y": round(y, 6),
        "transformed_neighbor_z": round(z, 6),
    }
