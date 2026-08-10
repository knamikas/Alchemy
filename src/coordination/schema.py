"""The three CSV schemas Alchemy publishes, and the rows written against them.

This is the output contract: ``metal_bonds_all.csv`` carries assigned
metal-donor contacts, ``metal_contact_candidates_all.csv`` the evidence behind
them, and ``metal_sites_all.csv`` the EDSTATS table with per-site columns
appended.
Rows are written by projecting them onto their column list, so a key a builder
gained without a matching column would be dropped in silence and one it lost
would surface downstream as a bare ``KeyError``. ``check_row_schema`` catches
both.

The builders are serialization only: every decision they report was made in
``coordination.analysis`` before the row is built.
"""

import hashlib
from typing import Any, ClassVar
from collections.abc import Iterable, Iterator, Mapping

import gemmi
from typing_extensions import override

from coordination.contact_record import Candidate, MultiDonorResult
from output_rows import CsvValue
from structure_analysis import NAN, AtomSite, StructureContext


# Cutoff for a reference-covered geometry outlier. Both CSVs publish it as a
# column, so a consumer reading an old file can see which threshold produced
# its `geometry_outlier` values.
ZSCORE_OUTLIER_CUTOFF = 6.0


# The driver's writers import these column lists, so the row builders and the
# CSV header cannot drift apart. "candidate" field names are kept for CSV
# compatibility and describe inferred first-sphere or source-declared contacts.
BOND_COLUMNS = [
    "pdbID",
    "metal_site_id",
    "contact_id",
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
    "metal_site_id",
    "contact_id",
    "assigned_as_bond",
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


# Appended after the dynamic EDSTATS header in metal_sites_all.csv.
STATS_EXTRA_COLUMNS = [
    "metal_site_id",
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
    "r_free",
    "reflection_count",
    "asu_volume",
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


def check_row_schema(row: Mapping[str, Any], columns: Iterable[str], name: str) -> None:
    """Fail loudly when a row builder and its CSV schema have drifted apart."""
    expected = set(columns)
    if set(row) == expected:
        return
    details: list[str] = []
    missing = sorted(expected - row.keys())
    unexpected = sorted(row.keys() - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise RuntimeError(
        f"{name} row does not match its column schema: " + "; ".join(details)
    )


class _CsvRow(Mapping[str, Any]):
    __slots__ = ("_values",)

    columns: ClassVar[tuple[str, ...]] = ()
    indices: ClassVar[dict[str, int]] = {}
    schema_name: ClassVar[str] = "CSV"

    def __init__(self, values: Mapping[str, Any]) -> None:
        check_row_schema(values, self.columns, self.schema_name)
        ordered: list[CsvValue] = []
        for column in self.columns:
            value = values[column]
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"{column} is not a CSV scalar: {type(value).__name__}")
            ordered.append(value)
        self._values = tuple(ordered)

    @override
    def __getitem__(self, key: str) -> Any:
        return self._values[self.indices[key]]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    @override
    def __len__(self) -> int:
        return len(self.columns)

    def as_dict(self) -> dict[str, CsvValue]:
        return dict(zip(self.columns, self._values, strict=True))


class BondRow(_CsvRow):
    columns = tuple(BOND_COLUMNS)
    indices = {column: index for index, column in enumerate(columns)}
    schema_name = "metal_bonds_all.csv"


class CandidateRow(_CsvRow):
    columns = tuple(CANDIDATE_COLUMNS)
    indices = {column: index for index, column in enumerate(columns)}
    schema_name = "metal_contact_candidates_all.csv"


def metal_site_identifier(pdb_id: str, metal: AtomSite) -> str:
    return (
        f"{pdb_id.lower()}:m{metal.model_index}:c{metal.output_chain_index}:"
        f"r{metal.output_residue_index}:a{metal.atom_index}"
    )


def contact_identifier(pdb_id: str, metal: AtomSite, contact: Candidate) -> str:
    neighbor = contact.neighbor
    identity = (
        neighbor.model_index,
        neighbor.output_chain_index,
        neighbor.output_residue_index,
        neighbor.atom_index,
        contact.contact_scope,
        contact.strict_ncs_operation_id,
        contact.symmetry_image_index,
        contact.symmetry_operation,
        *contact.translation,
    )
    # A fixed-width digest keeps joins manageable while retaining every field
    # that distinguishes generated images of the same deposited donor atom.
    payload = "\x1f".join(str(value) for value in identity).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{metal_site_identifier(pdb_id, metal)}:c{digest}"


def _bonded_to(is_water: bool = False) -> str:
    return "HOH" if is_water else "P"


def _connection_output_values(candidate: Candidate) -> dict[str, str | bool]:
    records = candidate.declared_connections
    inferred = candidate.eligibility().inferred_contact_eligible

    def joined(values: Iterable[object]) -> str:
        rendered: list[str] = []
        for value in values:
            text = "" if value is None else str(value)
            rendered.append("" if text in ("", "nan") else text)
        return "|".join(rendered)

    return {
        "coordination_status": (
            "declared" if records else ("inferred" if inferred else "unassigned")
        ),
        "coordination_source": (
            joined(record["source"] for record in records)
            if records
            else ("proximity_rule" if inferred else "")
        ),
        "declared_connection": bool(records),
        "connection_id": joined(record["connection_id"] for record in records),
        "connection_type": joined(record["connection_type"] for record in records),
        "connection_link_id": joined(
            record["connection_link_id"] for record in records
        ),
        "connection_asu": joined(record["connection_asu"] for record in records),
        "connection_reported_distance": joined(
            record["connection_reported_distance"] for record in records
        ),
    }


def _donor_output_values(candidate: Candidate) -> dict[str, bool | str | None]:
    policy = candidate.donor_policy()
    return {
        "inferred_donor_allowed": policy.inferred_allowed,
        "inferred_donor_rule": policy.rule,
        "donor_rule_override": policy.override,
    }


def _neighbor_class(neighbor: AtomSite) -> str:
    if neighbor.is_water:
        return "water"
    residue_info = gemmi.find_tabulated_residue(neighbor.residue_name)
    if residue_info.is_nucleic_acid():
        return "nucleotide"
    if residue_info.is_amino_acid():
        return "amino_acid"
    return "other"


def context_warning_values(
    candidate: Candidate,
    include_proximal: bool = False,
    multi_donor: MultiDonorResult | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    policy = candidate.donor_policy()
    eligibility = candidate.eligibility()
    if candidate.neighbor.occupancy_valid and candidate.neighbor.occupancy == 0.0:
        reasons.append("zero_occupancy_neighbor")
    if not policy.inferred_allowed:
        if candidate.declared_connections:
            reasons.append("declared_non_typical_donor")
        elif eligibility.first_sphere_eligible:
            reasons.append("non_typical_first_sphere_candidate")
        elif include_proximal:
            reasons.append("non_typical_proximal_candidate")
    if multi_donor is not None and multi_donor.contains_suspect_bond:
        reasons.append("suspect_multi_donor_group")
    reasons = list(dict.fromkeys(reasons))
    return {
        "context_warning": bool(reasons),
        "context_warning_reasons": "|".join(reasons),
    }


def stats_extra_values(
    pdb_id: str,
    structure: StructureContext,
    metal: AtomSite | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fixed per-site values appended to an EDSTATS row."""
    summary = summary or {}
    residue = structure.residue_for_atom(metal) if metal is not None else None
    values: dict[str, Any] = {
        "metal_site_id": metal_site_identifier(pdb_id, metal) if metal else "",
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


def bond_row(
    pdb_id: str,
    structure: StructureContext,
    metal: AtomSite,
    contact: Candidate,
    dpi: float,
    resolution: float,
    sigma: tuple[float, float, float],
    parent_type: str,
) -> BondRow:
    neighbor = contact.neighbor
    metal_residue = structure.residue_for_atom(metal)
    neighbor_residue = structure.residue_for_atom(neighbor)
    x, y, z = contact.transformed_position
    tx, ty, tz = contact.translation
    mag, neg, pos = sigma
    connection_values = _connection_output_values(contact)
    donor_values = _donor_output_values(contact)
    geometry = contact.geometry()
    multi_donor = contact.multi_donor()
    context_values = context_warning_values(contact, multi_donor=multi_donor)
    return BondRow(
        {
            "pdbID": pdb_id,
            "metal_site_id": metal_site_identifier(pdb_id, metal),
            "contact_id": contact_identifier(pdb_id, metal, contact),
            "metal_resname": metal.residue_name,
            "metal_chain": metal.chain_id,
            "metal_resnum": metal.resnum,
            "metal_element": metal.element,
            "neighbor_resname": neighbor.residue_name,
            "neighbor_atom": neighbor.atom_name,
            "neighbor_element": neighbor.element,
            "distance": geometry.distance,
            **connection_values,
            **donor_values,
            **context_values,
            "literature_distance": geometry.literature_distance,
            "literature_stdev": geometry.literature_stdev,
            "zscore": geometry.zscore,
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
            "metal_altloc_selection_fallback": (
                metal_residue.altloc_selection_fallback
            ),
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
            "reference_covered": geometry.reference_covered,
            "geometry_outlier": geometry.outlier,
            "geometry_consistent": geometry.consistent,
            "multi_donor_detected": multi_donor.detected,
            "multi_donor_contact_count": multi_donor.contact_count,
            "multi_donor_geometry_status": multi_donor.geometry_status,
            "multi_donor_contains_suspect_bond": multi_donor.contains_suspect_bond,
            "score_eligible": multi_donor.score_eligible,
            "score_exclusion_reason": multi_donor.score_exclusion_reason,
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
    )


def candidate_row(
    pdb_id: str,
    structure: StructureContext,
    metal: AtomSite,
    candidate: Candidate,
    *,
    assigned_as_bond: bool,
) -> CandidateRow:
    """Return one discovered or declared candidate as a candidate CSV row."""
    neighbor = candidate.neighbor
    x, y, z = candidate.transformed_position
    tx, ty, tz = candidate.translation
    connection_values = _connection_output_values(candidate)
    donor_values = _donor_output_values(candidate)
    context_values = context_warning_values(candidate, include_proximal=True)
    eligibility = candidate.eligibility()
    return CandidateRow(
        {
            "pdbID": pdb_id,
            "metal_site_id": metal_site_identifier(pdb_id, metal),
            "contact_id": contact_identifier(pdb_id, metal, candidate),
            "assigned_as_bond": assigned_as_bond,
            "candidate_source": "|".join(sorted(candidate.candidate_sources)),
            "eligibility_status": eligibility.status,
            "eligibility_reason": eligibility.reason,
            "first_sphere_eligible": eligibility.first_sphere_eligible,
            "candidate_distance": round(candidate.distance_raw, 3),
            "assignment_target": eligibility.assignment_target,
            "assignment_tolerance": eligibility.assignment_tolerance,
            "first_sphere_cutoff": eligibility.first_sphere_cutoff,
            "assignment_reference_kind": eligibility.assignment_reference_kind,
            "assignment_reference": eligibility.assignment_reference,
            "inferred_contact_eligible": eligibility.inferred_contact_eligible,
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
    )
