"""Identify metal-donor contacts and assess their bond distances.

Combine proximity candidates within 4 A with source struct_conn and LINK
declarations, then determine first-sphere eligibility and score geometry:

    z = (d_observed - mu) / sqrt(DPI**2 + sigma_lit**2)

Distances come from Harding (2006) and Zheng et al. (2008, Ni); DPI follows
Blow (2002), equation 7. Missing scoring inputs retain measured geometry
with NaN derived values. See docs/method.md for the scientific policy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from statistics import median
from typing import Any, cast

import gemmi

from codes import (
    CandidateSource,
    ContactScope,
    EligibilityReason,
    EligibilityStatus,
    GeometryStatus,
    MultiDonorStatus,
    ParentType,
    ReasonCode,
    ReferenceKind,
)
from coordination.contact_record import (
    ZSCORE_OUTLIER_CUTOFF,
    Candidate,
    DonorPolicy,
    EligibilityResult,
    GeometryResult,
    MultiDonorResult,
)
from coordination.declared_connections import collect_declared_candidates
from coordination.donor_chemistry import (
    AA,
    C_TERMINAL_DONOR_ATOMS,
    DONOR_ELEMENTS,
    INFERRED_DONOR_ATOMS,
    N_TERMINAL_DONOR_ATOMS,
)
from coordination.dpi import DpiComponents, DpiInputs, calculate_dpi_components
from coordination.schema import (
    STATS_EXTRA_COLUMNS,
    BondRow,
    CandidateRow,
    bond_row,
    candidate_row,
    contact_identifier,
    metal_site_identifier,
)
from metal_elements import METAL_ELEMENTS
from output_rows import CsvValue, MetalStatsRow
from reference_data import (
    cluster_ids,
    first_sphere_targets,
    heme_ids,
    literature_distances,
)
from structure_analysis import (
    NAN,
    AtomSite,
    ContactImage,
    StructureContext,
    count_deposited_ni,
    count_ni,
    load_structure,
    position_distance,
    spacegroup_or_none,
)

CANDIDATE_SEARCH_RADIUS = 4.0
# Context-only radius for nearby modeled metals. This does not assert a
# metal-metal bond or define a multinuclear active site.
NEARBY_METAL_RADIUS = 6.0
SEARCH_EPSILON = 1e-6
# The proximity accept filter is deliberately tighter than the neighbor-search
# radius pad above: the search casts CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON
# and the filter keeps CANDIDATE_SEARCH_RADIUS + CANDIDATE_ACCEPT_EPSILON.
# Changing this value admits contacts between 4 + 1e-9 and 4 + 1e-6 A.
CANDIDATE_ACCEPT_EPSILON = 1e-9

# First-sphere definition: donor distance <= target distance + 0.75 A.
# Harding, M. M. (2004), Acta Cryst. D60, 849-859.
# https://doi.org/10.1107/S0907444904004081
FIRST_SPHERE_TOLERANCE = 0.75
# The published reason strings embed the tolerance; fail at import if they drift.
if not (
    EligibilityReason.DISTANCE_WITHIN_TOLERANCE.endswith(f"{FIRST_SPHERE_TOLERANCE:g}")
    and EligibilityReason.DISTANCE_EXCEEDS_TOLERANCE.endswith(
        f"{FIRST_SPHERE_TOLERANCE:g}"
    )
):
    raise ValueError(
        "EligibilityReason distance values must end in FIRST_SPHERE_TOLERANCE"
    )

# Gemmi's ContactSearch uses 0.8 A by default to distinguish near-coincident
# symmetry images of an atom intended to occupy a special position.
# NeighborSearch returns those images unfiltered, so apply the cutoff here.
SPECIAL_POSITION_DEDUP_CUTOFF = 0.8

# Coordinate occupancies are commonly rounded in PDB/mmCIF output, so allow a
# small absolute tolerance when comparing them with 1 / site-symmetry order.
SPECIAL_POSITION_OCCUPANCY_TOLERANCE = 0.015

#: One deposited atom record, as ``AtomSite.source_key`` reports it.
AtomKey = tuple[int, int, int, int]

#: What makes two candidate records the same atom image around one metal.
_CandidateIdentity = tuple[AtomKey, str, tuple[int, int, int], tuple[float, ...]]

#: What makes two contacts share one donor-residue image around one metal.
_ResidueImageKey = tuple[
    tuple[int, int, int], ContactScope, str, int, str, tuple[int, int, int]
]

#: The per-site summary's share of ``STATS_EXTRA_COLUMNS``. Every assembled
#: site summary must cover exactly these; ``stats_extra_values`` in schema.py
#: fills the rest from the structure, the metal, and the density row.
SUMMARY_OWNED_STATS_EXTRA_COLUMNS: frozenset[str] = frozenset(
    (
        "metal_special_position",
        "metal_site_symmetry_order",
        "metal_expected_crystallographic_occupancy",
        "metal_occupancy_matches_site_symmetry",
        "entry_nonwater_median_b_iso",
        "donor_b_iso_count",
        "donor_median_b_iso",
        "metal_donor_b_ratio",
        "metal_minus_donor_b_iso",
        "metal_donor_b_similarity",
        "nearest_metal_distance",
        "nearest_metal_element",
        "nearest_metal_site_id",
        "nearby_metal_count_6a",
        "dpi",
        "resolution",
        "r_free",
        "reflection_count",
        "asu_volume",
        "occupancy_weighted_atom_count",
        "deposited_occupancy_weighted_atom_count",
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
        "metal_overfull_occupancy",
        "geometry_not_assessed_reason",
    )
)
if not SUMMARY_OWNED_STATS_EXTRA_COLUMNS.issubset(STATS_EXTRA_COLUMNS):
    raise ValueError(
        "SUMMARY_OWNED_STATS_EXTRA_COLUMNS names columns outside STATS_EXTRA_COLUMNS: "
        + ", ".join(
            sorted(SUMMARY_OWNED_STATS_EXTRA_COLUMNS - frozenset(STATS_EXTRA_COLUMNS))
        )
    )


def _check_site_summary_columns(summary: Mapping[str, Any]) -> None:
    """Fail loudly when a site summary drifts from its declared column set.

    ``stats_extra_values`` would otherwise blank a missing column or let a
    stray key overwrite a value it computed itself.
    """
    keys = frozenset(summary)
    if keys == SUMMARY_OWNED_STATS_EXTRA_COLUMNS:
        return
    details: list[str] = []
    missing = sorted(SUMMARY_OWNED_STATS_EXTRA_COLUMNS - keys)
    unexpected = sorted(keys - SUMMARY_OWNED_STATS_EXTRA_COLUMNS)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ValueError(
        "site summary does not match SUMMARY_OWNED_STATS_EXTRA_COLUMNS: "
        + "; ".join(details)
    )


def _metal_proximity_summaries(
    pdb_id: str, metals: Sequence[AtomSite]
) -> dict[AtomKey, dict[str, Any]]:
    """Summarize other modeled metals without crystallographic expansion.

    ``metals`` is the canonical analyzed-model selection, which already omits
    explicitly zero-occupancy sites. Non-finite coordinates remain in the result
    with unavailable proximity fields so their site rows keep a fixed schema.
    """
    unavailable: dict[str, Any] = {
        "nearest_metal_distance": NAN,
        "nearest_metal_element": "",
        "nearest_metal_site_id": "",
        "nearby_metal_count_6a": NAN,
    }
    summaries: dict[AtomKey, dict[str, Any]] = {
        metal.source_key: dict(unavailable) for metal in metals
    }
    spatial = [metal for metal in metals if metal.coordinates_valid]
    for metal in spatial:
        neighbors = sorted(
            (
                (
                    position_distance(metal.xyz, neighbor.xyz),
                    neighbor.source_key,
                    neighbor,
                )
                for neighbor in spatial
                if neighbor.source_key != metal.source_key
            ),
            key=lambda item: (item[0], item[1]),
        )
        summary = summaries[metal.source_key]
        summary["nearby_metal_count_6a"] = sum(
            distance <= NEARBY_METAL_RADIUS + SEARCH_EPSILON
            for distance, _, _ in neighbors
        )
        if neighbors:
            distance, _, nearest = neighbors[0]
            summary.update(
                {
                    "nearest_metal_distance": round(distance, 3),
                    "nearest_metal_element": nearest.element,
                    "nearest_metal_site_id": metal_site_identifier(pdb_id, nearest),
                }
            )
    return summaries


def _metal_special_position_summaries(
    structure: StructureContext, metals: Sequence[AtomSite]
) -> dict[AtomKey, dict[str, Any]]:
    """Report crystallographic site symmetry and its occupancy expectation.

    A fresh Gemmi cell is populated with space-group images only.  This keeps
    strict-NCS transforms, which are also present in ``structure.cell`` after
    ``setup_cell_images()``, from being mistaken for crystallographic site
    symmetry.
    """
    unavailable: dict[str, Any] = {
        "metal_special_position": "",
        "metal_site_symmetry_order": NAN,
        "metal_expected_crystallographic_occupancy": NAN,
        "metal_occupancy_matches_site_symmetry": "",
    }
    summaries: dict[AtomKey, dict[str, Any]] = {
        metal.source_key: dict(unavailable) for metal in metals
    }
    if not structure.symmetry_search_available:
        return summaries

    try:
        spacegroup = spacegroup_or_none(structure.structure)
        if spacegroup is None:
            return summaries
        source_cell = structure.structure.cell
        crystallographic_structure = gemmi.Structure()
        crystallographic_structure.cell = gemmi.UnitCell(
            source_cell.a,
            source_cell.b,
            source_cell.c,
            source_cell.alpha,
            source_cell.beta,
            source_cell.gamma,
        )
        crystallographic_structure.spacegroup_hm = spacegroup.xhm()
        crystallographic_structure.setup_cell_images()
    except Exception:
        return summaries

    for metal in metals:
        if not metal.coordinates_valid:
            continue
        try:
            coincident_nonidentity_images = int(
                crystallographic_structure.cell.is_special_position(
                    metal.pos, SPECIAL_POSITION_DEDUP_CUTOFF
                )
            )
        except Exception:
            continue
        site_symmetry_order = coincident_nonidentity_images + 1
        expected_occupancy = 1.0 / site_symmetry_order
        occupancy_matches: bool | str = ""
        if metal.occupancy_valid:
            occupancy_matches = math.isclose(
                metal.occupancy,
                expected_occupancy,
                rel_tol=0.0,
                abs_tol=SPECIAL_POSITION_OCCUPANCY_TOLERANCE,
            )
        summaries[metal.source_key] = {
            "metal_special_position": coincident_nonidentity_images > 0,
            "metal_site_symmetry_order": site_symmetry_order,
            "metal_expected_crystallographic_occupancy": round(expected_occupancy, 6),
            "metal_occupancy_matches_site_symmetry": occupancy_matches,
        }
    return summaries


def _entry_nonwater_median_b_iso(structure: StructureContext) -> float:
    """Median B for canonical, non-water, non-H modeled atoms in this entry."""
    values = [
        atom.b_iso
        for atom in structure.contact_atoms
        if not atom.is_water
        and not atom.is_hydrogen
        and not (atom.occupancy_valid and atom.occupancy == 0.0)
        and math.isfinite(atom.b_iso)
    ]
    return round(float(median(values)), 3) if values else NAN


def _donor_b_factor_summary(
    metal: AtomSite, contacts: Sequence[Candidate]
) -> dict[str, Any]:
    """Compare one metal B factor with the median of its assigned donors."""
    donor_values = [
        contact.neighbor.b_iso
        for contact in contacts
        if math.isfinite(contact.neighbor.b_iso)
    ]
    result: dict[str, Any] = {
        "donor_b_iso_count": len(donor_values),
        "donor_median_b_iso": NAN,
        "metal_donor_b_ratio": NAN,
        "metal_minus_donor_b_iso": NAN,
        "metal_donor_b_similarity": NAN,
    }
    if not donor_values:
        return result

    donor_median = float(median(donor_values))
    metal_b = metal.b_iso
    result["donor_median_b_iso"] = round(donor_median, 3)
    if not math.isfinite(metal_b):
        return result

    result["metal_minus_donor_b_iso"] = round(metal_b - donor_median, 3)
    if metal_b <= 0.0 or donor_median <= 0.0:
        return result

    ratio = metal_b / donor_median
    if not math.isfinite(ratio):
        return result
    result["metal_donor_b_ratio"] = round(ratio, 4)
    result["metal_donor_b_similarity"] = round(math.exp(-abs(math.log(ratio))), 4)
    return result


def _bonding_key(neighbor: AtomSite, metal_element: str) -> tuple[str, str, str]:
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


def _parent_type(structure: StructureContext, metal: AtomSite) -> ParentType:
    """Classify the component a metal belongs to for the bond rows.

    ``metal`` comes from ``metal_atoms(METAL_ELEMENTS)``, so its element is a
    metal by construction and only the component identity needs deciding.
    """
    if metal.residue_name in cluster_ids():
        return ParentType.CLUSTER
    if metal.residue_name in heme_ids():
        return ParentType.HEME
    residue = structure.residue_for_atom(metal)
    if residue.chemical_atom_site_count == 1:
        return ParentType.ION
    return ParentType.OTHER


def zscore(dist: float, mu: float, stdev: float, dpi: float) -> float:
    """Return the bond-distance z-score using sqrt(stdev**2 + dpi**2).

    Use one DPI for donor uncertainty, treating the heavy metal as well ordered.
    An independent two-atom error model would use a different denominator.
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
        contact.image.symmetry_operation,
        contact.image.translation,
        contact.image.position,
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
                        current.image.position,
                        candidate.image.position,
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
) -> tuple[float, float, ReferenceKind, str]:
    """Return target, cutoff, and provenance for proximity eligibility."""
    exact_key = _bonding_key(neighbor, metal.element)
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
        candidate.set_donor_policy(
            DonorPolicy(
                inferred_allowed=allowed,
                rule=rule,
                override=(
                    "declared_connection"
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
        if neighbor.element not in DONOR_ELEMENTS:
            continue
        if not (neighbor.occupancy_valid and neighbor.occupancy > 0.0):
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
    return (
        candidate.neighbor.source_key,
        candidate.image.symmetry_operation,
        candidate.image.translation,
        tuple(round(value, 5) for value in candidate.image.position),
    )


def _merge_candidates(*candidate_groups: Iterable[Candidate]) -> list[Candidate]:
    """Merge proximity and declaration provenance for the same atom image."""
    merged: dict[_CandidateIdentity, Candidate] = {}
    for candidates in candidate_groups:
        for candidate in candidates:
            key = _candidate_identity(candidate)
            existing = merged.get(key)
            if existing is None:
                # Copy provenance collections because merging appends to them.
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
        and not candidate.eligibility().inferred_contact_eligible
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
    """Attach reference-based geometry results to candidate contacts."""
    for contact in contacts:
        neighbor = contact.neighbor
        reported_distance = round(contact.image.distance, 3)
        literature = literature_distances().get(_bonding_key(neighbor, metal_element))
        if literature is None:
            mu = stdev = zscore_raw = NAN
        else:
            mu, stdev = literature
            zscore_raw = zscore(contact.image.distance, mu, stdev, dpi)
        rounded_zscore = round(zscore_raw, 4) if math.isfinite(zscore_raw) else NAN
        outlier: bool | str
        consistent: bool | str
        if math.isfinite(zscore_raw):
            magnitude = abs(zscore_raw)
            outlier = magnitude >= ZSCORE_OUTLIER_CUTOFF or math.isclose(
                magnitude,
                ZSCORE_OUTLIER_CUTOFF,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            consistent = not outlier
        else:
            outlier = ""
            consistent = ""
        contact.set_geometry(
            GeometryResult(
                distance=reported_distance,
                literature_distance=mu,
                literature_stdev=stdev,
                zscore=rounded_zscore,
                reference_covered=literature is not None,
                outlier=outlier,
                consistent=consistent,
            )
        )


def _residue_image_key(contact: Candidate) -> _ResidueImageKey:
    """Identity of one donor residue image around the current metal site."""
    return (
        contact.neighbor.residue_key,
        contact.image.scope,
        contact.image.strict_ncs_operation_id,
        contact.image.image_index,
        contact.image.symmetry_operation,
        contact.image.translation,
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
            geometry = contact.geometry()
            assessable = geometry.outlier is True or geometry.consistent is True
            contact.set_multi_donor(
                MultiDonorResult(
                    detected=False,
                    contact_count=1,
                    geometry_status=MultiDonorStatus.SINGLE_DONOR,
                    contains_suspect_bond=False,
                    score_eligible=assessable,
                    score_exclusion_reason=("" if assessable else "zscore_unavailable"),
                )
            )
            continue

        any_outlier = any(contact.geometry().outlier is True for contact in group)
        all_consistent = all(contact.geometry().consistent is True for contact in group)
        if all_consistent:
            status = MultiDonorStatus.CONSISTENT
        elif any_outlier:
            status = MultiDonorStatus.SUSPECT
        else:
            status = MultiDonorStatus.INDETERMINATE
        for contact in group:
            geometry = contact.geometry()
            assessable = geometry.outlier is True or geometry.consistent is True
            contact.set_multi_donor(
                MultiDonorResult(
                    detected=True,
                    contact_count=count,
                    geometry_status=status,
                    contains_suspect_bond=any_outlier,
                    score_eligible=assessable,
                    score_exclusion_reason=("" if assessable else "zscore_unavailable"),
                )
            )


@dataclass(frozen=True, slots=True)
class _ScopeSummary:
    """Contact counts and the geometry verdict for one search scope.

    Counts are integers when the scope was assessed and NaN when it was not;
    ``status`` is then blank instead of a ``GeometryStatus``.
    """

    candidate_count: int | float
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
            candidate_count=NAN,
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


def _scope_summary(contacts: Sequence[Candidate]) -> _ScopeSummary:
    """Count the assessed contacts of one scope and classify its geometry."""
    candidate = len(contacts)
    geometries = [contact.geometry() for contact in contacts]
    multi_donor = [contact.multi_donor() for contact in contacts]
    covered = sum(geometry.reference_covered for geometry in geometries)
    outlier = sum(geometry.outlier is True for geometry in geometries)
    consistent = sum(geometry.consistent is True for geometry in geometries)
    score_eligible = sum(result.score_eligible for result in multi_donor)
    score_excluded = candidate - score_eligible
    scored_outlier = sum(
        result.score_eligible and geometry.outlier is True
        for result, geometry in zip(multi_donor, geometries, strict=False)
    )
    scored_consistent = sum(
        result.score_eligible and geometry.consistent is True
        for result, geometry in zip(multi_donor, geometries, strict=False)
    )
    multi_donor_groups = {
        _residue_image_key(contact)
        for contact, result in zip(contacts, multi_donor, strict=False)
        if result.detected
    }
    suspect_multi_donor_groups = {
        _residue_image_key(contact)
        for contact, result in zip(contacts, multi_donor, strict=False)
        if result.geometry_status == MultiDonorStatus.SUSPECT
    }
    indeterminate_multi_donor_groups = {
        _residue_image_key(contact)
        for contact, result in zip(contacts, multi_donor, strict=False)
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
        candidate_count=candidate,
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
    """Return the contextual warnings that apply to one candidate contact."""
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
    return list(dict.fromkeys(reasons))


def _site_context_values(
    contacts: Sequence[Candidate], candidates: Sequence[Candidate]
) -> dict[str, bool | int | str]:
    """Aggregate coordination-relevant context without changing confidence."""
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
        reasons.append("non_typical_first_sphere_candidate")
    reasons = list(dict.fromkeys(reasons))
    return {
        "context_warning": bool(reasons),
        "context_warning_reasons": "|".join(reasons),
        "non_typical_first_sphere_candidate_count": len(non_typical_first_sphere),
        "declared_donor_override_contact_count": sum(
            contact.donor_policy().override == "declared_connection"
            for contact in contacts
        ),
    }


# Generated-contact scope and its two dependency flags, keyed by whether the
# site has any crystallographic and any strict-NCS contact. Read only once a
# site has generated contacts, and every generated contact is one or the other,
# so the (False, False) row is unreachable; it is kept so an inconsistent count
# degrades to the NCS scope instead of raising.
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
    dpi_components: DpiComponents,
    ni: float,
    deposited_ni: float,
    structure: StructureContext,
) -> dict[str, Any]:
    explicit = _scope_summary(explicit_contacts)
    image_search_available = image_contacts is not None
    image_inclusive = (
        _scope_summary(image_contacts)
        if image_contacts is not None
        else _ScopeSummary.unavailable()
    )
    primary = image_inclusive if image_search_available else explicit
    # Report overfull occupancy on the metal itself, separately from entry-wide warnings.
    metal_overfull = (
        metal.chemical_site_identity in structure.overfull_occupancy_site_keys
    )
    symmetry_count = (
        sum(contact.image.symmetry_contact for contact in image_contacts)
        if image_contacts is not None
        else NAN
    )
    crystallographic_count = (
        sum(contact.image.crystallographic_contact for contact in image_contacts)
        if image_contacts is not None
        else NAN
    )
    strict_ncs_count = (
        sum(contact.image.strict_ncs_contact for contact in image_contacts)
        if image_contacts is not None
        else NAN
    )
    combined_count = (
        sum(
            contact.image.crystallographic_contact and contact.image.strict_ncs_contact
            for contact in image_contacts
        )
        if image_contacts is not None
        else NAN
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
            else ("none", False, False)
        )

    reasons: list[str] = []
    if dpi_components.reason_code:
        reasons.append(dpi_components.reason_code)
    if not image_search_available:
        reasons.append(ReasonCode.SYMMETRY_SEARCH_UNAVAILABLE)
    if primary.scored_assessable_count == 0:
        reasons.append("no_assessable_reference_contacts")
    return {
        "dpi": dpi_components.dpi,
        "resolution": dpi_components.resolution,
        "r_free": dpi_components.r_free,
        "reflection_count": dpi_components.reflection_count,
        "asu_volume": dpi_components.asu_volume,
        "occupancy_weighted_atom_count": (round(ni, 6) if math.isfinite(ni) else NAN),
        "deposited_occupancy_weighted_atom_count": (
            round(deposited_ni, 6) if math.isfinite(deposited_ni) else NAN
        ),
        "dpi_unavailable_reason": dpi_components.reason_code,
        "candidate_contact_count": primary.candidate_count,
        "reference_covered_contact_count": primary.reference_covered_count,
        "geometry_outlier_contact_count": primary.outlier_count,
        "geometry_consistent_contact_count": primary.consistent_count,
        "score_eligible_contact_count": primary.score_eligible_count,
        "score_excluded_contact_count": primary.score_excluded_count,
        "scored_geometry_outlier_contact_count": primary.scored_outlier_count,
        "scored_geometry_consistent_contact_count": primary.scored_consistent_count,
        "multi_donor_residue_group_count": primary.multi_donor_group_count,
        "multi_donor_contact_count": primary.multi_donor_contact_count,
        "suspect_multi_donor_residue_group_count": (
            primary.suspect_multi_donor_group_count
        ),
        "indeterminate_multi_donor_residue_group_count": (
            primary.indeterminate_multi_donor_group_count
        ),
        "explicit_contact_count": explicit.candidate_count,
        "symmetry_contact_count": symmetry_count,
        "image_inclusive_contact_count": image_inclusive.candidate_count,
        "crystallographic_contact_count": crystallographic_count,
        "strict_ncs_contact_count": strict_ncs_count,
        "combined_ncs_crystallographic_contact_count": combined_count,
        "geometry_outlier_count_explicit": explicit.outlier_count,
        "geometry_outlier_count_image_inclusive": image_inclusive.outlier_count,
        "geometry_coverage_explicit": explicit.coverage,
        "geometry_coverage_image_inclusive": image_inclusive.coverage,
        "explicit_geometry_status": explicit.status,
        "image_inclusive_geometry_status": image_inclusive.status,
        "generated_contact_scope": generated_scope,
        "geometry_classification_changes_with_generated_images": changed,
        "coordination_depends_on_crystallographic_symmetry": (depends_crystallographic),
        "coordination_depends_on_strict_ncs": depends_strict_ncs,
        "metal_overfull_occupancy": metal_overfull,
        "geometry_not_assessed_reason": "|".join(dict.fromkeys(reasons)),
    }


@dataclass(frozen=True, slots=True)
class _MetalAnalysisResult:
    """Everything the analysis of one metal site contributes to the entry."""

    bond_rows: list[BondRow]
    candidate_rows: list[CandidateRow]
    summary: dict[str, Any]
    unsupported_pairs: set[tuple[str, str]]


#: The main-chain RSZD triple every bond row carries, in output order.
ZD_COLUMNS = ("ZDm", "ZD-m", "ZD+m")


@dataclass(frozen=True, slots=True)
class DensityZScoreIndex:
    """One entry's EDSTATS rows, addressable by metal site or author identity.

    ``by_site`` keys rows by the selected metal's coordinate ``source_key``;
    ``by_author`` keys them by ``(resname, chain, resnum)`` and omits every
    author identity that more than one row claims, so the fallback join can
    never pick an arbitrary residue. ``zd_indices`` locates :data:`ZD_COLUMNS`
    in the row fields, or is ``None`` when the header lacks any of them.
    """

    by_site: Mapping[tuple[Any, ...], Sequence[CsvValue]]
    by_author: Mapping[tuple[Any, ...], Sequence[CsvValue]]
    zd_indices: tuple[int, int, int] | None

    @classmethod
    def from_stats_rows(
        cls, stats_rows: Iterable[MetalStatsRow], header: Sequence[str] | None
    ) -> DensityZScoreIndex:
        """Index the extracted EDSTATS rows of one entry against their header."""
        by_site: dict[tuple[Any, ...], Sequence[CsvValue]] = {}
        by_author: dict[tuple[Any, ...], Sequence[CsvValue]] = {}
        ambiguous_authors: set[tuple[Any, ...]] = set()
        for row in stats_rows:
            if row.site_key is not None:
                by_site[tuple(row.site_key)] = row.fields
            author_key = (row.resname, str(row.chain), str(row.resnum))
            if author_key in by_author:
                ambiguous_authors.add(author_key)
            else:
                by_author[author_key] = row.fields
        for author_key in ambiguous_authors:
            del by_author[author_key]
        return cls(by_site, by_author, _zd_indices(header))

    def lookup(
        self, site_key: Sequence[Any] | None, author_key: tuple[str, str, str]
    ) -> tuple[float, float, float]:
        """Return ``(ZDm, ZD-m, ZD+m)`` for a site, or NaNs when unavailable.

        The site key wins; the author key is the fallback for rows that never
        received a site key. A missing row, a header without the ZD columns, or
        an unparsable value all yield the NaN triple.
        """
        resname, chain, resnum = author_key
        indexed: Sequence[CsvValue] | None = None
        if site_key is not None:
            indexed = self.by_site.get(tuple(site_key))
        if indexed is None:
            indexed = self.by_author.get((resname, str(chain), str(resnum)))
        if indexed is None or self.zd_indices is None:
            return NAN, NAN, NAN
        # ``zd_indices`` addresses EDSTATS header columns, which are parsed text;
        # only the site fields appended after them can hold non-string values.
        fields = cast(Sequence[str], indexed)
        try:
            return (
                float(fields[self.zd_indices[0]]),
                float(fields[self.zd_indices[1]]),
                float(fields[self.zd_indices[2]]),
            )
        except (IndexError, ValueError):
            return NAN, NAN, NAN


def _zd_indices(header: Sequence[str] | None) -> tuple[int, int, int] | None:
    """Header positions of :data:`ZD_COLUMNS`, or ``None`` when any is absent."""
    if not header:
        return None
    try:
        zdm, zd_minus, zd_plus = (header.index(name) for name in ZD_COLUMNS)
    except ValueError:
        return None
    return zdm, zd_minus, zd_plus


@dataclass(frozen=True, slots=True)
class _EntryContext:
    """Entry-level inputs that every metal site of one entry shares."""

    pdb_id: str
    structure: StructureContext
    explicit_search: gemmi.NeighborSearch
    image_search: gemmi.NeighborSearch | None
    dpi_components: DpiComponents
    ni: float
    deposited_ni: float
    density_z_scores: DensityZScoreIndex
    nonwater_median_b_iso: float


def _assess_scope(
    entry: _EntryContext,
    metal: AtomSite,
    search: gemmi.NeighborSearch,
    declarations: Sequence[Candidate],
    include_symmetry: bool,
) -> tuple[list[Candidate], list[Candidate], set[tuple[str, str]]]:
    """Discover, filter, and score the contacts of one metal in one scope.

    Returns ``(candidates, contacts, unsupported_pairs)``: every annotated
    candidate, the subset assigned as contacts with geometry and multi-donor
    results attached, and the metal-donor element pairs no reference covers.
    """
    candidates = _merge_candidates(
        collect_proximal_candidates(entry.structure, search, metal, include_symmetry),
        declarations,
    )
    _annotate_donor_policy(entry.structure, candidates)
    contacts, unsupported_pairs = _current_contacts_from_candidates(candidates, metal)
    annotate_contacts(contacts, metal.element, entry.dpi_components.dpi)
    _annotate_multi_donor_groups(contacts)
    return candidates, contacts, unsupported_pairs


def _analyze_metal_site(
    entry: _EntryContext,
    metal: AtomSite,
    metal_declarations: Sequence[Candidate],
) -> _MetalAnalysisResult:
    pdb_id = entry.pdb_id
    structure = entry.structure
    dpi_components = entry.dpi_components
    explicit_declarations = [
        candidate
        for candidate in metal_declarations
        if not candidate.image.symmetry_contact
    ]
    explicit_candidates, explicit_contacts, unsupported_pairs = _assess_scope(
        entry, metal, entry.explicit_search, explicit_declarations, False
    )

    image_candidates: list[Candidate] | None = None
    image_contacts: list[Candidate] | None = None
    if entry.image_search is not None:
        image_candidates, image_contacts, image_unsupported = _assess_scope(
            entry, metal, entry.image_search, metal_declarations, True
        )
        unsupported_pairs.update(image_unsupported)

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
        dpi_components,
        entry.ni,
        entry.deposited_ni,
        structure,
    )
    summary.update(_site_context_values(primary_contacts, primary_candidates))
    summary.update(_donor_b_factor_summary(metal, primary_contacts))

    sigma = entry.density_z_scores.lookup(
        metal.source_key, (metal.residue_name, metal.chain_id, metal.resnum)
    )
    parent_type = _parent_type(structure, metal)
    assigned_contact_ids = {
        contact_identifier(pdb_id, metal, contact) for contact in primary_contacts
    }
    return _MetalAnalysisResult(
        bond_rows=[
            bond_row(
                pdb_id,
                structure,
                metal,
                contact,
                dpi_components.dpi,
                dpi_components.resolution,
                sigma,
                parent_type,
                context_warning_reasons(contact, multi_donor=contact.multi_donor()),
            )
            for contact in primary_contacts
        ],
        candidate_rows=[
            candidate_row(
                pdb_id,
                structure,
                metal,
                candidate,
                assigned_as_bond=(
                    contact_identifier(pdb_id, metal, candidate) in assigned_contact_ids
                ),
                context_reasons=context_warning_reasons(
                    candidate, include_proximal=True
                ),
            )
            for candidate in primary_candidates
        ],
        summary=summary,
        unsupported_pairs=unsupported_pairs,
    )


@dataclass(slots=True)
class BondAnalysisMetadata:
    """Collect non-row outcomes from coordination analysis."""

    partial_reason_codes: list[str]
    warning_codes: list[str]
    messages: list[str]


@dataclass(frozen=True, slots=True)
class BondAnalysisResult:
    """Collect coordination rows, site summaries, and run metadata."""

    bond_rows: list[BondRow]
    candidate_rows: list[CandidateRow]
    site_summaries: dict[AtomKey, dict[str, Any]]
    metadata: BondAnalysisMetadata


def _declared_candidates_by_metal(
    structure: StructureContext,
    connection_path: str,
    spatial_metals: Sequence[AtomSite],
    metadata: BondAnalysisMetadata,
) -> dict[AtomKey, list[Candidate]]:
    """Resolve deposited connections, recording unresolved ones on ``metadata``."""
    declared_candidates, declared_issues, declared_warnings = (
        collect_declared_candidates(structure, connection_path, spatial_metals)
    )
    if declared_issues:
        metadata.partial_reason_codes.append(
            ReasonCode.DECLARED_CONNECTION_RESOLUTION_INCOMPLETE
        )
        metadata.messages.extend(declared_issues)
    metadata.warning_codes.extend(declared_warnings)
    declared_by_metal: dict[AtomKey, list[Candidate]] = {}
    for candidate in declared_candidates:
        # Every declaration-derived candidate carries the metal it was resolved
        # against; only proximity discovery leaves the field unset.
        declared_by_metal.setdefault(
            cast(AtomSite, candidate.metal).source_key, []
        ).append(candidate)
    return declared_by_metal


def _entry_context(
    pdb_id: str,
    structure: StructureContext,
    stats_rows: Sequence[MetalStatsRow],
    header: Sequence[str] | None,
    dpi_inputs: DpiInputs,
    metadata: BondAnalysisMetadata,
) -> _EntryContext:
    """Compute what every site shares, recording entry-level limitations."""
    dpi_components = calculate_dpi_components(structure, dpi_inputs)
    if dpi_components.reason_code:
        metadata.partial_reason_codes.append(dpi_components.reason_code)
        metadata.messages.append(f"DPI unavailable: {dpi_components.reason_code}")
    if not structure.symmetry_search_available:
        metadata.partial_reason_codes.append(ReasonCode.SYMMETRY_SEARCH_UNAVAILABLE)
        metadata.messages.append(
            "symmetry search unavailable: "
            + (structure.symmetry_search_failure_reason or "unknown reason")
        )
    image_search = None
    if structure.symmetry_search_available:
        image_search = structure.make_neighbor_search(
            CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON,
            include_symmetry=True,
            positive_occupancy_only=True,
        )
    return _EntryContext(
        pdb_id=pdb_id,
        structure=structure,
        explicit_search=structure.make_neighbor_search(
            CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON,
            include_symmetry=False,
            positive_occupancy_only=True,
        ),
        image_search=image_search,
        dpi_components=dpi_components,
        ni=count_ni(structure),
        deposited_ni=count_deposited_ni(structure),
        density_z_scores=DensityZScoreIndex.from_stats_rows(stats_rows, header),
        nonwater_median_b_iso=_entry_nonwater_median_b_iso(structure),
    )


def _unassessable_site_summary(entry: _EntryContext, metal: AtomSite) -> dict[str, Any]:
    """The site summary for a metal whose coordinates rule out any geometry."""
    summary = _site_summary(
        metal,
        [],
        [] if entry.structure.symmetry_search_available else None,
        entry.dpi_components,
        entry.ni,
        entry.deposited_ni,
        entry.structure,
    )
    summary["geometry_not_assessed_reason"] = ReasonCode.NON_FINITE_METAL_COORDINATES
    summary.update(_site_context_values([], []))
    summary.update(_donor_b_factor_summary(metal, []))
    return summary


def _note_unsupported_pairs(
    metadata: BondAnalysisMetadata, unsupported_pairs: set[tuple[str, str]]
) -> None:
    """Record metal-donor element pairs the reference table cannot score."""
    if not unsupported_pairs:
        return
    metadata.partial_reason_codes.append(ReasonCode.MISSING_FIRST_SPHERE_REFERENCE)
    pairs = ", ".join(
        f"{metal_element}-{donor_element}"
        for metal_element, donor_element in sorted(unsupported_pairs)
    )
    metadata.messages.append(f"first-sphere reference unavailable for {pairs}")


def run_bond_analysis(
    pdb_id: str,
    pdb_path: str,
    stats_rows: Sequence[MetalStatsRow],
    header: Sequence[str] | None,
    dpi_inputs: DpiInputs,
    structure: StructureContext | None = None,
    connection_path: str | None = None,
) -> BondAnalysisResult:
    """Return contact rows, candidate rows, site summaries, and metadata.

    Bond rows come from first-sphere-eligible candidates and from source
    ``struct_conn``/``LINK`` declarations, which are evaluated even outside the
    4 A discovery radius. Image-inclusive results are primary wherever symmetry
    metadata is available.
    """
    if structure is None:
        structure = load_structure(pdb_id, pdb_path)

    metals_in_model = structure.metal_atoms(METAL_ELEMENTS, canonical=True)
    metal_proximity = _metal_proximity_summaries(pdb_id, metals_in_model)
    metal_special_positions = _metal_special_position_summaries(
        structure, metals_in_model
    )
    spatial_metals = [metal for metal in metals_in_model if metal.coordinates_valid]
    non_finite_metals = [
        metal for metal in metals_in_model if not metal.coordinates_valid
    ]
    metadata = BondAnalysisMetadata(
        partial_reason_codes=[],
        warning_codes=list(structure.warning_codes),
        messages=[],
    )
    if not metals_in_model:
        return BondAnalysisResult([], [], {}, metadata)

    declared_by_metal = _declared_candidates_by_metal(
        structure, connection_path or pdb_path, spatial_metals, metadata
    )
    if non_finite_metals:
        metadata.partial_reason_codes.append(ReasonCode.NON_FINITE_METAL_COORDINATES)
        metadata.messages.append(
            "geometry unavailable for selected metal site(s) with non-finite "
            "coordinates: "
            + ", ".join(
                f"{metal.residue_name}/{metal.chain_id or '_'}/{metal.resnum}/"
                f"{metal.atom_name}"
                for metal in non_finite_metals
            )
        )
    entry = _entry_context(pdb_id, structure, stats_rows, header, dpi_inputs, metadata)

    rows: list[BondRow] = []
    candidate_rows: list[CandidateRow] = []
    summaries: dict[AtomKey, dict[str, Any]] = {}
    for metal in metals_in_model:
        if not metal.coordinates_valid:
            summary = _unassessable_site_summary(entry, metal)
        else:
            site_result = _analyze_metal_site(
                entry, metal, declared_by_metal.get(metal.source_key, ())
            )
            _note_unsupported_pairs(metadata, site_result.unsupported_pairs)
            summary = site_result.summary
            rows.extend(site_result.bond_rows)
            candidate_rows.extend(site_result.candidate_rows)
        summary.update(metal_proximity[metal.source_key])
        summary.update(metal_special_positions[metal.source_key])
        summary["entry_nonwater_median_b_iso"] = entry.nonwater_median_b_iso
        _check_site_summary_columns(summary)
        summaries[metal.source_key] = summary

    metadata.partial_reason_codes = list(dict.fromkeys(metadata.partial_reason_codes))
    metadata.warning_codes = list(dict.fromkeys(metadata.warning_codes))
    metadata.messages = list(dict.fromkeys(metadata.messages))
    return BondAnalysisResult(rows, candidate_rows, summaries, metadata)
