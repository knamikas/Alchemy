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

import json
import math
import os
import re

from metal_elements import METAL_ELEMENTS
from metal_identification import COFACTOR_CATALOG_PATH, DATA_DIR
from structure_analysis import (
    blank_if_missing,
    count_deposited_ni,
    count_ni,
    load_structure,
    position_distance,
)

# Derived from the shared DATA_DIR rather than recomputed, so the bundled data
# location has one definition (see metal_identification.DATA_DIR).
DONOR_DISTANCE_PATH = os.path.join(DATA_DIR, "metal_distances_info.txt")

# Recognized amino-acid donors. Waters are recognized separately with Gemmi's
# Residue.is_water(), which also handles WAT, H2O, and DOD.
AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

# Typical protein donors that Alchemy may infer from geometry alone. Every
# standard amino acid can donate through its backbone carbonyl O. The listed
# additions are established side-chain donor atoms; uncommon but chemically
# possible atoms remain visible candidates and require a source declaration to
# become bonds. Polymer-terminal atoms are handled conditionally below.
INFERRED_DONOR_ATOMS = {
    "ALA": frozenset(("O",)),
    "ARG": frozenset(("O",)),
    "ASN": frozenset(("O", "OD1")),
    "ASP": frozenset(("O", "OD1", "OD2")),
    "CYS": frozenset(("O", "SG")),
    "GLN": frozenset(("O", "OE1")),
    "GLU": frozenset(("O", "OE1", "OE2")),
    "GLY": frozenset(("O",)),
    "HIS": frozenset(("O", "ND1", "NE2")),
    "ILE": frozenset(("O",)),
    "LEU": frozenset(("O",)),
    "LYS": frozenset(("O", "NZ")),
    "MET": frozenset(("O", "SD")),
    "PHE": frozenset(("O",)),
    "PRO": frozenset(("O",)),
    "SER": frozenset(("O", "OG")),
    "THR": frozenset(("O", "OG1")),
    "TRP": frozenset(("O",)),
    "TYR": frozenset(("O", "OH")),
    "VAL": frozenset(("O",)),
}
if set(INFERRED_DONOR_ATOMS) != AA:
    raise ValueError("INFERRED_DONOR_ATOMS must cover every standard amino acid")

N_TERMINAL_DONOR_ATOMS = frozenset(("N",))
C_TERMINAL_DONOR_ATOMS = frozenset(("OXT", "OT1", "OT2"))

# Broad discovery retains all plausible donor elements. The atom-level table
# above, not this element set, controls geometry-only bond inference.
DONOR_ELEMENTS = frozenset(("N", "O", "S"))

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

NAN = float("nan")


def _load_cofactor_classes(path):
    """Return ``(cluster_ids, heme_ids)`` from the bundled catalog.

    Structural classes are derived from CCD connectivity when the catalog is built
    by ``tools/build_metallocofactor_catalog.py``, so they track the CCD rather
    than drifting behind a list maintained by hand here. They tag each metal's
    environment in ``parent_type``; CLUSTER takes precedence in
    ``_parent_type`` for any component present in both.
    """
    cluster, heme = set(), set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or not fields[0].strip():
                continue
            component_id = fields[0].strip()
            structural_class = fields[2].strip()
            if structural_class == "cluster":
                cluster.add(component_id)
            elif structural_class == "heme":
                heme.add(component_id)
    if not cluster or not heme:
        raise ValueError(
            f"{os.path.basename(path)} carries no structural classes; rebuild "
            "it with tools/build_metallocofactor_catalog.py"
        )
    return frozenset(cluster), frozenset(heme)


CLUSTER, HEMES = _load_cofactor_classes(COFACTOR_CATALOG_PATH)

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


def _load_literature(path):
    """Parse metal_distances_info.txt -> {(residue, atom, metal): (mu, stdev)}.

    Space-delimited ``residue atom metal avg_bond_dist st_dev``. The header line
    and blank separator lines are skipped naturally because their 4th/5th tokens
    do not parse as floats.
    """
    lit = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                mu, stdev = float(parts[3]), float(parts[4])
            except ValueError:
                continue  # header ("avg_bond_dist") or malformed line
            lit[(parts[0], parts[1], parts[2])] = (mu, stdev)
    return lit


LIT = _load_literature(DONOR_DISTANCE_PATH)
FIRST_SPHERE_TARGETS = {}
for (_, donor, metal_element), (target, _) in LIT.items():
    key = (metal_element, donor)
    FIRST_SPHERE_TARGETS[key] = max(target, FIRST_SPHERE_TARGETS.get(key, -math.inf))


# --------------------------------------------------------------------------- #
# DPI
# --------------------------------------------------------------------------- #
def _is_placeholder_cell(cell) -> bool:
    """Whether ``cell`` is Gemmi's stand-in for a file with no CRYST1 record.

    ``UnitCell.is_crystal()`` is false only for the exact 1 x 1 x 1 default,
    which is what a coordinate file carrying no usable cell parses to. That
    volume is smaller than a single non-hydrogen atom, so accepting it would
    hand the DPI calculation a physically impossible asymmetric unit and give
    every contact in the entry a confident-looking z-score derived from it.
    Reporting the metadata as missing is the honest outcome.

    Deliberately narrow: a small but genuine cell is still a crystal, and
    widening this into a plausibility threshold would mean inventing a cutoff
    nobody derived.
    """
    return not cell.is_crystal()


def _asu_volume(mtz_path, pdb_path):
    """Asymmetric-unit volume (A^3) = unit-cell volume / number of symmetry ops.

    Computing this from the cell and space group is exact and needs no header
    scrape. Prefer the MTZ (matching the diffraction data); fall back to PDB
    CRYST1 metadata.
    """
    import gemmi

    cell = sg = None
    try:
        mtz = gemmi.read_mtz_file(mtz_path)
        cell, sg = mtz.cell, mtz.spacegroup
    except Exception:
        cell = sg = None
    if cell is None or sg is None or cell.volume <= 0 or _is_placeholder_cell(cell):
        try:
            st = gemmi.read_structure(pdb_path)
            cell = st.cell
            sg = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
        except Exception:
            return NAN
    if cell is None or sg is None or cell.volume <= 0 or _is_placeholder_cell(cell):
        return NAN
    nops = len(list(sg.operations()))
    return cell.volume / nops if nops > 0 else NAN


def _rfree_from_pdb(pdb_path):
    """Fallback R-free scrape from a PDB REMARK 3 header (final R-free only)."""
    try:
        with open(pdb_path) as f:
            for line in f:
                if (
                    "FREE R VALUE" in line
                    and "TEST" not in line
                    and "ESTIMATED" not in line
                    and "BIN" not in line
                ):
                    m = re.search(r"FREE R VALUE\s*:\s*(\d+\.\d+)", line)
                    if m:
                        return float(m.group(1))
    except OSError:
        pass
    return NAN


def _calculate_dpi_details(structure, dpi_inputs):
    """Return ``(dpi, resolution, reason_code)``. Never raises.

    DPI = 1.28 * ni**0.5 * va**(1/3) * nobs**(-5/6) * rfree  (Blow 2002 eq. 7).
    Resolution is metadata only (it is implicit in va/nobs, not a separate term).
    Any missing/non-finite input yields ``(nan, resolution)`` so the caller still
    emits the measured bond geometry.
    """
    resolution = dpi_inputs.get("resolution", NAN)
    try:
        resolution = float(resolution)
    except (TypeError, ValueError):
        resolution = NAN

    data_json = dpi_inputs.get("data_json")
    if not data_json:
        # Manual input mode without --data-json: no metadata source exists, so
        # the reflection count can never be resolved and DPI is unavailable by
        # construction. Report that rather than letting open(None) raise a
        # TypeError into the catch-all below, which mislabelled a missing
        # argument as a failed calculation.
        return NAN, resolution, "missing_dpi_metadata_source"

    try:
        props = {}
        try:
            with open(data_json) as f:
                props = json.load(f).get("properties", {})
        except (OSError, ValueError):
            props = {}

        nobs = props.get("NREFCNT")
        rfree = props.get("RFFIN")
        rfree = (
            float(rfree)
            if rfree not in (None, "")
            else _rfree_from_pdb(dpi_inputs["pdb_path"])
        )
        nobs = float(nobs) if nobs not in (None, "") else NAN
        va = _asu_volume(dpi_inputs["mtz_path"], dpi_inputs["pdb_path"])
        ni = count_ni(structure)

        if not all(isinstance(x, (float, int)) for x in (nobs, rfree, va)):
            return NAN, resolution, "invalid_dpi_metadata"
        if not (
            math.isfinite(nobs)
            and math.isfinite(rfree)
            and math.isfinite(va)
            and nobs > 0
            and rfree > 0
            and va > 0
            and ni > 0
        ):
            if structure.occupancy_validation_failed:
                reason = "invalid_occupancy"
            elif not math.isfinite(nobs) or nobs <= 0:
                reason = "missing_or_invalid_reflection_count"
            elif not math.isfinite(rfree) or rfree <= 0:
                reason = "missing_or_invalid_rfree"
            elif not math.isfinite(va) or va <= 0:
                reason = "missing_or_invalid_asu_volume"
            else:
                reason = "invalid_dpi_atom_count"
            return NAN, resolution, reason
        dpi = 1.28 * (ni**0.5) * (va ** (1 / 3)) * (nobs ** (-5 / 6)) * rfree
        return round(dpi, 4), resolution, ""
    except Exception:
        return NAN, resolution, "dpi_calculation_failed"


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
    if metal_res in CLUSTER:
        return "cluster"
    if metal_res in HEMES:
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


def _sigma_index(stats_rows):
    """Map (resname, chain, resnum) -> edstats fields, for the sigma join."""
    return {
        (r["resname"], str(r["chain"]), str(r["resnum"])): r["fields"]
        for r in stats_rows
    }


ZD_COLUMNS = ("ZDm", "ZD-m", "ZD+m")


def _zd_indices(header):
    """Return column indices for ZDm/ZD-m/ZD+m, or None
    if the header is missing or doesn't contain all three names."""
    if not header:
        return None
    try:
        return tuple(header.index(name) for name in ZD_COLUMNS)
    except ValueError:
        return None


def _sigma_for(sig, resname, chain, resnum, zd_idx):
    fields = sig.get((resname, str(chain), str(resnum)))
    if fields is None or zd_idx is None:
        return NAN, NAN, NAN
    try:
        return (
            float(fields[zd_idx[0]]),
            float(fields[zd_idx[1]]),
            float(fields[zd_idx[2]]),
        )
    except (IndexError, ValueError):
        return NAN, NAN, NAN


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
    literature = LIT.get(exact_key)
    if literature is not None:
        target = literature[0]
        reference_kind = "exact"
        reference_key = exact_key
    else:
        # The scoring table may omit a residue-specific donor while still
        # defining the same donor element for this metal. Use the largest such
        # target only for sphere membership; exact references remain mandatory
        # for z-score calculation.
        target = FIRST_SPHERE_TARGETS.get((metal.element, neighbor.element))
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


def _connection_source(path):
    lower = str(path or "").lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    return "struct_conn" if lower.endswith((".cif", ".mmcif")) else "LINK"


def _enum_name(value):
    return str(getattr(value, "name", value)).lower()


def _analysis_chain_names(connection_path):
    """Map source chain names onto the analysis model's chain names.

    For ordinary structures, replaying Gemmi's ``setup_entities`` and
    ``shorten_chain_names`` calls recovers the conversion mapping instead of
    assuming it is the identity. Oversized structures use residue-level
    provenance and resolve source identities before this fallback is reached.
    A source PDB is extracted textually and keeps its chain names, so its
    mapping is empty.
    """
    import gemmi

    if _connection_source(connection_path) != "struct_conn":
        return {}
    copy = gemmi.read_structure(connection_path)
    if len(copy) == 0:
        return {}
    source_names = [str(chain.name) for chain in copy[0]]
    copy.setup_entities()
    copy.shorten_chain_names()
    return {
        source: str(chain.name)
        for source, chain in zip(source_names, copy[0])
        if source != str(chain.name)
    }


def _analysis_atom_for_partner(structure, cra, chain_names):
    """Resolve one declared partner to an analysis atom by author identity.

    The declaration's author identifiers survive conversion: only chain names
    are shortened, which ``chain_names`` reverses, and a component identifier
    too long for the legacy residue field is indexed under both its source and
    its converted name. Returns ``None`` when the identity matches no residue,
    matches more than one, or names an atom the analyzed model does not hold.
    """
    chain = getattr(cra, "chain", None)
    residue = getattr(cra, "residue", None)
    atom = getattr(cra, "atom", None)
    if chain is None or residue is None or atom is None:
        return None

    source_chain_id = str(chain.name)
    resnum = f"{residue.seqid.num}{blank_if_missing(residue.seqid.icode)}"
    source_lookup = (
        structure.residues_for_source_author
        if hasattr(type(structure), "residues_for_source_author")
        else structure.residues_for_author
    )
    matches = source_lookup(str(residue.name), source_chain_id, resnum)
    if len(matches) != 1:
        # Older analysis PDBs retain only Gemmi's chain-name shortening.  New
        # packed PDBs embed full residue provenance, so the source lookup above
        # succeeds without guessing how sequence numbers were remapped.
        chain_id = chain_names.get(source_chain_id, source_chain_id)
        if chain_id != source_chain_id:
            # The combined legacy index restores >3-character residue names
            # while using Gemmi's shortened chain name.  Packed structures do
            # not need this fallback because their full source identity is
            # indexed above.
            matches = structure.residues_for_author(str(residue.name), chain_id, resnum)
    if len(matches) != 1:
        return None

    atom_name = str(atom.name).strip()
    altloc = blank_if_missing(atom.altloc)
    named = [site for site in matches[0].source_atoms if site.atom_name == atom_name]
    if altloc:
        named = [site for site in named if site.altloc == altloc]
    return named[0] if named else None


def _selected_conformer_atom(structure, atom):
    """Return the selected-conformer record for ``atom``'s chemical site.

    A declaration names one deposited atom record, which may belong to an
    alternate conformer that per-residue selection did not choose. Contacts are
    measured on the selected conformer only, so re-point the declaration onto
    the same-named atom of that conformer rather than admitting a second record
    for a chemical site the proximity search already reports. Returns ``None``
    when the selected conformer has no atom of that name and element, because
    the declared contact then has no counterpart in the analyzed model.
    """
    if atom is None:
        return None
    selected = structure.atom_for_indices(
        atom.chain_index, atom.residue_index, atom.atom_index
    )
    if selected is not None:
        return selected
    for candidate in structure.residue_for_atom(atom).contact_atoms:
        if candidate.atom_name == atom.atom_name and candidate.element == atom.element:
            return candidate
    return None


def _declared_partner_is_metal(address, cra):
    """Whether a source connection partner unambiguously names a metal.

    Prefer the element of the atom resolved in the source model.  A malformed
    declaration may omit that atom or name a record no longer present, so fall
    back only to unambiguous component/atom identifiers such as ``ZN``.  Do not
    guess elements from prefixes (for example ``CA`` is commonly a protein
    alpha carbon): an indeterminate non-metal declaration is outside Alchemy's
    coordination-analysis scope.
    """
    source_atom = getattr(cra, "atom", None)
    if source_atom is not None:
        element = str(getattr(source_atom.element, "name", "")).upper()
        if element in METAL_ELEMENTS:
            return True

    residue_id = getattr(address, "res_id", None)
    residue_name = str(getattr(residue_id, "name", "")).strip().upper()
    atom_name = str(getattr(address, "atom_name", "")).strip().upper()
    return residue_name in METAL_ELEMENTS or atom_name in METAL_ELEMENTS


def _declared_candidate_geometry(structure, metal, neighbor, connection):
    """Return contact geometry for a resolved declared connection."""
    import gemmi

    asu = connection.asu
    if asu == gemmi.Asu.Same or (
        asu == gemmi.Asu.Any and not structure.symmetry_search_available
    ):
        return {
            "distance_raw": position_distance(metal.xyz, neighbor.xyz),
            "transformed_position": neighbor.xyz,
            "symmetry_contact": False,
            "crystallographic_contact": False,
            "strict_ncs_contact": False,
            "strict_ncs_operation_id": "",
            "contact_scope": "explicit",
            "symmetry_image_index": 0,
            "symmetry_operation": "1_555",
            "translation": (0, 0, 0),
        }

    if not structure.symmetry_search_available:
        raise ValueError(
            structure.symmetry_search_failure_reason or "symmetry metadata unavailable"
        )
    cell = structure.structure.cell
    nearest = cell.find_nearest_image(metal.pos, neighbor.pos, asu)
    transformed_fractional = cell.fract_image(nearest, cell.fractionalize(neighbor.pos))
    transformed = cell.orthogonalize(transformed_fractional)
    translation = tuple(int(value) for value in nearest.pbc_shift)
    image_index = int(nearest.sym_idx)
    # Classified exactly as the proximity path does. ``same_asu()`` alone
    # cannot tell the two apart: after ``setup_cell_images`` the image list
    # holds the strict-NCS transforms as well, so an NCS image is "not the same
    # ASU" and would otherwise be reported as crystallographic with no NCS
    # operation identifier.
    (
        crystallographic_contact,
        strict_ncs_contact,
        strict_ncs_operation_id,
        contact_scope,
    ) = structure.image_provenance(image_index, translation)
    return {
        "distance_raw": float(nearest.dist()),
        "transformed_position": (
            float(transformed.x),
            float(transformed.y),
            float(transformed.z),
        ),
        "symmetry_contact": crystallographic_contact or strict_ncs_contact,
        "crystallographic_contact": crystallographic_contact,
        "strict_ncs_contact": strict_ncs_contact,
        "strict_ncs_operation_id": strict_ncs_operation_id,
        "contact_scope": contact_scope,
        "symmetry_image_index": image_index,
        "symmetry_operation": nearest.symmetry_code(),
        "translation": translation,
    }


def _collect_declared_candidates(structure, connection_path, metals):
    """Resolve source ``struct_conn``/``LINK`` claims to analysis atoms.

    Partners are resolved by author identity -- chain, sequence number,
    insertion code, component, atom name, and altloc -- against the analysis
    model. Atom serials cannot carry this join: Gemmi's PDB writer emits a TER
    record after each polymer and every TER consumes a serial number, so the
    serials of an mmCIF-converted model run ahead of the source
    ``_atom_site.id`` by one for each preceding TER, and a partner past the
    first TER would resolve to a neighbouring atom. Conversion provenance
    restores source residue identities when the legacy PDB representation
    shortens a chain, truncates a component name, or packs an oversized model
    into synthetic chain and residue identifiers.

    A declaration names one deposited record, so either partner may be an
    alternate conformer that per-residue selection did not choose. Both are
    re-pointed onto their residue's selected conformer before use, so a
    declaration cannot introduce a second record for a chemical site the
    proximity search already reports and cannot inflate a coordination number
    with a conformer that is absent from the analyzed model.
    """
    import gemmi

    if not connection_path:
        return [], [], []
    source = _connection_source(connection_path)
    try:
        declared_structure = gemmi.read_structure(connection_path)
    except Exception as exc:
        return [], [f"{source} parse failed: {type(exc).__name__}: {exc}"], []
    if len(declared_structure) == 0:
        return [], [f"{source} contains no coordinate model"], []

    source_model = declared_structure[0]
    chain_names = _analysis_chain_names(connection_path)
    selected_metal_keys = {metal.source_key for metal in metals}
    candidates = []
    issues = []
    warnings = []
    for index, connection in enumerate(declared_structure.connections, start=1):
        connection_id = str(connection.name).strip() or f"{source}_{index}"
        addresses = (connection.partner1, connection.partner2)
        declared_metal_partner = any(
            _declared_partner_is_metal(address, None) for address in addresses
        )
        try:
            source_cras = [
                source_model.find_cra(address, ignore_segment=True)
                for address in addresses
            ]
            declared_metal_partner = declared_metal_partner or any(
                _declared_partner_is_metal(address, cra)
                for address, cra in zip(addresses, source_cras)
            )
            source_atoms = []
            deselected_partner = False
            substituted_partner = False
            for cra in source_cras:
                declared_atom = _analysis_atom_for_partner(structure, cra, chain_names)
                selected_atom = _selected_conformer_atom(structure, declared_atom)
                if declared_atom is not None:
                    if selected_atom is None:
                        deselected_partner = True
                    elif selected_atom is not declared_atom:
                        substituted_partner = True
                source_atoms.append(selected_atom)
        except Exception as exc:
            if declared_metal_partner:
                issues.append(
                    f"{source} {connection_id} resolution failed: {type(exc).__name__}"
                )
            continue

        if deselected_partner:
            if declared_metal_partner:
                issues.append(
                    f"{source} {connection_id} partner names a conformer whose "
                    f"selected alternative has no matching atom"
                )
            continue
        first, second = source_atoms
        first_is_metal = first is not None and first.source_key in selected_metal_keys
        second_is_metal = (
            second is not None and second.source_key in selected_metal_keys
        )
        connection_involves_metal = (
            declared_metal_partner or first_is_metal or second_is_metal
        )
        if substituted_partner and connection_involves_metal:
            warnings.append("declared_connection_conformer_substituted")
        if first is None and second is None:
            if connection_involves_metal:
                issues.append(f"{source} {connection_id} neither partner resolved")
            continue
        if not (first_is_metal or second_is_metal):
            if connection_involves_metal and (first is None or second is None):
                issues.append(f"{source} {connection_id} partner unresolved")
            continue
        if first is None or second is None:
            issues.append(f"{source} {connection_id} partner unresolved")
            continue
        if first_is_metal and second_is_metal:
            continue
        metal, neighbor = (first, second) if first_is_metal else (second, first)
        residue = structure.residue_for_atom(neighbor)
        if neighbor.element not in DONOR_ELEMENTS:
            # Not a donor-like atom at all, so there is no coordination
            # evidence to retain. Recorded rather than dropped in silence.
            warnings.append("declared_donor_element_unsupported")
            continue
        # Nucleic acids, modified residues and organic ligands are genuine
        # metal donors, but no bundled literature reference covers them, so
        # their geometry can never be z-scored. Retain them as candidate
        # evidence carrying the measured distance and the declaration's own
        # provenance: dropping them is indistinguishable from a metal with no
        # coordination at all, which is how a fully coordinated nucleic-acid
        # site came to report no contacts. They are deliberately not promoted
        # to bond rows -- that would raise coordination counts and enlarge the
        # confidence geometry-coverage denominator on the strength of a contact
        # nothing can assess.
        donor_class_supported = bool(residue.is_water or residue.residue_name in AA)
        if not donor_class_supported:
            warnings.append("declared_donor_outside_supported_classes")
        try:
            geometry = _declared_candidate_geometry(
                structure, metal, neighbor, connection
            )
        except Exception as exc:
            issues.append(
                f"{source} {connection_id} geometry unresolved: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if (
            neighbor.residue_key == metal.residue_key
            and not geometry["symmetry_contact"]
        ):
            continue

        reported_distance = float(connection.reported_distance)
        if not math.isfinite(reported_distance) or reported_distance <= 0:
            reported_distance = NAN

        record = {
            "source": source,
            "connection_id": connection_id,
            "connection_type": _enum_name(connection.type),
            "connection_link_id": str(connection.link_id).strip(),
            "connection_asu": _enum_name(connection.asu),
            "connection_reported_distance": reported_distance,
        }
        candidates.append(
            {
                "metal": metal,
                "neighbor": neighbor,
                **geometry,
                "candidate_sources": {source},
                "declared_connections": [record],
                "donor_class_supported": donor_class_supported,
            }
        )
    return candidates, issues, warnings


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
        literature = LIT.get(
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


def stats_extra_values(structure, metal=None, summary=None):
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
