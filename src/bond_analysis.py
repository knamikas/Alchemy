"""Metal-ligand bond-distance analysis for the Alchemy pipeline.

For one PDB entry this finds every metal atom in the first model, uses Gemmi to
search explicit, crystallographic, and strict-NCS neighbors within 4 A, filters
those candidates to the first coordination sphere, and computes a
resolution-aware z-score against the consolidated literature reference distances
in ``metal_distances_info.txt`` (Harding 2006 and Zheng et al. 2008 [Ni only]):

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

from metal_identification import metals, uncommonMetals
from structure_analysis import (
    count_deposited_ni,
    count_ni,
    load_structure,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Any element symbol we treat as a metal of interest (matched against the PDB
# element column, which is more reliable than residue-name matching).
METAL_ELEMENTS = set(metals) | set(uncommonMetals)

# Recognized amino-acid donors. Waters are recognized separately with Gemmi's
# Residue.is_water(), which also handles WAT, H2O, and DOD.
AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
      }

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

# Metallocofactor classifications, used only to tag each metal's environment in
# parent_type. CLUSTER is checked before HEMES for codes present in both sets.
CLUSTER = {
    '0KA', '1CL', '35L', '6ML', '82N', '8JU', '8P8', '9S8', 'A1CBX', 'B51', 'BF8', 
    'BJ8', 'CFM', 'CFN', 'CLF', 'CLP', 'CUV', 'CZL', 'D6N', 'ER2', 'F3S', 'F4S', 
    'FES', 'FNE', 'FS0', 'FS2', 'FS3', 'FS4', 'FS5', 'FSF', 'FSO', 'FSX', 'HC0', 
    'HC1', 'ICE', 'ICG', 'ICH', 'ICS', 'ICZ', 'LPJ', 'MSK', 'NFE', 'NFS', 'NUI', 
    'Q46', 'RQM', 'S3F', 'S5Q', 'SF3', 'SF4', 'T2N', 'UFF', 'VQ8', 'VV2', 'WCC', 
    'XCC', 'ZKP'

}
HEMES = {
    '1FH', '2FH', '4HE', '522', '6CO', '6CQ', '6HE', '7HE', '7OH', '83L', 
    '89R', 'A1ADT', 'A1JN4', 'CCH', 'CLN', 'DDH', 'DHE', 'FDD', 'FDE', 'FEC', 
    'FMI', 'HAS', 'HCO', 'HDD', 'HDE', 'HDM', 'HE6', 'HEA', 'HEB', 'HEC', 'HEM', 
    'HEO', 'HEV', 'HFM', 'HIF', 'HKL', 'HME', 'HP5', 'ISW', 'MH0', 'MHM', 'MQP', 
    'N7H', 'NTE', 'OBV', 'POR', 'SIK', 'SRM', 'UFE', 'VEA', 'VER', 'VOV', 'WC5', 
    'WPC', 'WUF', 'WUP', 'WVP', 'WXP', 'WYP'
}

# Fixed output schema; main.py imports this so the module and driver never
# drift. Legacy "candidate" field names are retained for CSV compatibility,
# but now describe contacts that passed the first-sphere filter.
BOND_COLUMNS = [
    "pdbID", "metal_resname", "metal_chain", "metal_resnum", "metal_element",
    "neighbor_resname", "neighbor_atom", "neighbor_element", "distance",
    "literature_distance", "literature_stdev", "zscore", "dpi", "resolution",
    "sigma_mag", "sigma_neg", "sigma_pos", "parent_type", "bonded_to",
    "model_id", "metal_model_index", "metal_chain_index",
    "metal_residue_index", "metal_atom_index", "metal_atom", "metal_icode",
    "metal_altloc", "metal_occupancy", "metal_occupancy_valid",
    "metal_occupancy_status", "metal_conformer_mean_occupancy",
    "metal_altloc_options", "metal_altloc_selection_fallback",
    "neighbor_chain", "neighbor_resnum", "neighbor_icode",
    "neighbor_model_index", "neighbor_chain_index", "neighbor_residue_index",
    "neighbor_atom_index", "neighbor_altloc", "neighbor_occupancy",
    "neighbor_occupancy_valid", "neighbor_occupancy_status",
    "neighbor_conformer_mean_occupancy", "neighbor_altloc_options",
    "neighbor_altloc_selection_fallback", "alternative_conformers_present",
    "altloc_selection_fallback", "neighbor_class", "candidate_contact",
    "reference_covered", "geometry_outlier", "geometry_consistent",
    "zscore_outlier_cutoff", "contact_scope", "symmetry_contact",
    "crystallographic_contact", "strict_ncs_contact",
    "strict_ncs_operation_id",
    "symmetry_image_index", "symmetry_operation", "cell_translation_x",
    "cell_translation_y", "cell_translation_z", "transformed_neighbor_x",
    "transformed_neighbor_y", "transformed_neighbor_z",
]


# Appended after the dynamic EDSTATS header in metal_stats_all.csv.
STATS_EXTRA_COLUMNS = [
    "model_policy", "input_model_count", "model_analyzed", "model_id",
    "multi_model_structure", "metal_model_index", "metal_chain_index",
    "metal_residue_index", "metal_atom_index", "metal_resname", "metal_chain",
    "metal_resnum", "metal_atom", "metal_element", "metal_icode",
    "metal_altloc", "metal_occupancy", "metal_occupancy_valid",
    "metal_occupancy_status",
    "metal_conformer_mean_occupancy", "metal_altloc_options",
    "alternative_conformers_present", "altloc_selection_fallback",
    "coordinate_mapping_status", "selected_metal_site_status",
    "dpi", "resolution", "occupancy_weighted_atom_count",
    "deposited_occupancy_weighted_atom_count", "dpi_atom_count_multiplier",
    "strict_ncs_operation_count", "crystallographic_operation_count",
    "dpi_unavailable_reason",
    "candidate_contact_count", "reference_covered_contact_count",
    "geometry_outlier_contact_count", "geometry_consistent_contact_count",
    "explicit_contact_count", "symmetry_contact_count",
    "image_inclusive_contact_count", "crystallographic_contact_count",
    "strict_ncs_contact_count", "combined_ncs_crystallographic_contact_count",
    "geometry_outlier_count_explicit",
    "geometry_outlier_count_image_inclusive",
    "geometry_coverage_explicit", "geometry_coverage_image_inclusive",
    "explicit_geometry_status", "image_inclusive_geometry_status",
    "generated_contact_scope",
    "geometry_classification_changes_with_generated_images",
    "coordination_depends_on_crystallographic_symmetry",
    "coordination_depends_on_strict_ncs",
    "symmetry_search_available", "symmetry_search_failure_reason",
    "occupancy_validation_failed", "missing_occupancy_count",
    "invalid_occupancy_count", "zero_occupancy_atom_count",
    "metal_zero_occupancy", "geometry_not_assessed_reason",
    "duplicate_atom_records_present", "duplicate_atom_record_count",
    "duplicate_atom_coordinate_conflict_count",
    "malformed_duplicate_atom_name_count", "raw_occupancy_mapping_failed",
    "raw_occupancy_mapping_failure_reason",
    "unknown_element_atom_count", "element_validation_warning",
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


LIT = _load_literature(os.path.join(DATA_DIR, "metal_distances_info.txt"))
FIRST_SPHERE_TARGETS = {}
for (_, donor, metal_element), (target, _) in LIT.items():
    key = (metal_element, donor)
    FIRST_SPHERE_TARGETS[key] = max(
        target, FIRST_SPHERE_TARGETS.get(key, -math.inf))


# --------------------------------------------------------------------------- #
# DPI
# --------------------------------------------------------------------------- #
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
    if cell is None or sg is None or cell.volume <= 0:
        try:
            st = gemmi.read_structure(pdb_path)
            cell = st.cell
            sg = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
        except Exception:
            return NAN
    if cell is None or sg is None or cell.volume <= 0:
        return NAN
    nops = len(list(sg.operations()))
    return cell.volume / nops if nops > 0 else NAN


def _rfree_from_pdb(pdb_path):
    """Fallback R-free scrape from a PDB REMARK 3 header (final R-free only)."""
    try:
        with open(pdb_path) as f:
            for line in f:
                if ("FREE R VALUE" in line and "TEST" not in line
                        and "ESTIMATED" not in line and "BIN" not in line):
                    m = re.search(r"FREE R VALUE\s*:\s*(\d+\.\d+)", line)
                    if m:
                        return float(m.group(1))
    except OSError:
        pass
    return NAN

def _count_ni(structure):
    """Compatibility wrapper for the shared first-model DPI atom count."""
    return count_ni(structure)


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
        rfree = float(rfree) if rfree not in (None, "") else _rfree_from_pdb(dpi_inputs["pdb_path"])
        nobs = float(nobs) if nobs not in (None, "") else NAN
        va = _asu_volume(dpi_inputs["mtz_path"], dpi_inputs["pdb_path"])
        ni = _count_ni(structure)

        if not all(isinstance(x, float) or isinstance(x, int) for x in (nobs, rfree, va)):
            return NAN, resolution, "invalid_dpi_metadata"
        if not (math.isfinite(nobs) and math.isfinite(rfree) and math.isfinite(va)
                and nobs > 0 and rfree > 0 and va > 0 and ni > 0):
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
        dpi = 1.28 * (ni ** 0.5) * (va ** (1 / 3)) * (nobs ** (-5 / 6)) * rfree
        return round(dpi, 4), resolution, ""
    except Exception:
        return NAN, resolution, "dpi_calculation_failed"


def calculate_dpi(structure, dpi_inputs):
    """Return ``(dpi, resolution)`` while retaining the historical public API."""
    dpi, resolution, _ = _calculate_dpi_details(structure, dpi_inputs)
    return dpi, resolution


# --------------------------------------------------------------------------- #
# Bond rows
# --------------------------------------------------------------------------- #
def _bonding_key(neighbor, nb_res, metal_el):
    """Exact (residue, atom, metal) key matching metal_distances_info.txt columns."""
    name = neighbor.atom_name.strip()
    if neighbor.is_water:
        return ("HOH", "O", metal_el)
    if name == "O":               # backbone carbonyl O -> literal "CA" row
        return ("CA", "O", metal_el)
    if name.startswith("O"):      # side-chain O (OD1/OE1/OG/OH/...)
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
    denom = math.sqrt(dpi ** 2 + stdev ** 2)
    return round((dist - mu) / denom, 4) if denom > 0 else NAN


def _sigma_index(stats_rows):
    """Map (resname, chain, resnum) -> edstats fields, for the sigma join."""
    return {(r["resname"], str(r["chain"]), str(r["resnum"])): r["fields"]
            for r in stats_rows}

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
        return float(fields[zd_idx[0]]), float(fields[zd_idx[1]]), float(fields[zd_idx[2]])
    except (IndexError, ValueError):
        return NAN, NAN, NAN


def _contact_sort_key(contact):
    neighbor = contact["neighbor"]
    return (neighbor.chain_index, neighbor.residue_index, neighbor.atom_index,
            contact["symmetry_operation"], contact["translation"],
            contact["transformed_position"])


def _transformed_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _special_position_preference(contact):
    """Choose a stable representative for near-coincident symmetry images.

    Prefer an explicit image when one exists so an off-axis refinement artifact
    cannot turn an otherwise explicit contact into a symmetry-dependent one.
    Within the same scope, retain the shortest contact and then use stable
    symmetry provenance to break any remaining tie.
    """
    return (contact["symmetry_contact"], contact["distance_raw"],
            contact["symmetry_image_index"], contact["symmetry_operation"],
            contact["translation"], contact["transformed_position"])


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
        for candidate in sorted(by_source[source_key],
                                key=_special_position_preference):
            if any(
                _transformed_distance(
                    current["transformed_position"],
                    candidate["transformed_position"],
                ) <= SPECIAL_POSITION_DEDUP_CUTOFF
                for current in retained
            ):
                continue
            retained.append(candidate)
        contacts.extend(retained)
    contacts.sort(key=_contact_sort_key)
    return contacts


def _first_sphere_cutoff(metal, neighbor):
    """Return the maximum direct metal-donor distance for this atom pair."""
    literature = LIT.get(_bonding_key(
        neighbor, neighbor.residue_name, metal.element))
    if literature is not None:
        target = literature[0]
    else:
        # The scoring table may omit a residue-specific donor while still
        # defining the same donor element for this metal. Use the largest such
        # target only for sphere membership; exact references remain mandatory
        # for z-score calculation.
        target = FIRST_SPHERE_TARGETS.get(
            (metal.element, neighbor.element))
        if target is None:
            return NAN
    return min(CUTOFF, target + FIRST_SPHERE_TOLERANCE)


def _retain_first_sphere(candidates, metal):
    """Discard broad-shell candidates that are not plausible direct bonds."""
    retained = []
    unsupported_pairs = set()
    for candidate in candidates:
        neighbor = candidate["neighbor"]
        cutoff = _first_sphere_cutoff(metal, neighbor)
        if not math.isfinite(cutoff):
            unsupported_pairs.add((metal.element, neighbor.element))
            continue
        if candidate["distance_raw"] <= cutoff + SEARCH_EPSILON:
            candidate["first_sphere_cutoff"] = cutoff
            retained.append(candidate)
    return retained, unsupported_pairs


def _collect_contacts(structure, search, metal, include_symmetry):
    """Return unique first-sphere contacts for one metal and search scope."""
    candidates = []
    marks = search.find_atoms(metal.pos, "\x00", min_dist=0.0,
                              radius=CUTOFF + SEARCH_EPSILON)
    for mark in marks:
        neighbor = structure.atom_for_mark(mark)
        if neighbor is None:
            continue
        residue = structure.residue_for_atom(neighbor)
        if neighbor.element not in ("N", "O", "S"):
            continue
        if not (neighbor.occupancy_valid and neighbor.occupancy > 0.0):
            continue
        if not (residue.is_water or residue.residue_name in AA):
            continue

        if include_symmetry:
            nearest = structure.structure.cell.find_nearest_pbc_image(
                metal.pos, neighbor.pos, mark.image_idx)
            transformed = structure.structure.cell.find_nearest_pbc_position(
                metal.pos, neighbor.pos, mark.image_idx)
            translation = tuple(int(value) for value in nearest.pbc_shift)
            image_index = int(nearest.sym_idx)
            (
                crystallographic_contact,
                strict_ncs_contact,
                strict_ncs_operation_id,
                contact_scope,
            ) = structure.image_provenance(image_index, translation)
            symmetry_contact = (
                crystallographic_contact or strict_ncs_contact)
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
            distance = math.sqrt(
                (metal.x - neighbor.x) ** 2 +
                (metal.y - neighbor.y) ** 2 +
                (metal.z - neighbor.z) ** 2)

        # A symmetry copy is a distinct residue image. Exclude only the actual
        # source residue in the explicit asymmetric unit.
        if neighbor.residue_key == metal.residue_key and not symmetry_contact:
            continue
        if not (0.0 < distance <= CUTOFF + 1e-9):
            continue

        position = (float(transformed.x), float(transformed.y), float(transformed.z))
        candidates.append({
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
        })

    first_sphere, unsupported_pairs = _retain_first_sphere(candidates, metal)
    return (_deduplicate_special_position_contacts(first_sphere),
            unsupported_pairs)


def _annotate_contacts(contacts, metal_element, dpi):
    for contact in contacts:
        neighbor = contact["neighbor"]
        reported_distance = round(contact["distance_raw"], 3)
        literature = LIT.get(_bonding_key(neighbor, neighbor.residue_name,
                                          metal_element))
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
            geometry_outlier=(abs(zscore) >= ZSCORE_OUTLIER_CUTOFF
                              if math.isfinite(zscore) else ""),
            geometry_consistent=(abs(zscore) < ZSCORE_OUTLIER_CUTOFF
                                 if math.isfinite(zscore) else ""),
        )


def _scope_summary(contacts, metal_zero_occupancy, unavailable=False):
    if unavailable:
        return {
            "candidate": NAN, "covered": NAN, "outlier": NAN,
            "consistent": NAN, "coverage": NAN, "status": "",
        }
    candidate = len(contacts)
    covered = sum(bool(contact["reference_covered"]) for contact in contacts)
    outlier = sum(contact["geometry_outlier"] is True for contact in contacts)
    consistent = sum(contact["geometry_consistent"] is True for contact in contacts)
    assessable = outlier + consistent
    if metal_zero_occupancy or assessable == 0:
        status = "insufficient data"
    elif outlier:
        status = "suspect"
    else:
        status = "plausible"
    coverage = round(covered / candidate, 4) if candidate else NAN
    return {
        "candidate": candidate,
        "covered": covered,
        "outlier": outlier,
        "consistent": consistent,
        "coverage": coverage,
        "status": status,
    }


def _site_summary(metal, explicit_contacts, image_contacts,
                  dpi, resolution, ni, deposited_ni, dpi_reason, structure):
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
        if image_search_available else NAN
    )
    crystallographic_count = (
        sum(contact["crystallographic_contact"] for contact in image_contacts)
        if image_search_available else NAN
    )
    strict_ncs_count = (
        sum(contact["strict_ncs_contact"] for contact in image_contacts)
        if image_search_available else NAN
    )
    combined_count = (
        sum(
            contact["crystallographic_contact"] and
            contact["strict_ncs_contact"]
            for contact in image_contacts
        )
        if image_search_available else NAN
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
    elif primary["consistent"] + primary["outlier"] == 0:
        reasons.append("no_assessable_reference_contacts")
    return {
        "dpi": dpi,
        "resolution": resolution,
        "occupancy_weighted_atom_count": (
            round(ni, 6) if math.isfinite(ni) else NAN),
        "deposited_occupancy_weighted_atom_count": (
            round(deposited_ni, 6)
            if math.isfinite(deposited_ni) else NAN),
        "dpi_atom_count_multiplier": structure.dpi_atom_count_multiplier,
        "strict_ncs_operation_count": structure.strict_ncs_operation_count,
        "crystallographic_operation_count": (
            structure.crystallographic_operation_count),
        "dpi_unavailable_reason": dpi_reason,
        "candidate_contact_count": primary["candidate"],
        "reference_covered_contact_count": primary["covered"],
        "geometry_outlier_contact_count": primary["outlier"],
        "geometry_consistent_contact_count": primary["consistent"],
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
        "coordination_depends_on_crystallographic_symmetry": (
            depends_crystallographic),
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
        "metal_chain_index": metal.chain_index if metal else "",
        "metal_residue_index": metal.residue_index if metal else "",
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
            residue.selected_conformer_mean_occupancy if residue else NAN),
        "metal_altloc_options": residue.altloc_options if residue else "",
        "alternative_conformers_present": (
            residue.alternative_conformers_present if residue else ""),
        "altloc_selection_fallback": (
            residue.altloc_selection_fallback if residue else ""),
        "symmetry_search_available": structure.symmetry_search_available,
        "symmetry_search_failure_reason": structure.symmetry_search_failure_reason,
        "strict_ncs_operation_count": structure.strict_ncs_operation_count,
        "crystallographic_operation_count": (
            structure.crystallographic_operation_count),
        "dpi_atom_count_multiplier": structure.dpi_atom_count_multiplier,
        "occupancy_validation_failed": structure.occupancy_validation_failed,
        "missing_occupancy_count": structure.missing_occupancy_count,
        "invalid_occupancy_count": structure.invalid_occupancy_count,
        "zero_occupancy_atom_count": structure.zero_occupancy_atom_count,
        "duplicate_atom_records_present": structure.duplicate_atom_records_present,
        "duplicate_atom_record_count": structure.duplicate_atom_record_count,
        "duplicate_atom_coordinate_conflict_count": (
            structure.duplicate_coordinate_conflict_count),
        "malformed_duplicate_atom_name_count": (
            structure.malformed_duplicate_atom_name_count),
        "raw_occupancy_mapping_failed": structure.raw_occupancy_mapping_failed,
        "raw_occupancy_mapping_failure_reason": (
            structure.raw_occupancy_mapping_failure_reason),
        "unknown_element_atom_count": structure.unknown_element_atom_count,
        "element_validation_warning": structure.element_validation_warning,
        "zscore_outlier_cutoff": ZSCORE_OUTLIER_CUTOFF,
    }
    for column in STATS_EXTRA_COLUMNS:
        values.setdefault(column, summary.get(column, ""))
    values.update({key: value for key, value in summary.items()
                   if key in STATS_EXTRA_COLUMNS})
    return values


def _bond_row(pdb_id, structure, metal, contact, dpi, resolution,
              sigma, parent_type):
    neighbor = contact["neighbor"]
    metal_residue = structure.residue_for_atom(metal)
    neighbor_residue = structure.residue_for_atom(neighbor)
    x, y, z = contact["transformed_position"]
    tx, ty, tz = contact["translation"]
    mag, neg, pos = sigma
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
        "literature_distance": contact["literature_distance"],
        "literature_stdev": contact["literature_stdev"],
        "zscore": contact["zscore"],
        "dpi": dpi,
        "resolution": resolution,
        "sigma_mag": mag, "sigma_neg": neg, "sigma_pos": pos,
        "parent_type": parent_type,
        "bonded_to": _bonded_to(neighbor.is_water),
        "model_id": structure.analyzed_model_id,
        "metal_model_index": metal.model_index,
        "metal_chain_index": metal.chain_index,
        "metal_residue_index": metal.residue_index,
        "metal_atom_index": metal.atom_index,
        "metal_atom": metal.atom_name,
        "metal_icode": metal.insertion_code,
        "metal_altloc": metal.altloc,
        "metal_occupancy": metal.occupancy,
        "metal_occupancy_valid": metal.occupancy_valid,
        "metal_occupancy_status": metal.occupancy_status,
        "metal_conformer_mean_occupancy": (
            metal_residue.selected_conformer_mean_occupancy),
        "metal_altloc_options": metal_residue.altloc_options,
        "metal_altloc_selection_fallback": (
            metal_residue.altloc_selection_fallback),
        "neighbor_chain": neighbor.chain_id,
        "neighbor_resnum": neighbor.resnum,
        "neighbor_icode": neighbor.insertion_code,
        "neighbor_model_index": neighbor.model_index,
        "neighbor_chain_index": neighbor.chain_index,
        "neighbor_residue_index": neighbor.residue_index,
        "neighbor_atom_index": neighbor.atom_index,
        "neighbor_altloc": neighbor.altloc,
        "neighbor_occupancy": neighbor.occupancy,
        "neighbor_occupancy_valid": neighbor.occupancy_valid,
        "neighbor_occupancy_status": neighbor.occupancy_status,
        "neighbor_conformer_mean_occupancy": (
            neighbor_residue.selected_conformer_mean_occupancy),
        "neighbor_altloc_options": neighbor_residue.altloc_options,
        "neighbor_altloc_selection_fallback": (
            neighbor_residue.altloc_selection_fallback),
        "alternative_conformers_present": (
            metal_residue.alternative_conformers_present or
            neighbor_residue.alternative_conformers_present),
        "altloc_selection_fallback": (
            metal_residue.altloc_selection_fallback or
            neighbor_residue.altloc_selection_fallback),
        "neighbor_class": "water" if neighbor.is_water else "amino_acid",
        "candidate_contact": True,
        "reference_covered": contact["reference_covered"],
        "geometry_outlier": contact["geometry_outlier"],
        "geometry_consistent": contact["geometry_consistent"],
        "zscore_outlier_cutoff": ZSCORE_OUTLIER_CUTOFF,
        "contact_scope": contact["contact_scope"],
        "symmetry_contact": contact["symmetry_contact"],
        "crystallographic_contact": contact["crystallographic_contact"],
        "strict_ncs_contact": contact["strict_ncs_contact"],
        "strict_ncs_operation_id": contact["strict_ncs_operation_id"],
        "symmetry_image_index": contact["symmetry_image_index"],
        "symmetry_operation": contact["symmetry_operation"],
        "cell_translation_x": tx, "cell_translation_y": ty,
        "cell_translation_z": tz,
        "transformed_neighbor_x": round(x, 6),
        "transformed_neighbor_y": round(y, 6),
        "transformed_neighbor_z": round(z, 6),
    }


def run_bond_analysis(pdbID, pdb_path, entry_dir, stats_rows, header,
                      dpi_inputs, structure=None):
    """Return ``(contact_rows, site_summaries, entry_metadata)``.

    Only external first-coordination-sphere contacts are emitted; atoms in the
    metal's own residue remain excluded. Image-inclusive contacts are the
    primary rows when symmetry metadata is available. Explicit-only and
    image-inclusive summaries are retained per metal site, and every generated
    contact is classified as crystallographic, strict NCS, or both. Missing DPI
    does not prevent first-sphere classification or distance reporting.
    """
    del entry_dir  # retained in the call signature for compatibility
    if structure is None:
        structure = load_structure(pdbID, pdb_path)

    metals_in_model = structure.metal_atoms(METAL_ELEMENTS, canonical=True)
    metadata = {
        "partial_reason_codes": [],
        "warning_codes": list(structure.warning_codes),
        "messages": [],
        "retryable": False,
    }
    if not metals_in_model:
        return [], {}, metadata

    dpi, resolution, dpi_reason = _calculate_dpi_details(structure, dpi_inputs)
    ni = _count_ni(structure)
    deposited_ni = count_deposited_ni(structure)
    if dpi_reason:
        metadata["partial_reason_codes"].append(dpi_reason)
        metadata["messages"].append(f"DPI unavailable: {dpi_reason}")
    if not structure.symmetry_search_available:
        metadata["partial_reason_codes"].append("symmetry_search_unavailable")
        metadata["messages"].append(
            "symmetry search unavailable: " +
            (structure.symmetry_search_failure_reason or "unknown reason"))

    explicit_search = structure.make_neighbor_search(
        CUTOFF + SEARCH_EPSILON, include_symmetry=False,
        positive_occupancy_only=True)
    image_search = None
    if structure.symmetry_search_available:
        image_search = structure.make_neighbor_search(
            CUTOFF + SEARCH_EPSILON, include_symmetry=True,
            positive_occupancy_only=True)
    sig = _sigma_index(stats_rows)
    zd_idx = _zd_indices(header)

    rows = []
    summaries = {}
    for metal in metals_in_model:
        explicit, unsupported_pairs = _collect_contacts(
            structure, explicit_search, metal, False)
        _annotate_contacts(explicit, metal.element, dpi)
        image_contacts = None
        if image_search is not None:
            image_contacts, image_unsupported = _collect_contacts(
                structure, image_search, metal, True)
            unsupported_pairs.update(image_unsupported)
            _annotate_contacts(image_contacts, metal.element, dpi)
        if unsupported_pairs:
            metadata["partial_reason_codes"].append(
                "missing_first_sphere_reference")
            pairs = ", ".join(
                f"{metal_element}-{donor_element}"
                for metal_element, donor_element in sorted(unsupported_pairs))
            metadata["messages"].append(
                f"first-sphere reference unavailable for {pairs}")
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
        summaries[metal.source_key] = summary
        if summary["metal_zero_occupancy"]:
            metadata["partial_reason_codes"].append("metal_zero_occupancy")
            metadata["messages"].append(
                f"zero-occupancy metal: {metal.chain_id}/{metal.resnum}/"
                f"{metal.atom_name}")

        primary_contacts = (
            image_contacts if image_contacts is not None else explicit)
        sigma = _sigma_for(sig, metal.residue_name, metal.chain_id,
                           metal.resnum, zd_idx)
        parent_type = _parent_type(structure, metal, metal.residue_name,
                                   metal.element)
        rows.extend(_bond_row(pdbID, structure, metal, contact, dpi,
                              resolution, sigma, parent_type)
                    for contact in primary_contacts)

    metadata["partial_reason_codes"] = list(dict.fromkeys(
        metadata["partial_reason_codes"]))
    metadata["warning_codes"] = list(dict.fromkeys(metadata["warning_codes"]))
    metadata["messages"] = list(dict.fromkeys(metadata["messages"]))
    return rows, summaries, metadata
