"""Metal-ligand bond-distance stage (generalized port of the legacy Fe script).

For one PDB entry this finds every metal atom, does a Biopython neighbor search,
and for each metal-ligand contact within 4 A computes the bond length and a
resolution-aware z-score against the consolidated literature reference distances
in ``metal_distances_info.txt`` (Harding 2006 and Zheng et al. 2008 [Ni only]):

    z = (d_observed - mu) / sqrt(DPI**2 + sigma_lit**2)

The DPI (Diffraction-component Precision Index, Blow 2002 eq. 7) is the model's
per-atom coordinate uncertainty; adding it in quadrature with the literature
spread makes the same absolute deviation more significant in a high-resolution
structure than in a low-resolution one.

This is a clean, self-contained generalization of
``fe_biopython_analysis_dpi_final.py`` (which only handled Fe, read a multi-source
literature file that no longer exists, and scraped DPI inputs that are absent from
PDB-REDO files). ``main.py`` calls ``run_bond_analysis`` from its per-entry worker.

Notable differences from the legacy script (all deliberate):
* All metals (``Analysisv2_kn.metals`` + ``uncommonMetals``), not just Fe.
* One z-score from the single Harding set; the old pdb2/pdb3/csd2/csd3 columns
  are gone.
* Bonding keys are exact ``(residue, atom, metal)`` tuples looked up in a dict --
  the old code built a comma-delimited key for N/S atoms and substring-matched,
  so His-N and Cys-S bonds silently never matched the space-delimited file.
* DPI inputs come from PDB-REDO ``data.json`` (NREFCNT, RFFIN) and the ASU volume
  is computed directly from the crystal cell / symmetry (gemmi), not from the
  unavailable Matthews coefficient. Missing inputs yield NaN rather than dropping.
* edstats sigma values are joined from the in-memory ``run_analysis`` rows, not
  from legacy per-structure ``{atom}_Data`` dump files.
"""
import json
import math
import os
import re

from Analysisv2_kn import metals, uncommonMetals

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Any element symbol we treat as a metal of interest (matched against the PDB
# element column, which is more reliable than residue-name matching).
METAL_ELEMENTS = set(metals) | set(uncommonMetals)

# Coordinating residues (and water) for which the reference table has data. The
# legacy list duplicated CYS; deduped here. A neighbor whose residue is not in
# this set is ignored.
AA = {"ASP", "CYS", "GLU", "HIS", "TYR", "THR", "SER", "HOH"}

# Max metal-ligand interaction distance (Gucwa et al. 2024 / CMM program).
CUTOFF = 4.0

NAN = float("nan")

# Metallocofactor classifications, used only to tag each metal's environment in parent_type. CLUSTER is
# checked before HEMES, matching the legacy precedence for the codes in both lists.
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

# Fixed output schema; main.py imports this so the module and driver never drift.
BOND_COLUMNS = [
    "pdbID", "metal_resname", "metal_chain", "metal_resnum", "metal_element",
    "neighbor_resname", "neighbor_atom", "neighbor_element", "distance",
    "literature_distance", "literature_stdev", "zscore", "dpi", "resolution",
    "sigma_mag", "sigma_neg", "sigma_pos", "parent_type", "bonded_to",
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


# --------------------------------------------------------------------------- #
# DPI
# --------------------------------------------------------------------------- #
def _asu_volume(mtz_path, pdb_path):
    """Asymmetric-unit volume (A^3) = unit-cell volume / number of symmetry ops.

    This is the quantity the legacy code approximated via ``mw * Matthews``;
    computing it from the cell + space group is exact and needs no header scrape.
    Prefer the MTZ (matches the diffraction data); fall back to the PDB CRYST1.
    """
    import gemmi  # lazy: only the metal/biotools/vinda envs have it
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

def load_structure(pdbID, pdb_path):
    """Parse pdb_path once via Biopython; shared by run_analysis and
    run_bond_analysis so a single entry's coordinate file is only read once."""
    from Bio.PDB import PDBParser  # lazy import
    return PDBParser(QUIET=True).get_structure(pdbID, pdb_path)

def _count_ni(structure):
    """Atoms in the model plus water-residue count.

    Ni represents the effective number of fully-occupied atom
    sites; a partial-occupancy atom contributes less scattering signal than
    a full atom. Each atom is weighted by its occupancy (0-1) rather than
    counted as a flat 1, as Blow (2002) indicates that the simplified equations assume
    full occupancy of every atom.
    """
    n_atoms = 0.0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.upper() == "H":
                        continue  # riding atoms, not independently refined
                    n_atoms += atom.get_occupancy()
    return n_atoms


def calculate_dpi(structure, dpi_inputs):
    """Return ``(dpi, resolution)``; either may be NaN. Never raises.

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
    try:
        props = {}
        try:
            with open(dpi_inputs["data_json"]) as f:
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
            return NAN, resolution
        if not (math.isfinite(nobs) and math.isfinite(rfree) and math.isfinite(va)
                and nobs > 0 and rfree > 0 and va > 0 and ni > 0):
            return NAN, resolution
        dpi = 1.28 * (ni ** 0.5) * (va ** (1 / 3)) * (nobs ** (-5 / 6)) * rfree
        return round(dpi, 4), resolution
    except Exception:
        return NAN, resolution


# --------------------------------------------------------------------------- #
# Bond rows
# --------------------------------------------------------------------------- #
def _bonding_key(neighbor, nb_res, metal_el):
    """Exact (residue, atom, metal) key matching metal_distances_info.txt columns."""
    name = neighbor.get_name()
    if nb_res == "HOH":
        return ("HOH", "O", metal_el)
    if name == "O":               # backbone carbonyl O -> literal "CA" row
        return ("CA", "O", metal_el)
    if name.startswith("O"):      # side-chain O (OD1/OE1/OG/OH/...)
        return (nb_res, "O", metal_el)
    return (nb_res, neighbor.element.upper(), metal_el)  # His N, Cys S, ...


def _parent_type(metal, metal_res, metal_el):
    if metal_res in CLUSTER:
        return "cluster"
    if metal_res in HEMES:
        return "heme"
    if metal_el not in METAL_ELEMENTS:
        return "other"  # defensive: shouldn't happen, metal_atoms is pre-filtered
    residue = metal.get_parent()
    if sum(1 for _ in residue) == 1:   # lone atom residue = free ion
        return "ion"
    return "other"


def _bonded_to(nb_res):
    if nb_res == "HOH":
        return "HOH"
    if nb_res in AA:
        return "P"
    return nb_res


def _zscore(dist, mu, stdev, dpi):
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


def run_bond_analysis(pdbID, pdb_path, entry_dir, stats_rows, header, dpi_inputs, structure=None):
    """Return a list of bond-row dicts (keys == BOND_COLUMNS) for one entry.

    Parses ``pdb_path`` (the coordinate file edstats consumed), finds all metal
    atoms, and emits one row per metal-ligand contact within CUTOFF to a
    coordinating residue/water. Rows with no literature reference or a NaN DPI
    keep the measured distance and carry NaN in the derived columns.

    `header` is the edstats column header (from run_analysis) used to locate
    the ZDm/ZD-m/ZD+m sigma columns; if those columns aren't present, sigma
    values are emitted as NaN rather than raising.

    aa_coverage_by_metal maps (resname, chain, resnum) -> aa_geometry_coverage: the
    fraction of a metal's coordinating contacts (N/O/S, within CUTOFF, to an
    AA-listed residue -- i.e. exactly the contacts that would produce a row
    below) that have a literature reference bond length. NaN if the metal has
    no such contacts at all; 0.0 if it has contacts but none have a reference.
    """
    from Bio.PDB import NeighborSearch  # lazy import

    if structure is None:
        structure = load_structure(pdbID, pdb_path)

    atoms = list(structure.get_atoms())

    metal_atoms = [a for a in atoms if a.element.upper() in METAL_ELEMENTS]
    if not metal_atoms:
        return [], {}

    dpi, resolution = calculate_dpi(structure, dpi_inputs)
    sig = _sigma_index(stats_rows)
    zd_idx = _zd_indices(header)
    ns = NeighborSearch(atoms)

    rows = []
    contacts_by_key = {}
    with_ref_by_key = {}


    for metal in metal_atoms:
        metal_el = metal.element.upper()
        metal_res = metal.get_parent().get_resname()
        chain = metal.get_full_id()[2]
        _, resseq, icode = metal.get_parent().get_id()
        resnum = f"{resseq}{icode.strip()}"
        ptype = _parent_type(metal, metal_res, metal_el)
        mag, neg, pos = _sigma_for(sig, metal_res, chain, resnum, zd_idx)

        key = (metal_res, str(chain), str(resnum))
        contacts_by_key.setdefault(key, 0)
        for neighbor in ns.search(metal.coord, CUTOFF, level="A"):
            nb_res = neighbor.get_parent().get_resname()
            if nb_res == metal_res:      # self / same cofactor
                continue
            if nb_res not in AA:         # no reference data for this residue
                continue
            if neighbor.element.upper() not in ("O", "N", "S"):
                continue                 # only N/O/S are ligand donors here (skip C, etc.)
            dist = metal - neighbor      # Biopython: Euclidean distance (A)
            if dist > CUTOFF:
                continue
            dist = round(dist, 3)
            if dist == 0.0:
                continue

            lit = LIT.get(_bonding_key(neighbor, nb_res, metal_el))
            contacts_by_key[key] = contacts_by_key.get(key, 0) + 1
            if lit is None:
                mu = stdev = zscore = NAN
            else:
                with_ref_by_key[key] = with_ref_by_key.get(key, 0) + 1
                mu, stdev = lit
                zscore = _zscore(dist, mu, stdev, dpi)

            rows.append({
                "pdbID": pdbID,
                "metal_resname": metal_res,
                "metal_chain": chain,
                "metal_resnum": resnum,
                "metal_element": metal_el,
                "neighbor_resname": nb_res,
                "neighbor_atom": neighbor.get_name(),
                "neighbor_element": neighbor.element,
                "distance": dist,
                "literature_distance": mu,
                "literature_stdev": stdev,
                "zscore": zscore,
                "dpi": dpi,
                "resolution": resolution,
                "sigma_mag": mag,
                "sigma_neg": neg,
                "sigma_pos": pos,
                "parent_type": ptype,
                "bonded_to": _bonded_to(nb_res),
            })
        
        aa_coverage_by_metal = {
        key: round(with_ref_by_key.get(key, 0) / n_contacts, 4) if n_contacts > 0 else NAN
        for key, n_contacts in contacts_by_key.items()
    }
    return rows, aa_coverage_by_metal
