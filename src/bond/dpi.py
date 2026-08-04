"""The diffraction-component precision index, and the crystal metadata it needs.

DPI is a whole-structure number -- how precisely this refinement located its
atoms, in Angstroms -- and it is what turns a measured metal-donor distance
into a z-score that means something. It touches no bond and knows nothing about
coordination chemistry, which is why it lives apart from ``bond_analysis``.

    DPI = 1.28 * ni**0.5 * va**(1/3) * nobs**(-5/6) * rfree     (Blow 2002, eq. 7)

Every input can be absent: manual runs may have no ``data.json``, a coordinate
file may carry no CRYST1 record, an older deposition may not report R-free.
Nothing here raises for any of that. Each function degrades to ``NAN`` and
``_calculate_dpi_details`` returns a reason code naming which input was
missing, so the caller still emits the geometry it did measure and the manifest
still says why the z-score is blank.
"""

import json
import math
import re

from structure_analysis import NAN, count_ni


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
