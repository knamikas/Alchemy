"""The diffraction-component precision index, and the crystal metadata it needs.

    DPI = 1.28 * ni**0.5 * va**(1/3) * nobs**(-5/6) * rfree     (Blow 2002, eq. 7)

Every input can be absent: manual runs may have no ``data.json``, a coordinate
file may carry no CRYST1 record, an older deposition may not report R-free.
Nothing here raises for any of that. Each function degrades to ``NAN`` and
``_calculate_dpi_details`` returns a reason code naming which input was
missing, so the caller still emits the geometry it did measure.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

from codes import ReasonCode
from structure_analysis import NAN, StructureContext, count_ni

if TYPE_CHECKING:
    import gemmi


def _is_placeholder_cell(cell: gemmi.UnitCell) -> bool:
    """Whether ``cell`` is Gemmi's stand-in for a file with no CRYST1 record.

    ``UnitCell.is_crystal()`` is false only for the exact 1 x 1 x 1 default a
    file with no usable cell parses to. That volume is smaller than one
    non-hydrogen atom, so accepting it would give every contact in the entry a
    confident-looking z-score off an impossible asymmetric unit.
    """
    return not cell.is_crystal()


def _asu_volume(mtz_path: str, pdb_path: str) -> float:
    """Asymmetric-unit volume (A^3) = unit-cell volume / number of symmetry ops.

    Prefer the MTZ, which matches the diffraction data; fall back to CRYST1.
    """
    import gemmi

    cell: Optional[gemmi.UnitCell]
    sg: Optional[gemmi.SpaceGroup]
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
            # Gemmi returns None for a name it does not recognize, but its
            # bundled stub declares a plain SpaceGroup return.
            sg = cast(
                Optional["gemmi.SpaceGroup"],
                gemmi.find_spacegroup_by_name(st.spacegroup_hm),
            )
        except Exception:
            return NAN
    # ``st.cell`` is a value member, so only the space group can still be
    # absent on the fallback path.
    if sg is None or cell.volume <= 0 or _is_placeholder_cell(cell):
        return NAN
    nops = len(list(sg.operations()))
    return cell.volume / nops if nops > 0 else NAN


def _rfree_from_pdb(pdb_path: str) -> float:
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


def _calculate_dpi_details(
    structure: StructureContext, dpi_inputs: Mapping[str, Any]
) -> tuple[float, float, str]:
    """Return ``(dpi, resolution, reason_code)``. Never raises.

    Resolution is metadata only: it is implicit in va and nobs, not a term of
    the formula.
    """
    resolution = dpi_inputs.get("resolution", NAN)
    try:
        resolution = float(resolution)
    except (TypeError, ValueError):
        resolution = NAN

    data_json = dpi_inputs.get("data_json")
    if not data_json:
        # Manual input mode without --data-json: the reflection count has no
        # source at all, which is a different answer from a calculation that
        # ran and failed.
        return NAN, resolution, ReasonCode.MISSING_DPI_METADATA_SOURCE

    try:
        props: dict[str, Any] = {}
        try:
            with open(data_json) as f:
                props = json.load(f).get("properties", {})
        except (OSError, ValueError):
            props = {}

        nobs = props.get("NREFCNT")
        rfree = props.get("RFFIN")
        rfree = (
            float(rfree)
            if rfree is not None and rfree != ""
            else _rfree_from_pdb(dpi_inputs["pdb_path"])
        )
        nobs = float(nobs) if nobs is not None and nobs != "" else NAN
        va = _asu_volume(dpi_inputs["mtz_path"], dpi_inputs["pdb_path"])
        ni = count_ni(structure)

        if not all(isinstance(x, (float, int)) for x in (nobs, rfree, va)):
            return NAN, resolution, ReasonCode.INVALID_DPI_METADATA
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
                reason = ReasonCode.INVALID_OCCUPANCY
            elif not math.isfinite(nobs) or nobs <= 0:
                reason = ReasonCode.MISSING_OR_INVALID_REFLECTION_COUNT
            elif not math.isfinite(rfree) or rfree <= 0:
                reason = ReasonCode.MISSING_OR_INVALID_RFREE
            elif not math.isfinite(va) or va <= 0:
                reason = ReasonCode.MISSING_OR_INVALID_ASU_VOLUME
            else:
                reason = ReasonCode.INVALID_DPI_ATOM_COUNT
            return NAN, resolution, reason
        dpi = 1.28 * (ni**0.5) * (va ** (1 / 3)) * (nobs ** (-5 / 6)) * rfree
        return round(dpi, 4), resolution, ""
    except Exception:
        return NAN, resolution, ReasonCode.DPI_CALCULATION_FAILED
