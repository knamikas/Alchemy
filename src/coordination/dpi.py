"""Calculate the diffraction-component precision index and its input metadata.

    DPI = 1.28 * ni**0.5 * va**(1/3) * nobs**(-5/6) * rfree

Uses Blow (2002), equation 7. Missing inputs yield NaN and a reason code so
measured geometry can still be reported.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, cast

from codes import ReasonCode
from structure_analysis import NAN, StructureContext, count_ni

if TYPE_CHECKING:
    import gemmi


@dataclass(frozen=True, slots=True)
class DpiComponents:
    """The DPI result plus the entry-level inputs used to calculate it."""

    dpi: float
    resolution: float
    reason_code: str
    r_free: float
    reflection_count: float
    asu_volume: float


def _is_placeholder_cell(cell: gemmi.UnitCell) -> bool:
    """Return whether cell is Gemmi's default for missing crystal metadata.

    The default 1 x 1 x 1 cell is too small for a physical asymmetric unit and
    must not be used to calculate DPI.
    """
    return not cell.is_crystal()


def asu_volume(mtz_path: str, pdb_path: str) -> float:
    """Asymmetric-unit volume (A^3) = unit-cell volume / number of symmetry ops.

    Prefer the MTZ, which matches the diffraction data; fall back to CRYST1.
    """
    import gemmi

    cell: gemmi.UnitCell | None
    sg: gemmi.SpaceGroup | None
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


def rfree_from_pdb(pdb_path: str) -> float:
    """Fallback R-free scrape from a PDB REMARK 3 header (final R-free only)."""
    try:
        with open(pdb_path, encoding="utf-8") as f:
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


def calculate_dpi_components(
    structure: StructureContext, dpi_inputs: Mapping[str, Any]
) -> DpiComponents:
    """Return the DPI, its status, and its reusable numeric inputs. Never raises.

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
        # Distinguish missing manual metadata from a failed DPI calculation.
        return DpiComponents(
            NAN,
            resolution,
            ReasonCode.MISSING_DPI_METADATA_SOURCE,
            NAN,
            NAN,
            NAN,
        )

    rfree_value = NAN
    nobs_value = NAN
    va_value = NAN
    try:
        props: dict[str, Any] = {}
        try:
            with open(data_json, encoding="utf-8") as f:
                props = json.load(f).get("properties", {})
        except (OSError, ValueError):
            props = {}

        nobs = props.get("NREFCNT")
        rfree = props.get("RFFIN")
        try:
            rfree = (
                float(rfree)
                if rfree is not None and rfree != ""
                else rfree_from_pdb(dpi_inputs["pdb_path"])
            )
            nobs = float(nobs) if nobs is not None and nobs != "" else NAN
        except (TypeError, ValueError):
            # Present but not numeric: a metadata defect, not a failed calculation.
            return DpiComponents(
                NAN,
                resolution,
                ReasonCode.INVALID_DPI_METADATA,
                NAN,
                NAN,
                NAN,
            )
        rfree_value = rfree
        nobs_value = nobs
        va = asu_volume(dpi_inputs["mtz_path"], dpi_inputs["pdb_path"])
        va_value = va
        ni = count_ni(structure)

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
            return DpiComponents(
                NAN,
                resolution,
                reason,
                rfree_value,
                nobs_value,
                va_value,
            )
        dpi = 1.28 * (ni**0.5) * (va ** (1 / 3)) * (nobs ** (-5 / 6)) * rfree
        return DpiComponents(
            round(dpi, 4),
            resolution,
            "",
            rfree_value,
            nobs_value,
            va_value,
        )
    except Exception:
        return DpiComponents(
            NAN,
            resolution,
            ReasonCode.DPI_CALCULATION_FAILED,
            rfree_value,
            nobs_value,
            va_value,
        )
