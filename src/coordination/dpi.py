"""Calculate the diffraction-component precision index and its input metadata.

    DPI = 1.28 * ni**0.5 * va**(1/3) * nobs**(-5/6) * rfree

Uses Blow (2002), equation 7. Missing inputs yield NaN and a reason code so
measured geometry can still be reported.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import gemmi

from codes import ReasonCode
from structure_analysis import NAN, StructureContext, count_ni, spacegroup_or_none


@dataclass(frozen=True, slots=True)
class DpiInputs:
    """Entry-level metadata the DPI formula and its fallbacks read.

    ``resolution`` is recorded alongside the DPI but is not a term of the
    formula. Without ``data_json`` the DPI is unavailable by construction.
    """

    resolution: float
    data_json: str | None = None
    pdb_path: str | None = None
    mtz_path: str | None = None


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

    This rejects the sentinel cell Gemmi reports when no cell record was
    present (it tests ``a == 1``), which must not be used to calculate DPI.
    """
    return not cell.is_crystal()


def asu_volume(
    mtz_path: str, pdb_path: str, structure: gemmi.Structure | None = None
) -> float:
    """Asymmetric-unit volume (A^3) = unit-cell volume / number of symmetry ops.

    Prefer the MTZ, which matches the diffraction data; fall back to CRYST1.
    Only the cell and symmetry are needed, so the reflections are not read.
    ``structure`` is the already-parsed coordinate file when the caller has
    one, which spares the fallback a second parse of ``pdb_path``.
    """
    cell: gemmi.UnitCell | None = None
    sg: gemmi.SpaceGroup | None = None
    try:
        mtz = gemmi.read_mtz_file(mtz_path, with_data=False)
        cell, sg = mtz.cell, mtz.spacegroup
    except Exception:
        cell = sg = None
    if cell is None or sg is None or cell.volume <= 0 or _is_placeholder_cell(cell):
        try:
            st = gemmi.read_structure(pdb_path) if structure is None else structure
            cell = st.cell
            # ``find_spacegroup`` reads the cell angles too, so rhombohedral
            # axes keep their three-operation setting instead of the
            # nine-operation hexagonal one a name lookup alone would pick.
            sg = spacegroup_or_none(st)
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
        with open(pdb_path, encoding="utf-8", errors="replace") as f:
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
    except (OSError, ValueError):
        pass
    return NAN


def _unavailable(
    reason: str,
    resolution: float,
    *,
    rfree: float = NAN,
    nobs: float = NAN,
    va: float = NAN,
) -> DpiComponents:
    """Return a NaN DPI with ``reason`` and whatever inputs were already read."""
    return DpiComponents(
        dpi=NAN,
        resolution=resolution,
        reason_code=reason,
        r_free=rfree,
        reflection_count=nobs,
        asu_volume=va,
    )


def _read_pdb_redo_properties(data_json: str) -> dict[str, Any]:
    """Return the ``properties`` block of a PDB-REDO ``data.json``.

    An unreadable, unparseable or wrongly shaped file yields an empty block, so
    every term is then reported as missing rather than as a failed calculation.
    """
    try:
        with open(data_json, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    properties = data.get("properties")
    block: dict[str, Any] = properties if isinstance(properties, dict) else {}
    return block


def _metadata_terms(properties: dict[str, Any], pdb_path: str) -> tuple[float, float]:
    """Return ``(rfree, nobs)`` from PDB-REDO properties, or the header fallback.

    Raises ``TypeError`` or ``ValueError`` when a present value is not numeric.
    """
    nobs = properties.get("NREFCNT")
    rfree = properties.get("RFFIN")
    rfree = (
        float(rfree) if rfree is not None and rfree != "" else rfree_from_pdb(pdb_path)
    )
    nobs = float(nobs) if nobs is not None and nobs != "" else NAN
    return rfree, nobs


def _invalid_term_reason(
    structure: StructureContext, nobs: float, rfree: float, va: float, ni: float
) -> str:
    """Name the first formula term that rules out the DPI, in reporting order.

    This is the only statement of what the formula needs: an empty string means
    every term is usable.
    """
    if structure.occupancy.validation_failed:
        return ReasonCode.INVALID_OCCUPANCY
    if not math.isfinite(nobs) or nobs <= 0:
        return ReasonCode.MISSING_OR_INVALID_REFLECTION_COUNT
    if not math.isfinite(rfree) or rfree <= 0:
        return ReasonCode.MISSING_OR_INVALID_RFREE
    if not math.isfinite(va) or va <= 0:
        return ReasonCode.MISSING_OR_INVALID_ASU_VOLUME
    if not math.isfinite(ni) or ni <= 0:
        return ReasonCode.INVALID_DPI_ATOM_COUNT
    return ""


def calculate_dpi_components(
    structure: StructureContext, dpi_inputs: DpiInputs
) -> DpiComponents:
    """Return the DPI, its status, and its reusable numeric inputs. Never raises.

    Resolution is metadata only: it is implicit in va and nobs, not a term of
    the formula.
    """
    resolution = float(dpi_inputs.resolution)
    data_json = dpi_inputs.data_json
    if not data_json:
        # Distinguish absent metadata from a failed DPI calculation.
        return _unavailable(ReasonCode.MISSING_DPI_METADATA_SOURCE, resolution)

    rfree = nobs = va = NAN
    try:
        properties = _read_pdb_redo_properties(data_json)
        try:
            rfree, nobs = _metadata_terms(properties, dpi_inputs.pdb_path or "")
        except (TypeError, ValueError):
            # Present but not numeric: a metadata defect, not a failed calculation.
            return _unavailable(ReasonCode.INVALID_DPI_METADATA, resolution)
        va = asu_volume(
            dpi_inputs.mtz_path or "",
            dpi_inputs.pdb_path or "",
            structure.structure,
        )
        ni = count_ni(structure)

        reason = _invalid_term_reason(structure, nobs, rfree, va, ni)
        if reason:
            return _unavailable(
                reason,
                resolution,
                rfree=rfree,
                nobs=nobs,
                va=va,
            )
        dpi = 1.28 * (ni**0.5) * (va ** (1 / 3)) * (nobs ** (-5 / 6)) * rfree
        return DpiComponents(
            dpi=round(dpi, 4),
            resolution=resolution,
            reason_code="",
            r_free=rfree,
            reflection_count=nobs,
            asu_volume=va,
        )
    except Exception:
        # Anything the guarded reads above did not anticipate.
        return _unavailable(
            ReasonCode.DPI_CALCULATION_FAILED,
            resolution,
            rfree=rfree,
            nobs=nobs,
            va=va,
        )
