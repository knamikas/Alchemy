"""Review regressions for the DPI inputs in ``src/coordination/dpi.py``.

Each test here pins one boundary the module reads through: the space-group
setting the CRYST1 fallback resolves, a header byte that is not UTF-8, a
``data.json`` whose shape is not the documented one, and the reason codes those
failures must produce instead of the catch-all.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import gemmi
import helpers
import pytest
from helpers import StructureBuilder, approx

import coordination.dpi as dpi_module
from codes import ReasonCode
from structure_analysis import StructureContext, load_structure

#: Cell edges and angles of an ordinary orthogonal 100 A cell.
_CUBIC_CELL = (100.0, 100.0, 100.0, 90.0, 90.0, 90.0)


def _metal_structure(
    tmp_path: Path,
    name: str = "dpi-review.pdb",
    *,
    cell: tuple[float, float, float, float, float, float] = _CUBIC_CELL,
    spacegroup: str = "P 1",
    occupancy: float = 1.0,
) -> str:
    """Write a one-metal coordinate file with a known cell and space group."""
    builder = StructureBuilder(cell=cell, spacegroup=spacegroup)
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0), occupancy=occupancy)
    return builder.write_pdb(tmp_path / name)


def _absent_mtz(tmp_path: Path) -> str:
    """A path with no MTZ on it, which forces the CRYST1 fallback."""
    return os.path.join(str(tmp_path), "absent.mtz")


def _components(
    context: StructureContext,
    path: str,
    data_json: str,
    mtz_path: str | None = None,
) -> dpi_module.DpiComponents:
    """Run the production entry point over one structure and its metadata."""
    return dpi_module.calculate_dpi_components(
        context,
        helpers.dpi_inputs(pdb_path=path, mtz_path=mtz_path, data_json=data_json),
    )


def test_rhombohedral_axes_keep_their_three_operation_setting(tmp_path: Path) -> None:
    """``R 3`` on rhombohedral axes has 3 symmetry operations, not 9.

    Resolving the space group by name alone answers with the hexagonal ``R 3:H``
    setting whatever the cell says; only the cell angles tell the two settings
    apart. Taking the 9-operation setting would make va three times too small
    and the DPI about 31% too small, with no reason code to show for it.
    """
    path = _metal_structure(
        tmp_path, cell=(60.0, 60.0, 60.0, 100.0, 100.0, 100.0), spacegroup="R 3"
    )
    structure = gemmi.read_structure(path)
    assert structure.spacegroup_hm == "R 3"
    by_name = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    assert len(list(by_name.operations())) == 9, "the name lookup is the hexagonal one"

    volume = dpi_module.asu_volume(_absent_mtz(tmp_path), path)

    assert volume == approx(structure.cell.volume / 3)


def test_the_rhombohedral_setting_reaches_the_published_asu_volume(
    tmp_path: Path,
) -> None:
    """The entry point publishes the same va the fallback resolved."""
    path = _metal_structure(
        tmp_path, cell=(60.0, 60.0, 60.0, 100.0, 100.0, 100.0), spacegroup="R 3"
    )
    context = load_structure("test", path)
    data_json = helpers.write_data_json(
        tmp_path / "data.json", nrefcnt=4096, rffin=0.20
    )

    components = _components(context, path, data_json, _absent_mtz(tmp_path))

    assert components.reason_code == ""
    assert components.asu_volume == approx(context.structure.cell.volume / 3)


def test_the_callers_structure_supplies_the_fallback_cell(tmp_path: Path) -> None:
    """A structure already in hand is used instead of re-parsing ``pdb_path``."""
    coordinates = _metal_structure(tmp_path)
    other = _metal_structure(
        tmp_path, "other.pdb", cell=(50.0, 50.0, 50.0, 90.0, 90.0, 90.0)
    )

    volume = dpi_module.asu_volume(
        _absent_mtz(tmp_path), coordinates, gemmi.read_structure(other)
    )

    assert volume == approx(50.0**3)


def test_a_readable_mtz_with_a_placeholder_cell_falls_back_to_cryst1(
    tmp_path: Path,
) -> None:
    """An MTZ that parses but carries no real cell is not a usable va source.

    Gemmi refuses to write an MTZ with no space group at all, so the placeholder
    cell is the reachable half of that guard.
    """
    pytest.importorskip("numpy")  # only needed to give the MTZ data

    path = _metal_structure(tmp_path)
    mtz_path = helpers.write_mtz(
        tmp_path / "data.mtz",
        {},
        [[0.0, 0.0, 1.0]],
        cell=(1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
        spacegroup="P 1",
    )

    assert dpi_module.asu_volume(mtz_path, path) == approx(100.0**3)


def test_rfree_survives_a_non_utf8_byte_in_the_header(tmp_path: Path) -> None:
    """One stray byte must not cost the whole R-free scrape.

    Deposited headers carry author names in legacy encodings; decoding strictly
    raised ``UnicodeDecodeError`` before the fix.
    """
    path = tmp_path / "latin1.pdb"
    path.write_bytes(
        b"REMARK   3   REFINEMENT BY CAF\xc9 REFINE\n"
        b"REMARK   3   FREE R VALUE                     : 0.21530\n"
    )

    assert dpi_module.rfree_from_pdb(str(path)) == approx(0.2153)


def test_a_non_utf8_header_keeps_every_dpi_term(tmp_path: Path) -> None:
    """The undecodable byte used to blank R-free, nobs and va together.

    The decode error escaped ``rfree_from_pdb`` and was caught as
    ``invalid_dpi_metadata``, which reports no inputs at all.
    """
    path = _metal_structure(tmp_path)
    with open(path, "rb") as handle:
        coordinates = handle.read()
    with open(path, "wb") as handle:
        handle.write(b"REMARK   3   REFINEMENT BY CAF\xc9 REFINE\n")
        handle.write(b"REMARK   3   FREE R VALUE                     : 0.21530\n")
        handle.write(coordinates)
    data_json = helpers.write_data_json(
        tmp_path / "data.json", nrefcnt=4096, rffin=None
    )

    components = _components(
        load_structure("test", path), path, data_json, _absent_mtz(tmp_path)
    )

    assert components.reason_code == ""
    assert components.r_free == approx(0.2153)
    assert components.reflection_count == approx(4096.0)
    assert components.asu_volume == approx(100.0**3)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",  # a top-level array has no ``get``
        '{"properties": null}',  # the key is present but not a block
        '{"properties": [1, 2]}',
        '"properties"',
    ],
)
def test_a_malformed_data_json_reports_missing_terms(
    tmp_path: Path, payload: str
) -> None:
    """A wrongly shaped ``data.json`` leaves the terms missing, not the DPI failed.

    ``dpi_calculation_failed`` is the catch-all for the unanticipated; a file
    Alchemy could read but not understand is anticipated.
    """
    path = _metal_structure(tmp_path)
    data_json = tmp_path / "data.json"
    data_json.write_text(payload, encoding="utf-8")

    assert dpi_module._read_pdb_redo_properties(str(data_json)) == {}  # pyright: ignore[reportPrivateUsage]

    components = _components(
        load_structure("test", path), path, str(data_json), _absent_mtz(tmp_path)
    )

    assert math.isnan(components.dpi)
    assert components.reason_code == ReasonCode.MISSING_OR_INVALID_REFLECTION_COUNT


def test_the_catch_all_reason_code_is_reachable(tmp_path: Path) -> None:
    """An unanticipated failure is reported as ``dpi_calculation_failed``.

    The terms already read are still published, because they are measured
    quantities that the failure did not invalidate.
    """
    path = _metal_structure(tmp_path)
    data_json = helpers.write_data_json(
        tmp_path / "data.json", nrefcnt=4096, rffin=0.20
    )

    def _explode(context: StructureContext) -> float:
        raise RuntimeError("unanticipated")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dpi_module, "count_ni", _explode)
        components = _components(
            load_structure("test", path), path, data_json, _absent_mtz(tmp_path)
        )

    assert math.isnan(components.dpi)
    assert components.reason_code == ReasonCode.DPI_CALCULATION_FAILED
    assert components.r_free == approx(0.20)
    assert components.reflection_count == approx(4096.0)
    assert components.asu_volume == approx(100.0**3)


def test_invalid_occupancy_is_named_by_the_entry_point(tmp_path: Path) -> None:
    """Occupancies unfit for Ni are named ahead of every other term."""
    path = _metal_structure(tmp_path)
    broken = tmp_path / "broken-occupancy.pdb"
    with open(path, encoding="utf-8") as handle:
        lines = [
            line[:54] + "  abcd" + line[60:]
            if line.startswith(("ATOM", "HETATM"))
            else line
            for line in handle
        ]
    broken.write_text("".join(lines), encoding="utf-8")
    context = load_structure("test", str(broken))
    assert context.occupancy.validation_failed is True
    data_json = helpers.write_data_json(
        tmp_path / "data.json", nrefcnt=4096, rffin=0.20
    )

    components = _components(context, str(broken), data_json, _absent_mtz(tmp_path))

    assert math.isnan(components.dpi)
    assert components.reason_code == ReasonCode.INVALID_OCCUPANCY
    # The terms that were readable are still published.
    assert components.r_free == approx(0.20)
    assert components.asu_volume == approx(100.0**3)


def test_the_invalid_term_reason_is_empty_when_every_term_is_usable(
    tmp_path: Path,
) -> None:
    """The reason helper is the single statement of what the formula needs."""
    context = load_structure("test", _metal_structure(tmp_path))

    assert dpi_module._invalid_term_reason(context, 4096.0, 0.20, 1.0e6, 1.0) == ""  # pyright: ignore[reportPrivateUsage]
    assert (
        dpi_module._invalid_term_reason(context, 4096.0, 0.20, 1.0e6, 0.0)  # pyright: ignore[reportPrivateUsage]
        == ReasonCode.INVALID_DPI_ATOM_COUNT
    )
    assert (
        dpi_module._invalid_term_reason(context, 4096.0, 0.20, 1.0e6, math.nan)  # pyright: ignore[reportPrivateUsage]
        == ReasonCode.INVALID_DPI_ATOM_COUNT
    )
