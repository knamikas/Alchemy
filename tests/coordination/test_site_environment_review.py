"""Direct unit tests for the site-environment summaries of a metal site.

Every fixture here is synthetic: the proximity and median-B helpers are called
with hand-built ``AtomSite`` records, and the special-position helper with small
crystal structures written to ``tmp_path``. Neither CCP4 nor the network is used.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gemmi
import pytest
from helpers import AtomSpec, StructureBuilder, approx, atom_site

import structure_analysis as sa
from codes import ParentType
from coordination import site_environment
from coordination.policy import NEARBY_METAL_RADIUS, SEARCH_EPSILON
from coordination.schema import metal_site_identifier
from coordination.site_environment import (
    MetalProximity,
    MetalSpecialPosition,
    entry_nonwater_median_b_iso,
    metal_proximity_summaries,
    metal_special_position_summaries,
    parent_type,
)
from metal_elements import METAL_ELEMENTS
from structure_analysis import NAN, AtomSite, StructureContext, position_distance

PDB_ID = "1abc"

AtomKey = tuple[int, int, int, int]


def _metal(
    index: int,
    pos: Sequence[float],
    element: str = "ZN",
    residue_name: str | None = None,
    **overrides: Any,
) -> AtomSite:
    """One metal atom record whose residue index makes its source key unique."""
    return atom_site(
        element,
        atom_name=element,
        residue_name=element if residue_name is None else residue_name,
        pos=pos,
        residue_index=index,
        **overrides,
    )


def _atom(element: str, b_iso: float, **overrides: Any) -> AtomSite:
    """One atom record carrying an explicit B factor."""
    atom = atom_site(element, atom_name=element, **overrides)
    atom.gemmi_atom.b_iso = b_iso
    return atom


def _load(builder: StructureBuilder, tmp_path: Path, name: str) -> StructureContext:
    """Write ``builder`` as mmCIF and load it as a production structure."""
    return sa.load_structure(PDB_ID, str(builder.write_cif(tmp_path / f"{name}.cif")))


def _reference_proximities(
    pdb_id: str, metals: Sequence[AtomSite]
) -> dict[AtomKey, MetalProximity]:
    """Brute-force proximities: sort every pair, then read the first element."""
    summaries: dict[AtomKey, MetalProximity] = {
        metal.source_key: MetalProximity.unavailable() for metal in metals
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
        count_within_6a = sum(
            distance <= NEARBY_METAL_RADIUS + SEARCH_EPSILON
            for distance, _, _ in neighbors
        )
        if not neighbors:
            summaries[metal.source_key] = MetalProximity(NAN, "", "", count_within_6a)
            continue
        distance, _, nearest = neighbors[0]
        summaries[metal.source_key] = MetalProximity(
            nearest_distance=round(distance, 3),
            nearest_element=nearest.element,
            nearest_site_id=metal_site_identifier(pdb_id, nearest),
            count_within_6a=count_within_6a,
        )
    return summaries


def _raise_spacegroup_lookup(_structure: object) -> object:
    """Stand in for a symmetry lookup that fails while the entry is analyzed."""
    raise RuntimeError("spacegroup lookup exploded")


class _ExplodingCell:
    """A unit cell whose special-position search always fails."""

    def is_special_position(self, _pos: object, _cutoff: float) -> int:
        raise ValueError("special position search exploded")


class _ExplodingStructure:
    """A Gemmi structure stand-in that only fails inside the per-metal loop."""

    def __init__(self) -> None:
        self._cell = _ExplodingCell()
        self.spacegroup_hm = ""

    @property
    def cell(self) -> _ExplodingCell:
        return self._cell

    @cell.setter
    def cell(self, _value: object) -> None:
        """Ignore the crystallographic cell the caller builds."""

    def setup_cell_images(self) -> None:
        """Accept the image setup the caller performs before searching."""


def test_parent_type_classifies_each_component_class(tmp_path: Path) -> None:
    """A cluster, a heme, a lone ion, and any other component are distinguished."""
    builder = StructureBuilder()
    builder.add_hetero_residue(
        "SF4",
        1,
        [
            AtomSpec(name="FE1", element="FE", pos=(0.0, 0.0, 0.0)),
            AtomSpec(name="S1", element="S", pos=(2.2, 0.0, 0.0)),
        ],
    )
    builder.add_hetero_residue(
        "HEM", 2, [AtomSpec(name="FE", element="FE", pos=(20.0, 0.0, 0.0))]
    )
    builder.add_metal("ZN", 3, pos=(40.0, 0.0, 0.0))
    builder.add_hetero_residue(
        "XYZ",
        4,
        [
            AtomSpec(name="MG", element="MG", pos=(60.0, 0.0, 0.0)),
            AtomSpec(name="O1", element="O", pos=(62.3, 0.0, 0.0)),
        ],
    )
    context = _load(builder, tmp_path, "components")

    types = {
        metal.residue_name: parent_type(context, metal)
        for metal in context.metal_atoms(METAL_ELEMENTS)
    }
    assert types == {
        "SF4": ParentType.CLUSTER,
        "HEM": ParentType.HEME,
        "ZN": ParentType.ION,
        "XYZ": ParentType.OTHER,
    }


@pytest.mark.parametrize(
    ("residue_name", "expected"),
    [
        ("sf4", ParentType.CLUSTER),
        (" SF4 ", ParentType.CLUSTER),
        ("hem", ParentType.HEME),
        (" Hem", ParentType.HEME),
    ],
)
def test_parent_type_normalizes_the_residue_name(
    tmp_path: Path, residue_name: str, expected: ParentType
) -> None:
    """Catalogued identifiers match however the deposited name was spelled."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1)
    context = _load(builder, tmp_path, "case")
    metal = _metal(0, (0.0, 0.0, 0.0), element="FE", residue_name=residue_name)

    assert parent_type(context, metal) is expected


def test_non_finite_metal_reports_unavailable_proximity() -> None:
    """An unsearchable metal keeps the schema with every proximity field blank."""
    broken = _metal(0, (float("nan"), 0.0, 0.0))
    partner = _metal(1, (3.0, 0.0, 0.0), element="FE")

    summaries = metal_proximity_summaries(PDB_ID, [broken, partner])

    proximity = summaries[broken.source_key]
    assert math.isnan(proximity.nearest_distance)
    assert math.isnan(float(proximity.count_within_6a))
    assert proximity.nearest_element == ""
    assert proximity.nearest_site_id == ""
    # The searchable metal never sees the unsearchable one.
    assert summaries[partner.source_key].nearest_element == ""
    assert summaries[partner.source_key].count_within_6a == 0


def test_lone_metal_reports_a_measured_zero_count() -> None:
    """A single metal has no neighbor, which is a count of zero, not a failure."""
    metal = _metal(0, (1.0, 2.0, 3.0))

    proximity = metal_proximity_summaries(PDB_ID, [metal])[metal.source_key]

    assert math.isnan(proximity.nearest_distance)
    assert proximity.nearest_element == ""
    assert proximity.nearest_site_id == ""
    assert proximity.count_within_6a == 0
    assert not math.isnan(float(proximity.count_within_6a))


def test_equidistant_neighbors_break_the_tie_by_source_key() -> None:
    """Two neighbors at the same distance resolve to the lowest source key."""
    metal = _metal(5, (0.0, 0.0, 0.0))
    low = _metal(1, (2.5, 0.0, 0.0), element="FE")
    high = _metal(9, (-2.5, 0.0, 0.0), element="MG")

    for order in ([metal, high, low], [metal, low, high]):
        proximity = metal_proximity_summaries(PDB_ID, order)[metal.source_key]
        assert proximity.nearest_element == "FE"
        assert proximity.nearest_site_id == metal_site_identifier(PDB_ID, low)
        assert proximity.nearest_distance == approx(2.5)
        assert proximity.count_within_6a == 2


def test_proximity_matches_a_brute_force_reference() -> None:
    """The single-pass search reproduces a fully sorted reference, ties included."""
    rng = random.Random(20260917)
    # A coarse grid makes exact distance ties common instead of vanishingly rare.
    metals = [
        _metal(
            index,
            [float(rng.randrange(-4, 5)) * 1.5 for _ in range(3)],
            element=rng.choice(("ZN", "FE", "MG", "CU")),
        )
        for index in range(40)
    ]

    assert metal_proximity_summaries(PDB_ID, metals) == _reference_proximities(
        PDB_ID, metals
    )


def test_median_b_iso_excludes_absent_and_unusable_atoms(tmp_path: Path) -> None:
    """Water, hydrogen, zero-occupancy, and non-finite B atoms leave the median."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1)
    context = _load(builder, tmp_path, "median")
    context.contact_atoms = (
        _atom("C", 10.0),
        _atom("N", 30.0),
        _atom("O", 90.0, is_water=True),
        _atom("H", 90.0),
        _atom("C", 90.0, occupancy=0.0),
        _atom("C", float("nan")),
    )

    assert entry_nonwater_median_b_iso(context) == approx(20.0)


def test_median_b_iso_without_usable_atoms_is_unavailable(tmp_path: Path) -> None:
    """An entry with nothing left to measure reports no median."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1)
    context = _load(builder, tmp_path, "median_empty")
    context.contact_atoms = ()

    assert math.isnan(entry_nonwater_median_b_iso(context))


def test_impossible_site_symmetry_order_is_reported_unavailable(
    tmp_path: Path,
) -> None:
    """A metal modeled off a four-fold cannot have a site symmetry order of three."""
    builder = StructureBuilder(
        cell=(20.0, 20.0, 30.0, 90.0, 90.0, 90.0), spacegroup="P 4"
    )
    builder.add_metal("ZN", 1, pos=(0.45, 0.0, 5.0), occupancy=0.33)
    builder.add_metal("FE", 2, pos=(0.0, 0.0, 5.0), occupancy=0.25)
    context = _load(builder, tmp_path, "off_axis")
    assert context.symmetry.crystallographic_operation_count == 4

    metals = {metal.element: metal for metal in context.metal_atoms(METAL_ELEMENTS)}
    summaries = metal_special_position_summaries(context, list(metals.values()))

    off_axis = summaries[metals["ZN"].source_key]
    assert off_axis.special_position is True
    assert math.isnan(off_axis.site_symmetry_order)
    assert math.isnan(off_axis.expected_occupancy)
    assert off_axis.occupancy_matches_site_symmetry == ""

    on_axis = summaries[metals["FE"].source_key]
    assert on_axis.special_position is True
    assert on_axis.site_symmetry_order == 4
    assert on_axis.expected_occupancy == approx(0.25)
    assert on_axis.occupancy_matches_site_symmetry is True


def _one_metal_crystal(tmp_path: Path, name: str) -> tuple[StructureContext, AtomSite]:
    """A P 2 crystal holding one metal on the two-fold axis."""
    builder = StructureBuilder(
        cell=(20.0, 20.0, 20.0, 90.0, 90.0, 90.0), spacegroup="P 2"
    )
    builder.add_metal("ZN", 1, pos=(0.0, 0.0, 0.0), occupancy=0.5)
    context = _load(builder, tmp_path, name)
    return context, context.metal_atoms(METAL_ELEMENTS)[0]


def test_special_position_setup_failure_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed cell setup is named, not read as an entry without site symmetry."""
    context, metal = _one_metal_crystal(tmp_path, "setup_failure")
    monkeypatch.setattr(
        site_environment, "spacegroup_or_none", _raise_spacegroup_lookup
    )
    messages: list[str] = []

    summaries = metal_special_position_summaries(context, [metal], messages)

    assert summaries[metal.source_key] == MetalSpecialPosition.unavailable()
    assert messages == [
        "special position evaluation failed: RuntimeError: spacegroup lookup exploded"
    ]


def test_special_position_search_failure_names_the_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metal whose own search crashes is reported with its site identifier."""
    context, metal = _one_metal_crystal(tmp_path, "search_failure")
    monkeypatch.setattr(gemmi, "Structure", _ExplodingStructure)
    messages: list[str] = []

    summaries = metal_special_position_summaries(context, [metal], messages)

    assert summaries[metal.source_key] == MetalSpecialPosition.unavailable()
    assert messages == [
        "special position evaluation failed for "
        f"{metal_site_identifier(PDB_ID, metal)}: "
        "ValueError: special position search exploded"
    ]


def test_special_position_failure_without_messages_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message list stays optional, so existing callers keep working."""
    context, metal = _one_metal_crystal(tmp_path, "silent_failure")
    monkeypatch.setattr(
        site_environment, "spacegroup_or_none", _raise_spacegroup_lookup
    )

    summaries = metal_special_position_summaries(context, [metal])

    assert summaries[metal.source_key] == MetalSpecialPosition.unavailable()
