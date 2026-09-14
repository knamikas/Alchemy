"""Check the synthetic builders and EDSTATS/DPI fixtures in ``helpers``."""

from __future__ import annotations

import math
from pathlib import Path

import helpers
import pytest
from helpers import (
    EDSTATS_HEADER,
    AtomSpec,
    StructureBuilder,
    approx,
    simple_metal_site,
)


@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_builder_output_loads_cleanly(tmp_path: Path, suffix: str) -> None:
    """A default synthetic structure loads with no complaints in either format.

    Every behavioural test starts from a builder structure, so a warning raised
    here would leak into unrelated assertions elsewhere.
    """
    from structure_analysis import load_structure

    builder = simple_metal_site()
    path = builder.write(tmp_path / f"site{suffix}")
    context = load_structure("test", path)

    assert context.analysis_coordinate_format == (
        "mmcif" if suffix == ".cif" else "pdb"
    )
    assert context.warning_codes == ()
    assert not context.occupancy_validation_failed
    assert context.unknown_element_atom_count == 0
    assert context.symmetry_search_available
    assert context.crystallographic_operation_count == 4

    metals = context.metal_atoms(["ZN"])
    assert [atom.element for atom in metals] == ["ZN"]
    assert metals[0].occupancy_valid and metals[0].occupancy == 1.0
    names = {residue.residue_name for residue in context.residues}
    assert names == {"ZN", "HIS", "ASP", "HOH"}


@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_simple_metal_site_places_donors_at_requested_distances(
    tmp_path: Path, suffix: str
) -> None:
    """Verify synthetic donor distances survive coordinate writing and analysis."""
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    donors = [("HIS", "NE2", 2.03), ("ASP", "OD1", 1.99), ("HOH", "O", 2.09)]
    path = simple_metal_site("ZN", donors).write(tmp_path / f"s{suffix}")
    context = load_structure("test", path)
    analysis = run_bond_analysis(
        "test", path, [], list(EDSTATS_HEADER), helpers.dpi_inputs(), structure=context
    )
    rows = analysis.bond_rows
    candidates = analysis.candidate_rows
    summaries = analysis.site_summaries
    metadata = analysis.metadata

    assert len(summaries) == 1
    measured = {
        (row["neighbor_resname"], row["neighbor_atom"]): row["distance"] for row in rows
    }
    for resname, atom_name, distance in donors:
        assert measured[(resname, atom_name)] == approx(distance, abs=1e-6)
    assert len(rows) == len(donors)
    assert candidates
    assert metadata.partial_reason_codes == ["missing_dpi_metadata_source"]


# A legacy LINK record carries no identifier, so gemmi regenerates one from the
# connection type; mmCIF _struct_conn.id round-trips the given name.
@pytest.mark.parametrize(
    "suffix,expected_source,expected_id",
    [(".pdb", "LINK", "metalc1"), (".cif", "struct_conn", "metal1")],
)
def test_declared_connection_is_reported_for_both_formats(
    tmp_path: Path, suffix: str, expected_source: str, expected_id: str
) -> None:
    """A declared connection is honoured from a PDB LINK and from mmCIF alike.

    LYS NZ at this distance is not a proximity-inferable Zn donor, so the row
    exists only because the connection was declared.
    """
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    builder = StructureBuilder()
    metal = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    lys = builder.add_amino_acid(
        "LYS", 10, chain="A", positions={"NZ": (2.10, 0.0, 0.0)}
    )
    builder.add_connection(
        metal.ref("ZN"), lys.ref("NZ"), name="metal1", reported_distance=2.10
    )
    path = builder.write(tmp_path / f"declared{suffix}")

    context = load_structure("test", path)
    analysis = run_bond_analysis(
        "test",
        path,
        [],
        list(EDSTATS_HEADER),
        helpers.dpi_inputs(),
        structure=context,
        connection_path=path,
    )
    rows = analysis.bond_rows
    metadata = analysis.metadata

    assert metadata.messages == ["DPI unavailable: missing_dpi_metadata_source"]
    declared = [row for row in rows if row["declared_connection"]]
    assert [row["neighbor_atom"] for row in declared] == ["NZ"]
    assert declared[0]["coordination_source"] == expected_source
    assert declared[0]["connection_id"] == expected_id
    assert float(declared[0]["connection_reported_distance"]) == approx(2.10)


def test_conformers_drive_the_altloc_selection_policy(tmp_path: Path) -> None:
    """Altloc selection picks the highest-occupancy conformer for contact search.

    Only the selected conformer reaches contact search, while both remain source
    atoms for the occupancy-weighted count; conflating the two sets would either
    double-count atoms in the DPI input or measure an unselected conformer.
    """
    from structure_analysis import load_structure

    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = next(r for r in builder.residues if r.name == "HIS")
    builder.add_conformers(
        his,
        [
            ("A", 0.35, {"NE2": (2.03, 0.0, 0.0)}),
            ("B", 0.65, {"NE2": (2.80, 0.0, 0.0)}),
        ],
        atom_names=["NE2"],
    )
    path = builder.write_pdb(tmp_path / "alt.pdb")

    context = load_structure("test", path)
    residue = next(r for r in context.residues if r.residue_name == "HIS")
    assert residue.alternative_conformers_present
    assert residue.selected_altloc == "B"
    assert residue.selected_conformer_mean_occupancy == approx(0.65)
    selected = [a for a in residue.contact_atoms if a.atom_name == "NE2"]
    assert [a.altloc for a in selected] == ["B"]
    assert sorted(a.altloc for a in residue.source_atoms if a.atom_name == "NE2") == [
        "A",
        "B",
    ]


def test_connection_can_name_a_specific_conformer(tmp_path: Path) -> None:
    """A declaration naming an unselected conformer is re-pointed and flagged.

    The connection names conformer A, but selection chose B, so the contact is
    measured against B and the substitution is recorded rather than silent.
    """
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    metal = next(r for r in builder.residues if r.name == "ZN")
    his = next(r for r in builder.residues if r.name == "HIS")
    builder.add_conformers(
        his,
        [
            ("A", 0.35, {"NE2": (2.03, 0.0, 0.0)}),
            ("B", 0.65, {"NE2": (2.80, 0.0, 0.0)}),
        ],
        atom_names=["NE2"],
    )
    builder.add_connection(metal.ref("ZN"), his.ref("NE2", "A"))
    path = builder.write_cif(tmp_path / "altdecl.cif")

    context = load_structure("test", path)
    analysis = run_bond_analysis(
        "test",
        path,
        [],
        list(EDSTATS_HEADER),
        helpers.dpi_inputs(),
        structure=context,
        connection_path=path,
    )
    rows = analysis.bond_rows
    metadata = analysis.metadata

    assert [row["neighbor_altloc"] for row in rows] == ["B"]
    assert rows[0]["distance"] == approx(2.80, abs=1e-6)
    assert "declared_connection_conformer_substituted" in metadata.warning_codes


def test_occupancy_and_altloc_survive_the_pdb_round_trip(tmp_path: Path) -> None:
    """Partial occupancies survive the PDB round trip and reach the atom count.

    ``count_deposited_ni`` is an occupancy-weighted sum, so an occupancy rounded
    to 1.00 on write would inflate the DPI input for every partial site.
    """
    from structure_analysis import count_deposited_ni, load_structure

    builder = StructureBuilder()
    builder.add_metal("MG", 1, chain="B", pos=(0.0, 0.0, 0.0), occupancy=0.5)
    builder.add_water(101, (0.0, 2.07, 0.0), chain="B", occupancy=0.25)
    path = builder.write_pdb(tmp_path / "occ.pdb")

    context = load_structure("test", path)
    occupancies = {a.atom_name: a.occupancy for a in context.source_atoms}
    assert occupancies == {"MG": 0.5, "O": 0.25}
    assert count_deposited_ni(context) == approx(0.75)


def test_hetero_residue_builds_a_multi_metal_cofactor(tmp_path: Path) -> None:
    """A het residue keeps its full atom composition and every element present.

    Cofactor handling keys off the residue's element set and chemical atom
    count, not its first atom.
    """
    from structure_analysis import load_structure

    cluster = [
        AtomSpec("FE1", "FE", (0.0, 0.0, 0.0)),
        AtomSpec("FE2", "FE", (2.7, 0.0, 0.0)),
        AtomSpec("S1", "S", (1.35, 1.8, 0.0)),
        AtomSpec("S2", "S", (1.35, -1.8, 0.0)),
    ]
    builder = StructureBuilder()
    builder.add_hetero_residue("FES", 1, cluster, chain="B")
    path = builder.write_cif(tmp_path / "fes.cif")

    context = load_structure("test", path)
    residue = context.residues[0]
    assert residue.residue_name == "FES"
    assert residue.chemical_atom_site_count == 4
    assert residue.elements == frozenset({"FE", "S"})


def test_builder_can_omit_symmetry_metadata(tmp_path: Path) -> None:
    """A structure with no cell reports symmetry search as unavailable, with cause.

    The loader must name the reason rather than fail later inside the contact
    search.
    """
    from structure_analysis import load_structure

    builder = StructureBuilder(cell=None, spacegroup=None)
    builder.add_metal("ZN", 1, chain="B")
    path = builder.write_pdb(tmp_path / "nocell.pdb")

    context = load_structure("test", path)
    assert not context.symmetry_search_available
    assert context.symmetry_search_failure_reason == "missing_or_invalid_unit_cell"


def test_edstats_row_matches_the_documented_schema() -> None:
    """Synthetic EDSTATS rows match the parser's own column list exactly.

    EDSTATS writes 42 columns for a residue row, 41 for a blank-chain row and
    39 for a separator row, and the helper must reproduce all three.
    """
    from metal_identification import EDSTATS_COLUMNS

    assert EDSTATS_HEADER == EDSTATS_COLUMNS
    assert len(EDSTATS_HEADER) == 42
    row = helpers.edstats_row("ZN", "B", "1", metrics={"ZDm": 1.5}, nr=3)
    assert len(row) == len(EDSTATS_HEADER)
    fields = dict(zip(EDSTATS_HEADER, row, strict=False))
    assert (fields["RT"], fields["CI"], fields["RN"]) == ("ZN", "B", "1")
    assert (fields["MN"], fields["CP"], fields["NR"]) == ("1", "B", "3")
    assert fields["ZDm"] == "1.5"

    blank = helpers.edstats_row("ZN", "_", "1", omit_cp=True)
    assert len(blank) == len(EDSTATS_HEADER) - 1
    assert len(helpers.edstats_separator_row()) == 39

    with pytest.raises(KeyError):
        helpers.edstats_row("ZN", "B", "1", metrics={"NOPE": 1})


def test_synthetic_edstats_satisfies_extract_metal_statistics(tmp_path: Path) -> None:
    """Helper-written EDSTATS output is accepted by the real statistics parser.

    This is the contract that lets the suite run without CCP4 at all.
    """
    from structure_analysis import load_structure

    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    rows, header, stats_path = helpers.stats_rows_for_structure(
        context, tmp_path / "stats.out", metrics={"ZDm": 2.5, "ZD-m": -2.5}
    )

    assert header == list(EDSTATS_HEADER)
    assert [(row["category"], row["resname"]) for row in rows] == [("metal", "ZN")]
    row = rows[0]
    assert row["chain"] == "B" and row["resnum"] == "1"
    assert row["selected_metal_site_status"] == "selected"
    assert row["coordinate_mapping_status"] == "matched"
    assert row["fields"][header.index("ZDm")] == "2.5"
    assert stats_path.endswith("stats.out")


def test_edstats_stats_rows_feed_the_bond_sigma_join(tmp_path: Path) -> None:
    """Per-residue EDSTATS sigmas reach every bond row for that residue.

    A missed join leaves the sigma columns blank rather than wrong, so it is
    invisible unless asserted.
    """
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats_rows, header, _ = helpers.stats_rows_for_structure(
        context, tmp_path / "stats.out", metrics={"ZDm": 3.0, "ZD-m": -1.0, "ZD+m": 4.0}
    )

    analysis = run_bond_analysis(
        "test", path, stats_rows, header, helpers.dpi_inputs(), structure=context
    )
    rows = analysis.bond_rows

    assert rows
    for row in rows:
        assert row["sigma_mag"] == approx(3.0)
        assert row["sigma_neg"] == approx(-1.0)
        assert row["sigma_pos"] == approx(4.0)


def test_blank_chain_rows_round_trip_through_the_parser(tmp_path: Path) -> None:
    """A blank chain id survives the whitespace-delimited EDSTATS format.

    EDSTATS separates columns by spaces, so a blank-chain residue produces a row
    with one field fewer; the parser has to attribute the missing field to the
    chain rather than shift every later column by one.
    """
    from metal_elements import METAL_ELEMENTS
    from metal_identification import extract_metal_statistics
    from structure_analysis import load_structure

    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="")
    builder.add_water(101, (0.0, 2.09, 0.0), chain="")
    path = builder.write_pdb(tmp_path / "blank.pdb")

    context = load_structure("test", path)
    assert {r.chain_id for r in context.residues} == {""}
    stats_path = helpers.write_edstats_for_structure(
        tmp_path / "stats.out", context, blank_chain_form=True
    )
    text = open(stats_path, encoding="utf-8").read().splitlines()
    assert [len(line.split()) for line in text[1:]] == [41, 41]

    rows, _ = extract_metal_statistics(
        "test", stats_path, set(METAL_ELEMENTS), set(), structure=context
    )
    assert [row["resname"] for row in rows] == ["ZN"]
    assert rows[0]["chain"] == ""


def test_cofactor_rows_repeat_once_per_metal_site(tmp_path: Path) -> None:
    """A two-metal cofactor yields one row per metal over one shared observation.

    EDSTATS measures the FES residue once but the cofactor holds two iron sites,
    so both rows share one density observation id and are marked shared, keeping
    the density from counting twice as independent evidence.
    """
    from metal_elements import METAL_ELEMENTS
    from metal_identification import extract_metal_statistics
    from structure_analysis import load_structure

    builder = StructureBuilder()
    builder.add_hetero_residue(
        "FES",
        1,
        [
            AtomSpec("FE1", "FE", (0.0, 0.0, 0.0)),
            AtomSpec("FE2", "FE", (2.7, 0.0, 0.0)),
            AtomSpec("S1", "S", (1.35, 1.8, 0.0)),
            AtomSpec("S2", "S", (1.35, -1.8, 0.0)),
        ],
        chain="B",
    )
    path = builder.write_cif(tmp_path / "fes.cif")

    context = load_structure("test", path)
    stats_path = helpers.write_edstats_for_structure(tmp_path / "stats.out", context)
    rows, _ = extract_metal_statistics(
        "test", stats_path, set(METAL_ELEMENTS), {"FES"}, structure=context
    )

    assert [row["category"] for row in rows] == ["cofactor", "cofactor"]
    assert [row["site"].atom_name for row in rows] == ["FE1", "FE2"]
    assert len({row["density_observation_id"] for row in rows}) == 1
    assert all(row["density_shared_site_count"] == 2 for row in rows)
    assert all(row["density_is_shared"] for row in rows)


def test_insertion_codes_survive_into_the_edstats_join(tmp_path: Path) -> None:
    """Insertion codes are preserved on the coordinate side of the join.

    The insertion code is part of the identity the statistics table joins on:
    dropping it would merge residue 10 and residue 10A into one.
    """
    from structure_analysis import load_structure

    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    builder.add_amino_acid(
        "HIS", 10, chain="A", icode="A", positions={"NE2": (2.03, 0.0, 0.0)}
    )
    path = builder.write_pdb(tmp_path / "icode.pdb")

    context = load_structure("test", path)
    resnums = {r.residue_name: r.resnum for r in context.residues}
    assert resnums == {"ZN": "1", "HIS": "10A"}
    rows, _, _ = helpers.stats_rows_for_structure(context, tmp_path / "stats.out")
    assert [row["resnum"] for row in rows] == ["1"]


def test_symmetry_image_contacts_are_reachable(tmp_path: Path) -> None:
    """A contact that exists only through a symmetry image is found and labelled.

    The deposited water is far from the metal, so the distance must be measured
    against the image and marked crystallographic, not deposited.
    """
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    # This small P 1 cell puts the water's -a image 1.0 A from the metal.
    builder = StructureBuilder(
        cell=(20.0, 20.0, 20.0, 90.0, 90.0, 90.0), spacegroup="P 1"
    )
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (19.5, 0.5, 0.5), chain="B")
    path = builder.write_pdb(tmp_path / "sym.pdb")

    context = load_structure("test", path)
    analysis = run_bond_analysis(
        "test", path, [], list(EDSTATS_HEADER), helpers.dpi_inputs(), structure=context
    )
    rows = analysis.bond_rows

    assert len(rows) == 1
    assert rows[0]["distance"] == approx(1.0, abs=1e-6)
    assert rows[0]["contact_scope"] == "crystallographic"
    assert rows[0]["crystallographic_contact"]
    assert rows[0]["symmetry_operation"] == "1_455"


def test_dpi_inputs_produce_a_finite_dpi_when_metadata_is_present(
    tmp_path: Path,
) -> None:
    """Complete refinement metadata yields a finite DPI and a finite z-score.

    The positive control for the many tests that run without a data.json: a
    blank z-score elsewhere means missing metadata, not a broken formula.
    """
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    data_json = helpers.write_data_json(
        tmp_path / "data.json", nrefcnt=50000, rffin=0.2
    )
    context = load_structure("test", path)
    analysis = run_bond_analysis(
        "test",
        path,
        [],
        list(EDSTATS_HEADER),
        helpers.dpi_inputs(
            pdb_path=path,
            mtz_path=str(tmp_path / "missing.mtz"),
            data_json=data_json,
            resolution=1.5,
        ),
        structure=context,
    )
    rows = analysis.bond_rows
    summary = next(iter(analysis.site_summaries.values()))
    metadata = analysis.metadata

    assert metadata.partial_reason_codes == []
    assert rows and math.isfinite(rows[0]["dpi"])
    assert rows[0]["resolution"] == approx(1.5)
    assert math.isfinite(rows[0]["zscore"])
    assert summary["r_free"] == approx(0.2)
    assert summary["reflection_count"] == approx(50000)
    assert math.isfinite(summary["asu_volume"])


def test_metal_site_output_retains_cartesian_coordinates(tmp_path: Path) -> None:
    """Metal and transformed-neighbor coordinates share a reconstructible frame."""
    from coordination.schema import stats_extra_values
    from structure_analysis import load_structure

    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(1.23456789, -2.34567891, 3.45678912))
    path = builder.write_cif(tmp_path / "coordinates.cif")
    context = load_structure("test", path)
    metal = context.metal_atoms(["ZN"])[0]

    values = stats_extra_values("test", context, metal)

    assert values["metal_coordinates_valid"] is True
    assert values["metal_x"] == approx(1.234568)
    assert values["metal_y"] == approx(-2.345679)
    assert values["metal_z"] == approx(3.456789)


def test_element_inference_rejects_ambiguous_names() -> None:
    """Atom-name element inference follows PDB naming and refuses metal names.

    The leading character is the element for protein atom names, but not for
    two-letter metals: ``ZN`` is zinc, not nitrogen.
    """
    assert helpers.element_for_atom_name("OD1") == "O"
    assert helpers.element_for_atom_name("ND1") == "N"
    assert helpers.element_for_atom_name("SG") == "S"
    assert helpers.element_for_atom_name("CB") == "C"
    with pytest.raises(ValueError):
        helpers.element_for_atom_name("ZN")
