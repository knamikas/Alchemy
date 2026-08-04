"""Scaffolding smoke tests.

These prove two things: the src modules import from a test process, and every
helper in ``tests/helpers.py`` produces input that the real pipeline accepts.
They are deliberately shallow -- behavioural coverage lives in the other test
modules.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import urllib.request

import gemmi
import pytest

import helpers
from helpers import (
    AtomSpec,
    EDSTATS_HEADER,
    StructureBuilder,
    simple_metal_site,
)


# sys.path wiring
def test_src_modules_import():
    """Every src module imports from a test process and exposes its entry points.

    ``src/`` is not a package, so this only works because ``conftest.py`` puts
    it on ``sys.path`` at import time. If that wiring breaks, every other module
    in the suite fails with a confusing ``ModuleNotFoundError``; failing here
    first names the real cause. ``driver`` *is* a package, reached the same way,
    so it is covered here too.
    """
    from bond import bond_analysis
    from bond import bond_schema
    import ccp4_setup
    import codes
    import confidence_score
    from bond import contact_record
    from bond import declared_connections
    import density_analysis
    from bond import donor_chemistry
    from bond import dpi
    import main
    import metal_elements
    import metal_identification  # noqa: F401 - imported to prove it loads
    import cli
    import structure_analysis
    import worker
    from driver import pool, progress, resources, runlog, writers

    assert "ZN" in metal_elements.METAL_ELEMENTS
    assert bond_analysis.CUTOFF == 4.0
    assert callable(structure_analysis.load_structure)
    assert callable(main.main), "src/main.py must keep working as the entry point"
    assert main.main is cli.main, "the entry point must delegate, not reimplement"
    assert callable(pool._run)
    assert callable(worker.process)
    assert writers.MANIFEST_COLUMNS[0] == "pdbID"
    assert callable(progress._ProgressReporter)
    assert callable(runlog._RunLog)
    assert resources.available_cpu_count() >= 1
    assert callable(dpi._calculate_dpi_details)
    assert callable(declared_connections._collect_declared_candidates)
    assert bond_schema.BOND_COLUMNS[0] == "pdbID"
    assert codes.GeometryStatus.SUSPECT == "suspect"
    assert callable(contact_record.Candidate)
    assert set(donor_chemistry.INFERRED_DONOR_ATOMS) == donor_chemistry.AA
    assert callable(density_analysis.run_density_analysis)
    assert callable(confidence_score.score_site)
    # Named explicitly rather than asserted truthy: a non-empty tuple literal is
    # always true, so `assert REQUIRED_CCP4_TOOLS` could never fail.
    assert set(ccp4_setup.REQUIRED_CCP4_TOOLS) == {
        "mtzfix",
        "fft",
        "mapmask",
        "edstats",
    }


def test_src_dir_fixture_points_at_the_modules(src_dir, repo_root, data_dir):
    """The path fixtures address the real checkout, not a copy or a stale root.

    ``data_dir`` in particular has to reach the bundled literature tables, which
    several modules read directly rather than through a fixture.
    """
    assert os.path.isfile(os.path.join(src_dir, "bond", "bond_analysis.py"))
    assert os.path.dirname(src_dir) == repo_root
    assert os.path.isfile(os.path.join(data_dir, "metal_distances_info.txt"))


# The isolation fixtures
def test_analysis_writes_nothing_into_the_current_directory(work_dir, tmp_path_factory):
    """A full in-memory analysis leaves the process working directory empty.

    The suite's release-critical promise is that a test run puts nothing inside
    the checkout, and pytest runs with the checkout as cwd. ``work_dir`` stands
    in for it: every input and output below is addressed absolutely under a
    *different* temporary directory, so any file appearing in the cwd is src
    code writing to a relative path -- which would land in the repository on a
    normal run.
    """
    from bond.bond_analysis import run_bond_analysis
    from structure_analysis import load_structure

    inputs = tmp_path_factory.mktemp("isolated_inputs")
    path = simple_metal_site().write_pdb(inputs / "site.pdb")
    context = load_structure("test", path)
    stats_rows, header, _ = helpers.stats_rows_for_structure(
        context, inputs / "stats.out", metrics={"ZDm": 2.5}
    )
    rows, _, _, _ = run_bond_analysis(
        "test", path, stats_rows, header, helpers.dpi_inputs(), structure=context
    )

    assert rows, "the analysis must have actually run for this to mean anything"
    assert os.path.realpath(os.getcwd()) == os.path.realpath(work_dir)
    assert os.listdir(work_dir) == []


def test_cache_directory_helper_uses_canonical_then_legacy(monkeypatch, tmp_path):
    """The shared helper prefers the canonical nonblank cache variable.

    ``ALCHEMY_TESTS_CACHE`` is the canonical name, matching the other
    ``ALCHEMY_TESTS_*`` switches; ``ALCHEMY_TEST_CACHE`` predates it and stays
    supported. Both the conftest cache fixture and the integration fixture
    resolve through this helper, so the precedence is suite-wide.
    """
    for variable in helpers.CACHE_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    assert helpers.cache_dir_from_env() is None

    legacy = str(tmp_path / "legacy")
    monkeypatch.setenv("ALCHEMY_TEST_CACHE", legacy)
    assert helpers.cache_dir_from_env() == legacy

    canonical = str(tmp_path / "canonical")
    monkeypatch.setenv("ALCHEMY_TESTS_CACHE", canonical)
    assert helpers.cache_dir_from_env() == canonical, "canonical name wins"

    # A blank export from a wrapper script means "unset", not "use the current
    # directory" -- which would scatter downloads through the checkout.
    monkeypatch.setenv("ALCHEMY_TESTS_CACHE", "   ")
    assert helpers.cache_dir_from_env() == legacy
    monkeypatch.delenv("ALCHEMY_TEST_CACHE")
    assert helpers.cache_dir_from_env() is None


def test_pdb_redo_cache_fixture_is_a_writable_directory(pdb_redo_cache):
    """The download cache fixture points somewhere downloads can actually go.

    It is either the configured warm cache -- created if it does not exist yet
    -- or a session tmp directory. A download test handed a missing or
    read-only path would fail with an ``OSError`` halfway through a fetch
    rather than skip or succeed.
    """
    assert os.path.isdir(pdb_redo_cache)
    assert os.access(pdb_redo_cache, os.W_OK | os.X_OK)

    configured = helpers.cache_dir_from_env()
    if configured:
        assert os.path.realpath(pdb_redo_cache) == os.path.realpath(configured)


# Structure helpers -> load_structure
@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_builder_output_loads_cleanly(tmp_path, suffix):
    """A default synthetic structure loads with no complaints in either format.

    Every behavioural test starts from a builder structure, so a clean build
    must produce no occupancy, element or duplicate-record warnings and must
    carry usable symmetry. Otherwise those warnings would leak into unrelated
    assertions and be mistaken for the behaviour under test.
    """
    from structure_analysis import load_structure

    builder = simple_metal_site()
    path = builder.write(tmp_path / f"site{suffix}")
    context = load_structure("test", path)

    assert context.analysis_coordinate_format == (
        "mmcif" if suffix == ".cif" else "pdb"
    )
    # No occupancy, element or duplicate-record complaints from a clean build.
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
def test_simple_metal_site_places_donors_at_requested_distances(tmp_path, suffix):
    """``simple_metal_site`` donor distances survive into measured bond rows.

    The helper's contract is that a donor requested at 2.03 Angstrom really is
    2.03 Angstrom from the metal after the file round-trip, in both formats.
    Distance assertions across the suite are only meaningful if that holds, and
    nothing but the requested donors may fall inside the 4 Angstrom search.
    """
    from bond.bond_analysis import run_bond_analysis
    from structure_analysis import load_structure

    donors = [("HIS", "NE2", 2.03), ("ASP", "OD1", 1.99), ("HOH", "O", 2.09)]
    path = simple_metal_site("ZN", donors).write(tmp_path / f"s{suffix}")
    context = load_structure("test", path)
    rows, candidates, summaries, metadata = run_bond_analysis(
        "test", path, [], list(EDSTATS_HEADER), helpers.dpi_inputs(), structure=context
    )

    assert len(summaries) == 1
    measured = {
        (row["neighbor_resname"], row["neighbor_atom"]): row["distance"] for row in rows
    }
    for resname, atom_name, distance in donors:
        assert measured[(resname, atom_name)] == pytest.approx(distance, abs=1e-6)
    # Nothing else of the schematic skeletons came within the 4 A search radius.
    assert len(rows) == len(donors)
    assert candidates
    assert metadata["partial_reason_codes"] == ["missing_dpi_metadata_source"]


def test_declared_connection_is_reported_for_both_formats(tmp_path):
    """A declared connection is honoured from a PDB LINK and from mmCIF alike.

    LYS NZ at this distance is not a proximity-inferable Zn donor, so the row
    exists only because the connection was declared. The two formats differ in
    what identity they preserve -- a LINK record carries no id and gemmi
    regenerates one from the connection type, while ``_struct_conn.id``
    round-trips -- and both must reach the same bond.
    """
    from bond.bond_analysis import run_bond_analysis
    from structure_analysis import load_structure

    # A legacy LINK record carries no identifier, so gemmi regenerates one from
    # the connection type; mmCIF _struct_conn.id round-trips the given name.
    cases = ((".pdb", "LINK", "metalc1"), (".cif", "struct_conn", "metal1"))
    for suffix, expected_source, expected_id in cases:
        builder = StructureBuilder()
        metal = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
        # LYS NZ is not a proximity-inferable donor at this distance for Zn:
        # the declaration is the only reason it becomes a bond.
        lys = builder.add_amino_acid(
            "LYS", 10, chain="A", positions={"NZ": (2.10, 0.0, 0.0)}
        )
        builder.add_connection(
            metal.ref("ZN"), lys.ref("NZ"), name="metal1", reported_distance=2.10
        )
        path = builder.write(tmp_path / f"declared{suffix}")

        context = load_structure("test", path)
        rows, _, _, metadata = run_bond_analysis(
            "test",
            path,
            [],
            list(EDSTATS_HEADER),
            helpers.dpi_inputs(),
            structure=context,
            connection_path=path,
        )

        assert metadata["messages"] == ["DPI unavailable: missing_dpi_metadata_source"]
        declared = [row for row in rows if row["declared_connection"]]
        assert [row["neighbor_atom"] for row in declared] == ["NZ"]
        assert declared[0]["coordination_source"] == expected_source
        assert declared[0]["connection_id"] == expected_id
        assert float(declared[0]["connection_reported_distance"]) == pytest.approx(2.10)


def test_conformers_drive_the_altloc_selection_policy(tmp_path):
    """Altloc selection picks the highest-occupancy conformer for contact search.

    The selected conformer is the only one offered to contact search, while both
    conformers remain available as source atoms for the occupancy-weighted atom
    count. Conflating the two sets would either double-count atoms in the DPI
    input or measure a bond to a conformer that was not selected.
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
    assert residue.selected_conformer_mean_occupancy == pytest.approx(0.65)
    # Only the selected conformer reaches contact search; both are kept as
    # source records for the occupancy-weighted atom count.
    selected = [a for a in residue.contact_atoms if a.atom_name == "NE2"]
    assert [a.altloc for a in selected] == ["B"]
    assert sorted(a.altloc for a in residue.source_atoms if a.atom_name == "NE2") == [
        "A",
        "B",
    ]


def test_connection_can_name_a_specific_conformer(tmp_path):
    """A declaration naming an unselected conformer is re-pointed and flagged.

    The connection names conformer A, but selection chose B. Alchemy measures
    the contact against B -- so the reported distance is B's -- and records
    ``declared_connection_conformer_substituted`` so the substitution is visible
    rather than silent.
    """
    from bond.bond_analysis import run_bond_analysis
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
    rows, _, _, metadata = run_bond_analysis(
        "test",
        path,
        [],
        list(EDSTATS_HEADER),
        helpers.dpi_inputs(),
        structure=context,
        connection_path=path,
    )

    # The declaration names conformer A, which selection did not choose, so the
    # contact is re-pointed onto the selected conformer B and flagged.
    assert [row["neighbor_altloc"] for row in rows] == ["B"]
    assert rows[0]["distance"] == pytest.approx(2.80, abs=1e-6)
    assert "declared_connection_conformer_substituted" in metadata["warning_codes"]


def test_occupancy_and_altloc_survive_the_pdb_round_trip(tmp_path):
    """Partial occupancies survive the PDB round trip and reach the atom count.

    ``count_deposited_ni`` is an occupancy-weighted sum, so an occupancy that
    was silently rounded to 1.00 on write would quietly inflate the DPI input
    for every partially occupied site.
    """
    from structure_analysis import count_deposited_ni, load_structure

    builder = StructureBuilder()
    builder.add_metal("MG", 1, chain="B", pos=(0.0, 0.0, 0.0), occupancy=0.5)
    builder.add_water(101, (0.0, 2.07, 0.0), chain="B", occupancy=0.25)
    path = builder.write_pdb(tmp_path / "occ.pdb")

    context = load_structure("test", path)
    occupancies = {a.atom_name: a.occupancy for a in context.source_atoms}
    assert occupancies == {"MG": 0.5, "O": 0.25}
    assert count_deposited_ni(context) == pytest.approx(0.75)


def test_hetero_residue_builds_a_multi_metal_cofactor(tmp_path):
    """A het residue keeps its full atom composition and every element present.

    Cofactor handling keys off the residue's element set and its chemical atom
    count, so an FeS cluster must report both elements and all four atoms rather
    than being flattened to its first atom.
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


def test_builder_can_omit_symmetry_metadata(tmp_path):
    """A structure with no cell reports symmetry search as unavailable, with cause.

    Several tests need an entry where symmetry expansion is impossible. The
    builder must be able to express that, and the loader must name the reason
    instead of failing later inside the contact search.
    """
    from structure_analysis import load_structure

    builder = StructureBuilder(cell=None, spacegroup=None)
    builder.add_metal("ZN", 1, chain="B")
    path = builder.write_pdb(tmp_path / "nocell.pdb")

    context = load_structure("test", path)
    assert not context.symmetry_search_available
    assert context.symmetry_search_failure_reason == "missing_or_invalid_unit_cell"


# EDSTATS helpers -> extract_metal_statistics
def test_edstats_row_matches_the_documented_schema():
    """Synthetic EDSTATS rows match the parser's own column list exactly.

    The helper writes the 42-column EDSTATS header the real program emits, and a
    metal row must be addressable by ``RT``/``CI``/``RN``. Two shapes matter
    besides: a blank-chain row is one field short, and a separator row is 39
    fields. An unknown metric name is a typo in the test, so it raises.
    """
    from metal_identification import EDSTATS_COLUMNS

    assert EDSTATS_HEADER == EDSTATS_COLUMNS
    assert len(EDSTATS_HEADER) == 42
    row = helpers.edstats_row("ZN", "B", "1", metrics={"ZDm": 1.5}, nr=3)
    assert len(row) == len(EDSTATS_HEADER)
    fields = dict(zip(EDSTATS_HEADER, row))
    assert (fields["RT"], fields["CI"], fields["RN"]) == ("ZN", "B", "1")
    assert (fields["MN"], fields["CP"], fields["NR"]) == ("1", "B", "3")
    assert fields["ZDm"] == "1.5"

    blank = helpers.edstats_row("ZN", "_", "1", omit_cp=True)
    assert len(blank) == len(EDSTATS_HEADER) - 1
    assert len(helpers.edstats_separator_row()) == 39

    with pytest.raises(KeyError):
        helpers.edstats_row("ZN", "B", "1", metrics={"NOPE": 1})


def test_synthetic_edstats_satisfies_extract_metal_statistics(tmp_path):
    """Helper-written EDSTATS output is accepted by the real statistics parser.

    This is the contract that lets the suite skip CCP4 entirely: whatever
    ``stats_rows_for_structure`` writes must be parseable, joined to the right
    residue, and marked as a matched, selected metal site.
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


def test_edstats_stats_rows_feed_the_bond_sigma_join(tmp_path):
    """Per-residue EDSTATS sigmas reach every bond row for that residue.

    The bond stage joins its rows to the statistics table on residue identity.
    If that join misses, the sigma columns are silently blank rather than wrong,
    which is why they are asserted explicitly here.
    """
    from bond.bond_analysis import run_bond_analysis
    from structure_analysis import load_structure

    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats_rows, header, _ = helpers.stats_rows_for_structure(
        context, tmp_path / "stats.out", metrics={"ZDm": 3.0, "ZD-m": -1.0, "ZD+m": 4.0}
    )

    rows, _, _, _ = run_bond_analysis(
        "test", path, stats_rows, header, helpers.dpi_inputs(), structure=context
    )

    assert rows
    for row in rows:
        assert row["sigma_mag"] == pytest.approx(3.0)
        assert row["sigma_neg"] == pytest.approx(-1.0)
        assert row["sigma_pos"] == pytest.approx(4.0)


def test_blank_chain_rows_round_trip_through_the_parser(tmp_path):
    """A blank chain id survives the whitespace-delimited EDSTATS format.

    EDSTATS writes columns separated by spaces, so a residue in the blank chain
    produces a row with one field fewer. The parser has to attribute the missing
    field to the chain rather than shifting every later column by one.
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


def test_cofactor_rows_repeat_once_per_metal_site(tmp_path):
    """A two-metal cofactor yields one row per metal over one shared observation.

    EDSTATS measures the FES residue once, but the cofactor holds two iron
    sites. Both must be reported, sharing a single density observation id and
    both marked shared, so density is not double-counted as independent
    evidence.
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
    # One density observation shared by two metal sites.
    assert len({row["density_observation_id"] for row in rows}) == 1
    assert all(row["density_shared_site_count"] == 2 for row in rows)
    assert all(row["density_is_shared"] for row in rows)


def test_insertion_codes_survive_into_the_edstats_join(tmp_path):
    """Insertion codes are preserved on the coordinate side of the join.

    ``load_structure`` reports HIS 10A with its insertion code attached, which is
    the identity the statistics table is joined on. Dropping the code would
    merge residue 10 and residue 10A into one.
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


def test_symmetry_image_contacts_are_reachable(tmp_path):
    """A contact that exists only through a symmetry image is found and labelled.

    In this small P 1 cell the water's ``-a`` image lies 1.0 Angstrom from the
    metal while the deposited copy is far away. The contact must be found, its
    distance measured against the image, and it must be marked crystallographic
    with the operation that produced it -- not reported as a deposited contact.
    """
    from bond.bond_analysis import run_bond_analysis
    from structure_analysis import load_structure

    # A small P 1 cell puts the water's -a image 1.0 A from the metal.
    builder = StructureBuilder(
        cell=(20.0, 20.0, 20.0, 90.0, 90.0, 90.0), spacegroup="P 1"
    )
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (19.5, 0.5, 0.5), chain="B")
    path = builder.write_pdb(tmp_path / "sym.pdb")

    context = load_structure("test", path)
    rows, _, _, _ = run_bond_analysis(
        "test", path, [], list(EDSTATS_HEADER), helpers.dpi_inputs(), structure=context
    )

    assert len(rows) == 1
    assert rows[0]["distance"] == pytest.approx(1.0, abs=1e-6)
    assert rows[0]["contact_scope"] == "crystallographic"
    assert rows[0]["crystallographic_contact"]
    assert rows[0]["symmetry_operation"] == "1_455"


def test_dpi_inputs_produce_a_finite_dpi_when_metadata_is_present(tmp_path):
    """Complete refinement metadata yields a finite DPI and a finite z-score.

    Most tests deliberately run without a data.json and expect the
    ``missing_dpi_metadata_source`` partial reason. This is the positive
    control: with reflection count, R-free and resolution supplied, the DPI
    chain must produce real numbers, so a blank z-score elsewhere means missing
    metadata rather than a broken formula.
    """
    from bond.bond_analysis import run_bond_analysis
    from structure_analysis import load_structure

    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    data_json = helpers.write_data_json(
        tmp_path / "data.json", nrefcnt=50000, rffin=0.2
    )
    context = load_structure("test", path)
    rows, _, _, metadata = run_bond_analysis(
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

    assert metadata["partial_reason_codes"] == []
    assert rows and math.isfinite(rows[0]["dpi"])
    assert rows[0]["resolution"] == pytest.approx(1.5)
    assert math.isfinite(rows[0]["zscore"])


# Capability probes and markers
def test_ccp4_capability_probe_is_boolean_and_does_not_raise():
    """The local CCP4 probe answers with a bool instead of raising.

    ``conftest`` calls it during collection to decide what to skip. A probe that
    raised -- no ``PATH`` or an unreadable config -- would abort collection of
    the entire suite rather than skip a few tests. The live network probe stays
    inside the ``network``-marked test below, so this unmarked smoke test never
    opens a real socket.
    """
    assert isinstance(helpers.ccp4_available(), bool)


def test_not_network_selection_does_not_call_the_network_probe(tmp_path, repo_root):
    """Marker deselection happens before the external capability probe.

    ``-m 'not network'`` is an offline selection boundary, not merely a promise
    to discard the test after collection. The child plugin replaces the probe
    with a function that raises, making a hidden DNS or socket attempt
    impossible while still exercising a real nested pytest collection.
    """
    test_file = tmp_path / "test_marker_selection.py"
    test_file.write_text(
        "import pytest\n\n"
        "@pytest.mark.network\n"
        "def test_live():\n"
        "    raise AssertionError('network test was not deselected')\n\n"
        "def test_local():\n"
        "    pass\n",
        encoding="utf-8",
    )
    plugin_file = tmp_path / "forbid_network_probe.py"
    plugin_file.write_text(
        "import helpers\n\n"
        "def pytest_configure(config):\n"
        "    def forbidden(*args, **kwargs):\n"
        "        raise AssertionError('network probe ran after -m exclusion')\n"
        "    helpers.network_available = forbidden\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    python_path = [str(tmp_path), os.path.join(repo_root, "tests")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            "-p",
            "conftest",
            "-p",
            "forbid_network_probe",
            str(test_file),
            "-m",
            "not network",
            "--no-ccp4",
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    assert "test_local" in completed.stdout
    assert "test_live" not in completed.stdout


@pytest.mark.ccp4
def test_ccp4_env_fixture_resolves_every_required_tool(ccp4_env):
    """The ``ccp4_env`` fixture hands back an environment CCP4 can actually run in.

    Each of the four programs is executed with no input, which proves the
    resolved ``PATH`` points at real, runnable binaries rather than at names
    that merely resolve. A CCP4 install that is present but broken -- missing
    shared libraries, a stale setup script -- is caught here instead of halfway
    through an end-to-end run: the dynamic loader fails before ``main`` runs, so
    the program never announces itself.
    """
    resolved = {tool: helpers.which(tool, ccp4_env) for tool in helpers.CCP4_TOOLS}
    assert all(resolved.values()), resolved

    # What the loader says when a shared object is missing or a symbol is
    # unresolved. Deliberately not "no such file or directory": fft and mapmask
    # print that themselves when given no input file, which is healthy.
    loader_failures = (
        "error while loading shared libraries",
        "cannot open shared object file",
        "symbol lookup error",
        "undefined symbol",
    )

    for tool, path in resolved.items():
        assert path is not None
        # No input and no arguments: every one of the four prints a banner
        # naming itself and exits promptly. The exit status differs between
        # them (edstats 0, the other three 1) and is not the point.
        completed = subprocess.run(
            [path],
            env=ccp4_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        output = completed.stdout.decode("utf-8", "replace")
        excerpt = output[:500]

        # Negative means killed by a signal: a SIGSEGV on startup is exactly
        # the "present but broken" install this test exists to catch.
        assert completed.returncode >= 0, (
            f"{tool} died on signal {-completed.returncode}: {excerpt!r}"
        )
        for marker in loader_failures:
            assert marker not in output.lower(), f"{tool}: {excerpt!r}"
        # mtzfix prints "MTZFIX", fft prints "FFTBIG", and mapmask and edstats
        # print their own names. Reaching the banner means the binary loaded
        # and its own code ran.
        assert tool in output.lower(), (
            f"{tool} produced no banner naming itself: {excerpt!r}"
        )


@pytest.mark.network
def test_network_marker_only_runs_with_connectivity():
    """PDB-REDO really answers, not merely a TCP handshake on port 443.

    The ``network`` marker gates tests that download entries, so the useful
    question is whether an entry file can actually be fetched. A captive portal
    or a proxy that accepts connections and then refuses them would pass the
    socket probe used for skipping but fail here.
    """
    assert helpers.network_available()

    url = "https://pdb-redo.eu/db/9myr/data.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        assert response.status == 200
        payload = json.loads(response.read().decode("utf-8"))
    # A JSON-formatted proxy error is still a non-empty dict, so require the
    # crystallographic metadata Alchemy actually consumes from data.json.
    assert isinstance(payload, dict)
    properties = payload.get("properties")
    assert isinstance(properties, dict) and properties
    assert float(properties["NREFCNT"]) > 0
    assert float(properties["DATARESH"]) > 0


def test_element_inference_rejects_ambiguous_names():
    """Atom-name element inference follows PDB naming and refuses metal names.

    The helpers infer an element from a protein atom name, where the leading
    character is the element. That rule does not hold for two-letter metals --
    ``ZN`` is zinc, not nitrogen -- so those must raise rather than be guessed.
    """
    assert helpers.element_for_atom_name("OD1") == "O"
    assert helpers.element_for_atom_name("ND1") == "N"
    assert helpers.element_for_atom_name("SG") == "S"
    assert helpers.element_for_atom_name("CB") == "C"
    with pytest.raises(ValueError):
        helpers.element_for_atom_name("ZN")


def test_gemmi_is_the_expected_flavour():
    # The helpers rely on Connection/AtomAddress and mmCIF writing.
    """The installed gemmi has the APIs the helpers and src depend on.

    ``Connection``/``AtomAddress`` are used to declare connections, and
    ``src/structure_analysis.py`` uses ``Model.num``, which arrived in gemmi
    0.7. An older build fails deep inside unrelated tests, so it is named here.
    """
    assert hasattr(gemmi, "Connection") and hasattr(gemmi, "AtomAddress")
    # Read via getattr: gemmi ships py.typed but its stubs omit __version__,
    # so a direct access is a type-checker error against a real runtime name.
    version = str(getattr(gemmi, "__version__", "unknown"))
    assert tuple(int(p) for p in version.split(".")[:2]) >= (0, 7)
