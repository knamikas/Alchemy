"""Scaffolding smoke tests: src imports, and helper output the pipeline accepts.

Shallow by design; behavioural coverage lives in the other test modules.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Protocol, cast

import gemmi
import pytest

import helpers
from helpers import (
    AtomSpec,
    EDSTATS_HEADER,
    StructureBuilder,
    simple_metal_site,
)


class _ApproxFactory(Protocol):
    """The concrete numeric subset of pytest's broadly typed approx helper."""

    def __call__(
        self,
        expected: object,
        rel: float | None = None,
        abs: float | None = None,
        nan_ok: bool = False,
    ) -> object: ...


class _PytestApi(Protocol):
    approx: _ApproxFactory


approx = cast(_PytestApi, pytest).approx


def test_src_modules_import() -> None:
    """Every src module imports from a test process and exposes its entry points.

    ``src/`` is not a package, so this works only because ``conftest.py`` puts
    it on ``sys.path``; failing here names that cause before the rest of the
    suite fails with ``ModuleNotFoundError``.
    """
    from coordination import analysis
    from coordination import schema
    import ccp4_setup
    import codes
    import confidence_score
    from coordination import contact_record
    from coordination import declared_connections
    import density_analysis
    from coordination import donor_chemistry
    from coordination import dpi
    import main
    import metal_elements
    import metal_identification
    import cli
    import structure_analysis
    import worker
    from driver import pool, progress, resources, runlog, writers

    assert "ZN" in metal_elements.METAL_ELEMENTS
    assert analysis.CANDIDATE_SEARCH_RADIUS == 4.0
    assert callable(structure_analysis.load_structure)
    assert callable(main.main), "src/main.py must keep working as the entry point"
    assert main.main is cli.main, "the entry point must delegate, not reimplement"
    assert callable(pool.run)
    assert callable(worker.process)
    assert writers.MANIFEST_COLUMNS[0] == "pdbID"
    assert callable(progress.ProgressReporter)
    assert callable(runlog.RunLog)
    assert resources.available_cpu_count() >= 1
    assert callable(dpi.calculate_dpi_details)
    assert callable(declared_connections.collect_declared_candidates)
    assert schema.BOND_COLUMNS[0] == "pdbID"
    # Compared as a bare string, not through ``.value``: that these members are
    # ``str`` is what lets every status comparison across src/ and the written
    # CSVs work, so a demotion to a plain Enum has to fail here. mypy reads the
    # two literals as non-overlapping and cannot see the StrEnum base.
    assert codes.GeometryStatus.SUSPECT == "suspect"  # type: ignore[comparison-overlap]
    assert callable(contact_record.Candidate)
    assert set(donor_chemistry.INFERRED_DONOR_ATOMS) == donor_chemistry.AA
    assert callable(density_analysis.run_density_analysis)
    assert callable(metal_identification.extract_metal_statistics)
    assert callable(confidence_score.score_site)
    assert set(ccp4_setup.REQUIRED_CCP4_TOOLS) == {
        "mtzfix",
        "fft",
        "mapmask",
        "edstats",
    }


def test_src_dir_fixture_points_at_the_modules(
    src_dir: str, repo_root: str, data_dir: str
) -> None:
    """The path fixtures address the real checkout, not a copy or a stale root."""
    assert os.path.isfile(os.path.join(src_dir, "coordination", "analysis.py"))
    assert os.path.dirname(src_dir) == repo_root
    assert os.path.isfile(os.path.join(data_dir, "metal_distances_info.txt"))


def test_root_launcher_works_outside_the_repository(
    repo_root: str, tmp_path: Path
) -> None:
    """``./alchemy`` resolves src from itself and delegates to the CLI."""
    launcher = os.path.join(repo_root, "alchemy")
    command = [launcher, "--help"]
    # Windows does not dispatch extensionless shebang files, but invoking the
    # same file through Python still exercises its repository-path resolution.
    if sys.platform == "win32":
        command.insert(0, sys.executable)

    result = subprocess.run(
        command,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Batch Alchemy core pipeline over PDB-REDO." in result.stdout
    assert "--pdb-redo-root" in result.stdout


def test_analysis_writes_nothing_into_the_current_directory(
    work_dir: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A full in-memory analysis leaves the process working directory empty.

    Every path below is absolute and under a different temporary directory, so
    a file appearing in the cwd is src code writing to a relative path, which on
    a normal run would land in the checkout.
    """
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    inputs = tmp_path_factory.mktemp("isolated_inputs")
    path = simple_metal_site().write_pdb(inputs / "site.pdb")
    context = load_structure("test", path)
    stats_rows, header, _ = helpers.stats_rows_for_structure(
        context, inputs / "stats.out", metrics={"ZDm": 2.5}
    )
    analysis = run_bond_analysis(
        "test", path, stats_rows, header, helpers.dpi_inputs(), structure=context
    )
    rows = analysis.bond_rows

    assert rows, "the analysis must have actually run for this to mean anything"
    assert os.path.realpath(os.getcwd()) == os.path.realpath(work_dir)
    assert os.listdir(work_dir) == []


def test_cache_directory_helper_uses_canonical_then_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shared helper prefers the canonical nonblank cache variable.

    Every cache fixture in the suite resolves through this helper, so the
    precedence between ``ALCHEMY_TESTS_CACHE`` and the older
    ``ALCHEMY_TEST_CACHE`` is suite-wide.
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

    # A blank export means "unset", not "use the current directory", which
    # would scatter downloads through the checkout.
    monkeypatch.setenv("ALCHEMY_TESTS_CACHE", "   ")
    assert helpers.cache_dir_from_env() == legacy
    monkeypatch.delenv("ALCHEMY_TEST_CACHE")
    assert helpers.cache_dir_from_env() is None


def test_pdb_redo_cache_fixture_is_a_writable_directory(pdb_redo_cache: str) -> None:
    """The download cache fixture points somewhere downloads can actually go.

    A download test handed a missing or read-only path fails with an ``OSError``
    halfway through a fetch rather than skipping.
    """
    assert os.path.isdir(pdb_redo_cache)
    assert os.access(pdb_redo_cache, os.W_OK | os.X_OK)

    configured = helpers.cache_dir_from_env()
    if configured:
        assert os.path.realpath(pdb_redo_cache) == os.path.realpath(configured)


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
    """``simple_metal_site`` donor distances survive into measured bond rows.

    Distance assertions across the suite are only meaningful if a donor
    requested at 2.03 Angstrom really is 2.03 Angstrom from the metal after the
    file round-trip, and if nothing else falls inside the 4 Angstrom search.
    """
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
    fields = dict(zip(EDSTATS_HEADER, row))
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
    metadata = analysis.metadata

    assert metadata.partial_reason_codes == []
    assert rows and math.isfinite(rows[0]["dpi"])
    assert rows[0]["resolution"] == approx(1.5)
    assert math.isfinite(rows[0]["zscore"])


def test_ccp4_capability_probe_is_boolean_and_does_not_raise() -> None:
    """The local CCP4 probe answers with a bool instead of raising.

    ``conftest`` calls it during collection, so a probe that raised on an
    unreadable config would abort collection of the whole suite.
    """
    assert isinstance(helpers.ccp4_available(), bool)


def test_not_network_selection_does_not_call_the_network_probe(
    tmp_path: Path, repo_root: str
) -> None:
    """Marker deselection happens before the external capability probe.

    ``-m 'not network'`` is an offline selection boundary, not just a promise to
    discard the test after collection, so the nested run replaces the probe with
    one that raises.
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
def test_ccp4_env_fixture_resolves_every_required_tool(
    ccp4_env: dict[str, str],
) -> None:
    """The ``ccp4_env`` fixture hands back an environment CCP4 can actually run in.

    An install that resolves but is broken -- missing shared libraries, a stale
    setup script -- must be caught here rather than halfway through an
    end-to-end run.
    """
    resolved = {tool: helpers.which(tool, ccp4_env) for tool in helpers.CCP4_TOOLS}
    assert all(resolved.values()), resolved

    # What the loader says when a shared object is missing or a symbol is
    # unresolved. Not "no such file or directory": fft and mapmask print that
    # themselves when given no input file, which is healthy.
    loader_failures = (
        "error while loading shared libraries",
        "cannot open shared object file",
        "symbol lookup error",
        "undefined symbol",
    )

    for tool, path in resolved.items():
        assert path is not None
        # With no input each of the four prints a banner naming itself and
        # exits; the status differs between them (edstats 0, the rest 1).
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

        # Negative means killed by a signal, so a SIGSEGV on startup is caught
        # here rather than read as a normal nonzero exit.
        assert completed.returncode >= 0, (
            f"{tool} died on signal {-completed.returncode}: {excerpt!r}"
        )
        for marker in loader_failures:
            assert marker not in output.lower(), f"{tool}: {excerpt!r}"
        # mtzfix prints "MTZFIX", fft prints "FFTBIG", mapmask and edstats print
        # their own names; reaching the banner means the binary loaded.
        assert tool in output.lower(), (
            f"{tool} produced no banner naming itself: {excerpt!r}"
        )


@pytest.mark.network
def test_network_marker_only_runs_with_connectivity() -> None:
    """PDB-REDO really answers, not merely a TCP handshake on port 443.

    A captive portal or a proxy that accepts connections and then refuses them
    passes the socket probe used for skipping but fails here.
    """
    assert helpers.network_available()

    url = "https://pdb-redo.eu/db/9myr/data.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        assert response.status == 200
        loaded: object = json.loads(response.read().decode("utf-8"))
    # A JSON-formatted proxy error is still a dict, so require the
    # crystallographic metadata Alchemy consumes from data.json.
    assert isinstance(loaded, dict)
    payload = cast("dict[object, object]", loaded)
    raw_properties = payload.get("properties")
    assert isinstance(raw_properties, dict) and raw_properties
    properties = cast("dict[object, object]", raw_properties)
    nrefcnt = properties["NREFCNT"]
    dataresh = properties["DATARESH"]
    assert isinstance(nrefcnt, (str, int, float))
    assert isinstance(dataresh, (str, int, float))
    assert float(nrefcnt) > 0
    assert float(dataresh) > 0


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


def test_gemmi_is_the_expected_flavour() -> None:
    """The installed gemmi has the APIs the helpers and src depend on.

    The helpers declare connections through ``Connection``/``AtomAddress``, and
    ``src/structure_analysis.py`` uses ``Model.num``, which arrived in 0.7.
    """
    assert hasattr(gemmi, "Connection") and hasattr(gemmi, "AtomAddress")
    # Read via getattr: gemmi ships py.typed but its stubs omit __version__,
    # so a direct access is a type error against a real runtime name.
    version = str(getattr(gemmi, "__version__", "unknown"))
    assert tuple(int(p) for p in version.split(".")[:2]) >= (0, 7)
