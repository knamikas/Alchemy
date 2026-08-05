"""The classification rules that decide what a metallocofactor is.

Scope: ``tools/build_metallocofactor_catalog.py``, whose output is bundled and
then read by every analysis run, where ``parent_type`` is ``cluster``, ``heme``
or neither for every metal site in the database. Rebuilding against the real
CCD takes a download and cannot be done in CI.

The rules are structural, not stoichiometric: an Fe/N/C count resembling a heme
is common in synthetic chelates, and sulfur in a formula says nothing about
whether it bridges two metals. Every test here therefore builds a CCD block
with real connectivity and asserts on what the graph says.

``CANONICAL_CLASSES`` in the tool pins the published cofactors against the real
CCD at build time. These tests pin the rules themselves, against synthetic
components small enough to reason about.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import gemmi
import pytest

from helpers import REPO_ROOT


def _load_tool():
    """Import the builder by path.

    ``tools/`` must stay off ``sys.path``: it holds developer utilities the
    pipeline never imports, and a ``src`` module importing one would go
    unnoticed for the whole session.
    """
    path = os.path.join(REPO_ROOT, "tools", "build_metallocofactor_catalog.py")
    spec = importlib.util.spec_from_file_location("_catalog_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


catalog = _load_tool()


def component_block(component_id: str, atoms, bonds) -> gemmi.cif.Block:
    """Return a CCD block for one component.

    ``atoms`` is ``{atom_id: element}``; ``bonds`` is pairs of atom ids. Only
    the columns ``_component_graph`` reads are written; a real CCD block
    carries dozens more that the classification never looks at.
    """
    document = gemmi.cif.Document()
    block = document.add_new_block(component_id)
    atom_loop = block.init_loop(
        "_chem_comp_atom.", ["comp_id", "atom_id", "type_symbol"]
    )
    for atom_id, element in atoms.items():
        atom_loop.add_row([component_id, atom_id, element])
    bond_loop = block.init_loop(
        "_chem_comp_bond.", ["comp_id", "atom_id_1", "atom_id_2"]
    )
    for atom_1, atom_2 in bonds:
        bond_loop.add_row([component_id, atom_1, atom_2])
    return block


def iron_sulfur_cube():
    """A 4Fe-4S cubane: every sulfur bridges three irons."""
    irons = [f"FE{index}" for index in range(1, 5)]
    sulfurs = [f"S{index}" for index in range(1, 5)]
    atoms = {atom: ("FE" if atom.startswith("FE") else "S") for atom in irons + sulfurs}
    bonds = [
        (iron, sulfur)
        for index, sulfur in enumerate(sulfurs)
        for iron in irons[:index] + irons[index + 1 :]
    ]
    return atoms, bonds


def porphyrin(
    *,
    iron: bool = True,
    nitrogen_bonds: int = 4,
    carbons: int = 20,
    oxygens: int = 0,
):
    """A porphyrin-sized macrocycle: one conjugated ring of C and N.

    The ring is a single cycle, so it is the one biconnected component the heme
    rule looks for; only its sizes matter. ``oxygens`` stands in for the ring
    heteroatoms of an oxaporphyrin, which count toward the core size but not
    toward the carbon floor.
    """
    ring = (
        [f"N{index}" for index in range(1, 5)]
        + [f"C{index}" for index in range(1, carbons + 1)]
        + [f"O{index}" for index in range(1, oxygens + 1)]
    )
    atoms = {
        atom: ("N" if atom.startswith("N") else "O" if atom.startswith("O") else "C")
        for atom in ring
    }
    bonds = [(ring[index], ring[(index + 1) % len(ring)]) for index in range(len(ring))]
    if iron:
        atoms["FE"] = "FE"
        bonds += [(f"N{index}", "FE") for index in range(1, nitrogen_bonds + 1)]
    return atoms, bonds


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("Fe4 S4", {"FE": 4, "S": 4}),
        # The CCD writes a count of one as a bare symbol.
        ("C34 H32 Fe N4 O4", {"C": 34, "H": 32, "FE": 1, "N": 4, "O": 4}),
        ("Fe2 S2", {"FE": 2, "S": 2}),
        ("C2 H6 C3", {"C": 5, "H": 6}),
        ("", {}),
        (None, {}),
    ],
)
def test_element_counts_parses_a_ccd_formula(formula, expected):
    """The formula parse decides which components are even considered."""
    assert catalog.element_counts(formula) == expected


def test_a_charge_suffix_is_not_read_as_an_element():
    """CCD formulae carry a trailing charge; it must not become an atom count."""
    counts = catalog.element_counts("Fe4 S4 2-")

    assert counts == {"FE": 4, "S": 4}


def test_the_parser_reads_element_case_the_way_the_ccd_writes_it():
    """Two-letter symbols must arrive capitalized, and the CCD writes them so.

    ``FE4`` is not a misspelling this parser absorbs: it reads as fluorine
    followed by four of element ``E``, silently rather than as an error.
    """
    assert catalog.element_counts("FE4 S4") == {"F": 1, "E": 4, "S": 4}
    assert "FE" not in catalog.element_counts("FE4 S4")


def test_a_simple_cycle_is_one_biconnected_component():
    """The porphyrinoid core is found as a cycle, so this is the base case."""
    adjacency = {
        "A": {"B", "D"},
        "B": {"A", "C"},
        "C": {"B", "D"},
        "D": {"C", "A"},
    }

    components = catalog._biconnected_components(adjacency, adjacency)

    assert components == [{"A", "B", "C", "D"}]


def test_a_bridge_separates_two_rings():
    """Two rings joined by a single bond are two components: a macrocycle fused
    to a substituent is not one oversized conjugated core."""
    adjacency = {
        "A": {"B", "C"},
        "B": {"A", "C"},
        "C": {"A", "B", "D"},
        "D": {"C", "E", "F"},
        "E": {"D", "F"},
        "F": {"D", "E"},
    }

    components = catalog._biconnected_components(adjacency, adjacency)
    sizes = sorted(len(component) for component in components)

    assert sizes == [2, 3, 3], "the two triangles and the bridge between them"
    assert {"A", "B", "C"} in components
    assert {"D", "E", "F"} in components


def test_atoms_outside_the_allowed_set_are_not_traversed():
    """The heme rule excludes metals and hydrogens before looking for a core:
    a four-coordinate metal bridges separate ligands into a false one."""
    adjacency = {
        "A": {"B", "FE"},
        "B": {"A", "FE"},
        "FE": {"A", "B"},
    }

    components = catalog._biconnected_components(adjacency, {"A", "B"})

    assert components == [{"A", "B"}]


def test_an_isolated_atom_contributes_no_component():
    """A component needs an edge; a lone atom is not a ring of one."""
    assert catalog._biconnected_components({"A": set()}, {"A"}) == []


def test_a_cubane_is_a_cluster():
    """The 4Fe-4S case, which is what the class exists for."""
    atoms, bonds = iron_sulfur_cube()

    assert catalog.classify_component(component_block("SF4", atoms, bonds)) == "cluster"


def test_a_single_metal_bonded_to_sulfur_is_not_a_cluster():
    """One Fe-S bond is a thiolate ligand, not a metal-sulfur cluster.

    Stoichiometry cannot tell the two apart, which is why the rule requires a
    chalcogen bridging *two* cluster metals.
    """
    atoms = {"FE": "FE", "S1": "S", "C1": "C"}
    bonds = [("FE", "S1"), ("S1", "C1")]

    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == ""


def test_two_metals_sharing_a_sulfur_is_a_cluster():
    """The minimum the rule accepts: one bridging chalcogen, two metals."""
    atoms = {"FE1": "FE", "FE2": "FE", "S1": "S"}
    bonds = [("FE1", "S1"), ("FE2", "S1")]

    assert catalog.classify_component(component_block("FES", atoms, bonds)) == "cluster"


def test_selenium_bridges_count_as_well_as_sulfur():
    """Selenium-substituted clusters are the same architecture."""
    atoms = {"FE1": "FE", "FE2": "FE", "SE1": "SE"}
    bonds = [("FE1", "SE1"), ("FE2", "SE1")]

    assert catalog.classify_component(component_block("SEC", atoms, bonds)) == "cluster"


def test_a_bridged_pair_of_non_cluster_metals_is_not_a_cluster():
    """Only the metals that form these cofactors qualify: zinc bridged by a
    sulfur is an ordinary binuclear thiolate site."""
    atoms = {"ZN1": "ZN", "ZN2": "ZN", "S1": "S"}
    bonds = [("ZN1", "S1"), ("ZN2", "S1")]

    assert catalog.classify_component(component_block("ZNS", atoms, bonds)) == ""


def test_an_iron_porphyrin_is_a_heme():
    """The canonical case: one Fe in a 24-atom conjugated macrocycle."""
    atoms, bonds = porphyrin()

    assert catalog.classify_component(component_block("HEM", atoms, bonds)) == "heme"


def test_a_porphyrin_without_iron_is_not_a_heme():
    """A free-base porphyrin has the ring but coordinates nothing."""
    atoms, bonds = porphyrin(iron=False)

    assert catalog.classify_component(component_block("POR", atoms, bonds)) == ""


def test_a_covalently_modified_heme_with_three_fe_n_bonds_is_still_a_heme():
    """Three Fe-N bonds is the floor: CCD ``WRK`` and its relatives lose a
    fourth Fe-N bond to a covalent link."""
    atoms, bonds = porphyrin(nitrogen_bonds=3)

    assert catalog.classify_component(component_block("WRK", atoms, bonds)) == "heme"


def test_two_fe_n_bonds_is_not_a_heme():
    """The other side of the same threshold."""
    atoms, bonds = porphyrin(nitrogen_bonds=2)

    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == ""


def test_a_ring_smaller_than_a_porphyrin_is_not_a_heme():
    """An Fe-N chelate with a small ring is a synthetic complex, not a heme."""
    atoms, bonds = porphyrin(carbons=8)

    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == ""


def test_a_core_one_atom_short_of_a_porphyrin_is_not_a_heme():
    """The size floor is checked separately from the rest.

    This ring passes the carbon, nitrogen and Fe-N bond counts -- 18 C, 4 N,
    four donors -- and is refused on size alone, at 22 atoms against the 24 a
    porphyrinoid has.
    """
    atoms, bonds = porphyrin(carbons=18)

    assert len([atom for atom in atoms if atom != "FE"]) == 22
    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == ""


def test_an_oxaporphyrin_reaches_the_size_floor_on_heteroatoms():
    """Ring oxygens count toward size, not toward carbon.

    Oxaporphyrins replace a ring carbon with a heteroatom, which is why the
    carbon floor (18) sits below the core size (24) rather than at it.
    """
    atoms, bonds = porphyrin(carbons=18, oxygens=2)

    assert catalog.classify_component(component_block("OXA", atoms, bonds)) == "heme"


def test_a_component_with_two_irons_is_not_a_heme():
    """The heme rule is single-iron: a second one means a polynuclear site,
    which is either a cluster or outside both populations."""
    atoms, bonds = porphyrin()
    atoms["FE2"] = "FE"
    bonds.append(("N1", "FE2"))

    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == ""


def test_cluster_wins_when_a_component_could_be_read_as_both():
    """Cluster wins, because ``_parent_type`` in the analysis resolves the same
    collision the same way and the two must agree."""
    atoms, bonds = porphyrin()
    atoms.update({"FE2": "FE", "S1": "S"})
    bonds += [("FE", "S1"), ("FE2", "S1")]

    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == "cluster"


def test_a_bond_naming_an_absent_atom_is_ignored_rather_than_fatal():
    """The CCD is 40,000 components; one bad bond row costs that component its
    classification, not the whole rebuild."""
    atoms = {"FE1": "FE", "FE2": "FE", "S1": "S"}
    bonds = [("FE1", "S1"), ("FE2", "S1"), ("S1", "GHOST")]

    assert catalog.classify_component(component_block("FES", atoms, bonds)) == "cluster"


def _load_stamp_tool():
    """Import the distance-table stamper by path, like the catalog builder."""
    path = os.path.join(REPO_ROOT, "tools", "stamp_distance_table.py")
    spec = importlib.util.spec_from_file_location("_distance_stamper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stamper = _load_stamp_tool()


def test_the_committed_sidecar_matches_the_committed_table(capsys):
    """``--check`` is what tells a maintainer the sidecar is still current."""
    assert stamper.main(["--check"]) == 0
    assert "sidecar matches" in capsys.readouterr().out


def test_a_corrupt_sidecar_is_reported_rather_than_raised(
    tmp_path, monkeypatch, capsys
):
    """A truncated sidecar is exactly what --check exists to catch.

    Only ``OSError`` was handled, so a half-written file raised
    ``JSONDecodeError`` -- a traceback, from the command whose job is to report
    that the sidecar cannot be trusted.
    """
    sidecar = tmp_path / "metal_distances_info.meta.json"
    sidecar.write_text('{"distance_table_sha256": "abc', encoding="utf-8")
    monkeypatch.setattr(stamper, "SIDECAR_PATH", str(sidecar))

    assert stamper.main(["--check"]) == 1
    assert "unreadable sidecar" in capsys.readouterr().out


def test_a_missing_sidecar_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stamper, "SIDECAR_PATH", str(tmp_path / "absent.json"))

    assert stamper.main(["--check"]) == 1
    assert "no sidecar" in capsys.readouterr().out


def test_the_recorded_row_count_comes_from_the_loader(tmp_path):
    """The count must describe what ``reference_data`` accepts, not raw lines.

    Counting lines would drift from the loader the moment a comment or blank
    line was added, and the sidecar is what the loader verifies itself against.
    """
    table = tmp_path / "distances.txt"
    table.write_text(
        "# a comment\n\nHOH O ZN 2.09 0.11\nCYS S ZN 2.31 0.10\n", encoding="utf-8"
    )

    metadata = stamper.build_metadata(str(table), "2026-08-05T00:00:00+00:00")

    assert metadata["reference_distance_count"] == 2
    assert metadata["distance_table_sha256"] == stamper.sha256(str(table))
    assert metadata["sources"], "citations are the point of the sidecar"
