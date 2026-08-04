"""The classification rules that decide what a metallocofactor is.

Scope: ``tools/build_metallocofactor_catalog.py``. It runs once in a while, by
hand, with the network -- and its output is bundled and then read by every
analysis run afterwards, where ``parent_type`` is ``cluster``, ``heme``, or
neither for every metal site in the database. Until this module existed the
tool had no tests at all: a rule change was checked only by rebuilding against
the real CCD, which takes a download and cannot be done in CI.

The rules are structural, not stoichiometric, and the reason is in
``classify_component``'s docstring: an Fe/N/C count resembling a heme is
common in synthetic chelates, and sulfur in a formula says nothing about
whether it bridges two metals. Every test here therefore builds a CCD block
with real connectivity and asserts on what the graph says.

``CANONICAL_CLASSES`` in the tool already pins the published cofactors against
the real CCD at build time. These tests pin the rules themselves, against
synthetic components small enough to reason about.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import gemmi
import pytest

from helpers import REPO_ROOT


def _load_tool():
    """Import the builder from ``tools/``, which is not on ``sys.path``.

    Loaded by path rather than by adding the directory: ``tools/`` holds
    developer utilities that the pipeline never imports, and putting it on the
    path for the whole test session would let a src module import one by
    accident and never be noticed.
    """
    path = os.path.join(REPO_ROOT, "tools", "build_metallocofactor_catalog.py")
    spec = importlib.util.spec_from_file_location("_catalog_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


catalog = _load_tool()


# Local scaffolding
def component_block(component_id: str, atoms, bonds) -> gemmi.cif.Block:
    """Return a CCD block for one component from atoms and bonds.

    ``atoms`` is ``{atom_id: element}``; ``bonds`` is pairs of atom ids. Only
    the four columns ``_component_graph`` reads are written -- a real CCD block
    carries dozens more, none of which the classification looks at.
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

    The ring is built as a single cycle so it is one biconnected component,
    which is what the heme rule looks for. Only the sizes matter to the rule,
    not which carbon is which. ``oxygens`` stands in for the ring heteroatoms
    of an oxaporphyrin, which count toward the core size but not toward the
    carbon floor.
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


# element_counts
@pytest.mark.parametrize(
    "formula,expected",
    [
        ("Fe4 S4", {"FE": 4, "S": 4}),
        # An element with no digit means one, which is how the CCD writes it.
        ("C34 H32 Fe N4 O4", {"C": 34, "H": 32, "FE": 1, "N": 4, "O": 4}),
        # The result is upper-cased to meet ``METAL_ELEMENTS``.
        ("Fe2 S2", {"FE": 2, "S": 2}),
        # A repeated symbol accumulates rather than replacing.
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

    ``FE4`` is not a misspelling this parser can absorb: it reads as fluorine
    followed by four of element ``E``. Recorded because the input contract is
    invisible at the call site -- the formula comes straight from
    ``_chem_comp.formula`` -- and because the result is silent, not an error.
    """
    assert catalog.element_counts("FE4 S4") == {"F": 1, "E": 4, "S": 4}
    assert "FE" not in catalog.element_counts("FE4 S4")


# _biconnected_components
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
    """Two rings joined by a single bond are two components, not one.

    This is the property the heme rule depends on: a macrocycle fused to a
    substituent must not be counted as one oversized conjugated core.
    """
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
    """The heme rule excludes metals and hydrogens before looking for a core.

    Without that exclusion the iron itself would join the ring it coordinates
    into one component, and a four-coordinate metal would bridge otherwise
    separate ligands into a false core.
    """
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


# classify_component: clusters
def test_a_cubane_is_a_cluster():
    """The 4Fe-4S case, which is what the class exists for."""
    atoms, bonds = iron_sulfur_cube()

    assert catalog.classify_component(component_block("SF4", atoms, bonds)) == "cluster"


def test_a_single_metal_bonded_to_sulfur_is_not_a_cluster():
    """One Fe-S bond is a thiolate ligand, not a metal-sulfur cluster.

    Stoichiometry cannot tell these apart -- both have iron and sulfur -- which
    is why the rule requires a chalcogen bridging *two* cluster metals.
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
    """Only the metals that form these cofactors qualify.

    Zinc bridged by a sulfur is an ordinary binuclear thiolate site, and
    calling it a cluster would put it in a reference population it does not
    belong to.
    """
    atoms = {"ZN1": "ZN", "ZN2": "ZN", "S1": "S"}
    bonds = [("ZN1", "S1"), ("ZN2", "S1")]

    assert catalog.classify_component(component_block("ZNS", atoms, bonds)) == ""


# classify_component: hemes
def test_an_iron_porphyrin_is_a_heme():
    """The canonical case: one Fe in a 24-atom conjugated macrocycle."""
    atoms, bonds = porphyrin()

    assert catalog.classify_component(component_block("HEM", atoms, bonds)) == "heme"


def test_a_porphyrin_without_iron_is_not_a_heme():
    """A free-base porphyrin has the ring but coordinates nothing."""
    atoms, bonds = porphyrin(iron=False)

    assert catalog.classify_component(component_block("POR", atoms, bonds)) == ""


def test_a_covalently_modified_heme_with_three_fe_n_bonds_is_still_a_heme():
    """Three Fe-N bonds is the documented floor, and it is deliberate.

    CCD ``WRK`` and its relatives lose a fourth Fe-N bond to a covalent link;
    demanding four would drop them out of the heme population.
    """
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
    """The size floor is a floor, and it is checked separately from the rest.

    This ring passes the carbon and nitrogen counts and the Fe-N bond count --
    18 C, 4 N, four donors -- and is refused on size alone, at 22 atoms
    against the 24 a porphyrinoid has. Without this the atom-count floor could
    be deleted and every other heme test would still pass.
    """
    atoms, bonds = porphyrin(carbons=18)

    assert len([atom for atom in atoms if atom != "FE"]) == 22
    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == ""


def test_an_oxaporphyrin_reaches_the_size_floor_on_heteroatoms():
    """The other side of it: ring oxygens count toward size, not toward carbon.

    Oxaporphyrins replace a ring carbon with a heteroatom, which is why the
    carbon floor (18) sits below the core size (24) rather than at it.
    """
    atoms, bonds = porphyrin(carbons=18, oxygens=2)

    assert catalog.classify_component(component_block("OXA", atoms, bonds)) == "heme"


def test_a_component_with_two_irons_is_not_a_heme():
    """The heme rule is single-iron by construction.

    A second iron means a polynuclear site, which either qualifies as a cluster
    by the rule above or is outside both populations.
    """
    atoms, bonds = porphyrin()
    atoms["FE2"] = "FE"
    bonds.append(("N1", "FE2"))

    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == ""


def test_cluster_wins_when_a_component_could_be_read_as_both():
    """Precedence is asserted, not left to the order of two ``if``s.

    ``_parent_type`` in the analysis resolves the same collision the same way,
    so a component present in both sets must be tagged identically by the two.
    """
    atoms, bonds = porphyrin()
    atoms.update({"FE2": "FE", "S1": "S"})
    bonds += [("FE", "S1"), ("FE2", "S1")]

    assert catalog.classify_component(component_block("XYZ", atoms, bonds)) == "cluster"


def test_a_bond_naming_an_absent_atom_is_ignored_rather_than_fatal():
    """A malformed CCD block must not take the whole rebuild down.

    The CCD is 40,000 components; one bad bond row should cost that component
    its classification, not the catalog.
    """
    atoms = {"FE1": "FE", "FE2": "FE", "S1": "S"}
    bonds = [("FE1", "S1"), ("FE2", "S1"), ("S1", "GHOST")]

    assert catalog.classify_component(component_block("FES", atoms, bonds)) == "cluster"
