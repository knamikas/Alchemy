"""Coordination chemistry and geometry in ``src/bond_analysis.py``.

Scope: the atom-level inferred-donor table, the polymer-terminal exceptions,
Harding's first-sphere rule and its same-element fallback, the resolution-aware
z-score and its |Z| >= 6 outlier boundary, the DPI that forms its denominator
(the Blow 2002 formula, the asymmetric-unit volume, the R-free scrape and every
reason a site can end up without one), the nullable geometry columns,
multi-donor grouping, and the contents and physical plausibility of the bundled
reference table ``src/data/metal_distances_info.txt``.

Out of scope here (owned elsewhere): the declared ``struct_conn``/``LINK``
resolution path, EDSTATS parsing, the manifest, and the driver.

Everything is built in memory with :mod:`helpers` and written to ``tmp_path``;
no network, no CCP4, no fixed execution order.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import gemmi
import pytest

import bond_analysis as ba
import helpers
from helpers import AtomSpec, EDSTATS_HEADER, StructureBuilder
from metal_elements import METAL_ELEMENTS
from structure_analysis import count_ni, load_structure


# --------------------------------------------------------------------------- #
# Expected donor table, transcribed from README.md ("The geometry-inference
# donor table covers all 20 standard amino acids...").
# --------------------------------------------------------------------------- #
README_SIDE_CHAIN_DONORS = {
    "ASN": {"OD1"},
    "ASP": {"OD1", "OD2"},
    "CYS": {"SG"},
    "GLN": {"OE1"},
    "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "MET": {"SD"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}

#: Atoms README names explicitly as *not* geometry-inferable donors.
README_EXCLUDED_ATOMS = [
    ("ALA", "N"),    # internal peptide nitrogen
    ("GLY", "N"),
    ("PRO", "N"),
    ("ASN", "ND2"),  # amide nitrogen
    ("GLN", "NE2"),  # amide nitrogen (same atom name as the HIS donor)
    ("TRP", "NE1"),  # pyrrole nitrogen
    ("ARG", "NE"),   # guanidinium nitrogens
    ("ARG", "NH1"),
    ("ARG", "NH2"),
]

REFERENCE_TABLE = os.path.join(helpers.SRC_DIR, "data", "metal_distances_info.txt")

#: Residue tokens the reference table is allowed to use: the 20 standard amino
#: acids plus ``HOH`` (water) and ``CA`` (the backbone-carbonyl pseudo residue
#: that ``_bonding_key`` maps every main-chain ``O`` onto).
REFERENCE_RESIDUE_TOKENS = helpers.STANDARD_AMINO_ACIDS | {"HOH", "CA"}

#: The one header line of ``metal_distances_info.txt``.
REFERENCE_TABLE_HEADER = ["residue", "atom", "metal", "avg_bond_dist", "st_dev"]

#: How many data rows the bundled table has. Pinned because a dropped row is
#: invisible at runtime: the pair just falls back to the same-element target and
#: loses its exact reference, its sigma and therefore its z-score.
EXPECTED_REFERENCE_ROW_COUNT = 79

#: The ten metals the table covers, in the order its blocks use.
REFERENCE_METALS = ("NA", "MG", "K", "CA", "MN", "FE", "CO", "CU", "ZN", "NI")

#: Keys whose loss would be most damaging and least visible: every metal's water
#: reference (the commonest first-sphere donor in the PDB), the thiolate and
#: imidazole rows that define transition-metal sites, the carboxylate rows, and
#: the backbone-carbonyl pseudo residue.
REQUIRED_REFERENCE_KEYS = frozenset(
    [("HOH", "O", metal) for metal in REFERENCE_METALS]
    + [("ASP", "O", metal) for metal in REFERENCE_METALS]
    + [("GLU", "O", metal) for metal in REFERENCE_METALS]
    + [("CA", "O", metal) for metal in REFERENCE_METALS]
    + [(residue, "O", metal)
       for residue in ("SER", "THR", "TYR")
       for metal in ("NA", "MG", "K", "CA", "MN", "FE", "CO", "CU", "ZN")]
    + [("HIS", "N", metal) for metal in ("MN", "FE", "CO", "CU", "ZN", "NI")]
    + [("CYS", "S", metal) for metal in ("MN", "FE", "CO", "CU", "ZN", "NI")]
)


# --------------------------------------------------------------------------- #
# Private builders (kept local so tests/helpers.py stays untouched)
# --------------------------------------------------------------------------- #
def _probe_structure(tmp_path, name, *, resname, positions, metal="ZN",
                     probe_index=1, chain_length=3, metal_kwargs=None):
    """Write a metal + polymer chain in which one residue reaches the metal.

    The metal sits at the origin in chain ``B``. Chain ``A`` holds
    ``chain_length`` copies of ``resname`` at widely separated origins, so only
    the atoms named in ``positions`` -- applied to residue ``probe_index`` -- are
    anywhere near the metal. ``probe_index`` therefore controls whether the
    donor residue is the polymer's first, last, or an internal residue.
    """
    builder = StructureBuilder()
    builder.add_metal(metal, 1, chain="B", pos=(0.0, 0.0, 0.0),
                      **(metal_kwargs or {}))
    for index in range(chain_length):
        builder.add_amino_acid(
            resname, 10 + index, chain="A",
            positions=dict(positions) if index == probe_index else None,
            origin=(20.0 + 14.0 * index, 20.0, 20.0))
    return builder.write_pdb(tmp_path / name)


def _analyze(path, *, data_json=None, resolution=1.50):
    """Run the real bond analysis on ``path`` and return its four outputs."""
    context = load_structure("test", path)
    if data_json is None:
        inputs = helpers.dpi_inputs(resolution=resolution)
    else:
        inputs = helpers.dpi_inputs(
            pdb_path=path,
            mtz_path=os.path.join(os.path.dirname(path), "absent.mtz"),
            data_json=data_json, resolution=resolution)
    return ba.run_bond_analysis("test", path, [], list(EDSTATS_HEADER), inputs,
                                structure=context)


def _dpi_metadata(tmp_path, *, nrefcnt=50000, rffin=0.20, name="data.json"):
    """A PDB-REDO-style ``data.json`` giving a finite DPI."""
    return helpers.write_data_json(tmp_path / name, nrefcnt=nrefcnt, rffin=rffin)


def _only(rows, atom_name):
    """The single row/candidate whose neighbour atom is ``atom_name``."""
    matches = [row for row in rows if row["neighbor_atom"] == atom_name]
    assert len(matches) == 1, (
        f"expected exactly one entry for {atom_name}, got "
        f"{[(m['neighbor_resnum'], m['neighbor_atom']) for m in matches]}")
    return matches[0]


class _StubAtom:
    """Minimal stand-in for an analysis atom, for pure-function tests."""

    def __init__(self, element, residue_name="", atom_name="", is_water=False):
        self.element = element
        self.residue_name = residue_name
        self.atom_name = atom_name
        self.is_water = is_water


def _parse_reference_table(path):
    """Strict, independent parse of metal_distances_info.txt -> list of records.

    ``bond_analysis._load_literature`` skips any line whose 4th/5th tokens do
    not parse as floats, which is how it drops the header. Copying that rule
    here would make ``parsed == ba.LIT`` a tautology: a corrupted ``CYS S CU``
    row would be dropped by both parsers alike, and the pair would silently
    degrade to the element fallback with the whole suite still green.

    So this parser skips nothing. The first line must be the documented header,
    every other line must be blank or exactly five well-formed columns, and
    anything else fails the test that called it.
    """
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    assert lines, f"{path} is empty"
    assert lines[0].split() == REFERENCE_TABLE_HEADER, (
        f"{path}:1: unexpected header {lines[0]!r}")

    records = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue                      # blank separator between metal blocks
        fields = line.split()
        assert len(fields) == 5, (
            f"{path}:{lineno}: expected 5 whitespace-separated columns, got "
            f"{len(fields)}: {line!r}")
        residue, atom, metal, mu_text, stdev_text = fields
        try:
            mu, stdev = float(mu_text), float(stdev_text)
        except ValueError as exc:
            raise AssertionError(
                f"{path}:{lineno}: distance columns are not numbers: "
                f"{line!r}") from exc
        records.append((lineno, residue, atom, metal, mu, stdev))
    return records


# --------------------------------------------------------------------------- #
# 1. The inferred-donor table
# --------------------------------------------------------------------------- #
def test_inferred_donor_table_matches_the_documented_atom_list():
    """The donor table is exactly README's list: backbone O plus named side chains.

    A single wrong entry silently changes coordination numbers for every entry
    in the database, so the whole table is compared element by element rather
    than spot-checked.
    """
    expected = {residue: {"O"} | README_SIDE_CHAIN_DONORS.get(residue, set())
                for residue in helpers.STANDARD_AMINO_ACIDS}
    actual = {residue: set(atoms)
              for residue, atoms in ba.INFERRED_DONOR_ATOMS.items()}

    assert actual == expected
    # The excluded atoms README calls out must not have crept in.
    for residue, atom in README_EXCLUDED_ATOMS:
        assert atom not in ba.INFERRED_DONOR_ATOMS[residue]
    assert set(ba.N_TERMINAL_DONOR_ATOMS) == {"N"}
    assert set(ba.C_TERMINAL_DONOR_ATOMS) == {"OXT", "OT1", "OT2"}


@pytest.mark.parametrize(
    "resname,atom_name",
    sorted((residue, atom)
           for residue, atoms in README_SIDE_CHAIN_DONORS.items()
           for atom in atoms))
def test_named_side_chain_donors_become_inferred_bonds(tmp_path, resname,
                                                       atom_name):
    """Each README side-chain donor is inferable from geometry alone.

    The atom is placed 2.0 A from a Zn, inside the first-sphere cutoff for every
    donor element, and must produce one inferred bond row attributed to the
    proximity rule.
    """
    path = _probe_structure(tmp_path, f"{resname}_{atom_name}.pdb",
                            resname=resname, positions={atom_name: (2.0, 0.0, 0.0)})
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, atom_name)
    assert candidate["inferred_donor_allowed"] is True
    assert candidate["inferred_donor_rule"] == "typical_sidechain_donor"
    assert candidate["inferred_contact_eligible"] is True
    # Nothing is flagged: a typical donor inside the sphere is unremarkable.
    assert candidate["context_warning"] is False

    row = _only(rows, atom_name)
    assert row["coordination_status"] == "inferred"
    assert row["coordination_source"] == "proximity_rule"
    assert row["declared_connection"] is False
    assert row["distance"] == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("resname", sorted(helpers.STANDARD_AMINO_ACIDS))
def test_backbone_carbonyl_oxygen_is_a_donor_for_every_residue(tmp_path, resname):
    """Main-chain ``O`` donates for all 20 residues and uses the generic reference.

    ``_bonding_key`` maps every backbone carbonyl onto the table's ``CA`` rows,
    not onto the residue's own side-chain row, so a Zn-O(backbone) contact must
    be scored against ``CA:O:ZN`` whatever the residue is.
    """
    path = _probe_structure(tmp_path, f"{resname}_O.pdb", resname=resname,
                            positions={"O": (2.0, 0.0, 0.0)})
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, "O")
    assert candidate["inferred_donor_allowed"] is True
    assert candidate["inferred_donor_rule"] == "backbone_carbonyl_oxygen"
    assert candidate["assignment_reference_kind"] == "exact"
    assert candidate["assignment_reference"] == "CA:O:ZN"

    row = _only(rows, "O")
    assert row["neighbor_resname"] == resname
    assert row["neighbor_class"] == "amino_acid"
    assert row["bonded_to"] == "P"
    assert row["literature_distance"] == pytest.approx(ba.LIT[("CA", "O", "ZN")][0])


def test_water_oxygen_is_a_donor_and_other_water_atoms_are_not(tmp_path):
    """Water donates through ``O`` only; a non-oxygen water atom is not inferable.

    Water is recognised by residue name rather than by the amino-acid table, so
    the element check is the only thing stopping a mis-modelled water atom from
    becoming a bond.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_water(101, (2.09, 0.0, 0.0), chain="B")
    path = builder.write_pdb(tmp_path / "water.pdb")
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, "O")
    assert candidate["inferred_donor_allowed"] is True
    assert candidate["inferred_donor_rule"] == "water_oxygen"
    row = _only(rows, "O")
    assert row["neighbor_class"] == "water"
    assert row["bonded_to"] == "HOH"

    # A water residue whose atom is not oxygen must not be inferable.
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_hetero_residue("HOH", 101, [AtomSpec("N", "N", (2.09, 0.0, 0.0))],
                               chain="B")
    path = builder.write_pdb(tmp_path / "odd_water.pdb")
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, "N")
    assert candidate["inferred_donor_allowed"] is False
    assert candidate["inferred_donor_rule"] == "outside_typical_donor_list"
    assert rows == []


@pytest.mark.parametrize("resname,atom_name", README_EXCLUDED_ATOMS)
def test_excluded_nitrogen_donors_never_become_inferred_bonds(tmp_path, resname,
                                                              atom_name):
    """Peptide, amide, pyrrole and guanidinium N stay candidates, never bonds.

    Each is placed 2.0 A from a Zn -- comfortably inside the first-sphere
    distance -- so only the atom-level chemical rule can keep it out. It must be
    retained as candidate evidence, marked non-typical, and produce no bond row.
    """
    path = _probe_structure(tmp_path, f"{resname}_{atom_name}.pdb",
                            resname=resname,
                            positions={atom_name: (2.0, 0.0, 0.0)})
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, atom_name)
    assert candidate["inferred_donor_allowed"] is False
    assert candidate["inferred_donor_rule"] == "outside_typical_donor_list"
    # Proximity alone still satisfies the distance rule; only the donor table
    # stops it, which is exactly what the eligibility status must say.
    assert candidate["first_sphere_eligible"] is True
    assert candidate["inferred_contact_eligible"] is False
    assert candidate["eligibility_status"] == "non_typical_donor"
    assert candidate["eligibility_reason"] == "atom_not_in_typical_inferred_donor_list"
    assert candidate["donor_rule_override"] == ""
    assert candidate["context_warning_reasons"] == "non_typical_first_sphere_candidate"
    assert rows == []


def test_a_distant_non_typical_atom_does_not_warn_the_site(tmp_path):
    """Only a *first-sphere* non-typical atom raises the site context warning.

    README: "A distant non-typical atom found only by the broad 4 A search does
    not by itself place the complete metal site under warning."
    """
    near = _probe_structure(tmp_path, "arg_near.pdb", resname="ARG",
                            positions={"NH1": (2.0, 0.0, 0.0)})
    far = _probe_structure(tmp_path, "arg_far.pdb", resname="ARG",
                           positions={"NH1": (3.6, 0.0, 0.0)})

    _, near_candidates, near_summaries, _ = _analyze(near)
    _, far_candidates, far_summaries, _ = _analyze(far)

    assert _only(near_candidates, "NH1")["first_sphere_eligible"] is True
    assert _only(far_candidates, "NH1")["first_sphere_eligible"] is False

    near_summary = next(iter(near_summaries.values()))
    far_summary = next(iter(far_summaries.values()))
    assert near_summary["non_typical_first_sphere_candidate_count"] == 1
    assert near_summary["context_warning"] is True
    assert far_summary["non_typical_first_sphere_candidate_count"] == 0
    assert far_summary["context_warning"] is False
    # The distant atom is still visible as candidate evidence.
    assert _only(far_candidates, "NH1")["context_warning_reasons"] == (
        "non_typical_proximal_candidate")


# --------------------------------------------------------------------------- #
# 2. Polymer-terminal donor exceptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("probe_index,allowed", [(0, True), (1, False), (2, False)])
def test_n_terminal_nitrogen_is_a_donor_only_at_the_chain_start(
        tmp_path, probe_index, allowed):
    """Backbone ``N`` donates from the first polymer residue and nowhere else."""
    path = _probe_structure(tmp_path, f"nterm{probe_index}.pdb", resname="ALA",
                            positions={"N": (2.0, 0.0, 0.0)},
                            probe_index=probe_index)
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, "N")
    assert candidate["neighbor_resnum"] == str(10 + probe_index)
    assert candidate["inferred_donor_allowed"] is allowed
    assert candidate["inferred_donor_rule"] == (
        "n_terminal_nitrogen" if allowed else "outside_typical_donor_list")
    assert len(rows) == (1 if allowed else 0)


@pytest.mark.parametrize("atom_name", ["OXT", "OT1", "OT2"])
@pytest.mark.parametrize("probe_index,allowed", [(0, False), (1, False), (2, True)])
def test_c_terminal_oxygens_are_donors_only_at_the_chain_end(
        tmp_path, atom_name, probe_index, allowed):
    """``OXT``/``OT1``/``OT2`` donate from the last polymer residue and nowhere else."""
    path = _probe_structure(tmp_path, f"cterm_{atom_name}{probe_index}.pdb",
                            resname="ALA",
                            positions={atom_name: (2.0, 0.0, 0.0)},
                            probe_index=probe_index)
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, atom_name)
    assert candidate["inferred_donor_allowed"] is allowed
    assert candidate["inferred_donor_rule"] == (
        "c_terminal_oxygen" if allowed else "outside_typical_donor_list")
    assert len(rows) == (1 if allowed else 0)
    if allowed:
        # No terminal-carboxylate reference is bundled, so the contact is
        # admitted through the element fallback rather than a misattributed
        # side-chain row.
        assert candidate["assignment_reference_kind"] == "element_fallback"
        assert candidate["assignment_reference"] == "*:O:ZN"


def test_a_single_residue_chain_is_both_polymer_termini(tmp_path):
    """A one-residue polymer is simultaneously the first and the last residue."""
    for atom_name, rule in (("N", "n_terminal_nitrogen"),
                            ("OXT", "c_terminal_oxygen")):
        path = _probe_structure(tmp_path, f"solo_{atom_name}.pdb", resname="ALA",
                                positions={atom_name: (2.0, 0.0, 0.0)},
                                probe_index=0, chain_length=1)
        rows, candidates, _, _ = _analyze(path)
        assert _only(candidates, atom_name)["inferred_donor_rule"] == rule
        assert len(rows) == 1


def test_terminal_rules_require_a_real_polymer_residue(tmp_path):
    """An amino acid modelled as a hetero component gets no terminal exception.

    ``_polymer_terminal_position`` keys on Gemmi's entity classification, so a
    free amino acid deposited as a HETATM component -- the first and only
    residue of its chain -- must still be refused, because it has no polymer
    boundary to sit at.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_hetero_residue("ALA", 20, [
        AtomSpec("N", "N", (2.0, 0.0, 0.0)),
        AtomSpec("CA", "C", (20.0, 20.0, 20.0)),
        AtomSpec("C", "C", (21.0, 20.0, 20.0)),
        AtomSpec("O", "O", (22.0, 20.0, 20.0)),
    ], chain="C")
    path = builder.write_pdb(tmp_path / "free_ala.pdb")

    rows, candidates, _, _ = _analyze(path)
    context = load_structure("test", path)
    entity_types = {residue.name: residue.entity_type
                    for chain in context.model for residue in chain}
    assert entity_types["ALA"] != gemmi.EntityType.Polymer

    candidate = _only(candidates, "N")
    assert candidate["inferred_donor_allowed"] is False
    assert candidate["inferred_donor_rule"] == "outside_typical_donor_list"
    assert rows == []


# --------------------------------------------------------------------------- #
# 3. _first_sphere_rule: Harding target + 0.75 A
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metal,residue,atom,element,is_water,expected_key", [
    ("ZN", "HIS", "NE2", "N", False, ("HIS", "N", "ZN")),
    ("ZN", "ASP", "OD2", "O", False, ("ASP", "O", "ZN")),
    ("ZN", "CYS", "SG", "S", False, ("CYS", "S", "ZN")),
    ("ZN", "ALA", "O", "O", False, ("CA", "O", "ZN")),
    ("MG", "HOH", "O", "O", True, ("HOH", "O", "MG")),
])
def test_first_sphere_cutoff_is_the_exact_target_plus_the_harding_tolerance(
        metal, residue, atom, element, is_water, expected_key):
    """An exact reference sets target = mu and cutoff = mu + 0.75 A."""
    stub_metal = _StubAtom(metal)
    neighbor = _StubAtom(element, residue_name=residue, atom_name=atom,
                         is_water=is_water)
    mu = ba.LIT[expected_key][0]

    target, cutoff, kind, key = ba._first_sphere_rule(stub_metal, neighbor)

    assert kind == "exact"
    assert key == ":".join(expected_key)
    assert target == pytest.approx(mu)
    assert cutoff == pytest.approx(mu + 0.75)
    assert ba.FIRST_SPHERE_TOLERANCE == 0.75


def test_missing_exact_reference_falls_back_to_the_largest_same_element_target():
    """The fallback uses the *largest* same-metal/same-element target, not any.

    ASN OD1 has no bundled Zn row, so sphere membership must use the widest Zn-O
    target in the table (water, 2.09 A) rather than, say, the Zn-Asp value. The
    fallback is flagged as such because it may not be used for a z-score.
    """
    stub_metal = _StubAtom("ZN")
    neighbor = _StubAtom("O", residue_name="ASN", atom_name="OD1")
    widest_zn_o = max(mu for (_, atom, metal), (mu, _sd) in ba.LIT.items()
                      if metal == "ZN" and atom == "O")

    target, cutoff, kind, key = ba._first_sphere_rule(stub_metal, neighbor)

    assert kind == "element_fallback"
    assert key == "*:O:ZN"
    assert target == pytest.approx(widest_zn_o)
    assert cutoff == pytest.approx(widest_zn_o + 0.75)
    # Strictly wider than the exact Zn-Asp cutoff, which is the point of "largest".
    exact_asp = ba.LIT[("ASP", "O", "ZN")][0]
    assert cutoff > exact_asp + 0.75


def test_first_sphere_targets_index_is_the_maximum_per_metal_and_donor_element():
    """``FIRST_SPHERE_TARGETS`` is exactly ``max`` over the reference table."""
    expected = {}
    for (_residue, atom, metal), (mu, _sd) in ba.LIT.items():
        expected[(metal, atom)] = max(mu, expected.get((metal, atom), -math.inf))
    assert ba.FIRST_SPHERE_TARGETS == expected


def test_a_metal_donor_pair_with_no_reference_is_never_inferred(tmp_path):
    """No target of any kind means candidate evidence only, plus a partial code.

    The table defines no K-S distance, so a potassium/cysteine contact has no
    sphere to belong to. It must stay a candidate with NaN geometry and the
    entry must report ``missing_first_sphere_reference``.
    """
    path = _probe_structure(tmp_path, "k_sg.pdb", resname="CYS", metal="K",
                            positions={"SG": (2.5, 0.0, 0.0)})
    rows, candidates, _, metadata = _analyze(path)

    candidate = _only(candidates, "SG")
    assert candidate["eligibility_status"] == "missing_assignment_reference"
    assert candidate["eligibility_reason"] == "no_metal_donor_assignment_reference"
    assert candidate["first_sphere_eligible"] is False
    assert candidate["assignment_reference_kind"] == "missing"
    assert math.isnan(candidate["assignment_target"])
    assert math.isnan(candidate["first_sphere_cutoff"])
    assert rows == []
    assert "missing_first_sphere_reference" in metadata["partial_reason_codes"]
    assert any("K-S" in message for message in metadata["messages"])


@pytest.mark.parametrize("distance,eligible", [(2.780, True), (2.781, False)])
def test_first_sphere_membership_at_the_exact_cutoff_boundary(tmp_path, distance,
                                                              eligible):
    """His NE2 at Zn: 2.030 + 0.750 = 2.780 A is inside, 2.781 A is outside."""
    path = _probe_structure(tmp_path, f"bound{distance}.pdb", resname="HIS",
                            positions={"NE2": (distance, 0.0, 0.0)})
    rows, candidates, _, _ = _analyze(path)

    candidate = _only(candidates, "NE2")
    assert candidate["candidate_distance"] == pytest.approx(distance)
    assert candidate["assignment_target"] == pytest.approx(2.03)
    assert candidate["first_sphere_cutoff"] == pytest.approx(2.78)
    assert candidate["first_sphere_eligible"] is eligible
    assert candidate["eligibility_status"] == (
        "first_sphere_eligible" if eligible else "outside_first_sphere")
    assert candidate["eligibility_reason"] == (
        "distance_within_target_plus_0.75" if eligible
        else "distance_exceeds_target_plus_0.75")
    assert len(rows) == (1 if eligible else 0)


def test_dpi_never_widens_the_chemical_cutoff(tmp_path):
    """A huge coordinate uncertainty must not admit an out-of-sphere contact.

    README: "DPI never expands the chemical cutoff." The same His NE2 at 2.781 A
    is analyzed with no DPI at all and with a deliberately terrible one (few
    reflections, high R-free); the cutoff and the verdict must be identical.
    """
    path = _probe_structure(tmp_path, "dpi_width.pdb", resname="HIS",
                            positions={"NE2": (2.781, 0.0, 0.0)})
    bad_dpi = _dpi_metadata(tmp_path, nrefcnt=200, rffin=0.45)

    no_dpi_rows, no_dpi_candidates, _, _ = _analyze(path)
    dpi_rows, dpi_candidates, summaries, _ = _analyze(
        path, data_json=bad_dpi, resolution=3.5)

    dpi_value = next(iter(summaries.values()))["dpi"]
    assert math.isfinite(dpi_value) and dpi_value > 0.5, (
        "the test needs a DPI far larger than the 0.75 A tolerance to be "
        "meaningful")

    for candidates in (no_dpi_candidates, dpi_candidates):
        candidate = _only(candidates, "NE2")
        assert candidate["first_sphere_cutoff"] == pytest.approx(2.78)
        assert candidate["assignment_tolerance"] == pytest.approx(0.75)
        assert candidate["first_sphere_eligible"] is False
    assert no_dpi_rows == [] and dpi_rows == []


def test_the_broad_search_radius_is_not_a_bond_cutoff(tmp_path):
    """A 3.5 A Zn-O contact is discovered but not assigned.

    Discovery uses 4 A; assignment uses the Harding cutoff. Conflating the two
    would turn every second-shell water into a bond.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_water(101, (3.5, 0.0, 0.0), chain="B")
    builder.add_water(102, (0.0, 2.09, 0.0), chain="B")
    path = builder.write_pdb(tmp_path / "second_shell.pdb")
    rows, candidates, summaries, _ = _analyze(path)

    assert ba.CUTOFF == 4.0
    distances = sorted(round(c["candidate_distance"], 3) for c in candidates)
    assert distances == [2.09, 3.5]
    assert [round(row["distance"], 3) for row in rows] == [2.09]
    summary = next(iter(summaries.values()))
    assert summary["candidate_contact_count"] == 1


# --------------------------------------------------------------------------- #
# 4. The z-score
# --------------------------------------------------------------------------- #
def test_zscore_matches_the_documented_formula():
    """z = (d - mu) / sqrt(DPI^2 + sigma^2), rounded to four decimals."""
    assert ba._zscore(2.30, 2.09, 0.05, 0.12) == pytest.approx(
        round((2.30 - 2.09) / math.sqrt(0.12 ** 2 + 0.05 ** 2), 4))
    # 0.12 / 0.05 / 0.13 is a right triangle, so this one is exact by hand.
    assert ba._zscore(2.87, 2.09, 0.05, 0.12) == pytest.approx(6.0)
    assert ba._zscore(1.31, 2.09, 0.05, 0.12) == pytest.approx(-6.0)
    assert ba._zscore(2.09, 2.09, 0.05, 0.12) == pytest.approx(0.0)


def test_zscore_denominator_carries_exactly_one_dpi():
    """Only one DPI enters the denominator, not ``sqrt(2) * DPI``.

    README calls this deliberate: the metal is assumed far better ordered than
    the donor, so a single DPI stands for the donor alone. The two-atom
    treatment would shrink this z-score by ~29% and move it across the outlier
    cutoff, so the difference is asserted, not just the value.
    """
    dist, mu, sigma, dpi = 2.87, 2.09, 0.05, 0.12
    single = (dist - mu) / math.sqrt(dpi ** 2 + sigma ** 2)
    two_atom = (dist - mu) / math.sqrt(2 * dpi ** 2 + sigma ** 2)

    assert ba._zscore(dist, mu, sigma, dpi) == pytest.approx(round(single, 4))
    assert single == pytest.approx(6.0)
    assert two_atom < ba.ZSCORE_OUTLIER_CUTOFF < single
    assert ba._zscore(dist, mu, sigma, dpi) != pytest.approx(round(two_atom, 4))


@pytest.mark.parametrize("mu,stdev,dpi", [
    (2.09, 0.05, float("nan")),   # DPI unavailable
    (float("nan"), 0.05, 0.12),   # no literature mean
    (2.09, float("nan"), 0.12),   # no literature spread
])
def test_zscore_propagates_missing_inputs_as_nan(mu, stdev, dpi):
    """Any missing input yields NaN rather than a fabricated number."""
    assert math.isnan(ba._zscore(2.30, mu, stdev, dpi))


def test_zscore_is_nan_when_the_denominator_vanishes():
    """A zero spread and a zero DPI give no scale, so no z-score."""
    assert math.isnan(ba._zscore(2.30, 2.09, 0.0, 0.0))
    # A non-zero spread alone is still a usable scale.
    assert ba._zscore(2.14, 2.09, 0.05, 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("distance,expected_z,outlier", [
    (2.870, 6.0, True),      # exactly at the cutoff -> outlier
    (2.869, 5.9923, False),  # one thousandth inside -> consistent
    (1.310, -6.0, True),     # the negative boundary is symmetric
    (1.311, -5.9923, False),
])
def test_outlier_flag_switches_at_absolute_z_of_six(tmp_path, distance,
                                                    expected_z, outlier):
    """``|Zbond| >= 6`` is an inclusive threshold, and it is symmetric.

    DPI 0.12 against the Zn-water spread of 0.05 gives a denominator of exactly
    0.13, so the boundary distances are hand-computable.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_water(101, (2.09, 0.0, 0.0), chain="B")
    context = load_structure("test", builder.write_pdb(tmp_path / "w.pdb"))
    water = next(residue for residue in context.residues if residue.is_water)
    contact = {"neighbor": water.contact_atoms[0], "distance_raw": distance}

    ba._annotate_contacts([contact], "ZN", 0.12)

    assert ba.ZSCORE_OUTLIER_CUTOFF == 6.0
    assert contact["literature_distance"] == pytest.approx(2.09)
    assert contact["literature_stdev"] == pytest.approx(0.05)
    assert contact["zscore"] == pytest.approx(expected_z)
    assert contact["reference_covered"] is True
    assert contact["geometry_outlier"] is outlier
    assert contact["geometry_consistent"] is (not outlier)


def test_end_to_end_zscore_uses_the_row_dpi_and_the_bundled_reference(tmp_path):
    """A pipeline row's z-score equals the formula applied to its own dpi column.

    The expected value is recomputed from ``(distance, mu, sigma, dpi)`` taken
    off the row, so the test checks the arithmetic rather than recording it.
    """
    path = _probe_structure(tmp_path, "zscore.pdb", resname="ASP",
                            positions={"OD1": (2.40, 0.0, 0.0)})
    rows, _, _, metadata = _analyze(path, data_json=_dpi_metadata(tmp_path))

    assert metadata["partial_reason_codes"] == []
    row = _only(rows, "OD1")
    mu, sigma = ba.LIT[("ASP", "O", "ZN")]
    assert (mu, sigma) == (1.99, 0.05)
    assert row["literature_distance"] == pytest.approx(mu)
    assert row["literature_stdev"] == pytest.approx(sigma)
    assert math.isfinite(row["dpi"]) and row["dpi"] > 0

    expected = round((row["distance"] - mu)
                     / math.sqrt(row["dpi"] ** 2 + sigma ** 2), 4)
    assert row["zscore"] == pytest.approx(expected)
    assert expected > ba.ZSCORE_OUTLIER_CUTOFF
    assert row["geometry_outlier"] is True
    assert row["geometry_consistent"] is False
    assert row["zscore_outlier_cutoff"] == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# 5. The DPI (Blow 2002 eq. 7)
#
# DPI = 1.28 * ni**(1/2) * va**(1/3) * nobs**(-5/6) * rfree
#
# It is the denominator of every Zbond, so a transposed exponent moves every
# bond in the database across the |Z| >= 6 boundary while leaving every
# "isfinite(dpi) and dpi > 0" assertion happy. The tests below therefore pin the
# value against inputs chosen so the arithmetic is exact by hand, then pin each
# exponent separately by measuring how DPI responds to one input at a time.
# --------------------------------------------------------------------------- #
#: Base inputs for the scaling tests. Every term is exact: ni = 16 (sqrt 4),
#: va = 200**3 A^3 (cube root 200), nobs = 2**12 (so nobs**(-5/6) = 2**-10),
#: rfree = 0.25. DPI = 1.28 * 4 * 200 * 2**-10 * 0.25 = 0.25 A exactly.
DPI_BASE = {"atom_count": 16, "cell_edge": 200.0, "nrefcnt": 4096, "rffin": 0.25}
DPI_BASE_VALUE = 0.25

#: ``_calculate_dpi_details`` rounds to four decimals, so any expected value is
#: only ever guaranteed to half a unit in the last place.
DPI_ROUNDING = 5e-5


def _atom_count_structure(tmp_path, name, atom_count, *, cell_edge=60.0,
                          spacegroup="P 1", occupancy=1.0, donor_distance=None):
    """A Zn plus waters totalling exactly ``atom_count`` non-hydrogen atoms.

    Every atom carries ``occupancy``, so ``count_ni`` -- the occupancy-weighted
    non-hydrogen count that feeds the DPI -- is ``atom_count * occupancy`` and
    is known to the test without asking ``src`` for it. The padding waters sit
    on a 3 A lattice starting 6 A out, well outside the 4 A search radius, so
    they contribute to ``ni`` without inventing coordination. ``P 1`` in a cubic
    cell makes the ASU volume the cell volume exactly, with no images to place.

    ``donor_distance`` puts one further water that far from the metal, which is
    the only contact the analysis can find.
    """
    assert atom_count >= 1
    builder = StructureBuilder(
        cell=(cell_edge, cell_edge, cell_edge, 90.0, 90.0, 90.0),
        spacegroup=spacegroup)
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0),
                      occupancy=occupancy)
    remaining = atom_count - 1
    if donor_distance is not None:
        assert remaining >= 1
        builder.add_water(100, (float(donor_distance), 0.0, 0.0), chain="B",
                          occupancy=occupancy)
        remaining -= 1
    for index in range(remaining):
        builder.add_water(200 + index,
                          (6.0 + 3.0 * (index % 8),
                           3.0 * ((index // 8) % 8),
                           3.0 * (index // 64)),
                          chain="B", occupancy=occupancy)
    return builder.write_pdb(tmp_path / name)


class _DpiRun:
    """What one ``_calculate_dpi_details`` call needed and produced."""

    def __init__(self, path, context, result):
        self.path = path
        self.context = context
        self.dpi, self.resolution, self.reason = result


def _dpi_details(tmp_path, *, atom_count, cell_edge, nrefcnt, rffin,
                 spacegroup="P 1", resolution=1.50, occupancy=1.0,
                 pdb_path=None, mtz_path=None, name="dpi"):
    """``_calculate_dpi_details`` over a structure with known ni and va.

    ``pdb_path``/``mtz_path`` default to the written structure and to a path
    that does not exist, which is what makes ``_asu_volume`` fall back to the
    coordinate file's CRYST1 record.
    """
    path = _atom_count_structure(tmp_path, f"{name}.pdb", atom_count,
                                 cell_edge=cell_edge, spacegroup=spacegroup,
                                 occupancy=occupancy)
    context = load_structure("test", path)
    inputs = helpers.dpi_inputs(
        pdb_path=path if pdb_path is None else pdb_path,
        mtz_path=(os.path.join(str(tmp_path), "absent.mtz")
                  if mtz_path is None else mtz_path),
        data_json=helpers.write_data_json(tmp_path / f"{name}.json",
                                          nrefcnt=nrefcnt, rffin=rffin),
        resolution=resolution)
    return _DpiRun(path, context, ba._calculate_dpi_details(context, inputs))


def _dpi_value(tmp_path, **kwargs):
    """Just the DPI, asserting the calculation reported no reason code."""
    run = _dpi_details(tmp_path, **kwargs)
    assert run.reason == "", f"unexpected DPI reason code {run.reason!r}"
    return run.dpi


def test_dpi_equals_the_hand_computed_blow_value(tmp_path):
    """The DPI value itself, not merely its finiteness, is pinned.

    Inputs are chosen so every factor is exact:

        ni    = 16          -> ni**(1/2)   = 4
        va    = 100**3      -> va**(1/3)   = 100
        nobs  = 4096 = 2**12 -> nobs**(-5/6) = 2**-10 = 1/1024
        rfree = 0.25

        DPI = 1.28 * 4 * 100 * (1/1024) * 0.25 = 0.125 A

    Any transposition -- ni**(1/3) with va**(1/2), -5/6 written as -6/5 or
    -5/8, a lost 1.28 -- lands somewhere else entirely.
    """
    run = _dpi_details(tmp_path, atom_count=16, cell_edge=100.0, nrefcnt=4096,
                       rffin=0.25, resolution=1.42)

    # The two inputs the test claims to know, confirmed against the model.
    assert count_ni(run.context) == pytest.approx(16.0)
    assert ba._asu_volume(os.path.join(str(tmp_path), "absent.mtz"),
                          run.path) == pytest.approx(100.0 ** 3)

    assert run.reason == ""
    assert run.dpi == pytest.approx(0.125, abs=DPI_ROUNDING)
    # Resolution is metadata only: it is passed through, never multiplied in.
    assert run.resolution == pytest.approx(1.42)


def test_dpi_at_realistic_crystallographic_magnitudes(tmp_path):
    """The same oracle with numbers a real entry could carry.

        ni    = 400 atoms            -> 20
        va    = 216000 A^3 (60 A)**3 -> 60
        nobs  = 46656 = 6**6         -> 6**-5 = 1/7776
        rfree = 0.243

        DPI = 1.28 * 20 * 60 * 0.243 / 7776 = 0.048 A

    0.048 A is a plausible coordinate precision for a ~1.9 A structure, and it
    is the same order as the literature spreads it is added to in quadrature --
    which is the whole reason the exponents have to be right.
    """
    run = _dpi_details(tmp_path, atom_count=400, cell_edge=60.0, nrefcnt=46656,
                       rffin=0.243, resolution=1.90)

    assert count_ni(run.context) == pytest.approx(400.0)
    assert ba._asu_volume(os.path.join(str(tmp_path), "absent.mtz"),
                          run.path) == pytest.approx(216000.0)
    assert run.reason == ""
    assert run.dpi == pytest.approx(0.048, abs=DPI_ROUNDING)
    assert run.resolution == pytest.approx(1.90)


@pytest.mark.parametrize("override,expected", [
    ({}, DPI_BASE_VALUE),                            # the base case itself
    ({"atom_count": 4}, 0.125),                      # ni / 4  -> DPI / 2
    ({"atom_count": 64}, 0.5),                       # ni * 4  -> DPI * 2
    ({"cell_edge": 100.0}, 0.125),                   # va / 8  -> DPI / 2
    ({"cell_edge": 400.0}, 0.5),                     # va * 8  -> DPI * 2
    ({"nrefcnt": 64}, 8.0),                          # nobs / 64 -> DPI * 32
    ({"nrefcnt": 262144}, 0.0078125),                # nobs * 64 -> DPI / 32
    ({"nrefcnt": 8192}, 0.25 * 2 ** (-5 / 6)),       # nobs * 2  -> 2**(-5/6)
    ({"rffin": 0.125}, 0.125),                       # rfree / 2 -> DPI / 2
    ({"rffin": 0.5}, 0.5),                           # rfree * 2 -> DPI * 2
])
def test_each_dpi_input_moves_the_result_by_its_own_exponent(tmp_path, override,
                                                             expected):
    """One input at a time, against a value computed by hand from the base case.

    The base case is DPI = 0.25 A exactly (see ``DPI_BASE``). Multiplying ni by
    four must double the DPI and nothing else; multiplying the cell volume by
    eight must double it; multiplying the reflection count by 64 must divide it
    by exactly 32; R-free enters linearly. Swapping any two exponents breaks at
    least two of these rows.
    """
    inputs = dict(DPI_BASE, **override)
    dpi = _dpi_value(tmp_path, **inputs)
    assert dpi == pytest.approx(expected, abs=DPI_ROUNDING)


def test_the_dpi_exponents_are_pinned_individually(tmp_path):
    """Each exponent is measured from the response to its own input.

    ``log(DPI(k*x) / DPI(x)) / log(k)`` recovers the exponent on ``x``. The
    tolerance is far tighter than the gap between 1/2, 1/3, 5/6 and 1, so a
    transposition of any pair fails here even if someone also adjusts 1.28 to
    keep one case looking right.
    """
    base = _dpi_value(tmp_path, name="base", **DPI_BASE)
    assert base == pytest.approx(DPI_BASE_VALUE, abs=DPI_ROUNDING)

    measured = {}
    for label, override, factor in (
            ("ni", {"atom_count": DPI_BASE["atom_count"] * 4}, 4),
            ("va", {"cell_edge": DPI_BASE["cell_edge"] * 2}, 8),  # volume * 8
            ("nobs", {"nrefcnt": DPI_BASE["nrefcnt"] * 2}, 2),
            ("rfree", {"rffin": DPI_BASE["rffin"] * 2}, 2),
    ):
        scaled = _dpi_value(tmp_path, name=label, **dict(DPI_BASE, **override))
        measured[label] = math.log(scaled / base) / math.log(factor)

    # The nearest wrong values are 1/3 vs 1/2 (0.167 away), -5/8 or -6/5 vs
    # -5/6 (0.21 and 0.37 away) and 5/6 vs 1 (0.167 away): every one of them is
    # two orders of magnitude outside this tolerance.
    assert measured["ni"] == pytest.approx(0.5, abs=1e-3)
    assert measured["va"] == pytest.approx(1 / 3, abs=1e-3)
    assert measured["nobs"] == pytest.approx(-5 / 6, abs=1e-3)
    assert measured["rfree"] == pytest.approx(1.0, abs=1e-3)


def test_dpi_carries_the_1_28_coefficient(tmp_path):
    """The leading constant is 1.28 (Blow 2002 eq. 7), not 1.0 and not 1.2.

    ni = 16 -> 4, va = 100**3 -> 100, nobs = 4096 -> 1/1024 and rfree = 0.64
    multiply to exactly 4 * 100 / 1024 * 0.64 = 0.25, so the DPI reads the
    coefficient straight off: 1.28 * 0.25 = 0.32 A. A dropped coefficient would
    give 0.25 and a 1.2 typo 0.30.
    """
    dpi = _dpi_value(tmp_path, atom_count=16, cell_edge=100.0, nrefcnt=4096,
                     rffin=0.64)
    assert dpi == pytest.approx(0.32, abs=DPI_ROUNDING)


def test_dpi_is_rounded_to_four_decimals(tmp_path):
    """The stored DPI is the rounded value, so consumers see a stable number."""
    dpi = _dpi_value(tmp_path, **dict(DPI_BASE, nrefcnt=8192))
    exact = DPI_BASE_VALUE * 2 ** (-5 / 6)
    assert exact == pytest.approx(0.1403077564, abs=1e-9)
    assert dpi == round(exact, 4) == 0.1403


def test_the_bond_row_dpi_is_the_hand_computed_value(tmp_path):
    """The pipeline's ``dpi`` column, and the z-score built on it, are pinned.

    Same construction as the realistic case (ni = 400, va = 60**3, nobs = 6**6,
    R-free 0.243 -> DPI = 0.048 A), run through ``run_bond_analysis`` so the row
    and the site summary are checked, and the z-score of a Zn-water contact at
    2.20 A is computed against the *pinned* DPI rather than against whatever the
    row happens to carry.
    """
    path = _atom_count_structure(tmp_path, "row_dpi.pdb", 400, cell_edge=60.0,
                                 donor_distance=2.20)
    context = load_structure("test", path)
    inputs = helpers.dpi_inputs(
        pdb_path=path, mtz_path=os.path.join(str(tmp_path), "absent.mtz"),
        data_json=helpers.write_data_json(tmp_path / "row.json",
                                          nrefcnt=46656, rffin=0.243),
        resolution=1.90)
    rows, _candidates, summaries, metadata = ba.run_bond_analysis(
        "test", path, [], list(EDSTATS_HEADER), inputs, structure=context)

    assert metadata["partial_reason_codes"] == []
    row = _only(rows, "O")
    assert row["dpi"] == pytest.approx(0.048, abs=DPI_ROUNDING)
    assert row["resolution"] == pytest.approx(1.90)

    mu, sigma = ba.LIT[("HOH", "O", "ZN")]
    assert (mu, sigma) == (2.09, 0.05)
    expected = round((2.20 - mu) / math.sqrt(0.048 ** 2 + sigma ** 2), 4)
    assert expected == pytest.approx(1.5871, abs=1e-4)
    assert row["zscore"] == pytest.approx(expected, abs=1e-3)
    assert row["geometry_consistent"] is True

    summary = next(iter(summaries.values()))
    assert summary["dpi"] == pytest.approx(0.048, abs=DPI_ROUNDING)
    assert summary["occupancy_weighted_atom_count"] == pytest.approx(400.0)
    assert summary["dpi_atom_count_multiplier"] == 1
    assert summary["dpi_unavailable_reason"] == ""


def test_dpi_counts_atoms_by_occupancy_not_by_record(tmp_path):
    """``ni`` is occupancy-weighted, so partly occupied atoms count in part.

    The same 16 atoms at occupancy 0.25 give ni = 4, not 16, and the DPI is
    halved: sqrt(4/16) = 1/2. Counting records instead would overstate the
    precision of exactly the disordered sites that need it most.
    """
    full = _dpi_value(tmp_path, name="full", atom_count=16, cell_edge=100.0,
                      nrefcnt=4096, rffin=0.25)
    quarter = _dpi_value(tmp_path, name="quarter", atom_count=16,
                         cell_edge=100.0, nrefcnt=4096, rffin=0.25,
                         occupancy=0.25)

    assert full == pytest.approx(0.125, abs=DPI_ROUNDING)
    assert quarter == pytest.approx(0.0625, abs=DPI_ROUNDING)


# --------------------------------------------------------------------------- #
# 5b. The asymmetric-unit volume
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spacegroup,operations", [
    ("P 1", 1),
    ("P 21 21 21", 4),
    ("C 2", 4),          # 2 symmetry operations x 2 centring translations
    ("P 43 21 2", 8),
])
def test_asu_volume_is_the_cell_volume_over_the_operation_count(
        tmp_path, spacegroup, operations):
    """va = V(cell) / number of symmetry operations, centring included.

    Counting only the point-group operations of a centred lattice would inflate
    va by the centring multiplicity -- for ``C 2`` by a factor of two, worth
    26% on the DPI through the cube root -- so a centred group is included
    deliberately.
    """
    builder = StructureBuilder(cell=(100.0, 100.0, 100.0, 90.0, 90.0, 90.0),
                               spacegroup=spacegroup)
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    path = builder.write_pdb(tmp_path / "cell.pdb")

    volume = ba._asu_volume(os.path.join(str(tmp_path), "absent.mtz"), path)

    assert volume == pytest.approx(1.0e6 / operations)


def test_asu_volume_prefers_the_mtz_cell_over_the_coordinate_file(tmp_path):
    """The reflection file wins, because it is the cell the data were scaled on.

    The two files are given deliberately different cells and space groups, so
    the returned volume says unambiguously which one was used.
    """
    numpy = pytest.importorskip("numpy")   # only needed to give the MTZ data

    builder = StructureBuilder(cell=(100.0, 100.0, 100.0, 90.0, 90.0, 90.0),
                               spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    pdb_path = builder.write_pdb(tmp_path / "coords.pdb")

    mtz = gemmi.Mtz(with_base=True)
    mtz.cell = gemmi.UnitCell(40.0, 40.0, 40.0, 90.0, 90.0, 90.0)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 21 21 21")
    mtz.set_data(numpy.array([[0.0, 0.0, 1.0]], dtype="float32"))
    mtz_path = str(tmp_path / "data.mtz")
    mtz.write_to_file(mtz_path)

    assert ba._asu_volume(mtz_path, pdb_path) == pytest.approx(40.0 ** 3 / 4)
    # Without the MTZ the same call falls back to CRYST1, which is a different
    # number -- so the assertion above really did read the MTZ.
    assert ba._asu_volume(os.path.join(str(tmp_path), "absent.mtz"),
                          pdb_path) == pytest.approx(100.0 ** 3)


def test_asu_volume_falls_back_when_the_mtz_is_unreadable(tmp_path):
    """A missing or corrupt MTZ degrades to CRYST1 rather than to NaN.

    Alchemy is routinely run on coordinates alone; losing the DPI because the
    reflection file was absent would blank every z-score in the entry.
    """
    builder = StructureBuilder(cell=(100.0, 100.0, 100.0, 90.0, 90.0, 90.0),
                               spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    pdb_path = builder.write_pdb(tmp_path / "coords.pdb")
    corrupt = tmp_path / "corrupt.mtz"
    corrupt.write_text("this is not an MTZ file\n", encoding="utf-8")

    for mtz in (os.path.join(str(tmp_path), "absent.mtz"), str(corrupt)):
        assert ba._asu_volume(mtz, pdb_path) == pytest.approx(100.0 ** 3)


@pytest.mark.parametrize("content", [
    None,                                       # the file does not exist
    "this is not a coordinate file\n",          # unparseable
    "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00 20.00"
    "           O\nEND\n",                      # parses, but has no CRYST1
])
def test_asu_volume_is_nan_when_no_source_supplies_cell_and_symmetry(tmp_path,
                                                                     content):
    """No cell or no space group means no volume -- and NaN, not a guess.

    A fabricated volume would propagate silently into every DPI and every
    z-score of the entry; NaN makes ``_calculate_dpi_details`` report
    ``missing_or_invalid_asu_volume`` instead.
    """
    pdb_path = os.path.join(str(tmp_path), "missing.pdb")
    if content is not None:
        with open(pdb_path, "w", encoding="utf-8") as handle:
            handle.write(content)

    volume = ba._asu_volume(os.path.join(str(tmp_path), "absent.mtz"), pdb_path)

    assert math.isnan(volume)


# --------------------------------------------------------------------------- #
# 5c. The R-free header fallback
# --------------------------------------------------------------------------- #
def _remark_3(*lines):
    """Render REMARK 3 refinement-statistics lines as a PDB header fragment."""
    return "".join(f"REMARK   3   {line}\n" for line in lines)


def test_rfree_is_read_from_the_remark_3_header(tmp_path):
    """The final R-free is scraped when PDB-REDO's ``data.json`` has none."""
    path = tmp_path / "header.pdb"
    path.write_text(
        _remark_3("FIT TO DATA USED IN REFINEMENT.",
                  "R VALUE     (WORKING SET) : 0.17800",
                  "FREE R VALUE                     : 0.21530"),
        encoding="utf-8")

    assert ba._rfree_from_pdb(str(path)) == pytest.approx(0.2153)


@pytest.mark.parametrize("line", [
    "FREE R VALUE TEST SET SIZE   (%) : 5.100",
    "FREE R VALUE TEST SET COUNT      : 2500.0",
    "ESTIMATED ERROR OF FREE R VALUE  : 0.008",
    "BIN FREE R VALUE                 : 0.2810",
    "FREE R VALUE (NO CUTOFF) BIN     : 0.3100",
])
def test_rfree_ignores_test_set_estimate_and_per_bin_lines(tmp_path, line):
    """Only the overall final R-free counts.

    A REMARK 3 block carries several near-identical lines. Picking up the test
    set *size* (5.1) or a per-bin value instead of the final R-free would
    inflate or deflate every DPI in the entry, and 5.1 would do it by a factor
    of 25.
    """
    path = tmp_path / "header.pdb"
    path.write_text(_remark_3(line, "FREE R VALUE : 0.21530"), encoding="utf-8")
    assert ba._rfree_from_pdb(str(path)) == pytest.approx(0.2153)

    decoy_only = tmp_path / "decoy.pdb"
    decoy_only.write_text(_remark_3(line), encoding="utf-8")
    assert math.isnan(ba._rfree_from_pdb(str(decoy_only)))


def test_rfree_takes_the_first_matching_line(tmp_path):
    """Deterministic when a file repeats the record: the first one wins."""
    path = tmp_path / "twice.pdb"
    path.write_text(_remark_3("FREE R VALUE : 0.21530",
                              "FREE R VALUE : 0.29900"), encoding="utf-8")
    assert ba._rfree_from_pdb(str(path)) == pytest.approx(0.2153)


@pytest.mark.parametrize("content", [
    None,                                             # no such file
    "",                                               # empty file
    "REMARK   2 RESOLUTION.    1.50 ANGSTROMS.\n",    # no refinement remarks
    "REMARK   3   FREE R VALUE : NULL\n",             # present but not a number
    "REMARK   3   FREE R VALUE : 0\n",                # no decimal point
    "REMARK   3   FREE R VALUE : .215\n",             # no integer part
])
def test_rfree_is_nan_when_absent_or_malformed(tmp_path, content):
    """A header that does not state a usable R-free yields NaN, never 0.0.

    0.0 would sail through as a number and produce a DPI of exactly zero, which
    makes every z-score infinite. NaN is what turns into
    ``missing_or_invalid_rfree`` instead. The two malformed spellings are the
    ones the scrape's ``\\d+\\.\\d+`` pattern deliberately does not accept.
    """
    path = os.path.join(str(tmp_path), "rfree.pdb")
    if content is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    assert math.isnan(ba._rfree_from_pdb(path))


def test_rfree_from_the_header_is_used_when_data_json_omits_it(tmp_path):
    """The two R-free sources are wired together, not merely both present.

    ``data.json`` without ``RFFIN`` must fall back to the coordinate header; the
    DPI that comes out is the hand-computed one for the header's value.
    """
    path = _atom_count_structure(tmp_path, "header_rfree.pdb", 16,
                                 cell_edge=100.0)
    # The scrape is a plain text scan, so where in the file the remark sits
    # does not matter; gemmi ignores it when it re-reads the cell.
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(_remark_3("FREE R VALUE : 0.25000"))
    context = load_structure("test", path)
    inputs = helpers.dpi_inputs(
        pdb_path=path, mtz_path=os.path.join(str(tmp_path), "absent.mtz"),
        data_json=helpers.write_data_json(tmp_path / "no_rffin.json",
                                          nrefcnt=4096, rffin=None))

    dpi, _resolution, reason = ba._calculate_dpi_details(context, inputs)

    assert reason == ""
    assert dpi == pytest.approx(0.125, abs=DPI_ROUNDING)


# --------------------------------------------------------------------------- #
# 5d. Why a site has no DPI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nrefcnt", [
    None,   # NREFCNT absent from data.json
    "",     # present but empty
    0,      # a count of zero is not a usable denominator
    -1,     # nor a negative one
])
def test_missing_reflection_count_is_named(tmp_path, nrefcnt):
    """``NREFCNT`` missing or unusable is reported as such, with every other
    term of the formula available."""
    run = _dpi_details(tmp_path, atom_count=16, cell_edge=100.0,
                       nrefcnt=nrefcnt, rffin=0.20)

    assert math.isnan(run.dpi)
    assert run.reason == "missing_or_invalid_reflection_count"
    # Not a mislabelled occupancy problem: the model itself is sound.
    assert run.context.occupancy_validation_failed is False
    assert count_ni(run.context) == pytest.approx(16.0)


def test_an_unreadable_data_json_reports_the_reflection_count(tmp_path):
    """A corrupt or absent metadata file is not a crash and not a silent zero."""
    path = _atom_count_structure(tmp_path, "site.pdb", 16, cell_edge=100.0)
    context = load_structure("test", path)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    for data_json in (str(broken), os.path.join(str(tmp_path), "absent.json")):
        inputs = helpers.dpi_inputs(
            pdb_path=path, mtz_path=os.path.join(str(tmp_path), "absent.mtz"),
            data_json=data_json)
        dpi, _resolution, reason = ba._calculate_dpi_details(context, inputs)
        assert math.isnan(dpi)
        assert reason == "missing_or_invalid_reflection_count"


@pytest.mark.parametrize("rffin", [
    None,   # RFFIN absent, and the coordinate file carries no REMARK 3 either
    "",     # present but empty
    0.0,    # a refinement never reaches R-free 0; treat it as missing
    -0.1,
])
def test_missing_rfree_is_named(tmp_path, rffin):
    """``NREFCNT`` alone, with no ``RFFIN`` and no REMARK 3 to fall back on."""
    run = _dpi_details(tmp_path, atom_count=16, cell_edge=100.0, nrefcnt=4096,
                       rffin=rffin)

    assert math.isnan(run.dpi)
    assert run.reason == "missing_or_invalid_rfree"


def test_missing_asu_volume_is_named(tmp_path):
    """Neither the MTZ nor the coordinate file yields a cell and a space group."""
    run = _dpi_details(tmp_path, atom_count=16, cell_edge=100.0, nrefcnt=4096,
                       rffin=0.20,
                       pdb_path=os.path.join(str(tmp_path), "absent.pdb"))

    assert math.isnan(run.dpi)
    assert run.reason == "missing_or_invalid_asu_volume"


def test_invalid_atom_count_is_named(tmp_path):
    """Every atom at zero occupancy weighs nothing, so ``ni`` is zero.

    ``ni = 0`` would make the DPI zero and every z-score infinite, so the
    calculation refuses and says which term failed. The occupancies are
    individually valid here, which is what separates this code from
    ``invalid_occupancy``.
    """
    run = _dpi_details(tmp_path, atom_count=16, cell_edge=100.0, nrefcnt=4096,
                       rffin=0.20, occupancy=0.0)

    assert math.isnan(run.dpi)
    assert run.reason == "invalid_dpi_atom_count"
    assert run.context.occupancy_validation_failed is False
    assert count_ni(run.context) == pytest.approx(0.0)


def test_the_missing_terms_are_reported_in_a_fixed_order(tmp_path):
    """With several terms missing at once the reason names the first of them.

    The order is fixed (reflection count, R-free, ASU volume, atom count) so a
    consumer aggregating reason codes over a batch sees a stable distribution
    rather than one that shifts with unrelated metadata.
    """
    absent_pdb = os.path.join(str(tmp_path), "absent.pdb")

    everything = _dpi_details(tmp_path, atom_count=16, cell_edge=100.0,
                              nrefcnt=None, rffin=None, pdb_path=absent_pdb)
    assert everything.reason == "missing_or_invalid_reflection_count"

    no_rfree = _dpi_details(tmp_path, atom_count=16, cell_edge=100.0,
                            nrefcnt=4096, rffin=None, pdb_path=absent_pdb,
                            name="second")
    assert no_rfree.reason == "missing_or_invalid_rfree"


def test_a_non_numeric_reflection_count_is_a_calculation_failure(tmp_path):
    """Garbage in ``data.json`` is caught and labelled, never raised.

    ``float("many")`` raises inside the calculation; the caller must still get a
    row, so the exception becomes ``dpi_calculation_failed``.
    """
    path = _atom_count_structure(tmp_path, "site.pdb", 16, cell_edge=100.0)
    context = load_structure("test", path)
    data_json = helpers.write_data_json(tmp_path / "bad.json", nrefcnt="many",
                                        rffin=0.20)
    inputs = helpers.dpi_inputs(
        pdb_path=path, mtz_path=os.path.join(str(tmp_path), "absent.mtz"),
        data_json=data_json)

    dpi, resolution, reason = ba._calculate_dpi_details(context, inputs)

    assert math.isnan(dpi)
    assert reason == "dpi_calculation_failed"
    assert resolution == pytest.approx(1.50)   # metadata survives the failure


def test_a_non_numeric_asu_volume_is_reported_as_invalid_metadata(tmp_path,
                                                                  monkeypatch):
    """The ``invalid_dpi_metadata`` guard keeps its own reason code.

    No real input reaches this branch -- ``nobs`` and ``rfree`` have been
    through ``float()`` by then and ``_asu_volume`` returns a float or NaN -- so
    the only way to exercise it is to make ``_asu_volume`` hand back something
    that is not a number. It is pinned because the branch is the difference
    between "the metadata was the wrong type" and the catch-all
    ``dpi_calculation_failed``, and because a future ``_asu_volume`` that
    returns ``None`` on failure would land here.
    """
    path = _atom_count_structure(tmp_path, "site.pdb", 16, cell_edge=100.0)
    context = load_structure("test", path)
    monkeypatch.setattr(ba, "_asu_volume", lambda mtz, pdb: "1000000")
    inputs = helpers.dpi_inputs(
        pdb_path=path, mtz_path=os.path.join(str(tmp_path), "absent.mtz"),
        data_json=helpers.write_data_json(tmp_path / "meta.json",
                                          nrefcnt=4096, rffin=0.25))

    dpi, _resolution, reason = ba._calculate_dpi_details(context, inputs)

    assert math.isnan(dpi)
    assert reason == "invalid_dpi_metadata"


def test_every_dpi_reason_code_reaches_the_site_summary(tmp_path):
    """A reason code is not internal: it lands on the row's site summary.

    ``dpi_unavailable_reason`` is what tells a consumer why a site carries no
    z-score, so the code produced by ``_calculate_dpi_details`` must survive
    into ``run_bond_analysis``'s summary and its partial-reason list.
    """
    path = _atom_count_structure(tmp_path, "summary.pdb", 16, cell_edge=100.0,
                                 donor_distance=2.09)
    context = load_structure("test", path)
    inputs = helpers.dpi_inputs(
        pdb_path=path, mtz_path=os.path.join(str(tmp_path), "absent.mtz"),
        data_json=helpers.write_data_json(tmp_path / "no_nref.json",
                                          nrefcnt=None, rffin=0.20))

    rows, _candidates, summaries, metadata = ba.run_bond_analysis(
        "test", path, [], list(EDSTATS_HEADER), inputs, structure=context)

    summary = next(iter(summaries.values()))
    assert summary["dpi_unavailable_reason"] == (
        "missing_or_invalid_reflection_count")
    assert "missing_or_invalid_reflection_count" in (
        metadata["partial_reason_codes"])
    assert math.isnan(summary["dpi"])
    # The measured geometry is still reported; only the derived value is gone.
    row = _only(rows, "O")
    assert row["distance"] == pytest.approx(2.09, abs=1e-6)
    assert math.isnan(row["zscore"])


# --------------------------------------------------------------------------- #
# 6. Nullable geometry columns
# --------------------------------------------------------------------------- #
def test_unassessable_geometry_renders_blank_and_never_false(tmp_path):
    """Blank means "not assessed"; ``False`` would mean "assessed and passed".

    Two independent ways of losing the z-score are covered: no exact literature
    reference (Lys NZ is admitted by the element fallback) and no DPI.
    """
    fallback = _probe_structure(tmp_path, "lys.pdb", resname="LYS",
                                positions={"NZ": (2.10, 0.0, 0.0)})
    fallback_rows, fallback_candidates, _, _ = _analyze(
        fallback, data_json=_dpi_metadata(tmp_path))

    row = _only(fallback_rows, "NZ")
    assert _only(fallback_candidates, "NZ")["assignment_reference_kind"] == (
        "element_fallback")
    assert row["coordination_status"] == "inferred"   # it is still a bond
    assert row["reference_covered"] is False
    assert math.isfinite(row["dpi"])                  # DPI is fine; the ref is not
    assert math.isnan(row["zscore"])
    for column in ("geometry_outlier", "geometry_consistent"):
        assert row[column] == ""
        assert row[column] is not False and row[column] is not True
    assert row["score_eligible"] is False
    assert row["score_exclusion_reason"] == "zscore_unavailable"

    covered = _probe_structure(tmp_path, "asp_nodpi.pdb", resname="ASP",
                               positions={"OD1": (1.99, 0.0, 0.0)})
    no_dpi_rows, _, _, metadata = _analyze(covered)

    row = _only(no_dpi_rows, "OD1")
    assert metadata["partial_reason_codes"] == ["missing_dpi_metadata_source"]
    assert row["reference_covered"] is True            # the reference exists
    assert math.isnan(row["dpi"]) and math.isnan(row["zscore"])
    for column in ("geometry_outlier", "geometry_consistent"):
        assert row[column] == ""
        assert row[column] is not False and row[column] is not True
    # Distance is still measured and reported.
    assert row["distance"] == pytest.approx(1.99, abs=1e-6)


def test_assessed_geometry_columns_are_complementary_booleans(tmp_path):
    """When geometry is assessed the two flags are real, opposite booleans."""
    path = _probe_structure(tmp_path, "assessed.pdb", resname="ASP",
                            positions={"OD1": (1.99, 0.0, 0.0)})
    rows, _, summaries, _ = _analyze(path, data_json=_dpi_metadata(tmp_path))

    row = _only(rows, "OD1")
    assert isinstance(row["geometry_outlier"], bool)
    assert isinstance(row["geometry_consistent"], bool)
    assert row["geometry_outlier"] is not row["geometry_consistent"]
    assert row["score_eligible"] is True
    assert row["score_exclusion_reason"] == ""
    summary = next(iter(summaries.values()))
    assert summary["score_eligible_contact_count"] == 1
    assert summary["score_excluded_contact_count"] == 0
    assert summary["image_inclusive_geometry_status"] == "plausible"


# --------------------------------------------------------------------------- #
# 7. Multi-donor groups
# --------------------------------------------------------------------------- #
def _bidentate(tmp_path, name, d1, d2, *, data_json=None):
    """Asp chelating a Zn through both carboxylate oxygens."""
    path = _probe_structure(tmp_path, name, resname="ASP",
                            positions={"OD1": (d1, 0.0, 0.0),
                                       "OD2": (0.0, d2, 0.0)})
    return _analyze(path, data_json=data_json)


def test_a_single_donor_contact_is_not_a_multi_donor_group(tmp_path):
    """One contact per residue image: detected False, count 1, ``single_donor``."""
    path = _probe_structure(tmp_path, "mono.pdb", resname="HIS",
                            positions={"NE2": (2.03, 0.0, 0.0)})
    rows, _, summaries, _ = _analyze(path, data_json=_dpi_metadata(tmp_path))

    row = _only(rows, "NE2")
    assert row["multi_donor_detected"] is False
    assert row["multi_donor_contact_count"] == 1
    assert row["multi_donor_geometry_status"] == "single_donor"
    assert row["multi_donor_contains_suspect_bond"] is False
    summary = next(iter(summaries.values()))
    assert summary["multi_donor_residue_group_count"] == 0
    assert summary["multi_donor_contact_count"] == 0


def test_two_donors_of_one_residue_form_a_consistent_group(tmp_path):
    """Two well-behaved contacts to one residue image: detected, not suspect."""
    rows, _, summaries, _ = _bidentate(tmp_path, "chelate_ok.pdb", 1.99, 2.00,
                                       data_json=_dpi_metadata(tmp_path))

    assert len(rows) == 2
    for row in rows:
        assert row["multi_donor_detected"] is True
        assert row["multi_donor_contact_count"] == 2
        assert row["multi_donor_geometry_status"] == "consistent"
        assert row["multi_donor_contains_suspect_bond"] is False
        assert row["geometry_consistent"] is True
        assert row["context_warning_reasons"] == ""
    summary = next(iter(summaries.values()))
    assert summary["multi_donor_residue_group_count"] == 1
    assert summary["multi_donor_contact_count"] == 2
    assert summary["suspect_multi_donor_residue_group_count"] == 0
    assert summary["indeterminate_multi_donor_residue_group_count"] == 0


def test_one_outlier_makes_the_whole_group_suspect_but_only_it_an_outlier(tmp_path):
    """Group status spreads; the per-bond outlier flag does not.

    README: "If any member is an outlier, every member of the group records
    multi_donor_geometry_status=suspect ...; the particular unusual bonds retain
    geometry_outlier=true." Neither bond is excluded from scoring.
    """
    rows, _, summaries, _ = _bidentate(tmp_path, "chelate_bad.pdb", 1.99, 2.40,
                                       data_json=_dpi_metadata(tmp_path))

    assert len(rows) == 2
    good = _only(rows, "OD1")
    bad = _only(rows, "OD2")

    assert good["geometry_outlier"] is False and good["geometry_consistent"] is True
    assert bad["geometry_outlier"] is True and bad["geometry_consistent"] is False
    for row in (good, bad):
        assert row["multi_donor_detected"] is True
        assert row["multi_donor_contact_count"] == 2
        assert row["multi_donor_geometry_status"] == "suspect"
        assert row["multi_donor_contains_suspect_bond"] is True
        # Contextual only: nothing is downweighted or dropped.
        assert row["score_eligible"] is True
        assert row["score_exclusion_reason"] == ""
        assert row["context_warning"] is True
        assert row["context_warning_reasons"] == "suspect_multi_donor_group"

    summary = next(iter(summaries.values()))
    assert summary["suspect_multi_donor_residue_group_count"] == 1
    assert summary["scored_geometry_outlier_contact_count"] == 1
    assert summary["scored_geometry_consistent_contact_count"] == 1
    assert summary["image_inclusive_geometry_status"] == "suspect"


def test_a_group_with_no_assessable_member_is_indeterminate(tmp_path):
    """All z-scores missing and no outlier -> ``indeterminate``, not ``consistent``.

    The same chelating geometry that is "suspect" with a DPI must not silently
    become "consistent" when the DPI is unavailable.
    """
    rows, _, summaries, _ = _bidentate(tmp_path, "chelate_nodpi.pdb", 1.99, 2.40)

    assert len(rows) == 2
    for row in rows:
        assert math.isnan(row["zscore"])
        assert row["multi_donor_detected"] is True
        assert row["multi_donor_geometry_status"] == "indeterminate"
        assert row["multi_donor_contains_suspect_bond"] is False
        assert row["score_eligible"] is False
        assert row["score_exclusion_reason"] == "zscore_unavailable"
        assert row["context_warning_reasons"] == ""

    summary = next(iter(summaries.values()))
    assert summary["indeterminate_multi_donor_residue_group_count"] == 1
    assert summary["suspect_multi_donor_residue_group_count"] == 0
    assert summary["score_eligible_contact_count"] == 0
    assert summary["image_inclusive_geometry_status"] == "insufficient data"


def test_backbone_and_side_chain_contacts_share_one_residue_group(tmp_path):
    """Grouping is by residue image, so a backbone O joins its own side chain.

    README: "Because backbone atoms carry the same residue identity in the
    coordinate model, this grouping automatically includes backbone-side-chain
    combinations without a separate backbone rule."
    """
    path = _probe_structure(tmp_path, "backbone_group.pdb", resname="ASP",
                            positions={"OD1": (1.99, 0.0, 0.0),
                                       "O": (0.0, 2.07, 0.0)})
    rows, _, summaries, _ = _analyze(path, data_json=_dpi_metadata(tmp_path))

    assert {row["neighbor_atom"] for row in rows} == {"OD1", "O"}
    for row in rows:
        assert row["multi_donor_detected"] is True
        assert row["multi_donor_contact_count"] == 2
        assert row["neighbor_resnum"] == "11"
    # The two atoms are scored against different references all the same.
    assert _only(rows, "OD1")["literature_distance"] == pytest.approx(1.99)
    assert _only(rows, "O")["literature_distance"] == pytest.approx(2.07)
    assert next(iter(summaries.values()))["multi_donor_residue_group_count"] == 1


def test_separate_residues_are_separate_groups(tmp_path):
    """Two donors from different residues are two single-donor groups."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_amino_acid("HIS", 10, chain="A", origin=(20.0, 20.0, 20.0),
                           positions={"NE2": (2.03, 0.0, 0.0)})
    builder.add_amino_acid("ASP", 11, chain="A", origin=(34.0, 20.0, 20.0),
                           positions={"OD1": (0.0, 1.99, 0.0)})
    path = builder.write_pdb(tmp_path / "two_residues.pdb")
    rows, _, summaries, _ = _analyze(path, data_json=_dpi_metadata(tmp_path))

    assert len(rows) == 2
    for row in rows:
        assert row["multi_donor_detected"] is False
        assert row["multi_donor_contact_count"] == 1
        assert row["multi_donor_geometry_status"] == "single_donor"
    summary = next(iter(summaries.values()))
    assert summary["multi_donor_residue_group_count"] == 0
    assert summary["candidate_contact_count"] == 2


def test_group_size_has_no_upper_limit(tmp_path):
    """A four-donor residue image reports all four; nothing is truncated."""
    path = _probe_structure(tmp_path, "tetra.pdb", resname="ASP",
                            positions={"OD1": (1.99, 0.0, 0.0),
                                       "OD2": (0.0, 1.99, 0.0),
                                       "O": (0.0, 0.0, 2.07),
                                       "N": (-2.03, 0.0, 0.0)},
                            probe_index=0)
    rows, _, summaries, _ = _analyze(path, data_json=_dpi_metadata(tmp_path))

    # N is allowed here because this residue is the polymer's first.
    assert {row["neighbor_atom"] for row in rows} == {"OD1", "OD2", "O", "N"}
    for row in rows:
        assert row["multi_donor_detected"] is True
        assert row["multi_donor_contact_count"] == 4
    assert next(iter(summaries.values()))["multi_donor_contact_count"] == 4


# --------------------------------------------------------------------------- #
# 8. The bundled reference table
# --------------------------------------------------------------------------- #
def test_reference_table_holds_exactly_the_expected_rows_and_keys():
    """Every bundled row is present, parsed strictly and reachable through ``LIT``.

    ``_load_literature`` silently drops any row whose numeric columns do not
    parse, so a corrupted ``CYS S CU`` line would simply vanish and demote
    Cu-thiolate bonds to the same-element fallback -- no exception, no failure,
    just a missing sigma and a missing z-score for every Cu-Cys site in the
    database. The count and the key set are therefore pinned against a parser
    that skips nothing.
    """
    records = _parse_reference_table(REFERENCE_TABLE)
    assert len(records) == EXPECTED_REFERENCE_ROW_COUNT

    keys = {(residue, atom, metal)
            for _lineno, residue, atom, metal, _mu, _sd in records}
    assert keys == REQUIRED_REFERENCE_KEYS
    # Named individually so a failure says which chemistry was lost.
    for key in (("CYS", "S", "CU"), ("CYS", "S", "ZN"), ("CYS", "S", "FE"),
                ("HIS", "N", "ZN"), ("HIS", "N", "NI"), ("HOH", "O", "MG"),
                ("ASP", "O", "CA"), ("CA", "O", "K")):
        assert key in keys, f"{key} is missing from {REFERENCE_TABLE}"

    # And the loader really exposes all of them, with the same numbers.
    assert len(ba.LIT) == EXPECTED_REFERENCE_ROW_COUNT
    assert set(ba.LIT) == keys
    assert {(residue, atom, metal): (mu, stdev)
            for _lineno, residue, atom, metal, mu, stdev in records} == ba.LIT


@pytest.mark.parametrize("corruption", [
    "CYS S CU 2.two 0.2",     # a typo in the mean
    "CYS S CU 2.20",          # a lost column
    "CYS S CU 2.20 0.2 0.3",  # a stray extra column
])
def test_the_strict_parser_rejects_a_corrupted_row(tmp_path, corruption):
    """The independence claim above is itself checked.

    ``_load_literature`` accepts all three of these files -- it just drops the
    bad line -- so if this test's parser did the same, the row-count and key
    assertions would be vacuous. A copy of the real table with one row damaged
    must fail to parse here while still loading cleanly in ``src``.
    """
    with open(REFERENCE_TABLE, encoding="utf-8") as handle:
        text = handle.read()
    damaged = tmp_path / "metal_distances_info.txt"
    damaged.write_text(text + corruption + "\n", encoding="utf-8")

    with pytest.raises(AssertionError):
        _parse_reference_table(str(damaged))

    # src, by contrast, accepts the damaged file without a word: the bad row is
    # either dropped or -- for the extra-column form -- silently overwrites the
    # real CYS/S/CU numbers. That is exactly why the parser above is strict.
    loaded = ba._load_literature(str(damaged))
    assert len(loaded) == EXPECTED_REFERENCE_ROW_COUNT


def test_reference_table_has_no_duplicate_keys():
    """A repeated (residue, atom, metal) key would be silently overwritten.

    ``_load_literature`` builds a dict, so a duplicate row is invisible at
    runtime; the raw file is re-parsed here to catch one.
    """
    records = _parse_reference_table(REFERENCE_TABLE)
    seen = {}
    duplicates = []
    for lineno, residue, atom, metal, _mu, _sd in records:
        key = (residue, atom, metal)
        if key in seen:
            duplicates.append((key, seen[key], lineno))
        seen[key] = lineno

    assert duplicates == []
    assert len(records) == len(ba.LIT)
    assert set(seen) == set(ba.LIT)


def test_reference_table_values_are_physically_plausible():
    """Every bundled distance is a positive, credible metal-ligand bond length.

    A negative or transposed column, a sigma of zero, or a distance outside
    roughly 1.5-3.5 A would silently corrupt every z-score for that pair.
    """
    records = _parse_reference_table(REFERENCE_TABLE)
    assert records, "the reference table must not be empty"

    for lineno, residue, atom, metal, mu, stdev in records:
        where = f"{REFERENCE_TABLE}:{lineno} ({residue} {atom} {metal})"
        assert residue in REFERENCE_RESIDUE_TOKENS, where
        assert atom in {"N", "O", "S"}, where
        assert metal in METAL_ELEMENTS, where
        assert 1.5 <= mu <= 3.5, f"{where}: implausible mean {mu}"
        assert 0.0 < stdev <= 0.5, f"{where}: implausible sigma {stdev}"
        # A spread that large relative to the mean would make |Z| meaningless.
        assert stdev < mu / 4.0, where
        # The 4 A discovery radius must never clip a first sphere.
        assert mu + ba.FIRST_SPHERE_TOLERANCE <= ba.CUTOFF, where

    parsed = {(residue, atom, metal): (mu, stdev)
              for _lineno, residue, atom, metal, mu, stdev in records}
    assert parsed == ba.LIT


def test_reference_table_is_internally_consistent_by_donor_element():
    """Chemistry cross-checks the numbers: S donors are longer than O donors.

    A metal-sulfur bond is longer than the same metal's oxygen bonds, and one
    metal's oxygen references should cluster rather than scatter. Both would
    break if a row's metal or donor column were transposed.
    """
    by_metal = defaultdict(lambda: defaultdict(list))
    for (_residue, atom, metal), (mu, _sd) in ba.LIT.items():
        by_metal[metal][atom].append(mu)

    assert set(by_metal) >= {"ZN", "CA", "MG", "K", "NA", "FE", "CU", "MN",
                             "CO", "NI"}
    for metal, by_atom in by_metal.items():
        oxygens = by_atom.get("O", [])
        assert oxygens, f"{metal} has no oxygen reference"
        assert max(oxygens) - min(oxygens) < 0.6, (
            f"{metal} oxygen references are implausibly scattered: {oxygens}")
        if "S" in by_atom:
            assert min(by_atom["S"]) > max(oxygens), (
                f"{metal}-S should exceed {metal}-O: "
                f"{by_atom['S']} vs {oxygens}")
        # Water is the reference every metal must define, since it is the most
        # common first-sphere donor in the PDB.
        assert ("HOH", "O", metal) in ba.LIT


def test_reference_table_ranks_ions_by_size():
    """Larger ions have longer water bonds: K > Na > Ca > Mn > Zn > Mg.

    This ordering is a basic property of ionic radius and catches a shuffled
    metal column that the per-row range checks would let through.
    """
    water = {metal: ba.LIT[("HOH", "O", metal)][0]
             for metal in ("K", "NA", "CA", "MN", "ZN", "MG")}
    assert (water["K"] > water["NA"] > water["CA"] > water["MN"] >
            water["ZN"] > water["MG"])
    assert water["ZN"] == pytest.approx(2.09)
    assert ba.LIT[("HIS", "N", "ZN")][0] == pytest.approx(2.03)
