"""Donor tables and the coarse neighbour classification in ``donor_chemistry``."""

from __future__ import annotations

import pytest
from helpers import atom_site

from codes import NeighborClass
from coordination.donor_chemistry import (
    AA,
    DONOR_ELEMENTS,
    INFERRED_DONOR_ATOMS,
    neighbor_class,
)

#: The twenty standard amino acids the donor table must cover. Spelled out so a
#: residue dropped from ``INFERRED_DONOR_ATOMS`` cannot silently shrink ``AA``.
STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


def test_aa_covers_every_standard_amino_acid() -> None:
    """``AA`` derives from the donor table, which must still cover all twenty."""
    assert AA == STANDARD_AMINO_ACIDS
    assert frozenset(INFERRED_DONOR_ATOMS) == AA


def test_donor_elements_are_exactly_n_o_and_s() -> None:
    """Discovery accepts only these three elements; a fourth would widen output."""
    assert frozenset({"N", "O", "S"}) == DONOR_ELEMENTS


def test_unclassified_component_falls_back_to_other() -> None:
    """A component that is neither nucleotide nor amino acid classifies as other."""
    neighbor = atom_site("N", atom_name="N1", residue_name="IMD")
    assert neighbor_class(neighbor) is NeighborClass.OTHER


@pytest.mark.parametrize("residue_name", ["MSE", "SEP"])
def test_gemmi_amino_acids_outside_aa_still_classify_as_amino_acid(
    residue_name: str,
) -> None:
    """The output class is Gemmi's coarse one, wider than the donor tables.

    ``MSE`` and ``SEP`` answer ``is_amino_acid()`` but carry no donor-atom
    entry, so the class must not be read as a statement about donor support.
    """
    neighbor = atom_site("O", atom_name="O", residue_name=residue_name)
    assert residue_name not in AA
    assert residue_name not in INFERRED_DONOR_ATOMS
    assert neighbor_class(neighbor) is NeighborClass.AMINO_ACID


def test_water_flag_takes_precedence_over_the_component_table() -> None:
    """The cached ``is_water`` flag decides before any table lookup happens."""
    tabulated_water = atom_site("O", atom_name="O", residue_name="HOH", is_water=True)
    assert neighbor_class(tabulated_water) is NeighborClass.WATER

    # A name the table would call an amino acid still classifies as water when
    # the loader marked the residue as one.
    flagged_water = atom_site("O", atom_name="O", residue_name="GLU", is_water=True)
    assert neighbor_class(flagged_water) is NeighborClass.WATER
    assert neighbor_class(atom_site("O", atom_name="O", residue_name="GLU")) is (
        NeighborClass.AMINO_ACID
    )


@pytest.mark.parametrize("residue_name", ["DA", "A"])
def test_nucleotides_classify_as_nucleotide(residue_name: str) -> None:
    """Deoxy- and ribonucleotide components both take the nucleotide class."""
    neighbor = atom_site("N", atom_name="N7", residue_name=residue_name)
    assert neighbor_class(neighbor) is NeighborClass.NUCLEOTIDE
