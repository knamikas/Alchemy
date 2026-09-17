"""Define donor residues and atoms, and classify a contact's donor residue.

The tables here drive geometry-only donor inference; ``neighbor_class`` is the
separate, coarser output classification published for every contact.
"""

from __future__ import annotations

import gemmi

from codes import NeighborClass
from structure_analysis import AtomSite

# Donors Alchemy may infer from geometry alone. Several uncommon but chemically
# possible nitrogen donors are omitted here, matching docs/method.md: amide
# nitrogens (ASN ND2, GLN NE2), the TRP pyrrole nitrogen (NE1), the ARG
# guanidinium nitrogens (NE, NH1, NH2), and internal peptide N. Each stays a
# visible candidate and needs a source declaration to become a bond.
# Polymer-terminal atoms are handled conditionally by the caller.
#
# This table and AA are keyed by upper-case, whitespace-stripped component
# names, which is the form ``ResidueIdentity`` normalizes every deposited
# component name to at load time, so callers look names up unchanged.
INFERRED_DONOR_ATOMS: dict[str, frozenset[str]] = {
    "ALA": frozenset(("O",)),
    "ARG": frozenset(("O",)),
    "ASN": frozenset(("O", "OD1")),
    "ASP": frozenset(("O", "OD1", "OD2")),
    "CYS": frozenset(("O", "SG")),
    "GLN": frozenset(("O", "OE1")),
    "GLU": frozenset(("O", "OE1", "OE2")),
    "GLY": frozenset(("O",)),
    "HIS": frozenset(("O", "ND1", "NE2")),
    "ILE": frozenset(("O",)),
    "LEU": frozenset(("O",)),
    "LYS": frozenset(("O", "NZ")),
    "MET": frozenset(("O", "SD")),
    "PHE": frozenset(("O",)),
    "PRO": frozenset(("O",)),
    "SER": frozenset(("O", "OG")),
    "THR": frozenset(("O", "OG1")),
    "TRP": frozenset(("O",)),
    "TYR": frozenset(("O", "OH")),
    "VAL": frozenset(("O",)),
}

# The standard amino acids the donor table covers, derived from it so the two
# cannot drift. Waters are recognized separately with Gemmi's Residue.is_water(),
# which also handles WAT, H2O, and DOD.
AA: frozenset[str] = frozenset(INFERRED_DONOR_ATOMS)

N_TERMINAL_DONOR_ATOMS: frozenset[str] = frozenset(("N",))
C_TERMINAL_DONOR_ATOMS: frozenset[str] = frozenset(("OXT", "OT1", "OT2"))

# Discovery only. The atom-level table above, not this element set, controls
# geometry-only bond inference.
DONOR_ELEMENTS: frozenset[str] = frozenset(("N", "O", "S"))


# Gemmi's component table is deliberately wider than AA/INFERRED_DONOR_ATOMS:
# MSE, SEP, PTR, and UNK all answer is_amino_acid() while the donor tables
# exclude them. An ``amino_acid`` class therefore never implies that the donor
# class is assessable; only the atom-level tables decide that.
def neighbor_class(neighbor: AtomSite) -> NeighborClass:
    """Classify a donor atom's residue coarsely for output.

    The water branch comes from the atom's cached ``is_water`` flag; every
    other class comes from Gemmi's tabulated component data.
    """
    if neighbor.is_water:
        return NeighborClass.WATER
    residue_info = gemmi.find_tabulated_residue(neighbor.residue_name)
    if residue_info.is_nucleic_acid():
        return NeighborClass.NUCLEOTIDE
    if residue_info.is_amino_acid():
        return NeighborClass.AMINO_ACID
    return NeighborClass.OTHER
