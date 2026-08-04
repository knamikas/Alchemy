"""Which residues and atoms Alchemy will accept as a metal donor.

This is vocabulary, not policy: the tables say what is chemically typical, and
the rules that consult them live with the code that measures geometry. Two
paths consult them -- the 4 A proximity search in ``bond_analysis`` and the
``struct_conn``/``LINK`` resolution in ``declared_connections`` -- and both
apply the same two tests, so the tables cannot live in either module without
the other importing it. They are shared here for the same reason
``metal_elements`` is: a vocabulary every layer may read and none owns.

``AA`` and ``INFERRED_DONOR_ATOMS`` are checked against each other at import,
which is cheap and catches an edit to one that forgot the other.
"""

# Recognized amino-acid donors. Waters are recognized separately with Gemmi's
# Residue.is_water(), which also handles WAT, H2O, and DOD.
AA = frozenset(
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

# Typical protein donors that Alchemy may infer from geometry alone. Every
# standard amino acid can donate through its backbone carbonyl O. The listed
# additions are established side-chain donor atoms; uncommon but chemically
# possible atoms remain visible candidates and require a source declaration to
# become bonds. Polymer-terminal atoms are handled conditionally by the caller.
INFERRED_DONOR_ATOMS = {
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
if set(INFERRED_DONOR_ATOMS) != AA:
    raise ValueError("INFERRED_DONOR_ATOMS must cover every standard amino acid")

N_TERMINAL_DONOR_ATOMS = frozenset(("N",))
C_TERMINAL_DONOR_ATOMS = frozenset(("OXT", "OT1", "OT2"))

# Broad discovery retains all plausible donor elements. The atom-level table
# above, not this element set, controls geometry-only bond inference.
DONOR_ELEMENTS = frozenset(("N", "O", "S"))
