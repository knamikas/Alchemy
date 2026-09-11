"""Define donor residues and atoms used by coordination-analysis rules."""

# Waters are recognized separately with Gemmi's Residue.is_water(), which also
# handles WAT, H2O, and DOD.
AA: frozenset[str] = frozenset(
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

# Donors Alchemy may infer from geometry alone. An uncommon but chemically
# possible atom is omitted here: it stays a visible candidate and needs a
# source declaration to become a bond. Polymer-terminal atoms are handled
# conditionally by the caller.
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
if frozenset(INFERRED_DONOR_ATOMS) != AA:
    raise ValueError("INFERRED_DONOR_ATOMS must cover every standard amino acid")

N_TERMINAL_DONOR_ATOMS: frozenset[str] = frozenset(("N",))
C_TERMINAL_DONOR_ATOMS: frozenset[str] = frozenset(("OXT", "OT1", "OT2"))

# Discovery only. The atom-level table above, not this element set, controls
# geometry-only bond inference.
DONOR_ELEMENTS: frozenset[str] = frozenset(("N", "O", "S"))
