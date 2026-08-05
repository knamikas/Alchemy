"""Metal element symbols shared by analysis and catalog maintenance."""

METAL_ELEMENTS = frozenset(
    (
        "NA",
        "MG",
        "K",
        "CA",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "CD",
        "HG",
        "PT",
        "MO",
        "AL",
        "BE",
        "BA",
        "RU",
        "V",
        "SR",
        "CS",
        "W",
        "AU",
        "IR",
        "YB",
        "LI",
        "GD",
        "PB",
        "TL",
        "U",
        "Y",
        "LR",
        "TI",
        "RB",
        "AG",
        "SM",
        "OS",
        "PR",
        "PD",
        "EU",
        "TB",
        "RE",
        "RH",
        "HF",
        "TA",
        "LU",
        "HO",
        "CR",
        "GA",
        "LA",
        "SN",
        "SB",
        "CE",
        "ZR",
        "ER",
        "TH",
        "IN",
        "SC",
        "DY",
        "BI",
        "PA",
        "PU",
        "AM",
        "CM",
        "CF",
        "GE",
        "NB",
        "TC",
        "ND",
        "PM",
        "TM",
        "PO",
        "FR",
        "RA",
        "AC",
        "NP",
        "BK",
        "ES",
        "FM",
        "MD",
        "NO",
        "RF",
        "DB",
        "SG",
    )
)

# Element symbols that the Chemical Component Dictionary also uses for a
# component that is not that element:
#
#   U   uranium (238 Da)   but the CCD's ``U`` is uridine 5'-monophosphate (324)
#   NO  nobelium (259 Da)  but the CCD's ``NO`` is nitric oxide (30)
#   CM  curium (247 Da)    but the CCD's ``CM`` is a ~60 Da buffer component
#
# Matching a residue name against METAL_ELEMENTS is only safe with these
# excluded. None of the three elements appears in a deposited macromolecular
# structure, so nothing real is lost by refusing to infer them from a name
# alone; an atom's own element symbol is unaffected and stays authoritative
# wherever it can be read.
#
# Derived rather than guessed: ``test_reference_data`` compares every symbol
# against Gemmi's component table and fails if this set is not exactly the
# symbols whose tabulated weight is not the element's own. Keeping the literal
# here keeps this module dependency-free, which ``tools/`` relies on.
AMBIGUOUS_METAL_COMPONENT_IDS = frozenset(("U", "NO", "CM"))

#: Metal symbols safe to match against a residue name with no element evidence.
UNAMBIGUOUS_METAL_COMPONENT_IDS = METAL_ELEMENTS - AMBIGUOUS_METAL_COMPONENT_IDS
