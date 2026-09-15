"""Behavioural tests for ``src/pdb_records.py``.

The PDB field helpers are exercised directly; the one loaded structure is
built in memory with the shared helpers.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import pytest
from helpers import StructureBuilder

import pdb_records
from structure_analysis import load_structure


@pytest.mark.parametrize(
    "field, expected",
    [
        ("   1", 1),
        ("  10", 10),
        ("9999", 9999),  # last decimal value representable in four columns
        ("A000", 10000),  # first hybrid-36 value: 9999 + 1
        ("A001", 10001),
        ("A00A", 10010),
        ("ZZZZ", 1223055),  # 36**4 - 1 - 10*36**3 + 10**4
        ("abcd", 24701),
    ],
)
def test_decode_pdb_resseq_matches_gemmi(
    field: str, expected: int, tmp_path: Path
) -> None:
    """Verify decoded residue numbers agree with Gemmi.

    EDSTATS rows and raw PDB atoms join on this number, so any divergence from
    Gemmi's ``Residue.seqid.num`` mismatches residues.
    """
    assert pdb_records.decode_pdb_resseq(field) == expected

    line = (
        f"HETATM    1 ZN    ZN B{field}       1.000   2.000   3.000"
        "  1.00 20.00          ZN  \n"
    )
    path = tmp_path / "resseq.pdb"
    path.write_text(line + "END\n", encoding="utf-8")
    structure = gemmi.read_structure(str(path))
    numbers = [residue.seqid.num for chain in structure[0] for residue in chain]
    assert numbers == [expected]


def test_decode_pdb_resseq_hybrid36_starts_immediately_after_9999() -> None:
    assert pdb_records.decode_pdb_resseq("9999") == 9999
    assert (
        pdb_records.decode_pdb_resseq("A000")
        == pdb_records.decode_pdb_resseq("9999") + 1
    )


def test_decode_pdb_resseq_is_case_insensitive() -> None:
    """Gemmi treats resSeq letter case equivalently, so Alchemy must too."""
    for upper, lower in (("A000", "a000"), ("ABCD", "abcd"), ("ZZZZ", "zzzz")):
        assert pdb_records.decode_pdb_resseq(upper) == pdb_records.decode_pdb_resseq(
            lower
        )


@pytest.mark.parametrize("field", ["  -1", "-999", " -12"])
def test_decode_pdb_resseq_accepts_negative_decimal(field: str) -> None:
    """Negative author numbering (expression tags) stays decimal, not base-36."""
    assert pdb_records.decode_pdb_resseq(field) == int(field.strip())


@pytest.mark.parametrize(
    "field",
    [
        "12345",  # wider than the four-column PDB field
        "",  # nothing to decode
        "   ",
        "A00-",  # not a base-36 digit
        "1 2",  # embedded space
        "1.5",
        "$$$$",
    ],
)
def test_decode_pdb_resseq_rejects_undecodable_fields(field: str) -> None:
    with pytest.raises(ValueError):
        pdb_records.decode_pdb_resseq(field)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("10", "10"),
        (" 10 ", "10"),  # surrounding whitespace is not significant
        ("10A", "10A"),  # compact insertion-code form
        ("10:A", "10A"),  # EDSTATS colon-separated form
        ("9999A", "9999A"),
        ("-5", "-5"),
        ("A000", "10000"),  # hybrid-36 number, no insertion code
        ("A00A", "10010"),  # trailing letter of a hybrid-36 number is a digit
        ("0", "0"),
    ],
)
def test_canonical_pdb_residue_id_normalizes_every_input_form(
    value: str, expected: str
) -> None:
    """All accepted spellings must collapse to ``<decimal><icode>``."""
    assert pdb_records.canonical_pdb_residue_id(value) == expected


@pytest.mark.parametrize("value", ["10:AB", "9999AB", "10:  A"])
def test_canonical_pdb_residue_id_rejects_multi_character_insertion(
    value: str,
) -> None:
    """A PDB insertion code is a single column; anything wider is malformed."""
    with pytest.raises(ValueError):
        pdb_records.canonical_pdb_residue_id(value)


def test_canonical_pdb_residue_id_equals_loaded_residue_resnum(
    tmp_path: Path,
) -> None:
    """Verify insertion-code identities agree across statistics and coordinates.

    The statistics table carries ``canonical_pdb_residue_id("10:A")`` and the
    coordinate side carries ``ResidueSelection.resnum``; a mismatch loses the
    residue during the sigma join.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    builder.add_amino_acid("HIS", 10, chain="A", icode="A")
    context = load_structure("test", builder.write_pdb(tmp_path / "ic.pdb"))

    (his,) = [residue for residue in context.residues if residue.residue_name == "HIS"]
    assert his.resnum == "10A"
    assert pdb_records.canonical_pdb_residue_id("10:A") == his.resnum
    assert pdb_records.canonical_pdb_residue_id("10A") == his.resnum


@pytest.mark.parametrize("token", list(pdb_records.MISSING_VALUE_TOKENS))
def test_blank_if_missing_blanks_every_missing_token(token: str) -> None:
    """Blank PDB columns, Gemmi's NUL altloc and mmCIF ``.``/``?`` all mean none."""
    assert pdb_records.blank_if_missing(token) == ""


def test_blank_if_missing_covers_the_documented_token_set() -> None:
    """The missing-value vocabulary is shared with the structure loader's mmCIF conversion."""
    assert set(pdb_records.MISSING_VALUE_TOKENS) == {"", " ", "\x00", ".", "?"}


@pytest.mark.parametrize("value", ["A", "B", "0", "1", "-", "..", "  ", "?!"])
def test_blank_if_missing_preserves_real_values(value: str) -> None:
    """EDSTATS uses ``"0"`` as an ordered-water chain group, so it is a value."""
    assert pdb_records.blank_if_missing(value) == value


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.0, True),
        (0.0, True),  # zero occupancy is a valid deposited value
        (-0.0, True),
        (0.5, True),
        ("0.5", True),  # deposited fields arrive as text
        ("1", True),
        (1.0000001, False),  # just above the physical maximum
        (-1e-9, False),  # just below the physical minimum
        (-0.5, False),
        (2.0, False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
        (None, False),
        (b"0.5", False),
        ("", False),
        ("abc", False),
        ([0.5], False),
    ],
)
def test_valid_occupancy_accepts_only_finite_values_in_unit_range(
    value: object, expected: bool
) -> None:
    assert pdb_records.valid_occupancy(value) is expected


@pytest.mark.parametrize(
    "field, element, status",
    [
        ("ZN", "ZN", "valid"),
        (" ZN ", "ZN", "valid"),
        ("zn", "ZN", "valid"),  # deposited case is normalized
        ("C", "C", "valid"),
        ("D", "D", "valid"),  # deuterium is a real element for Ni purposes
        ("", "", "missing"),
        ("  ", "", "missing"),
        ("X", "", "invalid"),  # Gemmi's unknown-element placeholder
        ("XX", "", "invalid"),
        ("Q", "", "invalid"),
    ],
)
def test_parse_pdb_element_reports_deposited_provenance(
    field: str, element: str, status: str
) -> None:
    """Blank and unrecognized element fields are flagged, never guessed."""
    assert pdb_records.parse_pdb_element(field) == (element, status)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("entry.pdb", "pdb"),
        ("entry.ent", "pdb"),
        ("entry.cif", "mmcif"),
        ("entry.CIF", "mmcif"),
        ("entry.mmcif", "mmcif"),
    ],
)
def test_analysis_format_for_path_follows_the_extension(
    path: str, expected: str
) -> None:
    assert pdb_records.analysis_format_for_path(path) == expected
