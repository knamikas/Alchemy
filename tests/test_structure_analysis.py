"""Behavioural tests for ``src/structure_analysis.py``.

Everything is built in memory with the shared helpers or from explicitly
constructed :class:`structure_analysis.AtomSite` records; nothing touches the
network, CCP4 or the PDB-REDO mirror.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import gemmi
import pytest

import structure_analysis as sa
from helpers import AtomSpec, StructureBuilder, simple_metal_site


_OCC_COLUMN = 54  # PDB occupancy, columns 55-60 (0-based 54:60)
_ELEMENT_COLUMN = 76  # PDB element, columns 77-78


def _rewrite_atom_field(
    source: str, destination, *, atom_name: str, column: int, text: str
) -> str:
    """Copy a PDB file, overwriting a fixed-width field of one named atom.

    Gemmi always writes a well-formed occupancy and element, so malformed
    deposited records have to be injected at the text level.
    """
    destination = str(destination)
    lines: List[str] = []
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            if (
                line[:6].strip() in ("ATOM", "HETATM")
                and line[12:16].strip() == atom_name
            ):
                line = line[:column] + text + line[column + len(text) :]
            lines.append(line)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return destination


def _site(
    atom_name: str,
    *,
    altloc: str = "",
    occupancy: float = 1.0,
    source_order: int = 0,
    element: str = "C",
    occupancy_valid: Optional[bool] = None,
    occupancy_status: Optional[str] = None,
    residue_index: int = 0,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
) -> sa.AtomSite:
    """Build one ``AtomSite`` directly; the builder cannot express bad occupancy."""
    if occupancy_valid is None:
        occupancy_valid = sa._valid_occupancy(occupancy)
    if occupancy_status is None:
        occupancy_status = "valid" if occupancy_valid else "invalid_value"
    atom = gemmi.Atom()
    atom.name = atom_name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*[float(value) for value in pos])
    atom.occ = float(occupancy) if math.isfinite(float(occupancy)) else 0.0
    if altloc:
        atom.altloc = altloc
    return sa.AtomSite(
        pdb_id="test",
        model_index=0,
        model_id="1",
        chain_index=0,
        chain_id="A",
        residue_index=residue_index,
        residue_name="HIS",
        coordinate_residue_name="HIS",
        residue_number=10,
        insertion_code="",
        resnum="10",
        atom_index=source_order,
        source_order=source_order,
        atom_name=atom_name,
        altloc=altloc,
        element=element,
        element_known=True,
        occupancy=float(occupancy),
        occupancy_valid=bool(occupancy_valid),
        occupancy_status=occupancy_status,
        serial=source_order + 1,
        x=float(pos[0]),
        y=float(pos[1]),
        z=float(pos[2]),
        is_water=False,
        is_hydrogen=element in ("H", "D"),
        gemmi_atom=atom,
    )


def _altloc_option_map(selection: sa.ResidueSelection) -> dict:
    """Parse ``ResidueSelection.altloc_options`` into ``{label: text}``."""
    if not selection.altloc_options:
        return {}
    return dict(part.split(":", 1) for part in selection.altloc_options.split("|"))


def _contact(selection: sa.ResidueSelection, atom_name: str) -> sa.AtomSite:
    matches = [atom for atom in selection.contact_atoms if atom.atom_name == atom_name]
    assert len(matches) == 1, (
        f"expected exactly one contact atom named {atom_name!r}, "
        f"got {[atom.altloc for atom in matches]}"
    )
    return matches[0]


def _residue(context: sa.StructureContext, name: str) -> sa.ResidueSelection:
    matches = [residue for residue in context.residues if residue.residue_name == name]
    assert len(matches) == 1, f"expected one {name} residue, got {len(matches)}"
    return matches[0]


def _write_pdb_with_ncs(
    builder: StructureBuilder, path, operations: Sequence[Tuple[str, bool]]
) -> str:
    """Write ``builder`` as PDB with MTRIX records for ``(id, given)`` ops."""
    structure = builder.to_gemmi()
    transform = gemmi.Transform()
    transform.mat.fromlist([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.vec.fromlist([10.0, 0.0, 0.0])
    for identifier, given in operations:
        structure.ncs.append(gemmi.NcsOp(transform, identifier, bool(given)))
    path = str(path)
    structure.write_pdb(path)
    return path


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
def test_decode_pdb_resseq_matches_gemmi(field, expected, tmp_path):
    """EDSTATS rows and raw PDB atoms are joined on this number, so any
    divergence from Gemmi's ``Residue.seqid.num`` mismatches residues."""
    assert sa._decode_pdb_resseq(field) == expected

    line = (
        "HETATM    1 ZN    ZN B{field}       1.000   2.000   3.000"
        "  1.00 20.00          ZN  \n"
    ).format(field=field)
    path = tmp_path / "resseq.pdb"
    path.write_text(line + "END\n", encoding="utf-8")
    structure = gemmi.read_structure(str(path))
    numbers = [residue.seqid.num for chain in structure[0] for residue in chain]
    assert numbers == [expected]


def test_decode_pdb_resseq_hybrid36_starts_immediately_after_9999():
    assert sa._decode_pdb_resseq("9999") == 9999
    assert sa._decode_pdb_resseq("A000") == sa._decode_pdb_resseq("9999") + 1


def test_decode_pdb_resseq_is_case_insensitive():
    """Gemmi treats resSeq letter case equivalently, so Alchemy must too."""
    for upper, lower in (("A000", "a000"), ("ABCD", "abcd"), ("ZZZZ", "zzzz")):
        assert sa._decode_pdb_resseq(upper) == sa._decode_pdb_resseq(lower)


@pytest.mark.parametrize("field", ["  -1", "-999", " -12"])
def test_decode_pdb_resseq_accepts_negative_decimal(field):
    """Negative author numbering (expression tags) stays decimal, not base-36."""
    assert sa._decode_pdb_resseq(field) == int(field.strip())


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
def test_decode_pdb_resseq_rejects_undecodable_fields(field):
    with pytest.raises(ValueError):
        sa._decode_pdb_resseq(field)


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
def test_canonical_pdb_residue_id_normalizes_every_input_form(value, expected):
    """All accepted spellings must collapse to ``<decimal><icode>``."""
    assert sa.canonical_pdb_residue_id(value) == expected


@pytest.mark.parametrize("value", ["10:AB", "9999AB", "10:  A"])
def test_canonical_pdb_residue_id_rejects_multi_character_insertion(value):
    """A PDB insertion code is a single column; anything wider is malformed."""
    with pytest.raises(ValueError):
        sa.canonical_pdb_residue_id(value)


def test_canonical_pdb_residue_id_equals_loaded_residue_resnum(tmp_path):
    """The stats table carries ``canonical_pdb_residue_id("10:A")`` and the
    coordinate side ``ResidueSelection.resnum``; if they differ the sigma join
    loses the residue."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    builder.add_amino_acid("HIS", 10, chain="A", icode="A")
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "ic.pdb"))

    his = _residue(context, "HIS")
    assert his.resnum == "10A"
    assert sa.canonical_pdb_residue_id("10:A") == his.resnum
    assert sa.canonical_pdb_residue_id("10A") == his.resnum


@pytest.mark.parametrize("token", list(sa.MISSING_VALUE_TOKENS))
def test_blank_if_missing_blanks_every_missing_token(token):
    """Blank PDB columns, Gemmi's NUL altloc and mmCIF ``.``/``?`` all mean none."""
    assert sa.blank_if_missing(token) == ""


def test_blank_if_missing_covers_the_documented_token_set():
    """The missing-value vocabulary is shared with main.py's mmCIF conversion."""
    assert set(sa.MISSING_VALUE_TOKENS) == {"", " ", "\x00", ".", "?"}


@pytest.mark.parametrize("value", ["A", "B", "0", "1", "-", "..", "  ", "?!"])
def test_blank_if_missing_preserves_real_values(value):
    """EDSTATS uses ``"0"`` as an ordered-water chain group, so it is a value."""
    assert sa.blank_if_missing(value) == value


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
def test_valid_occupancy_accepts_only_finite_values_in_unit_range(value, expected):
    assert sa._valid_occupancy(value) is expected


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
def test_parse_pdb_element_reports_deposited_provenance(field, element, status):
    """Blank and unrecognized element fields are flagged, never guessed."""
    assert sa._parse_pdb_element(field) == (element, status)


@pytest.mark.parametrize(
    "candidate, current, expected, why",
    [
        (
            _site("O", occupancy=0.1, source_order=5),
            _site("O", occupancy=float("nan"), source_order=0),
            True,
            "a valid occupancy always beats an invalid one",
        ),
        (
            _site("O", occupancy=float("nan"), source_order=0),
            _site("O", occupancy=0.1, source_order=5),
            False,
            "an invalid occupancy never beats a valid one",
        ),
        (
            _site("O", occupancy=0.9, source_order=5),
            _site("O", occupancy=0.4, source_order=0),
            True,
            "higher valid occupancy wins regardless of file order",
        ),
        (
            _site("O", occupancy=0.4, source_order=0),
            _site("O", occupancy=0.9, source_order=5),
            False,
            "lower valid occupancy loses regardless of file order",
        ),
        (
            _site("O", occupancy=0.5, source_order=1),
            _site("O", occupancy=0.5, source_order=3),
            True,
            "an exact tie keeps the earlier source record",
        ),
        (
            _site("O", occupancy=0.5, source_order=3),
            _site("O", occupancy=0.5, source_order=1),
            False,
            "an exact tie does not displace the earlier source record",
        ),
        (
            _site("O", occupancy=2.0, source_order=1),
            _site("O", occupancy=5.0, source_order=3),
            True,
            "two invalid records fall back to source order, not magnitude",
        ),
    ],
)
def test_site_is_better_ranks_validity_then_occupancy_then_order(
    candidate, current, expected, why
):
    assert sa._site_is_better(candidate, current) is expected, why


def test_select_residue_without_alternates_shares_every_blank_atom():
    atoms = [
        _site("N", occupancy=1.0, source_order=0),
        _site("CA", occupancy=0.8, source_order=1),
        _site("C", occupancy=0.6, source_order=2),
    ]
    selection = sa._select_residue(atoms)

    assert selection.selected_altloc == ""
    assert selection.alternative_conformers_present is False
    assert selection.altloc_selection_fallback is False
    assert [atom.atom_name for atom in selection.contact_atoms] == ["N", "CA", "C"]
    assert selection.selected_conformer_mean_occupancy == pytest.approx(
        (1.0 + 0.8 + 0.6) / 3
    )


def test_select_residue_shares_blank_atoms_with_the_selected_conformer():
    atoms = [
        _site("N", occupancy=1.0, source_order=0),
        _site("CA", occupancy=1.0, source_order=1),
        _site("NE2", altloc="A", occupancy=0.3, source_order=2),
        _site("NE2", altloc="B", occupancy=0.7, source_order=3),
    ]
    selection = sa._select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert [(atom.atom_name, atom.altloc) for atom in selection.contact_atoms] == [
        ("N", ""),
        ("CA", ""),
        ("NE2", "B"),
    ]


def test_select_residue_uses_the_conformer_mean_not_the_single_best_atom():
    """Selection compares mean occupancy per conformer, per the README.

    Conformer A holds the highest single atom (0.9) but the lower mean (0.5);
    per-atom maxima would build a chimeric residue.
    """
    atoms = [
        _site("CB", altloc="A", occupancy=0.9, source_order=0),
        _site("CG", altloc="A", occupancy=0.1, source_order=1),
        _site("CB", altloc="B", occupancy=0.6, source_order=2),
        _site("CG", altloc="B", occupancy=0.6, source_order=3),
    ]
    selection = sa._select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert selection.selected_conformer_mean_occupancy == pytest.approx(0.6)
    assert {atom.altloc for atom in selection.contact_atoms} == {"B"}


@pytest.mark.parametrize("order", ["ab", "ba"])
def test_select_residue_breaks_occupancy_ties_by_altloc_label(order):
    a_atoms = [
        _site("CB", altloc="A", occupancy=0.5, source_order=0),
        _site("CG", altloc="A", occupancy=0.5, source_order=1),
    ]
    b_atoms = [
        _site("CB", altloc="B", occupancy=0.5, source_order=2),
        _site("CG", altloc="B", occupancy=0.5, source_order=3),
    ]
    atoms = a_atoms + b_atoms if order == "ab" else b_atoms + a_atoms

    selection = sa._select_residue(atoms)

    assert selection.selected_altloc == "A"
    assert {atom.altloc for atom in selection.contact_atoms} == {"A"}


def test_select_residue_averages_only_valid_occupancies():
    """Conformer A is 0.4 plus an out-of-range 5.0: averaging the raw values
    gives 2.7 and hands A the selection, the valid subset gives 0.4 and B."""
    atoms = [
        _site("CB", altloc="A", occupancy=0.4, source_order=0),
        _site("CG", altloc="A", occupancy=5.0, source_order=1),
        _site("CB", altloc="B", occupancy=0.6, source_order=2),
        _site("CG", altloc="B", occupancy=0.6, source_order=3),
    ]
    selection = sa._select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert selection.selected_conformer_mean_occupancy == pytest.approx(0.6)
    assert float(_altloc_option_map(selection)["A"]) == pytest.approx(0.4)


def test_select_residue_records_every_available_alternative():
    atoms = [
        _site("CB", altloc="A", occupancy=0.25, source_order=0),
        _site("CB", altloc="B", occupancy=0.35, source_order=1),
        _site("CB", altloc="C", occupancy=0.40, source_order=2),
    ]
    selection = sa._select_residue(atoms)

    options = _altloc_option_map(selection)
    assert set(options) == {"A", "B", "C"}
    assert [float(options[label]) for label in ("A", "B", "C")] == pytest.approx(
        [0.25, 0.35, 0.40]
    )
    assert selection.selected_altloc == "C"


def test_select_residue_keeps_one_coherent_conformer_across_all_atoms():
    """Mixing conformers would hand the contact search an invented residue."""
    names = ("CB", "CG", "ND1", "CD2", "CE1", "NE2")
    atoms = []
    for index, name in enumerate(names):
        atoms.append(_site(name, altloc="A", occupancy=0.45, source_order=2 * index))
        atoms.append(
            _site(name, altloc="B", occupancy=0.55, source_order=2 * index + 1)
        )
    selection = sa._select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert {atom.altloc for atom in selection.contact_atoms} == {"B"}
    assert [atom.atom_name for atom in selection.contact_atoms] == list(names)
    assert selection.chemical_atom_site_count == len(names)
    assert len(selection.source_atoms) == 2 * len(names)


def test_select_residue_falls_back_when_no_conformer_has_valid_occupancy():
    atoms = [
        _site("CB", altloc="B", occupancy=2.5, source_order=0),
        _site("CB", altloc="A", occupancy=float("nan"), source_order=1),
    ]
    selection = sa._select_residue(atoms)

    assert selection.altloc_selection_fallback is True
    assert selection.selected_altloc == "A"  # lowest label, not file order
    assert selection.selected_conformer_mean_occupancy is None
    assert _altloc_option_map(selection) == {"A": "NA", "B": "NA"}
    assert {atom.altloc for atom in selection.contact_atoms} == {"A"}


def test_select_residue_prefers_a_selected_conformer_over_a_blank_duplicate():
    atoms = [
        _site("NE2", altloc="", occupancy=1.0, source_order=0),
        _site("NE2", altloc="A", occupancy=0.4, source_order=1),
        _site("NE2", altloc="B", occupancy=0.6, source_order=2),
    ]
    selection = sa._select_residue(atoms)

    contact = _contact(selection, "NE2")
    assert contact.altloc == "B"
    assert selection.selected_over_blank_duplicate_count == 1


def test_select_residue_resolves_repeated_atom_names_by_occupancy():
    atoms = [
        _site("NE2", altloc="A", occupancy=0.4, source_order=0, pos=(1.0, 0.0, 0.0)),
        _site("NE2", altloc="A", occupancy=0.9, source_order=1, pos=(2.0, 0.0, 0.0)),
    ]
    selection = sa._select_residue(atoms)

    contact = _contact(selection, "NE2")
    assert contact.occupancy == pytest.approx(0.9)
    assert contact.x == pytest.approx(2.0)
    assert selection.malformed_duplicate_atom_name_count == 1


def test_select_residue_orders_contact_atoms_by_source_order():
    """Deposited order keeps the downstream atom indices stable."""
    atoms = [
        _site("C", occupancy=1.0, source_order=7),
        _site("N", occupancy=1.0, source_order=2),
        _site("CA", occupancy=1.0, source_order=5),
    ]
    selection = sa._select_residue(atoms)

    assert [atom.atom_name for atom in selection.contact_atoms] == ["N", "CA", "C"]
    assert [atom.source_order for atom in selection.source_atoms] == [2, 5, 7]


def test_load_structure_selects_conformer_and_keeps_both_for_counting(tmp_path):
    """Occupancies 0.35/0.55 make the deposited sum (0.90) differ from both the
    selected conformer alone and from 1.0."""
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = builder.residues[1]
    builder.add_conformers(
        his,
        [
            ("A", 0.35, {"NE2": (2.03, 0.0, 0.0)}),
            ("B", 0.55, {"NE2": (2.80, 0.0, 0.0)}),
        ],
        atom_names=["NE2"],
    )
    path = builder.write_pdb(tmp_path / "conformers.pdb")

    context = sa.load_structure("test", path)
    selection = _residue(context, "HIS")

    assert selection.selected_altloc == "B"
    assert selection.alternative_conformers_present is True
    contact = _contact(selection, "NE2")
    assert contact.altloc == "B"
    assert contact.x == pytest.approx(2.80, abs=1e-6)

    # 9 blank HIS atoms + both NE2 alternates + ZN.
    assert sa.count_deposited_ni(context) == pytest.approx(9 + 0.35 + 0.55 + 1.0)


def test_load_structure_flags_altloc_fallback_from_the_file(tmp_path):
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = builder.residues[1]
    builder.add_conformers(his, [("A", 1.50, {}), ("B", 2.50, {})], atom_names=["NE2"])
    path = builder.write_pdb(tmp_path / "fallback.pdb")

    context = sa.load_structure("test", path)
    selection = _residue(context, "HIS")

    assert selection.altloc_selection_fallback is True
    assert selection.selected_altloc == "A"
    assert "altloc_selection_fallback" in context.warning_codes
    assert context.occupancy_validation_failed is True
    assert math.isnan(sa.count_deposited_ni(context))


def _two_model_pdb(path) -> str:
    """Write a two-model PDB whose second model holds an atom the first lacks."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    structure = builder.to_gemmi()
    model = gemmi.Model(2)
    chain = gemmi.Chain("B")
    residue = gemmi.Residue()
    residue.name = "FE"
    residue.seqid = gemmi.SeqId(2, " ")
    residue.het_flag = "H"
    atom = gemmi.Atom()
    atom.name = "FE"
    atom.element = gemmi.Element("Fe")
    atom.pos = gemmi.Position(5.0, 5.0, 5.0)
    atom.occ = 1.0
    atom.b_iso = 20.0
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    path = str(path)
    structure.write_pdb(path)
    return path


def test_load_structure_analyzes_the_first_model_only(tmp_path):
    """The README forbids combining models."""
    path = _two_model_pdb(tmp_path / "two_models.pdb")
    context = sa.load_structure("test", path)

    assert context.model_policy == "first"
    assert context.model_index == 0
    assert context.model_analyzed == 1
    assert context.analyzed_model_id == "1"
    assert context.input_model_count == 2
    assert context.multi_model_structure is True
    assert "multi_model_structure" in context.warning_codes

    assert {atom.element for atom in context.source_atoms} == {"ZN", "O"}
    assert all(atom.model_index == 0 for atom in context.source_atoms)
    assert all(atom.model_index == 0 for atom in context.contact_atoms)
    # ZN + water O only; the second model's FE is excluded.
    assert sa.count_deposited_ni(context) == pytest.approx(2.0)


def test_load_structure_reports_source_model_count_from_the_original_file(tmp_path):
    """``main`` strips MODEL/ENDMDL before EDSTATS, so the analysis file has one
    model where the deposited entry had several."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = builder.write_pdb(tmp_path / "stripped.pdb")

    context = sa.load_structure("test", path, source_model_count=8)

    assert context.input_model_count == 8
    assert context.multi_model_structure is True
    assert context.model_analyzed == 1
    assert "multi_model_structure" in context.warning_codes


def test_load_structure_rejects_a_source_model_count_below_the_file(tmp_path):
    path = _two_model_pdb(tmp_path / "two_models.pdb")
    with pytest.raises(ValueError, match="source model count"):
        sa.load_structure("test", path, source_model_count=1)


def test_load_structure_single_model_reports_no_multi_model_warning(tmp_path):
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "single.pdb"))

    assert context.input_model_count == 1
    assert context.multi_model_structure is False
    assert "multi_model_structure" not in context.warning_codes


@pytest.mark.parametrize(
    "label, field, status, missing, invalid",
    [
        ("missing", "      ", "missing", 1, 0),
        ("non_finite", "   nan", "invalid_non_finite", 0, 1),
        ("negative", " -0.50", "invalid_range", 0, 1),
        ("above_one", "  1.50", "invalid_range", 0, 1),
        ("non_numeric", "  abcd", "invalid_non_numeric", 0, 1),
    ],
)
def test_invalid_occupancy_makes_dpi_unavailable_without_repair(
    tmp_path, label, field, status, missing, invalid
):
    """The README makes malformed occupancy disable DPI rather than be repaired.

    The coordinates are retained so distance analysis can continue.
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    path = _rewrite_atom_field(
        clean, tmp_path / f"{label}.pdb", atom_name="ZN", column=_OCC_COLUMN, text=field
    )

    context = sa.load_structure("test", path)
    metal = [atom for atom in context.source_atoms if atom.element == "ZN"][0]

    assert metal.occupancy_status == status
    assert metal.occupancy_valid is False
    assert context.occupancy_validation_failed is True
    assert context.missing_occupancy_count == missing
    assert context.invalid_occupancy_count == invalid
    assert math.isnan(sa.count_deposited_ni(context))
    assert math.isnan(sa.count_ni(context))

    assert metal.occupancy != 1.0
    assert len(context.contact_atoms) == 2
    assert (metal.x, metal.y, metal.z) == (0.0, 0.0, 0.0)


def test_zero_occupancy_is_valid_for_ni(tmp_path):
    """Zero occupancy is a measured value: it keeps DPI available and adds 0."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    path = _rewrite_atom_field(
        clean, tmp_path / "zero.pdb", atom_name="ZN", column=_OCC_COLUMN, text="  0.00"
    )

    context = sa.load_structure("test", path)
    metal = [atom for atom in context.source_atoms if atom.element == "ZN"][0]

    assert metal.occupancy == pytest.approx(0.0)
    assert metal.occupancy_valid is True
    assert metal.occupancy_status == "valid"
    assert context.occupancy_validation_failed is False
    assert context.zero_occupancy_atom_count == 1
    assert "zero_occupancy_atoms" in context.warning_codes
    # Water oxygen contributes 1.0, the zero-occupancy metal contributes 0.0.
    assert sa.count_deposited_ni(context) == pytest.approx(1.0)


def test_missing_occupancy_is_not_counted_as_a_measured_zero(tmp_path):
    """Gemmi parses the blank column as 0.0, so only the raw provenance can tell
    "not deposited" from "deposited as zero"."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    path = _rewrite_atom_field(
        clean, tmp_path / "blank.pdb", atom_name="ZN", column=_OCC_COLUMN, text="      "
    )

    context = sa.load_structure("test", path)

    assert context.missing_occupancy_count == 1
    assert context.zero_occupancy_atom_count == 0
    assert "zero_occupancy_atoms" not in context.warning_codes
    assert math.isnan(sa.count_deposited_ni(context))


def test_unknown_element_makes_the_atom_count_indeterminate(tmp_path):
    """A blank element field is unknown, not inferred from the atom name."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    path = _rewrite_atom_field(
        clean,
        tmp_path / "noelement.pdb",
        atom_name="ZN",
        column=_ELEMENT_COLUMN,
        text="  ",
    )

    context = sa.load_structure("test", path)
    metal = [atom for atom in context.source_atoms if atom.atom_name == "ZN"][0]

    assert metal.element == ""
    assert metal.element_known is False
    assert context.unknown_element_atom_count == 1
    assert context.element_validation_warning == "unknown_element_atoms"
    assert "unknown_elements" in context.warning_codes
    assert context.occupancy_validation_failed is False  # occupancies are fine
    assert math.isnan(sa.count_deposited_ni(context))
    assert math.isnan(sa.count_ni(context))


def test_count_deposited_ni_is_the_occupancy_weighted_heavy_atom_sum(tmp_path):
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", occupancy=0.6)
    builder.add_water(101, (0.0, 2.09, 0.0), chain="B", occupancy=0.25)
    builder.add_water(102, (0.0, 4.20, 0.0), chain="B", occupancy=1.0)
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "ni.pdb"))

    assert sa.count_deposited_ni(context) == pytest.approx(0.6 + 0.25 + 1.0)


def test_count_deposited_ni_excludes_hydrogen_and_deuterium(tmp_path):
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    builder.add_hetero_residue(
        "HOH",
        101,
        [
            AtomSpec("O", "O", (0.0, 2.09, 0.0)),
            AtomSpec("H1", "H", (0.6, 2.60, 0.0)),
            AtomSpec("D2", "D", (-0.6, 2.60, 0.0)),
        ],
        chain="B",
    )
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "hydro.pdb"))

    assert len(context.source_atoms) == 4
    assert sum(atom.is_hydrogen for atom in context.source_atoms) == 2
    assert sa.count_deposited_ni(context) == pytest.approx(2.0)


def test_count_deposited_ni_counts_alternate_positions_separately(tmp_path):
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    water = builder.add_water(101, (0.0, 2.09, 0.0), chain="B")
    builder.add_conformers(water, [("A", 0.3, {}), ("B", 0.5, {"O": (0.0, 2.60, 0.0)})])
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "alt.pdb"))

    water_selection = _residue(context, "HOH")
    assert len(water_selection.source_atoms) == 2
    assert len(water_selection.contact_atoms) == 1  # only B is a contact
    assert sa.count_deposited_ni(context) == pytest.approx(1.0 + 0.3 + 0.5)


def test_count_ni_multiplies_by_non_given_strict_ncs_copies(tmp_path):
    """Operations flagged ``given`` are already deposited and are not counted
    a second time."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = _write_pdb_with_ncs(
        builder, tmp_path / "ncs.pdb", [("1", False), ("2", True), ("3", False)]
    )

    context = sa.load_structure("test", path)

    assert context.strict_ncs_operation_ids == ("1", "3")
    assert context.strict_ncs_operation_count == 2
    assert context.dpi_atom_count_multiplier == 3
    deposited = sa.count_deposited_ni(context)
    assert deposited == pytest.approx(2.0)
    assert sa.count_ni(context) == pytest.approx(6.0)
    assert sa.count_ni(context) == pytest.approx(
        deposited * context.dpi_atom_count_multiplier
    )


def test_count_ni_equals_deposited_without_strict_ncs(tmp_path):
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "plain.pdb"))

    assert context.strict_ncs_operation_ids == ()
    assert context.dpi_atom_count_multiplier == 1
    assert sa.count_ni(context) == pytest.approx(sa.count_deposited_ni(context))


def test_count_ni_stays_unavailable_when_the_deposited_count_is(tmp_path):
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = _write_pdb_with_ncs(builder, tmp_path / "ncs_clean.pdb", [("1", False)])
    path = _rewrite_atom_field(
        clean,
        tmp_path / "ncs_bad.pdb",
        atom_name="ZN",
        column=_OCC_COLUMN,
        text="  1.50",
    )

    context = sa.load_structure("test", path)

    assert context.dpi_atom_count_multiplier == 2
    assert math.isnan(sa.count_deposited_ni(context))
    assert math.isnan(sa.count_ni(context))


def test_duplicate_atom_records_collapse_to_the_higher_occupancy(tmp_path):
    builder = StructureBuilder()
    builder.add_hetero_residue(
        "ZN",
        1,
        [
            AtomSpec("ZN", "ZN", (0.0, 0.0, 0.0), occupancy=0.4),
            AtomSpec("ZN", "ZN", (0.0, 0.0, 0.0), occupancy=0.9),
        ],
        chain="B",
    )
    builder.add_water(101, (0.0, 2.09, 0.0), chain="B")
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "dup.pdb"))

    metals = [atom for atom in context.source_atoms if atom.element == "ZN"]
    assert len(metals) == 1
    assert metals[0].occupancy == pytest.approx(0.9)
    assert context.duplicate_atom_records_present is True
    assert context.duplicate_atom_record_count == 1
    assert context.duplicate_coordinate_conflict_count == 0
    assert "duplicate_atom_records" in context.warning_codes
    assert "duplicate_atom_coordinate_conflict" not in context.warning_codes
    assert sa.count_deposited_ni(context) == pytest.approx(0.9 + 1.0)


def test_duplicate_atom_records_at_different_positions_are_flagged(tmp_path):
    """Duplicates further apart than the 0.001 A tolerance are a real conflict."""
    builder = StructureBuilder()
    builder.add_hetero_residue(
        "ZN",
        1,
        [
            AtomSpec("ZN", "ZN", (0.0, 0.0, 0.0), occupancy=0.4),
            AtomSpec("ZN", "ZN", (0.5, 0.0, 0.0), occupancy=0.9),
        ],
        chain="B",
    )
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "conflict.pdb"))

    assert context.duplicate_atom_record_count == 1
    assert context.duplicate_coordinate_conflict_count == 1
    assert "duplicate_atom_coordinate_conflict" in context.warning_codes


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((0.0, 0.0, 0.0), (3.0, 4.0, 12.0), 13.0),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 0.0),
        ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), math.sqrt(12.0)),
        ((0.0, 0.0, 0.0), (2.03, 0.0, 0.0), 2.03),
        ((5.0, 0.0, 0.0), (0.0, 0.0, 0.0), 5.0),
    ],
)
def test_position_distance_is_euclidean(a, b, expected):
    assert sa.position_distance(a, b) == pytest.approx(expected, abs=1e-12)
    assert sa.position_distance(b, a) == pytest.approx(expected, abs=1e-12)


def test_position_distance_matches_gemmi_position_distance():
    first = (1.5, -2.25, 7.75)
    second = (-3.0, 4.5, 0.25)
    expected = gemmi.Position(*first).dist(gemmi.Position(*second))
    assert sa.position_distance(first, second) == pytest.approx(expected, abs=1e-12)


@pytest.fixture
def ncs_context(tmp_path) -> sa.StructureContext:
    """A loaded context with four symmetry operations and two strict-NCS ops."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = _write_pdb_with_ncs(
        builder, tmp_path / "prov.pdb", [("1", False), ("2", False)]
    )
    context = sa.load_structure("test", path)
    assert context.crystallographic_operation_count == 4
    assert context.strict_ncs_operation_ids == ("1", "2")
    return context


@pytest.mark.parametrize(
    "image_index, translation, expected",
    [
        (0, (0, 0, 0), (False, False, "", "explicit")),
        (1, (0, 0, 0), (True, False, "", "crystallographic")),
        (3, (0, 0, 0), (True, False, "", "crystallographic")),
        (0, (1, 0, 0), (True, False, "", "crystallographic")),
        (0, (0, 0, -1), (True, False, "", "crystallographic")),
        (4, (0, 0, 0), (False, True, "1", "strict_ncs")),
        (5, (0, 0, 0), (True, True, "1", "strict_ncs_and_crystallographic")),
        (4, (1, 0, 0), (True, True, "1", "strict_ncs_and_crystallographic")),
        (8, (0, 0, 0), (False, True, "2", "strict_ncs")),
        (11, (0, 0, 0), (True, True, "2", "strict_ncs_and_crystallographic")),
    ],
)
def test_image_provenance_classifies_symmetry_and_ncs(
    ncs_context, image_index, translation, expected
):
    """Gemmi's ``setup_cell_images()`` lists the identity, then the remaining
    space-group operations, then one full block per strict-NCS transform. A
    nonzero cell translation is crystallographic even at the identity."""
    assert ncs_context.image_provenance(image_index, tuple(translation)) == expected


def test_image_provenance_explicit_only_for_the_identity_image(ncs_context):
    crystallographic, strict_ncs, ncs_id, scope = ncs_context.image_provenance(
        0, (0, 0, 0)
    )
    assert (crystallographic, strict_ncs, ncs_id, scope) == (
        False,
        False,
        "",
        "explicit",
    )


def test_image_provenance_rejects_a_negative_image_index(ncs_context):
    with pytest.raises(ValueError, match="negative"):
        ncs_context.image_provenance(-1, (0, 0, 0))


def test_image_provenance_rejects_an_image_beyond_the_ncs_blocks(ncs_context):
    with pytest.raises(ValueError, match="strict-NCS"):
        ncs_context.image_provenance(12, (0, 0, 0))


def test_image_provenance_requires_symmetry_metadata(tmp_path):
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)], cell=None)
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "nocell.pdb"))

    assert context.symmetry_search_available is False
    assert context.symmetry_search_failure_reason == "missing_or_invalid_unit_cell"
    assert context.crystallographic_operation_count == 0
    with pytest.raises(ValueError, match="operation count"):
        context.image_provenance(0, (0, 0, 0))


def test_image_provenance_without_ncs_never_reports_a_strict_ncs_scope(tmp_path):
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "noncs.pdb"))

    scopes = {
        context.image_provenance(index, (0, 0, 0))[3]
        for index in range(context.crystallographic_operation_count)
    }
    assert scopes == {"explicit", "crystallographic"}
    with pytest.raises(ValueError, match="strict-NCS"):
        context.image_provenance(context.crystallographic_operation_count, (0, 0, 0))


def test_image_provenance_ncs_id_tracks_the_operation_block(tmp_path):
    """The reported NCS identifier is the deposited MTRIX id, not an index."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = _write_pdb_with_ncs(
        builder, tmp_path / "ids.pdb", [("7", False), ("9", False)]
    )
    context = sa.load_structure("test", path)

    operations = context.crystallographic_operation_count
    assert context.image_provenance(operations, (0, 0, 0))[2] == "7"
    assert context.image_provenance(2 * operations, (0, 0, 0))[2] == "9"


def test_load_structure_reads_mmcif_without_raw_pdb_matching(tmp_path):
    """For mmCIF input the parser is authoritative; no raw-record join runs."""
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)])
    context = sa.load_structure("test", builder.write_cif(tmp_path / "site.cif"))

    assert context.analysis_coordinate_format == "mmcif"
    assert context.raw_occupancy_mapping_failed is False
    assert context.raw_occupancy_mapping_failure_reason == ""
    assert context.occupancy_validation_failed is False
    assert all(atom.occupancy_status == "valid" for atom in context.source_atoms)
    assert sa.count_deposited_ni(context) == pytest.approx(
        float(len(context.source_atoms))
    )


def test_mmcif_occupancy_out_of_range_still_disables_dpi(tmp_path):
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", occupancy=1.5)
    builder.add_water(101, (0.0, 2.09, 0.0), chain="B")
    context = sa.load_structure("test", builder.write_cif(tmp_path / "bad.cif"))

    metal = [atom for atom in context.source_atoms if atom.element == "ZN"][0]
    assert metal.occupancy == pytest.approx(1.5)
    assert metal.occupancy_valid is False
    assert context.occupancy_validation_failed is True
    assert math.isnan(sa.count_deposited_ni(context))


def test_context_indexes_only_contact_atoms_for_neighbor_marks(tmp_path):
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = builder.residues[1]
    builder.add_conformers(
        his,
        [("A", 0.3, {"NE2": (2.03, 0.0, 0.0)}), ("B", 0.7, {"NE2": (2.80, 0.0, 0.0)})],
        atom_names=["NE2"],
    )
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "marks.pdb"))

    indexed = [
        context.atom_for_indices(atom.chain_index, atom.residue_index, atom.atom_index)
        for atom in context.contact_atoms
    ]
    assert all(found is not None for found in indexed)
    resolved = _contact(_residue(context, "HIS"), "NE2")
    assert (
        context.atom_for_indices(
            resolved.chain_index, resolved.residue_index, resolved.atom_index
        )
        is resolved
    )
    assert resolved.altloc == "B"
    deselected = next(
        atom
        for atom in context.source_atoms
        if atom.residue_name == "HIS" and atom.atom_name == "NE2" and atom.altloc == "A"
    )
    assert (
        context.atom_for_indices(
            deselected.chain_index,
            deselected.residue_index,
            deselected.atom_index,
        )
        is None
    )


def test_context_residues_are_addressable_by_author_identity(tmp_path):
    """Author (name, chain, resnum) lookup is how EDSTATS rows find residues."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    builder.add_amino_acid("HIS", 10, chain="A", icode="A")
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "auth.pdb"))

    found = context.residues_for_author("HIS", "A", "10A")
    assert len(found) == 1
    assert found[0] is _residue(context, "HIS")
    assert context.residues_for_author("HIS", "A", "10") == ()
    assert context.residue_for_atom(found[0].contact_atoms[0]) is found[0]


def test_metal_atoms_uses_selected_conformers_by_default(tmp_path):
    builder = StructureBuilder()
    metal = builder.add_metal("ZN", 1, chain="B")
    builder.add_conformers(metal, [("A", 0.4, {}), ("B", 0.6, {"ZN": (0.4, 0.0, 0.0)})])
    builder.add_water(101, (0.0, 2.09, 0.0), chain="B")
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "metal_alt.pdb"))

    canonical = context.metal_atoms(["ZN"])
    assert len(canonical) == 1
    assert canonical[0].altloc == "B"
    assert len(context.metal_atoms(["ZN"], canonical=False)) == 2
    assert context.metal_atoms(["zn"]) == canonical  # element case-insensitive


def _oxygen_neighbors_of_the_metal(context: sa.StructureContext) -> List[str]:
    search = context.make_neighbor_search(5.0, include_symmetry=False)
    metal = context.metal_atoms(["ZN"])[0]
    found = [
        context.atom_for_mark(mark)
        for mark in search.find_atoms(metal.pos, "\0", radius=5.0)
    ]
    return [
        atom.atom_name for atom in found if atom is not None and atom.element == "O"
    ]


def test_neighbor_search_skips_zero_occupancy_atoms(tmp_path):
    """Zero occupancy counts for Ni but is not evidence for a contact.

    The full-occupancy control proves the search would otherwise find the water.
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    assert _oxygen_neighbors_of_the_metal(sa.load_structure("test", clean)) == ["O"]

    path = _rewrite_atom_field(
        clean,
        tmp_path / "zero_water.pdb",
        atom_name="O",
        column=_OCC_COLUMN,
        text="  0.00",
    )
    context = sa.load_structure("test", path)

    assert _oxygen_neighbors_of_the_metal(context) == []
    assert context.zero_occupancy_atom_count == 1
    assert sa.count_deposited_ni(context) == pytest.approx(1.0)


def test_neighbor_search_requires_symmetry_metadata_when_asked(tmp_path):
    """An image-inclusive search must fail loudly without a usable space group.

    mmCIF is used because the PDB writer substitutes ``P 1`` on CRYST1.
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)], spacegroup=None)
    context = sa.load_structure("test", builder.write_cif(tmp_path / "nosg.cif"))

    assert context.symmetry_search_available is False
    assert context.symmetry_search_failure_reason == ("missing_or_invalid_space_group")
    with pytest.raises(ValueError, match="space_group"):
        context.make_neighbor_search(4.0, include_symmetry=True)
    assert context.make_neighbor_search(4.0, include_symmetry=False) is not None
