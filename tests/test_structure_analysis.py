"""Behavioural tests for ``src/structure_analysis.py``.

Every structure is built in memory with the shared helpers and loaded through
``load_structure``; nothing touches the network, CCP4 or the PDB-REDO mirror.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import gemmi
import helpers
import pytest
from helpers import AtomSpec, StructureBuilder, approx, simple_metal_site

import structure_analysis as sa

_OCC_COLUMN = 54  # PDB occupancy, columns 55-60 (0-based 54:60)
_ELEMENT_COLUMN = 76  # PDB element, columns 77-78

# Every writer below stringifies its argument, so callers pass ``tmp_path / ...``
# as readily as a plain string.
_StrPath = str | os.PathLike[str]


def _rewrite_atom_field(
    source: str, destination: _StrPath, *, atom_name: str, column: int, text: str
) -> str:
    """Copy a PDB file, overwriting a fixed-width field of one named atom.

    Gemmi always writes a well-formed occupancy and element, so malformed
    deposited records have to be injected at the text level.
    """
    destination = str(destination)
    lines: list[str] = []
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
    builder: StructureBuilder, path: _StrPath, operations: Sequence[tuple[str, bool]]
) -> str:
    """Write ``builder`` as PDB with MTRIX records for ``(id, given)`` ops."""
    structure = builder.to_gemmi()
    transform = gemmi.Transform()
    transform.mat.fromlist([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.vec.fromlist([10.0, 0.0, 0.0])
    for identifier, given in operations:
        structure.ncs.append(gemmi.NcsOp(transform, identifier, bool(given)))
    path = str(path)
    helpers.write_pdb(structure, path)
    return path


def test_load_structure_selects_conformer_and_keeps_both_for_counting(
    tmp_path: Path,
) -> None:
    """Verify deposited alternate occupancies determine the DPI atom count.

    Occupancies 0.35/0.55 make the deposited sum of 0.90 differ from both the
    selected conformer alone and 1.0.
    """
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
    assert contact.x == approx(2.80, abs=1e-6)

    # 9 blank HIS atoms + both NE2 alternates + ZN.
    assert sa.count_deposited_ni(context) == approx(9 + 0.35 + 0.55 + 1.0)


def test_rounded_alternate_occupancy_is_reported_but_keeps_the_dpi(
    tmp_path: Path,
) -> None:
    """Verify small alternate-occupancy rounding excess does not invalidate DPI.

    Independently rounded 0.46 and 0.55 occupancies can legitimately sum to 1.01.
    """
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = builder.residues[1]
    builder.add_conformers(
        his,
        [
            ("A", 0.46, {"NE2": (2.03, 0.0, 0.0)}),
            ("B", 0.55, {"NE2": (2.80, 0.0, 0.0)}),
        ],
        atom_names=["NE2"],
    )

    context = sa.load_structure(
        "test", builder.write_pdb(tmp_path / "rounded_conformers.pdb")
    )

    assert context.occupancy.overfull_site_count == 1
    assert context.occupancy.overfull_excess == approx(0.01)
    # The measurement is still reported, so the deposition oddity stays visible.
    assert "overfull_alternate_occupancy" in context.warning_codes
    assert context.occupancy.validation_failed is False
    assert math.isfinite(sa.count_deposited_ni(context))
    assert math.isfinite(sa.count_ni(context))


def test_overfull_excess_accumulated_across_sites_still_voids_the_dpi(
    tmp_path: Path,
) -> None:
    """The threshold is on total excess against Ni, so many small sites add up."""
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = builder.residues[1]
    names = [atom.name for atom in his.atoms if not atom.altloc]
    builder.add_conformers(
        his,
        [("A", 0.46, {}), ("B", 0.55, {})],
        atom_names=names,
    )

    context = sa.load_structure(
        "test", builder.write_pdb(tmp_path / "many_overfull.pdb")
    )

    assert context.occupancy.overfull_site_count == len(names)
    assert context.occupancy.overfull_excess == approx(0.01 * len(names))
    assert context.occupancy.validation_failed is True
    assert math.isnan(sa.count_ni(context))


def test_overfull_alternate_occupancy_makes_dpi_unavailable(tmp_path: Path) -> None:
    """A grossly overfull site still voids DPI: 0.6 excess against ~11 atoms."""
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = builder.residues[1]
    builder.add_conformers(
        his,
        [
            ("A", 0.8, {"NE2": (2.03, 0.0, 0.0)}),
            ("B", 0.8, {"NE2": (2.80, 0.0, 0.0)}),
        ],
        atom_names=["NE2"],
    )

    context = sa.load_structure(
        "test", builder.write_pdb(tmp_path / "overfull_conformers.pdb")
    )
    alternates = [
        atom
        for atom in context.source_atoms
        if atom.residue_name == "HIS" and atom.atom_name == "NE2"
    ]

    assert [atom.occupancy for atom in alternates] == approx([0.8, 0.8])
    assert all(atom.occupancy_valid for atom in alternates)
    assert context.occupancy.invalid_count == 0
    assert context.occupancy.overfull_site_count == 1
    assert context.occupancy.validation_failed is True
    assert "overfull_alternate_occupancy" in context.warning_codes
    assert math.isnan(sa.count_deposited_ni(context))
    assert math.isnan(sa.count_ni(context))


def test_load_structure_flags_altloc_fallback_from_the_file(tmp_path: Path) -> None:
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    his = builder.residues[1]
    builder.add_conformers(his, [("A", 1.50, {}), ("B", 2.50, {})], atom_names=["NE2"])
    path = builder.write_pdb(tmp_path / "fallback.pdb")

    context = sa.load_structure("test", path)
    selection = _residue(context, "HIS")

    assert selection.altloc_selection_fallback is True
    assert selection.selected_altloc == "A"
    assert "altloc_selection_fallback" in context.warning_codes
    assert context.occupancy.validation_failed is True
    assert math.isnan(sa.count_deposited_ni(context))


def _two_model_pdb(path: _StrPath) -> str:
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
    helpers.write_pdb(structure, path)
    return path


def test_load_structure_analyzes_the_first_model_only(tmp_path: Path) -> None:
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
    assert sa.count_deposited_ni(context) == approx(2.0)


def test_load_structure_reports_source_model_count_from_the_original_file(
    tmp_path: Path,
) -> None:
    """Verify first-model extraction preserves the source model count as provenance."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = builder.write_pdb(tmp_path / "stripped.pdb")

    context = sa.load_structure("test", path, source_model_count=8)

    assert context.input_model_count == 8
    assert context.multi_model_structure is True
    assert context.model_analyzed == 1
    assert "multi_model_structure" in context.warning_codes


def test_load_structure_rejects_a_source_model_count_below_the_file(
    tmp_path: Path,
) -> None:
    path = _two_model_pdb(tmp_path / "two_models.pdb")
    with pytest.raises(ValueError, match="source model count"):
        sa.load_structure("test", path, source_model_count=1)


def test_load_structure_single_model_reports_no_multi_model_warning(
    tmp_path: Path,
) -> None:
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
    tmp_path: Path, label: str, field: str, status: str, missing: int, invalid: int
) -> None:
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
    assert context.occupancy.validation_failed is True
    assert context.occupancy.missing_count == missing
    assert context.occupancy.invalid_count == invalid
    assert math.isnan(sa.count_deposited_ni(context))
    assert math.isnan(sa.count_ni(context))

    assert metal.occupancy != 1.0
    assert len(context.contact_atoms) == 2
    assert (metal.x, metal.y, metal.z) == (0.0, 0.0, 0.0)


def test_zero_occupancy_is_valid_for_ni(tmp_path: Path) -> None:
    """Zero occupancy is a measured value: it keeps DPI available and adds 0."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    path = _rewrite_atom_field(
        clean, tmp_path / "zero.pdb", atom_name="ZN", column=_OCC_COLUMN, text="  0.00"
    )

    context = sa.load_structure("test", path)
    metal = [atom for atom in context.source_atoms if atom.element == "ZN"][0]

    assert metal.occupancy == approx(0.0)
    assert metal.occupancy_valid is True
    assert metal.occupancy_status == "valid"
    assert context.occupancy.validation_failed is False
    assert context.occupancy.zero_atom_count == 1
    assert "zero_occupancy_atoms" in context.warning_codes
    assert context.metal_atoms(["ZN"]) == []
    assert context.metal_atoms(["ZN"], include_zero_occupancy=True) == [metal]
    # Water oxygen contributes 1.0, the zero-occupancy metal contributes 0.0.
    assert sa.count_deposited_ni(context) == approx(1.0)


def test_missing_occupancy_is_not_counted_as_a_measured_zero(tmp_path: Path) -> None:
    """Verify missing occupancy differs from a measured zero.

    Gemmi parses the blank column as 0.0, so only raw provenance distinguishes
    "not deposited" from "deposited as zero".
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    path = _rewrite_atom_field(
        clean, tmp_path / "blank.pdb", atom_name="ZN", column=_OCC_COLUMN, text="      "
    )

    context = sa.load_structure("test", path)

    assert context.occupancy.missing_count == 1
    assert context.occupancy.zero_atom_count == 0
    assert "zero_occupancy_atoms" not in context.warning_codes
    assert math.isnan(sa.count_deposited_ni(context))


def test_unknown_element_makes_the_atom_count_indeterminate(tmp_path: Path) -> None:
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
    assert context.records.unknown_element_atom_count == 1
    assert context.records.element_validation_warning == "unknown_element_atoms"
    assert "unknown_elements" in context.warning_codes
    assert context.occupancy.validation_failed is False
    assert math.isnan(sa.count_deposited_ni(context))
    assert math.isnan(sa.count_ni(context))


def test_count_deposited_ni_is_the_occupancy_weighted_heavy_atom_sum(
    tmp_path: Path,
) -> None:
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", occupancy=0.6)
    builder.add_water(101, (0.0, 2.09, 0.0), chain="B", occupancy=0.25)
    builder.add_water(102, (0.0, 4.20, 0.0), chain="B", occupancy=1.0)
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "ni.pdb"))

    assert sa.count_deposited_ni(context) == approx(0.6 + 0.25 + 1.0)


def test_count_deposited_ni_excludes_hydrogen_and_deuterium(tmp_path: Path) -> None:
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
    assert sa.count_deposited_ni(context) == approx(2.0)


def test_count_deposited_ni_counts_alternate_positions_separately(
    tmp_path: Path,
) -> None:
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    water = builder.add_water(101, (0.0, 2.09, 0.0), chain="B")
    builder.add_conformers(water, [("A", 0.3, {}), ("B", 0.5, {"O": (0.0, 2.60, 0.0)})])
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "alt.pdb"))

    water_selection = _residue(context, "HOH")
    assert len(water_selection.source_atoms) == 2
    assert len(water_selection.contact_atoms) == 1  # only B is a contact
    assert sa.count_deposited_ni(context) == approx(1.0 + 0.3 + 0.5)


def test_count_ni_multiplies_by_non_given_strict_ncs_copies(tmp_path: Path) -> None:
    """Verify the DPI multiplier excludes deposited strict-NCS copies.

    Operations flagged ``given`` are already deposited and are not counted a
    second time.
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = _write_pdb_with_ncs(
        builder, tmp_path / "ncs.pdb", [("1", False), ("2", True), ("3", False)]
    )

    context = sa.load_structure("test", path)

    assert context.symmetry.strict_ncs_operation_ids == ("1", "3")
    assert context.symmetry.strict_ncs_operation_count == 2
    assert context.symmetry.dpi_atom_count_multiplier == 3
    deposited = sa.count_deposited_ni(context)
    assert deposited == approx(2.0)
    assert sa.count_ni(context) == approx(6.0)
    assert sa.count_ni(context) == approx(
        deposited * context.symmetry.dpi_atom_count_multiplier
    )


def test_count_ni_equals_deposited_without_strict_ncs(tmp_path: Path) -> None:
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    context = sa.load_structure("test", builder.write_pdb(tmp_path / "plain.pdb"))

    assert context.symmetry.strict_ncs_operation_ids == ()
    assert context.symmetry.dpi_atom_count_multiplier == 1
    assert sa.count_ni(context) == approx(sa.count_deposited_ni(context))


def test_count_ni_stays_unavailable_when_the_deposited_count_is(
    tmp_path: Path,
) -> None:
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

    assert context.symmetry.dpi_atom_count_multiplier == 2
    assert math.isnan(sa.count_deposited_ni(context))
    assert math.isnan(sa.count_ni(context))


def test_duplicate_atom_records_collapse_to_the_higher_occupancy(
    tmp_path: Path,
) -> None:
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
    assert metals[0].occupancy == approx(0.9)
    assert context.records.duplicate_records_present is True
    assert context.records.duplicate_record_count == 1
    assert context.records.coordinate_conflict_count == 0
    assert "duplicate_atom_records" in context.warning_codes
    assert "duplicate_atom_coordinate_conflict" not in context.warning_codes
    assert sa.count_deposited_ni(context) == approx(0.9 + 1.0)


def test_duplicate_atom_records_at_different_positions_are_flagged(
    tmp_path: Path,
) -> None:
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

    assert context.records.duplicate_record_count == 1
    assert context.records.coordinate_conflict_count == 1
    assert "duplicate_atom_coordinate_conflict" in context.warning_codes


def test_load_structure_reads_mmcif_without_raw_pdb_matching(tmp_path: Path) -> None:
    """For mmCIF input the parser is authoritative; no raw-record join runs."""
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)])
    context = sa.load_structure("test", builder.write_cif(tmp_path / "site.cif"))

    assert context.analysis_coordinate_format == "mmcif"
    assert context.occupancy.raw_mapping_failed is False
    assert context.occupancy.raw_mapping_failure_reason == ""
    assert context.occupancy.validation_failed is False
    assert all(atom.occupancy_status == "valid" for atom in context.source_atoms)
    assert sa.count_deposited_ni(context) == approx(float(len(context.source_atoms)))


def test_mmcif_occupancy_out_of_range_still_disables_dpi(tmp_path: Path) -> None:
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", occupancy=1.5)
    builder.add_water(101, (0.0, 2.09, 0.0), chain="B")
    context = sa.load_structure("test", builder.write_cif(tmp_path / "bad.cif"))

    metal = [atom for atom in context.source_atoms if atom.element == "ZN"][0]
    assert metal.occupancy == approx(1.5)
    assert metal.occupancy_valid is False
    assert context.occupancy.validation_failed is True
    assert math.isnan(sa.count_deposited_ni(context))


def test_context_indexes_only_contact_atoms_for_neighbor_marks(tmp_path: Path) -> None:
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


def test_context_residues_are_addressable_by_author_identity(tmp_path: Path) -> None:
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


def test_metal_atoms_uses_selected_conformers_by_default(tmp_path: Path) -> None:
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


def _oxygen_neighbors_of_the_metal(context: sa.StructureContext) -> list[str]:
    search = context.make_neighbor_search(5.0, include_symmetry=False)
    metal = context.metal_atoms(["ZN"])[0]
    found = [
        context.atom_for_mark(mark)
        for mark in search.find_atoms(metal.pos, "\0", radius=5.0)
    ]
    return [
        atom.atom_name for atom in found if atom is not None and atom.element == "O"
    ]


def test_neighbor_search_skips_zero_occupancy_atoms(tmp_path: Path) -> None:
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
    assert context.occupancy.zero_atom_count == 1
    assert sa.count_deposited_ni(context) == approx(1.0)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_neighbor_search_excludes_non_finite_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, non_finite: float
) -> None:
    """Malformed coordinates remain auditable but never reach Gemmi's bins."""
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    path = builder.write_cif(tmp_path / "non_finite_donor.cif")
    parsed = gemmi.read_structure(path)
    donor = next(
        atom
        for chain in parsed[0]
        for residue in chain
        for atom in residue
        if atom.name == "NE2"
    )
    donor.pos = gemmi.Position(non_finite, donor.pos.y, donor.pos.z)

    def read_parsed_structure(_path: str) -> gemmi.Structure:
        return parsed

    monkeypatch.setattr(
        "structure_analysis.gemmi.read_structure", read_parsed_structure
    )

    context = sa.load_structure("test", path)
    invalid = next(atom for atom in context.source_atoms if atom.atom_name == "NE2")
    metal = context.metal_atoms(["ZN"])[0]

    assert invalid.coordinates_valid is False
    assert context.records.non_finite_coordinate_atom_count == 1
    assert "non_finite_coordinates" in context.warning_codes

    search = context.make_neighbor_search(5.0, include_symmetry=False)
    found = [
        context.atom_for_mark(mark) for mark in search.find_atoms(metal.pos, radius=5.0)
    ]
    assert invalid not in found


def test_neighbor_search_requires_symmetry_metadata_when_asked(tmp_path: Path) -> None:
    """An image-inclusive search must fail loudly without a usable space group.

    mmCIF is used because the PDB writer substitutes ``P 1`` on CRYST1.
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)], spacegroup=None)
    context = sa.load_structure("test", builder.write_cif(tmp_path / "nosg.cif"))

    assert context.symmetry.search_available is False
    assert context.symmetry.search_failure_reason == ("missing_or_invalid_space_group")
    with pytest.raises(ValueError, match="space_group"):
        context.make_neighbor_search(4.0, include_symmetry=True)
    assert context.make_neighbor_search(4.0, include_symmetry=False) is not None
