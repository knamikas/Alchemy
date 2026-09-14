"""Test the mmCIF-to-PDB and first-model coordinate-preparation converters.

The converted model is what EDSTATS and the structure analysis read, so these
tests check that atom identity, occupancy, and provenance remarks survive it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import gemmi
import helpers
import pytest
from helpers import approx

import coordinate_conversion as conversion
import edstats_statistics
import structure_analysis
from coordination import declared_connections
from pdb_remarks import RESIDUE_REMARK_PREFIX, RESNAME_REMARK_PREFIX

_CIF_HEADER = """data_TEST
_cell.length_a 60.0
_cell.length_b 70.0
_cell.length_c 80.0
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""
_CIF_HEADER_NO_OCCUPANCY = _CIF_HEADER.replace("_atom_site.occupancy\n", "")


def _cif_atom(
    serial: int | str,
    element: str,
    atom_name: str,
    comp_id: str,
    label_asym: str,
    entity: int,
    label_seq: int | str,
    xyz: tuple[float, float, float],
    occupancy: str | None,
    auth_seq: int,
    auth_asym: str,
    group: str = "ATOM",
    with_occupancy: bool = True,
) -> str:
    x, y, z = xyz
    fields = [
        group,
        str(serial),
        element,
        atom_name,
        ".",
        comp_id,
        label_asym,
        str(entity),
        str(label_seq),
        "?",
        f"{x}",
        f"{y}",
        f"{z}",
    ]
    if with_occupancy:
        # ``occupancy`` is None only where the caller also drops the column, so
        # the token appended here is always a string.
        fields.append(cast(str, occupancy))
    fields += ["20.0", str(auth_seq), auth_asym, "1"]
    return " ".join(fields)


def _write_cif(path: Path, atom_lines: Sequence[str], header: str = _CIF_HEADER) -> str:
    text = header + "".join(line + "\n" for line in atom_lines)
    with open(path, "w") as handle:
        handle.write(text)
    return str(path)


def _pdb_atom_lines(path: str | Path) -> list[str]:
    with open(path) as handle:
        return [
            line for line in handle if line[:6].strip().upper() in ("ATOM", "HETATM")
        ]


def _occupancy_field(line: str) -> str:
    """PDB columns 55-60, exactly as written."""
    return line[54:60]


def _element_field(line: str) -> str:
    """PDB columns 77-78, exactly as written."""
    return line[76:78].strip()


def _simple_structure(
    residues: Sequence[tuple[str, int, str, Sequence[tuple[str, str]]]],
) -> gemmi.Structure:
    """Build a one-model gemmi structure from (chain, seqid, name, atoms).

    gemmi's ``add_chain``/``add_residue`` copy their argument, so each chain is
    fully populated before it is attached.
    """
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain_order: list[str] = []
    chains: dict[str, gemmi.Chain] = {}
    for chain_name, seqid, resname, atoms in residues:
        if chain_name not in chains:
            chains[chain_name] = gemmi.Chain(chain_name)
            chain_order.append(chain_name)
        residue = gemmi.Residue()
        residue.name = resname
        residue.seqid = gemmi.SeqId(seqid, " ")
        for atom_name, element in atoms:
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(0.0, 0.0, 0.0)
            residue.add_atom(atom)
        chains[chain_name].add_residue(residue)
    for chain_name in chain_order:
        model.add_chain(chains[chain_name])
    structure.add_model(model)
    return structure


_BASE_RESIDUES = [
    ("A", 1, "GLY", [("N", "N"), ("CA", "C")]),
    ("B", 2, "ZN", [("ZN", "ZN")]),
]


class TestCifToPdb:
    """mmCIF -> analysis PDB conversion must not lose provenance."""

    @staticmethod
    def _standard_cif(tmp_path: Path, name: str = "in.cif") -> str:
        return _write_cif(
            tmp_path / name,
            [
                _cif_atom(
                    1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A"
                ),
                _cif_atom(
                    2, "C", "CA", "GLY", "A", 1, 1, (21.5, 20.0, 20.0), "?", 1, "A"
                ),
                _cif_atom(
                    3, "C", "C", "GLY", "A", 1, 1, (22.0, 21.4, 20.0), ".", 1, "A"
                ),
                _cif_atom(
                    4,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "0.75",
                    1,
                    "B",
                    group="HETATM",
                ),
                _cif_atom(
                    5,
                    "FE",
                    "FE1",
                    "SF4X",
                    "C",
                    3,
                    ".",
                    (5.0, 0.0, 0.0),
                    "1.00",
                    2,
                    "B",
                    group="HETATM",
                ),
            ],
        )

    def test_missing_occupancies_are_blank_not_one(self, tmp_path: Path) -> None:
        """README: '.' and '?' become blank PDB occupancy, never 1.00."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        lines = _pdb_atom_lines(out)
        assert len(lines) == 5
        occupancies = [_occupancy_field(line) for line in lines]
        assert occupancies[1].strip() == ""  # '?'
        assert occupancies[2].strip() == ""  # '.'
        assert occupancies[0].strip() == "1.00"
        assert float(occupancies[3]) == 0.75
        assert float(occupancies[4]) == 1.00

    def test_blanking_occupancy_does_not_shift_the_other_columns(
        self, tmp_path: Path
    ) -> None:
        """Only columns 55-60 change; coordinates, B and element stay put."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        blanked = _pdb_atom_lines(out)[1]
        assert blanked[30:38].strip() == "21.500"
        assert blanked[60:66].strip() == "20.00"
        assert _element_field(blanked) == "C"
        assert blanked[17:20].strip() == "GLY"
        assert len(blanked.rstrip("\n")) >= 78

    def test_absent_occupancy_column_uses_dictionary_default_with_provenance(
        self, tmp_path: Path
    ) -> None:
        cif = _write_cif(
            tmp_path / "noocc.cif",
            [
                _cif_atom(
                    1,
                    "N",
                    "N",
                    "GLY",
                    "A",
                    1,
                    1,
                    (20.0, 20.0, 20.0),
                    None,
                    1,
                    "A",
                    with_occupancy=False,
                ),
                _cif_atom(
                    2,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    None,
                    1,
                    "B",
                    group="HETATM",
                    with_occupancy=False,
                ),
            ],
            header=_CIF_HEADER_NO_OCCUPANCY,
        )
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        for line in _pdb_atom_lines(out):
            assert _occupancy_field(line).strip() == "1.00"

        context = structure_analysis.load_structure("test", out)
        assert context.occupancy.validation_failed is False
        assert context.occupancy.defaulted_atom_count == 2
        assert "occupancy_dictionary_default_applied" in context.warning_codes
        assert structure_analysis.count_deposited_ni(context) == approx(2.0)

    def test_type_symbol_is_written_into_the_pdb_element_field(
        self, tmp_path: Path
    ) -> None:
        """Alchemy reads the element column, never guesses from atom names."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        elements = [_element_field(line) for line in _pdb_atom_lines(out)]
        assert elements == ["N", "C", "C", "ZN", "FE"]

    def test_long_component_ids_are_truncated_but_recorded(
        self, tmp_path: Path
    ) -> None:
        """A >3-character CCD id cannot fit the legacy field, so map it."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        with open(out) as handle:
            remarks = [
                line.split()
                for line in handle
                if line.startswith(RESNAME_REMARK_PREFIX)
            ]
        assert len(remarks) == 1
        fields = remarks[0]
        # REMARK 950 ALCHEMY RESNAME <model> <chain> <resnum> <written> <source>
        assert fields[4:9] == ["1", "B", "2", "SF4", "SF4X"]
        written_names = {line[17:20].strip() for line in _pdb_atom_lines(out)}
        assert "SF4" in written_names
        assert "SF4X" not in written_names

    def test_no_mapping_remark_when_every_name_fits(self, tmp_path: Path) -> None:
        cif = _write_cif(
            tmp_path / "short.cif",
            [
                _cif_atom(
                    1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A"
                ),
                _cif_atom(
                    2,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "1.00",
                    1,
                    "B",
                    group="HETATM",
                ),
            ],
        )
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        with open(out) as handle:
            assert not any(line.startswith(RESNAME_REMARK_PREFIX) for line in handle)

    def test_conversion_round_trips_through_load_structure(
        self, tmp_path: Path
    ) -> None:
        """Verify packed residue names retain source identity.

        The mapping is reversible: the truncated name is what EDSTATS sees,
        while Alchemy's own output keeps the mmCIF identity.
        """
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        context = structure_analysis.load_structure("test", out)
        by_atom = {site.atom_name: site for site in context.source_atoms}

        assert by_atom["FE1"].residue_name == "SF4X"
        assert by_atom["FE1"].coordinate_residue_name == "SF4"
        assert by_atom["FE1"].element == "FE"
        assert by_atom["ZN"].residue_name == "ZN"

        assert by_atom["CA"].occupancy_valid is False
        assert by_atom["C"].occupancy_valid is False
        assert by_atom["N"].occupancy == approx(1.0)
        assert by_atom["ZN"].occupancy == approx(0.75)
        assert context.occupancy.defaulted_atom_count == 0
        assert "occupancy_dictionary_default_applied" not in context.warning_codes

    def test_more_than_62_chains_use_reversible_pdb_safe_residue_ids(
        self, tmp_path: Path
    ) -> None:
        """Every source site survives when one-character chain ids run out."""
        source_chains = [f"C{index:02d}" for index in range(62)]
        source_chains.insert(20, "A")
        cif = _write_cif(
            tmp_path / "many-chains.cif",
            [
                _cif_atom(
                    index + 1,
                    "ZN",
                    "ZN",
                    "ZN",
                    chain,
                    index + 1,
                    ".",
                    (float(index), 0.0, 0.0),
                    "1.00",
                    1,
                    chain,
                    group="HETATM",
                )
                for index, chain in enumerate(source_chains)
            ],
        )

        out = conversion.cif_to_pdb(cif, str(tmp_path / "many-chains.pdb"))
        atom_lines = _pdb_atom_lines(out)
        assert len(atom_lines) == 63
        assert {line[21:22] for line in atom_lines} == {"A"}
        with open(out) as handle:
            identity_remarks = [
                line for line in handle if line.startswith(RESIDUE_REMARK_PREFIX)
            ]
        assert len(identity_remarks) == 63

        context = structure_analysis.load_structure("test", out)
        metals = context.metal_atoms({"ZN"}, canonical=True)
        assert len(metals) == 63
        assert {metal.chain_id for metal in metals} == set(source_chains)
        assert {metal.coordinate_chain_id for metal in metals} == {"A"}
        assert {metal.coordinate_resnum for metal in metals} == {
            str(index) for index in range(1, 64)
        }
        assert {metal.chain_index for metal in metals} == {0}
        assert [metal.output_chain_index for metal in metals] == list(range(63))
        assert {metal.output_residue_index for metal in metals} == {0}
        assert context.occupancy.raw_mapping_failed is False
        assert "legacy_pdb_identifiers_packed" in context.warning_codes
        # A source key can coincide with a different residue's packed key, so
        # the two namespaces are indexed separately.
        assert len(context.residues_for_author("ZN", "A", "1")) == 2
        (source_a,) = context.residues_for_source_author("ZN", "A", "1")
        (coordinate_a1,) = context.residues_for_coordinate_author("ZN", "A", "1")
        assert source_a.chain_id == "A"
        assert coordinate_a1.chain_id == source_chains[0]

        # EDSTATS sees the packed identities, while aggregate rows restore the
        # original mmCIF author identifiers.
        stats_path = helpers.write_edstats_for_structure(
            tmp_path / "stats.out", context, metrics={"ZDm": 2.0}
        )
        rows = edstats_statistics.extract_metal_statistics(
            "test", stats_path, {"ZN"}, set(), structure=context
        ).rows
        assert len(rows) == 63
        assert {row.chain for row in rows} == set(source_chains)
        assert {row.fields[1] for row in rows} == set(source_chains)
        assert {row.resnum for row in rows} == {"1"}

        # Source struct_conn partners resolve through the same provenance,
        # without reproducing the packed numbering.
        source = gemmi.read_structure(cif)
        cra = next(item for item in source[0].all() if item.chain.name == "A")
        resolved = declared_connections.analysis_atom_for_partner(context, cra, {})
        assert resolved is not None
        assert resolved.chain_id == "A"
        assert resolved.resnum == "1"

    def test_duplicate_author_residue_ids_are_reversibly_packed(
        self, tmp_path: Path
    ) -> None:
        """Distinct glycan branches must not merge in the legacy PDB round trip."""
        header = _CIF_HEADER.replace(
            "loop_\n_atom_site.group_PDB",
            """loop_
_entity.id
_entity.type
3 branched
4 branched
6 branched
loop_
_struct_asym.id
_struct_asym.entity_id
G 3
H 4
K 6
loop_
_atom_site.group_PDB""",
        )
        cif = _write_cif(
            tmp_path / "duplicate-residues.cif",
            [
                _cif_atom(
                    1,
                    "C",
                    "C1",
                    "MAN",
                    "H",
                    4,
                    ".",
                    (0.0, 0.0, 0.0),
                    "1.00",
                    7,
                    "C",
                    group="HETATM",
                ),
                _cif_atom(
                    2,
                    "C",
                    "C1",
                    "MAN",
                    "G",
                    3,
                    ".",
                    (5.0, 0.0, 0.0),
                    "1.00",
                    1,
                    "D",
                    group="HETATM",
                ),
                _cif_atom(
                    3,
                    "C",
                    "C1",
                    "MAN",
                    "K",
                    6,
                    ".",
                    (10.0, 0.0, 0.0),
                    "1.00",
                    7,
                    "C",
                    group="HETATM",
                ),
            ],
            header=header,
        )

        out = conversion.cif_to_pdb(cif, str(tmp_path / "packed.pdb"))
        context = structure_analysis.load_structure("test", out)
        residues = context.residues_for_source_author("MAN", "C", "7")

        assert len(_pdb_atom_lines(out)) == 3
        assert len(residues) == 2
        assert {r.coordinate_chain_id for r in residues} == {"A"}
        assert len({r.coordinate_resnum for r in residues}) == 2
        assert "legacy_pdb_identifiers_packed" in context.warning_codes

    def test_missing_input_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            conversion.cif_to_pdb(str(tmp_path / "gone.cif"), str(tmp_path / "out.pdb"))

    def test_duplicate_atom_site_id_is_rejected(self, tmp_path: Path) -> None:
        """Serials key the occupancy restoration; duplicates make it ambiguous."""
        cif = _write_cif(
            tmp_path / "dup.cif",
            [
                _cif_atom(
                    1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A"
                ),
                _cif_atom(
                    1,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "?",
                    1,
                    "B",
                    group="HETATM",
                ),
            ],
        )
        with pytest.raises(ValueError, match="duplicate mmCIF atom_site id"):
            conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))

    def test_code_atom_site_ids_are_mapped_to_generated_pdb_serials(
        self, tmp_path: Path
    ) -> None:
        cif = _write_cif(
            tmp_path / "code-ids.cif",
            [
                _cif_atom(
                    "A1", "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "0.25", 1, "A"
                ),
                _cif_atom(
                    "metal-B",
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "0.75",
                    1,
                    "B",
                    group="HETATM",
                ),
                _cif_atom(
                    "atom-two",
                    "C",
                    "C",
                    "ALA",
                    "A",
                    1,
                    2,
                    (21.5, 20.0, 20.0),
                    "?",
                    2,
                    "A",
                ),
            ],
        )
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        lines = _pdb_atom_lines(out)

        # Gemmi groups the two A-chain rows even though the B-chain row was
        # interleaved in atom_site. Generated serials preserve the source row
        # association through that reorder.
        assert [line[12:16].strip() for line in lines] == ["N", "C", "ZN"]
        assert [int(line[6:11]) for line in lines] == [1, 2, 4]
        assert float(_occupancy_field(lines[0])) == approx(0.25)
        assert _occupancy_field(lines[1]).strip() == ""
        assert float(_occupancy_field(lines[2])) == approx(0.75)

    def test_multiple_atom_site_blocks_are_rejected(self, tmp_path: Path) -> None:
        atoms = [
            _cif_atom(1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A")
        ]
        text = (
            _CIF_HEADER
            + "".join(line + "\n" for line in atoms)
            + "\n"
            + _CIF_HEADER.replace("data_TEST", "data_SECOND")
            + "".join(line + "\n" for line in atoms)
        )
        path = tmp_path / "two.cif"
        path.write_text(text)
        with pytest.raises(ValueError, match="exactly one block"):
            conversion.cif_to_pdb(str(path), str(tmp_path / "out.pdb"))

    def test_cif_without_atom_records_is_rejected(self, tmp_path: Path) -> None:
        """A metadata-only mmCIF is not a coordinate file."""
        path = tmp_path / "meta.cif"
        path.write_text("data_TEST\n_cell.length_a 60.0\n")
        with pytest.raises(ValueError, match="exactly one block"):
            conversion.cif_to_pdb(str(path), str(tmp_path / "out.pdb"))

    def test_creates_the_destination_directory(self, tmp_path: Path) -> None:
        cif = self._standard_cif(tmp_path)
        dst = tmp_path / "nested" / "deeper" / "out.pdb"
        out = conversion.cif_to_pdb(cif, str(dst))
        assert out == str(dst)
        assert os.path.isfile(out)


class TestResidueConversionRecords:
    """The identity assertions ``cif_to_pdb`` enforces on gemmi's writer."""

    def test_identical_structures_produce_no_records(self) -> None:
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(_BASE_RESIDUES)
        assert conversion.residue_conversion_records(source, converted) == []

    def test_a_renamed_residue_is_recorded_with_its_author_identity(self) -> None:
        """The record must locate the residue the way the PDB reader will."""
        source = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("B", 7, "SF4X", [("FE1", "FE")]),
            ]
        )
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("B", 7, "SF4", [("FE1", "FE")]),
            ]
        )
        assert conversion.residue_conversion_records(source, converted) == [
            (1, "B", "7", "SF4", "SF4X")
        ]

    def test_reordering_is_rejected(self) -> None:
        """EDSTATS joins by position-independent identity only if order holds."""
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(list(reversed(_BASE_RESIDUES)))
        with pytest.raises(ValueError, match="changed residue ordering"):
            conversion.residue_conversion_records(source, converted)

    def test_changed_author_identifiers_are_rejected(self) -> None:
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N"), ("CA", "C")]),
                ("B", 99, "ZN", [("ZN", "ZN")]),
            ]
        )
        with pytest.raises(ValueError, match="ordering|author identifiers"):
            conversion.residue_conversion_records(source, converted)

    def test_changed_atom_membership_is_rejected(self) -> None:
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("B", 2, "ZN", [("ZN", "ZN")]),
            ]
        )
        with pytest.raises(ValueError, match="atom membership"):
            conversion.residue_conversion_records(source, converted)

    def test_changed_duplicate_multiplicity_is_rejected(self) -> None:
        source = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("A", 1, "ALA", [("N", "N")]),
            ]
        )
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
            ]
        )
        with pytest.raises(ValueError, match="ordering|multiplicity"):
            conversion.residue_conversion_records(source, converted)

    def test_the_index_keys_on_model_chain_and_author_resnum(self) -> None:
        """Residues are located by the identifiers EDSTATS also reports."""
        structure = _simple_structure(_BASE_RESIDUES)
        index, order = conversion.residue_index_by_author(structure, "mmCIF")
        assert order == [(0, "A", "1"), (0, "B", "2")]
        assert index[(0, "B", "2")] == [("ZN", (("ZN", "Zn"),))]

    def test_duplicate_author_ids_are_indexed_together_in_order(self) -> None:
        structure = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("A", 1, "ALA", [("N", "N")]),
            ]
        )
        index, order = conversion.residue_index_by_author(structure, "mmCIF")
        assert order == [(0, "A", "1"), (0, "A", "1")]
        assert [name for name, _ in index[(0, "A", "1")]] == ["GLY", "ALA"]


_MULTI_MODEL_PDB = """\
HEADER    TEST
NUMMDL    2
REMARK   3 SOMETHING
CRYST1   60.000   70.000   80.000  90.00  90.00  90.00 P 21 21 21
MODEL        1
ATOM      1  N   GLY A   1      20.000  20.000  20.000  1.00 20.00           N
ATOM      2  CA  GLY A   1      21.500  20.000  20.000  1.00 20.00           C
ENDMDL
MODEL        2
ATOM      1  N   GLY A   1      30.000  30.000  30.000  1.00 20.00           N
ATOM      2  CA  GLY A   1      31.500  30.000  30.000  1.00 20.00           C
ENDMDL
MASTER        0    0    0    0    0    0    0    0    2    1    0    0
END
"""


class TestFirstModelPdb:
    """Textual first-model extraction for EDSTATS."""

    def test_single_model_file_is_used_in_place(self, tmp_path: Path) -> None:
        builder = helpers.simple_metal_site(
            "ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)]
        )
        source = builder.write_pdb(tmp_path / "site.pdb")
        dst = tmp_path / "first.pdb"
        path, count = conversion.first_model_pdb(source, str(dst))
        assert path == source
        assert count == 1
        assert not dst.exists()

    def test_extracts_only_the_first_model(self, tmp_path: Path) -> None:
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        path, count = conversion.first_model_pdb(str(source), str(dst))
        assert path == str(dst)
        assert count == 2

        text = dst.read_text()
        atom_lines = _pdb_atom_lines(dst)
        assert len(atom_lines) == 2
        assert [line[30:38].strip() for line in atom_lines] == ["20.000", "21.500"]
        assert "30.000" not in text

    def test_model_wrappers_are_removed(self, tmp_path: Path) -> None:
        """EDSTATS emits a synthetic separator residue for any MODEL wrapper."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        for line in dst.read_text().splitlines():
            assert line[:6].strip().upper() not in ("MODEL", "ENDMDL")
        assert gemmi.read_structure(str(dst)).__len__() == 1

    def test_nummdl_is_dropped_but_crystallographic_header_is_kept(
        self, tmp_path: Path
    ) -> None:
        """NUMMDL would lie about a one-model file; CRYST1 must survive."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        lines = dst.read_text().splitlines()
        assert not any(line.startswith("NUMMDL") for line in lines)
        assert any(line.startswith("CRYST1") for line in lines)
        assert any(line.startswith("HEADER") for line in lines)
        assert any(line.startswith("REMARK   3") for line in lines)

    def test_trailing_records_after_the_first_model_are_dropped(
        self, tmp_path: Path
    ) -> None:
        """Bookkeeping records describe the full ensemble, not this extract."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        lines = dst.read_text().splitlines()
        assert not any(line.startswith("MASTER") for line in lines)
        assert lines[-1] == "END"

    def test_the_extract_preserves_records_byte_for_byte(self, tmp_path: Path) -> None:
        """Extraction is textual so occupancies and identifiers are untouched."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        source_atom_lines = [
            line for line in _MULTI_MODEL_PDB.splitlines() if line.startswith("ATOM")
        ]
        assert [
            line.rstrip("\n") for line in _pdb_atom_lines(dst)
        ] == source_atom_lines[:2]

    def test_creates_the_destination_directory(self, tmp_path: Path) -> None:
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "nested" / "first.pdb"
        path, _ = conversion.first_model_pdb(str(source), str(dst))
        assert os.path.isfile(path)

    def test_reported_model_count_is_the_source_ensemble_size(
        self, tmp_path: Path
    ) -> None:
        """The manifest's input_model_count describes the deposited file."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        _, count = conversion.first_model_pdb(str(source), str(tmp_path / "first.pdb"))
        assert count == 2
        assert len(gemmi.read_structure(str(source))) == count
