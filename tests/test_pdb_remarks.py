"""Test the ``REMARK 950 ALCHEMY`` provenance writer and parser together.

The writer and the parser share one set of record layouts, so these tests
check that what one writes the other reads back, and that malformed or
contradictory records are rejected rather than silently misassigned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pdb_remarks
from pdb_remarks import (
    OCCUPANCY_DEFAULT_LAYOUT,
    POLYMER_LAYOUT,
    RESIDUE_LAYOUT,
    RESNAME_LAYOUT,
    RemarkLayout,
    SourceResidueIdentity,
)

_BODY = "ATOM      1  N   GLY A   1      20.000  20.000  20.000  1.00 20.00           N\nEND\n"


def _write(path: Path, lines: list[str]) -> str:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)
    return str(path)


def _residue_mapping(
    path: str,
) -> dict[tuple[int, str, str, str], SourceResidueIdentity]:
    return pdb_remarks.read_conversion_provenance(path).residue_mapping


def _occupancy_counts(path: str) -> dict[int, int]:
    return pdb_remarks.read_conversion_provenance(path).defaulted_occupancy_counts


class TestRemarkLayout:
    @pytest.mark.parametrize(
        ("layout", "token_count"),
        [
            (RESNAME_LAYOUT, 9),
            (RESIDUE_LAYOUT, 15),
            (POLYMER_LAYOUT, 9),
            (OCCUPANCY_DEFAULT_LAYOUT, 7),
        ],
    )
    def test_token_counts_match_the_on_disk_format(
        self, layout: RemarkLayout, token_count: int
    ) -> None:
        assert layout.token_count == token_count
        line = layout.format(*range(len(layout.fields)))
        assert line.endswith("\n")
        assert len(line.split()) == token_count
        assert layout.matches(line.split())

    def test_format_rejects_the_wrong_number_of_fields(self) -> None:
        with pytest.raises(ValueError, match="takes 2 fields, got 1"):
            OCCUPANCY_DEFAULT_LAYOUT.format(1)

    @pytest.mark.parametrize("value", ["", " ", "A B", "SF4\t", " X"])
    def test_format_rejects_values_that_do_not_split_back_into_one_token(
        self, value: str
    ) -> None:
        with pytest.raises(ValueError, match="'source_name' must be one non-empty"):
            RESNAME_LAYOUT.format(1, "A", "1", "SF4", value)

    def test_field_reads_by_name(self) -> None:
        tokens = RESIDUE_LAYOUT.format(
            1, "A", "5B", "SF4", "CC", 12, "", "SF4X", 3, 4, "NC"
        ).split()
        assert RESIDUE_LAYOUT.field(tokens, "model") == "1"
        assert RESIDUE_LAYOUT.field(tokens, "source_name") == "SF4X"
        assert RESIDUE_LAYOUT.field(tokens, "source_insertion") == ""
        assert RESIDUE_LAYOUT.field(tokens, "source_polymer_position") == "NC"

    @pytest.mark.parametrize(
        ("value", "token"), [("", "_"), ("_", "__"), ("__", "___"), ("A_", "A_")]
    )
    def test_placeholder_fields_round_trip_empty_and_underscore_values(
        self, value: str, token: str
    ) -> None:
        tokens = POLYMER_LAYOUT.format(1, value, "1", "GLY", "N").split()
        assert tokens[5] == token
        assert POLYMER_LAYOUT.field(tokens, "chain") == value

    def test_non_placeholder_fields_keep_underscores_verbatim(self) -> None:
        tokens = RESNAME_LAYOUT.format(1, "A", "1", "SF4", "_").split()
        assert RESNAME_LAYOUT.field(tokens, "source_name") == "_"

    def test_placeholder_fields_must_be_layout_fields(self) -> None:
        with pytest.raises(ValueError, match="unknown placeholder fields"):
            RemarkLayout("REMARK 950 X", ("a",), frozenset({"b"}))

    def test_every_prefix_is_unique(self) -> None:
        assert len({layout.prefix for layout in pdb_remarks.LAYOUTS}) == 4


class TestWriter:
    def test_records_are_rendered_in_on_disk_order(self) -> None:
        remarks = pdb_remarks.conversion_provenance_remarks(
            [(1, "B", "2", "SF4", "SF4X")],
            [(1, "A", "7", "ZN", "C12", 5, "A", "ZN", 3, 0, "-")],
            [(1, "", "1", "GLY", "N"), (1, "B", "2", "SF4", "-")],
            [0, 4],
        )
        assert [line.rstrip("\n") for line in remarks] == [
            "REMARK 950 ALCHEMY RESNAME 1 B 2 SF4 SF4X",
            "REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 2 4",
            "REMARK 950 ALCHEMY RESIDUE 1 A 7 ZN C12 5 A ZN 3 0 -",
            "REMARK 950 ALCHEMY POLYMER 1 _ 1 GLY N",
            "REMARK 950 ALCHEMY POLYMER 1 B 2 SF4 -",
        ]

    def test_no_records_produce_no_lines(self) -> None:
        assert pdb_remarks.conversion_provenance_remarks([]) == []

    def test_negative_occupancy_counts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative defaulted occupancy count"):
            pdb_remarks.conversion_provenance_remarks(
                [], defaulted_occupancy_counts=[0, -1]
            )

    def test_unknown_polymer_positions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid polymer position 'X'"):
            pdb_remarks.conversion_provenance_remarks(
                [], polymer_records=[(1, "A", "1", "GLY", "X")]
            )
        with pytest.raises(ValueError, match="invalid polymer position 'X'"):
            pdb_remarks.conversion_provenance_remarks(
                [], [(1, "A", "1", "GLY", "B", 1, "", "GLY", 0, 0, "X")]
            )


class TestRoundTrip:
    def test_written_records_are_read_back(self, tmp_path: Path) -> None:
        remarks = pdb_remarks.conversion_provenance_remarks(
            [(1, "B", "2", "SF4", "SF4X")],
            [(1, "A", "7", "ZN", "C12", 5, "A", "ZN", 3, 0, "-")],
            [(1, "", "1", "GLY", "N"), (1, "B", "2", "SF4", "-")],
            [0, 4],
        )
        pdb = _write(tmp_path / "in.pdb", [*remarks, _BODY])
        provenance = pdb_remarks.read_conversion_provenance(pdb)
        assert provenance.residue_mapping == {
            (0, "SF4", "B", "2"): SourceResidueIdentity(
                residue_name="SF4X", chain_id="B", polymer_position="-"
            ),
            (0, "ZN", "A", "7"): SourceResidueIdentity(
                residue_name="ZN",
                chain_id="C12",
                residue_number=5,
                insertion_code="A",
                polymer_position="-",
                chain_index=3,
                residue_index=0,
            ),
            (0, "GLY", "", "1"): SourceResidueIdentity(
                residue_name="GLY", chain_id="", polymer_position="N"
            ),
        }
        assert provenance.defaulted_occupancy_counts == {1: 4}

    def test_empty_and_underscore_source_fields_are_restored(
        self, tmp_path: Path
    ) -> None:
        remarks = pdb_remarks.conversion_provenance_remarks(
            [],
            [
                (1, "A", "1", "ZN", "", 5, "", "ZN", 0, 0, "-"),
                (1, "B", "1", "ZN", "_", 6, "_", "ZN", 1, 0, "-"),
            ],
        )
        pdb = _write(tmp_path / "in.pdb", [*remarks, _BODY])
        mapping = _residue_mapping(pdb)
        assert (
            mapping[(0, "ZN", "A", "1")].chain_id,
            mapping[(0, "ZN", "A", "1")].insertion_code,
        ) == ("", "")
        assert (
            mapping[(0, "ZN", "B", "1")].chain_id,
            mapping[(0, "ZN", "B", "1")].insertion_code,
        ) == ("_", "_")

    def test_file_without_records_reads_as_empty(self, tmp_path: Path) -> None:
        pdb = _write(tmp_path / "in.pdb", [_BODY])
        provenance = pdb_remarks.read_conversion_provenance(pdb)
        assert provenance.residue_mapping == {}
        assert provenance.defaulted_occupancy_counts == {}

    def test_records_after_the_coordinates_are_honoured(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [_BODY, RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4X")],
        )
        assert _residue_mapping(pdb)[(0, "SF4", "A", "1")].residue_name == "SF4X"

    @pytest.mark.parametrize("residue_first", [False, True])
    def test_resname_and_residue_records_merge_in_either_order(
        self, tmp_path: Path, residue_first: bool
    ) -> None:
        """A renamed and packed residue carries both records; they must agree."""
        lines = [
            RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4X"),
            RESIDUE_LAYOUT.format(1, "A", "1", "SF4", "B", 9, "", "SF4X", 1, 2, "?"),
        ]
        if residue_first:
            lines.reverse()
        pdb = _write(tmp_path / "in.pdb", [*lines, _BODY])
        assert _residue_mapping(pdb)[(0, "SF4", "A", "1")] == SourceResidueIdentity(
            residue_name="SF4X",
            chain_id="B",
            residue_number=9,
            polymer_position="?",
            chain_index=1,
            residue_index=2,
        )

    def test_identical_duplicate_records_are_accepted(self, tmp_path: Path) -> None:
        line = RESIDUE_LAYOUT.format(1, "A", "1", "SF4", "B", 9, "", "SF4X", 1, 2, "M")
        pdb = _write(tmp_path / "in.pdb", [line, line, _BODY])
        assert _residue_mapping(pdb)[(0, "SF4", "A", "1")].residue_number == 9

    def test_polymer_record_agreeing_with_residue_record_is_accepted(
        self, tmp_path: Path
    ) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                POLYMER_LAYOUT.format(1, "A", "1", "ZN", "-"),
                RESIDUE_LAYOUT.format(1, "A", "1", "ZN", "B", 5, "", "ZN", 0, 0, "-"),
            ],
        )
        assert _residue_mapping(pdb)[(0, "ZN", "A", "1")].polymer_position == "-"


class TestRejectedRecords:
    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("REMARK 950 ALCHEMY RESNAME 1 A 1 SF4\n", "malformed Alchemy provenance"),
            ("REMARK 950 ALCHEMY POLYMER 1 A 1 GLY N extra\n", "malformed Alchemy"),
            ("REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 1\n", "malformed Alchemy"),
            ("REMARK 950 ALCHEMY RESNAME x A 1 SF4 SF4X\n", "invalid model"),
            ("REMARK 950 ALCHEMY RESNAME 1_0 A 1 SF4 SF4X\n", "invalid model"),
            ("REMARK 950 ALCHEMY RESNAME +1 A 1 SF4 SF4X\n", "invalid model"),
            ("REMARK 950 ALCHEMY RESNAME 0 A 1 SF4 SF4X\n", "model must be positive"),
            ("REMARK 950 ALCHEMY POLYMER 1 A 1 GLY X\n", "invalid polymer position"),
            (
                "REMARK 950 ALCHEMY RESIDUE 1 A 1 ZN B five _ ZN 0 0 -\n",
                "invalid source number",
            ),
            (
                "REMARK 950 ALCHEMY RESIDUE 1 A 1 ZN B 5 _ ZN 0 0 Q\n",
                "invalid polymer position",
            ),
            ("REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 1 x\n", "invalid count"),
            (
                "REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 1 0\n",
                "count must be positive",
            ),
            (
                "REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 0 3\n",
                "model must be positive",
            ),
        ],
    )
    def test_malformed_records(self, tmp_path: Path, line: str, message: str) -> None:
        pdb = _write(tmp_path / "in.pdb", [line, _BODY])
        with pytest.raises(ValueError, match=message):
            pdb_remarks.read_conversion_provenance(pdb)

    def test_negative_source_numbers_are_valid(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [RESIDUE_LAYOUT.format(1, "A", "1", "ZN", "B", -3, "", "ZN", 0, 0, "-")],
        )
        assert _residue_mapping(pdb)[(0, "ZN", "A", "1")].residue_number == -3

    def test_other_remark_950_records_are_ignored(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            ["REMARK 950 ALCHEMY FUTURE 1 2\n", "REMARK 950 OTHER TOOL\n", _BODY],
        )
        assert _residue_mapping(pdb) == {}

    def test_conflicting_resname_records(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4X"),
                RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4Y"),
            ],
        )
        with pytest.raises(
            ValueError,
            match="conflicting Alchemy residue mappings for model 1 residue SF4/A/1",
        ):
            pdb_remarks.read_conversion_provenance(pdb)

    def test_conflicting_residue_records(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                RESIDUE_LAYOUT.format(
                    1, "A", "1", "SF4", "B", 9, "", "SF4X", 1, 2, "M"
                ),
                RESIDUE_LAYOUT.format(
                    1, "A", "1", "SF4", "C", 9, "", "SF4X", 1, 2, "M"
                ),
            ],
        )
        with pytest.raises(ValueError, match="conflicting Alchemy residue mappings"):
            pdb_remarks.read_conversion_provenance(pdb)

    def test_resname_disagreeing_with_residue_record(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                RESIDUE_LAYOUT.format(
                    1, "A", "1", "SF4", "B", 9, "", "SF4X", 1, 2, "M"
                ),
                RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4Y"),
            ],
        )
        with pytest.raises(ValueError, match="conflicting Alchemy residue mappings"):
            pdb_remarks.read_conversion_provenance(pdb)

    def test_conflicting_polymer_records(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                POLYMER_LAYOUT.format(1, "", "1", "GLY", "N"),
                POLYMER_LAYOUT.format(1, "", "1", "GLY", "M"),
            ],
        )
        with pytest.raises(
            ValueError,
            match="conflicting Alchemy polymer mappings for model 1 residue GLY/_/1",
        ):
            pdb_remarks.read_conversion_provenance(pdb)

    def test_polymer_record_disagreeing_with_residue_record(
        self, tmp_path: Path
    ) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                RESIDUE_LAYOUT.format(1, "A", "1", "ZN", "B", 5, "", "ZN", 0, 0, "-"),
                POLYMER_LAYOUT.format(1, "A", "1", "ZN", "M"),
            ],
        )
        with pytest.raises(ValueError, match="conflicting Alchemy polymer mappings"):
            pdb_remarks.read_conversion_provenance(pdb)

    def test_duplicate_occupancy_record_for_one_model(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                OCCUPANCY_DEFAULT_LAYOUT.format(2, 3),
                OCCUPANCY_DEFAULT_LAYOUT.format(2, 3),
            ],
        )
        with pytest.raises(ValueError, match="duplicate .* for model 2"):
            pdb_remarks.read_conversion_provenance(pdb)
