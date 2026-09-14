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

    def test_field_reads_by_name(self) -> None:
        tokens = RESIDUE_LAYOUT.format(
            1, "A", "5B", "SF4", "CC", 12, "_", "SF4X", 3, 4, "NC"
        ).split()
        assert RESIDUE_LAYOUT.field(tokens, "model") == "1"
        assert RESIDUE_LAYOUT.field(tokens, "source_name") == "SF4X"
        assert RESIDUE_LAYOUT.field(tokens, "source_polymer_position") == "NC"

    def test_every_prefix_is_unique(self) -> None:
        prefixes = {
            layout.prefix
            for layout in (
                RESNAME_LAYOUT,
                RESIDUE_LAYOUT,
                POLYMER_LAYOUT,
                OCCUPANCY_DEFAULT_LAYOUT,
            )
        }
        assert len(prefixes) == 4


class TestRoundTrip:
    def test_written_records_are_prepended_in_order_and_read_back(
        self, tmp_path: Path
    ) -> None:
        pdb = _write(tmp_path / "in.pdb", [_BODY])
        pdb_remarks.write_conversion_provenance(
            pdb,
            [(1, "B", "2", "SF4", "SF4X")],
            [(1, "A", "7", "ZN", "C12", 5, "A", "ZN", 3, 0, "-")],
            [(1, "", "1", "GLY", "N"), (1, "B", "2", "SF4", "-")],
            [0, 4],
        )
        with open(pdb, encoding="utf-8", newline="") as handle:
            text = handle.read()
        assert text.endswith(_BODY)
        assert text.splitlines()[:5] == [
            "REMARK 950 ALCHEMY RESNAME 1 B 2 SF4 SF4X",
            "REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 2 4",
            "REMARK 950 ALCHEMY RESIDUE 1 A 7 ZN C12 5 A ZN 3 0 -",
            "REMARK 950 ALCHEMY POLYMER 1 _ 1 GLY N",
            "REMARK 950 ALCHEMY POLYMER 1 B 2 SF4 -",
        ]

        assert pdb_remarks.read_residue_mapping(pdb) == {
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
        assert pdb_remarks.read_defaulted_occupancy_counts(pdb) == {1: 4}

    def test_no_records_leaves_the_file_unchanged(self, tmp_path: Path) -> None:
        pdb = _write(tmp_path / "in.pdb", [_BODY])
        pdb_remarks.write_conversion_provenance(pdb, [])
        with open(pdb, encoding="utf-8", newline="") as handle:
            assert handle.read() == _BODY
        assert pdb_remarks.read_residue_mapping(pdb) == {}
        assert pdb_remarks.read_defaulted_occupancy_counts(pdb) == {}

    def test_residue_record_upgrades_a_matching_resname_record(
        self, tmp_path: Path
    ) -> None:
        """Older files carry both records for one residue; they must agree."""
        pdb = _write(
            tmp_path / "in.pdb",
            [
                RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4X"),
                RESIDUE_LAYOUT.format(
                    1, "A", "1", "SF4", "B", 9, "_", "SF4X", 1, 2, "?"
                ),
                _BODY,
            ],
        )
        mapping = pdb_remarks.read_residue_mapping(pdb)
        assert mapping[(0, "SF4", "A", "1")].residue_number == 9
        assert mapping[(0, "SF4", "A", "1")].polymer_position == "?"


class TestRejectedRecords:
    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("REMARK 950 ALCHEMY RESNAME 1 A 1 SF4\n", "malformed Alchemy residue"),
            ("REMARK 950 ALCHEMY POLYMER 1 A 1 GLY N extra\n", "malformed Alchemy"),
            ("REMARK 950 ALCHEMY RESNAME x A 1 SF4 SF4X\n", "invalid model"),
            ("REMARK 950 ALCHEMY RESNAME 0 A 1 SF4 SF4X\n", "must be positive"),
            ("REMARK 950 ALCHEMY POLYMER 1 A 1 GLY X\n", "invalid source polymer"),
            (
                "REMARK 950 ALCHEMY RESIDUE 1 A 1 ZN B five _ ZN 0 0 -\n",
                "invalid source numeric field",
            ),
            (
                "REMARK 950 ALCHEMY RESIDUE 1 A 1 ZN B 5 _ ZN 0 0 Q\n",
                "invalid source polymer",
            ),
        ],
    )
    def test_malformed_residue_mapping_records(
        self, tmp_path: Path, line: str, message: str
    ) -> None:
        pdb = _write(tmp_path / "in.pdb", [line, _BODY])
        with pytest.raises(ValueError, match=message):
            pdb_remarks.read_residue_mapping(pdb)

    def test_conflicting_resname_records(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4X"),
                RESNAME_LAYOUT.format(1, "A", "1", "SF4", "SF4Y"),
            ],
        )
        with pytest.raises(ValueError, match="conflicting Alchemy residue mappings"):
            pdb_remarks.read_residue_mapping(pdb)

    def test_conflicting_polymer_records(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                POLYMER_LAYOUT.format(1, "A", "1", "GLY", "N"),
                POLYMER_LAYOUT.format(1, "A", "1", "GLY", "M"),
            ],
        )
        with pytest.raises(ValueError, match="conflicting Alchemy polymer mappings"):
            pdb_remarks.read_residue_mapping(pdb)

    @pytest.mark.parametrize(
        ("line", "message"),
        [
            ("REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 1\n", "malformed Alchemy"),
            ("REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 1 x\n", "invalid Alchemy"),
            (
                "REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 1 0\n",
                "positive model and count",
            ),
            (
                "REMARK 950 ALCHEMY OCCUPANCY DEFAULTED 0 3\n",
                "positive model and count",
            ),
        ],
    )
    def test_malformed_occupancy_records(
        self, tmp_path: Path, line: str, message: str
    ) -> None:
        pdb = _write(tmp_path / "in.pdb", [line, _BODY])
        with pytest.raises(ValueError, match=message):
            pdb_remarks.read_defaulted_occupancy_counts(pdb)

    def test_duplicate_occupancy_record_for_one_model(self, tmp_path: Path) -> None:
        pdb = _write(
            tmp_path / "in.pdb",
            [
                OCCUPANCY_DEFAULT_LAYOUT.format(2, 3),
                OCCUPANCY_DEFAULT_LAYOUT.format(2, 3),
            ],
        )
        with pytest.raises(ValueError, match="duplicate .* for model 2"):
            pdb_remarks.read_defaulted_occupancy_counts(pdb)
