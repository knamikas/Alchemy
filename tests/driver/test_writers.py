"""Test the streamed CSV output writers: headers, row counts, and flushing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TextIO

import pytest
from helpers import entry_result, read_csv

import edstats_statistics
import score
from coordination import schema as coordination_schema
from driver import resume
from driver.writers import (
    MANIFEST_COLUMNS,
    STATS_COLUMNS,
    OutputWriters,
    manifest_row,
)
from output_rows import MetalStatsRow


class TestOutputWriters:
    """The streamed CSVs: headers on creation, running counts, schema guards."""

    @staticmethod
    def _handles(
        tmp_path: Path,
        bonds: bool = True,
        candidates: bool = True,
        score: bool | None = None,
    ) -> dict[str, TextIO]:
        """Open output files keyed as ``OutputTargets`` names them."""
        wanted = {
            "manifest": True,
            "stats": True,
            "bonds": bonds,
            "candidates": candidates,
            "score": bool(score),
        }
        return {
            name: open(tmp_path / f"{name}.csv", "w", newline="")
            for name, present in wanted.items()
            if present
        }

    @staticmethod
    def _close(handles: dict[str, TextIO]) -> None:
        for handle in handles.values():
            handle.close()

    def test_headers_survive_a_run_that_produced_no_rows(self, tmp_path: Path) -> None:
        """README: the CSVs keep their headers when nothing was found."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        writers.write_stats_rows([])
        writers.write_bond_rows([])
        writers.write_candidate_rows([])
        self._close(handles)

        assert read_csv(tmp_path / "manifest.csv") == [MANIFEST_COLUMNS]
        assert read_csv(tmp_path / "stats.csv") == [list(STATS_COLUMNS)]
        assert read_csv(tmp_path / "bonds.csv") == [
            list(coordination_schema.BOND_COLUMNS)
        ]
        assert read_csv(tmp_path / "candidates.csv") == [
            list(coordination_schema.CANDIDATE_COLUMNS)
        ]
        assert (writers.n_sites, writers.n_bonds, writers.n_candidates) == (0, 0, 0)

    def test_running_counts_track_the_rows_actually_written(
        self, tmp_path: Path
    ) -> None:
        """The end-of-run totals come from these counters, not from re-reading."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        stats_rows = [
            MetalStatsRow.from_output_fields(
                "109m", "metal", ["x"] * (len(STATS_COLUMNS) - 2)
            )
            for _ in range(3)
        ]
        bond_rows = [
            coordination_schema.BondRow(
                dict.fromkeys(coordination_schema.BOND_COLUMNS, "")
            )
            for _ in range(4)
        ]
        candidate_rows = [
            coordination_schema.CandidateRow(
                dict.fromkeys(coordination_schema.CANDIDATE_COLUMNS, "")
            )
            for _ in range(2)
        ]
        writers.write_stats_rows(stats_rows)
        writers.write_bond_rows(bond_rows)
        writers.write_candidate_rows(candidate_rows)
        writers.write_stats_rows(stats_rows)
        self._close(handles)

        assert writers.n_sites == 6
        assert writers.n_bonds == 4
        assert writers.n_candidates == 2
        assert len(read_csv(tmp_path / "stats.csv")) == 7
        assert len(read_csv(tmp_path / "bonds.csv")) == 5
        assert len(read_csv(tmp_path / "candidates.csv")) == 3

    def test_stats_rows_are_projected_onto_the_fixed_schema(
        self, tmp_path: Path
    ) -> None:
        """Id and category lead each row; the EDSTATS block follows verbatim."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        fields = [str(i) for i in range(len(STATS_COLUMNS) - 2)]
        writers.write_stats_rows(
            [MetalStatsRow.from_output_fields("109m", "cofactor", fields)]
        )
        self._close(handles)
        rows = read_csv(tmp_path / "stats.csv")
        assert rows[1] == ["109m", "cofactor"] + fields
        assert len(rows[1]) == len(STATS_COLUMNS)

    def test_density_context_distinguishes_uncomputed_from_empty_groups(
        self, tmp_path: Path
    ) -> None:
        """Every entry gets one row, while unavailable values remain blank."""
        handles = self._handles(tmp_path)
        context_handle = open(tmp_path / "density-context.csv", "w", newline="")
        writers = OutputWriters({**handles, "density_context": context_handle})
        writers.write_density_context_row(entry_result("109m"))
        measured: dict[str, Any] = dict.fromkeys(
            edstats_statistics.DENSITY_CONTEXT_COLUMNS, ""
        )
        measured.update(
            pdbID="1cll",
            density_context_status="available",
            edstats_residue_count=12,
            target_residue_count=1,
            ordinary_residue_count=11,
            ordinary_rszd_count=0,
        )
        writers.write_density_context_row(
            entry_result("1cll", density_context_row=measured)
        )
        self._close(handles)
        context_handle.close()

        with open(tmp_path / "density-context.csv", newline="") as context_input:
            rows = list(csv.DictReader(context_input))
        assert rows[0]["pdbID"] == "109m"
        assert rows[0]["density_context_status"] == "not_computed"
        assert rows[0]["ordinary_residue_count"] == ""
        assert rows[1]["density_context_status"] == "available"
        assert rows[1]["ordinary_residue_count"] == "11"
        assert rows[1]["ordinary_rszd_count"] == "0"
        assert writers.n_density_contexts == 2

    def test_stats_batch_is_validated_before_any_rows_are_written(
        self, tmp_path: Path
    ) -> None:
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        fields = [""] * (len(STATS_COLUMNS) - 2)
        rows = [
            MetalStatsRow.from_output_fields("109m", "metal", fields),
            MetalStatsRow.from_output_fields("1cll", "metal", fields[:-1]),
        ]
        try:
            with pytest.raises(ValueError):
                writers.write_stats_rows(rows)
        finally:
            self._close(handles)
        assert writers.n_sites == 0
        assert read_csv(tmp_path / "stats.csv") == [STATS_COLUMNS]

    def test_bond_rows_are_written_in_schema_order(self, tmp_path: Path) -> None:
        """Verify projected row values follow the CSV column order."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        row = {column: f"v-{column}" for column in coordination_schema.BOND_COLUMNS}
        shuffled = {key: row[key] for key in reversed(list(row))}
        writers.write_bond_rows([coordination_schema.BondRow(shuffled)])
        self._close(handles)
        written = read_csv(tmp_path / "bonds.csv")[1]
        assert written == [f"v-{column}" for column in coordination_schema.BOND_COLUMNS]

    def test_scientific_csv_scalars_use_portable_missing_and_boolean_values(
        self, tmp_path: Path
    ) -> None:
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        bond: dict[str, object] = dict.fromkeys(coordination_schema.BOND_COLUMNS, "")
        bond["declared_connection"] = True
        bond["geometry_outlier"] = False
        bond["zscore"] = float("nan")
        writers.write_bond_rows([coordination_schema.BondRow(bond)])
        self._close(handles)

        written = dict(
            zip(
                coordination_schema.BOND_COLUMNS,
                read_csv(tmp_path / "bonds.csv")[1],
                strict=True,
            )
        )
        assert written["declared_connection"] == "true"
        assert written["geometry_outlier"] == "false"
        assert written["zscore"] == ""

    def test_disabled_bond_outputs_are_a_no_op_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        """--no-bonds passes None handles; writes must be silently skipped."""
        handles = self._handles(tmp_path, bonds=False, candidates=False)
        writers = OutputWriters(handles)
        writers.write_bond_rows(
            [
                coordination_schema.BondRow(
                    dict.fromkeys(coordination_schema.BOND_COLUMNS, "")
                )
            ]
        )
        writers.write_candidate_rows(
            [
                coordination_schema.CandidateRow(
                    dict.fromkeys(coordination_schema.CANDIDATE_COLUMNS, "")
                )
            ]
        )
        self._close(handles)
        assert writers.n_bonds == 0
        assert writers.n_candidates == 0
        assert not (tmp_path / "bonds.csv").exists()
        assert not (tmp_path / "candidates.csv").exists()

    @pytest.mark.parametrize(
        "mutate,columns_name",
        [
            ("drop", "BOND_COLUMNS"),
            ("add", "BOND_COLUMNS"),
        ],
    )
    def test_bond_row_schema_drift_fails_loudly(
        self, tmp_path: Path, mutate: str, columns_name: str
    ) -> None:
        """A silently dropped or ignored column would corrupt every later row."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        row = dict.fromkeys(getattr(coordination_schema, columns_name), "")
        if mutate == "drop":
            drifted = next(iter(row))
            row.pop(drifted)
            expected_clause = f"missing {drifted}"
        else:
            drifted = "unexpected_column"
            row[drifted] = ""
            expected_clause = f"unexpected {drifted}"
        try:
            with pytest.raises(RuntimeError) as excinfo:
                coordination_schema.BondRow(row)
        finally:
            self._close(handles)
        message = str(excinfo.value)
        assert "metal_bonds_all.csv" in message
        # Name the mismatched column so schema failures are actionable.
        assert expected_clause in message
        assert writers.n_bonds == 0

    def test_candidate_row_schema_drift_fails_loudly(self, tmp_path: Path) -> None:
        """Same guard on the candidate stream, named for its own file."""
        row = dict.fromkeys(coordination_schema.CANDIDATE_COLUMNS, "")
        row["bogus"] = ""
        with pytest.raises(RuntimeError) as excinfo:
            coordination_schema.CandidateRow(row)
        assert "metal_contact_candidates_all.csv" in str(excinfo.value)
        assert "unexpected bogus" in str(excinfo.value)

    def test_score_output_requires_its_columns(self, tmp_path: Path) -> None:
        handles = self._handles(tmp_path, score=True)
        try:
            with pytest.raises(ValueError):
                OutputWriters(handles, score_columns=None)
        finally:
            self._close(handles)

    def test_score_inputs_require_scored_columns_to_cover_them(
        self, tmp_path: Path
    ) -> None:
        """A projection gap must fail before the two streams desynchronize."""
        columns = list(score.SCORE_INPUT_COLUMNS)[:-1]
        handles = self._handles(tmp_path, score=True)
        inputs_handle = open(tmp_path / "score_inputs.csv", "w", newline="")
        try:
            with pytest.raises(ValueError, match="every score input column"):
                OutputWriters(
                    {**handles, "score_inputs": inputs_handle},
                    score_columns=columns,
                )
        finally:
            self._close(handles)
            inputs_handle.close()

    def test_manifest_row_schema_mismatch_is_rejected(self, tmp_path: Path) -> None:
        """A missing manifest column must not be written as a silent blank."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        row = manifest_row(entry_result(status="ok"), False, True, {}, {})
        row.pop("status")
        try:
            with pytest.raises(RuntimeError):
                writers.write_manifest_row(row)
        finally:
            self._close(handles)
        assert read_csv(tmp_path / "manifest.csv") == [MANIFEST_COLUMNS]

    def test_score_header_and_counts(self, tmp_path: Path) -> None:
        columns = list(score.SCORE_INPUT_COLUMNS)
        handles = self._handles(tmp_path, score=True)
        writers = OutputWriters(handles, score_columns=columns)
        writers.write_score_rows([])
        assert writers.n_score == 0
        first: dict[str, object] = dict.fromkeys(columns, "")
        first["context_warning"] = True
        first["rszd_abs"] = float("nan")
        writers.write_score_rows([first, dict.fromkeys(columns, "")])
        self._close(handles)
        rows = read_csv(tmp_path / "score.csv")
        assert rows[0] == columns
        assert writers.n_score == 2
        assert len(rows) == 3
        written = dict(zip(columns, rows[1], strict=True))
        assert written["context_warning"] == "true"
        assert written["rszd_abs"] == ""

    def test_scored_score_stream_projects_synchronized_inputs(
        self, tmp_path: Path
    ) -> None:
        """A targeted scored resume updates its reusable input rows as well."""
        scored_columns = [
            *score.SCORE_INPUT_COLUMNS,
            *score.ANALYSIS_COLUMNS,
        ]
        handles = self._handles(tmp_path, score=True)
        inputs_handle = open(tmp_path / "score_inputs.csv", "w", newline="")
        try:
            writers = OutputWriters(
                {**handles, "score_inputs": inputs_handle},
                score_columns=scored_columns,
            )
            row = {column: f"value-{column}" for column in scored_columns}
            writers.write_score_rows([row])
        finally:
            self._close(handles)
            inputs_handle.close()

        scored = read_csv(tmp_path / "score.csv")
        inputs = read_csv(tmp_path / "score_inputs.csv")
        assert scored[0] == scored_columns
        assert inputs[0] == list(score.SCORE_INPUT_COLUMNS)
        assert inputs[1] == [row[column] for column in score.SCORE_INPUT_COLUMNS]

    def test_score_row_schema_mismatch_is_rejected(self, tmp_path: Path) -> None:
        columns = list(score.SCORE_INPUT_COLUMNS)
        handles = self._handles(tmp_path, score=True)
        writers = OutputWriters(handles, score_columns=columns)
        valid: dict[str, Any] = dict.fromkeys(columns, "")
        malformed = valid.copy()
        malformed.pop(columns[0])
        try:
            with pytest.raises(RuntimeError):
                writers.write_score_rows([valid, malformed])
        finally:
            self._close(handles)
        assert writers.n_score == 0
        assert read_csv(tmp_path / "score.csv") == [columns]

    def test_manifest_rows_round_trip_through_the_real_projection(
        self, tmp_path: Path
    ) -> None:
        """A written manifest is readable by load_done without reinterpretation."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        writers.write_manifest_row(
            manifest_row(
                entry_result(
                    "109m",
                    status="ok",
                    retryable=False,
                    n_metals=1,
                    n_bonds=0,
                    n_candidates=0,
                    runtime_s=1.0,
                ),
                False,
                True,
                {},
                {},
            )
        )
        writers.write_manifest_row(
            manifest_row(
                entry_result("1cll", status="error", retryable=True),
                False,
                True,
                {},
                {},
            )
        )
        self._close(handles)
        path = str(tmp_path / "manifest.csv")
        assert resume.load_done(path, bonds_required=True) == {"109m"}
        assert resume.manifest_values_by_id(path, "n_bonds") == {
            "109m": "0",
            "1cll": "",
        }

    def test_each_stream_is_flushed_before_the_manifest_marker(
        self, tmp_path: Path
    ) -> None:
        """An interrupted batch must retain the rows of completed entries."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(handles)
        writers.write_stats_rows(
            [
                MetalStatsRow.from_output_fields(
                    "109m", "metal", ["x"] * (len(STATS_COLUMNS) - 2)
                )
            ]
        )
        writers.write_bond_rows(
            [
                coordination_schema.BondRow(
                    dict.fromkeys(coordination_schema.BOND_COLUMNS, "")
                )
            ]
        )
        writers.write_manifest_row(
            manifest_row(entry_result(status="ok"), False, True, {}, {})
        )
        # Read before ``_close``: the rows must already be on disk.
        assert len(read_csv(tmp_path / "stats.csv")) == 2
        assert len(read_csv(tmp_path / "bonds.csv")) == 2
        assert len(read_csv(tmp_path / "manifest.csv")) == 2
        self._close(handles)
