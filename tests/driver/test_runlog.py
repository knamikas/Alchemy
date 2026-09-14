"""Test the run log: placement, content, and concurrent path claims."""

from __future__ import annotations

import csv
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import pytest
from helpers import entry_result, run_config

from driver import runlog
from driver.runlog import RunLog, RunSummary


class TestRunLog:
    """The written log is the only record of how a finished run behaved."""

    def test_the_log_goes_to_a_subdirectory_of_the_output_by_default(
        self, tmp_path: Path
    ) -> None:
        """Run diagnostics stay separate from the scientific result CSVs."""
        args = run_config(output_dir=str(tmp_path), log_dir=None, workers=1)

        path = RunLog(args, "pytest").write(0)

        assert os.path.dirname(path) == str(tmp_path / runlog.DEFAULT_LOG_DIRNAME)
        assert sorted(os.listdir(tmp_path)) == [runlog.DEFAULT_LOG_DIRNAME]
        diagnostics = Path(path.removesuffix(".log") + "_entries.csv")
        assert diagnostics.is_file()

    def test_an_explicit_log_dir_is_used_as_given(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "shared-logs"
        args = run_config(
            output_dir=str(tmp_path / "out"), log_dir=str(elsewhere), workers=1
        )

        path = RunLog(args, "pytest").write(0)

        assert os.path.dirname(path) == str(elsewhere)
        assert not (tmp_path / "out").exists(), "the output dir is not created here"

    @staticmethod
    def _log(tmp_path: Path, runtimes: Sequence[float]) -> RunLog:
        run_log = RunLog(
            run_config(output_dir=str(tmp_path), log_dir=None, workers=1),
            "pytest",
        )
        for index, runtime in enumerate(runtimes):
            run_log.record_entry(
                entry_result(
                    f"e{index}",
                    status="ok",
                    runtime_s=runtime,
                    n_metals=1,
                    n_bonds=0,
                    n_candidates=0,
                )
            )
        return run_log

    def test_the_slowest_entries_table_is_ordered_slowest_first(
        self, tmp_path: Path
    ) -> None:
        """Verify slowest-entry reporting sorts in descending order.

        Reversed, the table still looks plausible while naming the entries that
        mattered least.
        """
        text = open(self._log(tmp_path, [0.5, 9.0, 3.0]).write(0)).read()
        section = text.split("Slowest entries")[1].split("Entry diagnostics")[0]
        listed = [
            line.split(" | ")[0]
            for line in section.splitlines()
            if line.startswith("e")
        ]
        assert listed == ["e1", "e2", "e0"]

    def test_entry_diagnostics_are_a_sorted_machine_readable_companion(
        self, tmp_path: Path
    ) -> None:
        run_log = self._log(tmp_path, [])
        run_log.record_entry(
            entry_result(
                "2xyz",
                status="partial",
                runtime_s=3.25,
                n_metals=2,
                n_bonds=4,
                n_candidates=5,
                timings={"density_total_s": 2.5, "cleanup_s": 0.125},
                reason_codes=["missing_first_sphere_reference"],
                warning_codes=["multi_model_structure"],
                status_detail="first-sphere reference unavailable for ZN-N",
                density_map_scope_used="model-envelope",
                density_full_map_bytes=2048,
                density_edstats_map_bytes=1024,
            ),
            memory_estimate_bytes=3 * 1024**3,
        )
        run_log.record_entry(
            entry_result(
                "1abc",
                status="ok",
                runtime_s=1.0,
                n_metals=1,
                n_bonds=2,
                n_candidates=3,
                timings={"density_total_s": 0.75},
            )
        )

        log_path = run_log.write(0)
        diagnostics_path = Path(log_path.removesuffix(".log") + "_entries.csv")
        with diagnostics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert [row["pdbID"] for row in rows] == ["1abc", "2xyz"]
        assert rows[1]["density_total_s"] == "2.500"
        assert rows[1]["cleanup_s"] == "0.125"
        assert rows[1]["memory_estimate_bytes"] == str(3 * 1024**3)
        assert rows[1]["reason_codes"] == "missing_first_sphere_reference"
        assert rows[1]["warning_codes"] == "multi_model_structure"
        assert rows[1]["status_detail"] == (
            "first-sphere reference unavailable for ZN-N"
        )
        assert "error" not in rows[1]

        text = Path(log_path).read_text(encoding="utf-8")
        assert f"Entry diagnostics: {diagnostics_path}" in text
        assert "Per-entry results" not in text

    def test_log_records_policy_provenance_and_groups_routine_partials(
        self, tmp_path: Path
    ) -> None:
        run_log = self._log(tmp_path, [])
        run_log.details.update(
            alchemy_version="1.2.3",
            alchemy_commit="deadbeef1234",
            gemmi_version="0.7.3",
            ccp4_version="9.0",
            reference_data_id="test-reference",
        )
        run_log.record_entry(
            entry_result(
                "routine",
                status="partial",
                retryable=False,
                reason_codes=["missing_first_sphere_reference"],
            )
        )
        run_log.record_entry(
            entry_result(
                "dense",
                status="ok",
                retryable=False,
                n_metals=101,
                metal_site_limit_exceeded=True,
                reason_codes=["metal_site_limit_exceeded"],
            )
        )

        text = Path(run_log.write(0)).read_text(encoding="utf-8")

        assert "Alchemy commit: deadbeef1234" in text
        assert "Maximum selected metal sites per entry: 100" in text
        assert "Status counts: ok=1 partial=1 skip=0 error=0" in text
        assert "Policy exclusions above 100 metal sites: 1" in text
        assert "dense | 101" in text
        assert (
            "Terminal partials caused only by missing first-sphere references: 1"
            in text
        )
        exceptions = text.split("Exceptions and exclusions")[1].split(
            "Slowest entries"
        )[0]
        assert "routine | partial" not in exceptions

    @pytest.mark.parametrize(
        "written,finalized,expected",
        [
            # A run without confidence still accounts for its zero stream rows.
            (0, None, ["Additional completion details:", "  confidence_rows: 0"]),
            # A finalized database run reports the scores file total ...
            (40, 40, ["Confidence scores: scores.csv (40 rows)"]),
            # ... and, when resumed, how many of them this run streamed.
            (
                12,
                40,
                [
                    "Confidence scores: scores.csv (40 rows)",
                    "Additional completion details:",
                    "  confidence_rows_written: 12",
                ],
            ),
        ],
    )
    def test_confidence_counts_are_reported_beside_the_scores_file(
        self,
        tmp_path: Path,
        written: int,
        finalized: int | None,
        expected: list[str],
    ) -> None:
        run_log = self._log(tmp_path, [])
        run_log.summary.confidence_rows_written = written
        if finalized is not None:
            run_log.summary.confidence_rows = finalized
            run_log.summary.confidence_scores_path = "scores.csv"

        text = Path(run_log.write(0)).read_text(encoding="utf-8")

        section = text.split("Output files")[1].split("Stage timing")[0]
        assert [line for line in section.splitlines() if "confidence" in line] == (
            [line for line in expected if "confidence" in line]
        )
        assert ("Additional completion details:" in section) == (
            "Additional completion details:" in expected
        )

    def test_every_summary_field_is_rendered_or_listed_as_a_detail(self) -> None:
        """A field the report neither labels nor lists would vanish silently."""
        names = {field.name for field in fields(RunSummary)}
        assert names >= RunSummary.RENDERED_BY_NAME
        summary = RunSummary()
        for name in names - RunSummary.RENDERED_BY_NAME:
            setattr(summary, name, 1)
        assert set(summary.completion_details()) == names - RunSummary.RENDERED_BY_NAME

    def test_an_existing_log_is_never_overwritten(self, tmp_path: Path) -> None:
        first = self._log(tmp_path, [1.0]).write(0)
        second = self._log(tmp_path, [2.0]).write(1)
        assert first != second
        assert os.path.isfile(first) and os.path.isfile(second)
        assert Path(first.removesuffix(".log") + "_entries.csv").is_file()
        assert Path(second.removesuffix(".log") + "_entries.csv").is_file()
        assert "Exit code: 0" in open(first).read()
        assert "Exit code: 1" in open(second).read()


def test_concurrent_run_reports_reserve_distinct_file_pairs(tmp_path: Path) -> None:
    """A shared --log-dir cannot mix the two artifacts from separate runs."""
    run_count = 8
    ready = threading.Barrier(run_count)

    def claim(_index: int) -> tuple[str, str, str]:
        ready.wait()
        return runlog.claim_report_paths(str(tmp_path), "alchemy_run_20260805")

    with ThreadPoolExecutor(max_workers=run_count) as executor:
        claimed = list(executor.map(claim, range(run_count)))

    log_paths = [log_path for log_path, _diagnostics, _claim in claimed]
    diagnostics_paths = [diagnostics for _log_path, diagnostics, _claim in claimed]
    assert len(set(log_paths)) == run_count
    assert len(set(diagnostics_paths)) == run_count
    assert {Path(path).stem.removesuffix("_entries") for path in diagnostics_paths} == {
        Path(path).stem for path in log_paths
    }
    for _log_path, _diagnostics, claim_path in claimed:
        os.unlink(claim_path)
