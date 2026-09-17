"""Test resume bookkeeping: done sets, carry-forward, staging, and write policy."""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from helpers import entry_result, read_csv, run_config, write_manifest

import scratch
from codes import EntryStatus
from coordination import schema as coordination_schema
from driver import (
    confidence as driver_confidence,
    dispatch,
    layout as driver_layout,
    pool,
    resources,
    resume,
)
from driver.runlog import RunLog, RunSummary
from driver.writers import (
    STATS_COLUMNS,
    OutputTargets,
    manifest_row,
)
from worker import contracts


def _manifest_ids(path: str | Path, **kwargs: Any) -> set[str]:
    return resume.load_done(str(path), **kwargs)


class TestLoadDone:
    """Which manifest rows count as finished work that --resume may skip."""

    @pytest.mark.parametrize(
        "status,retryable,expected_done",
        [
            ("ok", "False", True),
            ("ok", "True", True),  # status ok is terminal regardless
            ("ok", "", True),
            ("partial", "False", True),  # terminal partial: nothing left to do
            ("partial", "0", True),
            ("partial", "no", True),
            ("partial", "True", False),
            ("partial", "", False),
            ("error", "True", False),
            # An error is never done: a resumed run may have been handed a
            # repaired input, and skipping would drop work already fixed.
            ("error", "False", False),
            ("skip", "False", False),
            ("skip", "True", False),
            ("", "", False),
        ],
    )
    def test_terminality_by_status_and_retryable(
        self, tmp_path: Path, status: str, retryable: str, expected_done: bool
    ) -> None:
        """Only ok and non-retryable partial rows are skippable."""
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": status,
                    "retryable": retryable,
                    "n_bonds": "3",
                    "n_candidates": "5",
                },
            ],
        )
        assert ("109m" in _manifest_ids(path)) is expected_done

    def test_an_error_is_retried_whatever_its_reason_code_claims(
        self, tmp_path: Path
    ) -> None:
        """A deterministic-looking failure is still offered to the next resume.

        The reason code says the failure recurs on the same inputs, but a
        resumed run may name a repaired file, so resume cannot treat the claim
        as licence to skip.
        """
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "error",
                    "retryable": "False",
                    "reason_codes": "deterministic_processing_error",
                },
            ],
        )
        assert _manifest_ids(path) == set()

    def test_a_partial_citing_a_retired_reason_code_is_retried(
        self, tmp_path: Path
    ) -> None:
        """A terminal partial from an older build cannot vouch for its entry.

        The stored ``retryable=false`` was decided by code that no longer
        emits the reason it cites, so the current code may complete the
        entry; the same row with a live reason code stays protected.
        """
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "1old",
                    "status": "partial",
                    "retryable": "False",
                    "reason_codes": "ambiguous_coordinate_residue_join",
                },
                {
                    "pdbID": "1mix",
                    "status": "partial",
                    "retryable": "False",
                    "reason_codes": "invalid_occupancy|ambiguous_coordinate_residue_join",
                },
                {
                    "pdbID": "1new",
                    "status": "partial",
                    "retryable": "False",
                    "reason_codes": "invalid_occupancy",
                },
            ],
        )
        assert _manifest_ids(path) == {"1new"}

    def test_ids_are_normalized_to_lowercase(self, tmp_path: Path) -> None:
        """Manifest IDs join against the driver's lowercased selection list."""
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": " 1CLL ", "status": "ok", "retryable": "False"},
            ],
        )
        assert _manifest_ids(path) == {"1cll"}

    def test_explicit_terminal_retry_unprotects_only_selected_partials(
        self, tmp_path: Path
    ) -> None:
        """Forced resume never turns an explicitly selected ``ok`` into work."""
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "1ok1",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "1",
                    "n_candidates": "1",
                },
                {
                    "pdbID": "1par",
                    "status": "partial",
                    "retryable": "False",
                    "n_bonds": "1",
                    "n_candidates": "1",
                },
                {
                    "pdbID": "2par",
                    "status": "partial",
                    "retryable": "False",
                    "n_bonds": "1",
                    "n_candidates": "1",
                },
                {
                    "pdbID": "1err",
                    "status": "error",
                    "retryable": "True",
                    "n_bonds": "",
                    "n_candidates": "",
                },
            ],
        )

        assert _manifest_ids(path, retry_partial_ids={"1OK1", "1PAR"}) == {
            "1ok1",
            "2par",
        }

    def test_missing_manifest_is_an_empty_done_set(self, tmp_path: Path) -> None:
        assert _manifest_ids(tmp_path / "absent.csv") == set()

    def test_manifest_without_the_required_columns_is_ignored(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "manifest.csv"
        with open(path, "w", newline="") as handle:
            csv.writer(handle).writerows([["id", "state"], ["109m", "ok"]])
        assert _manifest_ids(path) == set()

    def test_empty_manifest_file_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        path.write_text("")
        assert _manifest_ids(path) == set()

    def test_truncated_final_row_is_retried_without_losing_valid_rows(
        self, tmp_path: Path
    ) -> None:
        """An interrupted writer may leave only the first cells of its row."""
        path = write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok", "retryable": "False"}],
        )
        with open(path, "a", newline="") as handle:
            csv.writer(handle).writerow(["1cll", "ok"])

        assert _manifest_ids(path, bonds_required=False) == {"109m"}

    def test_row_without_a_pdb_id_is_not_done(self, tmp_path: Path) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "0",
                    "n_candidates": "0",
                },
            ],
        )
        assert _manifest_ids(path) == set()

    @pytest.mark.parametrize(
        "n_bonds,n_candidates,expected_done",
        [
            ("0", "0", True),  # measured zero: the stage ran and found none
            ("4", "9", True),
            ("", "0", False),  # blank: the stage never ran
            ("0", "", False),
            ("", "", False),
            ("   ", "0", False),  # whitespace is still blank
        ],
    )
    def test_blank_counts_mean_the_bond_stage_never_ran(
        self, tmp_path: Path, n_bonds: str, n_candidates: str, expected_done: bool
    ) -> None:
        """README: blank counts = not run, 0 = ran and found nothing."""
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": n_bonds,
                    "n_candidates": n_candidates,
                },
            ],
        )
        done = _manifest_ids(path, bonds_required=True)
        assert ("109m" in done) is expected_done

    def test_blank_counts_are_irrelevant_when_bonds_are_not_required(
        self, tmp_path: Path
    ) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "",
                    "n_candidates": "",
                },
            ],
        )
        assert _manifest_ids(path, bonds_required=False) == {"109m"}

    def test_indeterminate_metal_presence_needs_no_bond_stage(
        self, tmp_path: Path
    ) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "partial",
                    "retryable": "False",
                    "reason_codes": "metal_presence_indeterminate",
                    "n_bonds": "",
                    "n_candidates": "",
                }
            ],
        )

        assert _manifest_ids(path, bonds_required=True) == {"109m"}
        assert (
            resume.load_done(
                path,
                bonds_required=True,
                bond_output_present=False,
                candidate_output_present=True,
            )
            == set()
        )

    @pytest.mark.parametrize(
        "bond_present,candidate_present,expected_done",
        [
            (True, True, True),
            (False, True, False),
            (True, False, False),
            (False, False, False),
        ],
    )
    def test_absent_bond_outputs_make_the_result_incomplete(
        self,
        tmp_path: Path,
        bond_present: bool,
        candidate_present: bool,
        expected_done: bool,
    ) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "4",
                    "n_candidates": "9",
                },
            ],
        )
        done = resume.load_done(
            path,
            bonds_required=True,
            bond_output_present=bond_present,
            candidate_output_present=candidate_present,
        )
        assert ("109m" in done) is expected_done

    def test_selects_only_the_terminal_rows_of_a_mixed_manifest(
        self, tmp_path: Path
    ) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "6",
                    "n_candidates": "8",
                },
                {
                    "pdbID": "1cll",
                    "status": "partial",
                    "retryable": "False",
                    "n_bonds": "0",
                    "n_candidates": "0",
                },
                {
                    "pdbID": "1blu",
                    "status": "partial",
                    "retryable": "True",
                    "n_bonds": "1",
                    "n_candidates": "2",
                },
                {"pdbID": "2fha", "status": "error", "retryable": "True"},
                {"pdbID": "2cyp", "status": "skip", "retryable": "False"},
                {
                    "pdbID": "100d",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "",
                    "n_candidates": "",
                },
            ],
        )
        assert _manifest_ids(path, bonds_required=True) == {"109m", "1cll"}
        assert _manifest_ids(path, bonds_required=False) == {"109m", "1cll", "100d"}


class TestManifestValuesById:
    """Reading one prior manifest column for the resume carry-forward."""

    def test_returns_the_column_keyed_by_normalized_id(self, tmp_path: Path) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "109M", "status": "ok", "n_bonds": "7"},
                {"pdbID": " 1cll", "status": "ok", "n_bonds": "0"},
                {"pdbID": "1blu", "status": "error", "n_bonds": ""},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {
            "109m": "7",
            "1cll": "0",
            "1blu": "",
        }

    def test_missing_and_empty_files_yield_no_values(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        assert resume.manifest_values_by_id(str(tmp_path / "gone.csv"), "n_bonds") == {}
        assert resume.manifest_values_by_id(str(empty), "n_bonds") == {}

    def test_unknown_column_yields_blanks_not_an_error(self, tmp_path: Path) -> None:
        """An older manifest lacking the column must degrade to "not run"."""
        path = write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok"}],
            columns=["pdbID", "status"],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": ""}

    def test_truncated_rows_do_not_supply_carry_forward_values(
        self, tmp_path: Path
    ) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok", "n_bonds": "7"}],
        )
        with open(path, "a", newline="") as handle:
            csv.writer(handle).writerow(["1cll", "ok"])

        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "7"}

    def test_blank_ids_are_dropped(self, tmp_path: Path) -> None:
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "", "status": "ok", "n_bonds": "9"},
                {"pdbID": "109m", "status": "ok", "n_bonds": "1"},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "1"}

    def test_a_later_row_supersedes_an_earlier_one(self, tmp_path: Path) -> None:
        """Resume appends, so the last row for an ID is the current one."""
        path = write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "109m", "status": "error", "n_bonds": ""},
                {"pdbID": "109m", "status": "ok", "n_bonds": "5"},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "5"}


class TestUnrunBondStageChain:
    """Verify an unrun bond stage remains eligible after a density-only resume.

    Seed a pre-bond failure, resume without bonds, then resume with bonds.
    A placeholder zero must not mark the unrun stage complete.
    """

    def test_failed_bond_enabled_entry_is_not_marked_bond_complete(
        self, tmp_path: Path
    ) -> None:
        """Step 1: a pre-bond failure writes blank counts, so it is retried."""
        result = entry_result(
            "109m", status="error", retryable=True, status_detail="density stage failed"
        )
        row = manifest_row(result, False, True, {}, {})
        manifest = tmp_path / "manifest.csv"
        write_manifest(manifest, [row])

        assert resume.load_done(str(manifest), bonds_required=True) == set()
        assert resume.load_done(str(manifest), bonds_required=False) == set()
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_resume_no_bonds_recovery_does_not_fake_a_completed_bond_stage(
        self, tmp_path: Path
    ) -> None:
        """Verify a no-bonds retry does not mark bonds complete.

        Run 1 fails before the bond stage, run 2 recovers it under
        ``--no-bonds``, and run 3 must still schedule it: no bond analysis has
        ever run for the entry.
        """
        manifest = tmp_path / "manifest.csv"

        failed = manifest_row(
            entry_result(
                "109m", status="error", retryable=True, status_detail="edstats failed"
            ),
            resume=False,
            bonds_enabled=True,
            prior_bond_counts={},
            prior_candidate_counts={},
        )
        write_manifest(manifest, [failed])

        assert resume.load_done(str(manifest), bonds_required=False) == set()
        prior_bonds = resume.manifest_values_by_id(str(manifest), "n_bonds")
        prior_candidates = resume.manifest_values_by_id(str(manifest), "n_candidates")

        recovered = manifest_row(
            entry_result("109m", status="ok", retryable=False, n_metals=2),
            resume=True,
            bonds_enabled=False,
            prior_bond_counts=prior_bonds,
            prior_candidate_counts=prior_candidates,
        )
        assert recovered["status"] == "ok"
        assert recovered["n_metals"] == 2
        write_manifest(manifest, [recovered])

        assert resume.load_done(str(manifest), bonds_required=False) == {"109m"}
        assert resume.load_done(str(manifest), bonds_required=True) == set()

        # Only a genuine measured zero may let a later resume skip the entry.
        completed = manifest_row(
            entry_result(
                "109m",
                status="ok",
                retryable=False,
                n_metals=2,
                n_bonds=0,
                n_candidates=0,
            ),
            resume=True,
            bonds_enabled=True,
            prior_bond_counts={},
            prior_candidate_counts={},
        )
        write_manifest(manifest, [completed])
        assert resume.load_done(str(manifest), bonds_required=True) == {"109m"}


class TestResumeReplacementSucceeded:
    """Only a terminal retry may replace the rows it is retrying."""

    @pytest.mark.parametrize(
        "status,retryable,expected",
        [
            ("ok", False, True),
            ("ok", True, True),
            ("partial", False, True),
            ("partial", True, False),
            ("error", False, False),
            ("error", True, False),
            ("skip", False, False),
        ],
    )
    def test_terminality(self, status: str, retryable: bool, expected: bool) -> None:
        """A retryable or failed retry leaves the previous rows in place."""
        result = entry_result(status=status, retryable=retryable)
        assert resume.resume_replacement_succeeded(result) is expected

    @pytest.mark.parametrize("status", ["OK", " ok ", ""])
    def test_an_invalid_internal_status_is_rejected(self, status: str) -> None:
        with pytest.raises(ValueError):
            EntryStatus(status)

    def test_an_untouched_partial_is_assumed_unfinished(self) -> None:
        """Verify an untouched partial remains unfinished.

        ``retryable`` defaults to true, so a partial that cleared nothing cannot
        replace the previous result.
        """
        assert (
            resume.resume_replacement_succeeded(entry_result(status="partial")) is False
        )


class TestResumeStaging:
    """Staged retries replace rows only on a completed, terminal batch."""

    CORE_NAMES = ("manifest", "stats", "bonds", "candidates")
    ALWAYS_NAMES = (*CORE_NAMES, *OutputTargets.ALWAYS_WRITTEN_EXTRAS)

    def _outputs(
        self,
        output_dir: Path,
        *,
        confidence_inputs: bool = False,
        scores: bool = False,
    ) -> OutputTargets:
        """Create the always-written CSVs, plus any confidence ones, by name."""
        layout = driver_layout.OutputLayout(str(output_dir))
        targets = OutputTargets(
            manifest=layout.manifest,
            stats=layout.stats,
            bonds=layout.bonds,
            candidates=layout.candidates,
            crystallization_conditions=layout.crystallization_conditions,
            crystallization_summary=layout.crystallization_summary,
            density_context=layout.density_context,
            confidence=layout.confidence_scores if scores else None,
            confidence_inputs=layout.confidence_inputs if confidence_inputs else None,
        )
        for name, path in targets.present().items():
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value"])
                writer.writerow(["109m", f"old-109m-{name}"])
                writer.writerow(["1cll", f"old-1cll-{name}"])
        return targets

    @staticmethod
    def _bytes(
        targets: OutputTargets, names: Sequence[str] | None = None
    ) -> dict[str, bytes]:
        """The current content of the named outputs, or of every present one."""
        present = targets.present()
        chosen = present if names is None else {name: present[name] for name in names}
        return {name: Path(path).read_bytes() for name, path in chosen.items()}

    @staticmethod
    def _values(targets: OutputTargets, name: str) -> dict[str, str]:
        return {row[0]: row[1] for row in read_csv(targets.present()[name])[1:]}

    def _stage(
        self,
        staging: resume.ResumeStaging,
        names_and_rows: Mapping[str, Sequence[Sequence[str]]],
    ) -> None:
        """Write staged rows for the named outputs."""
        for name, rows in names_and_rows.items():
            with open(staging.staged.present()[name], "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value"])
                writer.writerows(rows)

    def test_staged_paths_mirror_the_targets_inside_the_output_dir(
        self, tmp_path: Path
    ) -> None:
        """Staging is a sibling temp dir so a merge is a same-filesystem move."""
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            assert os.path.isdir(staging.dir)
            assert os.path.dirname(staging.dir) == str(tmp_path)
            staged = staging.staged.present()
            assert set(staged) == set(targets.present())
            for name, path in targets.present().items():
                assert os.path.basename(staged[name]) == os.path.basename(path)
                assert os.path.dirname(staged[name]) == staging.dir
                assert read_csv(path)[1][1].startswith("old-109m")
            assert staging.replacement_ids == set()
        finally:
            staging.discard()

    def test_commit_without_replacement_ids_leaves_outputs_byte_identical(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        before = self._bytes(targets)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {"manifest": [["109m", "new"]]})
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        assert self._bytes(targets) == before

    @pytest.mark.parametrize("fail_merge", [False, True])
    def test_an_interrupted_batch_commits_the_entries_it_finished(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_merge: bool
    ) -> None:
        """The halted-run path is reached from ``process_entries`` itself.

        The loss lived in the wiring: staging was discarded whenever the batch
        failed to run to completion. A test that calls the commit helper
        directly still passes with that defect in place, so this one drives the
        real function and interrupts it mid-batch.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        layout = driver_layout.OutputLayout(str(output_dir))
        write_manifest(
            layout.manifest,
            [{"pdbID": "aaaa", "status": "ok", "retryable": "False"}],
        )
        headers = (
            (layout.stats, STATS_COLUMNS),
            (layout.bonds, coordination_schema.BOND_COLUMNS),
            (layout.candidates, coordination_schema.CANDIDATE_COLUMNS),
        )
        for path, columns in headers:
            with open(path, "w", newline="") as handle:
                csv.writer(handle).writerow(columns)

        def _dispatch(
            ids: Sequence[str],
            cfg: contracts.WorkerConfig,
            workers: int,
            memory_plan: resources.MemoryPlan,
            run_log: RunLog,
            sink: dispatch.ResultSink,
        ) -> dispatch.BatchTally:
            del cfg, workers, memory_plan, run_log
            for pdb_id in ids:
                sink(entry_result(pdb_id, status="ok", retryable=False))
            if not fail_merge:
                raise KeyboardInterrupt
            return dispatch.BatchTally()

        monkeypatch.setattr(dispatch, "dispatch_entries", _dispatch)
        if fail_merge:

            def failed_commit(*_args: object, **_kwargs: object) -> None:
                raise OSError("simulated full disk during resume merge")

            monkeypatch.setattr(resume.ResumeStaging, "commit", failed_commit)
        args = run_config(resume=True, bonds=True, output_dir=str(output_dir))
        run_log = RunLog(args, "pytest")

        with pytest.raises(OSError if fail_merge else KeyboardInterrupt):
            pool.process_entries(
                args,
                ["bbbb", "cccc"],
                # ``None`` proves the halted-batch path never reaches the worker
                # configuration: ``_dispatch_entries`` is replaced above.
                cast(contracts.WorkerConfig, None),
                1,
                layout,
                driver_confidence.ConfidencePlan(),
                run_log,
                resources.MemoryPlan(
                    [
                        resources.EntryMemoryEstimate("bbbb", 1, "test"),
                        resources.EntryMemoryEstimate("cccc", 1, "test"),
                    ],
                    None,
                    None,
                ),
            )

        if fail_merge:
            assert run_log.summary.resume_staging_recovery_dir is not None
            recovery = Path(run_log.summary.resume_staging_recovery_dir)
            assert recovery.is_dir()
            scratch.sweep_owned_scratch_directories(str(output_dir))
            assert recovery.is_dir()
            assert [r[0] for r in read_csv(str(recovery / "manifest.csv"))[1:]] == [
                "bbbb",
                "cccc",
            ]
            return

        assert [row[0] for row in read_csv(layout.manifest)[1:]] == [
            "aaaa",
            "bbbb",
            "cccc",
        ]
        assert run_log.summary.resume_entries_committed_after_interrupt == 2

    def _interrupt(
        self,
        staging: resume.ResumeStaging,
        tmp_path: Path,
        *,
        bonds: bool = True,
        confidence: bool = False,
    ) -> RunSummary:
        """Run the halted-resume path and return the run-log summary."""
        args = run_config(bonds=bonds, output_dir=str(tmp_path))
        # Any enabled plan will do: the commit only asks whether one exists.
        plan: driver_confidence.ConfidencePlan = (
            driver_confidence.ClassificationPlan(
                driver_layout.OutputLayout(str(tmp_path))
            )
            if confidence
            else driver_confidence.ConfidencePlan()
        )
        run_log = RunLog(args, "pytest")
        pool.commit_staged_entries(staging, args, plan, run_log, run_aborted=True)
        return run_log.summary

    def test_an_interrupted_resume_keeps_the_entries_it_completed(
        self, tmp_path: Path
    ) -> None:
        """A halted batch commits finished entries instead of destroying them.

        Discarding lost every entry the run had completed, including entries
        with no previous manifest row that staging never existed to protect,
        while the interrupt message promised they had been kept.
        """
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(
            staging,
            {name: [["8new", f"new-8new-{name}"]] for name in self.ALWAYS_NAMES},
        )
        staging.replacement_ids.add("8new")

        summary = self._interrupt(staging, tmp_path)

        for name, path in targets.present().items():
            ids = [row[0] for row in read_csv(path)[1:]]
            assert ids == ["109m", "1cll", "8new"], name
        assert summary.resume_entries_committed_after_interrupt == 1
        assert not os.path.isdir(staging.dir)

    def test_an_interrupted_resume_drops_an_entry_that_never_completed(
        self, tmp_path: Path
    ) -> None:
        """Rows without a manifest row are not promoted, so the entry retries.

        ``write_entry`` adds the id only after the manifest row, so an entry
        interrupted mid-write is absent from ``replacement_ids``.
        """
        targets = self._outputs(tmp_path)
        before = self._bytes(targets)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(staging, {"stats": [["8new", "half-written"]]})

        summary = self._interrupt(staging, tmp_path)

        assert self._bytes(targets) == before
        assert summary.resume_entries_committed_after_interrupt is None
        assert not os.path.isdir(staging.dir)

    def test_a_failed_interrupt_merge_leaves_the_staged_rows_on_disk(
        self, tmp_path: Path
    ) -> None:
        """A merge failure must not delete the only copy of the work."""
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(staging, {"manifest": [["8new", "new"]]})
        staging.replacement_ids.add("8new")
        # A staged file whose header disagrees makes ``commit`` raise.
        with open(staging.staged.stats, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["different", "header"])
            writer.writerow(["8new", "new"])

        try:
            summary = self._interrupt(staging, tmp_path)

            assert os.path.isdir(staging.dir)
            assert summary.resume_staging_recovery_dir == staging.dir
            assert summary.resume_staging_commit_error is not None
            assert "staged CSV schema" in summary.resume_staging_commit_error
            scratch.sweep_owned_scratch_directories(str(tmp_path))
            assert os.path.isdir(staging.dir)
        finally:
            staging.discard()

    def test_commit_replaces_only_the_retried_ids(self, tmp_path: Path) -> None:
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(
                staging,
                {name: [["109m", f"new-109m-{name}"]] for name in self.ALWAYS_NAMES},
            )
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        for name, path in targets.present().items():
            rows = read_csv(path)
            assert rows[0] == ["pdbID", "value"]
            assert self._values(targets, name) == {
                "109m": f"new-109m-{name}",
                "1cll": f"old-1cll-{name}",
            }
            assert len(rows) == 3

    def test_commit_drops_stale_rows_when_the_retry_produced_none(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {name: [] for name in self.ALWAYS_NAMES})
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        for name in targets.present():
            assert set(self._values(targets, name)) == {"1cll"}, name

    def test_commit_with_bonds_disabled_leaves_bond_outputs_untouched(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        bond_names = ("bonds", "candidates")
        before = self._bytes(targets, bond_names)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(
                staging,
                {
                    "manifest": [["109m", "new-manifest"]],
                    "stats": [["109m", "new-stats"]],
                    "bonds": [["109m", "SHOULD-NOT-APPEAR"]],
                    "candidates": [["109m", "SHOULD-NOT-APPEAR"]],
                    **{
                        name: [["109m", f"new-{name}"]]
                        for name in OutputTargets.ALWAYS_WRITTEN_EXTRAS
                    },
                },
            )
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=False)
        finally:
            staging.discard()
        assert self._bytes(targets, bond_names) == before
        assert self._values(targets, "manifest") == {
            "109m": "new-manifest",
            "1cll": "old-1cll-manifest",
        }

    def test_commit_replaces_confidence_rows_only_when_enabled(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path, confidence_inputs=True)
        before = self._bytes(targets, ("confidence_inputs",))
        staged_rows = {name: [["109m", "new"]] for name in targets.present()}

        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, staged_rows)
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=False)
        finally:
            staging.discard()
        assert self._bytes(targets, ("confidence_inputs",)) == before

        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, staged_rows)
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=True)
        finally:
            staging.discard()
        assert self._values(targets, "confidence_inputs") == {
            "109m": "new",
            "1cll": "old-1cll-confidence_inputs",
        }

    def test_commit_replaces_every_enabled_confidence_output(
        self, tmp_path: Path
    ) -> None:
        """Scored and input confidence rows participate in one staged commit."""
        targets = self._outputs(tmp_path, confidence_inputs=True, scores=True)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(
                staging,
                {name: [["109m", f"new-{name}"]] for name in targets.present()},
            )
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=True)
        finally:
            staging.discard()

        for name in OutputTargets.CONFIDENCE_OUTPUTS:
            assert self._values(targets, name) == {
                "109m": f"new-{name}",
                "1cll": f"old-1cll-{name}",
            }

    def test_discard_leaves_previous_rows_intact(self, tmp_path: Path) -> None:
        targets = self._outputs(tmp_path)
        before = self._bytes(targets)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(staging, {name: [["109m", "new"]] for name in self.CORE_NAMES})
        staging.replacement_ids.add("109m")
        staging.discard()
        assert not os.path.exists(staging.dir)
        assert self._bytes(targets) == before

    def test_discard_is_idempotent(self, tmp_path: Path) -> None:
        """Cleanup runs in a finally block and may be reached twice."""
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        staging.discard()
        staging.discard()
        assert not os.path.exists(staging.dir)

    def test_replacement_ids_are_matched_case_insensitively(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(
                staging, {name: [["109M", "new"]] for name in self.ALWAYS_NAMES}
            )
            staging.replacement_ids.add("109M")
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        values = {
            key.lower(): value
            for key, value in self._values(targets, "manifest").items()
        }
        assert values["109m"] == "new"
        assert len(values) == 2

    def test_a_staged_schema_mismatch_aborts_without_touching_the_target(
        self, tmp_path: Path
    ) -> None:
        """A header disagreement must fail loudly, not silently misalign rows."""
        targets = self._outputs(tmp_path)
        before = self._bytes(targets, ("manifest",))
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            with open(staging.staged.manifest, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value", "extra"])
                writer.writerow(["109m", "new", "x"])
            self._stage(
                staging,
                {
                    name: [["109m", "new"]]
                    for name in self.ALWAYS_NAMES
                    if name != "manifest"
                },
            )
            staging.replacement_ids.add("109m")
            with pytest.raises(ValueError):
                staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        assert self._bytes(targets, ("manifest",)) == before
        leftovers = [
            name for name in os.listdir(tmp_path) if name.startswith(".manifest.csv.")
        ]
        assert leftovers == []


def test_stale_bond_outputs_are_removed_only_by_a_fresh_disabled_run(
    tmp_path: Path,
) -> None:
    """Old bond rows must not be mistaken for this run's output.

    Resume is the exception: it retains completed entries, so their existing
    rows are still current.
    """
    paths = [str(tmp_path / "bonds.csv"), str(tmp_path / "candidates.csv")]
    for path in paths:
        open(path, "w", encoding="utf-8").close()

    assert (
        resume.remove_stale_disabled_bond_outputs(
            paths, resume=True, bonds_enabled=False
        )
        == []
    )
    assert (
        resume.remove_stale_disabled_bond_outputs(
            paths, resume=False, bonds_enabled=True
        )
        == []
    )
    assert all(os.path.exists(path) for path in paths)

    assert (
        resume.remove_stale_disabled_bond_outputs(
            paths, resume=False, bonds_enabled=False
        )
        == paths
    )
    assert not any(os.path.exists(path) for path in paths)


def test_removing_absent_bond_outputs_is_not_an_error(tmp_path: Path) -> None:
    paths = [str(tmp_path / "absent.csv")]
    assert (
        resume.remove_stale_disabled_bond_outputs(
            paths, resume=False, bonds_enabled=False
        )
        == []
    )


@pytest.mark.parametrize(
    "resuming, pdb_id, status, retryable, prior, expected, why",
    [
        (
            False,
            "1abc",
            "error",
            True,
            set[str](),
            True,
            "a fresh run writes everything",
        ),
        (True, "1abc", "ok", False, {"1abc"}, True, "an improved retry replaces"),
        (True, "1abc", "error", True, {"1abc"}, False, "a failed retry must not"),
        (True, "1ABC", "error", True, {"1abc"}, False, "ids compare case-folded"),
        (True, "2xyz", "error", True, {"1abc"}, True, "a new entry has none to keep"),
        (True, "2xyz", "skip", True, {"1abc"}, True, "including when it skipped"),
    ],
)
def test_a_resumed_run_writes_new_entries_even_when_they_fail(
    resuming: bool,
    pdb_id: str,
    status: str,
    retryable: bool,
    prior: set[str],
    expected: bool,
    why: str,
) -> None:
    """Suppression protects a previous row; a new entry has no previous row.

    Without the distinction a newly scheduled entry that failed left no
    manifest row at all, so the artifact --resume reads under-reported the set
    the run had actually scheduled.
    """
    result = entry_result(pdb_id, status=status, retryable=retryable)

    assert pool.should_write_entry(resuming, result, prior) is expected, why
