"""Test the manifest projection: result skeletons, rows, and write order."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from helpers import entry_result, run_config, worker_config

import worker
import worker_contracts
from driver import confidence as driver_confidence
from driver import environment, pool, resume
from driver.writers import (
    MANIFEST_COLUMNS,
    MANIFEST_FIELDS,
    OutputWriters,
    manifest_row,
)
from output_rows import MetalStatsRow
from run_config import RunConfig

CFG = worker_config()


# Columns ``manifest_row`` computes; every other column it copies.
DERIVED_MANIFEST_COLUMNS = frozenset(
    {
        "n_metals",
        "n_bonds",
        "n_candidates",
        "runtime_s",
        "reason_codes",
        "warning_codes",
    }
)


class TestInitialResult:
    """The per-entry skeleton that guarantees a complete manifest row."""

    def test_seeds_bond_counts_blank_not_zero(self) -> None:
        """``0`` is a measured result, so an unrun bond stage must stay blank."""
        result = worker.initial_result("109m", CFG, None)
        assert result.n_bonds is None
        assert result.n_candidates is None
        assert result.n_bonds != 0
        assert result.n_candidates != 0

    def test_every_non_derived_manifest_column_is_present_up_front(self) -> None:
        """A failure at any stage still projects onto a complete row."""
        result = worker.initial_result("109m", CFG, None)
        required = set(MANIFEST_COLUMNS) - DERIVED_MANIFEST_COLUMNS
        supplied = {
            column for column, name in MANIFEST_FIELDS.items() if hasattr(result, name)
        }
        assert required.issubset(supplied)

    def test_supplies_the_fields_manifest_row_reads_directly(self) -> None:
        """``manifest_row`` reads these without a default; they must exist."""
        result = worker.initial_result("109m", CFG, None)
        for name in ("n_metals", "runtime_s", "n_bonds", "n_candidates", "pdb_id"):
            assert hasattr(result, name)

    def test_defaults_to_a_retryable_error(self) -> None:
        result = worker.initial_result("109m", CFG, None)
        assert result.status == "error"
        assert result.retryable is True

    def test_carries_the_reference_data_identity(self) -> None:
        """Verify reference-data identity is preserved.

        Runs whose z-scores used different reference distances must remain
        distinguishable in the output.
        """
        result = worker.initial_result("109m", CFG, None)
        row = manifest_row(result, False, True, {}, {})

        assert result.reference_data_id == CFG.reference_data_id
        assert row["reference_data_id"] == CFG.reference_data_id

    def test_carries_the_analysis_configuration_identity(self) -> None:
        result = worker.initial_result("109m", CFG, None)
        row = manifest_row(result, False, True, {}, {})

        assert result.analysis_config_id == CFG.analysis_config_id
        assert row["analysis_config_id"] == CFG.analysis_config_id

    def test_carries_run_provenance_from_the_config(self) -> None:
        result = worker.initial_result("109m", CFG, None)
        assert result.alchemy_version == environment.ALCHEMY_VERSION
        assert result.alchemy_commit == CFG.alchemy_commit
        assert result.gemmi_version == CFG.gemmi_version
        assert result.ccp4_version == CFG.ccp4_version
        assert result.model_policy == worker_contracts.MODEL_POLICY
        assert result.altloc_policy == worker_contracts.ALTLOC_POLICY
        assert result.symmetry_contact_policy == worker_contracts.SYMMETRY_POLICY

    @pytest.mark.parametrize(
        "manual_inputs,expected",
        [
            (None, "final"),
            ({}, "final"),
            ({"pdb_file": "/x.pdb"}, "manual"),
        ],
    )
    def test_refinement_state_reflects_manual_inputs(
        self, manual_inputs: dict[str, str | None] | None, expected: str
    ) -> None:
        """Manual coordinate/MTZ input is not a PDB-REDO final re-refinement."""
        result = worker.initial_result("109m", CFG, manual_inputs)
        assert result.refinement_state == expected

    def test_mirror_source_paths_are_portable_but_manual_paths_are_preserved(
        self,
    ) -> None:
        mirror_path = "/srv/pdb-redo/09/109m/109m_final.cif"
        manual_path = "/research/inputs/custom.cif"

        assert (
            worker.source_coordinate_provenance_path(CFG, "109m", mirror_path)
            == "09/109m/109m_final.cif"
        )
        assert (
            worker.source_coordinate_provenance_path(
                worker_config(manual_inputs={"cif_file": manual_path}),
                "109m",
                manual_path,
            )
            == manual_path
        )

    def test_row_lists_are_independent_between_entries(self) -> None:
        first = worker.initial_result("109m", CFG, None)
        second = worker.initial_result("1cll", CFG, None)
        first.rows.append(MetalStatsRow.from_output_fields("109m", "metal", [1]))
        first.reason_codes.append("boom")
        assert second.rows == []
        assert second.reason_codes == []

    def test_a_misspelled_field_cannot_be_assigned(self) -> None:
        """Verify result records reject misspelled fields.

        With ``slots=True``, a misspelled assignment is an error rather than an
        unread field that leaves a stale value in the manifest.
        """
        result = worker.initial_result("109m", CFG, None)

        with pytest.raises(AttributeError, match="retryble"):
            result.retryble = True  # type: ignore[attr-defined]

        assert result.retryable is True, "the real field must be untouched"

    def test_unmeasured_fields_render_blank_not_none(self) -> None:
        """Verify unmeasured fields serialize as blanks.

        A ``None`` reaching CSV as ``"None"`` reads back to the next
        ``--resume`` as a completed stage.
        """
        result = worker.initial_result("109m", CFG, None)
        row = manifest_row(result, False, True, {}, {})

        for column in ("n_bonds", "n_candidates", "input_model_count"):
            assert row[column] == ""
        assert "None" not in set(map(str, row.values()))


class TestManifestRow:
    """Projection of a worker result onto the manifest schema."""

    def test_projects_exactly_the_manifest_columns(self) -> None:
        row = manifest_row(entry_result(), False, True, {}, {})
        assert set(row) == set(MANIFEST_COLUMNS)
        assert "rows" not in row
        assert "bond_rows" not in row
        assert "timings" not in row

    def test_renames_and_joins_the_derived_columns(self) -> None:
        """n/runtime become n_metals/runtime_s; code lists become pipe text."""
        result = entry_result(
            n_metals=3,
            runtime_s=12.5,
            n_bonds=7,
            n_candidates=11,
            reason_codes=["a", "b"],
            warning_codes=["w"],
        )
        row = manifest_row(result, False, True, {}, {})
        assert row["n_metals"] == 3
        assert row["runtime_s"] == "12.500"
        assert row["n_bonds"] == 7
        assert row["n_candidates"] == 11
        assert row["reason_codes"] == "a|b"
        assert row["warning_codes"] == "w"

    def test_booleans_and_status_detail_have_stable_csv_text(self) -> None:
        result = entry_result(
            retryable=False,
            no_metals=True,
            metal_site_limit_exceeded=False,
            pdb_redo_is_twin=True,
            error="analysis was incomplete",
        )

        row = manifest_row(result, False, True, {}, {})

        assert row["retryable"] == "false"
        assert row["no_metals"] == "true"
        assert row["metal_site_limit_exceeded"] == "false"
        assert row["pdb_redo_is_twin"] == "true"
        assert row["status_detail"] == "analysis was incomplete"
        assert "error" not in row

    def test_unmeasured_twin_status_is_blank(self) -> None:
        row = manifest_row(entry_result(), False, True, {}, {})
        assert row["pdb_redo_is_twin"] == ""

    def test_empty_code_lists_render_blank(self) -> None:
        """No codes must not become a spurious separator or literal '[]'."""
        row = manifest_row(entry_result(), False, True, {}, {})
        assert row["reason_codes"] == ""
        assert row["warning_codes"] == ""

    def test_bonds_enabled_reports_the_measured_zero(self) -> None:
        row = manifest_row(entry_result(n_bonds=0, n_candidates=0), False, True, {}, {})
        assert row["n_bonds"] == 0
        assert row["n_candidates"] == 0

    def test_fresh_no_bonds_run_writes_blank_counts(self) -> None:
        """Without --resume there is no prior stage to carry forward."""
        row = manifest_row(
            entry_result(n_bonds=4, n_candidates=6),
            False,
            False,
            {"109m": "99"},
            {"109m": "98"},
        )
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_resume_no_bonds_carries_the_prior_counts_forward(self) -> None:
        row = manifest_row(
            entry_result("109M"), True, False, {"109m": "6"}, {"109m": "8"}
        )
        assert row["n_bonds"] == "6"
        assert row["n_candidates"] == "8"

    def test_resume_no_bonds_carry_forward_preserves_a_prior_zero(self) -> None:
        row = manifest_row(entry_result(), True, False, {"109m": "0"}, {"109m": "0"})
        assert row["n_bonds"] == "0"
        assert row["n_candidates"] == "0"

    def test_resume_no_bonds_without_a_prior_row_stays_blank(self) -> None:
        row = manifest_row(
            entry_result("1cll"), True, False, {"109m": "6"}, {"109m": "8"}
        )
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_prior_blank_counts_are_not_upgraded(self) -> None:
        row = manifest_row(entry_result(), True, False, {"109m": ""}, {"109m": ""})
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""


class TestWriteEntry:
    """One entry's rows, with the manifest row written last.

    The manifest is the completion marker ``--resume`` reads, so writing it
    first would let an interruption mark an entry whose statistics never
    reached disk as finished.
    """

    class _RecordingWriters:
        """Records the writer calls in order, which is what is under test.

        The real ``OutputWriters`` writes files and remembers nothing, so the
        casts below carry this recorder into the precisely-typed parameter.
        """

        def __init__(self) -> None:
            self.calls: list[str] = []

        def __getattr__(self, name: str) -> Callable[..., None]:
            def record(*args: Any) -> None:
                self.calls.append(name)

            return record

    @staticmethod
    def _args(**overrides: Any) -> RunConfig:
        fields: dict[str, Any] = {"resume": False, "bonds": True}
        fields.update(overrides)
        return run_config(**fields)

    def test_the_manifest_row_is_written_after_every_data_row(self) -> None:
        writers = self._RecordingWriters()
        plan = driver_confidence.ConfidencePlan()
        pool.write_entry(
            entry_result(),
            plan,
            cast(OutputWriters, writers),
            None,
            ({}, {}),
            resume=False,
            bonds=True,
        )
        assert writers.calls[-1] == "write_manifest_row"
        assert set(writers.calls[:-1]) == {
            "write_stats_rows",
            "write_bond_rows",
            "write_candidate_rows",
            "write_crystallization_rows",
            "write_density_context_row",
        }

    def test_a_staged_entry_is_registered_for_replacement(self) -> None:
        """Staging replaces rows by id, so every written entry must be listed."""
        # A real ``ResumeStaging`` would create a scratch directory to hold
        # rows nothing here writes; registration touches only the id set.
        staging = cast(resume.ResumeStaging, SimpleNamespace(replacement_ids=set()))
        pool.write_entry(
            entry_result(),
            driver_confidence.ConfidencePlan(),
            cast(OutputWriters, self._RecordingWriters()),
            staging,
            ({}, {}),
            resume=True,
            bonds=True,
        )
        assert staging.replacement_ids == {"109m"}
