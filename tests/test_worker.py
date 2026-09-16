"""Test per-entry worker outcomes: statuses, reason codes, and stage results."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import helpers
import pytest
from helpers import entry_result, worker_config

import analysis_config
import density_analysis as density
import scratch
import structure_analysis
from codes import DensityMapScope, EntryStatus
from driver import confidence as driver_confidence
from driver.writers import manifest_row
from edstats_statistics import EdstatsExtraction
from inputs import PdbRedoMetadata
from worker import contracts, inputs as worker_inputs, lifecycle, stages


def _read_resolution_stub(
    entry_dir: str, mtz_path: str, data_json_path: str | None = None
) -> float:
    del entry_dir, mtz_path, data_json_path
    return 2.0


def _read_map_column_resolution_stub(mtz_path: str) -> tuple[float, float]:
    del mtz_path
    return 20.0, 2.0


def _empty_metal_statistics(*_args: Any, **_kwargs: Any) -> EdstatsExtraction:
    return EdstatsExtraction([], [], {}, [])


def _manual_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    density_stage: Callable[..., Any],
    *,
    structure_builder: helpers.StructureBuilder | None = None,
    pdb_transform: Callable[[Path], None] | None = None,
    bonds: bool = False,
    **cfg_overrides: Any,
) -> contracts.EntryResult:
    """Run one manual-input entry through ``lifecycle.process``.

    Only the three MTZ-dependent readers are stubbed; structure loading, result
    assembly and status computation all run for real.
    """
    builder = structure_builder or helpers.StructureBuilder()
    if structure_builder is None:
        builder.add_metal("ZN", 1, chain="A", pos=(0.0, 0.0, 0.0))
        builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.05, 0.0, 0.0)})
    pdb_path = tmp_path / "entry.pdb"
    builder.write_pdb(str(pdb_path))
    if pdb_transform is not None:
        pdb_transform(pdb_path)
    mtz_path = tmp_path / "entry.mtz"
    mtz_path.write_bytes(b"unused: the readers below are stubbed")

    monkeypatch.setattr(worker_inputs, "read_resolution", _read_resolution_stub)
    monkeypatch.setattr(
        worker_inputs, "read_map_column_resolution", _read_map_column_resolution_stub
    )
    monkeypatch.setattr(stages, "run_density_analysis", density_stage)

    cfg = worker_config(
        input_root=str(tmp_path),
        pdb_redo_root=str(tmp_path),
        pdb_redo_cache=str(tmp_path),
        output_dir=str(tmp_path),
        bonds=bonds,
        density_map_scope="full",
        ccp4_timeout=900,
        manual_inputs={
            "pdb_file": str(pdb_path),
            "mtz_file": str(mtz_path),
            "cif_file": None,
            "data_json": None,
        },
        alchemy_commit="test",
        gemmi_version="test",
        ccp4_version="test",
        **cfg_overrides,
    )
    # monkeypatch restores ``worker_config``; ``initialize_worker`` has no
    # teardown counterpart.
    monkeypatch.setattr(lifecycle, "worker_config", cfg)
    return lifecycle.process("1abc")


@pytest.mark.parametrize(
    "exception,expected_code",
    [
        (ValueError("no FWT column"), "deterministic_processing_error"),
        (KeyError("NR"), "deterministic_processing_error"),
        (TypeError("bad shape"), "deterministic_processing_error"),
        (
            density.Ccp4EntryLimitationError("MAPMASK maxsec"),
            "deterministic_processing_error",
        ),
        (OSError("disk full"), "unexpected_processing_error"),
        # Only the inputs module's MissingInputError is a skip; a stage that
        # loses a file it already located is an error, like any other OSError.
        (FileNotFoundError("stats.out vanished"), "unexpected_processing_error"),
        (MemoryError(), "unexpected_processing_error"),
        (RuntimeError("fft failed"), "unexpected_processing_error"),
    ],
)
def test_an_unanticipated_failure_reports_whether_it_will_recur(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    expected_code: str,
) -> None:
    """Verify exception types distinguish deterministic and potentially transient failures.

    Both remain eligible for resume because inputs or software may change.
    """

    def failing_stage(*args: Any, **kwargs: Any) -> None:
        raise exception

    result = _manual_entry(tmp_path, monkeypatch, failing_stage)

    assert result.status == "error"
    assert result.reason_codes == [expected_code]
    assert result.retryable is True
    assert type(exception).__name__ in result.status_detail


def test_a_missing_manual_input_skips_the_entry_as_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inputs module's ``MissingInputError`` is the skip signal.

    No stage is stubbed: ``resolve_manual_inputs`` itself reports the absent
    MTZ, and the worker maps that to ``skip`` / ``missing_input`` with the
    reader's message in ``status_detail``.
    """
    pdb_path = tmp_path / "entry.pdb"
    helpers.StructureBuilder().write_pdb(str(pdb_path))
    absent_mtz = tmp_path / "absent.mtz"
    cfg = worker_config(
        output_dir=str(tmp_path),
        manual_inputs={
            "pdb_file": str(pdb_path),
            "mtz_file": str(absent_mtz),
            "cif_file": None,
            "data_json": None,
        },
    )
    monkeypatch.setattr(lifecycle, "worker_config", cfg)

    result = lifecycle.process("1abc")

    assert result.status == "skip"
    assert result.reason_codes == ["missing_input"]
    assert result.status_detail == f"missing input: mtz file not found: {absent_mtz}"
    assert result.retryable is True


def _real_stats_density_stage(
    pdb_id: str, *, pdb_path: str, out_dir: str, **_kwargs: Any
) -> density.DensityResult:
    """Write a synthetic ``stats.out`` covering every residue of the model."""
    context = structure_analysis.load_structure(pdb_id, pdb_path)
    stats_out = helpers.write_edstats_for_structure(
        os.path.join(out_dir, "stats.out"), context
    )
    return density.DensityResult(
        stats_out=stats_out,
        rszd="/nonexistent/rszd.pdb",
        fo_map="/nonexistent/fo.map",
        df_map="/nonexistent/df.map",
        mtz_for_maps="/nonexistent/entry.mtz",
        mtzfix_log="/nonexistent/mtzfix.log",
        mtzfix_applied=False,
        timings={"edstats_s": 1.0},
        twin_coefficient_normalization_applied=False,
        twin_coefficient_normalization=None,
        density_map_scope_requested="model-envelope",
        density_map_scope_used=DensityMapScope.MODEL_ENVELOPE,
        full_map_bytes=8192,
        edstats_map_bytes=2048,
    )


def test_manifest_twin_flag_uses_the_density_routing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest and map route consume one authoritative ISTWIN value."""

    def twin_metadata(
        data_json_path: str | None, *, required: bool = False
    ) -> PdbRedoMetadata:
        del data_json_path, required
        return PdbRedoMetadata(is_twin=True, version="8.04", date="2024-02-08")

    monkeypatch.setattr(
        worker_inputs,
        "read_pdb_redo_metadata",
        twin_metadata,
    )

    result = _manual_entry(tmp_path, monkeypatch, _real_stats_density_stage)

    assert result.pdb_redo.pdb_redo_is_twin is True
    assert result.pdb_redo.pdb_redo_version == "8.04"
    assert result.pdb_redo.pdb_redo_date == "2024-02-08"


@pytest.mark.parametrize("bonds", [True, False], ids=["bonds", "no-bonds"])
def test_a_metal_outside_the_catalog_retains_density_with_or_without_bonds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bonds: bool
) -> None:
    """Coordinate and density selection must agree despite a catalog gap."""
    builder = helpers.StructureBuilder()
    builder.add_hetero_residue(
        "ZZZ",
        1,
        [
            helpers.AtomSpec(name="ZN", element="ZN", pos=(0.0, 0.0, 0.0)),
            helpers.AtomSpec(name="C1", element="C", pos=(1.5, 0.0, 0.0)),
        ],
        chain="B",
    )
    builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.03, 0.0, 0.0)})

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=bonds,
    )

    assert "metal_site_without_density" not in result.reason_codes
    assert "cofactor_catalog_fallback" in result.warning_codes
    assert result.status == ("partial" if bonds else "ok")
    assert result.retryable is False
    assert result.n_metals == 1
    if bonds:
        assert result.n_bonds, "bond analysis should retain the measured contact"
    else:
        assert result.n_bonds == 0
    assert len(result.rows) == 1
    assert result.rows[0].category == "cofactor"
    assert result.rows[0].resname == "ZZZ"
    assert result.rows[0].selected_metal_site_status == "selected"
    assert result.rows[0].density_scope == "cofactor_residue"


def test_a_zero_occupancy_metal_is_excluded_but_not_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``no_metals`` must not read as an authoritative negative about this file.

    A metal modeled at zero occupancy is not evidence for a site, so it is
    rightly excluded from selection. The exclusion used to be silent, leaving
    an entry whose only metal is modeled absent reporting no metals at all --
    and the generic ``zero_occupancy_atoms`` warning cannot say which atom.
    """
    builder = helpers.StructureBuilder()
    builder.add_hetero_residue(
        "ZN",
        1,
        [helpers.AtomSpec(name="ZN", element="ZN", pos=(0.0, 0.0, 0.0), occupancy=0.0)],
        chain="B",
    )

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=True,
    )

    assert result.n_metals == 0
    assert result.no_metals is True
    assert "zero_occupancy_metal_excluded" in result.warning_codes


def test_a_positive_occupancy_metal_raises_no_exclusion_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warning marks a real exclusion, not merely the presence of a metal."""
    builder = helpers.StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.03, 0.0, 0.0)})

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=True,
    )

    assert result.n_metals == 1
    assert "zero_occupancy_metal_excluded" not in result.warning_codes


def test_a_metal_dense_entry_finishes_before_density_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = helpers.StructureBuilder()
    for residue_number in range(1, analysis_config.MAX_ANALYZED_METAL_SITES + 2):
        builder.add_metal(
            "MG",
            residue_number,
            chain="A",
            pos=(float(residue_number), 0.0, 0.0),
        )

    def density_must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the metal-site pre-check must run before CCP4")

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        density_must_not_run,
        structure_builder=builder,
        bonds=True,
    )

    assert result.status == "ok"
    assert result.retryable is False
    assert result.no_metals is False
    assert result.metal_site_limit_exceeded is True
    assert result.reason_codes == ["metal_site_limit_exceeded"]
    assert result.n_metals == analysis_config.MAX_ANALYZED_METAL_SITES + 1
    assert result.n_bonds == 0
    assert result.n_candidates == 0
    assert result.rows == []
    assert result.bond_rows == []
    assert result.candidate_rows == []
    assert (
        driver_confidence.confidence_rows_for(
            result, driver_confidence.ConfidencePlan()
        )
        == []
    )


def test_the_metal_site_limit_includes_exactly_one_hundred_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = helpers.StructureBuilder()
    for residue_number in range(1, analysis_config.MAX_ANALYZED_METAL_SITES + 1):
        builder.add_metal(
            "MG",
            residue_number,
            chain="A",
            pos=(float(residue_number), 0.0, 0.0),
        )

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=False,
    )

    assert result.metal_site_limit_exceeded is False
    assert result.n_metals == analysis_config.MAX_ANALYZED_METAL_SITES
    assert len(result.rows) == analysis_config.MAX_ANALYZED_METAL_SITES


def test_a_catalogued_cofactor_metal_raises_no_such_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-atom ion is reported by both stages, so nothing is flagged."""
    builder = helpers.StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.03, 0.0, 0.0)})

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=True,
    )

    assert "metal_site_without_density" not in result.reason_codes
    assert result.n_metals == 1


def test_non_finite_selected_metal_is_partial_without_bond_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coordinate validation is not disabled together with bond output."""
    builder = helpers.StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))

    def make_metal_x_nan(path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        rewritten: list[str] = []
        for line in lines:
            if line[:6].strip() == "HETATM" and line[76:78].strip() == "ZN":
                line = line[:30] + "     nan" + line[38:]
            rewritten.append(line)
        path.write_text("".join(rewritten), encoding="utf-8")

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        pdb_transform=make_metal_x_nan,
        bonds=False,
    )

    assert result.status == "partial"
    assert result.retryable is False
    assert result.n_metals == 1
    assert "non_finite_coordinates" in result.warning_codes
    assert "non_finite_metal_coordinates" in result.reason_codes


def test_an_unanticipated_failure_logs_its_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The manifest keeps one line; the debug log must keep the stack.

    Without this the only way to locate an unanticipated failure was to rerun
    the entry by hand under ``--keep-intermediates``.
    """

    def failing_stage(*args: Any, **kwargs: Any) -> None:
        raise ValueError("no FWT column")

    with caplog.at_level(logging.DEBUG, logger="alchemy.lifecycle"):
        result = _manual_entry(tmp_path, monkeypatch, failing_stage)

    assert result.status == "error"
    records = [r for r in caplog.records if r.exc_info and "1abc" in r.getMessage()]
    assert records, "the failing entry logged no traceback"
    # The comprehension already selected on ``exc_info``; the cast only carries
    # that through to the subscript.
    assert cast(tuple[Any, ...], records[0].exc_info)[0] is ValueError


def test_bond_stage_failure_invalidates_confidence_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed geometry stage is not legitimate density-only evidence."""
    result = entry_result()
    inputs = worker_inputs.EntryInputs(
        work_dir="/nonexistent",
        mtz="/nonexistent/entry.mtz",
        pdb="/nonexistent/entry.pdb",
        data_json=None,
        data_reshi=2.0,
        map_reslo=50.0,
        map_reshi=2.0,
        pdb_redo_is_twin=False,
        source_coordinate_path="/nonexistent/entry.cif",
    )
    # The failing bond stage reads warnings only, so a full parsed context is unnecessary.
    structure = cast(
        structure_analysis.StructureContext, SimpleNamespace(warning_codes=[])
    )

    def fail_bond_analysis(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("geometry unavailable")

    monkeypatch.setattr(stages, "run_bond_analysis", fail_bond_analysis)

    outcome = stages.run_bond_stage(
        "109m", worker_config(bonds=True), inputs, structure, [], [], []
    )
    analysis = outcome.analysis

    assert (analysis.bond_rows, analysis.candidate_rows, analysis.site_summaries) == (
        [],
        [],
        {},
    )
    assert outcome.failed and outcome.status_detail.startswith("bond: RuntimeError")
    lifecycle._apply_bond_outcome(result, outcome)  # pyright: ignore[reportPrivateUsage]
    assert result.reason_codes == ["bond_stage_failure"]
    assert result.confidence_inputs_missing_reason == "bond_stage_failure"
    assert lifecycle.retryable_for(EntryStatus.PARTIAL, result.reason_codes) is True


def test_a_bond_failure_after_a_density_failure_keeps_both_on_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The later failure is appended; it must not erase the earlier one.

    A timed-out density stage names the stalled tool in ``status_detail`` and
    is the first reason confidence inputs are missing. A geometry crash that
    follows keeps its own code, but leaves the counts unmeasured rather than
    reporting a measured zero.
    """

    def fake_density(*args: Any, **kwargs: Any) -> None:
        raise density.Ccp4ToolTimeoutError(
            tool="edstats",
            timeout_s=900,
            elapsed_s=900.4,
            log_path="",
            timings={},
        )

    def fail_bond_analysis(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("geometry unavailable")

    monkeypatch.setattr(stages, "run_bond_analysis", fail_bond_analysis)
    result = _manual_entry(tmp_path, monkeypatch, fake_density, bonds=True)

    assert result.status == "partial"
    assert result.reason_codes == ["ccp4_tool_timeout", "bond_stage_failure"]
    assert "edstats" in result.status_detail
    assert "bond: RuntimeError: geometry unavailable" in result.status_detail
    assert result.confidence_inputs_missing_reason == "ccp4_tool_timeout"
    assert result.n_metals == 1
    assert result.n_bonds is None
    assert result.n_candidates is None
    assert result.retryable is True


def test_an_already_capped_detail_still_admits_appended_messages() -> None:
    """Capping the join again would silently drop the later stage's messages."""
    cap = stages.MAX_MANIFEST_STATUS_DETAIL_CHARS
    existing = "density unavailable: " + "x" * cap
    detail = lifecycle._appended_detail(  # pyright: ignore[reportPrivateUsage]
        existing, ["bond: RuntimeError: geometry unavailable"]
    )

    assert len(detail) <= cap
    assert detail.startswith("density unavailable: ")
    assert detail.endswith("bond: RuntimeError: geometry unavailable")


@pytest.mark.parametrize(
    "status, reason_codes, expected",
    [
        (EntryStatus.OK, [], False),
        (EntryStatus.OK, ["metal_site_limit_exceeded"], False),
        (EntryStatus.SKIP, ["missing_input"], True),
        (EntryStatus.ERROR, ["unexpected_processing_error"], True),
        (EntryStatus.ERROR, ["deterministic_processing_error"], True),
        (EntryStatus.ERROR, ["worker_process_died"], True),
        (EntryStatus.PARTIAL, ["ccp4_tool_timeout"], True),
        (EntryStatus.PARTIAL, ["bond_stage_failure"], True),
        (EntryStatus.PARTIAL, ["mtzfix_validation_failure"], False),
        (
            EntryStatus.PARTIAL,
            ["mtzfix_validation_failure", "bond_stage_failure"],
            True,
        ),
        (EntryStatus.PARTIAL, ["metal_presence_indeterminate"], False),
        (EntryStatus.PARTIAL, ["metal_site_without_density"], False),
    ],
)
def test_retry_policy_follows_from_status_and_reason_codes(
    status: EntryStatus, reason_codes: list[str], expected: bool
) -> None:
    """``retryable`` is derived once, so no stage can overwrite another's verdict.

    Deterministic errors stay retryable because a resume may read repaired
    inputs; a partial entry is retried only when the failed stage reported
    nothing about the entry itself.
    """
    assert lifecycle.retryable_for(status, reason_codes) is expected


class TestNoRecognizedMetalOutcome:
    @staticmethod
    def _density_must_not_run(*args: Any, **kwargs: Any) -> None:
        pytest.fail("density analysis ran without a recognized metal site")

    @staticmethod
    def _blank_metal_element(path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        rewritten: list[str] = []
        for line in lines:
            if line.startswith("HETATM") and line[12:16].strip() == "ZN":
                newline = "\n" if line.endswith("\n") else ""
                record = line.removesuffix("\n").ljust(80)
                line = record[:76] + "  " + record[78:] + newline
            rewritten.append(line)
        path.write_text("".join(rewritten), encoding="utf-8")

    def test_unknown_element_makes_metal_presence_indeterminate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builder = helpers.StructureBuilder()
        builder.add_metal("ZN", 1, chain="A")

        result = _manual_entry(
            tmp_path,
            monkeypatch,
            self._density_must_not_run,
            structure_builder=builder,
            pdb_transform=self._blank_metal_element,
        )

        assert result.status == "partial"
        assert result.retryable is False
        assert result.reason_codes == ["metal_presence_indeterminate"]
        assert "1 atom(s)" in result.status_detail
        assert result.n_metals == 0
        assert result.n_bonds is None
        assert result.n_candidates is None
        assert result.no_metals is False
        assert "unknown_elements" in result.warning_codes
        assert result.confidence_inputs_missing_reason == "metal_presence_indeterminate"
        assert (
            driver_confidence.confidence_rows_for(
                result, driver_confidence.ConfidencePlan()
            )
            == []
        )

    def test_known_nonmetal_structure_remains_an_authoritative_negative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builder = helpers.StructureBuilder()
        builder.add_amino_acid("GLY", 1, chain="A")

        result = _manual_entry(
            tmp_path,
            monkeypatch,
            self._density_must_not_run,
            structure_builder=builder,
        )

        assert result.status == "ok"
        assert result.retryable is False
        assert result.reason_codes == []
        assert result.status_detail == ""
        assert result.n_metals == 0
        assert result.n_bonds == 0
        assert result.n_candidates == 0
        assert result.no_metals is True

    def test_zero_occupancy_metal_is_not_an_analyzable_site(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builder = helpers.StructureBuilder()
        builder.add_metal("ZN", 1, chain="A", occupancy=0.0)

        result = _manual_entry(
            tmp_path,
            monkeypatch,
            self._density_must_not_run,
            structure_builder=builder,
        )

        assert result.status == "ok"
        assert result.retryable is False
        assert result.reason_codes == []
        assert result.n_metals == 0
        assert result.n_bonds == 0
        assert result.n_candidates == 0
        assert result.no_metals is True
        assert "zero_occupancy_atoms" in result.warning_codes


class TestDensityResultReachesTheResult:
    """The worker must report the density stage's own answers, not its request."""

    @staticmethod
    def density_result(**overrides: Any) -> density.DensityResult:
        fields: dict[str, Any] = {
            "stats_out": "/nonexistent/stats.out",
            "rszd": "/nonexistent/rszd.pdb",
            "fo_map": "/nonexistent/fo.map",
            "df_map": "/nonexistent/df.map",
            "mtz_for_maps": "/nonexistent/entry.mtz",
            "mtzfix_log": "/nonexistent/mtzfix.log",
            "mtzfix_applied": False,
            "timings": {"edstats_s": 12.5},
            "twin_coefficient_normalization_applied": False,
            "twin_coefficient_normalization": None,
            "density_map_scope_requested": "model-envelope",
            "density_map_scope_used": "model-envelope",
            "full_map_bytes": 8192,
            "edstats_map_bytes": 2048,
        }
        fields.update(overrides)
        fields["density_map_scope_used"] = DensityMapScope(
            fields["density_map_scope_used"]
        )
        return density.DensityResult(**fields)

    def _entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: density.DensityResult,
        **cfg_overrides: Any,
    ) -> contracts.EntryResult:
        monkeypatch.setattr(stages, "extract_metal_statistics", _empty_metal_statistics)

        def density_stage(*_args: Any, **_kwargs: Any) -> density.DensityResult:
            return result

        return _manual_entry(tmp_path, monkeypatch, density_stage, **cfg_overrides)

    def test_the_scope_reported_is_the_one_achieved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crop that could not be applied must not be reported as applied.

        Requested and achieved scope are separate fields because they disagree:
        an envelope larger than the map, or one past the cell edge, falls back
        to the full map.
        """
        result = self._entry(
            tmp_path,
            monkeypatch,
            self.density_result(
                density_map_scope_requested="model-envelope",
                density_map_scope_used="full-extent-fallback",
            ),
        )

        assert result.density.density_map_scope_used == "full-extent-fallback"
        assert result.density.density_full_map_bytes == 8192
        assert result.density.density_edstats_map_bytes == 2048

    def test_density_timings_reach_the_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """They are how a run log attributes an entry's cost to a CCP4 program."""
        result = self._entry(
            tmp_path, monkeypatch, self.density_result(timings={"edstats_s": 12.5})
        )

        assert result.timings["edstats_s"] == 12.5
        assert "density_total_s" in result.timings

    def test_a_normalized_twin_entry_is_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The normalization is a real modification and must stay visible."""
        result = self._entry(
            tmp_path,
            monkeypatch,
            self.density_result(
                twin_coefficient_normalization_applied=True,
                twin_coefficient_normalization={"usable_reflections": 10},
            ),
        )

        assert "twin_refmac_coefficients_normalized" in result.warning_codes

    @pytest.mark.parametrize("keep", [False, True])
    def test_keep_intermediates_decides_the_scratch_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: bool
    ) -> None:
        """The scratch_dir directory holds the maps, so only the flag may keep it."""
        self._entry(
            tmp_path, monkeypatch, self.density_result(), keep_intermediates=keep
        )

        scratch_dir = [
            name for name in os.listdir(tmp_path) if name.startswith(".alchemy-")
        ]

        assert bool(scratch_dir) is keep
        if keep:
            marker = tmp_path / scratch_dir[0] / scratch.SCRATCH_MARKER_FILENAME
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            assert metadata["preserve"] is True


def test_a_loaded_structure_fills_in_the_model_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify model provenance is populated only after structure loading.

    The model fields start unmeasured, like the bond counts, so an entry that
    failed before ``load_structure`` cannot claim a model count; a successful
    load must fill them in.
    """
    monkeypatch.setattr(stages, "extract_metal_statistics", _empty_metal_statistics)

    def density_stage(*_args: Any, **_kwargs: Any) -> density.DensityResult:
        return TestDensityResultReachesTheResult.density_result()

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        density_stage,
    )

    assert result.coordinates.input_model_count == 1
    assert result.coordinates.model_analyzed == 1
    assert result.coordinates.multi_model_structure is False

    row = manifest_row(result, False, True, {}, {})
    assert row["input_model_count"] == 1
    assert row["model_analyzed"] == 1


class TestCcp4TimeoutOutcome:
    """A stalled CCP4 program must be a named, retryable entry outcome.

    The generic ``except Exception`` handler would report
    ``unexpected_processing_error``, hiding a stalled installation behind an
    error code shared with real bugs.
    """

    @staticmethod
    def _entry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> contracts.EntryResult:
        """Run one manual-input entry whose density stage raises ``failure``."""

        def fake_density(*args: Any, **kwargs: Any) -> None:
            raise failure

        return _manual_entry(tmp_path, monkeypatch, fake_density)

    def test_timeout_is_reported_as_a_named_retryable_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="edstats",
                timeout_s=900,
                elapsed_s=900.4,
                log_path=str(tmp_path / "edstats.log"),
                timings={"edstats_s": 900.4},
            ),
        )

        assert result.reason_codes == ["ccp4_tool_timeout"]
        assert result.retryable is True, (
            "the program was killed for running too long and reported nothing "
            "about the entry, so it must stay eligible for retry"
        )
        assert result.status == "partial"
        assert "edstats" in result.status_detail
        assert result.confidence_inputs_missing_reason == "ccp4_tool_timeout"
        # The abandoned attempt still cost time, and the run log reports it.
        assert result.timings.get("edstats_s") == 900.4

    def test_the_partial_log_survives_scratch_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify timeout diagnostics preserve only the partial tool log.

        ``Ccp4ToolTimeoutError`` names a partial log, but ``process`` deletes
        the scratch directory unless --keep-intermediates is given, so the log
        alone is copied out: the maps beside it can be hundreds of megabytes.
        """
        scratch_log = tmp_path / "scratch_edstats.log"
        scratch_log.write_text("edstats output before the stall\n", encoding="utf-8")

        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="edstats",
                timeout_s=900,
                elapsed_s=900.4,
                log_path=str(scratch_log),
                timings={"edstats_s": 900.4},
            ),
        )

        kept = result.density.ccp4_timeout_log_path
        assert kept, "the timeout log path must be recorded on the result"
        assert os.path.isfile(kept), (
            f"the retained log is missing at {kept}; the path named in the "
            "timeout message must outlive the scratch directory"
        )
        assert stages.TIMEOUT_LOG_DIRNAME in kept
        with open(kept, encoding="utf-8") as handle:
            assert "before the stall" in handle.read()

    def test_only_the_log_is_kept_not_the_scratch_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "edstats.log").write_text("stalled\n", encoding="utf-8")
        huge_map = scratch / "1abc_fo.map"
        huge_map.write_bytes(b"0" * 4096)

        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="edstats",
                timeout_s=900,
                elapsed_s=900.4,
                log_path=str(scratch / "edstats.log"),
                timings={},
            ),
        )

        kept_dir = os.path.dirname(result.density.ccp4_timeout_log_path)
        assert os.listdir(kept_dir) == ["1abc_edstats_timeout.log"], (
            "only the log belongs in the retained directory"
        )

    def test_a_missing_log_is_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="fft",
                timeout_s=900,
                elapsed_s=900.1,
                log_path=str(tmp_path / "never_written.log"),
                timings={},
            ),
        )

        assert result.density.ccp4_timeout_log_path == ""
        assert result.reason_codes == ["ccp4_tool_timeout"]
        assert result.retryable is True

    def test_a_validation_failure_remains_terminal_and_distinct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The neighbouring handler keeps its own, non-retryable meaning."""
        result = self._entry(
            tmp_path,
            monkeypatch,
            density.MtzfixValidationError("bad coefficients", timings={}),
        )

        assert result.reason_codes == ["mtzfix_validation_failure"]
        assert result.retryable is False
