"""Test batch-driver data management.

Coverage includes argument validation, resume bookkeeping, manifest projection,
staged retries, and per-entry worker outcomes as the driver records them.

Streamed output writers, the coordinate-preparation converters, worker-death
recovery, and end-to-end pipeline runs are covered elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import helpers
import pytest
from helpers import entry_result, read_csv

import cli
import density_analysis as density
import main
import structure_analysis
import worker
import worker_contracts
from codes import EntryStatus
from coordination import schema as coordination_schema
from driver import dispatch, environment, output_lock, pool, resources, resume, runlog
from driver import pool as driver_pool
from driver.progress import ProgressReporter
from driver.runlog import RunLog
from driver.writers import (
    MANIFEST_COLUMNS,
    MANIFEST_FIELDS,
    STATS_COLUMNS,
    OutputTargets,
    OutputWriters,
    manifest_row,
)
from inputs import PdbRedoMetadata
from output_rows import MetalStatsRow
from run_config import RunConfig


def _read_resolution_stub(
    entry_dir: str, mtz_path: str, data_json_path: str | None = None
) -> float:
    del entry_dir, mtz_path, data_json_path
    return 2.0


def _read_map_column_resolution_stub(mtz_path: str) -> tuple[float, float]:
    del mtz_path
    return 20.0, 2.0


def _resolved_ccp4_environment(
    _args: RunConfig,
) -> tuple[dict[str, str], None]:
    return dict(os.environ), None


def _empty_metal_statistics(
    *_args: Any, **_kwargs: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    return [], []


def _cfg(**overrides: Any) -> worker_contracts.WorkerConfig:
    """A complete ``WorkerConfig`` with test placeholders."""
    fields: dict[str, Any] = {
        "root": "/nonexistent/root",
        "mirror_root": "/nonexistent/mirror",
        "cache_root": None,
        "env": {},
        "output_dir": "/nonexistent/output",
        "cofactors": frozenset(),
        "keep": False,
        "bonds": True,
        "density_map_scope": "model-envelope",
        "ccp4_timeout_s": density.CCP4_TOOL_TIMEOUT_S,
        "log_level": logging.INFO,
        "allow_download": False,
        "manual_inputs": None,
        "alchemy_commit": "abc123def456",
        "gemmi_version": "0.7.5",
        "ccp4_version": "9.0",
        "reference_data_id": "0123456789ab",
        "analysis_config_id": "alchemy-analysis-config-test",
    }
    fields.update(overrides)
    return worker_contracts.WorkerConfig(**fields)


CFG = _cfg()


def _run_config(**overrides: Any) -> RunConfig:
    return replace(cli.parse_args([]), **overrides)


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


def _manual_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    density_stage: Callable[..., Any],
    *,
    structure_builder: helpers.StructureBuilder | None = None,
    pdb_transform: Callable[[Path], None] | None = None,
    bonds: bool = False,
    **cfg_overrides: Any,
) -> worker_contracts.EntryResult:
    """Run one manual-input entry through ``worker.process``.

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

    monkeypatch.setattr(worker, "read_resolution", _read_resolution_stub)
    monkeypatch.setattr(
        worker, "read_map_column_resolution", _read_map_column_resolution_stub
    )
    monkeypatch.setattr(worker, "run_density_analysis", density_stage)

    cfg = _cfg(
        root=str(tmp_path),
        mirror_root=str(tmp_path),
        cache_root=str(tmp_path),
        output_dir=str(tmp_path),
        bonds=bonds,
        density_map_scope="full",
        ccp4_timeout_s=900,
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
    monkeypatch.setattr(worker, "worker_config", cfg)
    return worker.process("1abc")


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
    assert type(exception).__name__ in result.error


def _real_stats_density_stage(
    pdb_id: str, mtz: str, pdb: str, work_dir: str, *args: Any, **kwargs: Any
) -> density.DensityResult:
    """Write a synthetic ``stats.out`` covering every residue of the model."""
    context = structure_analysis.load_structure(pdb_id, pdb)
    stats_out = helpers.write_edstats_for_structure(
        os.path.join(work_dir, "stats.out"), context
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
        density_map_scope_used="model-envelope",
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
        worker,
        "read_pdb_redo_metadata",
        twin_metadata,
    )

    result = _manual_entry(tmp_path, monkeypatch, _real_stats_density_stage)

    assert result.pdb_redo_is_twin is True
    assert result.pdb_redo_version == "8.04"
    assert result.pdb_redo_date == "2024-02-08"


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
    for residue_number in range(1, worker_contracts.MAX_ANALYZED_METAL_SITES + 2):
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
    assert result.n_metals == worker_contracts.MAX_ANALYZED_METAL_SITES + 1
    assert result.n_bonds == 0
    assert result.n_candidates == 0
    assert result.rows == []
    assert result.bond_rows == []
    assert result.candidate_rows == []
    assert pool.confidence_rows_for(result, pool.ConfidencePlan()) == []


def test_the_metal_site_limit_includes_exactly_one_hundred_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = helpers.StructureBuilder()
    for residue_number in range(1, worker_contracts.MAX_ANALYZED_METAL_SITES + 1):
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
    assert result.n_metals == worker_contracts.MAX_ANALYZED_METAL_SITES
    assert len(result.rows) == worker_contracts.MAX_ANALYZED_METAL_SITES


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

    with caplog.at_level(logging.DEBUG, logger="alchemy.worker"):
        result = _manual_entry(tmp_path, monkeypatch, failing_stage)

    assert result.status == "error"
    records = [r for r in caplog.records if r.exc_info and "1abc" in r.getMessage()]
    assert records, "the failing entry logged no traceback"
    # The comprehension already selected on ``exc_info``; the cast only carries
    # that through to the subscript.
    assert cast(tuple[Any, ...], records[0].exc_info)[0] is ValueError


def _write_manifest(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str] | None = None,
) -> str:
    """Write a manifest CSV with the real schema and the given partial rows."""
    columns = list(columns if columns is not None else MANIFEST_COLUMNS)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            values = {
                "no_metals": "false",
                "metal_site_limit_exceeded": "false",
                **row,
            }
            writer.writerow({column: values.get(column, "") for column in columns})
    return str(path)


def _manifest_ids(path: str | Path, **kwargs: Any) -> set[str]:
    return resume.load_done(str(path), **kwargs)


def test_bond_stage_failure_invalidates_confidence_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed geometry stage is not legitimate density-only evidence."""
    result = entry_result()
    inputs = worker.EntryInputs(
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

    monkeypatch.setattr(worker, "run_bond_analysis", fail_bond_analysis)

    analysis = worker.run_bond_stage(
        result, _cfg(bonds=True), inputs, structure, [], []
    )

    assert (analysis.bond_rows, analysis.candidate_rows, analysis.site_summaries) == (
        [],
        [],
        {},
    )
    assert result.reason_codes == ["bond_stage_failure"]
    assert result.confidence_inputs_missing_reason == "bond_stage_failure"
    assert worker.retryable_for(EntryStatus.PARTIAL, result.reason_codes) is True


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
    assert worker.retryable_for(status, reason_codes) is expected


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
        assert "1 atom(s)" in result.error
        assert result.n_metals == 0
        assert result.n_bonds is None
        assert result.n_candidates is None
        assert result.no_metals is False
        assert "unknown_elements" in result.warning_codes
        assert result.confidence_inputs_missing_reason == "metal_presence_indeterminate"
        assert pool.confidence_rows_for(result, pool.ConfidencePlan()) == []

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
        assert result.error == ""
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


# Hand-written because the behaviours under test are raw ``.``/``?`` occupancy
# tokens and >3-character component ids, neither of which gemmi would write.
class TestPositiveInt:
    """``positive_int`` is the argparse gate for --workers/--max-pdbs/etc."""

    # Use Any to test non-string programmatic inputs outside the annotated CLI contract.
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", 1),
            ("2", 2),
            ("28", 28),
            ("  7  ", 7),  # int() tolerates surrounding whitespace
            ("+3", 3),
            ("1000000", 1000000),
            (5, 5),  # already an int (programmatic callers)
        ],
    )
    def test_accepts_integers_of_at_least_one(self, value: Any, expected: int) -> None:
        assert cli.positive_int(value) == expected
        assert isinstance(cli.positive_int(value), int)

    @pytest.mark.parametrize("value", ["0", "-1", "-28", 0, -3])
    def test_rejects_zero_and_negative(self, value: Any) -> None:
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            cli.positive_int(value)
        assert "at least 1" in str(excinfo.value)

    @pytest.mark.parametrize(
        "value",
        [
            "1.5",
            "2.0",
            "abc",
            "",
            " ",
            "1e3",
            "0x2",
            None,
            "nan",
            "inf",
            "1,000",
        ],
    )
    def test_rejects_non_integer_text(self, value: Any) -> None:
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            cli.positive_int(value)
        assert "positive integer" in str(excinfo.value)

    def test_boundary_is_one_not_zero(self) -> None:
        assert cli.positive_int("1") == 1
        with pytest.raises(argparse.ArgumentTypeError):
            cli.positive_int("0")


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
        path = _write_manifest(
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
        path = _write_manifest(
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

    def test_ids_are_normalized_to_lowercase(self, tmp_path: Path) -> None:
        """Manifest IDs join against the driver's lowercased selection list."""
        path = _write_manifest(
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
        path = _write_manifest(
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
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok", "retryable": "False"}],
        )
        with open(path, "a", newline="") as handle:
            csv.writer(handle).writerow(["1cll", "ok"])

        assert _manifest_ids(path, bonds_required=False) == {"109m"}

    def test_row_without_a_pdb_id_is_not_done(self, tmp_path: Path) -> None:
        path = _write_manifest(
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
        path = _write_manifest(
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
        path = _write_manifest(
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
        path = _write_manifest(
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
        path = _write_manifest(
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
        path = _write_manifest(
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
        path = _write_manifest(
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
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok"}],
            columns=["pdbID", "status"],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": ""}

    def test_truncated_rows_do_not_supply_carry_forward_values(
        self, tmp_path: Path
    ) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok", "n_bonds": "7"}],
        )
        with open(path, "a", newline="") as handle:
            csv.writer(handle).writerow(["1cll", "ok"])

        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "7"}

    def test_blank_ids_are_dropped(self, tmp_path: Path) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "", "status": "ok", "n_bonds": "9"},
                {"pdbID": "109m", "status": "ok", "n_bonds": "1"},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "1"}

    def test_a_later_row_supersedes_an_earlier_one(self, tmp_path: Path) -> None:
        """Resume appends, so the last row for an ID is the current one."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "109m", "status": "error", "n_bonds": ""},
                {"pdbID": "109m", "status": "ok", "n_bonds": "5"},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "5"}


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
                _cfg(manual_inputs={"cif_file": manual_path}),
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


class TestUnrunBondStageChain:
    """Verify an unrun bond stage remains eligible after a density-only resume.

    Seed a pre-bond failure, resume without bonds, then resume with bonds.
    A placeholder zero must not mark the unrun stage complete.
    """

    @staticmethod
    def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_failed_bond_enabled_entry_is_not_marked_bond_complete(
        self, tmp_path: Path
    ) -> None:
        """Step 1: a pre-bond failure writes blank counts, so it is retried."""
        result = entry_result(
            "109m", status="error", retryable=True, error="density stage failed"
        )
        row = manifest_row(result, False, True, {}, {})
        manifest = tmp_path / "manifest.csv"
        self._write_rows(manifest, [row])

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
                "109m", status="error", retryable=True, error="edstats failed"
            ),
            resume=False,
            bonds_enabled=True,
            prior_bond_counts={},
            prior_candidate_counts={},
        )
        self._write_rows(manifest, [failed])

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
        self._write_rows(manifest, [recovered])

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
        self._write_rows(manifest, [completed])
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
        layout = driver_pool.OutputLayout(str(output_dir))
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
        layout = driver_pool.OutputLayout(str(output_dir))
        headers = (
            (layout.manifest, MANIFEST_COLUMNS),
            (layout.stats, STATS_COLUMNS),
            (layout.bonds, coordination_schema.BOND_COLUMNS),
            (layout.candidates, coordination_schema.CANDIDATE_COLUMNS),
        )
        for path, columns in headers:
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
        prior = dict.fromkeys(MANIFEST_COLUMNS, "")
        prior.update(pdbID="aaaa", status="ok", retryable="False")
        with open(layout.manifest, "a", newline="") as handle:
            csv.writer(handle).writerow([prior[name] for name in MANIFEST_COLUMNS])

        def _dispatch(
            ids: Sequence[str],
            cfg: worker_contracts.WorkerConfig,
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
        args = _run_config(resume=True, bonds=True, output_dir=str(output_dir))
        run_log = RunLog(args, "pytest")

        with pytest.raises(OSError if fail_merge else KeyboardInterrupt):
            driver_pool.process_entries(
                args,
                ["bbbb", "cccc"],
                # ``None`` proves the halted-batch path never reaches the worker
                # configuration: ``_dispatch_entries`` is replaced above.
                cast(worker_contracts.WorkerConfig, None),
                1,
                layout,
                driver_pool.ConfidencePlan(),
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
            recovery = Path(run_log.summary["resume_staging_recovery_dir"])
            assert recovery.is_dir()
            output_lock.sweep_owned_scratch_directories(str(output_dir))
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
        assert run_log.summary["resume_entries_committed_after_interrupt"] == 2

    def _interrupt(
        self,
        staging: resume.ResumeStaging,
        tmp_path: Path,
        *,
        bonds: bool = True,
        confidence: bool = False,
    ) -> dict[str, Any]:
        """Run the halted-resume path and return the run-log summary."""
        args = _run_config(bonds=bonds, output_dir=str(tmp_path))
        plan = driver_pool.ConfidencePlan()
        # ``enabled`` is derived from the mode, which is what a scoring run sets.
        plan.mode = "reference" if confidence else None
        run_log = RunLog(args, "pytest")
        driver_pool.keep_completed_staging(staging, args, plan, run_log)
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
        assert summary["resume_entries_committed_after_interrupt"] == 1
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
        assert "resume_entries_committed_after_interrupt" not in summary
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
            assert summary["resume_staging_recovery_dir"] == staging.dir
            assert "staged CSV schema" in summary["resume_staging_commit_error"]
            output_lock.sweep_owned_scratch_directories(str(tmp_path))
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


class TestScheduleEntries:
    """The work list is reduced by resume first and capped by --max-pdbs after.

    Capping first would hand a resumed run the same already-finished prefix on
    every attempt, filter all of it away, and schedule nothing.
    """

    @staticmethod
    def _args(tmp_path: Path, ids: Sequence[str], **overrides: Any) -> RunConfig:
        id_file = tmp_path / "ids.txt"
        id_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
        fields: dict[str, Any] = {
            "id": None,
            "id_file": str(id_file),
            "pdb_file": None,
            "mtz_file": None,
            "cif_file": None,
            "data_json": None,
            "pdb_redo_root": str(tmp_path),
            "resume": False,
            "retry_partials": False,
            "bonds": True,
            "max_pdbs": None,
        }
        fields.update(overrides)
        return _run_config(**fields)

    @staticmethod
    def _manifest(tmp_path: Path, done_ids: Sequence[str]) -> Path:
        path = tmp_path / "manifest.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for pdb_id in done_ids:
                row = dict.fromkeys(MANIFEST_COLUMNS, "")
                row.update(pdbID=pdb_id, status="ok", n_bonds="0", n_candidates="0")
                writer.writerow(row)
        return path

    def _schedule(self, tmp_path: Path, args: RunConfig) -> tuple[list[str], RunLog]:
        run_log = RunLog(args, "pytest")
        layout = pool.OutputLayout(str(tmp_path))
        ids, _root, _manual = pool.schedule_entries(
            args, layout, str(tmp_path), run_log
        )
        return ids, run_log

    def test_a_capped_resume_reaches_entries_past_the_finished_prefix(
        self, tmp_path: Path
    ) -> None:
        """Two already-done entries must not consume the --max-pdbs budget."""
        all_ids = ["1abc", "2abc", "3abc", "4abc", "5abc"]
        self._manifest(tmp_path, ["1abc", "2abc"])
        (tmp_path / "metal_bonds_all.csv").write_text("", encoding="utf-8")
        (tmp_path / "metal_contact_candidates_all.csv").write_text("", encoding="utf-8")
        args = self._args(tmp_path, all_ids, resume=True, max_pdbs=2)

        ids, run_log = self._schedule(tmp_path, args)
        assert ids == ["3abc", "4abc"]
        assert run_log.details["entries_selected_before_resume"] == 5
        assert run_log.details["entries_scheduled"] == 2

    def test_without_resume_the_cap_takes_the_first_entries(
        self, tmp_path: Path
    ) -> None:
        args = self._args(tmp_path, ["1abc", "2abc", "3abc"], max_pdbs=2)
        ids, _run_log = self._schedule(tmp_path, args)
        assert ids == ["1abc", "2abc"]


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
        return _run_config(**fields)

    def test_the_manifest_row_is_written_after_every_data_row(self) -> None:
        writers = self._RecordingWriters()
        plan = pool.ConfidencePlan()
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
            pool.ConfidencePlan(),
            cast(OutputWriters, self._RecordingWriters()),
            staging,
            ({}, {}),
            resume=True,
            bonds=True,
        )
        assert staging.replacement_ids == {"109m"}


class TestProgressReporter:
    """The heartbeat is throttled differently for a terminal and a log file."""

    class _Stream(io.StringIO):
        def __init__(self, terminal: bool) -> None:
            super().__init__()
            self._terminal = terminal

        # typing.override requires Python 3.12; this project supports 3.11.
        def isatty(self) -> bool:  # type: ignore[explicit-override]
            return self._terminal

    @staticmethod
    def _counts() -> dict[str, int]:
        return {"ok": 1, "partial": 0, "skip": 0, "error": 0}

    def _reporter(
        self, terminal: bool, clock: Callable[[], float]
    ) -> tuple[ProgressReporter, TestProgressReporter._Stream]:
        stream = self._Stream(terminal)
        return ProgressReporter(total=10, stream=stream, clock=clock), stream

    def test_a_redirected_run_renders_far_less_often_than_a_terminal(self) -> None:
        """A run redirected to a file must not grow it by a line per second.

        The counts follow from stepping the clock by ``TERMINAL_INTERVAL_S``:
        a terminal renders every step, a log renders once.
        """
        now = [1000.0]
        for terminal, expected in ((True, 3), (False, 1)):
            reporter, stream = self._reporter(terminal, lambda: now[0])
            for _ in range(3):
                reporter.render(1, self._counts(), 0, 0)
                now[0] += ProgressReporter.TERMINAL_INTERVAL_S
            assert stream.getvalue().count("elapsed=") == expected, (
                f"terminal={terminal} rendered the wrong number of lines"
            )

    def test_force_renders_regardless_of_the_interval(self) -> None:
        reporter, stream = self._reporter(False, lambda: 1000.0)
        reporter.render(1, self._counts(), 0, 0)
        reporter.render(2, self._counts(), 0, 0)
        assert stream.getvalue().count("elapsed=") == 1
        reporter.render(10, self._counts(), 0, 0, force=True, final=True)
        assert stream.getvalue().count("elapsed=") == 2
        assert "[10/10 100.0%]" in stream.getvalue()

    def test_reports_policy_exclusions_beside_metal_free_entries(self) -> None:
        reporter, stream = self._reporter(False, lambda: 1000.0)
        reporter.render(2, self._counts(), 1, 1)

        assert "no_metals=1 metal_site_limit_exceeded=1" in stream.getvalue()


class TestRunLog:
    """The written log is the only record of how a finished run behaved."""

    def test_the_log_goes_to_a_subdirectory_of_the_output_by_default(
        self, tmp_path: Path
    ) -> None:
        """Run diagnostics stay separate from the scientific result CSVs."""
        args = _run_config(output_dir=str(tmp_path), log_dir=None, workers=1)

        path = RunLog(args, "pytest").write(0)

        assert os.path.dirname(path) == str(tmp_path / runlog.DEFAULT_LOG_DIRNAME)
        assert sorted(os.listdir(tmp_path)) == [runlog.DEFAULT_LOG_DIRNAME]
        diagnostics = Path(path.removesuffix(".log") + "_entries.csv")
        assert diagnostics.is_file()

    def test_an_explicit_log_dir_is_used_as_given(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "shared-logs"
        args = _run_config(
            output_dir=str(tmp_path / "out"), log_dir=str(elsewhere), workers=1
        )

        path = RunLog(args, "pytest").write(0)

        assert os.path.dirname(path) == str(elsewhere)
        assert not (tmp_path / "out").exists(), "the output dir is not created here"

    @staticmethod
    def _log(tmp_path: Path, runtimes: Sequence[float]) -> RunLog:
        run_log = RunLog(
            _run_config(output_dir=str(tmp_path), log_dir=None, workers=1),
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
                error="first-sphere reference unavailable for ZN-N",
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

    def test_an_existing_log_is_never_overwritten(self, tmp_path: Path) -> None:
        first = self._log(tmp_path, [1.0]).write(0)
        second = self._log(tmp_path, [2.0]).write(1)
        assert first != second
        assert os.path.isfile(first) and os.path.isfile(second)
        assert Path(first.removesuffix(".log") + "_entries.csv").is_file()
        assert Path(second.removesuffix(".log") + "_entries.csv").is_file()
        assert "Exit code: 0" in open(first).read()
        assert "Exit code: 1" in open(second).read()


class TestOutputDirectoryLock:
    def test_windows_backend_uses_a_nonblocking_lock_outside_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, int, int]] = []

        class FakeMsvcrt:
            LK_UNLCK = 0
            LK_NBLCK = 2

            @staticmethod
            def locking(descriptor: int, mode: int, size: int) -> None:
                position = os.lseek(descriptor, 0, os.SEEK_CUR)
                calls.append((mode, position, size))

        path = tmp_path / "lock"
        path.write_text("metadata", encoding="utf-8")
        monkeypatch.setattr(output_lock, "_IS_WINDOWS", True)
        monkeypatch.setattr(output_lock, "_platform_lock", FakeMsvcrt)

        with open(path, "r+", encoding="utf-8") as handle:
            output_lock.acquire_file_lock(handle)
            assert handle.tell() == 0
            output_lock.release_file_lock(handle)
            assert handle.tell() == 0

        assert calls == [
            (FakeMsvcrt.LK_NBLCK, output_lock.WINDOWS_LOCK_OFFSET, 1),
            (FakeMsvcrt.LK_UNLCK, output_lock.WINDOWS_LOCK_OFFSET, 1),
        ]

    def test_lock_path_symlink_is_refused_without_touching_its_target(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        target = tmp_path / "important.txt"
        target.write_text("important data\n", encoding="utf-8")
        lock_path = output_dir / output_lock.LOCK_FILENAME
        try:
            lock_path.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"this Windows account cannot create symlinks: {exc}")

        with (
            pytest.raises(
                output_lock.OutputDirectoryLockError,
                match="symbolic link|unsafe lock paths are refused",
            ),
            output_lock.OutputDirectoryLock(str(output_dir), "unsafe link"),
        ):
            pass

        assert lock_path.is_symlink()
        assert target.read_text(encoding="utf-8") == "important data\n"

    def test_multiply_linked_lock_is_refused_without_touching_its_target(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        target = tmp_path / "important.txt"
        target.write_text("important data\n", encoding="utf-8")
        lock_path = output_dir / output_lock.LOCK_FILENAME
        os.link(target, lock_path)

        with (
            pytest.raises(
                output_lock.OutputDirectoryLockError, match="multiple hard links"
            ),
            output_lock.OutputDirectoryLock(str(output_dir), "unsafe hard link"),
        ):
            pass

        assert target.read_text(encoding="utf-8") == "important data\n"

    def test_second_owner_fails_with_first_owners_details(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with (
            output_lock.OutputDirectoryLock(str(output_dir), "alchemy first"),
            pytest.raises(output_lock.OutputDirectoryBusyError) as caught,
            output_lock.OutputDirectoryLock(str(output_dir), "alchemy second"),
        ):
            pass

        message = str(caught.value)
        assert str(output_dir) in message
        assert f"pid={os.getpid()}" in message
        assert "command=alchemy first" in message
        assert (output_dir / output_lock.LOCK_FILENAME).is_file()

        # The stable file remains, but the kernel lease is reusable.
        with output_lock.OutputDirectoryLock(str(output_dir), "alchemy third"):
            pass

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
    def test_process_exit_releases_the_lock_without_deleting_its_file(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        ready_read, ready_write = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(ready_read)
            with output_lock.OutputDirectoryLock(str(output_dir), "crashing owner"):
                os.write(ready_write, b"1")
                os._exit(0)
        os.close(ready_write)
        try:
            assert os.read(ready_read, 1) == b"1"
        finally:
            os.close(ready_read)
            os.waitpid(pid, 0)

        with output_lock.OutputDirectoryLock(str(output_dir), "recovery run"):
            pass

    def test_rejected_run_does_not_sweep_or_replace_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            environment, "resolve_ccp4_environment", _resolved_ccp4_environment
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest = output_dir / "manifest.csv"
        manifest.write_bytes(b"existing manifest\n")
        scratch = output_lock.create_owned_scratch_directory(
            str(output_dir), prefix=".alchemy-109m-", kind="entry"
        )
        id_file = tmp_path / "ids.txt"
        id_file.write_text("109m\n", encoding="utf-8")

        with output_lock.OutputDirectoryLock(str(output_dir), "active batch"):
            exit_code = main.main(
                [
                    "--id-file",
                    str(id_file),
                    "--output-dir",
                    str(output_dir),
                    "--pdb-redo-root",
                    str(tmp_path / "absent-mirror"),
                    "--pdb-redo-cache",
                    str(tmp_path / "cache"),
                ]
            )

        assert exit_code == 1
        assert manifest.read_bytes() == b"existing manifest\n"
        assert os.path.isdir(scratch)
        error = capsys.readouterr().err
        assert "already in use by another Alchemy run" in error
        assert "active batch" in error


class TestLeakedWorkDirectorySweep:
    """The startup sweep removes only disposable scratch owned by Alchemy."""

    def test_removes_per_entry_and_staging_directories(self, tmp_path: Path) -> None:
        """Both scratch shapes are swept, with their contents.

        A per-entry directory is otherwise removed only on the normal
        completion path, and holds that entry's maps.
        """
        entry: str | Path = output_lock.create_owned_scratch_directory(
            str(tmp_path), prefix=".alchemy-109m-", kind="entry"
        )
        entry = tmp_path / os.path.basename(entry)
        (entry / "2mFo-DFc.map").write_text("stale", encoding="utf-8")
        staging: str | Path = output_lock.create_owned_scratch_directory(
            str(tmp_path), prefix=".alchemy-resume-", kind="resume"
        )
        staging = tmp_path / os.path.basename(staging)
        (staging / "manifest.csv").write_text("stale", encoding="utf-8")

        removed = output_lock.sweep_owned_scratch_directories(str(tmp_path))

        assert removed == 2
        assert sorted(os.listdir(tmp_path)) == []

    def test_leaves_real_output_alone(self, tmp_path: Path) -> None:
        """Names alone never authorize deleting output-directory contents.

        The unmarked directory has the exact shape of legacy entry scratch;
        it still belongs to the user as far as the conservative sweep knows.
        """
        (tmp_path / "manifest.csv").write_text("keep", encoding="utf-8")
        (tmp_path / "alchemy_run_20260101.log").write_text("keep", encoding="utf-8")
        (tmp_path / "109m").mkdir()  # a user directory named for an id
        (tmp_path / ".alchemyrc").write_text("keep", encoding="utf-8")
        (tmp_path / ".alchemy-109m-unmarked").mkdir()

        assert output_lock.sweep_owned_scratch_directories(str(tmp_path)) == 0
        assert sorted(os.listdir(tmp_path)) == [
            ".alchemy-109m-unmarked",
            ".alchemyrc",
            "109m",
            "alchemy_run_20260101.log",
            "manifest.csv",
        ]

    def test_preserved_scratch_is_not_swept(self, tmp_path: Path) -> None:
        kept = output_lock.create_owned_scratch_directory(
            str(tmp_path),
            prefix=".alchemy-109m-",
            kind="entry",
            preserve=True,
        )

        assert output_lock.sweep_owned_scratch_directories(str(tmp_path)) == 0
        assert os.path.isdir(kept)

    def test_symlink_is_not_followed_even_if_its_target_is_marked(
        self, tmp_path: Path
    ) -> None:
        target_root = tmp_path / "elsewhere"
        target_root.mkdir()
        target = output_lock.create_owned_scratch_directory(
            str(target_root), prefix=".alchemy-109m-", kind="entry"
        )
        link = tmp_path / ".alchemy-109m-link"
        link.symlink_to(target, target_is_directory=True)

        assert output_lock.sweep_owned_scratch_directories(str(tmp_path)) == 0
        assert link.is_symlink()
        assert os.path.isdir(target)

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert (
            output_lock.sweep_owned_scratch_directories(str(tmp_path / "absent")) == 0
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions required")
def test_unwritable_output_dir_exits_cleanly_naming_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A read-only destination exits like every other unusable input.

    CCP4 resolution is stubbed because it runs before the directory is created,
    and would otherwise fail on a machine with no CCP4 installed.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")

    monkeypatch.setattr(
        environment, "resolve_ccp4_environment", _resolved_ccp4_environment
    )
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    output_dir = parent / "output"
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    try:
        exit_code = main.main(
            [
                "--id-file",
                str(id_file),
                "--output-dir",
                str(output_dir),
                "--pdb-redo-root",
                str(tmp_path / "absent-mirror"),
                "--pdb-redo-cache",
                str(tmp_path / "cache"),
            ]
        )
    finally:
        parent.chmod(0o700)

    # 1 is the driver's code for a fixable failure; argparse owns 2.
    assert exit_code == 1
    message = capsys.readouterr().err
    assert str(output_dir) in message, message
    assert "Traceback" not in message


@pytest.mark.skipif(os.name != "posix", reason="uses a POSIX-only stub env")
def test_a_run_sweeps_leaked_scratch_before_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep is wired into the driver, not merely available to it.

    Driven through ``main.main`` so that deleting the call site fails the test.
    The run itself fails for want of a mirror: sweeping happens at startup, so
    even a failing run must leave the directory clean.
    """
    monkeypatch.setattr(
        environment, "resolve_ccp4_environment", _resolved_ccp4_environment
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    leaked: str | Path = output_lock.create_owned_scratch_directory(
        str(output_dir), prefix=".alchemy-109m-", kind="entry"
    )
    leaked = output_dir / os.path.basename(leaked)
    (leaked / "2mFo-DFc.map").write_text("stale map bytes", encoding="utf-8")
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    main.main(
        [
            "--id-file",
            str(id_file),
            "--output-dir",
            str(output_dir),
            "--pdb-redo-root",
            str(tmp_path / "absent-mirror"),
            "--pdb-redo-cache",
            str(tmp_path / "cache"),
        ]
    )

    leftovers = sorted(
        name for name in os.listdir(output_dir) if name.startswith(".alchemy-")
    )
    assert leftovers == [], f"the run left scratch behind: {leftovers}"


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
        return density.DensityResult(**fields)

    def _entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: density.DensityResult,
        **cfg_overrides: Any,
    ) -> worker_contracts.EntryResult:
        monkeypatch.setattr(worker, "extract_metal_statistics", _empty_metal_statistics)

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

        assert result.density_map_scope_used == "full-extent-fallback"
        assert result.density_full_map_bytes == 8192
        assert result.density_edstats_map_bytes == 2048

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
        """The scratch directory holds the maps, so only the flag may keep it."""
        self._entry(tmp_path, monkeypatch, self.density_result(), keep=keep)

        scratch = [
            name for name in os.listdir(tmp_path) if name.startswith(".alchemy-")
        ]

        assert bool(scratch) is keep
        if keep:
            marker = tmp_path / scratch[0] / output_lock.SCRATCH_MARKER_FILENAME
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
    monkeypatch.setattr(worker, "extract_metal_statistics", _empty_metal_statistics)

    def density_stage(*_args: Any, **_kwargs: Any) -> density.DensityResult:
        return TestDensityResultReachesTheResult.density_result()

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        density_stage,
    )

    assert result.input_model_count == 1
    assert result.model_analyzed == 1
    assert result.multi_model_structure is False

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
    ) -> worker_contracts.EntryResult:
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
        assert "edstats" in result.error
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

        kept = result.ccp4_timeout_log_path
        assert kept, "the timeout log path must be recorded on the result"
        assert os.path.isfile(kept), (
            f"the retained log is missing at {kept}; the path named in the "
            "timeout message must outlive the scratch directory"
        )
        assert worker.TIMEOUT_LOG_DIRNAME in kept
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

        kept_dir = os.path.dirname(result.ccp4_timeout_log_path)
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

        assert result.ccp4_timeout_log_path == ""
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
