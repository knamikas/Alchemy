"""Direct tests for the driver surfaces a batch run depends on.

Scope: the functions that decide *what a run does to existing data* and *which
entries it selects* -- resume-schema validation, the exit-code contract, ID
parsing, mirror enumeration, input preparation, worker autoscaling, and the
driver's requirement that CCP4 be complete before a run starts. Reached through
``main.main()`` they would run only in the ``ccp4``+``slow`` lane, which CI does
not run.

Out of scope here (owned elsewhere): argument *parsing* rules
(``test_cli_and_config``), manifest row content (``test_driver_manifest``), and
the pipeline itself (``test_pipeline_integration``).
"""

from __future__ import annotations

import argparse
import csv
import http.client
import gzip
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast
from collections.abc import Sequence

import pytest

from types import SimpleNamespace

from coordination import schema as coordination_schema
import ccp4_setup
import cli
import inputs
from driver import resources
from driver import writers
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS
from driver import resume
from driver import pool
from driver import runlog
import confidence_score

if TYPE_CHECKING:
    # Annotations only, so gemmi and numpy stay imported inside the handful of
    # helpers that build MTZ files, and ``worker`` is never imported at all.
    import numpy as np
    from numpy.typing import NDArray
    from worker_contracts import EntryResult


class _ApproxFactory(Protocol):
    """The concrete numeric subset of pytest's broadly typed approx helper."""

    def __call__(
        self,
        expected: object,
        rel: float | None = None,
        abs: float | None = None,
        nan_ok: bool = False,
    ) -> object: ...


class _PytestApi(Protocol):
    approx: _ApproxFactory


approx = cast(_PytestApi, pytest).approx


class _ResumeOutputs(TypedDict):
    """The four output paths ``resume_outputs`` hands to the resume validator."""

    manifest_path: str
    stats_path: str
    bonds_path: str
    candidates_path: str


def _write_header(path: Path, columns: Sequence[str]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(columns)
    return str(path)


def _append_csv_row(path: str, columns: Sequence[str], **values: str | int) -> None:
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writerow(
            {column: values.get(column, "") for column in columns}
        )


def _append_terminal_manifest(
    outputs: _ResumeOutputs,
    *,
    pdb_id: str = "1abc",
    status: str = "ok",
    retryable: str = "False",
    n_metals: str | int = 0,
    n_bonds: str | int = 0,
    n_candidates: str | int = 0,
    reason_codes: str = "",
    no_metals: str = "false",
    metal_site_limit_exceeded: str | None = None,
) -> None:
    _append_csv_row(
        outputs["manifest_path"],
        MANIFEST_COLUMNS,
        pdbID=pdb_id,
        status=status,
        retryable=retryable,
        n_metals=n_metals,
        n_bonds=n_bonds,
        n_candidates=n_candidates,
        reason_codes=reason_codes,
        no_metals=no_metals,
        metal_site_limit_exceeded=(
            metal_site_limit_exceeded
            if metal_site_limit_exceeded is not None
            else "true"
            if reason_codes == "metal_site_limit_exceeded"
            else "false"
        ),
    )


def _append_selected_stats(
    outputs: _ResumeOutputs, pdb_id: str = "1abc", atom_index: str = "0"
) -> None:
    _append_csv_row(
        outputs["stats_path"],
        STATS_COLUMNS,
        pdbID=pdb_id,
        category="metal",
        selected_metal_site_status="selected",
        metal_model_index="0",
        metal_chain_index="0",
        metal_residue_index="0",
        metal_atom_index=atom_index,
    )


@pytest.fixture
def resume_outputs(tmp_path: Path) -> _ResumeOutputs:
    """A set of output files whose headers all match the current schema."""
    return {
        "manifest_path": _write_header(tmp_path / "manifest.csv", MANIFEST_COLUMNS),
        "stats_path": _write_header(tmp_path / "stats.csv", STATS_COLUMNS),
        "bonds_path": _write_header(
            tmp_path / "bonds.csv", coordination_schema.BOND_COLUMNS
        ),
        "candidates_path": _write_header(
            tmp_path / "candidates.csv", coordination_schema.CANDIDATE_COLUMNS
        ),
    }


def test_matching_headers_are_accepted(resume_outputs: _ResumeOutputs) -> None:
    resume.validate_resume_schemas(**resume_outputs)


def test_a_policy_excluded_entry_requires_no_site_rows(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(
        resume_outputs,
        n_metals=101,
        reason_codes="metal_site_limit_exceeded",
    )

    resume.validate_resume_schemas(**resume_outputs)


def test_a_policy_excluded_entry_rejects_stray_site_rows(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(
        resume_outputs,
        n_metals=101,
        reason_codes="metal_site_limit_exceeded",
    )
    _append_selected_stats(resume_outputs)

    with pytest.raises(ValueError, match="policy-excluded rows"):
        resume.validate_resume_schemas(**resume_outputs)


@pytest.mark.parametrize(
    ("reason_codes", "flag"),
    [("metal_site_limit_exceeded", "false"), ("", "true")],
)
def test_policy_exclusion_flag_and_reason_must_agree(
    resume_outputs: _ResumeOutputs, reason_codes: str, flag: str
) -> None:
    _append_terminal_manifest(
        resume_outputs,
        n_metals=101,
        reason_codes=reason_codes,
        metal_site_limit_exceeded=flag,
    )

    with pytest.raises(ValueError, match="inconsistent metal-site exclusion"):
        resume.validate_resume_schemas(**resume_outputs)


def test_metal_free_flag_requires_a_successful_zero_site_result(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs, n_metals=1, no_metals="true")
    _append_selected_stats(resume_outputs)

    with pytest.raises(ValueError, match="invalid no_metals fields"):
        resume.validate_resume_schemas(**resume_outputs)


def test_absent_outputs_are_accepted(tmp_path: Path) -> None:
    """A first run has nothing to be incompatible with."""
    resume.validate_resume_schemas(
        manifest_path=str(tmp_path / "manifest.csv"),
        stats_path=str(tmp_path / "stats.csv"),
        bonds_path=str(tmp_path / "bonds.csv"),
        candidates_path=str(tmp_path / "candidates.csv"),
    )


@pytest.mark.parametrize(
    "target", ["manifest_path", "stats_path", "bonds_path", "candidates_path"]
)
def test_an_incompatible_header_is_refused(
    resume_outputs: _ResumeOutputs,
    tmp_path: Path,
    target: Literal["manifest_path", "stats_path", "bonds_path", "candidates_path"],
) -> None:
    """Appending beneath a foreign header would misalign every column, and
    nothing downstream could tell that column N had changed meaning."""
    _write_header(tmp_path / os.path.basename(resume_outputs[target]), ["unexpected"])

    with pytest.raises(ValueError, match="incompatible schema"):
        resume.validate_resume_schemas(**resume_outputs)


def test_a_manifest_without_the_reference_data_column_is_refused(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    """A manifest without the identity column cannot be resumed into.

    Its rows do not say which reference data produced them, so appending rows
    that do would make one file two datasets with no way to tell them apart.
    """
    older = [
        column for column in writers.MANIFEST_COLUMNS if column != "reference_data_id"
    ]
    _write_header(tmp_path / "manifest.csv", older)

    with pytest.raises(ValueError, match="missing reference_data_id"):
        resume.validate_resume_schemas(**resume_outputs)


def test_the_refusal_names_the_columns_that_differ(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    """The message names the columns that differ, leaving the operator with a
    cause rather than two headers to diff by hand."""
    _write_header(tmp_path / "manifest.csv", list(writers.MANIFEST_COLUMNS) + ["stray"])

    with pytest.raises(ValueError, match="unexpected stray"):
        resume.validate_resume_schemas(**resume_outputs)


def test_a_truncated_stats_header_is_refused(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    """A different EDSTATS build shifts the density block, and a dropped metric
    column misaligns every value after it with no other symptom."""
    _write_header(tmp_path / "stats.csv", list(STATS_COLUMNS)[:-1])

    with pytest.raises(ValueError, match="incompatible schema"):
        resume.validate_resume_schemas(**resume_outputs)


def test_bond_headers_are_ignored_when_the_bond_stage_is_disabled(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    """``--no-bonds`` writes no bond rows, so their schema cannot conflict."""
    _write_header(tmp_path / "bonds.csv", ["stale"])
    _write_header(tmp_path / "candidates.csv", ["stale"])

    resume.validate_resume_schemas(**resume_outputs, bonds_enabled=False)


def test_confidence_output_requires_its_columns(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    """A confidence path without its schema cannot be validated at all."""
    with pytest.raises(ValueError, match="confidence columns are required"):
        resume.validate_resume_schemas(
            **resume_outputs,
            confidence_path=str(tmp_path / "confidence.csv"),
            confidence_columns=None,
        )


def test_an_incompatible_confidence_header_is_refused(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    columns = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
    path = _write_header(tmp_path / "confidence.csv", columns[:-1])

    with pytest.raises(ValueError, match="incompatible schema"):
        resume.validate_resume_schemas(
            **resume_outputs, confidence_path=path, confidence_columns=columns
        )


@pytest.mark.parametrize("target", ["stats_path", "bonds_path", "candidates_path"])
def test_terminal_manifest_requires_every_enabled_output(
    resume_outputs: _ResumeOutputs,
    target: Literal["stats_path", "bonds_path", "candidates_path"],
) -> None:
    _append_terminal_manifest(resume_outputs)
    os.unlink(resume_outputs[target])

    with pytest.raises(ValueError, match="missing or empty"):
        resume.validate_resume_schemas(**resume_outputs)


def test_terminal_manifest_does_not_require_disabled_bond_outputs(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs, n_bonds="", n_candidates="")
    os.unlink(resume_outputs["bonds_path"])
    os.unlink(resume_outputs["candidates_path"])

    resume.validate_resume_schemas(**resume_outputs, bonds_enabled=False)


def test_a_consistent_terminal_artifact_set_is_accepted(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs, n_metals=1)
    _append_selected_stats(resume_outputs)

    resume.validate_resume_schemas(**resume_outputs)


def test_ok_manifest_metal_count_must_match_selected_stats(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs, n_metals=1)

    with pytest.raises(ValueError, match=r"n_metals=1.*has 0 selected row"):
        resume.validate_resume_schemas(**resume_outputs)


@pytest.mark.parametrize("n_metals", ["", "not-a-count", -1])
def test_terminal_manifest_requires_a_valid_metal_count(
    resume_outputs: _ResumeOutputs, n_metals: str | int
) -> None:
    _append_terminal_manifest(resume_outputs, n_metals=n_metals)

    with pytest.raises(ValueError, match="invalid n_metals"):
        resume.validate_resume_schemas(**resume_outputs)


def test_duplicate_selected_site_keys_are_refused(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs, n_metals=2)
    _append_selected_stats(resume_outputs)
    _append_selected_stats(resume_outputs)

    with pytest.raises(ValueError, match="duplicate selected-site key"):
        resume.validate_resume_schemas(**resume_outputs)


def test_terminal_partial_may_have_fewer_stats_than_detected_metals(
    resume_outputs: _ResumeOutputs,
) -> None:
    """A density-stage failure knows the metal count but writes no stats rows."""
    _append_terminal_manifest(
        resume_outputs,
        status="partial",
        n_metals=1,
        n_bonds="",
        n_candidates="",
    )

    resume.validate_resume_schemas(**resume_outputs)


@pytest.mark.parametrize(
    ("manifest_count", "output_key"),
    [("n_bonds", "bonds_path"), ("n_candidates", "candidates_path")],
)
def test_manifest_bond_stage_counts_must_match_rows(
    resume_outputs: _ResumeOutputs,
    manifest_count: str,
    output_key: Literal["bonds_path", "candidates_path"],
) -> None:
    if manifest_count == "n_bonds":
        _append_terminal_manifest(resume_outputs, n_bonds=1)
    else:
        _append_terminal_manifest(resume_outputs, n_candidates=1)

    with pytest.raises(
        ValueError,
        match=rf"{manifest_count}=1.*{os.path.basename(resume_outputs[output_key])}",
    ):
        resume.validate_resume_schemas(**resume_outputs)


def test_confidence_count_must_match_every_terminal_metal(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    columns = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
    path = _write_header(tmp_path / "confidence.csv", columns)
    _append_terminal_manifest(resume_outputs, n_metals=1)
    _append_selected_stats(resume_outputs)

    with pytest.raises(ValueError, match=r"n_metals=1.*confidence.csv has 0 row"):
        resume.validate_resume_schemas(
            **resume_outputs, confidence_path=path, confidence_columns=columns
        )

    _append_csv_row(path, columns, pdbID="1abc")
    resume.validate_resume_schemas(
        **resume_outputs, confidence_path=path, confidence_columns=columns
    )


def test_duplicate_complete_manifest_ids_are_refused(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs)
    _append_terminal_manifest(resume_outputs)

    with pytest.raises(ValueError, match=r"duplicate rows for 1abc"):
        resume.validate_resume_schemas(**resume_outputs)


def test_orphan_rows_from_a_pre_manifest_crash_remain_recoverable(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs)
    _append_selected_stats(resume_outputs, pdb_id="2def")
    _append_csv_row(
        resume_outputs["bonds_path"],
        coordination_schema.BOND_COLUMNS,
        pdbID="2def",
    )
    _append_csv_row(
        resume_outputs["candidates_path"],
        coordination_schema.CANDIDATE_COLUMNS,
        pdbID="2def",
    )

    resume.validate_resume_schemas(**resume_outputs)


@pytest.mark.parametrize(
    ("counts", "retryable_partials", "expected"),
    [
        ({"ok": 5, "partial": 0, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 4, "partial": 1, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 0, "partial": 0, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 4, "partial": 0, "skip": 0, "error": 1}, 0, 1),
        ({"ok": 4, "partial": 0, "skip": 1, "error": 0}, 0, 1),
        ({"ok": 4, "partial": 1, "skip": 0, "error": 0}, 1, 1),
    ],
)
def test_the_exit_code_reports_operational_incompleteness(
    counts: dict[str, int], retryable_partials: int, expected: int
) -> None:
    """Nonzero exactly when something remains to be done.

    A *terminal* partial is usable but incomplete science that no rerun can
    improve, so it exits zero; a *retryable* one is work still outstanding.
    """
    assert pool.batch_exit_code(counts, retryable_partials) == expected


def test_a_missing_status_key_is_treated_as_zero() -> None:
    """Counts are accumulated per status, so an absent key means none seen."""
    assert pool.batch_exit_code({}, 0) == 0
    assert pool.batch_exit_code({"error": 2}, 0) == 1


@pytest.mark.parametrize("value", ["9myr", "9MYR", "1abc", "0000"])
def test_pdb_ids_are_accepted_case_insensitively_and_normalized(value: str) -> None:
    """IDs are lowercased so cache paths and manifest keys cannot diverge."""
    assert cli.parse_pdb_id(value) == value.lower()


@pytest.mark.parametrize("value", ["abc", "abcde", "ab-c", "ab c", "", "9my_"])
def test_malformed_pdb_ids_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="four alphanumeric"):
        cli.parse_pdb_id(value)


def test_an_id_file_accepts_mixed_separators_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text(
        "9myr, 6NLR\n# a comment line\n\n9nxl 1abc   # trailing comment\n",
        encoding="utf-8",
    )

    assert pool.load_ids_from_file(str(path)) == ["9myr", "6nlr", "9nxl", "1abc"]


def test_an_id_file_reports_the_line_of_a_bad_id(tmp_path: Path) -> None:
    """A typo in a long ID list must name where it is, not just that it exists."""
    path = tmp_path / "ids.txt"
    path.write_text("9myr\n6nlr\nnot-an-id\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid PDB id .*ids\.txt:3"):
        pool.load_ids_from_file(str(path))


def test_a_missing_id_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="id file not found"):
        pool.load_ids_from_file(str(tmp_path / "absent.txt"))


def _make_entry(
    root: Path,
    pdb_id: str,
    *,
    mtz: bool = True,
    cif: bool = True,
    pdb: bool = False,
    compressed: bool = False,
    data_json: str | None = None,
) -> str:
    """Create a mirror-layout entry directory with the requested files.

    ``cif`` and ``pdb`` are independent so a mirror carrying both can be built.
    """
    entry_dir = inputs.entry_dir_for(str(root), pdb_id)
    os.makedirs(entry_dir, exist_ok=True)

    def write(name: str, payload: bytes = b"x") -> None:
        path = os.path.join(entry_dir, name)
        if compressed:
            path += ".gz"
            payload = gzip.compress(payload)
        with open(path, "wb") as handle:
            handle.write(payload)

    if mtz:
        write(f"{pdb_id}_final.mtz")
    if cif:
        write(f"{pdb_id}_final.cif")
    if pdb:
        write(f"{pdb_id}_final.pdb", b"END\n")
    if data_json is not None:
        with open(os.path.join(entry_dir, "data.json"), "w", encoding="utf-8") as fh:
            fh.write(data_json)
    return entry_dir


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_an_entry_is_final_only_with_both_inputs(
    tmp_path: Path, compressed: bool
) -> None:
    """Coordinates without map coefficients cannot be analyzed, and vice versa."""
    complete = _make_entry(tmp_path, "9myr", compressed=compressed)
    no_mtz = _make_entry(tmp_path, "6nlr", mtz=False, compressed=compressed)
    no_coords = _make_entry(
        tmp_path, "9nxl", cif=False, pdb=False, compressed=compressed
    )

    assert inputs.has_final_files(complete, "9myr")
    assert not inputs.has_final_files(no_mtz, "6nlr")
    assert not inputs.has_final_files(no_coords, "9nxl")


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_a_legacy_pdb_export_counts_as_usable_coordinates(
    tmp_path: Path, compressed: bool
) -> None:
    """``.pdb`` and ``.pdb.gz`` are accepted when the authoritative mmCIF is
    absent, so a mirror carrying only the legacy export is still analyzable."""
    entry_dir = _make_entry(
        tmp_path, "9myr", cif=False, pdb=True, compressed=compressed
    )
    assert inputs.has_final_files(entry_dir, "9myr")


def test_enumeration_returns_only_complete_entries(tmp_path: Path) -> None:
    """Incomplete entries are skipped, and the order follows the mirror layout.

    Ordering is by hash directory then entry, not by PDB ID: ``9myr`` lives
    under ``my`` and ``6nlr`` under ``nl``, so ``9myr`` comes first despite
    sorting later as a string.
    """
    _make_entry(tmp_path, "9myr")  # hashdir "my"
    _make_entry(tmp_path, "6nlr")  # hashdir "nl"
    _make_entry(tmp_path, "9nxl", mtz=False)  # hashdir "nx", incomplete

    assert inputs.enumerate_entries(str(tmp_path)) == ["9myr", "6nlr"]


def test_enumeration_stops_at_the_requested_limit(tmp_path: Path) -> None:
    """``--max-pdbs`` must not walk the whole mirror to return three ids."""
    for pdb_id in ("1aaa", "1bbb", "2ccc", "2ddd"):
        _make_entry(tmp_path, pdb_id)

    limited = inputs.enumerate_entries(str(tmp_path), limit=2)
    assert len(limited) == 2
    assert limited == inputs.enumerate_entries(str(tmp_path))[:2]


def test_an_unreadable_hashdir_is_skipped_rather_than_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partially-synced mirror must not abort the whole enumeration."""
    _make_entry(tmp_path, "9myr")
    _make_entry(tmp_path, "6nlr")
    real_listdir = os.listdir

    def failing_listdir(path: str) -> list[str]:
        if str(path).endswith(os.path.join(str(tmp_path), "nl")):
            raise PermissionError("locked down")
        return real_listdir(path)

    monkeypatch.setattr("inputs.os.listdir", failing_listdir)
    assert inputs.enumerate_entries(str(tmp_path)) == ["9myr"]


def test_missing_map_coefficients_are_reported_by_path(tmp_path: Path) -> None:
    """The error must name the file that was looked for, not just fail."""
    entry_dir = _make_entry(tmp_path, "9myr", mtz=False)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="9myr_final.mtz"):
        inputs.prepare_inputs("9myr", entry_dir, str(work_dir))


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_a_legacy_pdb_export_is_used_when_no_mmcif_exists(
    tmp_path: Path, compressed: bool
) -> None:
    """The fallback path returns the PDB directly, decompressing if needed."""
    entry_dir = _make_entry(
        tmp_path, "9myr", cif=False, pdb=True, compressed=compressed
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    _, pdb = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))

    assert pdb.endswith(".pdb"), "a gzipped export must be decompressed first"
    assert os.path.isfile(pdb)
    with open(pdb, encoding="ascii") as handle:
        assert handle.read() == "END\n"
    if compressed:
        assert os.path.dirname(pdb) == str(work_dir), (
            "the mirror must not be written to"
        )


def test_the_authoritative_mmcif_wins_over_the_legacy_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a mirror carries both, the mmCIF is the one converted: the legacy
    export loses identifiers it retains."""
    entry_dir = _make_entry(tmp_path, "9myr", cif=True, pdb=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    converted: list[str] = []

    def fake_cif_to_pdb(cif_path: str, destination: str) -> str:
        converted.append(cif_path)
        with open(destination, "w", encoding="ascii") as handle:
            handle.write("FROM CIF\n")
        return destination

    monkeypatch.setattr(inputs, "cif_to_pdb", fake_cif_to_pdb)
    _mtz, pdb = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))

    assert converted and converted[0].endswith("_final.cif")
    with open(pdb, encoding="ascii") as handle:
        assert handle.read() == "FROM CIF\n", "the legacy export was used instead"


def test_an_entry_with_neither_coordinate_format_names_both(tmp_path: Path) -> None:
    entry_dir = _make_entry(tmp_path, "9myr", cif=False, pdb=False)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"9myr_final\.cif or 9myr_final\.pdb"):
        inputs.prepare_inputs("9myr", entry_dir, str(work_dir))


def test_a_compressed_mirror_is_decompressed_into_the_work_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compressed mirrors are accepted, and never modified in place."""
    entry_dir = _make_entry(tmp_path, "9myr", compressed=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    converted: list[tuple[str, str]] = []

    def fake_cif_to_pdb(cif_path: str, destination: str) -> str:
        converted.append((cif_path, destination))
        with open(destination, "w", encoding="ascii") as handle:
            handle.write("END\n")
        return destination

    monkeypatch.setattr(inputs, "cif_to_pdb", fake_cif_to_pdb)
    mtz, pdb = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))

    assert os.path.dirname(mtz) == str(work_dir), "the mirror must not be written to"
    assert not mtz.endswith(".gz")
    assert os.path.isfile(pdb)
    assert converted, "the authoritative mmCIF should have been converted"


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
    assert resume.remove_stale_disabled_bond_outputs(paths, False, False) == []


def test_worker_limits_leave_headroom_and_respect_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both limits are floored at one, so a small machine still runs.

    The CPU limit leaves two cores for the driver and the OS; the memory limit
    exists because each worker holds whole maps in memory, and oversubscribing
    it invites the OOM killer.
    """
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 16)
    monkeypatch.setattr(
        resources,
        "available_memory_bytes",
        lambda: 8 * resources.AUTO_WORKER_MEMORY_BYTES,
    )
    assert resources.automatic_worker_limits() == (14, 6)

    monkeypatch.setattr(resources, "available_cpu_count", lambda: 1)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 0)
    assert resources.automatic_worker_limits() == (1, 1)


def test_unknown_memory_leaves_the_limit_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable ``/proc/meminfo`` must not silently cap parallelism."""
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 8)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: None)

    cpu_limit, memory_limit = resources.automatic_worker_limits()
    assert cpu_limit == 6
    assert memory_limit is None


def test_explicit_workers_are_still_capped_for_process_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(["--workers", "50", "--output-dir", str(tmp_path)])
    run_log = runlog.RunLog(args, "pytest")
    monkeypatch.setattr(pool, "automatic_worker_limits", lambda: (8, 3))

    workers = pool.choose_worker_count(args, entry_count=20, run_log=run_log)

    assert workers == 3
    assert run_log.details["Requested workers"] == 50
    assert run_log.details["Selected workers"] == 3


class _FailingResponse:
    """A response that opens successfully and then fails partway through."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self._served = False

    def getcode(self) -> int:
        return 200

    def read(self, _size: int) -> bytes:
        if self._served:
            raise self._error
        self._served = True
        return b"partial body"

    def __enter__(self) -> _FailingResponse:
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        return False


class _LengthResponse:
    """A cleanly ending response with an independently declared byte count."""

    def __init__(self, body: bytes, content_length: int) -> None:
        self._body = body
        self._content_length = content_length
        self._served = False

    def getcode(self) -> int:
        return 200

    def getheader(self, name: str) -> int | None:
        return self._content_length if name.lower() == "content-length" else None

    def read(self, _size: int) -> bytes:
        if self._served:
            return b""
        self._served = True
        return self._body

    def __enter__(self) -> _LengthResponse:
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        return False


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("connection reset by peer"),
        TimeoutError("read timed out"),
        http.client.IncompleteRead(b"partial body", 4096),
    ],
    ids=["reset", "timeout", "incomplete-read"],
)
def test_a_transfer_that_fails_midway_reports_no_usable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Only the opening request was guarded, so a mid-stream failure escaped.

    ``IncompleteRead`` is not even an ``OSError``, so a handler written for one
    would still have missed it. The caller's position is the same as a 404 --
    no file -- and the partial download must not be left behind.
    """

    def failing_response(_url: str, timeout: float | None = None) -> _FailingResponse:
        del timeout
        return _FailingResponse(error)

    monkeypatch.setattr(inputs, "urlopen", failing_response)
    destination = tmp_path / "9myr_final.mtz"

    with pytest.raises(FileNotFoundError) as excinfo:
        inputs.download_stream(
            "https://example.invalid/9myr_final.mtz", str(destination)
        )

    assert type(error).__name__ in str(excinfo.value)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == [], "a partial download was left behind"


def test_a_clean_early_eof_is_not_promoted_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTPResponse can return EOF without raising for a short response body."""
    body = b"nonempty but truncated"

    def short_response(_url: str, timeout: float | None = None) -> _LengthResponse:
        del timeout
        return _LengthResponse(body, len(body) + 4096)

    monkeypatch.setattr(inputs, "urlopen", short_response)
    destination = tmp_path / "9myr_final.mtz"

    with pytest.raises(FileNotFoundError, match=r"expected .* received"):
        inputs.download_stream(
            "https://example.invalid/9myr_final.mtz", str(destination)
        )

    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_a_body_matching_content_length_is_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"complete response"

    def complete_response(_url: str, timeout: float | None = None) -> _LengthResponse:
        del timeout
        return _LengthResponse(body, len(body))

    monkeypatch.setattr(inputs, "urlopen", complete_response)
    destination = tmp_path / "data.json"

    assert inputs.download_stream(
        "https://example.invalid/data.json", str(destination)
    ) == str(destination)
    assert destination.read_bytes() == body


def test_an_unwritable_cache_is_reported_as_a_driver_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local failure is not a missing entry, and is not a traceback either.

    ``ensure_entry_available`` creates directories and writes files, so an
    unwritable ``--pdb-redo-cache`` raised ``PermissionError`` straight out of
    the driver's pre-flight, past a handler that caught only
    ``FileNotFoundError``.
    """

    def unwritable(pdb_id: str, mirror_root: str, cache_root: str) -> str:
        raise PermissionError(13, "Permission denied", str(cache_root))

    monkeypatch.setattr(pool, "ensure_entry_available", unwritable)
    args = cli.parse_args(["--id", "9myr", "--pdb-redo-root", str(tmp_path / "mirror")])

    with pytest.raises(pool.DriverError) as excinfo:
        pool.select_entry_ids(args, str(tmp_path / "cache"))

    message = str(excinfo.value)
    assert "PermissionError" in message
    assert "not found" not in message, (
        "a local failure must not read as a missing entry"
    )


def _host_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_bytes: int
) -> Path:
    """Point the memory probe at a synthetic ``/proc/meminfo``."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {host_bytes // 1024} kB\n", encoding="ascii")
    monkeypatch.setattr(resources, "PROC_MEMINFO_PATH", str(meminfo))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(tmp_path / "no-cgroup"))
    monkeypatch.setattr("driver.resources.sys.platform", "linux")
    return meminfo


def test_available_memory_is_read_from_meminfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker sizing follows what the machine reports as free."""
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=7 * budget)

    assert resources.available_memory_bytes() == 7 * budget
    assert resources.automatic_worker_limits()[1] == 5


def test_an_unreadable_meminfo_leaves_worker_sizing_to_the_cpu_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memory probe that fails must not be read as zero available memory."""
    monkeypatch.setattr(
        resources, "PROC_MEMINFO_PATH", str(tmp_path / "definitely-absent")
    )
    monkeypatch.setattr("driver.resources.sys.platform", "linux")

    def unavailable_sysconf(_name: str) -> int:
        return 0

    monkeypatch.setattr(
        "driver.resources.os.sysconf", unavailable_sysconf, raising=False
    )

    assert resources.available_memory_bytes() is None
    assert resources.automatic_worker_limits()[1] is None


def test_a_cgroup_v2_memory_limit_caps_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containers must not schedule against host RAM they cannot access."""
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=64 * budget)
    # A limit far below host memory, of the kind SLURM or a container imposes.
    cgroup_root = tmp_path / "cgroup"
    (cgroup_root / "batch").mkdir(parents=True)
    (cgroup_root / "batch" / "memory.max").write_text(str(4 * budget), encoding="ascii")
    (cgroup_root / "batch" / "memory.current").write_text(str(budget), encoding="ascii")
    cgroup = tmp_path / "self-cgroup"
    cgroup.write_text("0::/batch\n", encoding="ascii")
    monkeypatch.setattr(resources, "CGROUP_ROOT", str(cgroup_root))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(cgroup))

    assert resources.available_memory_bytes() == 3 * budget


def test_unlimited_cgroup_leaves_host_memory_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=8 * budget)
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "memory.max").write_text("max", encoding="ascii")
    (cgroup_root / "memory.current").write_text("123", encoding="ascii")
    cgroup = tmp_path / "self-cgroup"
    cgroup.write_text("0::/\n", encoding="ascii")
    monkeypatch.setattr(resources, "CGROUP_ROOT", str(cgroup_root))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(cgroup))

    assert resources.available_memory_bytes() == 8 * budget


def test_parent_cgroup_limit_applies_when_child_is_unlimited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=32 * budget)
    cgroup_root = tmp_path / "cgroup"
    child = cgroup_root / "batch" / "task"
    child.mkdir(parents=True)
    (cgroup_root / "memory.max").write_text("max", encoding="ascii")
    (cgroup_root / "memory.current").write_text("0", encoding="ascii")
    (cgroup_root / "batch" / "memory.max").write_text(str(6 * budget), encoding="ascii")
    (cgroup_root / "batch" / "memory.current").write_text(
        str(2 * budget), encoding="ascii"
    )
    (child / "memory.max").write_text("max", encoding="ascii")
    (child / "memory.current").write_text(str(budget), encoding="ascii")
    cgroup = tmp_path / "self-cgroup"
    cgroup.write_text("0::/batch/task\n", encoding="ascii")
    monkeypatch.setattr(resources, "CGROUP_ROOT", str(cgroup_root))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(cgroup))

    assert resources.available_memory_bytes() == 4 * budget


def test_a_cgroup_v1_memory_limit_caps_host_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=32 * budget)
    cgroup_root = tmp_path / "cgroup"
    group = cgroup_root / "memory" / "batch"
    group.mkdir(parents=True)
    (group / "memory.limit_in_bytes").write_text(str(5 * budget), encoding="ascii")
    (group / "memory.usage_in_bytes").write_text(str(budget), encoding="ascii")
    cgroup = tmp_path / "self-cgroup"
    cgroup.write_text("7:cpu,cpuacct:/batch\n8:memory:/batch\n", encoding="ascii")
    monkeypatch.setattr(resources, "CGROUP_ROOT", str(cgroup_root))
    monkeypatch.setattr(resources, "PROC_SELF_CGROUP_PATH", str(cgroup))

    assert resources.available_memory_bytes() == 4 * budget


def test_density_memory_estimate_grows_with_cell_volume() -> None:
    small = resources.estimate_from_properties(
        "tiny",
        {"AAXIS": 30, "BAXIS": 40, "CAXIS": 50, "RESOLUTION": 2.0},
    )
    large = resources.estimate_from_properties(
        "huge",
        {"AAXIS": 210, "BAXIS": 450, "CAXIS": 620, "RESOLUTION": 2.5},
    )

    assert small is not None and large is not None
    assert small.bytes == resources.AUTO_WORKER_MEMORY_BYTES
    assert large.bytes > 8 * 1024**3
    assert large.combined_map_bytes is not None


def test_entry_estimate_reads_only_leading_properties(
    tmp_path: Path,
) -> None:
    pdb_id = "1abc"
    entry = tmp_path / "ab" / pdb_id
    entry.mkdir(parents=True)
    (entry / "data.json").write_text(
        '{"properties":{"AAXIS":200,"BAXIS":400,"CAXIS":600,'
        '"RESOLUTION":2.5},"large_later_value":"' + "x" * (5 * 1024**2) + '"}',
        encoding="utf-8",
    )

    estimate = resources.estimate_entry_memory(pdb_id, str(tmp_path))

    assert estimate.source == "data_json"
    assert estimate.bytes > resources.AUTO_WORKER_MEMORY_BYTES


def test_entry_estimate_falls_back_to_mtz_size(tmp_path: Path) -> None:
    pdb_id = "1abc"
    entry = tmp_path / "ab" / pdb_id
    entry.mkdir(parents=True)
    (entry / "data.json").write_text("", encoding="utf-8")
    mtz = entry / f"{pdb_id}_final.mtz"
    mtz.write_bytes(b"x" * 1024)

    estimate = resources.estimate_entry_memory(pdb_id, str(tmp_path))

    assert estimate.source == "mtz_size"
    assert estimate.bytes == resources.AUTO_WORKER_MEMORY_BYTES


def test_weighted_admission_skips_a_blocked_large_entry() -> None:
    gib = 1024**3
    active = [resources.EntryMemoryEstimate("active-small", 2 * gib, "test")]
    pending = [
        resources.EntryMemoryEstimate("large", 7 * gib, "test"),
        resources.EntryMemoryEstimate("small", gib, "test"),
    ]

    admitted = pool.pop_admissible_estimate(pending, 2 * gib, 5 * gib, active)

    assert admitted is not None and admitted.pdb_id == "small"
    assert [estimate.pdb_id for estimate in pending] == ["large"]


def test_oversized_entry_is_admitted_only_after_active_work_drains() -> None:
    gib = 1024**3
    active = [resources.EntryMemoryEstimate("active-small", 2 * gib, "test")]
    pending = [resources.EntryMemoryEstimate("large", 7 * gib, "test")]

    assert pool.pop_admissible_estimate(pending, 2 * gib, 5 * gib, active) is None
    admitted = pool.pop_admissible_estimate(pending, 0, 5 * gib, [])
    assert admitted is not None and admitted.pdb_id == "large"


def test_high_memory_entry_allows_two_ordinary_companions() -> None:
    gib = 1024**3
    large = resources.EntryMemoryEstimate("large", 3 * gib, "test")
    first = resources.EntryMemoryEstimate("first", 2 * gib, "test")
    second = resources.EntryMemoryEstimate("second", 2 * gib, "test")
    pending = [first, second]

    admitted = pool.pop_admissible_estimate(pending, 3 * gib, 20 * gib, [large])
    assert admitted == first

    admitted = pool.pop_admissible_estimate(pending, 5 * gib, 20 * gib, [large, first])
    assert admitted == second


def test_high_memory_entry_blocks_a_third_ordinary_companion() -> None:
    gib = 1024**3
    active = [
        resources.EntryMemoryEstimate("large", 3 * gib, "test"),
        resources.EntryMemoryEstimate("first", 2 * gib, "test"),
        resources.EntryMemoryEstimate("second", 2 * gib, "test"),
    ]
    pending = [resources.EntryMemoryEstimate("third", 2 * gib, "test")]

    assert pool.pop_admissible_estimate(pending, 7 * gib, 20 * gib, active) is None


def test_high_memory_entries_never_overlap() -> None:
    gib = 1024**3
    active = [resources.EntryMemoryEstimate("large", 3 * gib, "test")]
    pending = [resources.EntryMemoryEstimate("another-large", 4 * gib, "test")]

    assert pool.pop_admissible_estimate(pending, 3 * gib, 20 * gib, active) is None


def test_missing_ccp4_tools_are_named_with_a_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message must say which tools are absent and how to fix it."""

    def missing_edstats(tool: str, path: str | None = None) -> str | None:
        del path
        return None if tool == "edstats" else "/x"

    monkeypatch.setattr("ccp4_setup.shutil.which", missing_edstats)

    with pytest.raises(ccp4_setup.Ccp4SetupError) as excinfo:
        ccp4_setup.verify_ccp4({"PATH": "/x"})

    message = str(excinfo.value)
    assert "edstats" in message
    assert "--configure-ccp4" in message, "the error should name the remedy"


def test_a_complete_ccp4_installation_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    def installed_tool(tool: str, path: str | None = None) -> str:
        del path
        return f"/opt/{tool}"

    monkeypatch.setattr("ccp4_setup.shutil.which", installed_tool)
    ccp4_setup.verify_ccp4({"PATH": "/opt"})


def test_tool_availability_agrees_with_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ccp4_tools_available`` and ``verify_ccp4`` must not disagree, or the
    driver accepts an installation the setup helper rejects."""

    def missing_fft(tool: str, path: str | None = None) -> str | None:
        del path
        return None if tool == "fft" else "/x"

    monkeypatch.setattr("ccp4_setup.shutil.which", missing_fft)

    assert not ccp4_setup.ccp4_tools_available({"PATH": "/x"})
    with pytest.raises(ccp4_setup.Ccp4SetupError):
        ccp4_setup.verify_ccp4({"PATH": "/x"})


def test_a_library_caller_never_has_to_catch_systemexit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ccp4_setup`` raises an ordinary exception, not ``SystemExit``.

    ``SystemExit`` derives from ``BaseException``, so a caller outside a CLI
    process -- a notebook, a service -- cannot contain it with ``except
    Exception``. The CLI's conversion to an exit is ``test_cli_and_config``.
    """

    def no_tools(_tool: str, path: str | None = None) -> None:
        del path
        return None

    monkeypatch.setattr("ccp4_setup.shutil.which", no_tools)

    with pytest.raises(Exception) as excinfo:
        ccp4_setup.verify_ccp4({"PATH": "/x"})
    assert not isinstance(excinfo.value, SystemExit)


def _minimal_mtz(path: Path, high: float = 1.5) -> str:
    """An MTZ whose only purpose is to carry a known resolution limit."""
    import gemmi
    import numpy as np

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.cell = gemmi.UnitCell(high * 2, high * 2, high * 2, 90, 90, 90)
    dataset = mtz.add_dataset("test")
    mtz.add_column("F", "F", dataset.id)
    mtz.set_data(np.asarray([[1, 0, 0, 10.0], [0, 1, 0, 10.0]], dtype=np.float32))
    mtz.write_to_file(str(path))
    return str(path)


def _map_coefficient_mtz(
    path: Path,
    rows: Sequence[Sequence[float]],
    *,
    columns: Sequence[str] = inputs.MAP_COEFFICIENT_COLUMNS,
) -> str:
    """An MTZ carrying the four EDSTATS map-coefficient columns.

    ``rows`` are ``(h, k, l, *values)`` with one value per column, so a caller
    can make an individual coefficient non-finite without touching the others.
    """
    import gemmi
    import numpy as np

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.cell = gemmi.UnitCell(40, 40, 40, 90, 90, 90)
    dataset = mtz.add_dataset("test")
    for label in columns:
        mtz.add_column(
            label,
            "F" if label.startswith(("FWT", "DELFWT")) else "P",
            dataset.id,
        )
    mtz.set_data(np.asarray(rows, dtype=np.float32))
    mtz.write_to_file(str(path))
    return str(path)


def _d_spacing(mtz_path: str) -> NDArray[np.float32]:
    import gemmi

    return gemmi.read_mtz_file(mtz_path).make_d_array()


def test_map_column_resolution_spans_only_wholly_finite_reflections(
    tmp_path: Path,
) -> None:
    """EDSTATS is given the range where all four coefficients exist.

    ``docs/method.md`` states this deliberately: limits taken from the overall
    MTZ would describe reflections the maps were not calculated from. A bug in
    the whole-row mask would silently move the limits behind every RSZD in the
    database, so the mask is checked against a row that is finite in three
    columns and not the fourth.
    """
    nan = float("nan")
    path = _map_coefficient_mtz(
        tmp_path / "coefficients.mtz",
        [
            [1, 0, 0, 10.0, 20.0, 1.0, 30.0],
            [3, 0, 0, 10.0, 20.0, 1.0, 30.0],
            # Finite everywhere but DELFWT, and the highest-resolution row, so
            # a mask that did not span whole rows would extend reshi onto it.
            [4, 0, 0, 10.0, 20.0, nan, 30.0],
        ],
    )
    spacings = _d_spacing(path)

    reslo, reshi = inputs.read_map_column_resolution(path)

    assert (reslo, reshi) == approx(
        (float(max(spacings[0], spacings[1])), float(min(spacings[0], spacings[1])))
    )
    assert reshi > float(spacings[2])


def test_map_column_resolution_names_every_missing_column(tmp_path: Path) -> None:
    """The error says which coefficients are absent, not merely that one is."""
    path = _map_coefficient_mtz(
        tmp_path / "partial.mtz",
        [[1, 0, 0, 10.0, 20.0]],
        columns=("FWT", "PHWT"),
    )

    with pytest.raises(ValueError) as excinfo:
        inputs.read_map_column_resolution(path)

    message = str(excinfo.value)
    assert "DELFWT" in message and "PHDELWT" in message
    assert "FWT" in message


def test_map_column_resolution_rejects_a_wholly_unusable_set(tmp_path: Path) -> None:
    """No reflection finite in all four columns is an error, not an empty range."""
    nan = float("nan")
    path = _map_coefficient_mtz(
        tmp_path / "unusable.mtz",
        [
            [1, 0, 0, 10.0, 20.0, nan, 30.0],
            [2, 0, 0, nan, 20.0, 1.0, 30.0],
        ],
    )

    with pytest.raises(ValueError, match="no common finite reflections"):
        inputs.read_map_column_resolution(path)


def test_resolution_is_read_from_data_json_when_complete(tmp_path: Path) -> None:
    """data.json is authoritative because DPI metadata is derived from it."""
    entry_dir = _make_entry(
        tmp_path,
        "9myr",
        data_json=json.dumps({"properties": {"DATARESL": 47.1, "DATARESH": 1.72}}),
    )
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    assert inputs.read_resolution(entry_dir, mtz) == approx(1.72)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"properties": {"DATARESL": 47.1}}', "no DATARESH"),
        ('{"properties": {"DATARESH": 1.72}}', "no DATARESL"),
        ('{"properties": {}}', "neither limit"),
        ("{not valid json", "malformed"),
        ("", "empty"),
    ],
)
def test_incomplete_metadata_falls_back_to_the_mtz(
    tmp_path: Path, payload: str, reason: str
) -> None:
    """Both limits are required before data.json is believed, or a partial
    metadata source is mixed into DPI provenance."""
    entry_dir = _make_entry(tmp_path, "9myr", data_json=payload)
    mtz = _minimal_mtz(tmp_path / "fallback.mtz")

    resolution = inputs.read_resolution(entry_dir, mtz)
    assert resolution == approx(
        inputs.read_resolution(str(tmp_path / "absent"), mtz)
    ), f"{reason} should have fallen back to the MTZ"


def test_absent_metadata_falls_back_to_the_mtz(tmp_path: Path) -> None:
    """A mirror without data.json is still analyzable, DPI aside."""
    entry_dir = _make_entry(tmp_path, "9myr")
    mtz = _minimal_mtz(tmp_path / "only.mtz")

    assert inputs.read_resolution(entry_dir, mtz) > 0.0


def test_an_explicit_data_json_path_overrides_the_entry_directory(
    tmp_path: Path,
) -> None:
    """``--data-json`` supplies metadata for manual inputs with no entry dir.

    It must win over any file sitting beside the coordinates, or a manual run
    inside a populated mirror reads the wrong entry's metadata.
    """
    entry_dir = _make_entry(
        tmp_path,
        "9myr",
        data_json=json.dumps({"properties": {"DATARESL": 40.0, "DATARESH": 9.99}}),
    )
    explicit = tmp_path / "explicit.json"
    explicit.write_text(
        json.dumps({"properties": {"DATARESL": 47.1, "DATARESH": 1.72}}),
        encoding="utf-8",
    )
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    assert inputs.read_resolution(
        entry_dir, mtz, data_json_path=str(explicit)
    ) == approx(1.72)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "file not found"),
        ("{not valid json", "invalid JSON"),
        ("[]", "must contain a JSON object"),
        ("{}", "must contain a properties object"),
    ],
)
def test_explicit_invalid_data_json_does_not_fall_back_to_mtz(
    tmp_path: Path, payload: str | None, message: str
) -> None:
    """A requested metadata file is an input contract, not an optional probe."""
    data_json = tmp_path / "explicit.json"
    if payload is not None:
        data_json.write_text(payload, encoding="utf-8")
    mtz = _minimal_mtz(tmp_path / "fallback.mtz")

    with pytest.raises(ValueError, match=message):
        inputs.read_resolution(
            str(tmp_path),
            mtz,
            data_json_path=str(data_json),
        )


def test_manual_run_rejects_invalid_explicit_data_json_before_scheduling(
    tmp_path: Path,
) -> None:
    """A CLI typo fails as a driver input error instead of reaching a worker."""
    missing = tmp_path / "missing.json"
    args = cli.parse_args(
        [
            "--id",
            "9myr",
            "--pdb-file",
            "9myr.pdb",
            "--mtz-file",
            "9myr.mtz",
            "--data-json",
            str(missing),
        ]
    )

    with pytest.raises(pool.DriverError, match=r"Invalid --data-json:.*not found"):
        pool.select_entry_ids(args, str(tmp_path / "cache"))


def test_intermediates_are_discarded_unless_asked_for() -> None:
    """Per-entry maps are large, so retention is opt-in: ``process`` keys its
    scratch cleanup off this flag."""
    assert cli.parse_args([]).keep_intermediates is False
    assert cli.parse_args(["--keep-intermediates"]).keep_intermediates is True


def test_an_id_file_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    """A BOM belongs to the encoding, not to the first id.

    Windows editors and spreadsheet exports add one routinely, and under plain
    utf-8 it was glued to the first token, failing validation with a message
    that blamed the id.
    """
    path = tmp_path / "ids.txt"
    path.write_text("9myr, 6nlr\n", encoding="utf-8-sig")

    assert pool.load_ids_from_file(str(path)) == ["9myr", "6nlr"]


@pytest.mark.parametrize(
    "argv, fragment",
    [
        (["--pdb-file", "/tmp/1abc.pdb"], "requires --mtz-file"),
        (["--cif-file", "/tmp/1abc.cif"], "requires --mtz-file"),
        (["--mtz-file", "/tmp/1abc.mtz"], "requires --pdb-file or --cif-file"),
        (
            [
                "--pdb-file",
                "/tmp/1abc.pdb",
                "--cif-file",
                "/tmp/1abc.cif",
                "--mtz-file",
                "/tmp/1abc.mtz",
            ],
            "not both",
        ),
    ],
    ids=["pdb-alone", "cif-alone", "mtz-alone", "pdb-and-cif"],
)
def test_incomplete_manual_input_is_a_usage_error(
    argv: list[str], fragment: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Manual mode needs coordinates and reflections, and only one of each.

    These reached a worker before failing, and were reported as an unexpected
    processing error rather than as the usage mistake they are. Supplying both
    coordinate forms silently used the cif and ignored the pdb.
    """
    with pytest.raises(SystemExit):
        cli.parse_args(argv)

    assert fragment in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [b"", b"<!DOCTYPE html>\n<html><body>login</body></html>\n", b"  <html>\n"],
    ids=["empty", "captive-portal", "leading-space"],
)
def test_a_cached_body_that_is_not_entry_data_is_not_treated_as_cached(
    tmp_path: Path, content: bytes
) -> None:
    """A 200 response can still carry a login page under the entry's own name.

    Existence alone was the cache test, so such a body was reused forever: the
    entry failed identically on every resume with no way to recover short of
    deleting the cache by hand.
    """
    entry = tmp_path / "my" / "9myr"
    entry.mkdir(parents=True)
    (entry / "9myr_final.mtz").write_bytes(content)
    (entry / "9myr_final.cif").write_bytes(content)

    assert inputs.has_final_files(str(entry), "9myr") is False


def test_real_entry_bytes_are_accepted_as_cached(tmp_path: Path) -> None:
    """The validity check must not reject an ordinary entry."""
    entry = tmp_path / "my" / "9myr"
    entry.mkdir(parents=True)
    (entry / "9myr_final.mtz").write_bytes(b"MTZ \x00\x01binary payload")
    (entry / "9myr_final.cif").write_text(
        "data_9MYR\n_entry.id 9MYR\n", encoding="ascii"
    )

    assert inputs.has_final_files(str(entry), "9myr") is True


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
    # A stand-in for EntryResult: should_write_entry reads only these three
    # fields, and the real dataclass would need six unrelated ones supplied.
    result = cast(
        "EntryResult",
        SimpleNamespace(pdb_id=pdb_id, status=status, retryable=retryable),
    )

    assert pool.should_write_entry(resuming, result, prior) is expected, why


def test_an_uncapped_database_run_says_it_ignores_an_explicit_reference() -> None:
    """That run builds the reference, so it cannot be scored against one.

    The flag was accepted in silence, leaving an operator believing their
    reference had been used on the one run where it never could be. It is
    reported rather than refused: passing the flag uniformly across capped and
    uncapped runs is reasonable, and failing a multi-day run over an argument
    that changes nothing would be worse than saying so.
    """
    args = cli.parse_args(["--confidence-reference-dir", "/tmp/reference"])
    # Captured with a handler on the logger itself rather than through caplog:
    # the run configures ``alchemy`` not to propagate, so whether caplog sees
    # anything depends on which tests ran first.
    messages: list[str] = []

    class _Capture(logging.Handler):
        # No @override: typing.override arrived in 3.12 and this project still
        # supports 3.11, where importing it would fail at run time.
        def emit(  # type: ignore[explicit-override]
            self, record: logging.LogRecord
        ) -> None:
            messages.append(record.getMessage())

    alchemy_logger = logging.getLogger("alchemy.pool")
    handler = _Capture()
    previous_level = alchemy_logger.level
    alchemy_logger.addHandler(handler)
    alchemy_logger.setLevel(logging.WARNING)
    try:
        # ``None`` for the run log: the uncapped-database branch returns before
        # it is touched, and the parameter is not declared Optional.
        plan = pool.plan_confidence(
            args, pool.OutputLayout("/tmp/out"), True, cast("runlog.RunLog", None)
        )
    finally:
        alchemy_logger.removeHandler(handler)
        alchemy_logger.setLevel(previous_level)

    assert plan.mode == "database"
    assert plan.reference is None
    assert any("is ignored on an uncapped" in message for message in messages)
    assert any("/tmp/reference" in message for message in messages)


def test_targeted_run_without_reference_still_plans_classifications(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(["--id", "1abc", "--output-dir", str(tmp_path)])
    plan = pool.plan_confidence(
        args,
        pool.OutputLayout(str(tmp_path)),
        False,
        cast("runlog.RunLog", None),
    )
    assert plan.mode == "classification"
    assert plan.reference is None
    assert plan.stream_path == str(tmp_path / "confidence_scores_all.csv")
    assert plan.columns == (
        *confidence_score.CONFIDENCE_INPUT_COLUMNS,
        *confidence_score.ANALYSIS_COLUMNS,
    )


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
