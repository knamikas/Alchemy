"""Test the resume schema and artifact-consistency validation."""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

import pytest

import confidence_score
from coordination import schema as coordination_schema
from driver import resume, writers
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS


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
    """Verify resume rejects a foreign output header.

    Appending beneath a foreign header would misalign every column, and nothing
    downstream could tell that column N had changed meaning.
    """
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


def test_a_manifest_without_the_analysis_config_column_is_refused(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    older = [
        column for column in writers.MANIFEST_COLUMNS if column != "analysis_config_id"
    ]
    _write_header(tmp_path / "manifest.csv", older)

    with pytest.raises(ValueError, match="missing analysis_config_id"):
        resume.validate_resume_schemas(**resume_outputs)


def test_the_refusal_names_the_columns_that_differ(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    """Verify schema errors identify differing columns.

    Naming the columns that differ gives the operator a cause rather than two
    headers to diff by hand.
    """
    _write_header(tmp_path / "manifest.csv", list(writers.MANIFEST_COLUMNS) + ["stray"])

    with pytest.raises(ValueError, match="unexpected stray"):
        resume.validate_resume_schemas(**resume_outputs)


def test_a_truncated_stats_header_is_refused(
    resume_outputs: _ResumeOutputs, tmp_path: Path
) -> None:
    """Verify statistics resume rejects a shortened density schema.

    A different EDSTATS build shifts the density block, and a dropped metric
    column misaligns every later value with no other symptom.
    """
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


@pytest.mark.parametrize("outputs_present", [False, True])
@pytest.mark.parametrize("scored", [False, True])
def test_density_only_entries_can_resume_without_bond_or_confidence_rows(
    resume_outputs: _ResumeOutputs,
    tmp_path: Path,
    outputs_present: bool,
    scored: bool,
) -> None:
    columns: list[str] = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
    if scored:
        columns.extend(confidence_score.ANALYSIS_COLUMNS)
    path = _write_header(tmp_path / "confidence.csv", columns)
    _append_terminal_manifest(resume_outputs, n_metals=1, n_bonds="", n_candidates="")
    _append_selected_stats(resume_outputs)
    if not outputs_present:
        for output in (
            path,
            resume_outputs["bonds_path"],
            resume_outputs["candidates_path"],
        ):
            os.unlink(output)

    resume.validate_resume_schemas(
        **resume_outputs, confidence_path=path, confidence_columns=columns
    )
    assert (
        resume.load_done(resume_outputs["manifest_path"], bonds_required=True) == set()
    )


@pytest.mark.parametrize("confidence_present", [False, True])
def test_density_only_entry_does_not_hide_missing_completed_confidence(
    resume_outputs: _ResumeOutputs, tmp_path: Path, confidence_present: bool
) -> None:
    columns = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
    path = str(tmp_path / "confidence.csv")
    if confidence_present:
        _write_header(Path(path), columns)
    _append_terminal_manifest(resume_outputs, n_metals=1, n_bonds="", n_candidates="")
    _append_selected_stats(resume_outputs)
    _append_terminal_manifest(resume_outputs, pdb_id="2def", n_metals=1)
    _append_selected_stats(resume_outputs, pdb_id="2def")

    with pytest.raises(ValueError, match="confidence.csv"):
        resume.validate_resume_schemas(
            **resume_outputs, confidence_path=path, confidence_columns=columns
        )

    _write_header(Path(path), columns)
    _append_csv_row(path, columns, pdbID="2def")
    resume.validate_resume_schemas(
        **resume_outputs, confidence_path=path, confidence_columns=columns
    )


def test_density_only_entry_still_requires_its_selected_stats(
    resume_outputs: _ResumeOutputs,
) -> None:
    _append_terminal_manifest(resume_outputs, n_metals=1, n_bonds="", n_candidates="")
    with pytest.raises(ValueError, match=r"n_metals=1.*has 0 selected row"):
        resume.validate_resume_schemas(**resume_outputs)


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
