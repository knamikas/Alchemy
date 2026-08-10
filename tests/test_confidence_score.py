"""Behavioral tests for the final three-level confidence method."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Protocol, cast
from collections.abc import Mapping, Sequence

import pytest

import analysis_config
import confidence_score as cs
import helpers
import reference_data
from coordination.schema import STATS_EXTRA_COLUMNS
from output_rows import MetalStatsRow


class _ApproxFactory(Protocol):
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


STATS_ID_COLUMNS = ["pdbID", "category"]
STATS_FIELD_COLUMNS = (
    list(helpers.EDSTATS_HEADER) + ["aa_geometry_coverage"] + list(STATS_EXTRA_COLUMNS)
)
STATS_COLUMNS = STATS_ID_COLUMNS + STATS_FIELD_COLUMNS
ANALYSIS_CONFIG_ID = analysis_config.analysis_config_id(
    reference_data_id=reference_data.reference_data_id()
)

SITE = {
    "metal_model_index": "0",
    "metal_chain_index": "1",
    "metal_residue_index": "2",
    "metal_atom_index": "0",
}


def _stats_row(
    pdb_id: str = "1abc",
    zdm: object = 3.0,
    zd_neg: object = -1.0,
    zd_pos: object = 2.0,
    **overrides: str,
) -> dict[str, str]:
    row = {column: "" for column in cs.IDENTITY_COLUMNS}
    row.update(SITE)
    row.update(
        {
            "pdbID": pdb_id,
            "metal_site_id": f"{pdb_id}:m0:c1:r2:a0",
            "category": "metal",
            "density_observation_id": f"{pdb_id}/ZN",
            "density_scope": "site",
            "density_shared_site_count": "1",
            "density_is_shared": "False",
            "coordinate_mapping_status": "mapped",
            "selected_metal_site_status": "selected",
            "metal_resname": "ZN",
            "metal_chain": "B",
            "metal_resnum": "1",
            "metal_atom": "ZN",
            "metal_element": "ZN",
            "ZDm": "" if zdm is None else str(zdm),
            "ZD-m": "" if zd_neg is None else str(zd_neg),
            "ZD+m": "" if zd_pos is None else str(zd_pos),
            "suspect_multi_donor_residue_group_count": "0",
            "context_warning": "False",
            "context_warning_reasons": "",
        }
    )
    row.update(overrides)
    return row


def _bond_row(
    pdb_id: str = "1abc",
    covered: bool = True,
    zscore: float | None = 1.5,
    neighbor: str = "HIS",
    atom: str = "NE2",
    **overrides: str,
) -> dict[str, str]:
    row = dict(SITE)
    row.update(
        {
            "pdbID": pdb_id,
            "metal_site_id": f"{pdb_id}:m0:c1:r2:a0",
            "contact_id": f"{pdb_id}:m0:c1:r2:a0:c{neighbor}-{atom}",
            "parent_type": "ion",
            "metal_resname": "ZN",
            "metal_chain": "B",
            "metal_resnum": "1",
            "metal_atom": "ZN",
            "metal_element": "ZN",
            "neighbor_resname": neighbor,
            "neighbor_chain": "A",
            "neighbor_resnum": "10",
            "neighbor_atom": atom,
            "reference_covered": str(covered),
            "declared_connection": "False",
            "coordination_status": "inferred",
            "multi_donor_detected": "False",
            "score_eligible": str(zscore is not None),
            "score_exclusion_reason": ""
            if zscore is not None
            else "zscore_unavailable",
            "zscore": "" if zscore is None else str(zscore),
            "context_warning": "False",
            "context_warning_reasons": "",
        }
    )
    row.update(overrides)
    return row


def _input_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in cs.CONFIDENCE_INPUT_COLUMNS}
    row.update(SITE)
    row.update(
        {
            "pdbID": "1abc",
            "metal_site_id": "1abc:m0:c1:r2:a0",
            "category": "metal",
            "selected_metal_site_status": "selected",
            "metal_resname": "ZN",
            "metal_element": "ZN",
            "metal_chain": "B",
            "metal_resnum": "1",
            "metal_atom": "ZN",
            "rszd": "3",
            "rszd_abs": "3",
            "density_saturated": "False",
            "geometry_rms_zbond": "1.5",
            "geometry_max_abs_zbond": "2",
            "geometry_coverage": "1",
            "assigned_contact_count": "2",
            "reference_covered_contact_count": "2",
            "geometry_bond_count": "2",
            "context_warning": "False",
            "context_warning_reasons": "",
            "confidence_inputs_status": "complete",
            "confidence_inputs_missing_reasons": "",
        }
    )
    row.update(overrides)
    return row


def _write_input_csv(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str] = cs.CONFIDENCE_INPUT_COLUMNS,
) -> str:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _reference() -> cs.ConfidenceReference:
    return cs.ConfidenceReference(
        [1.0, 3.0, 6.0],
        [1, 2, 1],
        [0.5, 1.0, 2.0],
        [1, 2, 1],
        {
            "reference_id": "alchemy-confidence-test",
            "cohort_id": "alchemy-cohort-test",
            "input_row_count": 4,
        },
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "PASS"),
        (2.999999, "PASS"),
        (3.0, "REVIEW"),
        (5.999999, "REVIEW"),
        (6.0, "SUSPECT"),
        (99.9, "SUSPECT"),
        (math.nan, "INCOMPLETE"),
        (math.inf, "INCOMPLETE"),
        (-1.0, "INCOMPLETE"),
    ],
)
def test_density_levels_use_raw_final_thresholds(value: float, expected: str) -> None:
    assert cs.density_level(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "PASS"),
        (0.999999, "PASS"),
        (1.0, "REVIEW"),
        (1.999999, "REVIEW"),
        (2.0, "SUSPECT"),
        (20.0, "SUSPECT"),
        (math.nan, "INCOMPLETE"),
    ],
)
def test_geometry_levels_use_rms_final_thresholds(value: float, expected: str) -> None:
    assert cs.geometry_level(value) == expected


@pytest.mark.parametrize(
    ("density", "geometry", "overall"),
    [
        ("PASS", "PASS", "PASS"),
        ("REVIEW", "PASS", "REVIEW"),
        ("PASS", "REVIEW", "REVIEW"),
        ("REVIEW", "REVIEW", "SUSPECT"),
        ("SUSPECT", "PASS", "SUSPECT"),
        ("PASS", "SUSPECT", "SUSPECT"),
        ("SUSPECT", "REVIEW", "SUSPECT"),
        ("REVIEW", "SUSPECT", "SUSPECT"),
        ("SUSPECT", "SUSPECT", "SUSPECT"),
    ],
)
def test_complete_evidence_decision_matrix(
    density: str, geometry: str, overall: str
) -> None:
    representative = {"PASS": 0.5, "REVIEW": 1.5, "SUSPECT": 7.0}
    density_value = {"PASS": 1.0, "REVIEW": 4.0, "SUSPECT": 7.0}[density]
    result = cs.classify_site(density_value, representative[geometry])
    assert result["density_level"] == density
    assert result["geometry_level"] == geometry
    assert result["alchemy_level"] == overall
    assert result["evidence_basis"] == "density_and_geometry"


@pytest.mark.parametrize(
    ("rszd", "rms", "basis", "overall"),
    [
        (1.0, math.nan, "density_only", "PASS"),
        (4.0, math.nan, "density_only", "REVIEW"),
        (7.0, math.nan, "density_only", "SUSPECT"),
        (math.nan, 0.5, "geometry_only", "PASS"),
        (math.nan, 1.5, "geometry_only", "REVIEW"),
        (math.nan, 2.5, "geometry_only", "SUSPECT"),
        (math.nan, math.nan, "no_assessable_evidence", "INCOMPLETE"),
    ],
)
def test_missing_component_uses_available_evidence_directly(
    rszd: float, rms: float, basis: str, overall: str
) -> None:
    result = cs.classify_site(rszd, rms)
    assert result["evidence_basis"] == basis
    assert result["alchemy_level"] == overall


def test_verdict_reasons_distinguish_all_suspect_routes() -> None:
    assert cs.classify_site(7.0, 2.5)["verdict_reason"] == (
        "density_and_geometry_suspect"
    )
    assert cs.classify_site(7.0, 0.5)["verdict_reason"] == "density_suspect"
    assert cs.classify_site(1.0, 2.5)["verdict_reason"] == "geometry_suspect"
    assert cs.classify_site(4.0, 1.5)["verdict_reason"] == "review_plus_review"


def test_empirical_support_is_reverse_average_rank_with_ties() -> None:
    reference = _reference()
    assert reference.density.support_score(0.0) == approx(100.0)
    assert reference.density.support_score(1.0) == approx(87.5)
    assert reference.density.support_score(3.0) == approx(50.0)
    assert reference.density.support_score(4.0) == approx(25.0)
    assert reference.density.support_score(7.0) == approx(0.0)


def test_overall_ranking_score_is_minimum_available_support() -> None:
    reference = _reference()
    both = cs.score_site(1.0, 0.5, reference)
    assert both["density_score"] == approx(87.5)
    assert both["geometry_score"] == approx(87.5)
    assert both["alchemy_score"] == approx(87.5)

    density_only = cs.score_site(3.0, math.nan, reference)
    assert density_only["alchemy_score"] == approx(50.0)
    assert math.isnan(float(density_only["geometry_score"]))


def test_edstats_saturation_receives_zero_density_support() -> None:
    result = cs.score_site(99.9, math.nan, _reference())
    assert result["density_score"] == 0.0
    assert result["alchemy_score"] == 0.0


def test_ranking_score_does_not_define_review_plus_review_verdict() -> None:
    result = cs.score_site(4.0, 1.5, _reference())
    assert result["alchemy_level"] == "SUSPECT"
    assert float(result["alchemy_score"]) > 0.0


def test_classification_without_reference_keeps_levels_and_blanks_rankings() -> None:
    scored = cs.classify_without_reference(
        [_input_row(rszd_abs="4", geometry_rms_zbond="1.5")]
    )[0]
    assert scored["density_level"] == "REVIEW"
    assert scored["geometry_level"] == "REVIEW"
    assert scored["alchemy_level"] == "SUSPECT"
    assert scored["density_score"] == ""
    assert scored["geometry_score"] == ""
    assert scored["alchemy_score"] == ""
    assert scored["confidence_reference_version"] == ""


def test_geometry_summary_uses_rms_of_every_finite_score_eligible_contact() -> None:
    bonds = [
        _bond_row(zscore=1.0, neighbor="HIS", atom="NE2"),
        _bond_row(
            zscore=-2.0,
            neighbor="ASP",
            atom="OD1",
            declared_connection="True",
            coordination_status="declared",
        ),
        _bond_row(
            zscore=99.0,
            neighbor="GLU",
            atom="OE1",
            score_eligible="False",
            score_exclusion_reason="test_exclusion",
        ),
        _bond_row(zscore=None, neighbor="HOH", atom="O"),
    ]
    prepared = cs.prepare_confidence_inputs([_stats_row()], bonds)[0]

    assert int(prepared["geometry_bond_count"]) == 2
    assert float(prepared["geometry_rms_zbond"]) == approx(math.sqrt(2.5))
    assert float(prepared["geometry_mean_abs_zbond"]) == approx(1.5)
    assert float(prepared["geometry_mean_signed_zbond"]) == approx(-0.5)
    assert float(prepared["geometry_max_abs_zbond"]) == approx(2.0)
    assert prepared["worst_bond"].endswith(":cASP-OD1")
    assert prepared["worst_bond_source"] == "declared"
    assert prepared["declared_scored_bond_count"] == 1
    assert prepared["inferred_scored_bond_count"] == 1
    assert prepared["geometry_contact_basis"] == "declared_and_inferred"


def test_multiple_moderate_inferred_bonds_can_make_geometry_suspect() -> None:
    bonds = [
        _bond_row(zscore=value, neighbor=f"L{i}", atom="O")
        for i, value in enumerate((2.1, -2.2, 2.3, -2.4))
    ]
    prepared = cs.prepare_confidence_inputs([_stats_row(zdm=1.0)], bonds)[0]
    assert float(prepared["geometry_max_abs_zbond"]) < 3.0
    result = cs.classify_site(1.0, float(prepared["geometry_rms_zbond"]))
    assert result["geometry_level"] == "SUSPECT"
    assert result["alchemy_level"] == "SUSPECT"


def test_severe_declared_contact_can_dominate_site_rms() -> None:
    bonds = [_bond_row(zscore=0.2, neighbor=f"H{i}", atom="N") for i in range(5)]
    bonds.append(
        _bond_row(
            zscore=7.0,
            neighbor="ASP",
            atom="OD1",
            declared_connection="True",
            coordination_status="declared",
        )
    )
    prepared = cs.prepare_confidence_inputs([_stats_row(zdm=1.0)], bonds)[0]
    assert cs.geometry_level(float(prepared["geometry_rms_zbond"])) == "SUSPECT"
    assert prepared["worst_bond_source"] == "declared"


def test_geometry_coverage_is_annotation_only() -> None:
    complete = cs.prepare_confidence_inputs(
        [_stats_row(zdm=1.0)], [_bond_row(zscore=2.1)]
    )[0]
    partial = cs.prepare_confidence_inputs(
        [_stats_row(zdm=1.0)],
        [_bond_row(zscore=2.1), _bond_row(covered=False, zscore=None, neighbor="UNK")],
    )[0]
    assert float(complete["geometry_rms_zbond"]) == float(partial["geometry_rms_zbond"])
    assert complete["confidence_inputs_status"] == "complete"
    assert partial["confidence_inputs_status"] == "complete"
    assert float(partial["geometry_coverage"]) == approx(0.5)


def test_preparation_retains_density_signs_and_saturation_flag() -> None:
    prepared = cs.prepare_confidence_inputs(
        [_stats_row(zdm=-99.9, zd_neg=-99.9, zd_pos=0.0)], []
    )[0]
    assert prepared["rszd_abs"] == "99.9"
    assert prepared["rszd"] == "-99.9"
    assert prepared["rszd_negative"] == "-99.9"
    assert prepared["rszd_positive"] == "0"
    assert prepared["density_saturated"] is True


def test_missing_density_with_geometry_is_geometry_only() -> None:
    prepared = cs.prepare_confidence_inputs(
        [_stats_row(zdm=None)], [_bond_row(zscore=1.5)]
    )[0]
    assert prepared["rszd_abs"] == ""
    assert prepared["confidence_inputs_status"] == "geometry_only"
    assert prepared["geometry_rms_zbond"] != ""


def test_missing_geometry_with_density_is_density_only() -> None:
    prepared = cs.prepare_confidence_inputs([_stats_row(zdm=4.0)], [])[0]
    assert prepared["confidence_inputs_status"] == "density_only"
    assert prepared["geometry_rms_zbond"] == ""
    assert "no_assigned_contacts" in prepared["confidence_inputs_missing_reasons"]


def test_orphan_bond_site_can_be_scored_as_geometry_only() -> None:
    orphan = cs.prepare_confidence_inputs([], [_bond_row(zscore=2.5)])[0]
    assert orphan["confidence_inputs_status"] == "geometry_only"
    assert orphan["rszd_abs"] == ""
    assert float(orphan["geometry_rms_zbond"]) == approx(2.5)
    assert "density_row_unavailable" in orphan["confidence_inputs_missing_reasons"]


def test_prepare_emits_one_row_per_selected_site_and_rejects_duplicates() -> None:
    unselected = _stats_row(
        pdb_id="2def", selected_metal_site_status="diagnostic_unmatched"
    )
    assert len(cs.prepare_confidence_inputs([_stats_row(), unselected], [])) == 1
    with pytest.raises(ValueError, match="duplicate site key"):
        cs.prepare_confidence_inputs([_stats_row(), _stats_row()], [])


def test_prepare_result_rows_match_mapping_preparation() -> None:
    mapping = _stats_row()
    values = [mapping.get(column, "") for column in STATS_COLUMNS[2:]]
    result = MetalStatsRow.from_output_fields("1abc", "metal", values)
    expected = cs.prepare_confidence_inputs([mapping], [_bond_row()])
    assert (
        cs.prepare_result_confidence_inputs([result], [_bond_row()], STATS_COLUMNS)
        == expected
    )


def test_completion_retains_evidence_and_adds_unresolved_placeholders() -> None:
    rows = cs.prepare_confidence_inputs([_stats_row()], [])
    completed = cs.complete_confidence_site_count(
        rows, "1abc", 2, missing_reason="bond_stage_failure"
    )
    assert completed[0]["confidence_inputs_status"] == "density_only"
    assert "bond_stage_failure" in completed[0]["confidence_inputs_missing_reasons"]
    assert completed[1]["confidence_inputs_status"] == "unscorable"
    assert completed[1]["confidence_inputs_missing_reasons"].startswith(
        "rszd_unavailable"
    )
    assert rows[0]["confidence_inputs_missing_reasons"] == "no_assigned_contacts"


def test_completion_rejects_more_rows_than_selected_sites() -> None:
    with pytest.raises(ValueError, match="exceed selected metal count"):
        cs.complete_confidence_site_count([_input_row(), _input_row()], "1abc", 1)


def test_reference_round_trip_preserves_both_distributions(tmp_path: Path) -> None:
    provenance = {"cohort_id": "alchemy-cohort-test"}
    reference = cs.write_reference(
        str(tmp_path),
        Counter({1.0: 2, 3.0: 1}),
        Counter({0.5: 1, 2.0: 2}),
        4,
        provenance,
    )
    loaded = cs.load_reference(str(tmp_path))
    assert loaded.reference_id == reference.reference_id
    assert loaded.cohort_id == "alchemy-cohort-test"
    assert loaded.cohort_size == 4
    assert loaded.density.values == (1.0, 3.0)
    assert loaded.density.counts == (2, 1)
    assert loaded.geometry.values == (0.5, 2.0)
    assert loaded.geometry.counts == (1, 2)

    header, rows = _read_csv_rows(tmp_path / cs.REFERENCE_DISTRIBUTION_FILE)
    assert header == ["component", "value", "count"]
    assert {row["component"] for row in rows} == {"density", "geometry"}


def test_reference_identifier_tracks_either_component_distribution(
    tmp_path: Path,
) -> None:
    first = cs.write_reference(str(tmp_path / "a"), {1.0: 1}, {0.5: 1}, 1).reference_id
    same = cs.write_reference(str(tmp_path / "b"), {1.0: 1}, {0.5: 1}, 1).reference_id
    changed_density = cs.write_reference(
        str(tmp_path / "c"), {2.0: 1}, {0.5: 1}, 1
    ).reference_id
    changed_geometry = cs.write_reference(
        str(tmp_path / "d"), {1.0: 1}, {1.5: 1}, 1
    ).reference_id
    assert first == same
    assert len({first, changed_density, changed_geometry}) == 3


def test_write_reference_requires_at_least_one_component(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no evidence"):
        cs.write_reference(str(tmp_path), {}, {}, 1)


def test_load_reference_rejects_tampered_distribution(tmp_path: Path) -> None:
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    distribution = tmp_path / cs.REFERENCE_DISTRIBUTION_FILE
    distribution.write_text(
        "component,value,count\ndensity,1,2\ngeometry,0.5,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identifier"):
        cs.load_reference(str(tmp_path))


def test_load_reference_rejects_incompatible_policy(tmp_path: Path) -> None:
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    metadata_path = tmp_path / cs.REFERENCE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["geometry_thresholds"]["suspect"] = 6.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry_thresholds.*incompatible"):
        cs.load_reference(str(tmp_path))


def test_load_reference_rejects_non_integer_count(tmp_path: Path) -> None:
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    distribution = tmp_path / cs.REFERENCE_DISTRIBUTION_FILE
    distribution.write_text(
        "component,value,count\ndensity,1,1.5\ngeometry,0.5,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-integer"):
        cs.load_reference(str(tmp_path))


def test_finalize_builds_independent_component_cohorts_and_scores_every_basis(
    tmp_path: Path,
) -> None:
    rows = [
        _input_row(
            pdbID="1aaa", metal_site_id="1aaa:1", rszd_abs="1", geometry_rms_zbond="0.5"
        ),
        _input_row(
            pdbID="2bbb", metal_site_id="2bbb:1", rszd_abs="4", geometry_rms_zbond="1.5"
        ),
        _input_row(
            pdbID="3ccc",
            metal_site_id="3ccc:1",
            rszd_abs="",
            geometry_rms_zbond="3",
            confidence_inputs_status="geometry_only",
        ),
        _input_row(
            pdbID="4ddd",
            metal_site_id="4ddd:1",
            rszd_abs="9",
            geometry_rms_zbond="",
            confidence_inputs_status="density_only",
        ),
        _input_row(
            pdbID="5eee",
            metal_site_id="5eee:1",
            rszd_abs="",
            geometry_rms_zbond="",
            confidence_inputs_status="unscorable",
        ),
    ]
    input_path = _write_input_csv(tmp_path / "inputs.csv", rows)
    output_path = tmp_path / "scores.csv"
    reference_dir = tmp_path / "reference"

    total, scored, cohort = cs.finalize_database_confidence(
        input_path, str(output_path), str(reference_dir)
    )
    assert (total, scored, cohort) == (5, 4, 5)
    reference = cs.load_reference(str(reference_dir))
    assert reference.density_reference_size == 3
    assert reference.geometry_reference_size == 3

    columns, output = _read_csv_rows(output_path)
    assert columns == [*cs.CONFIDENCE_INPUT_COLUMNS, *cs.ANALYSIS_COLUMNS]
    by_id = {row["pdbID"]: row for row in output}
    assert by_id["1aaa"]["alchemy_level"] == "PASS"
    assert by_id["2bbb"]["alchemy_level"] == "SUSPECT"
    assert by_id["2bbb"]["verdict_reason"] == "review_plus_review"
    assert by_id["3ccc"]["evidence_basis"] == "geometry_only"
    assert by_id["3ccc"]["alchemy_level"] == "SUSPECT"
    assert by_id["4ddd"]["evidence_basis"] == "density_only"
    assert by_id["5eee"]["alchemy_level"] == "INCOMPLETE"
    assert by_id["5eee"]["alchemy_score"] == ""
    assert {row["score_policy_version"] for row in output} == {
        cs.CONFIDENCE_METHOD_VERSION
    }


def test_small_runs_use_the_frozen_reference_only(tmp_path: Path) -> None:
    reference = cs.write_reference(
        str(tmp_path / "reference"),
        {1.0: 1, 3.0: 2, 6.0: 1},
        {0.5: 1, 1.0: 2, 2.0: 1},
        4,
    )
    row = _input_row(rszd_abs="3", geometry_rms_zbond="1")
    alone = cs.score_against_reference([row], reference)[0]
    with_extremes = cs.score_against_reference(
        [
            _input_row(rszd_abs="0", geometry_rms_zbond="0"),
            row,
            _input_row(rszd_abs="99.9", geometry_rms_zbond="99.9"),
        ],
        reference,
    )[1]
    assert alone["density_score"] == with_extremes["density_score"] == "50"
    assert alone["geometry_score"] == with_extremes["geometry_score"] == "50"


def test_context_warning_is_carried_without_changing_result() -> None:
    plain = _input_row()
    warned = _input_row(
        context_warning="True", context_warning_reasons="declared_non_typical_donor"
    )
    scored_plain, scored_warned = cs.score_against_reference(
        [plain, warned], _reference()
    )
    for column in (
        "density_level",
        "geometry_level",
        "alchemy_level",
        "alchemy_score",
    ):
        assert scored_plain[column] == scored_warned[column]
    assert scored_warned["context_warning"] == "True"


def test_score_file_requires_new_evidence_columns_and_cleans_partial_output(
    tmp_path: Path,
) -> None:
    incomplete_columns = [
        column
        for column in cs.CONFIDENCE_INPUT_COLUMNS
        if column != "geometry_rms_zbond"
    ]
    complete = _input_row()
    incomplete = {column: complete[column] for column in incomplete_columns}
    input_path = _write_input_csv(
        tmp_path / "bad.csv", [incomplete], incomplete_columns
    )
    output_path = tmp_path / "scores.csv"
    with pytest.raises(ValueError, match="geometry_rms_zbond"):
        cs.score_file_against_reference(input_path, str(output_path), _reference())
    assert not output_path.exists()
    assert not (tmp_path / "scores.csv.tmp").exists()


def test_score_file_rejects_already_scored_input(tmp_path: Path) -> None:
    columns = [*cs.CONFIDENCE_INPUT_COLUMNS, *cs.ANALYSIS_COLUMNS]
    row = {**_input_row(), **{column: "" for column in cs.ANALYSIS_COLUMNS}}
    input_path = _write_input_csv(tmp_path / "scored.csv", [row], columns)
    with pytest.raises(ValueError, match="already contains analysis"):
        cs.score_file_against_reference(
            input_path, str(tmp_path / "out.csv"), _reference()
        )


def test_finalize_records_manifest_and_input_provenance(tmp_path: Path) -> None:
    input_path = Path(
        _write_input_csv(tmp_path / "inputs.csv", [_input_row(pdbID="1ABC")])
    )
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pdbID",
                "status",
                "no_metals",
                "metal_site_limit_exceeded",
                "n_metals",
                "analysis_config_id",
                "alchemy_version",
                "alchemy_commit",
                "gemmi_version",
                "ccp4_version",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pdbID": "1abc",
                "status": "ok",
                "no_metals": "false",
                "metal_site_limit_exceeded": "false",
                "n_metals": "1",
                "alchemy_version": "1.0",
                "alchemy_commit": "abc",
                "gemmi_version": "0.7",
                "ccp4_version": "9",
                "analysis_config_id": ANALYSIS_CONFIG_ID,
            }
        )
    reference_dir = tmp_path / "reference"
    cs.finalize_database_confidence(
        str(input_path),
        str(tmp_path / "scores.csv"),
        str(reference_dir),
        str(manifest_path),
    )
    metadata = json.loads(
        (reference_dir / cs.REFERENCE_METADATA_FILE).read_text(encoding="utf-8")
    )
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert metadata["confidence_inputs_sha256"] == input_hash
    assert metadata["cohort_id"] == "alchemy-cohort-" + input_hash[:20]
    assert metadata["input_entry_count"] == 1
    assert metadata["scorable_entry_count"] == 1
    assert metadata["source_entry_count"] == 1
    assert metadata["software_versions"]["alchemy_version"] == ["1.0"]
    assert metadata["analysis_config_id"] == ANALYSIS_CONFIG_ID


def test_finalize_removes_stale_completion_marker_on_failure(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    marker = reference_dir / cs.REFERENCE_METADATA_FILE
    marker.write_text("stale", encoding="utf-8")
    input_path = _write_input_csv(
        tmp_path / "empty.csv",
        [
            _input_row(
                rszd_abs="",
                geometry_rms_zbond="",
                confidence_inputs_status="unscorable",
            )
        ],
    )
    with pytest.raises(ValueError, match="no evidence"):
        cs.finalize_database_confidence(
            input_path, str(tmp_path / "scores.csv"), str(reference_dir)
        )
    assert not marker.exists()


def test_validate_scored_reference_checks_reference_and_cohort_ids(
    tmp_path: Path,
) -> None:
    reference = _reference()
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["confidence_reference_version", "confidence_cohort_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "confidence_reference_version": reference.reference_id,
                "confidence_cohort_id": reference.cohort_id,
            }
        )
    cs.validate_scored_reference(str(path), reference)

    path.write_text(
        "confidence_reference_version,confidence_cohort_id\n"
        "alchemy-confidence-other,alchemy-cohort-test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different database reference"):
        cs.validate_scored_reference(str(path), reference)


def test_main_finalize_and_score_commands(tmp_path: Path) -> None:
    input_path = _write_input_csv(tmp_path / "inputs.csv", [_input_row()])
    reference_dir = tmp_path / "reference"
    finalized = tmp_path / "finalized.csv"
    assert (
        cs.main(
            [
                "finalize",
                "--input",
                input_path,
                "--output",
                str(finalized),
                "--reference-dir",
                str(reference_dir),
            ]
        )
        == 0
    )
    rescored = tmp_path / "rescored.csv"
    assert (
        cs.main(
            [
                "score",
                "--input",
                input_path,
                "--output",
                str(rescored),
                "--reference-dir",
                str(reference_dir),
            ]
        )
        == 0
    )
    assert _read_csv_rows(finalized)[1] == _read_csv_rows(rescored)[1]


def test_reference_metadata_field_vocabulary_covers_emitted_metadata(
    tmp_path: Path,
) -> None:
    reference = cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    assert set(reference.metadata) <= cs.REFERENCE_METADATA_FIELDS


def test_all_input_and_analysis_columns_are_unique() -> None:
    assert len(cs.CONFIDENCE_INPUT_COLUMNS) == len(set(cs.CONFIDENCE_INPUT_COLUMNS))
    assert len(cs.ANALYSIS_COLUMNS) == len(set(cs.ANALYSIS_COLUMNS))
    assert not set(cs.CONFIDENCE_INPUT_COLUMNS) & set(cs.ANALYSIS_COLUMNS)


def test_scientific_boolean_columns_are_canonicalized_in_csv(tmp_path: Path) -> None:
    row = _input_row(
        density_is_shared="TRUE", density_saturated="TRUE", context_warning="FALSE"
    )
    input_path = _write_input_csv(tmp_path / "inputs.csv", [row])
    output_path = tmp_path / "output.csv"
    cs.score_file_against_reference(input_path, str(output_path), _reference())
    output = _read_csv_rows(output_path)[1][0]
    assert output["density_is_shared"] == "true"
    assert output["density_saturated"] == "true"
    assert output["context_warning"] == "false"


def test_reference_data_identity_is_part_of_reference_policy(tmp_path: Path) -> None:
    reference = cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    assert reference.metadata["reference_data_id"]


def test_analysis_configuration_identity_is_part_of_reference_policy(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    cs.write_reference(str(reference_dir), {1.0: 1}, {1.0: 1}, 1)
    metadata_path = reference_dir / cs.REFERENCE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["analysis_config_id"] = "alchemy-analysis-config-incompatible"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="analysis_config_id"):
        cs.load_reference(str(reference_dir))


def test_distribution_file_has_no_nonfinite_values(tmp_path: Path) -> None:
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    text = (tmp_path / cs.REFERENCE_DISTRIBUTION_FILE).read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    assert "inf" not in text.lower()


def test_confidence_input_status_vocabulary_matches_preparation() -> None:
    rows = [
        cs.prepare_confidence_inputs([_stats_row()], [_bond_row()])[0],
        cs.prepare_confidence_inputs([_stats_row()], [])[0],
        cs.prepare_confidence_inputs([_stats_row(zdm=None)], [_bond_row()])[0],
        cs.prepare_confidence_inputs([_stats_row(zdm=None)], [])[0],
    ]
    assert {row["confidence_inputs_status"] for row in rows} == (
        cs.CONFIDENCE_INPUT_STATUSES
    )


def test_database_reference_cohort_id_uses_exact_input_hash(tmp_path: Path) -> None:
    input_path = Path(_write_input_csv(tmp_path / "inputs.csv", [_input_row()]))
    reference_dir = tmp_path / "reference"
    cs.finalize_database_confidence(
        str(input_path), str(tmp_path / "scores.csv"), str(reference_dir)
    )
    reference = cs.load_reference(str(reference_dir))
    expected = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert reference.cohort_id == "alchemy-cohort-" + expected[:20]


def test_reference_output_is_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    a = cs.write_reference(str(first), {3.0: 2, 1.0: 1}, {2.0: 1, 0.5: 2}, 3)
    b = cs.write_reference(str(second), {1.0: 1, 3.0: 2}, {0.5: 2, 2.0: 1}, 3)
    assert a.reference_id == b.reference_id
    assert (first / cs.REFERENCE_DISTRIBUTION_FILE).read_bytes() == (
        second / cs.REFERENCE_DISTRIBUTION_FILE
    ).read_bytes()


def test_score_output_never_serializes_nan_or_infinity() -> None:
    scored = cs.score_against_reference(
        [
            _input_row(
                rszd_abs="",
                geometry_rms_zbond="",
                confidence_inputs_status="unscorable",
            )
        ],
        _reference(),
    )[0]
    assert scored["density_score"] == ""
    assert scored["geometry_score"] == ""
    assert scored["alchemy_score"] == ""
    assert scored["alchemy_level"] == "INCOMPLETE"


def test_metadata_completion_marker_is_written_last(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    cs.write_reference(str(reference_dir), {1.0: 1}, {0.5: 1}, 1)
    assert (reference_dir / cs.REFERENCE_DISTRIBUTION_FILE).is_file()
    assert (reference_dir / cs.REFERENCE_METADATA_FILE).is_file()
    assert not list(reference_dir.glob("*.tmp"))


def test_manifest_provenance_counts_no_metal_and_limited_entries(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pdbID",
                "status",
                "no_metals",
                "metal_site_limit_exceeded",
                "n_metals",
                "analysis_config_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "pdbID": "1aaa",
                    "status": "ok",
                    "no_metals": "true",
                    "metal_site_limit_exceeded": "false",
                    "n_metals": "0",
                    "analysis_config_id": ANALYSIS_CONFIG_ID,
                },
                {
                    "pdbID": "2bbb",
                    "status": "partial",
                    "no_metals": "false",
                    "metal_site_limit_exceeded": "true",
                    "n_metals": "101",
                    "analysis_config_id": ANALYSIS_CONFIG_ID,
                },
            ]
        )
    input_path = _write_input_csv(
        tmp_path / "inputs.csv",
        [_input_row(pdbID="2bbb", metal_site_id="2bbb:1")],
    )
    reference_dir = tmp_path / "reference"
    cs.finalize_database_confidence(
        input_path,
        str(tmp_path / "scores.csv"),
        str(reference_dir),
        str(manifest),
    )
    provenance = json.loads(
        (reference_dir / cs.REFERENCE_METADATA_FILE).read_text(encoding="utf-8")
    )
    assert provenance["no_metals_entry_count"] == 1
    assert provenance["metal_site_limit_exceeded_entry_count"] == 1
    assert provenance["metal_bearing_entry_count"] == 1


def test_reference_distribution_constructor_rejects_bad_shapes() -> None:
    metadata = {"input_row_count": 1}
    with pytest.raises(ValueError, match="differ in size"):
        cs.ConfidenceReference([1.0], [], [], [], metadata)
    with pytest.raises(ValueError, match="not increasing"):
        cs.ConfidenceReference([2.0, 1.0], [1, 1], [], [], metadata)
    with pytest.raises(ValueError, match="invalid count"):
        cs.ConfidenceReference([1.0], [0], [], [], metadata)


def test_main_reports_invalid_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = _write_input_csv(tmp_path / "inputs.csv", [_input_row()])
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    assert (
        cs.main(
            [
                "score",
                "--input",
                input_path,
                "--output",
                str(tmp_path / "scores.csv"),
                "--reference-dir",
                str(reference_dir),
            ]
        )
        == 1
    )
    assert "confidence score failed" in capsys.readouterr().err


def test_no_old_weighted_formula_fields_remain_in_public_schema() -> None:
    old = {
        "density_severity",
        "geometry_severity",
        "density_penalty_fraction",
        "geometry_penalty_fraction",
        "interaction_penalty_fraction",
        "confidence_score",
        "confidence_percentile",
    }
    assert not old & set(cs.ANALYSIS_COLUMNS)
    assert "geometry_rms_zbond" in cs.CONFIDENCE_INPUT_COLUMNS
    assert "alchemy_level" in cs.ANALYSIS_COLUMNS


def test_cli_output_paths_are_created(tmp_path: Path) -> None:
    input_path = _write_input_csv(tmp_path / "inputs.csv", [_input_row()])
    output_path = tmp_path / "nested" / "scores.csv"
    cs.score_file_against_reference(input_path, str(output_path), _reference())
    assert output_path.is_file()
    assert os.path.getsize(output_path) > 0
