"""Behavioral tests for the final three-level confidence method."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import confidence_oracle as oracle
import helpers
import pytest
from helpers import approx

import analysis_config
import confidence_score as cs
import reference_data
from confidence_score import cli as cs_cli, schema as confidence_schema
from coordination.schema import STATS_EXTRA_COLUMNS
from output_rows import MetalStatsRow


def test_bundled_reference_matches_the_archived_manuscript_bytes() -> None:
    directory = Path(helpers.SRC_DIR) / "confidence_score" / "confidence_reference"
    expected = {
        "component_distributions.csv": "92ca1c704db005172eee9111d1e0d7cad907d736bf20e7b54c160814a1314b4e",
        "metadata.json": "b62ceaf812d4512c77740290d5b7dcf9d986df1d385d711e6492c4e0ba9c0b4e",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest


def test_bundled_reference_loads_under_runtime_verification() -> None:
    """The shipped bytes still satisfy every check ``load_reference`` applies.

    ``reference_id`` is a digest over the scoring policy and every parsed
    distribution value and count, so this proves the bundled distributions,
    metadata, and current code agree, not only that the bytes are unchanged.
    """
    directory = Path(helpers.SRC_DIR) / "confidence_score" / "confidence_reference"
    reference = cs.load_reference(str(directory))
    assert reference.reference_id == "alchemy-confidence-8ba6808c816791ffbb87"
    assert reference.metadata["cohort_id"] == "alchemy-cohort-2e97cf013eefa9d8e0b4"
    assert reference.metadata["input_row_count"] == 330978
    assert reference.metadata["input_entry_count"] == 76954
    assert reference.density_reference_size == 330887
    assert reference.geometry_reference_size == 275870


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
    row: dict[str, str] = dict.fromkeys(cs.IDENTITY_COLUMNS, "")
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
    row: dict[str, str] = dict(SITE)
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
    row: dict[str, str] = dict.fromkeys(cs.CONFIDENCE_INPUT_COLUMNS, "")
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
    return helpers.read_csv(path)[0], helpers.read_csv_dicts(path)


def _reference() -> cs.ConfidenceReference:
    return cs.ConfidenceReference(
        density_values=[1.0, 3.0, 6.0],
        density_counts=[1, 2, 1],
        geometry_values=[0.5, 1.0, 2.0],
        geometry_counts=[1, 2, 1],
        metadata={
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
    assert result.density_level == density
    assert result.geometry_level == geometry
    assert result.alchemy_level == overall
    assert result.evidence_basis == "density_and_geometry"


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
    assert result.evidence_basis == basis
    assert result.alchemy_level == overall


def test_verdict_reasons_distinguish_all_suspect_routes() -> None:
    assert cs.classify_site(7.0, 2.5).verdict_reason == ("density_and_geometry_suspect")
    assert cs.classify_site(7.0, 0.5).verdict_reason == "density_suspect"
    assert cs.classify_site(1.0, 2.5).verdict_reason == "geometry_suspect"
    assert cs.classify_site(4.0, 1.5).verdict_reason == "review_plus_review"


def test_classification_alone_leaves_every_ranking_score_blank() -> None:
    verdict = cs.classify_site(4.0, 1.5)
    assert math.isnan(verdict.density_score)
    assert math.isnan(verdict.geometry_score)
    assert math.isnan(verdict.alchemy_score)


def test_verdict_row_uses_only_analysis_column_names() -> None:
    row = cs.score_site(1.0, 0.5, _reference()).as_row()
    assert set(row) < set(cs.ANALYSIS_COLUMNS)
    assert row["alchemy_level"] == "PASS"
    assert row["alchemy_score"] == approx(87.5)


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
    assert both.density_score == approx(87.5)
    assert both.geometry_score == approx(87.5)
    assert both.alchemy_score == approx(87.5)

    density_only = cs.score_site(3.0, math.nan, reference)
    assert density_only.alchemy_score == approx(50.0)
    assert math.isnan(density_only.geometry_score)


def test_edstats_saturation_receives_zero_density_support() -> None:
    result = cs.score_site(99.9, math.nan, _reference())
    assert result.density_score == 0.0
    assert result.alchemy_score == 0.0


def test_ranking_score_does_not_define_review_plus_review_verdict() -> None:
    result = cs.score_site(4.0, 1.5, _reference())
    assert result.alchemy_level == "SUSPECT"
    assert result.alchemy_score > 0.0


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
    assert result.geometry_level == "SUSPECT"
    assert result.alchemy_level == "SUSPECT"


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
    row = {**_input_row(), **dict.fromkeys(cs.ANALYSIS_COLUMNS, "")}
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
        cs_cli.main(
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
        cs_cli.main(
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
    # The scoring contract is every field the loader compares, and nothing
    # written is left out of either vocabulary.
    assert set(cs.SCORING_METADATA_FIELDS) <= set(reference.metadata)
    assert set(reference.metadata) - set(cs.SCORING_METADATA_FIELDS) <= set(
        cs.REFERENCE_PROVENANCE_FIELDS
    )


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
        cs.ConfidenceReference(
            density_values=[1.0],
            density_counts=[],
            geometry_values=[],
            geometry_counts=[],
            metadata=metadata,
        )
    with pytest.raises(ValueError, match="not increasing"):
        cs.ConfidenceReference(
            density_values=[2.0, 1.0],
            density_counts=[1, 1],
            geometry_values=[],
            geometry_counts=[],
            metadata=metadata,
        )
    with pytest.raises(ValueError, match="invalid count"):
        cs.ConfidenceReference(
            density_values=[1.0],
            density_counts=[0],
            geometry_values=[],
            geometry_counts=[],
            metadata=metadata,
        )


def test_main_reports_invalid_reference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = _write_input_csv(tmp_path / "inputs.csv", [_input_row()])
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    assert (
        cs_cli.main(
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


def test_the_scoring_policy_under_test_is_the_shipped_one(tmp_path: Path) -> None:
    """The raw thresholds ``confidence_oracle`` re-implements are Alchemy's own.

    ``assert_policy_was_applied`` is an independent oracle only while the two
    copies agree.
    """
    assert oracle.DENSITY_THRESHOLDS == (
        cs.DENSITY_REVIEW_THRESHOLD,
        cs.DENSITY_SUSPECT_THRESHOLD,
    )
    assert oracle.GEOMETRY_THRESHOLDS == (
        cs.GEOMETRY_REVIEW_THRESHOLD,
        cs.GEOMETRY_SUSPECT_THRESHOLD,
    )

    published = oracle.reference_metadata(oracle.frozen_reference(tmp_path / "policy"))
    assert published["density_thresholds"] == {"review": 3.0, "suspect": 6.0}
    assert published["geometry_thresholds"] == {"review": 1.0, "suspect": 2.0}
    assert published["geometry_statistic"] == "rms_finite_score_eligible_zbond"
    assert published["overall_rule"] == "any_suspect_or_review_plus_review"
    assert published["support_score_method"] == ("reverse_average_rank_empirical_cdf")


@pytest.mark.parametrize(
    ("value", "decimal_places", "expected"),
    [
        # Zero decimal places must not strip an integer's own zeros.
        (100.0, 0, "100"),
        (0.0, 0, "0"),
        (100.0, 6, "100"),
        (2.0, 6, "2"),
        (1.5, 6, "1.5"),
        (123.456000, 6, "123.456"),
        (0.1234567, 6, "0.123457"),
        (1e-9, 6, "0"),
        # Negative zero keeps its sign; the published rows were written with it.
        (-0.0, 6, "-0"),
        (-0.0, 0, "-0"),
        (math.nan, 6, ""),
        (math.inf, 6, ""),
        (-math.inf, 6, ""),
    ],
)
def test_format_decimal_strips_only_fractional_zeros(
    value: float, decimal_places: int, expected: str
) -> None:
    assert confidence_schema.format_decimal(value, decimal_places) == expected


def test_format_decimal_defaults_to_the_published_score_precision() -> None:
    assert confidence_schema.format_decimal(0.1234567) == "0.123457"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("  yes  ", True),
        (True, True),
        (False, False),
        ("", False),
        (None, False),
        # Unrecognized spellings read as false rather than raising.
        (1.0, False),
        ("1.0", False),
        (1, True),
        ("no", False),
        ("garbage", False),
    ],
)
def test_parse_csv_bool_reads_only_the_written_spellings(
    value: object, expected: bool
) -> None:
    assert confidence_schema.parse_csv_bool(value) is expected


def test_canonical_values_round_to_their_serialized_precision() -> None:
    assert cs.canonical_support_score(1.23456749) == float("1.234567")
    assert cs.canonical_support_score(100.0) == 100.0
    assert cs.canonical_metric(0.1234567890123456) == float("0.123456789012")
    assert cs.canonical_metric(3.0) == 3.0


def test_canonical_values_pass_non_finite_input_through() -> None:
    assert math.isnan(cs.canonical_support_score(math.nan))
    assert math.isnan(cs.canonical_metric(math.nan))
    assert cs.canonical_support_score(math.inf) == math.inf
    assert cs.canonical_metric(-math.inf) == -math.inf


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("density_saturated", "True", "true"),
        ("density_saturated", "FALSE", "false"),
        ("context_warning", "  True  ", "true"),
        ("density_is_shared", True, "true"),
        ("density_is_shared", False, "false"),
        # A non-boolean column is left to ``scientific_csv_value``.
        ("metal_resname", "True", "True"),
        ("rszd", math.nan, ""),
        ("rszd", 1.5, 1.5),
        # Text the boolean columns never carry is passed through unchanged.
        ("density_saturated", "maybe", "maybe"),
    ],
)
def test_confidence_csv_value_only_lowercases_boolean_columns(
    column: str, value: object, expected: object
) -> None:
    assert confidence_schema.confidence_csv_value(column, value) == expected


def test_site_key_uses_the_modern_scheme_when_the_site_id_is_present() -> None:
    row = {
        "pdbID": "1abc",
        "metal_site_id": " site-1 ",
        "metal_model_index": "1",
        "metal_atom_index": "7",
    }
    assert confidence_schema.site_key(row) == ("1abc", "site-1")


def test_site_key_falls_back_to_the_legacy_index_columns() -> None:
    row = {
        "pdbID": "1abc",
        "metal_site_id": "",
        "metal_model_index": "1",
        "metal_chain_index": "2",
        "metal_residue_index": "3",
        "metal_atom_index": "4",
    }
    assert confidence_schema.site_key(row) == ("1abc", "1", "2", "3", "4")


def test_site_key_of_a_row_without_identity_columns_is_all_blank() -> None:
    assert confidence_schema.site_key({}) == ("", "", "", "", "")


def test_require_columns_names_every_missing_column() -> None:
    with pytest.raises(ValueError) as excinfo:
        confidence_schema.require_columns(
            ["pdbID", "rszd"], ["pdbID", "rszd_abs", "geometry_rms_zbond"], "test table"
        )
    message = str(excinfo.value)
    assert message.startswith("test table is missing required columns: ")
    assert "rszd_abs, geometry_rms_zbond" in message


def test_require_columns_accepts_a_complete_header() -> None:
    confidence_schema.require_columns(["pdbID", "rszd"], ["pdbID"], "t")


def test_require_columns_treats_a_missing_header_as_missing_everything() -> None:
    with pytest.raises(ValueError, match="t is missing required columns: pdbID"):
        confidence_schema.require_columns(None, ["pdbID"], "t")


@pytest.mark.parametrize(
    ("rszd_abs", "expected"),
    [
        (99.9, True),
        (100.0, True),
        (99.8, False),
        (0.0, False),
        (math.nan, False),
        (math.inf, False),
    ],
)
def test_is_density_saturated_covers_the_edstats_ceiling(
    rszd_abs: float, expected: bool
) -> None:
    assert confidence_schema.is_density_saturated(rszd_abs) is expected


def test_prepared_input_status_matches_the_scored_evidence_basis() -> None:
    """Preparation and scoring must agree on which evidence a site has.

    ``confidence_inputs_status`` is decided by ``prepare_confidence_inputs``
    from the raw metrics, while ``evidence_basis`` is decided independently by
    ``classify_site`` from the parsed ones; a drift between the two modules
    would silently publish rows whose status contradicts their verdict.
    """
    prepared = [
        cs.prepare_confidence_inputs([_stats_row()], [_bond_row()])[0],
        cs.prepare_confidence_inputs([_stats_row()], [])[0],
        cs.prepare_confidence_inputs([_stats_row(zdm=None)], [_bond_row()])[0],
        cs.prepare_confidence_inputs([_stats_row(zdm=None)], [])[0],
    ]
    scored = cs.classify_without_reference(prepared)
    assert {
        row["confidence_inputs_status"]: row["evidence_basis"] for row in scored
    } == {
        "complete": "density_and_geometry",
        "density_only": "density_only",
        "geometry_only": "geometry_only",
        "unscorable": "no_assessable_evidence",
    }


@pytest.mark.parametrize("input_row_count", [None, [4], "4", 4.0, True, -1])
def test_reference_rejects_a_non_count_input_row_count(
    input_row_count: object,
) -> None:
    """Bad metadata must fail with ValueError, the contract callers catch.

    ``cli.py`` and ``driver/confidence.py`` guard reference loading with
    ``(OSError, ValueError)``, so a ``null`` or list in ``metadata.json`` must
    not escape as a ``TypeError``.
    """
    with pytest.raises(ValueError, match="input row count is invalid"):
        cs.ConfidenceReference(
            density_values=[1.0],
            density_counts=[1],
            geometry_values=[],
            geometry_counts=[],
            metadata={"input_row_count": input_row_count},
        )


@pytest.mark.parametrize("field", ["reference_id", "cohort_id"])
def test_reference_rejects_a_non_string_identifier(field: str) -> None:
    with pytest.raises(ValueError, match="is invalid"):
        cs.ConfidenceReference(
            density_values=[1.0],
            density_counts=[1],
            geometry_values=[],
            geometry_counts=[],
            metadata={"input_row_count": 1, field: None},
        )


def test_reference_metadata_is_a_read_only_copy() -> None:
    fields: dict[str, object] = {"input_row_count": 4, "cohort_id": "c"}
    reference = cs.ConfidenceReference(
        density_values=[1.0],
        density_counts=[1],
        geometry_values=[],
        geometry_counts=[],
        metadata=fields,
    )
    assert reference.metadata["cohort_id"] == "c"
    fields["cohort_id"] = "mutated"
    assert reference.metadata["cohort_id"] == "c"
    with pytest.raises(TypeError):
        reference.metadata["cohort_id"] = "mutated"  # type: ignore[index]


def test_reference_reports_its_distinct_value_counts() -> None:
    reference = _reference()
    assert reference.density_distinct_value_count == 3
    assert reference.geometry_distinct_value_count == 3
    assert reference.density_distinct_value_count == len(reference.density.values)
    assert reference.geometry.distinct_value_count == len(reference.geometry.values)


def test_empirical_distribution_from_counts_sorts_and_validates() -> None:
    from confidence_score.scoring import EmpiricalDistribution

    distribution = EmpiricalDistribution.from_counts({3.0: 1, 1.0: 2, 6.0: 1})
    assert distribution.values == (1.0, 3.0, 6.0)
    assert distribution.counts == (2, 1, 1)
    assert distribution.size == 4
    assert distribution.distinct_value_count == 3
    assert distribution.support_score(1.0) == approx(75.0)

    with pytest.raises(ValueError, match="invalid value"):
        EmpiricalDistribution.from_counts({-1.0: 1})
    with pytest.raises(ValueError, match="invalid count"):
        EmpiricalDistribution.from_counts({1.0: 0})


def test_reference_from_counts_matches_the_direct_constructor() -> None:
    metadata = {"input_row_count": 4}
    from_counts = cs.ConfidenceReference.from_counts(
        {3.0: 2, 1.0: 1, 6.0: 1}, {1.0: 2, 0.5: 1, 2.0: 1}, metadata
    )
    direct = cs.ConfidenceReference(
        density_values=[1.0, 3.0, 6.0],
        density_counts=[1, 2, 1],
        geometry_values=[0.5, 1.0, 2.0],
        geometry_counts=[1, 2, 1],
        metadata=metadata,
    )
    assert from_counts.density.values == direct.density.values
    assert from_counts.density.counts == direct.density.counts
    assert from_counts.geometry.values == direct.geometry.values
    assert from_counts.geometry.counts == direct.geometry.counts


def _write_manifest(path: Path, analysis_config_id: str = ANALYSIS_CONFIG_ID) -> str:
    """Write a one-entry completed run manifest for CLI provenance tests."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
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
                "analysis_config_id": analysis_config_id,
                "alchemy_version": "1.0",
                "alchemy_commit": "abc",
                "gemmi_version": "0.7",
                "ccp4_version": "9",
            }
        )
    return str(path)


def test_main_finalize_with_a_manifest_records_cohort_provenance(
    tmp_path: Path,
) -> None:
    """``--manifest`` is the documented recovery path, so exercise it end to end."""
    input_path = _write_input_csv(tmp_path / "inputs.csv", [_input_row()])
    manifest_path = _write_manifest(tmp_path / "manifest.csv")
    reference_dir = tmp_path / "reference"
    assert (
        cs_cli.main(
            [
                "finalize",
                "--input",
                input_path,
                "--output",
                str(tmp_path / "scores.csv"),
                "--reference-dir",
                str(reference_dir),
                "--manifest",
                manifest_path,
            ]
        )
        == 0
    )
    metadata = json.loads(
        (reference_dir / cs.REFERENCE_METADATA_FILE).read_text(encoding="utf-8")
    )
    assert metadata["source_manifest_file"] == "manifest.csv"
    assert (
        metadata["source_manifest_sha256"]
        == hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    )
    assert metadata["source_entry_count"] == 1
    assert metadata["manifest_status_counts"] == {"ok": 1}
    assert metadata["software_versions"]["alchemy_version"] == ["1.0"]
    assert metadata["analysis_config_id"] == ANALYSIS_CONFIG_ID


def test_main_rejects_a_manifest_from_another_analysis_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A manifest from different settings must not be recorded as this cohort."""
    input_path = _write_input_csv(tmp_path / "inputs.csv", [_input_row()])
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv", analysis_config_id="alchemy-config-other"
    )
    reference_dir = tmp_path / "reference"
    assert (
        cs_cli.main(
            [
                "finalize",
                "--input",
                input_path,
                "--output",
                str(tmp_path / "scores.csv"),
                "--reference-dir",
                str(reference_dir),
                "--manifest",
                manifest_path,
            ]
        )
        == 1
    )
    assert (
        "source manifest analysis configuration identity is incompatible"
        in capsys.readouterr().err
    )


def test_main_reports_an_oversized_csv_field_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``csv.Error`` is not a ``ValueError``, so the handler must name it."""
    row = _input_row(context_warning_reasons="x" * (csv.field_size_limit() + 1))
    input_path = tmp_path / "inputs.csv"
    with open(input_path, "w", newline="", encoding="utf-8") as handle:
        handle.write(",".join(cs.CONFIDENCE_INPUT_COLUMNS) + "\n")
        handle.write(",".join(row[column] for column in cs.CONFIDENCE_INPUT_COLUMNS))
        handle.write("\n")
    assert (
        cs_cli.main(
            [
                "finalize",
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "scores.csv"),
                "--reference-dir",
                str(tmp_path / "reference"),
            ]
        )
        == 1
    )
    assert "confidence finalize failed:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("resname", "parent_type", "expected"),
    [
        # HEM is a real entry in the frozen cofactor catalog.
        ("HEM", "other", "cofactor"),
        ("hem", "other", "cofactor"),
        ("ZN", "ion", "metal"),
        ("XXX", "other", ""),
    ],
)
def test_an_orphan_site_classifies_its_category_from_the_bond_row_alone(
    resname: str, parent_type: str, expected: str
) -> None:
    """The orphan path has only bond-row fields, so it has its own rule.

    ``edstats_statistics.classify_residue`` is unavailable here: no density row
    was joined, so the catalog name and ``parent_type`` are all there is. A
    site that is neither a catalog cofactor nor an ion therefore publishes a
    blank category, which is retained as published.
    """
    orphan = cs.prepare_confidence_inputs(
        [], [_bond_row(metal_resname=resname, parent_type=parent_type)]
    )[0]
    assert orphan["category"] == expected
    assert orphan["coordinate_mapping_status"] == "density_row_unavailable"
    assert orphan["selected_metal_site_status"] == "selected_without_density_row"
    # No density row was joined, so the density identity block stays blank.
    assert [
        orphan[column]
        for column in (
            "density_observation_id",
            "density_scope",
            "density_shared_site_count",
            "density_is_shared",
        )
    ] == ["", "", "", ""]


@pytest.mark.parametrize(
    ("bonds", "expected"),
    [
        ([("True", "declared"), ("False", "inferred")], "declared_and_inferred"),
        ([("True", "declared")], "declared_only"),
        ([("False", "inferred")], "inferred_only"),
        ([("False", "unassigned")], "none"),
        ([], "none"),
    ],
)
def test_geometry_contact_basis_describes_the_scored_contacts(
    bonds: list[tuple[str, str]], expected: str
) -> None:
    """The published basis counts scored contacts, not assigned ones."""
    rows = [
        _bond_row(
            contact_id=f"1abc:c{index}",
            declared_connection=declared,
            coordination_status=status,
        )
        for index, (declared, status) in enumerate(bonds)
    ]
    prepared = cs.prepare_confidence_inputs([_stats_row()], rows)[0]
    assert prepared["geometry_contact_basis"] == expected


def test_an_unscored_contact_leaves_no_basis_even_when_declared() -> None:
    """A declared contact with no z-score contributes no provenance."""
    prepared = cs.prepare_confidence_inputs(
        [_stats_row()],
        [
            _bond_row(
                zscore=None, declared_connection="True", coordination_status="declared"
            )
        ],
    )[0]
    assert prepared["geometry_contact_basis"] == "none"
    assert prepared["declared_contact_count"] == 1
    assert prepared["declared_scored_bond_count"] == 0


def test_missing_reason_vocabulary_keeps_its_published_spelling() -> None:
    """Pin the pipe-joined reasons, which mix three vocabularies."""
    no_contacts = cs.prepare_confidence_inputs([_stats_row(zdm=None)], [])[0]
    assert (
        no_contacts["confidence_inputs_missing_reasons"]
        == "rszd_unavailable|no_assigned_contacts"
    )
    invalid = cs.prepare_confidence_inputs(
        [_stats_row(metal_coordinates_valid="false")],
        [_bond_row(covered=False)],
    )[0]
    assert (
        invalid["confidence_inputs_missing_reasons"]
        == "non_finite_metal_coordinates|no_geometry_reference|partial_geometry_coverage"
    )
    partial = cs.prepare_confidence_inputs(
        [_stats_row()], [_bond_row(), _bond_row(contact_id="1abc:c2", covered=False)]
    )[0]
    assert partial["confidence_inputs_missing_reasons"] == "partial_geometry_coverage"
    unscored = cs.prepare_confidence_inputs([_stats_row()], [_bond_row(zscore=None)])[0]
    assert (
        unscored["confidence_inputs_missing_reasons"]
        == "zbond_unavailable_for_reference"
    )
    orphan = cs.prepare_confidence_inputs([], [_bond_row()])[0]
    assert (
        orphan["confidence_inputs_missing_reasons"]
        == "rszd_unavailable|density_row_unavailable"
    )


def test_a_placeholder_keeps_its_published_reasons_and_sentinel_index() -> None:
    """The placeholder's reasons, warning, and sentinel index are published."""
    completed = cs.complete_confidence_site_count([], "1abc", 2, "bond_stage_failure")
    assert [row["metal_atom_index"] for row in completed] == [
        "unresolved-1",
        "unresolved-2",
    ]
    for row in completed:
        assert row["context_warning"] is True
        assert row["context_warning_reasons"] == "site_evidence_unavailable"
        assert row["selected_metal_site_status"] == "selected_site_unresolved"
        assert row["confidence_inputs_missing_reasons"] == (
            "rszd_unavailable|site_identity_unavailable|"
            "site_evidence_unavailable|bond_stage_failure"
        )
    assert set(completed[0]) == set(cs.CONFIDENCE_INPUT_COLUMNS)
    assert (
        cs.complete_confidence_site_count([], "1abc", 1)[0][
            "confidence_inputs_missing_reasons"
        ]
        == "rszd_unavailable|site_identity_unavailable|site_evidence_unavailable"
    )


def test_every_prepared_row_carries_the_full_schema_in_order() -> None:
    """All three producers build rows through one builder, so order is fixed."""
    rows = cs.prepare_confidence_inputs([_stats_row()], [_bond_row()])
    rows += cs.prepare_confidence_inputs([], [_bond_row(pdb_id="9zzz")])
    rows = cs.complete_confidence_site_count(rows, "1abc", len(rows) + 1)
    for row in rows:
        assert list(row) == list(cs.CONFIDENCE_INPUT_COLUMNS)


def test_a_saturated_site_is_flagged_through_the_shared_predicate() -> None:
    """``density_saturated`` and the zero-support rule share one predicate."""
    saturated = cs.prepare_confidence_inputs([_stats_row(zdm=99.9)], [])[0]
    assert saturated["density_saturated"] is True
    assert confidence_schema.is_density_saturated(99.9)
    ordinary = cs.prepare_confidence_inputs([_stats_row(zdm=-3.0)], [])[0]
    assert ordinary["density_saturated"] is False
    negative_saturated = cs.prepare_confidence_inputs([_stats_row(zdm=-99.9)], [])[0]
    assert negative_saturated["density_saturated"] is True


def test_load_reference_rejects_metadata_that_is_not_an_object(tmp_path: Path) -> None:
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    (tmp_path / cs.REFERENCE_METADATA_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata is not a JSON object"):
        cs.load_reference(str(tmp_path))


@pytest.mark.parametrize("distribution_file", [["a"], "../x.csv", "sub/x.csv", ""])
def test_load_reference_rejects_a_distribution_file_that_is_not_a_sibling(
    tmp_path: Path, distribution_file: object
) -> None:
    """A hand-edited name must fail as ``ValueError``, not read another path."""
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    metadata_path = tmp_path / cs.REFERENCE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["distribution_file"] = distribution_file
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="distribution file name is invalid"):
        cs.load_reference(str(tmp_path))


@pytest.mark.parametrize("input_row_count", [None, "1", [1], True, -1])
def test_load_reference_rejects_a_metadata_row_count_that_is_not_a_count(
    tmp_path: Path, input_row_count: object
) -> None:
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    metadata_path = tmp_path / cs.REFERENCE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_row_count"] = input_row_count
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="input row count is invalid"):
        cs.load_reference(str(tmp_path))


def test_write_reference_rejects_a_cohort_smaller_than_its_evidence(
    tmp_path: Path,
) -> None:
    """The writer refuses what the loader would, and leaves nothing behind."""
    with pytest.raises(ValueError, match="input row count is invalid"):
        cs.write_reference(str(tmp_path / "reference"), {1.0: 2}, {0.5: 1}, 1)
    assert not (tmp_path / "reference").exists()


def test_write_reference_replaces_a_stale_completion_marker(tmp_path: Path) -> None:
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    marker = tmp_path / cs.REFERENCE_METADATA_FILE
    marker.write_text("stale", encoding="utf-8")
    cs.write_reference(str(tmp_path), {2.0: 1}, {0.5: 1}, 1)
    assert json.loads(marker.read_text(encoding="utf-8"))["input_row_count"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_load_reference_errors_name_the_file_row_and_offending_cell(
    tmp_path: Path,
) -> None:
    """A distribution failure must say where in which file it was found."""
    cs.write_reference(str(tmp_path), {1.0: 1}, {0.5: 1}, 1)
    distribution = tmp_path / cs.REFERENCE_DISTRIBUTION_FILE
    tampering = {
        "component,value,count\ndensity,1,1\nbonds,0.5,1\n": (
            "unknown component: 'bonds' at",
            "row 3",
        ),
        "component,value,count\ndensity,1,1\ndensity,1,2\n": (
            "duplicate value: '1' for density at",
            "row 3",
        ),
        "component,value,count\ndensity,1,x\n": (
            "non-integer count: 'x' for density '1' at",
            "row 2",
        ),
        "component,value\ndensity,1\n": ("invalid columns", "component, value"),
    }
    for text, (message, location) in tampering.items():
        distribution.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            cs.load_reference(str(tmp_path))
        assert message in str(excinfo.value)
        assert location in str(excinfo.value)
        assert str(distribution) in str(excinfo.value)


def test_finalize_requires_the_entry_identifier_column(tmp_path: Path) -> None:
    """Without ``pdbID`` the entry counts would silently be recorded as zero."""
    columns = [column for column in cs.CONFIDENCE_INPUT_COLUMNS if column != "pdbID"]
    row = _input_row()
    input_path = _write_input_csv(
        tmp_path / "inputs.csv",
        [{column: row[column] for column in columns}],
        columns,
    )
    with pytest.raises(ValueError, match="missing required columns: pdbID"):
        cs.finalize_database_confidence(
            input_path, str(tmp_path / "scores.csv"), str(tmp_path / "reference")
        )


def test_finalize_reports_its_counts_by_name(tmp_path: Path) -> None:
    input_path = _write_input_csv(
        tmp_path / "inputs.csv",
        [
            _input_row(pdbID="1aaa", metal_site_id="1aaa:1"),
            _input_row(
                pdbID="2bbb",
                metal_site_id="2bbb:1",
                rszd_abs="",
                geometry_rms_zbond="",
                confidence_inputs_status="unscorable",
            ),
        ],
    )
    finalized = cs.finalize_database_confidence(
        input_path, str(tmp_path / "scores.csv"), str(tmp_path / "reference")
    )
    assert (finalized.rows, finalized.scored_rows, finalized.cohort_size) == (2, 1, 2)
    assert tuple(finalized) == (2, 1, 2)


def test_finalize_rejects_an_incompatible_manifest_before_reading_the_input(
    tmp_path: Path,
) -> None:
    """The manifest is cheap to check, so it must fail before the input pass."""
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdbID", "analysis_config_id"])
        writer.writeheader()
        writer.writerow({"pdbID": "1abc", "analysis_config_id": "incompatible"})
    with pytest.raises(ValueError, match="source manifest analysis configuration"):
        cs.finalize_database_confidence(
            str(tmp_path / "absent-inputs.csv"),
            str(tmp_path / "scores.csv"),
            str(tmp_path / "reference"),
            str(manifest_path),
        )


def test_finalized_metadata_keeps_the_code_owned_analysis_config_id(
    tmp_path: Path,
) -> None:
    """Provenance describes the cohort; the policy identity stays the code's."""
    input_path = _write_input_csv(tmp_path / "inputs.csv", [_input_row()])
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdbID", "analysis_config_id"])
        writer.writeheader()
        writer.writerow({"pdbID": "1abc", "analysis_config_id": ANALYSIS_CONFIG_ID})
    reference_dir = tmp_path / "reference"
    cs.finalize_database_confidence(
        input_path,
        str(tmp_path / "scores.csv"),
        str(reference_dir),
        str(manifest_path),
    )
    metadata = json.loads(
        (reference_dir / cs.REFERENCE_METADATA_FILE).read_text(encoding="utf-8")
    )
    assert metadata["analysis_config_id"] == ANALYSIS_CONFIG_ID
    assert cs.load_reference(str(reference_dir)).metadata["source_entry_count"] == 1


def test_the_two_in_memory_entry_points_share_one_scoring_path() -> None:
    """Both wrappers add the same columns to the same rows; only ranking differs."""
    row = _input_row()
    ranked = cs.score_against_reference([row], _reference())[0]
    unranked = cs.classify_without_reference([row])[0]
    assert set(ranked) == set(unranked) == {*row, *cs.ANALYSIS_COLUMNS}
    assert ranked["alchemy_level"] == unranked["alchemy_level"]
    assert ranked["evidence_basis"] == unranked["evidence_basis"]
    assert ranked["verdict_reason"] == unranked["verdict_reason"]
    assert ranked["alchemy_score"] != ""
    assert unranked["alchemy_score"] == ""
    assert unranked["confidence_reference_version"] == ""
    assert row == _input_row(), "scoring must not edit the row it was handed"
