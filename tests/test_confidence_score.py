"""Behavioural tests for ``src/confidence_score.py``.

The module owns Alchemy's site-level confidence policy: the piecewise-linear
severity transform, the fixed scoring formula documented in README.md

    confidence = 100 * (1 - 0.50*SR - 0.35*QG*SG - 0.15*QG*sqrt(SR*SG))

the compact confidence-input rows derived from in-memory density and bond
evidence, the frozen database reference distribution, and the percentile that
small runs borrow from that frozen cohort.

Every numeric expectation here is derived by hand from the anchors and the
formula above, never from the code's current output.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter

import pytest

import confidence_score as cs
import helpers
from bond_schema import STATS_EXTRA_COLUMNS


# --------------------------------------------------------------------------- #
# Local row factories (compact rows are built directly; the pipeline is not run
# except in the one candidate-exclusion test, which needs real candidate rows).
# --------------------------------------------------------------------------- #
STATS_COLUMNS = (
    ["pdbID", "category"]
    + list(helpers.EDSTATS_HEADER)
    + ["aa_geometry_coverage"]
    + list(STATS_EXTRA_COLUMNS)
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
    **overrides,
):
    """A metal statistics row as ``prepare_confidence_inputs`` consumes it.

    The ZD parameters are typed ``object`` because tests pass blanks and
    unparsable text through them to exercise the non-numeric branches.
    """
    row = {column: "" for column in cs.IDENTITY_COLUMNS}
    row.update(SITE)
    row.update(
        {
            "pdbID": pdb_id,
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
            "metal_icode": "",
            "metal_altloc": "",
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
    **overrides,
):
    """An assigned bond row as ``bond_analysis`` emits it.

    ``zscore`` accepts ``None`` so tests can emit a reference-covered contact
    whose z-score is unavailable, which is how a missing DPI reaches the
    confidence inputs.
    """
    row = dict(SITE)
    row.update(
        {
            "pdbID": pdb_id,
            "parent_type": "metal",
            "metal_resname": "ZN",
            "metal_chain": "B",
            "metal_resnum": "1",
            "metal_atom": "ZN",
            "metal_element": "ZN",
            "metal_icode": "",
            "metal_altloc": "",
            "neighbor_resname": neighbor,
            "neighbor_chain": "A",
            "neighbor_resnum": "10",
            "neighbor_atom": atom,
            "reference_covered": str(covered),
            "declared_connection": "False",
            "coordination_status": "inferred",
            "multi_donor_detected": "False",
            "zscore": "" if zscore is None else str(zscore),
            "context_warning": "False",
            "context_warning_reasons": "",
        }
    )
    row.update(overrides)
    return row


def _input_row(**overrides):
    """A compact confidence-input row (all schema columns present)."""
    row = {column: "" for column in cs.CONFIDENCE_INPUT_COLUMNS}
    row.update(SITE)
    row.update(
        {
            "pdbID": "1abc",
            "category": "metal",
            "selected_metal_site_status": "selected",
            "metal_resname": "ZN",
            "metal_element": "ZN",
            "metal_chain": "B",
            "metal_resnum": "1",
            "metal_atom": "ZN",
            "rszd_magnitude": "3",
            "max_abs_zbond": "6",
            "geometry_coverage": "1",
            "assigned_contact_count": "2",
            "reference_covered_contact_count": "2",
            "zbond_available_contact_count": "2",
            "context_warning": "False",
            "context_warning_reasons": "",
            "confidence_inputs_status": "complete",
            "confidence_inputs_missing_reasons": "",
        }
    )
    row.update(overrides)
    return row


def _write_input_csv(path, rows, columns=cs.CONFIDENCE_INPUT_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        assert fieldnames is not None, f"CSV file {path} has no header"
        return list(fieldnames), list(reader)


# Hand-derived severities, straight off the anchor tables. These are the
# oracles the formula tests are written against.
#   density  anchors: (2,0.00) (3,0.25) (6,0.65) (12,1.00)
#   geometry anchors: (2,0.00) (3,0.35) (6,0.70) (12,1.00)
DENSITY_EXPECTED = {
    0.0: 0.0,
    1.0: 0.0,
    2.0: 0.0,
    2.5: 0.125,
    3.0: 0.25,
    4.5: 0.25 + 0.5 * 0.40,
    6.0: 0.65,
    9.0: 0.65 + 0.5 * 0.35,
    12.0: 1.0,
    40.0: 1.0,
}
GEOMETRY_EXPECTED = {
    0.0: 0.0,
    2.0: 0.0,
    2.5: 0.175,
    3.0: 0.35,
    4.5: 0.35 + 0.5 * 0.35,
    6.0: 0.70,
    9.0: 0.70 + 0.5 * 0.30,
    12.0: 1.0,
    40.0: 1.0,
}


def _expected_confidence(sr, sg, qg):
    """README formula, written out independently of the implementation."""
    return 100.0 * (1.0 - 0.50 * sr - 0.35 * qg * sg - 0.15 * qg * math.sqrt(sr * sg))


# --------------------------------------------------------------------------- #
# severity()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "anchors,expected",
    [
        (cs.DENSITY_ANCHORS, DENSITY_EXPECTED),
        (cs.GEOMETRY_ANCHORS, GEOMETRY_EXPECTED),
    ],
)
def test_severity_matches_hand_computed_anchor_interpolation(anchors, expected):
    """severity() is exactly the documented piecewise-linear anchor curve."""
    for value, want in expected.items():
        assert cs.severity(value, anchors) == pytest.approx(want, abs=1e-12), (
            f"severity({value})"
        )


@pytest.mark.parametrize("anchors", [cs.DENSITY_ANCHORS, cs.GEOMETRY_ANCHORS])
def test_severity_is_continuous_at_every_anchor(anchors):
    """Left and right limits agree at each anchor: no jump between segments."""
    eps = 1e-7
    for x, y in anchors:
        left = cs.severity(x - eps, anchors)
        right = cs.severity(x + eps, anchors)
        assert cs.severity(x, anchors) == pytest.approx(y, abs=1e-12)
        assert left == pytest.approx(y, abs=1e-6)
        assert right == pytest.approx(y, abs=1e-6)


@pytest.mark.parametrize("anchors", [cs.DENSITY_ANCHORS, cs.GEOMETRY_ANCHORS])
def test_severity_clamps_outside_the_end_anchors(anchors):
    """Below the first anchor and above the last, severity is flat, not linear."""
    low_x, low_y = anchors[0]
    high_x, high_y = anchors[-1]
    for value in (0.0, low_x / 2.0, low_x - 1e-9, low_x):
        assert cs.severity(value, anchors) == low_y
    for value in (high_x, high_x + 1e-9, high_x * 10.0, 1e9):
        assert cs.severity(value, anchors) == high_y


@pytest.mark.parametrize("anchors", [cs.DENSITY_ANCHORS, cs.GEOMETRY_ANCHORS])
def test_severity_is_monotonic_and_bounded_over_a_dense_sweep(anchors):
    """severity() never decreases and stays inside [0, 1] for any magnitude."""
    previous = -1.0
    value = 0.0
    while value <= 20.0:
        current = cs.severity(value, anchors)
        assert 0.0 <= current <= 1.0
        assert current >= previous - 1e-12, f"decreased at {value}"
        previous = current
        value += 0.01


@pytest.mark.parametrize("value", [-1e-9, -0.5, -50.0, math.nan, math.inf, -math.inf])
def test_severity_rejects_negative_and_nonfinite_magnitudes(value):
    """A magnitude that cannot be a |Z| yields NaN rather than a fake severity."""
    assert math.isnan(cs.severity(value, cs.DENSITY_ANCHORS))


# --------------------------------------------------------------------------- #
# score_site()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rszd,zbond,coverage,sr,sg",
    [
        (3.0, 6.0, 1.0, 0.25, 0.70),
        (2.0, 2.0, 1.0, 0.00, 0.00),
        (12.0, 12.0, 1.0, 1.00, 1.00),
        (6.0, 3.0, 0.5, 0.65, 0.35),
        (4.5, 9.0, 0.25, 0.45, 0.85),
        (2.5, 4.5, 1.0, 0.125, 0.525),
    ],
)
def test_score_site_matches_the_readme_formula(rszd, zbond, coverage, sr, sg):
    """confidence == 100*(1 - 0.50*SR - 0.35*QG*SG - 0.15*QG*sqrt(SR*SG))."""
    result = cs.score_site(rszd, zbond, coverage)
    assert result is not None
    assert result["density_severity"] == pytest.approx(sr, abs=1e-12)
    assert result["geometry_severity"] == pytest.approx(sg, abs=1e-12)
    assert result["density_score_component"] == pytest.approx(0.50 * sr)
    assert result["geometry_score_component"] == pytest.approx(0.35 * coverage * sg)
    assert result["interaction_score_component"] == pytest.approx(
        0.15 * coverage * math.sqrt(sr * sg)
    )
    assert result["confidence_score"] == pytest.approx(
        _expected_confidence(sr, sg, coverage), abs=1e-9
    )


def test_score_site_extreme_endpoints_are_exactly_zero_and_one_hundred():
    """Perfect evidence scores 100; worst-case full-coverage evidence scores 0."""
    perfect = cs.score_site(0.0, 0.0, 1.0)
    assert perfect is not None
    assert perfect["confidence_score"] == pytest.approx(100.0, abs=1e-12)
    worst = cs.score_site(12.0, 12.0, 1.0)
    assert worst is not None
    # 100*(1 - 0.50 - 0.35 - 0.15) == 0 exactly, with no clamping needed.
    assert worst["confidence_score"] == pytest.approx(0.0, abs=1e-9)


def test_score_site_with_zero_coverage_ignores_geometry_entirely():
    """QG == 0 removes both geometry terms, even when Zbond is unavailable."""
    density_only = cs.score_site(6.0, math.nan, 0.0)
    assert density_only is not None
    assert density_only["geometry_severity"] == 0.0
    assert density_only["geometry_score_component"] == 0.0
    assert density_only["interaction_score_component"] == 0.0
    # 100*(1 - 0.50*0.65) = 67.5
    assert density_only["confidence_score"] == pytest.approx(67.5, abs=1e-9)
    # A wild Zbond cannot change a zero-coverage score.
    wild_geometry = cs.score_site(6.0, 500.0, 0.0)
    assert wild_geometry is not None
    assert wild_geometry["confidence_score"] == pytest.approx(67.5, abs=1e-9)


@pytest.mark.parametrize("coverage,effective", [(1.5, 1.0), (99.0, 1.0)])
def test_score_site_clamps_coverage_above_one(coverage, effective):
    """A coverage above 1 cannot deduct more than full geometry coverage."""
    result = cs.score_site(3.0, 6.0, coverage)
    assert result is not None
    assert result["confidence_score"] == pytest.approx(
        _expected_confidence(0.25, 0.70, effective), abs=1e-9
    )


def test_score_site_clamps_negative_coverage_to_the_density_only_branch():
    """A negative coverage is treated as zero, never as negative weight."""
    result = cs.score_site(3.0, 6.0, -0.4)
    assert result is not None
    assert result["geometry_severity"] == 0.0
    assert result["confidence_score"] == pytest.approx(
        _expected_confidence(0.25, 0.0, 0.0), abs=1e-9
    )


@pytest.mark.parametrize(
    "rszd,zbond,coverage",
    [
        (math.nan, 6.0, 1.0),
        (-1.0, 6.0, 1.0),
        (math.inf, 6.0, 1.0),
        (3.0, math.nan, 1.0),
        (3.0, math.nan, 0.25),
        (3.0, -2.0, 1.0),
    ],
)
def test_score_site_refuses_to_score_missing_evidence(rszd, zbond, coverage):
    """Absent RSZD, or absent Zbond with non-zero coverage, must not be scored."""
    assert cs.score_site(rszd, zbond, coverage) is None


def test_score_site_stays_within_zero_and_one_hundred_without_clamping():
    """Across a broad sweep the raw formula is already bounded to [0, 100].

    The min/max clamp in score_site is a guard, not a correction: if it ever
    has to fire, the weights or anchors no longer sum to a bounded penalty.
    """
    coverages = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.4, -0.3)
    magnitudes = (0.0, 1.0, 2.0, 2.5, 3.0, 4.5, 6.0, 9.0, 12.0, 25.0, 1e6)
    for rszd in magnitudes:
        for zbond in magnitudes:
            for coverage in coverages:
                result = cs.score_site(rszd, zbond, coverage)
                assert result is not None
                raw = 100.0 * (
                    1.0
                    - result["density_score_component"]
                    - result["geometry_score_component"]
                    - result["interaction_score_component"]
                )
                assert -1e-9 <= raw <= 100.0 + 1e-9, (rszd, zbond, coverage)
                assert result["confidence_score"] == pytest.approx(raw, abs=1e-9)
                assert 0.0 <= result["confidence_score"] <= 100.0


def test_score_site_never_passes_a_negative_argument_to_sqrt():
    """sqrt() only ever sees SR*SG, a product of two severities in [0, 1].

    ``math.sqrt`` of a negative raises ``ValueError``, which would abort a
    database run mid-entry, so the interaction term's argument must be provably
    non-negative for every input the pipeline can produce -- including a
    coverage outside [0, 1], which the caller clamps.

    The argument is recovered from the reported components rather than by
    spying on ``math.sqrt``: ``cs.math`` *is* the stdlib module, so patching it
    would replace ``sqrt`` process-wide and let any unrelated call inside the
    sweep pollute the evidence.
    """
    checked = 0
    for rszd in (0.0, 2.5, 6.0, 1e6):
        for zbond in (0.0, 2.5, 6.0, 1e6):
            for coverage in (-5.0, 0.0, 0.5, 1.0, 7.0):
                result = cs.score_site(rszd, zbond, coverage)
                assert result is not None, (rszd, zbond, coverage)
                argument = result["density_severity"] * result["geometry_severity"]
                # Both severities are bounded transforms, so their product --
                # the only thing sqrt is given -- is in [0, 1].
                assert 0.0 <= argument <= 1.0, (rszd, zbond, coverage, argument)
                # And the reported interaction term really is that sqrt: if the
                # formula ever fed sqrt something else, this is where the
                # substitution shows up.
                clamped = min(1.0, max(0.0, coverage))
                assert result["interaction_score_component"] == pytest.approx(
                    0.15 * clamped * math.sqrt(argument), abs=1e-12
                )
                checked += 1
    assert checked == 4 * 4 * 5


# --------------------------------------------------------------------------- #
# geometry coverage (QG) in prepare_confidence_inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "covered_flags,expected_covered,expected_coverage",
    [
        ([True, True, True], 3, 1.0),
        ([True, False, False, False], 1, 0.25),
        ([True, True, False], 2, 2.0 / 3.0),
        ([False, False], 0, 0.0),
    ],
)
def test_geometry_coverage_is_reference_covered_over_assigned(
    covered_flags, expected_covered, expected_coverage
):
    """QG = reference-covered contacts / total assigned contacts."""
    bonds = [
        _bond_row(covered=flag, atom=f"N{index}")
        for index, flag in enumerate(covered_flags)
    ]
    (prepared,) = cs.prepare_confidence_inputs([_stats_row()], bonds)
    assert int(prepared["assigned_contact_count"]) == len(covered_flags)
    assert int(prepared["reference_covered_contact_count"]) == expected_covered
    assert float(prepared["geometry_coverage"]) == pytest.approx(
        expected_coverage, abs=1e-6
    )


def test_geometry_coverage_with_no_assigned_contacts_is_zero_not_an_error():
    """A metal with no assigned contacts has a zero denominator: QG is 0."""
    (prepared,) = cs.prepare_confidence_inputs([_stats_row()], [])
    assert int(prepared["assigned_contact_count"]) == 0
    assert float(prepared["geometry_coverage"]) == 0.0
    assert prepared["confidence_inputs_status"] == "density_only"
    assert "no_assigned_contacts" in prepared[
        "confidence_inputs_missing_reasons"
    ].split("|")
    # A zero denominator must still be scorable on density alone.
    assert cs.score_site(3.0, math.nan, 0.0) is not None


def test_rejected_broad_search_candidates_stay_out_of_the_qg_denominator(tmp_path):
    """Only assigned bond rows count; a rejected 4 A candidate must not.

    Built from real bond_analysis output so the candidate/bond split is the
    production one: HIS NE2 at 2.03 A is assigned, MET SD at 3.60 A is found
    by the broad search and rejected as outside the first sphere.
    """
    from structure_analysis import load_structure
    from bond_analysis import run_bond_analysis

    builder = helpers.simple_metal_site(
        "ZN", [("HIS", "NE2", 2.03), ("MET", "SD", 3.60)]
    )
    path = builder.write_pdb(tmp_path / "site.pdb")
    context = load_structure("1abc", path)
    bonds, candidates, _, _ = run_bond_analysis(
        "1abc",
        path,
        [],
        list(helpers.EDSTATS_HEADER),
        helpers.dpi_inputs(),
        structure=context,
    )

    rejected = [
        row
        for row in candidates
        if row["eligibility_status"] != "first_sphere_eligible"
    ]
    assert len(rejected) == 1 and rejected[0]["neighbor_atom"] == "SD"
    assert {row["neighbor_atom"] for row in bonds} == {"NE2"}

    stats = _stats_row(
        **{key: str(bonds[0][key]) for key in cs.SITE_KEY_COLUMNS if key != "pdbID"}
    )
    (prepared,) = cs.prepare_confidence_inputs([stats], bonds)
    assert int(prepared["assigned_contact_count"]) == 1
    assert float(prepared["geometry_coverage"]) == 1.0


def test_finite_zbond_count_is_recorded_apart_from_reference_coverage(tmp_path):
    """Missing DPI (no Zbond) must not masquerade as usable geometry evidence."""
    bonds = [
        _bond_row(covered=True, zscore=None, atom="NE2"),
        _bond_row(covered=True, zscore=None, atom="OD1"),
    ]
    (prepared,) = cs.prepare_confidence_inputs([_stats_row()], bonds)
    assert int(prepared["reference_covered_contact_count"]) == 2
    assert int(prepared["zbond_available_contact_count"]) == 0
    assert float(prepared["geometry_coverage"]) == 1.0
    assert prepared["max_abs_zbond"] == ""
    assert prepared["confidence_inputs_status"] == "partial_geometry"
    assert "zbond_unavailable_for_reference" in prepared[
        "confidence_inputs_missing_reasons"
    ].split("|")

    reference = cs.write_reference(str(tmp_path / "ref"), Counter({50.0: 3}), 3)
    (scored,) = cs.score_against_reference([prepared], reference)
    assert scored["confidence_scoring_status"] == "unscorable"
    assert scored["confidence_scoring_reason"] == (
        "zbond_unavailable_with_nonzero_coverage"
    )
    assert scored["confidence_score"] == ""


def test_max_abs_zbond_takes_the_largest_magnitude_and_names_its_contact():
    """The reported worst contact is the largest |Zbond|, sign-independent."""
    bonds = [
        _bond_row(zscore=3.0, neighbor="HIS", atom="NE2"),
        _bond_row(
            zscore=-7.5,
            neighbor="ASP",
            atom="OD1",
            neighbor_chain="C",
            neighbor_resnum="42",
        ),
        _bond_row(zscore=None, neighbor="HOH", atom="O"),
    ]
    (prepared,) = cs.prepare_confidence_inputs([_stats_row()], bonds)
    assert float(prepared["max_abs_zbond"]) == pytest.approx(7.5)
    assert prepared["max_abs_zbond_neighbor_resname"] == "ASP"
    assert prepared["max_abs_zbond_neighbor_atom"] == "OD1"
    assert prepared["max_abs_zbond_neighbor_chain"] == "C"
    assert prepared["max_abs_zbond_neighbor_resnum"] == "42"
    assert int(prepared["zbond_available_contact_count"]) == 2


# --------------------------------------------------------------------------- #
# prepare_confidence_inputs structure / status
# --------------------------------------------------------------------------- #
def test_prepare_emits_one_row_per_selected_site_and_skips_unselected():
    """Only sites the run actually selected enter the confidence cohort."""
    selected = _stats_row(pdb_id="1abc")
    other = _stats_row(
        pdb_id="1abc",
        metal_atom_index="7",
        selected_metal_site_status="alternate_conformer",
    )
    prepared = cs.prepare_confidence_inputs([selected, other], [])
    assert len(prepared) == 1
    assert prepared[0]["metal_atom_index"] == "0"
    assert set(prepared[0]) == set(cs.CONFIDENCE_INPUT_COLUMNS)


def test_prepare_rejects_duplicate_site_keys():
    """Two statistics rows for one site would double-count the cohort."""
    with pytest.raises(ValueError, match="duplicate site key"):
        cs.prepare_confidence_inputs([_stats_row(), _stats_row()], [])


@pytest.mark.parametrize(
    "zdm,expected_status,expected_reason",
    [
        ("", "unscorable", "rszd_unavailable"),
        ("n/a", "unscorable", "rszd_unavailable"),
        ("-4.0", "complete", ""),
    ],
)
def test_missing_rszd_makes_a_site_unscorable(zdm, expected_status, expected_reason):
    """RSZD is mandatory evidence; its magnitude is used, so sign is irrelevant."""
    stats = _stats_row(**{"ZDm": zdm})
    (prepared,) = cs.prepare_confidence_inputs([stats], [_bond_row()])
    assert prepared["confidence_inputs_status"] == expected_status
    reasons = [r for r in prepared["confidence_inputs_missing_reasons"].split("|") if r]
    assert reasons == ([expected_reason] if expected_reason else [])
    if expected_status == "complete":
        assert float(prepared["rszd_magnitude"]) == pytest.approx(4.0)


def test_partial_geometry_coverage_is_flagged_not_silently_averaged():
    """A site with some uncovered contacts is reported as partial geometry."""
    bonds = [_bond_row(covered=True, atom="NE2"), _bond_row(covered=False, atom="CB")]
    (prepared,) = cs.prepare_confidence_inputs([_stats_row()], bonds)
    assert prepared["confidence_inputs_status"] == "partial_geometry"
    assert "partial_geometry_coverage" in prepared[
        "confidence_inputs_missing_reasons"
    ].split("|")
    assert float(prepared["geometry_coverage"]) == pytest.approx(0.5)


def test_declared_inferred_and_multi_donor_contacts_are_counted():
    """Provenance counters describe how the assigned contacts were obtained."""
    bonds = [
        _bond_row(
            atom="NE2", declared_connection="True", coordination_status="declared"
        ),
        _bond_row(
            atom="OD1", coordination_status="inferred", multi_donor_detected="True"
        ),
        _bond_row(
            atom="OD2", coordination_status="inferred", multi_donor_detected="True"
        ),
    ]
    (prepared,) = cs.prepare_confidence_inputs([_stats_row()], bonds)
    assert int(prepared["declared_contact_count"]) == 1
    assert int(prepared["inferred_contact_count"]) == 2
    assert int(prepared["multi_donor_contact_count"]) == 2


def test_bond_rows_without_a_density_row_survive_as_unscorable_orphans():
    """A bond-bearing site whose density row is missing must not disappear."""
    bonds = [_bond_row(atom="NE2"), _bond_row(atom="OD1")]
    prepared = cs.prepare_confidence_inputs([], bonds)
    assert len(prepared) == 1
    orphan = prepared[0]
    assert orphan["coordinate_mapping_status"] == "density_row_unavailable"
    assert orphan["selected_metal_site_status"] == "selected_without_density_row"
    assert orphan["confidence_inputs_status"] == "unscorable"
    reasons = orphan["confidence_inputs_missing_reasons"].split("|")
    assert "rszd_unavailable" in reasons
    assert "density_row_unavailable" in reasons
    # The bond evidence it does have is preserved rather than discarded.
    assert int(orphan["assigned_contact_count"]) == 2
    assert orphan["metal_resname"] == "ZN"
    assert set(orphan) == set(cs.CONFIDENCE_INPUT_COLUMNS)


# --------------------------------------------------------------------------- #
# prepare_result_confidence_inputs agreement
# --------------------------------------------------------------------------- #
def _result_stats_row(flat):
    return {
        "pdbID": flat["pdbID"],
        "category": flat["category"],
        "fields": [flat.get(column, "") for column in STATS_COLUMNS[2:]],
    }


def test_result_preparation_agrees_with_row_preparation():
    """The in-memory worker path and the flat-row path produce identical rows."""
    flat = _stats_row()
    bonds = [
        _bond_row(atom="NE2", zscore=4.0),
        _bond_row(atom="OD1", covered=False, zscore=None),
    ]
    from_flat = cs.prepare_confidence_inputs([flat], bonds)
    from_result = cs.prepare_result_confidence_inputs(
        [_result_stats_row(flat)], bonds, STATS_COLUMNS
    )
    assert from_result == from_flat
    assert from_flat[0]["pdbID"] == "1abc"


def test_result_preparation_rejects_a_row_of_the_wrong_width():
    """A schema drift between driver and module must fail loudly, not silently."""
    flat = _stats_row()
    row = _result_stats_row(flat)
    row["fields"] = row["fields"][:-1]
    with pytest.raises(ValueError, match="confidence schema"):
        cs.prepare_result_confidence_inputs([row], [], STATS_COLUMNS)


# --------------------------------------------------------------------------- #
# complete_confidence_site_count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prepared_count,selected", [(0, 3), (1, 2), (2, 2), (0, 0)])
def test_every_manifest_counted_site_keeps_exactly_one_row(prepared_count, selected):
    """The streamed cohort holds exactly one row per manifest-counted metal."""
    rows = [_input_row(metal_atom_index=str(index)) for index in range(prepared_count)]
    completed = cs.complete_confidence_site_count(rows, "1abc", selected)
    assert len(completed) == selected
    placeholders = [
        row
        for row in completed
        if row["selected_metal_site_status"] == "selected_site_unresolved"
    ]
    assert len(placeholders) == selected - prepared_count
    for row in placeholders:
        assert set(row) == set(cs.CONFIDENCE_INPUT_COLUMNS)
        assert row["pdbID"] == "1abc"
        assert row["confidence_inputs_status"] == "unscorable"
        reasons = row["confidence_inputs_missing_reasons"].split("|")
        assert "rszd_unavailable" in reasons
        assert "site_identity_unavailable" in reasons
        assert "site_evidence_unavailable" in reasons
    # Placeholders are individually addressable, never collapsed.
    assert len({row["metal_atom_index"] for row in placeholders}) == len(placeholders)


def test_more_prepared_rows_than_selected_sites_is_an_error():
    """More confidence rows than selected metals means the join went wrong."""
    rows = [_input_row(metal_atom_index=str(i)) for i in range(3)]
    with pytest.raises(ValueError, match="exceed selected metal count"):
        cs.complete_confidence_site_count(rows, "1abc", 2)


def test_missing_reason_downgrades_real_rows_and_is_not_duplicated():
    """An entry-level failure marks every one of its rows unscorable, once."""
    rows = [_input_row(confidence_inputs_missing_reasons="already|bond_failure")]
    completed = cs.complete_confidence_site_count(
        rows, "1abc", 2, missing_reason="bond_failure"
    )
    assert completed[0]["confidence_inputs_status"] == "unscorable"
    assert completed[0]["confidence_inputs_missing_reasons"].split("|") == [
        "already",
        "bond_failure",
    ]
    assert "bond_failure" in completed[1]["confidence_inputs_missing_reasons"].split(
        "|"
    )


def test_completion_does_not_mutate_the_rows_it_was_given():
    """Completion copies: the caller's in-memory result rows stay untouched."""
    row = _input_row()
    original = dict(row)
    completed = cs.complete_confidence_site_count(
        [row], "1abc", 2, missing_reason="entry_failed"
    )
    assert row == original
    assert completed[0] is not row
    assert completed[0]["confidence_inputs_status"] == "unscorable"


# --------------------------------------------------------------------------- #
# Reference persistence, identity, percentiles
# --------------------------------------------------------------------------- #
def test_reference_round_trips_through_disk_unchanged(tmp_path):
    """write_reference / load_reference preserve the cohort exactly."""
    counts = Counter({12.5: 2, 56.72505: 1, 100.0: 5})
    written = cs.write_reference(str(tmp_path / "ref"), counts, input_row_count=11)
    loaded = cs.load_reference(str(tmp_path / "ref"))
    assert loaded.values == written.values == (12.5, 56.72505, 100.0)
    assert loaded.counts == written.counts == (2, 1, 5)
    assert loaded.cohort_size == 8
    assert loaded.reference_id == written.reference_id
    assert loaded.metadata["input_row_count"] == 11
    assert loaded.metadata["distinct_score_count"] == 3
    for score in written.values:
        assert loaded.percentile(score) == written.percentile(score)


def test_reference_id_is_deterministic_and_tracks_the_distribution(tmp_path):
    """The identifier is a pure function of policy plus the score distribution."""
    counts = Counter({10.0: 1, 20.0: 2})
    first = cs.write_reference(str(tmp_path / "a"), counts, 3)
    same = cs.write_reference(str(tmp_path / "b"), Counter({20.0: 2, 10.0: 1}), 3)
    assert first.reference_id == same.reference_id
    assert first.reference_id.startswith("alchemy-confidence-")

    different_count = cs.write_reference(
        str(tmp_path / "c"), Counter({10.0: 1, 20.0: 3}), 4
    )
    different_score = cs.write_reference(
        str(tmp_path / "d"), Counter({10.0: 1, 20.5: 2}), 3
    )
    extra_score = cs.write_reference(
        str(tmp_path / "e"), Counter({10.0: 1, 20.0: 2, 30.0: 1}), 4
    )
    assert (
        len(
            {
                first.reference_id,
                different_count.reference_id,
                different_score.reference_id,
                extra_score.reference_id,
            }
        )
        == 4
    )
    # The row count of the run is provenance, not part of the identity.
    other_input_count = cs.write_reference(str(tmp_path / "f"), counts, 900)
    assert other_input_count.reference_id == first.reference_id


def test_write_reference_refuses_an_empty_cohort(tmp_path):
    """A run with nothing scorable cannot publish a reference distribution."""
    with pytest.raises(ValueError, match="no scores"):
        cs.write_reference(str(tmp_path / "ref"), Counter(), 0)


@pytest.mark.parametrize(
    "score,expected",
    [
        (10.0, 12.5),  # 100*(0 + 0.5*1)/4
        (20.0, 50.0),  # 100*(1 + 0.5*2)/4
        (30.0, 87.5),  # 100*(3 + 0.5*1)/4
        (5.0, 0.0),  # below the cohort
        (15.0, 25.0),  # between 10 and 20: one score below, none equal
        (35.0, 100.0),  # above the cohort
    ],
)
def test_percentile_is_the_average_rank_ecdf_of_the_frozen_cohort(
    tmp_path, score, expected
):
    """Percentiles are hand-computed average-rank ECDF values, ties included."""
    reference = cs.write_reference(
        str(tmp_path / "ref"), Counter({10.0: 1, 20.0: 2, 30.0: 1}), 4
    )
    assert reference.cohort_size == 4
    assert reference.percentile(score) == pytest.approx(expected, abs=1e-9)


def test_percentile_is_monotonic_across_the_score_range(tmp_path):
    """A better score never receives a worse percentile."""
    reference = cs.write_reference(
        str(tmp_path / "ref"), Counter({0.0: 1, 33.5: 4, 67.0: 2, 100.0: 3}), 10
    )
    previous = -1.0
    score = -5.0
    while score <= 105.0:
        current = reference.percentile(score)
        assert 0.0 <= current <= 100.0
        assert current >= previous - 1e-12, f"decreased at {score}"
        previous = current
        score += 0.5


@pytest.mark.parametrize(
    "values,counts",
    [
        ((), ()),
        ((1.0, 2.0), (1,)),
        ((2.0, 1.0), (1, 1)),
        ((1.0, 1.0), (1, 1)),
        ((1.0, 2.0), (1, 0)),
    ],
)
def test_reference_construction_rejects_malformed_distributions(values, counts):
    """A distribution that is empty, unsorted, or non-positive is not usable."""
    with pytest.raises(ValueError):
        cs.ConfidenceReference(values, counts, {})


def test_load_reference_rejects_a_tampered_distribution(tmp_path):
    """The identifier detects a distribution edited after publication."""
    reference_dir = tmp_path / "ref"
    cs.write_reference(str(reference_dir), Counter({10.0: 1, 20.0: 2}), 3)
    distribution = reference_dir / cs.REFERENCE_DISTRIBUTION_FILE
    text = distribution.read_text(encoding="utf-8")
    distribution.write_text(text.replace("20.0,2", "20.0,5"), encoding="utf-8")
    with pytest.raises(ValueError, match="identifier does not match"):
        cs.load_reference(str(reference_dir))


def test_load_reference_rejects_incompatible_scoring_policy(tmp_path):
    """A reference built under different weights or anchors is refused."""
    reference_dir = tmp_path / "ref"
    cs.write_reference(str(reference_dir), Counter({10.0: 1}), 1)
    metadata_path = reference_dir / cs.REFERENCE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["weights"] = {"density": 0.6, "geometry": 0.3, "interaction": 0.1}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="weights is incompatible"):
        cs.load_reference(str(reference_dir))


def test_load_reference_rejects_a_cohort_size_that_contradicts_metadata(tmp_path):
    """Cohort size is cross-checked so percentiles cannot use a stale total."""
    reference_dir = tmp_path / "ref"
    cs.write_reference(str(reference_dir), Counter({10.0: 1, 20.0: 2}), 3)
    metadata_path = reference_dir / cs.REFERENCE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scorable_cohort_size"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="cohort size does not match"):
        cs.load_reference(str(reference_dir))


def test_load_reference_rejects_a_non_integer_count(tmp_path):
    """A corrupt count is an error, not a silently truncated weight."""
    reference_dir = tmp_path / "ref"
    cs.write_reference(str(reference_dir), Counter({10.0: 1}), 1)
    distribution = reference_dir / cs.REFERENCE_DISTRIBUTION_FILE
    distribution.write_text("confidence_score,count\n10.0,1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-integer count"):
        cs.load_reference(str(reference_dir))


# --------------------------------------------------------------------------- #
# Scoring against a frozen reference
# --------------------------------------------------------------------------- #
def _frozen_reference(tmp_path, name="ref"):
    """A 100-site cohort whose distribution differs from any test batch."""
    counts = Counter({float(value): 1 for value in range(1, 101)})
    return cs.write_reference(str(tmp_path / name), counts, 100)


def test_small_runs_take_percentiles_from_the_frozen_cohort_only(tmp_path):
    """A small run is ranked against the database, never against itself."""
    reference = _frozen_reference(tmp_path)
    row = _input_row(rszd_magnitude="3", max_abs_zbond="6", geometry_coverage="1")
    alone = cs.score_against_reference([row], reference)
    # 56.72505 -> 56 database scores below it, none equal.
    expected_score = _expected_confidence(0.25, 0.70, 1.0)
    assert float(alone[0]["confidence_score"]) == pytest.approx(
        expected_score, abs=1e-5
    )
    assert float(alone[0]["confidence_percentile"]) == pytest.approx(
        reference.percentile(expected_score), abs=1e-9
    )
    assert int(alone[0]["confidence_cohort_size"]) == 100

    # The same row inside a batch of very different sites is unchanged: the
    # batch's own distribution contributes nothing.
    batch = [
        _input_row(rszd_magnitude=str(value), max_abs_zbond="12", geometry_coverage="1")
        for value in (2.0, 2.0, 12.0, 12.0, 12.0)
    ] + [row]
    scored = cs.score_against_reference(batch, reference)
    assert scored[-1]["confidence_percentile"] == alone[0]["confidence_percentile"]
    assert {int(r["confidence_cohort_size"]) for r in scored} == {100}
    assert {r["confidence_reference_id"] for r in scored} == {reference.reference_id}


def test_scoring_status_distinguishes_density_only_from_partial_geometry(tmp_path):
    """The scoring status records how much geometry actually backed the score."""
    reference = _frozen_reference(tmp_path)
    rows = [
        _input_row(geometry_coverage="1"),
        _input_row(geometry_coverage="0.5"),
        _input_row(geometry_coverage="0", max_abs_zbond=""),
        _input_row(rszd_magnitude=""),
    ]
    scored = cs.score_against_reference(rows, reference)
    assert [row["confidence_scoring_status"] for row in scored] == [
        "complete",
        "partial_geometry",
        "density_only",
        "unscorable",
    ]
    assert scored[1]["confidence_scoring_reason"] == ("geometry_coverage_below_one")
    assert scored[2]["confidence_scoring_reason"] == "no_usable_geometry"
    assert scored[3]["confidence_scoring_reason"] == "rszd_unavailable"
    # An unscorable row still carries the reference identity for auditing.
    assert scored[3]["confidence_reference_id"] == reference.reference_id
    assert scored[3]["confidence_percentile"] == ""


def test_an_explicit_zero_coverage_is_scored_as_no_geometry(tmp_path):
    """A measured QG of zero is evidence, and contributes no geometry terms.

    This is the legitimate density-only case: the site really was assessed and
    really has no reference-covered contact. It is distinguished from a *blank*
    cell, which is damaged input -- see
    :func:`test_missing_or_corrupt_geometry_coverage_is_unscorable`.
    """
    reference = _frozen_reference(tmp_path)
    row = _input_row(geometry_coverage="0", max_abs_zbond="12")
    (scored,) = cs.score_against_reference([row], reference)
    assert scored["confidence_scoring_status"] == "density_only"
    assert scored["confidence_scoring_reason"] == "no_usable_geometry"
    # 100*(1 - 0.50*0.25) = 87.5, i.e. the geometry terms are absent.
    assert float(scored["confidence_score"]) == pytest.approx(87.5, abs=1e-6)


def test_missing_or_corrupt_geometry_coverage_is_unscorable(tmp_path):
    """Damaged QG is invalid input, not evidence that geometry is irrelevant.

    Regression: a non-finite coverage was coerced to zero, which removed both
    geometry terms from the formula, so a corrupt cell scored *higher* than the
    same site with real geometry evidence. Missing data must not move the score
    in either direction; the site is reported unscorable instead.

    A value outside [0, 1] is damaged too: ``score_site`` clamps it, which is
    the same silent repair by another route.
    """
    reference = _frozen_reference(tmp_path)
    for coverage in ("", "   ", "n/a", "nan", "inf", "-inf", "corrupt", "1.5", "-0.2"):
        row = _input_row(geometry_coverage=coverage, max_abs_zbond="12")
        (scored,) = cs.score_against_reference([row], reference)

        assert scored["confidence_scoring_status"] == "unscorable", coverage
        assert scored["confidence_scoring_reason"], coverage
        assert scored["confidence_score"] == "", coverage
        assert scored["confidence_percentile"] == "", coverage


def test_context_warning_is_carried_through_without_changing_the_score(tmp_path):
    """context_warning is interpretive annotation and deducts no points."""
    reference = _frozen_reference(tmp_path)
    plain = _input_row()
    warned = _input_row(
        context_warning="True", context_warning_reasons="suspect_multi_donor_group"
    )
    scored_plain, scored_warned = cs.score_against_reference([plain, warned], reference)
    assert scored_plain["confidence_score"] == scored_warned["confidence_score"]
    assert scored_warned["confidence_context_warning"] == "True"
    assert scored_warned["confidence_context_warning_reasons"] == (
        "suspect_multi_donor_group"
    )


def test_score_file_writes_input_columns_plus_analysis_columns(tmp_path):
    """The scored CSV extends the compact input rather than replacing it."""
    reference = _frozen_reference(tmp_path)
    input_path = _write_input_csv(
        tmp_path / "in.csv",
        [_input_row(), _input_row(rszd_magnitude="", metal_atom_index="1")],
    )
    output_path = str(tmp_path / "out.csv")
    total, scored = cs.score_file_against_reference(input_path, output_path, reference)
    assert (total, scored) == (2, 1)
    header, rows = _read_csv_rows(output_path)
    assert header == list(cs.CONFIDENCE_INPUT_COLUMNS) + list(cs.ANALYSIS_COLUMNS)
    assert not os.path.exists(output_path + ".tmp")
    assert rows[0]["metal_resname"] == "ZN"


def test_score_file_refuses_already_scored_input_and_leaves_no_partial_output(tmp_path):
    """Re-scoring an output file would stack analysis columns; it is refused."""
    reference = _frozen_reference(tmp_path)
    columns = list(cs.CONFIDENCE_INPUT_COLUMNS) + list(cs.ANALYSIS_COLUMNS)
    row = _input_row()
    row.update({column: "" for column in cs.ANALYSIS_COLUMNS})
    input_path = _write_input_csv(tmp_path / "scored.csv", [row], columns)
    output_path = str(tmp_path / "out.csv")
    with pytest.raises(ValueError, match="already contains analysis columns"):
        cs.score_file_against_reference(input_path, output_path, reference)
    assert not os.path.exists(output_path)
    assert not os.path.exists(output_path + ".tmp")


def test_score_file_requires_the_evidence_columns(tmp_path):
    """An input CSV missing the evidence columns cannot be scored blindly."""
    reference = _frozen_reference(tmp_path)
    columns = [c for c in cs.CONFIDENCE_INPUT_COLUMNS if c != "max_abs_zbond"]
    row = {column: "" for column in columns}
    input_path = _write_input_csv(tmp_path / "in.csv", [row], columns)
    with pytest.raises(ValueError, match="max_abs_zbond"):
        cs.score_file_against_reference(
            input_path, str(tmp_path / "out.csv"), reference
        )


# --------------------------------------------------------------------------- #
# finalize_database_confidence
# --------------------------------------------------------------------------- #
def test_finalize_builds_the_cohort_from_scorable_rows_only(tmp_path):
    """The frozen cohort counts scored sites; unscorable rows are kept but not
    counted."""
    rows = [
        _input_row(
            metal_atom_index="0",
            rszd_magnitude="3",
            max_abs_zbond="6",
            geometry_coverage="1",
        ),
        _input_row(
            metal_atom_index="1",
            rszd_magnitude="3",
            max_abs_zbond="6",
            geometry_coverage="1",
        ),
        _input_row(
            metal_atom_index="2",
            rszd_magnitude="2",
            max_abs_zbond="2",
            geometry_coverage="1",
        ),
        _input_row(metal_atom_index="3", rszd_magnitude=""),
        _input_row(
            metal_atom_index="4",
            rszd_magnitude="3",
            max_abs_zbond="",
            geometry_coverage="1",
        ),
    ]
    input_path = _write_input_csv(tmp_path / "in.csv", rows)
    output_path = str(tmp_path / "scores.csv")
    reference_dir = str(tmp_path / "ref")
    total, scored, cohort = cs.finalize_database_confidence(
        input_path, output_path, reference_dir
    )
    assert (total, scored, cohort) == (5, 3, 3)

    expected = Counter(
        {
            round(_expected_confidence(0.25, 0.70, 1.0), 12): 2,
            100.0: 1,
        }
    )
    reference = cs.load_reference(reference_dir)
    assert reference.cohort_size == 3
    assert reference.metadata["input_row_count"] == 5
    assert [round(value, 12) for value in reference.values] == sorted(expected)
    assert reference.counts == tuple(expected[key] for key in sorted(expected))

    _, out_rows = _read_csv_rows(output_path)
    assert len(out_rows) == 5
    # The two identical mid-range sites tie: average rank over a cohort of 3.
    assert float(out_rows[0]["confidence_percentile"]) == pytest.approx(
        100.0 * (0 + 0.5 * 2) / 3, abs=1e-6
    )
    assert float(out_rows[2]["confidence_percentile"]) == pytest.approx(
        100.0 * (2 + 0.5 * 1) / 3, abs=1e-6
    )
    assert {row["confidence_reference_id"] for row in out_rows} == {
        reference.reference_id
    }


def test_finalize_removes_a_stale_reference_when_it_cannot_publish(tmp_path):
    """A failed finalization must not leave an older reference looking current."""
    reference_dir = tmp_path / "ref"
    cs.write_reference(str(reference_dir), Counter({50.0: 4}), 4)
    metadata_path = reference_dir / cs.REFERENCE_METADATA_FILE
    assert metadata_path.is_file()
    input_path = _write_input_csv(tmp_path / "in.csv", [_input_row(rszd_magnitude="")])
    with pytest.raises(ValueError, match="no scores"):
        cs.finalize_database_confidence(
            input_path, str(tmp_path / "out.csv"), str(reference_dir)
        )
    assert not metadata_path.is_file()
    with pytest.raises(OSError):
        cs.load_reference(str(reference_dir))


def test_finalize_is_reproducible_for_the_same_inputs(tmp_path):
    """Two finalizations of identical inputs publish the same reference id."""
    rows = [
        _input_row(
            metal_atom_index=str(index),
            rszd_magnitude=str(2.0 + index),
            max_abs_zbond=str(3.0 + index),
            geometry_coverage="1",
        )
        for index in range(6)
    ]
    input_path = _write_input_csv(tmp_path / "in.csv", rows)
    first = cs.finalize_database_confidence(
        input_path, str(tmp_path / "a.csv"), str(tmp_path / "ref_a")
    )
    second = cs.finalize_database_confidence(
        input_path, str(tmp_path / "b.csv"), str(tmp_path / "ref_b")
    )
    assert first == second
    assert (
        cs.load_reference(str(tmp_path / "ref_a")).reference_id
        == cs.load_reference(str(tmp_path / "ref_b")).reference_id
    )
    assert (tmp_path / "a.csv").read_text(encoding="utf-8") == (
        tmp_path / "b.csv"
    ).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# validate_scored_reference
# --------------------------------------------------------------------------- #
def test_validate_scored_reference_accepts_output_from_the_same_reference(tmp_path):
    """Resuming onto output produced by the same frozen reference is allowed.

    The stamped identifier is asserted first: without it the acceptance below
    would also hold for a validator that had quietly become a no-op, or for
    output that carries no identifier at all (which
    ``test_validate_scored_reference_requires_the_identifier_column`` shows must
    be rejected).
    """
    reference = _frozen_reference(tmp_path)
    input_path = _write_input_csv(tmp_path / "in.csv", [_input_row()])
    output_path = str(tmp_path / "out.csv")
    cs.score_file_against_reference(input_path, output_path, reference)

    _, rows = _read_csv_rows(output_path)
    assert [row["confidence_reference_id"] for row in rows] == [reference.reference_id]

    # Acceptance is signalled by returning without raising.
    assert cs.validate_scored_reference(output_path, reference) is None

    # ... and it is a decision, not a no-op: the very same file is refused
    # against a different snapshot, so passing above means the identifiers were
    # read and matched.
    other = cs.write_reference(
        str(tmp_path / "ref_other"),
        Counter({float(value): 1 for value in range(1, 51)}),
        50,
    )
    assert other.reference_id != reference.reference_id
    with pytest.raises(ValueError, match="different database reference"):
        cs.validate_scored_reference(output_path, other)


def test_validate_scored_reference_rejects_another_database_snapshot(tmp_path):
    """Rows scored against a different snapshot must not be mixed on resume."""
    reference = _frozen_reference(tmp_path, "ref_a")
    other = cs.write_reference(
        str(tmp_path / "ref_b"), Counter({float(v): 1 for v in range(1, 51)}), 50
    )
    assert reference.reference_id != other.reference_id
    input_path = _write_input_csv(tmp_path / "in.csv", [_input_row()])
    output_path = str(tmp_path / "out.csv")
    cs.score_file_against_reference(input_path, output_path, other)
    with pytest.raises(ValueError, match="different database reference"):
        cs.validate_scored_reference(output_path, reference)


def test_validate_scored_reference_requires_the_identifier_column(tmp_path):
    """Output with no reference identifier cannot be proven compatible."""
    reference = _frozen_reference(tmp_path)
    path = _write_input_csv(tmp_path / "unscored.csv", [_input_row()])
    with pytest.raises(ValueError, match="no reference identifier"):
        cs.validate_scored_reference(path, reference)


def test_validate_scored_reference_tolerates_an_empty_scored_file(tmp_path):
    """A header-only output has no conflicting rows and must not block resume.

    A run that died before writing its first scored row leaves exactly this
    file. It carries the identifier column, so it is not the "no identifier"
    case, but it has no identifiers to disagree with -- resume must proceed.
    """
    reference = _frozen_reference(tmp_path)
    columns = list(cs.CONFIDENCE_INPUT_COLUMNS) + list(cs.ANALYSIS_COLUMNS)
    path = _write_input_csv(tmp_path / "empty.csv", [], columns)

    header, rows = _read_csv_rows(path)
    assert "confidence_reference_id" in header and rows == []

    # Acceptance is signalled by returning without raising.
    assert cs.validate_scored_reference(path, reference) is None

    # Tolerance comes from having no identifiers to disagree with, not from
    # the file being skipped: one foreign row in the same columns is refused.
    foreign = {column: "" for column in columns}
    foreign["confidence_reference_id"] = "alchemy-confidence-someotherrun"
    populated = _write_input_csv(tmp_path / "one_row.csv", [foreign], columns)
    with pytest.raises(ValueError, match="different database reference"):
        cs.validate_scored_reference(populated, reference)


# --------------------------------------------------------------------------- #
# Reference building must apply the same coverage policy as scoring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "coverage, label",
    [
        ("7.5", "out of range"),
        ("", "blank"),
        ("nan", "non-finite"),
        ("-0.5", "negative"),
        ("abc", "non-numeric"),
    ],
)
def test_damaged_coverage_is_excluded_from_the_reference_cohort(
    tmp_path, coverage, label
):
    """A site scoring refuses must not enter the cohort it is measured against.

    Regression: ``_score_prepared_row`` refused damaged coverage, but
    ``finalize_database_confidence`` coerced non-finite coverage to zero and
    applied no range check at all. The same row was therefore reported
    ``unscorable`` in ``confidence_scores_all.csv`` *and* scored into the frozen
    distribution, silently contaminating every percentile derived from it.
    """
    rows = [
        _input_row(pdbID="1aaa", metal_atom_index="1", geometry_coverage="1"),
        _input_row(pdbID="2bbb", metal_atom_index="2", geometry_coverage=coverage),
    ]
    inputs = tmp_path / "inputs.csv"
    _write_input_csv(inputs, rows)

    total, scored, cohort = cs.finalize_database_confidence(
        str(inputs), str(tmp_path / "scores.csv"), str(tmp_path / "ref")
    )

    assert total == 2
    assert scored == 1, f"{label} coverage was scored"
    assert cohort == 1, (
        f"{label} coverage entered the reference cohort; the frozen "
        "distribution and every percentile from it are contaminated"
    )

    distribution = list(
        csv.DictReader(
            open(tmp_path / "ref" / "score_distribution.csv", encoding="utf-8")
        )
    )
    assert sum(int(row["count"]) for row in distribution) == 1

    scored_rows = {
        row["pdbID"]: row
        for row in csv.DictReader(open(tmp_path / "scores.csv", encoding="utf-8"))
    }
    assert scored_rows["2bbb"]["confidence_scoring_status"] == "unscorable"
    assert scored_rows["2bbb"]["confidence_scoring_reason"] == (
        "geometry_coverage_invalid"
    )


def test_scoring_and_reference_building_agree_on_coverage_validity():
    """The two paths share one validity test rather than two copies of it.

    They disagreed once; a shared helper is what stops them drifting again.
    """
    for value in (0.0, 0.5, 1.0):
        assert cs.coverage_is_valid(value)
    for value in (-0.001, 1.001, 7.5, float("nan"), float("inf")):
        assert not cs.coverage_is_valid(value)


def test_coverage_policy_is_part_of_the_reference_identity(tmp_path):
    """A reference built under a different coverage policy is refused.

    Without this, a reference produced before the fix -- whose cohort may
    include sites the current code would refuse -- still validates, and nothing
    in the artifact records that it was built under the older rule.
    """
    rows = [_input_row(metal_atom_index="1", geometry_coverage="1")]
    inputs = tmp_path / "inputs.csv"
    _write_input_csv(inputs, rows)
    reference_dir = tmp_path / "ref"
    cs.finalize_database_confidence(
        str(inputs), str(tmp_path / "scores.csv"), str(reference_dir)
    )

    metadata_path = reference_dir / cs.REFERENCE_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["coverage_policy"] == cs.COVERAGE_POLICY

    metadata["coverage_policy"] = "coerce_invalid_coverage_to_zero"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage_policy"):
        cs.load_reference(str(reference_dir))
