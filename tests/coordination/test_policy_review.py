"""The shared thresholds in ``coordination.policy`` and the guards around them."""

from __future__ import annotations

from pathlib import Path

import helpers
import pytest
from helpers import StructureBuilder, approx

from coordination import policy, schema
from coordination.policy import (
    CANDIDATE_ACCEPT_EPSILON,
    CANDIDATE_SEARCH_RADIUS,
    FIRST_SPHERE_TOLERANCE,
    NEARBY_METAL_RADIUS,
    SEARCH_EPSILON,
    _check_reason_strings,
)
from metal_elements import METAL_ELEMENTS


def test_the_reason_string_guard_accepts_the_real_tolerance() -> None:
    """The published reason strings spell the tolerance the code applies."""
    _check_reason_strings(FIRST_SPHERE_TOLERANCE)


@pytest.mark.parametrize("tolerance", [5.0, 75.0, 0.7, 0.8])
def test_the_reason_string_guard_rejects_a_drifted_tolerance(tolerance: float) -> None:
    """A tolerance the strings do not spell fails, suffix coincidences included.

    5 and 75 are the cases a ``str.endswith`` test let through: both are
    suffixes of ``distance_within_target_plus_0.75`` while naming a different
    first sphere.
    """
    with pytest.raises(ValueError) as raised:
        _check_reason_strings(tolerance)

    message = str(raised.value)
    # Both the value in codes.py and the value the constant implies, so a
    # reader sees which of the two moved.
    assert "distance_within_target_plus_0.75" in message
    assert f"distance_within_target_plus_{tolerance:g}" in message
    assert f"distance_exceeds_target_plus_{tolerance:g}" in message


def test_the_nearby_metal_column_name_spells_the_radius() -> None:
    """The published column name is derived from NEARBY_METAL_RADIUS.

    ``site_summary`` and ``schema`` bake the radius into the column name, so a
    changed radius that left the name alone would mislabel every site row.
    """
    assert f"nearby_metal_count_{NEARBY_METAL_RADIUS:g}a" in schema.STATS_EXTRA_COLUMNS


def test_nearby_metal_count_includes_the_radius_and_stops_beyond_it(
    tmp_path: Path,
) -> None:
    """A metal at 5.99 A is counted as nearby; one at 6.01 A is not."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_metal("FE", 2, chain="B", pos=(NEARBY_METAL_RADIUS - 0.01, 0.0, 0.0))
    builder.add_metal("MG", 3, chain="B", pos=(-(NEARBY_METAL_RADIUS + 0.01), 0.0, 0.0))
    path = builder.write_cif(tmp_path / "nearby_metal_radius.cif")
    analysis = helpers.analyze_bonds(path)

    metals = {
        metal.element: metal for metal in analysis.context.metal_atoms(METAL_ELEMENTS)
    }
    summaries = {
        element: analysis.site_summaries[metal.source_key]
        for element, metal in metals.items()
    }

    # The zinc sees both metals, but only the one inside the radius counts.
    assert summaries["ZN"]["nearest_metal_distance"] == approx(5.99)
    assert summaries["ZN"]["nearby_metal_count_6a"] == 1
    assert summaries["FE"]["nearby_metal_count_6a"] == 1
    assert summaries["MG"]["nearest_metal_distance"] == approx(6.01)
    assert summaries["MG"]["nearby_metal_count_6a"] == 0


@pytest.mark.parametrize(
    ("occupancy", "matches"),
    [(0.51, True), (0.52, False), (0.49, True), (0.48, False)],
)
def test_special_position_occupancy_tolerance_band(
    tmp_path: Path, occupancy: float, matches: bool
) -> None:
    """Occupancy agreement is decided by an absolute tolerance of 0.015.

    The metal sits on a two-fold axis, so the symmetry-only expectation is
    0.5: 0.51 and 0.49 agree with it, 0.52 and 0.48 do not.
    """
    assert policy.SPECIAL_POSITION_OCCUPANCY_TOLERANCE == 0.015
    builder = StructureBuilder(
        cell=(20.0, 20.0, 20.0, 90.0, 90.0, 90.0), spacegroup="P 2"
    )
    builder.add_metal("ZN", 1, pos=(0.0, 0.0, 0.0), occupancy=occupancy)
    path = builder.write_cif(tmp_path / f"occupancy_{occupancy}.cif")
    summary = helpers.analyze_bonds(path).summary

    assert summary["metal_site_symmetry_order"] == 2
    assert summary["metal_expected_crystallographic_occupancy"] == approx(0.5)
    assert summary["metal_occupancy_matches_site_symmetry"] is matches


@pytest.mark.parametrize(
    ("overshoot", "accepted"),
    [(CANDIDATE_ACCEPT_EPSILON / 2, True), (SEARCH_EPSILON / 2, False)],
)
def test_candidate_acceptance_is_tighter_than_the_search_pad(
    tmp_path: Path, overshoot: float, accepted: bool
) -> None:
    """The accept filter uses CANDIDATE_ACCEPT_EPSILON, not the search pad.

    Both waters are returned by a search cast at ``CANDIDATE_SEARCH_RADIUS +
    SEARCH_EPSILON``; only the one inside ``CANDIDATE_SEARCH_RADIUS +
    CANDIDATE_ACCEPT_EPSILON`` becomes a candidate. mmCIF keeps the full
    coordinate precision, so the two cases differ by the epsilons alone.
    """
    assert CANDIDATE_ACCEPT_EPSILON < SEARCH_EPSILON
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_water(2, (CANDIDATE_SEARCH_RADIUS + overshoot, 0.0, 0.0))
    path = builder.write_cif(tmp_path / "accept_epsilon.cif")
    analysis = helpers.analyze_bonds(path)

    assert [row["neighbor_atom"] for row in analysis.candidate_rows] == (
        ["O"] if accepted else []
    )
