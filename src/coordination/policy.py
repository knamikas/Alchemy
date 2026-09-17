"""Search radii and scoring thresholds shared by the coordination stages.

Every threshold here changes which contacts are discovered, assigned, or
flagged, so each is published with the rows it governs or documented in
docs/method.md or docs/output-schema.md. The three exceptions are
``SEARCH_EPSILON``, ``CANDIDATE_ACCEPT_EPSILON`` and ``SAME_IMAGE_TOLERANCE``:
they are internal floating-point comparison pads, far below any resolvable
displacement, and are neither published in a row nor documented in the prose.
Keep them all in one place so the stages cannot drift apart.
"""

from __future__ import annotations

from codes import EligibilityReason

CANDIDATE_SEARCH_RADIUS = 4.0
# Context-only radius for nearby modeled metals. This does not assert a
# metal-metal bond or define a multinuclear active site.
NEARBY_METAL_RADIUS = 6.0
# A float-comparison pad with two roles. It widens the neighbor searches so an
# atom sitting exactly on a radius is still returned (analysis.py and
# candidates.py cast CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON), and it is the
# accept tolerance of two distance comparisons: the first-sphere cutoff in
# eligibility.py and the NEARBY_METAL_RADIUS count in site_environment.py.
SEARCH_EPSILON = 1e-6
# The proximity accept filter is deliberately tighter than the neighbor-search
# radius pad above: the search casts CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON
# and the filter keeps CANDIDATE_SEARCH_RADIUS + CANDIDATE_ACCEPT_EPSILON.
# Changing this value admits contacts between 4 + 1e-9 and 4 + 1e-6 A.
CANDIDATE_ACCEPT_EPSILON = 1e-9
# Two candidate records for the same deposited atom, symmetry operation, and
# lattice translation describe one point exactly; any difference between them
# is arithmetic noise. Declaration resolution reaches that point through
# ``fract_image``/``orthogonalize`` and proximity discovery through
# ``find_nearest_pbc_position``, which differ in the last bits (~1e-12 A on the
# coordinates a deposited cell reaches), so an exact or bucketed comparison can
# split one physical contact into two rows. 1e-6 A is far above that noise and
# far below any resolvable displacement, so it cannot merge images a
# crystallographer would report separately.
SAME_IMAGE_TOLERANCE = 1e-6

# First-sphere definition: donor distance <= target distance + 0.75 A.
# Harding, M. M. (2004), Acta Cryst. D60, 849-859.
# https://doi.org/10.1107/S0907444904004081
FIRST_SPHERE_TOLERANCE = 0.75


def _check_reason_strings(tolerance: float) -> None:
    """Fail if the published reason strings no longer spell ``tolerance``.

    The whole value is compared rather than its suffix: a suffix test passes
    for any tolerance the published string happens to end in, so 5 and 75
    would both satisfy ``"..._plus_0.75".endswith(...)`` while the rows kept
    saying 0.75.
    """
    drifted = [
        f"{reason.name}: {reason.value!r} should be {expected!r}"
        for reason, expected in (
            (
                EligibilityReason.DISTANCE_WITHIN_TOLERANCE,
                f"distance_within_target_plus_{tolerance:g}",
            ),
            (
                EligibilityReason.DISTANCE_EXCEEDS_TOLERANCE,
                f"distance_exceeds_target_plus_{tolerance:g}",
            ),
        )
        if reason.value != expected
    ]
    if drifted:
        raise ValueError(
            "EligibilityReason distance values must spell FIRST_SPHERE_TOLERANCE "
            f"({tolerance:g}): {'; '.join(drifted)}"
        )


# The published reason strings embed the tolerance; fail at import if they drift.
_check_reason_strings(FIRST_SPHERE_TOLERANCE)


# Gemmi's ContactSearch uses 0.8 A by default to distinguish near-coincident
# symmetry images of an atom intended to occupy a special position.
# NeighborSearch returns those images unfiltered, so apply the cutoff here.
# Two stages consume it: candidates.py collapses candidate images closer than
# this to one another, and site_environment.py passes it to gemmi's
# UnitCell.is_special_position, where it sets the reported site-symmetry order
# and so the expected crystallographic occupancy.
SPECIAL_POSITION_DEDUP_CUTOFF = 0.8

# Coordinate occupancies are commonly rounded in PDB/mmCIF output, so allow a
# small absolute tolerance when comparing them with 1 / site-symmetry order.
SPECIAL_POSITION_OCCUPANCY_TOLERANCE = 0.015

# |z| at or above this is a geometry outlier. It is published with every bond
# and site row so older result files retain the threshold they were scored with.
ZSCORE_OUTLIER_CUTOFF = 6.0
