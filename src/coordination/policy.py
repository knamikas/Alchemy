"""Search radii and scoring thresholds shared by the coordination stages.

Every constant here changes which contacts are discovered, assigned, or
flagged, so each is published with the rows it governs or documented in
docs/method.md. Keep them in one place so the stages cannot drift apart.
"""

from __future__ import annotations

from codes import EligibilityReason

CANDIDATE_SEARCH_RADIUS = 4.0
# Context-only radius for nearby modeled metals. This does not assert a
# metal-metal bond or define a multinuclear active site.
NEARBY_METAL_RADIUS = 6.0
SEARCH_EPSILON = 1e-6
# The proximity accept filter is deliberately tighter than the neighbor-search
# radius pad above: the search casts CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON
# and the filter keeps CANDIDATE_SEARCH_RADIUS + CANDIDATE_ACCEPT_EPSILON.
# Changing this value admits contacts between 4 + 1e-9 and 4 + 1e-6 A.
CANDIDATE_ACCEPT_EPSILON = 1e-9

# First-sphere definition: donor distance <= target distance + 0.75 A.
# Harding, M. M. (2004), Acta Cryst. D60, 849-859.
# https://doi.org/10.1107/S0907444904004081
FIRST_SPHERE_TOLERANCE = 0.75
# The published reason strings embed the tolerance; fail at import if they drift.
if not (
    EligibilityReason.DISTANCE_WITHIN_TOLERANCE.endswith(f"{FIRST_SPHERE_TOLERANCE:g}")
    and EligibilityReason.DISTANCE_EXCEEDS_TOLERANCE.endswith(
        f"{FIRST_SPHERE_TOLERANCE:g}"
    )
):
    raise ValueError(
        "EligibilityReason distance values must end in FIRST_SPHERE_TOLERANCE"
    )

# Gemmi's ContactSearch uses 0.8 A by default to distinguish near-coincident
# symmetry images of an atom intended to occupy a special position.
# NeighborSearch returns those images unfiltered, so apply the cutoff here.
SPECIAL_POSITION_DEDUP_CUTOFF = 0.8

# Coordinate occupancies are commonly rounded in PDB/mmCIF output, so allow a
# small absolute tolerance when comparing them with 1 / site-symmetry order.
SPECIAL_POSITION_OCCUPANCY_TOLERANCE = 0.015

# |z| at or above this is a geometry outlier. It is published with every bond
# and site row so older result files retain the threshold they were scored with.
ZSCORE_OUTLIER_CUTOFF = 6.0
