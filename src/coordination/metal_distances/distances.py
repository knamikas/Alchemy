"""Load and validate literature distances and cache immutable lookups."""

import math
import os
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from reference_integrity import verify_checksum

DISTANCE_DIR = os.path.dirname(os.path.abspath(__file__))
DONOR_DISTANCE_PATH = os.path.join(DISTANCE_DIR, "metal_distances_info.txt")
CHECKSUM_SIDECARS = {
    DONOR_DISTANCE_PATH: (
        os.path.join(DISTANCE_DIR, "metal_distances_info.meta.json"),
        "distance_table_sha256",
    ),
}


# Skipped by name rather than by failing to parse, which is what lets every
# other unparseable line be an error.
DISTANCE_TABLE_HEADER = ("residue", "atom", "metal", "avg_bond_dist", "st_dev")


def load_literature(
    path: str,
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Load literature distances as {(residue, atom, metal): (mean, stdev)}.

    The five columns are residue, atom, metal, mean distance, and standard
    deviation. In the residue column, CA means backbone carbonyl; in the metal
    column, CA means calcium. See _bonding_key.

    Reject malformed rows so damaged data cannot appear as missing references.
    """
    lit: dict[tuple[str, str, str], tuple[float, float]] = {}
    first_line_by_key: dict[tuple[str, str, str], int] = {}
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if tuple(parts) == DISTANCE_TABLE_HEADER:
                continue
            if len(parts) != 5:
                # Reject extra fields as well as missing ones to detect damaged rows.
                raise ValueError(
                    f"{os.path.basename(path)} line {number} has "
                    f"{len(parts)} fields, expected 5: {line.strip()!r}"
                )
            try:
                mu, stdev = float(parts[3]), float(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"{os.path.basename(path)} line {number} has non-numeric "
                    f"distance columns: {line.strip()!r}"
                ) from exc
            if not (math.isfinite(mu) and math.isfinite(stdev)):
                raise ValueError(
                    f"{os.path.basename(path)} line {number} has non-finite "
                    f"distance columns: {line.strip()!r}"
                )
            if mu <= 0.0 or stdev <= 0.0:
                raise ValueError(
                    f"{os.path.basename(path)} line {number} has non-positive "
                    f"distance columns: {line.strip()!r}"
                )
            key = (parts[0], parts[1], parts[2])
            if key in lit:
                first_line = first_line_by_key[key]
                raise ValueError(
                    f"{os.path.basename(path)} line {number} duplicates "
                    f"reference key {' '.join(key)!r} first defined on line "
                    f"{first_line}"
                )
            lit[key] = (mu, stdev)
            first_line_by_key[key] = number
    if not lit:
        raise ValueError(f"{os.path.basename(path)} carries no reference distances")
    return lit


@cache
def literature_distances(
    path: str = DONOR_DISTANCE_PATH,
) -> Mapping[tuple[str, str, str], tuple[float, float]]:
    """``{(residue, atom, metal): (mu, stdev)}`` from the literature table."""
    verify_checksum(path, CHECKSUM_SIDECARS)
    return MappingProxyType(load_literature(path))


@cache
def first_sphere_targets(
    path: str = DONOR_DISTANCE_PATH,
) -> Mapping[tuple[str, str], float]:
    """``{(metal_element, donor_element): longest reference distance}``."""
    targets: dict[tuple[str, str], float] = {}
    for (_, donor, metal_element), (target, _) in literature_distances(path).items():
        key = (metal_element, donor)
        targets[key] = max(target, targets.get(key, -math.inf))
    return MappingProxyType(targets)
