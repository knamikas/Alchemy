#!/usr/bin/env python3
"""Record the distance table's checksum, row count, and source citations.

Run after editing src/coordination/metal_distances/metal_distances_info.txt.
Use --check to verify the existing metadata without rewriting it.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(REPO_ROOT, "src")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from coordination.metal_distances.distances import (  # noqa: E402
    CHECKSUM_SIDECARS,
    DONOR_DISTANCE_PATH,
    load_literature,
)
from reference_integrity import sha256  # noqa: E402

# The runtime loader owns the bundled path and the sidecar key it verifies.
TABLE_PATH = DONOR_DISTANCE_PATH
SIDECAR_PATH, HASH_KEY = CHECKSUM_SIDECARS[DONOR_DISTANCE_PATH]

#: Published in the sidecar's ``sources``; the table carries no in-band citation.
SOURCES = [
    {
        "citation": (
            "Harding, M. M. (2006). Small revisions to predicted distances "
            "around metal sites in proteins. Acta Cryst. D62, 678-682."
        ),
        "doi": "10.1107/S0907444906014594",
        "covers": "every metal except Ni",
    },
    {
        "citation": (
            "Zheng, H., Chruszcz, M., Lasota, P., Lebioda, L. & Minor, W. "
            "(2008). Data mining of metal ion environments present in protein "
            "structures. J. Inorg. Biochem. 102, 1765-1776."
        ),
        "doi": "10.1016/j.jinorgbio.2008.05.006",
        "covers": "Ni",
    },
]


def count_rows(path: str) -> int:
    """Count the rows through the loader, so the count matches what it accepts."""
    return len(load_literature(path))


def build_metadata(path: str, generated: str) -> dict[str, object]:
    """Build the distance-table provenance sidecar payload."""
    return {
        "generated": generated,
        HASH_KEY: sha256(path),
        "reference_distance_count": count_rows(path),
        "sources": SOURCES,
        "format": (
            "Space-delimited: residue atom metal avg_bond_dist st_dev. Blank "
            "lines separate blocks. Column 1 'CA' is the backbone-carbonyl "
            "pseudo residue, not calcium; column 3 'CA' is calcium."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for stamping or checking the sidecar."""
    parser = argparse.ArgumentParser(
        description="Record what src/coordination/metal_distances/metal_distances_info.txt currently is."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the sidecar matches the table; write nothing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Stamp or verify the distance-table sidecar and return an exit status."""
    args = parse_args(argv)
    actual = sha256(TABLE_PATH)
    if args.check:
        try:
            with open(SIDECAR_PATH, encoding="utf-8") as handle:
                recorded = json.load(handle).get(HASH_KEY)
        except OSError:
            print(f"no sidecar at {SIDECAR_PATH}")
            return 1
        except ValueError as exc:
            # JSONDecodeError is a ValueError; report truncated sidecars without a traceback.
            print(f"unreadable sidecar at {SIDECAR_PATH}: {exc}")
            return 1
        if recorded != actual:
            print(f"stale sidecar: recorded {recorded}, table is {actual}")
            return 1
        print(f"sidecar matches: {actual}")
        return 0

    generated = datetime.now(UTC).isoformat()
    metadata = build_metadata(TABLE_PATH, generated)
    with open(SIDECAR_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(
        f"stamped {os.path.basename(TABLE_PATH)}: "
        f"{metadata['reference_distance_count']} distances, "
        f"sha256 {metadata[HASH_KEY]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
