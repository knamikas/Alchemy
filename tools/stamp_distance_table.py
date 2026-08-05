#!/usr/bin/env python3
"""Record what ``src/data/metal_distances_info.txt`` currently is.

Writes the sidecar ``src/reference_data`` verifies at load: the table's
SHA-256, its row count, and the citations the distances come from. Run it
after every hand edit of the table.

    python tools/stamp_distance_table.py            # rewrite the sidecar
    python tools/stamp_distance_table.py --check    # verify, change nothing
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, UTC


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "src", "data")
TABLE_PATH = os.path.join(DATA_DIR, "metal_distances_info.txt")
SIDECAR_PATH = os.path.join(DATA_DIR, "metal_distances_info.meta.json")

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


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path: str) -> int:
    """Count the rows through the loader, so the count matches what it accepts."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from reference_data import _load_literature

    return len(_load_literature(path))


def build_metadata(path: str, generated: str) -> dict[str, object]:
    return {
        "generated": generated,
        "distance_table_sha256": sha256(path),
        "reference_distance_count": count_rows(path),
        "sources": SOURCES,
        "format": (
            "Space-delimited: residue atom metal avg_bond_dist st_dev. Blank "
            "lines separate blocks. Column 1 'CA' is the backbone-carbonyl "
            "pseudo residue, not calcium; column 3 'CA' is calcium."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record what src/data/metal_distances_info.txt currently is."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the sidecar matches the table; write nothing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    actual = sha256(TABLE_PATH)
    if args.check:
        try:
            with open(SIDECAR_PATH, encoding="utf-8") as handle:
                recorded = json.load(handle).get("distance_table_sha256")
        except OSError:
            print(f"no sidecar at {SIDECAR_PATH}")
            return 1
        except ValueError as exc:
            # ``json.JSONDecodeError`` is a ValueError. A sidecar truncated by
            # an interrupted write is exactly what --check exists to report, so
            # it must not escape as a traceback.
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
        f"sha256 {metadata['distance_table_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
