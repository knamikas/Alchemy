"""The bundled reference data under ``src/data/``, and the only code reading it.

Nothing here runs at import: a malformed file must raise where a caller can
report it, not out of an import statement. Results are frozen because they are
process-wide and shared by every worker.
"""

import hashlib
import json
import math
import os
from functools import lru_cache
from types import MappingProxyType


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

COFACTOR_CATALOG_PATH = os.path.join(DATA_DIR, "metallocofactors_id.txt")
DONOR_DISTANCE_PATH = os.path.join(DATA_DIR, "metal_distances_info.txt")

# Each bundled file's sidecar and the key holding its SHA-256, both written by
# the tool that produces the file: tools/build_metallocofactor_catalog.py for
# the catalog, tools/stamp_distance_table.py for the literature distances.
CHECKSUM_SIDECARS = {
    COFACTOR_CATALOG_PATH: (
        os.path.join(DATA_DIR, "metallocofactors_id.meta.json"),
        "catalog_sha256",
    ),
    DONOR_DISTANCE_PATH: (
        os.path.join(DATA_DIR, "metal_distances_info.meta.json"),
        "distance_table_sha256",
    ),
}


class ReferenceDataError(RuntimeError):
    """Bundled reference data is missing, unreadable, or not what it claims."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checksum(path):
    """Fail unless a bundled file still hashes to what its sidecar recorded.

    Only the two bundled paths are checked; a caller supplying their own path
    owns it. This is identity, not correctness: it says the file is the one the
    tool wrote, not that the tool was right.
    """
    sidecar = CHECKSUM_SIDECARS.get(path)
    if sidecar is None:
        return
    sidecar_path, key = sidecar
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            recorded = json.load(handle).get(key)
    except OSError as exc:
        raise ReferenceDataError(
            f"{os.path.basename(path)} has no metadata sidecar at "
            f"{sidecar_path}; bundled reference data must be verifiable"
        ) from exc
    except ValueError as exc:
        raise ReferenceDataError(
            f"{os.path.basename(sidecar_path)} is not readable JSON"
        ) from exc
    if not recorded:
        raise ReferenceDataError(f"{os.path.basename(sidecar_path)} records no {key}")
    actual = _sha256(path)
    if actual != recorded:
        raise ReferenceDataError(
            f"{os.path.basename(path)} does not match the checksum recorded in "
            f"{os.path.basename(sidecar_path)} (expected {recorded}, found "
            f"{actual}). Rebuild the file with its tool, or re-stamp the "
            "sidecar if the edit was deliberate."
        )


def _parse_cofactor_catalog(path):
    """Return ``(component_ids, cluster_ids, heme_ids)`` from one pass.

    Tab-separated ``id<TAB>formula<TAB>structural_class``, written by
    ``tools/build_metallocofactor_catalog.py``. The classes tag each metal's
    environment in ``parent_type``, where ``cluster`` takes precedence over
    ``heme`` for a component carrying both.
    """
    ids, cluster, heme = set(), set(), set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            component_id = fields[0].strip()
            if not component_id:
                continue
            ids.add(component_id)
            if len(fields) < 3:
                continue
            structural_class = fields[2].strip()
            if structural_class == "cluster":
                cluster.add(component_id)
            elif structural_class == "heme":
                heme.add(component_id)
    if not ids:
        raise ValueError("bundled metallocofactor catalog is empty")
    if not cluster or not heme:
        raise ValueError(
            f"{os.path.basename(path)} carries no structural classes; rebuild "
            "it with tools/build_metallocofactor_catalog.py"
        )
    return frozenset(ids), frozenset(cluster), frozenset(heme)


@lru_cache(maxsize=None)
def reference_data_checksums():
    """``{filename: sha256}`` for both bundled files, verified as it goes."""
    return MappingProxyType(
        {os.path.basename(path): _verified_sha256(path) for path in CHECKSUM_SIDECARS}
    )


@lru_cache(maxsize=None)
def reference_data_id():
    """One short id for the reference data an entry was measured against.

    Both files decide results, so two rows are comparable only if both matched.
    Composed from the file hashes rather than the sidecars' recorded values, so
    the id describes what was actually read.
    """
    checksums = reference_data_checksums()
    digest = hashlib.sha256()
    for name in sorted(checksums):
        digest.update(f"{name}:{checksums[name]}\n".encode())
    # Twelve characters, matching the abbreviated alchemy_commit beside it in
    # the manifest. This identifies a build, it does not authenticate one.
    return digest.hexdigest()[:12]


def _verified_sha256(path):
    _verify_checksum(path)
    return _sha256(path)


@lru_cache(maxsize=None)
def _catalog(path=COFACTOR_CATALOG_PATH):
    _verify_checksum(path)
    return _parse_cofactor_catalog(path)


def cofactor_ids(path=COFACTOR_CATALOG_PATH):
    """Every component id Alchemy treats as a metal-containing cofactor."""
    return _catalog(path)[0]


def cluster_ids(path=COFACTOR_CATALOG_PATH):
    """Components whose metals sit in an iron-sulfur-style cluster."""
    return _catalog(path)[1]


def heme_ids(path=COFACTOR_CATALOG_PATH):
    """Components whose metals sit in a heme-style macrocycle."""
    return _catalog(path)[2]


# Skipped by name rather than by failing to parse, which is what lets every
# other unparseable line be an error.
DISTANCE_TABLE_HEADER = ("residue", "atom", "metal", "avg_bond_dist", "st_dev")


def _load_literature(path):
    """Parse metal_distances_info.txt -> {(residue, atom, metal): (mu, stdev)}.

    Space-delimited ``residue atom metal avg_bond_dist st_dev``, one row per
    metal-donor pair. Column 1 is a residue name **except** ``CA``, the
    backbone-carbonyl pseudo residue; column 3 ``CA`` is calcium. The two are
    unrelated and the file says so nowhere -- see ``_bonding_key``.

    Every non-blank, non-header line must parse. Skipping a malformed row would
    disable the z-score for its pair, which is indistinguishable in the output
    from a pair genuinely absent from the literature.
    """
    lit = {}
    first_line_by_key = {}
    with open(path) as f:
        for number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if tuple(parts) == DISTANCE_TABLE_HEADER:
                continue
            if len(parts) != 5:
                # Exactly five, not "at least": a damaged row that gained a
                # field would otherwise parse, silently replacing that pair's
                # real numbers with whatever the damaged line held.
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


@lru_cache(maxsize=None)
def literature_distances(path=DONOR_DISTANCE_PATH):
    """``{(residue, atom, metal): (mu, stdev)}`` from the literature table."""
    _verify_checksum(path)
    return MappingProxyType(_load_literature(path))


@lru_cache(maxsize=None)
def first_sphere_targets(path=DONOR_DISTANCE_PATH):
    """``{(metal_element, donor_element): longest reference distance}``."""
    targets: dict[tuple[str, str], float] = {}
    for (_, donor, metal_element), (target, _) in literature_distances(path).items():
        key = (metal_element, donor)
        targets[key] = max(target, targets.get(key, -math.inf))
    return MappingProxyType(targets)
