"""The bundled reference data, and the only code that reads it.

Two files travel with the checkout under ``src/data/``: the metallocofactor
catalog, which names every component Alchemy treats as a metal cofactor and
tags the clusters and hemes among them, and the literature table of
metal-donor distances that every z-score is measured against.

Nothing here runs at import. Each accessor is ``lru_cache``d on its path, so
the files are read the first time a value is actually wanted and never again,
and the results are frozen -- ``frozenset`` and ``MappingProxyType`` -- because
they are process-wide and shared by every worker.

That combination is the point of this module. Previously ``import
bond_analysis`` read both files, which meant ``--help``, a test-collection
pass, and every spawned worker paid for them; a malformed catalog raised
``ValueError`` out of an import statement, where there is no context to report
it with; and the results were mutable dicts one bad caller could edit for the
whole process. The catalog was also parsed twice by two parsers that disagreed
-- one required three tab-separated fields, the other accepted a single column
-- so a legacy file loaded cleanly in one module and hard-failed in the other.
One parser, one pass, one set of rules.
"""

import hashlib
import json
import math
import os
from functools import lru_cache
from types import MappingProxyType


# Single source for the bundled reference-data directory. Every file under it
# is named relative to this constant so the location is defined once; the
# directory travels with the checkout, which is how Alchemy is distributed.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

COFACTOR_CATALOG_PATH = os.path.join(DATA_DIR, "metallocofactors_id.txt")
DONOR_DISTANCE_PATH = os.path.join(DATA_DIR, "metal_distances_info.txt")

#: Each bundled file's metadata sidecar, and the key inside it holding the
#: SHA-256 of the file itself. Both are written by the tool that produces the
#: file: ``tools/build_metallocofactor_catalog.py`` for the catalog,
#: ``tools/stamp_distance_table.py`` for the literature distances.
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

    Only the two bundled paths are checked. A caller who passes a path of their
    own owns it -- the tests do this, and so would anyone running against a
    catalog they built themselves -- but the files that travel with the
    checkout are the ones every entry is measured against, and a hand edit to
    either silently changes results for every run afterwards with nothing in
    the output saying so.

    This is identity, not correctness: it says the file is the one the tool
    wrote, not that the tool was right.
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
    ``tools/build_metallocofactor_catalog.py``. The structural classes are
    derived from CCD connectivity when the catalog is built, so they track the
    CCD rather than drifting behind a list maintained by hand here. They tag
    each metal's environment in ``parent_type``; ``cluster`` takes precedence
    over ``heme`` in ``_parent_type`` for a component present in both.

    One parser rather than two. The ids and the classes come out of the same
    file, and reading it twice under different rules is what let a catalog be
    simultaneously valid and invalid depending on which module asked.
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
    """``{filename: sha256}`` for both bundled files, verified as it goes.

    The full hashes, for a run log or anyone asking *which* file differs.
    """
    return MappingProxyType(
        {os.path.basename(path): _verified_sha256(path) for path in CHECKSUM_SIDECARS}
    )


@lru_cache(maxsize=None)
def reference_data_id():
    """One short id for the reference data an entry was measured against.

    Both bundled files decide results -- the catalog decides what counts as a
    metal cofactor, the distance table sets every assignment cutoff and every
    z-score -- so a row is comparable with another only if both matched. One
    id makes that a grouping key rather than a two-column join, and the pieces
    stay recoverable from the sidecars and from the run log.

    Composed from the file hashes rather than from the sidecars' recorded
    values: the id then describes what was actually read.
    """
    checksums = reference_data_checksums()
    digest = hashlib.sha256()
    for name in sorted(checksums):
        digest.update(f"{name}:{checksums[name]}\n".encode())
    # Twelve characters, matching the abbreviated ``alchemy_commit`` beside it
    # in the manifest. This identifies a build, it does not authenticate one.
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


#: The literature table's own first line, skipped by name rather than by
#: failing to parse. Recognizing it explicitly is what lets every *other*
#: unparseable line be an error.
DISTANCE_TABLE_HEADER = ("residue", "atom", "metal", "avg_bond_dist", "st_dev")


def _load_literature(path):
    """Parse metal_distances_info.txt -> {(residue, atom, metal): (mu, stdev)}.

    Space-delimited ``residue atom metal avg_bond_dist st_dev``, one row per
    metal-donor pair, with blank lines separating the blocks. Column 1 is a
    residue name **except** ``CA``, which is the backbone-carbonyl pseudo
    residue; column 3 ``CA`` is calcium. The two are unrelated and the file
    says so nowhere -- see ``_bonding_key``, which builds the key.

    Every non-blank line that is not the header must parse. A row that does not
    used to be skipped in silence, which meant a typo in a reference distance
    disabled the z-score for that metal-donor pair and reported nothing: the
    contact simply became one no reference covered, which is a legitimate
    outcome for a pair genuinely absent from the literature. Refusing the file
    is the only way to tell those two apart.
    """
    lit = {}
    with open(path) as f:
        for number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if tuple(parts) == DISTANCE_TABLE_HEADER:
                continue
            if len(parts) != 5:
                # Exactly five, not "at least": a sixth field used to be
                # ignored, and a row that gained one still parsed -- silently
                # replacing that metal-donor pair's real numbers with whatever
                # the damaged line held.
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
            lit[(parts[0], parts[1], parts[2])] = (mu, stdev)
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
    """``{(metal_element, donor_element): longest reference distance}``.

    Built inside a function rather than by a loop at module scope, which used
    to leave its four loop variables behind in the module namespace.
    """
    targets = {}
    for (_, donor, metal_element), (target, _) in literature_distances(path).items():
        key = (metal_element, donor)
        targets[key] = max(target, targets.get(key, -math.inf))
    return MappingProxyType(targets)
