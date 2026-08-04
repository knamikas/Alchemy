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
def _catalog(path=COFACTOR_CATALOG_PATH):
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


def _load_literature(path):
    """Parse metal_distances_info.txt -> {(residue, atom, metal): (mu, stdev)}.

    Space-delimited ``residue atom metal avg_bond_dist st_dev``. The header line
    and blank separator lines are skipped naturally because their 4th/5th tokens
    do not parse as floats.
    """
    lit = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                mu, stdev = float(parts[3]), float(parts[4])
            except ValueError:
                continue  # header ("avg_bond_dist") or malformed line
            lit[(parts[0], parts[1], parts[2])] = (mu, stdev)
    return lit


@lru_cache(maxsize=None)
def literature_distances(path=DONOR_DISTANCE_PATH):
    """``{(residue, atom, metal): (mu, stdev)}`` from the literature table."""
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
