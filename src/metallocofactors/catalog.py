"""Load and validate the bundled metallocofactor catalog lazily."""

import os
from functools import cache
from typing import NamedTuple

from codes import ParentType
from reference_integrity import verify_checksum

CATALOG_DIR = os.path.dirname(os.path.abspath(__file__))
COFACTOR_CATALOG_PATH = os.path.join(CATALOG_DIR, "metallocofactors_id.txt")
CHECKSUM_SIDECARS = {
    COFACTOR_CATALOG_PATH: (
        os.path.join(CATALOG_DIR, "metallocofactors_id.meta.json"),
        "catalog_sha256",
    ),
}


class CofactorCatalog(NamedTuple):
    """The bundled metallocofactor catalog, split by structural class."""

    #: Every component id Alchemy treats as a metal-containing cofactor.
    ids: frozenset[str]
    #: Components whose metals sit in an iron-sulfur-style cluster.
    cluster: frozenset[str]
    #: Components whose metals sit in a heme-style macrocycle.
    heme: frozenset[str]


def _parse_cofactor_catalog(path: str) -> CofactorCatalog:
    """Return the catalog's id, cluster, and heme sets from one pass.

    Tab-separated ``id<TAB>formula<TAB>structural_class``, written by
    ``tools/build_metallocofactor_catalog.py``. The classes tag each metal's
    environment in ``parent_type``, where ``cluster`` takes precedence over
    ``heme`` for a component carrying both.
    """
    ids: set[str] = set()
    cluster: set[str] = set()
    heme: set[str] = set()
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
            if structural_class == ParentType.CLUSTER:
                cluster.add(component_id)
            elif structural_class == ParentType.HEME:
                heme.add(component_id)
    if not ids:
        raise ValueError("bundled metallocofactor catalog is empty")
    if not cluster or not heme:
        raise ValueError(
            f"{os.path.basename(path)} carries no structural classes; rebuild "
            "it with tools/build_metallocofactor_catalog.py"
        )
    return CofactorCatalog(frozenset(ids), frozenset(cluster), frozenset(heme))


@cache
def catalog(path: str = COFACTOR_CATALOG_PATH) -> CofactorCatalog:
    """Load and cache the verified metallocofactor catalog."""
    verify_checksum(path, CHECKSUM_SIDECARS)
    return _parse_cofactor_catalog(path)


def cofactor_ids(path: str = COFACTOR_CATALOG_PATH) -> frozenset[str]:
    """Every component id Alchemy treats as a metal-containing cofactor."""
    return catalog(path).ids


def cluster_ids(path: str = COFACTOR_CATALOG_PATH) -> frozenset[str]:
    """Components whose metals sit in an iron-sulfur-style cluster."""
    return catalog(path).cluster


def heme_ids(path: str = COFACTOR_CATALOG_PATH) -> frozenset[str]:
    """Components whose metals sit in a heme-style macrocycle."""
    return catalog(path).heme
