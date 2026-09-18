"""Verify and identify the combined bundled catalog and distance references."""

import hashlib
import os
from collections.abc import Mapping
from functools import cache
from types import MappingProxyType

from coordination.metal_distances import distances
from metallocofactors import catalog
from reference_integrity import sha256, verify_checksum

CHECKSUM_SIDECARS = {**catalog.CHECKSUM_SIDECARS, **distances.CHECKSUM_SIDECARS}


@cache
def reference_data_checksums() -> Mapping[str, str]:
    """``{filename: sha256}`` for both bundled files, verified as it goes."""
    return MappingProxyType(
        {os.path.basename(path): _verified_sha256(path) for path in CHECKSUM_SIDECARS}
    )


@cache
def reference_data_id() -> str:
    """Return a reference-data ID derived from both files' current hashes."""
    checksums = reference_data_checksums()
    digest = hashlib.sha256()
    for name in sorted(checksums):
        digest.update(f"{name}:{checksums[name]}\n".encode())
    # A compact reference identity for provenance, not authentication.
    return digest.hexdigest()[:12]


def _verified_sha256(path: str) -> str:
    verify_checksum(path, CHECKSUM_SIDECARS)
    return sha256(path)
