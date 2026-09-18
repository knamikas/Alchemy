"""Checksum and sidecar validation shared by bundled reference loaders."""

import hashlib
import json
import os
from collections.abc import Mapping


class ReferenceDataError(RuntimeError):
    """Bundled reference data is missing, unreadable, or inconsistent with its metadata."""


def sha256(path: str) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum(path: str, sidecars: Mapping[str, tuple[str, str]]) -> None:
    """Verify a bundled file against its recorded checksum.

    Custom paths are not checked. The checksum establishes file identity, not
    the scientific validity of its contents.
    """
    sidecar = sidecars.get(path)
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
    actual = sha256(path)
    if actual != recorded:
        raise ReferenceDataError(
            f"{os.path.basename(path)} does not match the checksum recorded in "
            f"{os.path.basename(sidecar_path)} (expected {recorded}, found "
            f"{actual}). Rebuild the file with its tool, or re-stamp the "
            "sidecar if the edit was deliberate."
        )
