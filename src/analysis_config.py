"""Deterministic identity for settings that shape scientific evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from worker_contracts import (
    ALTLOC_POLICY,
    MAX_ANALYZED_METAL_SITES,
    MODEL_POLICY,
    SYMMETRY_POLICY,
)

ANALYSIS_CONFIG_SCHEMA_VERSION = 1


def analysis_config_payload(*, reference_data_id: str) -> dict[str, Any]:
    """Return only settings that can change site evidence or cohort membership.

    Paths, worker count, logging, timeouts, caching, and other execution-only
    choices are intentionally absent: changing them must not create a new
    scientific configuration identity.
    """
    if not reference_data_id:
        raise ValueError("reference data identity is required for analysis identity")
    return {
        "schema_version": ANALYSIS_CONFIG_SCHEMA_VERSION,
        "model_policy": MODEL_POLICY,
        "altloc_policy": ALTLOC_POLICY,
        "symmetry_contact_policy": SYMMETRY_POLICY,
        "maximum_entry_metal_sites": MAX_ANALYZED_METAL_SITES,
        "reference_data_id": reference_data_id,
    }


def analysis_config_id(*, reference_data_id: str) -> str:
    """Return a stable, human-recognizable digest of the analysis payload."""
    payload = analysis_config_payload(reference_data_id=reference_data_id)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "alchemy-analysis-config-" + hashlib.sha256(encoded).hexdigest()[:20]


def analysis_configs_are_compatible(
    recorded_ids: Iterable[str], current_id: str
) -> bool:
    """Return whether existing rows can safely share the current output."""
    recorded = list(recorded_ids)
    return not recorded or {value.strip() for value in recorded} == {current_id}
