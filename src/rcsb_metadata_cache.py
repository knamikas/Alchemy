"""Fetch original-PDB crystallization metadata from RCSB and cache it on disk.

The driver warms the cache before workers start through
``prefetch_rcsb_crystallization_metadata``; workers only read it, through
``read_cache_payload`` via ``crystallization_conditions``. One JSON file per
entry holds the raw ``exptl_crystal_grow`` records with their provenance, so
that interpretation can change without another download. Nothing here
influences confidence scores.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
#: Attempts per GraphQL batch before the prefetch reports the batch failed.
RCSB_REQUEST_ATTEMPTS = 3
RCSB_REQUEST_TIMEOUT_S = 60
#: Exponential back-off base between attempts, in seconds.
RCSB_RETRY_BACKOFF_BASE_S = 2
RCSB_BATCH_SIZE = 200
RCSB_GRAPHQL_QUERY = """
query CrystallizationConditions($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { revision_date }
    exptl_crystal_grow {
      crystal_id
      method
      temp
      pH
      pdbx_pH_range
      pdbx_details
      temp_details
    }
  }
}
""".strip()


@dataclass(frozen=True, slots=True)
class CrystallizationPrefetchStats:
    """Count outcomes from prefetching original-PDB condition metadata.

    ``entry_unavailable`` counts entries RCSB reported it does not have;
    ``not_fetched`` counts entries never asked for because downloads were off.
    """

    requested: int
    cache_hits: int
    fetched: int
    available: int
    not_reported: int
    entry_unavailable: int
    not_fetched: int


class CrystallizationMetadataError(RuntimeError):
    """Original-PDB condition metadata could not be fetched or cached safely."""


def _cache_path(cache_root: str, pdb_id: str) -> str:
    """Locate an entry's cache file, sharded by the middle two ID characters."""
    pdb_id = pdb_id.strip().lower()
    return os.path.join(cache_root, pdb_id[1:3], f"{pdb_id}.json")


def _validated_cache_payload(pdb_id: str, value: object) -> dict[str, Any] | None:
    """Accept a loaded payload only when it has this schema and names this entry."""
    if not isinstance(value, dict):
        return None
    payload = cast("dict[str, Any]", value)
    if str(payload.get("pdb_id", "")).lower() != pdb_id:
        return None
    if not isinstance(payload.get("entry_available"), bool):
        return None
    if not isinstance(payload.get("conditions"), list):
        return None
    if not all(isinstance(condition, dict) for condition in payload["conditions"]):
        return None
    for key in ("metadata_source", "retrieved_at_utc", "entry_revision_date"):
        if not isinstance(payload.get(key), str):
            return None
    return payload


def read_cache_payload(cache_root: str, pdb_id: str) -> dict[str, Any] | None:
    """Return the validated cached payload for one entry, or ``None`` on a miss.

    Any unreadable or malformed file, or one naming another entry, is a miss.
    """
    pdb_id = pdb_id.strip().lower()
    path = _cache_path(cache_root, pdb_id)
    try:
        with open(path, encoding="utf-8") as handle:
            return _validated_cache_payload(pdb_id, json.load(handle))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache_payload(cache_root: str, payload: Mapping[str, Any]) -> None:
    """Write one payload atomically as compact, key-sorted JSON.

    A temporary file is renamed into place so a reader never sees a partial
    file, and is removed again when the write fails.
    """
    pdb_id = str(payload["pdb_id"])
    path = _cache_path(cache_root, pdb_id)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _entries_from_response(loaded: object) -> list[object]:
    """Extract ``data.entries`` from a GraphQL response, or refuse it."""
    if not isinstance(loaded, dict):
        raise CrystallizationMetadataError("RCSB response is not a JSON object")
    result = cast("dict[str, object]", loaded)
    if result.get("errors"):
        raise CrystallizationMetadataError(f"GraphQL errors: {result['errors']!r}")
    data_value = result.get("data")
    if not isinstance(data_value, dict):
        raise CrystallizationMetadataError("RCSB response has no data object")
    entries_value = cast("dict[str, object]", data_value).get("entries")
    if not isinstance(entries_value, list):
        raise CrystallizationMetadataError("RCSB response has no data.entries list")
    return cast("list[object]", entries_value)


def _fetch_graphql_batch(pdb_ids: Sequence[str]) -> list[object]:
    """Query RCSB for one batch of entries and return its ``data.entries`` list.

    Network failures, server errors, and undecodable bodies are retried with
    exponential back-off. A rejected request or a malformed reply would fail
    the same way again, so those raise ``CrystallizationMetadataError`` at
    once.
    """
    body = json.dumps(
        {
            "query": RCSB_GRAPHQL_QUERY,
            "variables": {"ids": [pdb_id.upper() for pdb_id in pdb_ids]},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        RCSB_GRAPHQL_URL,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Alchemy crystallization metadata cache",
        },
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(RCSB_REQUEST_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                request, timeout=RCSB_REQUEST_TIMEOUT_S
            ) as response:
                loaded: object = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise CrystallizationMetadataError(
                    f"RCSB Data API rejected the request: HTTP {exc.code} {exc.reason}"
                ) from None
            last_error = exc
        except (OSError, ValueError) as exc:
            # A network error, or a body cut short before it could be decoded.
            last_error = exc
        else:
            return _entries_from_response(loaded)
        if attempt < RCSB_REQUEST_ATTEMPTS - 1:
            time.sleep(RCSB_RETRY_BACKOFF_BASE_S**attempt)
    raise CrystallizationMetadataError(
        f"RCSB Data API request failed after {RCSB_REQUEST_ATTEMPTS} attempts: "
        f"{last_error}"
    )


def _payloads_from_graphql(
    pdb_ids: Sequence[str], entries: Sequence[object], retrieved_at_utc: str
) -> list[dict[str, object]]:
    """Build one cache payload per requested entry from a batch's entries.

    An entry RCSB did not return is recorded as unavailable, so that a later
    run does not ask for it again.
    """
    by_id: dict[str, Mapping[str, object]] = {}
    for value in entries:
        if not isinstance(value, dict):
            continue
        entry = cast("Mapping[str, object]", value)
        pdb_id = str(entry.get("rcsb_id", "")).strip().lower()
        if pdb_id:
            by_id[pdb_id] = entry
    payloads: list[dict[str, object]] = []
    for pdb_id in pdb_ids:
        matching_entry = by_id.get(pdb_id)
        accession_value = (
            matching_entry.get("rcsb_accession_info") if matching_entry else None
        )
        accession = (
            cast("dict[str, object]", accession_value)
            if isinstance(accession_value, dict)
            else {}
        )
        revision_value = accession.get("revision_date")
        revision = str(revision_value) if revision_value is not None else ""
        conditions_value: object = (
            matching_entry.get("exptl_crystal_grow") if matching_entry else None
        )
        if conditions_value is None:
            condition_values: list[object] = []
        elif isinstance(conditions_value, list):
            condition_values = cast("list[object]", conditions_value)
        else:
            raise CrystallizationMetadataError(
                f"RCSB Data API returned invalid conditions for {pdb_id}"
            )
        if not all(isinstance(condition, dict) for condition in condition_values):
            raise CrystallizationMetadataError(
                f"RCSB Data API returned invalid conditions for {pdb_id}"
            )
        conditions = [
            cast("dict[str, object]", condition) for condition in condition_values
        ]
        payloads.append(
            {
                "pdb_id": pdb_id,
                "metadata_source": "rcsb_data_api",
                "retrieved_at_utc": retrieved_at_utc,
                "entry_revision_date": revision,
                "entry_available": matching_entry is not None,
                "conditions": conditions,
            }
        )
    return payloads


def prefetch_rcsb_crystallization_metadata(
    pdb_ids: Iterable[str], cache_root: str, *, allow_download: bool
) -> CrystallizationPrefetchStats:
    """Populate the persistent RCSB cache before worker processes start.

    Entries already cached are not fetched again. With ``allow_download`` off,
    missing entries stay missing and count as unavailable in the returned
    statistics.
    """
    ids = tuple(dict.fromkeys(pdb_id.strip().lower() for pdb_id in pdb_ids))
    payloads: dict[str, Mapping[str, Any] | None] = {
        pdb_id: read_cache_payload(cache_root, pdb_id) for pdb_id in ids
    }
    missing = [pdb_id for pdb_id, payload in payloads.items() if payload is None]
    fetched = 0
    if allow_download:
        for start in range(0, len(missing), RCSB_BATCH_SIZE):
            batch = missing[start : start + RCSB_BATCH_SIZE]
            entries = _fetch_graphql_batch(batch)
            retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
            for payload in _payloads_from_graphql(batch, entries, retrieved_at):
                try:
                    _write_cache_payload(cache_root, payload)
                except OSError as exc:
                    raise CrystallizationMetadataError(
                        f"could not write crystallization metadata cache: {exc}"
                    ) from None
                payloads[str(payload["pdb_id"])] = payload
                fetched += 1
    available = not_reported = unavailable = not_fetched = 0
    for cached in payloads.values():
        if cached is None:
            not_fetched += 1
        elif not cached["entry_available"]:
            unavailable += 1
        elif cached["conditions"]:
            available += 1
        else:
            not_reported += 1
    return CrystallizationPrefetchStats(
        requested=len(ids),
        cache_hits=len(ids) - len(missing),
        fetched=fetched,
        available=available,
        not_reported=not_reported,
        entry_unavailable=unavailable,
        not_fetched=not_fetched,
    )
