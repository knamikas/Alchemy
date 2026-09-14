"""Tests for the RCSB crystallization metadata prefetch and its on-disk cache."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crystallization_conditions import cached_rcsb_crystallization_conditions
from rcsb_metadata_cache import (
    prefetch_rcsb_crystallization_metadata,
    read_cache_payload,
)


def _graphql_response(*entries: dict[str, Any]) -> dict[str, object]:
    return {"data": {"entries": list(entries)}}


def test_rcsb_prefetch_caches_conditions_and_provenance(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cache = tmp_path / "metadata"
    calls: list[tuple[str, ...]] = []

    def fetch(ids: tuple[str, ...] | list[str]) -> dict[str, Any]:
        calls.append(tuple(ids))
        return _graphql_response(
            {
                "rcsb_id": "1ABC",
                "rcsb_accession_info": {"revision_date": "2024-03-20"},
                "exptl_crystal_grow": [
                    {
                        "crystal_id": "1",
                        "method": "VAPOR DIFFUSION",
                        "temp": 293,
                        "pH": 6.5,
                        "pdbx_pH_range": None,
                        "pdbx_details": "0.1 M zinc chloride and sulfate",
                        "temp_details": None,
                    }
                ],
            },
            {
                "rcsb_id": "2DEF",
                "rcsb_accession_info": {"revision_date": "2023-01-01"},
                "exptl_crystal_grow": None,
            },
        )

    monkeypatch.setattr("rcsb_metadata_cache._fetch_graphql_batch", fetch)
    stats = prefetch_rcsb_crystallization_metadata(
        ["1ABC", "2def"], str(cache), allow_download=True
    )

    assert calls == [("1abc", "2def")]
    assert stats.fetched == 2
    assert stats.available == 1
    assert stats.not_reported == 1
    extraction = cached_rcsb_crystallization_conditions("1abc", str(cache))
    assert extraction is not None
    assert extraction.conditions[0]["pH"] == "6.5"
    assert extraction.conditions[0]["metadata_source"] == "rcsb_data_api"
    assert extraction.conditions[0]["entry_revision_date"] == "2024-03-20"
    assert extraction.summary["crystallization_detected_metals"] == "ZN"
    assert extraction.summary["crystallization_sulfate"] is True

    cached = prefetch_rcsb_crystallization_metadata(
        ["1abc", "2def"], str(cache), allow_download=False
    )
    assert cached.cache_hits == 2
    assert cached.fetched == 0


def test_cache_files_are_sharded_compact_key_sorted_json(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The cache is a stable on-disk format other runs and tools read back."""
    cache = tmp_path / "metadata"

    def fetch(_ids: Sequence[str]) -> dict[str, object]:
        return _graphql_response(
            {
                "rcsb_id": "2DEF",
                "rcsb_accession_info": {"revision_date": "2023-01-01"},
                "exptl_crystal_grow": None,
            }
        )

    monkeypatch.setattr(
        "rcsb_metadata_cache._fetch_graphql_batch",
        fetch,
    )

    prefetch_rcsb_crystallization_metadata(["2def"], str(cache), allow_download=True)

    path = cache / "de" / "2def.json"
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert text == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    assert set(payload) == {
        "schema_version",
        "pdb_id",
        "metadata_source",
        "retrieved_at_utc",
        "entry_revision_date",
        "entry_available",
        "conditions",
    }
    assert payload["entry_available"] is True
    assert payload["conditions"] == []
    assert read_cache_payload(str(cache), "2DEF") == payload


def test_invalid_cache_is_a_safe_offline_miss(tmp_path: Path) -> None:
    cache_file = tmp_path / "metadata" / "ab" / "1abc.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")

    assert read_cache_payload(str(cache_file.parents[1]), "1abc") is None
    assert (
        cached_rcsb_crystallization_conditions("1abc", str(cache_file.parents[1]))
        is None
    )
    stats = prefetch_rcsb_crystallization_metadata(
        ["1abc"], str(cache_file.parents[1]), allow_download=False
    )
    assert stats.cache_hits == 0
    assert stats.entry_unavailable == 1
