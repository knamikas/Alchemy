"""Tests for the RCSB crystallization metadata prefetch and its on-disk cache."""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Sequence
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

import rcsb_metadata_cache
from crystallization_conditions import cached_rcsb_crystallization_conditions
from rcsb_metadata_cache import (
    CrystallizationMetadataError,
    prefetch_rcsb_crystallization_metadata,
    read_cache_payload,
)


def _graphql_response(*entries: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``data.entries`` list a fetched batch yields."""
    return list(entries)


def test_rcsb_prefetch_caches_conditions_and_provenance(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cache = tmp_path / "metadata"
    calls: list[tuple[str, ...]] = []

    def fetch(ids: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
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

    def fetch(_ids: Sequence[str]) -> list[dict[str, Any]]:
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


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"pdb_id": "2def"},
        {"entry_available": "yes"},
        {"conditions": "invalid"},
        {"conditions": ["invalid"]},
        {"metadata_source": None},
        {"retrieved_at_utc": 123},
        {"entry_revision_date": None},
    ],
)
def test_invalid_cache_is_a_safe_offline_miss(
    tmp_path: Path, invalid_fields: dict[str, Any]
) -> None:
    cache_file = tmp_path / "metadata" / "ab" / "1abc.json"
    cache_file.parent.mkdir(parents=True)
    payload: dict[str, Any] = {
        "pdb_id": "1abc",
        "entry_available": True,
        "conditions": [],
        "metadata_source": "rcsb_data_api",
        "retrieved_at_utc": "2026-09-18T12:00:00+00:00",
        "entry_revision_date": "2024-03-20",
        **invalid_fields,
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    assert read_cache_payload(str(cache_file.parents[1]), "1abc") is None
    assert (
        cached_rcsb_crystallization_conditions("1abc", str(cache_file.parents[1]))
        is None
    )
    stats = prefetch_rcsb_crystallization_metadata(
        ["1abc"], str(cache_file.parents[1]), allow_download=False
    )
    assert stats.cache_hits == 0
    assert stats.entry_unavailable == 0
    assert stats.not_fetched == 1


def test_offline_prefetch_counts_misses_as_not_fetched_not_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Never asking RCSB is not the same as RCSB having no such entry."""
    cache = tmp_path / "metadata"

    def fetch(_ids: Sequence[str]) -> list[dict[str, Any]]:
        return _graphql_response(
            {"rcsb_id": "1ABC", "rcsb_accession_info": None, "exptl_crystal_grow": None}
        )

    monkeypatch.setattr("rcsb_metadata_cache._fetch_graphql_batch", fetch)
    prefetch_rcsb_crystallization_metadata(["1abc"], str(cache), allow_download=True)

    stats = prefetch_rcsb_crystallization_metadata(
        ["1abc", "2def"], str(cache), allow_download=False
    )

    assert stats.cache_hits == 1
    assert stats.not_reported == 1
    assert stats.entry_unavailable == 0
    assert stats.not_fetched == 1


def test_an_entry_rcsb_omits_is_cached_as_unavailable_and_not_asked_again(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cache = tmp_path / "metadata"
    calls: list[tuple[str, ...]] = []

    def fetch(ids: Sequence[str]) -> list[dict[str, Any]]:
        calls.append(tuple(ids))
        return _graphql_response(
            {"rcsb_id": "1ABC", "rcsb_accession_info": None, "exptl_crystal_grow": None}
        )

    monkeypatch.setattr("rcsb_metadata_cache._fetch_graphql_batch", fetch)
    first = prefetch_rcsb_crystallization_metadata(
        ["1abc", "9zzz"], str(cache), allow_download=True
    )
    second = prefetch_rcsb_crystallization_metadata(
        ["1abc", "9zzz"], str(cache), allow_download=True
    )

    assert calls == [("1abc", "9zzz")]
    assert first.fetched == 2
    assert first.entry_unavailable == 1
    assert second.cache_hits == 2
    assert second.entry_unavailable == 1
    payload = read_cache_payload(str(cache), "9zzz")
    assert payload is not None
    assert payload["entry_available"] is False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        rcsb_metadata_cache.RCSB_GRAPHQL_URL, code, "status", Message(), None
    )


def test_transient_failures_are_retried_with_backoff(monkeypatch: Any) -> None:
    outcomes: list[BaseException | bytes] = [
        urllib.error.URLError("connection reset"),
        _http_error(503),
        json.dumps({"data": {"entries": []}}).encode("utf-8"),
    ]
    sleeps: list[float] = []

    def urlopen(_request: Any, timeout: float) -> io.BytesIO:
        del timeout
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return io.BytesIO(outcome)

    monkeypatch.setattr("rcsb_metadata_cache.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("rcsb_metadata_cache.time.sleep", sleeps.append)

    assert rcsb_metadata_cache._fetch_graphql_batch(["1abc"]) == []  # pyright: ignore[reportPrivateUsage]
    assert sleeps == [1, 2]


@pytest.mark.parametrize(
    "body, match",
    [
        (b'{"errors": [{"message": "bad query"}]}', "GraphQL errors"),
        (b'{"data": {}}', "data.entries"),
    ],
)
def test_a_malformed_reply_is_not_retried(
    monkeypatch: Any, body: bytes, match: str
) -> None:
    calls = 0

    def urlopen(_request: Any, timeout: float) -> io.BytesIO:
        nonlocal calls
        del timeout
        calls += 1
        return io.BytesIO(body)

    monkeypatch.setattr("rcsb_metadata_cache.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("rcsb_metadata_cache.time.sleep", _never_sleep)

    with pytest.raises(CrystallizationMetadataError, match=match):
        rcsb_metadata_cache._fetch_graphql_batch(["1abc"])  # pyright: ignore[reportPrivateUsage]
    assert calls == 1


def test_a_rejected_request_is_not_retried(monkeypatch: Any) -> None:
    def urlopen(_request: Any, timeout: float) -> io.BytesIO:
        del timeout
        raise _http_error(400)

    monkeypatch.setattr("rcsb_metadata_cache.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("rcsb_metadata_cache.time.sleep", _never_sleep)

    with pytest.raises(CrystallizationMetadataError, match="HTTP 400"):
        rcsb_metadata_cache._fetch_graphql_batch(["1abc"])  # pyright: ignore[reportPrivateUsage]


def _never_sleep(seconds: float) -> None:
    raise AssertionError(f"unexpected back-off of {seconds}s")
