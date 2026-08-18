from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from confidence_score import ANALYSIS_COLUMNS, CONFIDENCE_INPUT_COLUMNS
from crystallization_conditions import (
    CONDITION_COLUMNS,
    REVIEW_CONTEXT_COLUMNS,
    SUMMARY_COLUMNS,
    cached_rcsb_crystallization_conditions,
    detected_metals,
    extract_crystallization_conditions,
    extract_crystallization_context,
    prefetch_rcsb_crystallization_metadata,
    write_review_queue,
)


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_crystallization_fields_are_not_confidence_inputs_or_outputs() -> None:
    assert all(
        not column.startswith("crystallization_")
        for column in (*CONFIDENCE_INPUT_COLUMNS, *ANALYSIS_COLUMNS)
    )


def test_mmcif_conditions_preserve_raw_metadata_and_build_context_flags(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "entry.cif",
        """data_1abc
loop_
_exptl_crystal_grow.crystal_id
_exptl_crystal_grow.method
_exptl_crystal_grow.temp
_exptl_crystal_grow.pH
_exptl_crystal_grow.pdbx_pH_range
_exptl_crystal_grow.pdbx_details
1 'VAPOR DIFFUSION, HANGING DROP' 293 7.5 ?
;0.1 M ZnCl2, sodium cacodylate, ammonium sulfate and acetate
;
2 batch 277 ? 6.0-6.5 '10 mM cadmium chloride'
""",
    )

    extraction = extract_crystallization_conditions("1ABC", path)

    assert len(extraction.conditions) == 2
    assert tuple(extraction.conditions[0]) == CONDITION_COLUMNS
    assert str(extraction.conditions[0]["raw_details"]).startswith("0.1 M ZnCl2")
    assert extraction.summary["crystallization_data_status"] == "available"
    assert extraction.summary["crystallization_condition_count"] == 2
    assert extraction.summary["crystallization_pH_min"] == 6.0
    assert extraction.summary["crystallization_pH_max"] == 7.5
    assert extraction.summary["crystallization_temperature_min_K"] == 277.0
    assert extraction.summary["crystallization_temperature_max_K"] == 293.0
    assert extraction.summary["crystallization_detected_metals"] == "CD|NA|ZN"
    assert extraction.summary["crystallization_promiscuous_transition_metal"]
    assert extraction.summary["crystallization_heavy_additive_phasing_metal"]
    assert extraction.summary["crystallization_cacodylate"]
    assert extraction.summary["crystallization_sulfate"]
    assert extraction.summary["crystallization_acetate"]


def test_absent_mmcif_category_is_unknown_not_negative_evidence(tmp_path: Path) -> None:
    path = _write(tmp_path / "entry.cif", "data_1abc\n_entry.id 1ABC\n")
    extraction = extract_crystallization_conditions("1abc", path)

    assert extraction.conditions == ()
    assert extraction.summary["crystallization_data_status"] == "not_reported"
    assert extraction.summary["crystallization_any_metal"] == ""


def test_pdb_remark_280_is_extracted(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "entry.pdb",
        "REMARK 280 CRYSTALLIZATION CONDITIONS: 50 MM MANGANESE CHLORIDE,\n"
        "REMARK 280 20% PEG 3350, PH 6.8, TEMPERATURE 291K, HANGING DROP\n"
        "END\n",
    )
    extraction = extract_crystallization_conditions("1abc", path)

    assert len(extraction.conditions) == 1
    row = extraction.conditions[0]
    assert row["method"] == "hanging drop"
    assert row["pH"] == "6.8"
    assert row["temperature_K"] == "291"
    assert extraction.summary["crystallization_detected_metals"] == "MN"


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

    monkeypatch.setattr("crystallization_conditions._fetch_graphql_batch", fetch)
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


def test_invalid_cache_is_a_safe_offline_miss(tmp_path: Path) -> None:
    cache_file = tmp_path / "metadata" / "ab" / "1abc.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")

    assert (
        cached_rcsb_crystallization_conditions("1abc", str(cache_file.parents[1]))
        is None
    )
    stats = prefetch_rcsb_crystallization_metadata(
        ["1abc"], str(cache_file.parents[1]), allow_download=False
    )
    assert stats.cache_hits == 0
    assert stats.entry_unavailable == 1


def test_source_precedence_distinguishes_database_and_manual_runs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    coordinate = _write(
        tmp_path / "entry.pdb",
        "REMARK 280 CRYSTALLIZATION CONDITIONS: 10 MM MANGANESE, PH 7.2\n",
    )
    cache = tmp_path / "metadata"

    def deposited_response(_ids: Sequence[str]) -> dict[str, object]:
        return _graphql_response(
            {
                "rcsb_id": "1ABC",
                "rcsb_accession_info": {"revision_date": "2025-01-02"},
                "exptl_crystal_grow": [
                    {
                        "crystal_id": "1",
                        "method": None,
                        "temp": None,
                        "pH": 4.2,
                        "pdbx_pH_range": None,
                        "pdbx_details": "pH 4.2",
                        "temp_details": None,
                    }
                ],
            }
        )

    monkeypatch.setattr(
        "crystallization_conditions._fetch_graphql_batch",
        deposited_response,
    )
    prefetch_rcsb_crystallization_metadata(["1abc"], str(cache), allow_download=True)

    database = extract_crystallization_context("1abc", coordinate, str(cache))
    manual = extract_crystallization_context(
        "1abc", coordinate, str(cache), prefer_coordinate_file=True
    )
    assert database.summary["crystallization_pH_min"] == 4.2
    assert database.summary["crystallization_metadata_source"] == "rcsb_data_api"
    assert manual.summary["crystallization_pH_min"] == 7.2
    assert manual.summary["crystallization_metadata_source"] == "manual_coordinate_file"


def test_coordinate_file_fills_a_deposited_record_without_conditions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    coordinate = _write(
        tmp_path / "entry.pdb",
        "REMARK 280 CRYSTALLIZATION CONDITIONS: 10 MM MANGANESE, PH 7.2\n",
    )
    cache = tmp_path / "metadata"

    def empty_response(_ids: Sequence[str]) -> dict[str, object]:
        return _graphql_response(
            {
                "rcsb_id": "1ABC",
                "rcsb_accession_info": {"revision_date": "2025-01-02"},
                "exptl_crystal_grow": None,
            }
        )

    monkeypatch.setattr(
        "crystallization_conditions._fetch_graphql_batch",
        empty_response,
    )
    prefetch_rcsb_crystallization_metadata(["1abc"], str(cache), allow_download=True)

    extraction = extract_crystallization_context("1abc", coordinate, str(cache))

    assert extraction.summary["crystallization_data_status"] == "available"
    assert (
        extraction.summary["crystallization_metadata_source"]
        == "pdb_redo_coordinate_file"
    )


def test_formula_detection_does_not_treat_ordinary_words_as_symbols() -> None:
    assert detected_metals("vapor diffusion in acetate buffer") == frozenset()
    assert detected_metals("SODIUM CACODYLATE AT 293 K") == frozenset({"NA"})
    assert detected_metals("25 mM NiCl2 and calcium acetate") == frozenset({"NI", "CA"})


def test_review_queue_filters_levels_and_joins_site_specific_context(
    tmp_path: Path,
) -> None:
    confidence_columns: list[str] = [*CONFIDENCE_INPUT_COLUMNS, *ANALYSIS_COLUMNS]
    scores_path = tmp_path / "confidence.csv"
    with open(scores_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=confidence_columns)
        writer.writeheader()
        for pdb_id, site_id, element, level in (
            ("1abc", "site-pass", "ZN", "PASS"),
            ("1abc", "site-review", "ZN", "REVIEW"),
            ("2def", "site-suspect", "FE", "SUSPECT"),
        ):
            writer.writerow(
                dict.fromkeys(confidence_columns, "")
                | {
                    "pdbID": pdb_id,
                    "metal_site_id": site_id,
                    "metal_element": element,
                    "alchemy_level": level,
                }
            )

    summary_path = tmp_path / "summary.csv"
    summary_columns: list[str] = list(SUMMARY_COLUMNS)
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerow(
            dict.fromkeys(summary_columns, "")
            | {
                "pdbID": "1abc",
                "crystallization_data_status": "available",
                "crystallization_detected_metals": "NI|ZN",
                "crystallization_any_metal": "true",
                "crystallization_promiscuous_transition_metal": "true",
                "crystallization_ni_co_like_metal": "true",
            }
        )
        writer.writerow(
            dict.fromkeys(summary_columns, "")
            | {
                "pdbID": "2def",
                "crystallization_data_status": "not_reported",
            }
        )

    output = tmp_path / "review.csv"
    assert (
        write_review_queue(
            str(scores_path), str(summary_path), str(output), confidence_columns
        )
        == 2
    )
    with open(output, newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [*confidence_columns, *REVIEW_CONTEXT_COLUMNS]
        rows = list(reader)

    review, suspect = rows
    assert review["metal_site_id"] == "site-review"
    assert review["crystallization_contains_modeled_metal"] == "true"
    assert (
        review["crystallization_contains_different_promiscuous_transition_metal"]
        == "true"
    )
    assert review["crystallization_context_flags"] == (
        "modeled_metal|different_promiscuous_transition_metal|ni_co_like_metal"
    )
    assert suspect["metal_site_id"] == "site-suspect"
    assert suspect["crystallization_contains_modeled_metal"] == ""
    assert suspect["crystallization_context_flags"] == ""
