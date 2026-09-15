from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from confidence_score import ANALYSIS_COLUMNS, CONFIDENCE_INPUT_COLUMNS
from crystallization_conditions import (
    CONDITION_COLUMNS,
    detected_metals,
    extract_crystallization_conditions,
    extract_crystallization_context,
)
from rcsb_metadata_cache import prefetch_rcsb_crystallization_metadata


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


def test_a_crystal_id_alone_is_not_a_condition(tmp_path: Path) -> None:
    """A grow row naming only its crystal says nothing; numbering skips it."""
    path = _write(
        tmp_path / "entry.cif",
        """data_1abc
loop_
_exptl_crystal_grow.crystal_id
_exptl_crystal_grow.method
_exptl_crystal_grow.pdbx_details
1 ? ?
2 batch ?
""",
    )

    extraction = extract_crystallization_conditions("1abc", path)

    assert [row["crystal_id"] for row in extraction.conditions] == ["2"]
    assert extraction.conditions[0]["crystallization_condition_id"] == (
        "1abc:condition:1"
    )
    assert extraction.summary["crystallization_condition_count"] == 1


def test_unparseable_input_is_logged_at_debug(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The status stays ``unparseable``; the debug log names the failure."""
    path = _write(tmp_path / "entry.pdb.gz", "not gzip data")

    with caplog.at_level(logging.DEBUG, logger="alchemy.crystallization_conditions"):
        extraction = extract_crystallization_conditions("1abc", path)

    assert extraction.conditions == ()
    assert extraction.summary["crystallization_data_status"] == "unparseable"
    assert extraction.summary["crystallization_source_format"] == "pdb"
    messages = [record.getMessage() for record in caplog.records]
    assert any("1abc" in m and "BadGzipFile" in m and path in m for m in messages)


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


@pytest.mark.parametrize(
    "remark, expected",
    [
        ("TEMPERATURE 291K", "291"),
        ("TEMPERATURE 277.0 KELVIN", "277.0"),
        ("TEMPERATURE 293", "293"),
        ("TEMPERATURE 20 C", "293.15"),
        ("TEMPERATURE 4\u00b0C", "277.15"),
        ("TEMPERATURE 18.5 DEG C", "291.65"),
        ("TEMPERATURE 20", ""),
        ("TEMPERATURE 293 CRYSTALS GREW IN A WEEK", "293"),
    ],
)
def test_remark_temperatures_are_recorded_in_kelvin(
    tmp_path: Path, remark: str, expected: str
) -> None:
    """Celsius is converted, and a unitless value counts only if it is kelvin."""
    path = _write(
        tmp_path / "entry.pdb",
        "REMARK 280 CRYSTALLIZATION CONDITIONS: 20% PEG 3350, PH 6.8,\n"
        f"REMARK 280 {remark}, HANGING DROP\n"
        "END\n",
    )
    extraction = extract_crystallization_conditions("1abc", path)

    assert extraction.conditions[0]["temperature_K"] == expected
    summary = extraction.summary
    expected_summary = float(expected) if expected else ""
    assert summary["crystallization_temperature_min_K"] == expected_summary
    assert summary["crystallization_temperature_max_K"] == expected_summary


def _graphql_response(*entries: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``data.entries`` list a fetched batch yields."""
    return list(entries)


def test_source_precedence_distinguishes_database_and_manual_runs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    coordinate = _write(
        tmp_path / "entry.pdb",
        "REMARK 280 CRYSTALLIZATION CONDITIONS: 10 MM MANGANESE, PH 7.2\n",
    )
    cache = tmp_path / "metadata"

    def deposited_response(_ids: Sequence[str]) -> list[dict[str, Any]]:
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
        "rcsb_metadata_cache._fetch_graphql_batch",
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

    def empty_response(_ids: Sequence[str]) -> list[dict[str, Any]]:
        return _graphql_response(
            {
                "rcsb_id": "1ABC",
                "rcsb_accession_info": {"revision_date": "2025-01-02"},
                "exptl_crystal_grow": None,
            }
        )

    monkeypatch.setattr(
        "rcsb_metadata_cache._fetch_graphql_batch",
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


def test_mass_units_do_not_spell_magnesium() -> None:
    """Deposited REMARK 280 text writes protein concentration as ``MG/ML``."""
    assert (
        detected_metals("PROTEIN 5 MG/ML, 0.1 M HEPES PH 7.5, TEMPERATURE 293K")
        == frozenset()
    )
    assert detected_metals("10MG/ML LYSOZYME, 50 MM NACL") == frozenset({"NA"})
    assert detected_metals("5 MG PROTEIN IN 0.2 M MG ACETATE") == frozenset({"MG"})
    assert detected_metals("2 mg/mL protein, 0.2 M Mg acetate") == frozenset({"MG"})
    assert detected_metals("10 mM MgCl2, 20 mM Mg2+") == frozenset({"MG"})
