from __future__ import annotations

import csv
from pathlib import Path

from confidence_score import ANALYSIS_COLUMNS, CONFIDENCE_INPUT_COLUMNS
from crystallization_conditions import (
    CONDITION_COLUMNS,
    REVIEW_CONTEXT_COLUMNS,
    SUMMARY_COLUMNS,
    detected_metals,
    extract_crystallization_conditions,
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


def test_formula_detection_does_not_treat_ordinary_words_as_symbols() -> None:
    assert detected_metals("vapor diffusion in acetate buffer") == frozenset()
    assert detected_metals("SODIUM CACODYLATE AT 293 K") == frozenset({"NA"})
    assert detected_metals("25 mM NiCl2 and calcium acetate") == frozenset({"NI", "CA"})


def test_review_queue_filters_levels_and_joins_site_specific_context(
    tmp_path: Path,
) -> None:
    confidence_columns = (*CONFIDENCE_INPUT_COLUMNS, *ANALYSIS_COLUMNS)
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
                {column: "" for column in confidence_columns}
                | {
                    "pdbID": pdb_id,
                    "metal_site_id": site_id,
                    "metal_element": element,
                    "alchemy_level": level,
                }
            )

    summary_path = tmp_path / "summary.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {column: "" for column in SUMMARY_COLUMNS}
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
            {column: "" for column in SUMMARY_COLUMNS}
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
