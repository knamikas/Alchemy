"""Extract crystallization metadata and attach it to confidence review rows.

Crystallization conditions are contextual evidence, never confidence-score
inputs.  This module therefore owns a separate entry-level schema and the
post-scoring REVIEW/SUSPECT projection that joins those annotations by PDB ID.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, cast
from collections.abc import Iterable, Mapping, Sequence

import gemmi

from output_rows import CsvValue, scientific_csv_value


CONDITION_COLUMNS = (
    "pdbID",
    "crystallization_condition_id",
    "source_format",
    "crystal_id",
    "method",
    "pH",
    "pH_range",
    "temperature_K",
    "temperature_details",
    "raw_details",
)

SUMMARY_COLUMNS = (
    "pdbID",
    "crystallization_data_status",
    "crystallization_condition_count",
    "crystallization_source_format",
    "crystallization_condition_ids",
    "crystallization_pH_min",
    "crystallization_pH_max",
    "crystallization_temperature_min_K",
    "crystallization_temperature_max_K",
    "crystallization_raw_text",
    "crystallization_detected_metals",
    "crystallization_any_metal",
    "crystallization_promiscuous_transition_metal",
    "crystallization_ni_co_like_metal",
    "crystallization_buffer_light_metal",
    "crystallization_heavy_additive_phasing_metal",
    "crystallization_sulfate",
    "crystallization_cacodylate",
    "crystallization_acetate",
)

REVIEW_CONTEXT_COLUMNS = tuple(
    column for column in SUMMARY_COLUMNS if column != "pdbID"
) + (
    "crystallization_contains_modeled_metal",
    "crystallization_contains_different_promiscuous_transition_metal",
    "crystallization_context_flags",
)

CRYSTALLIZATION_DATA_STATUSES = frozenset(
    {"available", "not_reported", "unparseable", "input_unavailable"}
)

PROMISCUOUS_TRANSITION_METALS = frozenset({"MN", "FE", "CO", "NI", "CU", "ZN", "CD"})
NI_CO_LIKE_METALS = frozenset({"NI", "CO"})
BUFFER_LIGHT_METALS = frozenset({"LI", "NA", "MG", "K", "CA"})
HEAVY_ADDITIVE_PHASING_METALS = frozenset(
    {
        "CD",
        "HG",
        "PT",
        "AU",
        "IR",
        "PB",
        "TL",
        "U",
        "AG",
        "OS",
        "PD",
        "GD",
        "YB",
        "SM",
        "EU",
        "TB",
        "LU",
        "HO",
        "LA",
        "CE",
        "ER",
        "DY",
        "ND",
        "PR",
    }
)

# Names and common reagent stems are intentionally positive evidence only.
# Symbols are handled separately with case-sensitive formula/word matching so
# ordinary words containing e.g. "ca" or "in" cannot become metal hits.
_METAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "LI": ("lithium",),
    "NA": ("sodium",),
    "MG": ("magnesium",),
    "K": ("potassium",),
    "CA": ("calcium",),
    "MN": ("manganese", "manganous", "permanganate"),
    "FE": ("iron", "ferrous", "ferric"),
    "CO": ("cobalt", "cobaltous"),
    "NI": ("nickel",),
    "CU": ("copper", "cupric", "cuprous"),
    "ZN": ("zinc",),
    "CD": ("cadmium",),
    "HG": ("mercury", "mercuric", "mercurous"),
    "PT": ("platinum",),
    "MO": ("molybdenum", "molybdate"),
    "AL": ("aluminum", "aluminium"),
    "BA": ("barium",),
    "RU": ("ruthenium",),
    "V": ("vanadium", "vanadate"),
    "SR": ("strontium",),
    "CS": ("cesium", "caesium"),
    "W": ("tungsten", "tungstate"),
    "AU": ("gold",),
    "IR": ("iridium",),
    "YB": ("ytterbium",),
    "GD": ("gadolinium",),
    "PB": ("lead",),
    "TL": ("thallium",),
    "U": ("uranium", "uranyl"),
    "Y": ("yttrium",),
    "TI": ("titanium",),
    "RB": ("rubidium",),
    "AG": ("silver",),
    "SM": ("samarium",),
    "OS": ("osmium",),
    "PR": ("praseodymium",),
    "PD": ("palladium",),
    "EU": ("europium",),
    "TB": ("terbium",),
    "LU": ("lutetium",),
    "HO": ("holmium",),
    "CR": ("chromium", "chromate", "dichromate"),
    "LA": ("lanthanum",),
    "SN": ("tin", "stannous", "stannic"),
    "SB": ("antimony", "antimonate"),
    "CE": ("cerium",),
    "ZR": ("zirconium",),
    "ER": ("erbium",),
    "TH": ("thorium",),
    "SC": ("scandium",),
    "DY": ("dysprosium",),
    "BI": ("bismuth",),
    "ND": ("neodymium",),
}

_FORMULA_SUFFIX = r"(?=[A-Z0-9(])[A-Za-z0-9()]*\d[A-Za-z0-9()]*"
_SIMPLE_FORMULAS: Mapping[str, tuple[str, ...]] = {
    "LI": ("LiCl", "LICL"),
    "NA": ("NaCl", "NACL"),
    "K": ("KCl", "KCL"),
    "AG": ("AgCl", "AGCL"),
}
_PH_RE = re.compile(r"(?i)\bp\s*h\s*(?:=|:)?\s*(-?\d+(?:\.\d+)?)")
_PH_RANGE_RE = re.compile(
    r"(?i)\bp\s*h\s*(?:range)?\s*(?:=|:)?\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)"
)
_TEMP_RE = re.compile(
    r"(?i)\b(?:temperature|temp\.?)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*(?:k|kelvin)?"
)


@dataclass(frozen=True, slots=True)
class CrystallizationExtraction:
    conditions: tuple[dict[str, CsvValue], ...]
    summary: dict[str, CsvValue]


class _MmcifCategoryBlock(Protocol):
    """Typed view over Gemmi's incompletely annotated category accessor."""

    def get_mmcif_category(
        self, name: str, raw: bool = False
    ) -> dict[str, list[str]]: ...


def _clean_cif_value(value: object) -> str:
    text = str(value).strip()
    return "" if text in {".", "?"} else text


def _category_value(
    category: Mapping[str, Sequence[str]], name: str, index: int
) -> str:
    values = category.get(name, ())
    return _clean_cif_value(values[index]) if index < len(values) else ""


def _finite_number(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _range(
    values: Iterable[object],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[CsvValue, CsvValue]:
    numbers = [
        number for value in values if (number := _finite_number(value)) is not None
    ]
    if minimum is not None:
        numbers = [number for number in numbers if number >= minimum]
    if maximum is not None:
        numbers = [number for number in numbers if number <= maximum]
    if not numbers:
        return "", ""
    return min(numbers), max(numbers)


def _method_from_text(text: str) -> str:
    lowered = text.lower()
    for label in (
        "hanging drop",
        "sitting drop",
        "vapor diffusion",
        "vapour diffusion",
        "microbatch",
        "batch",
        "dialysis",
        "free interface diffusion",
    ):
        if label in lowered:
            return label
    return ""


def _parse_text_measurements(text: str) -> tuple[str, str, str]:
    range_match = _PH_RANGE_RE.search(text)
    pH_range = ""
    pH = ""
    if range_match:
        pH_range = f"{range_match.group(1)}-{range_match.group(2)}"
    else:
        match = _PH_RE.search(text)
        if match:
            pH = match.group(1)
    temp_match = _TEMP_RE.search(text)
    temperature = temp_match.group(1) if temp_match else ""
    return pH, pH_range, temperature


def _mmcif_conditions(pdb_id: str, path: str) -> list[dict[str, CsvValue]]:
    document = gemmi.cif.read(path)
    rows: list[dict[str, CsvValue]] = []
    for block in document:
        category = cast(_MmcifCategoryBlock, block).get_mmcif_category(
            "_exptl_crystal_grow."
        )
        if not category:
            continue
        row_count = max((len(values) for values in category.values()), default=0)

        for index in range(row_count):
            crystal_id = _category_value(category, "crystal_id", index)
            row: dict[str, CsvValue] = {
                "pdbID": pdb_id,
                "crystallization_condition_id": (f"{pdb_id}:condition:{len(rows) + 1}"),
                "source_format": "mmcif",
                "crystal_id": crystal_id,
                "method": _category_value(category, "method", index),
                "pH": _category_value(category, "pH", index),
                "pH_range": _category_value(category, "pdbx_pH_range", index),
                "temperature_K": _category_value(category, "temp", index),
                "temperature_details": _category_value(category, "temp_details", index),
                "raw_details": _category_value(category, "pdbx_details", index),
            }
            if any(str(row[column]).strip() for column in CONDITION_COLUMNS[4:]):
                rows.append(row)
    return rows


def _pdb_conditions(pdb_id: str, path: str) -> list[dict[str, CsvValue]]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        pieces = [line[10:].strip() for line in handle if line.startswith("REMARK 280")]
    details = " ".join(piece for piece in pieces if piece).strip()
    marker = re.search(r"(?i)CRYSTALLIZATION CONDITIONS\s*:\s*", details)
    if marker:
        details = details[marker.end() :].strip()
    if not details:
        return []
    pH, pH_range, temperature = _parse_text_measurements(details)
    return [
        {
            "pdbID": pdb_id,
            "crystallization_condition_id": f"{pdb_id}:condition:1",
            "source_format": "pdb",
            "crystal_id": "",
            "method": _method_from_text(details),
            "pH": pH,
            "pH_range": pH_range,
            "temperature_K": temperature,
            "temperature_details": "",
            "raw_details": details,
        }
    ]


def detected_metals(text: str) -> frozenset[str]:
    """Return explicitly named or formula-like metals in condition text."""
    lowered = text.lower()
    detected = {
        symbol
        for symbol, aliases in _METAL_ALIASES.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases)
    }
    for symbol in _METAL_ALIASES:
        title_symbol = symbol.title()
        if len(symbol) > 1 and re.search(rf"\b(?:{symbol}|{title_symbol})\b", text):
            detected.add(symbol)
            continue
        formula_symbol = rf"(?:{re.escape(title_symbol)}|{re.escape(symbol)})"
        if re.search(rf"(?<![A-Za-z]){formula_symbol}{_FORMULA_SUFFIX}", text):
            detected.add(symbol)
            continue
        if any(
            re.search(rf"(?<![A-Za-z]){formula}(?![A-Za-z])", text)
            for formula in _SIMPLE_FORMULAS.get(symbol, ())
        ):
            detected.add(symbol)
    return frozenset(detected)


def unavailable_summary(
    pdb_id: str, status: str = "input_unavailable"
) -> dict[str, CsvValue]:
    row: dict[str, CsvValue] = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        pdbID=pdb_id,
        crystallization_data_status=status,
        crystallization_condition_count=0,
    )
    return row


def _summary(
    pdb_id: str,
    rows: Sequence[Mapping[str, CsvValue]],
    source_format: str,
) -> dict[str, CsvValue]:
    if not rows:
        return unavailable_summary(pdb_id, "not_reported") | {
            "crystallization_source_format": source_format
        }
    raw_text = " || ".join(
        dict.fromkeys(str(row.get("raw_details", "")).strip() for row in rows)
    ).strip(" |")
    searchable = " ".join(
        " ".join(str(row.get(column, "")) for column in CONDITION_COLUMNS[3:])
        for row in rows
    )
    metals = detected_metals(searchable)
    pH_values: list[object] = [row.get("pH", "") for row in rows]
    for row in rows:
        pH_range = str(row.get("pH_range", ""))
        pH_values.extend(re.findall(r"\d+(?:\.\d+)?", pH_range))
    pH_min, pH_max = _range(pH_values, minimum=0.0, maximum=14.0)
    temp_min, temp_max = _range(row.get("temperature_K", "") for row in rows)
    lowered = searchable.lower()
    return {
        "pdbID": pdb_id,
        "crystallization_data_status": "available",
        "crystallization_condition_count": len(rows),
        "crystallization_source_format": source_format,
        "crystallization_condition_ids": "|".join(
            str(row["crystallization_condition_id"]) for row in rows
        ),
        "crystallization_pH_min": pH_min,
        "crystallization_pH_max": pH_max,
        "crystallization_temperature_min_K": temp_min,
        "crystallization_temperature_max_K": temp_max,
        "crystallization_raw_text": raw_text,
        "crystallization_detected_metals": "|".join(sorted(metals)),
        "crystallization_any_metal": bool(metals),
        "crystallization_promiscuous_transition_metal": bool(
            metals & PROMISCUOUS_TRANSITION_METALS
        ),
        "crystallization_ni_co_like_metal": bool(metals & NI_CO_LIKE_METALS),
        "crystallization_buffer_light_metal": bool(metals & BUFFER_LIGHT_METALS),
        "crystallization_heavy_additive_phasing_metal": bool(
            metals & HEAVY_ADDITIVE_PHASING_METALS
        ),
        "crystallization_sulfate": "sulfate" in lowered or "sulphate" in lowered,
        "crystallization_cacodylate": "cacodylate" in lowered,
        "crystallization_acetate": "acetate" in lowered,
    }


def extract_crystallization_conditions(
    pdb_id: str, path: str
) -> CrystallizationExtraction:
    """Extract deposited conditions without making their absence an error."""
    pdb_id = pdb_id.strip().lower()
    if not path:
        return CrystallizationExtraction((), unavailable_summary(pdb_id))
    lower_path = path.lower()
    source_format = (
        "mmcif"
        if lower_path.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        else "pdb"
    )
    try:
        rows = (
            _mmcif_conditions(pdb_id, path)
            if source_format == "mmcif"
            else _pdb_conditions(pdb_id, path)
        )
    except (OSError, RuntimeError, ValueError):
        return CrystallizationExtraction(
            (),
            unavailable_summary(pdb_id, "unparseable")
            | {"crystallization_source_format": source_format},
        )
    return CrystallizationExtraction(tuple(rows), _summary(pdb_id, rows, source_format))


def write_review_queue(
    confidence_scores_path: str,
    crystallization_summary_path: str,
    output_path: str,
    confidence_columns: Sequence[str],
) -> int:
    """Write REVIEW/SUSPECT sites joined to entry-level condition context."""
    output_columns = (*confidence_columns, *REVIEW_CONTEXT_COLUMNS)
    summaries: dict[str, dict[str, str]] = {}
    if os.path.isfile(crystallization_summary_path):
        with open(crystallization_summary_path, newline="") as handle:
            summaries = {
                row["pdbID"].strip().lower(): row for row in csv.DictReader(handle)
            }
    count = 0
    with open(output_path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=output_columns)
        writer.writeheader()
        if not os.path.isfile(confidence_scores_path):
            return 0
        with open(confidence_scores_path, newline="") as scores:
            reader = csv.DictReader(scores)
            if reader.fieldnames != list(confidence_columns):
                raise ValueError(
                    "confidence score schema is incompatible with review queue"
                )
            for row in reader:
                if row.get("alchemy_level") not in {"REVIEW", "SUSPECT"}:
                    continue
                summary = summaries.get(row["pdbID"].strip().lower())
                context = {
                    column: summary.get(column, "") if summary else ""
                    for column in SUMMARY_COLUMNS
                    if column != "pdbID"
                }
                detected = {
                    value
                    for value in context["crystallization_detected_metals"].split("|")
                    if value
                }
                modeled = row.get("metal_element", "").strip().upper()
                available = context["crystallization_data_status"] == "available"
                same = modeled in detected if available and modeled else ""
                different = (
                    bool((detected - {modeled}) & PROMISCUOUS_TRANSITION_METALS)
                    if available
                    else ""
                )
                flags = [
                    name
                    for name, present in (
                        ("modeled_metal", same is True),
                        ("different_promiscuous_transition_metal", different is True),
                        (
                            "heavy_additive_phasing_metal",
                            context["crystallization_heavy_additive_phasing_metal"]
                            == "true",
                        ),
                        (
                            "ni_co_like_metal",
                            context["crystallization_ni_co_like_metal"] == "true",
                        ),
                        ("sulfate", context["crystallization_sulfate"] == "true"),
                        ("cacodylate", context["crystallization_cacodylate"] == "true"),
                        ("acetate", context["crystallization_acetate"] == "true"),
                    )
                    if present
                ]
                joined: dict[str, Any] = (
                    row
                    | context
                    | {
                        "crystallization_contains_modeled_metal": same,
                        (
                            "crystallization_contains_different_"
                            "promiscuous_transition_metal"
                        ): different,
                        "crystallization_context_flags": "|".join(flags),
                    }
                )
                writer.writerow(
                    {
                        column: scientific_csv_value(joined[column])
                        for column in output_columns
                    }
                )
                count += 1
    return count
