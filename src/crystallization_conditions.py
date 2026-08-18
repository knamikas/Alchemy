"""Extract crystallization metadata and attach it to confidence review rows.

Crystallization conditions are contextual evidence, never confidence-score
inputs.  This module therefore owns a separate entry-level schema and the
post-scoring REVIEW/SUSPECT projection that joins those annotations by PDB ID.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import gemmi

from output_rows import CsvValue, scientific_csv_value

CONDITION_COLUMNS = (
    "pdbID",
    "crystallization_condition_id",
    "source_format",
    "metadata_source",
    "metadata_retrieved_at_utc",
    "entry_revision_date",
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
    "crystallization_metadata_source",
    "crystallization_metadata_retrieved_at_utc",
    "crystallization_entry_revision_date",
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

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_CACHE_SCHEMA_VERSION = 1
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
class CrystallizationExtraction:
    """Collect normalized condition rows and their entry summary."""

    conditions: tuple[dict[str, CsvValue], ...]
    summary: dict[str, CsvValue]


@dataclass(frozen=True, slots=True)
class CrystallizationPrefetchStats:
    """Count outcomes from prefetching original-PDB condition metadata."""

    requested: int
    cache_hits: int
    fetched: int
    available: int
    not_reported: int
    entry_unavailable: int


class CrystallizationMetadataError(RuntimeError):
    """Original-PDB condition metadata could not be fetched or cached safely."""


class _MmcifCategoryBlock(Protocol):
    """Typed view over Gemmi's incompletely annotated category accessor."""

    def get_mmcif_category(
        self, name: str, raw: bool = False
    ) -> dict[str, list[str]]: ...


def _clean_cif_value(value: object) -> str:
    if value is None:
        return ""
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
    ph_range = ""
    ph = ""
    if range_match:
        ph_range = f"{range_match.group(1)}-{range_match.group(2)}"
    else:
        match = _PH_RE.search(text)
        if match:
            ph = match.group(1)
    temp_match = _TEMP_RE.search(text)
    temperature = temp_match.group(1) if temp_match else ""
    return ph, ph_range, temperature


def _provenance(
    metadata_source: str,
    retrieved_at_utc: str = "",
    entry_revision_date: str = "",
) -> dict[str, CsvValue]:
    return {
        "metadata_source": metadata_source,
        "metadata_retrieved_at_utc": retrieved_at_utc,
        "entry_revision_date": entry_revision_date,
    }


def _mmcif_conditions(
    pdb_id: str, path: str, metadata_source: str
) -> list[dict[str, CsvValue]]:
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
                **_provenance(metadata_source),
                "crystal_id": crystal_id,
                "method": _category_value(category, "method", index),
                "pH": _category_value(category, "pH", index),
                "pH_range": _category_value(category, "pdbx_pH_range", index),
                "temperature_K": _category_value(category, "temp", index),
                "temperature_details": _category_value(category, "temp_details", index),
                "raw_details": _category_value(category, "pdbx_details", index),
            }
            if any(str(row[column]).strip() for column in CONDITION_COLUMNS[7:]):
                rows.append(row)
    return rows


def _pdb_conditions(
    pdb_id: str, path: str, metadata_source: str
) -> list[dict[str, CsvValue]]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        pieces = [line[10:].strip() for line in handle if line.startswith("REMARK 280")]
    details = " ".join(piece for piece in pieces if piece).strip()
    marker = re.search(r"(?i)CRYSTALLIZATION CONDITIONS\s*:\s*", details)
    if marker:
        details = details[marker.end() :].strip()
    if not details:
        return []
    ph, ph_range, temperature = _parse_text_measurements(details)
    return [
        {
            "pdbID": pdb_id,
            "crystallization_condition_id": f"{pdb_id}:condition:1",
            "source_format": "pdb",
            **_provenance(metadata_source),
            "crystal_id": "",
            "method": _method_from_text(details),
            "pH": ph,
            "pH_range": ph_range,
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
    pdb_id: str,
    status: str = "input_unavailable",
    *,
    source_format: str = "",
    metadata_source: str = "",
    retrieved_at_utc: str = "",
    entry_revision_date: str = "",
) -> dict[str, CsvValue]:
    """Build a summary row for unavailable crystallization conditions."""
    row: dict[str, CsvValue] = dict.fromkeys(SUMMARY_COLUMNS, "")
    row.update(
        pdbID=pdb_id,
        crystallization_data_status=status,
        crystallization_condition_count=0,
        crystallization_source_format=source_format,
        crystallization_metadata_source=metadata_source,
        crystallization_metadata_retrieved_at_utc=retrieved_at_utc,
        crystallization_entry_revision_date=entry_revision_date,
    )
    return row


def _summary(
    pdb_id: str,
    rows: Sequence[Mapping[str, CsvValue]],
    source_format: str,
) -> dict[str, CsvValue]:
    first: Mapping[str, CsvValue] = rows[0] if rows else {}
    provenance = {
        "crystallization_source_format": source_format,
        "crystallization_metadata_source": str(first.get("metadata_source", "")),
        "crystallization_metadata_retrieved_at_utc": str(
            first.get("metadata_retrieved_at_utc", "")
        ),
        "crystallization_entry_revision_date": str(
            first.get("entry_revision_date", "")
        ),
    }
    if not rows:
        return unavailable_summary(pdb_id, "not_reported") | provenance
    raw_text = " || ".join(
        dict.fromkeys(str(row.get("raw_details", "")).strip() for row in rows)
    ).strip(" |")
    searchable = " ".join(
        " ".join(
            str(row.get(column, ""))
            for column in (
                "method",
                "pH_range",
                "temperature_details",
                "raw_details",
            )
        )
        for row in rows
    )
    metals = detected_metals(searchable)
    ph_values: list[object] = [row.get("pH", "") for row in rows]
    for row in rows:
        ph_range = str(row.get("pH_range", ""))
        ph_values.extend(re.findall(r"\d+(?:\.\d+)?", ph_range))
    ph_min, ph_max = _range(ph_values, minimum=0.0, maximum=14.0)
    temp_min, temp_max = _range(row.get("temperature_K", "") for row in rows)
    lowered = searchable.lower()
    return {
        "pdbID": pdb_id,
        "crystallization_data_status": "available",
        "crystallization_condition_count": len(rows),
        **provenance,
        "crystallization_condition_ids": "|".join(
            str(row["crystallization_condition_id"]) for row in rows
        ),
        "crystallization_pH_min": ph_min,
        "crystallization_pH_max": ph_max,
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
    pdb_id: str,
    path: str,
    *,
    metadata_source: str = "coordinate_file",
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
            _mmcif_conditions(pdb_id, path, metadata_source)
            if source_format == "mmcif"
            else _pdb_conditions(pdb_id, path, metadata_source)
        )
    except (OSError, RuntimeError, ValueError):
        return CrystallizationExtraction(
            (),
            unavailable_summary(
                pdb_id,
                "unparseable",
                source_format=source_format,
                metadata_source=metadata_source,
            ),
        )
    return CrystallizationExtraction(tuple(rows), _summary(pdb_id, rows, source_format))


def _cache_path(cache_root: str, pdb_id: str) -> str:
    pdb_id = pdb_id.strip().lower()
    return os.path.join(cache_root, pdb_id[1:3], f"{pdb_id}.json")


def _validated_cache_payload(pdb_id: str, value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    payload = cast(dict[str, Any], value)
    if payload.get("schema_version") != RCSB_CACHE_SCHEMA_VERSION:
        return None
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


def _read_cache_payload(cache_root: str, pdb_id: str) -> dict[str, Any] | None:
    path = _cache_path(cache_root, pdb_id)
    try:
        with open(path, encoding="utf-8") as handle:
            return _validated_cache_payload(pdb_id, json.load(handle))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache_payload(cache_root: str, payload: Mapping[str, Any]) -> None:
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


def _condition_rows_from_cache(
    pdb_id: str, payload: Mapping[str, Any]
) -> list[dict[str, CsvValue]]:
    provenance = _provenance(
        str(payload["metadata_source"]),
        str(payload["retrieved_at_utc"]),
        str(payload["entry_revision_date"]),
    )
    rows: list[dict[str, CsvValue]] = []
    conditions = cast(list[dict[str, Any]], payload["conditions"])
    for condition in conditions:
        row: dict[str, CsvValue] = {
            "pdbID": pdb_id,
            "crystallization_condition_id": f"{pdb_id}:condition:{len(rows) + 1}",
            "source_format": "json",
            **provenance,
            "crystal_id": _clean_cif_value(condition.get("crystal_id", "")),
            "method": _clean_cif_value(condition.get("method", "")),
            "pH": _clean_cif_value(condition.get("pH", "")),
            "pH_range": _clean_cif_value(condition.get("pdbx_pH_range", "")),
            "temperature_K": _clean_cif_value(condition.get("temp", "")),
            "temperature_details": _clean_cif_value(condition.get("temp_details", "")),
            "raw_details": _clean_cif_value(condition.get("pdbx_details", "")),
        }
        if any(str(row[column]).strip() for column in CONDITION_COLUMNS[7:]):
            rows.append(row)
    return rows


def cached_rcsb_crystallization_conditions(
    pdb_id: str, cache_root: str
) -> CrystallizationExtraction | None:
    """Return a validated cached RCSB extraction, or ``None`` on a cache miss."""
    pdb_id = pdb_id.strip().lower()
    if not cache_root or len(pdb_id) != 4:
        return None
    payload = _read_cache_payload(cache_root, pdb_id)
    if payload is None:
        return None
    rows = _condition_rows_from_cache(pdb_id, payload)
    if rows:
        return CrystallizationExtraction(tuple(rows), _summary(pdb_id, rows, "json"))
    status = "not_reported" if payload["entry_available"] else "input_unavailable"
    summary = unavailable_summary(
        pdb_id,
        status,
        source_format="json",
        metadata_source=str(payload["metadata_source"]),
        retrieved_at_utc=str(payload["retrieved_at_utc"]),
        entry_revision_date=str(payload["entry_revision_date"]),
    )
    return CrystallizationExtraction((), summary)


def extract_crystallization_context(
    pdb_id: str,
    coordinate_path: str,
    cache_root: str,
    *,
    prefer_coordinate_file: bool = False,
) -> CrystallizationExtraction:
    """Choose deposited or coordinate-file conditions with an explicit fallback."""
    local_source = (
        "manual_coordinate_file"
        if prefer_coordinate_file
        else "pdb_redo_coordinate_file"
    )
    local = extract_crystallization_conditions(
        pdb_id, coordinate_path, metadata_source=local_source
    )
    deposited = cached_rcsb_crystallization_conditions(pdb_id, cache_root)
    if deposited is None:
        return local
    local_available = local.summary["crystallization_data_status"] == "available"
    deposited_available = (
        deposited.summary["crystallization_data_status"] == "available"
    )
    if prefer_coordinate_file and local_available:
        return local
    if deposited_available:
        return deposited
    if local_available:
        return local
    return local if prefer_coordinate_file else deposited


def _fetch_graphql_batch(pdb_ids: Sequence[str]) -> dict[str, object]:
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
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                loaded: object = json.load(response)
            if not isinstance(loaded, dict):
                raise ValueError("response is not a JSON object")
            result = cast(dict[str, object], loaded)
            if result.get("errors"):
                raise ValueError(f"GraphQL errors: {result['errors']!r}")
            data_value = result.get("data")
            if not isinstance(data_value, dict):
                raise ValueError("response has no data object")
            data = cast(dict[str, object], data_value)
            if not isinstance(data.get("entries"), list):
                raise ValueError("response has no data.entries list")
            return result
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise CrystallizationMetadataError(
        f"RCSB Data API request failed after 3 attempts: {last_error}"
    )


def _payloads_from_graphql(
    pdb_ids: Sequence[str], response: Mapping[str, object], retrieved_at_utc: str
) -> list[dict[str, object]]:
    data_value = response["data"]
    if not isinstance(data_value, dict):
        raise CrystallizationMetadataError("RCSB Data API returned invalid data")
    data = cast(dict[str, object], data_value)
    entries_value = data.get("entries")
    if not isinstance(entries_value, list):
        raise CrystallizationMetadataError(
            "RCSB Data API returned an invalid entries list"
        )
    entries = cast(list[object], entries_value)
    by_id: dict[str, Mapping[str, object]] = {}
    for value in entries:
        if not isinstance(value, dict):
            continue
        entry = cast(Mapping[str, object], value)
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
            cast(dict[str, object], accession_value)
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
            condition_values = cast(list[object], conditions_value)
        else:
            raise CrystallizationMetadataError(
                f"RCSB Data API returned invalid conditions for {pdb_id}"
            )
        if not all(isinstance(condition, dict) for condition in condition_values):
            raise CrystallizationMetadataError(
                f"RCSB Data API returned invalid conditions for {pdb_id}"
            )
        conditions = [
            cast(dict[str, object], condition) for condition in condition_values
        ]
        payloads.append(
            {
                "schema_version": RCSB_CACHE_SCHEMA_VERSION,
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
    """Populate the persistent RCSB cache before worker processes start."""
    ids = tuple(dict.fromkeys(pdb_id.strip().lower() for pdb_id in pdb_ids))
    missing = [
        pdb_id for pdb_id in ids if _read_cache_payload(cache_root, pdb_id) is None
    ]
    fetched = 0
    if allow_download:
        for start in range(0, len(missing), RCSB_BATCH_SIZE):
            batch = missing[start : start + RCSB_BATCH_SIZE]
            response = _fetch_graphql_batch(batch)
            retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
            for payload in _payloads_from_graphql(batch, response, retrieved_at):
                try:
                    _write_cache_payload(cache_root, payload)
                except OSError as exc:
                    raise CrystallizationMetadataError(
                        f"could not write crystallization metadata cache: {exc}"
                    ) from None
                fetched += 1
    available = 0
    not_reported = 0
    unavailable = 0
    for pdb_id in ids:
        cached_payload = _read_cache_payload(cache_root, pdb_id)
        if cached_payload is None or not cached_payload["entry_available"]:
            unavailable += 1
        elif cached_payload["conditions"]:
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
    )


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
