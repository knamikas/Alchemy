"""Extract crystallization conditions from coordinate files and the RCSB cache.

Conditions supply review context only; they do not affect scores.
The RCSB client and its on-disk cache live in ``rcsb_metadata_cache``.
"""

from __future__ import annotations

import gzip
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import gemmi

from codes import CrystallizationDataStatus
from output_rows import CsvValue
from rcsb_metadata_cache import read_cache_payload
from run_logging import logger_for

logger = logger_for(__name__)

#: Which entry, source, and crystal a condition row describes, with provenance.
_CONDITION_IDENTITY_COLUMNS = (
    "pdbID",
    "crystallization_condition_id",
    "source_format",
    "metadata_source",
    "metadata_retrieved_at_utc",
    "entry_revision_date",
    "crystal_id",
)
#: What the condition says. A row with all of these blank is not recorded; a
#: crystal ID on its own does not describe a condition.
_CONDITION_CONTENT_COLUMNS = (
    "method",
    "pH",
    "pH_range",
    "temperature_K",
    "temperature_details",
    "raw_details",
)
CONDITION_COLUMNS = (*_CONDITION_IDENTITY_COLUMNS, *_CONDITION_CONTENT_COLUMNS)
#: Per-row column paired with the ``_exptl_crystal_grow`` item that fills it.
#: RCSB's GraphQL ``exptl_crystal_grow`` fields carry the same item names.
_EXPTL_CRYSTAL_GROW_ITEMS = (
    ("crystal_id", "crystal_id"),
    ("method", "method"),
    ("pH", "pH"),
    ("pH_range", "pdbx_pH_range"),
    ("temperature_K", "temp"),
    ("temperature_details", "temp_details"),
    ("raw_details", "pdbx_details"),
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
    "PB": (
        "lead acetate",
        "lead nitrate",
        "lead chloride",
        "lead ion",
        "lead ions",
        "trimethyllead",
        "trimethyl lead",
    ),
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
# Mass units in deposited condition text, such as ``PROTEIN 5 MG/ML``, spell
# the magnesium symbol. They are blanked before symbol detection.
_MASS_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:MG|Mg)\s*/\s*(?:[MDUmdu]|\u00b5)?L\b|\d[\d.]*\s*MG\b"
)
# Monovalent metals whose common salts have digit-free formulas (``NaBr``,
# ``KI``, ``CsCl``, ``NaOH``, ``KOAc``). ``_FORMULA_SUFFIX`` requires a digit,
# so these are matched against the anion list instead.
_DIGIT_FREE_SALT_METALS = frozenset({"LI", "NA", "K", "RB", "CS", "AG", "CU", "TL"})
_DIGIT_FREE_SALT_ANIONS = ("Cl", "Br", "I", "F", "OH", "OAc")
# Hyphenated prefixes such as ``CO-CRYSTALLIZED`` or ``Co-expressed`` spell a
# bare symbol without naming a metal.
_HYPHENATED_PREFIX = r"(?!-[A-Za-z])"
_PH_RE = re.compile(r"(?i)\bp\s*h\s*(?:=|:)?\s*(-?\d+(?:\.\d+)?)")
_PH_RANGE_RE = re.compile(
    r"(?i)\bp\s*h\s*(?:range)?\s*(?:=|:)?\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*"
    r"(-?\d+(?:\.\d+)?)(?![\d.]*\s*%)"
)
_TEMP_RE = re.compile(
    r"(?i)\b(?:temperature|temp\.?)\s*(?:=|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?:(?P<kelvin>k\b|kelvin\b)"
    r"|(?P<celsius>(?:deg(?:rees?)?\.?\s*|\u00b0\s*)?(?:c\b|celsius\b|centigrade\b)))?"
)
# A unitless temperature is taken as kelvin only inside this window; a value
# such as ``TEMPERATURE 20`` is ambiguous and is not recorded.
_UNITLESS_KELVIN_RANGE = (150.0, 400.0)


@dataclass(frozen=True, slots=True)
class CrystallizationExtraction:
    """Collect normalized condition rows and their entry summary."""

    conditions: tuple[dict[str, CsvValue], ...]
    summary: dict[str, CsvValue]


class _MmcifCategoryBlock(Protocol):
    """Typed view over Gemmi's incompletely annotated category accessor."""

    def get_mmcif_category(
        self, name: str, raw: bool = False
    ) -> dict[str, list[str]]: ...


def _clean_cif_value(value: object) -> str:
    """Blank the mmCIF placeholders.

    Gemmi's non-raw category view maps ``?`` to ``None`` and ``.`` to ``False``;
    the cache stores the item as JSON ``null`` or the literal placeholder.
    """
    if value is None or value is False:
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
    return ph, ph_range, _temperature_kelvin(text)


def _temperature_kelvin(text: str) -> str:
    """Read a stated temperature in kelvin, converting an explicit Celsius value."""
    match = _TEMP_RE.search(text)
    if match is None:
        return ""
    value = match.group("value")
    if match.group("celsius"):
        return f"{float(value) + 273.15:.2f}"
    low, high = _UNITLESS_KELVIN_RANGE
    if match.group("kelvin") or low <= float(value) <= high:
        return value
    return ""


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


def _condition_row(
    pdb_id: str,
    ordinal: int,
    source_format: str,
    provenance: Mapping[str, CsvValue],
    values: Mapping[str, str],
) -> dict[str, CsvValue]:
    """Assemble one condition row with its keys in ``CONDITION_COLUMNS`` order.

    ``ordinal`` numbers the entry's recorded rows from one. ``values`` holds the
    per-row cells, ``crystal_id`` and the ``_CONDITION_CONTENT_COLUMNS``, keyed
    by column; a column it omits is blank.
    """
    return {
        "pdbID": pdb_id,
        "crystallization_condition_id": f"{pdb_id}:condition:{ordinal}",
        "source_format": source_format,
        **provenance,
        "crystal_id": values.get("crystal_id", ""),
        **{column: values.get(column, "") for column in _CONDITION_CONTENT_COLUMNS},
    }


def _condition_rows(
    pdb_id: str,
    source_format: str,
    provenance: Mapping[str, CsvValue],
    values_per_row: Iterable[Mapping[str, str]],
) -> list[dict[str, CsvValue]]:
    """Number the rows that say anything about the condition and build them.

    A row whose content columns are all blank is dropped before numbering, so
    the recorded rows are always ``condition:1``, ``condition:2``, and so on.
    """
    rows: list[dict[str, CsvValue]] = []
    for values in values_per_row:
        if not any(
            values.get(column, "").strip() for column in _CONDITION_CONTENT_COLUMNS
        ):
            continue
        rows.append(
            _condition_row(pdb_id, len(rows) + 1, source_format, provenance, values)
        )
    return rows


def _mmcif_conditions(
    pdb_id: str, path: str, metadata_source: str
) -> list[dict[str, CsvValue]]:
    document = gemmi.cif.read(path)
    values_per_row: list[dict[str, str]] = []
    for block in document:
        category = cast(_MmcifCategoryBlock, block).get_mmcif_category(
            "_exptl_crystal_grow."
        )
        if not category:
            continue
        row_count = max((len(values) for values in category.values()), default=0)
        values_per_row.extend(
            {
                column: _category_value(category, item, index)
                for column, item in _EXPTL_CRYSTAL_GROW_ITEMS
            }
            for index in range(row_count)
        )
    return _condition_rows(
        pdb_id, "mmcif", _provenance(metadata_source), values_per_row
    )


def _pdb_conditions(
    pdb_id: str, path: str, metadata_source: str
) -> list[dict[str, CsvValue]]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        pieces = [line[10:].strip() for line in handle if line.startswith("REMARK 280")]
    details = " ".join(piece for piece in pieces if piece).strip()
    # REMARK 280 also carries the solvent content and Matthews coefficient;
    # only the text after the marker describes the crystallization.
    marker = re.search(r"(?i)CRYSTALLIZATION CONDITIONS\s*:\s*", details)
    if marker is None:
        return []
    details = details[marker.end() :].strip()
    if not details or details.upper() == "NULL":
        return []
    ph, ph_range, temperature = _parse_text_measurements(details)
    values = {
        "method": _method_from_text(details),
        "pH": ph,
        "pH_range": ph_range,
        "temperature_K": temperature,
        "raw_details": details,
    }
    return _condition_rows(pdb_id, "pdb", _provenance(metadata_source), [values])


def detected_metals(text: str) -> frozenset[str]:
    """Return explicitly named or formula-like metals in condition text."""
    text = _MASS_UNIT_RE.sub(" ", text)
    lowered = text.lower()
    detected = {
        symbol
        for symbol, aliases in _METAL_ALIASES.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases)
    }
    for symbol in _METAL_ALIASES:
        title_symbol = symbol.title()
        if len(symbol) > 1 and re.search(
            rf"\b(?:{symbol}|{title_symbol})\b{_HYPHENATED_PREFIX}", text
        ):
            detected.add(symbol)
            continue
        formula_symbol = rf"(?:{re.escape(title_symbol)}|{re.escape(symbol)})"
        if re.search(rf"(?<![A-Za-z]){formula_symbol}{_FORMULA_SUFFIX}", text):
            detected.add(symbol)
            continue
        if symbol in _DIGIT_FREE_SALT_METALS and re.search(
            _digit_free_salt_pattern(symbol), text
        ):
            detected.add(symbol)
    return frozenset(detected)


def _digit_free_salt_pattern(symbol: str) -> str:
    """Match ``NaBr``/``NABR``-style formulas of ``symbol`` as whole tokens."""
    title = "|".join(symbol.title() + anion for anion in _DIGIT_FREE_SALT_ANIONS)
    upper = "|".join(symbol + anion.upper() for anion in _DIGIT_FREE_SALT_ANIONS)
    return rf"(?<![A-Za-z])(?:{title}|{upper})(?![A-Za-z])"


def unavailable_summary(
    pdb_id: str,
    status: CrystallizationDataStatus = CrystallizationDataStatus.INPUT_UNAVAILABLE,
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
    """Summarize at least one recorded condition row.

    The rows share their provenance, so the first row supplies it; an entry
    with no rows gets ``unavailable_summary`` from the caller instead.
    """
    first = rows[0]
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
    raw_details = [str(row.get("raw_details", "")).strip() for row in rows]
    raw_text = " || ".join(dict.fromkeys(detail for detail in raw_details if detail))
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
        "crystallization_data_status": CrystallizationDataStatus.AVAILABLE,
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
    if not os.path.isfile(path):
        return CrystallizationExtraction(
            (),
            unavailable_summary(
                pdb_id,
                source_format=source_format,
                metadata_source=metadata_source,
            ),
        )
    try:
        rows = (
            _mmcif_conditions(pdb_id, path, metadata_source)
            if source_format == "mmcif"
            else _pdb_conditions(pdb_id, path, metadata_source)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug(
            "%s: crystallization conditions unparseable from %s: %s: %s",
            pdb_id,
            path,
            type(exc).__name__,
            exc,
        )
        return CrystallizationExtraction(
            (),
            unavailable_summary(
                pdb_id,
                CrystallizationDataStatus.UNPARSEABLE,
                source_format=source_format,
                metadata_source=metadata_source,
            ),
        )
    if not rows:
        return CrystallizationExtraction(
            (),
            unavailable_summary(
                pdb_id,
                CrystallizationDataStatus.NOT_REPORTED,
                source_format=source_format,
                metadata_source=metadata_source,
            ),
        )
    return CrystallizationExtraction(tuple(rows), _summary(pdb_id, rows, source_format))


def _condition_rows_from_cache(
    pdb_id: str, payload: Mapping[str, Any]
) -> list[dict[str, CsvValue]]:
    provenance = _provenance(
        str(payload["metadata_source"]),
        str(payload["retrieved_at_utc"]),
        str(payload["entry_revision_date"]),
    )
    conditions = cast(list[dict[str, Any]], payload["conditions"])
    values_per_row = (
        {
            column: _clean_cif_value(condition.get(item, ""))
            for column, item in _EXPTL_CRYSTAL_GROW_ITEMS
        }
        for condition in conditions
    )
    return _condition_rows(pdb_id, "json", provenance, values_per_row)


def cached_rcsb_crystallization_conditions(
    pdb_id: str, cache_root: str
) -> CrystallizationExtraction | None:
    """Return a validated cached RCSB extraction, or ``None`` on a cache miss."""
    pdb_id = pdb_id.strip().lower()
    if not cache_root or len(pdb_id) != 4:
        return None
    payload = read_cache_payload(cache_root, pdb_id)
    if payload is None:
        return None
    rows = _condition_rows_from_cache(pdb_id, payload)
    if rows:
        return CrystallizationExtraction(tuple(rows), _summary(pdb_id, rows, "json"))
    status = (
        CrystallizationDataStatus.NOT_REPORTED
        if payload["entry_available"]
        else CrystallizationDataStatus.INPUT_UNAVAILABLE
    )
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
    local_available = (
        local.summary["crystallization_data_status"]
        == CrystallizationDataStatus.AVAILABLE
    )
    deposited_available = (
        deposited.summary["crystallization_data_status"]
        == CrystallizationDataStatus.AVAILABLE
    )
    if prefer_coordinate_file and local_available:
        return local
    if deposited_available:
        return deposited
    if local_available:
        return local
    return local if prefer_coordinate_file else deposited
