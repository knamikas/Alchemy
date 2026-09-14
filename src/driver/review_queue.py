"""Write the review queue: REVIEW/SUSPECT sites joined to crystallization context.

A derived triage view regenerated from the completed confidence scores and the
crystallization summary. Nothing here feeds back into scoring.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from codes import ConfidenceLevel
from confidence_score.schema import parse_csv_bool
from crystallization_conditions import PROMISCUOUS_TRANSITION_METALS, SUMMARY_COLUMNS
from output_rows import CsvValue, scientific_csv_value

#: Summary columns copied onto every queued site; ``pdbID`` is already there.
_SUMMARY_CONTEXT_COLUMNS = tuple(
    column for column in SUMMARY_COLUMNS if column != "pdbID"
)
REVIEW_CONTEXT_COLUMNS = (
    *_SUMMARY_CONTEXT_COLUMNS,
    "crystallization_contains_modeled_metal",
    "crystallization_contains_different_promiscuous_transition_metal",
    "crystallization_context_flags",
)
_QUEUED_LEVELS = frozenset({ConfidenceLevel.REVIEW, ConfidenceLevel.SUSPECT})


@dataclass(frozen=True, slots=True)
class _ReviewContext:
    """Entry-level crystallization context for one queued site.

    ``cells`` carries the summary columns other than ``pdbID`` exactly as read
    from the summary CSV, blank when the entry has no summary row. The metal
    questions answer ``None`` while no condition data is available, so the
    written cell stays empty instead of asserting a negative.
    """

    cells: Mapping[str, str]
    available: bool
    detected_metals: frozenset[str]

    def flag(self, column: str) -> bool:
        """Read a serialized boolean cell; blank reads as false."""
        return parse_csv_bool(self.cells[column])

    def contains_modeled_metal(self, modeled_metal: str) -> bool | None:
        """Whether the conditions name the site's own metal; unknown without one."""
        if not self.available or not modeled_metal:
            return None
        return modeled_metal in self.detected_metals

    def contains_different_promiscuous_transition_metal(
        self, modeled_metal: str
    ) -> bool | None:
        """Whether the conditions name a promiscuous transition metal other than the modeled one."""
        if not self.available:
            return None
        other_metals = self.detected_metals - {modeled_metal}
        return bool(other_metals & PROMISCUOUS_TRANSITION_METALS)


def _read_summaries(crystallization_summary_path: str) -> dict[str, dict[str, str]]:
    """Index the crystallization summary by lower-case entry ID; empty without a file."""
    if not os.path.isfile(crystallization_summary_path):
        return {}
    with open(crystallization_summary_path, newline="", encoding="utf-8") as handle:
        return {row["pdbID"].strip().lower(): row for row in csv.DictReader(handle)}


def _review_context(summary: Mapping[str, str] | None) -> _ReviewContext:
    """Build the context for one site from its entry's summary row, if any."""
    cells = {
        column: summary.get(column, "") if summary else ""
        for column in _SUMMARY_CONTEXT_COLUMNS
    }
    detected_metals = frozenset(
        value for value in cells["crystallization_detected_metals"].split("|") if value
    )
    return _ReviewContext(
        cells=cells,
        available=cells["crystallization_data_status"] == "available",
        detected_metals=detected_metals,
    )


def _context_flags(context: _ReviewContext, modeled_metal: str) -> list[str]:
    """Name the condition facts that hold for this site, in a fixed order."""
    candidates: tuple[tuple[str, bool | None], ...] = (
        ("modeled_metal", context.contains_modeled_metal(modeled_metal)),
        (
            "different_promiscuous_transition_metal",
            context.contains_different_promiscuous_transition_metal(modeled_metal),
        ),
        (
            "heavy_additive_phasing_metal",
            context.flag("crystallization_heavy_additive_phasing_metal"),
        ),
        ("ni_co_like_metal", context.flag("crystallization_ni_co_like_metal")),
        ("sulfate", context.flag("crystallization_sulfate")),
        ("cacodylate", context.flag("crystallization_cacodylate")),
        ("acetate", context.flag("crystallization_acetate")),
    )
    return [name for name, present in candidates if present]


def _review_row(
    site: Mapping[str, str], context: _ReviewContext
) -> dict[str, CsvValue]:
    """Join one scored site to its entry context and the derived metal columns."""
    modeled_metal = site.get("metal_element", "").strip().upper()
    return {
        **site,
        **context.cells,
        "crystallization_contains_modeled_metal": context.contains_modeled_metal(
            modeled_metal
        ),
        "crystallization_contains_different_promiscuous_transition_metal": (
            context.contains_different_promiscuous_transition_metal(modeled_metal)
        ),
        "crystallization_context_flags": "|".join(
            _context_flags(context, modeled_metal)
        ),
    }


def write_review_queue(
    confidence_scores_path: str,
    crystallization_summary_path: str,
    output_path: str,
    confidence_columns: Sequence[str],
) -> int:
    """Write REVIEW/SUSPECT sites joined to entry-level condition context.

    The header is always written; a missing scores file leaves an empty queue,
    and a scores file with a different schema raises ``ValueError``.
    """
    output_columns = (*confidence_columns, *REVIEW_CONTEXT_COLUMNS)
    summaries = _read_summaries(crystallization_summary_path)
    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=output_columns)
        writer.writeheader()
        if not os.path.isfile(confidence_scores_path):
            return 0
        with open(confidence_scores_path, newline="", encoding="utf-8") as scores:
            reader = csv.DictReader(scores)
            if reader.fieldnames != list(confidence_columns):
                raise ValueError(
                    "confidence score schema is incompatible with review queue"
                )
            for site in reader:
                if site.get("alchemy_level") not in _QUEUED_LEVELS:
                    continue
                context = _review_context(summaries.get(site["pdbID"].strip().lower()))
                joined = _review_row(site, context)
                writer.writerow(
                    {
                        column: scientific_csv_value(joined[column])
                        for column in output_columns
                    }
                )
                count += 1
    return count
