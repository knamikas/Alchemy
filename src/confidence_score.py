"""Collect, classify, and rank Alchemy metal sites.

The authoritative PASS/REVIEW/SUSPECT verdict is determined from raw density
and geometry thresholds.  A complete database run also freezes independent
density and geometry distributions so later runs can receive empirical support
scores without ever defining their verdicts from the reference population.
"""

import argparse
import bisect
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TextIO

from analysis_config import analysis_config_id
from output_rows import MetalStatsRow, scientific_csv_value
from reference_data import cofactor_ids, reference_data_id
from worker_contracts import MAX_ANALYZED_METAL_SITES

REFERENCE_METADATA_FILE = "metadata.json"
REFERENCE_DISTRIBUTION_FILE = "component_distributions.csv"
CONFIDENCE_METHOD_VERSION = "three_level_rms_2026_v1"
CONFIDENCE_SCHEMA_VERSION = 3
COHORT_WEIGHTING = "per_metal_site"
SCORE_DECIMAL_PLACES = 6
METRIC_DECIMAL_PLACES = 12
DENSITY_REVIEW_THRESHOLD = 3.0
DENSITY_SUSPECT_THRESHOLD = 6.0
GEOMETRY_REVIEW_THRESHOLD = 1.0
GEOMETRY_SUSPECT_THRESHOLD = 2.0
EDSTATS_SATURATION_MAGNITUDE = 99.9

SITE_KEY_COLUMNS = (
    "pdbID",
    "metal_site_id",
)
LEGACY_SITE_KEY_COLUMNS = (
    "pdbID",
    "metal_model_index",
    "metal_chain_index",
    "metal_residue_index",
    "metal_atom_index",
)

IDENTITY_COLUMNS = (
    "pdbID",
    "metal_site_id",
    "category",
    "density_observation_id",
    "density_scope",
    "density_shared_site_count",
    "density_is_shared",
    "coordinate_mapping_status",
    "selected_metal_site_status",
    "metal_model_index",
    "metal_chain_index",
    "metal_residue_index",
    "metal_atom_index",
    "metal_resname",
    "metal_chain",
    "metal_resnum",
    "metal_atom",
    "metal_element",
    "metal_icode",
    "metal_altloc",
)

CONFIDENCE_INPUT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "rszd",
    "rszd_abs",
    "rszd_negative",
    "rszd_positive",
    "density_saturated",
    "assigned_contact_count",
    "reference_covered_contact_count",
    "geometry_bond_count",
    "geometry_coverage",
    "geometry_rms_zbond",
    "geometry_max_abs_zbond",
    "geometry_mean_abs_zbond",
    "geometry_mean_signed_zbond",
    "worst_bond",
    "worst_bond_source",
    "worst_bond_neighbor_resname",
    "worst_bond_neighbor_chain",
    "worst_bond_neighbor_resnum",
    "worst_bond_neighbor_atom",
    "declared_contact_count",
    "inferred_contact_count",
    "declared_scored_bond_count",
    "inferred_scored_bond_count",
    "geometry_contact_basis",
    "multi_donor_contact_count",
    "suspect_multi_donor_residue_group_count",
    "context_warning",
    "context_warning_reasons",
    "confidence_inputs_status",
    "confidence_inputs_missing_reasons",
)

ANALYSIS_COLUMNS = (
    "density_level",
    "density_score",
    "geometry_level",
    "geometry_score",
    "alchemy_level",
    "alchemy_score",
    "evidence_basis",
    "verdict_reason",
    "score_policy_version",
    "confidence_reference_version",
    "confidence_cohort_id",
    "confidence_cohort_size",
    "density_reference_size",
    "geometry_reference_size",
)

CONFIDENCE_INPUT_STATUSES = frozenset(
    {"complete", "density_only", "geometry_only", "unscorable"}
)
EVIDENCE_BASES = frozenset(
    {
        "density_and_geometry",
        "density_only",
        "geometry_only",
        "no_assessable_evidence",
    }
)
VERDICT_REASONS = frozenset(
    {
        "no_assessable_evidence",
        "density_and_geometry_suspect",
        "density_suspect",
        "geometry_suspect",
        "review_plus_review",
        "density_review",
        "geometry_review",
        "all_available_components_pass",
    }
)
INPUT_STATUS_POLICY = "independent_component_availability_v1"
CONFIDENCE_BOOLEAN_COLUMNS = frozenset(
    {"density_is_shared", "density_saturated", "context_warning"}
)
REFERENCE_METADATA_FIELDS = frozenset(
    {
        "confidence_method_version",
        "confidence_schema_version",
        "cohort_weighting",
        "score_decimal_places",
        "metric_decimal_places",
        "density_thresholds",
        "density_saturation_value",
        "density_saturation_policy",
        "geometry_thresholds",
        "support_score_method",
        "geometry_statistic",
        "overall_rule",
        "coverage_policy",
        "input_status_policy",
        "maximum_entry_metal_sites",
        "reference_data_id",
        "reference_id",
        "distribution_file",
        "density_distinct_value_count",
        "geometry_distinct_value_count",
        "density_reference_size",
        "geometry_reference_size",
        "cohort_id",
        "confidence_inputs_file",
        "confidence_inputs_sha256",
        "input_row_count",
        "input_entry_count",
        "scorable_entry_count",
        "input_status_counts",
        "source_manifest_file",
        "source_manifest_sha256",
        "source_entry_count",
        "manifest_status_counts",
        "no_metals_entry_count",
        "metal_site_limit_exceeded_entry_count",
        "metal_bearing_entry_count",
        "software_versions",
        "analysis_config_id",
    }
)


def _required_columns(
    fieldnames: Sequence[str] | None, required: Iterable[str], label: str
) -> None:
    missing = [column for column in required if column not in (fieldnames or ())]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _site_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    site_id = str(row.get("metal_site_id", "")).strip()
    columns = SITE_KEY_COLUMNS if site_id else LEGACY_SITE_KEY_COLUMNS
    return tuple(str(row.get(column, "")).strip() for column in columns)


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _format_decimal(value: float, decimal_places: int = 6) -> str:
    """Render a confidence number without trailing zeros, blanking non-finite.

    Distinct from ``structure_analysis._format_number``, which renders a fixed
    number of decimals and blanks ``None`` rather than non-finite values: the
    two obey different contracts and must not be merged.
    """
    if not math.isfinite(value):
        return ""
    return f"{value:.{decimal_places}f}".rstrip("0").rstrip(".")


def canonical_support_score(value: float) -> float:
    """Round a support score to its canonical serialized precision."""
    return float(f"{value:.{SCORE_DECIMAL_PLACES}f}")


def canonical_metric(value: float) -> float:
    """Round an input metric to its canonical serialized precision."""
    return float(f"{value:.{METRIC_DECIMAL_PLACES}f}")


def _confidence_csv_value(column: str, value: object) -> object:
    if column in CONFIDENCE_BOOLEAN_COLUMNS and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized
    return scientific_csv_value(value)


def _read_csv(path: str, label: str) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{label} has no CSV header")
        return tuple(reader.fieldnames), list(reader)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_provenance(path: str) -> dict[str, Any]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _required_columns(reader.fieldnames, ("analysis_config_id",), "source manifest")
        rows = list(reader)

    manifest_config_ids = [row.get("analysis_config_id", "").strip() for row in rows]
    analysis_config_ids = set(manifest_config_ids)
    if (
        not manifest_config_ids
        or "" in analysis_config_ids
        or len(analysis_config_ids) != 1
    ):
        raise ValueError(
            "source manifest must contain exactly one analysis configuration identity"
        )

    status_counts = Counter(row.get("status", "") for row in rows)
    software_columns = (
        "alchemy_version",
        "alchemy_commit",
        "gemmi_version",
        "ccp4_version",
    )
    software = {
        column: sorted({row.get(column, "") for row in rows if row.get(column, "")})
        for column in software_columns
    }
    return {
        "source_manifest_file": os.path.basename(path),
        "source_manifest_sha256": _file_sha256(path),
        "source_entry_count": len(rows),
        "manifest_status_counts": dict(sorted(status_counts.items())),
        "no_metals_entry_count": sum(_true(row.get("no_metals", "")) for row in rows),
        "metal_site_limit_exceeded_entry_count": sum(
            _true(row.get("metal_site_limit_exceeded", "")) for row in rows
        ),
        "metal_bearing_entry_count": sum(
            (_finite_float(row.get("n_metals", "")) > 0) for row in rows
        ),
        "software_versions": software,
        "analysis_config_id": next(iter(analysis_config_ids)),
    }


def _bond_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored_bonds: list[tuple[float, Mapping[str, Any]]] = []
    reference_covered = 0
    declared = 0
    inferred = 0
    declared_scored = 0
    inferred_scored = 0
    multi_donor = 0
    for row in rows:
        reference_covered += _true(row.get("reference_covered", ""))
        is_declared = _true(row.get("declared_connection", ""))
        is_inferred = row.get("coordination_status", "").strip() == "inferred"
        declared += is_declared
        inferred += is_inferred
        multi_donor += _true(row.get("multi_donor_detected", ""))
        zscore = _finite_float(row.get("zscore", ""))
        if _true(row.get("score_eligible", "")) and math.isfinite(zscore):
            scored_bonds.append((zscore, row))
            declared_scored += is_declared
            inferred_scored += is_inferred

    assigned = len(rows)
    coverage = reference_covered / assigned if assigned else 0.0
    largest = max(scored_bonds, key=lambda item: abs(item[0])) if scored_bonds else None
    largest_row: Mapping[str, Any] = largest[1] if largest else {}
    zscores = [item[0] for item in scored_bonds]
    geometry_contact_basis = (
        "declared_and_inferred"
        if declared_scored and inferred_scored
        else "declared_only"
        if declared_scored
        else "inferred_only"
        if inferred_scored
        else "none"
    )
    return {
        "assigned_contact_count": assigned,
        "reference_covered_contact_count": reference_covered,
        "geometry_bond_count": len(zscores),
        "geometry_coverage": _format_decimal(coverage),
        "geometry_rms_zbond": (
            _format_decimal(
                math.sqrt(sum(value * value for value in zscores) / len(zscores)),
                METRIC_DECIMAL_PLACES,
            )
            if zscores
            else ""
        ),
        "geometry_max_abs_zbond": (
            _format_decimal(abs(largest[0]), METRIC_DECIMAL_PLACES) if largest else ""
        ),
        "geometry_mean_abs_zbond": (
            _format_decimal(
                sum(abs(value) for value in zscores) / len(zscores),
                METRIC_DECIMAL_PLACES,
            )
            if zscores
            else ""
        ),
        "geometry_mean_signed_zbond": (
            _format_decimal(sum(zscores) / len(zscores), METRIC_DECIMAL_PLACES)
            if zscores
            else ""
        ),
        "worst_bond": largest_row.get("contact_id", ""),
        "worst_bond_source": (
            "declared"
            if largest and _true(largest_row.get("declared_connection", ""))
            else "inferred"
            if largest
            and largest_row.get("coordination_status", "").strip() == "inferred"
            else ""
        ),
        "worst_bond_neighbor_resname": largest_row.get("neighbor_resname", ""),
        "worst_bond_neighbor_chain": largest_row.get("neighbor_chain", ""),
        "worst_bond_neighbor_resnum": largest_row.get("neighbor_resnum", ""),
        "worst_bond_neighbor_atom": largest_row.get("neighbor_atom", ""),
        "declared_contact_count": declared,
        "inferred_contact_count": inferred,
        "declared_scored_bond_count": declared_scored,
        "inferred_scored_bond_count": inferred_scored,
        "geometry_contact_basis": geometry_contact_basis,
        "multi_donor_contact_count": multi_donor,
    }


def _orphan_bond_site_input(
    key: tuple[str, ...], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Preserve a bond-bearing site whose density row could not be joined."""
    first = rows[0]
    summary = _bond_summary(rows)
    warning_reasons: list[str] = []
    for row in rows:
        warning_reasons.extend(
            reason
            for reason in row.get("context_warning_reasons", "").split("|")
            if reason
        )
    missing_reasons = ["rszd_unavailable", "density_row_unavailable"]
    if summary["reference_covered_contact_count"] == 0:
        missing_reasons.append("no_geometry_reference")
    elif summary["geometry_bond_count"] < summary["reference_covered_contact_count"]:
        missing_reasons.append("zbond_unavailable_for_reference")
    if summary["reference_covered_contact_count"] < summary["assigned_contact_count"]:
        missing_reasons.append("partial_geometry_coverage")
    values: dict[str, Any] = dict.fromkeys(CONFIDENCE_INPUT_COLUMNS, "")
    values.update(
        {
            "pdbID": first.get("pdbID", key[0]),
            "category": (
                "cofactor"
                if str(first.get("metal_resname", "")).upper() in cofactor_ids()
                else "metal"
                if first.get("parent_type", "") == "ion"
                else ""
            ),
            "metal_site_id": first.get("metal_site_id", ""),
            "coordinate_mapping_status": "density_row_unavailable",
            "selected_metal_site_status": "selected_without_density_row",
            "metal_model_index": first.get("metal_model_index", ""),
            "metal_chain_index": first.get("metal_chain_index", ""),
            "metal_residue_index": first.get("metal_residue_index", ""),
            "metal_atom_index": first.get("metal_atom_index", ""),
            "metal_resname": first.get("metal_resname", ""),
            "metal_chain": first.get("metal_chain", ""),
            "metal_resnum": first.get("metal_resnum", ""),
            "metal_atom": first.get("metal_atom", ""),
            "metal_element": first.get("metal_element", ""),
            "metal_icode": first.get("metal_icode", ""),
            "metal_altloc": first.get("metal_altloc", ""),
            **summary,
            "context_warning": any(
                _true(row.get("context_warning", "")) for row in rows
            ),
            "context_warning_reasons": "|".join(dict.fromkeys(warning_reasons)),
            "confidence_inputs_status": (
                "geometry_only" if summary["geometry_bond_count"] else "unscorable"
            ),
            "confidence_inputs_missing_reasons": "|".join(missing_reasons),
        }
    )
    return values


def prepare_confidence_inputs(
    stats_rows: Sequence[Mapping[str, Any]],
    bond_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic confidence inputs for every selected metal site."""
    bonds_by_site: defaultdict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in bond_rows:
        bonds_by_site[_site_key(row)].append(row)

    seen_sites: set[tuple[str, ...]] = set()
    prepared: list[dict[str, Any]] = []
    for stats in stats_rows:
        if stats.get("selected_metal_site_status", "").strip() != "selected":
            continue
        key = _site_key(stats)
        if key in seen_sites:
            raise ValueError(
                "metal statistics contain duplicate site key: " + "/".join(key)
            )
        seen_sites.add(key)

        rszd = _finite_float(stats.get("ZDm", ""))
        rszd_abs = abs(rszd)
        negative = _finite_float(stats.get("ZD-m", ""))
        positive = _finite_float(stats.get("ZD+m", ""))
        summary = _bond_summary(bonds_by_site.pop(key, ()))
        missing_reasons: list[str] = []
        if str(stats.get("metal_coordinates_valid", "")).strip().lower() == "false":
            missing_reasons.append("non_finite_metal_coordinates")
        if not math.isfinite(rszd_abs):
            missing_reasons.append("rszd_unavailable")
        if summary["assigned_contact_count"] == 0:
            missing_reasons.append("no_assigned_contacts")
        else:
            if summary["reference_covered_contact_count"] == 0:
                missing_reasons.append("no_geometry_reference")
            elif (
                summary["geometry_bond_count"]
                < summary["reference_covered_contact_count"]
            ):
                missing_reasons.append("zbond_unavailable_for_reference")
            if (
                summary["reference_covered_contact_count"]
                < summary["assigned_contact_count"]
            ):
                missing_reasons.append("partial_geometry_coverage")

        density_available = math.isfinite(rszd_abs)
        geometry_available = summary["geometry_bond_count"] > 0
        status = (
            "complete"
            if density_available and geometry_available
            else "density_only"
            if density_available
            else "geometry_only"
            if geometry_available
            else "unscorable"
        )

        output = {column: stats.get(column, "") for column in IDENTITY_COLUMNS}
        output.update(
            {
                "rszd": _format_decimal(rszd, METRIC_DECIMAL_PLACES),
                "rszd_abs": _format_decimal(rszd_abs, METRIC_DECIMAL_PLACES),
                "rszd_negative": _format_decimal(negative),
                "rszd_positive": _format_decimal(positive),
                "density_saturated": (
                    math.isfinite(rszd_abs)
                    and math.isclose(rszd_abs, EDSTATS_SATURATION_MAGNITUDE)
                ),
                **summary,
                "suspect_multi_donor_residue_group_count": stats.get(
                    "suspect_multi_donor_residue_group_count", ""
                ),
                "context_warning": stats.get("context_warning", ""),
                "context_warning_reasons": stats.get("context_warning_reasons", ""),
                "confidence_inputs_status": status,
                "confidence_inputs_missing_reasons": "|".join(missing_reasons),
            }
        )
        prepared.append(output)

    for key, rows in sorted(bonds_by_site.items()):
        prepared.append(_orphan_bond_site_input(key, rows))
    return prepared


def prepare_result_confidence_inputs(
    stats_rows: Sequence[MetalStatsRow],
    bond_rows: Sequence[Mapping[str, Any]],
    stats_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Prepare confidence rows from one in-memory Alchemy worker result."""
    flattened = [row.as_output_dict(stats_columns) for row in stats_rows]
    return prepare_confidence_inputs(flattened, bond_rows)


def complete_confidence_site_count(
    rows: Sequence[Mapping[str, Any]],
    pdb_id: str,
    selected_site_count: int,
    missing_reason: str = "",
) -> list[dict[str, Any]]:
    """Retain unresolved placeholders so no manifest-counted site disappears."""
    if len(rows) > selected_site_count:
        raise ValueError(f"confidence inputs exceed selected metal count for {pdb_id}")
    completed = [dict(row) for row in rows]
    if missing_reason:
        for row in completed:
            reasons = [
                reason
                for reason in row.get("confidence_inputs_missing_reasons", "").split(
                    "|"
                )
                if reason
            ]
            if missing_reason not in reasons:
                reasons.append(missing_reason)
            row["confidence_inputs_missing_reasons"] = "|".join(reasons)
    for index in range(len(rows), selected_site_count):
        values: dict[str, Any] = dict.fromkeys(CONFIDENCE_INPUT_COLUMNS, "")
        values.update(
            {
                "pdbID": pdb_id,
                "selected_metal_site_status": "selected_site_unresolved",
                "metal_atom_index": f"unresolved-{index + 1}",
                "context_warning": True,
                "context_warning_reasons": "site_evidence_unavailable",
                "confidence_inputs_status": "unscorable",
                "confidence_inputs_missing_reasons": (
                    "rszd_unavailable|site_identity_unavailable|"
                    "site_evidence_unavailable"
                    + (f"|{missing_reason}" if missing_reason else "")
                ),
            }
        )
        completed.append(values)
    return completed


def component_level(value: float, review: float, suspect: float) -> str:
    """Classify one non-negative measurement at the final raw thresholds."""
    if not math.isfinite(value) or value < 0:
        return "INCOMPLETE"
    if value < review:
        return "PASS"
    if value < suspect:
        return "REVIEW"
    return "SUSPECT"


def density_level(rszd_abs: float) -> str:
    """Classify an absolute RSZD value at the density thresholds."""
    return component_level(
        rszd_abs, DENSITY_REVIEW_THRESHOLD, DENSITY_SUSPECT_THRESHOLD
    )


def geometry_level(geometry_rms_zbond: float) -> str:
    """Classify an RMS bond Z score at the geometry thresholds."""
    return component_level(
        geometry_rms_zbond, GEOMETRY_REVIEW_THRESHOLD, GEOMETRY_SUSPECT_THRESHOLD
    )


def classify_site(rszd_abs: float, geometry_rms_zbond: float) -> dict[str, str]:
    """Apply the non-compensatory final decision matrix to one site."""
    density = density_level(rszd_abs)
    geometry = geometry_level(geometry_rms_zbond)
    available = [level for level in (density, geometry) if level != "INCOMPLETE"]
    evidence_basis = (
        "density_and_geometry"
        if len(available) == 2
        else "density_only"
        if density != "INCOMPLETE"
        else "geometry_only"
        if geometry != "INCOMPLETE"
        else "no_assessable_evidence"
    )

    if not available:
        overall = "INCOMPLETE"
        reason = "no_assessable_evidence"
    elif density == "SUSPECT" and geometry == "SUSPECT":
        overall = "SUSPECT"
        reason = "density_and_geometry_suspect"
    elif density == "SUSPECT":
        overall = "SUSPECT"
        reason = "density_suspect"
    elif geometry == "SUSPECT":
        overall = "SUSPECT"
        reason = "geometry_suspect"
    elif density == "REVIEW" and geometry == "REVIEW":
        overall = "SUSPECT"
        reason = "review_plus_review"
    elif density == "REVIEW":
        overall = "REVIEW"
        reason = "density_review"
    elif geometry == "REVIEW":
        overall = "REVIEW"
        reason = "geometry_review"
    else:
        overall = "PASS"
        reason = "all_available_components_pass"

    return {
        "density_level": density,
        "geometry_level": geometry,
        "alchemy_level": overall,
        "evidence_basis": evidence_basis,
        "verdict_reason": reason,
    }


class _EmpiricalDistribution:
    """A compact average-rank survival distribution for one raw metric."""

    def __init__(self, values: Sequence[float], counts: Sequence[int]) -> None:
        if len(values) != len(counts):
            raise ValueError("confidence reference values and counts differ in size")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("confidence reference contains an invalid value")
        if any(isinstance(count, bool) or count < 1 for count in counts):
            raise ValueError("confidence reference contains an invalid count")
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("confidence reference values are not increasing")
        self.values = tuple(values)
        self.counts = tuple(counts)
        cumulative: list[int] = []
        running = 0
        for count in counts:
            cumulative.append(running)
            running += count
        self.cumulative_below = tuple(cumulative)
        self.size = running

    def support_score(self, value: float) -> float:
        """Return reverse average-rank ECDF support; ordinary values rank high."""
        if not math.isfinite(value) or value < 0 or not self.values:
            return math.nan
        index = bisect.bisect_left(self.values, value)
        if index < len(self.values) and self.values[index] == value:
            below = self.cumulative_below[index]
            equal = self.counts[index]
        else:
            below = (
                self.cumulative_below[index] if index < len(self.values) else self.size
            )
            equal = 0
        return 100.0 * (self.size - below - 0.5 * equal) / self.size


class ConfidenceReference:
    """Frozen empirical density and RMS-Zbond distributions."""

    def __init__(
        self,
        density_values: Sequence[float],
        density_counts: Sequence[int],
        geometry_values: Sequence[float],
        geometry_counts: Sequence[int],
        metadata: Mapping[str, Any],
    ) -> None:
        """Initialize the frozen empirical distributions and their metadata."""
        self.density = _EmpiricalDistribution(density_values, density_counts)
        self.geometry = _EmpiricalDistribution(geometry_values, geometry_counts)
        if self.density.size == 0 and self.geometry.size == 0:
            raise ValueError("confidence reference has no assessable evidence")
        self.metadata = dict(metadata)
        self.reference_id: str = self.metadata.get("reference_id", "")
        self.cohort_id: str = self.metadata.get("cohort_id", "")
        self.cohort_size = int(self.metadata.get("input_row_count", 0))

    @property
    def density_reference_size(self) -> int:
        """Return the number of density observations in the reference."""
        return self.density.size

    @property
    def geometry_reference_size(self) -> int:
        """Return the number of geometry observations in the reference."""
        return self.geometry.size


def score_site(
    rszd_abs: float,
    geometry_rms_zbond: float,
    reference: ConfidenceReference | None = None,
) -> dict[str, str | float]:
    """Return authoritative levels plus secondary empirical ranking scores."""
    result: dict[str, str | float] = dict(
        classify_site(rszd_abs, geometry_rms_zbond).items()
    )
    density_score = (
        0.0
        if reference and rszd_abs >= EDSTATS_SATURATION_MAGNITUDE
        else reference.density.support_score(rszd_abs)
        if reference
        else math.nan
    )
    geometry_score = (
        reference.geometry.support_score(geometry_rms_zbond) if reference else math.nan
    )
    available_scores = [
        score for score in (density_score, geometry_score) if math.isfinite(score)
    ]
    result.update(
        {
            "density_score": density_score,
            "geometry_score": geometry_score,
            "alchemy_score": min(available_scores) if available_scores else math.nan,
        }
    )
    return result


def _scoring_metadata() -> dict[str, Any]:
    """Everything a score depends on that is not the cohort itself.

    ``reference_data_id`` counts: every score in a distribution was measured
    against one catalog and one distance table, so a changed table must produce
    a different reference id rather than a quietly wrong empirical rank.
    """
    return {
        "confidence_method_version": CONFIDENCE_METHOD_VERSION,
        "confidence_schema_version": CONFIDENCE_SCHEMA_VERSION,
        "cohort_weighting": COHORT_WEIGHTING,
        "score_decimal_places": SCORE_DECIMAL_PLACES,
        "metric_decimal_places": METRIC_DECIMAL_PLACES,
        "density_thresholds": {
            "review": DENSITY_REVIEW_THRESHOLD,
            "suspect": DENSITY_SUSPECT_THRESHOLD,
        },
        "density_saturation_value": EDSTATS_SATURATION_MAGNITUDE,
        "density_saturation_policy": "suspect_with_zero_support",
        "geometry_thresholds": {
            "review": GEOMETRY_REVIEW_THRESHOLD,
            "suspect": GEOMETRY_SUSPECT_THRESHOLD,
        },
        "support_score_method": "reverse_average_rank_empirical_cdf",
        "geometry_statistic": "rms_finite_score_eligible_zbond",
        "overall_rule": "any_suspect_or_review_plus_review",
        "coverage_policy": "annotation_only_v1",
        "input_status_policy": INPUT_STATUS_POLICY,
        "maximum_entry_metal_sites": MAX_ANALYZED_METAL_SITES,
        "reference_data_id": reference_data_id(),
        "analysis_config_id": analysis_config_id(reference_data_id=reference_data_id()),
    }


def _reference_identifier(
    density_counts: Mapping[float, int], geometry_counts: Mapping[float, int]
) -> str:
    digest = hashlib.sha256()
    scoring_parameters = json.dumps(
        _scoring_metadata(), sort_keys=True, separators=(",", ":")
    )
    digest.update(scoring_parameters.encode("utf-8"))
    digest.update(b"\n")
    for component, counts in (
        ("density", density_counts),
        ("geometry", geometry_counts),
    ):
        for value in sorted(counts):
            digest.update(
                f"{component},{repr(value)},{counts[value]}\n".encode("ascii")
            )
    return "alchemy-confidence-" + digest.hexdigest()[:20]


def _normalized_metric_counts(counts: Mapping[float, int]) -> Counter[float]:
    normalized: Counter[float] = Counter()
    for value, count in counts.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError("confidence reference contains an invalid value")
        if isinstance(count, bool) or count < 1:
            raise ValueError("confidence reference contains an invalid count")
        normalized[canonical_metric(value)] += count
    return normalized


def write_reference(
    reference_dir: str,
    density_counts: Mapping[float, int],
    geometry_counts: Mapping[float, int],
    input_row_count: int,
    cohort_provenance: Mapping[str, Any] | None = None,
) -> "ConfidenceReference":
    """Write reusable component distributions and their policy metadata."""
    density_counts = _normalized_metric_counts(density_counts)
    geometry_counts = _normalized_metric_counts(geometry_counts)
    if not density_counts and not geometry_counts:
        raise ValueError("cannot build a confidence reference with no evidence")
    os.makedirs(reference_dir, exist_ok=True)
    distribution_path = os.path.join(reference_dir, REFERENCE_DISTRIBUTION_FILE)
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)

    distribution_tmp = distribution_path + ".tmp"
    with open(distribution_tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("component", "value", "count"))
        for component, counts in (
            ("density", density_counts),
            ("geometry", geometry_counts),
        ):
            for value in sorted(counts):
                writer.writerow(
                    (
                        component,
                        _format_decimal(value, METRIC_DECIMAL_PLACES),
                        counts[value],
                    )
                )
    os.replace(distribution_tmp, distribution_path)

    metadata = _scoring_metadata()
    metadata.update(
        {
            "reference_id": _reference_identifier(density_counts, geometry_counts),
            "input_row_count": input_row_count,
            "density_reference_size": sum(density_counts.values()),
            "geometry_reference_size": sum(geometry_counts.values()),
            "density_distinct_value_count": len(density_counts),
            "geometry_distinct_value_count": len(geometry_counts),
            "distribution_file": REFERENCE_DISTRIBUTION_FILE,
        }
    )
    provenance = dict(cohort_provenance or {})
    if "cohort_id" not in provenance:
        fallback_identity = hashlib.sha256(
            (
                f"{_reference_identifier(density_counts, geometry_counts)}\n"
                f"{input_row_count}\n"
            ).encode("ascii")
        ).hexdigest()
        provenance["cohort_id"] = "alchemy-cohort-" + fallback_identity[:20]
    metadata.update(provenance)
    metadata_tmp = metadata_path + ".tmp"
    with open(metadata_tmp, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(metadata_tmp, metadata_path)
    return ConfidenceReference(
        sorted(density_counts),
        [density_counts[value] for value in sorted(density_counts)],
        sorted(geometry_counts),
        [geometry_counts[value] for value in sorted(geometry_counts)],
        metadata,
    )


def load_reference(reference_dir: str) -> "ConfidenceReference":
    """Load and strictly validate a frozen database confidence reference."""
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = _scoring_metadata()
    for key in (
        "confidence_method_version",
        "confidence_schema_version",
        "cohort_weighting",
        "score_decimal_places",
        "metric_decimal_places",
        "density_thresholds",
        "density_saturation_value",
        "density_saturation_policy",
        "geometry_thresholds",
        "support_score_method",
        "geometry_statistic",
        "overall_rule",
        "coverage_policy",
        "input_status_policy",
        "maximum_entry_metal_sites",
        "analysis_config_id",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"confidence reference {key} is incompatible with this code"
            )
    if metadata.get("reference_data_id") != expected["reference_data_id"]:
        raise ValueError(
            "confidence reference was built against reference data "
            f"{metadata.get('reference_data_id') or 'nothing recorded'}, but "
            f"this run uses {expected['reference_data_id']}. Every score in it "
            "was measured against different reference distances; rebuild the "
            "reference with an uncapped database run."
        )
    cohort_id = metadata.get("cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id.startswith("alchemy-cohort-"):
        raise ValueError("confidence reference has no valid cohort identifier")
    inputs_sha256 = metadata.get("confidence_inputs_sha256")
    if inputs_sha256 is not None and (
        not isinstance(inputs_sha256, str)
        or len(inputs_sha256) != 64
        or cohort_id != "alchemy-cohort-" + inputs_sha256[:20]
    ):
        raise ValueError("confidence reference cohort identifier does not match input")
    distribution_path = os.path.join(
        reference_dir,
        metadata.get("distribution_file", REFERENCE_DISTRIBUTION_FILE),
    )
    header, rows = _read_csv(distribution_path, "confidence reference distribution")
    if header != ("component", "value", "count"):
        raise ValueError("confidence reference distribution has invalid columns")
    component_counts: dict[str, Counter[float]] = {
        "density": Counter(),
        "geometry": Counter(),
    }
    for row in rows:
        component = row["component"]
        value = _finite_float(row["value"])
        try:
            count = int(row["count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "confidence reference contains a non-integer count"
            ) from exc
        if component not in component_counts:
            raise ValueError("confidence reference contains an unknown component")
        if not math.isfinite(value) or value < 0:
            raise ValueError("confidence reference contains an invalid value")
        if value in component_counts[component]:
            raise ValueError("confidence reference contains a duplicate value")
        component_counts[component][value] = count
    density_counts = component_counts["density"]
    geometry_counts = component_counts["geometry"]
    if metadata.get("reference_id") != _reference_identifier(
        density_counts, geometry_counts
    ):
        raise ValueError("confidence reference identifier does not match data")
    reference = ConfidenceReference(
        sorted(density_counts),
        [density_counts[value] for value in sorted(density_counts)],
        sorted(geometry_counts),
        [geometry_counts[value] for value in sorted(geometry_counts)],
        metadata,
    )
    if reference.density_reference_size != metadata.get("density_reference_size"):
        raise ValueError("density reference size does not match metadata")
    if reference.geometry_reference_size != metadata.get("geometry_reference_size"):
        raise ValueError("geometry reference size does not match metadata")
    if len(reference.density.values) != metadata.get("density_distinct_value_count"):
        raise ValueError("density distinct-value count does not match metadata")
    if len(reference.geometry.values) != metadata.get("geometry_distinct_value_count"):
        raise ValueError("geometry distinct-value count does not match metadata")
    input_row_count = metadata.get("input_row_count")
    if (
        not isinstance(input_row_count, int)
        or input_row_count < reference.density_reference_size
        or input_row_count < reference.geometry_reference_size
    ):
        raise ValueError("confidence reference input row count is invalid")
    return reference


def _score_prepared_row(
    row: Mapping[str, Any], reference: "ConfidenceReference | None"
) -> tuple[dict[str, Any], float | None]:
    rszd = _finite_float(row.get("rszd_abs", ""))
    geometry_rms = _finite_float(row.get("geometry_rms_zbond", ""))
    result = score_site(rszd, geometry_rms, reference)
    output = dict(row)
    for key, value in result.items():
        output[key] = (
            _format_decimal(canonical_support_score(value))
            if isinstance(value, float)
            else value
        )
    output.update(
        {
            "score_policy_version": CONFIDENCE_METHOD_VERSION,
            "confidence_reference_version": reference.reference_id if reference else "",
            "confidence_cohort_id": reference.cohort_id if reference else "",
            "confidence_cohort_size": reference.cohort_size if reference else "",
            "density_reference_size": (
                reference.density_reference_size if reference else ""
            ),
            "geometry_reference_size": (
                reference.geometry_reference_size if reference else ""
            ),
        }
    )
    alchemy_score = _finite_float(output.get("alchemy_score", ""))
    return output, alchemy_score if math.isfinite(alchemy_score) else None


def score_against_reference(
    rows: Sequence[dict[str, Any]], reference: "ConfidenceReference"
) -> list[dict[str, Any]]:
    """Score prepared rows against a frozen database reference."""
    return [_score_prepared_row(row, reference)[0] for row in rows]


def classify_without_reference(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify prepared rows when no empirical ranking reference is installed."""
    return [_score_prepared_row(row, None)[0] for row in rows]


def _validated_input_reader(
    handle: TextIO,
) -> tuple[tuple[str, ...], "csv.DictReader[str]"]:
    reader = csv.DictReader(handle)
    input_columns = tuple(reader.fieldnames or ())
    _required_columns(
        input_columns,
        (
            "metal_site_id",
            "rszd_abs",
            "geometry_rms_zbond",
            "context_warning",
            "context_warning_reasons",
            "confidence_inputs_status",
        ),
        "confidence input CSV",
    )
    if any(column in input_columns for column in ANALYSIS_COLUMNS):
        raise ValueError("confidence input CSV already contains analysis columns")
    return input_columns, reader


def finalize_database_confidence(
    input_path: str,
    output_path: str,
    reference_dir: str,
    manifest_path: str | None = None,
) -> tuple[int, int, int]:
    """Build the database reference and assign final values from compact rows."""
    # Metadata is the completion marker. Remove it before rebuilding so a
    # failed finalization cannot leave an older reference looking current.
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)
    if os.path.isfile(metadata_path):
        os.unlink(metadata_path)
    density_counts: Counter[float] = Counter()
    geometry_counts: Counter[float] = Counter()
    input_row_count = 0
    input_entry_ids: set[str] = set()
    scorable_entry_ids: set[str] = set()
    input_status_counts: Counter[str] = Counter()
    with open(input_path, newline="", encoding="utf-8") as handle:
        _, rows = _validated_input_reader(handle)
        for row in rows:
            input_row_count += 1
            pdb_id = str(row.get("pdbID", "")).strip().lower()
            if pdb_id:
                input_entry_ids.add(pdb_id)
            input_status_counts[str(row.get("confidence_inputs_status", ""))] += 1
            rszd = _finite_float(row.get("rszd_abs", ""))
            geometry_rms = _finite_float(row.get("geometry_rms_zbond", ""))
            if math.isfinite(rszd) and rszd >= 0:
                density_counts[rszd] += 1
            if math.isfinite(geometry_rms) and geometry_rms >= 0:
                geometry_counts[geometry_rms] += 1
            if (math.isfinite(rszd) or math.isfinite(geometry_rms)) and pdb_id:
                scorable_entry_ids.add(pdb_id)
    inputs_sha256 = _file_sha256(input_path)
    provenance: dict[str, Any] = {
        "cohort_id": "alchemy-cohort-" + inputs_sha256[:20],
        "confidence_inputs_file": os.path.basename(input_path),
        "confidence_inputs_sha256": inputs_sha256,
        "input_entry_count": len(input_entry_ids),
        "scorable_entry_count": len(scorable_entry_ids),
        "input_status_counts": dict(sorted(input_status_counts.items())),
    }
    if manifest_path is not None:
        manifest_provenance = _manifest_provenance(manifest_path)
        if (
            manifest_provenance["analysis_config_id"]
            != _scoring_metadata()["analysis_config_id"]
        ):
            raise ValueError(
                "source manifest analysis configuration identity is "
                "incompatible with this code"
            )
        provenance.update(manifest_provenance)
    reference = write_reference(
        reference_dir,
        density_counts,
        geometry_counts,
        input_row_count,
        cohort_provenance=provenance,
    )
    total, scored = score_file_against_reference(input_path, output_path, reference)
    return total, scored, reference.cohort_size


def score_file_against_reference(
    input_path: str, output_path: str, reference: "ConfidenceReference"
) -> tuple[int, int]:
    """Score a compact input CSV against a loaded frozen reference."""
    output_tmp = output_path + ".tmp"
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    scored = 0
    try:
        with (
            open(input_path, newline="", encoding="utf-8") as source,
            open(output_tmp, "w", newline="", encoding="utf-8") as target,
        ):
            input_columns, rows = _validated_input_reader(source)
            writer = csv.DictWriter(
                target, fieldnames=(*input_columns, *ANALYSIS_COLUMNS)
            )
            writer.writeheader()
            for row in rows:
                output, score = _score_prepared_row(row, reference)
                writer.writerow(
                    {
                        column: _confidence_csv_value(column, value)
                        for column, value in output.items()
                    }
                )
                total += 1
                scored += score is not None
        os.replace(output_tmp, output_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(output_tmp)
        raise
    return total, scored


def validate_scored_reference(path: str, reference: "ConfidenceReference") -> None:
    """Refuse resume output containing rows from another frozen reference."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"confidence_reference_version", "confidence_cohort_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                "existing confidence output has no reference or cohort identifier"
            )
        identifiers: set[str] = set()
        cohort_identifiers: set[str] = set()
        for row in reader:
            identifier = (row.get("confidence_reference_version") or "").strip()
            cohort_id = (row.get("confidence_cohort_id") or "").strip()
            if not identifier or not cohort_id:
                raise ValueError(
                    "existing confidence output has a blank reference or cohort "
                    "identifier "
                    f"at CSV row {reader.line_num}"
                )
            identifiers.add(identifier)
            cohort_identifiers.add(cohort_id)
    if identifiers and identifiers != {reference.reference_id}:
        raise ValueError(
            "existing confidence output uses a different database reference"
        )
    if cohort_identifiers and cohort_identifiers != {reference.cohort_id}:
        raise ValueError("existing confidence output uses a different database cohort")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize or apply Alchemy confidence scores."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    finalize = commands.add_parser(
        "finalize", help="finalize a streamed complete-database cohort"
    )
    finalize.add_argument(
        "--input", required=True, help="streamed database confidence-input CSV"
    )
    finalize.add_argument(
        "--output", required=True, help="output database confidence-score CSV"
    )
    finalize.add_argument(
        "--reference-dir",
        required=True,
        help="output frozen database reference directory",
    )
    finalize.add_argument(
        "--manifest",
        help="optional completed run manifest to record as cohort provenance",
    )

    score = commands.add_parser(
        "score", help="score inputs against a frozen database reference"
    )
    score.add_argument("--input", required=True, help="prepared confidence-input CSV")
    score.add_argument(
        "--output", required=True, help="output confidence-score CSV path"
    )
    score.add_argument(
        "--reference-dir", required=True, help="frozen database reference directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the confidence-reference CLI and return an exit status."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "finalize":
            total, scored, cohort = finalize_database_confidence(
                args.input,
                args.output,
                args.reference_dir,
                manifest_path=args.manifest,
            )
            print(
                f"finalized {total} rows ({scored} scored; database cohort "
                f"{cohort}) to {args.output}"
            )
        else:
            reference = load_reference(args.reference_dir)
            total, scored = score_file_against_reference(
                args.input, args.output, reference
            )
            print(f"wrote {total} rows ({scored} scored) to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"confidence {args.command} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
