"""Column vocabularies, thresholds, and value helpers for confidence outputs."""

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from output_rows import scientific_csv_value

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
    """Format a confidence number without trailing zeros; blank non-finite values."""
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
