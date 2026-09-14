"""Column vocabularies, thresholds, and value helpers for confidence outputs."""

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from analysis_config import MAX_ANALYZED_METAL_SITES
from codes import ConfidenceInputStatus, EvidenceBasis, VerdictReason
from output_rows import scientific_csv_value

#: Policy-and-provenance file of a frozen reference directory. It is also the
#: completion marker: finalization removes it before rebuilding
#: (docs/output-schema.md, "confidence_reference/").
REFERENCE_METADATA_FILE = "metadata.json"
#: Per-component ``value``/``count`` table beside the metadata file.
REFERENCE_DISTRIBUTION_FILE = "component_distributions.csv"
#: Identity of the raw-threshold and verdict-matrix policy; written to every
#: row as ``score_policy_version`` and compared on reference load.
CONFIDENCE_METHOD_VERSION = "three_level_rms_2026_v1"
#: Layout version of the confidence input and analysis columns.
CONFIDENCE_SCHEMA_VERSION = 3
#: Each assessable metal site counts once in its component cohort; the ranks
#: are site-weighted, not structure-weighted (docs/output-schema.md).
COHORT_WEIGHTING = "per_metal_site"
#: Decimal places published for the 0-100 support scores.
SCORE_DECIMAL_PLACES = 6
#: Decimal places for raw metrics (|RSZD|, RMS Zbond) both in the compact rows
#: and when binning reference distribution values.
METRIC_DECIMAL_PLACES = 12
#: Absolute RSZD (a dimensionless density Z score) at or above which the
#: density component is REVIEW (docs/method.md, "Confidence scoring").
DENSITY_REVIEW_THRESHOLD = 3.0
#: Absolute RSZD at or above which the density component is SUSPECT.
DENSITY_SUSPECT_THRESHOLD = 6.0
#: Site RMS bond Z score at or above which the geometry component is REVIEW
#: (docs/method.md, "Confidence scoring").
GEOMETRY_REVIEW_THRESHOLD = 1.0
#: Site RMS bond Z score at or above which the geometry component is SUSPECT.
GEOMETRY_SUSPECT_THRESHOLD = 2.0
#: |RSZD| EDSTATS prints when its fixed-width field saturates; such a site is
#: SUSPECT with zero density support rather than ranked (docs/method.md).
EDSTATS_SATURATION_MAGNITUDE = 99.9
#: Columns that identify one site when ``metal_site_id`` is populated.
SITE_KEY_COLUMNS = (
    "pdbID",
    "metal_site_id",
)
#: Columns that identify one site in rows written before ``metal_site_id``.
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


CONFIDENCE_INPUT_STATUSES = frozenset(ConfidenceInputStatus)


EVIDENCE_BASES = frozenset(EvidenceBasis)


VERDICT_REASONS = frozenset(VerdictReason)


INPUT_STATUS_POLICY = "independent_component_availability_v1"


CONFIDENCE_BOOLEAN_COLUMNS = frozenset(
    {"density_is_shared", "density_saturated", "context_warning"}
)


#: Scoring-policy values recorded in ``metadata.json``. They are fixed by this
#: code, so a loaded reference must match every one of them; the reference
#: module adds the two bundled-data identity fields when it writes them.
SCORING_POLICY_METADATA: Mapping[str, Any] = MappingProxyType(
    {
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
    }
)

#: Identity of the bundled reference data and analysis configuration a
#: reference was built with; recorded beside the policy and compared on load.
REFERENCE_IDENTITY_FIELDS = ("reference_data_id", "analysis_config_id")

#: Every metadata key the scoring contract covers.
SCORING_METADATA_FIELDS = (*SCORING_POLICY_METADATA, *REFERENCE_IDENTITY_FIELDS)

#: Metadata keys that describe one particular cohort: its identity, sizes,
#: input and manifest hashes, and the software that produced it.
REFERENCE_PROVENANCE_FIELDS = (
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
)


REFERENCE_METADATA_FIELDS = frozenset(SCORING_METADATA_FIELDS) | frozenset(
    REFERENCE_PROVENANCE_FIELDS
)


def require_columns(
    fieldnames: Sequence[str] | None, required: Iterable[str], label: str
) -> None:
    """Raise ``ValueError`` naming every required column absent from a header."""
    missing = [column for column in required if column not in (fieldnames or ())]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def site_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the tuple that identifies a row's metal site across output tables."""
    site_id = str(row.get("metal_site_id", "")).strip()
    columns = SITE_KEY_COLUMNS if site_id else LEGACY_SITE_KEY_COLUMNS
    return tuple(str(row.get(column, "")).strip() for column in columns)


def finite_float(value: Any) -> float:
    """Parse a CSV number, returning NaN for blank, malformed, or non-finite text."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def parse_csv_bool(value: object) -> bool:
    """Return whether a serialized flag reads as true (``1``, ``true``, ``yes``)."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def format_decimal(value: float, decimal_places: int = 6) -> str:
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


def confidence_csv_value(column: str, value: object) -> object:
    """Serialize one confidence cell, canonicalizing boolean columns to lowercase."""
    if column in CONFIDENCE_BOOLEAN_COLUMNS and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized
    return scientific_csv_value(value)


def read_csv_table(
    path: str, label: str
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Read a CSV file into its header tuple and row dictionaries."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{label} has no CSV header")
        return tuple(reader.fieldnames), list(reader)
