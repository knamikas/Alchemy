"""Column vocabularies, thresholds, and value helpers for confidence outputs."""

import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from analysis_config import MAX_ANALYZED_METAL_SITES
from codes import ConfidenceInputStatus
from output_rows import scientific_csv_value

#: Policy-and-provenance file of a frozen reference directory. It is also the
#: completion marker: finalization removes it before rebuilding
#: (docs/output-schema.md, "confidence_reference/").
REFERENCE_METADATA_FILE = "metadata.json"
#: Per-component ``value``/``count`` table beside the metadata file.
REFERENCE_DISTRIBUTION_FILE = "component_distributions.csv"
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
    "confidence_reference_id",
    "confidence_cohort_id",
    "confidence_cohort_size",
    "density_reference_size",
    "geometry_reference_size",
)


CONFIDENCE_INPUT_STATUSES = frozenset(ConfidenceInputStatus)


INPUT_STATUS_POLICY = "independent_component_availability"


CONFIDENCE_BOOLEAN_COLUMNS = frozenset(
    {"density_is_shared", "density_saturated", "context_warning"}
)


#: Scoring-policy values recorded in ``metadata.json``. They are fixed by this
#: code, so a loaded reference must match every one of them; the reference
#: module adds the two bundled-data identity fields when it writes them.
SCORING_POLICY_METADATA: Mapping[str, Any] = MappingProxyType(
    {
        "cohort_weighting": COHORT_WEIGHTING,
        "score_decimal_places": SCORE_DECIMAL_PLACES,
        "metric_decimal_places": METRIC_DECIMAL_PLACES,
        # The "review"/"suspect" keys spell confidence levels by coincidence,
        # which is why tests/test_documentation.py allowlists this module in
        # ``_COINCIDENTAL_LITERALS``.
        "density_thresholds": MappingProxyType(
            {
                "review": DENSITY_REVIEW_THRESHOLD,
                "suspect": DENSITY_SUSPECT_THRESHOLD,
            }
        ),
        "density_saturation_value": EDSTATS_SATURATION_MAGNITUDE,
        "density_saturation_policy": "suspect_with_zero_support",
        "geometry_thresholds": MappingProxyType(
            {
                "review": GEOMETRY_REVIEW_THRESHOLD,
                "suspect": GEOMETRY_SUSPECT_THRESHOLD,
            }
        ),
        "support_score_method": "reverse_average_rank_empirical_cdf",
        "geometry_statistic": "rms_finite_score_eligible_zbond",
        "overall_rule": "any_suspect_or_review_plus_review",
        "coverage_policy": "annotation_only",
        "input_status_policy": INPUT_STATUS_POLICY,
        # Record the cohort site cap explicitly as well as in its analysis identity.
        "maximum_entry_metal_sites": MAX_ANALYZED_METAL_SITES,
    }
)

#: Identity of the bundled reference data and analysis configuration a
#: reference was built with; recorded beside the policy and compared on load.
REFERENCE_IDENTITY_FIELDS = ("reference_data_id", "analysis_config_id")

#: Every metadata key the scoring contract covers.
SCORING_METADATA_FIELDS = (
    *SCORING_POLICY_METADATA.keys(),
    *REFERENCE_IDENTITY_FIELDS,
)

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
    """Return the tuple that identifies a row's metal site across output tables.

    The scheme is chosen per row: a populated ``metal_site_id`` selects
    ``SITE_KEY_COLUMNS``, and anything else falls back to the index columns
    rows carried before that column existed. Callers that join two tables on
    this key therefore rely on every row of both tables using the same scheme
    (all rows carry ``metal_site_id`` or none do); a mixture produces keys of
    two different shapes that can never match. A row with none of the identity
    columns yields an all-blank key rather than an error.
    """
    site_id = str(row.get("metal_site_id", "")).strip()
    columns = SITE_KEY_COLUMNS if site_id else LEGACY_SITE_KEY_COLUMNS
    return tuple(str(row.get(column, "")).strip() for column in columns)


def parse_csv_bool(value: object) -> bool:
    """Return whether a serialized flag reads as true (``1``, ``true``, ``yes``).

    Every other text reads as false, numbers such as ``1.0`` and outright junk
    included. That is deliberate: this reads back flags this code wrote, and is
    not validation. A caller that must reject malformed input checks it first.
    """
    return str(value).strip().lower() in {"1", "true", "yes"}


def format_decimal(value: float, decimal_places: int = SCORE_DECIMAL_PLACES) -> str:
    """Format a confidence number without trailing zeros; blank non-finite values.

    Negative zero keeps its sign (``-0.0`` formats as ``"-0"``); that is
    retained deliberately, because the published rows were written with it.
    """
    if not math.isfinite(value):
        return ""
    text = f"{value:.{decimal_places}f}"
    # Only a fractional part has zeros to strip: stripping "100" formatted with
    # no decimal places would leave "1".
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def is_density_saturated(rszd_abs: float) -> bool:
    """Return whether an absolute RSZD is at the EDSTATS saturation ceiling.

    EDSTATS prints ``EDSTATS_SATURATION_MAGNITUDE`` when its fixed-width field
    saturates, so no larger magnitude is produced in practice; the check is
    non-strict so any value at or beyond the ceiling is treated the same way.
    Both the ``density_saturated`` input column and the zero-support scoring
    rule use this one predicate (docs/method.md, "Confidence scoring").
    """
    return math.isfinite(rszd_abs) and rszd_abs >= EDSTATS_SATURATION_MAGNITUDE


def canonical_support_score(value: float) -> float:
    """Round a support score to its canonical serialized precision."""
    return float(f"{value:.{SCORE_DECIMAL_PLACES}f}")


def canonical_metric(value: float) -> float:
    """Round an input metric to its canonical serialized precision."""
    return float(f"{value:.{METRIC_DECIMAL_PLACES}f}")


def confidence_csv_value(column: str, value: object) -> object:
    """Serialize one confidence cell for a confidence CSV row.

    A ``true``/``false`` spelling in one of the boolean columns is lowercased,
    so those flags read the same however they were captured. Everything else,
    including real booleans and non-boolean columns, is left to
    ``scientific_csv_value``.
    """
    if column in CONFIDENCE_BOOLEAN_COLUMNS and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized
    return scientific_csv_value(value)
