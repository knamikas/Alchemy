"""Classify metal sites and rank their density and geometry support.

Raw thresholds determine verdicts. Frozen database distributions provide
independent empirical rankings for later runs.

Unlike the stage packages, which export nothing and are imported by submodule,
this package re-exports its public surface here because the driver, the tests,
and the documentation checks consume it as one API. ``__all__`` is therefore a
published contract rather than a convenience. Submodule imports remain
acceptable for helpers outside that contract, such as
``confidence_score.schema.parse_csv_bool``, and for the command line, which
lives in ``confidence_score.cli`` so that importing a constant does not pull in
``argparse``.
"""

from confidence_score.prepare import (
    complete_confidence_site_count as complete_confidence_site_count,
    prepare_confidence_inputs as prepare_confidence_inputs,
    prepare_result_confidence_inputs as prepare_result_confidence_inputs,
)
from confidence_score.reference import (
    classify_without_reference as classify_without_reference,
    finalize_database_confidence as finalize_database_confidence,
    load_reference as load_reference,
    score_against_reference as score_against_reference,
    score_file_against_reference as score_file_against_reference,
    validate_scored_reference as validate_scored_reference,
    write_reference as write_reference,
)
from confidence_score.schema import (
    ANALYSIS_COLUMNS as ANALYSIS_COLUMNS,
    COHORT_WEIGHTING as COHORT_WEIGHTING,
    CONFIDENCE_BOOLEAN_COLUMNS as CONFIDENCE_BOOLEAN_COLUMNS,
    CONFIDENCE_INPUT_COLUMNS as CONFIDENCE_INPUT_COLUMNS,
    CONFIDENCE_INPUT_STATUSES as CONFIDENCE_INPUT_STATUSES,
    DENSITY_REVIEW_THRESHOLD as DENSITY_REVIEW_THRESHOLD,
    DENSITY_SUSPECT_THRESHOLD as DENSITY_SUSPECT_THRESHOLD,
    EDSTATS_SATURATION_MAGNITUDE as EDSTATS_SATURATION_MAGNITUDE,
    GEOMETRY_REVIEW_THRESHOLD as GEOMETRY_REVIEW_THRESHOLD,
    GEOMETRY_SUSPECT_THRESHOLD as GEOMETRY_SUSPECT_THRESHOLD,
    IDENTITY_COLUMNS as IDENTITY_COLUMNS,
    INPUT_STATUS_POLICY as INPUT_STATUS_POLICY,
    LEGACY_SITE_KEY_COLUMNS as LEGACY_SITE_KEY_COLUMNS,
    METRIC_DECIMAL_PLACES as METRIC_DECIMAL_PLACES,
    REFERENCE_DECIMAL_PLACES as REFERENCE_DECIMAL_PLACES,
    REFERENCE_DISTRIBUTION_FILE as REFERENCE_DISTRIBUTION_FILE,
    REFERENCE_METADATA_FIELDS as REFERENCE_METADATA_FIELDS,
    REFERENCE_METADATA_FILE as REFERENCE_METADATA_FILE,
    REFERENCE_PROVENANCE_FIELDS as REFERENCE_PROVENANCE_FIELDS,
    SCORE_DECIMAL_PLACES as SCORE_DECIMAL_PLACES,
    SCORING_METADATA_FIELDS as SCORING_METADATA_FIELDS,
    SITE_KEY_COLUMNS as SITE_KEY_COLUMNS,
    canonical_metric as canonical_metric,
    canonical_support_score as canonical_support_score,
)
from confidence_score.scoring import (
    ConfidenceReference as ConfidenceReference,
    SiteVerdict as SiteVerdict,
    classify_site as classify_site,
    density_level as density_level,
    geometry_level as geometry_level,
    score_site as score_site,
)

__all__ = [
    "ANALYSIS_COLUMNS",
    "COHORT_WEIGHTING",
    "CONFIDENCE_BOOLEAN_COLUMNS",
    "CONFIDENCE_INPUT_COLUMNS",
    "CONFIDENCE_INPUT_STATUSES",
    "ConfidenceReference",
    "DENSITY_REVIEW_THRESHOLD",
    "DENSITY_SUSPECT_THRESHOLD",
    "EDSTATS_SATURATION_MAGNITUDE",
    "GEOMETRY_REVIEW_THRESHOLD",
    "GEOMETRY_SUSPECT_THRESHOLD",
    "IDENTITY_COLUMNS",
    "INPUT_STATUS_POLICY",
    "LEGACY_SITE_KEY_COLUMNS",
    "METRIC_DECIMAL_PLACES",
    "REFERENCE_DECIMAL_PLACES",
    "REFERENCE_DISTRIBUTION_FILE",
    "REFERENCE_METADATA_FIELDS",
    "REFERENCE_METADATA_FILE",
    "REFERENCE_PROVENANCE_FIELDS",
    "SCORE_DECIMAL_PLACES",
    "SCORING_METADATA_FIELDS",
    "SITE_KEY_COLUMNS",
    "SiteVerdict",
    "canonical_metric",
    "canonical_support_score",
    "classify_site",
    "classify_without_reference",
    "complete_confidence_site_count",
    "density_level",
    "finalize_database_confidence",
    "geometry_level",
    "load_reference",
    "prepare_confidence_inputs",
    "prepare_result_confidence_inputs",
    "score_against_reference",
    "score_file_against_reference",
    "score_site",
    "validate_scored_reference",
    "write_reference",
]
