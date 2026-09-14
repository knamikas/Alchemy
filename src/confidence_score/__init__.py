"""Classify metal sites and rank their density and geometry support.

Raw thresholds determine verdicts. Frozen database distributions provide
independent empirical rankings for later runs.
"""

from confidence_score.cli import main as main
from confidence_score.prepare import (
    complete_confidence_site_count as complete_confidence_site_count,
)
from confidence_score.prepare import (
    prepare_confidence_inputs as prepare_confidence_inputs,
)
from confidence_score.prepare import (
    prepare_result_confidence_inputs as prepare_result_confidence_inputs,
)
from confidence_score.reference import (
    classify_without_reference as classify_without_reference,
)
from confidence_score.reference import (
    finalize_database_confidence as finalize_database_confidence,
)
from confidence_score.reference import load_reference as load_reference
from confidence_score.reference import (
    score_against_reference as score_against_reference,
)
from confidence_score.reference import (
    score_file_against_reference as score_file_against_reference,
)
from confidence_score.reference import (
    validate_scored_reference as validate_scored_reference,
)
from confidence_score.reference import write_reference as write_reference
from confidence_score.schema import ANALYSIS_COLUMNS as ANALYSIS_COLUMNS
from confidence_score.schema import COHORT_WEIGHTING as COHORT_WEIGHTING
from confidence_score.schema import (
    CONFIDENCE_BOOLEAN_COLUMNS as CONFIDENCE_BOOLEAN_COLUMNS,
)
from confidence_score.schema import CONFIDENCE_INPUT_COLUMNS as CONFIDENCE_INPUT_COLUMNS
from confidence_score.schema import (
    CONFIDENCE_INPUT_STATUSES as CONFIDENCE_INPUT_STATUSES,
)
from confidence_score.schema import (
    CONFIDENCE_METHOD_VERSION as CONFIDENCE_METHOD_VERSION,
)
from confidence_score.schema import (
    CONFIDENCE_SCHEMA_VERSION as CONFIDENCE_SCHEMA_VERSION,
)
from confidence_score.schema import DENSITY_REVIEW_THRESHOLD as DENSITY_REVIEW_THRESHOLD
from confidence_score.schema import (
    DENSITY_SUSPECT_THRESHOLD as DENSITY_SUSPECT_THRESHOLD,
)
from confidence_score.schema import (
    EDSTATS_SATURATION_MAGNITUDE as EDSTATS_SATURATION_MAGNITUDE,
)
from confidence_score.schema import EVIDENCE_BASES as EVIDENCE_BASES
from confidence_score.schema import (
    GEOMETRY_REVIEW_THRESHOLD as GEOMETRY_REVIEW_THRESHOLD,
)
from confidence_score.schema import (
    GEOMETRY_SUSPECT_THRESHOLD as GEOMETRY_SUSPECT_THRESHOLD,
)
from confidence_score.schema import IDENTITY_COLUMNS as IDENTITY_COLUMNS
from confidence_score.schema import INPUT_STATUS_POLICY as INPUT_STATUS_POLICY
from confidence_score.schema import LEGACY_SITE_KEY_COLUMNS as LEGACY_SITE_KEY_COLUMNS
from confidence_score.schema import METRIC_DECIMAL_PLACES as METRIC_DECIMAL_PLACES
from confidence_score.schema import (
    REFERENCE_DISTRIBUTION_FILE as REFERENCE_DISTRIBUTION_FILE,
)
from confidence_score.schema import (
    REFERENCE_METADATA_FIELDS as REFERENCE_METADATA_FIELDS,
)
from confidence_score.schema import REFERENCE_METADATA_FILE as REFERENCE_METADATA_FILE
from confidence_score.schema import SCORE_DECIMAL_PLACES as SCORE_DECIMAL_PLACES
from confidence_score.schema import SITE_KEY_COLUMNS as SITE_KEY_COLUMNS
from confidence_score.schema import VERDICT_REASONS as VERDICT_REASONS
from confidence_score.schema import canonical_metric as canonical_metric
from confidence_score.schema import canonical_support_score as canonical_support_score
from confidence_score.scoring import ConfidenceReference as ConfidenceReference
from confidence_score.scoring import classify_site as classify_site
from confidence_score.scoring import component_level as component_level
from confidence_score.scoring import density_level as density_level
from confidence_score.scoring import geometry_level as geometry_level
from confidence_score.scoring import score_site as score_site

__all__ = [
    "ANALYSIS_COLUMNS",
    "COHORT_WEIGHTING",
    "CONFIDENCE_BOOLEAN_COLUMNS",
    "CONFIDENCE_INPUT_COLUMNS",
    "CONFIDENCE_INPUT_STATUSES",
    "CONFIDENCE_METHOD_VERSION",
    "CONFIDENCE_SCHEMA_VERSION",
    "ConfidenceReference",
    "DENSITY_REVIEW_THRESHOLD",
    "DENSITY_SUSPECT_THRESHOLD",
    "EDSTATS_SATURATION_MAGNITUDE",
    "EVIDENCE_BASES",
    "GEOMETRY_REVIEW_THRESHOLD",
    "GEOMETRY_SUSPECT_THRESHOLD",
    "IDENTITY_COLUMNS",
    "INPUT_STATUS_POLICY",
    "LEGACY_SITE_KEY_COLUMNS",
    "METRIC_DECIMAL_PLACES",
    "REFERENCE_DISTRIBUTION_FILE",
    "REFERENCE_METADATA_FIELDS",
    "REFERENCE_METADATA_FILE",
    "SCORE_DECIMAL_PLACES",
    "SITE_KEY_COLUMNS",
    "VERDICT_REASONS",
    "canonical_metric",
    "canonical_support_score",
    "classify_site",
    "classify_without_reference",
    "complete_confidence_site_count",
    "component_level",
    "density_level",
    "finalize_database_confidence",
    "geometry_level",
    "load_reference",
    "main",
    "prepare_confidence_inputs",
    "prepare_result_confidence_inputs",
    "score_against_reference",
    "score_file_against_reference",
    "score_site",
    "validate_scored_reference",
    "write_reference",
]
