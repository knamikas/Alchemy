"""Collect, finalize, and apply confidence scores for Alchemy metal sites.

A complete database run streams prepared rows to disk and, on success, freezes
them into a reference distribution. Single-entry and small-batch runs load that
reference and score against the database cohort, never against themselves.
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, TextIO
from collections.abc import Iterable, Mapping, Sequence

from output_rows import MetalStatsRow, scientific_csv_value

from reference_data import cofactor_ids, reference_data_id
from worker_contracts import MAX_ANALYZED_METAL_SITES


REFERENCE_METADATA_FILE = "metadata.json"
REFERENCE_DISTRIBUTION_FILE = "score_distribution.csv"
DENSITY_ANCHORS = ((2.0, 0.0), (3.0, 0.25), (6.0, 0.65), (12.0, 1.0))
GEOMETRY_ANCHORS = ((2.0, 0.0), (3.0, 0.35), (6.0, 0.70), (12.0, 1.0))
CONFIDENCE_METHOD_VERSION = "provisional_june_2026_v2"
CONFIDENCE_SCHEMA_VERSION = 2
COHORT_WEIGHTING = "per_metal_site"
SCORE_DECIMAL_PLACES = 6

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
    "rszd_magnitude",
    "rszd_negative",
    "rszd_positive",
    "assigned_contact_count",
    "reference_covered_contact_count",
    "zbond_available_contact_count",
    "geometry_coverage",
    "max_abs_zbond",
    "max_abs_zbond_contact_id",
    "max_abs_zbond_neighbor_resname",
    "max_abs_zbond_neighbor_chain",
    "max_abs_zbond_neighbor_resnum",
    "max_abs_zbond_neighbor_atom",
    "declared_contact_count",
    "inferred_contact_count",
    "multi_donor_contact_count",
    "suspect_multi_donor_residue_group_count",
    "context_warning",
    "context_warning_reasons",
    "confidence_inputs_status",
    "confidence_inputs_missing_reasons",
)

ANALYSIS_COLUMNS = (
    "density_severity",
    "geometry_severity",
    "density_penalty_fraction",
    "geometry_penalty_fraction",
    "interaction_penalty_fraction",
    "confidence_score",
    "confidence_percentile",
    "confidence_scoring_status",
    "confidence_scoring_reason",
    "confidence_reference_id",
    "confidence_cohort_id",
    "confidence_cohort_size",
)

SCORABLE_INPUT_STATUSES = frozenset({"complete", "partial_geometry", "density_only"})
CONFIDENCE_INPUT_STATUSES = SCORABLE_INPUT_STATUSES | {"unscorable"}
INPUT_STATUS_POLICY = "explicit_scorable_statuses_v2"
CONFIDENCE_BOOLEAN_COLUMNS = frozenset({"density_is_shared", "context_warning"})
REFERENCE_METADATA_FIELDS = frozenset(
    {
        "confidence_method_version",
        "confidence_schema_version",
        "cohort_weighting",
        "score_decimal_places",
        "density_anchors",
        "geometry_anchors",
        "weights",
        "percentile_method",
        "coverage_policy",
        "input_status_policy",
        "maximum_entry_metal_sites",
        "reference_data_id",
        "reference_id",
        "distribution_file",
        "distinct_score_count",
        "scorable_cohort_size",
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


def _confidence_inputs_are_scorable(row: Mapping[str, Any]) -> bool:
    """Return whether upstream evidence collection authorized scoring."""
    return (
        str(row.get("confidence_inputs_status", "")).strip() in SCORABLE_INPUT_STATUSES
    )


def _format_decimal(value: float, decimal_places: int = 6) -> str:
    """Render a confidence number without trailing zeros, blanking non-finite.

    Distinct from ``structure_analysis._format_number``, which renders a fixed
    number of decimals and blanks ``None`` rather than non-finite values: the
    two obey different contracts and must not be merged.
    """
    if not math.isfinite(value):
        return ""
    return f"{value:.{decimal_places}f}".rstrip("0").rstrip(".")


def canonical_confidence_score(value: float) -> float:
    return float(f"{value:.{SCORE_DECIMAL_PLACES}f}")


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
        rows = list(csv.DictReader(handle))

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
    }


def _bond_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finite_bonds: list[tuple[float, Mapping[str, Any]]] = []
    reference_covered = 0
    declared = 0
    inferred = 0
    multi_donor = 0
    for row in rows:
        reference_covered += _true(row.get("reference_covered", ""))
        declared += _true(row.get("declared_connection", ""))
        inferred += row.get("coordination_status", "").strip() == "inferred"
        multi_donor += _true(row.get("multi_donor_detected", ""))
        zscore = _finite_float(row.get("zscore", ""))
        if math.isfinite(zscore):
            finite_bonds.append((abs(zscore), row))

    assigned = len(rows)
    coverage = reference_covered / assigned if assigned else 0.0
    largest = max(finite_bonds, key=lambda item: item[0]) if finite_bonds else None
    largest_row: Mapping[str, Any] = largest[1] if largest else {}
    return {
        "assigned_contact_count": assigned,
        "reference_covered_contact_count": reference_covered,
        "zbond_available_contact_count": len(finite_bonds),
        "geometry_coverage": _format_decimal(coverage),
        "max_abs_zbond": _format_decimal(largest[0]) if largest else "",
        "max_abs_zbond_contact_id": largest_row.get("contact_id", ""),
        "max_abs_zbond_neighbor_resname": largest_row.get("neighbor_resname", ""),
        "max_abs_zbond_neighbor_chain": largest_row.get("neighbor_chain", ""),
        "max_abs_zbond_neighbor_resnum": largest_row.get("neighbor_resnum", ""),
        "max_abs_zbond_neighbor_atom": largest_row.get("neighbor_atom", ""),
        "declared_contact_count": declared,
        "inferred_contact_count": inferred,
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
    elif (
        summary["zbond_available_contact_count"]
        < summary["reference_covered_contact_count"]
    ):
        missing_reasons.append("zbond_unavailable_for_reference")
    if summary["reference_covered_contact_count"] < summary["assigned_contact_count"]:
        missing_reasons.append("partial_geometry_coverage")
    values: dict[str, Any] = {column: "" for column in CONFIDENCE_INPUT_COLUMNS}
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
            "confidence_inputs_status": "unscorable",
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

        rszd = abs(_finite_float(stats.get("ZDm", "")))
        negative = _finite_float(stats.get("ZD-m", ""))
        positive = _finite_float(stats.get("ZD+m", ""))
        summary = _bond_summary(bonds_by_site.pop(key, ()))
        missing_reasons: list[str] = []
        if str(stats.get("metal_coordinates_valid", "")).strip().lower() == "false":
            missing_reasons.append("non_finite_metal_coordinates")
        if not math.isfinite(rszd):
            missing_reasons.append("rszd_unavailable")
        if summary["assigned_contact_count"] == 0:
            missing_reasons.append("no_assigned_contacts")
        else:
            if summary["reference_covered_contact_count"] == 0:
                missing_reasons.append("no_geometry_reference")
            elif (
                summary["zbond_available_contact_count"]
                < summary["reference_covered_contact_count"]
            ):
                missing_reasons.append("zbond_unavailable_for_reference")
            if (
                summary["reference_covered_contact_count"]
                < summary["assigned_contact_count"]
            ):
                missing_reasons.append("partial_geometry_coverage")

        if (
            "non_finite_metal_coordinates" in missing_reasons
            or "rszd_unavailable" in missing_reasons
        ):
            status = "unscorable"
        elif (
            summary["reference_covered_contact_count"] > 0
            and summary["zbond_available_contact_count"] == 0
        ):
            status = "unscorable"
        elif summary["reference_covered_contact_count"] == 0:
            status = "density_only"
        elif (
            "partial_geometry_coverage" in missing_reasons
            or "zbond_unavailable_for_reference" in missing_reasons
        ):
            status = "partial_geometry"
        else:
            status = "complete"

        output = {column: stats.get(column, "") for column in IDENTITY_COLUMNS}
        output.update(
            {
                "rszd_magnitude": _format_decimal(rszd),
                "rszd_negative": _format_decimal(negative),
                "rszd_positive": _format_decimal(positive),
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
            row["confidence_inputs_status"] = "unscorable"
            row["confidence_inputs_missing_reasons"] = "|".join(reasons)
    for index in range(len(rows), selected_site_count):
        values: dict[str, Any] = {column: "" for column in CONFIDENCE_INPUT_COLUMNS}
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


def severity(value: float, anchors: Sequence[tuple[float, float]]) -> float:
    """Piecewise-linear bounded severity for a non-negative magnitude."""
    if not math.isfinite(value) or value < 0:
        return math.nan
    if value <= anchors[0][0]:
        return anchors[0][1]
    for (low_x, low_y), (high_x, high_y) in zip(anchors, anchors[1:]):
        if value <= high_x:
            fraction = (value - low_x) / (high_x - low_x)
            return low_y + fraction * (high_y - low_y)
    return anchors[-1][1]


def score_site(
    rszd_magnitude: float, max_abs_zbond: float, geometry_coverage: float
) -> dict[str, float] | None:
    """Return the provisional score and its auditable weighted components."""
    density_severity = severity(rszd_magnitude, DENSITY_ANCHORS)
    if not math.isfinite(density_severity):
        return None
    coverage = min(1.0, max(0.0, geometry_coverage))
    if coverage == 0.0:
        geometry_severity = 0.0
    else:
        geometry_severity = severity(max_abs_zbond, GEOMETRY_ANCHORS)
        if not math.isfinite(geometry_severity):
            return None

    density_component = 0.50 * density_severity
    geometry_component = 0.35 * coverage * geometry_severity
    interaction_component = (
        0.15 * coverage * math.sqrt(density_severity * geometry_severity)
    )
    confidence = 100.0 * (
        1.0 - density_component - geometry_component - interaction_component
    )
    return {
        "density_severity": density_severity,
        "geometry_severity": geometry_severity,
        "density_penalty_fraction": density_component,
        "geometry_penalty_fraction": geometry_component,
        "interaction_penalty_fraction": interaction_component,
        "confidence_score": min(100.0, max(0.0, confidence)),
    }


class ConfidenceReference:
    """Frozen empirical confidence-score distribution from a database run."""

    def __init__(
        self,
        values: Sequence[float],
        counts: Sequence[int],
        metadata: Mapping[str, Any],
    ) -> None:
        if len(values) != len(counts) or not values:
            raise ValueError("confidence reference distribution is empty")
        if any(count < 1 for count in counts):
            raise ValueError("confidence reference contains an invalid count")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("confidence reference scores are not increasing")
        self.values = tuple(values)
        self.counts = tuple(counts)
        cumulative: list[int] = []
        running = 0
        for count in counts:
            cumulative.append(running)
            running += count
        self.cumulative_below = tuple(cumulative)
        self.cohort_size = running
        self.metadata = dict(metadata)
        self.reference_id: str = self.metadata.get("reference_id", "")
        self.cohort_id: str = self.metadata.get("cohort_id", "")

    def percentile(self, score: float) -> float:
        """Average-rank ECDF percentile against the frozen database cohort."""
        score = canonical_confidence_score(score)
        index = bisect.bisect_left(self.values, score)
        if index < len(self.values) and self.values[index] == score:
            below = self.cumulative_below[index]
            equal = self.counts[index]
        else:
            below = (
                self.cumulative_below[index]
                if index < len(self.values)
                else self.cohort_size
            )
            equal = 0
        return 100.0 * (below + 0.5 * equal) / self.cohort_size


# Part of ``reference_id``, because nothing else in a stored reference records
# whether corrupt sites entered its cohort. Bump it to make ``load_reference``
# refuse references built under an earlier policy.
COVERAGE_POLICY = "invalid_coverage_excluded_v1"


def coverage_is_valid(coverage: float) -> bool:
    """Whether ``coverage`` is usable geometry evidence.

    A blank, non-numeric or out-of-range coverage is damaged input, not
    evidence that geometry is irrelevant: coercing it to zero drops both
    geometry terms and scores the site higher than real evidence would. Both
    scoring and reference building apply this test, so the cohort can never
    contain a site that scoring itself refuses.
    """
    return math.isfinite(coverage) and 0.0 <= coverage <= 1.0


def _scoring_metadata() -> dict[str, Any]:
    """Everything a score depends on that is not the cohort itself.

    ``reference_data_id`` counts: every score in a distribution was measured
    against one catalog and one distance table, so a changed table must produce
    a different reference id rather than a quietly wrong percentile.
    """
    return {
        "confidence_method_version": CONFIDENCE_METHOD_VERSION,
        "confidence_schema_version": CONFIDENCE_SCHEMA_VERSION,
        "cohort_weighting": COHORT_WEIGHTING,
        "score_decimal_places": SCORE_DECIMAL_PLACES,
        "density_anchors": [list(anchor) for anchor in DENSITY_ANCHORS],
        "geometry_anchors": [list(anchor) for anchor in GEOMETRY_ANCHORS],
        "weights": {
            "density": 0.50,
            "geometry": 0.35,
            "interaction": 0.15,
        },
        "percentile_method": "average_rank_empirical_cdf",
        "coverage_policy": COVERAGE_POLICY,
        "input_status_policy": INPUT_STATUS_POLICY,
        "maximum_entry_metal_sites": MAX_ANALYZED_METAL_SITES,
        "reference_data_id": reference_data_id(),
    }


def _reference_identifier(score_counts: Mapping[float, int]) -> str:
    digest = hashlib.sha256()
    scoring_parameters = json.dumps(
        _scoring_metadata(), sort_keys=True, separators=(",", ":")
    )
    digest.update(scoring_parameters.encode("utf-8"))
    digest.update(b"\n")
    for score in sorted(score_counts):
        digest.update(f"{repr(score)},{score_counts[score]}\n".encode("ascii"))
    return "alchemy-confidence-" + digest.hexdigest()[:20]


def _normalized_score_counts(score_counts: Mapping[float, int]) -> Counter[float]:
    normalized: Counter[float] = Counter()
    for score, count in score_counts.items():
        normalized[canonical_confidence_score(score)] += count
    return normalized


def write_reference(
    reference_dir: str,
    score_counts: Mapping[float, int],
    input_row_count: int,
    cohort_provenance: Mapping[str, Any] | None = None,
) -> "ConfidenceReference":
    """Write a reusable database confidence distribution and its metadata."""
    score_counts = _normalized_score_counts(score_counts)
    if not score_counts:
        raise ValueError("cannot build a confidence reference with no scores")
    os.makedirs(reference_dir, exist_ok=True)
    distribution_path = os.path.join(reference_dir, REFERENCE_DISTRIBUTION_FILE)
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)

    distribution_tmp = distribution_path + ".tmp"
    with open(distribution_tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("confidence_score", "count"))
        for score in sorted(score_counts):
            writer.writerow((_format_decimal(score), score_counts[score]))
    os.replace(distribution_tmp, distribution_path)

    metadata = _scoring_metadata()
    metadata.update(
        {
            "reference_id": _reference_identifier(score_counts),
            "input_row_count": input_row_count,
            "scorable_cohort_size": sum(score_counts.values()),
            "distinct_score_count": len(score_counts),
            "distribution_file": REFERENCE_DISTRIBUTION_FILE,
        }
    )
    provenance = dict(cohort_provenance or {})
    if "cohort_id" not in provenance:
        fallback_identity = hashlib.sha256(
            f"{_reference_identifier(score_counts)}\n{input_row_count}\n".encode(
                "ascii"
            )
        ).hexdigest()
        provenance["cohort_id"] = "alchemy-cohort-" + fallback_identity[:20]
    metadata.update(provenance)
    metadata_tmp = metadata_path + ".tmp"
    with open(metadata_tmp, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(metadata_tmp, metadata_path)
    return ConfidenceReference(
        sorted(score_counts),
        [score_counts[value] for value in sorted(score_counts)],
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
        "density_anchors",
        "geometry_anchors",
        "weights",
        "percentile_method",
        "coverage_policy",
        "input_status_policy",
        "maximum_entry_metal_sites",
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
    if header != ("confidence_score", "count"):
        raise ValueError("confidence reference distribution has invalid columns")
    values: list[float] = []
    counts: list[int] = []
    score_counts: Counter[float] = Counter()
    for row in rows:
        score = _finite_float(row["confidence_score"])
        try:
            count = int(row["count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "confidence reference contains a non-integer count"
            ) from exc
        if not math.isfinite(score):
            raise ValueError("confidence reference contains a non-finite score")
        values.append(score)
        counts.append(count)
        score_counts[score] = count
    if metadata.get("reference_id") != _reference_identifier(score_counts):
        raise ValueError("confidence reference identifier does not match data")
    reference = ConfidenceReference(values, counts, metadata)
    if reference.cohort_size != metadata.get("scorable_cohort_size"):
        raise ValueError("confidence reference cohort size does not match metadata")
    if len(reference.values) != metadata.get("distinct_score_count"):
        raise ValueError(
            "confidence reference distinct score count does not match data"
        )
    input_row_count = metadata.get("input_row_count")
    if not isinstance(input_row_count, int) or input_row_count < reference.cohort_size:
        raise ValueError("confidence reference input row count is invalid")
    return reference


def _score_prepared_row(
    row: Mapping[str, Any], reference: "ConfidenceReference"
) -> tuple[dict[str, Any], float | None]:
    rszd = _finite_float(row.get("rszd_magnitude", ""))
    zbond = _finite_float(row.get("max_abs_zbond", ""))
    coverage = _finite_float(row.get("geometry_coverage", ""))
    coverage_valid = coverage_is_valid(coverage)
    inputs_scorable = _confidence_inputs_are_scorable(row)
    result = (
        score_site(rszd, zbond, coverage)
        if inputs_scorable and coverage_valid
        else None
    )
    output = dict(row)
    if result is None:
        input_status = str(row.get("confidence_inputs_status", "")).strip()
        missing_reasons = {
            reason
            for reason in str(row.get("confidence_inputs_missing_reasons", "")).split(
                "|"
            )
            if reason
        }
        output.update({column: "" for column in ANALYSIS_COLUMNS})
        output.update(
            {
                "confidence_scoring_status": "unscorable",
                "confidence_scoring_reason": (
                    "zbond_unavailable_with_nonzero_coverage"
                    if "zbond_unavailable_for_reference" in missing_reasons
                    and coverage_valid
                    and coverage > 0.0
                    and not math.isfinite(zbond)
                    else "confidence_inputs_unscorable"
                    if input_status == "unscorable"
                    else "confidence_inputs_status_invalid"
                    if not inputs_scorable
                    else "geometry_coverage_invalid"
                    if not coverage_valid
                    else "rszd_unavailable"
                    if not math.isfinite(rszd)
                    else "zbond_unavailable_with_nonzero_coverage"
                ),
                "confidence_reference_id": reference.reference_id,
                "confidence_cohort_id": reference.cohort_id,
                "confidence_cohort_size": reference.cohort_size,
            }
        )
        return output, None

    output.update({key: _format_decimal(value) for key, value in result.items()})
    score = canonical_confidence_score(result["confidence_score"])
    output["confidence_score"] = _format_decimal(score)
    output.update(
        {
            "confidence_percentile": _format_decimal(reference.percentile(score)),
            "confidence_scoring_status": (
                "density_only"
                if coverage == 0.0
                else ("partial_geometry" if coverage < 1.0 else "complete")
            ),
            "confidence_scoring_reason": (
                "no_usable_geometry"
                if coverage == 0.0
                else ("geometry_coverage_below_one" if coverage < 1.0 else "")
            ),
            "confidence_reference_id": reference.reference_id,
            "confidence_cohort_id": reference.cohort_id,
            "confidence_cohort_size": reference.cohort_size,
        }
    )
    return output, score


def score_against_reference(
    rows: Sequence[dict[str, Any]], reference: "ConfidenceReference"
) -> list[dict[str, Any]]:
    """Score prepared rows against a frozen database reference."""
    return [_score_prepared_row(row, reference)[0] for row in rows]


def _validated_input_reader(
    handle: TextIO,
) -> tuple[tuple[str, ...], "csv.DictReader[str]"]:
    reader = csv.DictReader(handle)
    input_columns = tuple(reader.fieldnames or ())
    _required_columns(
        input_columns,
        (
            "metal_site_id",
            "rszd_magnitude",
            "max_abs_zbond",
            "geometry_coverage",
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
    score_counts: Counter[float] = Counter()
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
            if not _confidence_inputs_are_scorable(row):
                continue
            rszd = _finite_float(row.get("rszd_magnitude", ""))
            zbond = _finite_float(row.get("max_abs_zbond", ""))
            coverage = _finite_float(row.get("geometry_coverage", ""))
            result = (
                score_site(rszd, zbond, coverage)
                if coverage_is_valid(coverage)
                else None
            )
            if result is not None:
                score_counts[
                    canonical_confidence_score(result["confidence_score"])
                ] += 1
                if pdb_id:
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
        provenance.update(_manifest_provenance(manifest_path))
    reference = write_reference(
        reference_dir,
        score_counts,
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
        try:
            os.unlink(output_tmp)
        except OSError:
            pass
        raise
    return total, scored


def validate_scored_reference(path: str, reference: "ConfidenceReference") -> None:
    """Refuse resume output containing rows from another frozen reference."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"confidence_reference_id", "confidence_cohort_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                "existing confidence output has no reference or cohort identifier"
            )
        identifiers: set[str] = set()
        cohort_identifiers: set[str] = set()
        for row in reader:
            identifier = (row.get("confidence_reference_id") or "").strip()
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
