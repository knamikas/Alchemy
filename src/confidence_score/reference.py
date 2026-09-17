"""Build, persist, load, and validate frozen confidence references."""

import contextlib
import csv
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

from analysis_config import analysis_config_id
from confidence_score.schema import (
    ANALYSIS_COLUMNS,
    CONFIDENCE_METHOD_VERSION,
    METRIC_DECIMAL_PLACES,
    REFERENCE_DISTRIBUTION_FILE,
    REFERENCE_METADATA_FILE,
    SCORING_POLICY_METADATA,
    canonical_metric,
    canonical_support_score,
    confidence_csv_value,
    format_decimal,
    parse_csv_bool,
    require_columns,
)
from confidence_score.scoring import (
    ConfidenceReference,
    score_site,
)
from output_rows import finite_float
from reference_data import reference_data_id, sha256


def _scoring_metadata() -> dict[str, Any]:
    """Return scoring-policy and reference-data identity, excluding the cohort.

    Include reference_data_id because changed reference tables change the metrics.
    """
    return {
        **SCORING_POLICY_METADATA,
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
) -> ConfidenceReference:
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
                        format_decimal(value, METRIC_DECIMAL_PLACES),
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
    return ConfidenceReference.from_counts(density_counts, geometry_counts, metadata)


def _read_csv_table(
    path: str, label: str
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Read a CSV file into its header tuple and row dictionaries."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{label} has no CSV header")
        return tuple(reader.fieldnames), list(reader)


def load_reference(reference_dir: str) -> ConfidenceReference:
    """Load and strictly validate a frozen database confidence reference."""
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = _scoring_metadata()
    # reference_data_id is checked separately below with a fuller explanation.
    for key in (key for key in expected if key != "reference_data_id"):
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
    header, rows = _read_csv_table(
        distribution_path, "confidence reference distribution"
    )
    if header != ("component", "value", "count"):
        raise ValueError("confidence reference distribution has invalid columns")
    component_counts: dict[str, Counter[float]] = {
        "density": Counter(),
        "geometry": Counter(),
    }
    for row in rows:
        component = row["component"]
        value = finite_float(row["value"])
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
    reference = ConfidenceReference.from_counts(
        density_counts, geometry_counts, metadata
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


def _format_support_score(score: float) -> str:
    """Serialize a support score at its published precision; NaN becomes blank."""
    return format_decimal(canonical_support_score(score))


def _score_prepared_row(
    row: Mapping[str, Any], reference: "ConfidenceReference | None"
) -> tuple[dict[str, Any], float | None]:
    rszd = finite_float(row.get("rszd_abs", ""))
    geometry_rms = finite_float(row.get("geometry_rms_zbond", ""))
    verdict = score_site(rszd, geometry_rms, reference)
    output = dict(row)
    output.update(
        {
            **verdict.as_row(),
            "density_score": _format_support_score(verdict.density_score),
            "geometry_score": _format_support_score(verdict.geometry_score),
            "alchemy_score": _format_support_score(verdict.alchemy_score),
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
    alchemy_score = verdict.alchemy_score
    return output, (
        canonical_support_score(alchemy_score) if math.isfinite(alchemy_score) else None
    )


def score_against_reference(
    rows: Sequence[dict[str, Any]], reference: ConfidenceReference
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
    require_columns(
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


def manifest_provenance(path: str) -> dict[str, Any]:
    """Summarize a completed run manifest as reference cohort provenance."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, ("analysis_config_id",), "source manifest")
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
        "source_manifest_sha256": sha256(path),
        "source_entry_count": len(rows),
        "manifest_status_counts": dict(sorted(status_counts.items())),
        "no_metals_entry_count": sum(
            parse_csv_bool(row.get("no_metals", "")) for row in rows
        ),
        "metal_site_limit_exceeded_entry_count": sum(
            parse_csv_bool(row.get("metal_site_limit_exceeded", "")) for row in rows
        ),
        "metal_bearing_entry_count": sum(
            (finite_float(row.get("n_metals", "")) > 0) for row in rows
        ),
        "software_versions": software,
        "analysis_config_id": next(iter(analysis_config_ids)),
    }


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
            rszd = finite_float(row.get("rszd_abs", ""))
            geometry_rms = finite_float(row.get("geometry_rms_zbond", ""))
            if math.isfinite(rszd) and rszd >= 0:
                density_counts[rszd] += 1
            if math.isfinite(geometry_rms) and geometry_rms >= 0:
                geometry_counts[geometry_rms] += 1
            if (math.isfinite(rszd) or math.isfinite(geometry_rms)) and pdb_id:
                scorable_entry_ids.add(pdb_id)
    inputs_sha256 = sha256(input_path)
    provenance: dict[str, Any] = {
        "cohort_id": "alchemy-cohort-" + inputs_sha256[:20],
        "confidence_inputs_file": os.path.basename(input_path),
        "confidence_inputs_sha256": inputs_sha256,
        "input_entry_count": len(input_entry_ids),
        "scorable_entry_count": len(scorable_entry_ids),
        "input_status_counts": dict(sorted(input_status_counts.items())),
    }
    if manifest_path is not None:
        manifest_summary = manifest_provenance(manifest_path)
        if (
            manifest_summary["analysis_config_id"]
            != _scoring_metadata()["analysis_config_id"]
        ):
            raise ValueError(
                "source manifest analysis configuration identity is "
                "incompatible with this code"
            )
        provenance.update(manifest_summary)
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
    input_path: str, output_path: str, reference: ConfidenceReference
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
                        column: confidence_csv_value(column, value)
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


def validate_scored_reference(path: str, reference: ConfidenceReference) -> None:
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
