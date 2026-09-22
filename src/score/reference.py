"""Build, persist, load, and apply frozen score references.

Two responsibilities live here, either side of the frozen reference. The
reference lifecycle writes a cohort's component distributions and their policy
metadata, reloads them under strict verification, and refuses output built
against a different reference. Applying a reference adds one verdict per
prepared row, in memory for a single entry or streaming a whole input file.
"""

import contextlib
import csv
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, TextIO, cast

from analysis_config import analysis_config_id
from output_rows import finite_float
from reference_data import reference_data_id
from reference_integrity import sha256
from score.schema import (
    ANALYSIS_COLUMNS,
    REFERENCE_DECIMAL_PLACES,
    REFERENCE_DISTRIBUTION_FILE,
    REFERENCE_METADATA_FILE,
    SCORING_POLICY_METADATA,
    canonical_reference_metric,
    canonical_score,
    format_decimal,
    parse_csv_bool,
    require_columns,
    score_csv_value,
)
from score.scoring import (
    EmpiricalDistribution,
    ScoreReference,
    score_site,
)

DEFAULT_SCORE_REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "score_reference"
)

#: The header every component distribution file carries.
DISTRIBUTION_COLUMNS = ("component", "value", "count")


class FinalizedScore(NamedTuple):
    """What finalizing a database reference produced.

    ``rows`` and ``scored_rows`` describe the scoring pass: every input row
    written to the scores file, and how many of those earned a finite ranking
    score. ``cohort_size`` is the frozen reference's ``input_row_count``, which
    the counting pass measured. Both passes read one file, so ``cohort_size``
    equals ``rows`` by construction today; the two can only differ if the input
    file changes between the passes, and they are reported separately so that
    such a change is visible rather than assumed away.
    """

    rows: int
    scored_rows: int
    cohort_size: int


def _scoring_metadata() -> dict[str, Any]:
    """Return scoring-policy and reference-data identity, excluding the cohort.

    Include reference_data_id because changed reference tables change the metrics.
    """
    return {
        # The nested threshold mappings are read-only proxies in the schema;
        # json.dumps cannot serialize those, so copy them into plain dicts.
        **{
            key: dict(cast(Mapping[str, Any], value))
            if isinstance(value, Mapping)
            else value
            for key, value in SCORING_POLICY_METADATA.items()
        },
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
    return "alchemy-score-" + digest.hexdigest()[:20]


def _normalized_metric_counts(counts: Mapping[float, int]) -> Counter[float]:
    """Validate raw observations, then merge values at reference precision."""
    # Validate before rounding and summing: neither a small negative metric nor
    # an invalid count may become valid just because it merges with another row.
    EmpiricalDistribution.from_counts(counts)
    normalized: Counter[float] = Counter()
    for value, count in counts.items():
        normalized[canonical_reference_metric(value)] += count
    return normalized


def _validated_input_row_count(value: object, minimum: int = 0) -> int:
    """Return a cohort size large enough to hold the observations it describes.

    ``value`` is read from metadata or handed in by a caller, so it is validated
    rather than coerced: the cohort size must be a real count, and no component
    can hold more observations than the cohort it was drawn from. Leaving
    ``minimum`` at zero checks only that the value is a count, which is what a
    caller that has not yet built the distributions can check.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"score reference input row count is invalid: {value!r}")
    if value < minimum:
        raise ValueError(
            f"score reference input row count is invalid: {value} is fewer "
            f"than the {minimum} observation(s) it is said to cover"
        )
    return value


def write_reference(
    reference_dir: str,
    density_counts: Mapping[float, int],
    geometry_counts: Mapping[float, int],
    input_row_count: int,
    cohort_provenance: Mapping[str, Any] | None = None,
) -> ScoreReference:
    """Write reusable component distributions and their policy metadata."""
    density_counts = _normalized_metric_counts(density_counts)
    geometry_counts = _normalized_metric_counts(geometry_counts)
    density = EmpiricalDistribution.from_counts(density_counts)
    geometry = EmpiricalDistribution.from_counts(geometry_counts)
    if not density.size and not geometry.size:
        raise ValueError("cannot build a score reference with no evidence")
    # Validate everything the reference claims before a byte is written, so a
    # rejected build cannot leave a half-written reference directory behind.
    input_row_count = _validated_input_row_count(
        input_row_count, max(density.size, geometry.size)
    )
    os.makedirs(reference_dir, exist_ok=True)
    distribution_path = os.path.join(reference_dir, REFERENCE_DISTRIBUTION_FILE)
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)

    # The metadata file is this directory's completion marker. Remove a stale
    # one first so that a failure part-way through cannot leave metadata
    # describing distributions the directory no longer holds.
    with contextlib.suppress(FileNotFoundError):
        os.unlink(metadata_path)

    distribution_tmp = distribution_path + ".tmp"
    try:
        with open(distribution_tmp, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(DISTRIBUTION_COLUMNS)
            for component, distribution in (
                ("density", density),
                ("geometry", geometry),
            ):
                for value, count in zip(
                    distribution.values, distribution.counts, strict=True
                ):
                    writer.writerow(
                        (
                            component,
                            format_decimal(value, REFERENCE_DECIMAL_PLACES),
                            count,
                        )
                    )
        os.replace(distribution_tmp, distribution_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(distribution_tmp)
        raise

    metadata = _scoring_metadata()
    reference_id = _reference_identifier(density_counts, geometry_counts)
    metadata.update(
        {
            "reference_id": reference_id,
            "input_row_count": input_row_count,
            "density_reference_size": density.size,
            "geometry_reference_size": geometry.size,
            "density_distinct_value_count": density.distinct_value_count,
            "geometry_distinct_value_count": geometry.distinct_value_count,
            "distribution_file": REFERENCE_DISTRIBUTION_FILE,
        }
    )
    provenance = dict(cohort_provenance or {})
    if "cohort_id" not in provenance:
        fallback_identity = hashlib.sha256(
            f"{reference_id}\n{input_row_count}\n".encode("ascii")
        ).hexdigest()
        provenance["cohort_id"] = "alchemy-cohort-" + fallback_identity[:20]
    metadata.update(provenance)
    metadata_tmp = metadata_path + ".tmp"
    try:
        with open(metadata_tmp, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(metadata_tmp, metadata_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(metadata_tmp)
        raise
    return ScoreReference(
        density_values=density.values,
        density_counts=density.counts,
        geometry_values=geometry.values,
        geometry_counts=geometry.counts,
        metadata=metadata,
    )


def _read_csv_table(
    path: str, label: str
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Read a CSV file into its header tuple and row dictionaries."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{label} has no CSV header")
        return tuple(reader.fieldnames), list(reader)


def _distribution_path(reference_dir: str, metadata: Mapping[str, Any]) -> str:
    """Resolve the distribution file a reference's metadata names.

    The name is a sibling of the metadata file, never a path: a hand-edited
    ``distribution_file`` must not be able to send the loader elsewhere.
    """
    name = metadata.get("distribution_file", REFERENCE_DISTRIBUTION_FILE)
    if not isinstance(name, str) or not name or os.path.basename(name) != name:
        raise ValueError(f"score reference distribution file name is invalid: {name!r}")
    return os.path.join(reference_dir, name)


def _distribution_counts(path: str) -> dict[str, Counter[float]]:
    """Read one distribution file into its per-component count mappings.

    Only the checks the file format itself owns live here -- an unknown
    component, a repeated value, a count that is not an integer. Every rule
    about the values and counts themselves belongs to ``EmpiricalDistribution``,
    which the caller builds from what this returns.
    """
    header, rows = _read_csv_table(path, "score reference distribution")
    if header != DISTRIBUTION_COLUMNS:
        raise ValueError(
            "score reference distribution has invalid columns: "
            f"{path} has {', '.join(header)}"
        )
    counts: dict[str, Counter[float]] = {"density": Counter(), "geometry": Counter()}
    # Row 1 is the header, so the first data row is CSV row 2.
    for line_number, row in enumerate(rows, start=2):
        location = f"{path} row {line_number}"
        component = row["component"]
        if component not in counts:
            raise ValueError(
                "score reference contains an unknown component: "
                f"{component!r} at {location}"
            )
        value = finite_float(row["value"])
        try:
            count = int(row["count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "score reference contains a non-integer count: "
                f"{row['count']!r} for {component} {row['value']!r} at {location}"
            ) from exc
        if value in counts[component]:
            raise ValueError(
                "score reference contains a duplicate value: "
                f"{row['value']!r} for {component} at {location}"
            )
        counts[component][value] = count
    return counts


def load_reference(reference_dir: str) -> ScoreReference:
    """Load and strictly validate a frozen database score reference."""
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("score reference metadata is not a JSON object")
    metadata = cast(dict[str, Any], metadata)
    expected = _scoring_metadata()
    # reference_data_id is checked separately below with a fuller explanation.
    for key in (key for key in expected if key != "reference_data_id"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"score reference {key} is incompatible with this code")
    if metadata.get("reference_data_id") != expected["reference_data_id"]:
        raise ValueError(
            "score reference was built against reference data "
            f"{metadata.get('reference_data_id') or 'nothing recorded'}, but "
            f"this run uses {expected['reference_data_id']}. Every score in it "
            "was measured against different reference distances; rebuild the "
            "reference with an uncapped database run."
        )
    cohort_id = metadata.get("cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id.startswith("alchemy-cohort-"):
        raise ValueError("score reference has no valid cohort identifier")
    inputs_sha256 = metadata.get("score_inputs_sha256")
    if inputs_sha256 is not None and (
        not isinstance(inputs_sha256, str)
        or len(inputs_sha256) != 64
        or cohort_id != "alchemy-cohort-" + inputs_sha256[:20]
    ):
        raise ValueError("score reference cohort identifier does not match input")
    distribution_path = _distribution_path(reference_dir, metadata)
    component_counts = _distribution_counts(distribution_path)
    density_counts = component_counts["density"]
    geometry_counts = component_counts["geometry"]
    # Report a malformed cohort size before the reference is built, so that the
    # loader's message, which says which metadata file it came from, is the one
    # an operator sees rather than the constructor's.
    _validated_input_row_count(metadata.get("input_row_count"))
    reference = ScoreReference.from_counts(density_counts, geometry_counts, metadata)
    if metadata.get("reference_id") != _reference_identifier(
        density_counts, geometry_counts
    ):
        raise ValueError("score reference identifier does not match data")
    if reference.density_reference_size != metadata.get("density_reference_size"):
        raise ValueError("density reference size does not match metadata")
    if reference.geometry_reference_size != metadata.get("geometry_reference_size"):
        raise ValueError("geometry reference size does not match metadata")
    if reference.density_distinct_value_count != metadata.get(
        "density_distinct_value_count"
    ):
        raise ValueError("density distinct-value count does not match metadata")
    if reference.geometry_distinct_value_count != metadata.get(
        "geometry_distinct_value_count"
    ):
        raise ValueError("geometry distinct-value count does not match metadata")
    _validated_input_row_count(
        metadata.get("input_row_count"),
        max(reference.density_reference_size, reference.geometry_reference_size),
    )
    return reference


def _format_score(score: float) -> str:
    """Serialize a score at its published precision; NaN becomes blank."""
    return format_decimal(canonical_score(score))


def _score_prepared_row(
    row: Mapping[str, Any], reference: ScoreReference | None
) -> tuple[dict[str, Any], bool]:
    """Return one prepared row with its verdict, and whether it was ranked."""
    rszd = finite_float(row.get("rszd_abs", ""))
    geometry_rms = finite_float(row.get("geometry_rms_zbond", ""))
    verdict = score_site(rszd, geometry_rms, reference)
    output = dict(row)
    output.update(
        {
            **verdict.as_row(),
            "density_score": _format_score(verdict.density_score),
            "geometry_score": _format_score(verdict.geometry_score),
            "alchemy_score": _format_score(verdict.alchemy_score),
            "score_reference_id": reference.reference_id if reference else "",
            "score_cohort_id": reference.cohort_id if reference else "",
            "score_cohort_size": reference.cohort_size if reference else "",
            "density_reference_size": (
                reference.density_reference_size if reference else ""
            ),
            "geometry_reference_size": (
                reference.geometry_reference_size if reference else ""
            ),
        }
    )
    return output, math.isfinite(verdict.alchemy_score)


def _score_rows(
    rows: Sequence[Mapping[str, Any]], reference: ScoreReference | None
) -> list[dict[str, Any]]:
    """Add a verdict to each prepared row, ranked against ``reference`` if given.

    The rows are not validated against the input schema: this path serves one
    entry's freshly prepared rows, which the preparation step already built to
    that schema, while the file paths validate the header they read.
    """
    return [_score_prepared_row(row, reference)[0] for row in rows]


def score_against_reference(
    rows: Sequence[Mapping[str, Any]], reference: ScoreReference
) -> list[dict[str, Any]]:
    """Score prepared rows against a frozen database reference."""
    return _score_rows(rows, reference)


def classify_without_reference(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Classify prepared rows when no empirical ranking reference is installed."""
    return _score_rows(rows, None)


def _validated_input_reader(
    handle: TextIO,
) -> tuple[tuple[str, ...], "csv.DictReader[str]"]:
    reader = csv.DictReader(handle)
    input_columns = tuple(reader.fieldnames or ())
    require_columns(
        input_columns,
        (
            "pdbID",
            "metal_site_id",
            "rszd_abs",
            "geometry_rms_zbond",
            "context_warning",
            "context_warning_reasons",
            "score_inputs_status",
        ),
        "score input CSV",
    )
    if any(column in input_columns for column in ANALYSIS_COLUMNS):
        raise ValueError("score input CSV already contains analysis columns")
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


def _validated_manifest_summary(manifest_path: str) -> dict[str, Any]:
    """Summarize a manifest built by this code, refusing one that was not.

    The ``analysis_config_id`` is compared and then dropped: it is a
    code-owned metadata key that ``_scoring_metadata`` already writes, and
    provenance must describe the cohort rather than restate the policy.
    """
    summary = manifest_provenance(manifest_path)
    if summary["analysis_config_id"] != _scoring_metadata()["analysis_config_id"]:
        raise ValueError(
            "source manifest analysis configuration identity is "
            "incompatible with this code"
        )
    summary.pop("analysis_config_id")
    return summary


def finalize_database_score(
    input_path: str,
    output_path: str,
    reference_dir: str,
    manifest_path: str | None = None,
) -> FinalizedScore:
    """Build the database reference and assign final values from compact rows."""
    # Metadata is the completion marker. Remove it before rebuilding so a
    # failed finalization cannot leave an older reference looking current.
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)
    if os.path.isfile(metadata_path):
        os.unlink(metadata_path)
    # Check the manifest before the input file is streamed: an incompatible
    # source manifest rejects the whole finalization, so it should cost one
    # read of the manifest rather than a pass over every input row.
    manifest_summary = (
        None if manifest_path is None else _validated_manifest_summary(manifest_path)
    )
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
            input_status_counts[str(row.get("score_inputs_status", ""))] += 1
            rszd = finite_float(row.get("rszd_abs", ""))
            geometry_rms = finite_float(row.get("geometry_rms_zbond", ""))
            if math.isfinite(rszd) and rszd >= 0:
                # Saturated |RSZD| observations (the 99.9 EDSTATS ceiling) are
                # counted into the density distribution even though they are
                # never ranked against it: they score a fixed 0. This is the
                # published policy and is retained deliberately.
                density_counts[canonical_reference_metric(rszd)] += 1
            if math.isfinite(geometry_rms) and geometry_rms >= 0:
                geometry_counts[canonical_reference_metric(geometry_rms)] += 1
            if (math.isfinite(rszd) or math.isfinite(geometry_rms)) and pdb_id:
                scorable_entry_ids.add(pdb_id)
    inputs_sha256 = sha256(input_path)
    provenance: dict[str, Any] = {
        "cohort_id": "alchemy-cohort-" + inputs_sha256[:20],
        "score_inputs_file": os.path.basename(input_path),
        "score_inputs_sha256": inputs_sha256,
        "input_entry_count": len(input_entry_ids),
        "scorable_entry_count": len(scorable_entry_ids),
        "input_status_counts": dict(sorted(input_status_counts.items())),
    }
    if manifest_summary is not None:
        provenance.update(manifest_summary)
    reference = write_reference(
        reference_dir,
        density_counts,
        geometry_counts,
        input_row_count,
        cohort_provenance=provenance,
    )
    total, scored = score_file_against_reference(input_path, output_path, reference)
    return FinalizedScore(total, scored, reference.cohort_size)


def score_file_against_reference(
    input_path: str, output_path: str, reference: ScoreReference
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
                output, row_scored = _score_prepared_row(row, reference)
                writer.writerow(
                    {
                        column: score_csv_value(column, value)
                        for column, value in output.items()
                    }
                )
                total += 1
                scored += row_scored
        os.replace(output_tmp, output_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(output_tmp)
        raise
    return total, scored


def validate_scored_reference(path: str, reference: ScoreReference) -> None:
    """Refuse resume output containing rows from another frozen reference."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"score_reference_id", "score_cohort_id"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                "existing score output has no reference or cohort identifier"
            )
        identifiers: set[str] = set()
        cohort_identifiers: set[str] = set()
        for row in reader:
            identifier = (row.get("score_reference_id") or "").strip()
            cohort_id = (row.get("score_cohort_id") or "").strip()
            if not identifier or not cohort_id:
                raise ValueError(
                    "existing score output has a blank reference or cohort "
                    "identifier "
                    f"at CSV row {reader.line_num}"
                )
            identifiers.add(identifier)
            cohort_identifiers.add(cohort_id)
    if identifiers and identifiers != {reference.reference_id}:
        raise ValueError("existing score output uses a different database reference")
    if cohort_identifiers and cohort_identifiers != {reference.cohort_id}:
        raise ValueError("existing score output uses a different database cohort")
