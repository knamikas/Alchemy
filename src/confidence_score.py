"""Collect, finalize, and apply confidence scores for Alchemy metal sites.

During a complete database run, ``main.py`` calls the row-level preparation
function while each entry's density and bond records are already in memory. The
compact inputs are streamed to disk. At successful completion this module builds
a frozen database reference distribution and assigns cohort-relative values.
Later single-entry and small-batch runs load that reference once and score their
prepared rows directly against the database rather than against themselves.
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
from typing import Any, Dict


REFERENCE_METADATA_FILE = "metadata.json"
REFERENCE_DISTRIBUTION_FILE = "score_distribution.csv"
DENSITY_ANCHORS = ((2.0, 0.0), (3.0, 0.25), (6.0, 0.65), (12.0, 1.0))
GEOMETRY_ANCHORS = ((2.0, 0.0), (3.0, 0.35), (6.0, 0.70), (12.0, 1.0))

SITE_KEY_COLUMNS = (
    "pdbID",
    "metal_model_index",
    "metal_chain_index",
    "metal_residue_index",
    "metal_atom_index",
)

IDENTITY_COLUMNS = (
    "pdbID",
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
    "density_score_component",
    "geometry_score_component",
    "interaction_score_component",
    "confidence_score",
    "confidence_percentile",
    "confidence_scoring_status",
    "confidence_scoring_reason",
    "confidence_reference_id",
    "confidence_cohort_size",
    "confidence_context_warning",
    "confidence_context_warning_reasons",
)


def _required_columns(fieldnames, required, label):
    missing = [column for column in required if column not in (fieldnames or ())]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {', '.join(missing)}")


def _site_key(row):
    return tuple(str(row[column]).strip() for column in SITE_KEY_COLUMNS)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _true(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _format_number(value, digits=6):
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _read_csv(path, label):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{label} has no CSV header")
        return tuple(reader.fieldnames), list(reader)


def _write_csv(path, fieldnames, rows):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bond_summary(rows):
    finite_bonds = []
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
    largest_row = largest[1] if largest else {}
    return {
        "assigned_contact_count": assigned,
        "reference_covered_contact_count": reference_covered,
        "zbond_available_contact_count": len(finite_bonds),
        "geometry_coverage": _format_number(coverage),
        "max_abs_zbond": _format_number(largest[0]) if largest else "",
        "max_abs_zbond_neighbor_resname": largest_row.get(
            "neighbor_resname", ""),
        "max_abs_zbond_neighbor_chain": largest_row.get("neighbor_chain", ""),
        "max_abs_zbond_neighbor_resnum": largest_row.get("neighbor_resnum", ""),
        "max_abs_zbond_neighbor_atom": largest_row.get("neighbor_atom", ""),
        "declared_contact_count": declared,
        "inferred_contact_count": inferred,
        "multi_donor_contact_count": multi_donor,
    }


def _orphan_bond_site_input(key, rows):
    """Preserve a bond-bearing site whose density row could not be joined."""
    first = rows[0]
    summary = _bond_summary(rows)
    warning_reasons = []
    for row in rows:
        warning_reasons.extend(
            reason for reason in row.get(
                "context_warning_reasons", "").split("|") if reason)
    missing_reasons = ["rszd_unavailable", "density_row_unavailable"]
    if summary["reference_covered_contact_count"] == 0:
        missing_reasons.append("no_geometry_reference")
    elif (summary["zbond_available_contact_count"] <
          summary["reference_covered_contact_count"]):
        missing_reasons.append("zbond_unavailable_for_reference")
    if (summary["reference_covered_contact_count"] <
            summary["assigned_contact_count"]):
        missing_reasons.append("partial_geometry_coverage")
    values: Dict[str, Any] = {
        column: "" for column in CONFIDENCE_INPUT_COLUMNS
    }
    values.update({
        "pdbID": key[0],
        "category": first.get("parent_type", ""),
        "coordinate_mapping_status": "density_row_unavailable",
        "selected_metal_site_status": "selected_without_density_row",
        "metal_model_index": key[1],
        "metal_chain_index": key[2],
        "metal_residue_index": key[3],
        "metal_atom_index": key[4],
        "metal_resname": first.get("metal_resname", ""),
        "metal_chain": first.get("metal_chain", ""),
        "metal_resnum": first.get("metal_resnum", ""),
        "metal_atom": first.get("metal_atom", ""),
        "metal_element": first.get("metal_element", ""),
        "metal_icode": first.get("metal_icode", ""),
        "metal_altloc": first.get("metal_altloc", ""),
        **summary,
        "context_warning": any(_true(row.get("context_warning", ""))
                               for row in rows),
        "context_warning_reasons": "|".join(dict.fromkeys(warning_reasons)),
        "confidence_inputs_status": "unscorable",
        "confidence_inputs_missing_reasons": "|".join(missing_reasons),
    })
    return values


def prepare_confidence_inputs(stats_rows, bond_rows):
    """Return deterministic confidence inputs for every selected metal site."""
    bonds_by_site = defaultdict(list)
    for row in bond_rows:
        bonds_by_site[_site_key(row)].append(row)

    seen_sites = set()
    prepared = []
    for stats in stats_rows:
        if stats.get("selected_metal_site_status", "").strip() != "selected":
            continue
        key = _site_key(stats)
        if key in seen_sites:
            raise ValueError(
                "metal statistics contain duplicate site key: " + "/".join(key))
        seen_sites.add(key)

        rszd = abs(_finite_float(stats.get("ZDm", "")))
        negative = _finite_float(stats.get("ZD-m", ""))
        positive = _finite_float(stats.get("ZD+m", ""))
        summary = _bond_summary(bonds_by_site.pop(key, ()))
        missing_reasons = []
        if not math.isfinite(rszd):
            missing_reasons.append("rszd_unavailable")
        if summary["assigned_contact_count"] == 0:
            missing_reasons.append("no_assigned_contacts")
        else:
            if summary["reference_covered_contact_count"] == 0:
                missing_reasons.append("no_geometry_reference")
            elif (summary["zbond_available_contact_count"] <
                  summary["reference_covered_contact_count"]):
                missing_reasons.append("zbond_unavailable_for_reference")
            if (summary["reference_covered_contact_count"] <
                    summary["assigned_contact_count"]):
                missing_reasons.append("partial_geometry_coverage")

        if "rszd_unavailable" in missing_reasons:
            status = "unscorable"
        elif summary["reference_covered_contact_count"] == 0:
            status = "density_only"
        elif ("partial_geometry_coverage" in missing_reasons or
              "zbond_unavailable_for_reference" in missing_reasons):
            status = "partial_geometry"
        else:
            status = "complete"

        output = {column: stats.get(column, "") for column in IDENTITY_COLUMNS}
        output.update({
            "rszd_magnitude": _format_number(rszd),
            "rszd_negative": _format_number(negative),
            "rszd_positive": _format_number(positive),
            **summary,
            "suspect_multi_donor_residue_group_count": stats.get(
                "suspect_multi_donor_residue_group_count", ""),
            "context_warning": stats.get("context_warning", ""),
            "context_warning_reasons": stats.get(
                "context_warning_reasons", ""),
            "confidence_inputs_status": status,
            "confidence_inputs_missing_reasons": "|".join(missing_reasons),
        })
        prepared.append(output)

    for key, rows in sorted(bonds_by_site.items()):
        prepared.append(_orphan_bond_site_input(key, rows))
    return prepared


def prepare_result_confidence_inputs(stats_rows, bond_rows, stats_columns):
    """Prepare confidence rows from one in-memory Alchemy worker result."""
    flattened = []
    for row in stats_rows:
        values = [row["pdbID"], row["category"], *row["fields"]]
        if len(values) != len(stats_columns):
            raise ValueError(
                "metal statistics row does not match the confidence schema")
        flattened.append(dict(zip(stats_columns, values)))
    return prepare_confidence_inputs(flattened, bond_rows)


def complete_confidence_site_count(rows, pdb_id, selected_site_count,
                                   missing_reason=""):
    """Retain unresolved placeholders so no manifest-counted site disappears."""
    if len(rows) > selected_site_count:
        raise ValueError(
            f"confidence inputs exceed selected metal count for {pdb_id}")
    completed = [dict(row) for row in rows]
    if missing_reason:
        for row in completed:
            reasons = [reason for reason in row.get(
                "confidence_inputs_missing_reasons", "").split("|") if reason]
            if missing_reason not in reasons:
                reasons.append(missing_reason)
            row["confidence_inputs_status"] = "unscorable"
            row["confidence_inputs_missing_reasons"] = "|".join(reasons)
    for index in range(len(rows), selected_site_count):
        values: Dict[str, Any] = {
            column: "" for column in CONFIDENCE_INPUT_COLUMNS
        }
        values.update({
            "pdbID": pdb_id,
            "selected_metal_site_status": "selected_site_unresolved",
            "metal_atom_index": f"unresolved-{index + 1}",
            "context_warning": True,
            "context_warning_reasons": "site_evidence_unavailable",
            "confidence_inputs_status": "unscorable",
            "confidence_inputs_missing_reasons": (
                "rszd_unavailable|site_identity_unavailable|"
                "site_evidence_unavailable" +
                (f"|{missing_reason}" if missing_reason else "")),
        })
        completed.append(values)
    return completed


def severity(value, anchors):
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


def score_site(rszd_magnitude, max_abs_zbond, geometry_coverage):
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
        0.15 * coverage * math.sqrt(density_severity * geometry_severity))
    confidence = 100.0 * (
        1.0 - density_component - geometry_component - interaction_component)
    return {
        "density_severity": density_severity,
        "geometry_severity": geometry_severity,
        "density_score_component": density_component,
        "geometry_score_component": geometry_component,
        "interaction_score_component": interaction_component,
        "confidence_score": min(100.0, max(0.0, confidence)),
    }


class ConfidenceReference:
    """Frozen empirical confidence-score distribution from a database run."""

    def __init__(self, values, counts, metadata):
        if len(values) != len(counts) or not values:
            raise ValueError("confidence reference distribution is empty")
        if any(count < 1 for count in counts):
            raise ValueError("confidence reference contains an invalid count")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("confidence reference scores are not increasing")
        self.values = tuple(values)
        self.counts = tuple(counts)
        cumulative = []
        running = 0
        for count in counts:
            cumulative.append(running)
            running += count
        self.cumulative_below = tuple(cumulative)
        self.cohort_size = running
        self.metadata = dict(metadata)
        self.reference_id = self.metadata.get("reference_id", "")

    def percentile(self, score):
        """Average-rank ECDF percentile against the frozen database cohort."""
        index = bisect.bisect_left(self.values, score)
        if index < len(self.values) and self.values[index] == score:
            below = self.cumulative_below[index]
            equal = self.counts[index]
        else:
            below = (self.cumulative_below[index]
                     if index < len(self.values) else self.cohort_size)
            equal = 0
        return 100.0 * (below + 0.5 * equal) / self.cohort_size


#: How damaged geometry coverage is treated. Part of the scoring policy, and so
#: part of ``reference_id``: a reference built before this policy existed may
#: have scored corrupt sites into its cohort, and nothing in the artifact
#: records whether it did. Bumping this makes ``load_reference`` refuse such a
#: reference instead of silently accepting it.
COVERAGE_POLICY = "invalid_coverage_excluded_v1"


def coverage_is_valid(coverage):
    """Whether ``coverage`` is usable geometry evidence.

    A blank, non-numeric or out-of-range coverage is damaged input, not
    evidence that geometry is irrelevant. Coercing it to zero removed the two
    geometry terms from the score, so a corrupt cell scored *higher* than the
    same site with real geometry evidence. Missing data must not move the score
    in either direction: the site is reported unscorable instead.

    Shared by scoring and reference building so the cohort can never contain a
    site that scoring itself would refuse. Those two disagreed once: the fix
    landed only in the scoring path, leaving `finalize_database_confidence`
    coercing non-finite coverage to zero and applying no range check at all.
    """
    return math.isfinite(coverage) and 0.0 <= coverage <= 1.0


def _scoring_metadata():
    return {
        "density_anchors": [list(anchor) for anchor in DENSITY_ANCHORS],
        "geometry_anchors": [list(anchor) for anchor in GEOMETRY_ANCHORS],
        "weights": {
            "density": 0.50,
            "geometry": 0.35,
            "interaction": 0.15,
        },
        "percentile_method": "average_rank_empirical_cdf",
        "coverage_policy": COVERAGE_POLICY,
    }


def _reference_identifier(score_counts):
    digest = hashlib.sha256()
    scoring_parameters = json.dumps(
        _scoring_metadata(), sort_keys=True, separators=(",", ":"))
    digest.update(scoring_parameters.encode("utf-8"))
    digest.update(b"\n")
    for score in sorted(score_counts):
        digest.update(f"{repr(score)},{score_counts[score]}\n".encode("ascii"))
    return "alchemy-confidence-" + digest.hexdigest()[:20]


def write_reference(reference_dir, score_counts, input_row_count):
    """Write a reusable database confidence distribution and its metadata."""
    if not score_counts:
        raise ValueError("cannot build a confidence reference with no scores")
    os.makedirs(reference_dir, exist_ok=True)
    distribution_path = os.path.join(
        reference_dir, REFERENCE_DISTRIBUTION_FILE)
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)

    distribution_tmp = distribution_path + ".tmp"
    with open(distribution_tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("confidence_score", "count"))
        for score in sorted(score_counts):
            writer.writerow((repr(score), score_counts[score]))
    os.replace(distribution_tmp, distribution_path)

    metadata = _scoring_metadata()
    metadata.update({
        "reference_id": _reference_identifier(score_counts),
        "input_row_count": input_row_count,
        "scorable_cohort_size": sum(score_counts.values()),
        "distinct_score_count": len(score_counts),
        "distribution_file": REFERENCE_DISTRIBUTION_FILE,
    })
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


def load_reference(reference_dir):
    """Load and strictly validate a frozen database confidence reference."""
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = _scoring_metadata()
    for key in ("density_anchors", "geometry_anchors", "weights",
                "percentile_method", "coverage_policy"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"confidence reference {key} is incompatible with this code")
    distribution_path = os.path.join(
        reference_dir,
        metadata.get("distribution_file", REFERENCE_DISTRIBUTION_FILE),
    )
    header, rows = _read_csv(distribution_path,
                             "confidence reference distribution")
    if header != ("confidence_score", "count"):
        raise ValueError("confidence reference distribution has invalid columns")
    values = []
    counts = []
    score_counts = Counter()
    for row in rows:
        score = _finite_float(row["confidence_score"])
        try:
            count = int(row["count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "confidence reference contains a non-integer count") from exc
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
    return reference


def _score_prepared_row(row, reference):
    rszd = _finite_float(row.get("rszd_magnitude", ""))
    zbond = _finite_float(row.get("max_abs_zbond", ""))
    coverage = _finite_float(row.get("geometry_coverage", ""))
    coverage_valid = coverage_is_valid(coverage)
    result = score_site(rszd, zbond, coverage) if coverage_valid else None
    output = dict(row)
    if result is None:
        output.update({column: "" for column in ANALYSIS_COLUMNS})
        output.update({
            "confidence_scoring_status": "unscorable",
            "confidence_scoring_reason": (
                "geometry_coverage_invalid" if not coverage_valid
                else "rszd_unavailable" if not math.isfinite(rszd)
                else "zbond_unavailable_with_nonzero_coverage"),
            "confidence_reference_id": reference.reference_id,
            "confidence_cohort_size": reference.cohort_size,
            "confidence_context_warning": row.get("context_warning", ""),
            "confidence_context_warning_reasons": row.get(
                "context_warning_reasons", ""),
        })
        return output, None

    output.update({key: _format_number(value) for key, value in result.items()})
    score = result["confidence_score"]
    output.update({
        "confidence_percentile": _format_number(reference.percentile(score)),
        "confidence_scoring_status": (
            "density_only" if coverage == 0.0 else
            ("partial_geometry" if coverage < 1.0 else "complete")),
        "confidence_scoring_reason": (
            "no_usable_geometry" if coverage == 0.0 else
            ("geometry_coverage_below_one" if coverage < 1.0 else "")),
        "confidence_reference_id": reference.reference_id,
        "confidence_cohort_size": reference.cohort_size,
        "confidence_context_warning": row.get("context_warning", ""),
        "confidence_context_warning_reasons": row.get(
            "context_warning_reasons", ""),
    })
    return output, score


def score_against_reference(rows, reference):
    """Score prepared rows against a frozen database reference."""
    return [_score_prepared_row(row, reference)[0] for row in rows]


def _validated_input_reader(handle):
    reader = csv.DictReader(handle)
    input_columns = tuple(reader.fieldnames or ())
    _required_columns(
        input_columns,
        ("rszd_magnitude", "max_abs_zbond", "geometry_coverage",
         "context_warning", "context_warning_reasons"),
        "confidence input CSV",
    )
    if any(column in input_columns for column in ANALYSIS_COLUMNS):
        raise ValueError("confidence input CSV already contains analysis columns")
    return input_columns, reader


def finalize_database_confidence(input_path, output_path, reference_dir):
    """Build the database reference and assign final values from compact rows."""
    # Metadata is the completion marker. Remove it before rebuilding so a
    # failed finalization cannot leave an older reference looking current.
    metadata_path = os.path.join(reference_dir, REFERENCE_METADATA_FILE)
    if os.path.isfile(metadata_path):
        os.unlink(metadata_path)
    score_counts = Counter()
    input_row_count = 0
    with open(input_path, newline="", encoding="utf-8") as handle:
        _, rows = _validated_input_reader(handle)
        for row in rows:
            input_row_count += 1
            rszd = _finite_float(row.get("rszd_magnitude", ""))
            zbond = _finite_float(row.get("max_abs_zbond", ""))
            coverage = _finite_float(row.get("geometry_coverage", ""))
            # Same validity test the scoring path applies, so a site reported
            # unscorable cannot also enter the cohort it is measured against.
            result = (score_site(rszd, zbond, coverage)
                      if coverage_is_valid(coverage) else None)
            if result is not None:
                score_counts[result["confidence_score"]] += 1
    reference = write_reference(reference_dir, score_counts, input_row_count)
    total, scored = score_file_against_reference(
        input_path, output_path, reference)
    return total, scored, reference.cohort_size


def score_file_against_reference(input_path, output_path, reference):
    """Score a compact input CSV against a loaded frozen reference."""
    output_tmp = output_path + ".tmp"
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    scored = 0
    try:
        with (open(input_path, newline="", encoding="utf-8") as source,
              open(output_tmp, "w", newline="", encoding="utf-8") as target):
            input_columns, rows = _validated_input_reader(source)
            writer = csv.DictWriter(
                target, fieldnames=(*input_columns, *ANALYSIS_COLUMNS))
            writer.writeheader()
            for row in rows:
                output, score = _score_prepared_row(row, reference)
                writer.writerow(output)
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


def validate_scored_reference(path, reference):
    """Refuse resume output containing rows from another frozen reference."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "confidence_reference_id" not in (reader.fieldnames or ()):
            raise ValueError(
                "existing confidence output has no reference identifier")
        identifiers = {
            row["confidence_reference_id"].strip()
            for row in reader if row["confidence_reference_id"].strip()
        }
    if identifiers and identifiers != {reference.reference_id}:
        raise ValueError(
            "existing confidence output uses a different database reference")


def _parser():
    parser = argparse.ArgumentParser(
        description="Finalize or apply Alchemy confidence scores.")
    commands = parser.add_subparsers(dest="command", required=True)

    finalize = commands.add_parser(
        "finalize", help="finalize a streamed complete-database cohort")
    finalize.add_argument("--input", required=True,
                          help="streamed database confidence-input CSV")
    finalize.add_argument("--output", required=True,
                          help="output database confidence-score CSV")
    finalize.add_argument("--reference-dir", required=True,
                          help="output frozen database reference directory")

    score = commands.add_parser(
        "score", help="score inputs against a frozen database reference")
    score.add_argument("--input", required=True,
                       help="prepared confidence-input CSV")
    score.add_argument("--output", required=True,
                       help="output confidence-score CSV path")
    score.add_argument("--reference-dir", required=True,
                       help="frozen database reference directory")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "finalize":
            total, scored, cohort = finalize_database_confidence(
                args.input, args.output, args.reference_dir)
            print(
                f"finalized {total} rows ({scored} scored; database cohort "
                f"{cohort}) to {args.output}")
        else:
            reference = load_reference(args.reference_dir)
            total, scored = score_file_against_reference(
                args.input, args.output, reference)
            print(f"wrote {total} rows ({scored} scored) to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"confidence {args.command} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
