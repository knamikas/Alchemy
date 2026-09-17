"""Derive compact confidence inputs from site and bond evidence."""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from codes import (
    ConfidenceInputStatus,
    CoordinateMappingStatus,
    CoordinationStatus,
    ParentType,
    ReasonCode,
    SelectedSiteStatus,
)
from confidence_score.schema import (
    CONFIDENCE_INPUT_COLUMNS,
    EDSTATS_SATURATION_MAGNITUDE,
    IDENTITY_COLUMNS,
    METRIC_DECIMAL_PLACES,
    format_decimal,
    parse_csv_bool,
    site_key,
)
from output_rows import MetalStatsRow, finite_float
from reference_data import cofactor_ids


def _bond_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored_bonds: list[tuple[float, Mapping[str, Any]]] = []
    reference_covered = 0
    declared = 0
    inferred = 0
    declared_scored = 0
    inferred_scored = 0
    multi_donor = 0
    for row in rows:
        reference_covered += parse_csv_bool(row.get("reference_covered", ""))
        is_declared = parse_csv_bool(row.get("declared_connection", ""))
        is_inferred = (
            row.get("coordination_status", "").strip() == CoordinationStatus.INFERRED
        )
        declared += is_declared
        inferred += is_inferred
        multi_donor += parse_csv_bool(row.get("multi_donor_detected", ""))
        zscore = finite_float(row.get("zscore", ""))
        if parse_csv_bool(row.get("score_eligible", "")) and math.isfinite(zscore):
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
        "geometry_coverage": format_decimal(coverage),
        "geometry_rms_zbond": (
            format_decimal(
                math.sqrt(sum(value * value for value in zscores) / len(zscores)),
                METRIC_DECIMAL_PLACES,
            )
            if zscores
            else ""
        ),
        "geometry_max_abs_zbond": (
            format_decimal(abs(largest[0]), METRIC_DECIMAL_PLACES) if largest else ""
        ),
        "geometry_mean_abs_zbond": (
            format_decimal(
                sum(abs(value) for value in zscores) / len(zscores),
                METRIC_DECIMAL_PLACES,
            )
            if zscores
            else ""
        ),
        "geometry_mean_signed_zbond": (
            format_decimal(sum(zscores) / len(zscores), METRIC_DECIMAL_PLACES)
            if zscores
            else ""
        ),
        "worst_bond": largest_row.get("contact_id", ""),
        "worst_bond_source": (
            CoordinationStatus.DECLARED
            if largest and parse_csv_bool(largest_row.get("declared_connection", ""))
            else CoordinationStatus.INFERRED
            if largest
            and largest_row.get("coordination_status", "").strip()
            == CoordinationStatus.INFERRED
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


def _geometry_missing_reasons(summary: Mapping[str, Any]) -> list[str]:
    """Explain which geometry evidence a bond summary lacks, worst first."""
    reasons: list[str] = []
    if summary["reference_covered_contact_count"] == 0:
        reasons.append("no_geometry_reference")
    elif summary["geometry_bond_count"] < summary["reference_covered_contact_count"]:
        reasons.append("zbond_unavailable_for_reference")
    if summary["reference_covered_contact_count"] < summary["assigned_contact_count"]:
        reasons.append("partial_geometry_coverage")
    return reasons


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
    missing_reasons = [
        "rszd_unavailable",
        CoordinateMappingStatus.DENSITY_ROW_UNAVAILABLE,
        *_geometry_missing_reasons(summary),
    ]
    values: dict[str, Any] = dict.fromkeys(CONFIDENCE_INPUT_COLUMNS, "")
    values.update(
        {
            "pdbID": first.get("pdbID", key[0]),
            "category": (
                "cofactor"
                if str(first.get("metal_resname", "")).upper() in cofactor_ids()
                else "metal"
                if first.get("parent_type", "") == ParentType.ION
                else ""
            ),
            "metal_site_id": first.get("metal_site_id", ""),
            "coordinate_mapping_status": CoordinateMappingStatus.DENSITY_ROW_UNAVAILABLE,
            "selected_metal_site_status": SelectedSiteStatus.SELECTED_WITHOUT_DENSITY_ROW,
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
                parse_csv_bool(row.get("context_warning", "")) for row in rows
            ),
            "context_warning_reasons": "|".join(dict.fromkeys(warning_reasons)),
            "confidence_inputs_status": (
                ConfidenceInputStatus.GEOMETRY_ONLY
                if summary["geometry_bond_count"]
                else ConfidenceInputStatus.UNSCORABLE
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
        bonds_by_site[site_key(row)].append(row)

    seen_sites: set[tuple[str, ...]] = set()
    prepared: list[dict[str, Any]] = []
    for stats in stats_rows:
        if (
            stats.get("selected_metal_site_status", "").strip()
            != SelectedSiteStatus.SELECTED
        ):
            continue
        key = site_key(stats)
        if key in seen_sites:
            raise ValueError(
                "metal statistics contain duplicate site key: " + "/".join(key)
            )
        seen_sites.add(key)

        rszd = finite_float(stats.get("ZDm", ""))
        rszd_abs = abs(rszd)
        negative = finite_float(stats.get("ZD-m", ""))
        positive = finite_float(stats.get("ZD+m", ""))
        summary = _bond_summary(bonds_by_site.pop(key, ()))
        missing_reasons: list[str] = []
        if str(stats.get("metal_coordinates_valid", "")).strip().lower() == "false":
            missing_reasons.append(ReasonCode.NON_FINITE_METAL_COORDINATES)
        if not math.isfinite(rszd_abs):
            missing_reasons.append("rszd_unavailable")
        if summary["assigned_contact_count"] == 0:
            missing_reasons.append("no_assigned_contacts")
        else:
            missing_reasons.extend(_geometry_missing_reasons(summary))

        density_available = math.isfinite(rszd_abs)
        geometry_available = summary["geometry_bond_count"] > 0
        status = (
            ConfidenceInputStatus.COMPLETE
            if density_available and geometry_available
            else ConfidenceInputStatus.DENSITY_ONLY
            if density_available
            else ConfidenceInputStatus.GEOMETRY_ONLY
            if geometry_available
            else ConfidenceInputStatus.UNSCORABLE
        )

        output = {column: stats.get(column, "") for column in IDENTITY_COLUMNS}
        output.update(
            {
                "rszd": format_decimal(rszd, METRIC_DECIMAL_PLACES),
                "rszd_abs": format_decimal(rszd_abs, METRIC_DECIMAL_PLACES),
                "rszd_negative": format_decimal(negative),
                "rszd_positive": format_decimal(positive),
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
                "selected_metal_site_status": SelectedSiteStatus.SELECTED_SITE_UNRESOLVED,
                "metal_atom_index": f"unresolved-{index + 1}",
                "context_warning": True,
                "context_warning_reasons": "site_evidence_unavailable",
                "confidence_inputs_status": ConfidenceInputStatus.UNSCORABLE,
                "confidence_inputs_missing_reasons": (
                    "rszd_unavailable|site_identity_unavailable|"
                    "site_evidence_unavailable"
                    + (f"|{missing_reason}" if missing_reason else "")
                ),
            }
        )
        completed.append(values)
    return completed
