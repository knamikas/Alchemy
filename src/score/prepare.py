"""Derive compact score inputs from site and bond evidence."""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from codes import (
    CoordinateMappingStatus,
    CoordinationStatus,
    GeometryContactBasis,
    ParentType,
    ReasonCode,
    ScoreInputStatus,
    ScoreMissingReason,
    SelectedSiteStatus,
)
from metallocofactors.catalog import cofactor_ids
from output_rows import MetalStatsRow, finite_float
from score.schema import (
    IDENTITY_COLUMNS,
    METRIC_DECIMAL_PLACES,
    SCORE_INPUT_COLUMNS,
    format_decimal,
    is_density_saturated,
    parse_csv_bool,
    site_key,
)

#: ``metal_atom_index`` of a placeholder row, prefixed to the site's one-based
#: position among the entry's unresolved sites. A placeholder names a site the
#: manifest counts but no output row identifies, so its true coordinate indices
#: are unknown; the sentinel occupies the column that would otherwise be blank
#: so that ``site_key`` still distinguishes two placeholders of one entry.
UNRESOLVED_SITE_INDEX_PREFIX = "unresolved-"


@dataclass(frozen=True)
class BondSummary:
    """Geometry evidence aggregated over one site's assigned contacts."""

    assigned_contact_count: int
    reference_covered_contact_count: int
    geometry_bond_count: int
    geometry_coverage: str
    geometry_rms_zbond: str
    geometry_max_abs_zbond: str
    geometry_mean_abs_zbond: str
    geometry_mean_signed_zbond: str
    #: Identity of the worst scored contact, copied from its bond row and so
    #: carrying whatever that row held; blank when nothing was scored.
    worst_bond: Any
    worst_bond_source: str
    worst_bond_neighbor_resname: Any
    worst_bond_neighbor_chain: Any
    worst_bond_neighbor_resnum: Any
    worst_bond_neighbor_atom: Any
    declared_contact_count: int
    inferred_contact_count: int
    declared_scored_bond_count: int
    inferred_scored_bond_count: int
    geometry_contact_basis: GeometryContactBasis
    multi_donor_contact_count: int

    def as_columns(self) -> dict[str, Any]:
        """The geometry block of a prepared row, in output-schema order."""
        return {
            "assigned_contact_count": self.assigned_contact_count,
            "reference_covered_contact_count": self.reference_covered_contact_count,
            "geometry_bond_count": self.geometry_bond_count,
            "geometry_coverage": self.geometry_coverage,
            "geometry_rms_zbond": self.geometry_rms_zbond,
            "geometry_max_abs_zbond": self.geometry_max_abs_zbond,
            "geometry_mean_abs_zbond": self.geometry_mean_abs_zbond,
            "geometry_mean_signed_zbond": self.geometry_mean_signed_zbond,
            "worst_bond": self.worst_bond,
            "worst_bond_source": self.worst_bond_source,
            "worst_bond_neighbor_resname": self.worst_bond_neighbor_resname,
            "worst_bond_neighbor_chain": self.worst_bond_neighbor_chain,
            "worst_bond_neighbor_resnum": self.worst_bond_neighbor_resnum,
            "worst_bond_neighbor_atom": self.worst_bond_neighbor_atom,
            "declared_contact_count": self.declared_contact_count,
            "inferred_contact_count": self.inferred_contact_count,
            "declared_scored_bond_count": self.declared_scored_bond_count,
            "inferred_scored_bond_count": self.inferred_scored_bond_count,
            "geometry_contact_basis": self.geometry_contact_basis,
            "multi_donor_contact_count": self.multi_donor_contact_count,
        }


def _bond_summary(rows: Sequence[Mapping[str, Any]]) -> BondSummary:
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
    return BondSummary(
        assigned_contact_count=assigned,
        reference_covered_contact_count=reference_covered,
        geometry_bond_count=len(zscores),
        # A site with no assigned contact publishes "0" rather than a blank:
        # retained as published, even though nothing was measured.
        geometry_coverage=format_decimal(coverage),
        geometry_rms_zbond=(
            format_decimal(
                math.sqrt(sum(value * value for value in zscores) / len(zscores)),
                METRIC_DECIMAL_PLACES,
            )
            if zscores
            else ""
        ),
        geometry_max_abs_zbond=(
            format_decimal(abs(largest[0]), METRIC_DECIMAL_PLACES) if largest else ""
        ),
        geometry_mean_abs_zbond=(
            format_decimal(
                sum(abs(value) for value in zscores) / len(zscores),
                METRIC_DECIMAL_PLACES,
            )
            if zscores
            else ""
        ),
        geometry_mean_signed_zbond=(
            format_decimal(sum(zscores) / len(zscores), METRIC_DECIMAL_PLACES)
            if zscores
            else ""
        ),
        worst_bond=largest_row.get("contact_id", ""),
        worst_bond_source=(
            CoordinationStatus.DECLARED
            if largest and parse_csv_bool(largest_row.get("declared_connection", ""))
            else CoordinationStatus.INFERRED
            if largest
            and largest_row.get("coordination_status", "").strip()
            == CoordinationStatus.INFERRED
            else ""
        ),
        worst_bond_neighbor_resname=largest_row.get("neighbor_resname", ""),
        worst_bond_neighbor_chain=largest_row.get("neighbor_chain", ""),
        worst_bond_neighbor_resnum=largest_row.get("neighbor_resnum", ""),
        worst_bond_neighbor_atom=largest_row.get("neighbor_atom", ""),
        declared_contact_count=declared,
        inferred_contact_count=inferred,
        declared_scored_bond_count=declared_scored,
        inferred_scored_bond_count=inferred_scored,
        geometry_contact_basis=(
            GeometryContactBasis.DECLARED_AND_INFERRED
            if declared_scored and inferred_scored
            else GeometryContactBasis.DECLARED_ONLY
            if declared_scored
            else GeometryContactBasis.INFERRED_ONLY
            if inferred_scored
            else GeometryContactBasis.NONE
        ),
        multi_donor_contact_count=multi_donor,
    )


def _geometry_missing_reasons(summary: BondSummary) -> list[str]:
    """Explain which geometry evidence a bond summary lacks, worst first."""
    reasons: list[str] = []
    if summary.reference_covered_contact_count == 0:
        reasons.append(ScoreMissingReason.NO_GEOMETRY_REFERENCE)
    elif summary.geometry_bond_count < summary.reference_covered_contact_count:
        reasons.append(ScoreMissingReason.ZBOND_UNAVAILABLE_FOR_REFERENCE)
    if summary.reference_covered_contact_count < summary.assigned_contact_count:
        reasons.append(ScoreMissingReason.PARTIAL_GEOMETRY_COVERAGE)
    return reasons


def _input_row(identity: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Build one prepared row: a blank schema, the identity block, overrides.

    Every row of ``score_inputs_all.csv`` is built here, so all three
    producers carry the full column set in schema order however little
    evidence they have.
    """
    values: dict[str, Any] = dict.fromkeys(SCORE_INPUT_COLUMNS, "")
    values.update({column: identity.get(column, "") for column in IDENTITY_COLUMNS})
    values.update(overrides)
    return values


def _orphan_bond_site_input(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    missing_reasons: list[str] = [
        ScoreMissingReason.RSZD_UNAVAILABLE,
        CoordinateMappingStatus.DENSITY_ROW_UNAVAILABLE,
        *_geometry_missing_reasons(summary),
    ]
    # A bond row carries none of the density identity columns, so the
    # ``density_*`` block stays blank; the three status columns below are the
    # only identity values this path knows better than the bond rows do.
    return _input_row(
        first,
        # Bond rows carry no residue classification, so this rule is not
        # ``edstats_statistics.classify_residue``: only the bond row's own
        # fields are available. A site that is neither a catalog cofactor nor
        # an ION therefore publishes a blank category, retained as published.
        category=(
            "cofactor"
            if str(first.get("metal_resname", "")).upper() in cofactor_ids()
            else "metal"
            if first.get("parent_type", "") == ParentType.ION
            else ""
        ),
        coordinate_mapping_status=CoordinateMappingStatus.DENSITY_ROW_UNAVAILABLE,
        selected_metal_site_status=SelectedSiteStatus.SELECTED_WITHOUT_DENSITY_ROW,
        **summary.as_columns(),
        # A real ``bool`` here and in the placeholder rows, but the raw stats
        # value on the main path: ``schema.score_csv_value`` and
        # ``output_rows.scientific_csv_value`` normalize both at write time,
        # and normalizing here would turn a blank (unknown) cell into "false".
        context_warning=any(
            parse_csv_bool(row.get("context_warning", "")) for row in rows
        ),
        context_warning_reasons="|".join(dict.fromkeys(warning_reasons)),
        score_inputs_status=(
            ScoreInputStatus.GEOMETRY_ONLY
            if summary.geometry_bond_count
            else ScoreInputStatus.UNSCORABLE
        ),
        score_inputs_missing_reasons="|".join(missing_reasons),
    )


def prepare_score_inputs(
    stats_rows: Sequence[Mapping[str, Any]],
    bond_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic score inputs for every selected metal site."""
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
        # Compared against "false" rather than parsed with ``parse_csv_bool``:
        # a blank means the validity was never determined, which this path
        # treats as valid, whereas ``parse_csv_bool`` reads a blank as false.
        if str(stats.get("metal_coordinates_valid", "")).strip().lower() == "false":
            missing_reasons.append(ReasonCode.NON_FINITE_METAL_COORDINATES)
        if not math.isfinite(rszd_abs):
            missing_reasons.append(ScoreMissingReason.RSZD_UNAVAILABLE)
        if summary.assigned_contact_count == 0:
            missing_reasons.append(ScoreMissingReason.NO_ASSIGNED_CONTACTS)
        else:
            missing_reasons.extend(_geometry_missing_reasons(summary))

        density_available = math.isfinite(rszd_abs)
        geometry_available = summary.geometry_bond_count > 0
        status = (
            ScoreInputStatus.COMPLETE
            if density_available and geometry_available
            else ScoreInputStatus.DENSITY_ONLY
            if density_available
            else ScoreInputStatus.GEOMETRY_ONLY
            if geometry_available
            else ScoreInputStatus.UNSCORABLE
        )

        prepared.append(
            _input_row(
                stats,
                rszd=format_decimal(rszd, METRIC_DECIMAL_PLACES),
                rszd_abs=format_decimal(rszd_abs, METRIC_DECIMAL_PLACES),
                # The signed pair keeps ``format_decimal``'s six-place default:
                # only the metrics the frozen reference bins -- |RSZD| and the
                # Zbond statistics -- are published with twelve places.
                rszd_negative=format_decimal(negative),
                rszd_positive=format_decimal(positive),
                # ``is_density_saturated`` accepts anything at or above the
                # ceiling where the published rows tested equality; EDSTATS
                # caps its field at 99.9 and parsed values have 0.1
                # granularity, so the two agree on every reachable value.
                density_saturated=is_density_saturated(rszd_abs),
                **summary.as_columns(),
                suspect_multi_donor_residue_group_count=stats.get(
                    "suspect_multi_donor_residue_group_count", ""
                ),
                # The raw stats value, not a ``bool``: see the note in
                # ``_orphan_bond_site_input``.
                context_warning=stats.get("context_warning", ""),
                context_warning_reasons=stats.get("context_warning_reasons", ""),
                score_inputs_status=status,
                score_inputs_missing_reasons="|".join(missing_reasons),
            )
        )

    # Bond rows whose site has no SELECTED statistics row land here as well as
    # bond rows with no statistics row at all: producers emit bonds only for
    # selected sites, so the first case is unreachable today.
    for _key, rows in sorted(bonds_by_site.items()):
        prepared.append(_orphan_bond_site_input(rows))
    return prepared


def prepare_result_score_inputs(
    stats_rows: Sequence[MetalStatsRow],
    bond_rows: Sequence[Mapping[str, Any]],
    stats_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Prepare score rows from one in-memory Alchemy worker result."""
    flattened = [row.as_output_dict(stats_columns) for row in stats_rows]
    return prepare_score_inputs(flattened, bond_rows)


def _joined_missing_reasons(reasons: list[str], missing_reason: str) -> str:
    """Join a row's reasons, adding the entry-level one it does not carry."""
    if missing_reason and missing_reason not in reasons:
        reasons.append(missing_reason)
    return "|".join(reasons)


def complete_score_site_count(
    rows: Sequence[Mapping[str, Any]],
    pdb_id: str,
    selected_site_count: int,
    missing_reason: str = "",
) -> list[dict[str, Any]]:
    """Retain unresolved placeholders so no manifest-counted site disappears.

    Copies of ``rows`` are returned, one placeholder appended for every
    manifest-counted site no prepared row identifies. An entry-level
    ``missing_reason`` -- the ``ReasonCode`` of whatever stopped the entry --
    is also annotated onto every existing row's
    ``score_inputs_missing_reasons``, so the reason a site is thin is
    recorded on the rows that exist as well as on the placeholders.
    """
    if len(rows) > selected_site_count:
        raise ValueError(f"score inputs exceed selected metal count for {pdb_id}")
    completed = [dict(row) for row in rows]
    if missing_reason:
        for row in completed:
            row["score_inputs_missing_reasons"] = _joined_missing_reasons(
                [
                    reason
                    for reason in row.get("score_inputs_missing_reasons", "").split("|")
                    if reason
                ],
                missing_reason,
            )
    for index in range(len(rows), selected_site_count):
        placeholder_reasons: list[str] = [
            ScoreMissingReason.RSZD_UNAVAILABLE,
            ScoreMissingReason.SITE_IDENTITY_UNAVAILABLE,
            ScoreMissingReason.SITE_EVIDENCE_UNAVAILABLE,
        ]
        completed.append(
            _input_row(
                {},
                pdbID=pdb_id,
                selected_metal_site_status=SelectedSiteStatus.SELECTED_SITE_UNRESOLVED,
                metal_atom_index=f"{UNRESOLVED_SITE_INDEX_PREFIX}{index + 1}",
                context_warning=True,
                context_warning_reasons=(ScoreMissingReason.SITE_EVIDENCE_UNAVAILABLE),
                score_inputs_status=ScoreInputStatus.UNSCORABLE,
                score_inputs_missing_reasons=_joined_missing_reasons(
                    placeholder_reasons, missing_reason
                ),
            )
        )
    return completed
