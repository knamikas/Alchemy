"""Extracting metal and cofactor real-space statistics from EDSTATS output.

EDSTATS' ``stats.out`` is a whitespace-separated table whose first non-empty
line is the column header.
"""

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from codes import WarningCode
from output_rows import MetalStatsRow
from structure_analysis import (
    NAN,
    AtomSite,
    ResidueSelection,
    StructureContext,
    canonical_pdb_residue_id,
)

# EDSTATS 1.0.9's standard residue-table schema, whose twelve metrics repeat
# for main-chain, side-chain and all atoms.
_EDSTATS_METRIC_STEMS = (
    "BA",
    "NP",
    "R",
    "RG",
    "SRG",
    "CCS",
    "CCP",
    "ZCCP",
    "ZO",
    "ZD",
    "ZD-",
    "ZD+",
)
EDSTATS_METRIC_COLUMNS = tuple(
    f"{stem}{atom_group}"
    for atom_group in ("m", "s", "a")
    for stem in _EDSTATS_METRIC_STEMS
)
EDSTATS_COLUMNS = (
    "RT",
    "CI",
    "RN",
    *EDSTATS_METRIC_COLUMNS,
    "MN",
    "CP",
    "NR",
)
# EDSTATS' documented marker for a statistic it could not calculate.
EDSTATS_NULL_VALUE = "n/a"
EDSTATS_MISSING_CHAIN_IDS = frozenset(("", ".", "?", "_"))
EDSTATS_GRID_POINT_COLUMNS = ("NPm", "NPs", "NPa")
EDSTATS_FIXED_WIDTH_OVERFLOW = "****"


_DENSITY_CONTEXT_GROUPS = ("ordinary", "ordinary_nonwater", "water")
_DENSITY_CONTEXT_GROUP_COLUMNS = (
    "residue_count",
    "rszd_count",
    "median_abs_rszd",
    "abs_rszd_ge_3_count",
    "abs_rszd_ge_3_fraction",
    "abs_rszd_ge_6_count",
    "abs_rszd_ge_6_fraction",
)
DENSITY_CONTEXT_COLUMNS = (
    "pdbID",
    "density_context_status",
    "edstats_residue_count",
    "target_residue_count",
    *(
        f"{group}_{column}"
        for group in _DENSITY_CONTEXT_GROUPS
        for column in _DENSITY_CONTEXT_GROUP_COLUMNS
    ),
)
DENSITY_CONTEXT_STATUSES = frozenset(("available", "not_computed"))


def _empty_float_values() -> list[float]:
    """Return a typed list for strict type checkers inspecting dataclass fields."""
    return []


@dataclass
class _DensityContextAccumulator:
    """Compact non-target RSZD distributions collected during table parsing."""

    edstats_residue_count: int = 0
    target_residue_count: int = 0
    ordinary_residue_count: int = 0
    ordinary_nonwater_residue_count: int = 0
    water_residue_count: int = 0
    ordinary_values: list[float] = field(default_factory=_empty_float_values)
    ordinary_nonwater_values: list[float] = field(default_factory=_empty_float_values)
    water_values: list[float] = field(default_factory=_empty_float_values)

    def observe(
        self,
        resolved: "_ResolvedEdstatsRow",
        metals_upper: set[str],
        cofactors: frozenset[str],
    ) -> None:
        self.edstats_residue_count += 1
        coordinate_resname, _chain, _resnum = resolved.coordinate_key
        target = coordinate_resname in cofactors or any(
            classify_residue(residue, metals_upper, cofactors)[0]
            for residue in resolved.matched_residues
        )
        if target:
            self.target_residue_count += 1
            return

        is_water = any(residue.is_water for residue in resolved.matched_residues)
        self.ordinary_residue_count += 1
        if is_water:
            self.water_residue_count += 1
        else:
            self.ordinary_nonwater_residue_count += 1

        value = resolved.fields[resolved.indices["ZDa"]]
        if value.lower() == EDSTATS_NULL_VALUE:
            return
        magnitude = abs(float(value))
        self.ordinary_values.append(magnitude)
        (self.water_values if is_water else self.ordinary_nonwater_values).append(
            magnitude
        )

    @staticmethod
    def _group_values(residue_count: int, values: list[float]) -> dict[str, Any]:
        finite_count = len(values)
        ge_3 = sum(value >= 3.0 for value in values)
        ge_6 = sum(value >= 6.0 for value in values)
        return {
            "residue_count": residue_count,
            "rszd_count": finite_count,
            "median_abs_rszd": statistics.median(values) if values else "",
            "abs_rszd_ge_3_count": ge_3,
            "abs_rszd_ge_3_fraction": ge_3 / finite_count if finite_count else "",
            "abs_rszd_ge_6_count": ge_6,
            "abs_rszd_ge_6_fraction": ge_6 / finite_count if finite_count else "",
        }

    def as_row(self, pdb_id: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "pdbID": pdb_id,
            "density_context_status": "available",
            "edstats_residue_count": self.edstats_residue_count,
            "target_residue_count": self.target_residue_count,
        }
        groups = {
            "ordinary": (self.ordinary_residue_count, self.ordinary_values),
            "ordinary_nonwater": (
                self.ordinary_nonwater_residue_count,
                self.ordinary_nonwater_values,
            ),
            "water": (self.water_residue_count, self.water_values),
        }
        for group, (residue_count, values) in groups.items():
            row.update(
                {
                    f"{group}_{name}": value
                    for name, value in self._group_values(residue_count, values).items()
                }
            )
        return row


def is_edstats_separator(fields: list[str]) -> bool:
    """Whether split fields are EDSTATS' synthetic model separator row.

    For a MODEL/ENDMDL-wrapped XYZIN, EDSTATS 1.0.9 emits a row whose residue
    name, residue number and chain-position fields are blank, so whitespace
    splitting yields ``_``, 36 ``n/a`` metrics, model and row number rather
    than the 42-column residue schema. Older captured output carries only the
    ``_`` marker.
    """
    if fields == ["_"]:
        return True
    metric_count = len(EDSTATS_METRIC_COLUMNS)
    return (
        len(fields) == metric_count + 3
        and fields[0] == "_"
        and all(
            value.lower() == EDSTATS_NULL_VALUE
            for value in fields[1 : metric_count + 1]
        )
        and all(value.isdigit() for value in fields[-2:])
    )


def normalize_edstats_row(
    fields: Sequence[str], header: Sequence[str], indices: Mapping[str, int]
) -> list[str]:
    """Restore and normalize EDSTATS' valid blank-chain representation.

    EDSTATS leaves the trailing chain field (CP) empty for a blank-chain
    residue, so whitespace splitting removes it. CP is the only field EDSTATS
    can legitimately omit, so restore it for that unambiguous shape -- one
    field short, ending in the integer MN and NR values -- and leave every
    other short row to fail row validation.

    The leading CI field cannot gate the restoration: CI is EDSTATS' own group
    label, reported as ``0`` for ordered waters whatever their actual chain,
    while CP carries the deposited one.
    """
    normalized = list(fields)
    if (
        len(normalized) == len(header) - 1
        and indices["MN"] == len(normalized) - 2
        and indices["CP"] == len(normalized) - 1
    ):
        try:
            int(normalized[-2])
            int(normalized[-1])
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            normalized.insert(indices["CP"], "")

    if len(normalized) == len(header):
        for name in ("CI", "CP"):
            index = indices[name]
            if normalized[index] in EDSTATS_MISSING_CHAIN_IDS:
                normalized[index] = ""
    return normalized


def _edstats_chain_and_altloc(value: str, line_number: int) -> tuple[str, str]:
    """Split EDSTATS' USEALT chain label into deposited chain and altloc.

    With ``USEALT=true``, EDSTATS 1.0.9 writes ``CI`` as ``<chain>:<altloc>``
    (for example ``A:B``) while leaving ``CP`` as the deposited chain.  PDB
    alternate-location identifiers are one character, so any other colon form
    is ambiguous and must not be silently joined to a coordinate residue.
    """
    text = str(value)
    if ":" not in text:
        chain = "" if text in EDSTATS_MISSING_CHAIN_IDS else text
        return chain, ""

    chain, altloc = text.rsplit(":", 1)
    if ":" in chain or len(altloc) != 1 or altloc in EDSTATS_MISSING_CHAIN_IDS:
        raise ValueError(
            f"invalid EDSTATS alternate-conformer CI value on row "
            f"{line_number}: {text!r}"
        )
    if chain in EDSTATS_MISSING_CHAIN_IDS:
        chain = ""
    return chain, altloc


def _row_matches_selected_altloc(
    row_altloc: str,
    matched_residues: Sequence[ResidueSelection],
    chain_residues: Sequence[ResidueSelection],
    row_number: int,
    line_number: int,
) -> bool:
    """Whether one USEALT row is the conformer selected by Alchemy.

    Author identifiers normally provide the residue.  The chain-local EDSTATS
    ordinal is also safe for this narrow decision when an identifier join
    failed: it lets an unmatched cofactor diagnostic discard its unselected
    alternate rows without pretending that the author-key join succeeded.
    """
    residue = matched_residues[0] if len(matched_residues) == 1 else None
    index = row_number - 1
    if residue is None and 0 <= index < len(chain_residues):
        residue = chain_residues[index]

    if residue is None:
        if row_altloc:
            raise ValueError(
                f"EDSTATS alternate conformer {row_altloc!r} on row "
                f"{line_number} cannot be matched to a coordinate residue"
            )
        return True

    selected = residue.selected_altloc
    if selected:
        if not row_altloc:
            # EDSTATS 1.0.9 retains a pooled summary row alongside the
            # conformer-specific rows requested by USEALT=true.  Ignore that
            # summary: the selected conformer's row below is the only density
            # observation Alchemy may use.  If it is absent for a metal or
            # cofactor, the completeness check at the end of extraction still
            # rejects the output.
            return False
        return row_altloc == selected
    if row_altloc:
        raise ValueError(
            f"EDSTATS row {line_number} reports alternate conformer "
            f"{row_altloc!r} for a residue with no selected alternate"
        )
    return True


def validated_edstats_header(fields: Sequence[str]) -> dict[str, int]:
    """Return column indices after validating the standard EDSTATS schema."""
    duplicates = sorted({name for name in fields if fields.count(name) > 1})
    if duplicates:
        raise ValueError(
            "EDSTATS header contains duplicate columns: " + ", ".join(duplicates)
        )

    missing = [name for name in EDSTATS_COLUMNS if name not in fields]
    unexpected = [name for name in fields if name not in EDSTATS_COLUMNS]
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError("invalid EDSTATS header: " + "; ".join(details))
    if tuple(fields) != EDSTATS_COLUMNS:
        raise ValueError("EDSTATS columns are not in the standard order")

    return {name: index for index, name in enumerate(fields)}


def validate_edstats_row(
    fields: Sequence[str],
    header: Sequence[str],
    indices: Mapping[str, int],
    line_number: int,
) -> int:
    """Validate one residue row and return its model number."""
    if len(fields) != len(header):
        raise ValueError(
            f"EDSTATS row {line_number} has {len(fields)} columns; "
            f"expected {len(header)}"
        )

    for name in EDSTATS_METRIC_COLUMNS:
        value = fields[indices[name]]
        if value.lower() == EDSTATS_NULL_VALUE:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"EDSTATS row {line_number} has a nonnumeric {name} value: {value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                f"EDSTATS row {line_number} has a non-finite {name} value: {value!r}"
            )

    model_value = fields[indices["MN"]]
    try:
        return int(model_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"invalid EDSTATS MN model value on row {line_number}: {model_value!r}"
        ) from exc


def classify_residue(
    residue: ResidueSelection, metals_upper: set[str], cofactor_set: Iterable[str]
) -> tuple[str, list[AtomSite]]:
    """Return ``(category, metal_sites)`` for one coordinate residue.

    ``category`` is ``"cofactor"``, ``"metal"``, or ``""`` when the residue is
    neither. The emitted rows and the EDSTATS completeness check below both
    derive from this one rule, so the set of sites Alchemy demands EDSTATS
    report cannot drift from the set it emits.
    """
    metal_sites = [
        atom
        for atom in residue.contact_atoms
        if atom.element_known
        and atom.element in metals_upper
        and not (atom.occupancy_valid and atom.occupancy == 0.0)
    ]
    if residue.residue_name in cofactor_set:
        return "cofactor", metal_sites
    if residue.chemical_atom_site_count == 1 and len(metal_sites) == 1:
        return "metal", metal_sites
    return "", metal_sites


def expected_edstats_residues(
    structure: StructureContext, metals_upper: set[str], cofactor_set: Iterable[str]
) -> Counter[tuple[str, str, str]]:
    """Coordinate residue-key multiplicities Alchemy expects EDSTATS to report."""
    return Counter(
        residue.coordinate_author_key
        for residue in structure.residues
        if classify_residue(residue, metals_upper, cofactor_set)[0]
    )


def _validated_edstats_row_number(
    fields: Sequence[str], indices: Mapping[str, int], line_number: int
) -> int:
    """Return EDSTATS' one-based residue ordinal within the row's chain part."""
    value = fields[indices["NR"]]
    try:
        row_number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"invalid EDSTATS NR row value on row {line_number}: {value!r}"
        ) from exc
    if row_number < 1:
        raise ValueError(
            f"invalid EDSTATS NR row value on row {line_number}: {value!r}"
        )
    return row_number


def _coordinate_residues_by_chain_part(
    structure: StructureContext,
) -> dict[str, tuple[ResidueSelection, ...]]:
    """Group coordinate residues by deposited chain, keeping coordinate order.

    EDSTATS restarts NR at 1 for each chain part it reports, so the ordinal
    locates a residue only within its own chain. CP carries the deposited chain
    and is the field that matches ``coordinate_author_key``; CI cannot be used
    here because EDSTATS reports it as ``0`` for ordered waters.
    """
    grouped: dict[str, list[ResidueSelection]] = {}
    for residue in structure.residues:
        grouped.setdefault(residue.coordinate_author_key[1], []).append(residue)
    return {chain: tuple(residues) for chain, residues in grouped.items()}


def _resolve_coordinate_residues(
    chain_residues: Sequence[ResidueSelection],
    matched_residues: tuple[ResidueSelection, ...],
    row_number: int,
    line_number: int,
    coordinate_key: tuple[str, str, str],
) -> tuple[ResidueSelection, ...]:
    """Use NR to reduce a repeated author identity to one coordinate residue.

    ``chain_residues`` are the coordinate residues of the row's own chain part,
    in coordinate order, because NR is numbered within a chain rather than
    across the model.
    """
    matched_residues = tuple(matched_residues)
    if len(matched_residues) <= 1:
        return matched_residues

    index = row_number - 1
    if index < len(chain_residues):
        row_residue_key = chain_residues[index].key
        for residue in matched_residues:
            if residue.key == row_residue_key:
                return (residue,)

    resname, chain, resnum = coordinate_key
    raise ValueError(
        f"EDSTATS NR {row_number} on row {line_number} does not resolve "
        f"the {len(matched_residues)} coordinate residues matching "
        f"{resname}/{chain or '_'}/{resnum}"
    )


def density_observation_id(
    pdb_id: str, fields: Sequence[str], indices: Mapping[str, int]
) -> str:
    """Return a stable identifier for one residue-level EDSTATS observation.

    EDSTATS reports one observation per coordinate residue, which Alchemy can
    expand into several metal-site rows, so the identifier derives from the
    EDSTATS row rather than from an individual metal atom. ``NR`` disambiguates
    repeated author residue identifiers within one chain, which is the scope it
    is numbered over; the chain itself keeps rows from different chains apart.
    """
    chain = fields[indices["CI"]] or "_"
    return "/".join(
        (
            str(pdb_id).lower(),
            f"model={fields[indices['MN']]}",
            f"chain={chain}",
            f"residue={fields[indices['RN']]}",
            f"component={fields[indices['RT']]}",
            f"edstats_row={fields[indices['NR']]}",
        )
    )


def _density_row(
    pdb_id: str,
    fields: Sequence[str],
    indices: Mapping[str, int],
    mapping_status: str,
    *,
    category: str,
    resname: str,
    site: AtomSite | None,
    residue_key: tuple[int, int, int] | None,
    shared_site_count: int,
) -> MetalStatsRow:
    """Build one site-level row from an extracted EDSTATS residue row.

    ``site`` is ``None`` for a cofactor residue with no selected metal site,
    which is also what makes the row's status ``no_selected_metal``.
    """
    return MetalStatsRow(
        pdb_id=pdb_id,
        category=category,
        resname=resname,
        chain=fields[indices["CI"]],
        resnum=fields[indices["RN"]],
        fields=tuple(fields),
        density_observation_id=density_observation_id(pdb_id, fields, indices),
        density_scope=(
            "cofactor_residue" if category == "cofactor" else "metal_residue"
        ),
        density_shared_site_count=shared_site_count,
        density_is_shared=shared_site_count > 1,
        coordinate_mapping_status=mapping_status,
        selected_metal_site_status=(
            "selected" if site is not None else "no_selected_metal"
        ),
        site=site,
        site_key=None if site is None else site.source_key,
        residue_key=residue_key,
    )


@dataclass(frozen=True)
class _ResolvedEdstatsRow:
    """One normalized row shared by duplicate, join and completeness checks."""

    fields: list[str]
    indices: Mapping[str, int]
    model: int
    row_number: int
    chain_part: str
    coordinate_key: tuple[str, str, str]
    matched_residues: tuple[ResidueSelection, ...]
    mapping_status: str


def _resolve_edstats_row(
    fields: Sequence[str],
    header: Sequence[str],
    indices: Mapping[str, int],
    line_number: int,
    structure: StructureContext,
    residues_by_chain_part: Mapping[str, tuple[ResidueSelection, ...]],
    grid_point_overflow_columns: set[str],
) -> _ResolvedEdstatsRow | None:
    """Return ``None`` for a valid row belonging to a discarded conformer."""
    normalized = normalize_edstats_row(fields, header, indices)
    if len(normalized) == len(header):
        for name in EDSTATS_GRID_POINT_COLUMNS:
            index = indices[name]
            if normalized[index] == EDSTATS_FIXED_WIDTH_OVERFLOW:
                normalized[index] = EDSTATS_NULL_VALUE
                grid_point_overflow_columns.add(name)
    row_model = validate_edstats_row(normalized, header, indices, line_number)
    row_number = _validated_edstats_row_number(normalized, indices, line_number)
    if row_model != structure.model_analyzed:
        raise ValueError(
            f"EDSTATS returned model {row_model}, but Alchemy's "
            f"model policy selected model {structure.model_analyzed}"
        )
    chain_part = normalized[indices["CP"]]
    try:
        normalized[indices["RN"]] = canonical_pdb_residue_id(normalized[indices["RN"]])
    except ValueError as exc:
        raise ValueError(
            f"invalid EDSTATS RN residue identifier on row {line_number}: "
            f"{normalized[indices['RN']]!r}"
        ) from exc

    coordinate_resname = normalized[indices["RT"]]
    _group_chain, row_altloc = _edstats_chain_and_altloc(
        normalized[indices["CI"]], line_number
    )
    # CP is the actual PDB chain. CI is normally the same chain plus an
    # optional altloc, but EDSTATS deliberately rewrites CI to ``0`` for every
    # ordered water so it can estimate that group's map-noise distribution.
    # Joining coordinates through CI therefore aliases equal-numbered waters
    # from different chains, especially when one real chain is itself ``0``.
    # Everything downstream addresses the deposited chain; the chosen altloc
    # remains explicit on the selected coordinate site's fields.
    chain = chain_part
    normalized[indices["CI"]] = chain
    resnum = normalized[indices["RN"]]
    coordinate_key = (coordinate_resname, chain, resnum)
    matched_residues = structure.residues_for_coordinate_author(*coordinate_key)
    chain_residues = residues_by_chain_part.get(chain_part, ())
    matched_residues = _resolve_coordinate_residues(
        chain_residues,
        matched_residues,
        row_number,
        line_number,
        coordinate_key,
    )
    if not _row_matches_selected_altloc(
        row_altloc,
        matched_residues,
        chain_residues,
        row_number,
        line_number,
    ):
        return None
    return _ResolvedEdstatsRow(
        fields=normalized,
        indices=indices,
        model=row_model,
        row_number=row_number,
        chain_part=chain_part,
        coordinate_key=coordinate_key,
        matched_residues=matched_residues,
        mapping_status=(
            "matched" if matched_residues else "coordinate_residue_not_found"
        ),
    )


def _append_density_rows(
    pdb_id: str,
    resolved: _ResolvedEdstatsRow,
    metals_upper: set[str],
    cofactors: frozenset[str],
    rows: list[MetalStatsRow],
) -> None:
    """Expand one residue observation into site or diagnostic rows."""
    coordinate_resname, _chain, _resnum = resolved.coordinate_key
    coordinate_name_is_cofactor = coordinate_resname in cofactors
    matched_cofactor_names: list[str] = []
    selected_sites: list[tuple[ResidueSelection, str, str, AtomSite]] = []
    for residue in resolved.matched_residues:
        resname = residue.residue_name
        category, metal_sites = classify_residue(residue, metals_upper, cofactors)
        if category == "cofactor":
            matched_cofactor_names.append(resname)
        if not category:
            continue
        for site in metal_sites:
            selected_sites.append((residue, resname, category, site))

    for residue, resname, category, site in selected_sites:
        output_fields = list(resolved.fields)
        output_fields[resolved.indices["RT"]] = resname
        if resolved.mapping_status == "matched":
            output_fields[resolved.indices["CI"]] = residue.chain_id
            output_fields[resolved.indices["RN"]] = residue.resnum
        rows.append(
            _density_row(
                pdb_id,
                output_fields,
                resolved.indices,
                resolved.mapping_status,
                category=category,
                resname=resname,
                site=site,
                residue_key=residue.key,
                shared_site_count=len(selected_sites),
            )
        )

    if not (coordinate_name_is_cofactor or matched_cofactor_names):
        return
    if selected_sites:
        return
    resname = (
        matched_cofactor_names[0] if matched_cofactor_names else coordinate_resname
    )
    output_fields = list(resolved.fields)
    output_fields[resolved.indices["RT"]] = resname
    matched_residue = (
        resolved.matched_residues[0] if len(resolved.matched_residues) == 1 else None
    )
    if matched_residue is not None:
        output_fields[resolved.indices["CI"]] = matched_residue.chain_id
        output_fields[resolved.indices["RN"]] = matched_residue.resnum
    rows.append(
        _density_row(
            pdb_id,
            output_fields,
            resolved.indices,
            resolved.mapping_status,
            category="cofactor",
            resname=resname,
            site=None,
            residue_key=None if matched_residue is None else matched_residue.key,
            shared_site_count=0,
        )
    )


def extract_metal_statistics(
    pdb_id: str,
    stats_out: str,
    metals_set: Iterable[str],
    cofactor_set: Iterable[str],
    structure: StructureContext,
    *,
    density_context_out: dict[str, Any] | None = None,
    warning_codes_out: list[str] | None = None,
) -> tuple[list[MetalStatsRow], list[str]]:
    """Parse an EDSTATS ``stats.out``, returning ``(rows, header)``.

    Cofactors match by CCD component name. A plain metal is a single-atom
    residue whose element, read from ``structure``, is in ``metals_set``:
    matching on the CCD id alone misclassifies RNA ``U`` and nitric oxide
    ``NO``, and misses metal ids like ``FE2``. Where the analysis PDB came from
    mmCIF the original component identifier is restored before matching, since
    a CCD id need not fit the three-character PDB residue field.

    Output is site-level: a multi-metal cofactor repeats its residue-level
    EDSTATS values once per selected metal site, sharing a
    ``density_observation_id`` and reporting the shared-site multiplicity so a
    density observation is counted only once. A cofactor with no matching
    coordinate residue or no selected metal site is retained once with
    ``site=None``, its row status distinguishing a failed identifier join from
    a matched cofactor that has no configured metal.

    When ``density_context_out`` is supplied, it is cleared immediately and
    populated only after the complete EDSTATS table passes validation. Its
    entry-level aggregates retain the non-target residue observations that are
    deliberately absent from the returned site-level rows.
    """
    metals_upper = {element.upper() for element in metals_set}
    cofactors = frozenset(cofactor_set)
    density_context = _DensityContextAccumulator()
    if density_context_out is not None:
        density_context_out.clear()

    rows: list[MetalStatsRow] = []
    schema: tuple[list[str], dict[str, int]] | None = None
    residue_row_count = 0
    observed_residues: Counter[tuple[str, str, str]] = Counter()
    observed_edstats_rows: set[tuple[int, str, int]] = set()
    residue_observations: dict[tuple[int, int, int], tuple[str, int]] = {}
    grid_point_overflow_columns: set[str] = set()
    residues_by_chain_part = _coordinate_residues_by_chain_part(structure)
    with open(stats_out, encoding="utf-8", errors="strict") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if schema is None:
                schema = (fields, validated_edstats_header(fields))
                continue

            if is_edstats_separator(fields):
                continue

            residue_row_count += 1
            header, indices = schema
            resolved = _resolve_edstats_row(
                fields,
                header,
                indices,
                line_number,
                structure,
                residues_by_chain_part,
                grid_point_overflow_columns,
            )
            if resolved is None:
                continue

            observation_key = (
                resolved.model,
                resolved.chain_part,
                resolved.row_number,
            )
            if observation_key in observed_edstats_rows:
                raise ValueError(
                    f"duplicate EDSTATS NR {resolved.row_number} for chain "
                    f"{resolved.chain_part or '_'} of model {resolved.model}"
                )
            observed_edstats_rows.add(observation_key)
            observed_residues[resolved.coordinate_key] += 1
            if resolved.matched_residues:
                residue_key = resolved.matched_residues[0].key
                previous = residue_observations.get(residue_key)
                if previous is not None:
                    previous_chain, previous_number = previous
                    coordinate_resname, chain, resnum = resolved.coordinate_key
                    raise ValueError(
                        f"EDSTATS NR {previous_number} of chain "
                        f"{previous_chain or '_'} and NR {resolved.row_number} "
                        f"of chain {resolved.chain_part or '_'} both map to "
                        "coordinate residue "
                        f"{coordinate_resname}/{chain or '_'}/{resnum}"
                    )
                residue_observations[residue_key] = (
                    resolved.chain_part,
                    resolved.row_number,
                )
            density_context.observe(resolved, metals_upper, cofactors)
            _append_density_rows(pdb_id, resolved, metals_upper, cofactors, rows)

    if schema is None:
        raise ValueError("EDSTATS output is empty")
    header, _indices = schema
    if residue_row_count == 0:
        raise ValueError("EDSTATS output contains no residue rows")

    missing_residues = sorted(
        (
            expected_edstats_residues(structure, metals_upper, cofactors)
            - observed_residues
        ).elements()
    )
    if missing_residues:
        preview = ", ".join(
            f"{resname}/{chain or '_'}/{resnum}"
            for resname, chain, resnum in missing_residues[:5]
        )
        suffix = (
            f" (and {len(missing_residues) - 5} more)"
            if len(missing_residues) > 5
            else ""
        )
        raise ValueError(
            "EDSTATS output is incomplete; missing expected residue"
            f"{'s' if len(missing_residues) != 1 else ''}: "
            f"{preview}{suffix}"
        )
    if density_context_out is not None:
        density_context_out.update(density_context.as_row(pdb_id))
    if grid_point_overflow_columns and warning_codes_out is not None:
        warning_codes_out[:] = list(
            dict.fromkeys(
                warning_codes_out + [WarningCode.EDSTATS_GRID_POINT_COUNT_OVERFLOW]
            )
        )
    return rows, header


# The density-sigma join reads Z-difference metrics back out of an extracted
# EDSTATS row, so it lives beside ``EDSTATS_COLUMNS``: a column-order change
# breaks both.
def sigma_index(
    stats_rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[tuple[Any, ...], Sequence[str]]]:
    """Index EDSTATS fields by site, with an unambiguous author-key fallback."""
    by_site: dict[tuple[Any, ...], Sequence[str]] = {}
    by_author: dict[tuple[Any, ...], Sequence[str]] = {}
    ambiguous_authors: set[tuple[Any, ...]] = set()
    for row in stats_rows:
        fields = row["fields"]
        site_key = row.get("site_key")
        if site_key is not None:
            by_site[tuple(site_key)] = fields
        author_key = (row["resname"], str(row["chain"]), str(row["resnum"]))
        if author_key in by_author:
            ambiguous_authors.add(author_key)
        else:
            by_author[author_key] = fields
    for author_key in ambiguous_authors:
        by_author.pop(author_key, None)
    return {"by_site": by_site, "by_author": by_author}


ZD_COLUMNS = ("ZDm", "ZD-m", "ZD+m")


def zd_indices(header: Sequence[str] | None) -> tuple[int, ...] | None:
    """Return column indices for ZDm/ZD-m/ZD+m, or ``None`` if any is absent."""
    if not header:
        return None
    try:
        return tuple(header.index(name) for name in ZD_COLUMNS)
    except ValueError:
        return None


def sigma_for(
    sig: Mapping[str, Mapping[tuple[Any, ...], Sequence[str]]],
    resname: str,
    chain: str,
    resnum: str,
    zd_idx: Sequence[int] | None,
    site_key: Sequence[Any] | None = None,
) -> tuple[float, float, float]:
    """Return the three density Z scores for a site or author identity."""
    fields: Sequence[str] | None = None
    if site_key is not None:
        fields = sig["by_site"].get(tuple(site_key))
    if fields is None:
        fields = sig["by_author"].get((resname, str(chain), str(resnum)))
    if fields is None or zd_idx is None:
        return NAN, NAN, NAN
    try:
        return (
            float(fields[zd_idx[0]]),
            float(fields[zd_idx[1]]),
            float(fields[zd_idx[2]]),
        )
    except (IndexError, ValueError):
        return NAN, NAN, NAN
