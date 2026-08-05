"""Extracting metal and cofactor real-space statistics from EDSTATS output.

EDSTATS' ``stats.out`` is a whitespace-separated table whose first non-empty
line is the column header.
"""

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence

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


def _is_edstats_separator(fields: list[str]) -> bool:
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


def _normalize_edstats_row(
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
            raise ValueError(
                f"EDSTATS row {line_number} pooled alternate conformers for a "
                f"residue whose selected conformer is {selected!r}; "
                "USEALT output is required"
            )
        return row_altloc == selected
    if row_altloc:
        raise ValueError(
            f"EDSTATS row {line_number} reports alternate conformer "
            f"{row_altloc!r} for a residue with no selected alternate"
        )
    return True


def _validated_edstats_header(fields: Sequence[str]) -> dict[str, int]:
    """Return column indices after validating the standard EDSTATS schema."""
    duplicates = sorted({name for name in fields if fields.count(name) > 1})
    if duplicates:
        raise ValueError(
            "EDSTATS header contains duplicate columns: " + ", ".join(duplicates)
        )

    missing = [name for name in EDSTATS_COLUMNS if name not in fields]
    unexpected = [name for name in fields if name not in EDSTATS_COLUMNS]
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError("invalid EDSTATS header: " + "; ".join(details))
    if tuple(fields) != EDSTATS_COLUMNS:
        raise ValueError("EDSTATS columns are not in the standard order")

    return {name: index for index, name in enumerate(fields)}


def _validate_edstats_row(
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


def _classify_residue(
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


def _expected_edstats_residues(
    structure: StructureContext, metals_upper: set[str], cofactor_set: Iterable[str]
) -> Counter[tuple[str, str, str]]:
    """Coordinate residue-key multiplicities Alchemy expects EDSTATS to report."""
    return Counter(
        residue.coordinate_author_key
        for residue in structure.residues
        if _classify_residue(residue, metals_upper, cofactor_set)[0]
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


def _density_observation_id(
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
    site: Optional[AtomSite],
    residue_key: Optional[tuple[int, int, int]],
    shared_site_count: int,
) -> dict[str, Any]:
    """Build one site-level row from an extracted EDSTATS residue row.

    ``site`` is ``None`` for a cofactor residue with no selected metal site,
    which is also what makes the row's status ``no_selected_metal``.
    """
    return {
        "pdbID": pdb_id,
        "category": category,
        "resname": resname,
        "chain": fields[indices["CI"]],
        "resnum": fields[indices["RN"]],
        "fields": fields,
        "density_observation_id": _density_observation_id(pdb_id, fields, indices),
        "density_scope": (
            "cofactor_residue" if category == "cofactor" else "metal_residue"
        ),
        "density_shared_site_count": shared_site_count,
        "density_is_shared": shared_site_count > 1,
        "coordinate_mapping_status": mapping_status,
        "selected_metal_site_status": (
            "selected" if site is not None else "no_selected_metal"
        ),
        "site": site,
        "site_key": None if site is None else site.source_key,
        "residue_key": residue_key,
    }


def extract_metal_statistics(
    pdb_id: str,
    stats_out: str,
    metals_set: Iterable[str],
    cofactor_set: Iterable[str],
    structure: StructureContext,
) -> tuple[list[dict[str, Any]], list[str]]:
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
    """

    metals_upper = {element.upper() for element in metals_set}

    rows: list[dict[str, Any]] = []
    schema: Optional[tuple[list[str], dict[str, int]]] = None
    residue_row_count = 0
    observed_residues: Counter[tuple[str, str, str]] = Counter()
    observed_edstats_rows: set[tuple[int, str, int]] = set()
    residue_observations: dict[tuple[int, int, int], tuple[str, int]] = {}
    residues_by_chain_part = _coordinate_residues_by_chain_part(structure)
    with open(stats_out, encoding="utf-8", errors="strict") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if schema is None:
                schema = (fields, _validated_edstats_header(fields))
                continue

            if _is_edstats_separator(fields):
                continue

            header, indices = schema

            fields = _normalize_edstats_row(fields, header, indices)
            row_model = _validate_edstats_row(fields, header, indices, line_number)
            row_number = _validated_edstats_row_number(fields, indices, line_number)
            if row_model != structure.model_analyzed:
                raise ValueError(
                    f"EDSTATS returned model {row_model}, but Alchemy's "
                    f"model policy selected model {structure.model_analyzed}"
                )
            chain_part = fields[indices["CP"]]
            try:
                fields[indices["RN"]] = canonical_pdb_residue_id(fields[indices["RN"]])
            except ValueError as exc:
                raise ValueError(
                    f"invalid EDSTATS RN residue identifier on row "
                    f"{line_number}: {fields[indices['RN']]!r}"
                ) from exc

            residue_row_count += 1
            coordinate_resname = fields[indices["RT"]]
            chain, row_altloc = _edstats_chain_and_altloc(
                fields[indices["CI"]], line_number
            )
            # Everything downstream addresses the deposited chain.  The chosen
            # altloc remains explicit on the selected coordinate site's fields.
            fields[indices["CI"]] = chain
            resnum = fields[indices["RN"]]
            coordinate_key = (coordinate_resname, chain, resnum)
            matched_residues = structure.residues_for_coordinate_author(
                coordinate_resname, chain, resnum
            )
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
                continue

            observation_key = (row_model, chain_part, row_number)
            if observation_key in observed_edstats_rows:
                raise ValueError(
                    f"duplicate EDSTATS NR {row_number} for chain "
                    f"{chain_part or '_'} of model {row_model}"
                )
            observed_edstats_rows.add(observation_key)
            observed_residues[coordinate_key] += 1
            if not matched_residues:
                mapping_status = "coordinate_residue_not_found"
            else:
                mapping_status = "matched"
                residue_key = matched_residues[0].key
                previous = residue_observations.get(residue_key)
                if previous is not None:
                    previous_chain, previous_number = previous
                    raise ValueError(
                        f"EDSTATS NR {previous_number} of chain "
                        f"{previous_chain or '_'} and NR {row_number} of chain "
                        f"{chain_part or '_'} both map to coordinate residue "
                        f"{coordinate_resname}/{chain or '_'}/{resnum}"
                    )
                residue_observations[residue_key] = (chain_part, row_number)

            coordinate_name_is_cofactor = coordinate_resname in cofactor_set
            matched_cofactor_names: list[str] = []
            selected_sites: list[tuple[ResidueSelection, str, str, AtomSite]] = []
            for residue in matched_residues:
                resname = residue.residue_name
                category, metal_sites = _classify_residue(
                    residue, metals_upper, cofactor_set
                )
                if category == "cofactor":
                    matched_cofactor_names.append(resname)
                if not category:
                    continue
                for site in metal_sites:
                    selected_sites.append((residue, resname, category, site))

            density_shared_site_count = len(selected_sites)
            for residue, resname, category, site in selected_sites:
                output_fields = list(fields)
                output_fields[indices["RT"]] = resname
                if mapping_status == "matched":
                    output_fields[indices["CI"]] = residue.chain_id
                    output_fields[indices["RN"]] = residue.resnum
                rows.append(
                    _density_row(
                        pdb_id,
                        output_fields,
                        indices,
                        mapping_status,
                        category=category,
                        resname=resname,
                        site=site,
                        residue_key=residue.key,
                        shared_site_count=density_shared_site_count,
                    )
                )

            if (
                coordinate_name_is_cofactor or matched_cofactor_names
            ) and not selected_sites:
                resname = (
                    matched_cofactor_names[0]
                    if matched_cofactor_names
                    else coordinate_resname
                )
                output_fields = list(fields)
                output_fields[indices["RT"]] = resname
                matched_residue = (
                    matched_residues[0] if len(matched_residues) == 1 else None
                )
                if matched_residue is not None:
                    output_fields[indices["CI"]] = matched_residue.chain_id
                    output_fields[indices["RN"]] = matched_residue.resnum
                rows.append(
                    _density_row(
                        pdb_id,
                        output_fields,
                        indices,
                        mapping_status,
                        category="cofactor",
                        resname=resname,
                        site=None,
                        residue_key=(
                            None if matched_residue is None else matched_residue.key
                        ),
                        shared_site_count=0,
                    )
                )

    if schema is None:
        raise ValueError("EDSTATS output is empty")
    header, _indices = schema
    if residue_row_count == 0:
        raise ValueError("EDSTATS output contains no residue rows")

    missing_residues = sorted(
        (
            _expected_edstats_residues(structure, metals_upper, cofactor_set)
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
    return rows, header


# The density-sigma join reads Z-difference metrics back out of an extracted
# EDSTATS row, so it lives beside ``EDSTATS_COLUMNS``: a column-order change
# breaks both.
def _sigma_index(
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


def _zd_indices(header: Optional[Sequence[str]]) -> Optional[tuple[int, ...]]:
    """Return column indices for ZDm/ZD-m/ZD+m, or ``None`` if any is absent."""
    if not header:
        return None
    try:
        return tuple(header.index(name) for name in ZD_COLUMNS)
    except ValueError:
        return None


def _sigma_for(
    sig: Mapping[str, Mapping[tuple[Any, ...], Sequence[str]]],
    resname: str,
    chain: str,
    resnum: str,
    zd_idx: Optional[Sequence[int]],
    site_key: Optional[Sequence[Any]] = None,
) -> tuple[float, float, float]:
    fields: Optional[Sequence[str]] = None
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
