# Analysis v2: extract metal / metallocofactor real-space stats from edstats output.
#
# edstats `stats.out` is a whitespace table: the first non-empty line is the column
# header, and each data line begins with a residue/component name. Column layout
# (0-indexed): 0 = residue name (RT), 1 = chain (CI), 2 = residue number (RN).
#
# `extract_metal_statistics` returns structured rows for metal ions and metal-containing
# cofactors; `main.py` aggregates these across many structures.

import math
import os
from typing import Any, Iterable, Optional

from structure_analysis import NAN, canonical_pdb_residue_id


# Single source for the bundled reference-data directory. Every file under it
# is named relative to this constant so the location is defined once; the
# directory travels with the checkout, which is how Alchemy is distributed.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

COFACTOR_CATALOG_PATH = os.path.join(DATA_DIR, "metallocofactors_id.txt")


def load_cofactor_ids() -> set[str]:
    """Load component IDs from Alchemy's fixed bundled catalog."""
    with open(COFACTOR_CATALOG_PATH, encoding="utf-8") as handle:
        cofactor_ids = {
            line.partition("\t")[0].strip() for line in handle if line.strip()
        }
    if not cofactor_ids:
        raise ValueError("bundled metallocofactor catalog is empty")
    return cofactor_ids


# EDSTATS 1.0.9 standard residue-table schema. The twelve metrics are repeated
# for main-chain, side-chain, and all atoms. ``n/a`` is EDSTATS' documented
# null marker when a statistic cannot be calculated for an atom group.
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
EDSTATS_NULL_VALUE = "n/a"
EDSTATS_MISSING_CHAIN_IDS = frozenset(("", ".", "?", "_"))


def _is_edstats_separator(fields):
    """Whether split fields are EDSTATS' synthetic model separator row.

    For a MODEL/ENDMDL-wrapped XYZIN, EDSTATS 1.0.9 emits a logical row with
    blank residue name, residue number, and chain-position fields. Whitespace
    splitting collapses those blanks, producing 39 fields rather than the
    42-column residue schema: ``_``, 36 ``n/a`` metrics, model, and row number.
    Older captured output may contain only the ``_`` marker. Recognize both
    forms semantically while leaving every other malformed row to validation.
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


def _normalize_edstats_row(fields, header, indices):
    """Restore and normalize EDSTATS' valid blank-chain representation.

    EDSTATS leaves the trailing chain field (CP) empty for a blank-chain
    residue, so whitespace splitting removes it and produces 41 fields. CP is
    the only field EDSTATS can legitimately omit, so restore it for that
    unambiguous shape: the row is exactly one field short, and its final two
    tokens are the integer MN and NR values. All other short rows remain short
    and are rejected by normal row validation.

    The leading CI field cannot gate this restoration. CI is EDSTATS' own group
    label rather than the deposited chain identifier -- ordered waters are
    reported as chain ``0`` whatever their actual chain, while CP carries the
    real one. A blank-chain entry therefore yields CI ``0`` for every water and
    CI ``_`` for every other residue, with CP omitted from both. Both chain
    fields then use an empty string as their canonical missing value.
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


def _validated_edstats_header(fields):
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


def _validate_edstats_row(fields, header, indices, line_number):
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


def _classify_residue(residue, metals_upper, cofactor_set):
    """Return ``(category, metal_sites)`` for one coordinate residue.

    ``category`` is ``"cofactor"``, ``"metal"``, or ``""`` when the residue is
    neither. The emitted rows and the EDSTATS completeness check below both
    derive from this single rule, so the set of sites Alchemy demands EDSTATS
    report cannot drift away from the set it actually emits.
    """
    metal_sites = [
        atom
        for atom in residue.contact_atoms
        if atom.element_known and atom.element in metals_upper
    ]
    if residue.residue_name in cofactor_set:
        return "cofactor", metal_sites
    if residue.chemical_atom_site_count == 1 and len(metal_sites) == 1:
        return "metal", metal_sites
    return "", metal_sites


def _expected_edstats_residues(structure, metals_upper, cofactor_set):
    """Coordinate residue keys for sites Alchemy expects EDSTATS to report."""
    return {
        residue.coordinate_author_key
        for residue in structure.residues
        if _classify_residue(residue, metals_upper, cofactor_set)[0]
    }


def _density_observation_id(pdb_id, fields, indices):
    """Return a stable identifier for one residue-level EDSTATS observation.

    EDSTATS reports one density observation per coordinate residue. Alchemy can
    expand that observation into several metal-site rows for a multi-metal
    cofactor, so the identifier deliberately derives from the EDSTATS row and
    not from an individual metal atom. ``NR`` disambiguates otherwise repeated
    author residue identifiers within the selected model.
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


def extract_metal_statistics(
    pdb_id: str,
    stats_out: str,
    metals_set: Iterable[str],
    cofactor_set: Iterable[str],
    structure: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse an edstats stats.out file, returning (rows, header).
    `structure` is the shared first-model Gemmi context used by bond analysis.

    Cofactors are matched by CCD component name (fields[0]) against
    cofactor_set, as before. Plain metals are matched by the residue's
    actual atom element(s), read from `structure` -- a single-atom residue
    whose element is in metals_set is classified as a metal. This avoids
    misclassifying components whose CCD id happens to look like an element
    symbol (RNA "U", nitric oxide "NO") and catches metal-ion CCD ids that
    don't themselves match an element string (e.g. "FE2").

    EDSTATS matching uses the legacy-PDB residue name. When the analysis PDB was
    converted from mmCIF, Alchemy restores the original component identifier
    before matching the cofactor catalog and writing the result. This supports
    CCD identifiers that cannot fit in the three-character PDB residue field.

    Output is site-level: a multi-metal cofactor repeats its residue-level
    EDSTATS values once per selected metal site. Repeated rows share a
    ``density_observation_id`` and report their shared-site multiplicity so
    downstream analyses can count the density observation only once. Each row
    also carries ``site`` and ``site_key`` internally so downstream contact
    summaries cannot collide for multiple metals or duplicate author residue
    identifiers.

    A cofactor row that has no matching coordinate residue or no selected metal
    site is retained once with ``site=None``. Machine-readable row status fields
    distinguish an identifier-join failure from a matched cofactor that simply
    has no selected configured metal. This preserves the residue-level EDSTATS
    observation without pretending that a metal site was available for geometry
    analysis.
    """

    metals_upper = {element.upper() for element in metals_set}

    rows = []
    # The column names and their positions are one value, not two: they are
    # produced together by header validation and are meaningless apart. Keeping
    # them in a single Optional makes the "not yet seen a header" state a
    # property of one variable, so every use below is reachable only after it
    # has been set -- which a reader and a type checker can both follow.
    schema: Optional[tuple[list[str], dict[str, int]]] = None
    residue_row_count = 0
    observed_residues = set()
    with open(stats_out, encoding="utf-8", errors="strict") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if schema is None:
                # first non-empty line is the edstats column header
                schema = (fields, _validated_edstats_header(fields))
                continue

            if _is_edstats_separator(fields):
                continue

            header, indices = schema

            fields = _normalize_edstats_row(fields, header, indices)
            row_model = _validate_edstats_row(fields, header, indices, line_number)
            if row_model != structure.model_analyzed:
                raise ValueError(
                    f"EDSTATS returned model {row_model}, but Alchemy's "
                    f"model policy selected model {structure.model_analyzed}"
                )
            try:
                fields[indices["RN"]] = canonical_pdb_residue_id(fields[indices["RN"]])
            except ValueError as exc:
                raise ValueError(
                    f"invalid EDSTATS RN residue identifier on row "
                    f"{line_number}: {fields[indices['RN']]!r}"
                ) from exc

            residue_row_count += 1
            coordinate_resname = fields[indices["RT"]]
            chain = fields[indices["CI"]]
            resnum = fields[indices["RN"]]
            observed_residues.add((coordinate_resname, chain, resnum))
            matched_residues = structure.residues_for_coordinate_author(
                coordinate_resname, chain, resnum
            )
            if not matched_residues:
                mapping_status = "coordinate_residue_not_found"
            elif len(matched_residues) == 1:
                mapping_status = "matched"
            else:
                mapping_status = "multiple_coordinate_residues"

            coordinate_name_is_cofactor = coordinate_resname in cofactor_set
            matched_cofactor_names = []
            selected_sites = []
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
            density_is_shared = density_shared_site_count > 1
            for residue, resname, category, site in selected_sites:
                output_fields = list(fields)
                output_fields[indices["RT"]] = resname
                if mapping_status == "matched":
                    output_fields[indices["CI"]] = residue.chain_id
                    output_fields[indices["RN"]] = residue.resnum
                density_observation_id = _density_observation_id(
                    pdb_id, output_fields, indices
                )
                output_chain = output_fields[indices["CI"]]
                output_resnum = output_fields[indices["RN"]]
                rows.append(
                    {
                        "pdbID": pdb_id,
                        "category": category,
                        "resname": resname,
                        "chain": output_chain,
                        "resnum": output_resnum,
                        "fields": output_fields,
                        "density_observation_id": density_observation_id,
                        "density_scope": (
                            "cofactor_residue"
                            if category == "cofactor"
                            else "metal_residue"
                        ),
                        "density_shared_site_count": density_shared_site_count,
                        "density_is_shared": density_is_shared,
                        "coordinate_mapping_status": mapping_status,
                        "selected_metal_site_status": "selected",
                        "site": site,
                        "site_key": site.source_key,
                        "residue_key": residue.key,
                    }
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
                output_chain = output_fields[indices["CI"]]
                output_resnum = output_fields[indices["RN"]]
                density_observation_id = _density_observation_id(
                    pdb_id, output_fields, indices
                )
                rows.append(
                    {
                        "pdbID": pdb_id,
                        "category": "cofactor",
                        "resname": resname,
                        "chain": output_chain,
                        "resnum": output_resnum,
                        "fields": output_fields,
                        "density_observation_id": density_observation_id,
                        "density_scope": "cofactor_residue",
                        "density_shared_site_count": 0,
                        "density_is_shared": False,
                        "coordinate_mapping_status": mapping_status,
                        "selected_metal_site_status": "no_selected_metal",
                        "site": None,
                        "site_key": None,
                        "residue_key": (
                            matched_residues[0].key
                            if len(matched_residues) == 1
                            else None
                        ),
                    }
                )

    if schema is None:
        raise ValueError("EDSTATS output is empty")
    header, _indices = schema
    if residue_row_count == 0:
        raise ValueError("EDSTATS output contains no residue rows")

    missing_residues = sorted(
        _expected_edstats_residues(structure, metals_upper, cofactor_set)
        - observed_residues
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


# --------------------------------------------------------------------------- #
# Density-sigma join
# --------------------------------------------------------------------------- #
# Reading the real-space Z-difference metrics back out of an extracted EDSTATS
# row. This is the same table ``EDSTATS_COLUMNS`` above describes, so the two
# live together: a column-order change breaks both, and having the reader in
# ``bond_analysis`` meant EDSTATS knowledge was split across two modules that
# had to agree without either one saying so.
def _sigma_index(stats_rows):
    """Map (resname, chain, resnum) -> edstats fields, for the sigma join."""
    return {
        (r["resname"], str(r["chain"]), str(r["resnum"])): r["fields"]
        for r in stats_rows
    }


ZD_COLUMNS = ("ZDm", "ZD-m", "ZD+m")


def _zd_indices(header):
    """Return column indices for ZDm/ZD-m/ZD+m, or None
    if the header is missing or doesn't contain all three names."""
    if not header:
        return None
    try:
        return tuple(header.index(name) for name in ZD_COLUMNS)
    except ValueError:
        return None


def _sigma_for(sig, resname, chain, resnum, zd_idx):
    fields = sig.get((resname, str(chain), str(resnum)))
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
