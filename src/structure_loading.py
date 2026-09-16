"""The steps ``structure_analysis.load_structure`` runs, in order.

Source provenance and raw fields are gathered for the analyzed model, joined
into ``AtomSite`` records, deduplicated into the atom inventory, and then
summarized as the occupancy, record, and symmetry audits and the residue
indexes that a ``StructureContext`` carries.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import gemmi

from codes import OccupancyStatus, WarningCode
from conformer_selection import select_residue, site_is_better
from pdb_records import (
    PDB_FORMAT,
    RawOccupancy,
    analysis_format_for_path,
    author_residue_number,
    blank_if_missing,
    element_for_atom,
    match_raw_occupancies,
    occupancy_for_atom,
    raw_pdb_occupancies,
)
from pdb_remarks import (
    SourceResidueIdentity,
    read_conversion_provenance,
)
from structure_model import (
    HYDROGEN_ELEMENTS,
    AtomRecordAudit,
    AtomSite,
    AuthorResidueKey,
    OccupancyValidation,
    ResidueIdentity,
    ResidueIndex,
    ResidueKey,
    ResidueSelection,
    SymmetryMetadata,
    position_distance,
    spacegroup_or_none,
)

#: Alchemy analyzes the first model of every coordinate file.
ANALYZED_MODEL_INDEX = 0
#: Duplicate records farther apart than this are a coordinate conflict.
DUPLICATE_ATOM_POSITION_TOLERANCE = 0.001

# DPI scales with sqrt(Ni), so relative atom-count error contributes half as
# much relative DPI error. This threshold limits that error to 0.1% while
# allowing routine two-decimal occupancy rounding, such as a sum of 1.01.
OVERFULL_OCCUPANCY_NI_FRACTION = 0.002

# Published ``SymmetryMetadata.search_failure_reason`` values.
MISSING_UNIT_CELL_REASON = "missing_or_invalid_unit_cell"
MISSING_SPACE_GROUP_REASON = "missing_or_invalid_space_group"
SYMMETRY_SETUP_FAILED_PREFIX = "symmetry_setup_failed:"
#: Names a strict-NCS operation whose deposited identifier is blank.
STRICT_NCS_ID_PREFIX = "strict_ncs_"

# Published ``OccupancyValidation.raw_mapping_failure_reason`` prefixes.
RAW_ATOM_COUNT_MISMATCH_REASON = "raw_atom_count_mismatch"
RAW_ATOM_IDENTITY_MISMATCH_REASON = "raw_atom_identity_mismatch"

#: ``AtomRecordAudit.element_validation_warning`` when any element is unknown.
UNKNOWN_ELEMENT_WARNING = "unknown_element_atoms"


@dataclass(frozen=True, slots=True)
class SourceModelData:
    """Raw fields Gemmi normalizes away and their first-model atom mapping."""

    analysis_format: str
    raw_first: tuple[RawOccupancy, ...]
    raw_matches: tuple[RawOccupancy | None, ...]
    source_residue_identities: Mapping[tuple[int, str, str, str], SourceResidueIdentity]
    defaulted_occupancy_count: int
    mapping_failed: bool
    mapping_reason: str
    legacy_identifiers_packed: bool


def source_model_data(path: str, model: gemmi.Model) -> SourceModelData:
    """Read the raw records and conversion provenance behind ``model``.

    For PDB input the deposited occupancy and element columns and any
    ``REMARK 950`` provenance are read from ``path`` and matched to Gemmi's
    atoms; mmCIF input has nothing to recover, so every match is ``None``.
    """
    analysis_format = analysis_format_for_path(path)
    raw_models: list[list[RawOccupancy]] = []
    raw_error = ""
    source_residue_identities: dict[
        tuple[int, str, str, str], SourceResidueIdentity
    ] = {}
    defaulted_occupancy_counts: dict[int, int] = {}
    if analysis_format == PDB_FORMAT:
        raw_models, raw_error = raw_pdb_occupancies(path)
        provenance = read_conversion_provenance(path)
        source_residue_identities = provenance.residue_mapping
        defaulted_occupancy_counts = provenance.defaulted_occupancy_counts

    legacy_identifiers_packed = any(
        identity.residue_number is not None
        and (
            identity.chain_id != coordinate_chain
            or f"{identity.residue_number}{identity.insertion_code}"
            != coordinate_resnum
        )
        for (
            _model_index,
            _coordinate_name,
            coordinate_chain,
            coordinate_resnum,
        ), identity in source_residue_identities.items()
    )
    raw_first = raw_models[0] if raw_models else []
    gemmi_atom_count = sum(
        1 for chain in model for residue in chain for _atom in residue
    )
    defaulted_occupancy_count = defaulted_occupancy_counts.get(ANALYZED_MODEL_INDEX, 0)
    if defaulted_occupancy_count not in (0, gemmi_atom_count):
        raise ValueError(
            "defaulted occupancy count does not match the analyzed model atom count"
        )

    if analysis_format == PDB_FORMAT:
        raw_matches, unmatched_gemmi, unmatched_raw = match_raw_occupancies(
            model, raw_first
        )
    else:
        raw_matches = [None] * gemmi_atom_count
        unmatched_gemmi = 0
        unmatched_raw = 0
    mapping_failed = analysis_format == PDB_FORMAT and bool(
        raw_error or unmatched_gemmi or unmatched_raw
    )
    mapping_reason = raw_error
    if mapping_failed and not mapping_reason:
        if len(raw_first) != gemmi_atom_count:
            mapping_reason = (
                f"{RAW_ATOM_COUNT_MISMATCH_REASON}:"
                f"{len(raw_first)}!={gemmi_atom_count};"
                f"unmatched_gemmi={unmatched_gemmi};unmatched_raw={unmatched_raw}"
            )
        else:
            mapping_reason = (
                f"{RAW_ATOM_IDENTITY_MISMATCH_REASON}:"
                f"unmatched_gemmi={unmatched_gemmi};unmatched_raw={unmatched_raw}"
            )
    return SourceModelData(
        analysis_format=analysis_format,
        raw_first=tuple(raw_first),
        raw_matches=tuple(raw_matches),
        source_residue_identities=source_residue_identities,
        defaulted_occupancy_count=defaulted_occupancy_count,
        mapping_failed=mapping_failed,
        mapping_reason=mapping_reason,
        legacy_identifiers_packed=legacy_identifiers_packed,
    )


def _residue_identity(
    model_id: str,
    chain_index: int,
    chain: gemmi.Chain,
    residue_index: int,
    residue: gemmi.Residue,
    source: SourceModelData,
) -> ResidueIdentity:
    """Join a Gemmi residue's coordinate identity with its source provenance.

    Without an embedded mapping the coordinate identity is the source identity.
    """
    coordinate_number = author_residue_number(residue)
    coordinate_insertion = blank_if_missing(residue.seqid.icode)
    coordinate_resnum = f"{coordinate_number}{coordinate_insertion}"
    coordinate_residue_name = str(residue.name)
    coordinate_chain_id = str(chain.name)
    source_identity = source.source_residue_identities.get(
        (
            ANALYZED_MODEL_INDEX,
            coordinate_residue_name,
            coordinate_chain_id,
            coordinate_resnum,
        ),
        SourceResidueIdentity(
            residue_name=coordinate_residue_name,
            chain_id=coordinate_chain_id,
            residue_number=coordinate_number,
            insertion_code=coordinate_insertion,
        ),
    )
    if source_identity.residue_number is None:
        source_number = coordinate_number
        source_insertion = coordinate_insertion
    else:
        source_number = source_identity.residue_number
        source_insertion = source_identity.insertion_code
    return ResidueIdentity(
        model_index=ANALYZED_MODEL_INDEX,
        model_id=model_id,
        chain_index=chain_index,
        chain_id=source_identity.chain_id,
        residue_index=residue_index,
        residue_name=source_identity.residue_name,
        coordinate_residue_name=coordinate_residue_name,
        residue_number=source_number,
        insertion_code=source_insertion,
        resnum=f"{source_number}{source_insertion}",
        coordinate_chain_id=coordinate_chain_id,
        coordinate_resnum=coordinate_resnum,
        source_polymer_position=source_identity.polymer_position,
        source_chain_index=source_identity.chain_index,
        source_residue_index=source_identity.residue_index,
    )


def build_atom_sites(
    model: gemmi.Model,
    model_id: str,
    source: SourceModelData,
) -> list[AtomSite]:
    """Return one ``AtomSite`` per Gemmi atom, in traversal order.

    Each site carries the deposited occupancy and element when a raw record
    matched; a PDB atom without a match is flagged as unmapped.
    """
    all_sites: list[AtomSite] = []
    gemmi_order = 0
    for chain_index, chain in enumerate(model):
        for residue_index, residue in enumerate(chain):
            identity = _residue_identity(
                model_id, chain_index, chain, residue_index, residue, source
            )
            is_water = bool(residue.is_water())
            for atom_index, atom in enumerate(residue):
                raw = (
                    source.raw_matches[gemmi_order]
                    if gemmi_order < len(source.raw_matches)
                    else None
                )
                occupancy, occupancy_valid, occupancy_status = occupancy_for_atom(
                    atom, raw
                )
                if source.analysis_format == PDB_FORMAT and raw is None:
                    occupancy_valid = False
                    occupancy_status = OccupancyStatus.RAW_MAPPING_FAILED
                source_order = (
                    raw.source_order
                    if raw is not None
                    else len(source.raw_first) + gemmi_order
                )
                element, element_known = element_for_atom(
                    atom, source.analysis_format, raw
                )
                all_sites.append(
                    AtomSite(
                        identity=identity,
                        atom_index=atom_index,
                        source_order=source_order,
                        atom_name=str(atom.name).strip(),
                        altloc=blank_if_missing(atom.altloc),
                        element=element,
                        element_known=element_known,
                        occupancy=occupancy,
                        occupancy_valid=occupancy_valid,
                        occupancy_status=occupancy_status,
                        x=float(atom.pos.x),
                        y=float(atom.pos.y),
                        z=float(atom.pos.z),
                        is_water=is_water,
                        is_hydrogen=(element_known and element in HYDROGEN_ELEMENTS),
                        gemmi_atom=atom,
                    )
                )
                gemmi_order += 1
    return all_sites


def _overfull_occupancy_summary(
    atoms: Iterable[AtomSite],
) -> tuple[int, float, frozenset[tuple[object, ...]]]:
    """Record sites whose alternate occupancies sum above one and their excess.

    Keep site identities so callers can locate the excess within a metal site.
    """
    by_chemical_site: dict[tuple[object, ...], list[AtomSite]] = defaultdict(list)
    for atom in atoms:
        by_chemical_site[atom.chemical_site_identity].append(atom)

    count = 0
    excess = 0.0
    identities: list[tuple[object, ...]] = []
    for identity, alternates in by_chemical_site.items():
        if len(alternates) <= 1:
            continue
        if not all(atom.occupancy_valid for atom in alternates):
            continue
        total = math.fsum(atom.occupancy for atom in alternates)
        if total > 1.0:
            count += 1
            excess += total - 1.0
            identities.append(identity)
    return count, excess, frozenset(identities)


def _occupancy_weighted_atom_count(atoms: Iterable[AtomSite]) -> float:
    """Count Ni without applying occupancy-validation flags.

    The raw count is needed to assess how much overfull sites affect DPI.
    """
    return math.fsum(
        atom.occupancy
        for atom in atoms
        if not atom.is_hydrogen and atom.occupancy_valid
    )


def _spatial_model(model: gemmi.Model) -> gemmi.Model:
    """Return ``model`` without non-finite atom positions, cloning only if needed.

    Gemmi bins every atom during NeighborSearch construction, including atoms
    Alchemy never adds explicitly, so non-finite coordinates must be absent.
    """
    invalid_positions = [
        (chain_index, residue_index, atom_index)
        for chain_index, chain in enumerate(model)
        for residue_index, residue in enumerate(chain)
        for atom_index, atom in enumerate(residue)
        if not all(
            math.isfinite(value) for value in (atom.pos.x, atom.pos.y, atom.pos.z)
        )
    ]
    if not invalid_positions:
        return model
    spatial_model = model.clone()
    # Deleting from the end keeps the earlier indices valid.
    for chain_index, residue_index, atom_index in reversed(invalid_positions):
        del spatial_model[chain_index][residue_index][atom_index]
    return spatial_model


@dataclass(frozen=True, slots=True)
class AtomInventory:
    """The deduplicated atom sets, selected residues, and the counts behind them."""

    source_atoms: tuple[AtomSite, ...]
    contact_atoms: tuple[AtomSite, ...]
    residues: tuple[ResidueSelection, ...]
    spatial_model: gemmi.Model
    missing_occupancy_count: int
    invalid_occupancy_count: int
    zero_occupancy_count: int
    duplicate_count: int
    coordinate_conflict_count: int
    overfull_site_count: int
    overfull_excess: float
    overfull_site_keys: frozenset[tuple[object, ...]]
    malformed_name_count: int
    unknown_element_count: int
    non_finite_coordinate_count: int


def prepare_atom_inventory(
    model: gemmi.Model, all_sites: Sequence[AtomSite]
) -> AtomInventory:
    """Collapse duplicate records, select conformers, and count what was found."""
    # Counted before exact duplicates are collapsed, so a valid twin cannot
    # hide an invalid deposited record.
    missing_count = sum(
        atom.occupancy_status == OccupancyStatus.MISSING for atom in all_sites
    )
    invalid_count = sum(
        not atom.occupancy_valid and atom.occupancy_status != OccupancyStatus.MISSING
        for atom in all_sites
    )
    zero_count = sum(
        atom.occupancy_valid and atom.occupancy == 0.0 for atom in all_sites
    )

    dedup: dict[tuple[object, ...], AtomSite] = {}
    duplicate_count = 0
    coordinate_conflicts = 0
    for site in all_sites:
        current = dedup.get(site.exact_identity)
        if current is None:
            dedup[site.exact_identity] = site
            continue
        duplicate_count += 1
        if position_distance(site.xyz, current.xyz) > DUPLICATE_ATOM_POSITION_TOLERANCE:
            coordinate_conflicts += 1
        if site_is_better(site, current):
            dedup[site.exact_identity] = site
    source_atoms = tuple(sorted(dedup.values(), key=lambda site: site.source_order))
    overfull_site_count, overfull_excess, overfull_site_keys = (
        _overfull_occupancy_summary(source_atoms)
    )

    residue_groups: dict[ResidueKey, list[AtomSite]] = defaultdict(list)
    for site in source_atoms:
        residue_groups[site.residue_key].append(site)
    residues = tuple(
        select_residue(residue_groups[key]) for key in sorted(residue_groups)
    )
    contact_atoms = tuple(
        atom for residue in residues for atom in residue.contact_atoms
    )
    malformed_names = sum(
        residue.malformed_duplicate_atom_name_count
        + residue.selected_over_blank_duplicate_count
        for residue in residues
    )
    unknown_count = sum(not atom.element_known for atom in source_atoms)
    non_finite_count = sum(not atom.coordinates_valid for atom in source_atoms)

    return AtomInventory(
        source_atoms=source_atoms,
        contact_atoms=contact_atoms,
        residues=residues,
        spatial_model=_spatial_model(model),
        missing_occupancy_count=missing_count,
        invalid_occupancy_count=invalid_count,
        zero_occupancy_count=zero_count,
        duplicate_count=duplicate_count,
        coordinate_conflict_count=coordinate_conflicts,
        overfull_site_count=overfull_site_count,
        overfull_excess=overfull_excess,
        overfull_site_keys=overfull_site_keys,
        malformed_name_count=malformed_names,
        unknown_element_count=unknown_count,
        non_finite_coordinate_count=non_finite_count,
    )


def symmetry_metadata(structure: gemmi.Structure) -> SymmetryMetadata:
    """Prepare cell images and record why symmetry searches are unavailable."""
    strict_ncs_ids = tuple(
        str(operation.id).strip() or f"{STRICT_NCS_ID_PREFIX}{index}"
        for index, operation in enumerate(
            (operation for operation in structure.ncs if not bool(operation.given)),
            start=1,
        )
    )

    def unavailable(reason: str) -> SymmetryMetadata:
        return SymmetryMetadata(False, reason, 0, strict_ncs_ids)

    try:
        cell = structure.cell
        if not cell.is_crystal() or cell.volume <= 0:
            return unavailable(MISSING_UNIT_CELL_REASON)
        spacegroup = spacegroup_or_none(structure)
        if spacegroup is None:
            return unavailable(MISSING_SPACE_GROUP_REASON)
        operation_count = len(list(spacegroup.operations()))
        if operation_count <= 0:
            return unavailable(MISSING_SPACE_GROUP_REASON)
        structure.setup_cell_images()
        return SymmetryMetadata(True, "", operation_count, strict_ncs_ids)
    except Exception as exc:  # malformed symmetry must fail one entry, not the batch
        return unavailable(f"{SYMMETRY_SETUP_FAILED_PREFIX}{type(exc).__name__}")


@dataclass(frozen=True, slots=True)
class ResidueIndexes:
    """Selected residues grouped by each author identity a caller may hold."""

    #: Source identity plus, where it differs, the coordinate identity.
    by_author: ResidueIndex
    by_source_author: ResidueIndex
    by_coordinate_author: ResidueIndex


def residue_indexes(residues: Iterable[ResidueSelection]) -> ResidueIndexes:
    """Index selected residues by source, coordinate, and either identity."""
    by_author: dict[AuthorResidueKey, list[ResidueSelection]] = defaultdict(list)
    by_source_author: dict[AuthorResidueKey, list[ResidueSelection]] = defaultdict(list)
    by_coordinate_author: dict[AuthorResidueKey, list[ResidueSelection]] = defaultdict(
        list
    )
    for selection in residues:
        by_source_author[selection.author_key].append(selection)
        by_coordinate_author[selection.coordinate_author_key].append(selection)
        by_author[selection.author_key].append(selection)
        if selection.coordinate_author_key != selection.author_key:
            by_author[selection.coordinate_author_key].append(selection)
    return ResidueIndexes(
        by_author={key: tuple(value) for key, value in by_author.items()},
        by_source_author={key: tuple(value) for key, value in by_source_author.items()},
        by_coordinate_author={
            key: tuple(value) for key, value in by_coordinate_author.items()
        },
    )


def _occupancy_validation_failed(
    inventory: AtomInventory, source: SourceModelData
) -> bool:
    """Whether the deposited occupancies are unfit for an occupancy-weighted Ni."""
    # Unreadable occupancy makes Ni unknown; a known excess is judged relative to Ni.
    countable_ni = _occupancy_weighted_atom_count(inventory.source_atoms)
    if inventory.overfull_excess <= 0.0:
        overfull_invalidates_dpi = False
    elif countable_ni > 0.0:
        overfull_invalidates_dpi = (
            inventory.overfull_excess / countable_ni > OVERFULL_OCCUPANCY_NI_FRACTION
        )
    else:
        overfull_invalidates_dpi = True
    return bool(
        inventory.missing_occupancy_count
        or inventory.invalid_occupancy_count
        or overfull_invalidates_dpi
        or source.mapping_failed
    )


def occupancy_validation(
    inventory: AtomInventory, source: SourceModelData
) -> OccupancyValidation:
    """Summarize the deposited occupancies and whether they disqualify Ni."""
    return OccupancyValidation(
        validation_failed=_occupancy_validation_failed(inventory, source),
        missing_count=inventory.missing_occupancy_count,
        invalid_count=inventory.invalid_occupancy_count,
        overfull_site_count=inventory.overfull_site_count,
        overfull_excess=round(inventory.overfull_excess, 6),
        overfull_site_keys=inventory.overfull_site_keys,
        defaulted_atom_count=source.defaulted_occupancy_count,
        zero_atom_count=inventory.zero_occupancy_count,
        raw_mapping_failed=source.mapping_failed,
        raw_mapping_failure_reason=source.mapping_reason,
    )


def record_audit(inventory: AtomInventory) -> AtomRecordAudit:
    """Report the record defects the inventory found."""
    return AtomRecordAudit(
        duplicate_records_present=inventory.duplicate_count > 0,
        duplicate_record_count=inventory.duplicate_count,
        coordinate_conflict_count=inventory.coordinate_conflict_count,
        malformed_duplicate_atom_name_count=inventory.malformed_name_count,
        unknown_element_atom_count=inventory.unknown_element_count,
        element_validation_warning=(
            UNKNOWN_ELEMENT_WARNING if inventory.unknown_element_count else ""
        ),
        non_finite_coordinate_atom_count=inventory.non_finite_coordinate_count,
    )


def warning_codes(
    inventory: AtomInventory, source: SourceModelData, input_model_count: int
) -> tuple[str, ...]:
    """Return the structure-level warning codes in their published order."""
    warnings: list[str] = []
    if input_model_count > 1:
        warnings.append(WarningCode.MULTI_MODEL_STRUCTURE)
    if inventory.duplicate_count:
        warnings.append(WarningCode.DUPLICATE_ATOM_RECORDS)
    if inventory.coordinate_conflict_count:
        warnings.append(WarningCode.DUPLICATE_ATOM_COORDINATE_CONFLICT)
    if inventory.malformed_name_count:
        warnings.append(WarningCode.MALFORMED_DUPLICATE_ATOM_NAMES)
    if any(residue.altloc_selection_fallback for residue in inventory.residues):
        warnings.append(WarningCode.ALTLOC_SELECTION_FALLBACK)
    if inventory.unknown_element_count:
        warnings.append(WarningCode.UNKNOWN_ELEMENTS)
    if inventory.non_finite_coordinate_count:
        warnings.append(WarningCode.NON_FINITE_COORDINATES)
    if inventory.zero_occupancy_count:
        warnings.append(WarningCode.ZERO_OCCUPANCY_ATOMS)
    if inventory.overfull_site_count:
        warnings.append(WarningCode.OVERFULL_ALTERNATE_OCCUPANCY)
    if source.defaulted_occupancy_count:
        warnings.append(WarningCode.OCCUPANCY_DICTIONARY_DEFAULT_APPLIED)
    if source.mapping_failed:
        warnings.append(WarningCode.RAW_OCCUPANCY_MAPPING_FAILED)
    if source.legacy_identifiers_packed:
        warnings.append(WarningCode.LEGACY_PDB_IDENTIFIERS_PACKED)
    return tuple(warnings)
