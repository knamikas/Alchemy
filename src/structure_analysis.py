"""Load deterministic first-model atom selections with Gemmi.

This module is the public face of structure loading: it re-exports the data
model from ``structure_model``, the PDB field helpers from ``pdb_records``,
and conformer selection from ``conformer_selection``, and it orchestrates the
steps in ``structure_loading`` behind ``load_structure``. ``source_atoms``
retains deduplicated alternate positions for occupancy-weighted DPI counts;
``contact_atoms`` selects one conformer per residue for contact searches.
"""

from __future__ import annotations

import math

import gemmi

from conformer_selection import select_residue, site_is_better
from pdb_records import (
    MISSING_VALUE_TOKENS,
    MMCIF_FORMAT,
    PDB_FORMAT,
    PDB_HYBRID36_DIGITS,
    RawOccupancy,
    analysis_format_for_path,
    blank_if_missing,
    canonical_pdb_residue_id,
    decode_pdb_resseq,
    parse_pdb_element,
    valid_occupancy,
)
from structure_loading import (
    ANALYZED_MODEL_INDEX,
    DUPLICATE_ATOM_POSITION_TOLERANCE,
    OVERFULL_OCCUPANCY_NI_FRACTION,
    build_atom_sites,
    occupancy_validation,
    prepare_atom_inventory,
    record_audit,
    residue_indexes,
    source_model_data,
    symmetry_metadata,
    warning_codes,
)
from structure_model import (
    NAN,
    AtomKey,
    AtomRecordAudit,
    AtomSite,
    ContactImage,
    ImageProvenance,
    OccupancyValidation,
    ResidueIdentity,
    ResidueSelection,
    StructureContext,
    SymmetryMetadata,
    pbc_translation,
    position_distance,
    spacegroup_or_none,
)

__all__ = [
    "ANALYZED_MODEL_COUNT",
    "ANALYZED_MODEL_INDEX",
    "DUPLICATE_ATOM_POSITION_TOLERANCE",
    "FIRST_MODEL_POLICY",
    "MISSING_VALUE_TOKENS",
    "NAN",
    "OVERFULL_OCCUPANCY_NI_FRACTION",
    "PDB_HYBRID36_DIGITS",
    "AtomKey",
    "AtomRecordAudit",
    "AtomSite",
    "ContactImage",
    "ImageProvenance",
    "OccupancyValidation",
    "RawOccupancy",
    "ResidueIdentity",
    "ResidueSelection",
    "StructureContext",
    "SymmetryMetadata",
    "blank_if_missing",
    "canonical_pdb_residue_id",
    "count_deposited_ni",
    "count_ni",
    "decode_pdb_resseq",
    "load_structure",
    "parse_pdb_element",
    "pbc_translation",
    "position_distance",
    "select_residue",
    "site_is_better",
    "spacegroup_or_none",
    "valid_occupancy",
]

#: Every entry is analyzed as its first model, whatever the file holds.
FIRST_MODEL_POLICY = "first"
ANALYZED_MODEL_COUNT = 1

#: Gemmi is told the format Alchemy decided on, so that the raw PDB fields are
#: never read from a file Gemmi parsed as something else.
_GEMMI_FORMATS = {
    PDB_FORMAT: gemmi.CoorFormat.Pdb,
    MMCIF_FORMAT: gemmi.CoorFormat.Mmcif,
}


def load_structure(
    pdb_id: str,
    path: str,
    source_model_count: int | None = None,
) -> StructureContext:
    """Parse ``path`` with Gemmi and build Alchemy's first-model atom sets."""
    analysis_format = analysis_format_for_path(path)
    structure = gemmi.read_structure(path, format=_GEMMI_FORMATS[analysis_format])
    if len(structure) == 0:
        raise ValueError(
            f"coordinate file contains no models when read as {analysis_format}"
        )
    input_model_count = (
        len(structure) if source_model_count is None else source_model_count
    )
    if input_model_count < len(structure):
        raise ValueError(
            "source model count cannot be smaller than the analysis model count"
        )
    model = structure[ANALYZED_MODEL_INDEX]
    model_id = str(model.num)
    source = source_model_data(path, model, analysis_format)
    all_sites = build_atom_sites(model, model_id, source)

    inventory = prepare_atom_inventory(model, all_sites)
    symmetry = symmetry_metadata(structure)
    indexes = residue_indexes(inventory.residues)
    occupancy = occupancy_validation(inventory, source)
    records = record_audit(inventory)
    deposited_ni = (
        NAN
        if occupancy.validation_failed or records.unknown_element_atom_count
        else inventory.occupancy_weighted_atom_count
    )
    return StructureContext(
        pdb_id=pdb_id,
        structure=structure,
        model=model,
        model_index=ANALYZED_MODEL_INDEX,
        model_policy=FIRST_MODEL_POLICY,
        input_model_count=input_model_count,
        model_analyzed=ANALYZED_MODEL_COUNT,
        analyzed_model_id=model_id,
        multi_model_structure=input_model_count > 1,
        source_atoms=inventory.source_atoms,
        contact_atoms=inventory.contact_atoms,
        residues=inventory.residues,
        occupancy=occupancy,
        records=records,
        symmetry=symmetry,
        analysis_coordinate_format=source.analysis_format,
        warning_codes=warning_codes(inventory, source, input_model_count),
        deposited_ni=deposited_ni,
        _spatial_model=inventory.spatial_model,
        _atom_by_indices={
            (atom.chain_index, atom.residue_index, atom.atom_index): atom
            for atom in inventory.contact_atoms
        },
        _residue_by_key={residue.key: residue for residue in inventory.residues},
        _residues_by_author=indexes.by_author,
        _residues_by_source_author=indexes.by_source_author,
        _residues_by_coordinate_author=indexes.by_coordinate_author,
    )


def count_deposited_ni(context: StructureContext) -> float:
    """Occupancy-weighted non-H/D count in deposited first-model records.

    The count is fixed when the structure is loaded: NaN when any counted
    atom's occupancy or element is unknown, otherwise the deposited sum.
    """
    return context.deposited_ni


def count_ni(context: StructureContext) -> float:
    """Occupancy-weighted non-H/D count for the complete asymmetric unit.

    Gemmi does not add strict-NCS copies to ``model``, but each non-given NCS
    operation is one more full copy that DPI must count.
    """
    deposited = count_deposited_ni(context)
    if not math.isfinite(deposited):
        return NAN
    return deposited * context.symmetry.dpi_atom_count_multiplier
