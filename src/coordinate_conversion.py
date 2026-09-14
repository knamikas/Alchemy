"""Convert mmCIF coordinates to the legacy PDB format required by EDSTATS.

Preserve source residue identities and occupancy provenance in REMARK records.
Reject conversions that change atom membership, ordering, or residue identity.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol, cast

import gemmi

from pdb_remarks import (
    PolymerPositionRecord,
    ResidueIdentityRecord,
    ResnameRecord,
    write_conversion_provenance,
)
from structure_analysis import blank_if_missing

# The one-character chain ids accepted by both Gemmi and the CCP4 tools.
LEGACY_PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class _PdbWritable(Protocol):
    """The fully typed positional overload of Gemmi's PDB writer."""

    def write_pdb(self, path: str, options: gemmi.PdbWriteOptions, /) -> None: ...


LEGACY_PDB_MAX_RESIDUE_NUMBER = 9999

# (atom name, element symbol) pairs, in residue order.
_ResidueAtoms = tuple[tuple[str, str], ...]
# Residue name plus its atoms, keyed elsewhere by author identifiers.
_ResidueEntry = tuple[str, _ResidueAtoms]
_SourceRecord = tuple[int, str, int, str, str, _ResidueAtoms, int, int, str]


def _structure_atom_signatures(
    structure: gemmi.Structure,
) -> tuple[tuple[object, ...], ...]:
    """Describe atom traversal without depending on source atom-site ids."""
    return tuple(
        (
            model_index,
            str(model.num),
            str(chain.name),
            str(residue.name),
            residue.seqid.num,
            blank_if_missing(str(residue.seqid.icode)),
            str(atom.name),
            blank_if_missing(str(atom.altloc)),
            str(atom.element.name),
            float(atom.pos.x),
            float(atom.pos.y),
            float(atom.pos.z),
        )
        for model_index, model in enumerate(structure)
        for chain in model
        for residue in chain
        for atom in residue
    )


def _cif_atom_data(
    cif_path: str,
) -> tuple[
    tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[tuple[object, ...], ...]
]:
    """Return occupancies and generated PDB serials in Gemmi traversal order.

    Gemmi represents ``.`` and ``?`` occupancy as 1.0 in a Structure, so the
    raw CIF loop must be read before conversion. ``_atom_site.id`` is an opaque
    code, not necessarily an integer; a temporary in-memory parse with generated
    numeric ids carries each source row number through any Gemmi reordering.
    """
    document = gemmi.cif.read(cif_path)
    atom_blocks: list[tuple[gemmi.cif.Block, list[str]]] = []
    for block in document:
        atom_ids = list(block.find_values("_atom_site.id"))
        if atom_ids:
            atom_blocks.append((block, atom_ids))
    if len(atom_blocks) != 1:
        raise ValueError(
            "mmCIF conversion requires exactly one block with atom_site records"
        )

    block, atom_ids = atom_blocks[0]
    seen_ids: set[str] = set()
    for atom_id in atom_ids:
        atom_id = str(atom_id)
        if not blank_if_missing(atom_id):
            raise ValueError("mmCIF atom_site id is missing")
        if atom_id in seen_ids:
            raise ValueError(f"duplicate mmCIF atom_site id: {atom_id}")
        seen_ids.add(atom_id)

    occupancies = list(block.find_values("_atom_site.occupancy"))
    occupancy_defaulted = not occupancies
    if occupancy_defaulted:
        occupancies = ["1.0"] * len(atom_ids)
    elif len(occupancies) != len(atom_ids):
        raise ValueError("mmCIF atom_site occupancy count does not match atom count")

    atom_id_column = block.find_values("_atom_site.id")
    for row_index in range(len(atom_ids)):
        atom_id_column[row_index] = str(row_index + 1)
    indexed_structure = gemmi.make_structure_from_block(block)
    # ``gemmi.read_structure`` performs this merge for mmCIF input. Mirror it
    # here so generated row serials follow the exact traversal later converted
    # to PDB, including files whose atom_site loop interleaves chain segments.
    indexed_structure.merge_chain_parts()
    indexed_atoms = [
        atom
        for model in indexed_structure
        for chain in model
        for residue in chain
        for atom in residue
    ]
    if len(indexed_atoms) != len(atom_ids):
        raise ValueError(
            "Gemmi structure atom count does not match mmCIF atom_site records"
        )

    pdb_serials = tuple(int(atom.serial) for atom in indexed_atoms)
    if set(pdb_serials) != set(range(1, len(atom_ids) + 1)):
        raise ValueError("generated atom_site row ids did not survive Gemmi parsing")
    ordered_occupancies = tuple(occupancies[serial - 1] for serial in pdb_serials)
    defaulted_counts = tuple(
        sum(1 for chain in model for residue in chain for _atom in residue)
        if occupancy_defaulted
        else 0
        for model in indexed_structure
    )
    return (
        ordered_occupancies,
        pdb_serials,
        defaulted_counts,
        _structure_atom_signatures(indexed_structure),
    )


def residue_index_by_author(
    structure: gemmi.Structure, label: str
) -> tuple[dict[tuple[int, str, str], list[_ResidueEntry]], list[tuple[int, str, str]]]:
    """Index residues by ``(model, chain, resnum)``, with the traversal order.

    The order lets a conversion be checked for reordering, not only for changed
    identifiers.
    """
    by_author: dict[tuple[int, str, str], list[_ResidueEntry]] = {}
    order: list[tuple[int, str, str]] = []
    for model_index, model in enumerate(structure):
        for chain in model:
            for residue in chain:
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"{label} residue {residue.name!r} has no author number"
                    )
                insertion = blank_if_missing(str(residue.seqid.icode))
                key = (model_index, str(chain.name), f"{number}{insertion}")
                order.append(key)
                by_author.setdefault(key, []).append(
                    (
                        str(residue.name),
                        tuple(
                            (str(atom.name), str(atom.element.name)) for atom in residue
                        ),
                    )
                )
    return by_author, order


def residue_conversion_records(
    structure: gemmi.Structure, converted_structure: gemmi.Structure
) -> list[ResnameRecord]:
    """Pair source mmCIF residue names with names written to legacy PDB."""
    source_by_author, source_order = residue_index_by_author(structure, "mmCIF")
    converted_by_author, converted_order = residue_index_by_author(
        converted_structure, "converted"
    )

    records: list[ResnameRecord] = []
    if converted_order != source_order:
        raise ValueError("PDB conversion changed residue ordering")
    if set(converted_by_author) != set(source_by_author):
        raise ValueError("PDB conversion changed residue author identifiers")
    for key, source_residues in source_by_author.items():
        converted_residues = converted_by_author[key]
        if len(converted_residues) != len(source_residues):
            raise ValueError("PDB conversion changed duplicate residue multiplicity")
        model_index, converted_chain, converted_resnum = key
        for source, converted in zip(source_residues, converted_residues, strict=False):
            source_name, source_atoms = source
            converted_name, converted_atoms = converted
            if source_atoms != converted_atoms:
                raise ValueError("PDB conversion changed residue atom membership")
            if converted_name != source_name:
                records.append(
                    (
                        model_index + 1,
                        converted_chain,
                        converted_resnum,
                        converted_name,
                        source_name,
                    )
                )
    return records


def _source_residue_records(structure: gemmi.Structure) -> list[_SourceRecord]:
    """Snapshot source-mmCIF residue identities before legacy conversion."""
    polymer_sequence_lengths = {
        str(entity.name): len(entity.full_sequence)
        for entity in structure.entities
        if entity.entity_type == gemmi.EntityType.Polymer and entity.full_sequence
    }
    records: list[_SourceRecord] = []
    for model_index, model in enumerate(structure, start=1):
        for source_chain_index, chain in enumerate(model):
            for residue_index, residue in enumerate(chain):
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"mmCIF residue {residue.name!r} has no author number"
                    )
                if residue.entity_type != gemmi.EntityType.Polymer:
                    polymer_position = "-"
                else:
                    label_seq = residue.label_seq
                    sequence_length = polymer_sequence_lengths.get(
                        str(residue.entity_id)
                    )
                    if (
                        label_seq is None
                        or sequence_length is None
                        or label_seq < 1
                        or label_seq > sequence_length
                    ):
                        # A modeled endpoint is not evidence of a chemical
                        # terminus when the deposited polymer extent is absent.
                        polymer_position = "?"
                    else:
                        is_first = label_seq == 1
                        is_last = label_seq == sequence_length
                        polymer_position = (
                            "NC"
                            if is_first and is_last
                            else ("N" if is_first else ("C" if is_last else "M"))
                        )
                records.append(
                    (
                        model_index,
                        str(chain.name),
                        int(number),
                        blank_if_missing(str(residue.seqid.icode)),
                        str(residue.name),
                        tuple(
                            (str(atom.name), str(atom.element.name)) for atom in residue
                        ),
                        source_chain_index,
                        residue_index,
                        polymer_position,
                    )
                )
    return records


def _legacy_identifiers_need_packing(structure: gemmi.Structure) -> bool:
    """Return whether residue identities need reversible PDB packing.

    Legacy PDB requires one-character chains and unique chain, number, and
    insertion-code keys. Gemmi merges residues sharing those keys on reread.
    """
    for model in structure:
        identities: set[tuple[str, int | None, str]] = set()
        for chain in model:
            chain_name = str(chain.name)
            if chain_name and chain_name not in LEGACY_PDB_CHAIN_IDS:
                return True
            for residue in chain:
                identity = (
                    chain_name,
                    residue.seqid.num,
                    blank_if_missing(str(residue.seqid.icode)),
                )
                if identity in identities:
                    return True
                identities.add(identity)
    return False


def _pack_legacy_pdb_residue_ids(structure: gemmi.Structure) -> None:
    """Assign a unique, one-character PDB identity to every residue.

    Multiple source chains may share one synthetic chain because EDSTATS needs
    only an unambiguous residue key, not polymer connectivity. Whole source
    chains stay together, TER records preserve their boundaries, and sequence
    numbers never exceed the portable four-column decimal PDB range.
    """
    for model in structure:
        chain_slot = 0
        next_residue_number = 1
        for chain in model:
            residue_count = len(chain)
            if residue_count > LEGACY_PDB_MAX_RESIDUE_NUMBER:
                raise ValueError(
                    "one mmCIF chain contains more residues than a portable "
                    "PDB chain can represent"
                )
            if next_residue_number + residue_count - 1 > LEGACY_PDB_MAX_RESIDUE_NUMBER:
                chain_slot += 1
                next_residue_number = 1
            if chain_slot >= len(LEGACY_PDB_CHAIN_IDS):
                raise ValueError(
                    "mmCIF model contains more residues than the portable "
                    "PDB surrogate namespace can represent"
                )
            chain.name = LEGACY_PDB_CHAIN_IDS[chain_slot]
            for residue in chain:
                residue.seqid = gemmi.SeqId(next_residue_number, " ")
                next_residue_number += 1


def _residue_identity_records(
    source_records: Sequence[_SourceRecord], converted_structure: gemmi.Structure
) -> list[ResidueIdentityRecord]:
    """Map packed PDB residue identities back to source-mmCIF identities."""
    converted_records: list[tuple[int, str, str, str, _ResidueAtoms]] = []
    for model_index, model in enumerate(converted_structure, start=1):
        for chain in model:
            for residue in chain:
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"converted residue {residue.name!r} has no number"
                    )
                converted_records.append(
                    (
                        model_index,
                        str(chain.name),
                        f"{number}{blank_if_missing(str(residue.seqid.icode))}",
                        str(residue.name),
                        tuple(
                            (str(atom.name), str(atom.element.name)) for atom in residue
                        ),
                    )
                )
    if len(source_records) != len(converted_records):
        raise ValueError("PDB conversion changed residue count")

    records: list[ResidueIdentityRecord] = []
    for source, converted in zip(source_records, converted_records, strict=False):
        (
            source_model,
            source_chain,
            source_number,
            source_insertion,
            source_name,
            source_atoms,
            source_chain_index,
            source_residue_index,
            source_polymer_position,
        ) = source
        (
            converted_model,
            converted_chain,
            converted_resnum,
            converted_name,
            converted_atoms,
        ) = converted
        if source_model != converted_model:
            raise ValueError("PDB conversion changed residue model ordering")
        if source_atoms != converted_atoms:
            raise ValueError("PDB conversion changed residue atom membership")
        source_resnum = f"{source_number}{source_insertion}"
        if (source_chain, source_resnum, source_name) == (
            converted_chain,
            converted_resnum,
            converted_name,
        ):
            continue
        records.append(
            (
                converted_model,
                converted_chain,
                converted_resnum,
                converted_name,
                source_chain,
                source_number,
                source_insertion,
                source_name,
                source_chain_index,
                source_residue_index,
                source_polymer_position,
            )
        )
    return records


def _polymer_position_records(
    source_records: Sequence[_SourceRecord], converted_structure: gemmi.Structure
) -> list[PolymerPositionRecord]:
    """Map every converted residue to its source polymer-boundary status."""
    converted_records: list[tuple[int, str, str, str]] = []
    for model_index, model in enumerate(converted_structure, start=1):
        for chain in model:
            for residue in chain:
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"converted residue {residue.name!r} has no number"
                    )
                converted_records.append(
                    (
                        model_index,
                        str(chain.name),
                        f"{number}{blank_if_missing(str(residue.seqid.icode))}",
                        str(residue.name),
                    )
                )
    if len(source_records) != len(converted_records):
        raise ValueError("PDB conversion changed residue count")
    positions: dict[tuple[int, str, str, str], str] = {}
    for source, converted in zip(source_records, converted_records, strict=False):
        position = source[-1]
        previous = positions.get(converted)
        if previous is not None and previous != position:
            # Ambiguous legacy keys cannot establish either source residue's boundary status.
            position = "?"
        positions[converted] = position
    return [(*converted, position) for converted, position in positions.items()]


def _blank_missing_occupancies(dst: str, missing_occupancies: Sequence[bool]) -> None:
    """Clear the occupancy columns of atoms whose mmCIF occupancy was null.

    Gemmi writes such atoms with occupancy 1.00; blank columns let the analysis
    report the value as unknown rather than as a full-occupancy site.
    """
    with open(dst, encoding="utf-8", errors="strict", newline="") as handle:
        lines = handle.readlines()

    atom_line_indices = [
        index
        for index, line in enumerate(lines)
        if line[:6].strip().upper() in ("ATOM", "HETATM")
    ]
    if len(atom_line_indices) != len(missing_occupancies):
        raise ValueError("PDB conversion output atom count does not match mmCIF input")
    if not any(missing_occupancies):
        return
    for line_index, missing in zip(
        atom_line_indices, missing_occupancies, strict=False
    ):
        if not missing:
            continue
        line = lines[line_index]
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        body = body.ljust(60)
        lines[line_index] = body[:54] + "      " + body[60:] + newline
    with open(dst, "w", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)


def cif_to_pdb(cif_path: str, dst: str) -> str:
    """Convert mmCIF to PDB without discarding occupancy or CCD provenance."""
    if not os.path.exists(cif_path):
        raise FileNotFoundError(cif_path)
    (
        occupancies,
        pdb_serials,
        defaulted_occupancy_counts,
        indexed_signatures,
    ) = _cif_atom_data(cif_path)
    structure = gemmi.read_structure(cif_path)
    structure_atoms = [
        atom
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    ]
    if len(structure_atoms) != len(occupancies):
        raise ValueError(
            "Gemmi structure atom count does not match mmCIF atom_site records"
        )
    if _structure_atom_signatures(structure) != indexed_signatures:
        raise ValueError("generated atom_site ids changed Gemmi atom traversal")
    for atom, serial in zip(structure_atoms, pdb_serials, strict=False):
        atom.serial = serial
    missing_occupancies = [occupancy in (".", "?") for occupancy in occupancies]

    structure.setup_entities()
    source_residues = _source_residue_records(structure)
    # Analyze the exact PDB passed to EDSTATS so both use the same identifiers.
    structure.shorten_chain_names()
    identifiers_packed = _legacy_identifiers_need_packing(structure)
    if identifiers_packed:
        _pack_legacy_pdb_residue_ids(structure)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    cast(_PdbWritable, structure).write_pdb(dst, gemmi.PdbWriteOptions())
    converted_structure = gemmi.read_structure(dst)
    residue_records = residue_conversion_records(structure, converted_structure)
    identity_records = (
        _residue_identity_records(source_residues, converted_structure)
        if identifiers_packed
        else []
    )
    # Preserve polymer position for every residue, independently of identity packing.
    polymer_records = _polymer_position_records(source_residues, converted_structure)
    _blank_missing_occupancies(dst, missing_occupancies)
    write_conversion_provenance(
        dst,
        residue_records,
        identity_records,
        polymer_records,
        defaulted_occupancy_counts,
    )
    return dst


def first_model_pdb(pdb_path: str, dst: str) -> tuple[str, int]:
    """Return a wrapper-free PDB containing the first coordinate model.

    The extraction is textual so atom records, occupancies, identifiers, and
    ordering remain exactly as deposited; Gemmi only determines and verifies
    the model count. MODEL/ENDMDL records are removed because EDSTATS emits a
    synthetic separator residue for even a one-model wrapper.
    """
    structure = gemmi.read_structure(pdb_path)
    model_count = len(structure)
    if model_count == 0:
        raise ValueError("coordinate file contains no models")

    with open(pdb_path, encoding="utf-8", errors="replace", newline="") as fh:
        lines = fh.readlines()
    model_starts = [
        index for index, line in enumerate(lines) if line[:6].strip().upper() == "MODEL"
    ]
    if not model_starts:
        if model_count == 1:
            return pdb_path, model_count
        raise ValueError(
            "Gemmi found multiple models but the PDB contains no MODEL records"
        )
    if len(model_starts) != model_count:
        raise ValueError("Gemmi model count does not match the PDB MODEL records")

    first_start = model_starts[0]
    next_start = model_starts[1] if len(model_starts) > 1 else len(lines)
    first_end = next(
        (
            index
            for index in range(first_start + 1, next_start)
            if lines[index][:6].strip().upper() == "ENDMDL"
        ),
        None,
    )
    if first_end is None:
        raise ValueError("the first PDB MODEL record has no matching ENDMDL")
    first_block = lines[first_start + 1 : first_end]

    # Drop the source ensemble count; retain cell and symmetry headers for EDSTATS.
    header = [
        line for line in lines[:first_start] if line[:6].strip().upper() != "NUMMDL"
    ]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(header)
        fh.writelines(first_block)
        fh.write("END\n")

    analysis_structure = gemmi.read_structure(dst)
    if len(analysis_structure) != 1:
        raise ValueError("failed to create a first-model-only analysis PDB")
    with open(dst, encoding="utf-8", errors="replace") as fh:
        if any(line[:6].strip().upper() in ("MODEL", "ENDMDL") for line in fh):
            raise ValueError("first-model analysis PDB still contains a model wrapper")
    return dst, model_count
