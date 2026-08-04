"""Converting a deposited mmCIF into the legacy PDB that EDSTATS can read.

EDSTATS consumes traditional PDB coordinates, whose chain field is one column
wide and whose residue numbers are four decimal digits; deposited mmCIF models
routinely exceed both. Conversion is therefore made reversible: each converted
residue's source identity is recorded as a REMARK in the PDB that Alchemy then
analyses, and every step that could silently change a residue's identity,
ordering, or atom membership raises instead. mmCIF ``.``/``?`` occupancy
becomes a blank PDB column rather than Gemmi's default 1.0, so missingness
survives the round trip.
"""

import os
from typing import Any, Dict, List, Tuple

from structure_analysis import (
    POLYMER_REMARK_PREFIX,
    RESIDUE_REMARK_PREFIX,
    RESNAME_REMARK_PREFIX,
    blank_if_missing,
)


# The one-character chain ids accepted by both Gemmi and the CCP4 tools.
LEGACY_PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
LEGACY_PDB_MAX_RESIDUE_NUMBER = 9999


def _cif_occupancy_by_serial(cif_path) -> Dict[int, str]:
    """Return raw mmCIF occupancy tokens keyed by ``_atom_site.id``.

    Gemmi represents ``.`` and ``?`` occupancy as 1.0 in a Structure, so the
    raw CIF loop must be read before conversion.
    """
    import gemmi

    document = gemmi.cif.read(cif_path)
    atom_blocks = []
    for block in document:
        atom_ids = list(block.find_values("_atom_site.id"))
        if atom_ids:
            atom_blocks.append((block, atom_ids))
    if len(atom_blocks) != 1:
        raise ValueError(
            "mmCIF conversion requires exactly one block with atom_site records"
        )

    block, atom_ids = atom_blocks[0]
    occupancies = list(block.find_values("_atom_site.occupancy"))
    if not occupancies:
        occupancies = ["?"] * len(atom_ids)
    elif len(occupancies) != len(atom_ids):
        raise ValueError("mmCIF atom_site occupancy count does not match atom count")

    by_serial: Dict[int, str] = {}
    for atom_id, occupancy in zip(atom_ids, occupancies):
        try:
            serial = int(atom_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"mmCIF atom_site id is not an integer: {atom_id!r}"
            ) from exc
        if serial in by_serial:
            raise ValueError(f"duplicate mmCIF atom_site id: {serial}")
        by_serial[serial] = occupancy
    return by_serial


def _residue_index_by_author(structure, label):
    """Index residues by ``(model, chain, resnum)``, with the traversal order.

    The order lets a conversion be checked for reordering, not only for changed
    identifiers.
    """
    by_author: dict[tuple[int, str, str], list[Any]] = {}
    order = []
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


def _residue_conversion_records(structure, converted_structure):
    """Pair source mmCIF residue names with names written to legacy PDB."""
    source_by_author, source_order = _residue_index_by_author(structure, "mmCIF")
    converted_by_author, converted_order = _residue_index_by_author(
        converted_structure, "converted"
    )

    records = []
    if converted_order != source_order:
        raise ValueError("PDB conversion changed residue ordering")
    if set(converted_by_author) != set(source_by_author):
        raise ValueError("PDB conversion changed residue author identifiers")
    for key, source_residues in source_by_author.items():
        converted_residues = converted_by_author[key]
        if len(converted_residues) != len(source_residues):
            raise ValueError("PDB conversion changed duplicate residue multiplicity")
        model_index, converted_chain, converted_resnum = key
        for source, converted in zip(source_residues, converted_residues):
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


def _source_residue_records(structure):
    """Snapshot source-mmCIF residue identities before legacy conversion."""
    import gemmi

    polymer_sequence_lengths = {
        str(entity.name): len(entity.full_sequence)
        for entity in structure.entities
        if entity.entity_type == gemmi.EntityType.Polymer
        and entity.full_sequence
    }
    records = []
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


def _legacy_identifiers_need_packing(structure):
    """Whether Gemmi could not shorten every chain to a portable PDB id."""
    return any(
        bool(str(chain.name)) and str(chain.name) not in LEGACY_PDB_CHAIN_IDS
        for model in structure
        for chain in model
    )


def _pack_legacy_pdb_residue_ids(structure):
    """Assign a unique, one-character PDB identity to every residue.

    Multiple source chains may share one synthetic chain because EDSTATS needs
    only an unambiguous residue key, not polymer connectivity. Whole source
    chains stay together, TER records preserve their boundaries, and sequence
    numbers never exceed the portable four-column decimal PDB range.
    """
    import gemmi

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


def _residue_identity_records(source_records, converted_structure):
    """Map packed PDB residue identities back to source-mmCIF identities."""
    converted_records = []
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

    records = []
    for source, converted in zip(source_records, converted_records):
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


def _polymer_position_records(source_records, converted_structure):
    """Map every converted residue to its source polymer-boundary status."""
    converted_records = []
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
    positions = {}
    for source, converted in zip(source_records, converted_records):
        position = source[-1]
        previous = positions.get(converted)
        if previous is not None and previous != position:
            # The legacy coordinate key cannot distinguish these source
            # residues, so retain uncertainty instead of assigning either
            # boundary classification to both.
            position = "?"
        positions[converted] = position
    return [(*converted, position) for converted, position in positions.items()]


def _write_cif_conversion_provenance(
    dst: str,
    missing_occupancies: List[bool],
    residue_records: List[Tuple[int, str, str, str, str]],
    identity_records=None,
    polymer_records=None,
) -> None:
    """Blank unknown occupancies and embed reversible residue mappings."""
    with open(dst, encoding="utf-8", errors="strict", newline="") as handle:
        lines = handle.readlines()

    atom_line_indices = [
        index
        for index, line in enumerate(lines)
        if line[:6].strip().upper() in ("ATOM", "HETATM")
    ]
    if len(atom_line_indices) != len(missing_occupancies):
        raise ValueError("PDB conversion output atom count does not match mmCIF input")
    for line_index, missing in zip(atom_line_indices, missing_occupancies):
        if not missing:
            continue
        line = lines[line_index]
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        body = body.ljust(60)
        lines[line_index] = body[:54] + "      " + body[60:] + newline

    remarks = [
        (
            f"{RESNAME_REMARK_PREFIX} {model_index} "
            f"{chain or '_'} {resnum} {converted_name} {source_name}\n"
        )
        for (model_index, chain, resnum, converted_name, source_name) in residue_records
    ]
    remarks.extend(
        (
            f"{RESIDUE_REMARK_PREFIX} {model_index} "
            f"{converted_chain or '_'} {converted_resnum} {converted_name} "
            f"{source_chain or '_'} {source_number} "
            f"{source_insertion or '_'} {source_name} "
            f"{source_chain_index} {source_residue_index} "
            f"{source_polymer_position}\n"
        )
        for (
            model_index,
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
        ) in (identity_records or ())
    )
    remarks.extend(
        (
            f"{POLYMER_REMARK_PREFIX} {model_index} "
            f"{converted_chain or '_'} {converted_resnum} {converted_name} "
            f"{polymer_position}\n"
        )
        for (
            model_index,
            converted_chain,
            converted_resnum,
            converted_name,
            polymer_position,
        ) in (polymer_records or ())
    )
    with open(dst, "w", encoding="utf-8", newline="") as handle:
        handle.writelines(remarks)
        handle.writelines(lines)


def _cif_to_pdb(cif_path, dst):
    """Convert mmCIF to PDB without discarding occupancy or CCD provenance."""
    import gemmi

    if not os.path.exists(cif_path):
        raise FileNotFoundError(cif_path)
    occupancy_by_serial = _cif_occupancy_by_serial(cif_path)
    structure = gemmi.read_structure(cif_path)
    structure_atoms = [
        atom
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    ]
    if len(structure_atoms) != len(occupancy_by_serial):
        raise ValueError(
            "Gemmi structure atom count does not match mmCIF atom_site records"
        )
    missing_occupancies = []
    for atom in structure_atoms:
        serial = atom.serial
        if serial is None or int(serial) not in occupancy_by_serial:
            raise ValueError(
                "Gemmi atom serial could not be matched to mmCIF atom_site id"
            )
        missing_occupancies.append(occupancy_by_serial[int(serial)] in (".", "?"))

    structure.setup_entities()
    source_residues = _source_residue_records(structure)
    # Shorten before writing and then analyse this exact PDB, so EDSTATS and
    # Alchemy never join identifiers from two different representations.
    structure.shorten_chain_names()
    identifiers_packed = _legacy_identifiers_need_packing(structure)
    if identifiers_packed:
        _pack_legacy_pdb_residue_ids(structure)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    structure.write_pdb(dst)
    converted_structure = gemmi.read_structure(dst)
    residue_records = _residue_conversion_records(structure, converted_structure)
    identity_records = (
        _residue_identity_records(source_residues, converted_structure)
        if identifiers_packed
        else []
    )
    # Every converted residue carries its source polymer position, even when
    # its PDB identifiers did not need packing. This deliberately remains
    # separate from reversible identity records: ordinary chain shortening is
    # part of the analysis namespace and must not look like identifier packing.
    polymer_records = _polymer_position_records(source_residues, converted_structure)
    _write_cif_conversion_provenance(
        dst,
        missing_occupancies,
        residue_records,
        identity_records,
        polymer_records,
    )
    return dst


def _first_model_pdb(pdb_path, dst):
    """Return a wrapper-free PDB containing the first coordinate model.

    The extraction is textual so atom records, occupancies, identifiers, and
    ordering remain exactly as deposited; Gemmi only determines and verifies
    the model count. MODEL/ENDMDL records are removed because EDSTATS emits a
    synthetic separator residue for even a one-model wrapper.
    """
    import gemmi

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

    # NUMMDL describes the source ensemble and would be false here. The other
    # header records stay: EDSTATS needs the same cell and symmetry metadata.
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
