"""Resolve one entry's coordinate and reflection inputs before any stage runs.

The worker hands each entry here once. Manual inputs and PDB-REDO mirror
entries converge on the same first-model PDB, the structure Gemmi loads from
it, and the provenance that records where both came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from coordinate_conversion import first_model_pdb
from inputs import (
    PdbRedoMetadata,
    ensure_entry_available,
    entry_dir_for,
    final_file_candidates,
    first_existing,
    prepare_inputs,
    read_map_column_resolution,
    read_pdb_redo_metadata,
    read_resolution,
    resolve_manual_inputs,
)
from structure_analysis import StructureContext, load_structure
from worker_contracts import CoordinateProvenance, WorkerConfig


@dataclass(frozen=True, slots=True)
class EntryInputs:
    """Resolved input paths, metadata, and resolution limits for one entry."""

    work_dir: str
    mtz: str
    # First-model analysis coordinates; source_coordinate_path retains the input.
    pdb: str
    # PDB-REDO metadata or manual --data-json; None if none was supplied.
    data_json: str | None
    # The diffraction data's own high-resolution limit, as distinct from the
    # map columns' range below.
    data_reshi: float
    map_reslo: float
    map_reshi: float
    pdb_redo_is_twin: bool
    source_coordinate_path: str


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Coordinate and PDB-REDO provenance the input stage establishes."""

    coordinates: CoordinateProvenance
    pdb_redo_metadata: PdbRedoMetadata


def _source_coordinate_format(cfg: WorkerConfig, source_path: str) -> tuple[str, bool]:
    """The deposited coordinate format and whether analysis converted it to PDB."""
    manual = cfg.manual_inputs
    if manual:
        converted = bool(manual.get("cif_file"))
    else:
        converted = source_path.lower().endswith((".cif", ".cif.gz"))
    return ("mmcif" if converted else "pdb", converted)


def _source_coordinate_path(
    cfg: WorkerConfig, pdb_id: str, entry: str, analysis_path: str
) -> str:
    """The deposited coordinate file, falling back to the analysis coordinates."""
    manual = cfg.manual_inputs
    if manual:
        return manual.get("cif_file") or manual.get("pdb_file") or ""
    return (
        first_existing(*final_file_candidates(entry, pdb_id, "coordinates"))
        or analysis_path
    )


def source_coordinate_provenance_path(
    cfg: WorkerConfig, pdb_id: str, source_path: str
) -> str:
    """Keep mirror provenance portable while preserving a manual input path."""
    if cfg.manual_inputs:
        return source_path
    return f"{pdb_id[1:3]}/{pdb_id}/{os.path.basename(source_path)}"


def resolve_entry_dir(pdb_id: str, cfg: WorkerConfig) -> str:
    """Locate an entry's PDB-REDO directory, downloading it when permitted."""
    if cfg.allow_download:
        available = ensure_entry_available(
            pdb_id, cfg.pdb_redo_root, cfg.pdb_redo_cache
        )
        return entry_dir_for(available.root, pdb_id)
    return entry_dir_for(cfg.input_root, pdb_id)


def prepare_analysis_inputs(
    pdb_id: str,
    cfg: WorkerConfig,
    entry: str,
    work_dir: str,
) -> tuple[EntryInputs, StructureContext, InputProvenance]:
    """Prepare the same first-model PDB for EDSTATS and Gemmi."""
    manual_inputs = cfg.manual_inputs
    data_json: str | None = None
    if manual_inputs:
        mtz, pdb = resolve_manual_inputs(
            pdb_id,
            pdb_file=manual_inputs.get("pdb_file"),
            mtz_file=manual_inputs.get("mtz_file"),
            cif_file=manual_inputs.get("cif_file"),
            work_dir=work_dir,
        )
        entry = os.path.dirname(pdb) or work_dir
        data_json = manual_inputs.get("data_json")
        data_reshi = read_resolution(entry, mtz, data_json_path=data_json)
    else:
        mtz, pdb = prepare_inputs(pdb_id, entry, work_dir)
        data_reshi = read_resolution(entry, mtz)

    # Feeds PDB-REDO provenance and the DPI stage; density never reads it.
    metadata_json = data_json if manual_inputs else os.path.join(entry, "data.json")
    pdb_redo_metadata = read_pdb_redo_metadata(
        metadata_json,
        required=bool(manual_inputs and data_json),
    )
    map_reslo, map_reshi = read_map_column_resolution(mtz)
    source_pdb = pdb
    source_coordinate_path = _source_coordinate_path(cfg, pdb_id, entry, source_pdb)
    source_format, converted = _source_coordinate_format(cfg, source_coordinate_path)
    model1_pdb = os.path.join(work_dir, f"{pdb_id}_model1.pdb")
    if os.path.realpath(model1_pdb) == os.path.realpath(source_pdb):
        model1_pdb = os.path.join(work_dir, f"{pdb_id}_analysis_model1.pdb")
    pdb, input_model_count = first_model_pdb(source_pdb, model1_pdb)
    inputs = EntryInputs(
        work_dir=work_dir,
        mtz=mtz,
        pdb=pdb,
        data_json=metadata_json,
        data_reshi=data_reshi,
        map_reslo=map_reslo,
        map_reshi=map_reshi,
        pdb_redo_is_twin=pdb_redo_metadata.is_twin,
        source_coordinate_path=source_coordinate_path,
    )
    structure = load_structure(pdb_id, pdb, source_model_count=input_model_count)
    provenance = InputProvenance(
        coordinates=CoordinateProvenance(
            source_coordinate_format=source_format,
            analysis_coordinate_format=structure.analysis_coordinate_format,
            coordinate_conversion_performed=converted,
            source_coordinate_path=source_coordinate_provenance_path(
                cfg, pdb_id, source_coordinate_path
            ),
            input_model_count=structure.input_model_count,
            model_analyzed=structure.model_analyzed,
            multi_model_structure=structure.multi_model_structure,
        ),
        pdb_redo_metadata=pdb_redo_metadata,
    )
    return inputs, structure, provenance
