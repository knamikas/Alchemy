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
    prepare_data_json,
    prepare_inputs,
    read_entry_metadata,
    read_map_column_resolution,
    resolve_manual_inputs,
)
from structure_analysis import StructureContext, load_structure
from worker.contracts import CoordinateProvenance, WorkerConfig


@dataclass(frozen=True, slots=True)
class EntryInputs:
    """Resolved input paths, metadata, and resolution limits for one entry."""

    work_dir: str
    mtz: str
    # First-model analysis coordinates; source_coordinate_path retains the input.
    pdb: str
    # The entry's PDB-REDO data.json or the manual --data-json; None when
    # the entry has none, so the DPI stage reports a missing metadata source.
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
    entry_dir: str | None,
    work_dir: str,
) -> tuple[EntryInputs, StructureContext, InputProvenance]:
    """Prepare the same first-model PDB for EDSTATS and Gemmi.

    ``entry_dir`` is the PDB-REDO entry of a mirror run; manual inputs have
    none and read only the files named on the command line, never a
    ``data.json`` that happens to sit beside them.
    """
    manual_inputs = cfg.manual_inputs
    if manual_inputs:
        prepared = resolve_manual_inputs(
            pdb_id,
            pdb_file=manual_inputs.get("pdb_file"),
            mtz_file=manual_inputs.get("mtz_file"),
            cif_file=manual_inputs.get("cif_file"),
            work_dir=work_dir,
        )
        data_json = manual_inputs.get("data_json")
        # An explicitly named metadata file is an input contract, not a probe.
        metadata_required = data_json is not None
    else:
        if entry_dir is None:
            raise ValueError("a PDB-REDO entry directory is required")
        prepared = prepare_inputs(pdb_id, entry_dir, work_dir)
        data_json = prepare_data_json(entry_dir, work_dir)
        metadata_required = False

    # Feeds PDB-REDO provenance and the DPI stage; density never reads it.
    metadata = read_entry_metadata(prepared.mtz, data_json, required=metadata_required)
    map_reslo, map_reshi = read_map_column_resolution(prepared.mtz)
    analysis_pdb, input_model_count = first_model_pdb(
        prepared.pdb, os.path.join(work_dir, f"{pdb_id}_model1.pdb")
    )
    inputs = EntryInputs(
        work_dir=work_dir,
        mtz=prepared.mtz,
        pdb=analysis_pdb,
        data_json=data_json,
        data_reshi=metadata.data_reshi,
        map_reslo=map_reslo,
        map_reshi=map_reshi,
        pdb_redo_is_twin=metadata.pdb_redo.is_twin,
        source_coordinate_path=prepared.coordinates,
    )
    structure = load_structure(
        pdb_id, analysis_pdb, source_model_count=input_model_count
    )
    provenance = InputProvenance(
        coordinates=CoordinateProvenance(
            source_coordinate_format=prepared.source_coordinate_format,
            analysis_coordinate_format=structure.analysis_coordinate_format,
            coordinate_conversion_performed=prepared.converted,
            source_coordinate_path=source_coordinate_provenance_path(
                cfg, pdb_id, prepared.coordinates
            ),
            input_model_count=structure.input_model_count,
            model_analyzed=structure.model_analyzed,
            multi_model_structure=structure.multi_model_structure,
        ),
        pdb_redo_metadata=metadata.pdb_redo,
    )
    return inputs, structure, provenance
