from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfig:
    id: str | None
    id_file: str | None
    pdb_file: str | None
    mtz_file: str | None
    cif_file: str | None
    data_json: str | None
    pdb_redo_root: str
    pdb_redo_cache: str
    pdb_metadata_cache: str
    crystallization_download: bool
    max_pdbs: int | None
    workers: int | None
    output_dir: str
    density_map_scope: str
    verbose: int
    quiet: bool
    log_dir: str | None
    log_file: str | None
    ccp4_timeout: int
    confidence_reference_dir: str | None
    ccp4_setup: str | None
    configure_ccp4: str | None
    keep_intermediates: bool
    resume: bool
    retry_partials: bool
    bonds: bool
