"""Keep process-boundary models independent from worker execution imports."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from _version import __version__
from codes import EntryStatus

if TYPE_CHECKING:
    from coordination.schema import BondRow, CandidateRow
    from output_rows import MetalStatsRow


class InflightEvent(NamedTuple):
    """A worker's notice that it started or finished holding an entry.

    Sent synchronously through the inflight queue, so the driver can name the
    entry a worker held if that process dies without returning a result.
    """

    state: Literal["start", "end"]
    pid: int
    pdb_id: str


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable so every worker in a run receives the same configuration.

    Field names follow ``RunConfig`` and the command line; ``input_root`` is
    the resolved directory entries are read from, which the CLI never names.
    """

    input_root: str
    pdb_redo_root: str | None
    pdb_redo_cache: str
    env: dict[str, str]
    output_dir: str
    cofactors: Collection[str]
    keep_intermediates: bool
    bonds: bool
    density_map_scope: str
    ccp4_timeout: int
    log_level: int
    allow_download: bool
    manual_inputs: dict[str, str | None] | None
    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str
    reference_data_id: str
    analysis_config_id: str
    pdb_metadata_cache: str = ""


@dataclass(frozen=True, slots=True)
class SoftwareProvenance:
    """The software and reference data every entry of a run was analyzed with."""

    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str
    reference_data_id: str
    analysis_config_id: str
    alchemy_version: str = __version__


@dataclass(frozen=True, slots=True)
class CoordinateProvenance:
    """Where the analyzed coordinates came from and which model was used.

    The model fields start unmeasured so an entry that failed before its
    structure loaded cannot claim a model count.
    """

    source_coordinate_format: str = ""
    analysis_coordinate_format: str = "pdb"
    coordinate_conversion_performed: bool = False
    source_coordinate_path: str = ""
    input_model_count: int | None = None
    model_analyzed: int | None = None
    multi_model_structure: bool | None = None


@dataclass(frozen=True, slots=True)
class PdbRedoProvenance:
    """The PDB-REDO refinement the entry's reflections and metadata describe."""

    refinement_state: str
    pdb_redo_is_twin: bool | None = None
    pdb_redo_version: str = ""
    pdb_redo_date: str = ""


@dataclass(frozen=True, slots=True)
class DensityProvenance:
    """What the density stage produced, sized for the run report."""

    density_map_scope_used: str = ""
    density_full_map_bytes: int = 0
    density_edstats_map_bytes: int = 0
    ccp4_timeout_log_path: str = ""


@dataclass(slots=True)
class EntryResult:
    """Keep an unrun stage distinguishable from one that measured zero.

    The provenance groups are frozen; a stage that establishes one replaces
    it whole, so a half-filled group never reaches the manifest.
    """

    pdb_id: str
    software: SoftwareProvenance
    pdb_redo: PdbRedoProvenance
    coordinates: CoordinateProvenance = field(default_factory=CoordinateProvenance)
    density: DensityProvenance = field(default_factory=DensityProvenance)

    status: EntryStatus = EntryStatus.ERROR
    retryable: bool = True
    n_metals: int = 0
    runtime_s: float = 0.0
    status_detail: str = ""
    no_metals: bool = False
    metal_site_limit_exceeded: bool = False

    rows: list[MetalStatsRow] = field(default_factory=list)
    bond_rows: list[BondRow] = field(default_factory=list)
    candidate_rows: list[CandidateRow] = field(default_factory=list)
    crystallization_condition_rows: list[dict[str, Any]] = field(default_factory=list)
    crystallization_summary_row: dict[str, Any] = field(default_factory=dict)
    density_context_row: dict[str, Any] = field(default_factory=dict)
    n_bonds: int | None = None
    n_candidates: int | None = None

    reason_codes: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    confidence_inputs_missing_reason: str = ""
