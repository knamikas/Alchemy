"""Keep process-boundary models independent from worker execution imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Collection

from _version import __version__
from codes import EntryStatus

if TYPE_CHECKING:
    from coordination.schema import BondRow, CandidateRow
    from output_rows import MetalStatsRow


MODEL_POLICY = "first"
ALTLOC_POLICY = "highest-mean-occupancy-residue-conformer"
SYMMETRY_POLICY = (
    "image-inclusive-primary-with-crystallographic-and-strict-ncs-provenance"
)

# A few metal-dense assemblies contribute thousands of correlated sites and
# can dominate a database reference built from otherwise small entries.
MAX_ANALYZED_METAL_SITES = 100


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable so every worker in a run receives the same configuration."""

    root: str
    mirror_root: str
    cache_root: str
    env: dict[str, str]
    output_dir: str
    cofactors: Collection[str]
    keep: bool
    bonds: bool
    density_map_scope: str
    ccp4_timeout_s: int
    log_level: int
    allow_download: bool
    manual_inputs: dict[str, str | None] | None
    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str
    reference_data_id: str


def blank_if_unmeasured(value: Any) -> Any:
    """Prevent an unmeasured value from looking measured after serialization."""
    return "" if value is None else value


def _empty_result_codes() -> list[str]:
    return []


def _empty_result_timings() -> dict[str, float]:
    return {}


def _empty_metal_stats_rows() -> list[MetalStatsRow]:
    return []


def _empty_bond_rows() -> list[BondRow]:
    return []


def _empty_candidate_rows() -> list[CandidateRow]:
    return []


@dataclass(slots=True)
class EntryResult:
    """Keep an unrun stage distinguishable from one that measured zero."""

    pdb_id: str
    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str
    reference_data_id: str
    refinement_state: str

    status: EntryStatus = EntryStatus.ERROR
    retryable: bool = True
    n_metals: int = 0
    runtime_s: float = 0.0
    error: str = ""
    no_metals: bool = False
    metal_site_limit_exceeded: bool = False

    rows: list[MetalStatsRow] = field(default_factory=_empty_metal_stats_rows)
    bond_rows: list[BondRow] = field(default_factory=_empty_bond_rows)
    candidate_rows: list[CandidateRow] = field(default_factory=_empty_candidate_rows)
    n_bonds: int | None = None
    n_candidates: int | None = None

    reason_codes: list[str] = field(default_factory=_empty_result_codes)
    warning_codes: list[str] = field(default_factory=_empty_result_codes)
    timings: dict[str, float] = field(default_factory=_empty_result_timings)

    density_map_scope_used: str = ""
    density_full_map_bytes: int = 0
    density_edstats_map_bytes: int = 0
    confidence_inputs_missing_reason: str = ""
    ccp4_timeout_log_path: str = ""

    alchemy_version: str = __version__
    pdb_redo_version: str = ""
    pdb_redo_date: str = ""
    source_coordinate_format: str = ""
    analysis_coordinate_format: str = "pdb"
    coordinate_conversion_performed: bool = False
    source_coordinate_path: str = ""

    model_policy: str = MODEL_POLICY
    altloc_policy: str = ALTLOC_POLICY
    symmetry_contact_policy: str = SYMMETRY_POLICY
    input_model_count: int | None = None
    model_analyzed: int | None = None
    multi_model_structure: bool | None = None
