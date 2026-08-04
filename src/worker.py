"""Running one PDB-REDO entry, from the worker process that owns it.

``process`` never raises: one bad entry must not take the batch down, so every
failure is recorded in the result as a status, a reason code and a bounded
message. Configuration reaches a worker once through ``_init_worker`` and is
held in ``_CFG``, because a pool initializer cannot return a value and passing
the config with every task would pickle it per entry.
"""

import contextlib
import math
import os
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Optional

from _version import __version__
from bond.bond_analysis import NAN, load_structure, run_bond_analysis
from codes import WarningCode
from bond.bond_schema import STATS_EXTRA_COLUMNS, _check_row_schema, stats_extra_values
from coordinate_conversion import _first_model_pdb
from density_analysis import (
    Ccp4ToolTimeoutError,
    MtzfixValidationError,
    run_density_analysis,
)
from inputs import (
    _first_existing,
    ensure_entry_available,
    entry_dir_for,
    prepare_inputs,
    read_map_column_resolution,
    read_pdb_redo_is_twin,
    read_resolution,
    resolve_manual_inputs,
)
from metal_elements import METAL_ELEMENTS
from metal_identification import extract_metal_statistics
from run_logging import configure_worker_logging, truncate


METALS_SET = set(METAL_ELEMENTS)

MODEL_POLICY = "first"
ALTLOC_POLICY = "highest-mean-occupancy-residue-conformer"
SYMMETRY_POLICY = (
    "image-inclusive-primary-with-crystallographic-and-strict-ncs-provenance"
)

# Reported alongside the machine-readable code in the manifest's error column.
IDENTIFICATION_REASON_MESSAGES = {
    "cofactor_coordinate_join_failed": (
        "cofactor EDSTATS row did not match a coordinate residue"
    ),
    "ambiguous_coordinate_residue_join": (
        "EDSTATS row matched multiple coordinate residues"
    ),
    "cofactor_without_selected_metal": (
        "matched cofactor has no selected configured metal site"
    ),
}

# The manifest's ``error`` column is a summary; the full message is logged.
MAX_MANIFEST_ERROR_CHARS = 300

TIMEOUT_LOG_DIRNAME = "ccp4_timeout_logs"


@dataclass(frozen=True)
class WorkerConfig:
    """Everything one run decides once and every worker then reads.

    The freeze is shallow: ``env`` and ``manual_inputs`` remain mutable dicts,
    and a worker mutating one diverges only its own unpickled copy.
    """

    root: str
    mirror_root: str
    cache_root: Optional[str]
    # Carries CCP4 on PATH, and is passed to every subprocess.
    env: Dict[str, str]
    output_dir: str
    # CCD component ids treated as metal cofactors.
    cofactors: Collection[str]
    # --keep-intermediates: retain each entry's scratch directory.
    keep: bool
    # Whether to run the contact stage, cleared by --no-bonds.
    bonds: bool
    density_map_scope: str
    ccp4_timeout_s: int
    # Level for this worker's logging handler; the driver owns the sink.
    log_level: int
    allow_download: bool
    # The four explicit input paths of a manual run, or ``None`` for a batch.
    manual_inputs: Optional[Dict[str, Optional[str]]]
    # Run provenance stamped into every manifest row.
    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str
    reference_data_id: str


_CFG: Optional[WorkerConfig] = None
_INFLIGHT: Optional[Any] = None


def _init_worker(cfg: WorkerConfig, inflight=None, log_queue=None) -> None:
    global _CFG, _INFLIGHT
    _CFG = cfg
    _INFLIGHT = inflight
    # A forked worker inherits the driver's SIGTERM-to-KeyboardInterrupt
    # handler, and raising there during ``pool.terminate`` interrupts whatever
    # the worker is doing, including the log queue's feeder-thread finalizer.
    with contextlib.suppress(AttributeError, OSError, ValueError):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    configure_worker_logging(log_queue, level=cfg.log_level if cfg else 20)


def _announce_inflight(state: str, pdb_id: str) -> None:
    """Tell the driver which entry this worker process currently holds.

    A worker killed by the OOM killer or a segfault runs no further Python and
    delivers no result, so these notifications are the driver's only record of
    which entry to fail retryably. ``SimpleQueue.put`` writes to the pipe
    synchronously, so a notification sent before the work begins is already
    readable when the process dies mid-entry.
    """
    if _INFLIGHT is None:
        return
    try:
        _INFLIGHT.put((state, os.getpid(), pdb_id))
    except Exception:  # noqa: BLE001 - bookkeeping must never fail an entry
        pass


def _preserve_timeout_log(timeout, pdb_id, output_dir):
    """Copy a timed-out program's partial log out of the scratch directory.

    Only the log: the maps beside it run to hundreds of megabytes per entry.
    Returns ``""`` when there was nothing to copy, and never raises, because
    failing to keep a diagnostic must not change the entry's outcome.
    """
    source = getattr(timeout, "log_path", "")
    if not source or not os.path.isfile(source):
        return ""
    try:
        destination_dir = os.path.join(output_dir, TIMEOUT_LOG_DIRNAME)
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(
            destination_dir, f"{pdb_id}_{timeout.tool}_timeout.log"
        )
        shutil.copyfile(source, destination)
        return destination
    except OSError:
        return ""


def blank_if_unmeasured(value: Any) -> Any:
    """Render a not-yet-measured field as the blank the outputs expect.

    A ``None`` reaching a CSV as the string ``"None"``, or a run log as
    ``null``, would be read by the next ``--resume`` as a real value.
    """
    return "" if value is None else value


@dataclass(slots=True)
class EntryResult:
    """One entry's outcome, with every manifest column present from the outset.

    Not-yet-measured is ``None``, never zero: zero ``n_bonds`` means the bond
    stage ran and found nothing, and a later ``--resume`` reads a non-blank
    count as proof the stage completed and skips the entry permanently.
    """

    # The CSV column keeps the deposited spelling ``pdbID``.
    pdb_id: str
    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str
    # The bundled catalog and distance table this row was measured against;
    # two rows are comparable only if they share it.
    reference_data_id: str
    refinement_state: str

    # ``ok``, ``partial``, ``skip`` or ``error``. Starts at ``error`` so a
    # worker that dies mid-entry cannot be mistaken for a success.
    status: str = "error"
    # Starts true for the same reason, and is cleared once a stage has produced
    # a terminal answer.
    retryable: bool = True
    n_metals: int = 0
    runtime_s: float = 0.0
    error: str = ""
    no_metals: bool = False

    rows: list[dict[str, Any]] = field(default_factory=list)
    bond_rows: list[dict[str, Any]] = field(default_factory=list)
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    # ``None`` until the bond stage runs -- see the class docstring.
    n_bonds: Optional[int] = None
    n_candidates: Optional[int] = None

    reason_codes: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    density_map_scope_used: str = ""
    density_full_map_bytes: int = 0
    density_edstats_map_bytes: int = 0
    confidence_inputs_missing_reason: str = ""
    ccp4_timeout_log_path: str = ""

    alchemy_version: str = __version__
    source_coordinate_format: str = ""
    analysis_coordinate_format: str = "pdb"
    coordinate_conversion_performed: bool = False
    source_coordinate_path: str = ""
    analysis_coordinate_path: str = ""

    # Copied onto every row so a CSV is readable without the code behind it.
    model_policy: str = MODEL_POLICY
    altloc_policy: str = ALTLOC_POLICY
    symmetry_contact_policy: str = SYMMETRY_POLICY
    # ``None`` until the coordinates load.
    input_model_count: Optional[int] = None
    model_analyzed: Optional[int] = None
    multi_model_structure: Optional[bool] = None


def _initial_result(pdb_id, cfg, manual_inputs):
    """Return the per-entry result skeleton, pre-filled with run provenance."""
    return EntryResult(
        pdb_id=pdb_id,
        alchemy_commit=cfg.alchemy_commit,
        gemmi_version=cfg.gemmi_version,
        ccp4_version=cfg.ccp4_version,
        reference_data_id=cfg.reference_data_id,
        refinement_state="manual" if manual_inputs else "final",
    )


def _worker_death_result(pdb_id, cfg, pid):
    """Synthesize the retryable result a killed worker could not return."""
    result = _initial_result(pdb_id, cfg, cfg.manual_inputs)
    result.status = "error"
    result.retryable = True
    result.reason_codes = ["worker_process_died"]
    result.error = (
        f"worker process {pid} terminated without returning a result "
        f"(out-of-memory kill or crash); {pdb_id} was not analyzed"
    )
    return result


def _coordinate_provenance(cfg, source_path):
    manual = cfg.manual_inputs
    if manual:
        converted = bool(manual.get("cif_file"))
        return ("mmcif" if converted else "pdb", "pdb", converted)
    coordinate_name = source_path.lower()
    converted = coordinate_name.endswith((".cif", ".cif.gz"))
    return ("mmcif" if converted else "pdb", "pdb", converted)


def _source_coordinate_path(cfg, pdb_id, entry, analysis_path):
    manual = cfg.manual_inputs
    if manual:
        return manual.get("cif_file") or manual.get("pdb_file") or ""
    return (
        _first_existing(
            os.path.join(entry, f"{pdb_id}_final.cif"),
            os.path.join(entry, f"{pdb_id}_final.cif.gz"),
            os.path.join(entry, f"{pdb_id}_final.pdb"),
            os.path.join(entry, f"{pdb_id}_final.pdb.gz"),
        )
        or analysis_path
    )


def _resolve_entry_dir(pdb_id, cfg):
    """Locate an entry's PDB-REDO directory, downloading it when permitted."""
    if cfg.allow_download:
        used_root = ensure_entry_available(pdb_id, cfg.mirror_root, cfg.cache_root)
        return entry_dir_for(used_root, pdb_id)
    return entry_dir_for(cfg.root, pdb_id)


@dataclass(frozen=True)
class _EntryInputs:
    """What the input stage resolved, and the later stages then read.

    Held together rather than threaded through each stage separately, so that
    the density and bond stages take one entry-shaped argument instead of a
    handful of loose paths and resolution limits.
    """

    work_dir: str
    mtz: str
    # First-model-only coordinates: what the analysis reads, which is not
    # necessarily what was deposited (see ``source_coordinate_path``).
    pdb: str
    # PDB-REDO's per-entry metadata, or a manual run's ``--data-json``, and
    # ``None`` when a manual run supplied none.
    data_json: Optional[str]
    # The diffraction data's own high-resolution limit, as distinct from the
    # map columns' range below.
    data_reshi: Any
    map_reslo: Any
    map_reshi: Any
    pdb_redo_is_twin: bool
    source_coordinate_path: str


def _run_density_stage(result, cfg, inputs, structure):
    """Calculate the maps, run EDSTATS, and extract this entry's statistics.

    Returns ``(rows, header)``, both empty when density could not be produced;
    the reason is recorded on ``result`` in that case, along with whether it is
    worth retrying.
    """
    rows: list[dict[str, Any]] = []
    header: list[str] = []
    density_started = time.monotonic()
    try:
        res = run_density_analysis(
            result.pdb_id,
            inputs.mtz,
            inputs.pdb,
            inputs.work_dir,
            inputs.map_reslo,
            inputs.map_reshi,
            env=cfg.env,
            map_scope=cfg.density_map_scope,
            keep_full_maps=cfg.keep,
            pdb_redo_is_twin=inputs.pdb_redo_is_twin,
            tool_timeout_s=cfg.ccp4_timeout_s,
        )
    except Ccp4ToolTimeoutError as exc:
        # A killed program says nothing about the entry, so unlike a
        # failure exit this is retryable, under its own reason code so a
        # stalled CCP4 install is visible in the manifest.
        rows, header = [], []
        kept_log = _preserve_timeout_log(exc, result.pdb_id, cfg.output_dir)
        result.retryable = True
        result.reason_codes = ["ccp4_tool_timeout"]
        result.error = truncate(f"density unavailable: {exc}", MAX_MANIFEST_ERROR_CHARS)
        result.confidence_inputs_missing_reason = "ccp4_tool_timeout"
        result.ccp4_timeout_log_path = kept_log
        result.timings.update(exc.timings)
    except MtzfixValidationError as exc:
        # MTZFIX could not make the Fourier coefficients internally
        # consistent, which retrying will not change; geometry stays
        # independently assessable.
        rows, header = [], []
        result.retryable = False
        result.reason_codes = ["mtzfix_validation_failure"]
        result.error = truncate(f"density unavailable: {exc}", MAX_MANIFEST_ERROR_CHARS)
        result.confidence_inputs_missing_reason = "mtzfix_validation_failure"
        result.timings.update(exc.timings)
    else:
        result.timings.update(res.timings)
        result.density_map_scope_used = res.density_map_scope_used
        result.density_full_map_bytes = res.full_map_bytes
        result.density_edstats_map_bytes = res.edstats_map_bytes
        if res.twin_coefficient_normalization_applied:
            result.warning_codes = list(
                dict.fromkeys(
                    result.warning_codes
                    + [WarningCode.TWIN_REFMAC_COEFFICIENTS_NORMALIZED]
                )
            )
        statistics_started = time.monotonic()
        rows, header = extract_metal_statistics(
            result.pdb_id,
            res.stats_out,
            METALS_SET,
            cfg.cofactors,
            structure=structure,
        )
        result.timings["statistics_extraction_s"] = round(
            time.monotonic() - statistics_started, 3
        )
        # Inputs and density succeeded; later limitations are terminal.
        result.retryable = False
    finally:
        result.timings["density_total_s"] = round(time.monotonic() - density_started, 3)
    return rows, header


def _identification_reason_codes(rows):
    """Deduplicated reason codes for EDSTATS rows that could not be joined."""
    codes = []
    for row in rows:
        mapping_status = row.get("coordinate_mapping_status", "")
        site_status = row.get("selected_metal_site_status", "")
        if mapping_status == "coordinate_residue_not_found":
            codes.append("cofactor_coordinate_join_failed")
        elif mapping_status == "multiple_coordinate_residues":
            codes.append("ambiguous_coordinate_residue_join")
        elif site_status == "no_selected_metal":
            codes.append("cofactor_without_selected_metal")
    return list(dict.fromkeys(codes))


def _append_site_fields(rows, site_summaries, structure):
    """Extend each EDSTATS row with its per-site contact and provenance values."""
    for index, row in enumerate(rows):
        summary = dict(site_summaries.get(row.get("site_key"), {}))
        for name in (
            "density_observation_id",
            "density_scope",
            "density_shared_site_count",
            "density_is_shared",
        ):
            summary[name] = row.get(name, "")
        summary["coordinate_mapping_status"] = row.get("coordinate_mapping_status", "")
        summary["selected_metal_site_status"] = row.get(
            "selected_metal_site_status", ""
        )
        coverage = summary.get("geometry_coverage_image_inclusive", NAN)
        if isinstance(coverage, float) and not math.isfinite(coverage):
            coverage = summary.get("geometry_coverage_explicit", NAN)
        extra = stats_extra_values(structure, row.get("site"), summary)
        if index == 0:
            _check_row_schema(extra, STATS_EXTRA_COLUMNS, "metal_stats_all.csv")
        row["fields"] = (
            row["fields"]
            + [coverage]
            + [extra[column] for column in STATS_EXTRA_COLUMNS]
        )


def _run_bond_stage(result, cfg, inputs, structure, rows, header):
    """Evaluate the contacts around each site, unless ``--no-bonds`` cleared it.

    Returns ``(bond_rows, candidate_rows, site_summaries, bond_meta)``. A
    bond-stage failure must not lose the EDSTATS rows already computed, so it
    is recorded on ``result`` and the empty defaults are returned instead.
    """
    bond_rows = []
    candidate_rows = []
    site_summaries = {}
    bond_meta = {
        "partial_reason_codes": [],
        "warning_codes": list(structure.warning_codes),
        "messages": [],
        "retryable": False,
    }
    if not cfg.bonds:
        return bond_rows, candidate_rows, site_summaries, bond_meta

    bond_started = time.monotonic()
    try:
        (bond_rows, candidate_rows, site_summaries, bond_meta) = run_bond_analysis(
            result.pdb_id,
            inputs.pdb,
            rows,
            header,
            {
                "data_json": inputs.data_json,
                "pdb_path": inputs.pdb,
                "mtz_path": inputs.mtz,
                "resolution": inputs.data_reshi,
            },
            structure=structure,
            connection_path=inputs.source_coordinate_path,
        )
    except Exception as e:  # noqa: BLE001
        result.error = truncate(
            f"bond: {type(e).__name__}: {e}", MAX_MANIFEST_ERROR_CHARS
        )
        result.reason_codes = list(
            dict.fromkeys(result.reason_codes + ["bond_stage_failure"])
        )
        result.retryable = True
    finally:
        result.timings["bond_analysis_s"] = round(time.monotonic() - bond_started, 3)
    return bond_rows, candidate_rows, site_summaries, bond_meta


def _finalize_result(result, identification_codes, bond_meta, structure):
    """Merge the stage outcomes into the final status, codes, and counts.

    The rows each stage produced are already on ``result``; only what no stage
    could record there is passed in.
    """
    result.reason_codes = list(
        dict.fromkeys(
            result.reason_codes
            + identification_codes
            + list(bond_meta["partial_reason_codes"])
        )
    )
    messages = [IDENTIFICATION_REASON_MESSAGES[code] for code in identification_codes]
    messages.extend(bond_meta["messages"])
    if messages:
        existing_error = result.error
        result.error = "; ".join(
            ([existing_error] if existing_error else []) + messages
        )
        result.error = truncate(result.error, MAX_MANIFEST_ERROR_CHARS)
    if bond_meta.get("retryable", False):
        result.retryable = True
    result.warning_codes = list(
        dict.fromkeys(result.warning_codes + bond_meta.get("warning_codes", []))
    )
    status = "partial" if result.reason_codes else "ok"
    if status == "ok":
        result.retryable = False
    result.status = status
    # Coordinate-model metal sites, not emitted statistics rows: a failed
    # EDSTATS join leaves a diagnostic row without a site even where bond
    # analysis found and evaluated the deposited metal.
    result.n_metals = len(structure.metal_atoms(METALS_SET, canonical=True))
    result.n_bonds = len(result.bond_rows)
    result.n_candidates = len(result.candidate_rows)


def process(pdb_id):
    """Run one entry in an initialized worker and return its result."""
    cfg = _CFG
    if cfg is None:
        raise RuntimeError("worker configuration has not been initialized")
    t0 = time.monotonic()
    # Only a directory created by this invocation may be removed in ``finally``:
    # a predictable <output-dir>/<pdbID> path could already hold user data.
    work_dir: Optional[str] = None
    manual_inputs = cfg.manual_inputs
    data_json = None
    result = _initial_result(pdb_id, cfg, manual_inputs)
    _announce_inflight("start", pdb_id)
    try:
        if manual_inputs:
            work_dir = tempfile.mkdtemp(
                prefix=f".alchemy-{pdb_id}-", dir=cfg.output_dir
            )
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
            # Resolved before any scratch space exists, so a missing entry
            # leaves no temporary directory behind.
            entry = _resolve_entry_dir(pdb_id, cfg)
            if not os.path.isdir(entry):
                result.status = "skip"
                result.error = "entry dir missing"
                return result
            work_dir = tempfile.mkdtemp(
                prefix=f".alchemy-{pdb_id}-", dir=cfg.output_dir
            )
            mtz, pdb = prepare_inputs(pdb_id, entry, work_dir)
            data_reshi = read_resolution(entry, mtz)
        density_data_json = (
            data_json if manual_inputs else os.path.join(entry, "data.json")
        )
        pdb_redo_is_twin = read_pdb_redo_is_twin(density_data_json)
        map_reslo, map_reshi = read_map_column_resolution(mtz)
        source_pdb = pdb
        source_coordinate_path = _source_coordinate_path(cfg, pdb_id, entry, source_pdb)
        source_format, analysis_format, converted = _coordinate_provenance(
            cfg, source_coordinate_path
        )
        model1_pdb = os.path.join(work_dir, f"{pdb_id}_model1.pdb")
        if os.path.realpath(model1_pdb) == os.path.realpath(source_pdb):
            model1_pdb = os.path.join(work_dir, f"{pdb_id}_analysis_model1.pdb")
        pdb, input_model_count = _first_model_pdb(source_pdb, model1_pdb)
        result.source_coordinate_format = source_format
        result.analysis_coordinate_format = analysis_format
        result.coordinate_conversion_performed = converted
        result.source_coordinate_path = source_coordinate_path
        result.analysis_coordinate_path = pdb
        inputs = _EntryInputs(
            work_dir=work_dir,
            mtz=mtz,
            pdb=pdb,
            data_json=density_data_json,
            data_reshi=data_reshi,
            map_reslo=map_reslo,
            map_reshi=map_reshi,
            pdb_redo_is_twin=pdb_redo_is_twin,
            source_coordinate_path=source_coordinate_path,
        )
        structure = load_structure(pdb_id, pdb, source_model_count=input_model_count)
        result.timings["input_structure_s"] = round(time.monotonic() - t0, 3)
        result.analysis_coordinate_format = structure.analysis_coordinate_format
        result.input_model_count = structure.input_model_count
        result.model_analyzed = structure.model_analyzed
        result.multi_model_structure = structure.multi_model_structure
        result.warning_codes = list(structure.warning_codes)
        if not structure.metal_atoms(METALS_SET, canonical=True):
            # Skip two FFT maps and EDSTATS: with no canonical metal site,
            # neither density nor contact analysis can produce output.
            result.status = "ok"
            result.retryable = False
            result.n_metals = 0
            result.rows = []
            result.bond_rows = []
            result.candidate_rows = []
            result.n_bonds = 0
            result.n_candidates = 0
            result.no_metals = True
            return result
        rows, header = _run_density_stage(result, cfg, inputs, structure)
        identification_reason_codes = _identification_reason_codes(rows)
        bond_rows, candidate_rows, site_summaries, bond_meta = _run_bond_stage(
            result, cfg, inputs, structure, rows, header
        )
        _append_site_fields(rows, site_summaries, structure)
        result.rows = rows
        result.bond_rows = bond_rows
        result.candidate_rows = candidate_rows
        _finalize_result(result, identification_reason_codes, bond_meta, structure)
    except FileNotFoundError as e:
        result.status = "skip"
        result.retryable = True
        result.reason_codes = ["missing_input"]
        result.error = truncate(f"missing input: {e}", MAX_MANIFEST_ERROR_CHARS)
    except Exception as e:  # noqa: BLE001 - one bad entry must not kill the batch
        result.status = "error"
        result.retryable = True
        result.reason_codes = ["unexpected_processing_error"]
        result.error = truncate(f"{type(e).__name__}: {e}", MAX_MANIFEST_ERROR_CHARS)
    finally:
        if not cfg.keep and work_dir is not None and os.path.isdir(work_dir):
            cleanup_started = time.monotonic()
            shutil.rmtree(work_dir, ignore_errors=True)
            result.timings["cleanup_s"] = round(time.monotonic() - cleanup_started, 3)
        result.runtime_s = round(time.monotonic() - t0, 2)
        _announce_inflight("end", pdb_id)
    return result
