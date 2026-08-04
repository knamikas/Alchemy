"""Running one PDB-REDO entry, from the worker process that owns it.

``process`` is the unit of work a pool worker executes: prepare the entry's
inputs, run the density stage, extract EDSTATS statistics for its metal sites,
run contact analysis, and return one result dictionary. It never raises -- a
single bad entry must not take the batch down with it -- so every failure is
recorded in the result as a status, a reason code, and a bounded message.

Two things here are the driver's, not the worker's. ``_initial_result`` builds
the manifest-complete result skeleton, which the driver also needs to
synthesize ``_worker_death_result`` for a process that was killed before it
could return anything. And ``_announce_inflight`` is the worker half of the
liveness protocol whose driver half is ``driver.pool._drain_inflight``.

Configuration reaches a worker once, through ``_init_worker``, and is held in
this module's ``_CFG``: a pool initializer cannot return a value, and passing
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
from bond_analysis import NAN, load_structure, run_bond_analysis
from codes import WarningCode
from bond_schema import STATS_EXTRA_COLUMNS, _check_row_schema, stats_extra_values
from coordinate_conversion import _first_model_pdb
from density_analysis import (
    Ccp4ToolTimeout,
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

# Human-readable text for each EDSTATS-join reason code, reported alongside the
# machine-readable code in the manifest's error column.
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

#: Longest text the manifest's free-text ``error`` column carries. The column
#: is a human-readable summary, not the full diagnostic: the complete message
#: is logged, and ``truncate`` marks the cut so a shortened one is not mistaken
#: for a complete one.
MAX_MANIFEST_ERROR_CHARS = 300

#: Where a stalled CCP4 program's partial log is copied so it outlives the
#: entry's scratch directory.
TIMEOUT_LOG_DIRNAME = "ccp4_timeout_logs"


@dataclass(frozen=True)
class WorkerConfig:
    """Everything one run decides once and every worker then reads.

    Frozen because it is shared state in the strongest sense: assembled by the
    driver, pickled into each worker's ``_init_worker``, and read from there by
    every stage of ``process``. Nothing in a worker may rebind a field, and a
    frozen dataclass says so where a dict could not.

    **The freeze is shallow, and ``env`` and ``manual_inputs`` are still
    mutable dicts.** ``cfg.env.clear()`` in a worker would succeed. Closing
    that needs an immutable mapping that also survives ``pickle``, which
    ``MappingProxyType`` does not -- spawn workers receive this config by
    pickling it -- so it wants a ``ManualInputs`` dataclass and a picklable
    frozen mapping for ``env``, not a one-line change. The hole is real but
    bounded: each worker unpickles its own copy, so such an edit diverges that
    one worker rather than the run.

    Declaring the fields also puts the shape in one place. Two test modules
    used to hand-build the same dict, and both had already drifted -- neither
    carried ``log_level``, and one was missing ``ccp4_timeout_s`` -- silently,
    because a missing key only fails at the moment some worker reads it.
    """

    #: Where entry directories are read from, and the mirror and cache the
    #: downloader falls back to when ``allow_download`` is set.
    root: str
    mirror_root: str
    cache_root: Optional[str]
    #: Process environment carrying CCP4 on PATH, passed to every subprocess.
    env: Dict[str, str]
    output_dir: str
    #: Component ids treated as metal cofactors, loaded once by the driver.
    cofactors: Collection[str]
    #: ``--keep-intermediates``: retain each entry's scratch directory.
    keep: bool
    #: ``--bonds``: run the contact stage at all.
    bonds: bool
    density_map_scope: str
    ccp4_timeout_s: int
    #: Level for this worker's logging handler; the driver owns the sink.
    log_level: int
    #: Whether a missing entry may be fetched, which is false for a manual run.
    allow_download: bool
    #: The four explicit input paths of a manual run, or ``None`` for a batch.
    manual_inputs: Optional[Dict[str, Optional[str]]]
    #: Run provenance stamped into every manifest row.
    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str


#: The configuration a pool worker was initialized with, and the queue it
#: reports its in-flight entry on. Both are set once per worker process by
#: ``_init_worker`` and are read-only thereafter.
_CFG: Optional[WorkerConfig] = None
_INFLIGHT: Optional[Any] = None


def _init_worker(cfg: WorkerConfig, inflight=None, log_queue=None) -> None:
    global _CFG, _INFLIGHT
    _CFG = cfg
    _INFLIGHT = inflight
    # A forked worker inherits the driver's SIGTERM handler, which converts the
    # signal into KeyboardInterrupt so the *driver* can unwind and write its run
    # log. A worker has nothing to unwind: ``pool.terminate`` SIGTERMs it during
    # normal shutdown, and raising there interrupts whatever it is doing --
    # including the log queue's feeder-thread finalizer, which printed a
    # spurious traceback on successful runs. Restore the default disposition so
    # a terminated worker simply exits.
    with contextlib.suppress(AttributeError, OSError, ValueError):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # Handlers inherited through fork would have several processes writing to
    # one stream; the queue makes the driver the only writer.
    configure_worker_logging(log_queue, level=cfg.log_level if cfg else 20)


def _announce_inflight(state: str, pdb_id: str) -> None:
    """Tell the driver which entry this worker process currently holds.

    A worker killed by the OOM killer or felled by a segfault in a compiled
    extension runs no further Python, so it cannot report its own death and the
    pool never delivers a result for the task it was running. These
    notifications are the only record of which entry a process held, letting the
    driver attribute a dead process to that entry and fail it retryably instead
    of waiting for a result that can never arrive.

    ``SimpleQueue.put`` writes to the pipe synchronously, so a notification sent
    before the work begins is already readable by the driver when the process is
    killed mid-entry.
    """
    if _INFLIGHT is None:
        return
    try:
        _INFLIGHT.put((state, os.getpid(), pdb_id))
    except Exception:  # noqa: BLE001 - bookkeeping must never fail an entry
        pass


def _preserve_timeout_log(timeout, pdb_id, output_dir):
    """Copy a timed-out program's partial log somewhere it will survive.

    ``process`` deletes the whole scratch directory unless --keep-intermediates
    is given, so the log named in the timeout message would be gone by the time
    anyone read the manifest. Only the log is copied: the maps beside it can run
    to hundreds of megabytes per entry, and a database run that starts timing
    out would fill the output directory with them.

    Returns the retained path, or ``""`` when there was nothing to copy. Never
    raises -- failing to keep a diagnostic must not change the entry's outcome.
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

    ``None`` on an ``EntryResult`` means "this stage never ran", which every
    output has always written as an empty cell. Keep that rendering in one
    place: a ``None`` reaching a CSV as the string ``"None"``, or a run log as
    ``null``, would be read by the next ``--resume`` as a real value.
    """
    return "" if value is None else value


@dataclass(slots=True)
class EntryResult:
    """One entry's outcome, from the skeleton a failure returns to a full run.

    Every manifest column is present from the outset so a failure at any stage
    still yields a complete row, and the worker fills the rest in as its stages
    complete -- which is why this is the one mutable contract in the pipeline.
    Fields are assigned directly rather than through a ``**fields`` helper: a
    helper takes every value as ``Any``, so ``n_bonds="0"`` would pass it and
    then break the blank-versus-measured rule below in a manifest column that
    ``--resume`` reads back. ``slots=True`` closes the other half -- a
    misspelled attribute is an error at the assignment, not a new attribute
    nobody reads.

    **Not-yet-measured is ``None``, and it is not zero.** ``n_bonds`` and
    ``n_candidates`` are the pair that matters: zero is a measured result
    meaning the bond stage ran and found nothing, so an entry that failed
    before that stage must not claim it. A later ``--resume`` reads a non-blank
    count as proof the stage completed and would skip the entry permanently.
    That invariant used to live only in a docstring, with ``""`` standing for
    "unmeasured" in a field otherwise holding integers; the three model fields
    below carry the same distinction for the same reason.
    """

    #: Run provenance, known before any work starts. The CSV column keeps the
    #: deposited spelling ``pdbID``; the identifier here is ``pdb_id``, as
    #: everywhere else in the codebase.
    pdb_id: str
    alchemy_commit: str
    gemmi_version: str
    ccp4_version: str
    refinement_state: str

    #: ``ok``, ``partial``, ``skip`` or ``error``. Starts at ``error`` so a
    #: worker that dies mid-entry cannot be mistaken for a success.
    status: str = "error"
    #: Whether this entry is worth another attempt. Starts true for the same
    #: reason, and is cleared once a stage has produced a terminal answer.
    retryable: bool = True
    n_metals: int = 0
    runtime_s: float = 0.0
    error: str = ""
    no_metals: bool = False

    #: The rows this entry contributes to each CSV.
    rows: list[dict[str, Any]] = field(default_factory=list)
    bond_rows: list[dict[str, Any]] = field(default_factory=list)
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    #: ``None`` until the bond stage runs -- see the class docstring.
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

    #: Fixed statements of what this run does, copied onto every row so a CSV
    #: is readable without the code that produced it.
    model_policy: str = MODEL_POLICY
    altloc_policy: str = ALTLOC_POLICY
    symmetry_contact_policy: str = SYMMETRY_POLICY
    #: ``None`` until the coordinates load.
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


def _finalize_result(
    result, identification_codes, bond_meta, structure, rows, bond_rows, candidate_rows
):
    """Merge the stage outcomes into the final status, codes, and counts."""
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
    # Count coordinate-model metal sites, not emitted statistics rows. A failed
    # EDSTATS join can leave a diagnostic row without a site even though bond
    # analysis still found and evaluated the deposited metal.
    result.status = status
    result.n_metals = len(structure.metal_atoms(METALS_SET, canonical=True))
    result.rows = rows
    result.bond_rows = bond_rows
    result.candidate_rows = candidate_rows
    result.n_bonds = len(bond_rows)
    result.n_candidates = len(candidate_rows)


def process(pdb_id):
    """Run one initialized worker entry and return its result dictionary."""
    cfg = _CFG
    if cfg is None:
        raise RuntimeError("worker configuration has not been initialized")
    t0 = time.monotonic()
    # Only a directory created by this invocation may be removed in ``finally``.
    # A predictable <output-dir>/<pdbID> path could already contain user data.
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
            # Resolved before any scratch space is created, so a missing entry
            # never leaves a temporary directory behind.
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
        structure = load_structure(pdb_id, pdb, source_model_count=input_model_count)
        result.timings["input_structure_s"] = round(time.monotonic() - t0, 3)
        result.analysis_coordinate_format = structure.analysis_coordinate_format
        result.input_model_count = structure.input_model_count
        result.model_analyzed = structure.model_analyzed
        result.multi_model_structure = structure.multi_model_structure
        result.warning_codes = list(structure.warning_codes)
        if not structure.metal_atoms(METALS_SET, canonical=True):
            # Density and contact analysis cannot produce metal-site output for
            # this structure. Avoid two FFT maps and EDSTATS when there is no
            # canonical metal site to assess.
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
        density_started = time.monotonic()
        try:
            res = run_density_analysis(
                pdb_id,
                mtz,
                pdb,
                work_dir,
                map_reslo,
                map_reshi,
                env=cfg.env,
                map_scope=cfg.density_map_scope,
                keep_full_maps=cfg.keep,
                pdb_redo_is_twin=pdb_redo_is_twin,
                tool_timeout_s=cfg.ccp4_timeout_s,
            )
        except Ccp4ToolTimeout as exc:
            # The program told us nothing about the entry -- it was killed for
            # running too long -- so this is retryable, unlike a failure exit.
            # It gets its own reason code rather than falling through to the
            # generic handler, so a stalled CCP4 install is visible in the
            # manifest instead of looking like an unexplained crash.
            rows, header = [], []
            kept_log = _preserve_timeout_log(exc, pdb_id, cfg.output_dir)
            result.retryable = True
            result.reason_codes = ["ccp4_tool_timeout"]
            result.error = truncate(
                f"density unavailable: {exc}", MAX_MANIFEST_ERROR_CHARS
            )
            result.confidence_inputs_missing_reason = "ccp4_tool_timeout"
            result.ccp4_timeout_log_path = kept_log
            result.timings.update(exc.timings)
        except MtzfixValidationError as exc:
            # The input is readable, but MTZFIX could not make its Fourier
            # coefficients internally consistent. Do not use those maps or
            # retry forever. Geometry remains independently assessable.
            rows, header = [], []
            result.retryable = False
            result.reason_codes = ["mtzfix_validation_failure"]
            result.error = truncate(
                f"density unavailable: {exc}", MAX_MANIFEST_ERROR_CHARS
            )
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
                pdb_id,
                res.stats_out,
                METALS_SET,
                cfg.cofactors,
                structure=structure,
            )
            result.timings["statistics_extraction_s"] = round(
                time.monotonic() - statistics_started, 3
            )
            # Reaching this point means the entry's core inputs and density
            # stage succeeded. Later deterministic limitations remain terminal.
            result.retryable = False
        finally:
            result.timings["density_total_s"] = round(
                time.monotonic() - density_started, 3
            )

        identification_reason_codes = _identification_reason_codes(rows)

        bond_rows = []
        candidate_rows = []
        site_summaries = {}
        bond_meta = {
            "partial_reason_codes": [],
            "warning_codes": list(structure.warning_codes),
            "messages": [],
            "retryable": False,
        }

        if cfg.bonds:
            # A bond-stage failure must not lose the edstats rows already computed.
            bond_started = time.monotonic()
            try:
                (bond_rows, candidate_rows, site_summaries, bond_meta) = (
                    run_bond_analysis(
                        pdb_id,
                        pdb,
                        rows,
                        header,
                        {
                            "data_json": (
                                data_json
                                if manual_inputs
                                else os.path.join(entry, "data.json")
                            ),
                            "pdb_path": pdb,
                            "mtz_path": mtz,
                            "resolution": data_reshi,
                        },
                        structure=structure,
                        connection_path=source_coordinate_path,
                    )
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
                result.timings["bond_analysis_s"] = round(
                    time.monotonic() - bond_started, 3
                )
        _append_site_fields(rows, site_summaries, structure)
        _finalize_result(
            result,
            identification_reason_codes,
            bond_meta,
            structure,
            rows,
            bond_rows,
            candidate_rows,
        )
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
