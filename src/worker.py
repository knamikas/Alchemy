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
liveness protocol whose driver half is ``main._drain_inflight``.

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
from typing import Any, Dict, Optional

from _version import __version__
from bond_analysis import (
    NAN,
    STATS_EXTRA_COLUMNS,
    load_structure,
    run_bond_analysis,
    stats_extra_values,
)
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

#: The configuration a pool worker was initialized with, and the queue it
#: reports its in-flight entry on. Both are set once per worker process by
#: ``_init_worker`` and are read-only thereafter.
_CFG: Optional[Dict[str, Any]] = None
_INFLIGHT: Optional[Any] = None


def _init_worker(cfg: Dict[str, Any], inflight=None, log_queue=None) -> None:
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
    configure_worker_logging(log_queue, level=(cfg or {}).get("log_level", 20))


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


def _initial_result(pdb_id, cfg, manual_inputs):
    """Return the per-entry result skeleton, pre-filled with run provenance.

    Every manifest column is present from the outset so a failure at any stage
    still yields a complete row.

    ``n_bonds`` and ``n_candidates`` start blank rather than zero. Zero is a
    measured result meaning the bond stage ran and found nothing, so an entry
    that fails before that stage must not claim it: a later ``--resume`` reads
    a non-blank count as proof the stage completed and would skip the entry
    permanently.
    """
    return {
        "pdbID": pdb_id,
        "status": "error",
        "n": 0,
        "runtime": 0.0,
        "error": "",
        "rows": [],
        "bond_rows": [],
        "candidate_rows": [],
        "n_bonds": "",
        "n_candidates": "",
        "no_metals": False,
        "timings": {},
        "density_map_scope_used": "",
        "density_full_map_bytes": 0,
        "density_edstats_map_bytes": 0,
        "retryable": True,
        "reason_codes": [],
        "warning_codes": [],
        "confidence_inputs_missing_reason": "",
        "ccp4_timeout_log_path": "",
        "alchemy_version": __version__,
        "alchemy_commit": cfg["alchemy_commit"],
        "gemmi_version": cfg["gemmi_version"],
        "ccp4_version": cfg["ccp4_version"],
        "refinement_state": "manual" if manual_inputs else "final",
        "source_coordinate_format": "",
        "analysis_coordinate_format": "pdb",
        "coordinate_conversion_performed": False,
        "source_coordinate_path": "",
        "analysis_coordinate_path": "",
        "model_policy": MODEL_POLICY,
        "input_model_count": "",
        "model_analyzed": "",
        "multi_model_structure": "",
        "altloc_policy": ALTLOC_POLICY,
        "symmetry_contact_policy": SYMMETRY_POLICY,
    }


def _worker_death_result(pdb_id, cfg, pid):
    """Synthesize the retryable result a killed worker could not return."""
    result = _initial_result(pdb_id, cfg, cfg.get("manual_inputs"))
    result.update(
        status="error",
        retryable=True,
        reason_codes=["worker_process_died"],
        error=(
            f"worker process {pid} terminated without returning a result "
            f"(out-of-memory kill or crash); {pdb_id} was not analyzed"
        ),
    )
    return result


def _coordinate_provenance(cfg, source_path):
    manual = cfg.get("manual_inputs")
    if manual:
        converted = bool(manual.get("cif_file"))
        return ("mmcif" if converted else "pdb", "pdb", converted)
    coordinate_name = source_path.lower()
    converted = coordinate_name.endswith((".cif", ".cif.gz"))
    return ("mmcif" if converted else "pdb", "pdb", converted)


def _source_coordinate_path(cfg, pdb_id, entry, analysis_path):
    manual = cfg.get("manual_inputs")
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
    if cfg["allow_download"]:
        used_root = ensure_entry_available(
            pdb_id, cfg["mirror_root"], cfg["cache_root"]
        )
        return entry_dir_for(used_root, pdb_id)
    return entry_dir_for(cfg["root"], pdb_id)


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


def _check_row_schema(row, columns, name):
    """Fail loudly when a row builder and its CSV schema have drifted apart.

    Rows are written by projecting them onto a fixed column list, so a key the
    builder gained without a matching column would be dropped silently and a
    column it lost would surface only as a bare KeyError.
    """
    expected = set(columns)
    if row.keys() == expected:
        return
    details = []
    missing = sorted(expected - row.keys())
    unexpected = sorted(row.keys() - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise RuntimeError(
        f"{name} row does not match its column schema: " + "; ".join(details)
    )


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
    result["reason_codes"] = list(
        dict.fromkeys(
            result["reason_codes"]
            + identification_codes
            + list(bond_meta["partial_reason_codes"])
        )
    )
    messages = [IDENTIFICATION_REASON_MESSAGES[code] for code in identification_codes]
    messages.extend(bond_meta["messages"])
    if messages:
        existing_error = result["error"]
        result["error"] = "; ".join(
            ([existing_error] if existing_error else []) + messages
        )
        result["error"] = truncate(result["error"], MAX_MANIFEST_ERROR_CHARS)
    if bond_meta.get("retryable", False):
        result["retryable"] = True
    result["warning_codes"] = list(
        dict.fromkeys(result["warning_codes"] + bond_meta.get("warning_codes", []))
    )
    status = "partial" if result["reason_codes"] else "ok"
    if status == "ok":
        result["retryable"] = False
    # Count coordinate-model metal sites, not emitted statistics rows. A failed
    # EDSTATS join can leave a diagnostic row without a site even though bond
    # analysis still found and evaluated the deposited metal.
    result.update(
        status=status,
        n=len(structure.metal_atoms(METALS_SET, canonical=True)),
        rows=rows,
        bond_rows=bond_rows,
        candidate_rows=candidate_rows,
        n_bonds=len(bond_rows),
        n_candidates=len(candidate_rows),
    )


def process(pdb_id):
    """Run one initialized worker entry and return its result dictionary."""
    cfg = _CFG
    if cfg is None:
        raise RuntimeError("worker configuration has not been initialized")
    t0 = time.monotonic()
    # Only a directory created by this invocation may be removed in ``finally``.
    # A predictable <output-dir>/<pdbID> path could already contain user data.
    work_dir: Optional[str] = None
    manual_inputs = cfg.get("manual_inputs")
    data_json = None
    result = _initial_result(pdb_id, cfg, manual_inputs)
    _announce_inflight("start", pdb_id)
    try:
        if manual_inputs:
            work_dir = tempfile.mkdtemp(
                prefix=f".alchemy-{pdb_id}-", dir=cfg["output_dir"]
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
                result.update(status="skip", error="entry dir missing")
                return result
            work_dir = tempfile.mkdtemp(
                prefix=f".alchemy-{pdb_id}-", dir=cfg["output_dir"]
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
        result.update(
            source_coordinate_format=source_format,
            analysis_coordinate_format=analysis_format,
            coordinate_conversion_performed=converted,
            source_coordinate_path=source_coordinate_path,
            analysis_coordinate_path=pdb,
        )
        structure = load_structure(pdb_id, pdb, source_model_count=input_model_count)
        result["timings"]["input_structure_s"] = round(time.monotonic() - t0, 3)
        result.update(
            analysis_coordinate_format=structure.analysis_coordinate_format,
            input_model_count=structure.input_model_count,
            model_analyzed=structure.model_analyzed,
            multi_model_structure=structure.multi_model_structure,
            warning_codes=list(structure.warning_codes),
        )
        if not structure.metal_atoms(METALS_SET, canonical=True):
            # Density and contact analysis cannot produce metal-site output for
            # this structure. Avoid two FFT maps and EDSTATS when there is no
            # canonical metal site to assess.
            result.update(
                status="ok",
                retryable=False,
                n=0,
                rows=[],
                bond_rows=[],
                candidate_rows=[],
                n_bonds=0,
                n_candidates=0,
                no_metals=True,
            )
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
                env=cfg["env"],
                map_scope=cfg["density_map_scope"],
                keep_full_maps=cfg["keep"],
                pdb_redo_is_twin=pdb_redo_is_twin,
                tool_timeout_s=cfg["ccp4_timeout_s"],
            )
        except Ccp4ToolTimeout as exc:
            # The program told us nothing about the entry -- it was killed for
            # running too long -- so this is retryable, unlike a failure exit.
            # It gets its own reason code rather than falling through to the
            # generic handler, so a stalled CCP4 install is visible in the
            # manifest instead of looking like an unexplained crash.
            rows, header = [], []
            kept_log = _preserve_timeout_log(exc, pdb_id, cfg["output_dir"])
            result.update(
                retryable=True,
                reason_codes=["ccp4_tool_timeout"],
                error=truncate(f"density unavailable: {exc}", MAX_MANIFEST_ERROR_CHARS),
                confidence_inputs_missing_reason="ccp4_tool_timeout",
                ccp4_timeout_log_path=kept_log,
            )
            result["timings"].update(exc.timings)
        except MtzfixValidationError as exc:
            # The input is readable, but MTZFIX could not make its Fourier
            # coefficients internally consistent. Do not use those maps or
            # retry forever. Geometry remains independently assessable.
            rows, header = [], []
            result.update(
                retryable=False,
                reason_codes=["mtzfix_validation_failure"],
                error=truncate(f"density unavailable: {exc}", MAX_MANIFEST_ERROR_CHARS),
                confidence_inputs_missing_reason=("mtzfix_validation_failure"),
            )
            result["timings"].update(exc.timings)
        else:
            result["timings"].update(res.get("timings", {}))
            result.update(
                density_map_scope_used=res["density_map_scope_used"],
                density_full_map_bytes=res["full_map_bytes"],
                density_edstats_map_bytes=res["edstats_map_bytes"],
            )
            if res.get("twin_coefficient_normalization_applied"):
                result["warning_codes"] = list(
                    dict.fromkeys(
                        result["warning_codes"]
                        + ["twin_refmac_coefficients_normalized"]
                    )
                )
            statistics_started = time.monotonic()
            rows, header = extract_metal_statistics(
                pdb_id,
                res["stats_out"],
                METALS_SET,
                cfg["cofactors"],
                structure=structure,
            )
            result["timings"]["statistics_extraction_s"] = round(
                time.monotonic() - statistics_started, 3
            )
            # Reaching this point means the entry's core inputs and density
            # stage succeeded. Later deterministic limitations remain terminal.
            result["retryable"] = False
        finally:
            result["timings"]["density_total_s"] = round(
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

        if cfg["bonds"]:
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
                result["error"] = truncate(
                    f"bond: {type(e).__name__}: {e}", MAX_MANIFEST_ERROR_CHARS
                )
                result["reason_codes"] = list(
                    dict.fromkeys(result["reason_codes"] + ["bond_stage_failure"])
                )
                result["retryable"] = True
            finally:
                result["timings"]["bond_analysis_s"] = round(
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
        result.update(
            status="skip",
            retryable=True,
            reason_codes=["missing_input"],
            error=truncate(f"missing input: {e}", MAX_MANIFEST_ERROR_CHARS),
        )
    except Exception as e:  # noqa: BLE001 - one bad entry must not kill the batch
        result.update(
            status="error",
            retryable=True,
            reason_codes=["unexpected_processing_error"],
            error=truncate(f"{type(e).__name__}: {e}", MAX_MANIFEST_ERROR_CHARS),
        )
    finally:
        if not cfg["keep"] and work_dir is not None and os.path.isdir(work_dir):
            cleanup_started = time.monotonic()
            shutil.rmtree(work_dir, ignore_errors=True)
            result["timings"]["cleanup_s"] = round(
                time.monotonic() - cleanup_started, 3
            )
        result["runtime"] = round(time.monotonic() - t0, 2)
        _announce_inflight("end", pdb_id)
    return result
