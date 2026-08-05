# Alchemy
# CCP4-based map calculation + edstats real-space statistics for one structure.
#
# Requires the CCP4 binaries `mtzfix`, `fft`, `mapmask`, and `edstats` on PATH
# (pass `env=` to point at a sourced CCP4 environment). The input MTZ must carry
# the FWT/PHWT/DELFWT/PHDELWT map-coefficient columns (PDB-REDO _final.mtz and
# refmac output have them).
import os
import subprocess
import shutil
import struct
import tempfile
import time

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any, Optional

import gemmi
import numpy as np
import numpy.typing as npt

from run_logging import logger_for, truncate

logger = logger_for(__name__)


MODEL_ENVELOPE_BORDER_ANGSTROM = 10

# Roughly five times the worst step observed across the July 2026 database runs
# (EDSTATS 185.7 s), so a genuine hang is bounded without killing a very large
# structure.
CCP4_TOOL_TIMEOUT_S = 15 * 60
DENSITY_MAP_SCOPES = ("model-envelope", "full")
REFMAC_TWIN_COLUMNS = {
    "FP": "F",
    "SIGFP": "Q",
    "FC_ALL": "F",
    "PHIC_ALL": "P",
    "FWT": "F",
    "PHWT": "P",
    "DELFWT": "F",
    "PHDELWT": "P",
    "FOM": "W",
}
REFMAC_TWIN_IDENTITY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class DensityResult:
    """What one entry's density stage produced, and how it produced it."""

    # stats_out is the only output the pipeline parses; the remaining paths
    # exist so the debug CLI and --keep-intermediates can name what was written.
    stats_out: str
    rszd: str
    fo_map: str
    df_map: str
    mtz_for_maps: str
    mtzfix_log: str
    mtzfix_applied: bool
    timings: dict[str, Any]
    twin_coefficient_normalization_applied: bool
    twin_coefficient_normalization: Optional[dict[str, Any]]
    density_map_scope_requested: str
    density_map_scope_used: str
    full_map_bytes: int
    edstats_map_bytes: int


class MtzfixValidationError(RuntimeError):
    """MTZFIX could not make map coefficients pass its consistency checks."""

    def __init__(
        self, message: str, timings: Optional[Mapping[str, float]] = None
    ) -> None:
        super().__init__(message)
        self.timings = dict(timings or {})


class Ccp4ToolTimeoutError(RuntimeError):
    """A CCP4 program ran past its time budget and was killed."""

    def __init__(
        self,
        tool: str,
        timeout_s: float,
        elapsed_s: float,
        log_path: str,
        timings: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__(
            f"{tool} exceeded its {timeout_s:g}s time budget "
            f"(killed after {elapsed_s:.1f}s); partial log kept at {log_path}"
        )
        self.tool = tool
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s
        self.log_path = log_path
        self.timings = dict(timings or {})


def _decoded_stderr(error_file: IO[bytes]) -> str:
    """Read a CCP4 program's captured stderr, tolerating any byte it wrote.

    Replacement decoding is deliberate: this text only ever reaches a log line
    or a manifest error string, so an undecodable byte should cost a character
    rather than the entry.
    """
    try:
        error_file.seek(0)
        return error_file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _complex_coefficients(
    amplitudes: npt.NDArray[np.float64], degree_phases: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex128]:
    # NumPy's stubs give every ufunc result dtype ``Any``, so the product below
    # is inferred as its float64 left operand rather than the complex128 it is.
    return amplitudes * np.exp(  # type: ignore[return-value]
        1j * np.deg2rad(degree_phases)
    )


def _amplitudes_and_degree_phases(
    coefficients: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    amplitudes = np.abs(coefficients)
    phases = np.rad2deg(np.angle(coefficients))
    phases[amplitudes == 0.0] = 0.0
    return amplitudes, phases


def _coefficient_residual_ratio(
    observed: npt.NDArray[np.complex128],
    expected: npt.NDArray[np.complex128],
    *scale_terms: npt.NDArray[np.complex128],
) -> float:
    """Largest scale-relative complex residual, with a 1-electron floor.

    ``scale_terms`` matter when ``observed`` is itself the small difference of
    two large coefficients. MTZ stores amplitudes and phases as float32, so the
    attainable precision of that subtraction is set by the original terms,
    not by the size of their nearly cancelled result.
    """
    scale = np.maximum.reduce(
        (
            np.abs(observed),
            np.abs(expected),
            np.ones(observed.shape),
            *(np.abs(term) for term in scale_terms),
        )
    )
    return float(np.max(np.abs(observed - expected) / scale))


def normalize_refmac_twin_coefficients(
    mtz_path: str, output_path: str
) -> dict[str, Any]:
    """Write a guarded Refmac-to-EDSTATS coefficient conversion.

    REFMAC writes both acentric and centric map coefficients using
    ``FWT = 2mFo-DFc`` and ``DELFWT = mFo-DFc``. EDSTATS instead requires the
    literature convention: ``2mFo-DFc`` and ``2(mFo-DFc)`` for acentric
    reflections, but ``mFo`` and ``mFo-DFc`` for centric reflections. Complex
    subtraction from ``FC_ALL`` performs that normalization without
    reconstructing m or Fo, which the untwinned identities behind MTZFIX's
    consistency re-test cannot do.
    """
    mtz_path = os.fspath(mtz_path)
    output_path = os.fspath(output_path)
    if os.path.realpath(mtz_path) == os.path.realpath(output_path):
        raise ValueError("twin coefficient output would overwrite its input")

    mtz = gemmi.read_mtz_file(mtz_path)
    if "refmac" not in (mtz.title or "").lower():
        raise ValueError("MTZ title does not identify Refmac output")
    if mtz.spacegroup is None:
        raise ValueError("MTZ has no space group for centric-reflection testing")

    columns = {}
    for label, expected_type in REFMAC_TWIN_COLUMNS.items():
        matches = [column for column in mtz.columns if column.label == label]
        if len(matches) != 1:
            raise ValueError(
                f"MTZ must contain exactly one {label} column; found {len(matches)}"
            )
        column = matches[0]
        if column.type != expected_type:
            raise ValueError(
                f"MTZ column {label} has type {column.type}, expected {expected_type}"
            )
        columns[label] = column

    coefficient_dataset_ids = {
        columns[label].dataset_id
        for label in ("FC_ALL", "PHIC_ALL", "FWT", "PHWT", "DELFWT", "PHDELWT")
    }
    if len(coefficient_dataset_ids) != 1:
        raise ValueError("Refmac map coefficients belong to different datasets")

    source = {
        label: np.asarray(columns[label].array, dtype=np.float64)
        for label in ("FC_ALL", "PHIC_ALL", "FWT", "PHWT", "DELFWT", "PHDELWT")
    }
    map_finite = np.ones(mtz.nreflections, dtype=bool)
    for label in ("FWT", "PHWT", "DELFWT", "PHDELWT"):
        map_finite &= np.isfinite(source[label])
    model_finite = np.isfinite(source["FC_ALL"]) & np.isfinite(source["PHIC_ALL"])
    if np.any(map_finite & ~model_finite):
        raise ValueError("finite Refmac map coefficients have missing FC_ALL values")
    usable = map_finite & model_finite
    usable_count = int(np.count_nonzero(usable))
    if usable_count == 0:
        raise ValueError("Refmac map coefficients have no common finite values")

    observed = _complex_coefficients(source["FWT"][usable], source["PHWT"][usable])
    calculated = _complex_coefficients(
        source["FC_ALL"][usable], source["PHIC_ALL"][usable]
    )
    difference = _complex_coefficients(
        source["DELFWT"][usable], source["PHDELWT"][usable]
    )
    raw_identity_residual = _coefficient_residual_ratio(
        observed - calculated, 2.0 * difference, observed, calculated
    )
    if raw_identity_residual > REFMAC_TWIN_IDENTITY_TOLERANCE:
        raise ValueError(
            "Refmac coefficients do not satisfy the guarded raw identity "
            f"(maximum relative residual {raw_identity_residual:.6g})"
        )

    centric = np.asarray(
        mtz.spacegroup.operations().centric_flag_array(mtz.make_miller_array()),
        dtype=bool,
    )[usable]
    normalized_observed = observed.copy()
    normalized_observed[centric] = (observed[centric] + calculated[centric]) / 2.0
    normalized_difference = normalized_observed - calculated
    fwt, phwt = _amplitudes_and_degree_phases(normalized_observed)
    delfwt, phdelwt = _amplitudes_and_degree_phases(normalized_difference)

    output_data = np.array(mtz.array, copy=True)
    for label, values in (
        ("FWT", fwt),
        ("PHWT", phwt),
        ("DELFWT", delfwt),
        ("PHDELWT", phdelwt),
    ):
        output_data[usable, columns[label].idx] = values
    mtz.set_data(output_data)
    mtz.history = [
        *mtz.history,
        "Alchemy: normalized twin Refmac map coefficients for EDSTATS",
    ]
    mtz.write_to_file(output_path)

    # EDSTATS relies on observed minus difference being the calculated map, so
    # verify that on the written file rather than on the in-memory arrays.
    written = gemmi.read_mtz_file(output_path)
    written_values = {
        label: np.asarray(written.column_with_label(label).array, dtype=np.float64)[
            usable
        ]
        for label in ("FC_ALL", "PHIC_ALL", "FWT", "PHWT", "DELFWT", "PHDELWT")
    }
    written_observed = _complex_coefficients(
        written_values["FWT"], written_values["PHWT"]
    )
    written_calculated = _complex_coefficients(
        written_values["FC_ALL"], written_values["PHIC_ALL"]
    )
    written_difference = _complex_coefficients(
        written_values["DELFWT"], written_values["PHDELWT"]
    )
    output_identity_residual = _coefficient_residual_ratio(
        written_observed - written_difference,
        written_calculated,
        written_observed,
        written_difference,
    )
    if output_identity_residual > REFMAC_TWIN_IDENTITY_TOLERANCE:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise ValueError(
            "written EDSTATS coefficients failed their identity check "
            f"(maximum relative residual {output_identity_residual:.6g})"
        )

    return {
        "usable_reflections": usable_count,
        "centric_reflections": int(np.count_nonzero(centric)),
        "raw_identity_max_relative_residual": raw_identity_residual,
        "output_identity_max_relative_residual": output_identity_residual,
    }


def run_density_analysis(
    pdb_id: str,
    mtz_path: str,
    pdb_path: str,
    out_dir: str,
    reslo: float,
    reshi: float,
    env: Optional[Mapping[str, str]] = None,
    map_scope: str = "model-envelope",
    keep_full_maps: bool = False,
    pdb_redo_is_twin: bool = False,
    tool_timeout_s: float = CCP4_TOOL_TIMEOUT_S,
) -> DensityResult:
    """Validate the MTZ, compute maps with `fft`, then run `edstats`.

    ``reslo``/``reshi`` are the common low/high limits of the four map columns,
    and ``pdb_redo_is_twin`` must be the exact PDB-REDO ``properties.ISTWIN``
    value. ``tool_timeout_s`` is a budget per CCP4 program, not per entry.
    """
    if map_scope not in DENSITY_MAP_SCOPES:
        raise ValueError(
            f"density map scope must be one of {', '.join(DENSITY_MAP_SCOPES)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    fixed_mtz = os.path.join(out_dir, f"{pdb_id}_mtzfix.mtz")
    mtzfix_log = os.path.join(out_dir, f"{pdb_id}_mtzfix.log")
    fo_map = os.path.join(out_dir, f"{pdb_id}_fo.map")
    df_map = os.path.join(out_dir, f"{pdb_id}_df.map")
    stats_out = os.path.join(out_dir, f"{pdb_id}_stats.out")
    rszd = os.path.join(out_dir, f"{pdb_id}_rszd.pdb")
    qq_out = os.path.join(out_dir, f"{pdb_id}_qq.out")
    twin_normalized_mtz = os.path.join(out_dir, f"{pdb_id}_twin_edstats.mtz")

    timings: dict[str, float] = {}

    def _run(
        cmd: Sequence[str],
        stdin: Optional[str],
        logname: str,
        timing_name: str,
        timeout_s: float = tool_timeout_s,
    ) -> None:
        log_path = os.path.join(out_dir, logname)
        resolved = shutil.which(cmd[0], path=(env or {}).get("PATH"))
        if resolved is None:
            raise RuntimeError(
                f"required CCP4 program {cmd[0]!r} was not found on PATH"
            )
        cmd = [resolved, *cmd[1:]]
        program = os.path.basename(resolved)
        logger.debug("%s: running %s (budget %gs)", pdb_id, program, timeout_s)
        started = time.monotonic()
        try:
            # stderr is captured to a file rather than a pipe. Decoding a pipe
            # under ``text=True`` is strict, so a single non-UTF-8 byte in a
            # Fortran runtime notice would raise before the exit status was read
            # and lose the entry to a retryable error that always recurs. And
            # with no output pipe to drain, the timeout bounds the process
            # rather than EOF, so a program that leaves a helper holding stderr
            # cannot report a false timeout at the full budget.
            #
            # The program is deliberately left in the worker's process group: a
            # group of its own would make a timeout kill its descendants too,
            # but would also let them escape the group-wide signal the driver
            # uses to stop a worker's CCP4 children at shutdown.
            with (
                open(log_path, "w", encoding="utf-8", errors="replace") as log,
                tempfile.TemporaryFile(dir=out_dir) as error_file,
            ):
                proc = subprocess.run(
                    cmd,
                    input=stdin,
                    text=True,
                    stdout=log,
                    stderr=error_file,
                    env=env,
                    timeout=timeout_s,
                )
                stderr_text = _decoded_stderr(error_file)
        except subprocess.TimeoutExpired as exc:
            logger.error(
                "%s: %s exceeded its %gs budget; partial log kept at %s",
                pdb_id,
                program,
                timeout_s,
                log_path,
            )
            # subprocess.run has already killed and reaped the child; the
            # partial log is the only evidence of where it stalled.
            elapsed = time.monotonic() - started
            timings[timing_name] = round(elapsed, 3)
            raise Ccp4ToolTimeoutError(
                tool=os.path.basename(cmd[0]),
                timeout_s=timeout_s,
                elapsed_s=elapsed,
                log_path=log_path,
                timings=timings,
            ) from exc
        finally:
            timings.setdefault(timing_name, round(time.monotonic() - started, 3))
            logger.debug(
                "%s: %s ended after %.3fs", pdb_id, program, timings[timing_name]
            )
        if proc.returncode != 0:
            detail = truncate(stderr_text)
            detail_suffix = f": {detail}" if detail else ""
            logger.warning(
                "%s: %s exited %d; see %s%s",
                pdb_id,
                program,
                proc.returncode,
                log_path,
                detail_suffix,
            )
            if os.path.basename(cmd[0]).lower() == "mtzfix":
                try:
                    with open(log_path, encoding="utf-8", errors="replace") as handle:
                        mtzfix_log_text = handle.read()
                except OSError:
                    mtzfix_log_text = ""
                if "FAILED a test on re-take" in mtzfix_log_text:
                    raise MtzfixValidationError(
                        "MTZFIX consistency re-test failed for "
                        f"{pdb_id}{detail_suffix}",
                        timings=timings,
                    )
            raise RuntimeError(
                f"{cmd[0]} failed for {pdb_id} (rc={proc.returncode}): "
                f"see {log_path}{detail_suffix}"
            )

    # MTZFIX writes HKLOUT only when it has corrections to make, so a retained
    # file from an earlier run would be read as this run's output.
    if os.path.lexists(fixed_mtz):
        if os.path.realpath(fixed_mtz) == os.path.realpath(mtz_path):
            raise RuntimeError(
                f"MTZFIX output path would overwrite its input: {mtz_path}"
            )
        os.remove(fixed_mtz)
    twin_normalization = None
    try:
        _run(
            ["mtzfix", "HKLIN", mtz_path, "HKLOUT", fixed_mtz],
            None,
            os.path.basename(mtzfix_log),
            "mtzfix_s",
        )
    except MtzfixValidationError as exc:
        if pdb_redo_is_twin is not True:
            raise
        started = time.monotonic()
        try:
            twin_normalization = normalize_refmac_twin_coefficients(
                mtz_path, twin_normalized_mtz
            )
        except (OSError, RuntimeError, ValueError) as normalization_error:
            timings["twin_coefficient_normalization_s"] = round(
                time.monotonic() - started, 3
            )
            raise MtzfixValidationError(
                f"{exc}; guarded twin coefficient normalization was refused: "
                f"{normalization_error}",
                timings=timings,
            ) from normalization_error
        else:
            timings["twin_coefficient_normalization_s"] = round(
                time.monotonic() - started, 3
            )

    if twin_normalization is not None:
        map_mtz = twin_normalized_mtz
        mtzfix_applied = False
    elif os.path.exists(fixed_mtz):
        if not os.path.isfile(fixed_mtz) or os.path.getsize(fixed_mtz) == 0:
            raise RuntimeError(f"mtzfix produced an invalid corrected MTZ for {pdb_id}")
        map_mtz = fixed_mtz
        mtzfix_applied = True
    else:
        map_mtz = mtz_path
        mtzfix_applied = False

    def _fft(
        map_path: str,
        f_label: str,
        phi_label: str,
        log_suffix: str,
        timing_name: str,
    ) -> None:
        _run(
            ["fft", "HKLIN", map_mtz, "MAPOUT", map_path],
            f"labi F1={f_label} PHI={phi_label}\nGRID SAMP=5\n",
            f"{pdb_id}_fft_{log_suffix}.log",
            timing_name,
        )

    def _model_envelope(
        full_map: str, envelope_map: str, log_suffix: str, timing_name: str
    ) -> None:
        _run(
            ["mapmask", "MAPIN", full_map, "MAPOUT", envelope_map, "XYZIN", pdb_path],
            f"BORDER {MODEL_ENVELOPE_BORDER_ANGSTROM}\nEND\n",
            f"{pdb_id}_mapmask_{log_suffix}.log",
            timing_name,
        )

    def _map_size(path: str, stage: str) -> int:
        if not os.path.isfile(path):
            raise RuntimeError(f"{stage} produced no map file for {pdb_id}: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(
                f"{stage} produced an empty map file for {pdb_id}: {path}"
            )
        return size

    def _map_extent_requires_full_map(path: str) -> bool:
        """Return whether a positive translated crop is unsafe for EDSTATS.

        EDSTATS accepts cropped grids that overlap or extend in the negative
        direction, but wraps lookups into the primary unit cell when an entire
        stored axis begins beyond its positive edge, which MAPMASK can produce
        for a translated deposited model.
        """
        with open(path, "rb") as handle:
            header = handle.read(1024)
        if len(header) != 1024:
            raise RuntimeError(
                f"MAPMASK produced an invalid CCP4 map header for {pdb_id}: {path}"
            )
        for byte_order in ("<", ">"):
            words = struct.unpack(f"{byte_order}256i", header)
            counts = words[0:3]
            mode = words[3]
            starts = words[4:7]
            sampling = words[7:10]
            axes = words[16:19]
            if (
                mode not in (0, 1, 2, 6, 12)
                or any(value <= 0 for value in counts + sampling)
                or sorted(axes) != [1, 2, 3]
            ):
                continue
            xyz_starts = [0, 0, 0]
            for start, _count, axis in zip(starts, counts, axes):
                xyz_starts[axis - 1] = start
            return any(
                start >= grid_size for start, grid_size in zip(xyz_starts, sampling)
            )
        raise RuntimeError(
            f"MAPMASK produced an unrecognized CCP4 map header for {pdb_id}: {path}"
        )

    map_scope_used = map_scope
    full_map_bytes = 0
    edstats_map_bytes = 0
    if map_scope == "full":
        _fft(fo_map, "FWT", "PHWT", "fo", "fft_2fofc_s")
        _fft(df_map, "DELFWT", "PHDELWT", "df", "fft_fofc_s")
        full_map_bytes = _map_size(fo_map, "2mFo-DFc FFT") + _map_size(
            df_map, "mFo-DFc FFT"
        )
        edstats_map_bytes = full_map_bytes
    else:
        full_fo_map = os.path.join(out_dir, f"{pdb_id}_fo_full.map")
        _fft(full_fo_map, "FWT", "PHWT", "fo", "fft_2fofc_s")
        _model_envelope(full_fo_map, fo_map, "fo", "mapmask_2fofc_s")
        full_fo_bytes = _map_size(full_fo_map, "2mFo-DFc FFT")
        envelope_fo_bytes = _map_size(fo_map, "2mFo-DFc MAPMASK")
        full_map_bytes += full_fo_bytes

        # A model spanning most or all of the cell can crop to an envelope
        # larger than FFT's own map. Either fallback keeps the full map and
        # skips MAPMASK for the second map too.
        fallback_scope = ""
        if envelope_fo_bytes >= full_fo_bytes:
            fallback_scope = "full-size-fallback"
        elif _map_extent_requires_full_map(fo_map):
            fallback_scope = "full-extent-fallback"
        if fallback_scope:
            os.remove(fo_map)
            os.replace(full_fo_map, fo_map)
            map_scope_used = fallback_scope
            _fft(df_map, "DELFWT", "PHDELWT", "df", "fft_fofc_s")
            full_map_bytes += _map_size(df_map, "mFo-DFc FFT")
            edstats_map_bytes = full_map_bytes
        else:
            edstats_map_bytes += envelope_fo_bytes
            if not keep_full_maps:
                os.remove(full_fo_map)
            full_df_map = os.path.join(out_dir, f"{pdb_id}_df_full.map")
            _fft(full_df_map, "DELFWT", "PHDELWT", "df", "fft_fofc_s")
            _model_envelope(full_df_map, df_map, "df", "mapmask_fofc_s")
            full_df_bytes = _map_size(full_df_map, "mFo-DFc FFT")
            envelope_df_bytes = _map_size(df_map, "mFo-DFc MAPMASK")
            if full_df_bytes != full_fo_bytes or envelope_df_bytes != envelope_fo_bytes:
                raise RuntimeError(
                    f"density map extents differ for {pdb_id}: "
                    f"full={full_fo_bytes}/{full_df_bytes}, "
                    f"model-envelope={envelope_fo_bytes}/{envelope_df_bytes}"
                )
            full_map_bytes += full_df_bytes
            edstats_map_bytes += envelope_df_bytes
            if not keep_full_maps:
                os.remove(full_df_map)

    _run(
        [
            "edstats",
            "XYZIN",
            pdb_path,
            "MAPIN1",
            fo_map,
            "MAPIN2",
            df_map,
            "XYZOUT",
            rszd,
            "OUT",
            stats_out,
            "QQDOUT",
            qq_out,
        ],
        # EDSTATS otherwise pools every named alternate conformer into one
        # residue observation (USEALT defaults to false).  Alchemy selects one
        # coherent residue conformer by occupancy, so request separate EDSTATS
        # rows and let ``extract_metal_statistics`` retain that same conformer.
        f"reslo={reslo},reshi={reshi},usealt=true\n",
        f"{pdb_id}_edstats.log",
        "edstats_s",
    )

    if not os.path.exists(stats_out):
        raise RuntimeError(f"edstats produced no stats file for {pdb_id}")
    return DensityResult(
        stats_out=stats_out,
        rszd=rszd,
        fo_map=fo_map,
        df_map=df_map,
        mtz_for_maps=map_mtz,
        mtzfix_log=mtzfix_log,
        mtzfix_applied=mtzfix_applied,
        timings=timings,
        twin_coefficient_normalization_applied=(twin_normalization is not None),
        twin_coefficient_normalization=twin_normalization,
        density_map_scope_requested=map_scope,
        density_map_scope_used=map_scope_used,
        full_map_bytes=full_map_bytes,
        edstats_map_bytes=edstats_map_bytes,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Run mtzfix + fft x2 + optional model-envelope mapmask "
            "+ edstats on a single structure."
        )
    )
    p.add_argument("pdb_id", metavar="pdbID")
    p.add_argument("mtz", help="MTZ with FWT/PHWT/DELFWT/PHDELWT columns")
    p.add_argument("pdb", help="coordinate file (edstats XYZIN)")
    # Must not default to this file's own directory: maps, mapmask output and
    # EDSTATS logs would land in src/, which .gitignore does not cover.
    p.add_argument(
        "--out-dir",
        default=".",
        help="directory for maps, logs and stats output (default: %(default)s)",
    )
    p.add_argument(
        "--reslo",
        type=float,
        required=True,
        help="low resolution limit (larger number)",
    )
    p.add_argument(
        "--reshi",
        type=float,
        required=True,
        help="high resolution limit (smaller number)",
    )
    p.add_argument(
        "--density-map-scope", choices=DENSITY_MAP_SCOPES, default="model-envelope"
    )
    p.add_argument("--keep-full-maps", action="store_true")
    args = p.parse_args()
    print(
        run_density_analysis(
            args.pdb_id,
            args.mtz,
            args.pdb,
            args.out_dir,
            args.reslo,
            args.reshi,
            map_scope=args.density_map_scope,
            keep_full_maps=args.keep_full_maps,
        )
    )
