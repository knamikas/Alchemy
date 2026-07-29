# Alchemy
# CCP4-based map calculation + edstats real-space statistics for one structure.
#
# Core pipeline (see main.py for batch orchestration over PDB-REDO):
#   1. mtzfix validates/corrects the input MTZ's Fourier map coefficients
#   2. fft  FWT/PHWT       -> {id}_fo.map   (2mFo-DFc "observed" map)
#   3. fft  DELFWT/PHDELWT -> {id}_df.map   (mFo-DFc difference map)
#   4. mapmask optionally limits both maps to the complete coordinate-model
#      envelope, retaining a 10 Angstrom border and every deposited atom
#   5. edstats XYZIN=pdb MAPIN1=fo MAPIN2=df -> {id}_stats.out per-atom stats
#
# Requires the CCP4 binaries `mtzfix`, `fft`, `mapmask`, and `edstats` on PATH
# (pass `env=` to point at a sourced CCP4 environment). The input MTZ must carry
# the FWT/PHWT/DELFWT/PHDELWT map-coefficient columns (PDB-REDO _final.mtz and
# refmac output have them).
import os
import subprocess
import shutil
import struct
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ENVELOPE_BORDER_ANGSTROM = 10
DENSITY_MAP_SCOPES = ("model-envelope", "full")


class MtzfixValidationError(RuntimeError):
    """MTZFIX could not make map coefficients pass its consistency checks."""

    def __init__(self, message, timings=None):
        super().__init__(message)
        self.timings = dict(timings or {})


def run_density_analysis(
        pdbID, mtz_path, pdb_path, out_dir, reslo, reshi, env=None,
        map_scope="model-envelope", keep_full_maps=False):
    """Validate the MTZ, compute maps with `fft`, then run `edstats`.

    Parameters
    ----------
    pdbID : str            -- used only to name output files
    mtz_path : str         -- MTZ with FWT/PHWT and DELFWT/PHDELWT columns
    pdb_path : str         -- coordinates (edstats XYZIN)
    out_dir : str          -- all outputs (maps, logs, stats) are written here
    reslo, reshi : float   -- common low/high limits of the four map columns
    env : dict | None      -- process environment (CCP4 on PATH) for subprocesses
    map_scope : str        -- ``model-envelope`` (default) or legacy ``full``
    keep_full_maps : bool  -- retain pre-crop maps with other intermediates

    MTZFIX creates HKLOUT only when it has corrections to write. If the input
    already passes its checks, no corrected file is produced and the original
    MTZ is used for both FFT calculations.

    Returns a dict of output paths and MTZFIX provenance. Raises RuntimeError if
    any CCP4 step exits non-zero or edstats produces no stats file.
    """
    if map_scope not in DENSITY_MAP_SCOPES:
        raise ValueError(
            f"density map scope must be one of {', '.join(DENSITY_MAP_SCOPES)}")

    os.makedirs(out_dir, exist_ok=True)
    fixed_mtz = os.path.join(out_dir, f"{pdbID}_mtzfix.mtz")
    mtzfix_log = os.path.join(out_dir, f"{pdbID}_mtzfix.log")
    fo_map = os.path.join(out_dir, f"{pdbID}_fo.map")
    df_map = os.path.join(out_dir, f"{pdbID}_df.map")
    stats_out = os.path.join(out_dir, f"{pdbID}_stats.out")
    rszd = os.path.join(out_dir, f"{pdbID}_rszd.pdb")
    qq_out = os.path.join(out_dir, f"{pdbID}_qq.out")

    timings = {}

    def _run(cmd, stdin, logname, timing_name):
        log_path = os.path.join(out_dir, logname)
        resolved = shutil.which(cmd[0], path=(env or {}).get("PATH"))
        if resolved is None:
            raise RuntimeError(
                f"required CCP4 program {cmd[0]!r} was not found on PATH")
        cmd = [resolved, *cmd[1:]]
        started = time.monotonic()
        try:
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, input=stdin, text=True, stdout=log,
                                      stderr=subprocess.PIPE, env=env)
        finally:
            timings[timing_name] = round(time.monotonic() - started, 3)
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()[:500]
            detail_suffix = f": {detail}" if detail else ""
            if os.path.basename(cmd[0]).lower() == "mtzfix":
                try:
                    with open(log_path, encoding="utf-8",
                              errors="replace") as handle:
                        mtzfix_log_text = handle.read()
                except OSError:
                    mtzfix_log_text = ""
                if "FAILED a test on re-take" in mtzfix_log_text:
                    raise MtzfixValidationError(
                        f"MTZFIX consistency re-test failed for {pdbID}"
                        f"{detail_suffix}", timings=timings)
            raise RuntimeError(
                f"{cmd[0]} failed for {pdbID} (rc={proc.returncode}): "
                f"see {log_path}{detail_suffix}")

    # MTZFIX deliberately does not create HKLOUT when the input passes all of
    # its consistency checks. Remove a result from any earlier retained run so
    # that its absence after this invocation cannot be mistaken for new output.
    if os.path.lexists(fixed_mtz):
        if os.path.realpath(fixed_mtz) == os.path.realpath(mtz_path):
            raise RuntimeError(
                f"MTZFIX output path would overwrite its input: {mtz_path}")
        os.remove(fixed_mtz)
    _run(["mtzfix", "HKLIN", mtz_path, "HKLOUT", fixed_mtz],
         None, os.path.basename(mtzfix_log), "mtzfix_s")

    if os.path.exists(fixed_mtz):
        if not os.path.isfile(fixed_mtz) or os.path.getsize(fixed_mtz) == 0:
            raise RuntimeError(
                f"mtzfix produced an invalid corrected MTZ for {pdbID}")
        map_mtz = fixed_mtz
        mtzfix_applied = True
    else:
        map_mtz = mtz_path
        mtzfix_applied = False

    def _fft(map_path, f_label, phi_label, log_suffix, timing_name):
        _run(["fft", "HKLIN", map_mtz, "MAPOUT", map_path],
             f"labi F1={f_label} PHI={phi_label}\nGRID SAMP=5\n",
             f"{pdbID}_fft_{log_suffix}.log", timing_name)

    def _model_envelope(full_map, envelope_map, log_suffix, timing_name):
        _run(
            ["mapmask", "MAPIN", full_map, "MAPOUT", envelope_map,
             "XYZIN", pdb_path],
            f"BORDER {MODEL_ENVELOPE_BORDER_ANGSTROM}\nEND\n",
            f"{pdbID}_mapmask_{log_suffix}.log", timing_name,
        )

    def _map_size(path, stage):
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{stage} produced no map file for {pdbID}: {path}")
        size = os.path.getsize(path)
        if size == 0:
            raise RuntimeError(
                f"{stage} produced an empty map file for {pdbID}: {path}")
        return size

    def _map_extent_requires_full_map(path):
        """Return whether a positive translated crop is unsafe for EDSTATS.

        EDSTATS accepts cropped grids that overlap or extend in the negative
        direction, but wraps lookups into the primary unit cell when an entire
        stored axis begins beyond its positive edge. MAPMASK can produce that
        layout for a translated deposited model. Reading only the 1024-byte
        CCP4 header keeps this preflight independent of map size.
        """
        with open(path, "rb") as handle:
            header = handle.read(1024)
        if len(header) != 1024:
            raise RuntimeError(
                f"MAPMASK produced an invalid CCP4 map header for {pdbID}: "
                f"{path}")
        for byte_order in ("<", ">"):
            words = struct.unpack(f"{byte_order}256i", header)
            counts = words[0:3]
            mode = words[3]
            starts = words[4:7]
            sampling = words[7:10]
            axes = words[16:19]
            if (mode not in (0, 1, 2, 6, 12) or
                    any(value <= 0 for value in counts + sampling) or
                    sorted(axes) != [1, 2, 3]):
                continue
            xyz_starts = [0, 0, 0]
            for start, _count, axis in zip(starts, counts, axes):
                xyz_starts[axis - 1] = start
            return any(
                start >= grid_size
                for start, grid_size in zip(xyz_starts, sampling)
            )
        raise RuntimeError(
            f"MAPMASK produced an unrecognized CCP4 map header for {pdbID}: "
            f"{path}")

    map_scope_used = map_scope
    full_map_bytes = 0
    edstats_map_bytes = 0
    if map_scope == "full":
        _fft(fo_map, "FWT", "PHWT", "fo", "fft_2fofc_s")
        _fft(df_map, "DELFWT", "PHDELWT", "df", "fft_fofc_s")
        full_map_bytes = (
            _map_size(fo_map, "2mFo-DFc FFT") +
            _map_size(df_map, "mFo-DFc FFT")
        )
        edstats_map_bytes = full_map_bytes
    else:
        full_fo_map = os.path.join(out_dir, f"{pdbID}_fo_full.map")
        _fft(full_fo_map, "FWT", "PHWT", "fo", "fft_2fofc_s")
        _model_envelope(
            full_fo_map, fo_map, "fo", "mapmask_2fofc_s")
        full_fo_bytes = _map_size(full_fo_map, "2mFo-DFc FFT")
        envelope_fo_bytes = _map_size(fo_map, "2mFo-DFc MAPMASK")
        full_map_bytes += full_fo_bytes

        # A model spanning most or all of the cell can produce an extended
        # envelope larger than FFT's original map. A translated model can also
        # produce a smaller envelope whose stored extent begins entirely beyond
        # the primary cell's positive edge; EDSTATS cannot safely consume that
        # layout. In either case retain the legacy map and skip MAPMASK for the
        # second map.
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
            full_df_map = os.path.join(out_dir, f"{pdbID}_df_full.map")
            _fft(full_df_map, "DELFWT", "PHDELWT", "df", "fft_fofc_s")
            _model_envelope(
                full_df_map, df_map, "df", "mapmask_fofc_s")
            full_df_bytes = _map_size(full_df_map, "mFo-DFc FFT")
            envelope_df_bytes = _map_size(df_map, "mFo-DFc MAPMASK")
            if (full_df_bytes != full_fo_bytes or
                    envelope_df_bytes != envelope_fo_bytes):
                raise RuntimeError(
                    f"density map extents differ for {pdbID}: "
                    f"full={full_fo_bytes}/{full_df_bytes}, "
                    f"model-envelope={envelope_fo_bytes}/{envelope_df_bytes}")
            full_map_bytes += full_df_bytes
            edstats_map_bytes += envelope_df_bytes
            if not keep_full_maps:
                os.remove(full_df_map)

    # real-space statistics (RSZD/RSR per atom/residue)
    _run(["edstats", "XYZIN", pdb_path, "MAPIN1", fo_map, "MAPIN2", df_map,
          "XYZOUT", rszd, "OUT", stats_out, "QQDOUT", qq_out],
         f"reslo={reslo},reshi={reshi}\n", f"{pdbID}_edstats.log",
         "edstats_s")

    if not os.path.exists(stats_out):
        raise RuntimeError(f"edstats produced no stats file for {pdbID}")
    return {"stats_out": stats_out, "rszd": rszd,
            "fo_map": fo_map, "df_map": df_map,
            "mtz_for_maps": map_mtz, "mtzfix_log": mtzfix_log,
            "mtzfix_applied": mtzfix_applied, "timings": timings,
            "density_map_scope_requested": map_scope,
            "density_map_scope_used": map_scope_used,
            "full_map_bytes": full_map_bytes,
            "edstats_map_bytes": edstats_map_bytes}


if __name__ == "__main__":
    # Thin single-structure CLI for manual testing. Batch runs go through main.py.
    import argparse
    p = argparse.ArgumentParser(
        description=("Run mtzfix + fft x2 + optional model-envelope mapmask "
                     "+ edstats on a single structure."))
    p.add_argument("pdbID")
    p.add_argument("mtz", help="MTZ with FWT/PHWT/DELFWT/PHDELWT columns")
    p.add_argument("pdb", help="coordinate file (edstats XYZIN)")
    p.add_argument("--out-dir", default=BASE_DIR)
    p.add_argument("--reslo", type=float, required=True,
                   help="low resolution limit (larger number)")
    p.add_argument("--reshi", type=float, required=True,
                   help="high resolution limit (smaller number)")
    p.add_argument("--density-map-scope", choices=DENSITY_MAP_SCOPES,
                   default="model-envelope")
    p.add_argument("--keep-full-maps", action="store_true")
    args = p.parse_args()
    print(run_density_analysis(args.pdbID, args.mtz, args.pdb, args.out_dir,
                      args.reslo, args.reshi,
                      map_scope=args.density_map_scope,
                      keep_full_maps=args.keep_full_maps))
