# Alchemy
# CCP4-based map calculation + edstats real-space statistics for one structure.
#
# Core pipeline (see main.py for batch orchestration over PDB-REDO):
#   1. mtzfix validates/corrects the input MTZ's Fourier map coefficients
#   2. fft  FWT/PHWT       -> {id}_fo.map   (2mFo-DFc "observed" map)
#   3. fft  DELFWT/PHDELWT -> {id}_df.map   (mFo-DFc difference map)
#   4. edstats XYZIN=pdb MAPIN1=fo MAPIN2=df -> {id}_stats.out per-atom stats
#
# Requires the CCP4 binaries `mtzfix`, `fft`, and `edstats` on PATH (pass `env=`
# to point at a sourced CCP4 environment). The input MTZ must carry the
# FWT/PHWT/DELFWT/PHDELWT map-coefficient columns (PDB-REDO _final.mtz and
# refmac output have them).
import os
import subprocess
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
directory = BASE_DIR


def run_density_analysis(pdbID, mtz_path, pdb_path, out_dir, reslo, reshi, env=None):
    """Validate the MTZ, compute maps with `fft`, then run `edstats`.

    Parameters
    ----------
    pdbID : str            -- used only to name output files
    mtz_path : str         -- MTZ with FWT/PHWT and DELFWT/PHDELWT columns
    pdb_path : str         -- coordinates (edstats XYZIN)
    out_dir : str          -- all outputs (maps, logs, stats) are written here
    reslo, reshi : float   -- common low/high limits of the four map columns
    env : dict | None      -- process environment (CCP4 on PATH) for subprocesses

    MTZFIX creates HKLOUT only when it has corrections to write. If the input
    already passes its checks, no corrected file is produced and the original
    MTZ is used for both FFT calculations.

    Returns a dict of output paths and MTZFIX provenance. Raises RuntimeError if
    any CCP4 step exits non-zero or edstats produces no stats file.
    """
    os.makedirs(out_dir, exist_ok=True)
    fixed_mtz = os.path.join(out_dir, f"{pdbID}_mtzfix.mtz")
    mtzfix_log = os.path.join(out_dir, f"{pdbID}_mtzfix.log")
    fo_map = os.path.join(out_dir, f"{pdbID}_fo.map")
    df_map = os.path.join(out_dir, f"{pdbID}_df.map")
    stats_out = os.path.join(out_dir, f"{pdbID}_stats.out")
    rszd = os.path.join(out_dir, f"{pdbID}_rszd.pdb")
    qq_out = os.path.join(out_dir, f"{pdbID}_qq.out")

    def _run(cmd, stdin, logname):
        log_path = os.path.join(out_dir, logname)
        resolved = shutil.which(cmd[0], path=(env or {}).get("PATH"))
        if resolved is None:
            raise RuntimeError(
                f"required CCP4 program {cmd[0]!r} was not found on PATH")
        cmd = [resolved, *cmd[1:]]
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, input=stdin, text=True, stdout=log,
                                  stderr=subprocess.PIPE, env=env)
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()[:500]
            detail_suffix = f": {detail}" if detail else ""
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
         None, os.path.basename(mtzfix_log))

    if os.path.exists(fixed_mtz):
        if not os.path.isfile(fixed_mtz) or os.path.getsize(fixed_mtz) == 0:
            raise RuntimeError(
                f"mtzfix produced an invalid corrected MTZ for {pdbID}")
        map_mtz = fixed_mtz
        mtzfix_applied = True
    else:
        map_mtz = mtz_path
        mtzfix_applied = False

    # 2mFo-DFc observed map
    _run(["fft", "HKLIN", map_mtz, "MAPOUT", fo_map],
         "labi F1=FWT PHI=PHWT\nGRID SAMP=5\n", f"{pdbID}_fft_fo.log")
    # mFo-DFc difference map
    _run(["fft", "HKLIN", map_mtz, "MAPOUT", df_map],
         "labi F1=DELFWT PHI=PHDELWT\nGRID SAMP=5\n", f"{pdbID}_fft_df.log")
    # real-space statistics (RSZD/RSR per atom/residue)
    _run(["edstats", "XYZIN", pdb_path, "MAPIN1", fo_map, "MAPIN2", df_map,
          "XYZOUT", rszd, "OUT", stats_out, "QQDOUT", qq_out],
         f"reslo={reslo},reshi={reshi}\n", f"{pdbID}_edstats.log")

    if not os.path.exists(stats_out):
        raise RuntimeError(f"edstats produced no stats file for {pdbID}")
    return {"stats_out": stats_out, "rszd": rszd,
            "fo_map": fo_map, "df_map": df_map,
            "mtz_for_maps": map_mtz, "mtzfix_log": mtzfix_log,
            "mtzfix_applied": mtzfix_applied}


if __name__ == "__main__":
    # Thin single-structure CLI for manual testing. Batch runs go through main.py.
    import argparse
    p = argparse.ArgumentParser(
        description="Run mtzfix + fft x2 + edstats on a single structure.")
    p.add_argument("pdbID")
    p.add_argument("mtz", help="MTZ with FWT/PHWT/DELFWT/PHDELWT columns")
    p.add_argument("pdb", help="coordinate file (edstats XYZIN)")
    p.add_argument("--out-dir", default=directory)
    p.add_argument("--reslo", type=float, required=True,
                   help="low resolution limit (larger number)")
    p.add_argument("--reshi", type=float, required=True,
                   help="high resolution limit (smaller number)")
    args = p.parse_args()
    print(run_density_analysis(args.pdbID, args.mtz, args.pdb, args.out_dir,
                      args.reslo, args.reshi))
