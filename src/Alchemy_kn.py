# Alchemy
# CCP4-based map calculation + edstats real-space statistics for one structure.
#
# Core pipeline (see main.py for batch orchestration over PDB-REDO):
#   1. fft  FWT/PHWT       -> {id}_fo.map   (2mFo-DFc "observed" map)
#   2. fft  DELFWT/PHDELWT -> {id}_df.map   (mFo-DFc difference map)
#   3. edstats XYZIN=pdb MAPIN1=fo MAPIN2=df -> {id}_stats.out per-atom stats
#
# Requires the CCP4 binaries `fft` and `edstats` on PATH (pass `env=` to point at
# a sourced CCP4 environment). The input MTZ must carry the FWT/PHWT/DELFWT/PHDELWT
# map-coefficient columns (PDB-REDO _final.mtz and refmac output have them).
import os
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
directory = BASE_DIR


def run_alchemy(pdbID, mtz_path, pdb_path, out_dir, reslo, reshi, env=None):
    """Compute Fo and difference maps with `fft`, then run `edstats`.

    Parameters
    ----------
    pdbID : str            -- used only to name output files
    mtz_path : str         -- MTZ with FWT/PHWT and DELFWT/PHDELWT columns
    pdb_path : str         -- coordinates (edstats XYZIN)
    out_dir : str          -- all outputs (maps, logs, stats) are written here
    reslo, reshi : float   -- low / high resolution limits passed to edstats
    env : dict | None      -- process environment (CCP4 on PATH) for subprocesses

    Returns a dict of output paths. Raises RuntimeError if any CCP4 step exits
    non-zero or edstats produces no stats file.
    """
    os.makedirs(out_dir, exist_ok=True)
    fo_map = os.path.join(out_dir, f"{pdbID}_fo.map")
    df_map = os.path.join(out_dir, f"{pdbID}_df.map")
    stats_out = os.path.join(out_dir, f"{pdbID}_stats.out")
    rszd = os.path.join(out_dir, f"{pdbID}_rszd.pdb")
    qq_out = os.path.join(out_dir, f"{pdbID}_qq.out")

    def _run(cmd, stdin, logname):
        log_path = os.path.join(out_dir, logname)
        resolved = shutil.which(cmd[0], path=(env or {}).get("PATH"))
        if resolved:
            cmd = [resolved, *cmd[1:]]
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, input=stdin, text=True, stdout=log,
                                  stderr=subprocess.PIPE, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{cmd[0]} failed for {pdbID} (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[:500]}")

    # 2mFo-DFc observed map
    _run(["fft", "HKLIN", mtz_path, "MAPOUT", fo_map],
         "labi F1=FWT PHI=PHWT\nGRID SAMP=5\n", f"{pdbID}_fft_fo.log")
    # mFo-DFc difference map
    _run(["fft", "HKLIN", mtz_path, "MAPOUT", df_map],
         "labi F1=DELFWT PHI=PHDELWT\nGRID SAMP=5\n", f"{pdbID}_fft_df.log")
    # real-space statistics (RSZD/RSR per atom/residue)
    _run(["edstats", "XYZIN", pdb_path, "MAPIN1", fo_map, "MAPIN2", df_map,
          "XYZOUT", rszd, "OUT", stats_out, "QQDOUT", qq_out],
         f"reslo={reslo},reshi={reshi}\n", f"{pdbID}_edstats.log")

    if not os.path.exists(stats_out):
        raise RuntimeError(f"edstats produced no stats file for {pdbID}")
    return {"stats_out": stats_out, "rszd": rszd,
            "fo_map": fo_map, "df_map": df_map}


if __name__ == "__main__":
    # Thin single-structure CLI for manual testing. Batch runs go through main.py.
    import argparse
    p = argparse.ArgumentParser(
        description="Run fft x2 + edstats on a single structure.")
    p.add_argument("pdbID")
    p.add_argument("mtz", help="MTZ with FWT/PHWT/DELFWT/PHDELWT columns")
    p.add_argument("pdb", help="coordinate file (edstats XYZIN)")
    p.add_argument("--out-dir", default=directory)
    p.add_argument("--reslo", type=float, required=True,
                   help="low resolution limit (larger number)")
    p.add_argument("--reshi", type=float, required=True,
                   help="high resolution limit (smaller number)")
    args = p.parse_args()
    print(run_alchemy(args.pdbID, args.mtz, args.pdb, args.out_dir,
                      args.reslo, args.reshi))
