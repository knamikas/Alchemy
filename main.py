#!/usr/bin/env python
"""Batch-run the Alchemy core pipeline over PDB-REDO entries.

For each PDB entry this computes 2mFo-DFc and mFo-DFc maps (CCP4 `fft`) and runs
`edstats`, then extracts per-atom real-space statistics for metal ions and
metal-containing cofactors. Results are streamed to two CSVs under --output-dir:

  metal_stats_all.csv  -- one row per metal/cofactor atom (pdbID + edstats columns)
  manifest.csv         -- one row per entry: pdbID,status,n_metals,runtime_s,error

Requirements
------------
* CCP4 `fft` and `edstats` on PATH -- either already sourced, or via --ccp4-setup
  pointing at a CCP4 setup script (e.g. <CCP4>/bin/ccp4.setup-sh).
* Run under a Python env with gemmi + Biopython (e.g. `conda run -n metal python
  main.py ...`). gemmi is needed only for the 0cyc/besttls states (gunzip +
  CIF->PDB) and the resolution fallback.

Examples
--------
  conda run -n metal python main.py --id 109m \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
  conda run -n metal python main.py --max-pdbs 20 --workers 4 \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
"""
import argparse
import csv
import gzip
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count

from Alchemy_kn import run_alchemy
from Analysisv2_kn import metals, uncommonMetals, load_cofactors, run_analysis

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = "/datasets/bioinfo/pdb-redo"
METALS_SET = set(metals) | set(uncommonMetals)

# config dict shared with worker processes (set once per worker by _init_worker)
_CFG = None


# --------------------------------------------------------------------------- #
# CCP4 environment
# --------------------------------------------------------------------------- #
def resolve_env(ccp4_setup):
    """Return the environment dict to run CCP4 under.

    If `ccp4_setup` is given, source it in a bash subshell and capture the
    resulting environment; otherwise inherit the current environment.
    """
    if not ccp4_setup:
        return os.environ.copy()
    if not os.path.exists(ccp4_setup):
        raise SystemExit(f"--ccp4-setup not found: {ccp4_setup}")
    cmd = f"source {shlex.quote(ccp4_setup)} >/dev/null 2>&1 && env -0"
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"Failed to source CCP4 setup {ccp4_setup}:\n{out.stderr}")
    env = {}
    for chunk in out.stdout.split("\0"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            env[k] = v
    return env


def verify_ccp4(env):
    missing = [t for t in ("fft", "edstats")
               if shutil.which(t, path=env.get("PATH")) is None]
    if missing:
        raise SystemExit(
            f"CCP4 tool(s) not found on PATH: {missing}. "
            f"Provide --ccp4-setup <ccp4.setup-sh> or source CCP4 before running.")


# --------------------------------------------------------------------------- #
# Per-entry input preparation
# --------------------------------------------------------------------------- #
def entry_dir_for(root, pdbID):
    """PDB-REDO layout: <root>/<middle two chars of id>/<id>/."""
    return os.path.join(root, pdbID[1:3], pdbID)


def _gunzip_to(src_gz, dst):
    if not os.path.exists(src_gz):
        raise FileNotFoundError(src_gz)
    with gzip.open(src_gz, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return dst


def _cif_to_pdb(cif_path, dst):
    """Convert a (optionally gzipped) mmCIF coordinate file to PDB via gemmi."""
    import gemmi  # imported lazily; only the 0cyc state needs it
    if not os.path.exists(cif_path):
        raise FileNotFoundError(cif_path)
    structure = gemmi.read_structure(cif_path)
    structure.setup_entities()
    structure.write_pdb(dst)
    return dst


def prepare_inputs(pdbID, entry_dir, state, work_dir):
    """Return (mtz_path, pdb_path) for the requested refinement state.

    `final` is read directly from the dataset (uncompressed PDB + MTZ). `besttls`
    and `0cyc` are decompressed/converted into work_dir. Raises FileNotFoundError
    (-> "skip") when the required source files are absent.
    """
    if state == "final":
        mtz = os.path.join(entry_dir, f"{pdbID}_final.mtz")
        pdb = os.path.join(entry_dir, f"{pdbID}_final.pdb")
        for p in (mtz, pdb):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        return mtz, pdb
    if state == "besttls":
        mtz = _gunzip_to(os.path.join(entry_dir, f"{pdbID}_besttls.mtz.gz"),
                         os.path.join(work_dir, f"{pdbID}_besttls.mtz"))
        pdb = _gunzip_to(os.path.join(entry_dir, f"{pdbID}_besttls.pdb.gz"),
                         os.path.join(work_dir, f"{pdbID}_besttls.pdb"))
        return mtz, pdb
    if state == "0cyc":
        mtz = _gunzip_to(os.path.join(entry_dir, f"{pdbID}_0cyc.mtz.gz"),
                         os.path.join(work_dir, f"{pdbID}_0cyc.mtz"))
        pdb = _cif_to_pdb(os.path.join(entry_dir, f"{pdbID}_0cyc.cif.gz"),
                          os.path.join(work_dir, f"{pdbID}_0cyc.pdb"))
        return mtz, pdb
    raise ValueError(f"unknown refine state: {state}")


def read_resolution(entry_dir, mtz_path):
    """Return (reslo, reshi) -- low/high resolution limits for edstats.

    Prefer PDB-REDO data.json (DATARESL/DATARESH); fall back to the MTZ via gemmi.
    """
    dj = os.path.join(entry_dir, "data.json")
    if os.path.exists(dj):
        try:
            props = json.load(open(dj)).get("properties", {})
            lo, hi = props.get("DATARESL"), props.get("DATARESH")
            if lo and hi:
                return float(lo), float(hi)
        except (ValueError, KeyError, OSError):
            pass
    import gemmi
    m = gemmi.read_mtz_file(mtz_path)
    return m.resolution_low(), m.resolution_high()


def has_state_files(entry_dir, pdbID, state):
    req = {
        "final": [f"{pdbID}_final.mtz", f"{pdbID}_final.pdb"],
        "besttls": [f"{pdbID}_besttls.mtz.gz", f"{pdbID}_besttls.pdb.gz"],
        "0cyc": [f"{pdbID}_0cyc.mtz.gz", f"{pdbID}_0cyc.cif.gz"],
    }[state]
    return all(os.path.exists(os.path.join(entry_dir, f)) for f in req)


def enumerate_entries(root, state, limit=None):
    """All PDB ids under `root` that have the required files for `state`.

    If `limit` is given, stop after collecting that many (sorted) ids -- this
    keeps small --max-pdbs debug runs fast instead of walking all ~24k entries.
    """
    ids = []
    for hashdir in sorted(os.listdir(root)):
        hp = os.path.join(root, hashdir)
        if not os.path.isdir(hp):
            continue
        for pid in sorted(os.listdir(hp)):
            ep = os.path.join(hp, pid)
            if os.path.isdir(ep) and has_state_files(ep, pid, state):
                ids.append(pid)
                if limit is not None and len(ids) >= limit:
                    return ids
    return ids


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def process(pdbID):
    """Run the core pipeline for one entry. Returns a result dict (never raises)."""
    cfg = _CFG
    t0 = time.monotonic()
    out_dir = os.path.join(cfg["output_dir"], pdbID)
    result = {"pdbID": pdbID, "status": "error", "n": 0,
              "runtime": 0.0, "error": "", "rows": [], "header": None}
    try:
        entry = entry_dir_for(cfg["root"], pdbID)
        if not os.path.isdir(entry):
            result.update(status="skip", error="entry dir missing")
            return result
        os.makedirs(out_dir, exist_ok=True)
        mtz, pdb = prepare_inputs(pdbID, entry, cfg["state"], out_dir)
        reslo, reshi = read_resolution(entry, mtz)
        res = run_alchemy(pdbID, mtz, pdb, out_dir, reslo, reshi, env=cfg["env"])
        rows, header = run_analysis(pdbID, res["stats_out"],
                                    METALS_SET, cfg["cofactors"])
        result.update(status="ok", n=len(rows), rows=rows, header=header)
    except FileNotFoundError as e:
        result.update(status="skip", error=f"missing input: {e}"[:300])
    except Exception as e:  # noqa: BLE001 - one bad entry must not kill the batch
        result.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
    finally:
        result["runtime"] = round(time.monotonic() - t0, 2)
        if not cfg["keep"] and os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_done(manifest_path):
    """PDB ids already recorded in an existing manifest (for --resume)."""
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if row:
                    done.add(row[0])
    return done


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Batch Alchemy core pipeline over PDB-REDO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--id", help="process a single PDB id (else batch the root)")
    ap.add_argument("--pdb-redo-root", default=DEFAULT_ROOT,
                    help="root of the PDB-REDO mirror")
    ap.add_argument("--refine-state", choices=["final", "0cyc", "besttls"],
                    default="final", help="which refinement state to analyze")
    ap.add_argument("--max-pdbs", type=int, default=None,
                    help="debug cap: process only the first N entries")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                    help="number of worker processes")
    ap.add_argument("--output-dir", default=os.path.join(REPO_DIR, "output"))
    ap.add_argument("--ccp4-setup", default=None,
                    help="CCP4 setup script to source (e.g. .../bin/ccp4.setup-sh)")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="keep per-entry maps/logs (default: delete after extract)")
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in manifest.csv")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    env = resolve_env(args.ccp4_setup)
    verify_ccp4(env)
    os.makedirs(args.output_dir, exist_ok=True)
    cofactors = load_cofactors()

    root = args.pdb_redo_root
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    stats_path = os.path.join(args.output_dir, "metal_stats_all.csv")

    if args.id:
        ids = [args.id]
    else:
        print(f"Enumerating entries under {root} (state={args.refine_state}) ...",
              flush=True)
        # Early-stop only when capping and not resuming (resume needs the full set).
        limit = args.max_pdbs if (args.max_pdbs and not args.resume) else None
        ids = enumerate_entries(root, args.refine_state, limit=limit)
    if args.resume:
        done = load_done(manifest_path)
        ids = [i for i in ids if i not in done]
    if args.max_pdbs is not None:
        ids = ids[:args.max_pdbs]

    if not ids:
        print("No entries to process.", flush=True)
        return 0
    print(f"Processing {len(ids)} entr{'y' if len(ids) == 1 else 'ies'} "
          f"with {args.workers} worker(s) ...", flush=True)

    cfg = {"root": root, "state": args.refine_state, "env": env,
           "output_dir": args.output_dir, "cofactors": cofactors,
           "keep": args.keep_intermediates}

    append = args.resume and os.path.exists(manifest_path)
    man_fh = open(manifest_path, "a" if append else "w", newline="")
    stats_fh = open(stats_path, "a" if append else "w", newline="")
    man_w = csv.writer(man_fh)
    stats_w = csv.writer(stats_fh)
    if not append:
        man_w.writerow(["pdbID", "status", "n_metals", "runtime_s", "error"])
    stats_header_written = append and os.path.getsize(stats_path) > 0
    stats_header = None

    counts = {"ok": 0, "skip": 0, "error": 0}
    n_rows = 0
    try:
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for k, r in enumerate(pool.imap_unordered(process, ids, chunksize=1), 1):
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                man_w.writerow([r["pdbID"], r["status"], r["n"],
                                r["runtime"], r["error"]])
                man_fh.flush()
                if r["rows"]:
                    if not stats_header_written and r["header"]:
                        stats_header = ["pdbID", "category"] + r["header"]
                        stats_w.writerow(stats_header)
                        stats_header_written = True
                    for row in r["rows"]:
                        stats_w.writerow([row["pdbID"], row["category"]] + row["fields"])
                        n_rows += 1
                    stats_fh.flush()
                if k % 200 == 0 or k == len(ids):
                    print(f"[{k}/{len(ids)}] ok={counts['ok']} "
                          f"skip={counts['skip']} error={counts['error']} "
                          f"rows={n_rows}", flush=True)
    finally:
        man_fh.close()
        stats_fh.close()

    print(f"Done. ok={counts['ok']} skip={counts['skip']} error={counts['error']}; "
          f"{n_rows} metal/cofactor rows -> {stats_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
