#!/usr/bin/env python
"""Batch-run the Alchemy core pipeline over PDB-REDO entries.

For each PDB entry this computes 2mFo-DFc and mFo-DFc maps (CCP4 `fft`) and runs
`edstats`, then extracts per-atom real-space statistics for metal ions and
metal-containing cofactors. Results are streamed to three CSVs under --output-dir:

  metal_stats_all.csv  -- one row per selected metal site
  metal_bonds_all.csv  -- one row per retained candidate contact
  manifest.csv         -- one row per entry with status and provenance

Requirements
------------
* CCP4 `fft` and `edstats` on PATH -- either already sourced, or via --ccp4-setup
  pointing at a CCP4 setup script (e.g. <CCP4>/bin/ccp4.setup-sh).
* Run under a Python environment with gemmi>=0.7.0 and requests.

Examples
--------
  conda run -n metal python src/main.py --id 109m \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
  conda run -n metal python src/main.py --max-pdbs 20 --workers 4 \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
"""
import argparse
import csv
import gzip
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count
import requests

from density_analysis import run_density_analysis
from metal_identification import metals, uncommonMetals, load_cofactor_ids, extract_metal_statistics
from build_metallocofactor_catalog import refresh_cofactors_if_needed, active_cofactors_path
from bond_analysis import (
    BOND_COLUMNS,
    NAN,
    STATS_EXTRA_COLUMNS,
    load_structure,
    run_bond_analysis,
    stats_extra_values,
)
from ccp4_setup import (
    ccp4_tools_available,
    find_ccp4_setup,
    load_ccp4_setup_config,
    save_ccp4_setup,
)


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = "/datasets/bioinfo/pdb-redo"
METALS_SET = set(metals) | set(uncommonMetals)

MODEL_POLICY = "first"
ALTLOC_POLICY = "highest-mean-occupancy-residue-conformer"
SYMMETRY_POLICY = "crystal-inclusive-primary-with-explicit-summary"
ALCHEMY_VERSION = "0.1.0"

MANIFEST_COLUMNS = [
    "pdbID", "status", "retryable", "n_metals", "n_bonds", "runtime_s",
    "reason_codes", "warning_codes", "error", "alchemy_version", "alchemy_commit",
    "gemmi_version", "ccp4_version", "refinement_state",
    "source_coordinate_format", "analysis_coordinate_format",
    "coordinate_conversion_performed", "source_coordinate_path",
    "analysis_coordinate_path", "model_policy", "input_model_count",
    "model_analyzed", "multi_model_structure", "altloc_policy",
    "symmetry_contact_policy",
]

# config dict shared with worker processes (set once per worker by _init_worker)
_CFG = None


# --------------------------------------------------------------------------- #
# CCP4 environment
# --------------------------------------------------------------------------- #
def _normalize_path_key(env):
    """Ensure the PATH variable is accessible under the exact key "PATH".

    Different platforms/shells report it with different casing (Windows'
    `set` reports "Path"; Unix shells report "PATH"). Python dict lookups
    are case-sensitive, so downstream code that does env.get("PATH") would
    silently miss it if the key came back in a different case. This finds
    any case-variant of "PATH" and consolidates it under the exact
    all-caps key, leaving every other variable untouched.
    """
    for k in list(env):
        if k.upper() == "PATH" and k != "PATH":
            env["PATH"] = env.pop(k)
    return env


def resolve_env(ccp4_setup):
    """Return the environment dict to run CCP4 under.

    If `ccp4_setup` is given and looks like a bash setup script, source it in a
    bash subshell and capture the resulting environment. If it is a Windows batch
    launcher, run it in a cmd shell and capture the resulting environment instead.
    If no setup script is provided, fall back to the current environment.
    """
    if not ccp4_setup:
        return os.environ.copy()
    if not os.path.exists(ccp4_setup):
        raise SystemExit(f"CCP4 setup file not found: {ccp4_setup}")

    if os.path.splitext(ccp4_setup)[1].lower() == ".bat":
        tmp_cmd = os.path.join(os.environ.get("TEMP", os.getcwd()), "ccp4_env.cmd")
        with open(tmp_cmd, "w", encoding="utf-8") as fh:
            fh.write(f'@echo off\r\ncall "{ccp4_setup}"\r\nset\r\n')
        out = subprocess.run(["cmd", "/c", tmp_cmd], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"Failed to run CCP4 setup {ccp4_setup}:\n{out.stderr}")
        env = {}
        for line in out.stdout.splitlines():
            if "=" in line and not line.startswith("CMD") and not line.startswith("C:\\"):
                k, v = line.split("=", 1)
                env[k] = v
        return _normalize_path_key({**os.environ.copy(), **env})

    cmd = f"source {shlex.quote(ccp4_setup)} >/dev/null 2>&1 && env -0"
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"Failed to source CCP4 setup {ccp4_setup}:\n{out.stderr}")
    env = {}
    for chunk in out.stdout.split("\0"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            env[k] = v
    return _normalize_path_key({**os.environ.copy(), **env})


def verify_ccp4(env):
    missing = [t for t in ("fft", "edstats")
               if shutil.which(t, path=env.get("PATH")) is None]
    if missing:
        raise SystemExit(
            "CCP4 tools (fft, edstats) were not found on PATH. "
            "Set them up once with --configure-ccp4 /path/to/ccp4.setup-sh, "
            "export CCP4_SETUP=/path/to/ccp4.setup-sh, or source CCP4 in your shell before running."
        )


def default_ccp4_config_files(repo_dir=REPO_DIR):
    return [
        os.path.expanduser("~/.config/alchemy/ccp4.json"),
        os.path.expanduser("~/.alchemy/ccp4.json"),
        os.path.join(repo_dir, ".alchemy", "ccp4.json"),
    ]


def resolve_ccp4_environment(args):
    config_files = default_ccp4_config_files(REPO_DIR)
    config = load_ccp4_setup_config(config_files=config_files)
    if args.configure_ccp4:
        setup_path = os.path.abspath(os.path.expanduser(args.configure_ccp4))
        if not os.path.exists(setup_path):
            raise SystemExit(f"CCP4 setup file not found: {setup_path}")

        env = resolve_env(setup_path)
        try:
            verify_ccp4(env)
        except SystemExit as e:
            raise SystemExit(
                f"Ran {setup_path}, but CCP4 tools are still not available. {e}"
            ) from None

        saved = save_ccp4_setup(setup_path, config_files=config_files)
        print(f"Verified fft and edstats are available; saved CCP4 setup "
              f"path to {', '.join(saved)}", flush=True)
        return None, None

    environment = os.environ.copy()
    if ccp4_tools_available(environment):
        return environment, None

    if args.ccp4_setup:
        if not os.path.exists(args.ccp4_setup):
            raise SystemExit(f"CCP4 setup file not found: {args.ccp4_setup}")
        env = resolve_env(args.ccp4_setup)
        verify_ccp4(env)
        return env, args.ccp4_setup

    ccp4_setup = find_ccp4_setup(
        explicit_setup=None,
        env=environment,
        config=config,
        config_files=config_files,
    )
    if ccp4_setup is None:
        raise SystemExit(
            "CCP4 tools (fft, edstats) were not found on PATH and no setup file could be auto-detected. "
            "Set them up once with --configure-ccp4 /path/to/ccp4.setup-sh, "
            "export CCP4_SETUP=/path/to/ccp4.setup-sh, or source CCP4 in your shell before running."
        )
    env = resolve_env(ccp4_setup)
    verify_ccp4(env)
    return env, ccp4_setup


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
    # EDSTATS consumes PDB coordinates, whose chain field is one character.
    # Shorten deterministically before writing, then analyze this exact PDB so
    # EDSTATS and Alchemy never join identifiers from different representations.
    structure.shorten_chain_names()
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


def read_resolution(entry_dir, mtz_path, data_json_path=None):
    """Return (reslo, reshi) -- low/high resolution limits for edstats.

    Prefer a supplied data.json (or PDB-REDO data.json) when available;
    fall back to the MTZ via gemmi.
    """
    dj = data_json_path or os.path.join(entry_dir, "data.json")
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


def _download_stream(url, dst, timeout=30):
    """Download URL to dst. Raise FileNotFoundError on non-200."""
    try:
        r = requests.get(url, stream=True, timeout=timeout)
    except Exception as e:  # network/connection
        raise FileNotFoundError(f"{url}: {e}")
    if r.status_code != 200:
        raise FileNotFoundError(f"{url}: status {r.status_code}")
    
    tmp = f"{dst}.{os.getpid()}.part"
    try:
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(8192):
                if chunk:
                    fh.write(chunk)
        os.replace(tmp, dst)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dst


def download_entry_to_cache(pdbID, cache_root, state):
    """Download required files for `pdbID` into a mirror-like `cache_root`.

    This tries common PDB-REDO filenames for each refinement `state` and
    uncompresses when needed so the cache matches the layout expected by the
    rest of the pipeline.
    """
    base = f"https://pdb-redo.eu/db/{pdbID}/"
    entry = entry_dir_for(cache_root, pdbID)
    os.makedirs(entry, exist_ok=True)
    got = []
    # helper to try url and optionally un-gzip into final name
    def try_fetch(name, want_uncompress=False):
        url = base + name
        dst = os.path.join(entry, name)
        try:
            _download_stream(url, dst)
            got.append(name)
            return True
        except FileNotFoundError:
            return False

    # Download per-state expected files
    if state == "final":
        # prefer uncompressed final files, fall back to .gz then uncompress
        for fname in (f"{pdbID}_final.mtz", f"{pdbID}_final.pdb", "data.json"):
            if try_fetch(fname):
                continue
            gz = fname + ".gz"
            if try_fetch(gz):
                # if we need uncompressed final files, gunzip them
                if fname.endswith(".mtz") or fname.endswith(".pdb"):
                    _gunzip_to(os.path.join(entry, gz), os.path.join(entry, fname))
                    os.remove(os.path.join(entry, gz))
                    got.append(fname)
    elif state == "besttls":
        for fname in (f"{pdbID}_besttls.mtz.gz", f"{pdbID}_besttls.pdb.gz", "data.json"):
            if try_fetch(fname):
                continue
            # try uncompressed data.json
            if fname == "data.json":
                try_fetch("data.json")
    elif state == "0cyc":
        for fname in (f"{pdbID}_0cyc.mtz.gz", f"{pdbID}_0cyc.cif.gz", "data.json"):
            if try_fetch(fname):
                continue
            if fname == "data.json":
                try_fetch("data.json")

    # Verify we have the files required for the state
    if not has_state_files(entry, pdbID, state):
        raise FileNotFoundError(f"PDB-REDO entry {pdbID} missing files for state={state}")


def ensure_entry_available(pdbID, mirror_root, cache_root, state):
    """Return the root (mirror or cache) that contains the required files.

    Preference order: mirror_root (full local mirror) -> cache_root (auto-download).
    Raises FileNotFoundError when unavailable.
    """
    # 1) check full mirror specified by user
    mirror_entry = entry_dir_for(mirror_root, pdbID)
    if os.path.isdir(mirror_entry) and has_state_files(mirror_entry, pdbID, state):
        return mirror_root
    # 2) check cache
    cache_entry = entry_dir_for(cache_root, pdbID)
    if os.path.isdir(cache_entry) and has_state_files(cache_entry, pdbID, state):
        return cache_root
    # 3) try to download into cache
    download_entry_to_cache(pdbID, cache_root, state)
    if os.path.isdir(cache_entry) and has_state_files(cache_entry, pdbID, state):
        return cache_root
    raise FileNotFoundError(pdbID)


def resolve_manual_inputs(pdbID, pdb_file=None, mtz_file=None, cif_file=None, work_dir=None):
    """Return (mtz_path, pdb_path) for a manually supplied local input set."""
    if not mtz_file:
        raise ValueError("manual mode requires --mtz-file")
    if not os.path.exists(mtz_file):
        raise FileNotFoundError(f"mtz file not found: {mtz_file}")

    if pdb_file:
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"pdb file not found: {pdb_file}")
        return mtz_file, pdb_file

    if cif_file:
        if not os.path.exists(cif_file):
            raise FileNotFoundError(f"cif file not found: {cif_file}")
        target_pdb = os.path.join(work_dir or os.getcwd(), f"{pdbID}.pdb")
        return mtz_file, _cif_to_pdb(cif_file, target_pdb)

    raise ValueError("manual mode requires --pdb-file or --cif-file")


def infer_pdb_id_from_path(path):
    """Infer a 4-char PDB id from a local file name if possible."""
    if not path:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"([A-Za-z0-9]{4})(?:_.*)?$", stem)
    return m.group(1).lower() if m else None


def enumerate_entries(root, state, limit=None):
    """All PDB ids under `root` that have the required files for `state`.

    If `limit` is given, stop after collecting that many (sorted) ids -- this
    keeps small --max-pdbs debug runs fast instead of walking all ~24k entries.
    """
    ids = []
    skipped = 0
    for hashdir in sorted(os.listdir(root)):
        hp = os.path.join(root, hashdir)
        if not os.path.isdir(hp):
            continue
        try:
            entries = sorted(os.listdir(hp))
        except (PermissionError, OSError) as e:
            # Common with partially-synced/locked-down mirrors: skip rather than
            # aborting the whole enumeration on one unreadable hashdir.
            skipped += 1
            print(f"  warning: skipping unreadable dir {hp}: {e}", flush=True)
            continue
        for pid in entries:
            ep = os.path.join(hp, pid)
            if os.path.isdir(ep) and has_state_files(ep, pid, state):
                ids.append(pid)
                if limit is not None and len(ids) >= limit:
                    return ids
    if skipped:
        print(f"  note: skipped {skipped} unreadable hashdir(s) under {root}",
              flush=True)
    return ids


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _coordinate_provenance(cfg):
    manual = cfg.get("manual_inputs")
    if manual:
        converted = bool(manual.get("cif_file") and not manual.get("pdb_file"))
        return ("mmcif" if converted else "pdb", "pdb", converted)
    converted = cfg["state"] == "0cyc"
    return ("mmcif" if converted else "pdb", "pdb", converted)


def _source_coordinate_path(cfg, pdb_id, entry, analysis_path):
    manual = cfg.get("manual_inputs")
    if manual:
        return manual.get("pdb_file") or manual.get("cif_file") or ""
    if cfg["state"] == "besttls":
        return os.path.join(entry, f"{pdb_id}_besttls.pdb.gz")
    if cfg["state"] == "0cyc":
        return os.path.join(entry, f"{pdb_id}_0cyc.cif.gz")
    return analysis_path


def _alchemy_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_DIR,
            capture_output=True, text=True, check=True)
        commit = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_DIR, capture_output=True, text=True, check=True)
        return commit + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _gemmi_version():
    try:
        import gemmi
        return gemmi.__version__
    except Exception:
        return "unknown"


def _ccp4_version(env):
    for key in ("CCP4_VERSION", "CCP4_VERSION_CODE", "CCP4VER"):
        if env.get(key):
            return env[key]
    ccp4_root = env.get("CCP4", "")
    return os.path.basename(ccp4_root.rstrip(os.sep)) if ccp4_root else "unknown"


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def process(pdbID):
    """Run the core pipeline for one entry. Returns a result dict (never raises)."""
    cfg = _CFG
    t0 = time.monotonic()
    out_dir = os.path.join(cfg["output_dir"], pdbID)
    source_format, analysis_format, converted = _coordinate_provenance(cfg)
    result = {"pdbID": pdbID, "status": "error", "n": 0,
              "runtime": 0.0, "error": "", "rows": [], "header": None,
              "bond_rows": [], "n_bonds": 0, "retryable": True,
              "reason_codes": [], "warning_codes": [],
              "alchemy_version": ALCHEMY_VERSION,
              "alchemy_commit": cfg["alchemy_commit"],
              "gemmi_version": cfg["gemmi_version"],
              "ccp4_version": cfg["ccp4_version"],
              "refinement_state": cfg["state"],
              "source_coordinate_format": source_format,
              "analysis_coordinate_format": analysis_format,
              "coordinate_conversion_performed": converted,
              "source_coordinate_path": "", "analysis_coordinate_path": "",
              "model_policy": MODEL_POLICY, "input_model_count": "",
              "model_analyzed": "", "multi_model_structure": "",
              "altloc_policy": ALTLOC_POLICY,
              "symmetry_contact_policy": SYMMETRY_POLICY}
    try:
        if cfg.get("manual_inputs"):
            os.makedirs(out_dir, exist_ok=True)
            mtz, pdb = resolve_manual_inputs(
                pdbID,
                pdb_file=cfg["manual_inputs"].get("pdb_file"),
                mtz_file=cfg["manual_inputs"].get("mtz_file"),
                cif_file=cfg["manual_inputs"].get("cif_file"),
                work_dir=out_dir,
            )
            entry = os.path.dirname(pdb) or out_dir
            data_json = cfg["manual_inputs"].get("data_json")
            reslo, reshi = read_resolution(entry, mtz, data_json_path=data_json)
        else:
            if cfg["allow_download"]:
                used_root = ensure_entry_available(pdbID, cfg["mirror_root"], cfg["cache_root"], cfg["state"])
                entry = entry_dir_for(used_root, pdbID)
            else:
                entry = entry_dir_for(cfg["root"], pdbID)
            if not os.path.isdir(entry):
                result.update(status="skip", error="entry dir missing")
                return result
            os.makedirs(out_dir, exist_ok=True)
            mtz, pdb = prepare_inputs(pdbID, entry, cfg["state"], out_dir)
            reslo, reshi = read_resolution(entry, mtz)
        result.update(
            source_coordinate_path=_source_coordinate_path(
                cfg, pdbID, entry, pdb),
            analysis_coordinate_path=pdb,
        )
        res = run_density_analysis(pdbID, mtz, pdb, out_dir, reslo, reshi, env=cfg["env"])
        structure = load_structure(pdbID, pdb)
        result.update(
            analysis_coordinate_format=structure.analysis_coordinate_format,
            input_model_count=structure.input_model_count,
            model_analyzed=structure.model_analyzed,
            multi_model_structure=structure.multi_model_structure,
            warning_codes=list(structure.warning_codes),
        )

        rows, header = extract_metal_statistics(pdbID, res["stats_out"],
                                    METALS_SET, cfg["cofactors"], structure=structure)
        # Reaching this point means the entry's core inputs and density stage
        # succeeded. Any limitations discovered below are terminal unless a
        # later stage explicitly identifies a transient failure.
        result["retryable"] = False

        identification_reason_codes = []
        for row in rows:
            mapping_status = row.get("coordinate_mapping_status", "")
            site_status = row.get("selected_metal_site_status", "")
            if mapping_status == "coordinate_residue_not_found":
                identification_reason_codes.append(
                    "cofactor_coordinate_join_failed")
            elif mapping_status == "multiple_coordinate_residues":
                identification_reason_codes.append(
                    "ambiguous_coordinate_residue_join")
            elif site_status == "no_selected_metal":
                identification_reason_codes.append(
                    "cofactor_without_selected_metal")
        identification_reason_codes = list(dict.fromkeys(
            identification_reason_codes))

        bond_rows = []
        site_summaries = {}
        bond_meta = {"partial_reason_codes": [],
                     "warning_codes": list(structure.warning_codes),
                     "messages": [], "retryable": False}
        
        if cfg["bonds"]:
            # A bond-stage failure must not lose the edstats rows already computed.
            try:
                bond_rows, site_summaries, bond_meta = run_bond_analysis(
                    pdbID, pdb, entry, rows, header,
                    {"data_json": data_json if cfg.get("manual_inputs") else os.path.join(entry, "data.json"),
                     "pdb_path": pdb, "mtz_path": mtz, "resolution": reshi}, structure=structure)
            except Exception as e:  # noqa: BLE001
                result["error"] = f"bond: {type(e).__name__}: {e}"[:300]
                result["reason_codes"] = ["bond_stage_failure"]
                result["retryable"] = True
        if header:
            header = header + ["aa_geometry_coverage"] + STATS_EXTRA_COLUMNS
        for row in rows:
            summary = dict(site_summaries.get(row.get("site_key"), {}))
            summary["coordinate_mapping_status"] = row.get(
                "coordinate_mapping_status", "")
            summary["selected_metal_site_status"] = row.get(
                "selected_metal_site_status", "")
            coverage = summary.get("geometry_coverage_crystal", NAN)
            if isinstance(coverage, float) and not math.isfinite(coverage):
                coverage = summary.get("geometry_coverage_explicit", NAN)
            extra = stats_extra_values(structure, row.get("site"), summary)
            row["fields"] = (row["fields"] + [coverage] +
                             [extra[column] for column in STATS_EXTRA_COLUMNS])

        partial_reason_codes = list(dict.fromkeys(
            identification_reason_codes +
            list(bond_meta["partial_reason_codes"])))
        result["reason_codes"] = list(dict.fromkeys(
            result["reason_codes"] + partial_reason_codes))
        reason_messages = {
            "cofactor_coordinate_join_failed":
                "cofactor EDSTATS row did not match a coordinate residue",
            "ambiguous_coordinate_residue_join":
                "EDSTATS row matched multiple coordinate residues",
            "cofactor_without_selected_metal":
                "matched cofactor has no selected configured metal site",
        }
        messages = [reason_messages[code]
                    for code in identification_reason_codes]
        messages.extend(bond_meta["messages"])
        if messages:
            existing_error = result["error"]
            result["error"] = "; ".join(
                ([existing_error] if existing_error else []) + messages)[:300]
        if bond_meta.get("retryable", False):
            result["retryable"] = True
        result["warning_codes"] = list(dict.fromkeys(
            result["warning_codes"] + bond_meta.get("warning_codes", [])))
        status = "partial" if result["reason_codes"] else "ok"
        if status == "ok":
            result["retryable"] = False
        # Count coordinate-model metal sites, not emitted statistics rows.
        # A failed EDSTATS join can leave a diagnostic row without a site even
        # though bond analysis still found and evaluated the deposited metal.
        selected_site_count = len(structure.metal_atoms(
            METALS_SET, canonical=True))
        result.update(status=status, n=selected_site_count,
                      rows=rows, header=header,
                      bond_rows=bond_rows, n_bonds=len(bond_rows))
    except FileNotFoundError as e:
        result.update(status="skip", retryable=True,
                      reason_codes=["missing_input"],
                      error=f"missing input: {e}"[:300])
    except Exception as e:  # noqa: BLE001 - one bad entry must not kill the batch
        result.update(status="error", retryable=True,
                      reason_codes=["unexpected_processing_error"],
                      error=f"{type(e).__name__}: {e}"[:300])
    finally:
        result["runtime"] = round(time.monotonic() - t0, 2)
        if not cfg["keep"] and os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_done(manifest_path):
    """PDB IDs whose result is terminal in an existing manifest."""
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and {"pdbID", "status"}.issubset(reader.fieldnames):
                for row in reader:
                    status = row.get("status", "").strip().lower()
                    retryable = row.get("retryable", "").strip().lower()
                    terminal_partial = (status == "partial" and
                                        retryable in ("false", "0", "no"))
                    if status == "ok" or terminal_partial:
                        pdbID = row.get("pdbID", "").strip().lower()
                        if pdbID:
                            done.add(pdbID)
    return done


def remove_csv_rows_for_ids(path, pdb_ids):
    """Atomically remove data rows for retried PDB IDs, preserving the header."""
    retry_ids = {pdbID.lower() for pdbID in pdb_ids}
    if not retry_ids or not os.path.exists(path):
        return

    directory = os.path.dirname(os.path.abspath(path))
    original_mode = os.stat(path).st_mode
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",
                                    suffix=".tmp", dir=directory, text=True)
    try:
        removed = False
        with open(path, newline="") as src, os.fdopen(fd, "w", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            header = next(reader, None)
            if header is not None:
                writer.writerow(header)
            for row in reader:
                if row and row[0].strip().lower() in retry_ids:
                    removed = True
                    continue
                writer.writerow(row)
        if removed:
            os.chmod(tmp_path, original_mode)
            os.replace(tmp_path, path)
        else:
            os.unlink(tmp_path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _csv_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="") as handle:
        return next(csv.reader(handle), None)


def validate_resume_schemas(manifest_path, stats_path, bonds_path,
                            bonds_enabled=True):
    """Refuse to append migration rows beneath an incompatible old header."""
    manifest_header = _csv_header(manifest_path)
    if manifest_header is not None and manifest_header != MANIFEST_COLUMNS:
        raise ValueError(
            "Existing manifest.csv uses an incompatible schema; choose a new "
            "--output-dir for this Gemmi migration run.")

    stats_header = _csv_header(stats_path)
    if stats_header is not None:
        expected_suffix = ["aa_geometry_coverage"] + STATS_EXTRA_COLUMNS
        if (stats_header[:2] != ["pdbID", "category"] or
                stats_header[-len(expected_suffix):] != expected_suffix):
            raise ValueError(
                "Existing metal_stats_all.csv uses an incompatible schema; "
                "choose a new --output-dir for this Gemmi migration run.")

    if bonds_enabled:
        bonds_header = _csv_header(bonds_path)
        if bonds_header is not None and bonds_header != BOND_COLUMNS:
            raise ValueError(
                "Existing metal_bonds_all.csv uses an incompatible schema; "
                "choose a new --output-dir for this Gemmi migration run.")

def parse_pdb_id(value):
    if not re.fullmatch(r"[A-Za-z0-9]{4}", value):
        raise argparse.ArgumentTypeError(
            "PDB ID must contain exactly four alphanumeric characters")
    return value.lower()

def load_ids_from_file(path):
    """Return a list of PDB ids from a comma/newline-separated text file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"id file not found: {path}")
    ids = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in re.split(r"[,\s]+", line):
                token = token.strip()
                if not token:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9]{4}", token):
                    raise ValueError(f"invalid PDB id {token!r} at {path}:{lineno}")
                ids.append(token.lower())
    return list(dict.fromkeys(ids))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Batch Alchemy core pipeline over PDB-REDO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--id", type=parse_pdb_id, help="process a single PDB id (else batch the root)")
    ap.add_argument("--id-file", help="path to a file of PDB ids (comma- and/or newline-separated)")
    ap.add_argument("--pdb-file", help="path to a local PDB file for manual input mode")
    ap.add_argument("--mtz-file", help="path to a local MTZ file for manual input mode")
    ap.add_argument("--cif-file", help="path to a local mmCIF file for manual input mode")
    ap.add_argument("--data-json", help="optional path to a local data.json for manual input mode")
    ap.add_argument("--pdb-redo-root", default=DEFAULT_ROOT,
                    help="root of the PDB-REDO mirror")
    ap.add_argument("--pdb-redo-cache", default=os.path.join(REPO_DIR, "pdb-redo-cache"),
                    help="root of local cache for auto-downloaded PDB-REDO entries")
    ap.add_argument("--refine-state", choices=["final", "0cyc", "besttls"],
                    default="final", help="which refinement state to analyze")
    ap.add_argument("--max-pdbs", type=int, default=None,
                    help="debug cap: process only the first N entries")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                    help="number of worker processes")
    ap.add_argument("--output-dir", default=os.path.join(REPO_DIR, "output"))
    ap.add_argument("--ccp4-setup", default=None,
                    help="optional CCP4 setup script override (e.g. .../bin/ccp4.setup-sh)")
    ap.add_argument("--configure-ccp4", default=None,
                    help="save a CCP4 setup script path for future runs")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="keep per-entry maps/logs (default: delete after extract)")
    ap.add_argument("--resume", action="store_true",
                    help="skip terminal ok/partial results; retry retryable incomplete ids")
    ap.add_argument("--no-bonds", dest="bonds", action="store_false",
                    help="skip the metal-ligand bond-distance stage (edstats stats only)")
    ap.add_argument("--refresh-cofactors", action="store_true",
                    help="force a refresh of metallocofactors_id.txt from the CCD before running")
    ap.set_defaults(bonds=True)
    
    args = ap.parse_args(argv)
    if args.id and args.id_file:
        raise SystemExit("use either --id or --id-file, not both")
    return args


def main(argv=None):
    args = parse_args(argv)

    env, _ = resolve_ccp4_environment(args)
    if env is None:
        return 0
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        refresh_cofactors_if_needed(force=args.refresh_cofactors)
    except RuntimeError as e:
        print(str(e), flush=True)
        return 1
    cofactors = load_cofactor_ids(active_cofactors_path())

    root = args.pdb_redo_root
    cache_root = args.pdb_redo_cache
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    stats_path = os.path.join(args.output_dir, "metal_stats_all.csv")
    bonds_path = os.path.join(args.output_dir, "metal_bonds_all.csv")
    if args.resume:
        try:
            validate_resume_schemas(manifest_path, stats_path, bonds_path,
                                    bonds_enabled=args.bonds)
        except ValueError as exc:
            print(str(exc), flush=True)
            return 1

    manual_inputs = None
    if args.pdb_file or args.mtz_file or args.cif_file:
        pdbID = args.id or infer_pdb_id_from_path(args.pdb_file) or infer_pdb_id_from_path(args.mtz_file) or infer_pdb_id_from_path(args.cif_file)
        if not pdbID:
            print("Manual input mode requires --id or a file name that contains a 4-character PDB id.", flush=True)
            return 1
        manual_inputs = {
            "pdb_file": args.pdb_file,
            "mtz_file": args.mtz_file,
            "cif_file": args.cif_file,
            "data_json": args.data_json,
        }
        ids = [pdbID]
    elif args.id:
        # Ensure requested single entry is available locally (mirror or cache).
        try:
            used_root = ensure_entry_available(args.id, args.pdb_redo_root, cache_root, args.refine_state)
            if used_root != args.pdb_redo_root:
                print(f"Auto-downloaded {args.id} into cache at {cache_root}", flush=True)
            root = used_root
            ids = [args.id]
        except FileNotFoundError:
            print(f"Entry {args.id} not found locally and download failed.", flush=True)
            return 1
    elif args.id_file:
        try:
            ids = load_ids_from_file(args.id_file)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), flush=True)
            return 1
        print(f"Loaded {len(ids)} IDs from {args.id_file}", flush=True)
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

    if args.resume:
        # Retried partial/error/skip entries may already have result rows from a
        # previous attempt. Remove only those IDs so the replacement rows do not
        # create duplicates; each file is rewritten atomically.
        remove_csv_rows_for_ids(manifest_path, ids)
        remove_csv_rows_for_ids(stats_path, ids)
        remove_csv_rows_for_ids(bonds_path, ids)

    print(f"Processing {len(ids)} entr{'y' if len(ids) == 1 else 'ies'} "
          f"with {args.workers} worker(s) ...", flush=True)

    cfg = {"root": root, "mirror_root": args.pdb_redo_root,
           "cache_root": cache_root, "state": args.refine_state, "env": env,
           "output_dir": args.output_dir, "cofactors": cofactors,
           "keep": args.keep_intermediates, "bonds": args.bonds,
           "allow_download": bool(args.id or args.id_file),
           "manual_inputs": manual_inputs,
           "alchemy_commit": _alchemy_commit(),
           "gemmi_version": _gemmi_version(),
           "ccp4_version": _ccp4_version(env)}

    append = (args.resume and os.path.exists(manifest_path) and
              os.path.getsize(manifest_path) > 0)
    man_fh = open(manifest_path, "a" if append else "w", newline="")
    stats_fh = open(stats_path, "a" if append else "w", newline="")
    bonds_fh = open(bonds_path, "a" if append else "w", newline="") if args.bonds else None
    man_w = csv.DictWriter(man_fh, fieldnames=MANIFEST_COLUMNS)
    stats_w = csv.writer(stats_fh)
    bonds_w = csv.writer(bonds_fh) if bonds_fh else None
    if not append:
        man_w.writeheader()
    stats_header_written = append and os.path.getsize(stats_path) > 0
    bonds_header_written = bool(bonds_fh) and append and os.path.getsize(bonds_path) > 0

    counts = {"ok": 0, "partial": 0, "skip": 0, "error": 0}
    n_rows = 0
    n_bonds = 0
    try:
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for k, r in enumerate(pool.imap_unordered(process, ids, chunksize=1), 1):
                if r["rows"]:
                    if not stats_header_written and r["header"]:
                        stats_w.writerow(["pdbID", "category"] + r["header"])
                        stats_header_written = True
                    for row in r["rows"]:
                        stats_w.writerow([row["pdbID"], row["category"]] + row["fields"])
                        n_rows += 1
                    stats_fh.flush()
                if bonds_w and r["bond_rows"]:
                    if not bonds_header_written:
                        bonds_w.writerow(BOND_COLUMNS)
                        bonds_header_written = True
                    for b in r["bond_rows"]:
                        bonds_w.writerow([b[c] for c in BOND_COLUMNS])
                        n_bonds += 1
                    bonds_fh.flush()
                # The manifest is the completion marker for --resume, so write
                # it only after this entry's result rows have been flushed.
                manifest_row = {column: r.get(column, "")
                                for column in MANIFEST_COLUMNS}
                manifest_row.update(
                    n_metals=r["n"], n_bonds=r["n_bonds"],
                    runtime_s=r["runtime"],
                    reason_codes="|".join(r.get("reason_codes", [])),
                    warning_codes="|".join(r.get("warning_codes", [])),
                )
                man_w.writerow(manifest_row)
                man_fh.flush()
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                if k % 200 == 0 or k == len(ids):
                    print(f"[{k}/{len(ids)}] ok={counts['ok']} "
                          f"partial={counts['partial']} skip={counts['skip']} "
                          f"error={counts['error']} "
                          f"rows={n_rows} bonds={n_bonds}", flush=True)
    finally:
        man_fh.close()
        stats_fh.close()
        if bonds_fh:
            bonds_fh.close()

    print(f"Done. ok={counts['ok']} partial={counts['partial']} "
          f"skip={counts['skip']} error={counts['error']}; "
          f"{n_rows} metal/cofactor rows -> {stats_path}", flush=True)
    if args.bonds:
        print(f"      {n_bonds} bond rows -> {bonds_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
