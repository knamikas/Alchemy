#!/usr/bin/env python
"""Batch-run the Alchemy core pipeline over PDB-REDO entries.

For each PDB entry this validates/corrects its Fourier map coefficients with
CCP4 `mtzfix`, computes 2mFo-DFc and mFo-DFc maps with `fft`, and runs
`edstats`, then extracts per-atom real-space statistics for metal ions and
metal-containing cofactors. Results are streamed to three CSVs under
--output-dir:

  metal_stats_all.csv  -- one row per selected metal site
  metal_bonds_all.csv  -- one row per retained first-sphere contact
  manifest.csv         -- one row per entry with status and provenance

Requirements
------------
* CCP4 `mtzfix`, `fft`, and `edstats` on PATH -- either already sourced, or via
  --ccp4-setup pointing at a CCP4 setup script
  (e.g. <CCP4>/bin/ccp4.setup-sh).
* Run under a Python environment with gemmi>=0.7.0.

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
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import urlopen

from density_analysis import run_density_analysis
from metal_identification import metals, uncommonMetals, load_cofactor_ids, extract_metal_statistics
from build_metallocofactor_catalog import refresh_cofactors_if_needed, active_cofactors_path
from structure_analysis import RESNAME_REMARK_PREFIX
from bond_analysis import (
    BOND_COLUMNS,
    NAN,
    STATS_EXTRA_COLUMNS,
    load_structure,
    run_bond_analysis,
    stats_extra_values,
)
from ccp4_setup import (
    REQUIRED_CCP4_TOOLS,
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
SYMMETRY_POLICY = (
    "image-inclusive-primary-with-crystallographic-and-strict-ncs-provenance"
)
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
_CFG: Optional[Dict[str, Any]] = None


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
    missing = [t for t in REQUIRED_CCP4_TOOLS
               if shutil.which(t, path=env.get("PATH")) is None]
    if missing:
        raise SystemExit(
            f"Required CCP4 tools were not found on PATH: {', '.join(missing)}. "
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
        print(f"Verified {', '.join(REQUIRED_CCP4_TOOLS)} are available; saved CCP4 setup "
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
            f"Required CCP4 tools ({', '.join(REQUIRED_CCP4_TOOLS)}) were not "
            "found on PATH and no setup file could be auto-detected. "
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
    with gzip.GzipFile(src_gz, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return dst


def _cif_occupancy_by_serial(cif_path) -> Dict[int, str]:
    """Return raw mmCIF occupancy tokens keyed by ``_atom_site.id``.

    Gemmi represents ``.`` and ``?`` occupancy as 1.0 in a Structure, so the
    raw CIF loop must be read before conversion if missingness is to survive.
    """
    import gemmi

    document = gemmi.cif.read(cif_path)
    atom_blocks = []
    for block in document:
        atom_ids = list(block.find_values("_atom_site.id"))
        if atom_ids:
            atom_blocks.append((block, atom_ids))
    if len(atom_blocks) != 1:
        raise ValueError(
            "mmCIF conversion requires exactly one block with atom_site records")

    block, atom_ids = atom_blocks[0]
    occupancies = list(block.find_values("_atom_site.occupancy"))
    if not occupancies:
        occupancies = ["?"] * len(atom_ids)
    elif len(occupancies) != len(atom_ids):
        raise ValueError(
            "mmCIF atom_site occupancy count does not match atom count")

    by_serial: Dict[int, str] = {}
    for atom_id, occupancy in zip(atom_ids, occupancies):
        try:
            serial = int(atom_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"mmCIF atom_site id is not an integer: {atom_id!r}") from exc
        if serial in by_serial:
            raise ValueError(f"duplicate mmCIF atom_site id: {serial}")
        by_serial[serial] = occupancy
    return by_serial


def _residue_conversion_records(structure, converted_structure):
    """Pair source mmCIF residue names with names written to legacy PDB."""
    source_by_author = {}
    source_order = []
    for model_index, model in enumerate(structure):
        for chain in model:
            for residue in chain:
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"mmCIF residue {residue.name!r} has no author number")
                insertion = str(residue.seqid.icode)
                if insertion in ("", " ", "\x00", ".", "?"):
                    insertion = ""
                key = (model_index, str(chain.name), f"{number}{insertion}")
                source_order.append(key)
                source_by_author.setdefault(key, []).append((
                    str(residue.name),
                    tuple((str(atom.name), str(atom.element.name))
                          for atom in residue),
                ))

    records = []
    converted_by_author = {}
    converted_order = []
    for model_index, model in enumerate(converted_structure):
        for chain in model:
            for residue in chain:
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"converted residue {residue.name!r} has no author number")
                insertion = str(residue.seqid.icode)
                if insertion in ("", " ", "\x00", ".", "?"):
                    insertion = ""
                converted_chain = str(chain.name)
                converted_resnum = f"{number}{insertion}"
                key = (model_index, converted_chain, converted_resnum)
                converted_order.append(key)
                converted_by_author.setdefault(key, []).append((
                    str(residue.name),
                    tuple((str(atom.name), str(atom.element.name))
                          for atom in residue),
                ))

    if converted_order != source_order:
        raise ValueError("PDB conversion changed residue ordering")
    if set(converted_by_author) != set(source_by_author):
        raise ValueError("PDB conversion changed residue author identifiers")
    for key, source_residues in source_by_author.items():
        converted_residues = converted_by_author[key]
        if len(converted_residues) != len(source_residues):
            raise ValueError(
                "PDB conversion changed duplicate residue multiplicity")
        model_index, converted_chain, converted_resnum = key
        for source, converted in zip(source_residues, converted_residues):
            source_name, source_atoms = source
            converted_name, converted_atoms = converted
            if source_atoms != converted_atoms:
                raise ValueError(
                    "PDB conversion changed residue atom membership")
            if converted_name != source_name:
                records.append((
                    model_index + 1,
                    converted_chain,
                    converted_resnum,
                    converted_name,
                    source_name,
                ))
    return records


def _write_cif_conversion_provenance(
        dst: str,
        missing_occupancies: List[bool],
        residue_records: List[Tuple[int, str, str, str, str]],
        ) -> None:
    """Blank unknown occupancies and embed reversible residue-name mappings."""
    with open(dst, encoding="utf-8", errors="strict", newline="") as handle:
        lines = handle.readlines()

    atom_line_indices = [
        index for index, line in enumerate(lines)
        if line[:6].strip().upper() in ("ATOM", "HETATM")
    ]
    if len(atom_line_indices) != len(missing_occupancies):
        raise ValueError(
            "PDB conversion output atom count does not match mmCIF input")
    for line_index, missing in zip(atom_line_indices, missing_occupancies):
        if not missing:
            continue
        line = lines[line_index]
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        body = body.ljust(60)
        lines[line_index] = body[:54] + "      " + body[60:] + newline

    remarks = [
        (
            f"{RESNAME_REMARK_PREFIX} {model_index} "
            f"{chain or '_'} {resnum} {converted_name} {source_name}\n"
        )
        for (model_index, chain, resnum, converted_name,
             source_name) in residue_records
    ]
    with open(dst, "w", encoding="utf-8", newline="") as handle:
        handle.writelines(remarks)
        handle.writelines(lines)


def _cif_to_pdb(cif_path, dst):
    """Convert mmCIF to PDB without discarding occupancy or CCD provenance."""
    import gemmi

    if not os.path.exists(cif_path):
        raise FileNotFoundError(cif_path)
    occupancy_by_serial = _cif_occupancy_by_serial(cif_path)
    structure = gemmi.read_structure(cif_path)
    structure_atoms = [
        atom for model in structure for chain in model
        for residue in chain for atom in residue
    ]
    if len(structure_atoms) != len(occupancy_by_serial):
        raise ValueError(
            "Gemmi structure atom count does not match mmCIF atom_site records")
    missing_occupancies = []
    for atom in structure_atoms:
        serial = atom.serial
        if serial is None or int(serial) not in occupancy_by_serial:
            raise ValueError(
                "Gemmi atom serial could not be matched to mmCIF atom_site id")
        missing_occupancies.append(
            occupancy_by_serial[int(serial)] in (".", "?"))

    structure.setup_entities()
    # EDSTATS consumes PDB coordinates, whose chain field is one character.
    # Shorten deterministically before writing, then analyze this exact PDB so
    # EDSTATS and Alchemy never join identifiers from different representations.
    structure.shorten_chain_names()
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    structure.write_pdb(dst)
    converted_structure = gemmi.read_structure(dst)
    residue_records = _residue_conversion_records(
        structure, converted_structure)
    _write_cif_conversion_provenance(
        dst, missing_occupancies, residue_records)
    return dst


def _first_model_pdb(pdb_path, dst):
    """Return a PDB containing only the deposited first coordinate model.

    The extraction is textual so atom records, occupancies, identifiers, and
    ordering remain exactly as deposited. Gemmi is used only to determine and
    verify the model count.
    """
    import gemmi

    structure = gemmi.read_structure(pdb_path)
    model_count = len(structure)
    if model_count == 0:
        raise ValueError("coordinate file contains no models")
    if model_count == 1:
        return pdb_path, model_count

    with open(pdb_path, encoding="utf-8", errors="replace", newline="") as fh:
        lines = fh.readlines()
    model_starts = [
        index for index, line in enumerate(lines)
        if line[:6].strip().upper() == "MODEL"
    ]
    if len(model_starts) < 2:
        raise ValueError(
            "Gemmi found multiple models but the PDB MODEL records could not "
            "be isolated")

    first_start, second_start = model_starts[:2]
    first_end = next(
        (index for index in range(first_start, second_start)
         if lines[index][:6].strip().upper() == "ENDMDL"),
        None,
    )
    if first_end is None:
        first_block = lines[first_start:second_start]
        first_block.append("ENDMDL\n")
    else:
        first_block = lines[first_start:first_end + 1]

    # NUMMDL describes the source ensemble and would be false in this
    # first-model-only analysis file. Other crystallographic header records are
    # retained because EDSTATS needs the same cell and symmetry metadata.
    header = [
        line for line in lines[:first_start]
        if line[:6].strip().upper() != "NUMMDL"
    ]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(header)
        fh.writelines(first_block)
        fh.write("END\n")

    if len(gemmi.read_structure(dst)) != 1:
        raise ValueError("failed to create a first-model-only analysis PDB")
    return dst, model_count


def _first_existing(*paths):
    return next((path for path in paths if os.path.exists(path)), None)


def prepare_inputs(pdbID, entry_dir, work_dir):
    """Return the final PDB-REDO ``(mtz_path, pdb_path)`` analysis inputs.

    Prefer the authoritative final mmCIF and convert it under Alchemy's
    provenance-preserving policy for EDSTATS. Use the PDB compatibility export
    only when mmCIF is unavailable. Compressed mirrors are accepted for either
    format.
    """
    mtz = _first_existing(
        os.path.join(entry_dir, f"{pdbID}_final.mtz"),
        os.path.join(entry_dir, f"{pdbID}_final.mtz.gz"),
    )
    if mtz is None:
        raise FileNotFoundError(
            os.path.join(entry_dir, f"{pdbID}_final.mtz"))
    if mtz.endswith(".gz"):
        mtz = _gunzip_to(
            mtz, os.path.join(work_dir, f"{pdbID}_final.mtz"))

    cif = _first_existing(
        os.path.join(entry_dir, f"{pdbID}_final.cif"),
        os.path.join(entry_dir, f"{pdbID}_final.cif.gz"),
    )
    if cif is not None:
        pdb = _cif_to_pdb(
            cif, os.path.join(work_dir, f"{pdbID}_final_from_cif.pdb"))
        return mtz, pdb

    pdb = _first_existing(
        os.path.join(entry_dir, f"{pdbID}_final.pdb"),
        os.path.join(entry_dir, f"{pdbID}_final.pdb.gz"),
    )
    if pdb is None:
        raise FileNotFoundError(
            f"{pdbID}_final.cif or {pdbID}_final.pdb")
    if pdb.endswith(".gz"):
        pdb = _gunzip_to(
            pdb, os.path.join(work_dir, f"{pdbID}_final.pdb"))
    return mtz, pdb


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


def has_final_files(entry_dir, pdbID):
    """Whether an entry has final map coefficients and usable coordinates."""
    mtz = _first_existing(
        os.path.join(entry_dir, f"{pdbID}_final.mtz"),
        os.path.join(entry_dir, f"{pdbID}_final.mtz.gz"),
    )
    coordinates = _first_existing(
        os.path.join(entry_dir, f"{pdbID}_final.cif"),
        os.path.join(entry_dir, f"{pdbID}_final.cif.gz"),
        os.path.join(entry_dir, f"{pdbID}_final.pdb"),
        os.path.join(entry_dir, f"{pdbID}_final.pdb.gz"),
    )
    return mtz is not None and coordinates is not None


def _download_stream(url, dst, timeout=30):
    """Download URL to dst. Raise FileNotFoundError on non-200."""
    try:
        response = urlopen(url, timeout=timeout)
    except HTTPError as e:
        raise FileNotFoundError(f"{url}: status {e.code}") from e
    except (OSError, ValueError) as e:  # network/connection or invalid URL
        raise FileNotFoundError(f"{url}: {e}") from e

    tmp = f"{dst}.{os.getpid()}.part"
    try:
        with response:
            status = response.getcode()
            if status != 200:
                raise FileNotFoundError(f"{url}: status {status}")
            with open(tmp, "wb") as fh:
                while chunk := response.read(8192):
                    fh.write(chunk)
        os.replace(tmp, dst)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dst


def download_entry_to_cache(pdbID, cache_root):
    """Download final PDB-REDO files into a mirror-like ``cache_root``."""
    base = f"https://pdb-redo.eu/db/{pdbID}/"
    entry = entry_dir_for(cache_root, pdbID)
    os.makedirs(entry, exist_ok=True)

    def try_fetch(name):
        url = base + name
        dst = os.path.join(entry, name)
        try:
            _download_stream(url, dst)
            return True
        except FileNotFoundError:
            return False

    def fetch_variant(name):
        if (_first_existing(
                os.path.join(entry, name),
                os.path.join(entry, name + ".gz")) is not None):
            return True
        return try_fetch(name) or try_fetch(name + ".gz")

    fetch_variant(f"{pdbID}_final.mtz")
    if not fetch_variant(f"{pdbID}_final.cif"):
        fetch_variant(f"{pdbID}_final.pdb")
    if not os.path.exists(os.path.join(entry, "data.json")):
        try_fetch("data.json")

    if not has_final_files(entry, pdbID):
        raise FileNotFoundError(
            f"PDB-REDO entry {pdbID} is missing final model files")


def ensure_entry_available(pdbID, mirror_root, cache_root):
    """Return the root (mirror or cache) containing the final model files.

    Preference order: mirror_root (full local mirror) -> cache_root (auto-download).
    Raises FileNotFoundError when unavailable.
    """
    # 1) check full mirror specified by user
    mirror_entry = entry_dir_for(mirror_root, pdbID)
    if os.path.isdir(mirror_entry) and has_final_files(mirror_entry, pdbID):
        return mirror_root
    # 2) check cache
    cache_entry = entry_dir_for(cache_root, pdbID)
    if os.path.isdir(cache_entry) and has_final_files(cache_entry, pdbID):
        return cache_root
    # 3) try to download into cache
    download_entry_to_cache(pdbID, cache_root)
    if os.path.isdir(cache_entry) and has_final_files(cache_entry, pdbID):
        return cache_root
    raise FileNotFoundError(pdbID)


def resolve_manual_inputs(pdbID, pdb_file=None, mtz_file=None, cif_file=None, work_dir=None):
    """Return (mtz_path, pdb_path) for a manually supplied local input set."""
    if not mtz_file:
        raise ValueError("manual mode requires --mtz-file")
    if not os.path.exists(mtz_file):
        raise FileNotFoundError(f"mtz file not found: {mtz_file}")

    if cif_file:
        if not os.path.exists(cif_file):
            raise FileNotFoundError(f"cif file not found: {cif_file}")
        target_pdb = os.path.join(work_dir or os.getcwd(), f"{pdbID}.pdb")
        return mtz_file, _cif_to_pdb(cif_file, target_pdb)

    if pdb_file:
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"pdb file not found: {pdb_file}")
        return mtz_file, pdb_file

    raise ValueError("manual mode requires --pdb-file or --cif-file")


def infer_pdb_id_from_path(path):
    """Infer a 4-char PDB id from a local file name if possible."""
    if not path:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"([A-Za-z0-9]{4})(?:_.*)?$", stem)
    return m.group(1).lower() if m else None


def enumerate_entries(root, limit=None):
    """All PDB ids under ``root`` that have final model files.

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
            if os.path.isdir(ep) and has_final_files(ep, pid):
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
    return _first_existing(
        os.path.join(entry, f"{pdb_id}_final.cif"),
        os.path.join(entry, f"{pdb_id}_final.cif.gz"),
        os.path.join(entry, f"{pdb_id}_final.pdb"),
        os.path.join(entry, f"{pdb_id}_final.pdb.gz"),
    ) or analysis_path


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
        return str(getattr(gemmi, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _ccp4_version(env):
    for key in ("CCP4_VERSION", "CCP4_VERSION_CODE", "CCP4VER"):
        if env.get(key):
            return env[key]
    ccp4_root = env.get("CCP4", "")
    return os.path.basename(ccp4_root.rstrip(os.sep)) if ccp4_root else "unknown"


def _init_worker(cfg: Dict[str, Any]) -> None:
    global _CFG
    _CFG = cfg


def process(pdbID):
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
    result = {"pdbID": pdbID, "status": "error", "n": 0,
              "runtime": 0.0, "error": "", "rows": [], "header": None,
              "bond_rows": [], "n_bonds": 0, "retryable": True,
              "reason_codes": [], "warning_codes": [],
              "alchemy_version": ALCHEMY_VERSION,
              "alchemy_commit": cfg["alchemy_commit"],
              "gemmi_version": cfg["gemmi_version"],
              "ccp4_version": cfg["ccp4_version"],
              "refinement_state": "manual" if manual_inputs else "final",
              "source_coordinate_format": "",
              "analysis_coordinate_format": "pdb",
              "coordinate_conversion_performed": False,
              "source_coordinate_path": "", "analysis_coordinate_path": "",
              "model_policy": MODEL_POLICY, "input_model_count": "",
              "model_analyzed": "", "multi_model_structure": "",
              "altloc_policy": ALTLOC_POLICY,
              "symmetry_contact_policy": SYMMETRY_POLICY}
    try:
        if manual_inputs:
            work_dir = tempfile.mkdtemp(
                prefix=f".alchemy-{pdbID}-", dir=cfg["output_dir"])
            mtz, pdb = resolve_manual_inputs(
                pdbID,
                pdb_file=manual_inputs.get("pdb_file"),
                mtz_file=manual_inputs.get("mtz_file"),
                cif_file=manual_inputs.get("cif_file"),
                work_dir=work_dir,
            )
            entry = os.path.dirname(pdb) or work_dir
            data_json = manual_inputs.get("data_json")
            reslo, reshi = read_resolution(entry, mtz, data_json_path=data_json)
        else:
            if cfg["allow_download"]:
                used_root = ensure_entry_available(
                    pdbID, cfg["mirror_root"], cfg["cache_root"])
                entry = entry_dir_for(used_root, pdbID)
            else:
                entry = entry_dir_for(cfg["root"], pdbID)
            if not os.path.isdir(entry):
                result.update(status="skip", error="entry dir missing")
                return result
            work_dir = tempfile.mkdtemp(
                prefix=f".alchemy-{pdbID}-", dir=cfg["output_dir"])
            mtz, pdb = prepare_inputs(pdbID, entry, work_dir)
            reslo, reshi = read_resolution(entry, mtz)
        source_pdb = pdb
        source_coordinate_path = _source_coordinate_path(
            cfg, pdbID, entry, source_pdb)
        source_format, analysis_format, converted = _coordinate_provenance(
            cfg, source_coordinate_path)
        model1_pdb = os.path.join(work_dir, f"{pdbID}_model1.pdb")
        if os.path.realpath(model1_pdb) == os.path.realpath(source_pdb):
            model1_pdb = os.path.join(
                work_dir, f"{pdbID}_analysis_model1.pdb")
        pdb, input_model_count = _first_model_pdb(source_pdb, model1_pdb)
        result.update(
            source_coordinate_format=source_format,
            analysis_coordinate_format=analysis_format,
            coordinate_conversion_performed=converted,
            source_coordinate_path=source_coordinate_path,
            analysis_coordinate_path=pdb,
        )
        res = run_density_analysis(
            pdbID, mtz, pdb, work_dir, reslo, reshi, env=cfg["env"])
        structure = load_structure(
            pdbID, pdb, source_model_count=input_model_count)
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
                    {"data_json": data_json if manual_inputs else os.path.join(entry, "data.json"),
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
            coverage = summary.get("geometry_coverage_image_inclusive", NAN)
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
        if (not cfg["keep"] and work_dir is not None and
                os.path.isdir(work_dir)):
            shutil.rmtree(work_dir, ignore_errors=True)
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


def _resume_replacement_succeeded(result):
    """Whether a retry produced a terminal result suitable for replacement."""
    status = str(result.get("status", "")).strip().lower()
    return (
        status == "ok" or
        (status == "partial" and not bool(result.get("retryable", True)))
    )


def _manifest_values_by_id(path, column):
    """Return one manifest column keyed by normalized PDB ID."""
    values = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return values
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = row.get("pdbID", "").strip().lower()
            if pdb_id:
                values[pdb_id] = row.get(column, "")
    return values


def _merge_csv_replacements(path, staged_path, pdb_ids):
    """Atomically replace selected IDs with rows from a completed staging file.

    The existing destination is never changed until the full replacement file
    has been written successfully. Rows for IDs absent from ``pdb_ids`` are
    copied verbatim.
    """
    replacement_ids = {pdb_id.lower() for pdb_id in pdb_ids}
    if not replacement_ids:
        return

    directory = os.path.dirname(os.path.abspath(path))
    original_mode = os.stat(path).st_mode if os.path.exists(path) else None
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",
                                    suffix=".tmp", dir=directory, text=True)
    try:
        destination_header = None
        with os.fdopen(fd, "w", newline="") as dst:
            writer = csv.writer(dst)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, newline="") as src:
                    reader = csv.reader(src)
                    destination_header = next(reader, None)
                    if destination_header is not None:
                        writer.writerow(destination_header)
                    for row in reader:
                        if (row and
                                row[0].strip().lower() in replacement_ids):
                            continue
                        writer.writerow(row)

            if os.path.exists(staged_path) and os.path.getsize(staged_path) > 0:
                with open(staged_path, newline="") as staged:
                    reader = csv.reader(staged)
                    staged_header = next(reader, None)
                    if destination_header is None and staged_header is not None:
                        destination_header = staged_header
                        writer.writerow(staged_header)
                    elif (staged_header is not None and
                          staged_header != destination_header):
                        raise ValueError(
                            f"staged CSV schema does not match {path}")
                    for row in reader:
                        if (row and
                                row[0].strip().lower() in replacement_ids):
                            writer.writerow(row)
            dst.flush()
            os.fsync(dst.fileno())

        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, path)
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
        pdbID = (
            args.id or
            infer_pdb_id_from_path(args.cif_file) or
            infer_pdb_id_from_path(args.pdb_file) or
            infer_pdb_id_from_path(args.mtz_file)
        )
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
            used_root = ensure_entry_available(
                args.id, args.pdb_redo_root, cache_root)
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
        print(f"Enumerating final PDB-REDO entries under {root} ...", flush=True)
        # Early-stop only when capping and not resuming (resume needs the full set).
        limit = args.max_pdbs if (args.max_pdbs and not args.resume) else None
        ids = enumerate_entries(root, limit=limit)
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

    cfg = {"root": root, "mirror_root": args.pdb_redo_root,
           "cache_root": cache_root, "env": env,
           "output_dir": args.output_dir, "cofactors": cofactors,
           "keep": args.keep_intermediates, "bonds": args.bonds,
           "allow_download": bool(args.id or args.id_file),
           "manual_inputs": manual_inputs,
           "alchemy_commit": _alchemy_commit(),
           "gemmi_version": _gemmi_version(),
           "ccp4_version": _ccp4_version(env)}

    resume_stage_dir = None
    replacement_ids = set()
    prior_bond_counts = (
        _manifest_values_by_id(manifest_path, "n_bonds")
        if args.resume and not args.bonds else {}
    )
    if args.resume:
        resume_stage_dir = tempfile.mkdtemp(
            prefix=".alchemy-resume-", dir=args.output_dir)
        write_manifest_path = os.path.join(
            resume_stage_dir, "manifest.csv")
        write_stats_path = os.path.join(
            resume_stage_dir, "metal_stats_all.csv")
        write_bonds_path = os.path.join(
            resume_stage_dir, "metal_bonds_all.csv")
    else:
        write_manifest_path = manifest_path
        write_stats_path = stats_path
        write_bonds_path = bonds_path

    man_fh = open(write_manifest_path, "w", newline="")
    stats_fh = open(write_stats_path, "w", newline="")
    bonds_fh = (open(write_bonds_path, "w", newline="")
                if args.bonds else None)
    man_w = csv.DictWriter(man_fh, fieldnames=MANIFEST_COLUMNS)
    stats_w = csv.writer(stats_fh)
    bonds_w = csv.writer(bonds_fh) if bonds_fh is not None else None
    man_w.writeheader()
    stats_header_written = False
    bonds_header_written = False

    counts = {"ok": 0, "partial": 0, "skip": 0, "error": 0}
    n_rows = 0
    n_bonds = 0
    processing_completed = False
    try:
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for k, r in enumerate(pool.imap_unordered(process, ids, chunksize=1), 1):
                persist_result = (
                    not args.resume or _resume_replacement_succeeded(r)
                )
                if persist_result and r["rows"]:
                    if not stats_header_written and r["header"]:
                        stats_w.writerow(["pdbID", "category"] + r["header"])
                        stats_header_written = True
                    for row in r["rows"]:
                        stats_w.writerow([row["pdbID"], row["category"]] + row["fields"])
                        n_rows += 1
                    stats_fh.flush()
                if (persist_result and bonds_w is not None and
                        bonds_fh is not None and
                        r["bond_rows"]):
                    if not bonds_header_written:
                        bonds_w.writerow(BOND_COLUMNS)
                        bonds_header_written = True
                    for b in r["bond_rows"]:
                        bonds_w.writerow([b[c] for c in BOND_COLUMNS])
                        n_bonds += 1
                    bonds_fh.flush()
                if persist_result:
                    # The manifest is the completion marker, so stage it only
                    # after this entry's result rows have been flushed.
                    manifest_row = {column: r.get(column, "")
                                    for column in MANIFEST_COLUMNS}
                    manifest_n_bonds = r["n_bonds"]
                    if args.resume and not args.bonds:
                        manifest_n_bonds = prior_bond_counts.get(
                            r["pdbID"].lower(), manifest_n_bonds)
                    manifest_row.update(
                        n_metals=r["n"], n_bonds=manifest_n_bonds,
                        runtime_s=r["runtime"],
                        reason_codes="|".join(r.get("reason_codes", [])),
                        warning_codes="|".join(r.get("warning_codes", [])),
                    )
                    man_w.writerow(manifest_row)
                    man_fh.flush()
                    if args.resume:
                        replacement_ids.add(r["pdbID"].lower())
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                if k % 200 == 0 or k == len(ids):
                    print(f"[{k}/{len(ids)}] ok={counts['ok']} "
                          f"partial={counts['partial']} skip={counts['skip']} "
                          f"error={counts['error']} "
                          f"rows={n_rows} bonds={n_bonds}", flush=True)
        processing_completed = True
    finally:
        man_fh.close()
        stats_fh.close()
        if bonds_fh:
            bonds_fh.close()
        if (resume_stage_dir is not None and not processing_completed and
                os.path.isdir(resume_stage_dir)):
            shutil.rmtree(resume_stage_dir, ignore_errors=True)

    if resume_stage_dir is not None:
        try:
            if replacement_ids:
                # Data files are committed before the manifest completion
                # marker. If an interruption occurs between replacements, the
                # old manifest causes the entry to be retried safely.
                _merge_csv_replacements(
                    stats_path, write_stats_path, replacement_ids)
                if args.bonds:
                    _merge_csv_replacements(
                        bonds_path, write_bonds_path, replacement_ids)
                _merge_csv_replacements(
                    manifest_path, write_manifest_path, replacement_ids)
        finally:
            if os.path.isdir(resume_stage_dir):
                shutil.rmtree(resume_stage_dir, ignore_errors=True)

    print(f"Done. ok={counts['ok']} partial={counts['partial']} "
          f"skip={counts['skip']} error={counts['error']}; "
          f"{n_rows} metal/cofactor rows -> {stats_path}", flush=True)
    if args.bonds:
        print(f"      {n_bonds} bond rows -> {bonds_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
