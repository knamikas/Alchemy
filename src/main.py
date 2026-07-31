#!/usr/bin/env python
"""Batch-run the Alchemy core pipeline over PDB-REDO entries.

For each PDB entry this validates/corrects its Fourier map coefficients with
CCP4 `mtzfix`, computes 2mFo-DFc and mFo-DFc maps with `fft`, and runs
`edstats`, then extracts per-atom real-space statistics for metal ions and
metal-containing cofactors. Core results are streamed to four CSVs under
--output-dir:

  metal_stats_all.csv  -- one row per selected metal site
  metal_bonds_all.csv  -- one row per inferred or declared contact
  metal_candidates_all.csv -- one row per discovered or declared candidate
  manifest.csv         -- one row per entry with status and provenance

An uncapped database run additionally streams compact confidence inputs and,
after successful completion, writes final confidence scores plus a reusable
database reference. Smaller runs use an installed frozen reference when one is
available.

Requirements
------------
* CCP4 `mtzfix`, `fft`, `mapmask`, and `edstats` on PATH -- either already
  sourced, or via --ccp4-setup pointing at a CCP4 setup script
  (e.g. <CCP4>/bin/ccp4.setup-sh).
* Run under Python 3.11+ with gemmi>=0.7.0 and numpy>=1.17. Both are
  required; gemmi does not install numpy.

Examples
--------
  python src/main.py --id 109m \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
  python src/main.py --max-pdbs 20 --workers 4 \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
"""
import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing import (
    Pool,
    SimpleQueue,
    TimeoutError as MultiprocessingTimeoutError,
    cpu_count,
)
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import urlopen

from density_analysis import (
    DENSITY_MAP_SCOPES,
    MODEL_ENVELOPE_BORDER_ANGSTROM,
    MtzfixValidationError,
    run_density_analysis,
)
from metal_identification import (
    EDSTATS_COLUMNS,
    extract_metal_statistics,
    load_cofactor_ids,
)
from metal_elements import METAL_ELEMENTS
from structure_analysis import (
    RESIDUE_REMARK_PREFIX,
    RESNAME_REMARK_PREFIX,
    blank_if_missing,
)
from bond_analysis import (
    BOND_COLUMNS,
    CANDIDATE_COLUMNS,
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
from confidence_score import (
    ANALYSIS_COLUMNS as CONFIDENCE_ANALYSIS_COLUMNS,
    CONFIDENCE_INPUT_COLUMNS,
    REFERENCE_METADATA_FILE,
    finalize_database_confidence,
    complete_confidence_site_count,
    load_reference as load_confidence_reference,
    prepare_result_confidence_inputs,
    score_against_reference,
    validate_scored_reference,
)


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = "/datasets/bioinfo/pdb-redo"
DEFAULT_CONFIDENCE_REFERENCE_DIR = os.path.join(
    REPO_DIR, "confidence_reference")
METALS_SET = set(METAL_ELEMENTS)

MODEL_POLICY = "first"
ALTLOC_POLICY = "highest-mean-occupancy-residue-conformer"
SYMMETRY_POLICY = (
    "image-inclusive-primary-with-crystallographic-and-strict-ncs-provenance"
)
MAP_COEFFICIENT_COLUMNS = ("FWT", "PHWT", "DELFWT", "PHDELWT")
ALCHEMY_VERSION = "1.0.0"
AUTO_WORKER_MEMORY_BYTES = 1280 * 1024 * 1024
# Seconds of no completed entry, after a worker died without naming the entry
# it held, before the remaining outstanding entries are failed retryably.
WORKER_STALL_GRACE_S = 600.0
# Seconds to let a worker pool shut down cleanly before its children are killed
# outright. Every result has already been collected by then and the workers are
# idle, so a healthy pool finishes this in milliseconds and never approaches the
# deadline; it exists only to bound the hang described in ``_shutdown_pool``.
WORKER_SHUTDOWN_GRACE_S = 5.0

# Marker echoed by the Windows setup wrapper so the CCP4 launcher's own banner
# is never mistaken for environment variables.
ENV_SENTINEL = "__ALCHEMY_CCP4_ENV__"

# Human-readable text for each EDSTATS-join reason code, reported alongside the
# machine-readable code in the manifest's error column.
IDENTIFICATION_REASON_MESSAGES = {
    "cofactor_coordinate_join_failed":
        "cofactor EDSTATS row did not match a coordinate residue",
    "ambiguous_coordinate_residue_join":
        "EDSTATS row matched multiple coordinate residues",
    "cofactor_without_selected_metal":
        "matched cofactor has no selected configured metal site",
}

MANIFEST_COLUMNS = [
    "pdbID", "status", "retryable", "n_metals", "n_bonds", "n_candidates",
    "runtime_s",
    "reason_codes", "warning_codes", "error", "alchemy_version", "alchemy_commit",
    "gemmi_version", "ccp4_version", "refinement_state",
    "source_coordinate_format", "analysis_coordinate_format",
    "coordinate_conversion_performed", "source_coordinate_path",
    "analysis_coordinate_path", "model_policy", "input_model_count",
    "model_analyzed", "multi_model_structure", "altloc_policy",
    "symmetry_contact_policy",
]

# metal_stats_all.csv schema. The middle block is the EDSTATS residue table,
# whose column set and order `extract_metal_statistics` validates against
# EDSTATS_COLUMNS before emitting any row, so the full header is fixed. Defining
# it once keeps the written header and the --resume compatibility check from
# disagreeing about the columns between them.
STATS_COLUMNS = (
    ["pdbID", "category"] + list(EDSTATS_COLUMNS) +
    ["aa_geometry_coverage"] + list(STATS_EXTRA_COLUMNS)
)

# config dict shared with worker processes (set once per worker by _init_worker)
_CFG: Optional[Dict[str, Any]] = None
_INFLIGHT: Optional[Any] = None


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


def _parse_windows_set_output(stdout):
    """Return the ``set`` variables printed after ENV_SENTINEL.

    The CCP4 batch launcher prints its own banner before the variables, and any
    of those lines can contain "=". Everything before the sentinel is therefore
    discarded rather than guessed at by prefix.
    """
    env = {}
    seen_sentinel = False
    for line in stdout.splitlines():
        if not seen_sentinel:
            # With `echo on` the sentinel command is echoed before its output;
            # only the output line compares equal.
            seen_sentinel = line.strip() == ENV_SENTINEL
            continue
        key, separator, value = line.partition("=")
        if separator and key:
            env[key] = value
    return env, seen_sentinel


def _resolve_env_windows(ccp4_setup):
    """Capture the environment a Windows CCP4 batch launcher establishes."""
    # Driving cmd.exe through a temporary script avoids its quoting rules,
    # which differ from the ones subprocess applies when building a command
    # line, and so would mis-handle an install path containing spaces.
    handle, script_path = tempfile.mkstemp(prefix="alchemy-ccp4-", suffix=".cmd")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\r\n") as fh:
            fh.write("@echo off\n"
                     f'call "{ccp4_setup}"\n'
                     f"echo {ENV_SENTINEL}\n"
                     "set\n")
        out = subprocess.run(["cmd", "/c", script_path],
                             capture_output=True, text=True)
    finally:
        # A fixed name in %TEMP% left one file behind per run and let
        # concurrent runs overwrite each other's script.
        try:
            os.unlink(script_path)
        except OSError:
            pass

    if out.returncode != 0:
        raise SystemExit(
            f"Failed to run CCP4 setup {ccp4_setup}:\n{out.stderr}")
    env, seen_sentinel = _parse_windows_set_output(out.stdout)
    if not seen_sentinel:
        raise SystemExit(
            f"CCP4 setup {ccp4_setup} did not report its environment; "
            f"expected `set` output after the marker.\n{out.stderr}")
    return _normalize_path_key({**os.environ.copy(), **env})


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

    if os.path.splitext(ccp4_setup)[1].lower() in (".bat", ".cmd"):
        return _resolve_env_windows(ccp4_setup)

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

    # An explicit --ccp4-setup is checked before the ambient environment, not
    # after. A user passing it is overriding whatever the shell already has --
    # typically because the wrong CCP4 version is sourced -- so honouring PATH
    # first would silently run against the very installation they were
    # replacing, and record that installation's version as the run's
    # provenance. A path that does not exist is an error for the same reason:
    # falling through would make a typo indistinguishable from success.
    if args.ccp4_setup:
        setup_path = os.path.abspath(os.path.expanduser(args.ccp4_setup))
        if not os.path.exists(setup_path):
            raise SystemExit(f"CCP4 setup file not found: {args.ccp4_setup}")
        env = resolve_env(setup_path)
        try:
            verify_ccp4(env)
        except SystemExit as exc:
            raise SystemExit(
                f"Ran {setup_path}, but CCP4 tools are still not available. "
                f"{exc}") from None
        return env, setup_path

    if ccp4_tools_available(environment):
        return environment, None

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


def _residue_index_by_author(structure, label):
    """Index residues by ``(model, chain, resnum)`` with their atom membership.

    Returns the index and the traversal order, so a conversion can be checked
    for reordering as well as for changed identifiers. ``label`` names the
    structure in any error raised here.
    """
    by_author = {}
    order = []
    for model_index, model in enumerate(structure):
        for source_chain_index, chain in enumerate(model):
            for residue in chain:
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"{label} residue {residue.name!r} has no author number")
                insertion = blank_if_missing(str(residue.seqid.icode))
                key = (model_index, str(chain.name), f"{number}{insertion}")
                order.append(key)
                by_author.setdefault(key, []).append((
                    str(residue.name),
                    tuple((str(atom.name), str(atom.element.name))
                          for atom in residue),
                ))
    return by_author, order


def _residue_conversion_records(structure, converted_structure):
    """Pair source mmCIF residue names with names written to legacy PDB."""
    source_by_author, source_order = _residue_index_by_author(
        structure, "mmCIF")
    converted_by_author, converted_order = _residue_index_by_author(
        converted_structure, "converted")

    records = []
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


# The traditional PDB chain field is one column wide.  These are the portable
# identifiers accepted by both Gemmi and the CCP4 tools used by Alchemy.
LEGACY_PDB_CHAIN_IDS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
LEGACY_PDB_MAX_RESIDUE_NUMBER = 9999


def _source_residue_records(structure):
    """Snapshot source-mmCIF residue identities before legacy conversion."""
    import gemmi

    records = []
    for model_index, model in enumerate(structure, start=1):
        for source_chain_index, chain in enumerate(model):
            polymer_indices_by_subchain = {}
            for residue_index, residue in enumerate(chain):
                if residue.entity_type == gemmi.EntityType.Polymer:
                    polymer_indices_by_subchain.setdefault(
                        str(residue.subchain), []).append(residue_index)
            for residue_index, residue in enumerate(chain):
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"mmCIF residue {residue.name!r} has no author number")
                polymer_indices = polymer_indices_by_subchain.get(
                    str(residue.subchain), [])
                if not polymer_indices:
                    polymer_position = "-"
                else:
                    is_first = residue_index == polymer_indices[0]
                    is_last = residue_index == polymer_indices[-1]
                    polymer_position = (
                        "NC" if is_first and is_last else
                        ("N" if is_first else ("C" if is_last else "M")))
                records.append((
                    model_index,
                    str(chain.name),
                    int(number),
                    blank_if_missing(str(residue.seqid.icode)),
                    str(residue.name),
                    tuple((str(atom.name), str(atom.element.name))
                          for atom in residue),
                    source_chain_index,
                    residue_index,
                    polymer_position,
                ))
    return records


def _legacy_identifiers_need_packing(structure):
    """Whether Gemmi could not shorten every chain to a portable PDB id."""
    return any(
        bool(str(chain.name)) and str(chain.name) not in LEGACY_PDB_CHAIN_IDS
        for model in structure for chain in model
    )


def _pack_legacy_pdb_residue_ids(structure):
    """Assign a unique, one-character PDB identity to every residue.

    Multiple source chains may occupy one synthetic chain because EDSTATS only
    needs an unambiguous residue key, not polymer connectivity.  Whole source
    chains stay together, TER records preserve their boundaries, and sequence
    numbers never exceed the portable four-column decimal PDB range.
    """
    import gemmi

    for model in structure:
        chain_slot = 0
        next_residue_number = 1
        for chain in model:
            residue_count = len(chain)
            if residue_count > LEGACY_PDB_MAX_RESIDUE_NUMBER:
                raise ValueError(
                    "one mmCIF chain contains more residues than a portable "
                    "PDB chain can represent")
            if (next_residue_number + residue_count - 1 >
                    LEGACY_PDB_MAX_RESIDUE_NUMBER):
                chain_slot += 1
                next_residue_number = 1
            if chain_slot >= len(LEGACY_PDB_CHAIN_IDS):
                raise ValueError(
                    "mmCIF model contains more residues than the portable "
                    "PDB surrogate namespace can represent")
            chain.name = LEGACY_PDB_CHAIN_IDS[chain_slot]
            for residue in chain:
                residue.seqid = gemmi.SeqId(next_residue_number, " ")
                next_residue_number += 1


def _residue_identity_records(source_records, converted_structure):
    """Map packed PDB residue identities back to source-mmCIF identities."""
    converted_records = []
    for model_index, model in enumerate(converted_structure, start=1):
        for chain in model:
            for residue in chain:
                number = residue.seqid.num
                if number is None:
                    raise ValueError(
                        f"converted residue {residue.name!r} has no number")
                converted_records.append((
                    model_index,
                    str(chain.name),
                    f"{number}{blank_if_missing(str(residue.seqid.icode))}",
                    str(residue.name),
                    tuple((str(atom.name), str(atom.element.name))
                          for atom in residue),
                ))
    if len(source_records) != len(converted_records):
        raise ValueError("PDB conversion changed residue count")

    records = []
    for source, converted in zip(source_records, converted_records):
        (source_model, source_chain, source_number, source_insertion,
         source_name, source_atoms, source_chain_index,
         source_residue_index, source_polymer_position) = source
        (converted_model, converted_chain, converted_resnum,
         converted_name, converted_atoms) = converted
        if source_model != converted_model:
            raise ValueError("PDB conversion changed residue model ordering")
        if source_atoms != converted_atoms:
            raise ValueError("PDB conversion changed residue atom membership")
        source_resnum = f"{source_number}{source_insertion}"
        if ((source_chain, source_resnum, source_name) ==
                (converted_chain, converted_resnum, converted_name)):
            continue
        records.append((
            converted_model,
            converted_chain,
            converted_resnum,
            converted_name,
            source_chain,
            source_number,
            source_insertion,
            source_name,
            source_chain_index,
            source_residue_index,
            source_polymer_position,
        ))
    return records


def _write_cif_conversion_provenance(
        dst: str,
        missing_occupancies: List[bool],
        residue_records: List[Tuple[int, str, str, str, str]],
        identity_records=None,
        ) -> None:
    """Blank unknown occupancies and embed reversible residue mappings."""
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
    remarks.extend(
        (
            f"{RESIDUE_REMARK_PREFIX} {model_index} "
            f"{converted_chain or '_'} {converted_resnum} {converted_name} "
            f"{source_chain or '_'} {source_number} "
            f"{source_insertion or '_'} {source_name} "
            f"{source_chain_index} {source_residue_index} "
            f"{source_polymer_position}\n"
        )
        for (model_index, converted_chain, converted_resnum, converted_name,
             source_chain, source_number, source_insertion,
             source_name, source_chain_index, source_residue_index,
             source_polymer_position) in (identity_records or ())
    )
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
    source_residues = _source_residue_records(structure)
    # EDSTATS consumes PDB coordinates, whose chain field is one character.
    # Shorten deterministically before writing, then analyze this exact PDB so
    # EDSTATS and Alchemy never join identifiers from different representations.
    structure.shorten_chain_names()
    identifiers_packed = _legacy_identifiers_need_packing(structure)
    if identifiers_packed:
        _pack_legacy_pdb_residue_ids(structure)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    structure.write_pdb(dst)
    converted_structure = gemmi.read_structure(dst)
    residue_records = _residue_conversion_records(
        structure, converted_structure)
    identity_records = (
        _residue_identity_records(source_residues, converted_structure)
        if identifiers_packed else [])
    _write_cif_conversion_provenance(
        dst, missing_occupancies, residue_records, identity_records)
    return dst


def _first_model_pdb(pdb_path, dst):
    """Return a wrapper-free PDB containing the first coordinate model.

    The extraction is textual so atom records, occupancies, identifiers, and
    ordering remain exactly as deposited. Gemmi is used only to determine and
    verify the model count. Explicit MODEL/ENDMDL records are removed because
    EDSTATS emits a synthetic separator residue for even a one-model wrapper.
    """
    import gemmi

    structure = gemmi.read_structure(pdb_path)
    model_count = len(structure)
    if model_count == 0:
        raise ValueError("coordinate file contains no models")

    with open(pdb_path, encoding="utf-8", errors="replace", newline="") as fh:
        lines = fh.readlines()
    model_starts = [
        index for index, line in enumerate(lines)
        if line[:6].strip().upper() == "MODEL"
    ]
    if not model_starts:
        if model_count == 1:
            return pdb_path, model_count
        raise ValueError(
            "Gemmi found multiple models but the PDB contains no MODEL records")
    if len(model_starts) != model_count:
        raise ValueError(
            "Gemmi model count does not match the PDB MODEL records")

    first_start = model_starts[0]
    next_start = model_starts[1] if len(model_starts) > 1 else len(lines)
    first_end = next(
        (index for index in range(first_start + 1, next_start)
         if lines[index][:6].strip().upper() == "ENDMDL"),
        None,
    )
    if first_end is None:
        raise ValueError("the first PDB MODEL record has no matching ENDMDL")
    else:
        first_block = lines[first_start + 1:first_end]

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

    analysis_structure = gemmi.read_structure(dst)
    if len(analysis_structure) != 1:
        raise ValueError("failed to create a first-model-only analysis PDB")
    with open(dst, encoding="utf-8", errors="replace") as fh:
        if any(line[:6].strip().upper() in ("MODEL", "ENDMDL") for line in fh):
            raise ValueError("first-model analysis PDB still contains a model wrapper")
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
    """Return the overall diffraction-data high-resolution limit.

    Prefer a supplied data.json (or PDB-REDO data.json) when available;
    fall back to the MTZ via gemmi. Only the high-resolution limit is reported
    because that is what the DPI metadata records; EDSTATS is given the map
    columns' own range by ``read_map_column_resolution`` instead.
    """
    dj = data_json_path or os.path.join(entry_dir, "data.json")
    if os.path.exists(dj):
        try:
            with open(dj) as handle:
                props = json.load(handle).get("properties", {})
            lo, hi = props.get("DATARESL"), props.get("DATARESH")
            # Both limits are still required before trusting data.json, so a
            # half-populated record falls back to the MTZ as it always has.
            if lo and hi:
                return float(hi)
        except (ValueError, KeyError, OSError):
            pass
    import gemmi
    return gemmi.read_mtz_file(mtz_path).resolution_high()


def read_pdb_redo_is_twin(data_json_path):
    """Return only an explicit boolean PDB-REDO ``properties.ISTWIN`` value.

    Missing, malformed, and string-valued metadata are deliberately false: the
    twin coefficient fallback is a guarded processing path, not something to
    infer from a filename or from an MTZFIX failure alone.
    """
    if not data_json_path:
        return False
    try:
        with open(data_json_path) as handle:
            value = json.load(handle).get("properties", {}).get("ISTWIN")
    except (AttributeError, OSError, ValueError):
        return False
    return value is True


def read_map_column_resolution(mtz_path):
    """Return the common finite resolution range of both EDSTATS maps.

    EDSTATS receives maps calculated from FWT/PHWT and DELFWT/PHDELWT, so its
    limits must describe reflections for which all four values are present,
    rather than the overall range of unrelated columns in the MTZ.
    """
    import gemmi
    import numpy as np

    mtz = gemmi.read_mtz_file(mtz_path)
    columns = []
    missing = []
    for label in MAP_COEFFICIENT_COLUMNS:
        column = mtz.column_with_label(label)
        if column is None:
            missing.append(label)
        else:
            columns.append(column)
    if missing:
        raise ValueError(
            "MTZ is missing required map coefficient column(s): "
            + ", ".join(missing))

    d_values = mtz.make_d_array()
    row_count = len(d_values)
    if any(len(column) != row_count for column in columns):
        raise ValueError(
            "MTZ map coefficient columns do not match the reflection count")

    # A reflection counts only where its d-spacing and all four map
    # coefficients are finite, so the mask is built across whole rows.
    usable = np.isfinite(d_values) & (d_values > 0.0)
    for column in columns:
        usable &= np.isfinite(column.array)
    usable_d = d_values[usable]
    if usable_d.size == 0:
        raise ValueError(
            "MTZ map coefficient columns have no common finite reflections")
    return float(usable_d.max()), float(usable_d.min())


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


def _init_worker(cfg: Dict[str, Any], inflight=None) -> None:
    global _CFG, _INFLIGHT
    _CFG = cfg
    _INFLIGHT = inflight


def _announce_inflight(state: str, pdbID: str) -> None:
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
        _INFLIGHT.put((state, os.getpid(), pdbID))
    except Exception:  # noqa: BLE001 - bookkeeping must never fail an entry
        pass


def _initial_result(pdbID, cfg, manual_inputs):
    """Return the per-entry result skeleton, pre-filled with run provenance.

    Every manifest column is present from the outset so a failure at any stage
    still yields a complete row.

    ``n_bonds`` and ``n_candidates`` start blank rather than zero. Zero is a
    measured result meaning the bond stage ran and found nothing, so an entry
    that fails before that stage must not claim it: a later ``--resume`` reads
    a non-blank count as proof the stage completed and would skip the entry
    permanently.
    """
    return {"pdbID": pdbID, "status": "error", "n": 0,
            "runtime": 0.0, "error": "", "rows": [],
            "bond_rows": [], "candidate_rows": [],
            "n_bonds": "", "n_candidates": "", "no_metals": False,
            "timings": {},
            "density_map_scope_used": "",
            "density_full_map_bytes": 0,
            "density_edstats_map_bytes": 0,
            "retryable": True,
            "reason_codes": [], "warning_codes": [],
            "confidence_inputs_missing_reason": "",
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


def _drain_inflight(inflight, assignments):
    """Apply pending worker notifications to the pid -> entry assignment map."""
    while True:
        try:
            if inflight.empty():
                return
            state, pid, pdbID = inflight.get()
        except (OSError, EOFError):  # pragma: no cover - pipe torn down
            return
        if state == "start":
            assignments[pid] = pdbID
        else:
            assignments.pop(pid, None)


def _dead_worker_pids(pool, known_pids):
    """Return worker pids that have disappeared since the last check.

    ``Pool`` silently replaces a worker that died, so a pid leaving the pool's
    roster is the only signal that its task will never produce a result.
    """
    current = {
        process.pid for process in getattr(pool, "_pool", ()) or ()
        if process.pid is not None
    }
    if not current:
        return set()
    dead = known_pids - current
    known_pids.clear()
    known_pids.update(current)
    return dead


def _install_termination_handler():
    """Route SIGTERM through the same unwind path as Ctrl-C.

    SIGTERM's default disposition kills the process immediately: no ``finally``
    runs, so the worker pool is never shut down and its children -- which are
    daemonic but not yet signalled -- are reparented to init and keep working,
    driving CCP4 subprocesses and holding their scratch directories open.
    Raising ``KeyboardInterrupt`` instead makes a scheduler stop or a plain
    ``kill`` unwind exactly like an interactive interrupt, so the pool is
    terminated and the run log is still written.

    Returns the previous handler, or ``None`` where SIGTERM cannot be trapped
    (a non-main thread, or a platform without it).
    """
    def _raise_interrupt(signum, frame):  # noqa: ARG001 - signal API
        raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGTERM, _raise_interrupt)
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        return None


def _sweep_leaked_work_dirs(output_dir):
    """Remove per-entry scratch directories a previous run left behind.

    Each entry works inside ``<output-dir>/.alchemy-<id>-XXXX`` and deletes it
    once its rows are extracted, but a run that is interrupted -- Ctrl-C,
    SIGTERM, an out-of-memory kill -- never reaches that cleanup, and the
    directory survives holding that entry's maps. Nothing removed them, so they
    accumulated over every interrupted attempt.

    Sweeping at startup is safe because the directories belong to a run that is
    no longer executing: a second Alchemy process sharing one ``--output-dir``
    would already be overwriting the first one's manifest and CSVs.

    Returns the number of directories removed.
    """
    removed = 0
    try:
        names = os.listdir(output_dir)
    except OSError:
        return 0
    for name in names:
        if not name.startswith((".alchemy-", ".alchemy-resume-")):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path) or os.path.islink(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += not os.path.exists(path)
    return removed


def _shutdown_pool(pool):
    """Close a worker pool without letting a dead worker hang the driver.

    A pool worker waiting for its next task blocks inside the task queue's
    ``get()`` while holding that queue's lock. A process killed there -- by the
    out-of-memory killer, or by a crash in a compiled extension -- never
    releases it, and ``Pool.terminate`` in turn blocks acquiring the same lock.
    Shutdown then hangs *after* every result has already been collected and
    written, so the batch is complete but the run log, confidence finalization
    and exit code never happen.

    Run the clean shutdown on a side thread and give it a deadline. If it does
    not return, cancel the finalizer that would repeat the same wait at
    interpreter exit and kill the remaining children directly. The thread is a
    daemon, so an abandoned one cannot keep the process alive.

    Returns ``True`` when shutdown had to be forced.
    """
    children = [process for process in getattr(pool, "_pool", ()) or ()
                if getattr(process, "pid", None)]
    closer = threading.Thread(target=pool.terminate, daemon=True)
    closer.start()
    closer.join(WORKER_SHUTDOWN_GRACE_S)
    if not closer.is_alive():
        return False

    finalizer = getattr(pool, "_terminate", None)
    if finalizer is not None:
        try:
            finalizer.cancel()
        except Exception:  # noqa: BLE001 - best effort, shutdown must proceed
            pass
    for process in children:
        try:
            # Process.kill is SIGKILL on POSIX and TerminateProcess on Windows,
            # where signal.SIGKILL does not exist.
            process.kill()
        except (OSError, ValueError, AttributeError):  # already reaped
            pass
    # The lock belongs to a process that is already gone, so killing the
    # remaining children cannot release it and the thread will not return.
    # Abandon it: a daemon thread does not keep the interpreter alive.
    return True


def _worker_death_result(pdbID, cfg, pid):
    """Synthesize the retryable result a killed worker could not return."""
    result = _initial_result(pdbID, cfg, cfg.get("manual_inputs"))
    result.update(
        status="error",
        retryable=True,
        reason_codes=["worker_process_died"],
        error=(f"worker process {pid} terminated without returning a result "
               f"(out-of-memory kill or crash); {pdbID} was not analyzed"),
    )
    return result


def _resolve_entry_dir(pdbID, cfg):
    """Locate an entry's PDB-REDO directory, downloading it when permitted."""
    if cfg["allow_download"]:
        used_root = ensure_entry_available(
            pdbID, cfg["mirror_root"], cfg["cache_root"])
        return entry_dir_for(used_root, pdbID)
    return entry_dir_for(cfg["root"], pdbID)


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
        f"{name} row does not match its column schema: " + "; ".join(details))


def _append_site_fields(rows, site_summaries, structure):
    """Extend each EDSTATS row with its per-site contact and provenance values."""
    for index, row in enumerate(rows):
        summary = dict(site_summaries.get(row.get("site_key"), {}))
        for name in (
            "density_observation_id", "density_scope",
            "density_shared_site_count", "density_is_shared",
        ):
            summary[name] = row.get(name, "")
        summary["coordinate_mapping_status"] = row.get(
            "coordinate_mapping_status", "")
        summary["selected_metal_site_status"] = row.get(
            "selected_metal_site_status", "")
        coverage = summary.get("geometry_coverage_image_inclusive", NAN)
        if isinstance(coverage, float) and not math.isfinite(coverage):
            coverage = summary.get("geometry_coverage_explicit", NAN)
        extra = stats_extra_values(structure, row.get("site"), summary)
        if index == 0:
            _check_row_schema(extra, STATS_EXTRA_COLUMNS,
                              "metal_stats_all.csv")
        row["fields"] = (row["fields"] + [coverage] +
                         [extra[column] for column in STATS_EXTRA_COLUMNS])


def _finalize_result(result, identification_codes, bond_meta, structure,
                     rows, bond_rows, candidate_rows):
    """Merge the stage outcomes into the final status, codes, and counts."""
    result["reason_codes"] = list(dict.fromkeys(
        result["reason_codes"] + identification_codes +
        list(bond_meta["partial_reason_codes"])))
    messages = [IDENTIFICATION_REASON_MESSAGES[code]
                for code in identification_codes]
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
    # Count coordinate-model metal sites, not emitted statistics rows. A failed
    # EDSTATS join can leave a diagnostic row without a site even though bond
    # analysis still found and evaluated the deposited metal.
    result.update(status=status,
                  n=len(structure.metal_atoms(METALS_SET, canonical=True)),
                  rows=rows, bond_rows=bond_rows,
                  candidate_rows=candidate_rows,
                  n_bonds=len(bond_rows),
                  n_candidates=len(candidate_rows))


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
    result = _initial_result(pdbID, cfg, manual_inputs)
    _announce_inflight("start", pdbID)
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
            data_reshi = read_resolution(
                entry, mtz, data_json_path=data_json)
        else:
            # Resolved before any scratch space is created, so a missing entry
            # never leaves a temporary directory behind.
            entry = _resolve_entry_dir(pdbID, cfg)
            if not os.path.isdir(entry):
                result.update(status="skip", error="entry dir missing")
                return result
            work_dir = tempfile.mkdtemp(
                prefix=f".alchemy-{pdbID}-", dir=cfg["output_dir"])
            mtz, pdb = prepare_inputs(pdbID, entry, work_dir)
            data_reshi = read_resolution(entry, mtz)
        density_data_json = (
            data_json if manual_inputs else os.path.join(entry, "data.json"))
        pdb_redo_is_twin = read_pdb_redo_is_twin(density_data_json)
        map_reslo, map_reshi = read_map_column_resolution(mtz)
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
        structure = load_structure(
            pdbID, pdb, source_model_count=input_model_count)
        result["timings"]["input_structure_s"] = round(
            time.monotonic() - t0, 3)
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
                pdbID, mtz, pdb, work_dir, map_reslo, map_reshi,
                env=cfg["env"], map_scope=cfg["density_map_scope"],
                keep_full_maps=cfg["keep"],
                pdb_redo_is_twin=pdb_redo_is_twin)
        except MtzfixValidationError as exc:
            # The input is readable, but MTZFIX could not make its Fourier
            # coefficients internally consistent. Do not use those maps or
            # retry forever. Geometry remains independently assessable.
            rows, header = [], []
            result.update(
                retryable=False,
                reason_codes=["mtzfix_validation_failure"],
                error=f"density unavailable: {exc}"[:300],
                confidence_inputs_missing_reason=(
                    "mtzfix_validation_failure"),
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
                result["warning_codes"] = list(dict.fromkeys(
                    result["warning_codes"] +
                    ["twin_refmac_coefficients_normalized"]))
            statistics_started = time.monotonic()
            rows, header = extract_metal_statistics(
                pdbID, res["stats_out"], METALS_SET, cfg["cofactors"],
                structure=structure)
            result["timings"]["statistics_extraction_s"] = round(
                time.monotonic() - statistics_started, 3)
            # Reaching this point means the entry's core inputs and density
            # stage succeeded. Later deterministic limitations remain terminal.
            result["retryable"] = False
        finally:
            result["timings"]["density_total_s"] = round(
                time.monotonic() - density_started, 3)

        identification_reason_codes = _identification_reason_codes(rows)

        bond_rows = []
        candidate_rows = []
        site_summaries = {}
        bond_meta = {"partial_reason_codes": [],
                     "warning_codes": list(structure.warning_codes),
                     "messages": [], "retryable": False}

        if cfg["bonds"]:
            # A bond-stage failure must not lose the edstats rows already computed.
            bond_started = time.monotonic()
            try:
                (bond_rows, candidate_rows, site_summaries,
                 bond_meta) = run_bond_analysis(
                    pdbID, pdb, rows, header,
                    {"data_json": data_json if manual_inputs else os.path.join(entry, "data.json"),
                     "pdb_path": pdb, "mtz_path": mtz,
                     "resolution": data_reshi}, structure=structure,
                    connection_path=source_coordinate_path)
            except Exception as e:  # noqa: BLE001
                result["error"] = f"bond: {type(e).__name__}: {e}"[:300]
                result["reason_codes"] = list(dict.fromkeys(
                    result["reason_codes"] + ["bond_stage_failure"]))
                result["retryable"] = True
            finally:
                result["timings"]["bond_analysis_s"] = round(
                    time.monotonic() - bond_started, 3)
        _append_site_fields(rows, site_summaries, structure)
        _finalize_result(result, identification_reason_codes, bond_meta,
                         structure, rows, bond_rows, candidate_rows)
    except FileNotFoundError as e:
        result.update(status="skip", retryable=True,
                      reason_codes=["missing_input"],
                      error=f"missing input: {e}"[:300])
    except Exception as e:  # noqa: BLE001 - one bad entry must not kill the batch
        result.update(status="error", retryable=True,
                      reason_codes=["unexpected_processing_error"],
                      error=f"{type(e).__name__}: {e}"[:300])
    finally:
        if (not cfg["keep"] and work_dir is not None and
                os.path.isdir(work_dir)):
            cleanup_started = time.monotonic()
            shutil.rmtree(work_dir, ignore_errors=True)
            result["timings"]["cleanup_s"] = round(
                time.monotonic() - cleanup_started, 3)
        result["runtime"] = round(time.monotonic() - t0, 2)
        _announce_inflight("end", pdbID)
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_done(manifest_path, bonds_required=False, bond_output_present=True,
              candidate_output_present=True, retry_partial_ids=()):
    """PDB IDs whose requested result is terminal in an existing manifest.

    Blank ``n_bonds`` and ``n_candidates`` values mean bond analysis was
    disabled, while ``0`` means it ran successfully and found no rows of that
    type. When bonds are requested, a terminal density result with either blank
    count still needs processing. Absent bond or candidate CSVs also make
    bond-stage results incomplete. IDs in ``retry_partial_ids`` are removed
    from the done set only when their row is a non-retryable ``partial``;
    successful ``ok`` rows remain protected from accidental reprocessing.
    """
    retry_partial_ids = {
        str(pdb_id).strip().lower() for pdb_id in retry_partial_ids
        if str(pdb_id).strip()
    }
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
                    pdbID = row.get("pdbID", "").strip().lower()
                    bonds_complete = (
                        not bonds_required or
                        (bond_output_present and candidate_output_present and
                         row.get("n_bonds", "").strip() != "" and
                         row.get("n_candidates", "").strip() != "")
                    )
                    protected_terminal = (
                        status == "ok" or
                        (terminal_partial and pdbID not in retry_partial_ids)
                    )
                    if protected_terminal and bonds_complete and pdbID:
                        done.add(pdbID)
    return done


def resolve_confidence_reference_dir(output_dir, configured_dir=None):
    """Find a frozen confidence reference, honoring an explicit override."""
    if configured_dir is not None:
        candidates = (configured_dir,)
    else:
        candidates = (
            os.path.join(output_dir, "confidence_reference"),
            DEFAULT_CONFIDENCE_REFERENCE_DIR,
        )
    for candidate in candidates:
        metadata_path = os.path.join(candidate, REFERENCE_METADATA_FILE)
        if os.path.isfile(metadata_path):
            return candidate, candidates
    return None, candidates


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
        # os.fdopen takes ownership of the descriptor and closes it on exit, so
        # it is closed here only if the wrapping itself failed. Closing it again
        # afterwards could target a descriptor the runtime has since reused.
        try:
            staged = os.fdopen(fd, "w", newline="")
        except BaseException:
            os.close(fd)
            raise
        destination_header = None
        with staged as dst:
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
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _csv_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, newline="") as handle:
        return next(csv.reader(handle), None)


def _batch_exit_code(counts, retryable_partial_count):
    """Return failure when one or more entries remain operationally incomplete."""
    incomplete = (
        counts.get("error", 0) +
        counts.get("skip", 0) +
        retryable_partial_count
    )
    return 1 if incomplete else 0


def validate_resume_schemas(manifest_path, stats_path, bonds_path,
                            candidates_path, bonds_enabled=True,
                            confidence_path=None, confidence_columns=None):
    """Refuse to append migration rows beneath an incompatible old header.

    Whole headers are compared, including the EDSTATS block of
    metal_stats_all.csv. Appending rows beneath a header from a different
    EDSTATS build would misalign every density column without any other
    symptom.
    """
    checks = [(manifest_path, MANIFEST_COLUMNS), (stats_path, STATS_COLUMNS)]
    if bonds_enabled:
        checks.extend(((bonds_path, BOND_COLUMNS),
                       (candidates_path, CANDIDATE_COLUMNS)))
    if confidence_path is not None:
        if confidence_columns is None:
            raise ValueError(
                "confidence columns are required with a confidence output")
        checks.append((confidence_path, list(confidence_columns)))
    for path, expected in checks:
        header = _csv_header(path)
        if header is not None and header != expected:
            raise ValueError(
                f"Existing {os.path.basename(path)} uses an incompatible "
                "schema; choose a new --output-dir for this Gemmi migration "
                "run.")


def remove_stale_disabled_bond_outputs(paths, resume, bonds_enabled):
    """Remove previous bond-stage CSVs before a fresh disabled run.

    A non-resume run replaces the manifest and statistics outputs, so retaining
    older bond-stage files would falsely associate them with the new run.
    Resume mode is different: completed entries and their existing rows are
    retained.
    """
    if resume or bonds_enabled:
        return []
    removed = []
    for path in paths:
        if os.path.lexists(path):
            os.unlink(path)
            removed.append(path)
    return removed


def available_cpu_count():
    """Return the number of CPUs this process is actually permitted to use.

    ``multiprocessing.cpu_count()`` reports every logical CPU on the machine
    and ignores CPU affinity, so inside a container or under a scheduler
    allocation it can report far more than the job was granted -- defaulting a
    batch run to dozens of workers on a handful of cores. The affinity-aware
    interfaces are preferred where the platform provides them.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    if hasattr(os, "sched_getaffinity"):  # Linux
        try:
            count = len(os.sched_getaffinity(0))
        except OSError:
            count = 0
        if count:
            return count
    try:
        return cpu_count()
    except NotImplementedError:
        return 1


def available_memory_bytes():
    """Return currently available physical memory, or ``None`` if unknown.

    Available memory is used instead of total RAM so an automatic batch run
    does not compete with memory already committed to other applications. Keep
    this dependency-free because worker selection happens before the analysis
    environment is initialized.
    """
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                    ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, ValueError):
            pass

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            if page_size > 0 and available_pages > 0:
                return int(page_size * available_pages)
        except (OSError, TypeError, ValueError):
            pass
    return None


def automatic_worker_limits():
    """Return the CPU and optional memory limits for automatic parallelism."""
    cpu_limit = max(1, available_cpu_count() - 2)
    available_memory = available_memory_bytes()
    memory_limit = None
    if available_memory is not None:
        memory_limit = max(1, available_memory // AUTO_WORKER_MEMORY_BYTES)
    return cpu_limit, memory_limit


def parse_pdb_id(value):
    if not re.fullmatch(r"[A-Za-z0-9]{4}", value):
        raise argparse.ArgumentTypeError(
            "PDB ID must contain exactly four alphanumeric characters")
    return value.lower()


def positive_int(value):
    """Argparse type for integer options that must be at least one."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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
    ap.add_argument("--max-pdbs", type=positive_int, default=None,
                    help="process only the first N entries (minimum: 1)")
    ap.add_argument(
        "--workers", type=positive_int, default=None,
        help=("number of worker processes (minimum: 1); by default Alchemy "
              "uses the lower CPU or available-memory limit"),
    )
    ap.add_argument("--output-dir", default=os.path.join(REPO_DIR, "output"))
    ap.add_argument(
        "--density-map-scope", choices=DENSITY_MAP_SCOPES,
        default="model-envelope",
        help=("map extent supplied to EDSTATS; model-envelope retains every "
              f"coordinate plus a {MODEL_ENVELOPE_BORDER_ANGSTROM} Angstrom "
              "border and falls back to full when cropping would be unsafe or "
              "larger"),
    )
    ap.add_argument(
        "--confidence-reference-dir",
        default=None,
        help=("explicit frozen full-database confidence reference for single, "
              "ID-file, manual, and capped runs; otherwise Alchemy searches "
              "the output directory and repository default"),
    )
    ap.add_argument("--ccp4-setup", default=None,
                    help="optional CCP4 setup script override (e.g. .../bin/ccp4.setup-sh)")
    ap.add_argument("--configure-ccp4", default=None,
                    help="save a CCP4 setup script path for future runs")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="keep per-entry maps/logs (default: delete after extract)")
    ap.add_argument("--resume", action="store_true",
                    help="skip terminal ok/partial results; retry retryable incomplete ids")
    ap.add_argument(
        "--retry-partials", action="store_true",
        help=("with --resume, reprocess non-retryable partial entries from "
              "the manifest while still skipping successful entries; --id "
              "or --id-file may restrict the retry set"),
    )
    # ArgumentDefaultsHelpFormatter appends the default of ``bonds``, not of
    # the flag, so an unqualified help string renders "(default: True)" -- the
    # negation of what --no-bonds does. Naming %(default)s explicitly suppresses
    # that append and lets the value be labelled with the setting it belongs to.
    ap.add_argument("--no-bonds", dest="bonds", action="store_false",
                    help="skip the metal-ligand bond-distance stage (edstats "
                         "stats only); bond analysis is enabled by default "
                         "(bonds=%(default)s)")
    ap.set_defaults(bonds=True)
    
    args = ap.parse_args(argv)
    if args.id and args.id_file:
        raise SystemExit("use either --id or --id-file, not both")
    if args.retry_partials and not args.resume:
        ap.error("--retry-partials requires --resume")
    if args.retry_partials and (args.pdb_file or args.mtz_file or args.cif_file):
        ap.error("--retry-partials cannot be used with manual structure inputs")
    return args


class _DriverError(Exception):
    """A user-facing driver failure: the message is printed and the run exits 1."""


def _select_entry_ids(args, cache_root):
    """Resolve the run's work list, returning ``(ids, root, manual_inputs)``.

    Raises ``_DriverError`` when the request cannot be satisfied at all.
    """
    root = args.pdb_redo_root
    if args.pdb_file or args.mtz_file or args.cif_file:
        pdbID = (
            args.id or
            infer_pdb_id_from_path(args.cif_file) or
            infer_pdb_id_from_path(args.pdb_file) or
            infer_pdb_id_from_path(args.mtz_file)
        )
        if not pdbID:
            raise _DriverError(
                "Manual input mode requires --id or a file name that contains "
                "a 4-character PDB id.")
        return [pdbID], root, {
            "pdb_file": args.pdb_file,
            "mtz_file": args.mtz_file,
            "cif_file": args.cif_file,
            "data_json": args.data_json,
        }

    if args.id:
        # Ensure requested single entry is available locally (mirror or cache).
        try:
            used_root = ensure_entry_available(
                args.id, args.pdb_redo_root, cache_root)
        except FileNotFoundError:
            raise _DriverError(
                f"Entry {args.id} not found locally and download failed.")
        if used_root != args.pdb_redo_root:
            print(f"Auto-downloaded {args.id} into cache at {cache_root}", flush=True)
        return [args.id], used_root, None

    if args.id_file:
        try:
            ids = load_ids_from_file(args.id_file)
        except (FileNotFoundError, ValueError) as exc:
            raise _DriverError(str(exc))
        print(f"Loaded {len(ids)} IDs from {args.id_file}", flush=True)
        return ids, root, None

    print(f"Enumerating final PDB-REDO entries under {root} ...", flush=True)
    # Early-stop only when capping and not resuming (resume needs the full set).
    limit = args.max_pdbs if (args.max_pdbs and not args.resume) else None
    return enumerate_entries(root, limit=limit), root, None


def _manifest_row(result, resume, bonds_enabled, prior_bond_counts,
                  prior_candidate_counts):
    """Project one worker result onto the manifest schema."""
    row = {column: result.get(column, "") for column in MANIFEST_COLUMNS}
    n_bonds = result["n_bonds"]
    n_candidates = result["n_candidates"]
    if not bonds_enabled:
        n_bonds = (
            prior_bond_counts.get(result["pdbID"].lower(), "")
            if resume else ""
        )
        n_candidates = (
            prior_candidate_counts.get(result["pdbID"].lower(), "")
            if resume else ""
        )
    row.update(
        n_metals=result["n"], n_bonds=n_bonds,
        n_candidates=n_candidates,
        runtime_s=result["runtime"],
        reason_codes="|".join(result.get("reason_codes", [])),
        warning_codes="|".join(result.get("warning_codes", [])),
    )
    return row


class _ResumeStaging:
    """Holds retried entries' rows until the whole retry batch has succeeded.

    Rows go to a temporary directory and are merged into the real outputs only
    once the batch completes, so a failed or interrupted retry leaves the
    previous rows untouched.
    """

    def __init__(self, output_dir, targets):
        self.targets = targets
        self.dir = tempfile.mkdtemp(prefix=".alchemy-resume-", dir=output_dir)
        self.staged = tuple(os.path.join(self.dir, os.path.basename(path))
                            for path in targets)
        self.replacement_ids = set()

    def commit(self, bonds_enabled, confidence_enabled=False):
        """Replace the retried entries' rows in the real output files."""
        if not self.replacement_ids:
            return
        manifest_path, stats_path, bonds_path, candidates_path = self.targets[:4]
        (staged_manifest, staged_stats, staged_bonds,
         staged_candidates) = self.staged[:4]
        # Data files are committed before the manifest completion marker. If an
        # interruption occurs between replacements, the old manifest causes the
        # entry to be retried safely.
        _merge_csv_replacements(stats_path, staged_stats, self.replacement_ids)
        if bonds_enabled:
            _merge_csv_replacements(bonds_path, staged_bonds,
                                    self.replacement_ids)
            _merge_csv_replacements(candidates_path, staged_candidates,
                                    self.replacement_ids)
        if confidence_enabled:
            for target, staged in zip(self.targets[4:], self.staged[4:]):
                _merge_csv_replacements(
                    target, staged, self.replacement_ids)
        _merge_csv_replacements(manifest_path, staged_manifest,
                                self.replacement_ids)

    def discard(self):
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)


class _OutputWriters:
    """The streamed CSV outputs, with running row counts.

    Each stream is flushed after every entry so an interrupted batch run
    retains the results it already completed. Headers are written on creation.
    """

    def __init__(self, manifest_fh, stats_fh, bonds_fh, candidates_fh,
                 confidence_fh=None, confidence_columns=None,
                 confidence_inputs_fh=None):
        self._manifest_fh = manifest_fh
        self._stats_fh = stats_fh
        self._bonds_fh = bonds_fh
        self._candidates_fh = candidates_fh
        self._confidence_fh = confidence_fh
        self._confidence_inputs_fh = confidence_inputs_fh
        self._manifest = csv.DictWriter(manifest_fh,
                                        fieldnames=MANIFEST_COLUMNS)
        self._stats = csv.writer(stats_fh)
        self._bonds = csv.writer(bonds_fh) if bonds_fh is not None else None
        self._candidates = (
            csv.writer(candidates_fh) if candidates_fh is not None else None)
        if confidence_fh is not None and confidence_columns is None:
            raise ValueError(
                "confidence columns are required with a confidence output")
        self._confidence = None
        self._confidence_inputs = None
        if confidence_fh is not None and confidence_columns is not None:
            self._confidence = csv.DictWriter(
                confidence_fh, fieldnames=confidence_columns)
        if confidence_inputs_fh is not None:
            if confidence_fh is None:
                raise ValueError(
                    "confidence inputs synchronization requires scored output")
            self._confidence_inputs = csv.DictWriter(
                confidence_inputs_fh, fieldnames=CONFIDENCE_INPUT_COLUMNS)
        self._confidence_columns = confidence_columns
        self.n_rows = 0
        self.n_bonds = 0
        self.n_candidates = 0
        self.n_confidence = 0
        self._manifest.writeheader()
        self._stats.writerow(STATS_COLUMNS)
        if self._bonds is not None:
            self._bonds.writerow(BOND_COLUMNS)
        if self._candidates is not None:
            self._candidates.writerow(CANDIDATE_COLUMNS)
        if self._confidence is not None:
            self._confidence.writeheader()
        if self._confidence_inputs is not None:
            self._confidence_inputs.writeheader()

    def write_stats_rows(self, rows):
        if not rows:
            return
        for row in rows:
            self._stats.writerow(
                [row["pdbID"], row["category"]] + row["fields"])
            self.n_rows += 1
        self._stats_fh.flush()

    def write_bond_rows(self, bond_rows):
        if self._bonds is None or not bond_rows:
            return
        _check_row_schema(bond_rows[0], BOND_COLUMNS, "metal_bonds_all.csv")
        for bond in bond_rows:
            self._bonds.writerow([bond[column] for column in BOND_COLUMNS])
            self.n_bonds += 1
        self._bonds_fh.flush()

    def write_candidate_rows(self, candidate_rows):
        if self._candidates is None or not candidate_rows:
            return
        _check_row_schema(candidate_rows[0], CANDIDATE_COLUMNS,
                          "metal_candidates_all.csv")
        for candidate in candidate_rows:
            self._candidates.writerow(
                [candidate[column] for column in CANDIDATE_COLUMNS])
            self.n_candidates += 1
        self._candidates_fh.flush()

    def write_manifest_row(self, row):
        self._manifest.writerow(row)
        self._manifest_fh.flush()

    def write_confidence_rows(self, rows):
        if self._confidence is None or not rows:
            return
        if self._confidence_columns is None or self._confidence_fh is None:
            raise RuntimeError("confidence output is not fully configured")
        expected = set(self._confidence_columns)
        if rows[0].keys() != expected:
            raise RuntimeError(
                "confidence row does not match its output schema")
        self._confidence.writerows(rows)
        if self._confidence_inputs is not None:
            self._confidence_inputs.writerows({
                column: row[column] for column in CONFIDENCE_INPUT_COLUMNS
            } for row in rows)
        self.n_confidence += len(rows)
        self._confidence_fh.flush()
        if self._confidence_inputs_fh is not None:
            self._confidence_inputs_fh.flush()


class _ProgressReporter:
    """Render a throttled parent-process heartbeat without worker overhead."""

    TERMINAL_INTERVAL_S = 1.0
    REDIRECTED_INTERVAL_S = 30.0

    def __init__(self, total, stream=None, clock=None):
        self.total = total
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.monotonic
        self.started = self.clock()
        self.last_rendered = float("-inf")
        self.last_width = 0
        self.terminal = bool(self.stream.isatty())
        self.line_open = False

    @staticmethod
    def _elapsed_text(elapsed_s):
        elapsed = max(0, int(elapsed_s))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def render(self, completed, counts, no_metal_count, force=False,
               final=False):
        now = self.clock()
        interval = (self.TERMINAL_INTERVAL_S if self.terminal else
                    self.REDIRECTED_INTERVAL_S)
        if not force and now - self.last_rendered < interval:
            return
        percent = 100.0 * completed / self.total if self.total else 100.0
        line = (
            f"[{completed}/{self.total} {percent:5.1f}%] "
            f"elapsed={self._elapsed_text(now - self.started)} | "
            f"ok={counts['ok']} partial={counts['partial']} "
            f"skip={counts['skip']} error={counts['error']} | "
            f"no_metals={no_metal_count}"
        )
        if self.terminal:
            padded = line.ljust(self.last_width)
            print(f"\r{padded}", end="\n" if final else "",
                  file=self.stream, flush=True)
            self.last_width = len(line)
            self.line_open = not final
        else:
            print(line, file=self.stream, flush=True)
        self.last_rendered = now

    def close(self):
        """Finish an in-place terminal line after success or an exception."""
        if self.terminal and self.line_open:
            print(file=self.stream, flush=True)
            self.line_open = False


class _RunLog:
    """Collect compact run diagnostics and write one human-readable log."""

    def __init__(self, args, command):
        self.args = args
        self.command = command
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.details = {
            "initial_available_memory_bytes": available_memory_bytes(),
        }
        self.summary = {}
        self.entries = []
        self.driver_error = ""

    def record_entry(self, result):
        """Retain diagnostic fields without keeping large result-row payloads."""
        self.entries.append({
            "pdbID": result.get("pdbID", ""),
            "status": result.get("status", "unknown"),
            "retryable": bool(result.get("retryable", False)),
            "no_metals": bool(result.get("no_metals", False)),
            "n_metals": result.get("n", 0),
            "n_bonds": result.get("n_bonds", 0),
            "n_candidates": result.get("n_candidates", 0),
            "runtime_s": float(result.get("runtime", 0.0)),
            "timings": dict(result.get("timings", {})),
            "reason_codes": list(result.get("reason_codes", [])),
            "warning_codes": list(result.get("warning_codes", [])),
            "error": str(result.get("error", "")),
            "density_map_scope_used": result.get(
                "density_map_scope_used", ""),
            "density_full_map_bytes": result.get(
                "density_full_map_bytes", 0),
            "density_edstats_map_bytes": result.get(
                "density_edstats_map_bytes", 0),
        })

    @staticmethod
    def _clean(value):
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _counter_text(counter):
        if not counter:
            return "none"
        return ", ".join(
            f"{name}={count}" for name, count in
            sorted(counter.items(), key=lambda item: (-item[1], item[0])))

    def _render(self, exit_code, finished_at, elapsed_s):
        lines = [
            "Alchemy detailed run log",
            "========================",
            f"Started (UTC): {self.started_at.isoformat()}",
            f"Finished (UTC): {finished_at.isoformat()}",
            f"Wall time: {elapsed_s:.3f} s",
            f"Exit code: {exit_code}",
            f"Command: {self.command}",
            "",
            "System",
            "------",
            f"Platform: {platform.platform()}",
            f"Python: {platform.python_version()}",
            f"Available CPUs at startup: {available_cpu_count()}",
        ]
        initial_memory = self.details.get("initial_available_memory_bytes")
        if initial_memory is None:
            lines.append("Available memory at startup: unknown")
        else:
            lines.append(
                "Available memory at startup: "
                f"{initial_memory / (1024 ** 3):.2f} GiB")
        final_memory = available_memory_bytes()
        lines.append(
            "Available memory at finish: " +
            (f"{final_memory / (1024 ** 3):.2f} GiB"
             if final_memory is not None else "unknown"))

        lines.extend(["", "Configuration", "-------------"])
        for name, value in sorted(vars(self.args).items()):
            lines.append(f"{name}: {self._clean(value)}")
        for name, value in sorted(self.details.items()):
            if name == "initial_available_memory_bytes":
                continue
            lines.append(f"{name}: {self._clean(value)}")

        status_counts = Counter(
            entry["status"] for entry in self.entries)
        reason_counts = Counter(
            reason for entry in self.entries
            for reason in entry["reason_codes"])
        warning_counts = Counter(
            warning for entry in self.entries
            for warning in entry["warning_codes"])
        retryable_count = sum(
            entry["retryable"] for entry in self.entries)
        no_metal_count = sum(
            entry["no_metals"] for entry in self.entries)
        map_scope_counts = Counter(
            entry["density_map_scope_used"] for entry in self.entries
            if entry["density_map_scope_used"])
        total_entry_s = sum(
            entry["runtime_s"] for entry in self.entries)
        throughput = (
            len(self.entries) * 60.0 / elapsed_s if elapsed_s > 0 else 0.0)

        lines.extend([
            "",
            "Summary",
            "-------",
            f"Entries returned: {len(self.entries)}",
            f"Status counts: {self._counter_text(status_counts)}",
            f"Retryable entries: {retryable_count}",
            f"Metal-free entries: {no_metal_count}",
            f"Summed entry runtime: {total_entry_s:.3f} s",
            f"Throughput: {throughput:.2f} entries/minute",
            f"Reason codes: {self._counter_text(reason_counts)}",
            f"Warning codes: {self._counter_text(warning_counts)}",
            f"Density map scopes used: {self._counter_text(map_scope_counts)}",
        ])
        for name, value in sorted(self.summary.items()):
            lines.append(f"{name}: {self._clean(value)}")
        if self.driver_error:
            lines.append(f"Driver error: {self._clean(self.driver_error)}")

        stage_values = {}
        for entry in self.entries:
            for name, value in entry["timings"].items():
                try:
                    stage_values.setdefault(name, []).append(float(value))
                except (TypeError, ValueError):
                    continue
        lines.extend(["", "Stage timing", "------------"])
        if not stage_values:
            lines.append("No completed stage timings were recorded.")
        else:
            lines.append(
                "Totals sum per-entry measurements; density_total_s contains "
                "its subprocess stages, and parallel totals are not wall time.")
            lines.append(
                "stage | entries | total_s | mean_s | max_s | max_entry")
            for name in sorted(stage_values):
                values = stage_values[name]
                stage_entries = [
                    entry for entry in self.entries
                    if name in entry["timings"]]
                max_entry = max(
                    stage_entries,
                    key=lambda entry: float(entry["timings"][name]))
                lines.append(
                    f"{name} | {len(values)} | {sum(values):.3f} | "
                    f"{sum(values) / len(values):.3f} | {max(values):.3f} | "
                    f"{max_entry['pdbID']}")

        incomplete_entries = [
            entry for entry in self.entries if entry["status"] != "ok"]
        lines.extend(["", "Incomplete entries", "------------------"])
        if not incomplete_entries:
            lines.append("None.")
        else:
            lines.append("pdbID | status | retryable | reasons | error")
            for entry in incomplete_entries:
                lines.append(
                    f"{entry['pdbID']} | {entry['status']} | "
                    f"{self._clean(entry['retryable'])} | "
                    f"{'|'.join(entry['reason_codes']) or '-'} | "
                    f"{self._clean(entry['error']) or '-'}")

        lines.extend(["", "Slowest entries", "---------------"])
        if not self.entries:
            lines.append("No entries were processed.")
        else:
            lines.append(
                "pdbID | status | runtime_s | metals | bonds | candidates | "
                "reasons")
            for entry in sorted(
                    self.entries, key=lambda item: item["runtime_s"],
                    reverse=True)[:20]:
                lines.append(
                    f"{entry['pdbID']} | {entry['status']} | "
                    f"{entry['runtime_s']:.2f} | {entry['n_metals']} | "
                    f"{entry['n_bonds']} | {entry['n_candidates']} | "
                    f"{'|'.join(entry['reason_codes']) or '-'}")

        lines.extend(["", "Per-entry results", "-----------------"])
        if not self.entries:
            lines.append("No entries were processed.")
        for entry in self.entries:
            timing_text = ",".join(
                f"{name}={float(value):.3f}"
                for name, value in sorted(entry["timings"].items())) or "-"
            lines.append(
                f"{entry['pdbID']} | status={entry['status']} | "
                f"retryable={self._clean(entry['retryable'])} | "
                f"no_metals={self._clean(entry['no_metals'])} | "
                f"runtime_s={entry['runtime_s']:.2f} | "
                f"metals={entry['n_metals']} | bonds={entry['n_bonds']} | "
                f"candidates={entry['n_candidates']} | timings={timing_text} | "
                f"density_map_scope={entry['density_map_scope_used'] or '-'} | "
                f"full_map_bytes={entry['density_full_map_bytes']} | "
                f"edstats_map_bytes={entry['density_edstats_map_bytes']} | "
                f"reasons={'|'.join(entry['reason_codes']) or '-'} | "
                f"warnings={'|'.join(entry['warning_codes']) or '-'} | "
                f"error={self._clean(entry['error']) or '-'}")
        lines.append("")
        return "\n".join(lines)

    def write(self, exit_code):
        """Atomically write the final timestamped log and return its path."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        finished_at = datetime.now(timezone.utc)
        elapsed_s = time.monotonic() - self.started_monotonic
        run_date = self.started_at.strftime("%Y%m%d")
        log_stem = f"alchemy_run_{run_date}"
        path = os.path.join(self.args.output_dir, f"{log_stem}.log")
        suffix = 2
        while os.path.lexists(path):
            path = os.path.join(
                self.args.output_dir, f"{log_stem}_{suffix}.log")
            suffix += 1
        handle, temporary_path = tempfile.mkstemp(
            prefix=".alchemy-run-log-", dir=self.args.output_dir, text=True)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as log:
                log.write(self._render(exit_code, finished_at, elapsed_s))
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
        return path


def _run(args, run_log):

    try:
        cofactors = load_cofactor_ids()
    except (OSError, UnicodeError, ValueError) as exc:
        message = f"Invalid bundled metallocofactor catalog: {exc}"
        run_log.driver_error = message
        print(message, flush=True)
        return 1

    env, _ = resolve_ccp4_environment(args)
    if env is None:
        return 0
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as exc:
        # A read-only mount or someone else's directory is a fixable user
        # mistake, so it exits the way every other unusable input does rather
        # than as a traceback that reads like an Alchemy bug.
        raise SystemExit(
            f"Cannot use --output-dir {args.output_dir}: {exc.strerror or exc}"
        ) from None
    _sweep_leaked_work_dirs(args.output_dir)

    cache_root = args.pdb_redo_cache
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    stats_path = os.path.join(args.output_dir, "metal_stats_all.csv")
    bonds_path = os.path.join(args.output_dir, "metal_bonds_all.csv")
    candidates_path = os.path.join(
        args.output_dir, "metal_candidates_all.csv")
    confidence_inputs_path = os.path.join(
        args.output_dir, "confidence_inputs_all.csv")
    confidence_scores_path = os.path.join(
        args.output_dir, "confidence_scores_all.csv")
    database_reference_dir = os.path.join(
        args.output_dir, "confidence_reference")

    manual_requested = bool(args.pdb_file or args.mtz_file or args.cif_file)
    database_run = (
        not args.id and not args.id_file and not manual_requested and
        args.max_pdbs is None
    )
    run_log.details["run_mode"] = (
        "manual" if manual_requested else
        "single" if args.id else
        "id_file" if args.id_file else
        "database" if database_run else "capped_database")
    confidence_mode = None
    confidence_reference = None
    confidence_stream_path = None
    confidence_columns = None
    synchronize_confidence_inputs = False
    if args.bonds and database_run:
        confidence_mode = "database"
        confidence_stream_path = confidence_inputs_path
        confidence_columns = CONFIDENCE_INPUT_COLUMNS
    elif args.bonds:
        confidence_reference_dir, searched_reference_dirs = (
            resolve_confidence_reference_dir(
                args.output_dir, args.confidence_reference_dir)
        )
        if confidence_reference_dir is not None:
            try:
                confidence_reference = load_confidence_reference(
                    confidence_reference_dir)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                message = f"Invalid confidence reference: {exc}"
                run_log.driver_error = message
                print(message, flush=True)
                return 1
            run_log.details["confidence_reference_dir"] = (
                confidence_reference_dir)
            confidence_mode = "reference"
            confidence_stream_path = confidence_scores_path
            confidence_columns = (
                *CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS)
        else:
            # Expected on a fresh checkout: no reference is distributed with
            # Alchemy because the confidence score is not finalized. Say so
            # plainly -- naming the searched directories alone read as a
            # misconfiguration the user was supposed to fix.
            print(
                "Confidence scoring is not enabled: no frozen reference is "
                "distributed with Alchemy, because the score is not yet "
                "finalized. All other outputs are unaffected. To enable it, "
                "complete an uncapped full-database run or pass "
                "--confidence-reference-dir. (Searched: "
                f"{', '.join(searched_reference_dirs)}.)",
                flush=True,
            )
    if args.resume:
        if (confidence_mode is not None and
                (confidence_stream_path is None or
                 not os.path.isfile(confidence_stream_path))):
            message = (
                "Cannot resume confidence-aware output because "
                f"{confidence_stream_path} is missing; use a fresh output "
                "directory.")
            run_log.driver_error = message
            print(message, flush=True)
            return 1
        try:
            validate_resume_schemas(
                manifest_path, stats_path, bonds_path, candidates_path,
                bonds_enabled=args.bonds,
                confidence_path=confidence_stream_path,
                confidence_columns=confidence_columns)
            synchronize_confidence_inputs = (
                confidence_mode == "reference" and
                os.path.isfile(confidence_inputs_path)
            )
            if synchronize_confidence_inputs:
                validate_resume_schemas(
                    manifest_path, stats_path, bonds_path, candidates_path,
                    bonds_enabled=args.bonds,
                    confidence_path=confidence_inputs_path,
                    confidence_columns=CONFIDENCE_INPUT_COLUMNS)
        except ValueError as exc:
            run_log.driver_error = str(exc)
            print(str(exc), flush=True)
            return 1
        if confidence_mode == "reference":
            try:
                validate_scored_reference(
                    confidence_stream_path, confidence_reference)
            except (OSError, ValueError) as exc:
                message = f"Cannot resume confidence output: {exc}"
                run_log.driver_error = message
                print(message, flush=True)
                return 1

    try:
        ids, root, manual_inputs = _select_entry_ids(args, cache_root)
    except _DriverError as exc:
        run_log.driver_error = str(exc)
        print(str(exc), flush=True)
        return 1

    run_log.details["entries_selected_before_resume"] = len(ids)
    run_log.details["resolved_input_root"] = root

    if args.resume:
        done_kwargs = {
            "bonds_required": args.bonds,
            "bond_output_present": os.path.isfile(bonds_path),
            "candidate_output_present": os.path.isfile(candidates_path),
        }
        normally_done = load_done(manifest_path, **done_kwargs)
        if args.retry_partials:
            done = load_done(
                manifest_path, retry_partial_ids=ids, **done_kwargs)
            reselected = normally_done - done
            run_log.details["terminal_partials_reselected"] = len(reselected)
            print(
                f"Selected {len(reselected)} terminal partial "
                f"entr{'y' if len(reselected) == 1 else 'ies'} for retry.",
                flush=True,
            )
        else:
            done = normally_done
        ids = [i for i in ids if i not in done]
    if args.max_pdbs is not None:
        ids = ids[:args.max_pdbs]
    run_log.details["entries_scheduled"] = len(ids)

    if not ids:
        if (args.resume and confidence_mode == "database" and
                os.path.isfile(confidence_inputs_path)):
            try:
                total, scored, cohort = finalize_database_confidence(
                    confidence_inputs_path,
                    confidence_scores_path,
                    database_reference_dir,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                message = f"Confidence finalization failed: {exc}"
                run_log.driver_error = message
                print(message, file=sys.stderr, flush=True)
                return 1
            print(
                "No entries required retry; finalized "
                f"{total} confidence rows ({scored} scored; database cohort "
                f"{cohort}) -> {confidence_scores_path}",
                flush=True,
            )
            print(f"Confidence reference -> {database_reference_dir}",
                  flush=True)
            return 0
        print("No entries to process.", flush=True)
        return 0

    try:
        removed_stale_bond_outputs = remove_stale_disabled_bond_outputs(
            (bonds_path, candidates_path), resume=args.resume,
            bonds_enabled=args.bonds)
    except OSError as exc:
        message = f"Could not remove stale bond-stage output: {exc}"
        run_log.driver_error = message
        print(message, flush=True)
        return 1
    for removed_path in removed_stale_bond_outputs:
        print(f"Removed stale bond-stage output: {removed_path}", flush=True)
    if not args.resume:
        # Reference metadata is its completion marker. Fresh runs must not leave
        # incompatible confidence artifacts beside newly replaced core output.
        if confidence_mode == "database":
            stale_confidence_paths = (
                confidence_scores_path,
                os.path.join(database_reference_dir, REFERENCE_METADATA_FILE),
            )
        elif confidence_mode == "reference":
            stale_confidence_paths = (confidence_inputs_path,)
        else:
            stale_confidence_paths = (
                confidence_inputs_path,
                confidence_scores_path,
                os.path.join(database_reference_dir, REFERENCE_METADATA_FILE),
            )
        try:
            for path in stale_confidence_paths:
                if os.path.isfile(path):
                    os.unlink(path)
        except OSError as exc:
            message = f"Could not clear stale confidence output: {exc}"
            run_log.driver_error = message
            print(message, flush=True)
            return 1

    # A Pool creates every worker up front, so asking for more than there are
    # entries just pays each one's startup for no work. Costly under the spawn
    # start method, where each worker re-imports gemmi into its own interpreter.
    if args.workers is None:
        cpu_limit, memory_limit = automatic_worker_limits()
        automatic_limit = cpu_limit
        if memory_limit is not None:
            automatic_limit = min(automatic_limit, memory_limit)
        workers = min(automatic_limit, len(ids))
        run_log.details["worker_selection"] = "automatic"
        run_log.details["CPU worker limit"] = cpu_limit
        run_log.details["Memory worker limit"] = (
            memory_limit if memory_limit is not None else "unavailable")
        run_log.details["Selected workers"] = workers
        print("Automatic worker selection:", flush=True)
        print(f"  CPU worker limit: {cpu_limit}", flush=True)
        print(
            "  Memory worker limit: "
            f"{memory_limit if memory_limit is not None else 'unavailable'}",
            flush=True,
        )
        print(f"  Selected workers: {workers}", flush=True)
    else:
        workers = min(args.workers, len(ids))
        run_log.details["worker_selection"] = "explicit"
        run_log.details["Selected workers"] = workers
    print(f"Processing {len(ids)} entr{'y' if len(ids) == 1 else 'ies'} "
          f"with {workers} worker(s) ...", flush=True)

    cfg = {"root": root, "mirror_root": args.pdb_redo_root,
           "cache_root": cache_root, "env": env,
           "output_dir": args.output_dir, "cofactors": cofactors,
           "keep": args.keep_intermediates, "bonds": args.bonds,
           "density_map_scope": args.density_map_scope,
           "allow_download": bool(args.id or args.id_file),
           "manual_inputs": manual_inputs,
           "alchemy_commit": _alchemy_commit(),
           "gemmi_version": _gemmi_version(),
           "ccp4_version": _ccp4_version(env)}
    run_log.details.update(
        alchemy_version=ALCHEMY_VERSION,
        gemmi_version=cfg["gemmi_version"],
        ccp4_version=cfg["ccp4_version"],
        confidence_mode=confidence_mode or "disabled",
    )

    prior_bond_counts = (
        _manifest_values_by_id(manifest_path, "n_bonds")
        if args.resume and not args.bonds else {}
    )
    prior_candidate_counts = (
        _manifest_values_by_id(manifest_path, "n_candidates")
        if args.resume and not args.bonds else {}
    )
    output_paths = [manifest_path, stats_path, bonds_path, candidates_path]
    if confidence_mode is not None:
        if confidence_stream_path is None:
            raise RuntimeError("confidence output path is not configured")
        output_paths.append(confidence_stream_path)
    if synchronize_confidence_inputs:
        output_paths.append(confidence_inputs_path)
    output_paths = tuple(output_paths)
    staging = (_ResumeStaging(args.output_dir, output_paths)
               if args.resume else None)
    write_paths = staging.staged if staging is not None else output_paths
    (write_manifest_path, write_stats_path, write_bonds_path,
     write_candidates_path) = write_paths[:4]
    write_confidence_path = (
        write_paths[4] if confidence_mode is not None else None)
    write_confidence_inputs_path = (
        write_paths[5] if synchronize_confidence_inputs else None)

    counts = {"ok": 0, "partial": 0, "skip": 0, "error": 0}
    no_metal_count = 0
    retryable_partial_count = 0
    processing_completed = False
    writers = None
    progress = _ProgressReporter(len(ids))
    try:
        # ExitStack closes whichever handles were opened, so a failure partway
        # through opening them cannot leak the earlier ones.
        with contextlib.ExitStack() as handles:
            writers = _OutputWriters(
                handles.enter_context(
                    open(write_manifest_path, "w", newline="")),
                handles.enter_context(
                    open(write_stats_path, "w", newline="")),
                handles.enter_context(
                    open(write_bonds_path, "w", newline=""))
                if args.bonds else None,
                handles.enter_context(
                    open(write_candidates_path, "w", newline=""))
                if args.bonds else None,
                handles.enter_context(
                    open(write_confidence_path, "w", newline=""))
                if write_confidence_path is not None else None,
                confidence_columns,
                handles.enter_context(
                    open(write_confidence_inputs_path, "w", newline=""))
                if write_confidence_inputs_path is not None else None,
            )
            inflight = SimpleQueue()
            assignments: Dict[int, str] = {}
            worker_pids: set = set()
            lost_ids: set = set()
            completed_ids: set = set()
            unattributed_deaths = 0
            last_progress = time.monotonic()
            # Not a ``with`` block: ``Pool.__exit__`` calls the same
            # ``terminate`` that a killed idle worker can wedge forever, so
            # shutdown is bounded explicitly in the ``finally`` below.
            pool = Pool(workers, initializer=_init_worker,
                        initargs=(cfg, inflight))
            try:
                results = pool.imap_unordered(process, ids, chunksize=1)
                completed = 0
                progress.render(
                    completed, counts, no_metal_count, force=True)
                while completed < len(ids):
                    batch = []
                    try:
                        batch.append(results.next(timeout=1.0))
                    except MultiprocessingTimeoutError:
                        pass
                    # A killed worker never delivers a result, so its entry is
                    # recovered from the pool roster rather than waited on.
                    _drain_inflight(inflight, assignments)
                    for dead_pid in _dead_worker_pids(pool, worker_pids):
                        dead_id = assignments.pop(dead_pid, None)
                        if dead_id is None:
                            unattributed_deaths += 1
                        elif dead_id not in lost_ids:
                            lost_ids.add(dead_id)
                            batch.append(_worker_death_result(
                                dead_id, cfg, dead_pid))
                    if not batch:
                        stalled = time.monotonic() - last_progress
                        remaining = len(ids) - completed
                        if (unattributed_deaths
                                and stalled > WORKER_STALL_GRACE_S
                                and remaining <= unattributed_deaths):
                            # Every entry still outstanding can only be held by
                            # a process that died before naming its entry.
                            # Entries that already returned a result are not
                            # outstanding: blaming one of those would write a
                            # second, failed row for an entry that succeeded and
                            # leave the entry that actually died unreported.
                            for stuck_id in ids:
                                if (stuck_id in lost_ids
                                        or stuck_id in completed_ids):
                                    continue
                                lost_ids.add(stuck_id)
                                batch.append(_worker_death_result(
                                    stuck_id, cfg, 0))
                                if len(batch) >= remaining:
                                    break
                        else:
                            progress.render(
                                completed, counts, no_metal_count)
                            continue
                    last_progress = time.monotonic()
                    for r in batch:
                        if (r["pdbID"] in lost_ids and r.get("reason_codes")
                                != ["worker_process_died"]):
                            # A real result arrived for an entry already
                            # declared lost. The synthesized row stands, so
                            # this one is dropped rather than written twice.
                            continue
                        completed += 1
                        completed_ids.add(r["pdbID"])
                        run_log.record_entry(r)
                        if (not args.resume
                                or _resume_replacement_succeeded(r)):
                            writers.write_stats_rows(r["rows"])
                            writers.write_bond_rows(r["bond_rows"])
                            writers.write_candidate_rows(r["candidate_rows"])
                            confidence_rows = []
                            if confidence_mode is not None:
                                confidence_rows = (
                                    prepare_result_confidence_inputs(
                                        r["rows"], r["bond_rows"],
                                        STATS_COLUMNS))
                                confidence_rows = (
                                    complete_confidence_site_count(
                                        confidence_rows, r["pdbID"], r["n"],
                                        r.get(
                                            "confidence_inputs_missing_reason",
                                            "")))
                                if confidence_mode == "reference":
                                    confidence_rows = score_against_reference(
                                        confidence_rows, confidence_reference)
                                writers.write_confidence_rows(confidence_rows)
                            # The manifest is the completion marker, so write
                            # it only after this entry's rows have flushed.
                            writers.write_manifest_row(_manifest_row(
                                r, args.resume, args.bonds, prior_bond_counts,
                                prior_candidate_counts))
                            if staging is not None:
                                staging.replacement_ids.add(
                                    r["pdbID"].lower())
                        counts[r["status"]] = counts.get(r["status"], 0) + 1
                        if r.get("no_metals", False):
                            no_metal_count += 1
                        if (r["status"] == "partial"
                                and r.get("retryable", False)):
                            retryable_partial_count += 1
                        finished = completed == len(ids)
                        progress.render(
                            completed, counts, no_metal_count,
                            force=progress.terminal or finished,
                            final=finished)
            finally:
                if _shutdown_pool(pool):
                    run_log.summary["worker_pool_forced_shutdown"] = True
                    print("Warning: a worker pool shutdown had to be forced "
                          "after a worker died holding the task-queue lock; "
                          "results above are complete.", flush=True)
            processing_completed = True
    finally:
        progress.close()
        if staging is not None and not processing_completed:
            staging.discard()

    n_rows = writers.n_rows if writers is not None else 0
    n_bonds = writers.n_bonds if writers is not None else 0
    n_candidates = writers.n_candidates if writers is not None else 0
    run_log.summary.update(
        metal_rows_written=n_rows,
        bond_rows_written=n_bonds,
        candidate_rows_written=n_candidates,
        confidence_rows_written=(
            writers.n_confidence if writers is not None else 0),
        manifest_path=manifest_path,
        metal_stats_path=stats_path,
        metal_bonds_path=bonds_path if args.bonds else "disabled",
        metal_candidates_path=(candidates_path if args.bonds else "disabled"),
    )

    if staging is not None:
        try:
            staging.commit(
                args.bonds, confidence_enabled=confidence_mode is not None)
        finally:
            staging.discard()

    print(f"Done. ok={counts['ok']} partial={counts['partial']} "
          f"skip={counts['skip']} error={counts['error']} "
          f"no_metals={no_metal_count}; "
          f"{n_rows} metal/cofactor rows -> {stats_path}", flush=True)
    if args.bonds:
        print(f"      {n_bonds} bond rows -> {bonds_path}", flush=True)
        print(f"      {n_candidates} candidate rows -> {candidates_path}",
              flush=True)
    exit_code = _batch_exit_code(counts, retryable_partial_count)
    if confidence_mode == "database":
        if exit_code == 0:
            try:
                total, scored, cohort = finalize_database_confidence(
                    confidence_inputs_path,
                    confidence_scores_path,
                    database_reference_dir,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                message = f"Confidence finalization failed: {exc}"
                run_log.driver_error = message
                print(message, file=sys.stderr, flush=True)
                return 1
            run_log.summary.update(
                confidence_status="finalized",
                confidence_rows=total,
                confidence_scored_rows=scored,
                confidence_reference_cohort=cohort,
                confidence_scores_path=confidence_scores_path,
                confidence_reference_path=database_reference_dir,
            )
            print(
                f"      {total} confidence rows ({scored} scored; "
                f"reference cohort {cohort}) -> {confidence_scores_path}",
                flush=True,
            )
            print(f"      confidence reference -> {database_reference_dir}",
                  flush=True)
        else:
            run_log.summary["confidence_status"] = (
                "not_finalized_incomplete_run")
            print(
                "      confidence inputs were retained, but the database "
                "reference was not finalized because the run is incomplete.",
                flush=True,
            )
    elif confidence_mode == "reference":
        if confidence_reference is None:
            raise RuntimeError("confidence reference is not configured")
        print(
            f"      {writers.n_confidence} confidence rows compared with "
            f"database cohort {confidence_reference.cohort_size} -> "
            f"{confidence_scores_path}",
            flush=True,
        )
        run_log.summary.update(
            confidence_status="scored_against_reference",
            confidence_reference_cohort=confidence_reference.cohort_size,
            confidence_scores_path=confidence_scores_path,
        )
    if exit_code:
        print(
            "Alchemy completed with incomplete entries: "
            f"errors={counts['error']}, skips={counts['skip']}, "
            f"retryable_partials={retryable_partial_count}.",
            file=sys.stderr,
            flush=True,
        )
    return exit_code


def main(argv=None):
    """Parse arguments, execute the driver, and always emit a run log."""
    raw_args = None if argv is None else list(argv)
    args = parse_args(raw_args)
    command_parts = (
        list(sys.argv) if raw_args is None else [sys.argv[0], *raw_args])
    run_log = _RunLog(args, shlex.join(command_parts))
    exit_code = 1
    previous_term = _install_termination_handler()
    try:
        exit_code = _run(args, run_log)
        return exit_code
    except KeyboardInterrupt:
        # Ctrl-C or SIGTERM. The pool has already been shut down by _run's
        # own finally, so this only decides how the driver reports it: a
        # conventional interrupt status and one line, rather than a traceback
        # that looks like a crash.
        run_log.driver_error = "interrupted before completion"
        print("\nInterrupted: workers stopped; rows already flushed are kept "
              "and --resume will continue.", file=sys.stderr, flush=True)
        exit_code = 130
        return exit_code
    except BaseException as exc:
        run_log.driver_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if previous_term is not None:
            with contextlib.suppress(AttributeError, OSError, ValueError):
                signal.signal(signal.SIGTERM, previous_term)
        try:
            log_path = run_log.write(exit_code)
        except OSError as exc:
            print(f"Could not write detailed run log: {exc}",
                  file=sys.stderr, flush=True)
        else:
            print(f"Detailed run log -> {log_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
