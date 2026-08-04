"""Getting one PDB-REDO entry ready for analysis, and reading its metadata.

Everything between "here is a PDB id" and "here are the two files the pipeline
runs on": locating an entry in a mirror, downloading it into the cache when it
is absent, decompressing what arrived, converting the coordinates, and reading
the resolution limits off the result.
"""

import gzip
import json
import os
import re
import shutil
from urllib.error import HTTPError
from urllib.request import urlopen

from coordinate_conversion import _cif_to_pdb
from run_logging import logger_for


# The four Fourier coefficients EDSTATS' two maps are calculated from; a
# reflection is usable only where all four are present.
MAP_COEFFICIENT_COLUMNS = ("FWT", "PHWT", "DELFWT", "PHDELWT")

logger = logger_for(__name__)


def entry_dir_for(root, pdb_id):
    """PDB-REDO layout: <root>/<middle two chars of id>/<id>/."""
    return os.path.join(root, pdb_id[1:3], pdb_id)


def _gunzip_to(src_gz, dst):
    if not os.path.exists(src_gz):
        raise FileNotFoundError(src_gz)
    with gzip.GzipFile(src_gz, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return dst


def _first_existing(*paths):
    return next((path for path in paths if os.path.exists(path)), None)


def prepare_inputs(pdb_id, entry_dir, work_dir):
    """Return the final PDB-REDO ``(mtz_path, pdb_path)`` analysis inputs.

    The authoritative final mmCIF is preferred and converted for EDSTATS; the
    PDB compatibility export is used only when no mmCIF exists. Compressed
    mirrors are accepted for either format.
    """
    mtz = _first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.mtz"),
        os.path.join(entry_dir, f"{pdb_id}_final.mtz.gz"),
    )
    if mtz is None:
        raise FileNotFoundError(os.path.join(entry_dir, f"{pdb_id}_final.mtz"))
    if mtz.endswith(".gz"):
        mtz = _gunzip_to(mtz, os.path.join(work_dir, f"{pdb_id}_final.mtz"))

    cif = _first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.cif"),
        os.path.join(entry_dir, f"{pdb_id}_final.cif.gz"),
    )
    if cif is not None:
        pdb = _cif_to_pdb(cif, os.path.join(work_dir, f"{pdb_id}_final_from_cif.pdb"))
        return mtz, pdb

    pdb = _first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.pdb"),
        os.path.join(entry_dir, f"{pdb_id}_final.pdb.gz"),
    )
    if pdb is None:
        raise FileNotFoundError(f"{pdb_id}_final.cif or {pdb_id}_final.pdb")
    if pdb.endswith(".gz"):
        pdb = _gunzip_to(pdb, os.path.join(work_dir, f"{pdb_id}_final.pdb"))
    return mtz, pdb


def read_resolution(entry_dir, mtz_path, data_json_path=None):
    """Return the overall diffraction-data high-resolution limit.

    Only the high-resolution limit is reported, because that is what the DPI
    metadata records; EDSTATS is given the map columns' own range by
    ``read_map_column_resolution`` instead.
    """
    dj = data_json_path or os.path.join(entry_dir, "data.json")
    if os.path.exists(dj):
        try:
            with open(dj) as handle:
                props = json.load(handle).get("properties", {})
            lo, hi = props.get("DATARESL"), props.get("DATARESH")
            # A half-populated record is not trusted; fall back to the MTZ.
            if lo and hi:
                return float(hi)
        except (ValueError, KeyError, OSError):
            pass
    import gemmi

    return gemmi.read_mtz_file(mtz_path).resolution_high()


def read_pdb_redo_is_twin(data_json_path):
    """Return only an explicit boolean PDB-REDO ``properties.ISTWIN`` value.

    Missing, malformed and string-valued metadata are false: the twin
    coefficient fallback must not be inferred from a filename or from an
    MTZFIX failure.
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
    missing: list[str] = []
    for label in MAP_COEFFICIENT_COLUMNS:
        column = mtz.column_with_label(label)
        if column is None:
            missing.append(label)
        else:
            columns.append(column)
    if missing:
        raise ValueError(
            "MTZ is missing required map coefficient column(s): " + ", ".join(missing)
        )

    d_values = mtz.make_d_array()
    row_count = len(d_values)
    if any(len(column) != row_count for column in columns):
        raise ValueError(
            "MTZ map coefficient columns do not match the reflection count"
        )

    # A reflection counts only where its d-spacing and all four coefficients
    # are finite, so the mask spans whole rows.
    usable = np.isfinite(d_values) & (d_values > 0.0)
    for column in columns:
        usable &= np.isfinite(column.array)
    usable_d = d_values[usable]
    if usable_d.size == 0:
        raise ValueError(
            "MTZ map coefficient columns have no common finite reflections"
        )
    return float(usable_d.max()), float(usable_d.min())


def has_final_files(entry_dir, pdb_id):
    """Whether an entry has final map coefficients and usable coordinates."""
    mtz = _first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.mtz"),
        os.path.join(entry_dir, f"{pdb_id}_final.mtz.gz"),
    )
    coordinates = _first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.cif"),
        os.path.join(entry_dir, f"{pdb_id}_final.cif.gz"),
        os.path.join(entry_dir, f"{pdb_id}_final.pdb"),
        os.path.join(entry_dir, f"{pdb_id}_final.pdb.gz"),
    )
    return mtz is not None and coordinates is not None


def _download_stream(url, dst, timeout=30):
    """Download URL to dst. Raise FileNotFoundError on non-200."""
    try:
        response = urlopen(url, timeout=timeout)
    except HTTPError as e:
        raise FileNotFoundError(f"{url}: status {e.code}") from e
    except (OSError, ValueError) as e:
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


def download_entry_to_cache(pdb_id, cache_root):
    """Download final PDB-REDO files into a mirror-like ``cache_root``."""
    base = f"https://pdb-redo.eu/db/{pdb_id}/"
    entry = entry_dir_for(cache_root, pdb_id)
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
        if (
            _first_existing(
                os.path.join(entry, name), os.path.join(entry, name + ".gz")
            )
            is not None
        ):
            return True
        return try_fetch(name) or try_fetch(name + ".gz")

    fetch_variant(f"{pdb_id}_final.mtz")
    if not fetch_variant(f"{pdb_id}_final.cif"):
        fetch_variant(f"{pdb_id}_final.pdb")
    if not os.path.exists(os.path.join(entry, "data.json")):
        try_fetch("data.json")

    if not has_final_files(entry, pdb_id):
        raise FileNotFoundError(f"PDB-REDO entry {pdb_id} is missing final model files")


def ensure_entry_available(pdb_id, mirror_root, cache_root):
    """Return the root containing the final model files: mirror, then cache.

    A cache miss triggers a download into ``cache_root``.
    """
    mirror_entry = entry_dir_for(mirror_root, pdb_id)
    if os.path.isdir(mirror_entry) and has_final_files(mirror_entry, pdb_id):
        return mirror_root
    cache_entry = entry_dir_for(cache_root, pdb_id)
    if os.path.isdir(cache_entry) and has_final_files(cache_entry, pdb_id):
        return cache_root
    download_entry_to_cache(pdb_id, cache_root)
    if os.path.isdir(cache_entry) and has_final_files(cache_entry, pdb_id):
        return cache_root
    raise FileNotFoundError(pdb_id)


def resolve_manual_inputs(
    pdb_id, pdb_file=None, mtz_file=None, cif_file=None, work_dir=None
):
    """Return (mtz_path, pdb_path) for a manually supplied local input set."""
    if not mtz_file:
        raise ValueError("manual mode requires --mtz-file")
    if not os.path.exists(mtz_file):
        raise FileNotFoundError(f"mtz file not found: {mtz_file}")

    if cif_file:
        if not os.path.exists(cif_file):
            raise FileNotFoundError(f"cif file not found: {cif_file}")
        target_pdb = os.path.join(work_dir or os.getcwd(), f"{pdb_id}.pdb")
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

    ``limit`` stops the walk once that many sorted ids are collected, so a
    small --max-pdbs run does not traverse all ~24k entries.
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
            # Common on a partially-synced mirror; one unreadable hashdir must
            # not abort the whole enumeration.
            skipped += 1
            logger.warning("skipping unreadable directory %s: %s", hp, e)
            continue
        for pid in entries:
            ep = os.path.join(hp, pid)
            if os.path.isdir(ep) and has_final_files(ep, pid):
                ids.append(pid)
                if limit is not None and len(ids) >= limit:
                    return ids
    if skipped:
        logger.warning("skipped %d unreadable hashdir(s) under %s", skipped, root)
    return ids
