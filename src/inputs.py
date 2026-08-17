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
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any, Protocol, cast
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import numpy.typing as npt

from coordinate_conversion import cif_to_pdb
from run_logging import logger_for


# The four Fourier coefficients EDSTATS' two maps are calculated from; a
# reflection is usable only where all four are present.
MAP_COEFFICIENT_COLUMNS = ("FWT", "PHWT", "DELFWT", "PHDELWT")

logger = logger_for(__name__)


@dataclass(frozen=True)
class PdbRedoMetadata:
    is_twin: bool = False
    version: str = ""
    date: str = ""


class _MtzColumnArray(Protocol):
    """The concrete array member omitted from part of Gemmi's column stub."""

    array: npt.NDArray[np.float32]


def read_data_json_properties(data_json_path: str) -> dict[str, Any]:
    """Return a validated PDB-REDO ``properties`` object.

    Callers use this strict reader when a user explicitly supplied the file.
    Automatically discovered metadata may catch ``ValueError`` and retain its
    documented fallback behavior.
    """
    try:
        with open(data_json_path, encoding="utf-8") as handle:
            loaded: object = json.load(handle)
    except FileNotFoundError:
        raise ValueError(f"data.json file not found: {data_json_path}") from None
    except OSError as exc:
        raise ValueError(f"could not read data.json {data_json_path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in data.json {data_json_path}: {exc.msg}"
        ) from None

    if not isinstance(loaded, dict):
        raise ValueError(f"data.json must contain a JSON object: {data_json_path}")
    payload = cast("dict[str, object]", loaded)
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(
            f"data.json must contain a properties object: {data_json_path}"
        )
    return cast("dict[str, Any]", properties)


def entry_dir_for(root: str, pdb_id: str) -> str:
    """PDB-REDO layout: <root>/<middle two chars of id>/<id>/."""
    return os.path.join(root, pdb_id[1:3], pdb_id)


def _gunzip_to(src_gz: str, dst: str) -> str:
    if not os.path.exists(src_gz):
        raise FileNotFoundError(src_gz)
    with gzip.GzipFile(src_gz, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    return dst


def first_existing(*paths: str) -> str | None:
    return next((path for path in paths if os.path.exists(path)), None)


# A proxy notice or captive-portal login page is served with status 200, so the
# transfer succeeds and the bytes land under the entry's own file name. Only
# the opening bytes are inspected: enough to tell a document from a structure
# file without reading a 100 MB MTZ to decide whether to keep it.
_HTML_PREFIXES = (b"<!doctype", b"<html", b"<?xml")

# A served notice is a page, and pages are small: the largest captive-portal or
# proxy-error body is orders of magnitude under this. Size alone therefore
# clears any bigger file without opening it, which is what makes enumerating a
# full local mirror affordable -- probing the content of every candidate file
# cost a ~195k-entry run about two hours of startup, because each open pulls a
# readahead window off the filesystem to read 64 bytes.
_MAX_WEB_PAGE_BYTES = 1024 * 1024


def _looks_like_a_web_page(path: str, size: int | None = None) -> bool:
    """Whether a cached file holds a served document rather than entry data.

    ``size`` lets a caller that has already stat-ed the file skip the read for
    anything too large to be a page.
    """
    if size is not None and size > _MAX_WEB_PAGE_BYTES:
        return False
    try:
        with open(path, "rb") as handle:
            head = handle.read(64).lstrip().lower()
    except OSError:
        return False
    return head.startswith(_HTML_PREFIXES)


def _is_usable_entry_file(path: str | None) -> bool:
    """Whether a cached path is worth reading rather than re-fetching.

    Existence alone is not enough. A truncated or wrong-status body written
    under the right name would be cached permanently: every later run reads the
    same unusable file, fails identically, and no ``--resume`` recovers it short
    of deleting the cache by hand.
    """
    if path is None:
        return False
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if size == 0:
        return False
    return not _looks_like_a_web_page(path, size=size)


def prepare_inputs(pdb_id: str, entry_dir: str, work_dir: str) -> tuple[str, str]:
    """Return the final PDB-REDO ``(mtz_path, pdb_path)`` analysis inputs.

    The authoritative final mmCIF is preferred and converted for EDSTATS; the
    PDB compatibility export is used only when no mmCIF exists. Compressed
    mirrors are accepted for either format.
    """
    mtz = first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.mtz"),
        os.path.join(entry_dir, f"{pdb_id}_final.mtz.gz"),
    )
    if mtz is None:
        raise FileNotFoundError(os.path.join(entry_dir, f"{pdb_id}_final.mtz"))
    if mtz.endswith(".gz"):
        mtz = _gunzip_to(mtz, os.path.join(work_dir, f"{pdb_id}_final.mtz"))

    cif = first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.cif"),
        os.path.join(entry_dir, f"{pdb_id}_final.cif.gz"),
    )
    pdb: str | None
    if cif is not None:
        pdb = cif_to_pdb(cif, os.path.join(work_dir, f"{pdb_id}_final_from_cif.pdb"))
        return mtz, pdb

    pdb = first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.pdb"),
        os.path.join(entry_dir, f"{pdb_id}_final.pdb.gz"),
    )
    if pdb is None:
        raise FileNotFoundError(f"{pdb_id}_final.cif or {pdb_id}_final.pdb")
    if pdb.endswith(".gz"):
        pdb = _gunzip_to(pdb, os.path.join(work_dir, f"{pdb_id}_final.pdb"))
    return mtz, pdb


def read_resolution(
    entry_dir: str, mtz_path: str, data_json_path: str | None = None
) -> float:
    """Return the overall diffraction-data high-resolution limit.

    Only the high-resolution limit is reported, because that is what the DPI
    metadata records; EDSTATS is given the map columns' own range by
    ``read_map_column_resolution`` instead.
    """
    explicit = data_json_path is not None
    dj = (
        data_json_path
        if data_json_path is not None
        else os.path.join(entry_dir, "data.json")
    )
    if explicit or os.path.exists(dj):
        try:
            props = read_data_json_properties(dj)
            lo, hi = props.get("DATARESL"), props.get("DATARESH")
            # A half-populated record is not trusted; fall back to the MTZ.
            if lo and hi:
                return float(hi)
        except (TypeError, ValueError):
            if explicit:
                raise
    import gemmi

    return gemmi.read_mtz_file(mtz_path).resolution_high()


def read_pdb_redo_metadata(
    data_json_path: str | None, *, required: bool = False
) -> PdbRedoMetadata:
    """Read the PDB-REDO fields that affect analysis or source provenance.

    Automatically discovered missing or malformed metadata, and string-valued
    flags, are false: the twin coefficient fallback must not be inferred from
    a filename or from an MTZFIX failure. When ``required`` is true, the path
    came from ``--data-json`` and read or structural errors are fatal instead.
    """
    if not data_json_path:
        if required:
            raise ValueError("an explicit data.json path is required")
        return PdbRedoMetadata()
    try:
        properties = read_data_json_properties(data_json_path)
    except ValueError:
        if required:
            raise
        return PdbRedoMetadata()

    def text(name: str) -> str:
        value = properties.get(name)
        if value is None or isinstance(value, (bool, dict, list)):
            return ""
        return str(value).strip()

    return PdbRedoMetadata(
        is_twin=properties.get("ISTWIN") is True,
        version=text("VERSION"),
        date=text("TIME"),
    )


def read_pdb_redo_is_twin(
    data_json_path: str | None, *, required: bool = False
) -> bool:
    return read_pdb_redo_metadata(data_json_path, required=required).is_twin


def read_map_column_resolution(mtz_path: str) -> tuple[float, float]:
    """Return the common finite resolution range of both EDSTATS maps.

    EDSTATS receives maps calculated from FWT/PHWT and DELFWT/PHDELWT, so its
    limits must describe reflections for which all four values are present,
    rather than the overall range of unrelated columns in the MTZ.
    """
    import gemmi

    mtz = gemmi.read_mtz_file(mtz_path)
    columns: list[gemmi.Mtz.Column] = []
    missing: list[str] = []
    for label in MAP_COEFFICIENT_COLUMNS:
        # gemmi's stub declares a plain Column here, but the binding returns
        # None for a label the file does not carry.
        column = cast("gemmi.Mtz.Column | None", mtz.column_with_label(label))
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
        usable &= np.isfinite(cast(_MtzColumnArray, column).array)
    usable_d = d_values[usable]
    if usable_d.size == 0:
        raise ValueError(
            "MTZ map coefficient columns have no common finite reflections"
        )
    return float(usable_d.max()), float(usable_d.min())


def has_final_files(entry_dir: str, pdb_id: str) -> bool:
    """Whether an entry has final map coefficients and usable coordinates."""
    mtz = first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.mtz"),
        os.path.join(entry_dir, f"{pdb_id}_final.mtz.gz"),
    )
    coordinates = first_existing(
        os.path.join(entry_dir, f"{pdb_id}_final.cif"),
        os.path.join(entry_dir, f"{pdb_id}_final.cif.gz"),
        os.path.join(entry_dir, f"{pdb_id}_final.pdb"),
        os.path.join(entry_dir, f"{pdb_id}_final.pdb.gz"),
    )
    return _is_usable_entry_file(mtz) and _is_usable_entry_file(coordinates)


def _remove_partial_download(tmp: str) -> None:
    """Drop a ``.part`` file so no caller can mistake it for a complete one."""
    try:
        os.remove(tmp)
    except OSError:
        pass


def _response_content_length(response: object, url: str) -> int | None:
    """Return a validated HTTP Content-Length, or ``None`` when absent."""
    getheader = getattr(response, "getheader", None)
    if callable(getheader):
        value = getheader("Content-Length")
    else:
        headers = getattr(response, "headers", None)
        value = headers.get("Content-Length") if headers is not None else None
    if value in (None, ""):
        return None
    text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    try:
        length = int(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FileNotFoundError(
            f"{url}: invalid Content-Length header {value!r}"
        ) from exc
    if length < 0:
        raise FileNotFoundError(f"{url}: invalid Content-Length header {value!r}")
    return length


def download_stream(url: str, dst: str, timeout: float = 30) -> str:
    """Download URL to dst. Raise FileNotFoundError if no usable file results.

    A transfer that begins and then fails -- a reset connection, a read
    timeout, a body shorter than its Content-Length -- leaves the caller in the
    same position as a 404: no file. Guarding the opening request alone leaves
    them raising out of the driver's pre-flight as a bare traceback, and
    ``http.client.IncompleteRead`` is not even an ``OSError``, so it escapes
    handlers written for one.
    """
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
            expected_bytes = _response_content_length(response, url)
            received_bytes = 0
            with open(tmp, "wb") as fh:
                while chunk := response.read(8192):
                    fh.write(chunk)
                    received_bytes += len(chunk)
            if expected_bytes is not None and received_bytes != expected_bytes:
                raise FileNotFoundError(
                    f"{url}: incomplete response body: expected "
                    f"{expected_bytes} bytes, received {received_bytes}"
                )
        os.replace(tmp, dst)
    except FileNotFoundError:
        _remove_partial_download(tmp)
        raise
    except (OSError, HTTPException) as e:
        _remove_partial_download(tmp)
        raise FileNotFoundError(f"{url}: {type(e).__name__}: {e}") from e
    except Exception:
        _remove_partial_download(tmp)
        raise
    return dst


def download_entry_to_cache(pdb_id: str, cache_root: str) -> None:
    """Download final PDB-REDO files into a mirror-like ``cache_root``."""
    base = f"https://pdb-redo.eu/db/{pdb_id}/"
    entry = entry_dir_for(cache_root, pdb_id)
    os.makedirs(entry, exist_ok=True)

    def try_fetch(name: str) -> bool:
        url = base + name
        dst = os.path.join(entry, name)
        try:
            download_stream(url, dst)
            return True
        except FileNotFoundError:
            return False

    def fetch_variant(name: str) -> bool:
        # Reuse a cached file only if it is worth reading: short-circuiting on
        # existence alone makes an empty or served-document body permanent.
        cached = first_existing(
            os.path.join(entry, name), os.path.join(entry, name + ".gz")
        )
        if _is_usable_entry_file(cached):
            return True
        return try_fetch(name) or try_fetch(name + ".gz")

    fetch_variant(f"{pdb_id}_final.mtz")
    if not fetch_variant(f"{pdb_id}_final.cif"):
        fetch_variant(f"{pdb_id}_final.pdb")
    if not os.path.exists(os.path.join(entry, "data.json")):
        try_fetch("data.json")

    if not has_final_files(entry, pdb_id):
        raise FileNotFoundError(f"PDB-REDO entry {pdb_id} is missing final model files")


def ensure_entry_available(pdb_id: str, mirror_root: str, cache_root: str) -> str:
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
    pdb_id: str,
    pdb_file: str | None = None,
    mtz_file: str | None = None,
    cif_file: str | None = None,
    work_dir: str | None = None,
) -> tuple[str, str]:
    """Return (mtz_path, pdb_path) for a manually supplied local input set."""
    if not mtz_file:
        raise ValueError("manual mode requires --mtz-file")
    if not os.path.exists(mtz_file):
        raise FileNotFoundError(f"mtz file not found: {mtz_file}")

    if cif_file:
        if not os.path.exists(cif_file):
            raise FileNotFoundError(f"cif file not found: {cif_file}")
        target_pdb = os.path.join(work_dir or os.getcwd(), f"{pdb_id}.pdb")
        return mtz_file, cif_to_pdb(cif_file, target_pdb)

    if pdb_file:
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"pdb file not found: {pdb_file}")
        return mtz_file, pdb_file

    raise ValueError("manual mode requires --pdb-file or --cif-file")


def infer_pdb_id_from_path(path: str | None) -> str | None:
    """Infer a 4-char PDB id from a local file name if possible."""
    if not path:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"([A-Za-z0-9]{4})(?:_.*)?$", stem)
    return m.group(1).lower() if m else None


def enumerate_entries(root: str, limit: int | None = None) -> list[str]:
    """All PDB ids under ``root`` that have final model files.

    ``limit`` stops the walk once that many sorted ids are collected, so a
    small --max-pdbs run does not traverse all ~24k entries.
    """
    ids: list[str] = []
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
