"""Locate, download, and prepare entry files and read their metadata."""

import contextlib
import gzip
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any, Literal, NamedTuple, Protocol, cast
from urllib.error import HTTPError
from urllib.request import urlopen

import gemmi
import numpy as np

from coordinate_conversion import cif_to_pdb
from gemmi_typing import mtz_column_data
from run_logging import logger_for

# The four Fourier coefficients EDSTATS' two maps are calculated from; a
# reflection is usable only where all four are present.
MAP_COEFFICIENT_COLUMNS = ("FWT", "PHWT", "DELFWT", "PHDELWT")

logger = logger_for(__name__)

#: A PDB identifier: four alphanumeric characters, compared case-insensitively.
PDB_ID_PATTERN = r"[A-Za-z0-9]{4}"


@dataclass(frozen=True)
class PdbRedoMetadata:
    """Store the PDB-REDO metadata used for provenance and map handling."""

    is_twin: bool = False
    version: str = ""
    date: str = ""


class MissingInputError(FileNotFoundError):
    """An entry's input file could not be located, downloaded, or read.

    Raised as a domain signal, distinct from an OS-level ``FileNotFoundError``,
    so the worker can skip the entry as missing input. It subclasses
    ``FileNotFoundError`` so existing handlers keep working.
    """


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
    """Decompress ``src_gz`` to ``dst``; an unreadable archive is a missing input.

    A truncated archive raises ``EOFError`` rather than an ``OSError``, and
    either must not leave a partial ``dst`` for a later reader to trust.
    """
    try:
        with gzip.open(src_gz, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    except (OSError, EOFError) as exc:
        with contextlib.suppress(OSError):
            os.remove(dst)
        raise MissingInputError(f"{src_gz}: {type(exc).__name__}: {exc}") from exc
    return dst


def first_existing(*paths: str) -> str | None:
    """Return the first existing path, or ``None`` when none exists."""
    return next((path for path in paths if os.path.exists(path)), None)


def final_file_candidates(
    entry_dir: str, pdb_id: str, kind: Literal["mtz", "coordinates"]
) -> tuple[str, ...]:
    """Return the paths a final PDB-REDO file may be stored under, in priority order.

    Plain files win over their gzipped mirrors, and for coordinates the
    authoritative mmCIF wins over the PDB compatibility export. The first
    usable candidate is the one an analysis uses, so the order matters.
    """
    extensions = ("mtz",) if kind == "mtz" else ("cif", "pdb")
    return tuple(
        os.path.join(entry_dir, f"{pdb_id}_final.{extension}{suffix}")
        for extension in extensions
        for suffix in ("", ".gz")
    )


# Detect HTML error pages returned with HTTP 200 before accepting cached data.
_HTML_PREFIXES = (b"<!doctype", b"<html", b"<?xml")

# Skip content probes for large files; opening every mirror entry adds
# substantial I/O just to reject small proxy or login pages.
MAX_WEB_PAGE_BYTES = 1024 * 1024

#: Seconds to wait for each PDB-REDO download before giving the entry up.
DOWNLOAD_TIMEOUT_S = 30.0


def _looks_like_a_web_page(path: str, size: int | None = None) -> bool:
    """Whether a cached file holds a served document rather than entry data.

    ``size`` lets a caller that has already stat-ed the file skip the read for
    anything too large to be a page.
    """
    if size is not None and size > MAX_WEB_PAGE_BYTES:
        return False
    try:
        with open(path, "rb") as handle:
            head = handle.read(64).lstrip().lower()
    except OSError:
        return False
    return head.startswith(_HTML_PREFIXES)


def _is_usable_entry_file(path: str | None) -> bool:
    """Check that a cached file is usable before reusing it.

    Reject empty files and served documents so failed downloads can be retried.
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


def first_usable(*paths: str) -> str | None:
    """Return the first candidate holding entry data, or ``None`` when none does.

    A served page or empty file cached under an earlier candidate must not
    hide a usable later one, so every candidate is checked in turn.
    """
    return next((path for path in paths if _is_usable_entry_file(path)), None)


class PreparedInputs(NamedTuple):
    """The analysis inputs of one entry and the coordinates they came from.

    Mirror and manual inputs converge here so the worker need not know which
    files it was given or whether the analysis PDB was converted from mmCIF.
    """

    mtz: str
    pdb: str
    # The deposited coordinate file the analysis PDB was derived from.
    coordinates: str
    converted: bool

    @property
    def source_coordinate_format(self) -> str:
        """The deposited coordinate format."""
        return "mmcif" if self.converted else "pdb"


def prepare_inputs(pdb_id: str, entry_dir: str, work_dir: str) -> PreparedInputs:
    """Return the final PDB-REDO analysis inputs.

    The authoritative final mmCIF is preferred and converted for EDSTATS; the
    PDB compatibility export is used only when no mmCIF exists. Compressed
    mirrors are accepted for either format.
    """
    mtz_candidates = final_file_candidates(entry_dir, pdb_id, "mtz")
    mtz = first_usable(*mtz_candidates)
    if mtz is None:
        raise MissingInputError(mtz_candidates[0])
    if mtz.endswith(".gz"):
        mtz = _gunzip_to(mtz, os.path.join(work_dir, f"{pdb_id}_final.mtz"))

    coordinates = first_usable(*final_file_candidates(entry_dir, pdb_id, "coordinates"))
    if coordinates is None:
        raise MissingInputError(f"{pdb_id}_final.cif or {pdb_id}_final.pdb")
    converted = coordinates.endswith((".cif", ".cif.gz"))
    if converted:
        pdb = cif_to_pdb(
            coordinates, os.path.join(work_dir, f"{pdb_id}_final_from_cif.pdb")
        )
    elif coordinates.endswith(".gz"):
        pdb = _gunzip_to(coordinates, os.path.join(work_dir, f"{pdb_id}_final.pdb"))
    else:
        pdb = coordinates
    return PreparedInputs(mtz, pdb, coordinates, converted)


def prepare_data_json(entry_dir: str, work_dir: str) -> str | None:
    """Return an entry's readable ``data.json``, or ``None`` when it has none.

    A gzipped mirror copy is decompressed into ``work_dir`` so every reader
    can open the plain file.
    """
    plain = os.path.join(entry_dir, "data.json")
    if os.path.exists(plain):
        return plain
    compressed = plain + ".gz"
    if os.path.exists(compressed):
        return _gunzip_to(compressed, os.path.join(work_dir, "data.json"))
    return None


class EntryMetadata(NamedTuple):
    """What one ``data.json`` read supplies: the resolution limit and provenance."""

    # The diffraction data's own high-resolution limit.
    data_reshi: float
    pdb_redo: PdbRedoMetadata


def _data_json_properties(
    data_json_path: str | None, *, required: bool
) -> dict[str, Any] | None:
    """Read a data.json's properties once, or ``None`` when there is nothing to trust.

    When ``required`` is true, the path came from ``--data-json`` and a
    missing path or read or structural error is fatal instead.
    """
    if not data_json_path:
        if required:
            raise ValueError("an explicit data.json path is required")
        return None
    try:
        return read_data_json_properties(data_json_path)
    except ValueError:
        if required:
            raise
        return None


def _high_resolution_limit(
    properties: dict[str, Any] | None, mtz_path: str, *, required: bool
) -> float:
    """Return the overall diffraction-data high-resolution limit.

    Only the high-resolution limit is reported, because that is what the DPI
    metadata records; EDSTATS is given the map columns' own range by
    ``read_map_column_resolution`` instead. Without trusted metadata the MTZ
    is read.
    """
    if properties is not None:
        try:
            lo, hi = properties.get("DATARESL"), properties.get("DATARESH")
            # A half-populated or unphysical record is not trusted.
            if lo and hi:
                limit = float(hi)
                if math.isfinite(limit) and limit > 0.0:
                    return limit
        except (TypeError, ValueError):
            if required:
                raise
    return gemmi.read_mtz_file(mtz_path).resolution_high()


def pdb_redo_metadata_from(properties: dict[str, Any] | None) -> PdbRedoMetadata:
    """Extract the PDB-REDO fields that affect analysis or source provenance.

    Missing metadata and string-valued flags are false: the twin coefficient
    fallback must not be inferred from a filename or from an MTZFIX failure.
    """
    if properties is None:
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


def read_entry_metadata(
    mtz_path: str, data_json_path: str | None = None, *, required: bool = False
) -> EntryMetadata:
    """Read an entry's resolution limit and PDB-REDO provenance in one pass.

    Automatically discovered missing or malformed metadata falls back to the
    MTZ and to no provenance. When ``required`` is true, the path came from
    ``--data-json`` and read or structural errors are fatal instead.
    """
    properties = _data_json_properties(data_json_path, required=required)
    return EntryMetadata(
        data_reshi=_high_resolution_limit(properties, mtz_path, required=required),
        pdb_redo=pdb_redo_metadata_from(properties),
    )


def read_map_column_resolution(mtz_path: str) -> tuple[float, float]:
    """Return the common finite resolution range of both EDSTATS maps.

    EDSTATS receives maps calculated from FWT/PHWT and DELFWT/PHDELWT, so its
    limits must describe reflections for which all four values are present,
    rather than the overall range of unrelated columns in the MTZ.
    """
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
        usable &= np.isfinite(mtz_column_data(column).array)
    usable_d = d_values[usable]
    if usable_d.size == 0:
        raise ValueError(
            "MTZ map coefficient columns have no common finite reflections"
        )
    return float(usable_d.max()), float(usable_d.min())


def has_final_files(entry_dir: str, pdb_id: str) -> bool:
    """Whether an entry has final map coefficients and usable coordinates."""
    return (
        first_usable(*final_file_candidates(entry_dir, pdb_id, "mtz")) is not None
        and first_usable(*final_file_candidates(entry_dir, pdb_id, "coordinates"))
        is not None
    )


def _remove_partial_download(tmp: str) -> None:
    """Drop a ``.part`` file so no caller can mistake it for a complete one."""
    with contextlib.suppress(OSError):
        os.remove(tmp)


class _HttpResponse(Protocol):
    """The one header accessor ``download_stream`` needs from ``urlopen``."""

    def getheader(self, name: str) -> object: ...


def _response_content_length(response: _HttpResponse, url: str) -> int | None:
    """Return a validated HTTP Content-Length, or ``None`` when absent."""
    value = response.getheader("Content-Length")
    if value in (None, ""):
        return None
    text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    try:
        length = int(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MissingInputError(
            f"{url}: invalid Content-Length header {value!r}"
        ) from exc
    if length < 0:
        raise MissingInputError(f"{url}: invalid Content-Length header {value!r}")
    return length


def download_stream(url: str, dst: str, timeout: float = DOWNLOAD_TIMEOUT_S) -> str:
    """Download a URL to dst, raising MissingInputError if no usable file results.

    Handle both request failures and interrupted or truncated response bodies,
    including IncompleteRead, which is not an OSError.
    """
    try:
        response = urlopen(url, timeout=timeout)
    except HTTPError as e:
        raise MissingInputError(f"{url}: status {e.code}") from e
    except (OSError, ValueError) as e:
        raise MissingInputError(f"{url}: {e}") from e

    tmp = f"{dst}.{os.getpid()}.part"
    try:
        with response:
            status = response.getcode()
            if status != 200:
                raise MissingInputError(f"{url}: status {status}")
            expected_bytes = _response_content_length(response, url)
            received_bytes = 0
            with open(tmp, "wb") as fh:
                while chunk := response.read(8192):
                    fh.write(chunk)
                    received_bytes += len(chunk)
            if expected_bytes is not None and received_bytes != expected_bytes:
                raise MissingInputError(
                    f"{url}: incomplete response body: expected "
                    f"{expected_bytes} bytes, received {received_bytes}"
                )
        os.replace(tmp, dst)
    except MissingInputError:
        raise
    except (OSError, HTTPException) as e:
        raise MissingInputError(f"{url}: {type(e).__name__}: {e}") from e
    finally:
        # Once promoted the temporary name is gone; otherwise nothing may trust it.
        _remove_partial_download(tmp)
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
        except MissingInputError:
            return False

    def fetch_variant(name: str) -> bool:
        # Reject empty or served-document cache entries so downloads can be retried.
        if first_usable(os.path.join(entry, name), os.path.join(entry, name + ".gz")):
            return True
        return try_fetch(name) or try_fetch(name + ".gz")

    fetch_variant(f"{pdb_id}_final.mtz")
    if not fetch_variant(f"{pdb_id}_final.cif"):
        fetch_variant(f"{pdb_id}_final.pdb")
    if (
        first_existing(
            os.path.join(entry, "data.json"), os.path.join(entry, "data.json.gz")
        )
        is None
    ):
        try_fetch("data.json")

    if not has_final_files(entry, pdb_id):
        raise MissingInputError(f"PDB-REDO entry {pdb_id} is missing final model files")


class AvailableEntry(NamedTuple):
    """Where an entry's final files were found, and whether they had to be fetched."""

    root: str
    downloaded: bool


def ensure_entry_available(
    pdb_id: str, mirror_root: str | None, cache_root: str
) -> AvailableEntry:
    """Return the root containing the final model files: mirror, then cache.

    A cache miss triggers a download into ``cache_root``.
    """
    if mirror_root:
        mirror_entry = entry_dir_for(mirror_root, pdb_id)
        if os.path.isdir(mirror_entry) and has_final_files(mirror_entry, pdb_id):
            return AvailableEntry(mirror_root, downloaded=False)
    cache_entry = entry_dir_for(cache_root, pdb_id)
    if os.path.isdir(cache_entry) and has_final_files(cache_entry, pdb_id):
        return AvailableEntry(cache_root, downloaded=False)
    download_entry_to_cache(pdb_id, cache_root)
    if os.path.isdir(cache_entry) and has_final_files(cache_entry, pdb_id):
        return AvailableEntry(cache_root, downloaded=True)
    raise MissingInputError(pdb_id)


def resolve_manual_inputs(
    pdb_id: str,
    pdb_file: str | None = None,
    mtz_file: str | None = None,
    cif_file: str | None = None,
    work_dir: str | None = None,
) -> PreparedInputs:
    """Return the analysis inputs of a manually supplied local input set."""
    if not mtz_file:
        raise ValueError("manual mode requires --mtz-file")
    if not os.path.isfile(mtz_file):
        raise MissingInputError(f"mtz file not found: {mtz_file}")

    if cif_file:
        if not os.path.isfile(cif_file):
            raise MissingInputError(f"cif file not found: {cif_file}")
        target_pdb = os.path.join(work_dir or os.getcwd(), f"{pdb_id}.pdb")
        return PreparedInputs(
            mtz_file, cif_to_pdb(cif_file, target_pdb), cif_file, converted=True
        )

    if pdb_file:
        if not os.path.isfile(pdb_file):
            raise MissingInputError(f"pdb file not found: {pdb_file}")
        return PreparedInputs(mtz_file, pdb_file, pdb_file, converted=False)

    raise ValueError("manual mode requires --pdb-file or --cif-file")


def infer_pdb_id_from_path(path: str | None) -> str | None:
    """Infer a 4-char PDB id from a local file name if possible."""
    if not path:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(rf"({PDB_ID_PATTERN})(?:_.*)?$", stem)
    return m.group(1).lower() if m else None


def enumerate_entries(root: str, limit: int | None = None) -> list[str]:
    """Return sorted PDB IDs with final model files, stopping at limit if supplied."""
    ids: list[str] = []
    skipped = 0
    for hashdir in sorted(os.listdir(root)):
        hp = os.path.join(root, hashdir)
        if not os.path.isdir(hp):
            continue
        try:
            entries = sorted(os.listdir(hp))
        except OSError as e:
            # Common on a partially-synced mirror; one unreadable hashdir must
            # not abort the whole enumeration.
            skipped += 1
            logger.warning("skipping unreadable directory %s: %s", hp, e)
            continue
        for pid in entries:
            # Only the mirror layout is an entry: <root>/<id[1:3]>/<id>/.
            if not re.fullmatch(PDB_ID_PATTERN, pid) or pid[1:3] != hashdir:
                continue
            ep = os.path.join(hp, pid)
            if os.path.isdir(ep) and has_final_files(ep, pid):
                ids.append(pid)
                if limit is not None and len(ids) >= limit:
                    return ids
    if skipped:
        logger.warning("skipped %d unreadable hashdir(s) under %s", skipped, root)
    return ids
