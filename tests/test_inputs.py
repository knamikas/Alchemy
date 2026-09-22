"""Test mirror enumeration, input preparation, downloads, and MTZ readers."""

from __future__ import annotations

import builtins
import gzip
import http.client
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, cast

import gemmi
import pytest
from helpers import approx, write_mtz

import inputs

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _make_entry(
    root: Path,
    pdb_id: str,
    *,
    mtz: bool = True,
    cif: bool = True,
    pdb: bool = False,
    compressed: bool = False,
    data_json: str | None = None,
) -> str:
    """Create a mirror-layout entry directory with the requested files.

    ``cif`` and ``pdb`` are independent so a mirror carrying both can be built.
    """
    entry_dir = inputs.entry_dir_for(str(root), pdb_id)
    os.makedirs(entry_dir, exist_ok=True)

    def write(name: str, payload: bytes = b"x") -> None:
        path = os.path.join(entry_dir, name)
        if compressed:
            path += ".gz"
            payload = gzip.compress(payload)
        with open(path, "wb") as handle:
            handle.write(payload)

    if mtz:
        write(f"{pdb_id}_final.mtz")
    if cif:
        write(f"{pdb_id}_final.cif")
    if pdb:
        write(f"{pdb_id}_final.pdb", b"END\n")
    if data_json is not None:
        with open(os.path.join(entry_dir, "data.json"), "w", encoding="utf-8") as fh:
            fh.write(data_json)
    return entry_dir


def _data_json(entry_dir: str) -> str:
    return os.path.join(entry_dir, "data.json")


def _resolution(mtz: str, data_json: str | None) -> float:
    return inputs.read_entry_metadata(mtz, data_json).data_reshi


def _stub_cif_to_pdb(_cif: str, dst: str) -> str:
    """Return the conversion path for tests that only exercise input selection."""
    return dst


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_an_entry_is_final_only_with_both_inputs(
    tmp_path: Path, compressed: bool
) -> None:
    """Coordinates without map coefficients cannot be analyzed, and vice versa."""
    complete = _make_entry(tmp_path, "9myr", compressed=compressed)
    no_mtz = _make_entry(tmp_path, "6nlr", mtz=False, compressed=compressed)
    no_coords = _make_entry(
        tmp_path, "9nxl", cif=False, pdb=False, compressed=compressed
    )

    assert inputs.has_final_files(complete, "9myr")
    assert not inputs.has_final_files(no_mtz, "6nlr")
    assert not inputs.has_final_files(no_coords, "9nxl")


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_a_legacy_pdb_export_counts_as_usable_coordinates(
    tmp_path: Path, compressed: bool
) -> None:
    """Verify legacy PDB mirrors remain analyzable.

    ``.pdb`` and ``.pdb.gz`` are accepted when the authoritative mmCIF is
    absent, so a mirror carrying only the legacy export remains usable.
    """
    entry_dir = _make_entry(
        tmp_path, "9myr", cif=False, pdb=True, compressed=compressed
    )
    assert inputs.has_final_files(entry_dir, "9myr")


def test_final_file_candidates_rank_plain_over_gzipped_and_mmcif_over_pdb(
    tmp_path: Path,
) -> None:
    """The first existing candidate is the file an analysis uses.

    Every caller that probes for an entry's final files shares this order, so
    the mirror, the worker's provenance path, and the driver's memory estimate
    all agree on which file wins.
    """
    entry_dir = tmp_path / "mirror" / "my" / "9myr"
    assert inputs.final_file_candidates(str(entry_dir), "9myr", "mtz") == (
        str(entry_dir / "9myr_final.mtz"),
        str(entry_dir / "9myr_final.mtz.gz"),
    )
    assert inputs.final_file_candidates(str(entry_dir), "9myr", "coordinates") == (
        str(entry_dir / "9myr_final.cif"),
        str(entry_dir / "9myr_final.cif.gz"),
        str(entry_dir / "9myr_final.pdb"),
        str(entry_dir / "9myr_final.pdb.gz"),
    )


def test_enumeration_returns_only_complete_entries(tmp_path: Path) -> None:
    """Incomplete entries are skipped, and the order follows the mirror layout.

    Ordering is by hash directory then entry, not by PDB ID: ``9myr`` lives
    under ``my`` and ``6nlr`` under ``nl``, so ``9myr`` comes first despite
    sorting later as a string.
    """
    _make_entry(tmp_path, "9myr")  # hashdir "my"
    _make_entry(tmp_path, "6nlr")  # hashdir "nl"
    _make_entry(tmp_path, "9nxl", mtz=False)  # hashdir "nx", incomplete

    assert inputs.enumerate_entries(str(tmp_path)) == ["9myr", "6nlr"]


def test_enumeration_stops_at_the_requested_limit(tmp_path: Path) -> None:
    """``--max-pdbs`` must not walk the whole mirror to return three ids."""
    for pdb_id in ("1aaa", "1bbb", "2ccc", "2ddd"):
        _make_entry(tmp_path, pdb_id)

    limited = inputs.enumerate_entries(str(tmp_path), limit=2)
    assert len(limited) == 2
    assert limited == inputs.enumerate_entries(str(tmp_path))[:2]


def test_an_unreadable_hashdir_is_skipped_rather_than_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partially-synced mirror must not abort the whole enumeration."""
    _make_entry(tmp_path, "9myr")
    _make_entry(tmp_path, "6nlr")
    real_listdir = os.listdir

    def failing_listdir(path: str) -> list[str]:
        if str(path).endswith(os.path.join(str(tmp_path), "nl")):
            raise PermissionError("locked down")
        return real_listdir(path)

    monkeypatch.setattr("inputs.os.listdir", failing_listdir)
    assert inputs.enumerate_entries(str(tmp_path)) == ["9myr"]


def test_enumeration_trusts_only_the_mirror_layout(tmp_path: Path) -> None:
    """A directory outside ``<id[1:3]>/<id>`` is not an entry, whatever it holds."""
    _make_entry(tmp_path, "9myr")
    stray = tmp_path / "my" / "scratch"
    stray.mkdir()
    (stray / "scratch_final.mtz").write_bytes(b"x")
    (stray / "scratch_final.cif").write_bytes(b"x")
    misfiled = tmp_path / "zz" / "6nlr"
    misfiled.mkdir(parents=True)
    (misfiled / "6nlr_final.mtz").write_bytes(b"x")
    (misfiled / "6nlr_final.cif").write_bytes(b"x")

    assert inputs.enumerate_entries(str(tmp_path)) == ["9myr"]


def test_an_unusable_plain_file_does_not_hide_its_gzipped_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A served page cached under the plain name must not shadow the real data."""
    entry_dir = _make_entry(tmp_path, "9myr", mtz=False)
    with open(os.path.join(entry_dir, "9myr_final.mtz"), "wb") as handle:
        handle.write(b"<html>sign in</html>")
    with open(os.path.join(entry_dir, "9myr_final.mtz.gz"), "wb") as handle:
        handle.write(gzip.compress(b"MTZ real"))
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setattr(inputs, "cif_to_pdb", _stub_cif_to_pdb)

    assert inputs.has_final_files(entry_dir, "9myr")
    mtz = inputs.prepare_inputs("9myr", entry_dir, str(work_dir)).mtz
    with open(mtz, "rb") as handle:
        assert handle.read() == b"MTZ real"


def test_a_corrupt_archive_is_a_missing_input_and_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    """A truncated gzip raises ``EOFError``, which is not an ``OSError``."""
    entry_dir = _make_entry(tmp_path, "9myr", mtz=False)
    with open(os.path.join(entry_dir, "9myr_final.mtz.gz"), "wb") as handle:
        handle.write(b"\x1f\x8b\x08\x00truncated")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(inputs.MissingInputError, match="EOFError"):
        inputs.prepare_inputs("9myr", entry_dir, str(work_dir))
    assert list(work_dir.iterdir()) == []


def test_missing_map_coefficients_are_reported_by_path(tmp_path: Path) -> None:
    """The error must name the file that was looked for, not just fail."""
    entry_dir = _make_entry(tmp_path, "9myr", mtz=False)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(inputs.MissingInputError, match="9myr_final.mtz"):
        inputs.prepare_inputs("9myr", entry_dir, str(work_dir))


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_a_legacy_pdb_export_is_used_when_no_mmcif_exists(
    tmp_path: Path, compressed: bool
) -> None:
    """The fallback path returns the PDB directly, decompressing if needed."""
    entry_dir = _make_entry(
        tmp_path, "9myr", cif=False, pdb=True, compressed=compressed
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    prepared = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))
    pdb = prepared.pdb

    assert prepared.coordinates.startswith(entry_dir)
    assert not prepared.converted
    assert pdb.endswith(".pdb"), "a gzipped export must be decompressed first"
    assert os.path.isfile(pdb)
    with open(pdb, encoding="ascii") as handle:
        assert handle.read() == "END\n"
    if compressed:
        assert os.path.dirname(pdb) == str(work_dir), (
            "the mirror must not be written to"
        )


def test_the_authoritative_mmcif_wins_over_the_legacy_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify mmCIF wins when both coordinate formats exist.

    The mmCIF is converted because the legacy export loses identifiers that it
    retains.
    """
    entry_dir = _make_entry(tmp_path, "9myr", cif=True, pdb=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    converted: list[str] = []

    def fake_cif_to_pdb(cif_path: str, destination: str) -> str:
        converted.append(cif_path)
        with open(destination, "w", encoding="ascii") as handle:
            handle.write("FROM CIF\n")
        return destination

    monkeypatch.setattr(inputs, "cif_to_pdb", fake_cif_to_pdb)
    prepared = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))
    pdb = prepared.pdb

    assert converted and converted[0].endswith("_final.cif")
    assert prepared.coordinates == converted[0]
    with open(pdb, encoding="ascii") as handle:
        assert handle.read() == "FROM CIF\n", "the legacy export was used instead"


def test_an_entry_with_neither_coordinate_format_names_both(tmp_path: Path) -> None:
    entry_dir = _make_entry(tmp_path, "9myr", cif=False, pdb=False)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(
        inputs.MissingInputError, match=r"9myr_final\.cif or 9myr_final\.pdb"
    ):
        inputs.prepare_inputs("9myr", entry_dir, str(work_dir))


def test_a_compressed_mirror_is_decompressed_into_the_work_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compressed mirrors are accepted, and never modified in place."""
    entry_dir = _make_entry(tmp_path, "9myr", compressed=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    converted: list[tuple[str, str]] = []

    def fake_cif_to_pdb(cif_path: str, destination: str) -> str:
        converted.append((cif_path, destination))
        with open(destination, "w", encoding="ascii") as handle:
            handle.write("END\n")
        return destination

    monkeypatch.setattr(inputs, "cif_to_pdb", fake_cif_to_pdb)
    mtz, pdb, coordinates, was_converted = inputs.prepare_inputs(
        "9myr", entry_dir, str(work_dir)
    )

    assert was_converted
    assert coordinates.endswith("_final.cif.gz")
    assert os.path.dirname(mtz) == str(work_dir), "the mirror must not be written to"
    assert not mtz.endswith(".gz")
    assert os.path.isfile(pdb)
    assert converted, "the authoritative mmCIF should have been converted"


@pytest.mark.parametrize(
    "content",
    [b"", b"<!DOCTYPE html>\n<html><body>login</body></html>\n", b"  <html>\n"],
    ids=["empty", "captive-portal", "leading-space"],
)
def test_a_cached_body_that_is_not_entry_data_is_not_treated_as_cached(
    tmp_path: Path, content: bytes
) -> None:
    """A 200 response can still carry a login page under the entry's own name.

    Existence alone was the cache test, so such a body was reused forever: the
    entry failed identically on every resume with no way to recover short of
    deleting the cache by hand.
    """
    entry = tmp_path / "my" / "9myr"
    entry.mkdir(parents=True)
    (entry / "9myr_final.mtz").write_bytes(content)
    (entry / "9myr_final.cif").write_bytes(content)

    assert inputs.has_final_files(str(entry), "9myr") is False


def test_enumeration_does_not_open_files_too_large_to_be_a_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size alone must clear a real entry file, without reading it.

    Probing content costs one open per candidate, and each open pulls a
    readahead window off the filesystem to inspect 64 bytes. Over a ~195k-entry
    mirror that put about two hours in front of every run, including a
    ``--resume`` with only a handful of entries left to do.
    """
    entry = tmp_path / "my" / "9myr"
    entry.mkdir(parents=True)
    big = b"MTZ " + b"\0" * (inputs.MAX_WEB_PAGE_BYTES + 1)
    (entry / "9myr_final.mtz").write_bytes(big)
    (entry / "9myr_final.cif").write_bytes(big)

    opened: list[str] = []
    real_open = builtins.open

    def counting_open(path: Any, *args: Any, **kwargs: Any) -> IO[Any]:
        if str(path).startswith(str(entry)):
            opened.append(str(path))
        return cast(IO[Any], real_open(path, *args, **kwargs))

    monkeypatch.setattr(builtins, "open", counting_open)

    assert inputs.has_final_files(str(entry), "9myr") is True
    assert opened == [], f"enumeration should not read entry files, opened {opened}"


def test_a_page_sized_html_body_is_still_rejected(tmp_path: Path) -> None:
    """The size shortcut must not let a served notice through."""
    entry = tmp_path / "my" / "9myr"
    entry.mkdir(parents=True)
    page = b"<!DOCTYPE html>\n<html><body>proxy error</body></html>\n"
    assert len(page) <= inputs.MAX_WEB_PAGE_BYTES
    (entry / "9myr_final.mtz").write_bytes(page)
    (entry / "9myr_final.cif").write_bytes(page)

    assert inputs.has_final_files(str(entry), "9myr") is False


def test_real_entry_bytes_are_accepted_as_cached(tmp_path: Path) -> None:
    """The validity check must not reject an ordinary entry."""
    entry = tmp_path / "my" / "9myr"
    entry.mkdir(parents=True)
    (entry / "9myr_final.mtz").write_bytes(b"MTZ \x00\x01binary payload")
    (entry / "9myr_final.cif").write_text(
        "data_9MYR\n_entry.id 9MYR\n", encoding="ascii"
    )

    assert inputs.has_final_files(str(entry), "9myr") is True


class _FailingResponse:
    """A response that opens successfully and then fails partway through."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self._served = False

    def getcode(self) -> int:
        return 200

    def getheader(self, name: str) -> None:
        del name
        return None

    def read(self, _size: int) -> bytes:
        if self._served:
            raise self._error
        self._served = True
        return b"partial body"

    def __enter__(self) -> _FailingResponse:
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        return False


class _LengthResponse:
    """A cleanly ending response with an independently declared byte count."""

    def __init__(self, body: bytes, content_length: int) -> None:
        self._body = body
        self._content_length = content_length
        self._served = False

    def getcode(self) -> int:
        return 200

    def getheader(self, name: str) -> int | None:
        return self._content_length if name.lower() == "content-length" else None

    def read(self, _size: int) -> bytes:
        if self._served:
            return b""
        self._served = True
        return self._body

    def __enter__(self) -> _LengthResponse:
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        return False


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("connection reset by peer"),
        TimeoutError("read timed out"),
        http.client.IncompleteRead(b"partial body", 4096),
    ],
    ids=["reset", "timeout", "incomplete-read"],
)
def test_a_transfer_that_fails_midway_reports_no_usable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Only the opening request was guarded, so a mid-stream failure escaped.

    ``IncompleteRead`` is not even an ``OSError``, so a handler written for one
    would still have missed it. The caller's position is the same as a 404 --
    no file -- and the partial download must not be left behind.
    """

    def failing_response(_url: str, timeout: float | None = None) -> _FailingResponse:
        del timeout
        return _FailingResponse(error)

    monkeypatch.setattr(inputs, "urlopen", failing_response)
    destination = tmp_path / "9myr_final.mtz"

    with pytest.raises(inputs.MissingInputError) as excinfo:
        inputs.download_stream(
            "https://example.invalid/9myr_final.mtz", str(destination)
        )

    assert type(error).__name__ in str(excinfo.value)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == [], "a partial download was left behind"


def test_a_clean_early_eof_is_not_promoted_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTPResponse can return EOF without raising for a short response body."""
    body = b"nonempty but truncated"

    def short_response(_url: str, timeout: float | None = None) -> _LengthResponse:
        del timeout
        return _LengthResponse(body, len(body) + 4096)

    monkeypatch.setattr(inputs, "urlopen", short_response)
    destination = tmp_path / "9myr_final.mtz"

    with pytest.raises(inputs.MissingInputError, match=r"expected .* received"):
        inputs.download_stream(
            "https://example.invalid/9myr_final.mtz", str(destination)
        )

    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_a_missing_manual_file_is_reported_as_missing_input(tmp_path: Path) -> None:
    """A manual path that does not exist is a domain signal, not an OS error.

    The worker skips the entry on ``MissingInputError`` alone, so the manual
    readers must raise that class rather than a bare ``FileNotFoundError``.
    """
    mtz = tmp_path / "entry.mtz"
    mtz.write_bytes(b"MTZ ")

    with pytest.raises(inputs.MissingInputError, match="mtz file not found"):
        inputs.resolve_manual_inputs("9myr", mtz_file=str(tmp_path / "absent.mtz"))
    with pytest.raises(inputs.MissingInputError, match="pdb file not found"):
        inputs.resolve_manual_inputs(
            "9myr", mtz_file=str(mtz), pdb_file=str(tmp_path / "absent.pdb")
        )
    with pytest.raises(inputs.MissingInputError, match="cif file not found"):
        inputs.resolve_manual_inputs(
            "9myr", mtz_file=str(mtz), cif_file=str(tmp_path / "absent.cif")
        )
    # A directory is not an input file either.
    with pytest.raises(inputs.MissingInputError, match="pdb file not found"):
        inputs.resolve_manual_inputs("9myr", mtz_file=str(mtz), pdb_file=str(tmp_path))


def test_manual_inputs_report_their_source_and_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker learns the deposited file and format from the reader itself."""
    mtz = tmp_path / "entry.mtz"
    mtz.write_bytes(b"MTZ ")
    pdb = tmp_path / "entry.pdb"
    pdb.write_bytes(b"END\n")
    cif = tmp_path / "entry.cif"
    cif.write_bytes(b"data_entry\n")
    monkeypatch.setattr(inputs, "cif_to_pdb", _stub_cif_to_pdb)

    from_pdb = inputs.resolve_manual_inputs(
        "9myr", mtz_file=str(mtz), pdb_file=str(pdb)
    )
    from_cif = inputs.resolve_manual_inputs(
        "9myr", mtz_file=str(mtz), cif_file=str(cif), work_dir=str(tmp_path)
    )

    assert from_pdb == inputs.PreparedInputs(str(mtz), str(pdb), str(pdb), False)
    assert from_pdb.source_coordinate_format == "pdb"
    assert from_cif.coordinates == str(cif)
    assert from_cif.converted and from_cif.source_coordinate_format == "mmcif"
    assert from_cif.pdb == os.path.join(str(tmp_path), "9myr.pdb")


def test_a_body_matching_content_length_is_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"complete response"

    def complete_response(_url: str, timeout: float | None = None) -> _LengthResponse:
        del timeout
        return _LengthResponse(body, len(body))

    monkeypatch.setattr(inputs, "urlopen", complete_response)
    destination = tmp_path / "data.json"

    assert inputs.download_stream(
        "https://example.invalid/data.json", str(destination)
    ) == str(destination)
    assert destination.read_bytes() == body


def _minimal_mtz(path: Path, high: float = 1.5) -> str:
    """An MTZ whose only purpose is to carry a known resolution limit."""
    return write_mtz(
        path,
        {"F": "F"},
        [[1, 0, 0, 10.0], [0, 1, 0, 10.0]],
        cell=(high * 2, high * 2, high * 2, 90, 90, 90),
    )


def _map_coefficient_mtz(
    path: Path,
    rows: Sequence[Sequence[float]],
    *,
    columns: Sequence[str] = inputs.MAP_COEFFICIENT_COLUMNS,
) -> str:
    """An MTZ carrying the four EDSTATS map-coefficient columns.

    ``rows`` are ``(h, k, l, *values)`` with one value per column, so a caller
    can make an individual coefficient non-finite without touching the others.
    """
    return write_mtz(
        path,
        {
            label: "F" if label.startswith(("FWT", "DELFWT")) else "P"
            for label in columns
        },
        rows,
    )


def _d_spacing(mtz_path: str) -> NDArray[np.float32]:
    return gemmi.read_mtz_file(mtz_path).make_d_array()


def test_map_column_resolution_spans_only_wholly_finite_reflections(
    tmp_path: Path,
) -> None:
    """Verify map resolution uses reflections with all four finite coefficients.

    A highest-resolution row missing DELFWT must not extend the map limits.
    """
    nan = float("nan")
    path = _map_coefficient_mtz(
        tmp_path / "coefficients.mtz",
        [
            [1, 0, 0, 10.0, 20.0, 1.0, 30.0],
            [3, 0, 0, 10.0, 20.0, 1.0, 30.0],
            # Finite everywhere but DELFWT, and the highest-resolution row, so
            # a mask that did not span whole rows would extend reshi onto it.
            [4, 0, 0, 10.0, 20.0, nan, 30.0],
        ],
    )
    spacings = _d_spacing(path)

    reslo, reshi = inputs.read_map_column_resolution(path)

    assert (reslo, reshi) == approx(
        (float(max(spacings[0], spacings[1])), float(min(spacings[0], spacings[1])))
    )
    assert reshi > float(spacings[2])


def test_map_column_resolution_names_every_missing_column(tmp_path: Path) -> None:
    """The error says which coefficients are absent, not merely that one is."""
    path = _map_coefficient_mtz(
        tmp_path / "partial.mtz",
        [[1, 0, 0, 10.0, 20.0]],
        columns=("FWT", "PHWT"),
    )

    with pytest.raises(ValueError) as excinfo:
        inputs.read_map_column_resolution(path)

    message = str(excinfo.value)
    assert "DELFWT" in message and "PHDELWT" in message
    assert "FWT" in message


def test_map_column_resolution_rejects_a_wholly_unusable_set(tmp_path: Path) -> None:
    """No reflection finite in all four columns is an error, not an empty range."""
    nan = float("nan")
    path = _map_coefficient_mtz(
        tmp_path / "unusable.mtz",
        [
            [1, 0, 0, 10.0, 20.0, nan, 30.0],
            [2, 0, 0, nan, 20.0, 1.0, 30.0],
        ],
    )

    with pytest.raises(ValueError, match="no common finite reflections"):
        inputs.read_map_column_resolution(path)


def test_resolution_is_read_from_data_json_when_complete(tmp_path: Path) -> None:
    """data.json is authoritative because DPI metadata is derived from it."""
    entry_dir = _make_entry(
        tmp_path,
        "9myr",
        data_json=json.dumps({"properties": {"DATARESL": 47.1, "DATARESH": 1.72}}),
    )
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    assert _resolution(mtz, _data_json(entry_dir)) == approx(1.72)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"properties": {"DATARESL": 47.1}}', "no DATARESH"),
        ('{"properties": {"DATARESH": 1.72}}', "no DATARESL"),
        ('{"properties": {}}', "neither limit"),
        ("{not valid json", "malformed"),
        ("", "empty"),
    ],
)
def test_incomplete_metadata_falls_back_to_the_mtz(
    tmp_path: Path, payload: str, reason: str
) -> None:
    """Verify partial resolution metadata does not override the MTZ.

    Both limits are required before data.json is trusted; otherwise a partial
    metadata source would be mixed into DPI provenance.
    """
    entry_dir = _make_entry(tmp_path, "9myr", data_json=payload)
    mtz = _minimal_mtz(tmp_path / "fallback.mtz")

    resolution = _resolution(mtz, _data_json(entry_dir))
    assert resolution == approx(_resolution(mtz, None)), (
        f"{reason} should have fallen back to the MTZ"
    )


def test_absent_metadata_falls_back_to_the_mtz(tmp_path: Path) -> None:
    """A mirror without data.json is still analyzable, DPI aside."""
    entry_dir = _make_entry(tmp_path, "9myr")
    mtz = _minimal_mtz(tmp_path / "only.mtz")

    assert inputs.prepare_data_json(entry_dir, str(tmp_path)) is None
    assert _resolution(mtz, None) > 0.0


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_a_mirror_data_json_is_located_and_readable(
    tmp_path: Path, compressed: bool
) -> None:
    """A gzipped mirror copy is decompressed into the work directory, never in place."""
    payload = json.dumps({"properties": {"DATARESL": 47.1, "DATARESH": 1.72}})
    entry_dir = _make_entry(tmp_path, "9myr")
    name = "data.json.gz" if compressed else "data.json"
    with open(os.path.join(entry_dir, name), "wb") as handle:
        handle.write(
            gzip.compress(payload.encode()) if compressed else payload.encode()
        )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    data_json = inputs.prepare_data_json(entry_dir, str(work_dir))

    assert data_json is not None
    expected_dir = str(work_dir) if compressed else entry_dir
    assert os.path.dirname(data_json) == expected_dir
    assert _resolution(mtz, data_json) == approx(1.72)
    assert sorted(os.listdir(entry_dir)) == sorted(
        ["9myr_final.mtz", "9myr_final.cif", name]
    )


@pytest.mark.parametrize("limit", [-1.72, 0.0, "nan", "inf"])
def test_an_unphysical_resolution_limit_is_not_trusted(
    tmp_path: Path, limit: object
) -> None:
    """A record that parses but cannot be a resolution falls back to the MTZ."""
    entry_dir = _make_entry(
        tmp_path,
        "9myr",
        data_json=json.dumps({"properties": {"DATARESL": 47.1, "DATARESH": limit}}),
    )
    mtz = _minimal_mtz(tmp_path / "fallback.mtz")

    assert _resolution(mtz, _data_json(entry_dir)) == approx(_resolution(mtz, None))


def test_one_read_supplies_both_the_limit_and_the_provenance(
    tmp_path: Path,
) -> None:
    entry_dir = _make_entry(
        tmp_path,
        "9myr",
        data_json=json.dumps(
            {"properties": {"DATARESL": 47.1, "DATARESH": 1.72, "ISTWIN": True}}
        ),
    )
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    metadata = inputs.read_entry_metadata(mtz, _data_json(entry_dir))

    assert metadata.data_reshi == approx(1.72)
    assert metadata.pdb_redo.is_twin is True


def test_a_required_data_json_must_be_named(tmp_path: Path) -> None:
    """``--data-json`` semantics never apply to a path that was not given."""
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    with pytest.raises(ValueError, match="explicit data.json path is required"):
        inputs.read_entry_metadata(mtz, None, required=True)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "file not found"),
        ("{not valid json", "invalid JSON"),
        ("[]", "must contain a JSON object"),
        ("{}", "must contain a properties object"),
    ],
)
def test_explicit_invalid_data_json_does_not_fall_back_to_mtz(
    tmp_path: Path, payload: str | None, message: str
) -> None:
    """A requested metadata file is an input contract, not an optional probe."""
    data_json = tmp_path / "explicit.json"
    if payload is not None:
        data_json.write_text(payload, encoding="utf-8")
    mtz = _minimal_mtz(tmp_path / "fallback.mtz")

    with pytest.raises(ValueError, match=message):
        inputs.read_entry_metadata(mtz, str(data_json), required=True)
