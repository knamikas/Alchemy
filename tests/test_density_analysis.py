"""Guarded density-map coefficient normalization and twin routing."""

import json
import os
import struct
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, NoReturn, Protocol, cast

import gemmi
import numpy as np
import pytest

import density_analysis as density
import inputs

_Ccp4Runner = density._Ccp4Runner  # pyright: ignore[reportPrivateUsage]


class _ApproxFactory(Protocol):
    """The concrete numeric subset of pytest's broadly typed approx helper."""

    def __call__(
        self,
        expected: object,
        rel: float | None = None,
        abs: float | None = None,
        nan_ok: bool = False,
    ) -> object: ...


class _PytestApi(Protocol):
    approx: _ApproxFactory


approx = cast(_PytestApi, pytest).approx


def _found_command(command: str, path: str | None = None) -> str:
    """Return a typed successful ``shutil.which`` lookup for CCP4 stubs."""
    del path
    return command


def _write_refmac_mtz(
    path: Path,
    *,
    title: str = "Output mtz file from refmac",
    difference: tuple[float, float] = (2.0, 2.0),
) -> str:
    """Write two raw-Refmac reflections: one centric and one acentric."""
    mtz = gemmi.Mtz(with_base=True)
    mtz.title = title
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 21")
    mtz.cell = gemmi.UnitCell(40, 50, 60, 90, 100, 90)
    dataset = mtz.add_dataset("refmac")
    for label, column_type in density.REFMAC_TWIN_COLUMNS.items():
        mtz.add_column(label, column_type, dataset.id)

    # For both rows C=10, raw D=2 and raw A=14, hence A-C=2D. In P 21,
    # (1,0,0) is centric while (1,1,0) is acentric.
    rows = np.asarray(
        [
            [1, 0, 0, 20, 1, 10, 0, 14, 0, difference[0], 0, 0.8],
            [1, 1, 0, 20, 1, 10, 0, 14, 0, difference[1], 0, 0.8],
        ],
        dtype=np.float32,
    )
    mtz.set_data(rows)
    mtz.write_to_file(str(path))
    return str(path)


def test_refmac_twin_normalization_uses_edstats_centric_convention(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mtz"
    output = tmp_path / "normalized.mtz"
    _write_refmac_mtz(source)
    source_bytes = source.read_bytes()

    provenance = density.normalize_refmac_twin_coefficients(str(source), str(output))

    assert source.read_bytes() == source_bytes
    normalized = gemmi.read_mtz_file(str(output))
    np.testing.assert_allclose(
        normalized.column_with_label("FWT").array, [12, 14], atol=1e-5
    )
    np.testing.assert_allclose(
        normalized.column_with_label("DELFWT").array, [2, 4], atol=1e-5
    )
    np.testing.assert_allclose(
        normalized.column_with_label("PHWT").array, [0, 0], atol=1e-5
    )
    np.testing.assert_allclose(
        normalized.column_with_label("PHDELWT").array, [0, 0], atol=1e-5
    )
    assert provenance["usable_reflections"] == 2
    assert provenance["centric_reflections"] == 1
    assert provenance["raw_identity_max_relative_residual"] == approx(0)
    assert provenance["output_identity_max_relative_residual"] == approx(0)
    assert any("Alchemy: normalized twin Refmac" in line for line in normalized.history)


def test_refmac_twin_normalization_rejects_inconsistent_coefficients(
    tmp_path: Path,
) -> None:
    source = _write_refmac_mtz(tmp_path / "inconsistent.mtz", difference=(3.0, 2.0))
    output = tmp_path / "normalized.mtz"

    with pytest.raises(ValueError, match="guarded raw identity"):
        density.normalize_refmac_twin_coefficients(source, str(output))

    assert not output.exists()


def test_refmac_twin_normalization_requires_refmac_provenance(tmp_path: Path) -> None:
    source = _write_refmac_mtz(
        tmp_path / "unknown.mtz", title="unidentified map coefficients"
    )

    with pytest.raises(ValueError, match="does not identify Refmac"):
        density.normalize_refmac_twin_coefficients(
            source, str(tmp_path / "normalized.mtz")
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", False), (1, False), (None, False)],
)
def test_twin_routing_requires_explicit_boolean_metadata(
    tmp_path: Path, value: object, expected: bool
) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"properties": {"ISTWIN": value}}), encoding="utf-8")
    assert inputs.read_pdb_redo_is_twin(str(path)) is expected


def test_pdb_redo_metadata_includes_the_source_revision(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            {
                "properties": {
                    "ISTWIN": True,
                    "VERSION": 8.04,
                    "TIME": "2024-02-08",
                }
            }
        ),
        encoding="utf-8",
    )

    metadata = inputs.read_pdb_redo_metadata(str(path))

    assert metadata.is_twin is True
    assert metadata.version == "8.04"
    assert metadata.date == "2024-02-08"


@pytest.mark.parametrize("payload", [None, "{not valid json", "[]", "{}"])
def test_explicit_twin_metadata_read_failures_are_not_non_twin(
    tmp_path: Path, payload: str | None
) -> None:
    """Unreadable explicit metadata cannot be collapsed into ``ISTWIN=false``."""
    path = tmp_path / "data.json"
    if payload is not None:
        path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        inputs.read_pdb_redo_is_twin(str(path), required=True)


def _fake_ccp4_run_factory(mtzfix_log_text: str) -> Callable[..., SimpleNamespace]:
    """Return a subprocess stub sufficient for the full-map density path."""

    def fake_run(
        cmd: Sequence[str],
        input: str | None = None,
        text: bool | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[bytes] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SimpleNamespace:
        program = cmd[0].rsplit("/", 1)[-1]
        if program == "mtzfix":
            assert stdout is not None, "the runner must redirect stdout to a log"
            stdout.write(mtzfix_log_text)
            return SimpleNamespace(returncode=1, stderr="")
        if program == "fft":
            map_path = cmd[cmd.index("MAPOUT") + 1]
            with open(map_path, "wb") as handle:
                handle.write(b"map")
        elif program == "edstats":
            stats_path = cmd[cmd.index("OUT") + 1]
            with open(stats_path, "w", encoding="utf-8") as handle:
                handle.write("stats\n")
        return SimpleNamespace(returncode=0, stderr="")

    return fake_run


def _stub_ccp4_program(tmp_path: Path, name: str, script: str) -> Path:
    """Install a real executable standing in for one CCP4 program."""
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{script}\n", encoding="ascii")
    path.chmod(0o755)
    return path


def test_non_utf8_stderr_is_reported_rather_than_losing_the_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CCP4 program may write any byte to stderr; that must not raise.

    Decoding a pipe under ``text=True`` is strict, so one such byte used to
    raise ``UnicodeDecodeError`` before the exit status was read. The worker's
    catch-all then recorded a retryable ``unexpected_processing_error`` naming
    nothing, and every later ``--resume`` reprocessed the entry and failed
    identically.
    """

    def non_utf8_failure(
        *args: object, stderr: IO[bytes] | None = None, **kwargs: object
    ) -> SimpleNamespace:
        assert stderr is not None
        stderr.write(b"bad \xff\xfe byte")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("density_analysis.shutil.which", _found_command)
    monkeypatch.setattr("density_analysis.subprocess.run", non_utf8_failure)
    source = tmp_path / "source.mtz"
    source.write_bytes(b"source")
    pdb = tmp_path / "model.pdb"
    pdb.write_text("END\n", encoding="ascii")

    with pytest.raises(RuntimeError) as excinfo:
        density.run_density_analysis(
            "test",
            str(source),
            str(pdb),
            str(tmp_path / "out"),
            20,
            2,
            map_scope="full",
        )

    message = str(excinfo.value)
    assert "rc=1" in message
    # The undecodable bytes cost characters, not the entry.
    assert "bad" in message and "byte" in message
    assert not isinstance(excinfo.value, UnicodeDecodeError)


@pytest.mark.parametrize(
    "detail",
    [
        "mapmask: ccpmapin - Map section > maxsec: change maxsec and recompile",
        "FFTBIG: No reflexions pass acceptance criteria! Check RESOLUTION",
    ],
)
def test_known_ccp4_entry_limitations_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detail: str
) -> None:
    """Known input/build limitations must not look like transient machine errors."""

    def failed_program(
        *args: object, stderr: IO[bytes] | None = None, **kwargs: object
    ) -> SimpleNamespace:
        assert stderr is not None
        stderr.write(detail.encode("ascii"))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("density_analysis.shutil.which", _found_command)
    monkeypatch.setattr("density_analysis.subprocess.run", failed_program)
    runner = _Ccp4Runner("1abc", str(tmp_path), None, 900, {})

    with pytest.raises(density.Ccp4EntryLimitationError, match="rc=1"):
        runner.run(["mapmask"], None, "mapmask.log", "mapmask_s")


def test_an_unknown_ccp4_failure_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unfamiliar diagnostic must not silently become a terminal exclusion."""

    def failed_program(
        *args: object, stderr: IO[bytes] | None = None, **kwargs: object
    ) -> SimpleNamespace:
        assert stderr is not None
        stderr.write(b"temporary filesystem I/O failure")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("density_analysis.shutil.which", _found_command)
    monkeypatch.setattr("density_analysis.subprocess.run", failed_program)
    runner = _Ccp4Runner("1abc", str(tmp_path), None, 900, {})

    with pytest.raises(RuntimeError) as excinfo:
        runner.run(["mapmask"], None, "mapmask.log", "mapmask_s")
    assert not isinstance(excinfo.value, density.Ccp4EntryLimitationError)


@pytest.mark.skipif(
    os.name != "posix", reason="requires POSIX shell background-process semantics"
)
def test_a_helper_holding_stderr_does_not_fake_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget bounds the program, not the lifetime of an inherited pipe.

    With stderr on a pipe, ``communicate`` waits for EOF, so a program that
    forks a helper and exits at once still raised ``TimeoutExpired`` at the
    full budget and recorded a timeout that could never be reproduced.
    """
    program = _stub_ccp4_program(
        tmp_path, "mtzfix", "echo note >&2\n(sleep 30) &\nexit 1"
    )

    def find_program(_command: str, path: str | None = None) -> str:
        del path
        return str(program)

    monkeypatch.setattr("density_analysis.shutil.which", find_program)
    source = tmp_path / "source.mtz"
    source.write_bytes(b"source")
    pdb = tmp_path / "model.pdb"
    pdb.write_text("END\n", encoding="ascii")

    started = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        density.run_density_analysis(
            "test",
            str(source),
            str(pdb),
            str(tmp_path / "out"),
            20,
            2,
            map_scope="full",
            tool_timeout_s=10,
        )
    elapsed = time.monotonic() - started

    assert not isinstance(excinfo.value, density.Ccp4ToolTimeoutError)
    assert elapsed < 5, f"waited {elapsed:.1f}s on a program that exited at once"


def test_mtzfix_retest_failure_normalizes_only_explicit_twins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr("density_analysis.shutil.which", _found_command)
    monkeypatch.setattr(
        "density_analysis.subprocess.run",
        _fake_ccp4_run_factory("FAILED a test on re-take\n"),
    )
    normalized_calls: list[tuple[str, str]] = []

    def fake_normalize(mtz_path: str, output_path: str) -> dict[str, Any]:
        normalized_calls.append((mtz_path, output_path))
        with open(output_path, "wb") as handle:
            handle.write(b"normalized")
        return {"usable_reflections": 2, "centric_reflections": 1}

    monkeypatch.setattr(density, "normalize_refmac_twin_coefficients", fake_normalize)

    with pytest.raises(density.MtzfixValidationError):
        density.run_density_analysis(
            "test",
            str(source),
            str(pdb),
            str(tmp_path / "not_twin"),
            20,
            2,
            map_scope="full",
            pdb_redo_is_twin=False,
        )
    assert normalized_calls == []

    result = density.run_density_analysis(
        "test",
        str(source),
        str(pdb),
        str(tmp_path / "twin"),
        20,
        2,
        map_scope="full",
        pdb_redo_is_twin=True,
    )
    assert len(normalized_calls) == 1
    assert result.twin_coefficient_normalization_applied is True
    assert result.twin_coefficient_normalization == {
        "usable_reflections": 2,
        "centric_reflections": 1,
    }
    assert result.mtz_for_maps.endswith("test_twin_edstats.mtz")


def test_generic_mtzfix_error_never_uses_twin_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr("density_analysis.shutil.which", _found_command)
    monkeypatch.setattr(
        "density_analysis.subprocess.run",
        _fake_ccp4_run_factory("some unrelated failure\n"),
    )

    def fail_normalize(_mtz_path: str, _output_path: str) -> NoReturn:
        pytest.fail("unsafe twin fallback was attempted")

    monkeypatch.setattr(density, "normalize_refmac_twin_coefficients", fail_normalize)

    with pytest.raises(RuntimeError, match="mtzfix failed"):
        density.run_density_analysis(
            "test",
            str(source),
            str(pdb),
            str(tmp_path / "out"),
            20,
            2,
            map_scope="full",
            pdb_redo_is_twin=True,
        )


def test_a_stalled_ccp4_program_is_killed_and_reported_with_its_partial_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung CCP4 step raises ``Ccp4ToolTimeoutError`` and keeps what it wrote.

    The driver detects workers that have died, but nothing detects one that is
    merely stuck, and the partial log is the only evidence of where it stalled.
    """
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr("density_analysis.shutil.which", _found_command)

    def fake_run(
        cmd: Sequence[str],
        input: str | None = None,
        text: bool | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[bytes] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SimpleNamespace:
        if cmd[0].rsplit("/", 1)[-1] == "mtzfix":
            assert stdout is not None, "the runner must redirect stdout to a log"
            assert timeout is not None, "the runner must pass a budget"
            stdout.write("mtzfix progress before the stall\n")
            stdout.flush()
            raise subprocess.TimeoutExpired(cmd, timeout)
        raise AssertionError("no program should run after a timeout")

    monkeypatch.setattr("density_analysis.subprocess.run", fake_run)

    with pytest.raises(density.Ccp4ToolTimeoutError) as excinfo:
        density.run_density_analysis(
            "1abc",
            str(source),
            str(pdb),
            str(tmp_path / "out"),
            20.0,
            2.0,
            map_scope="full",
            pdb_redo_is_twin=False,
            tool_timeout_s=900,
        )

    error = excinfo.value
    assert error.tool == "mtzfix"
    assert error.timeout_s == 900
    assert "mtzfix" in str(error) and "900" in str(error)
    assert error.timings.get("mtzfix_s") is not None

    with open(error.log_path, encoding="utf-8") as handle:
        assert "progress before the stall" in handle.read(), (
            "the partial log must survive; it is the only record of where the "
            "program stalled"
        )


def test_the_ccp4_budget_applies_to_each_program_not_to_the_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every CCP4 step gets the full budget rather than sharing one deadline."""
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr("density_analysis.shutil.which", _found_command)

    seen: list[tuple[str, float | None, str | None]] = []

    def fake_run(
        cmd: Sequence[str],
        input: str | None = None,
        text: bool | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[bytes] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SimpleNamespace:
        program = cmd[0].rsplit("/", 1)[-1]
        seen.append((program, timeout, input))
        if program == "fft":
            with open(cmd[cmd.index("MAPOUT") + 1], "wb") as handle:
                handle.write(b"map")
        elif program == "edstats":
            with open(cmd[cmd.index("OUT") + 1], "w", encoding="utf-8") as handle:
                handle.write("stats\n")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("density_analysis.subprocess.run", fake_run)
    density.run_density_analysis(
        "1abc",
        str(source),
        str(pdb),
        str(tmp_path / "out"),
        20.0,
        2.0,
        map_scope="full",
        pdb_redo_is_twin=False,
        tool_timeout_s=123,
    )

    assert len(seen) >= 3, seen
    assert {timeout for _, timeout, _ in seen} == {123}, (
        f"every program must receive the same per-program budget: {seen}"
    )
    edstats_inputs = [stdin for program, _, stdin in seen if program == "edstats"]
    assert edstats_inputs == ["reslo=20.0,reshi=2.0,usealt=true\n"], (
        "EDSTATS must emit separate observations for the residue conformer "
        f"Alchemy selects: {edstats_inputs}"
    )


def _ccp4_map(
    size: int,
    *,
    starts: tuple[int, int, int] = (0, 0, 0),
    sampling: tuple[int, int, int] = (64, 64, 64),
    mode: int = 2,
) -> bytes:
    """A byte string that ``_map_extent_requires_full_map`` will parse.

    Only the first 1024 bytes are a real CCP4 header; the rest is padding to
    reach ``size``, which is all ``_map_size`` inspects.
    """
    words = [0] * 256
    words[0:3] = [10, 10, 10]  # NC, NR, NS
    words[3] = mode
    words[4:7] = list(starts)  # NCSTART, NRSTART, NSSTART
    words[7:10] = list(sampling)  # NX, NY, NZ
    words[16:19] = [1, 2, 3]  # MAPC, MAPR, MAPS
    header = struct.pack("<256i", *words)
    if size < len(header):
        raise ValueError("a CCP4 map cannot be smaller than its header")
    return header + b"\0" * (size - len(header))


def _envelope_run_factory(
    *,
    full_size: int,
    envelope_size: int,
    envelope_starts: tuple[int, int, int] = (0, 0, 0),
) -> Callable[..., SimpleNamespace]:
    """A stub CCP4 suite whose FFT and MAPMASK outputs have chosen extents."""

    def fake_run(
        cmd: Sequence[str],
        input: str | None = None,
        text: bool | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[bytes] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SimpleNamespace:
        program = cmd[0].rsplit("/", 1)[-1]
        if program == "fft":
            with open(cmd[cmd.index("MAPOUT") + 1], "wb") as handle:
                handle.write(_ccp4_map(full_size))
        elif program == "mapmask":
            with open(cmd[cmd.index("MAPOUT") + 1], "wb") as handle:
                handle.write(_ccp4_map(envelope_size, starts=envelope_starts))
        elif program == "edstats":
            with open(cmd[cmd.index("OUT") + 1], "w", encoding="utf-8") as handle:
                handle.write("stats\n")
        return SimpleNamespace(returncode=0, stderr="")

    return fake_run


def _run_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_run: Callable[..., SimpleNamespace],
    **kwargs: Any,
) -> density.DensityResult:
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr("density_analysis.shutil.which", _found_command)
    monkeypatch.setattr("density_analysis.subprocess.run", fake_run)
    return density.run_density_analysis(
        "1abc",
        str(source),
        str(pdb),
        str(tmp_path / "out"),
        20.0,
        2.0,
        map_scope="model-envelope",
        pdb_redo_is_twin=False,
        **kwargs,
    )


def test_a_usable_envelope_is_cropped_and_the_full_maps_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path: a smaller, readable crop is what EDSTATS receives."""
    result = _run_envelope(
        tmp_path,
        monkeypatch,
        _envelope_run_factory(full_size=8192, envelope_size=2048),
    )

    assert result.density_map_scope_used == "model-envelope"
    assert result.full_map_bytes == 8192 * 2
    assert result.edstats_map_bytes == 2048 * 2
    assert not os.path.exists(tmp_path / "out" / "1abc_fo_full.map"), (
        "the pre-crop map is redundant once cropping succeeded"
    )
    assert not os.path.exists(tmp_path / "out" / "1abc_df_full.map")


def test_keeping_intermediates_retains_the_pre_crop_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--keep-intermediates`` must preserve what cropping replaced."""
    _run_envelope(
        tmp_path,
        monkeypatch,
        _envelope_run_factory(full_size=8192, envelope_size=2048),
        keep_full_maps=True,
    )

    assert os.path.exists(tmp_path / "out" / "1abc_fo_full.map")
    assert os.path.exists(tmp_path / "out" / "1abc_df_full.map")


def test_an_envelope_larger_than_the_original_falls_back_to_the_full_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model spanning the cell can produce a crop bigger than its input.

    Cropping is then pointless: the original is kept and MAPMASK is skipped for
    the second map.
    """
    result = _run_envelope(
        tmp_path,
        monkeypatch,
        _envelope_run_factory(full_size=2048, envelope_size=4096),
    )

    assert result.density_map_scope_used == "full-size-fallback"
    assert result.edstats_map_bytes == result.full_map_bytes
    assert os.path.getsize(tmp_path / "out" / "1abc_fo.map") == 2048, (
        "the retained map must be the original, not the larger crop"
    )


def test_an_envelope_beyond_the_cell_edge_falls_back_to_the_full_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EDSTATS wraps lookups when a stored axis starts past the positive edge.

    A translated model makes MAPMASK emit exactly that: a crop that is smaller
    yet unusable, which size alone cannot detect.
    """
    result = _run_envelope(
        tmp_path,
        monkeypatch,
        _envelope_run_factory(
            full_size=8192,
            envelope_size=2048,
            # Exactly at the grid size: the first start the check rejects.
            envelope_starts=(64, 0, 0),
        ),
    )

    assert result.density_map_scope_used == "full-extent-fallback"
    assert result.edstats_map_bytes == result.full_map_bytes
    assert os.path.getsize(tmp_path / "out" / "1abc_fo.map") == 8192


def test_a_crop_starting_inside_the_cell_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last start the check accepts, one below the grid size.

    Paired with the test above, 63 accepted and 64 rejected pins the comparison
    at exactly the boundary.
    """
    result = _run_envelope(
        tmp_path,
        monkeypatch,
        _envelope_run_factory(
            full_size=8192, envelope_size=2048, envelope_starts=(63, 0, 0)
        ),
    )

    assert result.density_map_scope_used == "model-envelope"


def test_an_unreadable_map_header_is_reported_rather_than_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated or unrecognizable header must not be silently accepted."""

    def fake_run(
        cmd: Sequence[str],
        input: str | None = None,
        text: bool | None = None,
        stdout: IO[str] | None = None,
        stderr: IO[bytes] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SimpleNamespace:
        program = cmd[0].rsplit("/", 1)[-1]
        if program == "fft":
            with open(cmd[cmd.index("MAPOUT") + 1], "wb") as handle:
                handle.write(_ccp4_map(8192))
        elif program == "mapmask":
            with open(cmd[cmd.index("MAPOUT") + 1], "wb") as handle:
                handle.write(b"\0" * 512)  # shorter than a CCP4 header
        return SimpleNamespace(returncode=0, stderr="")

    with pytest.raises(RuntimeError, match="invalid CCP4 map header"):
        _run_envelope(tmp_path, monkeypatch, fake_run)
