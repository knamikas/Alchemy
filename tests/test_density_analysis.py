"""Guarded density-map coefficient normalization and twin routing."""

import json
from types import SimpleNamespace

import gemmi
import numpy as np
import pytest

import density_analysis as density
import main


def _write_refmac_mtz(
    path, *, title="Output mtz file from refmac", difference=(2.0, 2.0)
):
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


def test_refmac_twin_normalization_uses_edstats_centric_convention(tmp_path):
    source = tmp_path / "source.mtz"
    output = tmp_path / "normalized.mtz"
    _write_refmac_mtz(source)
    source_bytes = source.read_bytes()

    provenance = density.normalize_refmac_twin_coefficients(source, output)

    # The input remains byte-for-byte untouched; only the temporary output is
    # normalized. The centric main coefficient becomes mFo=12 and keeps D=2,
    # while the acentric main coefficient stays 14 and difference becomes 4.
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
    assert provenance["raw_identity_max_relative_residual"] == pytest.approx(0)
    assert provenance["output_identity_max_relative_residual"] == pytest.approx(0)
    assert any("Alchemy: normalized twin Refmac" in line for line in normalized.history)


def test_refmac_twin_normalization_rejects_inconsistent_coefficients(tmp_path):
    source = _write_refmac_mtz(tmp_path / "inconsistent.mtz", difference=(3.0, 2.0))
    output = tmp_path / "normalized.mtz"

    with pytest.raises(ValueError, match="guarded raw identity"):
        density.normalize_refmac_twin_coefficients(source, output)

    assert not output.exists()


def test_refmac_twin_normalization_requires_refmac_provenance(tmp_path):
    source = _write_refmac_mtz(
        tmp_path / "unknown.mtz", title="unidentified map coefficients"
    )

    with pytest.raises(ValueError, match="does not identify Refmac"):
        density.normalize_refmac_twin_coefficients(source, tmp_path / "normalized.mtz")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", False), (1, False), (None, False)],
)
def test_twin_routing_requires_explicit_boolean_metadata(tmp_path, value, expected):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"properties": {"ISTWIN": value}}), encoding="utf-8")
    assert main.read_pdb_redo_is_twin(path) is expected


def _fake_ccp4_run_factory(mtzfix_log_text):
    """Return a subprocess stub sufficient for the full-map density path."""

    def fake_run(
        cmd, input=None, text=None, stdout=None, stderr=None, env=None, timeout=None
    ):
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


def test_mtzfix_retest_failure_normalizes_only_explicit_twins(tmp_path, monkeypatch):
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr(density.shutil, "which", lambda command, path=None: command)
    monkeypatch.setattr(
        density.subprocess, "run", _fake_ccp4_run_factory("FAILED a test on re-take\n")
    )
    normalized_calls = []

    def fake_normalize(mtz_path, output_path):
        normalized_calls.append((mtz_path, output_path))
        with open(output_path, "wb") as handle:
            handle.write(b"normalized")
        return {"usable_reflections": 2, "centric_reflections": 1}

    monkeypatch.setattr(density, "normalize_refmac_twin_coefficients", fake_normalize)

    with pytest.raises(density.MtzfixValidationError):
        density.run_density_analysis(
            "test",
            source,
            pdb,
            tmp_path / "not_twin",
            20,
            2,
            map_scope="full",
            pdb_redo_is_twin=False,
        )
    assert normalized_calls == []

    result = density.run_density_analysis(
        "test",
        source,
        pdb,
        tmp_path / "twin",
        20,
        2,
        map_scope="full",
        pdb_redo_is_twin=True,
    )
    assert len(normalized_calls) == 1
    assert result["twin_coefficient_normalization_applied"] is True
    assert result["twin_coefficient_normalization"] == {
        "usable_reflections": 2,
        "centric_reflections": 1,
    }
    assert result["mtz_for_maps"].endswith("test_twin_edstats.mtz")


def test_generic_mtzfix_error_never_uses_twin_fallback(tmp_path, monkeypatch):
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr(density.shutil, "which", lambda command, path=None: command)
    monkeypatch.setattr(
        density.subprocess, "run", _fake_ccp4_run_factory("some unrelated failure\n")
    )
    monkeypatch.setattr(
        density,
        "normalize_refmac_twin_coefficients",
        lambda *_args: pytest.fail("unsafe twin fallback was attempted"),
    )

    with pytest.raises(RuntimeError, match="mtzfix failed"):
        density.run_density_analysis(
            "test",
            source,
            pdb,
            tmp_path / "out",
            20,
            2,
            map_scope="full",
            pdb_redo_is_twin=True,
        )


# --------------------------------------------------------------------------- #
# Timeouts
# --------------------------------------------------------------------------- #
def test_a_stalled_ccp4_program_is_killed_and_reported_with_its_partial_log(
    tmp_path, monkeypatch
):
    """A hung CCP4 step raises ``Ccp4ToolTimeout`` and keeps what it wrote.

    Without a budget a stalled ``edstats`` holds its worker slot for the rest of
    the run: the driver detects workers that have *died*, but nothing detects
    one that is merely stuck. The partial log is retained deliberately -- what
    the program printed before it stalled is the only evidence of where.
    """
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr(density.shutil, "which", lambda command, path=None: command)

    def fake_run(
        cmd, input=None, text=None, stdout=None, stderr=None, env=None, timeout=None
    ):
        if cmd[0].rsplit("/", 1)[-1] == "mtzfix":
            assert stdout is not None, "the runner must redirect stdout to a log"
            assert timeout is not None, "the runner must pass a budget"
            stdout.write("mtzfix progress before the stall\n")
            stdout.flush()
            raise density.subprocess.TimeoutExpired(cmd, timeout)
        raise AssertionError("no program should run after a timeout")

    monkeypatch.setattr(density.subprocess, "run", fake_run)

    with pytest.raises(density.Ccp4ToolTimeout) as excinfo:
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
    # The cost of the abandoned attempt is still recorded.
    assert error.timings.get("mtzfix_s") is not None

    with open(error.log_path, encoding="utf-8") as handle:
        assert "progress before the stall" in handle.read(), (
            "the partial log must survive; it is the only record of where the "
            "program stalled"
        )


def test_the_ccp4_budget_applies_to_each_program_not_to_the_entry(
    tmp_path, monkeypatch
):
    """Every CCP4 step gets the full budget rather than sharing one deadline.

    An entry runs four programs; charging them against a single deadline would
    make the last one fail for the sum of the others' legitimate work.
    """
    source = tmp_path / "source.mtz"
    pdb = tmp_path / "model.pdb"
    source.write_bytes(b"source")
    pdb.write_text("END\n", encoding="ascii")
    monkeypatch.setattr(density.shutil, "which", lambda command, path=None: command)

    seen = []

    def fake_run(
        cmd, input=None, text=None, stdout=None, stderr=None, env=None, timeout=None
    ):
        seen.append((cmd[0].rsplit("/", 1)[-1], timeout))
        program = cmd[0].rsplit("/", 1)[-1]
        if program == "fft":
            with open(cmd[cmd.index("MAPOUT") + 1], "wb") as handle:
                handle.write(b"map")
        elif program == "edstats":
            with open(cmd[cmd.index("OUT") + 1], "w", encoding="utf-8") as handle:
                handle.write("stats\n")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(density.subprocess, "run", fake_run)
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
    assert {timeout for _, timeout in seen} == {123}, (
        f"every program must receive the same per-program budget: {seen}"
    )
