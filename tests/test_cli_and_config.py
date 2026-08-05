"""Command-line contracts and CCP4 configuration precedence.

Everything here is offline: argparse rejects these argument combinations before
any CCP4 or network capability is probed, so no test needs a marker.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence

import pytest

import ccp4_setup
import cli
import density_analysis as density
from driver import pool
import confidence_score


def _option_help(option: str) -> str:
    """The rendered ``--help`` paragraph for one option."""
    captured = io.StringIO()
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(captured):
            cli.parse_args(["--help"])
    text = captured.getvalue()
    # The option name also appears in the usage banner, so the description block
    # is its last occurrence.
    start = text.rindex(option)
    end = text.find("\n  --", start)
    return text[start : end if end != -1 else len(text)]


def test_help_does_not_claim_that_no_bonds_defaults_to_true() -> None:
    """``--help`` must not read as "bonds are already skipped by default".

    ``ArgumentDefaultsHelpFormatter`` would append the default of the ``bonds``
    destination rather than of the flag that clears it.
    """
    paragraph = _option_help("--no-bonds")
    assert "skip the metal-ligand bond-distance stage" in paragraph
    assert "(default: True)" not in paragraph, paragraph
    assert "bonds=True" in paragraph, paragraph


@pytest.mark.parametrize("value", ["0", "-5"])
def test_max_pdbs_rejects_non_positive_caps(value: str) -> None:
    """``--max-pdbs`` is a count, so zero and negatives are rejected by name."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as excinfo:
            cli.parse_args(["--id", "109m", "--max-pdbs", value])

    # ``argparse.error`` puts the diagnostic on stderr and only the status on the
    # exception, while ``raise SystemExit(message)`` does the reverse; accept both.
    assert excinfo.value.code not in (0, None)
    message = f"{stderr.getvalue()}\n{excinfo.value}"
    assert "max-pdbs" in message, (
        f"--max-pdbs {value} was rejected, but not by name:\n" + message
    )


def test_negative_max_pdbs_does_not_silently_drop_entries_from_the_end() -> None:
    """A negative cap is rejected, not applied as a tail-trimming ``ids[:n]``."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as excinfo:
            cli.parse_args(["--id-file", "ids.txt", "--max-pdbs", "-3"])

    assert excinfo.value.code not in (0, None)
    message = f"{stderr.getvalue()}\n{excinfo.value}"
    assert "max-pdbs" in message, message
    assert "No entries to process" not in message


@pytest.mark.parametrize(
    "arguments,fragment",
    [
        (["--id", "109m", "--id-file", "ids.txt"], "either --id or --id-file"),
        (["--id", "109m", "--retry-partials"], "requires --resume"),
        (
            [
                "--id",
                "109m",
                "--pdb-file",
                "109m.pdb",
                "--mtz-file",
                "109m.mtz",
                "--resume",
                "--retry-partials",
            ],
            "manual structure inputs",
        ),
        (
            [
                "--id-file",
                "ids.txt",
                "--pdb-file",
                "109m.pdb",
                "--mtz-file",
                "109m.mtz",
            ],
            "--id-file cannot be combined with manual structure inputs",
        ),
        (
            ["--id", "109m", "--data-json", "data.json"],
            "--data-json requires manual structure inputs",
        ),
    ],
)
def test_an_unusable_combination_of_arguments_exits_two(
    arguments: list[str], fragment: str
) -> None:
    """Every unusable argument combination exits 2, whichever check caught it.

    A script branching on the status must not have to know which one did.
    """
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as excinfo:
            cli.parse_args(arguments)
    assert excinfo.value.code == 2
    assert fragment in stderr.getvalue()


@pytest.mark.parametrize(
    "selector",
    [
        [],
        ["--id", "109m"],
        ["--id-file", "ids.txt"],
    ],
)
def test_retry_partials_accepts_optional_resume_selectors(selector: list[str]) -> None:
    args = cli.parse_args([*selector, "--resume", "--retry-partials"])
    assert args.resume is True
    assert args.retry_partials is True


@pytest.mark.parametrize(
    "selector",
    [[], ["--id", "109m"], ["--id-file", "ids.txt"]],
    ids=["database", "single", "id-file"],
)
def test_data_json_is_rejected_outside_manual_mode(selector: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.parse_args([*selector, "--data-json", "data.json"])

    assert excinfo.value.code == 2


def test_manual_mode_accepts_an_optional_single_id() -> None:
    args = cli.parse_args(
        [
            "--id",
            "109m",
            "--pdb-file",
            "model.pdb",
            "--mtz-file",
            "data.mtz",
            "--data-json",
            "data.json",
        ]
    )

    assert args.id == "109m"
    assert args.id_file is None


def test_confidence_reference_is_discovered_in_output_before_repo_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    output_reference = output_dir / "confidence_reference"
    output_reference.mkdir(parents=True)
    (output_reference / confidence_score.REFERENCE_METADATA_FILE).write_text("{}")
    repository_reference = tmp_path / "repository-reference"
    repository_reference.mkdir()
    (repository_reference / confidence_score.REFERENCE_METADATA_FILE).write_text("{}")
    monkeypatch.setattr(
        pool, "DEFAULT_CONFIDENCE_REFERENCE_DIR", str(repository_reference)
    )

    selected, searched = pool.resolve_confidence_reference_dir(str(output_dir))

    assert selected == str(output_reference)
    assert searched == (str(output_reference), str(repository_reference))


def test_explicit_confidence_reference_is_authoritative(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    automatic_reference = output_dir / "confidence_reference"
    automatic_reference.mkdir(parents=True)
    (automatic_reference / confidence_score.REFERENCE_METADATA_FILE).write_text("{}")
    explicit_reference = tmp_path / "explicit-reference"

    selected, searched = pool.resolve_confidence_reference_dir(
        str(output_dir), str(explicit_reference)
    )

    assert selected is None
    assert searched == (str(explicit_reference),)


def test_saved_ccp4_setup_is_the_one_that_gets_loaded_back(tmp_path: Path) -> None:
    """Save and load agree that ``config_files[0]`` is authoritative.

    ``save_ccp4_setup`` writes only that file, so a later-file win would leave
    ``--configure-ccp4`` reporting success and the next run using the stale path.
    """
    primary = tmp_path / "user" / "ccp4.json"
    shadowing = tmp_path / "repo" / "ccp4.json"
    shadowing.parent.mkdir(parents=True)
    shadowing.write_text(
        '{"ccp4_setup": "/stale/repo/ccp4.setup-sh"}\n', encoding="utf-8"
    )
    config_files = [str(primary), str(shadowing)]

    chosen = "/the/path/the/user/configured/ccp4.setup-sh"
    written = ccp4_setup.save_ccp4_setup(chosen, config_files=config_files)
    assert written == [str(primary)]

    loaded = ccp4_setup.load_ccp4_setup_config(config_files=config_files)
    assert loaded.get("ccp4_setup") == chosen


def test_later_config_files_still_supply_keys_the_primary_omits(tmp_path: Path) -> None:
    """First-wins precedence shadows only the keys the primary file defines."""
    primary = tmp_path / "user" / "ccp4.json"
    primary.parent.mkdir(parents=True)
    primary.write_text('{"ccp4_setup": "/user/ccp4.setup-sh"}\n', encoding="utf-8")
    secondary = tmp_path / "repo" / "ccp4.json"
    secondary.parent.mkdir(parents=True)
    secondary.write_text(
        '{"ccp4_setup": "/repo/ccp4.setup-sh", "other_key": "kept"}\n', encoding="utf-8"
    )

    loaded = ccp4_setup.load_ccp4_setup_config(
        config_files=[str(primary), str(secondary)]
    )

    assert loaded["ccp4_setup"] == "/user/ccp4.setup-sh"
    assert loaded["other_key"] == "kept"


def test_the_driver_reads_the_setup_path_configuration_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--configure-ccp4`` and the next run must use the same file list.

    Redirecting the single ``DEFAULT_CONFIG_FILES`` is what proves it: a driver
    carrying a list of its own would save where the run never reads.
    """
    primary = tmp_path / "user" / "ccp4.json"
    monkeypatch.setattr(ccp4_setup, "DEFAULT_CONFIG_FILES", [str(primary)])

    setup = tmp_path / "ccp4.setup-sh"
    setup.write_text("# a setup script that changes nothing\n", encoding="utf-8")

    monkeypatch.setattr(pool, "resolve_env", lambda path: {"PATH": str(path)})
    monkeypatch.setattr(pool, "verify_ccp4", lambda env: None)

    configure = argparse.Namespace(configure_ccp4=str(setup), ccp4_setup=None)
    assert pool.resolve_ccp4_environment(configure) == (None, None)
    assert primary.exists(), "--configure-ccp4 wrote outside the configured list"

    monkeypatch.setattr(ccp4_setup, "ccp4_tools_available", lambda env=None: False)
    monkeypatch.setattr(pool, "ccp4_tools_available", lambda env=None: False)
    monkeypatch.delenv("CCP4_SETUP", raising=False)

    run = argparse.Namespace(configure_ccp4=None, ccp4_setup=None)
    _, used = pool.resolve_ccp4_environment(run)

    assert used == str(setup)


def _stub_ccp4_dir(root: Path, marker: str) -> Path:
    """A directory holding the four CCP4 tool names, each echoing ``marker``."""
    bindir = root / f"ccp4-{marker}"
    bindir.mkdir(parents=True)
    for tool in ccp4_setup.REQUIRED_CCP4_TOOLS:
        script = bindir / tool
        script.write_text(f"#!/bin/sh\necho {marker}\n", encoding="utf-8")
        script.chmod(0o755)
        if sys.platform == "win32":  # pragma: no cover - POSIX dev host
            (bindir / f"{tool}.bat").write_text(
                f"@echo off\r\necho {marker}\r\n", encoding="utf-8"
            )
    return bindir


def test_nonexistent_ccp4_setup_is_an_error_even_with_ccp4_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A typo'd ``--ccp4-setup`` must fail, not fall through to ``PATH``."""
    on_path = _stub_ccp4_dir(tmp_path, "ambient")
    monkeypatch.setenv("PATH", str(on_path))
    assert ccp4_setup.ccp4_tools_available(os.environ), (
        "the stub must satisfy the PATH probe, or this test proves nothing"
    )

    args = argparse.Namespace(
        configure_ccp4=None, ccp4_setup="/nonexistent/ccp4.setup-sh"
    )
    with pytest.raises(pool.DriverError, match="not found"):
        pool.resolve_ccp4_environment(args)


@pytest.mark.skipif(sys.platform == "win32", reason="writes a POSIX sh setup script")
def test_explicit_ccp4_setup_overrides_the_installation_already_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The requested installation is used, not the one the shell had sourced.

    Honouring ``PATH`` first would run the wrong binaries and record their
    version as the run's provenance, so the output looks internally consistent.
    """
    ambient = _stub_ccp4_dir(tmp_path, "ambient")
    requested = _stub_ccp4_dir(tmp_path, "requested")
    # Sourcing the setup script runs through ``bash``, so the system directories
    # must stay on PATH behind the stub.
    monkeypatch.setenv("PATH", f"{ambient}{os.pathsep}{os.defpath}")
    assert ccp4_setup.ccp4_tools_available(os.environ), (
        "the ambient stub must satisfy the PATH probe, or the override this "
        "test checks would never have been bypassed in the first place"
    )

    setup = tmp_path / "ccp4.setup-sh"
    setup.write_text(f'export PATH="{requested}:$PATH"\n', encoding="utf-8")

    args = argparse.Namespace(configure_ccp4=None, ccp4_setup=str(setup))
    env, used = pool.resolve_ccp4_environment(args)

    assert env is not None
    assert used == str(setup)
    resolved = shutil.which("edstats", path=env.get("PATH"))
    assert resolved is not None
    assert str(requested) in resolved, (
        f"the ambient PATH installation won over the requested one: {resolved}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="writes a POSIX sh setup script")
def test_posix_ccp4_setup_can_remove_an_inherited_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = tmp_path / "ccp4.setup-sh"
    setup.write_text("unset ALCHEMY_TEST_INHERITED\n", encoding="utf-8")
    monkeypatch.setenv("ALCHEMY_TEST_INHERITED", "must disappear")

    env = ccp4_setup.resolve_env(str(setup))

    assert "ALCHEMY_TEST_INHERITED" not in env


def test_windows_ccp4_setup_output_is_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = tmp_path / "ccp4.setup.cmd"
    setup.write_text("@set ALCHEMY_TEST_INHERITED=\r\n", encoding="utf-8")
    monkeypatch.setenv("ALCHEMY_TEST_INHERITED", "must disappear")

    completed = subprocess.CompletedProcess(
        args=["cmd"],
        returncode=0,
        stdout=f"launcher banner\n{ccp4_setup.ENV_SENTINEL}\nPath=C:\\CCP4\\bin\n",
        stderr="",
    )
    monkeypatch.setattr("ccp4_setup.subprocess.run", lambda *a, **k: completed)

    env = ccp4_setup.resolve_env(str(setup))

    assert "ALCHEMY_TEST_INHERITED" not in env
    assert env["PATH"] == r"C:\CCP4\bin"


def test_the_three_timeout_budgets_are_distinct_and_ordered() -> None:
    """Each class of subprocess gets a budget matched to its own work."""
    assert (
        pool.PROVENANCE_COMMAND_TIMEOUT_S
        < ccp4_setup.SETUP_SHELL_TIMEOUT_S
        < density.CCP4_TOOL_TIMEOUT_S
    )
    # The July 2026 database runs peaked at EDSTATS 185.7 s and FFT 54.2 s.
    assert density.CCP4_TOOL_TIMEOUT_S >= 4 * 186


@pytest.mark.parametrize(
    ("setup_name", "shell"), [("ccp4.setup-sh", "bash"), ("ccp4.setup.bat", "cmd")]
)
def test_a_hanging_setup_script_aborts_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup_name: str, shell: str
) -> None:
    """A setup script that blocks stops the run rather than failing one entry."""
    setup = tmp_path / setup_name
    setup.write_text("sleep forever\n", encoding="utf-8")

    def fake_run(cmd: Sequence[str], **kwargs: Any) -> NoReturn:
        timeout = kwargs.get("timeout")
        assert timeout == ccp4_setup.SETUP_SHELL_TIMEOUT_S
        raise subprocess.TimeoutExpired(cmd, float(timeout))

    monkeypatch.setattr("ccp4_setup.subprocess.run", fake_run)

    with pytest.raises(ccp4_setup.Ccp4SetupError) as excinfo:
        ccp4_setup.resolve_env(str(setup))

    message = str(excinfo.value)
    assert str(ccp4_setup.SETUP_SHELL_TIMEOUT_S) in message
    assert "stops the run" in message


def test_a_hanging_git_probe_costs_the_commit_hash_not_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance degrades to ``unknown`` instead of failing anything.

    ``git`` only stamps the run log, so a stuck index lock must not take the
    analysis with it.
    """
    calls: list[Optional[float]] = []

    def fake_run(cmd: Sequence[str], **kwargs: Any) -> NoReturn:
        timeout = kwargs.get("timeout")
        calls.append(timeout)
        assert timeout is not None, "provenance probes must be bounded"
        raise subprocess.TimeoutExpired(cmd, float(timeout))

    monkeypatch.setattr("driver.pool.subprocess.run", fake_run)

    assert pool._alchemy_commit() == "unknown"
    assert calls and set(calls) == {pool.PROVENANCE_COMMAND_TIMEOUT_S}


def test_ccp4_timeout_accepts_a_custom_budget_and_rejects_nonsense() -> None:
    """``--ccp4-timeout`` is settable, and its default is the module constant."""
    assert cli.parse_args([]).ccp4_timeout == density.CCP4_TOOL_TIMEOUT_S
    assert cli.parse_args(["--ccp4-timeout", "3600"]).ccp4_timeout == 3600

    for bad in ("0", "-1", "not-a-number"):
        with pytest.raises(SystemExit):
            cli.parse_args(["--ccp4-timeout", bad])


def test_ccp4_timeout_help_states_its_default() -> None:
    """A reader must see the budget without reading the source."""
    help_text = _option_help("--ccp4-timeout")
    assert f"default: {density.CCP4_TOOL_TIMEOUT_S}" in help_text
