"""Test argument validation and CCP4 configuration precedence without external tools."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

import pytest
from helpers import resolved_ccp4_environment

import cli
import confidence_score
import density_analysis as density
import scratch
from driver import ccp4_setup, environment, errors
from driver import confidence as driver_confidence
from driver.runlog import RunLog


def _option_help(option: str) -> str:
    """The rendered ``--help`` paragraph for one option."""
    captured = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(captured):
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


def test_help_does_not_claim_that_no_crystallization_download_defaults_to_true() -> (
    None
):
    """The same guard as for ``--no-bonds``, for the other ``store_false`` flag."""
    paragraph = _option_help("--no-crystallization-download")
    assert "do not fetch missing original-PDB crystallization" in paragraph
    assert "(default: True)" not in paragraph, paragraph
    assert "crystallization_download=True" in paragraph, paragraph


def test_help_omits_the_default_of_unset_options() -> None:
    """A ``None`` default is "unset", so the formatter must not print it."""
    assert "(default: None)" not in _option_help("--memory-limit")
    assert "(default: None)" not in _option_help("--log-dir")


def test_crystallization_metadata_download_can_be_disabled() -> None:
    default = cli.parse_args([])
    offline = cli.parse_args(
        [
            "--pdb-metadata-cache",
            "/tmp/alchemy-metadata",
            "--no-crystallization-download",
        ]
    )

    assert default.crystallization_download is True
    assert default.pdb_metadata_cache.endswith("pdb-metadata-cache")
    assert offline.crystallization_download is False
    assert offline.pdb_metadata_cache == "/tmp/alchemy-metadata"


def test_memory_controls_keep_detection_as_the_default() -> None:
    default = cli.parse_args([])
    configured = cli.parse_args(
        ["--memory-limit", "240G", "--memory-utilization", "0.9"]
    )

    assert default.memory_limit is None
    assert default.memory_utilization == 0.8
    assert configured.memory_limit == 240 * 1024**3
    assert configured.memory_utilization == 0.9


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("1024", 1024),
        ("512M", 512 * 1024**2),
        ("1.5GiB", int(1.5 * 1024**3)),
        ("2TB", 2 * 1024**4),
    ],
)
def test_memory_limit_accepts_readable_byte_sizes(spelling: str, expected: int) -> None:
    assert cli.memory_size_bytes(spelling) == expected


@pytest.mark.parametrize("value", ["0", "-1G", "lots", "1PB"])
def test_memory_limit_rejects_invalid_sizes(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.memory_size_bytes(value)


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "nan", "inf"])
def test_memory_utilization_rejects_values_outside_its_fraction(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.utilization_fraction(value)


@pytest.mark.parametrize("value", ["0", "-5"])
def test_max_pdbs_rejects_non_positive_caps(value: str) -> None:
    """``--max-pdbs`` is a count, so zero and negatives are rejected by name."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
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
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
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
    """Verify all invalid argument combinations exit with code 2."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as excinfo:
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
        driver_confidence, "DEFAULT_CONFIDENCE_REFERENCE_DIR", str(repository_reference)
    )

    selected, searched = driver_confidence.resolve_confidence_reference_dir(
        str(output_dir)
    )

    assert selected == str(output_reference)
    assert searched == (str(output_reference), str(repository_reference))


def test_explicit_confidence_reference_is_authoritative(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    automatic_reference = output_dir / "confidence_reference"
    automatic_reference.mkdir(parents=True)
    (automatic_reference / confidence_score.REFERENCE_METADATA_FILE).write_text("{}")
    explicit_reference = tmp_path / "explicit-reference"

    selected, searched = driver_confidence.resolve_confidence_reference_dir(
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
    assert written == str(primary)

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

    def resolve_env(path: str | None) -> dict[str, str]:
        return {"PATH": str(path)}

    def verify_ccp4(_env: Mapping[str, str]) -> None:
        return None

    monkeypatch.setattr(environment, "resolve_env", resolve_env)
    monkeypatch.setattr(environment, "verify_ccp4", verify_ccp4)

    configure = cli.parse_args(["--configure-ccp4", str(setup)])
    assert environment.configure_ccp4(configure) is True
    assert primary.exists(), "--configure-ccp4 wrote outside the configured list"
    assert environment.configure_ccp4(cli.parse_args([])) is False

    def tools_unavailable(_env: Mapping[str, str] | None = None) -> bool:
        return False

    monkeypatch.setattr(ccp4_setup, "ccp4_tools_available", tools_unavailable)
    monkeypatch.setattr(environment, "ccp4_tools_available", tools_unavailable)
    monkeypatch.delenv("CCP4_SETUP", raising=False)

    run = cli.parse_args([])
    env = environment.resolve_ccp4_environment(run)

    # The stubbed ``resolve_env`` records which setup script was sourced.
    assert env == {"PATH": str(setup)}


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

    args = cli.parse_args(["--ccp4-setup", "/nonexistent/ccp4.setup-sh"])
    with pytest.raises(errors.DriverError, match="not found"):
        environment.resolve_ccp4_environment(args)


@pytest.mark.skipif(sys.platform == "win32", reason="writes a POSIX sh setup script")
def test_explicit_ccp4_setup_overrides_the_installation_already_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify explicit CCP4 setup overrides the shell's existing installation."""
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

    args = cli.parse_args(["--ccp4-setup", str(setup)])
    env = environment.resolve_ccp4_environment(args)

    resolved = shutil.which("edstats", path=env.get("PATH"))
    assert resolved is not None
    assert str(requested) in resolved, (
        f"the ambient PATH installation won over the requested one: {resolved}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="writes a POSIX sh setup script")
def test_a_failing_posix_ccp4_setup_reports_its_own_error_output(
    tmp_path: Path,
) -> None:
    """The script's stderr is the only clue to why sourcing failed."""
    setup = tmp_path / "ccp4.setup-sh"
    setup.write_text('echo "libccp4 missing" >&2\nexit 1\n', encoding="utf-8")

    with pytest.raises(ccp4_setup.Ccp4SetupError, match="libccp4 missing"):
        ccp4_setup.resolve_env(str(setup))


def test_saving_the_ccp4_setup_replaces_a_config_file_of_the_wrong_shape(
    tmp_path: Path,
) -> None:
    """The loader tolerates a non-object file, so the saver must as well."""
    config = tmp_path / "ccp4.json"
    config.write_text("[1, 2, 3]\n", encoding="utf-8")

    ccp4_setup.save_ccp4_setup("/opt/ccp4/bin/ccp4.setup-sh", [str(config)])

    assert json.loads(config.read_text(encoding="utf-8")) == {
        "ccp4_setup": "/opt/ccp4/bin/ccp4.setup-sh"
    }


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

    def run_stub(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr("driver.ccp4_setup.subprocess.run", run_stub)

    env = ccp4_setup.resolve_env(str(setup))

    assert "ALCHEMY_TEST_INHERITED" not in env
    assert env["PATH"] == r"C:\CCP4\bin"


def test_the_three_timeout_budgets_are_distinct_and_ordered() -> None:
    """Each class of subprocess gets a budget matched to its own work."""
    assert (
        environment.PROVENANCE_COMMAND_TIMEOUT_S
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

    monkeypatch.setattr("driver.ccp4_setup.subprocess.run", fake_run)

    with pytest.raises(ccp4_setup.Ccp4SetupError) as excinfo:
        ccp4_setup.resolve_env(str(setup))

    message = str(excinfo.value)
    assert str(ccp4_setup.SETUP_SHELL_TIMEOUT_S) in message
    assert "stops the run" in message


def test_a_hanging_git_probe_costs_the_commit_hash_not_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify failed Git provenance probes report unknown without failing the run."""
    calls: list[float | None] = []

    def fake_run(cmd: Sequence[str], **kwargs: Any) -> NoReturn:
        timeout = kwargs.get("timeout")
        calls.append(timeout)
        assert timeout is not None, "provenance probes must be bounded"
        raise subprocess.TimeoutExpired(cmd, float(timeout))

    monkeypatch.setattr("driver.environment.subprocess.run", fake_run)

    assert environment.alchemy_commit() == "unknown"
    assert calls and set(calls) == {environment.PROVENANCE_COMMAND_TIMEOUT_S}


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


class TestPositiveInt:
    """``positive_int`` is the argparse gate for --workers/--max-pdbs/etc."""

    # Use Any to test non-string programmatic inputs outside the annotated CLI contract.
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", 1),
            ("2", 2),
            ("28", 28),
            ("  7  ", 7),  # int() tolerates surrounding whitespace
            ("+3", 3),
            ("1000000", 1000000),
            (5, 5),  # already an int (programmatic callers)
        ],
    )
    def test_accepts_integers_of_at_least_one(self, value: Any, expected: int) -> None:
        assert cli.positive_int(value) == expected
        assert isinstance(cli.positive_int(value), int)

    @pytest.mark.parametrize("value", ["0", "-1", "-28", 0, -3])
    def test_rejects_zero_and_negative(self, value: Any) -> None:
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            cli.positive_int(value)
        assert "at least 1" in str(excinfo.value)

    @pytest.mark.parametrize(
        "value",
        [
            "1.5",
            "2.0",
            "abc",
            "",
            " ",
            "1e3",
            "0x2",
            None,
            "nan",
            "inf",
            "1,000",
        ],
    )
    def test_rejects_non_integer_text(self, value: Any) -> None:
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            cli.positive_int(value)
        assert "positive integer" in str(excinfo.value)

    def test_boundary_is_one_not_zero(self) -> None:
        assert cli.positive_int("1") == 1
        with pytest.raises(argparse.ArgumentTypeError):
            cli.positive_int("0")


@pytest.mark.parametrize("value", ["9myr", "9MYR", "1abc", "0000"])
def test_pdb_ids_are_accepted_case_insensitively_and_normalized(value: str) -> None:
    """IDs are lowercased so cache paths and manifest keys cannot diverge."""
    assert cli.pdb_id(value) == value.lower()


@pytest.mark.parametrize("value", ["abc", "abcde", "ab-c", "ab c", "", "9my_"])
def test_malformed_pdb_ids_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="four alphanumeric"):
        cli.pdb_id(value)


def test_intermediates_are_discarded_unless_asked_for() -> None:
    """Verify intermediate retention is opt-in.

    Per-entry maps are large, so ``process`` keys its scratch cleanup off this
    flag.
    """
    assert cli.parse_args([]).keep_intermediates is False
    assert cli.parse_args(["--keep-intermediates"]).keep_intermediates is True


@pytest.mark.parametrize(
    "argv, fragment",
    [
        (["--pdb-file", "/tmp/1abc.pdb"], "requires --mtz-file"),
        (["--cif-file", "/tmp/1abc.cif"], "requires --mtz-file"),
        (["--mtz-file", "/tmp/1abc.mtz"], "requires --pdb-file or --cif-file"),
        (
            [
                "--pdb-file",
                "/tmp/1abc.pdb",
                "--cif-file",
                "/tmp/1abc.cif",
                "--mtz-file",
                "/tmp/1abc.mtz",
            ],
            "not both",
        ),
    ],
    ids=["pdb-alone", "cif-alone", "mtz-alone", "pdb-and-cif"],
)
def test_incomplete_manual_input_is_a_usage_error(
    argv: list[str], fragment: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Manual mode needs coordinates and reflections, and only one of each.

    These reached a worker before failing, and were reported as an unexpected
    processing error rather than as the usage mistake they are. Supplying both
    coordinate forms silently used the cif and ignored the pdb.
    """
    with pytest.raises(SystemExit):
        cli.parse_args(argv)

    assert fragment in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions required")
def test_unwritable_output_dir_exits_cleanly_naming_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A read-only destination exits like every other unusable input.

    CCP4 resolution is stubbed because it runs before the directory is created,
    and would otherwise fail on a machine with no CCP4 installed.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")

    monkeypatch.setattr(
        environment, "resolve_ccp4_environment", resolved_ccp4_environment
    )
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    output_dir = parent / "output"
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    try:
        exit_code = cli.main(
            [
                "--id-file",
                str(id_file),
                "--output-dir",
                str(output_dir),
                "--pdb-redo-root",
                str(tmp_path / "absent-mirror"),
                "--pdb-redo-cache",
                str(tmp_path / "cache"),
            ]
        )
    finally:
        parent.chmod(0o700)

    # 1 is the driver's code for a fixable failure; argparse owns 2.
    assert exit_code == 1
    message = capsys.readouterr().err
    assert str(output_dir) in message, message
    assert "Traceback" not in message


@pytest.mark.skipif(os.name != "posix", reason="uses a POSIX-only stub env")
def test_a_run_sweeps_leaked_scratch_before_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep is wired into the driver, not merely available to it.

    Driven through ``cli.main`` so that deleting the call site fails the test.
    The run itself fails for want of a mirror: sweeping happens at startup, so
    even a failing run must leave the directory clean.
    """
    monkeypatch.setattr(
        environment, "resolve_ccp4_environment", resolved_ccp4_environment
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    leaked: str | Path = scratch.create_owned_scratch_directory(
        str(output_dir), prefix=".alchemy-109m-", kind="entry"
    )
    leaked = output_dir / os.path.basename(leaked)
    (leaked / "2mFo-DFc.map").write_text("stale map bytes", encoding="utf-8")
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    cli.main(
        [
            "--id-file",
            str(id_file),
            "--output-dir",
            str(output_dir),
            "--pdb-redo-root",
            str(tmp_path / "absent-mirror"),
            "--pdb-redo-cache",
            str(tmp_path / "cache"),
        ]
    )

    leftovers = sorted(
        name for name in os.listdir(output_dir) if name.startswith(".alchemy-")
    )
    assert leftovers == [], f"the run left scratch behind: {leftovers}"


def _report_text(output_dir: Path) -> str:
    """The single run report ``main`` wrote under ``output_dir``."""
    reports = sorted((output_dir / "logs").glob("alchemy_run_*.log"))
    assert len(reports) == 1, reports
    return reports[0].read_text(encoding="utf-8")


def _main_argv(tmp_path: Path) -> list[str]:
    """Arguments for a ``main`` call whose run is stubbed out."""
    return ["--id", "109m", "--output-dir", str(tmp_path / "out")]


@pytest.mark.parametrize("delivery", ["ctrl-c", "sigterm"])
def test_an_interrupted_run_exits_130_and_still_writes_its_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    delivery: str,
) -> None:
    """Ctrl-C and SIGTERM both end as exit 130 with the interruption on record.

    The SIGTERM handler must also be gone again by the time ``main`` returns,
    whichever way the run ended.
    """
    before = signal.getsignal(signal.SIGTERM)

    def interrupted_run(*_args: object) -> NoReturn:
        if delivery == "sigterm":
            signal.raise_signal(signal.SIGTERM)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", interrupted_run)

    exit_code = cli.main(_main_argv(tmp_path))

    assert exit_code == 130
    assert signal.getsignal(signal.SIGTERM) is before
    assert "Interrupted: workers stopped" in capsys.readouterr().err
    report = _report_text(tmp_path / "out")
    assert "Exit code: 130" in report
    assert "Driver error: interrupted before completion" in report


def test_an_unexpected_exception_is_on_record_before_it_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash still leaves a report, as a failure, naming the exception."""
    before = signal.getsignal(signal.SIGTERM)

    def crashing_run(*_args: object) -> NoReturn:
        raise RuntimeError("the pool fell over")

    monkeypatch.setattr(cli, "run", crashing_run)

    with pytest.raises(RuntimeError, match="fell over"):
        cli.main(_main_argv(tmp_path))

    assert signal.getsignal(signal.SIGTERM) is before
    report = _report_text(tmp_path / "out")
    assert "Exit code: 1" in report
    assert "Driver error: RuntimeError: the pool fell over" in report


def _run_returning_zero(*_args: object) -> int:
    """Stand in for ``cli.run`` with a successful, instant run."""
    return 0


def test_a_completed_run_reports_its_own_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report carries the run's code and the report path is announced."""
    monkeypatch.setattr(cli, "run", _run_returning_zero)

    assert cli.main(_main_argv(tmp_path)) == 0

    assert "Exit code: 0" in _report_text(tmp_path / "out")
    assert "run report -> " in capsys.readouterr().err


def test_a_report_that_cannot_be_written_is_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Losing the report costs one error line, never the run's own exit code."""
    monkeypatch.setattr(cli, "run", _run_returning_zero)

    def unwritable(_self: RunLog, _exit_code: int) -> NoReturn:
        raise OSError("disk full")

    monkeypatch.setattr(RunLog, "write", unwritable)

    assert cli.main(_main_argv(tmp_path)) == 0

    error = capsys.readouterr().err
    assert "ERROR" in error and "could not write run report: disk full" in error
