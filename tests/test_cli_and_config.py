"""Command-line contracts and CCP4 configuration precedence.

These are the promises a user reads off ``--help`` or infers from
``--configure-ccp4``, as distinct from what the pipeline computes. They are
cheap, offline, and none of them needs CCP4 or the network: the argument
failures here are raised by ``argparse`` before any capability is probed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys

import pytest

import ccp4_setup
import main


def _option_help(option: str) -> str:
    """The rendered ``--help`` paragraph for one option."""
    captured = io.StringIO()
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(captured):
            main.parse_args(["--help"])
    text = captured.getvalue()
    # The option name also appears in the usage banner; the description block is
    # the last occurrence.
    start = text.rindex(option)
    end = text.find("\n  --", start)
    return text[start : end if end != -1 else len(text)]


# --------------------------------------------------------------------------- #
# --help must describe the flag, not the dest it clears
# --------------------------------------------------------------------------- #
def test_help_does_not_claim_that_no_bonds_defaults_to_true():
    """``--help`` must not tell the user that ``--no-bonds`` defaults to True.

    Bond analysis is on by default and ``--no-bonds`` turns it off.

    Regression: ``ArgumentDefaultsHelpFormatter`` appends the default of the
    ``bonds`` destination rather than of the flag, so the paragraph rendered
    "(default: True)" -- which a user can only read as "bonds are already
    skipped by default", the opposite of the truth.
    """
    paragraph = _option_help("--no-bonds")
    assert "skip the metal-ligand bond-distance stage" in paragraph
    assert "(default: True)" not in paragraph, paragraph
    # The real default is still stated, attached to the setting it belongs to.
    assert "bonds=True" in paragraph, paragraph


# --------------------------------------------------------------------------- #
# --max-pdbs is a count, so it must be positive
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["0", "-5"])
def test_max_pdbs_rejects_non_positive_caps(value):
    """``--max-pdbs`` is documented as "process only the first N entries".

    Zero and negative values have no meaning under that description.
    ``--workers`` already rejects them through ``positive_int``; ``--max-pdbs``
    is validated the same way.

    The rejection has to be *about* ``--max-pdbs``: a bare
    ``pytest.raises(SystemExit)`` would also be satisfied by an exit provoked by
    something else in the argument list.
    """
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as excinfo:
            main.parse_args(["--id", "109m", "--max-pdbs", value])

    # ``argparse.error`` exits 2 with the diagnostic on stderr and nothing but
    # the code on the exception (``str(SystemExit(2)) == "2"``, which is why
    # ``pytest.raises(match=...)`` cannot be used here); an explicit
    # ``raise SystemExit(message)`` carries it the other way round. Accept both.
    assert excinfo.value.code not in (0, None)
    message = f"{stderr.getvalue()}\n{excinfo.value}"
    assert "max-pdbs" in message, (
        f"--max-pdbs {value} was rejected, but not by name:\n" + message
    )


def test_negative_max_pdbs_does_not_silently_drop_entries_from_the_end():
    """A negative cap must not quietly discard the tail of the id list.

    Regression: the cap was applied as ``ids[:args.max_pdbs]``, and a negative
    slice trims from the *end*. ``--max-pdbs -3`` over three ids yielded an
    empty list, so the driver reported "No entries to process" and exited 0 --
    a run that silently did nothing while looking like a success.

    Rejection now happens in ``parse_args``, before any entry is enumerated,
    so no mirror or CCP4 access is involved.
    """
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as excinfo:
            main.parse_args(["--id-file", "ids.txt", "--max-pdbs", "-3"])

    assert excinfo.value.code not in (0, None)
    message = f"{stderr.getvalue()}\n{excinfo.value}"
    assert "max-pdbs" in message, message
    assert "No entries to process" not in message


# --------------------------------------------------------------------------- #
# Terminal-partial retries must be resume-safe
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arguments,fragment",
    [
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
    ],
)
def test_retry_partials_rejects_unsafe_invocations(arguments, fragment):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as excinfo:
            main.parse_args(arguments)
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
def test_retry_partials_accepts_optional_resume_selectors(selector):
    args = main.parse_args([*selector, "--resume", "--retry-partials"])
    assert args.resume is True
    assert args.retry_partials is True


# --------------------------------------------------------------------------- #
# Confidence references are discovered beside the current output
# --------------------------------------------------------------------------- #
def test_confidence_reference_is_discovered_in_output_before_repo_default(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "output"
    output_reference = output_dir / "confidence_reference"
    output_reference.mkdir(parents=True)
    (output_reference / main.REFERENCE_METADATA_FILE).write_text("{}")
    repository_reference = tmp_path / "repository-reference"
    repository_reference.mkdir()
    (repository_reference / main.REFERENCE_METADATA_FILE).write_text("{}")
    monkeypatch.setattr(
        main, "DEFAULT_CONFIDENCE_REFERENCE_DIR", str(repository_reference)
    )

    selected, searched = main.resolve_confidence_reference_dir(str(output_dir))

    assert selected == str(output_reference)
    assert searched == (str(output_reference), str(repository_reference))


def test_explicit_confidence_reference_is_authoritative(tmp_path):
    output_dir = tmp_path / "output"
    automatic_reference = output_dir / "confidence_reference"
    automatic_reference.mkdir(parents=True)
    (automatic_reference / main.REFERENCE_METADATA_FILE).write_text("{}")
    explicit_reference = tmp_path / "explicit-reference"

    selected, searched = main.resolve_confidence_reference_dir(
        str(output_dir), str(explicit_reference)
    )

    assert selected is None
    assert searched == (str(explicit_reference),)


# --------------------------------------------------------------------------- #
# Saving and loading the CCP4 setup path must agree on precedence
# --------------------------------------------------------------------------- #
def test_saved_ccp4_setup_is_the_one_that_gets_loaded_back(tmp_path):
    """``--configure-ccp4`` must actually take effect on the next run.

    ``save_ccp4_setup`` writes the user-level file (``config_files[0]``).

    Regression: ``load_ccp4_setup_config`` merged every file in order and let
    the *last* one win, while ``main.default_ccp4_config_files`` puts the
    in-repo ``.alchemy/ccp4.json`` last. A user who ran ``--configure-ccp4``
    was told the path had been saved and then kept getting the stale one, with
    no diagnostic. Save and load must agree on which file is authoritative.
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


def test_later_config_files_still_supply_keys_the_primary_omits(tmp_path):
    """First-wins precedence must not stop later files contributing at all.

    Only a key already present earlier is shadowed; anything the primary file
    does not define is still picked up from a later one.
    """
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


# --------------------------------------------------------------------------- #
# An explicit --ccp4-setup must beat whatever the shell already has
# --------------------------------------------------------------------------- #
def _stub_ccp4_dir(root, marker):
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
    monkeypatch, tmp_path
):
    """A typo'd ``--ccp4-setup`` must fail, not fall through to ``PATH``.

    Regression: ``resolve_ccp4_environment`` returned as soon as
    ``ccp4_tools_available`` was true, before it ever looked at
    ``args.ccp4_setup``. A nonexistent path therefore exited 0 and the run
    proceeded against the ambient environment, so a typo was indistinguishable
    from success.
    """
    on_path = _stub_ccp4_dir(tmp_path, "ambient")
    monkeypatch.setenv("PATH", str(on_path))
    assert ccp4_setup.ccp4_tools_available(os.environ), (
        "the stub must satisfy the PATH probe, or this test proves nothing"
    )

    args = argparse.Namespace(
        configure_ccp4=None, ccp4_setup="/nonexistent/ccp4.setup-sh"
    )
    with pytest.raises(SystemExit, match="not found"):
        main.resolve_ccp4_environment(args)


@pytest.mark.skipif(sys.platform == "win32", reason="writes a POSIX sh setup script")
def test_explicit_ccp4_setup_overrides_the_installation_already_on_path(
    monkeypatch, tmp_path
):
    """The requested installation is used, not the one the shell had sourced.

    This is the reason the option exists: a user passes ``--ccp4-setup``
    precisely because the wrong CCP4 is already on ``PATH``. Honouring ``PATH``
    first ran the wrong binaries *and* recorded their version as the run's
    provenance, so the output looked internally consistent while describing an
    installation the user had tried to replace.
    """
    ambient = _stub_ccp4_dir(tmp_path, "ambient")
    requested = _stub_ccp4_dir(tmp_path, "requested")
    # The stub goes first but the system directories stay: sourcing the setup
    # script runs through ``bash``, which must still be resolvable.
    monkeypatch.setenv("PATH", f"{ambient}{os.pathsep}{os.defpath}")
    assert ccp4_setup.ccp4_tools_available(os.environ), (
        "the ambient stub must satisfy the PATH probe, or the override this "
        "test checks would never have been bypassed in the first place"
    )

    setup = tmp_path / "ccp4.setup-sh"
    setup.write_text(f'export PATH="{requested}:$PATH"\n', encoding="utf-8")

    args = argparse.Namespace(configure_ccp4=None, ccp4_setup=str(setup))
    env, used = main.resolve_ccp4_environment(args)

    assert env is not None
    assert used == str(setup)
    resolved = shutil.which("edstats", path=env.get("PATH"))
    assert resolved is not None
    assert str(requested) in resolved, (
        f"the ambient PATH installation won over the requested one: {resolved}"
    )
