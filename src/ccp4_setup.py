"""Locating CCP4 and resolving the environment its programs run under.

Failures are ``Ccp4SetupError``, never ``SystemExit``, so this is usable
outside a CLI process; ``driver.pool.resolve_ccp4_environment`` is the one
place that turns one into an exit.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence


class Ccp4SetupError(Exception):
    """CCP4 could not be located, sourced, or verified.

    The CLI prints the message verbatim, so it is written for a terminal.
    """


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_CCP4_TOOLS = ("mtzfix", "fft", "mapmask", "edstats")

# Consulted for a saved CCP4 setup path, highest precedence first.
# ``save_ccp4_setup`` writes only to the first, and ``load_ccp4_setup_config``
# merges with the first file winning.
DEFAULT_CONFIG_FILES = [
    os.path.expanduser("~/.config/alchemy/ccp4.json"),
    os.path.expanduser("~/.alchemy/ccp4.json"),
    os.path.join(REPO_DIR, ".alchemy", "ccp4.json"),
]

WINDOWS_CCP4_SETUP_NAMES = ("ccp4.setup.bat", "ccp4.setup.cmd")

# Marker echoed by the Windows setup wrapper so the CCP4 launcher's own banner
# is never mistaken for environment variables.
ENV_SENTINEL = "__ALCHEMY_CCP4_ENV__"

# Sourcing a CCP4 setup script takes well under a second even on a network
# filesystem; one that blocks past this is hung, most often on an interactive
# prompt the shell has no terminal to answer.
SETUP_SHELL_TIMEOUT_S = 30


def _windows_ccp4_setup_candidates() -> list[str]:
    """Best-effort Windows install locations for the CCP4 batch launcher.

    ``%CCP4%`` comes first because the installer sets it, making it the only
    root that is not a guess; the rest are the usual installer defaults.
    """
    roots = []
    ccp4_root = os.environ.get("CCP4")
    if ccp4_root:
        roots.append(ccp4_root)
    system_drive = os.environ.get("SystemDrive", "C:")
    for base in (
        system_drive + os.sep,
        os.environ.get("ProgramFiles", ""),
        os.path.expanduser("~"),
    ):
        if not base:
            continue
        for version in ("CCP4-9", "CCP4-8", "CCP4"):
            roots.append(os.path.join(base, version))
    return [
        os.path.join(root, name) for root in roots for name in WINDOWS_CCP4_SETUP_NAMES
    ]


COMMON_CCP4_SETUP_CANDIDATES = (
    _windows_ccp4_setup_candidates()
    if sys.platform == "win32"
    else [
        "/opt/ccp4/bin/ccp4.setup-sh",
        "/usr/local/ccp4/bin/ccp4.setup-sh",
        "/opt/ccp4/ccp4.setup-sh",
        "/usr/local/ccp4/ccp4.setup-sh",
        os.path.expanduser("~/CCP4-9/CCP4/ccp4.setup-sh"),
        os.path.expanduser("~/CCP4/ccp4.setup-sh"),
    ]
)


def load_ccp4_setup_config(
    config_files: Sequence[str] | None = None,
) -> dict[str, str]:
    """Merge the CCP4 configuration files, earliest file winning.

    The order must match ``save_ccp4_setup``, which writes only to
    ``config_files[0]``; with the last file winning, a path stored by
    ``--configure-ccp4`` is shadowed and silently ignored.
    """
    config_files = config_files or DEFAULT_CONFIG_FILES
    config: dict[str, str] = {}
    for config_file in config_files:
        if not config_file:
            continue
        path = Path(config_file)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for key, value in data.items():
                    config.setdefault(key, value)
        except (OSError, json.JSONDecodeError):
            continue
    return config


def save_ccp4_setup(
    setup_path: str, config_files: Sequence[str] | None = None
) -> list[str]:
    config_files = config_files or DEFAULT_CONFIG_FILES
    target = config_files[0]
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except (OSError, json.JSONDecodeError):
            data = {}
    data["ccp4_setup"] = setup_path
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return [str(path)]


def find_ccp4_setup(
    env: Mapping[str, str] | None = None,
    config: Mapping[str, str] | None = None,
    config_files: Sequence[str] | None = None,
) -> str | None:
    """Locate a CCP4 setup script, or return ``None``.

    ``None`` means two different things and the caller must distinguish them:
    CCP4 is already on PATH and no script is needed, or nothing could be found
    at all. ``driver.pool.resolve_ccp4_environment`` treats the second as fatal.
    """
    env = os.environ.copy() if env is None else env
    config = (
        load_ccp4_setup_config(config_files=config_files) if config is None else config
    )

    if ccp4_tools_available(env):
        return None

    if env.get("CCP4_SETUP") and os.path.exists(env["CCP4_SETUP"]):
        return env["CCP4_SETUP"]

    if config.get("ccp4_setup") and os.path.exists(config["ccp4_setup"]):
        return config["ccp4_setup"]

    for candidate in COMMON_CCP4_SETUP_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate

    return None


def missing_ccp4_tools(env: Mapping[str, str] | None = None) -> list[str]:
    """Return the required CCP4 programs absent from ``env``'s PATH."""
    env = os.environ.copy() if env is None else env
    return [
        tool
        for tool in REQUIRED_CCP4_TOOLS
        if shutil.which(tool, path=env.get("PATH")) is None
    ]


def ccp4_tools_available(env: Mapping[str, str] | None = None) -> bool:
    return not missing_ccp4_tools(env)


def verify_ccp4(env: Mapping[str, str]) -> None:
    """Raise ``Ccp4SetupError`` naming the tools ``env`` cannot resolve."""
    missing = missing_ccp4_tools(env)
    if missing:
        raise Ccp4SetupError(
            f"Required CCP4 tools were not found on PATH: {', '.join(missing)}. "
            "Set them up once with --configure-ccp4 /path/to/ccp4.setup-sh, "
            "export CCP4_SETUP=/path/to/ccp4.setup-sh, or source CCP4 in\n"
            "your shell before running."
        )


def _normalize_path_key(env: dict[str, str]) -> dict[str, str]:
    """Consolidate any case-variant of PATH under the exact key ``PATH``.

    Windows' ``set`` reports ``Path``; dict lookups are case-sensitive, so
    ``env.get("PATH")`` downstream would silently miss it.
    """
    for k in list(env):
        if k.upper() == "PATH" and k != "PATH":
            env["PATH"] = env.pop(k)
    return env


def _parse_windows_set_output(stdout: str) -> tuple[dict[str, str], bool]:
    """Return the ``set`` variables printed after ENV_SENTINEL.

    The CCP4 batch launcher prints its own banner first, and any of those lines
    can contain "=", so everything before the sentinel is discarded.
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


def _resolve_env_windows(ccp4_setup: str) -> dict[str, str]:
    """Capture the environment a Windows CCP4 batch launcher establishes."""
    # cmd.exe's quoting rules differ from the ones subprocess applies when
    # building a command line, so an install path containing spaces survives
    # only via a temporary script.
    handle, script_path = tempfile.mkstemp(prefix="alchemy-ccp4-", suffix=".cmd")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\r\n") as fh:
            fh.write(f'@echo off\ncall "{ccp4_setup}"\necho {ENV_SENTINEL}\nset\n')
        try:
            out = subprocess.run(
                ["cmd", "/c", script_path],
                capture_output=True,
                text=True,
                timeout=SETUP_SHELL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise Ccp4SetupError(
                f"CCP4 setup {ccp4_setup} did not finish within "
                f"{SETUP_SHELL_TIMEOUT_S}s and was stopped. A setup script "
                "that blocks usually waits on input the launcher cannot "
                "provide. Alchemy cannot run without CCP4, so this stops the "
                "run rather than failing one entry."
            ) from None
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    if out.returncode != 0:
        raise Ccp4SetupError(f"Failed to run CCP4 setup {ccp4_setup}:\n{out.stderr}")
    env, seen_sentinel = _parse_windows_set_output(out.stdout)
    if not seen_sentinel:
        raise Ccp4SetupError(
            f"CCP4 setup {ccp4_setup} did not report its environment; "
            f"expected `set` output after the marker.\n{out.stderr}"
        )
    # ``cmd`` inherited the parent environment before calling the setup file,
    # so this listing is authoritative: merging the parent back would restore
    # variables the setup intentionally removed.
    return _normalize_path_key(env)


def resolve_env(ccp4_setup: str | None) -> dict[str, str]:
    """Return the environment dict to run CCP4 under.

    Sourcing ``ccp4_setup`` in a subshell and capturing the result is the only
    way to reproduce what it exports; without one, the current environment is
    assumed to be set up already.
    """
    if not ccp4_setup:
        return os.environ.copy()
    if not os.path.exists(ccp4_setup):
        raise Ccp4SetupError(f"CCP4 setup file not found: {ccp4_setup}")

    if os.path.splitext(ccp4_setup)[1].lower() in (".bat", ".cmd"):
        return _resolve_env_windows(ccp4_setup)

    cmd = f"source {shlex.quote(ccp4_setup)} >/dev/null 2>&1 && env -0"
    try:
        out = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=SETUP_SHELL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise Ccp4SetupError(
            f"CCP4 setup {ccp4_setup} did not finish within "
            f"{SETUP_SHELL_TIMEOUT_S}s and was stopped. A setup script that "
            "blocks usually waits on input the shell cannot provide. Alchemy "
            "cannot run without CCP4, so this stops the run rather than "
            "failing one entry."
        ) from None
    if out.returncode != 0:
        raise Ccp4SetupError(f"Failed to source CCP4 setup {ccp4_setup}:\n{out.stderr}")
    env = {}
    for chunk in out.stdout.split("\0"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            env[k] = v
    # The shell inherited the parent before sourcing the setup file. Its final
    # ``env`` output is therefore authoritative, including deliberate unsets.
    return _normalize_path_key(env)
