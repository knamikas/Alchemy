"""Locating CCP4 and resolving the environment its programs run under.

Discovery (where is the setup script?) and resolution (what environment does
running it produce?) are one responsibility, not two: every caller that needs
one needs the other, and splitting them across two modules produced four
concrete defects -- two disagreeing definitions of the default config files,
two implementations of the tool-availability probe, a dead ``explicit_setup``
parameter, and library code raising ``SystemExit``.

Nothing here raises ``SystemExit``. Failures are ``Ccp4SetupError``, so CCP4
resolution is usable outside a CLI process;
``driver.pool.resolve_ccp4_environment`` is the single place that turns one
into an exit.
"""

import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence
import shlex
import shutil
import subprocess
import sys
import tempfile


class Ccp4SetupError(Exception):
    """CCP4 could not be located, sourced, or verified.

    Carries a message written for a user at a terminal -- the CLI prints it
    verbatim -- but is an ordinary exception so a library caller can catch it.
    """


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_CCP4_TOOLS = ("mtzfix", "fft", "mapmask", "edstats")

#: Every file consulted for a saved CCP4 setup path, highest precedence first.
#: ``save_ccp4_setup`` writes only to the first, and ``load_ccp4_setup_config``
#: merges with the first file winning, so the two agree about which file is
#: authoritative. There is deliberately one list: a second definition elsewhere
#: meant the test harness and the application disagreed about where
#: configuration lives.
DEFAULT_CONFIG_FILES = [
    os.path.expanduser("~/.config/alchemy/ccp4.json"),
    os.path.expanduser("~/.alchemy/ccp4.json"),
    os.path.join(REPO_DIR, ".alchemy", "ccp4.json"),
]

WINDOWS_CCP4_SETUP_NAMES = ("ccp4.setup.bat", "ccp4.setup.cmd")

# Marker echoed by the Windows setup wrapper so the CCP4 launcher's own banner
# is never mistaken for environment variables.
ENV_SENTINEL = "__ALCHEMY_CCP4_ENV__"

# Sourcing a CCP4 setup script should take well under a second. A budget this
# small still leaves room for a slow network filesystem, and a setup script that
# blocks past it is hung -- most often on an interactive prompt no one can
# answer, since the shell has no terminal. Exceeding it aborts the run rather
# than failing one entry: without CCP4 there is nothing to process.
SETUP_SHELL_TIMEOUT_S = 30


def _windows_ccp4_setup_candidates() -> list[str]:
    """Best-effort Windows install locations for the CCP4 batch launcher.

    ``%CCP4%`` is checked first because the CCP4 installer sets it, which makes
    it the only non-guessed root available. The remaining directories are the
    usual installer defaults; auto-detection is a convenience, and
    ``--configure-ccp4`` remains the reliable way to record the real path.
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
    config_files: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    """Merge the CCP4 configuration files, earliest file winning.

    The order must match ``save_ccp4_setup``, which writes only to
    ``config_files[0]``. Merging with the last file winning meant a path stored
    by ``--configure-ccp4`` was shadowed by any later file that also carried
    one, so configuration appeared to succeed and was then silently ignored.
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
    setup_path: str, config_files: Optional[Sequence[str]] = None
) -> list[str]:
    config_files = config_files or DEFAULT_CONFIG_FILES
    target = config_files[0]  # write only to the primary user-level location
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
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
    env: Optional[Mapping[str, str]] = None,
    config: Optional[Mapping[str, str]] = None,
    config_files: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Locate a CCP4 setup script, or return ``None``.

    ``None`` means two different things and the caller must distinguish them:
    CCP4 is already on PATH and no script is needed, or nothing could be found
    at all. ``driver.pool.resolve_ccp4_environment`` treats the second as fatal.

    An explicitly requested script is deliberately not handled here. The caller
    that has one overrides the ambient environment with it and errors on a path
    that does not exist -- stronger semantics than a first-branch return -- so
    a parameter for it would be a second implementation of one concept.
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


def missing_ccp4_tools(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Return the required CCP4 programs absent from ``env``'s PATH.

    The single availability probe. ``ccp4_tools_available`` and
    ``verify_ccp4`` differ only in what they do with the answer; when they were
    separate sweeps, the driver could accept an installation the setup helper
    rejected.
    """
    env = os.environ.copy() if env is None else env
    return [
        tool
        for tool in REQUIRED_CCP4_TOOLS
        if shutil.which(tool, path=env.get("PATH")) is None
    ]


def ccp4_tools_available(env: Optional[Mapping[str, str]] = None) -> bool:
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
    """Ensure the PATH variable is accessible under the exact key "PATH".

    Different platforms/shells report it with different casing (Windows'
    `set` reports "Path"; Unix shells report "PATH"). Python dict lookups
    are case-sensitive, so downstream code that does env.get("PATH") would
    silently miss it if the key came back in a different case. This finds
    any case-variant of "PATH" and consolidates it under the exact
    all-caps key, leaving every other variable untouched.
    """
    for k in list(env):
        if k.upper() == "PATH" and k != "PATH":
            env["PATH"] = env.pop(k)
    return env


def _parse_windows_set_output(stdout: str) -> tuple[dict[str, str], bool]:
    """Return the ``set`` variables printed after ENV_SENTINEL.

    The CCP4 batch launcher prints its own banner before the variables, and any
    of those lines can contain "=". Everything before the sentinel is therefore
    discarded rather than guessed at by prefix.
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
    # Driving cmd.exe through a temporary script avoids its quoting rules,
    # which differ from the ones subprocess applies when building a command
    # line, and so would mis-handle an install path containing spaces.
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
        # A fixed name in %TEMP% left one file behind per run and let
        # concurrent runs overwrite each other's script.
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
    return _normalize_path_key({**os.environ.copy(), **env})


def resolve_env(ccp4_setup: Optional[str]) -> dict[str, str]:
    """Return the environment dict to run CCP4 under.

    If `ccp4_setup` is given and looks like a bash setup script, source it in a
    bash subshell and capture the resulting environment. If it is a Windows batch
    launcher, run it in a cmd shell and capture the resulting environment instead.
    If no setup script is provided, fall back to the current environment.
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
    return _normalize_path_key({**os.environ.copy(), **env})
