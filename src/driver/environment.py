"""Resolve the CCP4 environment and record software provenance for a run."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping

from _version import __version__
from ccp4_setup import (
    REPO_DIR,
    REQUIRED_CCP4_TOOLS,
    Ccp4SetupError,
    ccp4_tools_available,
    find_ccp4_setup,
    load_ccp4_setup_config,
    resolve_env,
    save_ccp4_setup,
    verify_ccp4,
)
from driver.errors import DriverError
from run_config import RunConfig
from run_logging import logger_for

ALCHEMY_VERSION = __version__
# If checking the Git commit takes too long, record it as unknown
# and continue the analysis.
PROVENANCE_COMMAND_TIMEOUT_S = 1

logger = logger_for(__name__)


def _verify_resolved_ccp4(env: Mapping[str, str], setup_path: str) -> None:
    """Verify CCP4 tools and include the setup path in any failure message."""
    try:
        verify_ccp4(env)
    except Ccp4SetupError as exc:
        raise Ccp4SetupError(
            f"Ran {setup_path}, but CCP4 tools are still not available. {exc}"
        ) from None


def _existing_setup_path(configured: str) -> str:
    """Expand a configured setup path, failing when nothing is there."""
    setup_path = os.path.abspath(os.path.expanduser(configured))
    if not os.path.exists(setup_path):
        raise Ccp4SetupError(f"CCP4 setup file not found: {setup_path}")
    return setup_path


def configure_ccp4(args: RunConfig) -> bool:
    """Save the ``--configure-ccp4`` setup path, or return False when not asked to.

    The path is verified before it is saved, so a later run never reads a
    setup file this one could not use. Raise ``DriverError`` on any failure.
    """
    if not args.configure_ccp4:
        return False
    try:
        setup_path = _existing_setup_path(args.configure_ccp4)
        env = resolve_env(setup_path)
        _verify_resolved_ccp4(env, setup_path)
        saved = save_ccp4_setup(setup_path)
    except Ccp4SetupError as exc:
        raise DriverError(str(exc)) from None
    logger.info(
        "verified %s are available; saved CCP4 setup path to %s",
        ", ".join(REQUIRED_CCP4_TOOLS),
        saved,
    )
    return True


def _resolve_ccp4_environment(args: RunConfig) -> dict[str, str]:
    """Resolve the CCP4 environment, raising ``Ccp4SetupError`` on any failure."""
    inherited = os.environ.copy()

    # Validate an explicit setup script before looking at PATH, so that a bad
    # override cannot silently select another install.
    if args.ccp4_setup:
        setup_path = _existing_setup_path(args.ccp4_setup)
        env = resolve_env(setup_path)
        _verify_resolved_ccp4(env, setup_path)
        return env

    if ccp4_tools_available(inherited):
        return inherited

    detected = find_ccp4_setup(env=inherited, config=load_ccp4_setup_config())
    if detected is None:
        raise Ccp4SetupError(
            f"Required CCP4 tools ({', '.join(REQUIRED_CCP4_TOOLS)}) were not "
            "found on PATH and no setup file could be auto-detected. "
            "Set them up once with --configure-ccp4 /path/to/ccp4.setup-sh, "
            "export CCP4_SETUP=/path/to/ccp4.setup-sh, or source CCP4 in "
            "your shell before running."
        )
    env = resolve_env(detected)
    _verify_resolved_ccp4(env, detected)
    return env


def resolve_ccp4_environment(args: RunConfig) -> dict[str, str]:
    """Return the environment the CCP4 tools run in, or raise ``DriverError``."""
    try:
        return _resolve_ccp4_environment(args)
    except Ccp4SetupError as exc:
        raise DriverError(str(exc)) from None


def _git_output(*arguments: str) -> str | None:
    """Return a git command's stdout, or None when it fails or times out."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=PROVENANCE_COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def alchemy_commit() -> str:
    """Return the abbreviated source commit with a worktree-state marker.

    ``+dirty`` marks tracked changes. A commit whose worktree state could not
    be determined in time keeps its hash and is marked ``+unknown-state``.
    """
    commit = _git_output("rev-parse", "--short=12", "HEAD")
    if not commit:
        return "unknown"
    status = _git_output("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return commit + "+unknown-state"
    return commit + ("+dirty" if status else "")


def gemmi_version() -> str:
    """Return the installed Gemmi version, or ``unknown``."""
    try:
        import gemmi

        return str(getattr(gemmi, "__version__", "unknown"))
    except ImportError:
        return "unknown"


def ccp4_version(env: Mapping[str, str]) -> str:
    """Return the CCP4 version exposed by a resolved environment."""
    for key in ("CCP4_VERSION", "CCP4_VERSION_CODE", "CCP4VER"):
        if env.get(key):
            return env[key]
    ccp4_root = env.get("CCP4", "")
    return os.path.basename(ccp4_root.rstrip(os.sep)) if ccp4_root else "unknown"
