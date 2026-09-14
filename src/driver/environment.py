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


def _resolve_ccp4_environment(
    args: RunConfig,
) -> tuple[dict[str, str] | None, str | None]:
    """Resolve the CCP4 environment, raising ``Ccp4SetupError`` on any failure."""
    if args.configure_ccp4:
        setup_path = _existing_setup_path(args.configure_ccp4)
        env = resolve_env(setup_path)
        _verify_resolved_ccp4(env, setup_path)

        saved = save_ccp4_setup(setup_path)
        logger.info(
            "verified %s are available; saved CCP4 setup path to %s",
            ", ".join(REQUIRED_CCP4_TOOLS),
            ", ".join(saved),
        )
        return None, None

    environment = os.environ.copy()

    # Validate explicit setup before PATH so a bad override cannot select another install.
    if args.ccp4_setup:
        setup_path = _existing_setup_path(args.ccp4_setup)
        env = resolve_env(setup_path)
        _verify_resolved_ccp4(env, setup_path)
        return env, setup_path

    if ccp4_tools_available(environment):
        return environment, None

    ccp4_setup = find_ccp4_setup(env=environment, config=load_ccp4_setup_config())
    if ccp4_setup is None:
        raise Ccp4SetupError(
            f"Required CCP4 tools ({', '.join(REQUIRED_CCP4_TOOLS)}) were not "
            "found on PATH and no setup file could be auto-detected. "
            "Set them up once with --configure-ccp4 /path/to/ccp4.setup-sh, "
            "export CCP4_SETUP=/path/to/ccp4.setup-sh, or source CCP4 in\n"
            "your shell before running."
        )
    env = resolve_env(ccp4_setup)
    verify_ccp4(env)
    return env, ccp4_setup


def resolve_ccp4_environment(
    args: RunConfig,
) -> tuple[dict[str, str] | None, str | None]:
    """Return ``(env, setup_path)`` for this run, or raise ``DriverError``."""
    try:
        return _resolve_ccp4_environment(args)
    except Ccp4SetupError as exc:
        raise DriverError(str(exc)) from None


def alchemy_commit() -> str:
    """Return the abbreviated source commit with a dirty-worktree marker."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=PROVENANCE_COMMAND_TIMEOUT_S,
        )
        commit = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=PROVENANCE_COMMAND_TIMEOUT_S,
        )
        return commit + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


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
