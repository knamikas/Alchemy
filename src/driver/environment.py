"""Resolve the CCP4 environment and record software provenance for a run."""

from __future__ import annotations

import os
from collections.abc import Mapping

from _version import __version__
from driver.ccp4_setup import (
    CCP4_SETUP_HINT,
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
            "found on PATH and no setup file could be auto-detected. " + CCP4_SETUP_HINT
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
