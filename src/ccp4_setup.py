import json
import os
from pathlib import Path
import shutil
import sys


REQUIRED_CCP4_TOOLS = ("mtzfix", "fft", "edstats")

DEFAULT_CONFIG_FILES = [
    os.path.expanduser("~/.config/alchemy/ccp4.json"),
    os.path.expanduser("~/.alchemy/ccp4.json"),
]

WINDOWS_CCP4_SETUP_NAMES = ("ccp4.setup.bat", "ccp4.setup.cmd")


def _windows_ccp4_setup_candidates():
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
    for base in (system_drive + os.sep, os.environ.get("ProgramFiles", ""),
                 os.path.expanduser("~")):
        if not base:
            continue
        for version in ("CCP4-9", "CCP4-8", "CCP4"):
            roots.append(os.path.join(base, version))
    return [os.path.join(root, name)
            for root in roots for name in WINDOWS_CCP4_SETUP_NAMES]


COMMON_CCP4_SETUP_CANDIDATES = (
    _windows_ccp4_setup_candidates() if sys.platform == "win32" else [
        "/opt/ccp4/bin/ccp4.setup-sh",
        "/usr/local/ccp4/bin/ccp4.setup-sh",
        "/opt/ccp4/ccp4.setup-sh",
        "/usr/local/ccp4/ccp4.setup-sh",
        os.path.expanduser("~/CCP4-9/CCP4/ccp4.setup-sh"),
        os.path.expanduser("~/CCP4/ccp4.setup-sh"),
    ]
)


def load_ccp4_setup_config(config_files=None):
    config_files = config_files or DEFAULT_CONFIG_FILES
    config = {}
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
                config.update(data)
        except (OSError, json.JSONDecodeError):
            continue
    return config


def save_ccp4_setup(setup_path, config_files=None):
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


def find_ccp4_setup(explicit_setup=None, env=None, config=None, config_files=None, common_candidates=None):
    env = env or os.environ.copy()
    config = config or load_ccp4_setup_config(config_files=config_files)
    common_candidates = common_candidates or COMMON_CCP4_SETUP_CANDIDATES

    if explicit_setup:
        return explicit_setup

    if ccp4_tools_available(env):
        return None

    if env.get("CCP4_SETUP") and os.path.exists(env["CCP4_SETUP"]):
        return env["CCP4_SETUP"]

    if config.get("ccp4_setup") and os.path.exists(config["ccp4_setup"]):
        return config["ccp4_setup"]

    for candidate in common_candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return None


def ccp4_tools_available(env=None):
    env = env or os.environ.copy()
    return all(
        shutil.which(tool, path=env.get("PATH")) is not None
        for tool in REQUIRED_CCP4_TOOLS
    )
