import json
import os
from pathlib import Path
import shutil
import sys


DEFAULT_CONFIG_FILES = [
    os.path.expanduser("~/.config/alchemy/ccp4.json"),
    os.path.expanduser("~/.alchemy/ccp4.json"),
]

COMMON_CCP4_SETUP_CANDIDATES = [
    "/opt/ccp4/bin/ccp4.setup-sh",
    "/usr/local/ccp4/bin/ccp4.setup-sh",
    "/opt/ccp4/ccp4.setup-sh",
    "/usr/local/ccp4/ccp4.setup-sh",
    os.path.expanduser("~/CCP4-9/CCP4/ccp4.setup-sh"),
    os.path.expanduser("~/CCP4/ccp4.setup-sh"),
]


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


def prompt_for_ccp4_setup(config_files=None):
    config_files = config_files or DEFAULT_CONFIG_FILES

    print("CCP4 tools were not found on PATH and no saved setup path is available.", flush=True)
    while True:
        try:
            raw_path = input("Enter the path to the CCP4 setup script (for example /opt/ccp4/bin/ccp4.setup-sh), or press Enter to cancel: ").strip()
        except EOFError:
            return None

        if not raw_path:
            return None

        setup_path = os.path.expanduser(raw_path)
        if os.path.exists(setup_path):
            saved = save_ccp4_setup(setup_path, config_files=config_files)
            print(f"Saved CCP4 setup path to {', '.join(saved)}", flush=True)
            return setup_path

        print("That path does not exist. Please enter a valid CCP4 setup script path.", flush=True)


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
    return all(shutil.which(tool, path=env.get("PATH")) is not None for tool in ("fft", "edstats"))
