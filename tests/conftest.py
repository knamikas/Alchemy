"""Pytest configuration shared by the whole suite.

Responsibilities:

1. Put ``src`` on ``sys.path`` so tests can ``import main``, ``import
   bond_analysis`` and friends. ``src`` is not a package.
2. Turn the ``ccp4``, ``entry_data``, ``network`` and ``slow`` markers into
   clean skips for an ordinary local run, with opt-in strict modes for
   capability-specific lanes.
3. Expose fixtures for the things a test cannot build itself: a CCP4
   environment, a PDB-REDO download cache, and an isolated working directory.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import pytest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")

# Must happen at import time, before any test module is collected.
for _path in (SRC_DIR, TESTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gemmi  # noqa: E402  (needs the sys.path entries above)

import helpers  # noqa: E402  (needs the sys.path entries above)


#: ``src/structure_analysis.py`` uses ``gemmi.Model.num``, which arrived in
#: 0.7. Older builds import fine and then fail deep inside ~200 unrelated tests
#: with ``AttributeError``, which reads as a broken suite rather than a
#: too-old dependency.
MINIMUM_GEMMI = (0, 7)


def _gemmi_version_string():
    """Return gemmi's reported version.

    Read through ``getattr`` because gemmi ships ``py.typed`` while its stubs
    omit ``__version__``, so a direct attribute access is a type-checker error
    against a name that exists at runtime. ``src/main.py`` reads it the same way.
    """
    return str(getattr(gemmi, "__version__", "unknown"))


def _gemmi_version():
    parts = []
    for field in _gemmi_version_string().split(".")[:2]:
        digits = "".join(c for c in field if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def pytest_configure(config):
    """Fail fast, and legibly, on a gemmi too old for ``src`` to run against."""
    version = _gemmi_version()
    if version < MINIMUM_GEMMI:
        raise pytest.UsageError(
            f"gemmi {_gemmi_version_string()} is too old for Alchemy: "
            f"{'.'.join(str(p) for p in MINIMUM_GEMMI)} or newer is required "
            "(src/structure_analysis.py uses gemmi.Model.num). Install it with "
            "'python3 -m pip install --upgrade gemmi'. Note that a changed HOME "
            "can hide a newer user-site gemmi behind an older system one."
        )

    for name in ("ccp4", "network"):
        disabled = config.getoption(f"--no-{name}")
        required = config.getoption(f"--require-{name}")
        if disabled and required:
            raise pytest.UsageError(
                f"--no-{name} and --require-{name} are mutually exclusive"
            )
        if disabled:
            os.environ[f"ALCHEMY_TESTS_NO_{name.upper()}"] = "1"

    config._alchemy_required_capabilities = tuple(
        name for name in ("ccp4", "network") if config.getoption(f"--require-{name}")
    )
    if config.getoption("--require-entry-data"):
        config._alchemy_required_capabilities += ("entry_data",)
    config._alchemy_capability_tests_ran = {
        name: False for name in config._alchemy_required_capabilities
    }


# Options
def pytest_addoption(parser):
    group = parser.getgroup("alchemy")
    group.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="skip tests marked slow (end-to-end pipeline runs)",
    )
    group.addoption(
        "--no-ccp4",
        action="store_true",
        default=False,
        help="pretend CCP4 is unavailable; skip tests marked ccp4",
    )
    group.addoption(
        "--no-network",
        action="store_true",
        default=False,
        help="pretend the network is down; skip tests marked network",
    )
    group.addoption(
        "--require-ccp4",
        action="store_true",
        default=False,
        help=(
            "require CCP4 and at least one non-skipped ccp4 test; fail instead "
            "of silently passing a skipped capability lane"
        ),
    )
    group.addoption(
        "--require-network",
        action="store_true",
        default=False,
        help=(
            "require live network access and at least one non-skipped network "
            "test; fail instead of silently passing a skipped capability lane"
        ),
    )
    group.addoption(
        "--require-entry-data",
        action="store_true",
        default=False,
        help=(
            "require at least one non-skipped test backed by the pinned "
            "PDB-REDO entry snapshot"
        ),
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Skip marked tests whose external requirement is not satisfied.

    Running after pytest's built-in ``-m``/``-k`` selection is essential: a
    deselected network test must not cause the capability probe it was selected
    out to avoid. Availability is probed at most once per session, and only when
    a surviving test carries the relevant marker.
    """
    # Annotated because the two probes have different signatures --
    # ``network_available`` takes three defaulted arguments, ``ccp4_available``
    # none -- so the inferred value type is their join, which is not callable.
    probes: dict[str, tuple[Callable[[], bool], str]] = {
        "ccp4": (
            helpers.ccp4_available,
            "CCP4 (mtzfix/fft/mapmask/edstats) is not available",
        ),
        "network": (helpers.network_available, "no network access to pdb-redo.eu"),
    }
    results = {}
    for name, (probe, reason) in probes.items():
        if any(item.get_closest_marker(name) for item in items):
            available = probe()
            if config.getoption(f"--require-{name}") and not available:
                raise pytest.UsageError(f"--require-{name} was requested, but {reason}")
            results[name] = (available, reason)

    skip_slow = config.getoption("--skip-slow")
    for item in items:
        for name, (available, reason) in results.items():
            if item.get_closest_marker(name) and not available:
                item.add_marker(pytest.mark.skip(reason=reason))
        if skip_slow and item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.skip(reason="--skip-slow"))


def pytest_collection_finish(session):
    """Reject a strict capability lane that selected no matching tests."""
    required = session.config._alchemy_required_capabilities
    for name in required:
        if not any(item.get_closest_marker(name) for item in session.items):
            label = name.replace("_", "-")
            raise pytest.UsageError(
                f"--require-{label} was requested, but no {label} tests were selected"
            )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Remember whether a strict lane reached a non-skipped test call."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.skipped:
        return
    ran = item.config._alchemy_capability_tests_ran
    for name in ran:
        if item.get_closest_marker(name):
            ran[name] = True


def pytest_sessionfinish(session, exitstatus):
    """Make an all-skipped strict capability lane fail at process level."""
    if exitstatus != pytest.ExitCode.OK or session.config.getoption("collectonly"):
        return
    missing = [
        name
        for name, ran in session.config._alchemy_capability_tests_ran.items()
        if not ran
    ]
    if not missing:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    message = "required capability lane ran no non-skipped tests: " + ", ".join(
        name.replace("_", "-") for name in missing
    )
    if reporter is not None:
        reporter.write_sep("=", f"ERROR: {message}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


# Paths
@pytest.fixture(scope="session")
def repo_root() -> str:
    """Absolute path of the repository checkout under test."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def src_dir() -> str:
    """Absolute path of ``src/``, already on ``sys.path``."""
    return SRC_DIR


@pytest.fixture(scope="session")
def data_dir() -> str:
    """Absolute path of ``src/data/`` (bundled literature and catalog files)."""
    return os.path.join(SRC_DIR, "data")


@pytest.fixture
def work_dir(tmp_path, monkeypatch) -> str:
    """A tmp directory that is also the process cwd for the duration of a test.

    Use it for code that writes relative paths, and to prove that code under
    test writes nothing into whatever directory pytest happened to start in --
    nothing may be left in the repository. ``monkeypatch`` restores the previous
    cwd at teardown even when the test fails.
    """
    monkeypatch.chdir(tmp_path)
    return str(tmp_path)


# External capabilities
@pytest.fixture(scope="session")
def ccp4_env():
    """Environment mapping with the CCP4 binaries on ``PATH``.

    Skips the test when CCP4 cannot be resolved, so it is safe to request even
    without the ``ccp4`` marker -- but mark the test ``ccp4`` anyway so the
    reason is visible in a plain collection.
    """
    env = helpers.ccp4_env()
    if env is None:
        pytest.skip("CCP4 (mtzfix/fft/mapmask/edstats) is not available")
    return env


@pytest.fixture(scope="session")
def pdb_redo_cache(tmp_path_factory) -> str:
    """Directory for PDB-REDO downloads, shared across the session.

    Honours the cache environment variable when set (the orchestrator points it
    at a warm cache); otherwise a session tmp directory is used, which means a
    real download. ``helpers.cache_dir_from_env`` is the suite-wide reader, so
    the canonical ``ALCHEMY_TESTS_CACHE`` and older ``ALCHEMY_TEST_CACHE`` both
    work, with the first nonblank value winning in canonical-then-legacy order.
    Treat the contents as best-effort: entries may be missing.
    """
    configured = helpers.cache_dir_from_env()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    return str(tmp_path_factory.mktemp("pdb_redo_cache"))
