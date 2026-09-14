"""Check imports, launchers, and shared fixture compatibility."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import cast

import gemmi
import helpers
import pytest
from helpers import simple_metal_site


def test_src_modules_import() -> None:
    """Verify source modules import and expose their entry points."""
    import ccp4_setup
    import cli
    import codes
    import confidence_score
    import density_analysis
    import main
    import metal_elements
    import metal_identification
    import structure_analysis
    import worker
    from coordination import (
        analysis,
        contact_record,
        declared_connections,
        donor_chemistry,
        dpi,
        schema,
    )
    from driver import pool, progress, resources, runlog, writers

    assert "ZN" in metal_elements.METAL_ELEMENTS
    assert analysis.CANDIDATE_SEARCH_RADIUS == 4.0
    assert callable(structure_analysis.load_structure)
    assert callable(main.main), "src/main.py must keep delegating to the CLI"
    assert main.main is cli.main, "the entry point must delegate, not reimplement"
    assert callable(pool.run)
    assert callable(worker.process)
    assert writers.MANIFEST_COLUMNS[0] == "pdbID"
    assert callable(progress.ProgressReporter)
    assert callable(runlog.RunLog)
    assert resources.available_cpu_count() >= 1
    assert callable(dpi.calculate_dpi_components)
    assert callable(declared_connections.collect_declared_candidates)
    assert schema.BOND_COLUMNS[0] == "pdbID"
    # Compare enum members directly with strings to verify serialized compatibility.
    # Mypy cannot infer the intended cross-enum comparison.
    assert codes.GeometryStatus.SUSPECT == "suspect"  # type: ignore[comparison-overlap]
    assert callable(contact_record.Candidate)
    assert set(donor_chemistry.INFERRED_DONOR_ATOMS) == donor_chemistry.AA
    assert callable(density_analysis.run_density_analysis)
    assert callable(metal_identification.extract_metal_statistics)
    assert callable(confidence_score.score_site)
    assert set(ccp4_setup.REQUIRED_CCP4_TOOLS) == {
        "mtzfix",
        "fft",
        "mapmask",
        "edstats",
    }


def test_src_dir_fixture_points_at_the_modules(
    src_dir: str, repo_root: str, data_dir: str
) -> None:
    """The path fixtures address the real checkout, not a copy or a stale root."""
    assert os.path.isfile(os.path.join(src_dir, "coordination", "analysis.py"))
    assert os.path.dirname(src_dir) == repo_root
    assert os.path.isfile(os.path.join(data_dir, "metal_distances_info.txt"))


def test_root_launcher_works_outside_the_repository(
    repo_root: str, tmp_path: Path
) -> None:
    """``./alchemy`` resolves src from itself and delegates to the CLI."""
    launcher = os.path.join(repo_root, "alchemy")
    command = [launcher, "--help"]
    # Windows does not dispatch extensionless shebang files, but invoking the
    # same file through Python still exercises its repository-path resolution.
    if sys.platform == "win32":
        command.insert(0, sys.executable)

    result = subprocess.run(
        command,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Batch Alchemy core pipeline over PDB-REDO." in result.stdout
    assert "--pdb-redo-root" in result.stdout


def test_analysis_writes_nothing_into_the_current_directory(
    work_dir: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A full in-memory analysis leaves the process working directory empty.

    Every path below is absolute and under a different temporary directory, so
    a file appearing in the cwd is src code writing to a relative path, which on
    a normal run would land in the checkout.
    """
    from coordination.analysis import run_bond_analysis
    from structure_analysis import load_structure

    inputs = tmp_path_factory.mktemp("isolated_inputs")
    path = simple_metal_site().write_pdb(inputs / "site.pdb")
    context = load_structure("test", path)
    stats_rows, header, _ = helpers.stats_rows_for_structure(
        context, inputs / "stats.out", metrics={"ZDm": 2.5}
    )
    analysis = run_bond_analysis(
        "test", path, stats_rows, header, helpers.dpi_inputs(), structure=context
    )
    rows = analysis.bond_rows

    assert rows, "the analysis must have actually run for this to mean anything"
    assert os.path.realpath(os.getcwd()) == os.path.realpath(work_dir)
    assert os.listdir(work_dir) == []


def test_cache_directory_helper_uses_canonical_then_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shared helper prefers the canonical nonblank cache variable.

    Every cache fixture in the suite resolves through this helper, so the
    precedence between ``ALCHEMY_TESTS_CACHE`` and the older
    ``ALCHEMY_TEST_CACHE`` is suite-wide.
    """
    for variable in helpers.CACHE_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    assert helpers.cache_dir_from_env() is None

    legacy = str(tmp_path / "legacy")
    monkeypatch.setenv("ALCHEMY_TEST_CACHE", legacy)
    assert helpers.cache_dir_from_env() == legacy

    canonical = str(tmp_path / "canonical")
    monkeypatch.setenv("ALCHEMY_TESTS_CACHE", canonical)
    assert helpers.cache_dir_from_env() == canonical, "canonical name wins"

    # A blank export means "unset", not "use the current directory", which
    # would scatter downloads through the checkout.
    monkeypatch.setenv("ALCHEMY_TESTS_CACHE", "   ")
    assert helpers.cache_dir_from_env() == legacy
    monkeypatch.delenv("ALCHEMY_TEST_CACHE")
    assert helpers.cache_dir_from_env() is None


def test_pdb_redo_cache_fixture_is_a_writable_directory(pdb_redo_cache: str) -> None:
    """The download cache fixture points somewhere downloads can actually go.

    A download test handed a missing or read-only path fails with an ``OSError``
    halfway through a fetch rather than skipping.
    """
    assert os.path.isdir(pdb_redo_cache)
    assert os.access(pdb_redo_cache, os.W_OK | os.X_OK)

    configured = helpers.cache_dir_from_env()
    if configured:
        assert os.path.realpath(pdb_redo_cache) == os.path.realpath(configured)


def test_ccp4_capability_probe_is_boolean_and_does_not_raise() -> None:
    """The local CCP4 probe answers with a bool instead of raising.

    ``conftest`` calls it during collection, so a probe that raised on an
    unreadable config would abort collection of the whole suite.
    """
    assert isinstance(helpers.ccp4_available(), bool)


def test_not_network_selection_does_not_call_the_network_probe(
    tmp_path: Path, repo_root: str
) -> None:
    """Marker deselection happens before the external capability probe.

    ``-m 'not network'`` is an offline selection boundary, not just a promise to
    discard the test after collection, so the nested run replaces the probe with
    one that raises.
    """
    test_file = tmp_path / "test_marker_selection.py"
    test_file.write_text(
        "import pytest\n\n"
        "@pytest.mark.network\n"
        "def test_live():\n"
        "    raise AssertionError('network test was not deselected')\n\n"
        "def test_local():\n"
        "    pass\n",
        encoding="utf-8",
    )
    plugin_file = tmp_path / "forbid_network_probe.py"
    plugin_file.write_text(
        "import helpers\n\n"
        "def pytest_configure(config):\n"
        "    def forbidden(*args, **kwargs):\n"
        "        raise AssertionError('network probe ran after -m exclusion')\n"
        "    helpers.network_available = forbidden\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    python_path = [str(tmp_path), os.path.join(repo_root, "tests")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            "-p",
            "conftest",
            "-p",
            "forbid_network_probe",
            str(test_file),
            "-m",
            "not network",
            "--no-ccp4",
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout
    assert "test_local" in completed.stdout
    assert "test_live" not in completed.stdout


@pytest.mark.ccp4
def test_ccp4_env_fixture_resolves_every_required_tool(
    ccp4_env: dict[str, str],
) -> None:
    """Verify the resolved CCP4 programs start successfully."""
    resolved = {tool: helpers.which(tool, ccp4_env) for tool in helpers.CCP4_TOOLS}
    assert all(resolved.values()), resolved

    # Detect loader errors; missing-input messages are expected for these probes.
    loader_failures = (
        "error while loading shared libraries",
        "cannot open shared object file",
        "symbol lookup error",
        "undefined symbol",
    )

    for tool, path in resolved.items():
        assert path is not None
        # With no input each of the four prints a banner naming itself and
        # exits; the status differs between them (edstats 0, the rest 1).
        completed = subprocess.run(
            [path],
            env=ccp4_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        output = completed.stdout.decode("utf-8", "replace")
        excerpt = output[:500]

        # Negative return codes indicate termination by a signal, including startup crashes.
        assert completed.returncode >= 0, (
            f"{tool} died on signal {-completed.returncode}: {excerpt!r}"
        )
        for marker in loader_failures:
            assert marker not in output.lower(), f"{tool}: {excerpt!r}"
        # mtzfix prints "MTZFIX", fft prints "FFTBIG", mapmask and edstats print
        # their own names; reaching the banner means the binary loaded.
        assert tool in output.lower(), (
            f"{tool} produced no banner naming itself: {excerpt!r}"
        )


@pytest.mark.network
def test_network_marker_only_runs_with_connectivity() -> None:
    """PDB-REDO really answers, not merely a TCP handshake on port 443.

    A captive portal or a proxy that accepts connections and then refuses them
    passes the socket probe used for skipping but fails here.
    """
    assert helpers.network_available()

    url = "https://pdb-redo.eu/db/9myr/data.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        assert response.status == 200
        loaded: object = json.loads(response.read().decode("utf-8"))
    # A JSON-formatted proxy error is still a dict, so require the
    # crystallographic metadata Alchemy consumes from data.json.
    assert isinstance(loaded, dict)
    payload = cast("dict[object, object]", loaded)
    raw_properties = payload.get("properties")
    assert isinstance(raw_properties, dict) and raw_properties
    properties = cast("dict[object, object]", raw_properties)
    nrefcnt = properties["NREFCNT"]
    dataresh = properties["DATARESH"]
    assert isinstance(nrefcnt, (str, int, float))
    assert isinstance(dataresh, (str, int, float))
    assert float(nrefcnt) > 0
    assert float(dataresh) > 0


def test_gemmi_is_the_expected_flavour() -> None:
    """The installed gemmi has the APIs the helpers and src depend on.

    The helpers declare connections through ``Connection``/``AtomAddress``, and
    ``src/structure_analysis.py`` uses ``Model.num``, which arrived in 0.7.
    """
    assert hasattr(gemmi, "Connection") and hasattr(gemmi, "AtomAddress")
    # Read via getattr: gemmi ships py.typed but its stubs omit __version__,
    # so a direct access is a type error against a real runtime name.
    version = str(getattr(gemmi, "__version__", "unknown"))
    assert tuple(int(p) for p in version.split(".")[:2]) >= (0, 7)
