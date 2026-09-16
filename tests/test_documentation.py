"""Check documentation against dependencies, CLI options, and output schemas."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
from helpers import REPO_ROOT, SRC_DIR

from codes import ReasonCode, WarningCode
from driver import confidence as driver_confidence, environment
from driver.runlog import (
    ENTRY_DIAGNOSTIC_BASE_COLUMNS,
    ENTRY_DIAGNOSTIC_TRAILING_COLUMNS,
    PREFERRED_TIMING_COLUMNS,
)
from driver.writers import STATS_COLUMNS

README_PATH = os.path.join(REPO_ROOT, "README.md")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
# Every prose file a reader is sent to.
DOC_PATHS = [README_PATH, os.path.join(REPO_ROOT, "tests", "README.md")] + [
    os.path.join(DOCS_DIR, name)
    for name in (
        "usage.md",
        "architecture.md",
        "method.md",
        "output-schema.md",
        "operations.md",
        "maintenance.md",
    )
]
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")
LAUNCHER_PATH = os.path.join(REPO_ROOT, "alchemy")

_STDLIB = set(sys.stdlib_module_names)

# Import name -> distribution name, spelled out so the mapping does not depend
# on what happens to be installed.
_IMPORT_TO_DISTRIBUTION = {"gemmi": "gemmi", "numpy": "numpy"}


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_every_reason_code_is_documented() -> None:
    """Verify every manifest reason code is documented in docs/operations.md."""
    documented = _read(os.path.join(DOCS_DIR, "operations.md"))
    missing = sorted(
        code.value for code in ReasonCode if f"`{code.value}`" not in documented
    )
    assert not missing, f"reason codes absent from docs/operations.md: {missing}"


def test_every_warning_code_is_documented() -> None:
    """Verify every manifest warning code is documented in docs/operations.md."""
    documented = _read(os.path.join(DOCS_DIR, "operations.md"))
    missing = sorted(
        code.value for code in WarningCode if f"`{code.value}`" not in documented
    )
    assert not missing, f"warning codes absent from docs/operations.md: {missing}"


def test_no_reason_code_is_emitted_outside_the_shared_vocabulary() -> None:
    """Verify reason codes use shared constants so producer and resume logic agree."""
    # Two words are both a reason code and an output column, and the column is
    # legitimately written as a literal dict key. Excluding them costs coverage
    # of those two codes rather than reporting correct code as a defect.
    also_column_names = {code.value for code in ReasonCode} & set(STATS_COLUMNS)
    known = {code.value for code in ReasonCode} - also_column_names
    offenders: list[str] = []
    for module in (
        "worker/lifecycle.py",
        "coordination/analysis.py",
        "coordination/site_summary.py",
        "coordination/dpi.py",
    ):
        source = _read(os.path.join(SRC_DIR, module))
        for literal in re.findall(r'"([a-z][a-z0-9_]{6,})"', source):
            if literal in known:
                offenders.append(f"{module}: {literal!r}")
    assert not offenders, (
        "reason codes written as string literals rather than ReasonCode members: "
        f"{sorted(offenders)}"
    )


def _normalized_distribution_name(name: str) -> str:
    """The PEP 503 form of a distribution name, as wheel metadata spells it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_dependencies() -> set[str]:
    """Distribution names in ``[project.dependencies]``, lowercased."""
    with open(PYPROJECT_PATH, "rb") as handle:
        project = tomllib.load(handle)["project"]
    return {
        re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0].strip().lower()
        for spec in project["dependencies"]
    }


def _documented_dependencies() -> set[str]:
    """Distribution names the README tells a reader to install.

    ``pip install .`` delegates to ``pyproject.toml`` and so covers every
    declared dependency by construction.
    """
    readme = _read(README_PATH)
    documented: set[str] = set()
    for command in re.findall(r"python -m pip install ([^\n]+)", readme):
        if re.match(r"^\.\s*$", command):
            return _declared_dependencies()
        for token in re.findall(r'"([^"]+)"|(\S+)', command):
            spec = token[0] or token[1]
            name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower()
            if name and not name.startswith("-"):
                documented.add(name)
    return documented


def _local_module_names() -> set[str]:
    """Every name ``src/`` itself provides: flat modules and sub-packages.

    ``src`` is on ``sys.path`` rather than being a package, so a sub-package is
    a top-level import name too, and omitting it here would report Alchemy's
    own modules as undeclared distributions.
    """
    local: set[str] = set()
    for name in os.listdir(SRC_DIR):
        path = os.path.join(SRC_DIR, name)
        if name.endswith(".py"):
            local.add(name[:-3])
        elif os.path.isfile(os.path.join(path, "__init__.py")):
            local.add(name)
    return local


def _source_files() -> list[str]:
    """Every ``.py`` file under ``src/``, sub-packages included."""
    found: list[str] = []
    for directory, _subdirs, names in os.walk(SRC_DIR):
        found.extend(
            os.path.join(directory, name) for name in names if name.endswith(".py")
        )
    return sorted(found)


def _third_party_imports() -> set[str]:
    """Distributions imported by ``src/``, whether at module or function scope."""
    found: set[str] = set()
    local = _local_module_names()
    for path in _source_files():
        tree = ast.parse(_read(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]] if not node.level else []
            else:
                continue
            for root in roots:
                if root and root not in _STDLIB and root not in local:
                    found.add(root)
    return found


def test_readme_documents_every_declared_dependency() -> None:
    """Following the README installs everything ``pyproject.toml`` declares."""
    missing = _declared_dependencies() - _documented_dependencies()
    assert not missing, (
        "README setup instructions omit declared dependencies: "
        f"{sorted(missing)}. A reader following them cannot run Alchemy."
    )


def test_declared_dependencies_cover_every_third_party_import() -> None:
    """Nothing in ``src/`` imports a distribution that is not declared."""
    declared = _declared_dependencies()
    undeclared = {
        root
        for root in _third_party_imports()
        if _IMPORT_TO_DISTRIBUTION.get(root, root).lower() not in declared
    }
    assert not undeclared, (
        f"src/ imports undeclared distribution(s): {sorted(undeclared)}. "
        "Add them to [project.dependencies] and to the README."
    )


def test_launcher_docstring_points_to_project_documentation() -> None:
    """Verify the launcher directs readers to existing project documentation."""
    docstring = ast.get_docstring(ast.parse(_read(LAUNCHER_PATH)))
    assert docstring is not None
    for relative_path in ("README.md", "docs/"):
        assert relative_path in docstring
        assert os.path.exists(os.path.join(REPO_ROOT, relative_path))


def test_default_paths_resolve_from_the_checkout_root() -> None:
    """Verify modules resolve REPO_DIR to the project root."""
    import paths

    assert paths.REPO_DIR == REPO_ROOT
    # Read out of the module namespace: what is under test is the value
    # ``driver.confidence`` itself binds, so importing the name from ``paths``
    # to satisfy no-implicit-reexport would assert the line above twice.
    assert vars(driver_confidence)["REPO_DIR"] == REPO_ROOT, (
        "driver.confidence must import REPO_DIR rather than recompute it: two "
        f"dirname calls from {os.path.join(SRC_DIR, 'driver')} name src/, "
        "not the checkout"
    )
    assert (
        os.path.join(REPO_ROOT, "src", "data", "confidence_reference")
        == driver_confidence.DEFAULT_CONFIDENCE_REFERENCE_DIR
    )


@pytest.mark.parametrize(
    "path",
    DOC_PATHS + [LAUNCHER_PATH],
    ids=[
        os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        for path in DOC_PATHS
    ]
    + ["launcher-docstring"],
)
def test_examples_do_not_assume_a_private_environment(path: str) -> None:
    """Verify documented commands do not assume a private environment.

    Alchemy requires no named environment, and readers cannot follow an example
    that depends on one they have no way to create.
    """
    offending = [
        line.strip()
        for line in _read(path).splitlines()
        if re.search(r"conda (run|activate)", line)
    ]
    assert not offending, (
        f"{os.path.basename(path)} documents a specific conda environment: "
        f"{offending}. Use a plain `./alchemy` invocation instead."
    )


def test_setuptools_declares_nothing_installable() -> None:
    """``pyproject.toml`` declares an empty distribution.

    The half of the empty-wheel invariant that needs no build toolchain, so it
    runs everywhere.
    """
    with open(PYPROJECT_PATH, "rb") as handle:
        config = tomllib.load(handle)
    setuptools = config.get("tool", {}).get("setuptools", {})
    assert setuptools.get("packages") == [], (
        "[tool.setuptools] packages must stay empty; otherwise auto-discovery "
        "publishes main, data and six other generic names into site-packages"
    )
    assert setuptools.get("py-modules") == [], (
        "[tool.setuptools] py-modules must stay empty for the same reason"
    )
    assert "packages" not in setuptools.get("dynamic", {}), (
        "dynamic package discovery would defeat the empty declaration"
    )


def test_built_wheel_contains_only_distribution_metadata(tmp_path: Path) -> None:
    """Verify built wheels contain metadata and dependencies without Alchemy modules.

    Build a temporary copy so build artifacts stay outside the repository.
    """
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(SRC_DIR, source / "src")
    for name in ("pyproject.toml", "README.md"):
        shutil.copyfile(os.path.join(REPO_ROOT, name), source / name)

    outdir = tmp_path / "wheel"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(outdir),
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "the wheel build failed, so nothing verified what `pip install .` ships; "
        "the [test] extra installs setuptools and wheel for this check:\n"
        + (completed.stderr or completed.stdout).strip()[-2000:]
    )

    wheels = sorted(outdir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    members = [
        name for name in zipfile.ZipFile(wheels[0]).namelist() if not name.endswith("/")
    ]
    assert members, "wheel is empty; the build produced no metadata at all"

    payload = [
        name for name in members if not re.match(r"^alchemy-[^/]*\.dist-info/", name)
    ]
    assert not payload, (
        "wheel ships importable content outside alchemy-*.dist-info/: "
        f"{sorted(payload)}. Alchemy is run from a clone; `pip install .` must "
        "install dependencies and metadata only."
    )

    metadata = (
        zipfile.ZipFile(wheels[0])
        .read(next(n for n in members if n.endswith(".dist-info/METADATA")))
        .decode()
    )
    required = {
        _normalized_distribution_name(
            re.split(r"[<>=!~; \[]", spec, maxsplit=1)[0].strip()
        )
        for spec in re.findall(r"^Requires-Dist: (.+)$", metadata, re.M)
    }
    declared = {_normalized_distribution_name(n) for n in _declared_dependencies()}
    assert declared <= required, (
        "wheel metadata dropped declared runtime dependencies: "
        f"declared {sorted(declared)}, wheel has {sorted(required)}"
    )


def test_version_has_a_single_definition() -> None:
    """``pyproject.toml`` derives the version from ``src/_version.py``.

    The number is stamped into every manifest row as ``alchemy_version``, so a
    second literal could drift and mislabel a whole database run.
    """
    import _version

    with open(PYPROJECT_PATH, "rb") as handle:
        config = tomllib.load(handle)
    project = config["project"]

    assert "version" not in project, (
        "a literal version in [project] would be a second definition; "
        'declare `dynamic = ["version"]` instead'
    )
    assert "version" in project.get("dynamic", []), (
        "the version must be declared dynamic so it comes from _version.py"
    )
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "_version.__version__"
    }

    assert _version.__version__ == environment.ALCHEMY_VERSION, (
        "driver.environment.ALCHEMY_VERSION must come from _version.py, not a second literal"
    )


def test_every_documentation_link_resolves() -> None:
    """The README is a table of contents, so a moved page breaks a link."""
    broken: list[str] = []
    for path in DOC_PATHS:
        directory = os.path.dirname(path)
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", _read(path)):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not os.path.exists(os.path.join(directory, target)):
                broken.append(f"{os.path.relpath(path, REPO_ROOT)} -> {target}")

    assert not broken, f"documentation links point at missing files: {broken}"


def test_the_readme_points_at_every_documentation_page() -> None:
    """No page is orphaned: each one is reachable from the front door."""
    readme = _read(README_PATH)
    unlinked = [
        name
        for name in sorted(os.listdir(DOCS_DIR))
        if name.endswith(".md") and f"docs/{name}" not in readme
    ]

    assert not unlinked, f"docs/ pages nothing links to: {unlinked}"


def _all_prose() -> str:
    return "".join(_read(path) for path in DOC_PATHS)


def _output_prose() -> str:
    """The pages describing what a run produces.

    ``tests/README.md`` is excluded: it documents the suite, so its backticked
    names are pytest fixtures rather than anything a result file contains.
    """
    return "".join(_read(path) for path in DOC_PATHS if "tests" not in path)


def _declared_cli_flags() -> set[str]:
    """Every ``--flag`` ``src/cli.py`` accepts, read from the source.

    The parser is built inside ``parse_args`` and not returned, so it is read
    statically rather than constructed.
    """
    tree = ast.parse(_read(os.path.join(SRC_DIR, "cli.py")))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == (
            "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    flags.add(str(arg.value))
    return flags


def test_every_cli_flag_appears_in_the_prose() -> None:
    """Verify every CLI flag appears in the documentation."""
    prose = _all_prose()
    undocumented = sorted(flag for flag in _declared_cli_flags() if flag not in prose)

    assert not undocumented, (
        f"CLI flags documented nowhere: {undocumented}. Add them to "
        "docs/usage.md, or to the page that owns the behaviour."
    )


# Backticked snake_case words in the prose that name something other than an
# output field or a status code: a formula symbol, an mmCIF item, a function,
# and the informational subset of ``ok``. Listed so the check below can be
# strict about everything else.
_NON_FIELD_TERMS = frozenset(
    (
        "confidence_score",
        "sigma_lit",
        "label_seq_id",
        "extract_metal_statistics",
        "exptl_crystal_grow",
        "manual_coordinate_file",
        "pdb_redo_coordinate_file",
        "rcsb_data_api",
    )
)


def test_every_field_name_in_the_prose_still_exists() -> None:
    """Verify documented output fields and status codes still exist."""
    from enum import StrEnum

    import codes as codes_module
    from confidence_score import (
        ANALYSIS_COLUMNS,
        CONFIDENCE_INPUT_COLUMNS,
        CONFIDENCE_INPUT_STATUSES,
        EVIDENCE_BASES,
        REFERENCE_METADATA_FIELDS,
        VERDICT_REASONS,
    )
    from coordination.schema import BOND_COLUMNS, CANDIDATE_COLUMNS
    from crystallization_conditions import (
        CONDITION_COLUMNS,
        CRYSTALLIZATION_DATA_STATUSES,
        SUMMARY_COLUMNS,
    )
    from driver.review_queue import REVIEW_CONTEXT_COLUMNS
    from driver.writers import MANIFEST_COLUMNS
    from edstats_statistics import (
        DENSITY_CONTEXT_COLUMNS,
        DENSITY_CONTEXT_STATUSES,
    )

    known: set[str] = set()
    for columns in (
        MANIFEST_COLUMNS,
        STATS_COLUMNS,
        BOND_COLUMNS,
        CANDIDATE_COLUMNS,
        CONFIDENCE_INPUT_COLUMNS,
        ANALYSIS_COLUMNS,
        CONDITION_COLUMNS,
        SUMMARY_COLUMNS,
        REVIEW_CONTEXT_COLUMNS,
        DENSITY_CONTEXT_COLUMNS,
        ENTRY_DIAGNOSTIC_BASE_COLUMNS,
        PREFERRED_TIMING_COLUMNS,
        ENTRY_DIAGNOSTIC_TRAILING_COLUMNS,
    ):
        known |= set(columns)
    known |= CONFIDENCE_INPUT_STATUSES | REFERENCE_METADATA_FIELDS
    known |= EVIDENCE_BASES | VERDICT_REASONS
    known |= CRYSTALLIZATION_DATA_STATUSES
    known |= DENSITY_CONTEXT_STATUSES
    for name in dir(codes_module):
        candidate = getattr(codes_module, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, StrEnum)
            and candidate is not StrEnum
        ):
            known |= {member.value for member in candidate}

    cited = {
        term
        for term in re.findall(r"`([a-z][a-z0-9_]{6,})`", _output_prose())
        if "_" in term
    }
    unknown = sorted(cited - known - _NON_FIELD_TERMS)

    assert not unknown, (
        f"the prose names fields or codes that no longer exist: {unknown}. "
        "Rename them, or add genuinely non-field terms to _NON_FIELD_TERMS."
    )


def test_documented_thresholds_match_the_constants() -> None:
    """Verify documented thresholds match the constants used by the code."""
    from coordination.policy import (
        CANDIDATE_SEARCH_RADIUS,
        FIRST_SPHERE_TOLERANCE,
        ZSCORE_OUTLIER_CUTOFF,
    )
    from density_analysis import (
        CCP4_TOOL_TIMEOUT_S,
        MODEL_ENVELOPE_BORDER_ANGSTROM,
    )
    from driver.resources import AUTO_WORKER_MEMORY_BYTES
    from structure_analysis import OVERFULL_OCCUPANCY_NI_FRACTION

    prose = _all_prose()
    documented: list[tuple[str, int | float, str]] = [
        ("broad candidate search radius", CANDIDATE_SEARCH_RADIUS, "4 Å"),
        ("z-score outlier cutoff", ZSCORE_OUTLIER_CUTOFF, ">= 6"),
        ("first-sphere tolerance", FIRST_SPHERE_TOLERANCE, "0.75"),
        ("model-envelope border", MODEL_ENVELOPE_BORDER_ANGSTROM, "10 Angstrom"),
        ("per-program CCP4 budget", CCP4_TOOL_TIMEOUT_S, "900"),
        ("per-worker memory budget", AUTO_WORKER_MEMORY_BYTES, "2 GiB"),
        ("overfull occupancy fraction", OVERFULL_OCCUPANCY_NI_FRACTION, "0.2%"),
    ]

    missing = [
        f"{label} ({value}) is documented as {text!r}, which no page contains"
        for label, value, text in documented
        if text not in prose
    ]
    assert not missing, missing

    assert (CANDIDATE_SEARCH_RADIUS, ZSCORE_OUTLIER_CUTOFF, FIRST_SPHERE_TOLERANCE) == (
        4.0,
        6.0,
        0.75,
    )
    assert (MODEL_ENVELOPE_BORDER_ANGSTROM, CCP4_TOOL_TIMEOUT_S) == (10, 15 * 60)
    assert AUTO_WORKER_MEMORY_BYTES == 2 * 1024**3
    assert OVERFULL_OCCUPANCY_NI_FRACTION == 0.002
