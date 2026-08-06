"""Keep the setup instructions honest.

Scope: the contract between three places that each describe the same
environment -- ``pyproject.toml``, the README, and ``src/main.py``'s module
docstring. The instructions are compared against the declaration, and the
declaration against the code, since a correct declaration cannot catch
instructions that install less than it declares.

Out of scope here (owned elsewhere): argument parsing (``test_cli_and_config``)
and the pipeline itself (``test_pipeline_integration``).
"""

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
from codes import ReasonCode
from driver import pool
from driver.writers import STATS_COLUMNS


README_PATH = os.path.join(REPO_ROOT, "README.md")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
# Every prose file a reader is sent to.
DOC_PATHS = [README_PATH, os.path.join(REPO_ROOT, "tests", "README.md")] + [
    os.path.join(DOCS_DIR, name)
    for name in ("usage.md", "method.md", "operations.md", "maintenance.md")
]
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")

_STDLIB = set(sys.stdlib_module_names)

# Import name -> distribution name, spelled out so the mapping does not depend
# on what happens to be installed.
_IMPORT_TO_DISTRIBUTION = {"gemmi": "gemmi", "numpy": "numpy"}


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_every_reason_code_is_documented() -> None:
    """The manifest vocabulary is the most user-visible text Alchemy writes.

    ``docs/operations.md`` is where an operator looks a code up, so a code that
    reaches the manifest without an entry there is undocumented in the only
    place it matters. Checking the enum rather than the source also means a
    renamed code cannot quietly leave the documentation describing the old one.
    """
    documented = _read(os.path.join(DOCS_DIR, "operations.md"))
    missing = sorted(
        code.value for code in ReasonCode if f"`{code.value}`" not in documented
    )
    assert not missing, f"reason codes absent from docs/operations.md: {missing}"


def test_no_reason_code_is_emitted_outside_the_shared_vocabulary() -> None:
    """Reason codes are assigned from ``ReasonCode``, not spelled out again.

    ``driver.resume`` reads one of these back to decide whether a bond stage was
    applicable, so a literal at one end and a constant at the other would let a
    rename reprocess those entries on every resume with nothing to catch it.
    """
    # Two words are both a reason code and an output column, and the column is
    # legitimately written as a literal dict key. Excluding them costs coverage
    # of those two codes rather than reporting correct code as a defect.
    also_column_names = {code.value for code in ReasonCode} & set(STATS_COLUMNS)
    known = {code.value for code in ReasonCode} - also_column_names
    offenders: list[str] = []
    for module in ("worker.py", "coordination/analysis.py", "coordination/dpi.py"):
        source = _read(os.path.join(SRC_DIR, module))
        for literal in re.findall(r'"([a-z][a-z0-9_]{6,})"', source):
            if literal in known:
                offenders.append(f"{module}: {literal!r}")
    assert not offenders, (
        "reason codes written as string literals rather than ReasonCode members: "
        f"{sorted(offenders)}"
    )


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


def test_main_docstring_requirements_match_the_declaration() -> None:
    """``src/main.py``'s Requirements section names every declared dependency.

    That docstring is the third copy of the list, and this is what holds it to
    the other two.
    """
    import main

    requirements = (main.__doc__ or "").lower()
    missing = [name for name in _declared_dependencies() if name not in requirements]
    assert not missing, (
        f"main.py's module docstring omits declared dependencies: {missing}"
    )


def test_default_paths_land_in_the_checkout_not_inside_src() -> None:
    """``REPO_DIR`` names the checkout root from every module that reads it.

    It walks up from ``__file__``, so the number of ``dirname`` calls is right
    only for a module sitting directly in ``src/``. Recomputed one directory
    deeper it resolves to ``src/``, nothing raises, and the run writes its
    output into the source tree.
    """
    import ccp4_setup
    from driver import pool

    assert ccp4_setup.REPO_DIR == REPO_ROOT
    # Read out of the module namespace: what is under test is the value
    # ``driver.pool`` itself binds, so importing the name from ``ccp4_setup``
    # to satisfy no-implicit-reexport would assert the line above twice.
    assert vars(pool)["REPO_DIR"] == REPO_ROOT, (
        "driver.pool must import REPO_DIR rather than recompute it: two "
        f"dirname calls from {os.path.join(SRC_DIR, 'driver')} name src/, "
        "not the checkout"
    )
    assert pool.DEFAULT_CONFIDENCE_REFERENCE_DIR == os.path.join(
        REPO_ROOT, "confidence_reference"
    )


def test_the_density_cli_does_not_default_its_output_into_the_source_tree() -> None:
    """``density_analysis.py``'s own CLI must not write where the code lives.

    It is a debugging entry point documented nowhere, and it once defaulted
    ``--out-dir`` to this file's directory, so a run without the flag dropped
    FFT maps, mapmask output and EDSTATS logs into ``src/`` -- which
    ``.gitignore`` does not cover. Read from the source because the parser is
    built under ``if __name__ == "__main__"`` and never imported.
    """
    tree = ast.parse(_read(os.path.join(SRC_DIR, "density_analysis.py")))
    defaults = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "add_argument"
        and any(
            isinstance(arg, ast.Constant) and arg.value == "--out-dir"
            for arg in node.args
        )
        for keyword in node.keywords
        if keyword.arg == "default"
    ]

    assert len(defaults) == 1, "expected exactly one --out-dir default"
    default = defaults[0]
    assert isinstance(default, ast.Constant), (
        "--out-dir must default to a literal path, not to a module constant "
        "derived from __file__"
    )
    default_path = default.value
    assert isinstance(default_path, str), (
        f"--out-dir default is not a path: {default!r}"
    )
    assert not os.path.isabs(default_path), default_path


@pytest.mark.parametrize(
    "path",
    DOC_PATHS + [os.path.join(SRC_DIR, "main.py")],
    ids=[
        os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        for path in DOC_PATHS
    ]
    + ["main-docstring"],
)
def test_examples_do_not_assume_a_private_environment(path: str) -> None:
    """Documented commands run anywhere: Alchemy requires no named environment,
    and one a reader has no way to create is one they cannot follow."""
    offending = [
        line.strip()
        for line in _read(path).splitlines()
        if re.search(r"conda (run|activate)", line)
    ]
    assert not offending, (
        f"{os.path.basename(path)} documents a specific conda environment: "
        f"{offending}. Use a plain `python src/main.py` invocation instead."
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
    """A built wheel ships metadata and dependencies, but no Alchemy code.

    Nothing else detects the loss of that invariant: the suite reaches ``src``
    through ``conftest.py``'s ``sys.path`` insertion and CI installs editable,
    so both stay green even if modules are published again.

    Built from a copy because ``pip wheel`` leaves ``build/`` and
    ``*.egg-info/`` beside the sources it builds, and this suite writes nothing
    inside the repository.
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
    if completed.returncode != 0:
        pytest.skip(
            "no usable wheel-build toolchain in this environment: "
            + (completed.stderr or completed.stdout).strip()[:200]
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
        re.split(r"[<>=!~; \[]", spec, maxsplit=1)[0].strip().lower()
        for spec in re.findall(r"^Requires-Dist: (.+)$", metadata, re.M)
    }
    assert _declared_dependencies() <= required, (
        "wheel metadata dropped declared runtime dependencies: "
        f"declared {sorted(_declared_dependencies())}, wheel has {sorted(required)}"
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

    assert pool.ALCHEMY_VERSION == _version.__version__, (
        "driver.pool.ALCHEMY_VERSION must come from _version.py, not a second literal"
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
    """A flag nobody documents is a flag nobody can find.

    ``--confidence-reference-dir`` was absent from every page for exactly as
    long as nothing checked, and it is the flag that enables confidence scoring
    at all. Reading the flags from the parser means adding one without a
    sentence about it fails here rather than in a user's search.
    """
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
    ("sigma_lit", "label_seq_id", "extract_metal_statistics", "no_metals")
)


def test_every_field_name_in_the_prose_still_exists() -> None:
    """A renamed column must not leave the documentation describing the old one.

    The prose names specific fields and codes throughout, and a reader looks
    them up in an output file. Checking the direction that matters -- that
    everything named still exists -- catches a rename without demanding that
    all 134 statistics columns be documented.
    """
    from enum import StrEnum

    import codes as codes_module
    from confidence_score import ANALYSIS_COLUMNS, CONFIDENCE_INPUT_COLUMNS
    from coordination.schema import BOND_COLUMNS, CANDIDATE_COLUMNS
    from driver.writers import MANIFEST_COLUMNS

    known: set[str] = set()
    for columns in (
        MANIFEST_COLUMNS,
        STATS_COLUMNS,
        BOND_COLUMNS,
        CANDIDATE_COLUMNS,
        CONFIDENCE_INPUT_COLUMNS,
        ANALYSIS_COLUMNS,
    ):
        known |= set(columns)
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
    """Numbers in the prose are the ones the code actually uses.

    Every one of these was verified by hand once. Pinning them means a
    threshold cannot be tuned in ``src/`` while the documentation goes on
    quoting the old value, which is the failure mode that leaves a reader
    confidently wrong about what a run did.
    """
    from coordination.analysis import CANDIDATE_SEARCH_RADIUS, FIRST_SPHERE_TOLERANCE
    from coordination.schema import ZSCORE_OUTLIER_CUTOFF
    from density_analysis import CCP4_TOOL_TIMEOUT_S, MODEL_ENVELOPE_BORDER_ANGSTROM
    from driver.resources import AUTO_WORKER_MEMORY_BYTES
    from structure_analysis import OVERFULL_OCCUPANCY_NI_FRACTION

    prose = _all_prose()
    documented: list[tuple[str, int | float, str]] = [
        ("broad candidate search radius", CANDIDATE_SEARCH_RADIUS, "4 Å"),
        ("z-score outlier cutoff", ZSCORE_OUTLIER_CUTOFF, ">= 6"),
        ("first-sphere tolerance", FIRST_SPHERE_TOLERANCE, "0.75"),
        ("model-envelope border", MODEL_ENVELOPE_BORDER_ANGSTROM, "10 Angstrom"),
        ("per-program CCP4 budget", CCP4_TOOL_TIMEOUT_S, "900"),
        ("per-worker memory budget", AUTO_WORKER_MEMORY_BYTES, "1.25 GiB"),
        ("overfull occupancy fraction", OVERFULL_OCCUPANCY_NI_FRACTION, "0.2%"),
    ]

    missing = [
        f"{label} ({value}) is documented as {text!r}, which no page contains"
        for label, value, text in documented
        if text not in prose
    ]
    assert not missing, missing

    # The constants themselves, so a change to one is visible here as well as
    # in the prose that quotes it.
    assert (CANDIDATE_SEARCH_RADIUS, ZSCORE_OUTLIER_CUTOFF, FIRST_SPHERE_TOLERANCE) == (
        4.0,
        6.0,
        0.75,
    )
    assert (MODEL_ENVELOPE_BORDER_ANGSTROM, CCP4_TOOL_TIMEOUT_S) == (10, 15 * 60)
    assert AUTO_WORKER_MEMORY_BYTES == 1280 * 1024 * 1024
    assert OVERFULL_OCCUPANCY_NI_FRACTION == 0.002
