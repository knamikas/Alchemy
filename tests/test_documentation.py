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
from typing import Set

import pytest

from helpers import REPO_ROOT, SRC_DIR
from driver import pool


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


def _declared_dependencies() -> Set[str]:
    """Distribution names in ``[project.dependencies]``, lowercased."""
    with open(PYPROJECT_PATH, "rb") as handle:
        project = tomllib.load(handle)["project"]
    return {
        re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0].strip().lower()
        for spec in project["dependencies"]
    }


def _documented_dependencies() -> Set[str]:
    """Distribution names the README tells a reader to install.

    ``pip install .`` delegates to ``pyproject.toml`` and so covers every
    declared dependency by construction.
    """
    readme = _read(README_PATH)
    documented: Set[str] = set()
    for command in re.findall(r"python -m pip install ([^\n]+)", readme):
        if re.match(r"^\.\s*$", command):
            return _declared_dependencies()
        for token in re.findall(r'"([^"]+)"|(\S+)', command):
            spec = token[0] or token[1]
            name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower()
            if name and not name.startswith("-"):
                documented.add(name)
    return documented


def _local_module_names() -> Set[str]:
    """Every name ``src/`` itself provides: flat modules and sub-packages.

    ``src`` is on ``sys.path`` rather than being a package, so a sub-package is
    a top-level import name too, and omitting it here would report Alchemy's
    own modules as undeclared distributions.
    """
    local: Set[str] = set()
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


def _third_party_imports() -> Set[str]:
    """Distributions imported by ``src/``, whether at module or function scope."""
    found: Set[str] = set()
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


def test_readme_documents_every_declared_dependency():
    """Following the README installs everything ``pyproject.toml`` declares."""
    missing = _declared_dependencies() - _documented_dependencies()
    assert not missing, (
        "README setup instructions omit declared dependencies: "
        f"{sorted(missing)}. A reader following them cannot run Alchemy."
    )


def test_declared_dependencies_cover_every_third_party_import():
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


def test_main_docstring_requirements_match_the_declaration():
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


def test_default_paths_land_in_the_checkout_not_inside_src():
    """``REPO_DIR`` names the checkout root from every module that reads it.

    It walks up from ``__file__``, so the number of ``dirname`` calls is right
    only for a module sitting directly in ``src/``. Recomputed one directory
    deeper it resolves to ``src/``, nothing raises, and the run writes its
    output into the source tree.
    """
    import ccp4_setup
    from driver import pool

    assert ccp4_setup.REPO_DIR == REPO_ROOT
    assert pool.REPO_DIR == REPO_ROOT, (
        "driver.pool must import REPO_DIR rather than recompute it: two "
        f"dirname calls from {os.path.join(SRC_DIR, 'driver')} name src/, "
        "not the checkout"
    )
    assert pool.DEFAULT_CONFIDENCE_REFERENCE_DIR == os.path.join(
        REPO_ROOT, "confidence_reference"
    )


@pytest.mark.parametrize(
    "path",
    DOC_PATHS + [os.path.join(SRC_DIR, "main.py")],
    ids=[
        os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        for path in DOC_PATHS
    ]
    + ["main-docstring"],
)
def test_examples_do_not_assume_a_private_environment(path):
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


def test_setuptools_declares_nothing_installable():
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


def test_built_wheel_contains_only_distribution_metadata(tmp_path):
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


def test_version_has_a_single_definition():
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


def test_every_documentation_link_resolves():
    """The README is a table of contents, so a moved page breaks a link."""
    broken = []
    for path in DOC_PATHS:
        directory = os.path.dirname(path)
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", _read(path)):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not os.path.exists(os.path.join(directory, target)):
                broken.append(f"{os.path.relpath(path, REPO_ROOT)} -> {target}")

    assert not broken, f"documentation links point at missing files: {broken}"


def test_the_readme_points_at_every_documentation_page():
    """No page is orphaned: each one is reachable from the front door."""
    readme = _read(README_PATH)
    unlinked = [
        name
        for name in sorted(os.listdir(DOCS_DIR))
        if name.endswith(".md") and f"docs/{name}" not in readme
    ]

    assert not unlinked, f"docs/ pages nothing links to: {unlinked}"
