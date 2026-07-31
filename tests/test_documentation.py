"""Keep the setup instructions honest.

Scope: the documented way to get Alchemy running must actually work. These are
not tests of behaviour but of the contract between three places that each
describe the same environment -- ``pyproject.toml``, the README, and
``src/main.py``'s module docstring -- and which drifted apart once already.

Out of scope here (owned elsewhere): argument parsing (``test_cli_and_config``)
and the pipeline itself (``test_pipeline_integration``).

Regression: the README instructed ``pip install "gemmi>=0.7.0"`` while
``density_analysis`` imports numpy at module scope, so a reader who followed the
documented setup exactly got ``ImportError: No module named 'numpy'`` from their
first command. A test that merely imported ``main`` under the *declared*
dependencies could not have caught this -- numpy was declared correctly all
along; only the instructions were wrong. These tests therefore compare the
instructions against the declaration, and the declaration against the code.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import tomllib
from typing import Set

import pytest

from helpers import REPO_ROOT, SRC_DIR


README_PATH = os.path.join(REPO_ROOT, "README.md")
PYPROJECT_PATH = os.path.join(REPO_ROOT, "pyproject.toml")

#: Distributions that ship with CPython and so are never declared.
_STDLIB = set(sys.stdlib_module_names)

#: Import name -> distribution name, where the two differ. Kept explicit rather
#: than resolved at runtime so the test does not depend on what is installed.
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

    Recognises both the explicit ``pip install "pkg>=x"`` form and
    ``pip install .``, which delegates to ``pyproject.toml`` and therefore
    covers every declared dependency by construction.
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


def _third_party_imports() -> Set[str]:
    """Distributions imported by ``src/``, whether at module or function scope."""
    found: Set[str] = set()
    local = {
        name[:-3]
        for name in os.listdir(SRC_DIR)
        if name.endswith(".py")
    }
    for name in sorted(os.listdir(SRC_DIR)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(_read(os.path.join(SRC_DIR, name)))
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
    """Following the README installs everything ``pyproject.toml`` declares.

    Regression: the README listed only gemmi, so numpy -- declared, and imported
    at module scope by ``density_analysis`` -- was never installed.
    """
    missing = _declared_dependencies() - _documented_dependencies()
    assert not missing, (
        "README setup instructions omit declared dependencies: "
        f"{sorted(missing)}. A reader following them cannot run Alchemy."
    )


def test_declared_dependencies_cover_every_third_party_import():
    """Nothing in ``src/`` imports a distribution that is not declared."""
    declared = _declared_dependencies()
    undeclared = {
        root for root in _third_party_imports()
        if _IMPORT_TO_DISTRIBUTION.get(root, root).lower() not in declared
    }
    assert not undeclared, (
        f"src/ imports undeclared distribution(s): {sorted(undeclared)}. "
        "Add them to [project.dependencies] and to the README."
    )


def test_main_docstring_requirements_match_the_declaration():
    """``src/main.py``'s Requirements section names every declared dependency.

    It is the third copy of the same list and drifts as easily as the README.
    """
    import main

    requirements = (main.__doc__ or "").lower()
    missing = [
        name for name in _declared_dependencies() if name not in requirements
    ]
    assert not missing, (
        f"main.py's module docstring omits declared dependencies: {missing}"
    )


@pytest.mark.parametrize(
    "path",
    [README_PATH, os.path.join(SRC_DIR, "main.py")],
    ids=["readme", "main-docstring"],
)
def test_examples_do_not_assume_a_private_environment(path):
    """Documented commands run anywhere, not only on the author's machine.

    Regression: every Quick Start example invoked ``conda run -n metal``. That
    environment name appeared in six documentation lines and nowhere else in the
    repository -- no code, no tests, no CI -- so a reader was told to use an
    environment that only existed on one machine, with no way to create it.
    Alchemy requires no conda environment at all.
    """
    offending = [
        line.strip()
        for line in _read(path).splitlines()
        if re.search(r"conda (run|activate)", line)
    ]
    assert not offending, (
        f"{os.path.basename(path)} documents a specific conda environment: "
        f"{offending}. Use a plain `python src/main.py` invocation instead."
    )
