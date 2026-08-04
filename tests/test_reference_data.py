"""The bundled reference data: when it is read, and what the caller gets back.

Scope: ``src/reference_data.py`` as a loading policy, not as crystallography.
Whether a particular reference distance is right belongs to
``test_bond_geometry``; whether reading it costs an import, can be mutated by
one caller for the whole process, or can be simultaneously valid and invalid
depending on which module asked, belongs here.

Regression: ``import bond_analysis`` used to read both bundled files. Every
``--help``, every test-collection pass and every spawned worker paid for them,
a malformed catalog raised ``ValueError`` out of an import statement, the
results were mutable dicts shared process-wide, and the catalog was parsed
twice by two parsers that disagreed about what a valid catalog was.
"""

from __future__ import annotations

import subprocess
import sys
from types import MappingProxyType
from typing import Any

import pytest

import reference_data
from helpers import SRC_DIR


# The modules that used to read a file merely by being imported.
_PREVIOUSLY_EAGER = ("bond_analysis", "metal_identification", "reference_data")


def _catalog(tmp_path, lines, name="catalog.txt"):
    path = tmp_path / name
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# No import-time I/O
# --------------------------------------------------------------------------- #
def test_importing_the_analysis_modules_reads_no_files():
    """Importing must not touch the disk, in a fresh interpreter.

    Run in a subprocess because this test process has already imported all
    three: a second ``import`` is a dict lookup and would pass no matter what
    the module body does.
    """
    program = f"""
import builtins, sys
sys.path.insert(0, {SRC_DIR!r})
opened = []
real_open = builtins.open
def spy(*args, **kwargs):
    opened.append(args[0] if args else kwargs.get("file"))
    return real_open(*args, **kwargs)
builtins.open = spy
for name in {list(_PREVIOUSLY_EAGER)!r}:
    __import__(name)
builtins.open = real_open
print("|".join(str(path) for path in opened))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    opened = [path for path in completed.stdout.strip().split("|") if path]
    assert not opened, (
        f"importing {', '.join(_PREVIOUSLY_EAGER)} read {opened}. Bundled data "
        "must be loaded on demand: --help, test collection and every spawned "
        "worker pay for anything done at import."
    )


def test_a_malformed_catalog_raises_from_the_call_not_the_import(tmp_path):
    """The failure has to be attributable to a caller.

    Raised out of an import there is no context to report it with -- the
    traceback names an import statement in whichever module happened to be
    loaded first.
    """
    path = _catalog(tmp_path, ["ABC\t'C1'\t", "DEF\t'C2'\t"])
    with pytest.raises(ValueError, match="no structural classes"):
        reference_data.cofactor_ids(path)


# --------------------------------------------------------------------------- #
# One parser
# --------------------------------------------------------------------------- #
def test_ids_and_classes_come_from_one_pass_over_one_file(tmp_path):
    """Every classified component is also a known component.

    Regression: two parsers read this file under different rules -- one
    required three tab-separated fields, the other accepted a single column --
    so a legacy catalog loaded cleanly in ``metal_identification`` and
    hard-failed at import in ``bond_analysis``.
    """
    path = _catalog(
        tmp_path,
        ["SF4\t'Fe4 S4'\tcluster", "HEM\t'C34'\theme", "ZN\t'Zn'\t"],
    )
    assert reference_data.cofactor_ids(path) == {"SF4", "HEM", "ZN"}
    assert reference_data.cluster_ids(path) == {"SF4"}
    assert reference_data.heme_ids(path) == {"HEM"}
    assert reference_data.cluster_ids(path) <= reference_data.cofactor_ids(path)
    assert reference_data.heme_ids(path) <= reference_data.cofactor_ids(path)


def test_a_short_row_is_a_component_but_never_a_classification(tmp_path):
    """A legacy row without a class column contributes an id and nothing else.

    The two-parser era is what makes this worth pinning: a short row was a
    valid component to one reader and skipped entirely by the other. Under one
    parser it has to be exactly one thing -- known, unclassified -- because
    guessing a class from a missing column would put a component into
    ``parent_type`` on no evidence at all.
    """
    path = _catalog(
        tmp_path,
        ["SF4\t'Fe4 S4'\tcluster", "HEM\t'C34'\theme", "OLD\t'C2'", "BARE"],
    )
    assert reference_data.cofactor_ids(path) == {"SF4", "HEM", "OLD", "BARE"}
    assert reference_data.cluster_ids(path) == {"SF4"}
    assert reference_data.heme_ids(path) == {"HEM"}


@pytest.mark.parametrize(
    "accessor", ["cofactor_ids", "cluster_ids", "heme_ids"], ids=str
)
def test_every_accessor_rejects_the_same_bad_catalog(tmp_path, accessor):
    """One set of rules, so a catalog cannot be valid to one caller only."""
    path = _catalog(tmp_path, ["ABC\t'C1'\t"])
    with pytest.raises(ValueError):
        getattr(reference_data, accessor)(path)


def test_an_empty_catalog_is_named_as_empty(tmp_path):
    """The two failures stay distinguishable: empty is not unclassified."""
    with pytest.raises(ValueError, match="catalog is empty"):
        reference_data.cofactor_ids(_catalog(tmp_path, ["", "   "]))


# --------------------------------------------------------------------------- #
# Frozen, and read once
# --------------------------------------------------------------------------- #
def test_the_loaded_data_cannot_be_edited_by_one_caller(tmp_path):
    """These objects are process-wide and shared by every worker.

    A mutable dict handed to every caller means one of them can change what
    every later z-score is measured against, with nothing in the output saying
    the reference moved.
    """
    assert isinstance(reference_data.cofactor_ids(), frozenset)
    assert isinstance(reference_data.cluster_ids(), frozenset)
    assert isinstance(reference_data.heme_ids(), frozenset)
    assert isinstance(reference_data.literature_distances(), MappingProxyType)
    assert isinstance(reference_data.first_sphere_targets(), MappingProxyType)

    # Bound through ``Any`` deliberately. A type checker rejects assignment
    # into a ``MappingProxyType`` outright, which is the guarantee under test:
    # the point here is that it also fails at runtime, for a caller who never
    # ran one.
    distances: Any = reference_data.literature_distances()
    targets: Any = reference_data.first_sphere_targets()
    with pytest.raises(TypeError):
        distances[("HIS", "N", "ZN")] = (0.0, 0.0)
    with pytest.raises(TypeError):
        targets[("ZN", "N")] = 0.0


def test_a_file_is_read_once_however_many_callers_ask(tmp_path, monkeypatch):
    """Caching is what makes on-demand loading affordable.

    Without it, moving the read out of the import would trade one read per
    process for one per call -- and ``literature_distances`` is consulted for
    every contact of every metal site.
    """
    path = _catalog(tmp_path, ["SF4\t'Fe4 S4'\tcluster", "HEM\t'C34'\theme"])
    reads = []
    real_open = open

    def counting_open(file, *args, **kwargs):
        reads.append(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    for _ in range(3):
        reference_data.cofactor_ids(path)
        reference_data.cluster_ids(path)
        reference_data.heme_ids(path)
    assert reads.count(path) == 1


# --------------------------------------------------------------------------- #
# first_sphere_targets
# --------------------------------------------------------------------------- #
def test_first_sphere_targets_is_the_longest_distance_per_metal_and_donor():
    """It is exactly ``max`` over the literature table, keyed the other way."""
    expected: dict[tuple[str, str], float] = {}
    for (_residue, donor, metal), (
        mu,
        _stdev,
    ) in reference_data.literature_distances().items():
        key = (metal, donor)
        expected[key] = max(mu, expected.get(key, float("-inf")))
    assert dict(reference_data.first_sphere_targets()) == expected


def test_building_the_targets_leaves_nothing_in_the_module_namespace():
    """Regression: a module-scope ``for`` loop built this and leaked four names.

    ``donor``, ``metal_element``, ``target`` and ``key`` stayed bound in
    ``bond_analysis`` afterwards, where they read as module constants.
    """
    import bond_analysis

    leaked = [
        name
        for name in ("donor", "metal_element", "target", "key")
        if hasattr(reference_data, name) or hasattr(bond_analysis, name)
    ]
    assert not leaked, f"loop variables left in a module namespace: {leaked}"
