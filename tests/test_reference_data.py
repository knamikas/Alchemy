"""The bundled reference data: when it is read, and what the caller gets back.

Scope: ``src/reference_data.py`` as a loading policy, not as crystallography.
Whether a particular reference distance is right belongs to
``test_bond_geometry``; whether reading it costs an import, can be mutated by
one caller for the whole process, or can be simultaneously valid and invalid
depending on which module asked, belongs here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import MappingProxyType
from typing import Any

import pytest

import reference_data
from helpers import SRC_DIR


_MUST_NOT_READ_AT_IMPORT = (
    "coordination.analysis",
    "metal_identification",
    "reference_data",
)


def _clear_reference_data_caches():
    """Drop every memoized read in ``reference_data``."""
    for cached in (
        reference_data.reference_data_checksums,
        reference_data.reference_data_id,
        reference_data._catalog,
        reference_data.literature_distances,
        reference_data.first_sphere_targets,
    ):
        cached.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_reference_data_caches():
    """Clear the memoized reads around every test in this module.

    ``reference_data_checksums`` and ``reference_data_id`` take no arguments, so
    a value computed from stubbed files outlives the ``monkeypatch`` that
    produced it: the patched ``CHECKSUM_SIDECARS`` is restored while the value
    derived from it is not. Clearing only before use therefore left whichever
    identity the last stubbing test computed visible for the rest of the
    session, which failed unrelated tests in other modules and made this file's
    own results depend on collection order.
    """
    _clear_reference_data_caches()
    yield
    _clear_reference_data_caches()


def _catalog(tmp_path, lines, name="catalog.txt"):
    path = tmp_path / name
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return str(path)


def test_importing_the_analysis_modules_reads_no_files():
    """Importing must not touch the disk.

    Run in a subprocess: this process has already imported all three, so a
    second ``import`` is a dict lookup that passes whatever the module does.
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
for name in {list(_MUST_NOT_READ_AT_IMPORT)!r}:
    __import__(name)
builtins.open = real_open
print("|".join(str(path) for path in opened))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    opened = [path for path in completed.stdout.strip().split("|") if path]
    assert not opened, (
        f"importing {', '.join(_MUST_NOT_READ_AT_IMPORT)} read {opened}. Bundled data "
        "must be loaded on demand: --help, test collection and every spawned "
        "worker pay for anything done at import."
    )


def test_a_malformed_catalog_raises_from_the_call_not_the_import(tmp_path):
    """The failure has to be attributable to a caller: raised out of an import,
    the traceback names whichever module happened to be loaded first."""
    path = _catalog(tmp_path, ["ABC\t'C1'\t", "DEF\t'C2'\t"])
    with pytest.raises(ValueError, match="no structural classes"):
        reference_data.cofactor_ids(path)


def test_ids_and_classes_come_from_one_pass_over_one_file(tmp_path):
    """Every classified component is also a known component."""
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
    """A row without a class column is known and unclassified, nothing else:
    guessing a class would fill ``parent_type`` on no evidence at all."""
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


def test_the_loaded_data_cannot_be_edited_by_one_caller(tmp_path):
    """These objects are process-wide, so a mutable one would let a single
    caller change what every later z-score is measured against."""
    assert isinstance(reference_data.cofactor_ids(), frozenset)
    assert isinstance(reference_data.cluster_ids(), frozenset)
    assert isinstance(reference_data.heme_ids(), frozenset)
    assert isinstance(reference_data.literature_distances(), MappingProxyType)
    assert isinstance(reference_data.first_sphere_targets(), MappingProxyType)

    # Bound through ``Any``: mypy rejects assignment into a MappingProxyType
    # outright, and what is under test is that it also fails at runtime.
    distances: Any = reference_data.literature_distances()
    targets: Any = reference_data.first_sphere_targets()
    with pytest.raises(TypeError):
        distances[("HIS", "N", "ZN")] = (0.0, 0.0)
    with pytest.raises(TypeError):
        targets[("ZN", "N")] = 0.0


def test_a_file_is_read_once_however_many_callers_ask(tmp_path, monkeypatch):
    """Caching is what makes on-demand loading affordable: the tables are
    consulted for every contact of every metal site."""
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
    """A loop variable left bound at module scope reads as a module constant."""
    from coordination import analysis as coordination_analysis

    leaked = [
        name
        for name in ("donor", "metal_element", "target", "key")
        if hasattr(reference_data, name) or hasattr(coordination_analysis, name)
    ]
    assert not leaked, f"loop variables left in a module namespace: {leaked}"


def test_both_bundled_files_match_their_recorded_checksums():
    """The shipped data is the data the sidecars record: identity, not
    correctness. A checkout that has drifted produces unattributable results."""
    for path, (sidecar, key) in reference_data.CHECKSUM_SIDECARS.items():
        with open(sidecar, encoding="utf-8") as handle:
            recorded = json.load(handle)[key]
        assert reference_data._sha256(path) == recorded, (
            f"{os.path.basename(path)} does not match {os.path.basename(sidecar)}; "
            "rebuild it with its tool or re-stamp the sidecar"
        )


@pytest.mark.parametrize("target", ["catalog", "distances"])
def test_an_edited_bundled_file_is_refused_at_load(tmp_path, monkeypatch, target):
    """A hand edit to either file stops the run rather than changing results."""
    if target == "catalog":
        path = reference_data.COFACTOR_CATALOG_PATH
        accessor = reference_data.cofactor_ids
        cached = reference_data._catalog
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        edited = text + "ZZZ\tZn\tcluster\n"
    else:
        path = reference_data.DONOR_DISTANCE_PATH
        accessor = reference_data.literature_distances
        cached = reference_data.literature_distances
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        edited = text.replace("HOH O ZN 2.09", "HOH O ZN 2.50")
        assert edited != text, "the edit must actually change the file"

    copied = tmp_path / os.path.basename(path)
    copied.write_text(edited, encoding="utf-8")
    # The copy keeps the original's sidecar, which is what a hand-edited
    # checkout looks like.
    sidecar = reference_data.CHECKSUM_SIDECARS[path]
    monkeypatch.setitem(reference_data.CHECKSUM_SIDECARS, str(copied), sidecar)
    cached.cache_clear()

    with pytest.raises(reference_data.ReferenceDataError, match="does not match"):
        accessor(str(copied))


def test_a_caller_supplied_file_is_not_checksummed(tmp_path):
    """Only the bundled paths are verified; a path you passed is yours."""
    catalog = tmp_path / "mine.txt"
    catalog.write_text("ABC\tZn\tcluster\nDEF\tFe\theme\n", encoding="utf-8")

    assert reference_data.cofactor_ids(str(catalog)) == frozenset({"ABC", "DEF"})


def test_a_missing_sidecar_is_an_error_for_a_bundled_file(tmp_path, monkeypatch):
    """Deleting the sidecar must not silently disable the check."""
    catalog = tmp_path / "catalog.txt"
    catalog.write_text("ABC\tZn\tcluster\nDEF\tFe\theme\n", encoding="utf-8")
    monkeypatch.setitem(
        reference_data.CHECKSUM_SIDECARS,
        str(catalog),
        (str(tmp_path / "gone.meta.json"), "catalog_sha256"),
    )
    reference_data._catalog.cache_clear()

    with pytest.raises(reference_data.ReferenceDataError, match="no metadata sidecar"):
        reference_data.cofactor_ids(str(catalog))


@pytest.mark.parametrize(
    "row,message",
    [
        ("CYS S CU 2.two 0.2", "non-numeric"),
        ("CYS S CU 2.20", "fields, expected 5"),
        ("CYS S CU 2.20 0.2 0.3", "fields, expected 5"),
    ],
)
def test_a_damaged_distance_row_fails_the_parse(tmp_path, row, message):
    """A typo in a reference distance is an error, not a missing z-score.

    Dropping the row yields a contact "no reference covers", which is also the
    legitimate outcome for a pair genuinely absent from the literature.
    """
    table = tmp_path / "distances.txt"
    table.write_text(f"HOH O ZN 2.09 0.11\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        reference_data._load_literature(str(table))


@pytest.mark.parametrize(
    "mu,stdev,message",
    [
        ("nan", "0.11", "non-finite"),
        ("2.09", "inf", "non-finite"),
        ("-2.09", "0.11", "non-positive"),
        ("2.09", "-0.11", "non-positive"),
        ("0", "0.11", "non-positive"),
        ("2.09", "0", "non-positive"),
    ],
)
def test_distance_values_must_be_finite_and_positive(tmp_path, mu, stdev, message):
    table = tmp_path / "distances.txt"
    table.write_text(f"HOH O ZN {mu} {stdev}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        reference_data._load_literature(str(table))


def test_duplicate_reference_keys_are_rejected(tmp_path):
    table = tmp_path / "distances.txt"
    table.write_text("HOH O ZN 2.09 0.11\nHOH O ZN 2.10 0.12\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2 duplicates.*first defined on line 1"):
        reference_data._load_literature(str(table))


def test_the_table_header_is_skipped_by_name():
    """It is skipped by name, which is what allows every other unparseable line
    to be an error rather than a shrug."""
    assert reference_data.DISTANCE_TABLE_HEADER == (
        "residue",
        "atom",
        "metal",
        "avg_bond_dist",
        "st_dev",
    )
    with open(reference_data.DONOR_DISTANCE_PATH, encoding="utf-8") as handle:
        assert tuple(handle.readline().split()) == reference_data.DISTANCE_TABLE_HEADER


def test_a_table_with_no_distances_is_named_as_empty(tmp_path):
    """A file of comments and blanks parses to nothing and must say so."""
    table = tmp_path / "distances.txt"
    table.write_text("# only a comment\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no reference distances"):
        reference_data._load_literature(str(table))


def _stub_reference_data(tmp_path, monkeypatch, catalog_text, distance_text):
    """Point the checksum map at a writable pair of files and their sidecars."""
    paths = {}
    for name, text, key in (
        ("catalog.txt", catalog_text, "catalog_sha256"),
        ("distances.txt", distance_text, "distance_table_sha256"),
    ):
        data = tmp_path / name
        data.write_text(text, encoding="utf-8")
        sidecar = tmp_path / f"{name}.meta.json"
        sidecar.write_text(
            json.dumps({key: reference_data._sha256(str(data))}), encoding="utf-8"
        )
        paths[str(data)] = (str(sidecar), key)
    monkeypatch.setattr(reference_data, "CHECKSUM_SIDECARS", paths)
    # Cleared here as well as by the autouse fixture: a test may re-stub
    # mid-run, and the previous stub's identity would otherwise be returned.
    _clear_reference_data_caches()
    return paths


def test_the_identity_covers_both_files(tmp_path, monkeypatch):
    """Editing either file changes the id.

    The catalog decides what counts as a metal cofactor and the distance table
    sets every cutoff and z-score, so an id derived from one of them alone
    would report two incomparable runs as comparable.
    """
    _stub_reference_data(
        tmp_path, monkeypatch, "ABC\tZn\tcluster\n", "HOH O ZN 2.09 0.11\n"
    )
    original = reference_data.reference_data_id()

    for name, text in (
        ("catalog.txt", "ABC\tZn\tcluster\nDEF\tFe\theme\n"),
        ("distances.txt", "HOH O ZN 2.50 0.11\n"),
    ):
        _stub_reference_data(
            tmp_path,
            monkeypatch,
            text if name == "catalog.txt" else "HOH O ZN 2.09 0.11\n",
            text if name == "distances.txt" else "ABC\tZn\tcluster\n",
        )
        assert reference_data.reference_data_id() != original, (
            f"editing {name} left the identity unchanged"
        )


def test_the_identity_is_stable_for_unchanged_data(tmp_path, monkeypatch):
    """Same bytes, same id -- otherwise it cannot be a grouping key."""
    args = (tmp_path, monkeypatch, "ABC\tZn\tcluster\n", "HOH O ZN 2.09 0.11\n")
    _stub_reference_data(*args)
    first = reference_data.reference_data_id()
    _stub_reference_data(*args)

    assert reference_data.reference_data_id() == first


def test_the_identity_verifies_before_it_reports(tmp_path, monkeypatch):
    """An id for data that failed its own checksum would be a false claim."""
    paths = _stub_reference_data(
        tmp_path, monkeypatch, "ABC\tZn\tcluster\n", "HOH O ZN 2.09 0.11\n"
    )
    catalog = next(path for path in paths if path.endswith("catalog.txt"))
    with open(catalog, "a", encoding="utf-8") as handle:
        handle.write("GHI\tCu\t\n")
    reference_data.reference_data_checksums.cache_clear()
    reference_data.reference_data_id.cache_clear()

    with pytest.raises(reference_data.ReferenceDataError, match="does not match"):
        reference_data.reference_data_id()


def test_the_bundled_identity_is_short_and_hexadecimal():
    """It sits beside ``alchemy_commit`` in the manifest and reads like one."""
    identity = reference_data.reference_data_id()

    assert len(identity) == 12
    assert set(identity) <= set("0123456789abcdef")
    assert set(reference_data.reference_data_checksums()) == {
        "metallocofactors_id.txt",
        "metal_distances_info.txt",
    }


def test_the_ambiguous_component_ids_are_exactly_the_impostors():
    """Derive the denylist from Gemmi rather than trusting a hand-written one.

    An element symbol is safe to match against a residue name only if the CCD
    does not also use it for something else. Three do: ``U`` is uridine
    5'-monophosphate, ``NO`` is nitric oxide, ``CM`` is a small buffer
    component -- and the last was missed when this set was first written from
    the two that were already known. Comparing each tabulated component's
    weight against its element's own finds them without anyone having to
    remember, and fails if the CCD ever adds a fourth.
    """
    import gemmi

    from metal_elements import AMBIGUOUS_METAL_COMPONENT_IDS, METAL_ELEMENTS

    impostors = set()
    for symbol in METAL_ELEMENTS:
        info = gemmi.find_tabulated_residue(symbol)
        if info is None or str(info.kind) == "ResidueKind.UNKNOWN":
            continue
        # A tabulated component whose weight is not its element's is a
        # different molecule wearing the same two letters.
        if abs(info.weight - gemmi.Element(symbol).weight) > 0.5:
            impostors.add(symbol)

    assert impostors == set(AMBIGUOUS_METAL_COMPONENT_IDS), (
        "AMBIGUOUS_METAL_COMPONENT_IDS no longer matches the component table: "
        f"table says {sorted(impostors)}"
    )


def test_the_unambiguous_ids_still_cover_the_metals_that_matter():
    """Excluding the impostors must not remove a metal Alchemy analyses."""
    from metal_elements import UNAMBIGUOUS_METAL_COMPONENT_IDS

    for symbol in ("ZN", "FE", "MG", "CA", "MN", "CU", "NI", "CO", "K", "NA"):
        assert symbol in UNAMBIGUOUS_METAL_COMPONENT_IDS, symbol
