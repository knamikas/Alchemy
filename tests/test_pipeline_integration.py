"""End-to-end pipeline behaviour: ``src/main.py`` driven as a whole run.

Most tests here execute the real driver over a checksum-pinned PDB-REDO entry
snapshot, so they need CCP4 and either a warm entry cache or network access to
fetch those exact bytes. Entry-dependent tests acquire a ``network`` marker only
when the configured cache is not already complete. Consequently
``pytest -m 'not network'`` can use a warm cache but never opens a socket, while
a plain run on a bare machine downloads the snapshot or skips cleanly when the
network is unavailable.

The entry set is deliberately tiny, chemically well understood, and limited to
structures from the authors' research group:

``9myr``  Sal-Shy(DUF35) aldolase. Two structural zinc sites, each declared as
          a Cys3-His zinc-ribbon tetrad in ``_struct_conn``.
``6nlr``  the Snell-group PIXE-remodelled histidinol phosphatase. Eleven metal
          sites (Mn, Co, Fe, and Ca) across three chains provide the multi-site
          and multi-element stress case.
``9nxl``  steroid aldehyde dehydrogenase with no modelled metal: the documented
          ``no_metals`` negative control.

Nothing is written inside the repository: every run receives ``--output-dir``,
``--pdb-redo-cache``, ``--pdb-redo-root`` and ``--confidence-reference-dir``
under a tmp path. ``--confidence-reference-dir`` points at a non-existent
directory unless a test deliberately installs a frozen reference, so confidence
scoring is off by default and on exactly where it is under test.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import socket
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

import confidence_score
import helpers
import inputs
import main
from driver import runlog
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS
from bond.bond_schema import BOND_COLUMNS, CANDIDATE_COLUMNS


#: The three entries every test in this module draws on, and the set the cache
#: discovery below is keyed by.
ENTRY_IDS = ("9myr", "6nlr", "9nxl")

#: Byte-level identity of the compact PDB-REDO snapshot used by the numerical
#: assertions below. PDB-REDO's live entry URLs are mutable; pinning the three
#: files required from each entry turns an upstream revision into an explicit
#: snapshot-drift failure instead of a mysterious change in Alchemy's numbers.
#: The test fixture deliberately requires uncompressed mmCIF, MTZ and data.json
#: files. Production accepts compressed and legacy-PDB variants, but those are
#: not interchangeable numerical regression inputs and the manual-mode tests
#: require data.json.
_ENTRY_SNAPSHOT_SHA256 = {
    "9myr": {
        "9myr_final.mtz": "7d47d24a5a1e1cafc003adde878082aa9a2f4da8e947031b8228e1cf2234d21a",  # noqa: E501 - a SHA-256 digest cannot be wrapped
        "9myr_final.cif": "1013a862cd408d87d7ecfa0bf7521818454150fed7681dee392cde7bc2d9090e",  # noqa: E501 - a SHA-256 digest cannot be wrapped
        "data.json": "063a97b17a3fba9f970e39f99833322329879d1a76ae2ebad6d366ae12c74e3a",
    },
    "6nlr": {
        "6nlr_final.mtz": "a44d757d49a054738831e2b3d93f38d69f420513f92ecf77cff4a6b9cc995f5b",  # noqa: E501 - a SHA-256 digest cannot be wrapped
        "6nlr_final.cif": "e85475aeb6b89b95f8bb8df4ef1dfb16a77c12e059591541c3b25f64ec7aaa21",  # noqa: E501 - a SHA-256 digest cannot be wrapped
        "data.json": "6f7e3efca7d75d4b419b332445646082ee1b2fb5671ed465ce45ade3df7634cb",
    },
    "9nxl": {
        "9nxl_final.mtz": "87948fbae05c75fe8d03dde231b821b125b11badbf4595c2ade955e5a6026991",  # noqa: E501 - a SHA-256 digest cannot be wrapped
        "9nxl_final.cif": "185f55ec47be2b7ec42ea35a2986615f1ab289a435cea8ff16569c2053840e2f",  # noqa: E501 - a SHA-256 digest cannot be wrapped
        "data.json": "ded26ff60eee2b0b2faa730468443641eecb0c5f226375e53390b752d695e79d",
    },
}

#: Reference row for a zinc-ribbon histidine, read from the bundled literature
#: table rather than hard-coded, so the assertion tracks the data.
_HIS_N_ZN = ("HIS", "N", "ZN")

#: Real-space difference-density z-scores measured by this pipeline for every
#: selected metal site of the two metal-bearing entries, as
#: ``(entry, chain, residue number) -> (ZDm, ZD-m, ZD+m)``.
#:
#: These are the output of the whole density arm -- mtzfix, both FFTs, mapmask
#: and edstats -- over the checksum-pinned PDB-REDO snapshot above, so they are
#: the numbers that arm exists to produce and the only genuine oracle for it.
_MEASURED_RSZD = {
    ("9myr", "B", "201"): (2.1, -2.1, 1.6),
    ("9myr", "D", "201"): (1.2, -1.0, 1.2),
    ("6nlr", "A", "301"): (4.9, -4.9, 3.0),
    ("6nlr", "A", "302"): (1.5, -1.1, 1.5),
    ("6nlr", "A", "303"): (4.8, -4.8, 0.0),
    ("6nlr", "B", "301"): (2.8, -2.8, 0.0),
    ("6nlr", "B", "302"): (8.3, -8.3, 2.5),
    ("6nlr", "B", "303"): (4.7, -4.7, 0.2),
    ("6nlr", "B", "304"): (6.6, -6.6, 0.0),
    ("6nlr", "C", "301"): (8.2, -8.2, 3.0),
    ("6nlr", "C", "302"): (3.8, -3.8, 0.2),
    ("6nlr", "C", "303"): (1.5, -1.5, 0.1),
    ("6nlr", "C", "304"): (12.0, -12.0, 0.0),
}

#: EDSTATS reports RSZD to one decimal, so 0.05 is already lost to printing.
#: 0.3 leaves room on top of that for map-grid and rounding differences between
#: CCP4 builds while still pinning every value to ~7 percent -- tight enough
#: that a doubled, halved or sign-flipped RSZD, or a site scored against the
#: wrong map, cannot pass.
_RSZD_TOLERANCE = 0.3


def entry_file(cache_root: str, pdb_id: str, suffix: str) -> str:
    """Path to a required file in the checksum-pinned integration snapshot."""
    expected = _ENTRY_SNAPSHOT_SHA256.get(pdb_id)
    if expected is None or suffix not in expected:
        raise KeyError(f"{pdb_id}/{suffix} is not part of the integration snapshot")
    return os.path.join(inputs.entry_dir_for(cache_root, pdb_id), suffix)


def _entry_ready(cache_root: str, pdb_id: str) -> bool:
    """Whether the exact files required by this suite exist for ``pdb_id``."""
    return all(
        os.path.isfile(entry_file(cache_root, pdb_id, suffix))
        for suffix in _ENTRY_SNAPSHOT_SHA256[pdb_id]
    )


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_entry_snapshot(cache_root: str) -> None:
    """Fail clearly if a complete-looking cache is not the pinned snapshot."""
    mismatches = []
    for pdb_id, files in _ENTRY_SNAPSHOT_SHA256.items():
        for suffix, expected in files.items():
            path = entry_file(cache_root, pdb_id, suffix)
            if not os.path.isfile(path):
                mismatches.append(f"{pdb_id}/{suffix}: missing")
                continue
            actual = _file_sha256(path)
            if actual != expected:
                mismatches.append(
                    f"{pdb_id}/{suffix}: expected {expected}, got {actual}"
                )
    if mismatches:
        raise AssertionError(
            "PDB-REDO integration input differs from the checksum-pinned "
            "snapshot:\n  " + "\n  ".join(mismatches)
        )


def _network_available_now(timeout: float = 5.0) -> bool:
    """Fresh PDB-REDO reachability probe, bypassing helpers' session memo."""
    if os.environ.get("ALCHEMY_TESTS_NO_NETWORK"):
        return False
    try:
        with socket.create_connection(("pdb-redo.eu", 443), timeout=timeout):
            return True
    except OSError:
        return False


# Add the network marker only when acquiring the pinned entry data could touch
# the network. With a complete configured cache the same tests remain eligible
# under ``-m 'not network'`` and the fixture performs no capability probe.
_CONFIGURED_ENTRY_CACHE = helpers.cache_dir_from_env()
_ENTRY_SNAPSHOT_IS_WARM = bool(
    _CONFIGURED_ENTRY_CACHE
    and all(_entry_ready(_CONFIGURED_ENTRY_CACHE, pdb_id) for pdb_id in ENTRY_IDS)
)


def _requires_entry_data(test):
    """Mark a snapshot-backed test, plus its conditional network requirement."""
    test = pytest.mark.entry_data(test)
    if not _ENTRY_SNAPSHOT_IS_WARM:
        test = pytest.mark.network(test)
    return test


@pytest.fixture(scope="session")
def entry_cache(tmp_path_factory) -> str:
    """A PDB-REDO cache root holding every entry in :data:`ENTRY_IDS`.

    Uses :func:`helpers.cache_dir_from_env`, whose canonical
    ``ALCHEMY_TESTS_CACHE`` spelling takes precedence over the legacy alias.
    Otherwise downloads into a session tmp directory.

    A missing capability skips, but a broken download fails. ``pdb-redo.eu``
    being unreachable is an environment fact this suite cannot assert against;
    ``download_entry_to_cache`` returning without the final model files is a
    regression, and it must not be able to hide as a skip.
    """
    configured = helpers.cache_dir_from_env()
    if _ENTRY_SNAPSHOT_IS_WARM:
        assert configured is not None
        snapshot_cache = _CONFIGURED_ENTRY_CACHE
        assert snapshot_cache is not None
        if configured != snapshot_cache or not all(
            _entry_ready(snapshot_cache, pdb_id) for pdb_id in ENTRY_IDS
        ):
            raise AssertionError(
                "the configured PDB-REDO cache changed after test collection; "
                "refusing a network fallback for tests collected as offline"
            )
        _assert_entry_snapshot(configured)
        return configured

    target = configured or str(tmp_path_factory.mktemp("pdb_redo_cache"))
    os.makedirs(target, exist_ok=True)
    missing = [pdb_id for pdb_id in ENTRY_IDS if not _entry_ready(target, pdb_id)]
    if missing and not helpers.network_available():
        pytest.skip(
            f"PDB-REDO entries {', '.join(missing)} are not cached and there is "
            "no network access to fetch them"
        )
    for pdb_id in missing:
        # download_entry_to_cache funnels every failure -- a dead socket, a 404
        # and a renamed file alike -- into FileNotFoundError, so the exception
        # alone cannot separate "no network" from "the downloader is broken".
        # A fresh (non-memoized) re-probe separates them: if pdb-redo.eu answers,
        # the fetch itself regressed. Every other exception (OSError from an
        # unwritable cache, say) propagates untouched and fails the run.
        try:
            inputs.download_entry_to_cache(pdb_id, target)
        except FileNotFoundError as exc:
            if _network_available_now():
                raise AssertionError(
                    f"pdb-redo.eu is reachable but PDB-REDO entry {pdb_id} "
                    f"could not be downloaded into {target}: {exc}"
                ) from exc
            pytest.skip(
                f"network access to pdb-redo.eu was lost while fetching {pdb_id}: {exc}"
            )
        if not _entry_ready(target, pdb_id):
            raise AssertionError(
                f"download_entry_to_cache reported success for {pdb_id} but "
                f"{inputs.entry_dir_for(target, pdb_id)} does not contain the "
                "uncompressed mmCIF, MTZ, and data.json snapshot files"
            )
    _assert_entry_snapshot(target)
    return target


# Running the driver
@dataclass
class RunResult:
    """One completed ``main.main`` invocation."""

    exit_code: int
    stdout: str
    stderr: str
    output_dir: str

    @property
    def text(self) -> str:
        return self.stdout + self.stderr


@contextlib.contextmanager
def _environment(overrides: Optional[Dict[str, str]]):
    """Temporarily install ``overrides`` into ``os.environ``."""
    saved = os.environ.copy()
    if overrides:
        os.environ.update(overrides)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def run_alchemy(
    output_dir,
    *args,
    ccp4_environ=None,
    cache=None,
    tmp_root=None,
    mirror=None,
    reference_dir=None,
) -> RunResult:
    """Invoke the driver in-process and return its exit code and output.

    ``--output-dir`` plus every path option that otherwise defaults into the
    repository is supplied here, so no test can leave anything in the checkout.

    ``mirror`` sets ``--pdb-redo-root``; without it the root is a path that does
    not exist, so entries can only come from the cache. ``reference_dir`` sets
    ``--confidence-reference-dir``; without it that too is a non-existent path,
    which is what turns confidence scoring off.
    """
    output_dir = str(output_dir)
    argv = ["--output-dir", output_dir]
    if cache is not None:
        argv += ["--pdb-redo-cache", str(cache)]
    root = (
        str(mirror)
        if mirror is not None
        else os.path.join(str(tmp_root or output_dir), "absent-mirror")
    )
    argv += ["--pdb-redo-root", root]
    argv += [
        "--confidence-reference-dir",
        str(reference_dir)
        if reference_dir is not None
        else os.path.join(str(tmp_root or output_dir), "absent-reference"),
    ]
    argv += [str(a) for a in args]

    out, err = io.StringIO(), io.StringIO()
    code = 0
    with _environment(ccp4_environ):
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return RunResult(int(code or 0), out.getvalue(), err.getvalue(), output_dir)


def id_file(directory, pdb_ids: Sequence[str]) -> str:
    """Write a whitespace-separated ``--id-file`` and return its path."""
    os.makedirs(str(directory), exist_ok=True)
    path = os.path.join(str(directory), "ids.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(pdb_ids) + "\n")
    return path


# Output readers
def read_rows(output_dir, name: str) -> List[Dict[str, str]]:
    with open(os.path.join(str(output_dir), name), newline="") as handle:
        return list(csv.DictReader(handle))


def read_header(output_dir, name: str) -> List[str]:
    with open(os.path.join(str(output_dir), name), newline="") as handle:
        return next(csv.reader(handle))


def manifest_by_id(output_dir) -> Dict[str, Dict[str, str]]:
    return {row["pdbID"]: row for row in read_rows(output_dir, "manifest.csv")}


def rows_for(rows: Sequence[Dict[str, str]], pdb_id: str) -> List[Dict[str, str]]:
    return [row for row in rows if row["pdbID"] == pdb_id]


def log_paths(output_dir) -> List[str]:
    """Every run log this output directory has, wherever the default puts them."""
    directory = os.path.join(str(output_dir), runlog.DEFAULT_LOG_DIRNAME)
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if re.fullmatch(r"alchemy_run_\d{8}(_\d+)?\.log", name)
    )


def literature_reference(residue: str, atom_element: str, metal: str):
    """(mu, sigma) for one row of ``src/data/metal_distances_info.txt``."""
    path = os.path.join(helpers.SRC_DIR, "data", "metal_distances_info.txt")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            if (parts[0].upper(), parts[1].upper(), parts[2].upper()) == (
                residue,
                atom_element,
                metal,
            ):
                return float(parts[3]), float(parts[4])
    raise AssertionError(
        f"no literature reference for {residue}/{atom_element}/{metal}"
    )


@dataclass
class Batch:
    """Parsed outputs of one completed multi-entry run."""

    result: RunResult
    output_dir: str
    manifest_rows: List[Dict[str, str]]
    manifest: Dict[str, Dict[str, str]]
    stats: List[Dict[str, str]]
    bonds: List[Dict[str, str]]
    candidates: List[Dict[str, str]]


def run_batch(
    output_dir,
    cache,
    ccp4_environ,
    pdb_ids=ENTRY_IDS,
    extra=(),
    tmp_root=None,
    reference_dir=None,
) -> Batch:
    """Run ``--id-file`` over ``pdb_ids`` and parse every output CSV."""
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ids_path = id_file(tmp_root or output_dir, pdb_ids)
    result = run_alchemy(
        output_dir,
        "--id-file",
        ids_path,
        "--workers",
        "3",
        *extra,
        ccp4_environ=ccp4_environ,
        cache=cache,
        tmp_root=tmp_root,
        reference_dir=reference_dir,
    )
    manifest_rows = read_rows(output_dir, "manifest.csv")
    return Batch(
        result=result,
        output_dir=output_dir,
        manifest_rows=manifest_rows,
        manifest={row["pdbID"]: row for row in manifest_rows},
        stats=read_rows(output_dir, "metal_stats_all.csv"),
        bonds=read_rows(output_dir, "metal_bonds_all.csv"),
        candidates=read_rows(output_dir, "metal_candidates_all.csv"),
    )


@pytest.fixture(scope="session")
def batch(tmp_path_factory, entry_cache, ccp4_env) -> Batch:
    """A single completed three-entry run, shared read-only by several tests."""
    root = tmp_path_factory.mktemp("batch")
    result = run_batch(root / "output", entry_cache, ccp4_env, tmp_root=root)
    assert result.result.exit_code == 0, result.result.text
    return result


# 1. A single-entry run produces the documented files and columns
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_single_entry_run_writes_documented_outputs(tmp_path, entry_cache, ccp4_env):
    """One ``--id`` run writes exactly the documented output set.

    README "Results are written to": manifest, statistics, bond and candidate
    CSVs plus one timestamped run log. Each CSV must carry its module's full
    fixed schema, and the per-entry scratch directory must be cleaned up.
    """
    output_dir = tmp_path / "output"
    result = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert result.exit_code == 0, result.text

    written = set(os.listdir(output_dir))
    assert {
        "manifest.csv",
        "metal_stats_all.csv",
        "metal_bonds_all.csv",
        "metal_candidates_all.csv",
    } <= written
    assert len(log_paths(output_dir)) == 1

    # No confidence artefacts: no frozen reference is installed for this run.
    assert "confidence_scores_all.csv" not in written
    assert "confidence_inputs_all.csv" not in written
    assert "no frozen reference is distributed with Alchemy" in result.text

    # Per-entry working directories are removed once their rows are extracted.
    assert [name for name in written if name.startswith(".alchemy-")] == []

    assert read_header(output_dir, "manifest.csv") == MANIFEST_COLUMNS
    assert read_header(output_dir, "metal_stats_all.csv") == STATS_COLUMNS
    assert read_header(output_dir, "metal_bonds_all.csv") == BOND_COLUMNS
    assert read_header(output_dir, "metal_candidates_all.csv") == CANDIDATE_COLUMNS

    # Columns the README names explicitly must really be present, so a schema
    # that silently drifted away from the documentation is caught here and not
    # only by comparison with the code's own constant.
    stats_header = read_header(output_dir, "metal_stats_all.csv")
    assert set(helpers.EDSTATS_HEADER) <= set(stats_header)
    assert {
        "density_observation_id",
        "density_scope",
        "density_shared_site_count",
        "density_is_shared",
        "coordinate_mapping_status",
        "selected_metal_site_status",
        "dpi",
        "resolution",
    } <= set(stats_header)
    assert {
        "distance",
        "zscore",
        "geometry_outlier",
        "geometry_consistent",
        "declared_connection",
        "contact_scope",
        "symmetry_contact",
        "crystallographic_contact",
        "strict_ncs_contact",
        "multi_donor_detected",
        "context_warning",
        "context_warning_reasons",
    } <= set(BOND_COLUMNS)
    assert {
        "first_sphere_eligible",
        "eligibility_status",
        "eligibility_reason",
        "candidate_source",
        "inferred_donor_allowed",
        "first_sphere_cutoff",
        "assignment_reference_kind",
    } <= set(CANDIDATE_COLUMNS)
    assert {
        "n_metals",
        "n_bonds",
        "n_candidates",
        "status",
        "retryable",
        "reason_codes",
        "runtime_s",
    } <= set(MANIFEST_COLUMNS)


# 2. 9myr and 6nlr: the chemistry and density have to come out right
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_9myr_reports_two_chemically_sane_zinc_ribbon_sites(batch):
    """9myr yields two Cys3-His zinc sites with measured density and geometry.

    Each site supplies four independently declared bonds. Their z-scores are
    re-derived from the README formula and bundled literature table, so a
    change in the scoring path -- not merely a changed recorded number -- makes
    this fail.
    """
    stats = {row["CI"]: row for row in rows_for(batch.stats, "9myr")}
    assert set(stats) == {"B", "D"}

    bonds = rows_for(batch.bonds, "9myr")
    by_chain = {
        chain: [row for row in bonds if row["metal_chain"] == chain] for chain in stats
    }
    expected_donors = {
        ("CYS", "SG", "31"),
        ("HIS", "ND1", "35"),
        ("CYS", "SG", "44"),
        ("CYS", "SG", "47"),
    }

    for chain, site in stats.items():
        assert site["metal_element"] == "ZN"
        assert site["category"] == "metal"
        assert site["RT"] == "ZN"
        assert site["RN"] == "201"
        assert site["coordinate_mapping_status"] == "matched"
        assert site["selected_metal_site_status"] == "selected"

        expected_density = _MEASURED_RSZD[("9myr", chain, "201")]
        for column, value in zip(("ZDm", "ZD-m", "ZD+m"), expected_density):
            assert float(site[column]) == pytest.approx(value, abs=_RSZD_TOLERANCE), (
                chain,
                column,
            )
        rszd = float(site["ZDm"])
        assert math.isfinite(rszd)
        assert abs(rszd) == pytest.approx(
            max(abs(float(site["ZD-m"])), abs(float(site["ZD+m"]))), abs=0.05
        )

        site_bonds = by_chain[chain]
        assert len(site_bonds) == 4
        assert {
            (row["neighbor_resname"], row["neighbor_atom"], row["neighbor_resnum"])
            for row in site_bonds
        } == expected_donors
        assert all(row["declared_connection"] == "True" for row in site_bonds)

        for bond in site_bonds:
            distance = float(bond["distance"])
            mu, sigma = literature_reference(
                bond["neighbor_resname"], bond["neighbor_element"], "ZN"
            )
            assert float(bond["literature_distance"]) == pytest.approx(mu)
            assert float(bond["literature_stdev"]) == pytest.approx(sigma)
            dpi = float(bond["dpi"])
            assert math.isfinite(dpi) and dpi > 0.0
            expected_z = (distance - mu) / math.sqrt(dpi**2 + sigma**2)
            assert float(bond["zscore"]) == pytest.approx(expected_z, abs=5e-4)
            assert abs(expected_z) < float(bond["zscore_outlier_cutoff"])
            assert bond["geometry_outlier"] == "False"
            assert bond["geometry_consistent"] == "True"
            assert float(bond["sigma_mag"]) == pytest.approx(rszd)
            assert float(bond["sigma_neg"]) == pytest.approx(float(site["ZD-m"]))
            assert float(bond["sigma_pos"]) == pytest.approx(float(site["ZD+m"]))

    # The histidine reference used above is an explicit shipped-data oracle.
    assert literature_reference(*_HIS_N_ZN) == pytest.approx((2.03, 0.05))


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_6nlr_multi_element_sites_report_their_measured_difference_density(batch):
    """All eleven PIXE-remodelled 6nlr sites carry their own RSZD triple.

    This stresses the density join with four elements, three chains, and
    repeated residue numbers. A map that is mis-scaled, mis-cropped, or joined
    using an incomplete site key cannot pass.
    """
    sites = {(row["CI"], row["RN"]): row for row in rows_for(batch.stats, "6nlr")}
    expected_keys = {key[1:] for key in _MEASURED_RSZD if key[0] == "6nlr"}
    assert set(sites) == expected_keys
    assert Counter(row["metal_element"] for row in sites.values()) == Counter(
        {"MN": 3, "CO": 3, "FE": 3, "CA": 2}
    )

    for (chain, resnum), row in sites.items():
        expected = _MEASURED_RSZD[("6nlr", chain, resnum)]
        assert row["selected_metal_site_status"] == "selected", (chain, resnum)
        assert row["coordinate_mapping_status"] == "matched", (chain, resnum)
        for column, value in zip(("ZDm", "ZD-m", "ZD+m"), expected):
            assert float(row[column]) == pytest.approx(value, abs=_RSZD_TOLERANCE), (
                chain,
                resnum,
                column,
            )
        assert abs(float(row["ZDm"])) == pytest.approx(
            max(abs(float(row["ZD-m"])), abs(float(row["ZD+m"]))), abs=0.05
        ), (chain, resnum)

    # The spread is a property of the snapshot, not merely a broad sanity
    # bound: A302 and C303 tie at the low end while C304 reaches the top anchor.
    minimum = min(float(row["ZDm"]) for row in sites.values())
    assert {key for key, row in sites.items() if float(row["ZDm"]) == minimum} == {
        ("A", "302"),
        ("C", "303"),
    }
    assert max(sites, key=lambda key: float(sites[key]["ZDm"])) == ("C", "304")


# 3. Regression: declared connections must bind to the atoms they name
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_declared_connections_measure_their_own_reported_distance(batch):
    """Every declared contact's measured distance matches the one it declares.

    Regression for 733d8ec. ``_collect_declared_candidates`` used to translate
    source-mmCIF atom serials into analysis-PDB serial space; gemmi's PDB writer
    emits a TER after each polymer and every TER consumes a serial, so every
    partner after the first TER resolved to the wrong atom. Both 6nlr's ions and
    9myr's zinc sites are numbered after their polymer chains and therefore
    place the metals behind TER records.

    A ``_struct_conn`` row carries the distance the depositor measured, so a
    declaration bound to the wrong atom disagrees with its own number -- the
    reported symptom was Ca-to-donor "contacts" at 9-50 Angstrom.
    """
    declared = [row for row in batch.bonds if row["declared_connection"] == "True"]
    assert declared, "the batch must contain declared metal connections"

    for row in declared:
        where = (
            f"{row['pdbID']} {row['metal_resname']}{row['metal_resnum']}"
            f"-{row['neighbor_resname']}{row['neighbor_resnum']}"
            f":{row['neighbor_atom']}"
        )
        reported = float(row["connection_reported_distance"])
        measured = float(row["distance"])
        assert reported > 0.0, where
        # Reported distances are written to two decimals, measured to three.
        assert measured == pytest.approx(reported, abs=0.02), where
        # A first-sphere metal-donor contact is a chemical bond length, never a
        # lattice-scale separation.
        assert 1.5 < measured < 3.0, where
        assert row["coordination_source"] == "struct_conn", where
        zscore = float(row["zscore"])
        if math.isfinite(zscore):
            assert abs(zscore) < float(row["zscore_outlier_cutoff"]), where

    # Every declaration in both metal-bearing entries survives resolution.
    # Losing some of them is the other half of a broken partner lookup.
    assert len(declared) == 49, "declared connections went missing or duplicated"
    assert len(rows_for(declared, "6nlr")) == 41
    assert len(rows_for(declared, "9myr")) == 8


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_declared_contacts_reach_the_expected_zinc_ribbon_donors(batch):
    """Both 9myr zinc sites resolve to the deposited Cys3-His tetrad.

    A serial-space mismatch does not only distort distances: it re-points the
    contact at whatever atom happens to hold the shifted serial. Checking donor
    identities pins the mapping independently of geometry.
    """
    zinc = rows_for(batch.bonds, "9myr")
    assert len(zinc) == 8
    expected = {
        ("CYS", "SG", "S", "31"),
        ("HIS", "ND1", "N", "35"),
        ("CYS", "SG", "S", "44"),
        ("CYS", "SG", "S", "47"),
    }
    for chain in ("B", "D"):
        rows = [row for row in zinc if row["metal_chain"] == chain]
        assert {
            (
                row["neighbor_resname"],
                row["neighbor_atom"],
                row["neighbor_element"],
                row["neighbor_resnum"],
            )
            for row in rows
        } == expected
        assert {row["neighbor_chain"] for row in rows} == {chain}
        assert all(row["neighbor_class"] == "amino_acid" for row in rows)
        assert all(row["declared_connection"] == "True" for row in rows)


# 4. The no-metal control
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_no_metal_entry_is_reported_not_dropped(batch):
    """9nxl finishes as a successful negative result with zeroed counts.

    README: "structures with no recognized metal atoms finish with n_metals=0
    without running mtzfix, either FFT, or edstats", and those entries are
    reported as ``no_metals`` -- an informational subset of ``ok``, never
    ``skip``. Because the stage did run to completion, the bond counts are a
    measured ``0`` rather than blank.
    """
    row = batch.manifest["9nxl"]
    assert row["status"] == "ok"
    assert row["retryable"] == "False"
    assert row["reason_codes"] == ""
    assert row["error"] == ""
    assert row["n_metals"] == "0"
    assert row["n_bonds"] == "0"
    assert row["n_candidates"] == "0"

    assert rows_for(batch.stats, "9nxl") == []
    assert rows_for(batch.bonds, "9nxl") == []
    assert rows_for(batch.candidates, "9nxl") == []

    # Headers survive even when an entry contributes no rows.
    assert read_header(batch.output_dir, "metal_stats_all.csv") == STATS_COLUMNS

    # The density stages are skipped entirely for a metal-free entry.
    log_text = open(log_paths(batch.output_dir)[0], encoding="utf-8").read()
    entry_line = next(
        line for line in log_text.splitlines() if line.startswith("9nxl | status=")
    )
    assert "no_metals=true" in entry_line
    assert "density_map_scope=-" in entry_line
    assert "edstats_s" not in entry_line
    assert "mtzfix_s" not in entry_line
    assert "Metal-free entries: 1" in log_text


# 5. Manifest bookkeeping
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_multi_entry_batch_aggregates_every_entry_exactly_once(batch):
    """A batch writes one manifest row per requested id and no cross-talk."""
    manifest_ids = [row["pdbID"] for row in batch.manifest_rows]
    assert len(batch.manifest_rows) == len(ENTRY_IDS)
    assert Counter(manifest_ids) == Counter(ENTRY_IDS)
    assert set(batch.manifest) == set(ENTRY_IDS)
    assert all(row["status"] == "ok" for row in batch.manifest.values())
    assert batch.result.exit_code == 0

    for table in (batch.stats, batch.bonds, batch.candidates):
        assert {row["pdbID"] for row in table} <= set(ENTRY_IDS)

    # Aggregate totals are the sum of the per-entry manifest counts.
    assert len(batch.bonds) == sum(
        int(row["n_bonds"]) for row in batch.manifest.values()
    )
    assert len(batch.candidates) == sum(
        int(row["n_candidates"]) for row in batch.manifest.values()
    )

    # Each entry's rows really belong to it: metals only ever appear in the
    # coordinate file they were read from.
    assert {row["metal_element"] for row in rows_for(batch.bonds, "9myr")} == {"ZN"}
    assert {row["metal_element"] for row in rows_for(batch.bonds, "6nlr")} == {
        "MN",
        "CO",
        "FE",
        "CA",
    }


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
@pytest.mark.parametrize("pdb_id", ENTRY_IDS)
def test_manifest_counts_match_the_rows_actually_written(batch, pdb_id):
    """``n_metals``/``n_bonds``/``n_candidates`` agree with the CSV contents.

    README: ``n_metals`` counts distinct selected coordinate-model sites, not
    diagnostic or repeated EDSTATS rows, while ``n_bonds`` and ``n_candidates``
    are assigned-output and candidate-evidence sizes.
    """
    row = batch.manifest[pdb_id]
    assert int(row["n_bonds"]) == len(rows_for(batch.bonds, pdb_id))
    assert int(row["n_candidates"]) == len(rows_for(batch.candidates, pdb_id))

    selected = [
        stat
        for stat in rows_for(batch.stats, pdb_id)
        if stat["selected_metal_site_status"] == "selected"
    ]
    assert int(row["n_metals"]) == len(selected)

    # Candidate evidence is a superset of assigned contacts: every bond was
    # either discovered by the 4 A search or declared, and both are recorded.
    assert int(row["n_candidates"]) >= int(row["n_bonds"])


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_each_run_writes_its_own_immutable_log(tmp_path, entry_cache, ccp4_env):
    """A second invocation adds a suffixed log instead of replacing the first.

    README: "Additional runs on the same UTC date receive a numeric suffix ...
    Resume runs create another log rather than replacing the original record."
    """
    output_dir = tmp_path / "output"
    first = run_alchemy(
        output_dir,
        "--id",
        "9nxl",
        "--workers",
        "1",
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert first.exit_code == 0, first.text
    original = log_paths(output_dir)
    assert len(original) == 1
    original_text = open(original[0], encoding="utf-8").read()

    second = run_alchemy(
        output_dir,
        "--id",
        "9nxl",
        "--workers",
        "1",
        "--resume",
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert second.exit_code == 0, second.text
    assert len(log_paths(output_dir)) == 2
    assert open(original[0], encoding="utf-8").read() == original_text
    assert "no entries to process" in second.text


# 6. Resume semantics
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_resume_over_a_completed_batch_adds_no_duplicate_rows(
    tmp_path, entry_cache, ccp4_env
):
    """Resuming a finished batch reprocesses nothing and rewrites nothing.

    README: ``--resume`` skips ``ok`` and terminal ``partial`` outcomes without
    duplicating their previous rows.
    """
    output_dir = tmp_path / "output"
    first = run_batch(output_dir, entry_cache, ccp4_env, tmp_root=tmp_path)
    assert first.result.exit_code == 0, first.result.text

    names = (
        "manifest.csv",
        "metal_stats_all.csv",
        "metal_bonds_all.csv",
        "metal_candidates_all.csv",
    )
    before = {
        name: open(os.path.join(str(output_dir), name), "rb").read() for name in names
    }

    second = run_batch(
        output_dir, entry_cache, ccp4_env, extra=("--resume",), tmp_root=tmp_path
    )
    assert second.result.exit_code == 0, second.result.text
    assert "no entries to process" in second.result.text

    after = {
        name: open(os.path.join(str(output_dir), name), "rb").read() for name in names
    }
    assert after == before

    # And, stated as row counts rather than bytes, nothing was appended.
    assert len(second.manifest) == len(ENTRY_IDS)
    assert len(second.bonds) == len(first.bonds)
    assert len(second.candidates) == len(first.candidates)
    assert len(second.stats) == len(first.stats)


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_fresh_no_bonds_run_clears_bond_outputs_and_resume_restores_them(
    tmp_path, entry_cache, ccp4_env
):
    """``--no-bonds`` erases stale bond output; a later resume fills it back in.

    README: "A fresh --no-bonds run removes pre-existing metal_bonds_all.csv and
    metal_candidates_all.csv ... Entries originating from a bond-disabled run
    retain blank n_bonds and n_candidates, so a later bond-enabled resume will
    process them."
    """
    output_dir = tmp_path / "output"
    complete = run_batch(output_dir, entry_cache, ccp4_env, tmp_root=tmp_path)
    assert complete.result.exit_code == 0, complete.result.text
    assert complete.bonds, "the seeding run must produce bond rows"

    disabled = run_alchemy(
        output_dir,
        "--id-file",
        id_file(tmp_path, ENTRY_IDS),
        "--workers",
        "3",
        "--no-bonds",
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert disabled.exit_code == 0, disabled.text
    assert "removed stale bond-stage output" in disabled.text
    assert not os.path.exists(os.path.join(str(output_dir), "metal_bonds_all.csv"))
    assert not os.path.exists(os.path.join(str(output_dir), "metal_candidates_all.csv"))

    blank = manifest_by_id(output_dir)
    assert set(blank) == set(ENTRY_IDS)
    for pdb_id, row in blank.items():
        assert row["status"] == "ok", pdb_id
        assert row["n_bonds"] == "", pdb_id
        assert row["n_candidates"] == "", pdb_id
    # Statistics are still produced by the bond-disabled run.
    assert read_rows(output_dir, "metal_stats_all.csv")

    restored = run_batch(
        output_dir, entry_cache, ccp4_env, extra=("--resume",), tmp_root=tmp_path
    )
    assert restored.result.exit_code == 0, restored.result.text
    assert set(restored.manifest) == set(ENTRY_IDS)
    for pdb_id, row in restored.manifest.items():
        assert row["n_bonds"] != "", pdb_id
        assert int(row["n_bonds"]) == len(rows_for(restored.bonds, pdb_id))
        assert int(row["n_candidates"]) == len(rows_for(restored.candidates, pdb_id))
    # The retry replaced rows rather than duplicating them.
    assert len(restored.bonds) == len(complete.bonds)
    assert len(restored.stats) == len(complete.stats)
    assert len(restored.manifest) == len(ENTRY_IDS)


# 7. Regression: an unrun bond stage is blank, not a measured zero
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_entry_failing_before_the_bond_stage_is_never_resumed_away(
    tmp_path, entry_cache, ccp4_env
):
    """A pre-bond failure leaves blank counts, so bonds are still filled later.

    Regression for cf55dd5. ``_initial_result`` used to seed ``n_bonds`` and
    ``n_candidates`` with ``0``. An entry that failed before the bond stage then
    claimed a measured zero; ``--resume --no-bonds`` carried that stale ``0``
    onto the now-``ok`` row, and the following bond-enabled ``--resume`` saw
    ``status=ok`` with non-blank counts, judged the stage complete and skipped
    the entry permanently while the bond CSVs held no rows for it.

    The three runs below are exactly that sequence, with the third asserting
    that the bond rows do arrive.
    """
    output_dir = tmp_path / "output"
    good_mtz = entry_file(entry_cache, "9myr", "9myr_final.mtz")
    cif = entry_file(entry_cache, "9myr", "9myr_final.cif")
    data_json = entry_file(entry_cache, "9myr", "data.json")
    broken_mtz = _mtz_without_map_coefficients(good_mtz, tmp_path / "broken.mtz")

    base = [
        "--id",
        "9myr",
        "--cif-file",
        cif,
        "--data-json",
        data_json,
        "--workers",
        "1",
    ]

    failed = run_alchemy(
        output_dir,
        *base,
        "--mtz-file",
        broken_mtz,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert failed.exit_code == 1, failed.text
    row = manifest_by_id(output_dir)["9myr"]
    assert row["status"] == "error"
    assert row["n_bonds"] == "", "an unrun bond stage must not claim zero rows"
    assert row["n_candidates"] == ""

    carried = run_alchemy(
        output_dir,
        *base,
        "--mtz-file",
        good_mtz,
        "--resume",
        "--no-bonds",
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert carried.exit_code == 0, carried.text
    row = manifest_by_id(output_dir)["9myr"]
    assert row["status"] == "ok"
    assert row["n_bonds"] == "", "a bond-disabled resume must keep counts blank"
    assert row["n_candidates"] == ""

    filled = run_alchemy(
        output_dir,
        *base,
        "--mtz-file",
        good_mtz,
        "--resume",
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert filled.exit_code == 0, filled.text
    assert "no entries to process" not in filled.text, (
        "the bond stage never ran for this entry, so resume must process it"
    )
    bonds = rows_for(read_rows(output_dir, "metal_bonds_all.csv"), "9myr")
    assert bonds, "the bond-enabled resume must actually write bond rows"
    row = manifest_by_id(output_dir)["9myr"]
    assert int(row["n_bonds"]) == len(bonds)
    assert float(bonds[0]["distance"]) == pytest.approx(2.311, abs=0.02)


# 8. Manual-file mode
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_manual_files_with_data_json_reproduce_the_cached_entry_run(
    tmp_path, entry_cache, ccp4_env, batch
):
    """Manual ``--cif-file/--mtz-file/--data-json`` matches the automatic run.

    Manual mode is an input-selection path, not a different analysis: given the
    same coordinates, reflections and metadata it must produce the same geometry
    and the same DPI-derived numbers as the mirror/cache workflow.
    """
    output_dir = tmp_path / "output"
    result = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        "--cif-file",
        entry_file(entry_cache, "9myr", "9myr_final.cif"),
        "--mtz-file",
        entry_file(entry_cache, "9myr", "9myr_final.mtz"),
        "--data-json",
        entry_file(entry_cache, "9myr", "data.json"),
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert result.exit_code == 0, result.text

    row = manifest_by_id(output_dir)["9myr"]
    assert row["status"] == "ok"
    assert row["refinement_state"] == "manual"
    assert row["source_coordinate_format"] == "mmcif"
    assert row["coordinate_conversion_performed"] == "True"

    manual_bonds = read_rows(output_dir, "metal_bonds_all.csv")
    reference_bonds = rows_for(batch.bonds, "9myr")
    assert len(manual_bonds) == len(reference_bonds) == 8
    for manual, reference in zip(manual_bonds, reference_bonds):
        for column in (
            "metal_chain",
            "metal_element",
            "neighbor_resname",
            "neighbor_atom",
            "neighbor_resnum",
            "distance",
            "dpi",
            "zscore",
            "literature_distance",
            "connection_reported_distance",
        ):
            assert manual[column] == reference[column], column


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_manual_files_without_data_json_are_a_terminal_partial(
    tmp_path, entry_cache, ccp4_env, batch
):
    """Omitting ``--data-json`` costs DPI only, and is not a failure.

    README: "Without --data-json there is no reflection count, so DPI and every
    value derived from it are unavailable. Bond geometry is still measured and
    emitted, and the omission is reported as ``missing_dpi_metadata_source``
    rather than as a calculation failure." A deterministic limitation is a
    terminal (non-retryable) partial, and terminal partials exit successfully.
    """
    output_dir = tmp_path / "output"
    result = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        "--cif-file",
        entry_file(entry_cache, "9myr", "9myr_final.cif"),
        "--mtz-file",
        entry_file(entry_cache, "9myr", "9myr_final.mtz"),
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert result.exit_code == 0, result.text

    row = manifest_by_id(output_dir)["9myr"]
    assert row["status"] == "partial"
    assert row["retryable"] == "False"
    assert row["reason_codes"] == "missing_dpi_metadata_source"
    assert row["n_metals"] == "2"
    assert row["n_bonds"] == "8"

    site = read_rows(output_dir, "metal_stats_all.csv")[0]
    assert site["dpi_unavailable_reason"] == "missing_dpi_metadata_source"
    assert not math.isfinite(float(site["dpi"]))
    # Density is unaffected: only the DPI-derived numbers are lost.
    assert float(site["ZDm"]) == pytest.approx(
        float(rows_for(batch.stats, "9myr")[0]["ZDm"])
    )

    bond = read_rows(output_dir, "metal_bonds_all.csv")[0]
    reference = rows_for(batch.bonds, "9myr")[0]
    assert float(bond["distance"]) == pytest.approx(float(reference["distance"]))
    assert bond["neighbor_atom"] == "SG"
    assert not math.isfinite(float(bond["dpi"]))
    assert not math.isfinite(float(bond["zscore"]))
    # The measured, non-derived evidence is still complete.
    assert float(bond["literature_distance"]) > 0.0
    assert bond["geometry_outlier"] in ("", "False")


# 9. The map-scope equivalence claim
@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_full_and_model_envelope_map_scopes_give_identical_statistics(
    tmp_path, entry_cache, ccp4_env
):
    """Cropping the FFT map to the model envelope changes no EDSTATS number.

    README: "Model-envelope mode still calculates each complete FFT map before
    cropping, so map values come from the same Fourier calculation as legacy
    full-map mode." That makes the default a pure performance optimisation, and
    it is the pipeline's strongest numerical claim.
    """
    cropped = run_batch(
        tmp_path / "envelope",
        entry_cache,
        ccp4_env,
        extra=("--density-map-scope", "model-envelope"),
        tmp_root=tmp_path / "envelope-tmp",
    )
    full = run_batch(
        tmp_path / "full",
        entry_cache,
        ccp4_env,
        extra=("--density-map-scope", "full"),
        tmp_root=tmp_path / "full-tmp",
    )
    assert cropped.result.exit_code == 0, cropped.result.text
    assert full.result.exit_code == 0, full.result.text

    # The comparison is only meaningful if cropping actually happened for at
    # least one entry; otherwise both runs took the same code path.
    cropped_log = open(log_paths(cropped.output_dir)[0], encoding="utf-8").read()
    full_log = open(log_paths(full.output_dir)[0], encoding="utf-8").read()
    assert re.search(r"density_map_scope=model-envelope\b", cropped_log), (
        "no entry exercised model-envelope cropping"
    )
    assert not re.search(r"density_map_scope=model-envelope\b", full_log)
    assert re.search(r"density_map_scope=full\b", full_log)

    def key(row):
        return (row["pdbID"], row["RT"], row["CI"], row["RN"])

    cropped_sites = {key(row): row for row in cropped.stats}
    full_sites = {key(row): row for row in full.stats}
    assert cropped_sites and set(cropped_sites) == set(full_sites)

    for site_key, row in cropped_sites.items():
        other = full_sites[site_key]
        for metric in helpers.EDSTATS_METRICS:
            assert row[metric] == other[metric], (site_key, metric)

    # The downstream numbers that depend on those statistics agree too.
    def bond_key(row):
        return (
            row["pdbID"],
            row["metal_resnum"],
            row["neighbor_resnum"],
            row["neighbor_atom"],
        )

    cropped_bonds = {bond_key(row): row for row in cropped.bonds}
    full_bonds = {bond_key(row): row for row in full.bonds}
    assert set(cropped_bonds) == set(full_bonds)
    for bond_id, row in cropped_bonds.items():
        for column in ("distance", "zscore", "sigma_mag", "sigma_neg", "sigma_pos"):
            assert row[column] == full_bonds[bond_id][column], (bond_id, column)


# 10. Clean failure, not a traceback or a hang
def _mtz_without_map_coefficients(source_mtz: str, destination) -> str:
    """Write an MTZ with real indices but none of FWT/PHWT/DELFWT/PHDELWT."""
    import gemmi
    import numpy as np

    source = gemmi.read_mtz_file(str(source_mtz))
    stripped = gemmi.Mtz(with_base=True)
    stripped.cell = source.cell
    stripped.spacegroup = source.spacegroup
    stripped.add_dataset("synthetic")
    stripped.add_column("FP", "F")
    stripped.add_column("SIGFP", "Q")
    indices = source.array[:, :3]
    rows = len(indices)
    stripped.set_data(
        np.hstack(
            [
                indices,
                np.full((rows, 1), 100.0),
                np.full((rows, 1), 1.0),
            ]
        )
    )
    stripped.write_to_file(str(destination))
    return str(destination)


@pytest.mark.ccp4
@pytest.mark.slow
def test_unknown_pdb_id_fails_with_a_message_and_no_output_tables(
    tmp_path, ccp4_env, monkeypatch
):
    """An id that exists nowhere ends the run with an explanation and exit 1.

    The downloader is replaced at its boundary: this is an error-reporting test,
    not a live-network test, and an unmarked test must never open a socket under
    ``-m 'not network'``.
    """

    def unavailable(pdb_id, cache_root):
        raise FileNotFoundError(pdb_id)

    monkeypatch.setattr(inputs, "download_entry_to_cache", unavailable)
    output_dir = tmp_path / "output"
    result = run_alchemy(
        output_dir,
        "--id",
        "zzzz",
        "--workers",
        "1",
        ccp4_environ=ccp4_env,
        cache=tmp_path / "empty-cache",
        tmp_root=tmp_path,
    )
    assert result.exit_code == 1
    assert "not found locally and download failed" in result.text
    assert "Traceback" not in result.text
    # The run is abandoned before any output table is created, but the run log
    # still records the attempt.
    assert not os.path.exists(os.path.join(str(output_dir), "manifest.csv"))
    assert len(log_paths(output_dir)) == 1
    assert "Driver error:" in open(log_paths(output_dir)[0], encoding="utf-8").read()


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_mtz_without_map_coefficients_fails_the_entry_cleanly(
    tmp_path, entry_cache, ccp4_env
):
    """A missing FWT/PHWT/DELFWT/PHDELWT set fails one entry, not the process.

    README: "The MTZ input must contain FWT, PHWT, DELFWT, and PHDELWT columns."
    The failure must name them, be retryable, and exit nonzero -- and it must
    not be mistaken for a bond stage that ran and found nothing.
    """
    output_dir = tmp_path / "output"
    broken = _mtz_without_map_coefficients(
        entry_file(entry_cache, "9myr", "9myr_final.mtz"), tmp_path / "bad.mtz"
    )
    result = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        "--cif-file",
        entry_file(entry_cache, "9myr", "9myr_final.cif"),
        "--mtz-file",
        broken,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.text
    assert "completed with incomplete entries" in result.text

    row = manifest_by_id(output_dir)["9myr"]
    assert row["status"] == "error"
    assert row["retryable"] == "True"
    assert row["reason_codes"] == "unexpected_processing_error"
    for column in inputs.MAP_COEFFICIENT_COLUMNS:
        assert column in row["error"], row["error"]
    assert row["n_bonds"] == "" and row["n_candidates"] == ""

    # The other outputs exist and are empty rather than absent or malformed.
    assert read_rows(output_dir, "metal_stats_all.csv") == []
    assert read_header(output_dir, "metal_bonds_all.csv") == BOND_COLUMNS
    assert [
        name for name in os.listdir(str(output_dir)) if name.startswith(".alchemy-")
    ] == []


@pytest.mark.parametrize("value", ["0", "-1"])
def test_workers_below_one_is_rejected_before_any_work(tmp_path, value):
    """``--workers`` must be at least 1 and is refused during argument parsing.

    README: "--workers <n> ... must be at least 1". Rejection happens before the
    run starts, so no output directory and no log are created. This test needs
    neither CCP4 nor the network.
    """
    output_dir = tmp_path / "output"
    result = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        value,
        cache=tmp_path / "cache",
        tmp_root=tmp_path,
    )
    assert result.exit_code == 2
    assert "--workers" in result.stderr
    assert "at least 1" in result.stderr
    assert not os.path.exists(output_dir)


# 11. Database-referenced confidence scoring. The published severity anchors
# below are restated here rather than imported, so this module scores sites
# independently of the code it is checking.
# ``test_the_scoring_policy_under_test_is_the_shipped_one`` keeps the two
# copies honest; the anchors are part of the frozen policy that
# ``confidence_reference_id`` is derived from, so they cannot drift quietly.
_DENSITY_ANCHORS = ((2.0, 0.0), (3.0, 0.25), (6.0, 0.65), (12.0, 1.0))
_GEOMETRY_ANCHORS = ((2.0, 0.0), (3.0, 0.35), (6.0, 0.70), (12.0, 1.0))
_WEIGHTS = {"density": 0.50, "geometry": 0.35, "interaction": 0.15}

#: Evidence grid the frozen test reference is built from: eight RSZD values
#: spread over every density-anchor interval, crossed with five maximum
#: ``|Zbond|`` values, so the cohort exercises the geometry and interaction
#: terms as well as the density one and spans nearly the whole 0-100 range.
_COHORT_RSZD = (0.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0)
_COHORT_ZBOND = (1.0, 2.5, 4.0, 7.0, 10.0)

#: A second cohort: same policy, a different database snapshot.
_ALTERNATE_RSZD = (1.0, 2.2, 3.3, 4.4, 5.5, 7.7)
_ALTERNATE_ZBOND = (0.5, 3.5, 6.0, 9.0)

#: 9myr chain B's zinc scores 98.75: RSZD 2.1 is one tenth of the way
#: through the 2.0-3.0 density interval, so SR = 0.025, while every |Zbond| is
#: below the 2.0 geometry floor. Only the density term contributes:
#: ``100 * (1 - 0.50 * 0.025)``.
_9MYR_B_CONFIDENCE = 98.75
#: ``_RSZD_TOLERANCE`` of RSZD slack inside the 2.0-3.0 interval is
#: ``0.3 * 0.25 * 0.50 * 100 = 3.75`` points of score.
_9MYR_B_CONFIDENCE_TOLERANCE = 3.75


def _severity(value: float, anchors) -> float:
    """Piecewise-linear bounded severity, re-derived from ``anchors``."""
    if not math.isfinite(value) or value < 0.0:
        return math.nan
    if value <= anchors[0][0]:
        return anchors[0][1]
    for (low_x, low_y), (high_x, high_y) in zip(anchors, anchors[1:]):
        if value <= high_x:
            fraction = (value - low_x) / (high_x - low_x)
            return low_y + fraction * (high_y - low_y)
    return anchors[-1][1]


def readme_confidence(rszd: float, max_abs_zbond: float, coverage: float) -> float:
    """``100 * (1 - 0.50*SR - 0.35*QG*SG - 0.15*QG*sqrt(SR*SG))``.

    The README's published score, computed here from the site's own recorded
    evidence so a run's numbers are checked against the documentation rather
    than against the implementation that produced them.
    """
    coverage = min(1.0, max(0.0, coverage))
    density = _severity(rszd, _DENSITY_ANCHORS)
    geometry = _severity(max_abs_zbond, _GEOMETRY_ANCHORS) if coverage else 0.0
    score = 100.0 * (
        1.0
        - _WEIGHTS["density"] * density
        - _WEIGHTS["geometry"] * coverage * geometry
        - _WEIGHTS["interaction"] * coverage * math.sqrt(density * geometry)
    )
    return min(100.0, max(0.0, score))


def frozen_cohort(reference_dir) -> List[Tuple[float, int]]:
    """``[(score, count)]`` read straight from a published distribution file."""
    path = os.path.join(
        str(reference_dir), confidence_score.REFERENCE_DISTRIBUTION_FILE
    )
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            (float(row["confidence_score"]), int(row["count"]))
            for row in csv.DictReader(handle)
        ]


def reference_metadata(reference_dir) -> Dict[str, object]:
    """The published policy metadata of a frozen reference directory."""
    path = os.path.join(str(reference_dir), confidence_score.REFERENCE_METADATA_FILE)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def average_rank_percentile(cohort, score: float, tolerance: float = 1e-6) -> float:
    """Average-rank ECDF of ``score`` in ``cohort``, computed independently.

    ``tolerance`` groups a printed score with the cohort value it came from:
    scores reach the CSV rounded to six decimals while the frozen distribution
    keeps them at full precision. Distinct cohort scores are three orders of
    magnitude further apart than that, so nothing else can be merged by it.
    """
    total = sum(count for _, count in cohort)
    below = sum(count for value, count in cohort if value < score - tolerance)
    equal = sum(count for value, count in cohort if abs(value - score) <= tolerance)
    return 100.0 * (below + 0.5 * equal) / total


def frozen_reference(
    directory, rszd_values=_COHORT_RSZD, zbond_values=_COHORT_ZBOND
) -> str:
    """Publish a frozen confidence reference and return its directory.

    Built the way a real one is: a compact confidence-input CSV is finalized by
    ``src/confidence_score.py finalize`` -- the documented recovery entry point
    -- which is what writes ``metadata.json`` and ``score_distribution.csv``.
    Two rows carry no RSZD, so the scorable cohort is genuinely smaller than the
    input row count and the two cannot be confused.
    """
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    rows = []
    for index, (rszd, zbond) in enumerate(itertools.product(rszd_values, zbond_values)):
        row = {column: "" for column in confidence_score.CONFIDENCE_INPUT_COLUMNS}
        row.update(
            pdbID=f"c{index:03d}",
            category="ion",
            selected_metal_site_status="selected",
            metal_element="ZN",
            rszd_magnitude=repr(rszd),
            max_abs_zbond=repr(zbond),
            geometry_coverage="1",
            assigned_contact_count="4",
            reference_covered_contact_count="4",
            zbond_available_contact_count="4",
            confidence_inputs_status="complete",
        )
        rows.append(row)
    for index in range(2):
        row = {column: "" for column in confidence_score.CONFIDENCE_INPUT_COLUMNS}
        row.update(
            pdbID=f"u{index:03d}",
            selected_metal_site_status="selected",
            confidence_inputs_status="unscorable",
            confidence_inputs_missing_reasons="rszd_unavailable",
        )
        rows.append(row)

    inputs = os.path.join(directory, "confidence_inputs.csv")
    with open(inputs, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=confidence_score.CONFIDENCE_INPUT_COLUMNS
        )
        writer.writeheader()
        writer.writerows(rows)

    reference_dir = os.path.join(directory, "confidence_reference")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = confidence_score.main(
            [
                "finalize",
                "--input",
                inputs,
                "--output",
                os.path.join(directory, "confidence_scores.csv"),
                "--reference-dir",
                reference_dir,
            ]
        )
    assert code == 0, out.getvalue() + err.getvalue()
    return reference_dir


CONFIDENCE_COLUMNS = list(confidence_score.CONFIDENCE_INPUT_COLUMNS) + list(
    confidence_score.ANALYSIS_COLUMNS
)


def assert_policy_was_applied(rows, reference_dir):
    """Every scored row obeys the README formula and the frozen cohort's ECDF.

    This is the check the whole confidence feature reduces to: the score is the
    published function of the site's own evidence, and the percentile places
    that score in the *frozen database* cohort rather than in the handful of
    sites this run happened to look at.
    """
    metadata = reference_metadata(reference_dir)
    cohort = frozen_cohort(reference_dir)
    cohort_size = sum(count for _, count in cohort)
    assert cohort_size == metadata["scorable_cohort_size"]

    scored = 0
    for row in rows:
        where = (
            f"{row['pdbID']} {row['metal_resname']}"
            f"{row['metal_resnum']}/{row['metal_atom']}"
        )
        # Which snapshot produced this row is recorded on the row itself.
        assert row["confidence_reference_id"] == metadata["reference_id"], where
        assert int(row["confidence_cohort_size"]) == cohort_size, where
        if row["confidence_scoring_status"] == "unscorable":
            assert row["confidence_score"] == "", where
            assert row["confidence_percentile"] == "", where
            continue

        score = float(row["confidence_score"])
        assert 0.0 <= score <= 100.0, where
        expected = readme_confidence(
            float(row["rszd_magnitude"]),
            float(row["max_abs_zbond"]) if row["max_abs_zbond"] else math.nan,
            float(row["geometry_coverage"]),
        )
        # 1e-3 absorbs the six-decimal rounding of geometry_coverage in the CSV;
        # the score itself is reproduced to well under that.
        assert score == pytest.approx(expected, abs=1e-3), where
        assert float(row["confidence_percentile"]) == pytest.approx(
            average_rank_percentile(cohort, expected), abs=1e-6
        ), where
        scored += 1
    return scored


def test_the_scoring_policy_under_test_is_the_shipped_one(tmp_path):
    """The anchors and weights this module re-implements are Alchemy's own.

    ``assert_policy_was_applied`` deliberately recomputes scores from its own
    copy of the policy instead of calling ``confidence_score``. That is only an
    independent oracle while the two copies agree, and the policy is frozen: it
    is published in every reference's ``metadata.json`` and hashed into every
    ``confidence_reference_id``, so changing it invalidates every reference
    already in the wild. Needs neither CCP4 nor the network.
    """
    assert _DENSITY_ANCHORS == confidence_score.DENSITY_ANCHORS
    assert _GEOMETRY_ANCHORS == confidence_score.GEOMETRY_ANCHORS

    published = reference_metadata(frozen_reference(tmp_path / "policy"))
    assert published["density_anchors"] == [list(anchor) for anchor in _DENSITY_ANCHORS]
    assert published["geometry_anchors"] == [
        list(anchor) for anchor in _GEOMETRY_ANCHORS
    ]
    assert published["weights"] == _WEIGHTS
    assert published["percentile_method"] == "average_rank_empirical_cdf"


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_installed_reference_scores_every_selected_site_against_the_database(
    tmp_path, entry_cache, ccp4_env, batch
):
    """``--confidence-reference-dir`` makes a run emit its confidence scores.

    README: "For later single-entry, ID-file, manual, or capped runs, place the
    published database reference files in ... `confidence_reference/`, or select
    another copy with `--confidence-reference-dir`. Alchemy loads that reference
    once, derives each new site's compact inputs while its normal result is
    still in memory, and writes `confidence_scores_all.csv` directly. These runs
    are compared with the frozen database and never generate percentiles from
    their own small cohort."

    This is the product's differentiating output, so it is asserted as output:
    one row per manifest-counted selected metal site, a README-formula score in
    [0, 100] derived from that site's own density and geometry evidence, a
    percentile relative to the frozen cohort rather than to these thirteen sites,
    and the reference identifier recorded on every row.
    """
    reference_dir = frozen_reference(tmp_path / "installed")
    output_dir = tmp_path / "output"
    scored_run = run_batch(
        output_dir,
        entry_cache,
        ccp4_env,
        tmp_root=tmp_path,
        reference_dir=reference_dir,
    )
    assert scored_run.result.exit_code == 0, scored_run.result.text

    written = set(os.listdir(str(output_dir)))
    assert "confidence_scores_all.csv" in written
    # A referenced run scores in place; it neither streams compact inputs nor
    # publishes a reference of its own, both of which belong to database runs.
    assert "confidence_inputs_all.csv" not in written
    assert "confidence_reference" not in written
    assert read_header(output_dir, "confidence_scores_all.csv") == (CONFIDENCE_COLUMNS)

    rows = read_rows(output_dir, "confidence_scores_all.csv")
    # One row per selected metal site, entry by entry -- the guarantee
    # complete_confidence_site_count exists to keep.
    for pdb_id, manifest_row in scored_run.manifest.items():
        assert len(rows_for(rows, pdb_id)) == int(manifest_row["n_metals"]), pdb_id
    assert len(rows) == 13
    assert rows_for(rows, "9nxl") == []

    metadata = reference_metadata(reference_dir)
    cohort_size = metadata["scorable_cohort_size"]
    assert isinstance(cohort_size, int)
    assert cohort_size == len(_COHORT_RSZD) * len(_COHORT_ZBOND)
    assert metadata["input_row_count"] == cohort_size + 2
    assert assert_policy_was_applied(rows, reference_dir) == len(rows)

    # The cohort is the frozen database, not this run's thirteen sites.
    ranked = sorted(
        (float(row["confidence_score"]), float(row["confidence_percentile"]))
        for row in rows
    )
    percentiles = [percentile for _, percentile in ranked]
    assert all(0.0 < value < 100.0 for value in percentiles)
    assert len(set(percentiles)) > 1
    # A percentile is a rank, so it never disagrees with the score ordering.
    assert percentiles == sorted(percentiles)

    # 9myr chain B zinc: its confidence follows from the RSZD the density arm
    # measured for it, and from the density term alone.
    zinc = next(
        row for row in rows if row["pdbID"] == "9myr" and row["metal_chain"] == "B"
    )
    site = next(row for row in rows_for(batch.stats, "9myr") if row["CI"] == "B")
    assert float(zinc["rszd_magnitude"]) == pytest.approx(
        abs(float(site["ZDm"])), abs=1e-6
    )
    assert float(zinc["max_abs_zbond"]) < _GEOMETRY_ANCHORS[0][0]
    assert float(zinc["geometry_coverage"]) == 1.0
    assert zinc["confidence_scoring_status"] == "complete"
    assert float(zinc["confidence_score"]) == pytest.approx(
        _9MYR_B_CONFIDENCE, abs=_9MYR_B_CONFIDENCE_TOLERANCE
    )

    assert (
        f"13 confidence rows compared with database cohort {cohort_size}"
        in scored_run.result.stdout
    )
    log_text = open(log_paths(output_dir)[0], encoding="utf-8").read()
    assert "confidence_mode: reference" in log_text
    assert "confidence_status: scored_against_reference" in log_text


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_uncapped_database_run_finalizes_and_publishes_its_own_reference(
    tmp_path, entry_cache, ccp4_env
):
    """A run over a whole mirror streams inputs and then freezes a reference.

    README: confidence inputs are collected "as part of an uncapped
    full-database run: no `--id`, `--id-file`, manual coordinate arguments, or
    `--max-pdbs`", streamed to `confidence_inputs_all.csv`, and "only after the
    database run completes without operationally incomplete entries does Alchemy
    finalize confidence", writing `confidence_scores_all.csv` and a reusable
    `confidence_reference/`.

    The mirror here links the three cached entries where directory symlinks are
    supported and copies them otherwise. It is a complete database as far as the
    driver is concerned: enumeration finds exactly those entries and nothing
    caps the run. That makes the database arm -- the one that produces the
    reference everybody else consumes -- reachable without a 24k-entry mirror.
    """
    mirror = os.path.join(str(tmp_path), "mirror")
    for pdb_id in ENTRY_IDS:
        entry = inputs.entry_dir_for(mirror, pdb_id)
        os.makedirs(os.path.dirname(entry), exist_ok=True)
        source = inputs.entry_dir_for(entry_cache, pdb_id)
        try:
            os.symlink(source, entry, target_is_directory=True)
        except (NotImplementedError, OSError):
            shutil.copytree(source, entry)

    output_dir = tmp_path / "output"
    result = run_alchemy(
        output_dir,
        "--workers",
        "3",
        mirror=mirror,
        ccp4_environ=ccp4_env,
        cache=tmp_path / "cache",
        tmp_root=tmp_path,
    )
    assert result.exit_code == 0, result.text
    assert f"enumerating final PDB-REDO entries under {mirror}" in result.text

    written = set(os.listdir(str(output_dir)))
    assert {
        "confidence_inputs_all.csv",
        "confidence_scores_all.csv",
        "confidence_reference",
    } <= written
    assert read_header(output_dir, "confidence_inputs_all.csv") == list(
        confidence_score.CONFIDENCE_INPUT_COLUMNS
    )
    assert read_header(output_dir, "confidence_scores_all.csv") == (CONFIDENCE_COLUMNS)

    reference_dir = os.path.join(str(output_dir), "confidence_reference")
    manifest = manifest_by_id(output_dir)
    streamed_inputs = read_rows(output_dir, "confidence_inputs_all.csv")
    rows = read_rows(output_dir, "confidence_scores_all.csv")

    # Every selected metal site of every entry reaches both files exactly once.
    assert (
        len(streamed_inputs)
        == len(rows)
        == sum(int(row["n_metals"]) for row in manifest.values())
        == 13
    )
    for pdb_id, manifest_row in manifest.items():
        assert len(rows_for(rows, pdb_id)) == int(manifest_row["n_metals"]), pdb_id
    # Scoring adds columns to the streamed inputs; it does not re-derive them.
    for streamed, scored in zip(streamed_inputs, rows):
        for column in confidence_score.CONFIDENCE_INPUT_COLUMNS:
            assert scored[column] == streamed[column], column

    metadata = reference_metadata(reference_dir)
    assert metadata["input_row_count"] == len(rows)
    assert assert_policy_was_applied(rows, reference_dir) == len(rows)
    # These sites *are* the cohort, so their average empirical rank is 50 even
    # when tied scores share a percentile.
    percentiles = [float(row["confidence_percentile"]) for row in rows]
    assert sum(percentiles) / len(percentiles) == pytest.approx(50.0)

    assert (
        f"13 confidence rows (13 scored; reference cohort 13) -> "
        f"{os.path.join(str(output_dir), 'confidence_scores_all.csv')}" in result.stdout
    )
    log_text = open(log_paths(output_dir)[0], encoding="utf-8").read()
    assert "confidence_mode: database" in log_text
    assert "confidence_status: finalized" in log_text

    # The published reference is directly reusable by a later scored run.
    reused = tmp_path / "reused"
    second = run_alchemy(
        reused,
        "--id",
        "9myr",
        "--workers",
        "1",
        reference_dir=reference_dir,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path / "reused-tmp",
    )
    assert second.exit_code == 0, second.text
    reused_rows = read_rows(reused, "confidence_scores_all.csv")
    assert len(reused_rows) == 2
    assert assert_policy_was_applied(reused_rows, reference_dir) == 2
    assert [row["confidence_score"] for row in reused_rows] == [
        row["confidence_score"] for row in rows_for(rows, "9myr")
    ]


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_resume_refuses_to_mix_two_database_snapshots(tmp_path, entry_cache, ccp4_env):
    """Resuming against a different frozen reference is refused, not blended.

    README: "A deterministic `confidence_reference_id` identifies the exact
    frozen policy and score distribution and prevents resumed output from mixing
    database snapshots." A percentile is only meaningful relative to one cohort,
    so half a file scored against another database is worse than no file: it
    reads as a single comparable column and is not one.

    The same-reference resume is asserted first, so a failure of the mismatch
    case cannot be a resume that was broken for some unrelated reason.
    """
    installed = frozen_reference(tmp_path / "installed")
    other = frozen_reference(tmp_path / "other", _ALTERNATE_RSZD, _ALTERNATE_ZBOND)
    assert (
        reference_metadata(installed)["reference_id"]
        != reference_metadata(other)["reference_id"]
    )

    output_dir = tmp_path / "output"
    first = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        reference_dir=installed,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert first.exit_code == 0, first.text
    scores_path = os.path.join(str(output_dir), "confidence_scores_all.csv")
    original = open(scores_path, "rb").read()

    same = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        "--resume",
        reference_dir=installed,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert same.exit_code == 0, same.text
    assert "no entries to process" in same.text

    mismatched = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        "--resume",
        reference_dir=other,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert mismatched.exit_code == 1
    assert "Traceback" not in mismatched.text
    assert "Cannot resume confidence output" in mismatched.text
    assert "different database reference" in mismatched.text
    # Refused before anything was rewritten.
    assert open(scores_path, "rb").read() == original

    # And the converse gap: resuming a run that has no confidence output at all
    # would silently score only the entries the resume happens to touch.
    unscored = tmp_path / "unscored"
    seed = run_alchemy(
        unscored,
        "--id",
        "9myr",
        "--workers",
        "1",
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path / "unscored-tmp",
    )
    assert seed.exit_code == 0, seed.text
    assert not os.path.exists(os.path.join(str(unscored), "confidence_scores_all.csv"))
    upgraded = run_alchemy(
        unscored,
        "--id",
        "9myr",
        "--workers",
        "1",
        "--resume",
        reference_dir=installed,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path / "unscored-tmp",
    )
    assert upgraded.exit_code == 1
    assert "Cannot resume confidence-aware output" in upgraded.text
    assert "use a fresh output directory" in upgraded.text


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
@pytest.mark.parametrize(
    "damage,message",
    [
        ("weights", "weights is incompatible with this code"),
        ("distribution", "identifier does not match data"),
    ],
)
def test_an_incompatible_reference_stops_the_run_instead_of_scoring(
    tmp_path, entry_cache, ccp4_env, damage, message
):
    """A reference that is not the one it claims to be aborts the run.

    Two ways a reference directory can be wrong: its policy no longer matches
    this code (``weights``), and its distribution no longer matches the
    identifier published beside it (``distribution``). Either would produce
    scores that are silently not comparable with anything, so the driver must
    refuse the run rather than emit them -- and refuse before any entry is
    processed, so nothing half-scored is left behind.
    """
    reference_dir = frozen_reference(tmp_path / "installed")
    metadata_path = os.path.join(
        reference_dir, confidence_score.REFERENCE_METADATA_FILE
    )
    if damage == "weights":
        metadata = reference_metadata(reference_dir)
        weights = metadata.get("weights")
        assert isinstance(weights, dict)
        weights["density"] = 0.6
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)
    else:
        distribution_path = os.path.join(
            reference_dir, confidence_score.REFERENCE_DISTRIBUTION_FILE
        )
        cohort = frozen_cohort(reference_dir)
        with open(distribution_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("confidence_score", "count"))
            for index, (score, count) in enumerate(cohort):
                writer.writerow((repr(score), count + (1 if index == 0 else 0)))

    output_dir = tmp_path / "output"
    result = run_alchemy(
        output_dir,
        "--id",
        "9myr",
        "--workers",
        "1",
        reference_dir=reference_dir,
        ccp4_environ=ccp4_env,
        cache=entry_cache,
        tmp_root=tmp_path,
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.text
    assert "Invalid confidence reference" in result.text
    assert message in result.text

    # Nothing was analysed, so no output table and no partially scored file.
    assert not os.path.exists(os.path.join(str(output_dir), "manifest.csv"))
    assert not os.path.exists(
        os.path.join(str(output_dir), "confidence_scores_all.csv")
    )
    assert len(log_paths(output_dir)) == 1
