"""End-to-end pipeline behaviour: ``src/main.py`` driven as a whole run.

Most tests here execute the real driver over a checksum-pinned PDB-REDO entry
snapshot, so they need CCP4 and either a warm entry cache or network access.
The ``network`` marker is applied only when the configured cache is not already
complete, so ``pytest -m 'not network'`` can use a warm cache and still never
opens a socket.

``9myr``  Sal-Shy(DUF35) aldolase; two zinc sites, each declared as a Cys3-His
          zinc-ribbon tetrad in ``_struct_conn``.
``6nlr``  PIXE-remodelled histidinol phosphatase; eleven Mn, Co, Fe and Ca
          sites across three chains, the multi-site multi-element case.
``9nxl``  steroid aldehyde dehydrogenase with no modelled metal, the
          ``no_metals`` control.
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
from pathlib import Path
from typing import (
    Protocol,
    TypeVar,
    cast,
)
from collections.abc import Callable, Generator, Sequence

import pytest

import confidence_score
import crystallization_conditions
import helpers
import inputs
import main
from driver import runlog
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS
from coordination.schema import BOND_COLUMNS, CANDIDATE_COLUMNS


class _ApproxFactory(Protocol):
    """The concrete numeric subset of pytest's broadly typed approx helper."""

    def __call__(
        self,
        expected: object,
        rel: float | None = None,
        abs: float | None = None,
        nan_ok: bool = False,
    ) -> object: ...


class _PytestApi(Protocol):
    approx: _ApproxFactory


approx = cast(_PytestApi, pytest).approx


_StrPath = str | os.PathLike[str]

ENTRY_IDS = ("9myr", "6nlr", "9nxl")

# PDB-REDO's live entry URLs are mutable, so pinning the bytes turns an upstream
# revision into an explicit snapshot-drift failure instead of a mysterious change
# in Alchemy's numbers. Uncompressed mmCIF, MTZ and data.json only: production
# also accepts compressed and legacy-PDB variants, but those are not
# interchangeable numerical regression inputs.
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

_HIS_N_ZN = ("HIS", "N", "ZN")

# ``(entry, chain, residue number) -> (ZDm, ZD-m, ZD+m)`` measured by the whole
# density arm -- mtzfix, both FFTs, mapmask and edstats -- over the pinned
# snapshot above, the only genuine oracle that arm has.
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

# EDSTATS reports RSZD to one decimal, so 0.05 is already lost to printing; 0.3
# also absorbs map-grid differences between CCP4 builds while still pinning every
# value to ~7 percent.
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
    mismatches: list[str] = []
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


_CONFIGURED_ENTRY_CACHE = helpers.cache_dir_from_env()
_ENTRY_SNAPSHOT_IS_WARM = bool(
    _CONFIGURED_ENTRY_CACHE
    and all(_entry_ready(_CONFIGURED_ENTRY_CACHE, pdb_id) for pdb_id in ENTRY_IDS)
)


_TestFunc = TypeVar("_TestFunc", bound=Callable[..., None])


def _requires_entry_data(test: _TestFunc) -> _TestFunc:
    """Mark a snapshot-backed test, plus its conditional network requirement."""
    test = pytest.mark.entry_data(test)
    if not _ENTRY_SNAPSHOT_IS_WARM:
        test = pytest.mark.network(test)
    return test


@pytest.fixture(scope="session")
def entry_cache(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A PDB-REDO cache root holding every entry in :data:`ENTRY_IDS`.

    An unreachable ``pdb-redo.eu`` skips, but a ``download_entry_to_cache`` that
    returns without the final model files is a regression and must not be able
    to hide as a skip.
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
        # download_entry_to_cache funnels a dead socket, a 404 and a renamed file
        # alike into FileNotFoundError, so only a fresh re-probe separates "no
        # network" from a broken downloader.
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
def _environment(overrides: dict[str, str] | None) -> Generator[None, None, None]:
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
    output_dir: _StrPath,
    *args: object,
    ccp4_environ: dict[str, str] | None = None,
    cache: _StrPath | None = None,
    tmp_root: _StrPath | None = None,
    mirror: _StrPath | None = None,
    reference_dir: _StrPath | None = None,
) -> RunResult:
    """Invoke the driver in-process and return its exit code and output.

    Every path option that otherwise defaults into the repository is supplied
    here, so no test can write into the checkout. Omitting ``mirror`` or
    ``reference_dir`` points that option at a non-existent path, which is what
    forces cache-only input and leaves confidence scoring off.
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


def id_file(directory: _StrPath, pdb_ids: Sequence[str]) -> str:
    """Write a whitespace-separated ``--id-file`` and return its path."""
    os.makedirs(str(directory), exist_ok=True)
    path = os.path.join(str(directory), "ids.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(pdb_ids) + "\n")
    return path


def read_rows(output_dir: _StrPath, name: str) -> list[dict[str, str]]:
    with open(os.path.join(str(output_dir), name), newline="") as handle:
        return list(csv.DictReader(handle))


def read_header(output_dir: _StrPath, name: str) -> list[str]:
    with open(os.path.join(str(output_dir), name), newline="") as handle:
        return next(csv.reader(handle))


def read_text(path: _StrPath) -> str:
    return Path(path).read_text(encoding="utf-8")


def manifest_by_id(output_dir: _StrPath) -> dict[str, dict[str, str]]:
    return {row["pdbID"]: row for row in read_rows(output_dir, "manifest.csv")}


def rows_for(rows: Sequence[dict[str, str]], pdb_id: str) -> list[dict[str, str]]:
    return [row for row in rows if row["pdbID"] == pdb_id]


def log_paths(output_dir: _StrPath) -> list[str]:
    """Every run log in this output directory."""
    directory = os.path.join(str(output_dir), runlog.DEFAULT_LOG_DIRNAME)
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if re.fullmatch(r"alchemy_run_\d{8}(_\d+)?\.log", name)
    )


def literature_reference(
    residue: str, atom_element: str, metal: str
) -> tuple[float, float]:
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
    manifest_rows: list[dict[str, str]]
    manifest: dict[str, dict[str, str]]
    stats: list[dict[str, str]]
    bonds: list[dict[str, str]]
    candidates: list[dict[str, str]]


def run_batch(
    output_dir: _StrPath,
    cache: _StrPath,
    ccp4_environ: dict[str, str] | None,
    pdb_ids: Sequence[str] = ENTRY_IDS,
    extra: Sequence[str] = (),
    tmp_root: _StrPath | None = None,
    reference_dir: _StrPath | None = None,
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
        stats=read_rows(output_dir, "metal_sites_all.csv"),
        bonds=read_rows(output_dir, "metal_bonds_all.csv"),
        candidates=read_rows(output_dir, "metal_contact_candidates_all.csv"),
    )


@pytest.fixture(scope="session")
def batch(
    tmp_path_factory: pytest.TempPathFactory, entry_cache: str, ccp4_env: dict[str, str]
) -> Batch:
    """A single completed three-entry run, shared read-only by several tests."""
    root = tmp_path_factory.mktemp("batch")
    result = run_batch(root / "output", entry_cache, ccp4_env, tmp_root=root)
    assert result.result.exit_code == 0, result.result.text
    return result


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_single_entry_run_writes_documented_outputs(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """One ``--id`` run writes exactly the documented output set.

    README "Results are written to": manifest, statistics, bond and candidate
    CSVs plus one timestamped run log, each with its module's full schema.
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
        "metal_sites_all.csv",
        "metal_bonds_all.csv",
        "metal_contact_candidates_all.csv",
        "crystallization_conditions_all.csv",
        "crystallization_summary_all.csv",
        "review_queue_all.csv",
    } <= written
    assert len(log_paths(output_dir)) == 1

    # No frozen reference is installed, so classifications are emitted without
    # empirical component rankings.
    assert "confidence_scores_all.csv" in written
    assert "confidence_inputs_all.csv" not in written
    assert "no frozen confidence reference is installed" in result.text
    confidence_rows = read_rows(output_dir, "confidence_scores_all.csv")
    assert read_header(output_dir, "confidence_scores_all.csv") == CONFIDENCE_COLUMNS
    assert all(
        row["alchemy_level"] in {"PASS", "REVIEW", "SUSPECT"} for row in confidence_rows
    )
    assert all(row["alchemy_score"] == "" for row in confidence_rows)

    assert [name for name in written if name.startswith(".alchemy-")] == []

    assert read_header(output_dir, "manifest.csv") == MANIFEST_COLUMNS
    assert read_header(output_dir, "metal_sites_all.csv") == STATS_COLUMNS
    assert read_header(output_dir, "metal_bonds_all.csv") == BOND_COLUMNS
    assert (
        read_header(output_dir, "metal_contact_candidates_all.csv") == CANDIDATE_COLUMNS
    )
    assert read_header(output_dir, "crystallization_conditions_all.csv") == list(
        crystallization_conditions.CONDITION_COLUMNS
    )
    assert read_header(output_dir, "crystallization_summary_all.csv") == list(
        crystallization_conditions.SUMMARY_COLUMNS
    )
    assert read_header(output_dir, "review_queue_all.csv") == [
        *CONFIDENCE_COLUMNS,
        *crystallization_conditions.REVIEW_CONTEXT_COLUMNS,
    ]

    # Columns the README names must be present, not merely equal to the code's
    # own constant, which would drift away from the documentation unnoticed.
    stats_header = read_header(output_dir, "metal_sites_all.csv")
    assert set(helpers.EDSTATS_HEADER) <= set(stats_header)
    assert {
        "metal_site_id",
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
        "metal_site_id",
        "contact_id",
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
        "metal_site_id",
        "contact_id",
        "assigned_as_bond",
        "first_sphere_eligible",
        "eligibility_status",
        "eligibility_reason",
        "candidate_source",
        "inferred_donor_allowed",
        "first_sphere_cutoff",
        "assignment_reference_kind",
    } <= set(CANDIDATE_COLUMNS)
    assert {
        "no_metals",
        "metal_site_limit_exceeded",
        "n_metals",
        "n_bonds",
        "n_candidates",
        "status",
        "retryable",
        "reason_codes",
        "status_detail",
        "runtime_s",
        "pdb_redo_version",
        "pdb_redo_date",
        "source_coordinate_path",
    } <= set(MANIFEST_COLUMNS)
    assert "analysis_coordinate_path" not in MANIFEST_COLUMNS

    manifest = manifest_by_id(output_dir)["9myr"]
    assert manifest["source_coordinate_path"] == "my/9myr/9myr_final.cif"
    assert manifest["pdb_redo_version"]
    assert manifest["pdb_redo_date"]


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_9myr_reports_two_chemically_sane_zinc_ribbon_sites(batch: Batch) -> None:
    """9myr yields two Cys3-His zinc sites with measured density and geometry.

    The z-scores are re-derived from the README formula and the bundled
    literature table, so a change in the scoring path fails here too.
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
            assert float(site[column]) == approx(value, abs=_RSZD_TOLERANCE), (
                chain,
                column,
            )
        rszd = float(site["ZDm"])
        assert math.isfinite(rszd)
        assert abs(rszd) == approx(
            max(abs(float(site["ZD-m"])), abs(float(site["ZD+m"]))), abs=0.05
        )

        site_bonds = by_chain[chain]
        assert len(site_bonds) == 4
        assert {
            (row["neighbor_resname"], row["neighbor_atom"], row["neighbor_resnum"])
            for row in site_bonds
        } == expected_donors
        assert all(row["declared_connection"] == "true" for row in site_bonds)

        for bond in site_bonds:
            distance = float(bond["distance"])
            mu, sigma = literature_reference(
                bond["neighbor_resname"], bond["neighbor_element"], "ZN"
            )
            assert float(bond["literature_distance"]) == approx(mu)
            assert float(bond["literature_stdev"]) == approx(sigma)
            dpi = float(bond["dpi"])
            assert math.isfinite(dpi) and dpi > 0.0
            denominator = math.sqrt(dpi**2 + sigma**2)
            expected_z = (distance - mu) / denominator
            # The z-score is computed from the full-precision distance, while
            # the CSV reports it rounded to three decimals, so recomputing here
            # inherits up to half a unit in that last place. Divided by the
            # denominator that is worth far more than the z-score's own
            # rounding, and a fixed tolerance would be tuned to one entry.
            tolerance = 0.0005 / denominator + 5e-5
            assert float(bond["zscore"]) == approx(expected_z, abs=tolerance)
            assert abs(expected_z) < float(bond["zscore_outlier_cutoff"])
            assert bond["geometry_outlier"] == "false"
            assert bond["geometry_consistent"] == "true"
            assert float(bond["sigma_mag"]) == approx(rszd)
            assert float(bond["sigma_neg"]) == approx(float(site["ZD-m"]))
            assert float(bond["sigma_pos"]) == approx(float(site["ZD+m"]))

    assert literature_reference(*_HIS_N_ZN) == approx((2.03, 0.05))


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_6nlr_multi_element_sites_report_their_measured_difference_density(
    batch: Batch,
) -> None:
    """All eleven PIXE-remodelled 6nlr sites carry their own RSZD triple.

    Four elements, three chains and repeated residue numbers: a map that is
    mis-scaled, mis-cropped or joined on an incomplete site key cannot pass.
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
            assert float(row[column]) == approx(value, abs=_RSZD_TOLERANCE), (
                chain,
                resnum,
                column,
            )
        assert abs(float(row["ZDm"])) == approx(
            max(abs(float(row["ZD-m"])), abs(float(row["ZD+m"]))), abs=0.05
        ), (chain, resnum)

    # The spread is a property of the snapshot, not a broad sanity bound.
    minimum = min(float(row["ZDm"]) for row in sites.values())
    assert {key for key, row in sites.items() if float(row["ZDm"]) == minimum} == {
        ("A", "302"),
        ("C", "303"),
    }
    assert max(sites, key=lambda key: float(sites[key]["ZDm"])) == ("C", "304")


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_declared_connections_measure_their_own_reported_distance(
    batch: Batch,
) -> None:
    """Every declared contact's measured distance matches the one it declares.

    gemmi's PDB writer emits a TER after each polymer and every TER consumes a
    serial, so translating source-mmCIF atom serials into analysis-PDB serial
    space re-points every partner behind a TER -- and both entries number their
    metals after their polymer chains. A ``_struct_conn`` row carries the
    distance the depositor measured, so a misbound declaration disagrees with
    its own number.
    """
    declared = [row for row in batch.bonds if row["declared_connection"] == "true"]
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
        assert measured == approx(reported, abs=0.02), where
        # A first-sphere metal-donor contact is a bond length, not a
        # lattice-scale separation.
        assert 1.5 < measured < 3.0, where
        assert row["coordination_source"] == "struct_conn", where
        zscore = float(row["zscore"])
        if math.isfinite(zscore):
            assert abs(zscore) < float(row["zscore_outlier_cutoff"]), where

    # Losing declarations is the other half of a broken partner lookup.
    assert len(declared) == 49, "declared connections went missing or duplicated"
    assert len(rows_for(declared, "6nlr")) == 41
    assert len(rows_for(declared, "9myr")) == 8


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_declared_contacts_reach_the_expected_zinc_ribbon_donors(
    batch: Batch,
) -> None:
    """Both 9myr zinc sites resolve to the deposited Cys3-His tetrad.

    Donor identity pins the serial mapping independently of geometry.
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
        assert all(row["declared_connection"] == "true" for row in rows)


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_no_metal_entry_is_reported_not_dropped(batch: Batch) -> None:
    """9nxl finishes as a successful negative result with zeroed counts.

    README: "structures with no recognized metal atoms finish with n_metals=0
    without running mtzfix, either FFT, or edstats", reported as ``no_metals``,
    an informational subset of ``ok``. The stage ran, so the counts are a
    measured ``0`` rather than blank.
    """
    row = batch.manifest["9nxl"]
    assert row["status"] == "ok"
    assert row["retryable"] == "false"
    assert row["no_metals"] == "true"
    assert row["metal_site_limit_exceeded"] == "false"
    assert row["reason_codes"] == ""
    assert row["status_detail"] == ""
    assert row["n_metals"] == "0"
    assert row["n_bonds"] == "0"
    assert row["n_candidates"] == "0"

    assert rows_for(batch.stats, "9nxl") == []
    assert rows_for(batch.bonds, "9nxl") == []
    assert rows_for(batch.candidates, "9nxl") == []

    # Headers survive even when an entry contributes no rows.
    assert read_header(batch.output_dir, "metal_sites_all.csv") == STATS_COLUMNS

    log_path = log_paths(batch.output_dir)[0]
    diagnostics_path = log_path.removesuffix(".log") + "_entries.csv"
    with open(diagnostics_path, newline="", encoding="utf-8") as handle:
        entry = next(row for row in csv.DictReader(handle) if row["pdbID"] == "9nxl")
    assert entry["no_metals"] == "true"
    assert entry["density_map_scope"] == ""
    assert entry["edstats_s"] == ""
    assert entry["mtzfix_s"] == ""
    log_text = read_text(log_path)
    assert "Metal-free entries: 1" in log_text


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_multi_entry_batch_aggregates_every_entry_exactly_once(batch: Batch) -> None:
    """A batch writes one manifest row per requested id and no cross-talk."""
    manifest_ids = [row["pdbID"] for row in batch.manifest_rows]
    assert len(batch.manifest_rows) == len(ENTRY_IDS)
    assert Counter(manifest_ids) == Counter(ENTRY_IDS)
    assert set(batch.manifest) == set(ENTRY_IDS)
    assert all(row["status"] == "ok" for row in batch.manifest.values())
    assert batch.result.exit_code == 0

    for table in (batch.stats, batch.bonds, batch.candidates):
        assert {row["pdbID"] for row in table} <= set(ENTRY_IDS)

    assert len(batch.bonds) == sum(
        int(row["n_bonds"]) for row in batch.manifest.values()
    )
    assert len(batch.candidates) == sum(
        int(row["n_candidates"]) for row in batch.manifest.values()
    )

    # Metals only ever appear in the entry they were read from.
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
def test_manifest_counts_match_the_rows_actually_written(
    batch: Batch, pdb_id: str
) -> None:
    """``n_metals``/``n_bonds``/``n_candidates`` agree with the CSV contents.

    README: ``n_metals`` counts distinct selected coordinate-model sites, not
    diagnostic or repeated EDSTATS rows.
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

    # Every bond was either discovered by the 4 A search or declared, and both
    # are recorded as candidate evidence.
    assert int(row["n_candidates"]) >= int(row["n_bonds"])


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_each_run_writes_its_own_immutable_log(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """A second invocation adds a suffixed log instead of replacing the first.

    README: "Additional runs on the same UTC date receive a numeric suffix ...
    Resume runs create another log rather than replacing the original record".
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
    original_text = read_text(original[0])

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
    assert read_text(original[0]) == original_text
    assert "no entries to process" in second.text


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_resume_over_a_completed_batch_adds_no_duplicate_rows(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """Resuming a finished batch reprocesses nothing and rewrites nothing.

    README: ``--resume`` skips ``ok`` and terminal ``partial`` outcomes without
    duplicating their previous rows.
    """
    output_dir = tmp_path / "output"
    first = run_batch(output_dir, entry_cache, ccp4_env, tmp_root=tmp_path)
    assert first.result.exit_code == 0, first.result.text

    names = (
        "manifest.csv",
        "metal_sites_all.csv",
        "metal_bonds_all.csv",
        "metal_contact_candidates_all.csv",
        "crystallization_conditions_all.csv",
        "crystallization_summary_all.csv",
        "review_queue_all.csv",
    )
    before = {name: Path(str(output_dir), name).read_bytes() for name in names}

    second = run_batch(
        output_dir, entry_cache, ccp4_env, extra=("--resume",), tmp_root=tmp_path
    )
    assert second.result.exit_code == 0, second.result.text
    assert "no entries to process" in second.result.text

    after = {name: Path(str(output_dir), name).read_bytes() for name in names}
    assert after == before

    assert len(second.manifest) == len(ENTRY_IDS)
    assert len(second.bonds) == len(first.bonds)
    assert len(second.candidates) == len(first.candidates)
    assert len(second.stats) == len(first.stats)


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_fresh_no_bonds_run_clears_bond_outputs_and_resume_restores_them(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """``--no-bonds`` erases stale bond output; a later resume fills it back in.

    README: "Entries originating from a bond-disabled run retain blank n_bonds
    and n_candidates, so a later bond-enabled resume will process them".
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
    assert not os.path.exists(
        os.path.join(str(output_dir), "metal_contact_candidates_all.csv")
    )

    blank = manifest_by_id(output_dir)
    assert set(blank) == set(ENTRY_IDS)
    for pdb_id, row in blank.items():
        assert row["status"] == "ok", pdb_id
        assert row["n_bonds"] == "", pdb_id
        assert row["n_candidates"] == "", pdb_id
    assert read_rows(output_dir, "metal_sites_all.csv")

    restored = run_batch(
        output_dir, entry_cache, ccp4_env, extra=("--resume",), tmp_root=tmp_path
    )
    assert restored.result.exit_code == 0, restored.result.text
    assert set(restored.manifest) == set(ENTRY_IDS)
    for pdb_id, row in restored.manifest.items():
        assert row["n_bonds"] != "", pdb_id
        assert int(row["n_bonds"]) == len(rows_for(restored.bonds, pdb_id))
        assert int(row["n_candidates"]) == len(rows_for(restored.candidates, pdb_id))
    assert len(restored.bonds) == len(complete.bonds)
    assert len(restored.stats) == len(complete.stats)
    assert len(restored.manifest) == len(ENTRY_IDS)


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_entry_failing_before_the_bond_stage_is_never_resumed_away(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """A pre-bond failure leaves blank counts, so bonds are still filled later.

    A seeded ``0`` would read as a measured zero: ``--resume --no-bonds``
    carries it onto the now-``ok`` row, and the next bond-enabled resume then
    judges the stage complete and skips the entry permanently with no bond rows
    written for it. The three runs below are that sequence.
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
    assert float(bonds[0]["distance"]) == approx(2.311, abs=0.02)


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_manual_files_with_data_json_reproduce_the_cached_entry_run(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str], batch: Batch
) -> None:
    """Manual ``--cif-file/--mtz-file/--data-json`` matches the automatic run.

    Manual mode selects inputs; it is not a different analysis.
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
    assert row["coordinate_conversion_performed"] == "true"

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
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str], batch: Batch
) -> None:
    """Omitting ``--data-json`` costs DPI only, and is not a failure.

    README: "Without --data-json there is no reflection count, so DPI and every
    value derived from it are unavailable ... the omission is reported as
    ``missing_dpi_metadata_source`` rather than as a calculation failure". A
    deterministic limitation is a terminal partial, and those exit successfully.
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
    assert row["retryable"] == "false"
    assert row["reason_codes"] == "missing_dpi_metadata_source"
    assert row["n_metals"] == "2"
    assert row["n_bonds"] == "8"

    site = read_rows(output_dir, "metal_sites_all.csv")[0]
    assert site["dpi_unavailable_reason"] == "missing_dpi_metadata_source"
    assert site["dpi"] == ""
    # Only the DPI-derived numbers are lost; density is unaffected.
    assert float(site["ZDm"]) == approx(float(rows_for(batch.stats, "9myr")[0]["ZDm"]))

    bond = read_rows(output_dir, "metal_bonds_all.csv")[0]
    reference = rows_for(batch.bonds, "9myr")[0]
    assert float(bond["distance"]) == approx(float(reference["distance"]))
    assert bond["neighbor_atom"] == "SG"
    assert bond["dpi"] == ""
    assert bond["zscore"] == ""
    assert float(bond["literature_distance"]) > 0.0
    assert bond["geometry_outlier"] == ""


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_full_and_model_envelope_map_scopes_give_identical_statistics(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """Cropping the FFT map to the model envelope changes no EDSTATS number.

    README: "Model-envelope mode still calculates each complete FFT map before
    cropping, so map values come from the same Fourier calculation as legacy
    full-map mode", which makes the default a pure performance optimisation.
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

    # Without cropping on at least one entry both runs took the same path and
    # the comparison proves nothing.
    cropped_log = read_text(log_paths(cropped.output_dir)[0])
    full_log = read_text(log_paths(full.output_dir)[0])
    assert re.search(r"density_map_scope=model-envelope\b", cropped_log), (
        "no entry exercised model-envelope cropping"
    )
    assert not re.search(r"density_map_scope=model-envelope\b", full_log)
    assert re.search(r"density_map_scope=full\b", full_log)

    def key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return (row["pdbID"], row["RT"], row["CI"], row["RN"])

    cropped_sites = {key(row): row for row in cropped.stats}
    full_sites = {key(row): row for row in full.stats}
    assert cropped_sites and set(cropped_sites) == set(full_sites)

    for site_key, row in cropped_sites.items():
        other = full_sites[site_key]
        for metric in helpers.EDSTATS_METRICS:
            assert row[metric] == other[metric], (site_key, metric)

    def bond_key(row: dict[str, str]) -> tuple[str, str, str, str]:
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


def _mtz_without_map_coefficients(source_mtz: str, destination: _StrPath) -> str:
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
    tmp_path: Path, ccp4_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An id that exists nowhere ends the run with an explanation and exit 1.

    The downloader is stubbed because this test carries no ``network`` marker
    and so must never open a socket.
    """

    def unavailable(pdb_id: str, cache_root: str) -> None:
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
    # Abandoned before any output table exists, but the log records the attempt.
    assert not os.path.exists(os.path.join(str(output_dir), "manifest.csv"))
    assert len(log_paths(output_dir)) == 1
    assert "Driver error:" in read_text(log_paths(output_dir)[0])


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_mtz_without_map_coefficients_fails_the_entry_cleanly(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """A missing FWT/PHWT/DELFWT/PHDELWT set fails one entry, not the process.

    README: "The MTZ input must contain FWT, PHWT, DELFWT, and PHDELWT
    columns". The failure names them and must not read as a bond stage that ran
    and found nothing. Its reason code reports that the failure will recur while
    the same file is supplied, without preventing a resume from retrying it.
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
    assert row["retryable"] == "true"
    assert row["reason_codes"] == "deterministic_processing_error"
    for column in inputs.MAP_COEFFICIENT_COLUMNS:
        assert column in row["status_detail"], row["status_detail"]
    assert row["n_bonds"] == "" and row["n_candidates"] == ""

    assert read_rows(output_dir, "metal_sites_all.csv") == []
    assert read_header(output_dir, "metal_bonds_all.csv") == BOND_COLUMNS
    assert [
        name for name in os.listdir(str(output_dir)) if name.startswith(".alchemy-")
    ] == []


@pytest.mark.parametrize("value", ["0", "-1"])
def test_workers_below_one_is_rejected_before_any_work(
    tmp_path: Path, value: str
) -> None:
    """``--workers`` must be at least 1, and is refused during argument parsing.

    Rejection precedes the run, so no output directory and no log are created.
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


# The final raw thresholds are restated here so this module remains an
# independent oracle. The policy test below keeps this copy aligned with the
# implementation and the same values are frozen into every reference version.
_DENSITY_THRESHOLDS = (3.0, 6.0)
_GEOMETRY_THRESHOLDS = (1.0, 2.0)

# Evidence grid used to build a small frozen test reference. It crosses all
# three levels in both components and includes missing-evidence rows.
_COHORT_RSZD = (0.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0)
_COHORT_RMS_ZBOND = (0.25, 0.75, 1.0, 1.5, 2.5)

# A second cohort: same policy, a different database snapshot.
_ALTERNATE_RSZD = (1.0, 2.2, 3.3, 4.4, 5.5, 7.7)
_ALTERNATE_RMS_ZBOND = (0.5, 1.2, 2.0, 3.0)


def raw_level(value: float, thresholds: tuple[float, float]) -> str:
    if not math.isfinite(value) or value < 0:
        return "INCOMPLETE"
    if value < thresholds[0]:
        return "PASS"
    if value < thresholds[1]:
        return "REVIEW"
    return "SUSPECT"


def frozen_cohort(
    reference_dir: _StrPath,
) -> dict[str, list[tuple[float, int]]]:
    """Read both raw component distributions from a published reference."""
    path = os.path.join(
        str(reference_dir), confidence_score.REFERENCE_DISTRIBUTION_FILE
    )
    result: dict[str, list[tuple[float, int]]] = {"density": [], "geometry": []}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["component"]].append((float(row["value"]), int(row["count"])))
    return result


def reference_metadata(reference_dir: _StrPath) -> dict[str, object]:
    """The published policy metadata of a frozen reference directory."""
    path = os.path.join(str(reference_dir), confidence_score.REFERENCE_METADATA_FILE)
    with open(path, encoding="utf-8") as handle:
        metadata: dict[str, object] = json.load(handle)
    return metadata


def reverse_average_rank_support(
    cohort: Sequence[tuple[float, int]], value: float, tolerance: float = 1e-9
) -> float:
    """Reverse average-rank ECDF, computed independently of production code."""
    total = sum(count for _, count in cohort)
    below = sum(count for item, count in cohort if item < value - tolerance)
    equal = sum(count for item, count in cohort if abs(item - value) <= tolerance)
    return 100.0 * (total - below - 0.5 * equal) / total


def frozen_reference(
    directory: _StrPath,
    rszd_values: Sequence[float] = _COHORT_RSZD,
    rms_zbond_values: Sequence[float] = _COHORT_RMS_ZBOND,
) -> str:
    """Publish a frozen confidence reference and return its directory.

    Built the way a real one is, through the documented
    ``src/confidence_score.py finalize`` entry point. Two rows carry no
    assessable component, so component sizes and total input size cannot be
    confused.
    """
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    rows: list[dict[str, str]] = []
    for index, (rszd, rms) in enumerate(
        itertools.product(rszd_values, rms_zbond_values)
    ):
        row = {column: "" for column in confidence_score.CONFIDENCE_INPUT_COLUMNS}
        row.update(
            pdbID=f"c{index:03d}",
            category="ion",
            selected_metal_site_status="selected",
            metal_element="ZN",
            rszd_abs=repr(rszd),
            geometry_rms_zbond=repr(rms),
            geometry_coverage="1",
            assigned_contact_count="4",
            reference_covered_contact_count="4",
            geometry_bond_count="4",
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
        fieldnames: list[str] = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def assert_policy_was_applied(
    rows: Sequence[dict[str, str]], reference_dir: _StrPath
) -> int:
    """Every row obeys the raw decision matrix and frozen component ranks."""
    metadata = reference_metadata(reference_dir)
    cohorts = frozen_cohort(reference_dir)
    density_size = sum(count for _, count in cohorts["density"])
    geometry_size = sum(count for _, count in cohorts["geometry"])
    assert density_size == metadata["density_reference_size"]
    assert geometry_size == metadata["geometry_reference_size"]

    scored = 0
    for row in rows:
        where = (
            f"{row['pdbID']} {row['metal_resname']}"
            f"{row['metal_resnum']}/{row['metal_atom']}"
        )
        assert row["confidence_reference_version"] == metadata["reference_id"], where
        assert int(row["confidence_cohort_size"]) == metadata["input_row_count"], where
        rszd = float(row["rszd_abs"]) if row["rszd_abs"] else math.nan
        rms = (
            float(row["geometry_rms_zbond"]) if row["geometry_rms_zbond"] else math.nan
        )
        density_level = raw_level(rszd, _DENSITY_THRESHOLDS)
        geometry_level = raw_level(rms, _GEOMETRY_THRESHOLDS)
        available = [
            level for level in (density_level, geometry_level) if level != "INCOMPLETE"
        ]
        overall = (
            "INCOMPLETE"
            if not available
            else "SUSPECT"
            if "SUSPECT" in available or available.count("REVIEW") == 2
            else "REVIEW"
            if "REVIEW" in available
            else "PASS"
        )
        assert row["density_level"] == density_level, where
        assert row["geometry_level"] == geometry_level, where
        assert row["alchemy_level"] == overall, where

        component_scores: list[float] = []
        for component, value in (("density", rszd), ("geometry", rms)):
            score_text = row[f"{component}_score"]
            if not math.isfinite(value):
                assert score_text == "", where
                continue
            expected = (
                0.0
                if component == "density" and value >= 99.9
                else reverse_average_rank_support(cohorts[component], value)
            )
            assert float(score_text) == approx(expected, abs=1e-6), where
            component_scores.append(expected)
        if component_scores:
            assert float(row["alchemy_score"]) == approx(
                min(component_scores), abs=1e-6
            ), where
            scored += 1
        else:
            assert row["alchemy_score"] == "", where
    return scored


def test_the_scoring_policy_under_test_is_the_shipped_one(tmp_path: Path) -> None:
    """The raw thresholds this module re-implements are Alchemy's own.

    ``assert_policy_was_applied`` is an independent oracle only while the two
    copies agree.
    """
    assert _DENSITY_THRESHOLDS == (
        confidence_score.DENSITY_REVIEW_THRESHOLD,
        confidence_score.DENSITY_SUSPECT_THRESHOLD,
    )
    assert _GEOMETRY_THRESHOLDS == (
        confidence_score.GEOMETRY_REVIEW_THRESHOLD,
        confidence_score.GEOMETRY_SUSPECT_THRESHOLD,
    )

    published = reference_metadata(frozen_reference(tmp_path / "policy"))
    assert published["density_thresholds"] == {"review": 3.0, "suspect": 6.0}
    assert published["geometry_thresholds"] == {"review": 1.0, "suspect": 2.0}
    assert published["geometry_statistic"] == "rms_finite_score_eligible_zbond"
    assert published["overall_rule"] == "any_suspect_or_review_plus_review"
    assert published["support_score_method"] == ("reverse_average_rank_empirical_cdf")


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_installed_reference_scores_every_selected_site_against_the_database(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str], batch: Batch
) -> None:
    """``--confidence-reference-dir`` makes a run emit its confidence scores.

    Such runs load the reference once, derive each new site's compact inputs
    while its normal result is still in memory, and never generate empirical
    rankings from their own small cohort.
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
    # Streaming compact inputs and publishing a reference belong to database runs.
    assert "confidence_inputs_all.csv" not in written
    assert "confidence_reference" not in written
    assert read_header(output_dir, "confidence_scores_all.csv") == (CONFIDENCE_COLUMNS)

    rows = read_rows(output_dir, "confidence_scores_all.csv")
    # One row per selected metal site: the complete_confidence_site_count
    # guarantee.
    for pdb_id, manifest_row in scored_run.manifest.items():
        assert len(rows_for(rows, pdb_id)) == int(manifest_row["n_metals"]), pdb_id
    assert len(rows) == 13
    assert rows_for(rows, "9nxl") == []

    metadata = reference_metadata(reference_dir)
    cohort_size = metadata["input_row_count"]
    assert isinstance(cohort_size, int)
    assessable_size = len(_COHORT_RSZD) * len(_COHORT_RMS_ZBOND)
    assert cohort_size == assessable_size + 2
    assert metadata["density_reference_size"] == assessable_size
    assert metadata["geometry_reference_size"] == assessable_size
    assert assert_policy_was_applied(rows, reference_dir) == len(rows)

    # The cohort is the frozen database, not this run's thirteen sites.
    rankings = [float(row["alchemy_score"]) for row in rows]
    assert all(0.0 <= value <= 100.0 for value in rankings)
    assert len(set(rankings)) > 1

    # 9myr chain B zinc: raw density and RMS geometry both pass.
    zinc = next(
        row for row in rows if row["pdbID"] == "9myr" and row["metal_chain"] == "B"
    )
    site = next(row for row in rows_for(batch.stats, "9myr") if row["CI"] == "B")
    assert float(zinc["rszd_abs"]) == approx(abs(float(site["ZDm"])), abs=1e-6)
    assert float(zinc["geometry_rms_zbond"]) < _GEOMETRY_THRESHOLDS[0]
    assert float(zinc["geometry_coverage"]) == 1.0
    assert zinc["density_level"] == "PASS"
    assert zinc["geometry_level"] == "PASS"
    assert zinc["alchemy_level"] == "PASS"

    assert (
        f"13 confidence rows compared with database cohort {cohort_size}"
        in scored_run.result.stdout
    )
    log_text = read_text(log_paths(output_dir)[0])
    assert "confidence_mode: reference" in log_text
    assert "confidence_status: scored_against_reference" in log_text


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_uncapped_database_run_finalizes_and_publishes_its_own_reference(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """A run over a whole mirror streams inputs and then freezes a reference.

    README: confidence inputs are collected "as part of an uncapped
    full-database run: no `--id`, `--id-file`, manual coordinate arguments, or
    `--max-pdbs`", and "only after the database run completes without
    operationally incomplete entries does Alchemy finalize confidence".

    The three-entry mirror below is a complete database as far as the driver is
    concerned, which reaches the database arm without a 24k-entry mirror.
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

    # Every selected metal site reaches both files exactly once.
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
    # These sites are each component cohort, so each component's average
    # reverse empirical rank is 50 even when tied values share a score.
    density_scores = [
        float(row["density_score"]) for row in rows if row["density_score"]
    ]
    geometry_scores = [
        float(row["geometry_score"]) for row in rows if row["geometry_score"]
    ]
    assert sum(density_scores) / len(density_scores) == approx(50.0)
    assert sum(geometry_scores) / len(geometry_scores) == approx(50.0)

    assert (
        f"13 confidence rows (13 scored; reference cohort 13) -> "
        f"{os.path.join(str(output_dir), 'confidence_scores_all.csv')}" in result.stdout
    )
    log_text = read_text(log_paths(output_dir)[0])
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
    assert [row["alchemy_score"] for row in reused_rows] == [
        row["alchemy_score"] for row in rows_for(rows, "9myr")
    ]


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
def test_resume_refuses_to_mix_two_database_snapshots(
    tmp_path: Path, entry_cache: str, ccp4_env: dict[str, str]
) -> None:
    """Resuming against a different frozen reference is refused, not blended.

    A ranking is meaningful only relative to one cohort, so half a file scored
    against another database would be misleading. The same-reference resume is
    asserted first, so the mismatch cannot be an unrelated resume failure.
    """
    installed = frozen_reference(tmp_path / "installed")
    other = frozen_reference(tmp_path / "other", _ALTERNATE_RSZD, _ALTERNATE_RMS_ZBOND)
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
    original = Path(scores_path).read_bytes()

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
    assert Path(scores_path).read_bytes() == original

    # Classification-only output cannot be silently upgraded to a referenced
    # ranking during resume, because untouched rows would remain unranked.
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
    assert os.path.exists(os.path.join(str(unscored), "confidence_scores_all.csv"))
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
    assert "Cannot resume confidence output" in upgraded.text
    assert "blank reference or cohort identifier" in upgraded.text


@_requires_entry_data
@pytest.mark.ccp4
@pytest.mark.slow
@pytest.mark.parametrize(
    "damage,message",
    [
        ("thresholds", "geometry_thresholds is incompatible with this code"),
        ("distribution", "identifier does not match data"),
    ],
)
def test_an_incompatible_reference_stops_the_run_instead_of_scoring(
    tmp_path: Path,
    entry_cache: str,
    ccp4_env: dict[str, str],
    damage: str,
    message: str,
) -> None:
    """A reference that is not the one it claims to be aborts the run.

    Either damage -- a policy that no longer matches this code, a distribution
    that no longer matches its published identifier -- yields scores comparable
    with nothing, so the run must stop before any entry is processed.
    """
    reference_dir = frozen_reference(tmp_path / "installed")
    metadata_path = os.path.join(
        reference_dir, confidence_score.REFERENCE_METADATA_FILE
    )
    if damage == "thresholds":
        metadata = reference_metadata(reference_dir)
        thresholds = metadata.get("geometry_thresholds")
        assert isinstance(thresholds, dict)
        thresholds["suspect"] = 6.0
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)
    else:
        distribution_path = os.path.join(
            reference_dir, confidence_score.REFERENCE_DISTRIBUTION_FILE
        )
        cohorts = frozen_cohort(reference_dir)
        with open(distribution_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("component", "value", "count"))
            for component, cohort in cohorts.items():
                for index, (value, count) in enumerate(cohort):
                    writer.writerow(
                        (
                            component,
                            repr(value),
                            count + (1 if component == "density" and index == 0 else 0),
                        )
                    )

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

    assert not os.path.exists(os.path.join(str(output_dir), "manifest.csv"))
    assert not os.path.exists(
        os.path.join(str(output_dir), "confidence_scores_all.csv")
    )
    assert len(log_paths(output_dir)) == 1
