"""Direct tests for the driver surfaces a batch run depends on.

Scope: the functions that decide *what a run does to existing data* and *which
entries it selects* -- resume-schema validation, the exit-code contract, ID
parsing, mirror enumeration, input preparation, worker autoscaling, and the
driver's requirement that CCP4 be complete before a run starts. Reached through
``main.main()`` they would run only in the ``ccp4``+``slow`` lane, which CI does
not run.

Out of scope here (owned elsewhere): argument *parsing* rules
(``test_cli_and_config``), manifest row content (``test_driver_manifest``), and
the pipeline itself (``test_pipeline_integration``).
"""

from __future__ import annotations

import argparse
import csv
import http.client
import gzip
import json
import os

import pytest

from coordination import schema as coordination_schema
import ccp4_setup
import cli
import inputs
from driver import resources
from driver import writers
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS
from driver import resume
from driver import pool
import confidence_score


def _write_header(path, columns):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(columns)
    return str(path)


@pytest.fixture
def resume_outputs(tmp_path):
    """A set of output files whose headers all match the current schema."""
    return {
        "manifest_path": _write_header(tmp_path / "manifest.csv", MANIFEST_COLUMNS),
        "stats_path": _write_header(tmp_path / "stats.csv", STATS_COLUMNS),
        "bonds_path": _write_header(
            tmp_path / "bonds.csv", coordination_schema.BOND_COLUMNS
        ),
        "candidates_path": _write_header(
            tmp_path / "candidates.csv", coordination_schema.CANDIDATE_COLUMNS
        ),
    }


def test_matching_headers_are_accepted(resume_outputs):
    resume.validate_resume_schemas(**resume_outputs)


def test_absent_outputs_are_accepted(tmp_path):
    """A first run has nothing to be incompatible with."""
    resume.validate_resume_schemas(
        manifest_path=str(tmp_path / "manifest.csv"),
        stats_path=str(tmp_path / "stats.csv"),
        bonds_path=str(tmp_path / "bonds.csv"),
        candidates_path=str(tmp_path / "candidates.csv"),
    )


@pytest.mark.parametrize(
    "target", ["manifest_path", "stats_path", "bonds_path", "candidates_path"]
)
def test_an_incompatible_header_is_refused(resume_outputs, tmp_path, target):
    """Appending beneath a foreign header would misalign every column, and
    nothing downstream could tell that column N had changed meaning."""
    _write_header(tmp_path / os.path.basename(resume_outputs[target]), ["unexpected"])

    with pytest.raises(ValueError, match="incompatible schema"):
        resume.validate_resume_schemas(**resume_outputs)


def test_a_manifest_without_the_reference_data_column_is_refused(
    resume_outputs, tmp_path
):
    """A manifest without the identity column cannot be resumed into.

    Its rows do not say which reference data produced them, so appending rows
    that do would make one file two datasets with no way to tell them apart.
    """
    older = [
        column for column in writers.MANIFEST_COLUMNS if column != "reference_data_id"
    ]
    _write_header(tmp_path / "manifest.csv", older)

    with pytest.raises(ValueError, match="missing reference_data_id"):
        resume.validate_resume_schemas(**resume_outputs)


def test_the_refusal_names_the_columns_that_differ(resume_outputs, tmp_path):
    """The message names the columns that differ, leaving the operator with a
    cause rather than two headers to diff by hand."""
    _write_header(tmp_path / "manifest.csv", list(writers.MANIFEST_COLUMNS) + ["stray"])

    with pytest.raises(ValueError, match="unexpected stray"):
        resume.validate_resume_schemas(**resume_outputs)


def test_a_truncated_stats_header_is_refused(resume_outputs, tmp_path):
    """A different EDSTATS build shifts the density block, and a dropped metric
    column misaligns every value after it with no other symptom."""
    _write_header(tmp_path / "stats.csv", list(STATS_COLUMNS)[:-1])

    with pytest.raises(ValueError, match="incompatible schema"):
        resume.validate_resume_schemas(**resume_outputs)


def test_bond_headers_are_ignored_when_the_bond_stage_is_disabled(
    resume_outputs, tmp_path
):
    """``--no-bonds`` writes no bond rows, so their schema cannot conflict."""
    _write_header(tmp_path / "bonds.csv", ["stale"])
    _write_header(tmp_path / "candidates.csv", ["stale"])

    resume.validate_resume_schemas(**resume_outputs, bonds_enabled=False)


def test_confidence_output_requires_its_columns(resume_outputs, tmp_path):
    """A confidence path without its schema cannot be validated at all."""
    with pytest.raises(ValueError, match="confidence columns are required"):
        resume.validate_resume_schemas(
            **resume_outputs,
            confidence_path=str(tmp_path / "confidence.csv"),
            confidence_columns=None,
        )


def test_an_incompatible_confidence_header_is_refused(resume_outputs, tmp_path):
    columns = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
    path = _write_header(tmp_path / "confidence.csv", columns[:-1])

    with pytest.raises(ValueError, match="incompatible schema"):
        resume.validate_resume_schemas(
            **resume_outputs, confidence_path=path, confidence_columns=columns
        )


@pytest.mark.parametrize(
    ("counts", "retryable_partials", "expected"),
    [
        ({"ok": 5, "partial": 0, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 4, "partial": 1, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 0, "partial": 0, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 4, "partial": 0, "skip": 0, "error": 1}, 0, 1),
        ({"ok": 4, "partial": 0, "skip": 1, "error": 0}, 0, 1),
        ({"ok": 4, "partial": 1, "skip": 0, "error": 0}, 1, 1),
    ],
)
def test_the_exit_code_reports_operational_incompleteness(
    counts, retryable_partials, expected
):
    """Nonzero exactly when something remains to be done.

    A *terminal* partial is usable but incomplete science that no rerun can
    improve, so it exits zero; a *retryable* one is work still outstanding.
    """
    assert pool._batch_exit_code(counts, retryable_partials) == expected


def test_a_missing_status_key_is_treated_as_zero():
    """Counts are accumulated per status, so an absent key means none seen."""
    assert pool._batch_exit_code({}, 0) == 0
    assert pool._batch_exit_code({"error": 2}, 0) == 1


@pytest.mark.parametrize("value", ["9myr", "9MYR", "1abc", "0000"])
def test_pdb_ids_are_accepted_case_insensitively_and_normalized(value):
    """IDs are lowercased so cache paths and manifest keys cannot diverge."""
    assert cli.parse_pdb_id(value) == value.lower()


@pytest.mark.parametrize("value", ["abc", "abcde", "ab-c", "ab c", "", "9my_"])
def test_malformed_pdb_ids_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError, match="four alphanumeric"):
        cli.parse_pdb_id(value)


def test_an_id_file_accepts_mixed_separators_and_comments(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text(
        "9myr, 6NLR\n# a comment line\n\n9nxl 1abc   # trailing comment\n",
        encoding="utf-8",
    )

    assert pool.load_ids_from_file(str(path)) == ["9myr", "6nlr", "9nxl", "1abc"]


def test_an_id_file_reports_the_line_of_a_bad_id(tmp_path):
    """A typo in a long ID list must name where it is, not just that it exists."""
    path = tmp_path / "ids.txt"
    path.write_text("9myr\n6nlr\nnot-an-id\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid PDB id .*ids\.txt:3"):
        pool.load_ids_from_file(str(path))


def test_a_missing_id_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="id file not found"):
        pool.load_ids_from_file(str(tmp_path / "absent.txt"))


def _make_entry(
    root, pdb_id, *, mtz=True, cif=True, pdb=False, compressed=False, data_json=None
):
    """Create a mirror-layout entry directory with the requested files.

    ``cif`` and ``pdb`` are independent so a mirror carrying both can be built.
    """
    entry_dir = inputs.entry_dir_for(str(root), pdb_id)
    os.makedirs(entry_dir, exist_ok=True)

    def write(name, payload=b"x"):
        path = os.path.join(entry_dir, name)
        if compressed:
            path += ".gz"
            payload = gzip.compress(payload)
        with open(path, "wb") as handle:
            handle.write(payload)

    if mtz:
        write(f"{pdb_id}_final.mtz")
    if cif:
        write(f"{pdb_id}_final.cif")
    if pdb:
        write(f"{pdb_id}_final.pdb", b"END\n")
    if data_json is not None:
        with open(os.path.join(entry_dir, "data.json"), "w", encoding="utf-8") as fh:
            fh.write(data_json)
    return entry_dir


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_an_entry_is_final_only_with_both_inputs(tmp_path, compressed):
    """Coordinates without map coefficients cannot be analyzed, and vice versa."""
    complete = _make_entry(tmp_path, "9myr", compressed=compressed)
    no_mtz = _make_entry(tmp_path, "6nlr", mtz=False, compressed=compressed)
    no_coords = _make_entry(
        tmp_path, "9nxl", cif=False, pdb=False, compressed=compressed
    )

    assert inputs.has_final_files(complete, "9myr")
    assert not inputs.has_final_files(no_mtz, "6nlr")
    assert not inputs.has_final_files(no_coords, "9nxl")


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_a_legacy_pdb_export_counts_as_usable_coordinates(tmp_path, compressed):
    """``.pdb`` and ``.pdb.gz`` are accepted when the authoritative mmCIF is
    absent, so a mirror carrying only the legacy export is still analyzable."""
    entry_dir = _make_entry(
        tmp_path, "9myr", cif=False, pdb=True, compressed=compressed
    )
    assert inputs.has_final_files(entry_dir, "9myr")


def test_enumeration_returns_only_complete_entries(tmp_path):
    """Incomplete entries are skipped, and the order follows the mirror layout.

    Ordering is by hash directory then entry, not by PDB ID: ``9myr`` lives
    under ``my`` and ``6nlr`` under ``nl``, so ``9myr`` comes first despite
    sorting later as a string.
    """
    _make_entry(tmp_path, "9myr")  # hashdir "my"
    _make_entry(tmp_path, "6nlr")  # hashdir "nl"
    _make_entry(tmp_path, "9nxl", mtz=False)  # hashdir "nx", incomplete

    assert inputs.enumerate_entries(str(tmp_path)) == ["9myr", "6nlr"]


def test_enumeration_stops_at_the_requested_limit(tmp_path):
    """``--max-pdbs`` must not walk the whole mirror to return three ids."""
    for pdb_id in ("1aaa", "1bbb", "2ccc", "2ddd"):
        _make_entry(tmp_path, pdb_id)

    limited = inputs.enumerate_entries(str(tmp_path), limit=2)
    assert len(limited) == 2
    assert limited == inputs.enumerate_entries(str(tmp_path))[:2]


def test_an_unreadable_hashdir_is_skipped_rather_than_fatal(tmp_path, monkeypatch):
    """A partially-synced mirror must not abort the whole enumeration."""
    _make_entry(tmp_path, "9myr")
    _make_entry(tmp_path, "6nlr")
    real_listdir = os.listdir

    def failing_listdir(path):
        if str(path).endswith(os.path.join(str(tmp_path), "nl")):
            raise PermissionError("locked down")
        return real_listdir(path)

    monkeypatch.setattr(inputs.os, "listdir", failing_listdir)
    assert inputs.enumerate_entries(str(tmp_path)) == ["9myr"]


def test_missing_map_coefficients_are_reported_by_path(tmp_path):
    """The error must name the file that was looked for, not just fail."""
    entry_dir = _make_entry(tmp_path, "9myr", mtz=False)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="9myr_final.mtz"):
        inputs.prepare_inputs("9myr", entry_dir, str(work_dir))


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzipped"])
def test_a_legacy_pdb_export_is_used_when_no_mmcif_exists(tmp_path, compressed):
    """The fallback path returns the PDB directly, decompressing if needed."""
    entry_dir = _make_entry(
        tmp_path, "9myr", cif=False, pdb=True, compressed=compressed
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    mtz, pdb = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))

    assert pdb.endswith(".pdb"), "a gzipped export must be decompressed first"
    assert os.path.isfile(pdb)
    with open(pdb, encoding="ascii") as handle:
        assert handle.read() == "END\n"
    if compressed:
        assert os.path.dirname(pdb) == str(work_dir), (
            "the mirror must not be written to"
        )


def test_the_authoritative_mmcif_wins_over_the_legacy_export(tmp_path, monkeypatch):
    """When a mirror carries both, the mmCIF is the one converted: the legacy
    export loses identifiers it retains."""
    entry_dir = _make_entry(tmp_path, "9myr", cif=True, pdb=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    converted = []

    def fake_cif_to_pdb(cif_path, destination):
        converted.append(cif_path)
        with open(destination, "w", encoding="ascii") as handle:
            handle.write("FROM CIF\n")
        return destination

    monkeypatch.setattr(inputs, "_cif_to_pdb", fake_cif_to_pdb)
    _mtz, pdb = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))

    assert converted and converted[0].endswith("_final.cif")
    with open(pdb, encoding="ascii") as handle:
        assert handle.read() == "FROM CIF\n", "the legacy export was used instead"


def test_an_entry_with_neither_coordinate_format_names_both(tmp_path):
    entry_dir = _make_entry(tmp_path, "9myr", cif=False, pdb=False)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"9myr_final\.cif or 9myr_final\.pdb"):
        inputs.prepare_inputs("9myr", entry_dir, str(work_dir))


def test_a_compressed_mirror_is_decompressed_into_the_work_directory(
    tmp_path, monkeypatch
):
    """Compressed mirrors are accepted, and never modified in place."""
    entry_dir = _make_entry(tmp_path, "9myr", compressed=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    converted = []

    def fake_cif_to_pdb(cif_path, destination):
        converted.append((cif_path, destination))
        with open(destination, "w", encoding="ascii") as handle:
            handle.write("END\n")
        return destination

    monkeypatch.setattr(inputs, "_cif_to_pdb", fake_cif_to_pdb)
    mtz, pdb = inputs.prepare_inputs("9myr", entry_dir, str(work_dir))

    assert os.path.dirname(mtz) == str(work_dir), "the mirror must not be written to"
    assert not mtz.endswith(".gz")
    assert os.path.isfile(pdb)
    assert converted, "the authoritative mmCIF should have been converted"


def test_stale_bond_outputs_are_removed_only_by_a_fresh_disabled_run(tmp_path):
    """Old bond rows must not be mistaken for this run's output.

    Resume is the exception: it retains completed entries, so their existing
    rows are still current.
    """
    paths = [str(tmp_path / "bonds.csv"), str(tmp_path / "candidates.csv")]
    for path in paths:
        open(path, "w", encoding="utf-8").close()

    assert (
        resume.remove_stale_disabled_bond_outputs(
            paths, resume=True, bonds_enabled=False
        )
        == []
    )
    assert (
        resume.remove_stale_disabled_bond_outputs(
            paths, resume=False, bonds_enabled=True
        )
        == []
    )
    assert all(os.path.exists(path) for path in paths)

    assert (
        resume.remove_stale_disabled_bond_outputs(
            paths, resume=False, bonds_enabled=False
        )
        == paths
    )
    assert not any(os.path.exists(path) for path in paths)


def test_removing_absent_bond_outputs_is_not_an_error(tmp_path):
    paths = [str(tmp_path / "absent.csv")]
    assert resume.remove_stale_disabled_bond_outputs(paths, False, False) == []


def test_worker_limits_leave_headroom_and_respect_memory(monkeypatch):
    """Both limits are floored at one, so a small machine still runs.

    The CPU limit leaves two cores for the driver and the OS; the memory limit
    exists because each worker holds whole maps in memory, and oversubscribing
    it invites the OOM killer.
    """
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 16)
    monkeypatch.setattr(
        resources,
        "available_memory_bytes",
        lambda: 8 * resources.AUTO_WORKER_MEMORY_BYTES,
    )
    assert resources.automatic_worker_limits() == (14, 8)

    monkeypatch.setattr(resources, "available_cpu_count", lambda: 1)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 0)
    assert resources.automatic_worker_limits() == (1, 1)


def test_unknown_memory_leaves_the_limit_unset(monkeypatch):
    """An unreadable ``/proc/meminfo`` must not silently cap parallelism."""
    monkeypatch.setattr(resources, "available_cpu_count", lambda: 8)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: None)

    cpu_limit, memory_limit = resources.automatic_worker_limits()
    assert cpu_limit == 6
    assert memory_limit is None


class _FailingResponse:
    """A response that opens successfully and then fails partway through."""

    def __init__(self, error):
        self._error = error
        self._served = False

    def getcode(self):
        return 200

    def read(self, _size):
        if self._served:
            raise self._error
        self._served = True
        return b"partial body"

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("connection reset by peer"),
        TimeoutError("read timed out"),
        http.client.IncompleteRead(b"partial body", 4096),
    ],
    ids=["reset", "timeout", "incomplete-read"],
)
def test_a_transfer_that_fails_midway_reports_no_usable_file(
    tmp_path, monkeypatch, error
):
    """Only the opening request was guarded, so a mid-stream failure escaped.

    ``IncompleteRead`` is not even an ``OSError``, so a handler written for one
    would still have missed it. The caller's position is the same as a 404 --
    no file -- and the partial download must not be left behind.
    """
    monkeypatch.setattr(
        inputs, "urlopen", lambda url, timeout=None: _FailingResponse(error)
    )
    destination = tmp_path / "9myr_final.mtz"

    with pytest.raises(FileNotFoundError) as excinfo:
        inputs._download_stream(
            "https://example.invalid/9myr_final.mtz", str(destination)
        )

    assert type(error).__name__ in str(excinfo.value)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == [], "a partial download was left behind"


def test_an_unwritable_cache_is_reported_as_a_driver_error(tmp_path, monkeypatch):
    """A local failure is not a missing entry, and is not a traceback either.

    ``ensure_entry_available`` creates directories and writes files, so an
    unwritable ``--pdb-redo-cache`` raised ``PermissionError`` straight out of
    the driver's pre-flight, past a handler that caught only
    ``FileNotFoundError``.
    """

    def unwritable(pdb_id, mirror_root, cache_root):
        raise PermissionError(13, "Permission denied", str(cache_root))

    monkeypatch.setattr(pool, "ensure_entry_available", unwritable)
    args = argparse.Namespace(
        id="9myr",
        id_file=None,
        pdb_file=None,
        mtz_file=None,
        cif_file=None,
        data_json=None,
        pdb_redo_root=str(tmp_path / "mirror"),
    )

    with pytest.raises(pool.DriverError) as excinfo:
        pool._select_entry_ids(args, str(tmp_path / "cache"))

    message = str(excinfo.value)
    assert "PermissionError" in message
    assert "not found" not in message, (
        "a local failure must not read as a missing entry"
    )


def _host_meminfo(tmp_path, monkeypatch, host_bytes):
    """Point the memory probe at a synthetic ``/proc/meminfo``."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {host_bytes // 1024} kB\n", encoding="ascii")
    monkeypatch.setattr(resources, "PROC_MEMINFO_PATH", str(meminfo))
    monkeypatch.setattr(resources.sys, "platform", "linux")
    return meminfo


def test_available_memory_is_read_from_meminfo(tmp_path, monkeypatch):
    """Worker sizing follows what the machine reports as free."""
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=7 * budget)

    assert resources.available_memory_bytes() == 7 * budget
    assert resources.automatic_worker_limits()[1] == 7


def test_an_unreadable_meminfo_leaves_worker_sizing_to_the_cpu_limit(
    tmp_path, monkeypatch
):
    """A memory probe that fails must not be read as zero available memory."""
    monkeypatch.setattr(
        resources, "PROC_MEMINFO_PATH", str(tmp_path / "definitely-absent")
    )
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources.os, "sysconf", lambda name: 0)

    assert resources.available_memory_bytes() is None
    assert resources.automatic_worker_limits()[1] is None


def test_a_cgroup_memory_limit_is_not_detected(tmp_path, monkeypatch):
    """Alchemy sizes from host memory, so a scheduler job needs --workers.

    Detecting a cgroup allowance was removed rather than repaired: it was
    roughly half of this module, was reached by nobody running Alchemy from a
    clone, and could not affect a single-structure run at all because workers
    are capped at the entry count. ``docs/usage.md`` states the consequence, so
    this test holds the code to what the documentation promises.
    """
    budget = resources.AUTO_WORKER_MEMORY_BYTES
    _host_meminfo(tmp_path, monkeypatch, host_bytes=64 * budget)
    # A limit far below host memory, of the kind SLURM or a container imposes.
    cgroup_root = tmp_path / "cgroup"
    (cgroup_root / "batch").mkdir(parents=True)
    (cgroup_root / "batch" / "memory.max").write_text(str(2 * budget), encoding="ascii")

    assert resources.available_memory_bytes() == 64 * budget
    assert not hasattr(resources, "CGROUP_ROOT")


def test_missing_ccp4_tools_are_named_with_a_remedy(monkeypatch):
    """The message must say which tools are absent and how to fix it."""
    monkeypatch.setattr(
        ccp4_setup.shutil,
        "which",
        lambda tool, path=None: None if tool == "edstats" else "/x",
    )

    with pytest.raises(ccp4_setup.Ccp4SetupError) as excinfo:
        ccp4_setup.verify_ccp4({"PATH": "/x"})

    message = str(excinfo.value)
    assert "edstats" in message
    assert "--configure-ccp4" in message, "the error should name the remedy"


def test_a_complete_ccp4_installation_passes(monkeypatch):
    monkeypatch.setattr(
        ccp4_setup.shutil, "which", lambda tool, path=None: f"/opt/{tool}"
    )
    ccp4_setup.verify_ccp4({"PATH": "/opt"})


def test_tool_availability_agrees_with_verification(monkeypatch):
    """``ccp4_tools_available`` and ``verify_ccp4`` must not disagree, or the
    driver accepts an installation the setup helper rejects."""
    monkeypatch.setattr(
        ccp4_setup.shutil,
        "which",
        lambda tool, path=None: None if tool == "fft" else "/x",
    )

    assert not ccp4_setup.ccp4_tools_available({"PATH": "/x"})
    with pytest.raises(ccp4_setup.Ccp4SetupError):
        ccp4_setup.verify_ccp4({"PATH": "/x"})


def test_a_library_caller_never_has_to_catch_systemexit(monkeypatch):
    """``ccp4_setup`` raises an ordinary exception, not ``SystemExit``.

    ``SystemExit`` derives from ``BaseException``, so a caller outside a CLI
    process -- a notebook, a service -- cannot contain it with ``except
    Exception``. The CLI's conversion to an exit is ``test_cli_and_config``.
    """
    monkeypatch.setattr(ccp4_setup.shutil, "which", lambda tool, path=None: None)

    with pytest.raises(Exception) as excinfo:
        ccp4_setup.verify_ccp4({"PATH": "/x"})
    assert not isinstance(excinfo.value, SystemExit)


def _minimal_mtz(path, high=1.5):
    """An MTZ whose only purpose is to carry a known resolution limit."""
    import gemmi
    import numpy as np

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.cell = gemmi.UnitCell(high * 2, high * 2, high * 2, 90, 90, 90)
    dataset = mtz.add_dataset("test")
    mtz.add_column("F", "F", dataset.id)
    mtz.set_data(np.asarray([[1, 0, 0, 10.0], [0, 1, 0, 10.0]], dtype=np.float32))
    mtz.write_to_file(str(path))
    return str(path)


def _map_coefficient_mtz(path, rows, *, columns=inputs.MAP_COEFFICIENT_COLUMNS):
    """An MTZ carrying the four EDSTATS map-coefficient columns.

    ``rows`` are ``(h, k, l, *values)`` with one value per column, so a caller
    can make an individual coefficient non-finite without touching the others.
    """
    import gemmi
    import numpy as np

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    mtz.cell = gemmi.UnitCell(40, 40, 40, 90, 90, 90)
    dataset = mtz.add_dataset("test")
    for label in columns:
        mtz.add_column(
            label,
            "F" if label.startswith(("FWT", "DELFWT")) else "P",
            dataset.id,
        )
    mtz.set_data(np.asarray(rows, dtype=np.float32))
    mtz.write_to_file(str(path))
    return str(path)


def _d_spacing(mtz_path):
    import gemmi

    return gemmi.read_mtz_file(mtz_path).make_d_array()


def test_map_column_resolution_spans_only_wholly_finite_reflections(tmp_path):
    """EDSTATS is given the range where all four coefficients exist.

    ``docs/method.md`` states this deliberately: limits taken from the overall
    MTZ would describe reflections the maps were not calculated from. A bug in
    the whole-row mask would silently move the limits behind every RSZD in the
    database, so the mask is checked against a row that is finite in three
    columns and not the fourth.
    """
    nan = float("nan")
    path = _map_coefficient_mtz(
        tmp_path / "coefficients.mtz",
        [
            [1, 0, 0, 10.0, 20.0, 1.0, 30.0],
            [3, 0, 0, 10.0, 20.0, 1.0, 30.0],
            # Finite everywhere but DELFWT, and the highest-resolution row, so
            # a mask that did not span whole rows would extend reshi onto it.
            [4, 0, 0, 10.0, 20.0, nan, 30.0],
        ],
    )
    spacings = _d_spacing(path)

    reslo, reshi = inputs.read_map_column_resolution(path)

    assert (reslo, reshi) == pytest.approx(
        (float(max(spacings[0], spacings[1])), float(min(spacings[0], spacings[1])))
    )
    assert reshi > float(spacings[2])


def test_map_column_resolution_names_every_missing_column(tmp_path):
    """The error says which coefficients are absent, not merely that one is."""
    path = _map_coefficient_mtz(
        tmp_path / "partial.mtz",
        [[1, 0, 0, 10.0, 20.0]],
        columns=("FWT", "PHWT"),
    )

    with pytest.raises(ValueError) as excinfo:
        inputs.read_map_column_resolution(path)

    message = str(excinfo.value)
    assert "DELFWT" in message and "PHDELWT" in message
    assert "FWT" in message


def test_map_column_resolution_rejects_a_wholly_unusable_set(tmp_path):
    """No reflection finite in all four columns is an error, not an empty range."""
    nan = float("nan")
    path = _map_coefficient_mtz(
        tmp_path / "unusable.mtz",
        [
            [1, 0, 0, 10.0, 20.0, nan, 30.0],
            [2, 0, 0, nan, 20.0, 1.0, 30.0],
        ],
    )

    with pytest.raises(ValueError, match="no common finite reflections"):
        inputs.read_map_column_resolution(path)


def test_resolution_is_read_from_data_json_when_complete(tmp_path):
    """data.json is authoritative because DPI metadata is derived from it."""
    entry_dir = _make_entry(
        tmp_path,
        "9myr",
        data_json=json.dumps({"properties": {"DATARESL": 47.1, "DATARESH": 1.72}}),
    )
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    assert inputs.read_resolution(entry_dir, mtz) == pytest.approx(1.72)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"properties": {"DATARESL": 47.1}}', "no DATARESH"),
        ('{"properties": {"DATARESH": 1.72}}', "no DATARESL"),
        ('{"properties": {}}', "neither limit"),
        ("{not valid json", "malformed"),
        ("", "empty"),
    ],
)
def test_incomplete_metadata_falls_back_to_the_mtz(tmp_path, payload, reason):
    """Both limits are required before data.json is believed, or a partial
    metadata source is mixed into DPI provenance."""
    entry_dir = _make_entry(tmp_path, "9myr", data_json=payload)
    mtz = _minimal_mtz(tmp_path / "fallback.mtz")

    resolution = inputs.read_resolution(entry_dir, mtz)
    assert resolution == pytest.approx(
        inputs.read_resolution(tmp_path / "absent", mtz)
    ), f"{reason} should have fallen back to the MTZ"


def test_absent_metadata_falls_back_to_the_mtz(tmp_path):
    """A mirror without data.json is still analyzable, DPI aside."""
    entry_dir = _make_entry(tmp_path, "9myr")
    mtz = _minimal_mtz(tmp_path / "only.mtz")

    assert inputs.read_resolution(entry_dir, mtz) > 0.0


def test_an_explicit_data_json_path_overrides_the_entry_directory(tmp_path):
    """``--data-json`` supplies metadata for manual inputs with no entry dir.

    It must win over any file sitting beside the coordinates, or a manual run
    inside a populated mirror reads the wrong entry's metadata.
    """
    entry_dir = _make_entry(
        tmp_path,
        "9myr",
        data_json=json.dumps({"properties": {"DATARESL": 40.0, "DATARESH": 9.99}}),
    )
    explicit = tmp_path / "explicit.json"
    explicit.write_text(
        json.dumps({"properties": {"DATARESL": 47.1, "DATARESH": 1.72}}),
        encoding="utf-8",
    )
    mtz = _minimal_mtz(tmp_path / "unused.mtz")

    assert inputs.read_resolution(
        entry_dir, mtz, data_json_path=str(explicit)
    ) == pytest.approx(1.72)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "file not found"),
        ("{not valid json", "invalid JSON"),
        ("[]", "must contain a JSON object"),
        ("{}", "must contain a properties object"),
    ],
)
def test_explicit_invalid_data_json_does_not_fall_back_to_mtz(
    tmp_path, payload, message
):
    """A requested metadata file is an input contract, not an optional probe."""
    data_json = tmp_path / "explicit.json"
    if payload is not None:
        data_json.write_text(payload, encoding="utf-8")
    mtz = _minimal_mtz(tmp_path / "fallback.mtz")

    with pytest.raises(ValueError, match=message):
        inputs.read_resolution(
            tmp_path,
            mtz,
            data_json_path=str(data_json),
        )


def test_manual_run_rejects_invalid_explicit_data_json_before_scheduling(tmp_path):
    """A CLI typo fails as a driver input error instead of reaching a worker."""
    missing = tmp_path / "missing.json"
    args = cli.parse_args(
        [
            "--id",
            "9myr",
            "--pdb-file",
            "9myr.pdb",
            "--mtz-file",
            "9myr.mtz",
            "--data-json",
            str(missing),
        ]
    )

    with pytest.raises(pool.DriverError, match=r"Invalid --data-json:.*not found"):
        pool._select_entry_ids(args, str(tmp_path / "cache"))


def test_intermediates_are_discarded_unless_asked_for():
    """Per-entry maps are large, so retention is opt-in: ``process`` keys its
    scratch cleanup off this flag."""
    assert cli.parse_args([]).keep_intermediates is False
    assert cli.parse_args(["--keep-intermediates"]).keep_intermediates is True
