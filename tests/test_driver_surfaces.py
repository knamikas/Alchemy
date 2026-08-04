"""Direct tests for the driver surfaces a batch run depends on.

Scope: the functions that decide *what a run does to existing data* and *which
entries it selects* -- resume-schema validation, the exit-code contract, ID
parsing, mirror enumeration, input preparation, and worker autoscaling.

Every one of these was previously reached only through ``main.main()``, which
runs solely in the ``ccp4``+``slow`` lane. That lane is deliberately not run in
CI (finding 1.7), so a regression in any of them surfaced either as a confusing
downstream failure or not at all.

Four of them -- ``verify_ccp4``, ``prepare_inputs``, ``read_resolution`` and
``has_final_files`` -- are relocated by the Phase E split. Pinning them first
means a regression there is attributed to the move rather than discovered later
in unrelated work. ``verify_ccp4`` has since moved to ``ccp4_setup``; the tests
followed it here rather than being split off, since what they check is the
driver's requirement that CCP4 be complete before a run starts.

Out of scope here (owned elsewhere): argument *parsing* rules
(``test_cli_and_config``), manifest row content (``test_driver_manifest``), and
the pipeline itself (``test_pipeline_integration``).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os

import pytest

import ccp4_setup
import inputs
import main
from driver import resources
from driver.writers import MANIFEST_COLUMNS, STATS_COLUMNS


# --------------------------------------------------------------------------- #
# Resume schema validation -- refusing to corrupt existing output
# --------------------------------------------------------------------------- #
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
        "bonds_path": _write_header(tmp_path / "bonds.csv", main.BOND_COLUMNS),
        "candidates_path": _write_header(
            tmp_path / "candidates.csv", main.CANDIDATE_COLUMNS
        ),
    }


def test_matching_headers_are_accepted(resume_outputs):
    main.validate_resume_schemas(**resume_outputs)


def test_absent_outputs_are_accepted(tmp_path):
    """A first run has nothing to be incompatible with."""
    main.validate_resume_schemas(
        manifest_path=str(tmp_path / "manifest.csv"),
        stats_path=str(tmp_path / "stats.csv"),
        bonds_path=str(tmp_path / "bonds.csv"),
        candidates_path=str(tmp_path / "candidates.csv"),
    )


@pytest.mark.parametrize(
    "target", ["manifest_path", "stats_path", "bonds_path", "candidates_path"]
)
def test_an_incompatible_header_is_refused(resume_outputs, tmp_path, target):
    """Appending beneath a foreign header would misalign every column.

    This is the check that stands between a schema migration and silently
    corrupted output: the rows would be written, and nothing downstream could
    tell that column N of the new rows means something else than column N of
    the old ones.
    """
    _write_header(tmp_path / os.path.basename(resume_outputs[target]), ["unexpected"])

    with pytest.raises(ValueError, match="incompatible schema"):
        main.validate_resume_schemas(**resume_outputs)


def test_a_truncated_stats_header_is_refused(resume_outputs, tmp_path):
    """A different EDSTATS build would shift the density block.

    The whole header is compared rather than its length or first few names,
    because a dropped metric column misaligns every value after it without any
    other symptom.
    """
    _write_header(tmp_path / "stats.csv", list(STATS_COLUMNS)[:-1])

    with pytest.raises(ValueError, match="incompatible schema"):
        main.validate_resume_schemas(**resume_outputs)


def test_bond_headers_are_ignored_when_the_bond_stage_is_disabled(
    resume_outputs, tmp_path
):
    """``--no-bonds`` writes no bond rows, so their schema cannot conflict."""
    _write_header(tmp_path / "bonds.csv", ["stale"])
    _write_header(tmp_path / "candidates.csv", ["stale"])

    main.validate_resume_schemas(**resume_outputs, bonds_enabled=False)


def test_confidence_output_requires_its_columns(resume_outputs, tmp_path):
    """A confidence path without its schema cannot be validated at all."""
    with pytest.raises(ValueError, match="confidence columns are required"):
        main.validate_resume_schemas(
            **resume_outputs,
            confidence_path=str(tmp_path / "confidence.csv"),
            confidence_columns=None,
        )


def test_an_incompatible_confidence_header_is_refused(resume_outputs, tmp_path):
    columns = list(main.CONFIDENCE_INPUT_COLUMNS)
    path = _write_header(tmp_path / "confidence.csv", columns[:-1])

    with pytest.raises(ValueError, match="incompatible schema"):
        main.validate_resume_schemas(
            **resume_outputs, confidence_path=path, confidence_columns=columns
        )


# --------------------------------------------------------------------------- #
# Exit-code contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("counts", "retryable_partials", "expected"),
    [
        ({"ok": 5, "partial": 0, "skip": 0, "error": 0}, 0, 0),
        # A terminal partial is usable output, not an incomplete run.
        ({"ok": 4, "partial": 1, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 0, "partial": 0, "skip": 0, "error": 0}, 0, 0),
        ({"ok": 4, "partial": 0, "skip": 0, "error": 1}, 0, 1),
        ({"ok": 4, "partial": 0, "skip": 1, "error": 0}, 0, 1),
        # A retryable partial means the entry can still be repaired.
        ({"ok": 4, "partial": 1, "skip": 0, "error": 0}, 1, 1),
    ],
)
def test_the_exit_code_reports_operational_incompleteness(
    counts, retryable_partials, expected
):
    """Nonzero exactly when something remains to be done.

    The distinction that matters is between a *terminal* partial -- usable but
    incomplete science, which no rerun can improve -- and a *retryable* one.
    Treating the first as failure would make every database run look broken;
    treating the second as success would hide work still outstanding.
    """
    assert main._batch_exit_code(counts, retryable_partials) == expected


def test_a_missing_status_key_is_treated_as_zero():
    """Counts are accumulated per status, so an absent key means none seen."""
    assert main._batch_exit_code({}, 0) == 0
    assert main._batch_exit_code({"error": 2}, 0) == 1


# --------------------------------------------------------------------------- #
# Entry selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["9myr", "9MYR", "1abc", "0000"])
def test_pdb_ids_are_accepted_case_insensitively_and_normalized(value):
    """IDs are lowercased so cache paths and manifest keys cannot diverge."""
    assert main.parse_pdb_id(value) == value.lower()


@pytest.mark.parametrize("value", ["abc", "abcde", "ab-c", "ab c", "", "9my_"])
def test_malformed_pdb_ids_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError, match="four alphanumeric"):
        main.parse_pdb_id(value)


def test_an_id_file_accepts_mixed_separators_and_comments(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text(
        "9myr, 6NLR\n# a comment line\n\n9nxl 1abc   # trailing comment\n",
        encoding="utf-8",
    )

    assert main.load_ids_from_file(str(path)) == ["9myr", "6nlr", "9nxl", "1abc"]


def test_an_id_file_reports_the_line_of_a_bad_id(tmp_path):
    """A typo in a long ID list must name where it is, not just that it exists."""
    path = tmp_path / "ids.txt"
    path.write_text("9myr\n6nlr\nnot-an-id\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid PDB id .*ids\.txt:3"):
        main.load_ids_from_file(str(path))


def test_a_missing_id_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="id file not found"):
        main.load_ids_from_file(str(tmp_path / "absent.txt"))


def _make_entry(
    root, pdb_id, *, mtz=True, cif=True, pdb=False, compressed=False, data_json=None
):
    """Create a mirror-layout entry directory with the requested files.

    ``cif`` and ``pdb`` are independent so a mirror carrying both formats can
    be built: production prefers the authoritative mmCIF and keeps the legacy
    PDB export only as a fallback.
    """
    entry_dir = inputs.entry_dir_for(str(root), pdb_id)
    os.makedirs(entry_dir, exist_ok=True)

    def write(name, payload=b"x"):
        # gzip.compress rather than gzip.open: it is unambiguously bytes-in,
        # bytes-out, so both branches share one plainly binary write.
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
    """Mirrors carrying only the legacy PDB export are still analyzable.

    Production accepts ``.pdb`` and ``.pdb.gz`` as a fallback when the
    authoritative mmCIF is absent. Testing only mmCIF would let a later
    extraction drop that fallback with every test still green, silently making
    a class of mirror unprocessable.
    """
    entry_dir = _make_entry(
        tmp_path, "9myr", cif=False, pdb=True, compressed=compressed
    )
    assert inputs.has_final_files(entry_dir, "9myr")


def test_enumeration_returns_only_complete_entries(tmp_path):
    """Incomplete entries are skipped, and the order follows the mirror layout.

    Ordering is by hash directory then entry, not by PDB ID: ``9myr`` lives
    under ``my`` and ``6nlr`` under ``nl``, so ``9myr`` comes first despite
    sorting later as a string. That is what makes ``--max-pdbs`` reproducible --
    it takes a prefix of a directory walk, not of a sorted ID list.
    """
    _make_entry(tmp_path, "9myr")  # hashdir "my"
    _make_entry(tmp_path, "6nlr")  # hashdir "nl"
    _make_entry(tmp_path, "9nxl", mtz=False)  # hashdir "nx", incomplete

    assert main.enumerate_entries(str(tmp_path)) == ["9myr", "6nlr"]


def test_enumeration_stops_at_the_requested_limit(tmp_path):
    """``--max-pdbs`` must not walk the whole mirror to return three ids."""
    for pdb_id in ("1aaa", "1bbb", "2ccc", "2ddd"):
        _make_entry(tmp_path, pdb_id)

    limited = main.enumerate_entries(str(tmp_path), limit=2)
    assert len(limited) == 2
    assert limited == main.enumerate_entries(str(tmp_path))[:2]


def test_an_unreadable_hashdir_is_skipped_rather_than_fatal(tmp_path, monkeypatch):
    """A partially-synced mirror must not abort the whole enumeration."""
    _make_entry(tmp_path, "9myr")
    _make_entry(tmp_path, "6nlr")
    real_listdir = os.listdir

    def failing_listdir(path):
        if str(path).endswith(os.path.join(str(tmp_path), "nl")):
            raise PermissionError("locked down")
        return real_listdir(path)

    monkeypatch.setattr(main.os, "listdir", failing_listdir)
    assert main.enumerate_entries(str(tmp_path)) == ["9myr"]


# --------------------------------------------------------------------------- #
# Input preparation
# --------------------------------------------------------------------------- #
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
    """When a mirror carries both, the mmCIF is the one converted.

    The legacy export loses identifiers the mmCIF retains, so preferring it
    would silently degrade every result for such an entry.
    """
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


# --------------------------------------------------------------------------- #
# Housekeeping and autoscaling
# --------------------------------------------------------------------------- #
def test_stale_bond_outputs_are_removed_only_by_a_fresh_disabled_run(tmp_path):
    """Old bond rows must not be mistaken for this run's output.

    Resume is the exception: it retains completed entries, so their existing
    rows are still current.
    """
    paths = [str(tmp_path / "bonds.csv"), str(tmp_path / "candidates.csv")]
    for path in paths:
        open(path, "w", encoding="utf-8").close()

    assert (
        main.remove_stale_disabled_bond_outputs(paths, resume=True, bonds_enabled=False)
        == []
    )
    assert (
        main.remove_stale_disabled_bond_outputs(paths, resume=False, bonds_enabled=True)
        == []
    )
    assert all(os.path.exists(path) for path in paths)

    assert (
        main.remove_stale_disabled_bond_outputs(
            paths, resume=False, bonds_enabled=False
        )
        == paths
    )
    assert not any(os.path.exists(path) for path in paths)


def test_removing_absent_bond_outputs_is_not_an_error(tmp_path):
    paths = [str(tmp_path / "absent.csv")]
    assert main.remove_stale_disabled_bond_outputs(paths, False, False) == []


def test_worker_limits_leave_headroom_and_respect_memory(monkeypatch):
    """Both limits are floored at one, so a small machine still runs.

    The CPU limit deliberately leaves two cores for the driver and the OS; the
    memory limit exists because each worker holds whole maps in memory, and
    oversubscribing it is what invites the OOM killer the driver has to
    recover from.
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


# --------------------------------------------------------------------------- #
# CCP4 tool verification
# --------------------------------------------------------------------------- #
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
    """``ccp4_tools_available`` and ``verify_ccp4`` must not disagree.

    They were separate implementations of the same question (finding 2.2), and
    a divergence would make the driver accept an installation the setup helper
    rejects, or the reverse. They now share ``missing_ccp4_tools``; this pins
    the agreement so a future edit cannot split them again.
    """
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

    ``SystemExit`` derives from ``BaseException``, so a caller reusing CCP4
    resolution outside a CLI process -- a notebook, a service -- would have its
    interpreter torn down by a bare ``except Exception``. The CLI's own
    conversion to an exit is covered by ``test_cli_and_config``.
    """
    monkeypatch.setattr(ccp4_setup.shutil, "which", lambda tool, path=None: None)

    with pytest.raises(Exception) as excinfo:
        ccp4_setup.verify_ccp4({"PATH": "/x"})
    assert not isinstance(excinfo.value, SystemExit)


# --------------------------------------------------------------------------- #
# Resolution metadata
# --------------------------------------------------------------------------- #
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
    """A half-populated record must not be trusted for part of the answer.

    Both limits are required before data.json is believed: reporting a
    high-resolution limit from a record missing its low-resolution counterpart
    would silently mix a partial metadata source into DPI provenance.
    """
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

    The override must win over any file that happens to sit beside the
    coordinates, or a manual run inside a populated mirror would silently read
    the wrong entry's metadata.
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


# --------------------------------------------------------------------------- #
# Intermediate retention
# --------------------------------------------------------------------------- #
def test_intermediates_are_discarded_unless_asked_for():
    """Per-entry maps are large, so retention is opt-in.

    ``process`` keys its scratch cleanup off this flag; defaulting it to true
    would fill an output directory over a database run.
    """
    assert main.parse_args([]).keep_intermediates is False
    assert main.parse_args(["--keep-intermediates"]).keep_intermediates is True
