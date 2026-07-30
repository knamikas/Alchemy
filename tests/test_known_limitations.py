"""Validated-but-unfixed defects, pinned so a fix cannot land unnoticed.

Every executable defect pin in this module describes behaviour Alchemy *should*
have and is marked ``xfail(strict=True)``.  Unsafe process-level reproductions
are documented as explicit skips instead, so the reported outcome is:

``XFAIL``
    The defect is still present. This is the expected, quiet state.
``XPASS`` -> **the suite fails**
    Someone fixed the defect. That is good news, and it is deliberately loud:
    delete the ``xfail`` marker, move the test into the module that owns the
    behaviour, and update the finding list below.

Three properties keep this module honest:

* it asserts the *desired* behaviour, never the buggy behaviour, so a fix turns
  a red mark green rather than requiring the assertion to be inverted;
* nothing here needs CCP4 or the network -- CCP4 is stubbed with four
  do-nothing executables where the driver only probes ``PATH`` -- so the
  findings stay visible on a bare laptop checkout; every strict assertion is
  narrow enough that only a fix to *that* defect can satisfy it, because under
  ``strict=True`` an incidental pass is a red suite claiming a fix nobody made;
* findings that can only be reproduced by killing a process, filling a disk or
  exhausting memory are recorded as ``skip`` with the manual reproduction in the
  docstring, rather than as a fragile automated approximation.

The four open findings are pinned across this module and the module that can
exercise them most directly.  Corrupt confidence coverage is pinned in
``test_confidence_score.py``.  This module owns the remaining findings:
SIGTERM cleanup, leaked scratch directories and unwritable output handling
are all incorrect.
"""

from __future__ import annotations

import contextlib
import csv
import io
import os
import sys

import pytest

import ccp4_setup
import main
from bond_analysis import BOND_COLUMNS, CANDIDATE_COLUMNS


# --------------------------------------------------------------------------- #
# Shared scaffolding
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_ccp4_path(tmp_path, monkeypatch) -> str:
    """Put four do-nothing CCP4 executables on ``PATH`` and return their dir.

    The code paths exercised here only ask ``shutil.which`` whether the four
    programs exist; none of them is ever executed. Stubbing keeps these tests
    identical on a machine with CCP4 and on one without, which is the whole
    point of a known-limitations module.

    Two details matter for the strict xfails that depend on this fixture. First,
    ``shutil.which`` honours ``PATHEXT`` on Windows and will not resolve an
    extensionless file, so a ``.bat`` shim is written alongside each script
    there. Second, the fixture then verifies the stub through
    ``ccp4_setup.ccp4_tools_available`` -- the very predicate ``src`` uses -- and
    *skips* if it did not take effect. Without that check a platform where the
    stub is invisible makes ``resolve_ccp4_environment`` fail for an unrelated
    reason, which would XPASS a strict xfail and turn the suite red while
    announcing that someone had fixed a defect that is still there.
    """
    bindir = tmp_path / "ccp4-stub-bin"
    bindir.mkdir()
    for tool in ccp4_setup.REQUIRED_CCP4_TOOLS:
        script = bindir / tool
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(0o755)
        if sys.platform == "win32":  # pragma: no cover - POSIX development host
            shim = bindir / f"{tool}.bat"
            shim.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(bindir))
    if not ccp4_setup.ccp4_tools_available(os.environ):  # pragma: no cover
        pytest.skip(
            "the CCP4 stub executables in "
            f"{bindir} are not resolvable through PATH on this platform, so "
            "the driver would fail for a reason unrelated to the defect under "
            "test")
    return str(bindir)


def _run_driver(argv):
    """Call ``main.main`` in-process, capturing output and the exit code."""
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main.main(list(argv))
    except SystemExit as exc:
        if isinstance(exc.code, int):
            code = exc.code
        else:
            code = 1
            if exc.code:
                print(exc.code, file=err)
    return int(code or 0), out.getvalue() + err.getvalue()


def _isolated_paths(tmp_path):
    """Driver options that keep every default path out of the repository."""
    return [
        "--pdb-redo-cache", str(tmp_path / "cache"),
        "--pdb-redo-root", str(tmp_path / "absent-mirror"),
        "--confidence-reference-dir", str(tmp_path / "absent-reference"),
    ]


def _write_csv(path, columns, rows=()):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _completed_output_dir(tmp_path, pdb_id="109m"):
    """An output directory whose manifest marks ``pdb_id`` terminally ``ok``.

    A ``--resume`` run over it therefore schedules no entries at all, which is
    what makes the resume-related test fast and network-free.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_csv(output_dir / "manifest.csv", main.MANIFEST_COLUMNS, [
        {"pdbID": pdb_id, "status": "ok", "retryable": "False",
         "n_metals": "1", "n_bonds": "1", "n_candidates": "1"}])
    _write_csv(output_dir / "metal_stats_all.csv", main.STATS_COLUMNS,
               [{"pdbID": pdb_id}])
    _write_csv(output_dir / "metal_bonds_all.csv", BOND_COLUMNS,
               [{"pdbID": pdb_id}])
    _write_csv(output_dir / "metal_candidates_all.csv", CANDIDATE_COLUMNS,
               [{"pdbID": pdb_id}])
    return output_dir


def _id_file(tmp_path, pdb_ids):
    path = tmp_path / "ids.txt"
    path.write_text(" ".join(pdb_ids) + "\n", encoding="utf-8")
    return str(path)


# --ccp4-setup is now honoured ahead of the ambient PATH; the regression
# tests live in tests/test_cli_and_config.py.

# --------------------------------------------------------------------------- #
# 2. SIGTERM orphans workers (not automatable)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason=(
    "requires SIGTERM to a real multi-worker driver and inspection of orphaned "
    "child processes; an in-process approximation would be timing-dependent "
    "and could leave stray processes behind on a failure -- reproduce by hand, "
    "see the docstring"))
def test_sigterm_to_the_driver_reaps_its_workers():
    """SIGTERM to the driver should stop workers and remove their scratch dirs.

    Manual reproduction:

    1. Start a multi-entry run with several workers::

           python3 src/main.py --id-file ids.txt --workers 4 \\
               --output-dir /tmp/alchemy-sigterm

    2. Once ``.alchemy-<id>-*`` directories appear under the output directory,
       send ``SIGTERM`` to the driver process only (not the process group).
    3. Observed: the driver exits, its worker children keep running to
       completion as orphans, and every ``.alchemy-<id>-*`` directory they were
       using is left behind under ``--output-dir``.
    4. Expected: the driver installs a SIGTERM handler that terminates the pool,
       waits for the workers, and removes their working directories before
       exiting non-zero.

    Related to the leak in
    :func:`test_resume_sweeps_a_leaked_per_entry_working_directory`, but with a
    different trigger: there the run is interrupted, here it is signalled.
    """


# Killing an idle Pool worker used to deadlock Pool shutdown. Fixed: the driver
# no longer relies on ``Pool.__exit__``, bounding teardown and force-killing any
# survivor. Covered by
# tests/test_worker_recovery.py::test_idle_worker_death_does_not_wedge_pool_shutdown


# Declared contacts on strict-NCS images used to be mislabelled
# crystallographic, and a declaration with two unresolved partners used to
# vanish silently. Both fixed; the regression tests moved to
# tests/test_declared_connections.py.


# Declared donors outside the 20 amino acids + water used to disappear
# entirely. Fixed: they are retained as candidate evidence with their
# measured distance and declaration provenance, but are never scored. See
# tests/test_declared_connections.py::test_declared_donor_outside_the_standard_residues_is_kept_as_evidence


# --------------------------------------------------------------------------- #
# 5. An interrupted run leaks its per-entry working directory
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "per-entry scratch directories are created with tempfile.mkdtemp("
    "prefix='.alchemy-<id>-', dir=output_dir) (src/main.py ~1142/1162) and only "
    "removed on the normal completion path (~1292); nothing sweeps a directory "
    "left behind by an interrupted run, and --resume does not either")
)
def test_resume_sweeps_a_leaked_per_entry_working_directory(
        tmp_path, stub_ccp4_path):
    """A resume run should clear scratch directories a dead run left behind.

    Each entry gets a ``.alchemy-<id>-XXXX`` directory inside ``--output-dir``
    holding its maps, which are deleted once its rows are extracted. Kill the
    run in between -- Ctrl-C, SIGTERM, an OOM kill -- and that directory
    survives, holding tens of megabytes of map per entry. ``--resume`` is the
    documented way to pick a run back up and is the natural place to clean up,
    but it walks straight past the leftovers, so they accumulate over every
    interrupted attempt until someone notices by hand.

    Simulated by planting a leftover directory rather than by killing a real
    run, so the test is deterministic and leaves no stray processes.
    """
    output_dir = _completed_output_dir(tmp_path)
    leaked = output_dir / ".alchemy-109m-leaked"
    leaked.mkdir()
    (leaked / "2mFo-DFc.map").write_text("stale map bytes", encoding="utf-8")

    exit_code, text = _run_driver(
        ["--id-file", _id_file(tmp_path, ["109m"]), "--resume",
         "--output-dir", str(output_dir), *_isolated_paths(tmp_path)])
    assert exit_code == 0, text

    leftovers = sorted(name for name in os.listdir(output_dir)
                       if name.startswith(".alchemy-"))
    assert leftovers == [], f"resume left scratch directories behind: {leftovers}"


# --------------------------------------------------------------------------- #
# 6. An unwritable --output-dir produces a traceback
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/main.py ~2191 calls os.makedirs(args.output_dir, exist_ok=True) "
    "unguarded, so an unwritable destination escapes main() as a raw "
    "PermissionError traceback instead of the clean SystemExit every other "
    "unusable-input path produces")
)
def test_unwritable_output_dir_fails_with_a_clean_message(
        tmp_path, stub_ccp4_path):
    """A read-only ``--output-dir`` is a user error, not a crash.

    Every other bad input -- an unknown id, both ``--id`` and ``--id-file``, a
    missing CCP4 setup file -- exits via ``SystemExit`` with a sentence the user
    can act on. Pointing ``--output-dir`` somewhere unwritable (a read-only
    mount, someone else's directory) instead dumps a traceback, which reads as
    an Alchemy bug rather than a fixable mistake.

    "Acts on" is what is actually asserted: the exit has to be non-zero -- a
    failed run must not look like a success -- and the message has to name the
    directory it could not create, so the user knows which path to fix.
    """
    if sys.platform == "win32":  # pragma: no cover - POSIX-only permissions
        pytest.skip("POSIX directory permissions required")
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")

    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    output_dir = parent / "output"
    argv = ["--id-file", _id_file(tmp_path, ["109m"]),
            "--output-dir", str(output_dir),
            *_isolated_paths(tmp_path)]
    try:
        try:
            exit_code, message = _run_driver(argv)
        except OSError as exc:
            pytest.fail(
                "a raw filesystem exception escaped the CLI boundary: "
                f"{type(exc).__name__}: {exc}")
    finally:
        parent.chmod(0o700)

    assert exit_code != 0, "an unusable --output-dir must fail the run"
    assert str(output_dir) in message, (
        "the message must name the directory Alchemy could not create; got:\n"
        + message)
    assert "Traceback (most recent call last)" not in message, message


# --no-bonds help text, non-positive --max-pdbs and CCP4 config precedence
# were all fixed; their regression tests live in
# tests/test_cli_and_config.py.


# The placeholder 1 x 1 x 1 unit cell is now rejected for DPI; the
# regression test lives in tests/test_bond_geometry.py.
