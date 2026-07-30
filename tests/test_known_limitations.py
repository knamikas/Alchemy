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

The fourteen open findings are pinned across this module and the module that can
exercise them most directly.  The watchdog-attribution pin lives in
``test_worker_recovery.py`` and corrupt confidence coverage is pinned in
``test_confidence_score.py``.  This module owns the remaining findings:

* declarations with two unresolved partners disappear without an audit trace;
* declared non-protein donors disappear, and strict-NCS declarations are
  mislabelled;
* explicit CCP4 selection, SIGTERM cleanup, leaked scratch directories and
  unwritable output handling are incorrect;
* non-positive ``--max-pdbs``, ``--no-bonds`` help and CCP4 config precedence
  violate their CLI contracts;
* an idle worker killed while holding CPython's Pool queue lock can deadlock
  shutdown (documented skip);
* a degenerate 1-cubic-Angstrom unit cell is accepted for DPI.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import sys
import types

import gemmi
import pytest

import bond_analysis as ba
import ccp4_setup
import helpers
import main
from bond_analysis import BOND_COLUMNS, CANDIDATE_COLUMNS
from helpers import AtomSpec, EDSTATS_HEADER, StructureBuilder
from structure_analysis import load_structure


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


def _option_help(option: str) -> str:
    """The rendered ``--help`` paragraph for one option."""
    captured = io.StringIO()
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(captured):
            main.parse_args(["--help"])
    text = captured.getvalue()
    # The option name also appears in the usage banner; the description block is
    # the last occurrence.
    start = text.rindex(option)
    end = text.find("\n  --", start)
    return text[start:end if end != -1 else len(text)]


# --------------------------------------------------------------------------- #
# 1. --ccp4-setup is ignored when CCP4 is already on PATH
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/main.py ~305: resolve_ccp4_environment returns as soon as "
    "ccp4_tools_available(os.environ) is true, before it ever consults "
    "args.ccp4_setup, so an explicit override -- even a nonexistent path -- is "
    "silently discarded"))
def test_explicit_ccp4_setup_is_honoured_even_when_ccp4_is_on_path(
        stub_ccp4_path):
    """An explicit ``--ccp4-setup`` must not be silently ignored.

    A user who passes ``--ccp4-setup`` is overriding whatever is on ``PATH``,
    usually because the shell has the wrong CCP4 version sourced. Pointing the
    option at a path that does not exist has to be an error: the run otherwise
    proceeds against the very environment the user was trying to replace, and
    reports nothing.
    """
    args = argparse.Namespace(configure_ccp4=None,
                              ccp4_setup="/nonexistent/ccp4.setup-sh")
    with pytest.raises(SystemExit, match="not found"):
        main.resolve_ccp4_environment(args)


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


# --------------------------------------------------------------------------- #
# Killing an idle Pool worker can deadlock Pool shutdown (not automatable)
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason=(
    "destructively reproducing this CPython multiprocessing deadlock can leave "
    "pytest itself blocked in Pool._terminate_pool; keep the manual process-"
    "level recipe in the docstring instead"))
def test_killing_an_idle_pool_worker_does_not_deadlock_driver_shutdown():
    """An idle-worker death must not prevent an otherwise complete run exiting.

    Manual reproduction on POSIX:

    1. Start a batch with at least two workers and enough unevenly sized entries
       that one worker becomes idle while another is still processing.
    2. Identify the idle child and send it ``SIGKILL`` while it is blocked in
       the Pool task queue's ``inqueue.get()``.
    3. Let the remaining entries finish.  The current driver writes a complete
       manifest but can then remain inside the ``with Pool(...)`` teardown:
       the killed child held ``SimpleQueue._rlock``, so CPython's
       ``Pool._terminate_pool`` waits forever trying to drain that queue.
    4. Expected: worker supervision and teardown are designed so the driver
       exits with a run log and status code even when an idle worker disappears.

    This is deliberately not approximated inside pytest.  A timeout around the
    driver would detect the hang but would not make it safe to tear down the
    wedged Pool, and a failed test could strand descendants on the test host.
    """


# --------------------------------------------------------------------------- #
# 3. Declared contacts on strict-NCS images are mislabelled
# --------------------------------------------------------------------------- #
def _structure_with_one_ncs_operation(tmp_path):
    """A Zn and a water whose +10 Angstrom NCS image sits 2.09 A from the Zn.

    The water is 7.91 Angstrom away inside the asymmetric unit, so the only
    contact-range approach to the metal is through strict-NCS operation ``1``.
    """
    builder = StructureBuilder()
    metal = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    water = builder.add_water(101, (-7.91, 0.0, 0.0), chain="W")
    structure = builder.to_gemmi()
    transform = gemmi.Transform()
    transform.mat.fromlist([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.vec.fromlist([10.0, 0.0, 0.0])
    structure.ncs.append(gemmi.NcsOp(transform, "1", False))
    path = str(tmp_path / "ncs.pdb")
    structure.write_pdb(path)
    return path, metal, water


@pytest.mark.xfail(strict=True, reason=(
    "src/bond_analysis.py ~871 and ~896: both return branches of "
    "_declared_candidate_geometry hardcode strict_ncs_contact False and an "
    "empty strict_ncs_operation_id, so a declared contact reached through a "
    "strict-NCS image is reported as an ordinary crystallographic contact"))
def test_declared_contact_through_an_ncs_image_is_labelled_strict_ncs(tmp_path):
    """A declared contact must get the same provenance as an inferred one.

    ``StructureContext.image_provenance`` is the single source of truth for
    whether a Gemmi image is crystallographic, strict-NCS or both, and the
    proximity search already uses it. The declared-connection path computes its
    own answer and never reports strict NCS, so the same physical contact is
    labelled ``crystallographic`` when it is declared and ``strict_ncs`` when it
    is inferred. Downstream that changes ``contact_scope`` and the per-site
    ``coordination_depends_on_strict_ncs`` summary.

    The connection is built in-process rather than round-tripped through a file
    because Gemmi normalises ``Connection.asu`` back to ``Same`` on write, which
    would never reach the symmetry branch.
    """
    path, _, _ = _structure_with_one_ncs_operation(tmp_path)
    context = load_structure("test", path)
    assert context.strict_ncs_operation_ids == ("1",)

    metal = context.metal_atoms(["ZN"])[0]
    neighbor = next(atom for atom in context.contact_atoms
                    if atom.atom_name == "O")
    connection = types.SimpleNamespace(asu=gemmi.Asu.Any)

    geometry = ba._declared_candidate_geometry(
        context, metal, neighbor, connection)

    # The proximity path's verdict for the very same image is the oracle.
    crystallographic, strict_ncs, ncs_id, scope = context.image_provenance(
        geometry["symmetry_image_index"], tuple(geometry["translation"]))
    assert (strict_ncs, ncs_id, scope) == (True, "1", "strict_ncs")

    assert geometry["strict_ncs_contact"] == strict_ncs
    assert geometry["strict_ncs_operation_id"] == ncs_id
    assert geometry["crystallographic_contact"] == crystallographic
    assert geometry["contact_scope"] == scope


# --------------------------------------------------------------------------- #
# Declaration whose two partners are both unresolved is dropped silently
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/bond_analysis.py ~984 tests metal membership before unresolved "
    "partners; when both source identities are absent from the analysis model, "
    "neither looks like a selected metal and the declaration is silently "
    "continued without a row, reason code or message"))
def test_declaration_with_two_unresolved_partners_leaves_an_audit_trace(
        tmp_path):
    """Failure to map either declared partner must make the result incomplete."""
    source_builder = StructureBuilder()
    donor = source_builder.add_amino_acid(
        "HIS", 10, chain="A", positions={"NE2": (2.03, 0.0, 0.0)})
    metal = source_builder.add_metal(
        "ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    source_builder.add_connection(
        metal.ref("ZN"), donor.ref("NE2"), name="unmapped",
        reported_distance=2.03)
    source = source_builder.write_cif(tmp_path / "source.cif")

    # Bond analysis still has a selected metal site, but neither source author
    # identity exists in this deliberately unrelated analysis model.
    analysis_builder = StructureBuilder()
    analysis_builder.add_metal(
        "ZN", 99, chain="Z", pos=(20.0, 20.0, 20.0))
    analysis = analysis_builder.write_pdb(tmp_path / "analysis.pdb")
    context = load_structure("test", analysis)
    rows, candidates, _, metadata = ba.run_bond_analysis(
        "test", analysis, [], list(EDSTATS_HEADER), helpers.dpi_inputs(),
        structure=context, connection_path=source)

    trace = (
        "declared_connection_resolution_incomplete"
        in metadata["partial_reason_codes"]
        or any("unmapped" in message for message in metadata["messages"])
    )
    assert trace, (
        "the declaration produced no audit trace when both partners were "
        f"unresolved (rows={rows!r}, candidates={candidates!r})")


# --------------------------------------------------------------------------- #
# Declared donors outside the 20 amino acids + water are dropped silently
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/bond_analysis.py ~995: a declared partner whose residue is neither "
    "water nor one of the 20 standard amino acids hits a bare `continue`, so "
    "the depositor's own bond produces no bond row, no candidate row and no "
    "reason code or coordination evidence"))
def test_declared_donor_outside_the_standard_residues_leaves_a_trace(tmp_path):
    """A dropped declaration must still be accounted for somewhere.

    Nucleic acids, modified residues and organic ligands are legitimate metal
    donors and the depositor declared this one explicitly with its own distance.
    Alchemy may well decide not to score such a contact -- there is no
    literature reference distance for it -- but dropping it without a bond row,
    a candidate row or a reason code is indistinguishable from "this metal has
    no coordination at all".

    The assertion is deliberately permissive about *how* the contact is
    accounted for, so any of the plausible fixes turns it green.
    """
    builder = StructureBuilder()
    metal = builder.add_metal("MN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    nucleotide = builder.add_hetero_residue("DG", 10, [
        AtomSpec("P", "P", (2.20, 1.20, 0.0)),
        AtomSpec("OP1", "O", (2.15, 0.0, 0.0)),
        AtomSpec("OP2", "O", (3.30, 2.00, 0.0)),
    ], chain="C")
    builder.add_connection(metal.ref("MN"), nucleotide.ref("OP1"),
                           name="metalc1", reported_distance=2.15)
    path = builder.write_cif(tmp_path / "nucleotide.cif")

    context = load_structure("test", path)
    rows, candidates, _, metadata = ba.run_bond_analysis(
        "test", path, [], list(EDSTATS_HEADER), helpers.dpi_inputs(),
        structure=context, connection_path=path)

    declared_codes = {
        code for code in (
            *metadata["warning_codes"], *metadata["partial_reason_codes"])
        if "declared" in code or "connection" in code
    }
    declared_message = any(
        "metalc1" in message or "declared" in message.lower()
        for message in metadata["messages"])
    accounted_for = (
        bool(rows) or bool(candidates) or bool(declared_codes)
        or declared_message)
    assert accounted_for, (
        "the declared MN-OP1 connection produced no bond row, no candidate row "
        "and no declaration-specific audit trace")


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


# --------------------------------------------------------------------------- #
# 7. --help describes --no-bonds as defaulting to True
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/main.py ~1662: --no-bonds is a store_false into dest='bonds' with "
    "set_defaults(bonds=True) at ~1664, and the parser is built with "
    "ArgumentDefaultsHelpFormatter at ~1621, so argparse appends the default of "
    "`bonds` to the help of `--no-bonds` and prints '(default: True)' -- the "
    "exact negation of what passing the flag does")
)
def test_help_does_not_claim_that_no_bonds_defaults_to_true():
    """``--help`` must not tell the user that ``--no-bonds`` defaults to True.

    Bond analysis is on by default and ``--no-bonds`` turns it off. The rendered
    help says the opposite, which a user can only read as "bonds are already
    skipped by default". Whatever the fix -- a per-option formatter, an explicit
    ``help=`` suffix, or splitting the flag pair -- the rendered paragraph must
    not carry a bare ``(default: True)``.
    """
    paragraph = _option_help("--no-bonds")
    assert "skip the metal-ligand bond-distance stage" in paragraph
    assert "(default: True)" not in paragraph, paragraph


# --------------------------------------------------------------------------- #
# 8. --max-pdbs accepts 0 and negative caps
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/main.py ~1632 declares --max-pdbs as a plain `type=int` with no "
    "validation, unlike --workers which uses positive_int, so 0 and negative "
    "caps are accepted")
)
@pytest.mark.parametrize("value", ["0", "-5"])
def test_max_pdbs_rejects_non_positive_caps(value):
    """``--max-pdbs`` is documented as "process only the first N entries".

    Zero and negative values have no meaning under that description.
    ``--workers`` already rejects them through ``positive_int``; ``--max-pdbs``
    should be validated the same way instead of silently accepting a cap that
    cannot express a positive number of entries.

    The rejection has to be *about* ``--max-pdbs``: a bare
    ``pytest.raises(SystemExit)`` would also be satisfied by an exit provoked by
    something else in the argument list, and this xfail is strict, so that would
    read as "someone fixed --max-pdbs" when nobody had.
    """
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with pytest.raises(SystemExit) as excinfo:
            main.parse_args(["--id", "109m", "--max-pdbs", value])

    # ``argparse.error`` exits 2 with the diagnostic on stderr and nothing but
    # the code on the exception (``str(SystemExit(2)) == "2"``, which is why
    # ``pytest.raises(match=...)`` cannot be used here); an explicit
    # ``raise SystemExit(message)`` carries it the other way round. Accept both.
    assert excinfo.value.code not in (0, None)
    message = f"{stderr.getvalue()}\n{excinfo.value}"
    assert "max-pdbs" in message, (
        f"--max-pdbs {value} was rejected, but not by name:\n" + message)


@pytest.mark.xfail(strict=True, reason=(
    "src/main.py ~2297 applies the cap as `ids = ids[:args.max_pdbs]`, so a "
    "negative cap is a Python slice from the end: --max-pdbs -3 over three ids "
    "quietly schedules nothing at all instead of being rejected")
)
def test_negative_max_pdbs_does_not_silently_drop_entries_from_the_end(
        tmp_path, stub_ccp4_path):
    """A negative cap must not quietly discard the tail of the id list.

    ``ids[:args.max_pdbs]`` with a negative cap trims from the *end*, so
    ``--max-pdbs -3`` over three ids yields an empty list and the driver reports
    "No entries to process" and exits 0 -- a run that silently did nothing while
    looking like a success. The user asked for a cap and got a truncation from
    the wrong end.

    The driver is offline here (the mirror root points at a directory that does
    not exist), so a bare ``exit_code != 0`` would be satisfied by any unrelated
    failure -- a mirror lookup, a download -- and this xfail is strict: that
    would turn the suite red announcing a fix to ``--max-pdbs`` that nobody
    made. The check is therefore that the run is rejected *by name*.
    """
    output_dir = tmp_path / "output"
    exit_code, text = _run_driver(
        ["--id-file", _id_file(tmp_path, ["109m", "1cll", "100d"]),
         "--max-pdbs", "-3", "--output-dir", str(output_dir),
         *_isolated_paths(tmp_path)])

    assert "max-pdbs" in text, (
        "a negative --max-pdbs must be refused with a message naming the "
        f"option; the run said:\n{text}")
    assert exit_code != 0, (
        "a negative --max-pdbs silently produced a successful empty run:\n"
        + text)
    assert "No entries to process" not in text, (
        "the negative cap trimmed the id list from the end instead of being "
        f"rejected:\n{text}")


# --------------------------------------------------------------------------- #
# 9. The CCP4 config loader is last-wins, the writer targets the first file
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/ccp4_setup.py ~66: load_ccp4_setup_config merges with "
    "config.update(data) over the whole list, so the LAST readable file wins, "
    "while save_ccp4_setup writes only config_files[0] (~74); with the "
    "three-entry list "
    "from main.default_ccp4_config_files a saved setup path is shadowed by any "
    "later file")
)
def test_saved_ccp4_setup_is_the_one_that_gets_loaded_back(tmp_path):
    """``--configure-ccp4`` must actually take effect on the next run.

    ``save_ccp4_setup`` writes the user-level file (``config_files[0]``), but
    ``load_ccp4_setup_config`` merges every file in order and lets the last one
    win -- and ``main.default_ccp4_config_files`` puts the in-repo
    ``.alchemy/ccp4.json`` last. A user who runs ``--configure-ccp4`` is told
    "saved CCP4 setup path to ~/.config/alchemy/ccp4.json" and then keeps
    getting the stale path from the repository file, with no diagnostic.

    Save and load must agree on which file is authoritative.
    """
    primary = tmp_path / "user" / "ccp4.json"
    shadowing = tmp_path / "repo" / "ccp4.json"
    shadowing.parent.mkdir(parents=True)
    shadowing.write_text('{"ccp4_setup": "/stale/repo/ccp4.setup-sh"}\n',
                         encoding="utf-8")
    config_files = [str(primary), str(shadowing)]

    chosen = "/the/path/the/user/configured/ccp4.setup-sh"
    written = ccp4_setup.save_ccp4_setup(chosen, config_files=config_files)
    assert written == [str(primary)]

    loaded = ccp4_setup.load_ccp4_setup_config(config_files=config_files)
    assert loaded.get("ccp4_setup") == chosen


# --------------------------------------------------------------------------- #
# A degenerate default unit cell is accepted for DPI
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "src/bond_analysis.py ~292 accepts every finite positive cell volume; "
    "Gemmi's placeholder CRYST1 cell is 1 x 1 x 1 Angstrom in P 1, so a "
    "structure without meaningful crystallographic metadata supplies a bogus "
    "1 A^3 ASU volume instead of missing_or_invalid_asu_volume"))
def test_placeholder_one_angstrom_cell_is_not_used_for_dpi(tmp_path):
    """Gemmi's degenerate default cell is missing metadata, not a real ASU.

    A one-cubic-Angstrom cell cannot contain even one atom.  Accepting it makes
    the DPI calculation succeed with a physically impossible volume, which is
    worse than returning NaN and the existing
    ``missing_or_invalid_asu_volume`` reason code.
    """
    builder = StructureBuilder(
        cell=(1.0, 1.0, 1.0, 90.0, 90.0, 90.0), spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    pdb_path = builder.write_pdb(tmp_path / "placeholder-cell.pdb")

    volume = ba._asu_volume(str(tmp_path / "absent.mtz"), pdb_path)

    assert math.isnan(volume), (
        f"a placeholder 1 A^3 cell was accepted as ASU volume {volume!r}")
