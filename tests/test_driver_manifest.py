"""Batch-driver data management in ``src/main.py``.

Scope: argument validation (``positive_int``), the resume bookkeeping that
decides which entries are already finished (``load_done``,
``_manifest_values_by_id``, ``_resume_replacement_succeeded``), the manifest
row projection and its provenance skeleton (``_manifest_row``,
``_initial_result``), the staged-retry and streamed-output machinery
(``_ResumeStaging``, ``_OutputWriters``), and the two coordinate-preparation
converters (``_cif_to_pdb``, ``_first_model_pdb``).

Worker-death recovery and end-to-end pipeline runs are covered elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List

import gemmi
import pytest

import helpers
import main
import structure_analysis


# --------------------------------------------------------------------------- #
# Local fixtures / builders
# --------------------------------------------------------------------------- #
CFG = {
    "alchemy_commit": "abc123def456",
    "gemmi_version": "0.7.5",
    "ccp4_version": "9.0",
}

# Columns that ``_manifest_row`` recomputes from the worker result rather than
# copying straight out of it.
DERIVED_MANIFEST_COLUMNS = frozenset({
    "n_metals", "n_bonds", "n_candidates", "runtime_s",
    "reason_codes", "warning_codes",
})


def _result(pdb_id="109m", **overrides):
    """A worker result skeleton plus overrides, as the driver would see it."""
    result = main._initial_result(pdb_id, CFG, None)
    result.update(overrides)
    return result


def _write_manifest(path, rows, columns=None):
    """Write a manifest CSV with the real schema and the given partial rows."""
    columns = list(columns if columns is not None else main.MANIFEST_COLUMNS)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return str(path)


def _manifest_ids(path, **kwargs):
    return main.load_done(str(path), **kwargs)


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


# mmCIF scaffolding for the conversion tests. Written by hand rather than by
# gemmi because the behaviours under test are about raw ``.``/``?`` occupancy
# tokens and >3-character component ids, neither of which a gemmi round trip
# would produce.
_CIF_HEADER = """data_TEST
_cell.length_a 60.0
_cell.length_b 70.0
_cell.length_c 80.0
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""

_CIF_HEADER_NO_OCCUPANCY = _CIF_HEADER.replace("_atom_site.occupancy\n", "")


def _cif_atom(serial, element, atom_name, comp_id, label_asym, entity,
              label_seq, xyz, occupancy, auth_seq, auth_asym,
              group="ATOM", with_occupancy=True):
    x, y, z = xyz
    fields = [group, str(serial), element, atom_name, ".", comp_id,
              label_asym, str(entity), str(label_seq), "?",
              f"{x}", f"{y}", f"{z}"]
    if with_occupancy:
        fields.append(occupancy)
    fields += ["20.0", str(auth_seq), auth_asym, "1"]
    return " ".join(fields)


def _write_cif(path, atom_lines, header=_CIF_HEADER):
    text = header + "".join(line + "\n" for line in atom_lines)
    with open(path, "w") as handle:
        handle.write(text)
    return str(path)


def _pdb_atom_lines(path):
    with open(path) as handle:
        return [line for line in handle
                if line[:6].strip().upper() in ("ATOM", "HETATM")]


def _occupancy_field(line):
    """PDB columns 55-60, exactly as written."""
    return line[54:60]


def _element_field(line):
    """PDB columns 77-78, exactly as written."""
    return line[76:78].strip()


def _simple_structure(residues):
    """Build a one-model gemmi structure from (chain, seqid, name, atoms).

    ``add_chain``/``add_residue`` copy their argument, so each chain is fully
    populated before it is attached.
    """
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain_order = []
    chains = {}
    for chain_name, seqid, resname, atoms in residues:
        if chain_name not in chains:
            chains[chain_name] = gemmi.Chain(chain_name)
            chain_order.append(chain_name)
        residue = gemmi.Residue()
        residue.name = resname
        residue.seqid = gemmi.SeqId(seqid, " ")
        for atom_name, element in atoms:
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(0.0, 0.0, 0.0)
            residue.add_atom(atom)
        chains[chain_name].add_residue(residue)
    for chain_name in chain_order:
        model.add_chain(chains[chain_name])
    structure.add_model(model)
    return structure


_BASE_RESIDUES = [
    ("A", 1, "GLY", [("N", "N"), ("CA", "C")]),
    ("B", 2, "ZN", [("ZN", "ZN")]),
]


# --------------------------------------------------------------------------- #
# positive_int
# --------------------------------------------------------------------------- #
class TestPositiveInt:
    """``positive_int`` is the argparse gate for --workers/--max-pdbs/etc."""

    @pytest.mark.parametrize("value,expected", [
        ("1", 1),
        ("2", 2),
        ("28", 28),
        ("  7  ", 7),        # int() tolerates surrounding whitespace
        ("+3", 3),
        ("1000000", 1000000),
        (5, 5),              # already an int (programmatic callers)
    ])
    def test_accepts_integers_of_at_least_one(self, value, expected):
        """Any representation of an integer >= 1 is accepted and normalized."""
        assert main.positive_int(value) == expected
        assert isinstance(main.positive_int(value), int)

    @pytest.mark.parametrize("value", ["0", "-1", "-28", 0, -3])
    def test_rejects_zero_and_negative(self, value):
        """Zero workers or a zero cap would silently do nothing; reject them."""
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            main.positive_int(value)
        assert "at least 1" in str(excinfo.value)

    @pytest.mark.parametrize("value", [
        "1.5", "2.0", "abc", "", " ", "1e3", "0x2", None, "nan", "inf",
        "1,000",
    ])
    def test_rejects_non_integer_text(self, value):
        """Argparse hands over raw strings; anything non-integral is a usage error."""
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            main.positive_int(value)
        assert "positive integer" in str(excinfo.value)

    def test_boundary_is_one_not_zero(self):
        """1 is the smallest accepted value; 0 is the first rejected one."""
        assert main.positive_int("1") == 1
        with pytest.raises(argparse.ArgumentTypeError):
            main.positive_int("0")


# --------------------------------------------------------------------------- #
# load_done
# --------------------------------------------------------------------------- #
class TestLoadDone:
    """Which manifest rows count as finished work that --resume may skip."""

    @pytest.mark.parametrize("status,retryable,expected_done", [
        ("ok", "False", True),
        ("ok", "True", True),          # status ok is terminal regardless
        ("ok", "", True),
        ("partial", "False", True),    # terminal partial: nothing left to do
        ("partial", "0", True),
        ("partial", "no", True),
        ("partial", "True", False),    # retryable partial must be retried
        ("partial", "", False),
        ("error", "True", False),
        ("error", "False", False),     # even a terminal error is not "done"
        ("skip", "False", False),
        ("skip", "True", False),
        ("", "", False),
    ])
    def test_terminality_by_status_and_retryable(self, tmp_path, status,
                                                 retryable, expected_done):
        """Only ok and non-retryable partial rows are skippable."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "109m", "status": status, "retryable": retryable,
             "n_bonds": "3", "n_candidates": "5"},
        ])
        assert ("109m" in _manifest_ids(path)) is expected_done

    def test_ids_are_normalized_to_lowercase(self, tmp_path):
        """Manifest IDs join against the driver's lowercased selection list."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": " 1CLL ", "status": "ok", "retryable": "False"},
        ])
        assert _manifest_ids(path) == {"1cll"}

    def test_missing_manifest_is_an_empty_done_set(self, tmp_path):
        """A first run has no manifest; resume must not crash on that."""
        assert _manifest_ids(tmp_path / "absent.csv") == set()

    def test_manifest_without_the_required_columns_is_ignored(self, tmp_path):
        """A foreign CSV cannot be mistaken for completion evidence."""
        path = tmp_path / "manifest.csv"
        with open(path, "w", newline="") as handle:
            csv.writer(handle).writerows([["id", "state"], ["109m", "ok"]])
        assert _manifest_ids(path) == set()

    def test_empty_manifest_file_is_ignored(self, tmp_path):
        """A zero-byte manifest has no header and therefore no done rows."""
        path = tmp_path / "manifest.csv"
        path.write_text("")
        assert _manifest_ids(path) == set()

    def test_row_without_a_pdb_id_is_not_done(self, tmp_path):
        """A blank ID cannot mark anything complete."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "", "status": "ok", "retryable": "False",
             "n_bonds": "0", "n_candidates": "0"},
        ])
        assert _manifest_ids(path) == set()

    @pytest.mark.parametrize("n_bonds,n_candidates,expected_done", [
        ("0", "0", True),      # measured zero: the stage ran and found none
        ("4", "9", True),
        ("", "0", False),      # blank: the stage never ran
        ("0", "", False),
        ("", "", False),
        ("   ", "0", False),   # whitespace is still blank
    ])
    def test_blank_counts_mean_the_bond_stage_never_ran(
            self, tmp_path, n_bonds, n_candidates, expected_done):
        """README: blank counts = not run, 0 = ran and found nothing."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "109m", "status": "ok", "retryable": "False",
             "n_bonds": n_bonds, "n_candidates": n_candidates},
        ])
        done = _manifest_ids(path, bonds_required=True)
        assert ("109m" in done) is expected_done

    def test_blank_counts_are_irrelevant_when_bonds_are_not_required(
            self, tmp_path):
        """A --no-bonds resume must not re-run density-complete entries."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "109m", "status": "ok", "retryable": "False",
             "n_bonds": "", "n_candidates": ""},
        ])
        assert _manifest_ids(path, bonds_required=False) == {"109m"}

    @pytest.mark.parametrize("bond_present,candidate_present,expected_done", [
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ])
    def test_absent_bond_outputs_make_the_result_incomplete(
            self, tmp_path, bond_present, candidate_present, expected_done):
        """Counts in the manifest are worthless if the CSV rows are gone."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "109m", "status": "ok", "retryable": "False",
             "n_bonds": "4", "n_candidates": "9"},
        ])
        done = main.load_done(path, bonds_required=True,
                              bond_output_present=bond_present,
                              candidate_output_present=candidate_present)
        assert ("109m" in done) is expected_done

    def test_selects_only_the_terminal_rows_of_a_mixed_manifest(self, tmp_path):
        """A realistic manifest yields exactly the finished subset."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "109m", "status": "ok", "retryable": "False",
             "n_bonds": "6", "n_candidates": "8"},
            {"pdbID": "1cll", "status": "partial", "retryable": "False",
             "n_bonds": "0", "n_candidates": "0"},
            {"pdbID": "1blu", "status": "partial", "retryable": "True",
             "n_bonds": "1", "n_candidates": "2"},
            {"pdbID": "2fha", "status": "error", "retryable": "True"},
            {"pdbID": "2cyp", "status": "skip", "retryable": "False"},
            {"pdbID": "100d", "status": "ok", "retryable": "False",
             "n_bonds": "", "n_candidates": ""},
        ])
        assert _manifest_ids(path, bonds_required=True) == {"109m", "1cll"}
        assert _manifest_ids(path, bonds_required=False) == {
            "109m", "1cll", "100d"}


# --------------------------------------------------------------------------- #
# _manifest_values_by_id
# --------------------------------------------------------------------------- #
class TestManifestValuesById:
    """Reading one prior manifest column for the resume carry-forward."""

    def test_returns_the_column_keyed_by_normalized_id(self, tmp_path):
        """Values are returned verbatim under a lowercased, stripped key."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "109M", "status": "ok", "n_bonds": "7"},
            {"pdbID": " 1cll", "status": "ok", "n_bonds": "0"},
            {"pdbID": "1blu", "status": "error", "n_bonds": ""},
        ])
        assert main._manifest_values_by_id(path, "n_bonds") == {
            "109m": "7", "1cll": "0", "1blu": ""}

    def test_missing_and_empty_files_yield_no_values(self, tmp_path):
        """A first run or a truncated manifest carries nothing forward."""
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        assert main._manifest_values_by_id(str(tmp_path / "gone.csv"),
                                           "n_bonds") == {}
        assert main._manifest_values_by_id(str(empty), "n_bonds") == {}

    def test_unknown_column_yields_blanks_not_an_error(self, tmp_path):
        """An older manifest lacking the column must degrade to "not run"."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok"}],
            columns=["pdbID", "status"])
        assert main._manifest_values_by_id(path, "n_bonds") == {"109m": ""}

    def test_blank_ids_are_dropped(self, tmp_path):
        """An unattributable row must not become a wildcard carry-forward."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "", "status": "ok", "n_bonds": "9"},
            {"pdbID": "109m", "status": "ok", "n_bonds": "1"},
        ])
        assert main._manifest_values_by_id(path, "n_bonds") == {"109m": "1"}

    def test_a_later_row_supersedes_an_earlier_one(self, tmp_path):
        """Resume appends, so the last row for an ID is the current one."""
        path = _write_manifest(tmp_path / "manifest.csv", [
            {"pdbID": "109m", "status": "error", "n_bonds": ""},
            {"pdbID": "109m", "status": "ok", "n_bonds": "5"},
        ])
        assert main._manifest_values_by_id(path, "n_bonds") == {"109m": "5"}


# --------------------------------------------------------------------------- #
# _initial_result
# --------------------------------------------------------------------------- #
class TestInitialResult:
    """The per-entry skeleton that guarantees a complete manifest row."""

    def test_seeds_bond_counts_blank_not_zero(self):
        """REGRESSION (cf55dd5): an unrun bond stage must not claim a zero.

        ``0`` is a measured result. Seeding it made an entry that failed
        before the bond stage look like a completed bond analysis.
        """
        result = main._initial_result("109m", CFG, None)
        assert result["n_bonds"] == ""
        assert result["n_candidates"] == ""
        assert result["n_bonds"] != 0
        assert result["n_candidates"] != 0

    def test_every_non_derived_manifest_column_is_present_up_front(self):
        """A failure at any stage still projects onto a complete row."""
        result = main._initial_result("109m", CFG, None)
        required = set(main.MANIFEST_COLUMNS) - DERIVED_MANIFEST_COLUMNS
        assert required.issubset(result.keys())

    def test_supplies_the_keys_manifest_row_reads_directly(self):
        """``_manifest_row`` indexes these without a default; they must exist."""
        result = main._initial_result("109m", CFG, None)
        for key in ("n", "runtime", "n_bonds", "n_candidates", "pdbID"):
            assert key in result

    def test_defaults_to_a_retryable_error(self):
        """An entry that dies before setting a status must be retried."""
        result = main._initial_result("109m", CFG, None)
        assert result["status"] == "error"
        assert result["retryable"] is True

    def test_carries_run_provenance_from_the_config(self):
        """Version provenance is stamped once and shared by every row."""
        result = main._initial_result("109m", CFG, None)
        assert result["alchemy_version"] == main.ALCHEMY_VERSION
        assert result["alchemy_commit"] == CFG["alchemy_commit"]
        assert result["gemmi_version"] == CFG["gemmi_version"]
        assert result["ccp4_version"] == CFG["ccp4_version"]
        assert result["model_policy"] == main.MODEL_POLICY
        assert result["altloc_policy"] == main.ALTLOC_POLICY
        assert result["symmetry_contact_policy"] == main.SYMMETRY_POLICY

    @pytest.mark.parametrize("manual_inputs,expected", [
        (None, "final"),
        ({}, "final"),
        ({"pdb_file": "/x.pdb"}, "manual"),
    ])
    def test_refinement_state_reflects_manual_inputs(self, manual_inputs,
                                                     expected):
        """Manual coordinate/MTZ input is not a PDB-REDO final re-refinement."""
        result = main._initial_result("109m", CFG, manual_inputs)
        assert result["refinement_state"] == expected

    def test_row_lists_are_independent_between_entries(self):
        """Two skeletons must not share mutable row accumulators."""
        first = main._initial_result("109m", CFG, None)
        second = main._initial_result("1cll", CFG, None)
        first["rows"].append({"x": 1})
        first["reason_codes"].append("boom")
        assert second["rows"] == []
        assert second["reason_codes"] == []


# --------------------------------------------------------------------------- #
# _manifest_row
# --------------------------------------------------------------------------- #
class TestManifestRow:
    """Projection of a worker result onto the manifest schema."""

    def test_projects_exactly_the_manifest_columns(self):
        """No worker-internal key leaks into the CSV and none is missing."""
        row = main._manifest_row(_result(), False, True, {}, {})
        assert set(row) == set(main.MANIFEST_COLUMNS)
        assert "rows" not in row
        assert "bond_rows" not in row
        assert "timings" not in row

    def test_renames_and_joins_the_derived_columns(self):
        """n/runtime become n_metals/runtime_s; code lists become pipe text."""
        result = _result(n=3, runtime=12.5, n_bonds=7, n_candidates=11,
                         reason_codes=["a", "b"], warning_codes=["w"])
        row = main._manifest_row(result, False, True, {}, {})
        assert row["n_metals"] == 3
        assert row["runtime_s"] == 12.5
        assert row["n_bonds"] == 7
        assert row["n_candidates"] == 11
        assert row["reason_codes"] == "a|b"
        assert row["warning_codes"] == "w"

    def test_empty_code_lists_render_blank(self):
        """No codes must not become a spurious separator or literal '[]'."""
        row = main._manifest_row(_result(), False, True, {}, {})
        assert row["reason_codes"] == ""
        assert row["warning_codes"] == ""

    def test_bonds_enabled_reports_the_measured_zero(self):
        """A bond run that found nothing records 0, distinct from blank."""
        row = main._manifest_row(_result(n_bonds=0, n_candidates=0),
                                 False, True, {}, {})
        assert row["n_bonds"] == 0
        assert row["n_candidates"] == 0

    def test_fresh_no_bonds_run_writes_blank_counts(self):
        """Without --resume there is no prior stage to carry forward."""
        row = main._manifest_row(_result(n_bonds=4, n_candidates=6),
                                 False, False,
                                 {"109m": "99"}, {"109m": "98"})
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_resume_no_bonds_carries_the_prior_counts_forward(self):
        """--resume --no-bonds preserves an earlier run's bond-stage counts."""
        row = main._manifest_row(_result("109M"), True, False,
                                 {"109m": "6"}, {"109m": "8"})
        assert row["n_bonds"] == "6"
        assert row["n_candidates"] == "8"

    def test_resume_no_bonds_carry_forward_preserves_a_prior_zero(self):
        """A prior measured zero stays a zero, not a blank."""
        row = main._manifest_row(_result(), True, False,
                                 {"109m": "0"}, {"109m": "0"})
        assert row["n_bonds"] == "0"
        assert row["n_candidates"] == "0"

    def test_resume_no_bonds_without_a_prior_row_stays_blank(self):
        """An entry new to this output dir has no bond stage to inherit."""
        row = main._manifest_row(_result("1cll"), True, False,
                                 {"109m": "6"}, {"109m": "8"})
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_prior_blank_counts_are_not_upgraded(self):
        """Carrying forward a blank must keep it blank, never zero."""
        row = main._manifest_row(_result(), True, False,
                                 {"109m": ""}, {"109m": ""})
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""


# --------------------------------------------------------------------------- #
# REGRESSION (bug 4): the unrun-bond-stage chain
# --------------------------------------------------------------------------- #
class TestUnrunBondStageChain:
    """REGRESSION (cf55dd5): an unrun bond stage must never look complete.

    The bug is the whole chain, not the literal seed value:
      1. an entry fails before the bond stage,
      2. its manifest row is later carried forward by ``--resume --no-bonds``,
      3. a subsequent bond-enabled ``--resume`` reads that row.
    With ``0`` seeded, step 3 judged the bond stage complete and skipped the
    entry permanently while the bond CSVs held no rows for it.
    """

    @staticmethod
    def _write_rows(path, rows):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=main.MANIFEST_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_failed_bond_enabled_entry_is_not_marked_bond_complete(
            self, tmp_path):
        """Step 1: a pre-bond failure writes blank counts, so it is retried."""
        result = _result("109m", status="error", retryable=True,
                         error="density stage failed")
        row = main._manifest_row(result, False, True, {}, {})
        manifest = tmp_path / "manifest.csv"
        self._write_rows(manifest, [row])

        # Observable consequence first: the entry is still work to do.
        assert main.load_done(str(manifest), bonds_required=True) == set()
        assert main.load_done(str(manifest), bonds_required=False) == set()
        # ...because the row records "the bond stage did not run", not "zero".
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_resume_no_bonds_recovery_does_not_fake_a_completed_bond_stage(
            self, tmp_path):
        """The full chain: fail, resume --no-bonds to ok, then resume bonds.

        The entry must still be scheduled by the bond-enabled resume, because
        no bond analysis has ever run for it.
        """
        manifest = tmp_path / "manifest.csv"

        # Run 1: bonds enabled, but the entry fails before the bond stage.
        failed = main._manifest_row(
            _result("109m", status="error", retryable=True,
                    error="edstats failed"),
            resume=False, bonds_enabled=True,
            prior_bond_counts={}, prior_candidate_counts={})
        self._write_rows(manifest, [failed])

        # Run 2: --resume --no-bonds. The entry is scheduled (not done) and
        # this time the density stage succeeds.
        assert main.load_done(str(manifest), bonds_required=False) == set()
        prior_bonds = main._manifest_values_by_id(str(manifest), "n_bonds")
        prior_candidates = main._manifest_values_by_id(
            str(manifest), "n_candidates")

        recovered = main._manifest_row(
            _result("109m", status="ok", retryable=False, n=2),
            resume=True, bonds_enabled=False,
            prior_bond_counts=prior_bonds,
            prior_candidate_counts=prior_candidates)
        assert recovered["status"] == "ok"
        assert recovered["n_metals"] == 2
        self._write_rows(manifest, [recovered])

        # Run 3: --resume with bonds. status is ok and the density stage is
        # done, but the bond stage has never run, so the entry must still be
        # scheduled rather than skipped forever with empty bond CSVs.
        assert main.load_done(str(manifest), bonds_required=False) == {"109m"}
        assert main.load_done(str(manifest), bonds_required=True) == set()

        # Run 3 completes the bond stage with a genuine measured zero; only
        # now may a later bond-enabled resume skip the entry.
        completed = main._manifest_row(
            _result("109m", status="ok", retryable=False, n=2,
                    n_bonds=0, n_candidates=0),
            resume=True, bonds_enabled=True,
            prior_bond_counts={}, prior_candidate_counts={})
        self._write_rows(manifest, [completed])
        assert main.load_done(str(manifest), bonds_required=True) == {"109m"}


# --------------------------------------------------------------------------- #
# _resume_replacement_succeeded
# --------------------------------------------------------------------------- #
class TestResumeReplacementSucceeded:
    """Only a terminal retry may replace the rows it is retrying."""

    @pytest.mark.parametrize("status,retryable,expected", [
        ("ok", False, True),
        ("ok", True, True),
        ("OK", True, True),
        (" ok ", False, True),
        ("partial", False, True),
        ("partial", True, False),
        ("error", False, False),
        ("error", True, False),
        ("skip", False, False),
        ("", True, False),
    ])
    def test_terminality(self, status, retryable, expected):
        """A retryable or failed retry leaves the previous rows in place."""
        result = {"status": status, "retryable": retryable}
        assert main._resume_replacement_succeeded(result) is expected

    def test_missing_retryable_defaults_to_retryable(self):
        """An unspecified partial is assumed unfinished, so it cannot replace."""
        assert main._resume_replacement_succeeded({"status": "partial"}) is False


# --------------------------------------------------------------------------- #
# _ResumeStaging
# --------------------------------------------------------------------------- #
class TestResumeStaging:
    """Staged retries replace rows only on a completed, terminal batch."""

    TARGET_NAMES = ("manifest.csv", "metal_stats_all.csv",
                    "metal_bonds_all.csv", "metal_candidates_all.csv")

    def _outputs(self, output_dir, confidence=False):
        """Create the four (or five) output CSVs with two entries' rows."""
        names: List[str] = list(self.TARGET_NAMES)
        if confidence:
            names.append("confidence_inputs_all.csv")
        paths = []
        for name in names:
            path = output_dir / name
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value"])
                writer.writerow(["109m", f"old-109m-{name}"])
                writer.writerow(["1cll", f"old-1cll-{name}"])
            paths.append(str(path))
        return tuple(paths)

    def _stage(self, staging, names_and_rows):
        """Write staged rows for the given staged-file indices."""
        for index, rows in names_and_rows.items():
            with open(staging.staged[index], "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value"])
                for row in rows:
                    writer.writerow(row)

    def test_staged_paths_mirror_the_targets_inside_the_output_dir(
            self, tmp_path):
        """Staging is a sibling temp dir so a merge is a same-filesystem move."""
        targets = self._outputs(tmp_path)
        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            assert os.path.isdir(staging.dir)
            assert os.path.dirname(staging.dir) == str(tmp_path)
            assert [os.path.basename(p) for p in staging.staged] == \
                [os.path.basename(p) for p in targets]
            assert all(os.path.dirname(p) == staging.dir
                       for p in staging.staged)
            assert staging.replacement_ids == set()
            # Nothing has touched the real outputs yet.
            for target in targets:
                assert _read_csv(target)[1][1].startswith("old-109m")
        finally:
            staging.discard()

    def test_commit_without_replacement_ids_leaves_outputs_byte_identical(
            self, tmp_path):
        """A batch in which no retry succeeded must change nothing."""
        targets = self._outputs(tmp_path)
        before = {p: open(p, "rb").read() for p in targets}
        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {0: [["109m", "new"]]})
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        assert {p: open(p, "rb").read() for p in targets} == before

    def test_commit_replaces_only_the_retried_ids(self, tmp_path):
        """Rows for untouched entries are copied through verbatim."""
        targets = self._outputs(tmp_path)
        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {
                index: [["109m", f"new-109m-{name}"]]
                for index, name in enumerate(self.TARGET_NAMES)
            })
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        for target, name in zip(targets, self.TARGET_NAMES):
            rows = _read_csv(target)
            assert rows[0] == ["pdbID", "value"]
            values = {row[0]: row[1] for row in rows[1:]}
            assert values["109m"] == f"new-109m-{name}"
            assert values["1cll"] == f"old-1cll-{name}"
            assert len(rows) == 3

    def test_commit_drops_stale_rows_when_the_retry_produced_none(
            self, tmp_path):
        """A retry that yields no rows must not leave the old ones behind."""
        targets = self._outputs(tmp_path)
        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {index: [] for index in range(4)})
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        for target in targets:
            values = {row[0] for row in _read_csv(target)[1:]}
            assert values == {"1cll"}

    def test_commit_with_bonds_disabled_leaves_bond_outputs_untouched(
            self, tmp_path):
        """--resume --no-bonds must preserve existing bond and candidate rows."""
        targets = self._outputs(tmp_path)
        bond_paths = targets[2:4]
        before = {p: open(p, "rb").read() for p in bond_paths}
        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {
                0: [["109m", "new-manifest"]],
                1: [["109m", "new-stats"]],
                2: [["109m", "SHOULD-NOT-APPEAR"]],
                3: [["109m", "SHOULD-NOT-APPEAR"]],
            })
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=False)
        finally:
            staging.discard()
        assert {p: open(p, "rb").read() for p in bond_paths} == before
        manifest_values = {row[0]: row[1]
                           for row in _read_csv(targets[0])[1:]}
        assert manifest_values == {
            "109m": "new-manifest",
            "1cll": "old-1cll-manifest.csv",
        }

    def test_commit_replaces_confidence_rows_only_when_enabled(self, tmp_path):
        """The optional fifth output participates only in confidence runs."""
        targets = self._outputs(tmp_path, confidence=True)
        confidence_path = targets[4]
        before = open(confidence_path, "rb").read()

        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {index: [["109m", "new"]] for index in range(5)})
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=False)
        finally:
            staging.discard()
        assert open(confidence_path, "rb").read() == before

        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {index: [["109m", "new"]] for index in range(5)})
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=True)
        finally:
            staging.discard()
        values = {row[0]: row[1] for row in _read_csv(confidence_path)[1:]}
        assert values == {"109m": "new", "1cll": (
            "old-1cll-confidence_inputs_all.csv")}

    def test_discard_leaves_previous_rows_intact(self, tmp_path):
        """An interrupted retry batch must not damage the existing outputs."""
        targets = self._outputs(tmp_path)
        before = {p: open(p, "rb").read() for p in targets}
        staging = main._ResumeStaging(str(tmp_path), targets)
        self._stage(staging, {index: [["109m", "new"]] for index in range(4)})
        staging.replacement_ids.add("109m")
        staging.discard()
        assert not os.path.exists(staging.dir)
        assert {p: open(p, "rb").read() for p in targets} == before

    def test_discard_is_idempotent(self, tmp_path):
        """Cleanup runs in a finally block and may be reached twice."""
        targets = self._outputs(tmp_path)
        staging = main._ResumeStaging(str(tmp_path), targets)
        staging.discard()
        staging.discard()
        assert not os.path.exists(staging.dir)

    def test_replacement_ids_are_matched_case_insensitively(self, tmp_path):
        """Manifest IDs and worker IDs may differ in case; the merge must not."""
        targets = self._outputs(tmp_path)
        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {index: [["109M", "new"]]
                                  for index in range(4)})
            staging.replacement_ids.add("109M")
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        values = {row[0].lower(): row[1] for row in _read_csv(targets[0])[1:]}
        assert values["109m"] == "new"
        assert len(values) == 2

    def test_a_staged_schema_mismatch_aborts_without_touching_the_target(
            self, tmp_path):
        """A header disagreement must fail loudly, not silently misalign rows."""
        targets = self._outputs(tmp_path)
        before = open(targets[0], "rb").read()
        staging = main._ResumeStaging(str(tmp_path), targets)
        try:
            with open(staging.staged[0], "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value", "extra"])
                writer.writerow(["109m", "new", "x"])
            self._stage(staging, {1: [["109m", "new"]], 2: [], 3: []})
            staging.replacement_ids.add("109m")
            with pytest.raises(ValueError):
                staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        assert open(targets[0], "rb").read() == before
        # No temporary merge file was left behind next to the target.
        leftovers = [name for name in os.listdir(tmp_path)
                     if name.startswith(".manifest.csv.")]
        assert leftovers == []


# --------------------------------------------------------------------------- #
# _OutputWriters
# --------------------------------------------------------------------------- #
class TestOutputWriters:
    """The streamed CSVs: headers on creation, running counts, schema guards."""

    @staticmethod
    def _handles(tmp_path, bonds=True, candidates=True, confidence=None):
        manifest = open(tmp_path / "manifest.csv", "w", newline="")
        stats = open(tmp_path / "stats.csv", "w", newline="")
        bonds_fh = (open(tmp_path / "bonds.csv", "w", newline="")
                    if bonds else None)
        candidates_fh = (open(tmp_path / "candidates.csv", "w", newline="")
                         if candidates else None)
        confidence_fh = (open(tmp_path / "confidence.csv", "w", newline="")
                         if confidence else None)
        return manifest, stats, bonds_fh, candidates_fh, confidence_fh

    @staticmethod
    def _close(handles):
        for handle in handles:
            if handle is not None:
                handle.close()

    def test_headers_survive_a_run_that_produced_no_rows(self, tmp_path):
        """README: the CSVs keep their headers when nothing was found."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        writers.write_stats_rows([])
        writers.write_bond_rows([])
        writers.write_candidate_rows([])
        self._close(handles)

        assert _read_csv(tmp_path / "manifest.csv") == [main.MANIFEST_COLUMNS]
        assert _read_csv(tmp_path / "stats.csv") == [list(main.STATS_COLUMNS)]
        assert _read_csv(tmp_path / "bonds.csv") == [list(main.BOND_COLUMNS)]
        assert _read_csv(tmp_path / "candidates.csv") == [
            list(main.CANDIDATE_COLUMNS)]
        assert (writers.n_rows, writers.n_bonds, writers.n_candidates) == (
            0, 0, 0)

    def test_running_counts_track_the_rows_actually_written(self, tmp_path):
        """The end-of-run totals come from these counters, not from re-reading."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        stats_rows = [{"pdbID": "109m", "category": "metal",
                       "fields": ["x"] * (len(main.STATS_COLUMNS) - 2)}
                      for _ in range(3)]
        bond_rows = [dict.fromkeys(main.BOND_COLUMNS, "") for _ in range(4)]
        candidate_rows = [dict.fromkeys(main.CANDIDATE_COLUMNS, "")
                          for _ in range(2)]
        writers.write_stats_rows(stats_rows)
        writers.write_bond_rows(bond_rows)
        writers.write_candidate_rows(candidate_rows)
        writers.write_stats_rows(stats_rows)
        self._close(handles)

        assert writers.n_rows == 6
        assert writers.n_bonds == 4
        assert writers.n_candidates == 2
        assert len(_read_csv(tmp_path / "stats.csv")) == 7   # header + 6
        assert len(_read_csv(tmp_path / "bonds.csv")) == 5
        assert len(_read_csv(tmp_path / "candidates.csv")) == 3

    def test_stats_rows_are_projected_onto_the_fixed_schema(self, tmp_path):
        """id and category lead each row; the EDSTATS block follows verbatim."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        fields = [str(i) for i in range(len(main.STATS_COLUMNS) - 2)]
        writers.write_stats_rows(
            [{"pdbID": "109m", "category": "cofactor", "fields": fields}])
        self._close(handles)
        rows = _read_csv(tmp_path / "stats.csv")
        assert rows[1] == ["109m", "cofactor"] + fields
        assert len(rows[1]) == len(main.STATS_COLUMNS)

    def test_bond_rows_are_written_in_schema_order(self, tmp_path):
        """Columns are positional, so the projection order is load-bearing."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        row = {column: f"v-{column}" for column in main.BOND_COLUMNS}
        # Feed it in a deliberately different key order.
        shuffled = {key: row[key] for key in reversed(list(row))}
        writers.write_bond_rows([shuffled])
        self._close(handles)
        written = _read_csv(tmp_path / "bonds.csv")[1]
        assert written == [f"v-{column}" for column in main.BOND_COLUMNS]

    def test_disabled_bond_outputs_are_a_no_op_not_a_crash(self, tmp_path):
        """--no-bonds passes None handles; writes must be silently skipped."""
        handles = self._handles(tmp_path, bonds=False, candidates=False)
        writers = main._OutputWriters(*handles)
        writers.write_bond_rows([dict.fromkeys(main.BOND_COLUMNS, "")])
        writers.write_candidate_rows([dict.fromkeys(main.CANDIDATE_COLUMNS, "")])
        self._close(handles)
        assert writers.n_bonds == 0
        assert writers.n_candidates == 0
        assert not (tmp_path / "bonds.csv").exists()
        assert not (tmp_path / "candidates.csv").exists()

    @pytest.mark.parametrize("mutate,columns_name", [
        ("drop", "BOND_COLUMNS"),
        ("add", "BOND_COLUMNS"),
    ])
    def test_bond_row_schema_drift_fails_loudly(self, tmp_path, mutate,
                                                columns_name):
        """A silently dropped or ignored column would corrupt every later row."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        row = dict.fromkeys(getattr(main, columns_name), "")
        if mutate == "drop":
            row.pop(next(iter(row)))
        else:
            row["unexpected_column"] = ""
        try:
            with pytest.raises(RuntimeError) as excinfo:
                writers.write_bond_rows([row])
        finally:
            self._close(handles)
        assert "metal_bonds_all.csv" in str(excinfo.value)
        assert writers.n_bonds == 0

    def test_candidate_row_schema_drift_fails_loudly(self, tmp_path):
        """Same guard on the candidate stream, named for its own file."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        row = dict.fromkeys(main.CANDIDATE_COLUMNS, "")
        row["bogus"] = ""
        try:
            with pytest.raises(RuntimeError) as excinfo:
                writers.write_candidate_rows([row])
        finally:
            self._close(handles)
        assert "metal_candidates_all.csv" in str(excinfo.value)

    def test_confidence_output_requires_its_columns(self, tmp_path):
        """A confidence stream without a schema cannot be written safely."""
        handles = self._handles(tmp_path, confidence=True)
        try:
            with pytest.raises(ValueError):
                main._OutputWriters(*handles[:4], confidence_fh=handles[4],
                                    confidence_columns=None)
        finally:
            self._close(handles)

    def test_confidence_header_and_counts(self, tmp_path):
        """The optional fifth stream behaves like the others."""
        columns = list(main.CONFIDENCE_INPUT_COLUMNS)
        handles = self._handles(tmp_path, confidence=True)
        writers = main._OutputWriters(*handles[:4], confidence_fh=handles[4],
                                      confidence_columns=columns)
        writers.write_confidence_rows([])
        assert writers.n_confidence == 0
        writers.write_confidence_rows(
            [dict.fromkeys(columns, ""), dict.fromkeys(columns, "")])
        self._close(handles)
        rows = _read_csv(tmp_path / "confidence.csv")
        assert rows[0] == columns
        assert writers.n_confidence == 2
        assert len(rows) == 3

    def test_confidence_row_schema_mismatch_is_rejected(self, tmp_path):
        """A drifted confidence row must not be written under the old header."""
        columns = list(main.CONFIDENCE_INPUT_COLUMNS)
        handles = self._handles(tmp_path, confidence=True)
        writers = main._OutputWriters(*handles[:4], confidence_fh=handles[4],
                                      confidence_columns=columns)
        row = dict.fromkeys(columns, "")
        row.pop(columns[0])
        try:
            with pytest.raises(RuntimeError):
                writers.write_confidence_rows([row])
        finally:
            self._close(handles)
        assert writers.n_confidence == 0

    def test_manifest_rows_round_trip_through_the_real_projection(self,
                                                                  tmp_path):
        """A written manifest is readable by load_done without reinterpretation."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        writers.write_manifest_row(main._manifest_row(
            _result("109m", status="ok", retryable=False, n=1,
                    n_bonds=0, n_candidates=0, runtime=1.0),
            False, True, {}, {}))
        writers.write_manifest_row(main._manifest_row(
            _result("1cll", status="error", retryable=True), False, True,
            {}, {}))
        self._close(handles)
        path = str(tmp_path / "manifest.csv")
        assert main.load_done(path, bonds_required=True) == {"109m"}
        assert main._manifest_values_by_id(path, "n_bonds") == {
            "109m": "0", "1cll": ""}

    def test_each_stream_is_flushed_before_the_manifest_marker(self, tmp_path):
        """An interrupted batch must retain the rows of completed entries."""
        handles = self._handles(tmp_path)
        writers = main._OutputWriters(*handles)
        writers.write_stats_rows([{"pdbID": "109m", "category": "metal",
                                   "fields": ["x"] * (
                                       len(main.STATS_COLUMNS) - 2)}])
        writers.write_bond_rows([dict.fromkeys(main.BOND_COLUMNS, "")])
        writers.write_manifest_row(
            main._manifest_row(_result(status="ok"), False, True, {}, {}))
        # Without closing the handles, the data must already be on disk.
        assert len(_read_csv(tmp_path / "stats.csv")) == 2
        assert len(_read_csv(tmp_path / "bonds.csv")) == 2
        assert len(_read_csv(tmp_path / "manifest.csv")) == 2
        self._close(handles)


# --------------------------------------------------------------------------- #
# _cif_to_pdb
# --------------------------------------------------------------------------- #
class TestCifToPdb:
    """mmCIF -> analysis PDB conversion must not lose provenance."""

    @staticmethod
    def _standard_cif(tmp_path, name="in.cif"):
        return _write_cif(tmp_path / name, [
            _cif_atom(1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0),
                      "1.00", 1, "A"),
            _cif_atom(2, "C", "CA", "GLY", "A", 1, 1, (21.5, 20.0, 20.0),
                      "?", 1, "A"),
            _cif_atom(3, "C", "C", "GLY", "A", 1, 1, (22.0, 21.4, 20.0),
                      ".", 1, "A"),
            _cif_atom(4, "ZN", "ZN", "ZN", "B", 2, ".", (0.0, 0.0, 0.0),
                      "0.75", 1, "B", group="HETATM"),
            _cif_atom(5, "FE", "FE1", "SF4X", "C", 3, ".", (5.0, 0.0, 0.0),
                      "1.00", 2, "B", group="HETATM"),
        ])

    def test_missing_occupancies_are_blank_not_one(self, tmp_path):
        """README: '.' and '?' become blank PDB occupancy, never 1.00."""
        cif = self._standard_cif(tmp_path)
        out = main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        lines = _pdb_atom_lines(out)
        assert len(lines) == 5
        occupancies = [_occupancy_field(line) for line in lines]
        assert occupancies[1].strip() == ""      # '?'
        assert occupancies[2].strip() == ""      # '.'
        assert occupancies[0].strip() == "1.00"
        assert float(occupancies[3]) == 0.75
        assert float(occupancies[4]) == 1.00

    def test_blanking_occupancy_does_not_shift_the_other_columns(self,
                                                                 tmp_path):
        """Only columns 55-60 change; coordinates, B and element stay put."""
        cif = self._standard_cif(tmp_path)
        out = main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        blanked = _pdb_atom_lines(out)[1]
        assert blanked[30:38].strip() == "21.500"
        assert blanked[60:66].strip() == "20.00"
        assert _element_field(blanked) == "C"
        assert blanked[17:20].strip() == "GLY"
        assert len(blanked.rstrip("\n")) >= 78

    def test_absent_occupancy_column_blanks_every_atom(self, tmp_path):
        """An mmCIF with no occupancy loop item asserts nothing about it."""
        cif = _write_cif(tmp_path / "noocc.cif", [
            _cif_atom(1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0),
                      None, 1, "A", with_occupancy=False),
            _cif_atom(2, "ZN", "ZN", "ZN", "B", 2, ".", (0.0, 0.0, 0.0),
                      None, 1, "B", group="HETATM", with_occupancy=False),
        ], header=_CIF_HEADER_NO_OCCUPANCY)
        out = main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        for line in _pdb_atom_lines(out):
            assert _occupancy_field(line).strip() == ""

    def test_type_symbol_is_written_into_the_pdb_element_field(self, tmp_path):
        """Alchemy reads the element column, never guesses from atom names."""
        cif = self._standard_cif(tmp_path)
        out = main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        elements = [_element_field(line) for line in _pdb_atom_lines(out)]
        assert elements == ["N", "C", "C", "ZN", "FE"]

    def test_long_component_ids_are_truncated_but_recorded(self, tmp_path):
        """A >3-character CCD id cannot fit the legacy field, so map it."""
        cif = self._standard_cif(tmp_path)
        out = main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        with open(out) as handle:
            remarks = [line.split() for line in handle
                       if line.startswith(main.RESNAME_REMARK_PREFIX)]
        assert len(remarks) == 1
        fields = remarks[0]
        # REMARK 950 ALCHEMY RESNAME <model> <chain> <resnum> <written> <source>
        assert fields[4:9] == ["1", "B", "2", "SF4", "SF4X"]
        written_names = {line[17:20].strip() for line in _pdb_atom_lines(out)}
        assert "SF4" in written_names
        assert "SF4X" not in written_names

    def test_no_mapping_remark_when_every_name_fits(self, tmp_path):
        """Short component ids need no provenance record."""
        cif = _write_cif(tmp_path / "short.cif", [
            _cif_atom(1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0),
                      "1.00", 1, "A"),
            _cif_atom(2, "ZN", "ZN", "ZN", "B", 2, ".", (0.0, 0.0, 0.0),
                      "1.00", 1, "B", group="HETATM"),
        ])
        out = main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        with open(out) as handle:
            assert not any(line.startswith(main.RESNAME_REMARK_PREFIX)
                           for line in handle)

    def test_conversion_round_trips_through_load_structure(self, tmp_path):
        """The mapping is reversible: the analysis load restores the CCD id.

        This is the end of the provenance contract -- the truncated name is
        what EDSTATS sees, while Alchemy's own output keeps the mmCIF identity.
        """
        cif = self._standard_cif(tmp_path)
        out = main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        context = structure_analysis.load_structure("test", out)
        by_atom = {site.atom_name: site for site in context.source_atoms}

        assert by_atom["FE1"].residue_name == "SF4X"
        assert by_atom["FE1"].coordinate_residue_name == "SF4"
        assert by_atom["FE1"].element == "FE"
        assert by_atom["ZN"].residue_name == "ZN"

        # Occupancy missingness survived as missingness, not as 1.0.
        assert by_atom["CA"].occupancy_valid is False
        assert by_atom["C"].occupancy_valid is False
        assert by_atom["N"].occupancy == pytest.approx(1.0)
        assert by_atom["ZN"].occupancy == pytest.approx(0.75)

    def test_missing_input_file_raises_file_not_found(self, tmp_path):
        """A vanished mirror file must not be reported as a conversion bug."""
        with pytest.raises(FileNotFoundError):
            main._cif_to_pdb(str(tmp_path / "gone.cif"),
                             str(tmp_path / "out.pdb"))

    def test_duplicate_atom_site_id_is_rejected(self, tmp_path):
        """Serials key the occupancy restoration; duplicates make it ambiguous."""
        cif = _write_cif(tmp_path / "dup.cif", [
            _cif_atom(1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0),
                      "1.00", 1, "A"),
            _cif_atom(1, "ZN", "ZN", "ZN", "B", 2, ".", (0.0, 0.0, 0.0),
                      "?", 1, "B", group="HETATM"),
        ])
        with pytest.raises(ValueError, match="duplicate mmCIF atom_site id"):
            main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))

    def test_non_integer_atom_site_id_is_rejected(self, tmp_path):
        """gemmi exposes an integer serial; a non-integer id cannot be joined."""
        cif = _write_cif(tmp_path / "bad.cif", [
            _cif_atom("A1", "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0),
                      "1.00", 1, "A"),
        ])
        with pytest.raises(ValueError, match="not an integer"):
            main._cif_to_pdb(cif, str(tmp_path / "out.pdb"))

    def test_multiple_atom_site_blocks_are_rejected(self, tmp_path):
        """Ambiguous input must fail rather than silently convert one block."""
        atoms = [_cif_atom(1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0),
                           "1.00", 1, "A")]
        text = (_CIF_HEADER + "".join(line + "\n" for line in atoms) + "\n" +
                _CIF_HEADER.replace("data_TEST", "data_SECOND") +
                "".join(line + "\n" for line in atoms))
        path = tmp_path / "two.cif"
        path.write_text(text)
        with pytest.raises(ValueError, match="exactly one block"):
            main._cif_to_pdb(str(path), str(tmp_path / "out.pdb"))

    def test_cif_without_atom_records_is_rejected(self, tmp_path):
        """A metadata-only mmCIF is not a coordinate file."""
        path = tmp_path / "meta.cif"
        path.write_text("data_TEST\n_cell.length_a 60.0\n")
        with pytest.raises(ValueError, match="exactly one block"):
            main._cif_to_pdb(str(path), str(tmp_path / "out.pdb"))

    def test_creates_the_destination_directory(self, tmp_path):
        """Work directories are created lazily around the converted file."""
        cif = self._standard_cif(tmp_path)
        dst = tmp_path / "nested" / "deeper" / "out.pdb"
        out = main._cif_to_pdb(cif, str(dst))
        assert out == str(dst)
        assert os.path.isfile(out)


class TestResidueConversionRecords:
    """The identity assertions ``_cif_to_pdb`` enforces on gemmi's writer."""

    def test_identical_structures_produce_no_records(self):
        """No renaming means no provenance remark is needed."""
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(_BASE_RESIDUES)
        assert main._residue_conversion_records(source, converted) == []

    def test_a_renamed_residue_is_recorded_with_its_author_identity(self):
        """The record must locate the residue the way the PDB reader will."""
        source = _simple_structure([
            ("A", 1, "GLY", [("N", "N")]),
            ("B", 7, "SF4X", [("FE1", "FE")]),
        ])
        converted = _simple_structure([
            ("A", 1, "GLY", [("N", "N")]),
            ("B", 7, "SF4", [("FE1", "FE")]),
        ])
        assert main._residue_conversion_records(source, converted) == [
            (1, "B", "7", "SF4", "SF4X")]

    def test_reordering_is_rejected(self):
        """EDSTATS joins by position-independent identity only if order holds."""
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(list(reversed(_BASE_RESIDUES)))
        with pytest.raises(ValueError, match="changed residue ordering"):
            main._residue_conversion_records(source, converted)

    def test_changed_author_identifiers_are_rejected(self):
        """A renumbered or rechained residue would break every downstream join."""
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure([
            ("A", 1, "GLY", [("N", "N"), ("CA", "C")]),
            ("B", 99, "ZN", [("ZN", "ZN")]),
        ])
        with pytest.raises(ValueError, match="ordering|author identifiers"):
            main._residue_conversion_records(source, converted)

    def test_changed_atom_membership_is_rejected(self):
        """Dropping or renaming an atom silently would corrupt coordination."""
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure([
            ("A", 1, "GLY", [("N", "N")]),
            ("B", 2, "ZN", [("ZN", "ZN")]),
        ])
        with pytest.raises(ValueError, match="atom membership"):
            main._residue_conversion_records(source, converted)

    def test_changed_duplicate_multiplicity_is_rejected(self):
        """Two residues sharing an author id must stay two after conversion."""
        source = _simple_structure([
            ("A", 1, "GLY", [("N", "N")]),
            ("A", 1, "ALA", [("N", "N")]),
        ])
        converted = _simple_structure([
            ("A", 1, "GLY", [("N", "N")]),
        ])
        with pytest.raises(ValueError, match="ordering|multiplicity"):
            main._residue_conversion_records(source, converted)

    def test_the_index_keys_on_model_chain_and_author_resnum(self):
        """Residues are located by the identifiers EDSTATS also reports."""
        structure = _simple_structure(_BASE_RESIDUES)
        index, order = main._residue_index_by_author(structure, "mmCIF")
        assert order == [(0, "A", "1"), (0, "B", "2")]
        assert index[(0, "B", "2")] == [("ZN", (("ZN", "Zn"),))]

    def test_duplicate_author_ids_are_indexed_together_in_order(self):
        """Two residues with one author id must both survive the index."""
        structure = _simple_structure([
            ("A", 1, "GLY", [("N", "N")]),
            ("A", 1, "ALA", [("N", "N")]),
        ])
        index, order = main._residue_index_by_author(structure, "mmCIF")
        assert order == [(0, "A", "1"), (0, "A", "1")]
        assert [name for name, _ in index[(0, "A", "1")]] == ["GLY", "ALA"]


# --------------------------------------------------------------------------- #
# _first_model_pdb
# --------------------------------------------------------------------------- #
_MULTI_MODEL_PDB = """\
HEADER    TEST
NUMMDL    2
REMARK   3 SOMETHING
CRYST1   60.000   70.000   80.000  90.00  90.00  90.00 P 21 21 21
MODEL        1
ATOM      1  N   GLY A   1      20.000  20.000  20.000  1.00 20.00           N
ATOM      2  CA  GLY A   1      21.500  20.000  20.000  1.00 20.00           C
ENDMDL
MODEL        2
ATOM      1  N   GLY A   1      30.000  30.000  30.000  1.00 20.00           N
ATOM      2  CA  GLY A   1      31.500  30.000  30.000  1.00 20.00           C
ENDMDL
MASTER        0    0    0    0    0    0    0    0    2    1    0    0
END
"""


class TestFirstModelPdb:
    """Textual first-model extraction for EDSTATS."""

    def test_single_model_file_is_used_in_place(self, tmp_path):
        """A file with no MODEL wrapper needs no rewrite at all."""
        builder = helpers.simple_metal_site(
            "ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)])
        source = builder.write_pdb(tmp_path / "site.pdb")
        dst = tmp_path / "first.pdb"
        path, count = main._first_model_pdb(source, str(dst))
        assert path == source
        assert count == 1
        assert not dst.exists()

    def test_extracts_only_the_first_model(self, tmp_path):
        """Later models must not contribute a single coordinate record."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        path, count = main._first_model_pdb(str(source), str(dst))
        assert path == str(dst)
        assert count == 2

        text = dst.read_text()
        atom_lines = _pdb_atom_lines(dst)
        assert len(atom_lines) == 2
        assert [line[30:38].strip() for line in atom_lines] == [
            "20.000", "21.500"]
        assert "30.000" not in text

    def test_model_wrappers_are_removed(self, tmp_path):
        """EDSTATS emits a synthetic separator residue for any MODEL wrapper."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        main._first_model_pdb(str(source), str(dst))
        for line in dst.read_text().splitlines():
            assert line[:6].strip().upper() not in ("MODEL", "ENDMDL")
        assert gemmi.read_structure(str(dst)).__len__() == 1

    def test_nummdl_is_dropped_but_crystallographic_header_is_kept(self,
                                                                   tmp_path):
        """NUMMDL would lie about a one-model file; CRYST1 must survive."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        main._first_model_pdb(str(source), str(dst))
        lines = dst.read_text().splitlines()
        assert not any(line.startswith("NUMMDL") for line in lines)
        assert any(line.startswith("CRYST1") for line in lines)
        assert any(line.startswith("HEADER") for line in lines)
        assert any(line.startswith("REMARK   3") for line in lines)

    def test_trailing_records_after_the_first_model_are_dropped(self, tmp_path):
        """Bookkeeping records describe the full ensemble, not this extract."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        main._first_model_pdb(str(source), str(dst))
        lines = dst.read_text().splitlines()
        assert not any(line.startswith("MASTER") for line in lines)
        assert lines[-1] == "END"

    def test_the_extract_preserves_records_byte_for_byte(self, tmp_path):
        """Extraction is textual so occupancies and identifiers are untouched."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        main._first_model_pdb(str(source), str(dst))
        source_atom_lines = [line for line in _MULTI_MODEL_PDB.splitlines()
                             if line.startswith("ATOM")]
        assert [line.rstrip("\n") for line in _pdb_atom_lines(dst)] == \
            source_atom_lines[:2]

    def test_creates_the_destination_directory(self, tmp_path):
        """The analysis file may be the first thing written to a work dir."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "nested" / "first.pdb"
        path, _ = main._first_model_pdb(str(source), str(dst))
        assert os.path.isfile(path)

    def test_reported_model_count_is_the_source_ensemble_size(self, tmp_path):
        """The manifest's input_model_count describes the deposited file."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        _, count = main._first_model_pdb(str(source),
                                         str(tmp_path / "first.pdb"))
        assert count == 2
        assert len(gemmi.read_structure(str(source))) == count


# --------------------------------------------------------------------------- #
# Cleaning up after a run that did not finish
# --------------------------------------------------------------------------- #
class TestLeakedWorkDirectorySweep:
    """``_sweep_leaked_work_dirs`` clears scratch a dead run left behind."""

    def test_removes_per_entry_and_staging_directories(self, tmp_path):
        """Both scratch shapes are swept, with their contents.

        Regression: a per-entry ``.alchemy-<id>-XXXX`` directory is deleted
        only on the normal completion path, so an interrupted run left one
        behind holding that entry's maps -- tens of megabytes each -- and
        nothing, including ``--resume``, ever removed them.
        """
        entry = tmp_path / ".alchemy-109m-abcd"
        entry.mkdir()
        (entry / "2mFo-DFc.map").write_text("stale", encoding="utf-8")
        staging = tmp_path / ".alchemy-resume-wxyz"
        staging.mkdir()
        (staging / "manifest.csv").write_text("stale", encoding="utf-8")

        removed = main._sweep_leaked_work_dirs(str(tmp_path))

        assert removed == 2
        assert sorted(os.listdir(tmp_path)) == []

    def test_leaves_real_output_alone(self, tmp_path):
        """Only the dotted scratch prefixes are swept, never results.

        The sweep runs at startup against a directory that normally holds the
        four result CSVs and every previous run log, so a prefix match that was
        even slightly too broad would delete a completed run's output.
        """
        (tmp_path / "manifest.csv").write_text("keep", encoding="utf-8")
        (tmp_path / "alchemy_run_20260101.log").write_text(
            "keep", encoding="utf-8")
        (tmp_path / "109m").mkdir()          # a user directory named for an id
        (tmp_path / ".alchemyrc").write_text("keep", encoding="utf-8")

        assert main._sweep_leaked_work_dirs(str(tmp_path)) == 0
        assert sorted(os.listdir(tmp_path)) == [
            ".alchemyrc", "109m", "alchemy_run_20260101.log", "manifest.csv"]

    def test_missing_directory_is_not_an_error(self, tmp_path):
        """A sweep of a directory that does not exist yet is a no-op."""
        assert main._sweep_leaked_work_dirs(str(tmp_path / "absent")) == 0


# --------------------------------------------------------------------------- #
# An unusable --output-dir is a user error, not a crash
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions required")
def test_unwritable_output_dir_exits_cleanly_naming_the_path(
        tmp_path, monkeypatch, capsys):
    """A read-only destination exits like every other unusable input.

    Regression: ``os.makedirs(args.output_dir, exist_ok=True)`` was unguarded,
    so a read-only mount or someone else's directory escaped ``main()`` as a
    raw ``PermissionError`` traceback -- which reads as an Alchemy bug rather
    than a path the user can fix.

    CCP4 resolution is stubbed out because it runs before the directory is
    created; without that the test would fail for an unrelated reason on a
    machine with no CCP4 installed.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")

    monkeypatch.setattr(main, "resolve_ccp4_environment",
                        lambda args: (dict(os.environ), None))
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    output_dir = parent / "output"
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    try:
        with pytest.raises(SystemExit) as excinfo:
            main.main(["--id-file", str(id_file),
                       "--output-dir", str(output_dir),
                       "--pdb-redo-root", str(tmp_path / "absent-mirror"),
                       "--pdb-redo-cache", str(tmp_path / "cache")])
    finally:
        parent.chmod(0o700)

    assert excinfo.value.code not in (0, None)
    message = f"{excinfo.value}\n{capsys.readouterr().err}"
    assert str(output_dir) in message, message
    assert "Traceback" not in message


@pytest.mark.skipif(os.name != "posix", reason="uses a POSIX-only stub env")
def test_a_run_sweeps_leaked_scratch_before_processing(tmp_path, monkeypatch):
    """The sweep is wired into the driver, not merely available to it.

    Exercises ``main.main`` rather than ``_sweep_leaked_work_dirs`` directly:
    the defect was that nothing ever *called* a sweep, so a test of the helper
    alone would stay green with the call site deleted.

    The run itself fails -- there is no mirror and no network -- which is the
    point: sweeping happens at startup, so even a run that goes on to fail must
    leave the directory clean.
    """
    monkeypatch.setattr(main, "resolve_ccp4_environment",
                        lambda args: (dict(os.environ), None))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    leaked = output_dir / ".alchemy-109m-leaked"
    leaked.mkdir()
    (leaked / "2mFo-DFc.map").write_text("stale map bytes", encoding="utf-8")
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    main.main(["--id-file", str(id_file), "--output-dir", str(output_dir),
               "--pdb-redo-root", str(tmp_path / "absent-mirror"),
               "--pdb-redo-cache", str(tmp_path / "cache")])

    leftovers = sorted(name for name in os.listdir(output_dir)
                       if name.startswith(".alchemy-"))
    assert leftovers == [], f"the run left scratch behind: {leftovers}"
