"""Batch-driver data management: argument validation, resume bookkeeping, the
manifest projection, the staged-retry and streamed-output machinery, and the
two coordinate-preparation converters.

Worker-death recovery and end-to-end pipeline runs are covered elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol
from collections.abc import Callable
from collections.abc import Mapping
from typing import NamedTuple
from collections.abc import Sequence
from typing import cast

import gemmi
import pytest

import helpers
from coordination import schema as coordination_schema
import coordinate_conversion as conversion
from coordination import declared_connections
import density_analysis as density
import main
import metal_identification
import structure_analysis
import worker
import worker_contracts
from codes import EntryStatus
from driver.progress import ProgressReporter
from driver import runlog
from driver import resources
from driver.runlog import RunLog
from driver.writers import (
    MANIFEST_COLUMNS,
    MANIFEST_FIELDS,
    STATS_COLUMNS,
    manifest_row,
    OutputWriters,
)
from structure_analysis import RESIDUE_REMARK_PREFIX, RESNAME_REMARK_PREFIX
import cli
from driver import pool as driver_pool
from driver import resume
from driver import pool
from driver import output_lock
import confidence_score
from output_rows import MetalStatsRow
from run_config import RunConfig


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


def _read_resolution_stub(
    entry_dir: str, mtz_path: str, data_json_path: str | None = None
) -> float:
    del entry_dir, mtz_path, data_json_path
    return 2.0


def _read_map_column_resolution_stub(mtz_path: str) -> tuple[float, float]:
    del mtz_path
    return 20.0, 2.0


def _resolved_ccp4_environment(
    _args: RunConfig,
) -> tuple[dict[str, str], None]:
    return dict(os.environ), None


def _empty_metal_statistics(
    *_args: Any, **_kwargs: Any
) -> tuple[list[dict[str, Any]], list[str]]:
    return [], []


def _cfg(**overrides: Any) -> worker_contracts.WorkerConfig:
    """A complete ``WorkerConfig`` with test placeholders."""
    fields: dict[str, Any] = {
        "root": "/nonexistent/root",
        "mirror_root": "/nonexistent/mirror",
        "cache_root": None,
        "env": {},
        "output_dir": "/nonexistent/output",
        "cofactors": frozenset(),
        "keep": False,
        "bonds": True,
        "density_map_scope": "model-envelope",
        "ccp4_timeout_s": density.CCP4_TOOL_TIMEOUT_S,
        "log_level": logging.INFO,
        "allow_download": False,
        "manual_inputs": None,
        "alchemy_commit": "abc123def456",
        "gemmi_version": "0.7.5",
        "ccp4_version": "9.0",
        "reference_data_id": "0123456789ab",
    }
    fields.update(overrides)
    return worker_contracts.WorkerConfig(**fields)


CFG = _cfg()


def _run_config(**overrides: Any) -> RunConfig:
    return replace(cli.parse_args([]), **overrides)


# Columns ``manifest_row`` computes; every other column it copies.
DERIVED_MANIFEST_COLUMNS = frozenset(
    {
        "n_metals",
        "n_bonds",
        "n_candidates",
        "runtime_s",
        "reason_codes",
        "warning_codes",
    }
)


def _result(pdb_id: str = "109m", **overrides: Any) -> worker_contracts.EntryResult:
    """A worker result skeleton plus overrides, as the driver would see it."""
    result = worker.initial_result(pdb_id, CFG, None)
    for name, value in overrides.items():
        if name == "status":
            value = EntryStatus(value)
        setattr(result, name, value)
    return result


def _manual_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    density_stage: Callable[..., Any],
    *,
    structure_builder: helpers.StructureBuilder | None = None,
    pdb_transform: Callable[[Path], None] | None = None,
    bonds: bool = False,
    **cfg_overrides: Any,
) -> worker_contracts.EntryResult:
    """Run one manual-input entry through ``worker.process``.

    Only the three MTZ-dependent readers are stubbed; structure loading, result
    assembly and status computation all run for real.
    """
    builder = structure_builder or helpers.StructureBuilder()
    if structure_builder is None:
        builder.add_metal("ZN", 1, chain="A", pos=(0.0, 0.0, 0.0))
        builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.05, 0.0, 0.0)})
    pdb_path = tmp_path / "entry.pdb"
    builder.write_pdb(str(pdb_path))
    if pdb_transform is not None:
        pdb_transform(pdb_path)
    mtz_path = tmp_path / "entry.mtz"
    mtz_path.write_bytes(b"unused: the readers below are stubbed")

    monkeypatch.setattr(worker, "read_resolution", _read_resolution_stub)
    monkeypatch.setattr(
        worker, "read_map_column_resolution", _read_map_column_resolution_stub
    )
    monkeypatch.setattr(worker, "run_density_analysis", density_stage)

    cfg = _cfg(
        root=str(tmp_path),
        mirror_root=str(tmp_path),
        cache_root=str(tmp_path),
        output_dir=str(tmp_path),
        bonds=bonds,
        density_map_scope="full",
        ccp4_timeout_s=900,
        manual_inputs={
            "pdb_file": str(pdb_path),
            "mtz_file": str(mtz_path),
            "cif_file": None,
            "data_json": None,
        },
        alchemy_commit="test",
        gemmi_version="test",
        ccp4_version="test",
        **cfg_overrides,
    )
    # monkeypatch restores ``worker_config``; ``initialize_worker`` has no
    # teardown counterpart.
    monkeypatch.setattr(worker, "worker_config", cfg)
    return worker.process("1abc")


@pytest.mark.parametrize(
    "exception,expected_code",
    [
        (ValueError("no FWT column"), "deterministic_processing_error"),
        (KeyError("NR"), "deterministic_processing_error"),
        (TypeError("bad shape"), "deterministic_processing_error"),
        (OSError("disk full"), "unexpected_processing_error"),
        (MemoryError(), "unexpected_processing_error"),
        (RuntimeError("fft failed"), "unexpected_processing_error"),
    ],
)
def test_an_unanticipated_failure_reports_whether_it_will_recur(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    expected_code: str,
) -> None:
    """The exception type separates a permanent failure from a possible fluke.

    A parse, lookup or type error describes the data or Alchemy's code and will
    recur while both are unchanged. An OSError or a CCP4 RuntimeError may
    describe the machine instead. The distinction is for triage: both stay
    retryable, because nothing here can establish that a later run reads the
    same bytes.
    """

    def failing_stage(*args: Any, **kwargs: Any) -> None:
        raise exception

    result = _manual_entry(tmp_path, monkeypatch, failing_stage)

    assert result.status == "error"
    assert result.reason_codes == [expected_code]
    assert result.retryable is True
    assert type(exception).__name__ in result.error


def _real_stats_density_stage(
    pdb_id: str, mtz: str, pdb: str, work_dir: str, *args: Any, **kwargs: Any
) -> density.DensityResult:
    """Write a synthetic ``stats.out`` covering every residue of the model."""
    context = structure_analysis.load_structure(pdb_id, pdb)
    stats_out = helpers.write_edstats_for_structure(
        os.path.join(work_dir, "stats.out"), context
    )
    return density.DensityResult(
        stats_out=stats_out,
        rszd="/nonexistent/rszd.pdb",
        fo_map="/nonexistent/fo.map",
        df_map="/nonexistent/df.map",
        mtz_for_maps="/nonexistent/entry.mtz",
        mtzfix_log="/nonexistent/mtzfix.log",
        mtzfix_applied=False,
        timings={"edstats_s": 1.0},
        twin_coefficient_normalization_applied=False,
        twin_coefficient_normalization=None,
        density_map_scope_requested="model-envelope",
        density_map_scope_used="model-envelope",
        full_map_bytes=8192,
        edstats_map_bytes=2048,
    )


@pytest.mark.parametrize("bonds", [True, False], ids=["bonds", "no-bonds"])
def test_a_metal_outside_the_catalog_is_reported_not_half_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bonds: bool
) -> None:
    """The two stages select metals by different rules, and must say when they differ.

    Coordinate analysis takes every metal atom by element; statistics reports
    a residue only when it is a catalog cofactor or a single-atom ion. A metal
    inside a multi-atom residue absent from the bundled catalog therefore has
    no density evidence. The mismatch must be reported whether or not bond
    analysis was requested.
    """
    builder = helpers.StructureBuilder()
    builder.add_hetero_residue(
        "ZZZ",
        1,
        [
            helpers.AtomSpec(name="ZN", element="ZN", pos=(0.0, 0.0, 0.0)),
            helpers.AtomSpec(name="C1", element="C", pos=(1.5, 0.0, 0.0)),
        ],
        chain="B",
    )
    builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.03, 0.0, 0.0)})

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=bonds,
    )

    assert "metal_site_without_density" in result.reason_codes
    assert result.status == "partial"
    assert result.retryable is False
    # The manifest counts the coordinate-model site while metal_stats_all.csv
    # carries no row for it, so a join on statistics would silently lose it.
    # When requested, bond analysis still retains the site's contact evidence.
    assert result.n_metals == 1
    if bonds:
        assert result.n_bonds, "bond analysis should retain the measured contact"
    else:
        assert result.n_bonds == 0
    assert result.rows == []


def test_a_zero_occupancy_metal_is_excluded_but_not_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``no_metals`` must not read as an authoritative negative about this file.

    A metal modeled at zero occupancy is not evidence for a site, so it is
    rightly excluded from selection. The exclusion used to be silent, leaving
    an entry whose only metal is modeled absent reporting no metals at all --
    and the generic ``zero_occupancy_atoms`` warning cannot say which atom.
    """
    builder = helpers.StructureBuilder()
    builder.add_hetero_residue(
        "ZN",
        1,
        [helpers.AtomSpec(name="ZN", element="ZN", pos=(0.0, 0.0, 0.0), occupancy=0.0)],
        chain="B",
    )

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=True,
    )

    assert result.n_metals == 0
    assert result.no_metals is True
    assert "zero_occupancy_metal_excluded" in result.warning_codes


def test_a_positive_occupancy_metal_raises_no_exclusion_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warning marks a real exclusion, not merely the presence of a metal."""
    builder = helpers.StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.03, 0.0, 0.0)})

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=True,
    )

    assert result.n_metals == 1
    assert "zero_occupancy_metal_excluded" not in result.warning_codes


def test_a_catalogued_cofactor_metal_raises_no_such_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-atom ion is reported by both stages, so nothing is flagged."""
    builder = helpers.StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_amino_acid("HIS", 2, chain="A", positions={"NE2": (2.03, 0.0, 0.0)})

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        bonds=True,
    )

    assert "metal_site_without_density" not in result.reason_codes
    assert result.n_metals == 1


def test_non_finite_selected_metal_is_partial_without_bond_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coordinate validation is not disabled together with bond output."""
    builder = helpers.StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))

    def make_metal_x_nan(path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        rewritten: list[str] = []
        for line in lines:
            if line[:6].strip() == "HETATM" and line[76:78].strip() == "ZN":
                line = line[:30] + "     nan" + line[38:]
            rewritten.append(line)
        path.write_text("".join(rewritten), encoding="utf-8")

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        _real_stats_density_stage,
        structure_builder=builder,
        pdb_transform=make_metal_x_nan,
        bonds=False,
    )

    assert result.status == "partial"
    assert result.retryable is False
    assert result.n_metals == 1
    assert "non_finite_coordinates" in result.warning_codes
    assert "non_finite_metal_coordinates" in result.reason_codes


def test_an_unanticipated_failure_logs_its_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The manifest keeps one line; the debug log must keep the stack.

    Without this the only way to locate an unanticipated failure was to rerun
    the entry by hand under ``--keep-intermediates``.
    """

    def failing_stage(*args: Any, **kwargs: Any) -> None:
        raise ValueError("no FWT column")

    with caplog.at_level(logging.DEBUG, logger="alchemy.worker"):
        result = _manual_entry(tmp_path, monkeypatch, failing_stage)

    assert result.status == "error"
    records = [r for r in caplog.records if r.exc_info and "1abc" in r.getMessage()]
    assert records, "the failing entry logged no traceback"
    # The comprehension already selected on ``exc_info``; the cast only carries
    # that through to the subscript.
    assert cast(tuple[Any, ...], records[0].exc_info)[0] is ValueError


def _write_manifest(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str] | None = None,
) -> str:
    """Write a manifest CSV with the real schema and the given partial rows."""
    columns = list(columns if columns is not None else MANIFEST_COLUMNS)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return str(path)


def _manifest_ids(path: str | Path, **kwargs: Any) -> set[str]:
    return resume.load_done(str(path), **kwargs)


def _read_csv(path: str | Path) -> list[list[str]]:
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


def test_bond_stage_failure_invalidates_confidence_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed geometry stage is not legitimate density-only evidence."""
    result = _result()
    inputs = worker.EntryInputs(
        work_dir="/nonexistent",
        mtz="/nonexistent/entry.mtz",
        pdb="/nonexistent/entry.pdb",
        data_json=None,
        data_reshi=2.0,
        map_reslo=50.0,
        map_reshi=2.0,
        pdb_redo_is_twin=False,
        source_coordinate_path="/nonexistent/entry.cif",
    )
    # A real ``StructureContext`` only comes from parsing a file on disk, which
    # this test has no use for: the bond stage fails before reading anything
    # but the warning codes it carries forward.
    structure = cast(
        structure_analysis.StructureContext, SimpleNamespace(warning_codes=[])
    )

    def fail_bond_analysis(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("geometry unavailable")

    monkeypatch.setattr(worker, "run_bond_analysis", fail_bond_analysis)

    analysis = worker.run_bond_stage(
        result, _cfg(bonds=True), inputs, structure, [], []
    )

    assert (analysis.bond_rows, analysis.candidate_rows, analysis.site_summaries) == (
        [],
        [],
        {},
    )
    assert result.reason_codes == ["bond_stage_failure"]
    assert result.confidence_inputs_missing_reason == "bond_stage_failure"
    assert result.retryable is True


class TestNoRecognizedMetalOutcome:
    @staticmethod
    def _density_must_not_run(*args: Any, **kwargs: Any) -> None:
        pytest.fail("density analysis ran without a recognized metal site")

    @staticmethod
    def _blank_metal_element(path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        rewritten: list[str] = []
        for line in lines:
            if line.startswith("HETATM") and line[12:16].strip() == "ZN":
                newline = "\n" if line.endswith("\n") else ""
                record = line.removesuffix("\n").ljust(80)
                line = record[:76] + "  " + record[78:] + newline
            rewritten.append(line)
        path.write_text("".join(rewritten), encoding="utf-8")

    def test_unknown_element_makes_metal_presence_indeterminate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builder = helpers.StructureBuilder()
        builder.add_metal("ZN", 1, chain="A")

        result = _manual_entry(
            tmp_path,
            monkeypatch,
            self._density_must_not_run,
            structure_builder=builder,
            pdb_transform=self._blank_metal_element,
        )

        assert result.status == "partial"
        assert result.retryable is False
        assert result.reason_codes == ["metal_presence_indeterminate"]
        assert "1 atom(s)" in result.error
        assert result.n_metals == 0
        assert result.n_bonds is None
        assert result.n_candidates is None
        assert result.no_metals is False
        assert "unknown_elements" in result.warning_codes
        assert result.confidence_inputs_missing_reason == "metal_presence_indeterminate"
        assert pool.confidence_rows_for(result, pool.ConfidencePlan()) == []

    def test_known_nonmetal_structure_remains_an_authoritative_negative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builder = helpers.StructureBuilder()
        builder.add_amino_acid("GLY", 1, chain="A")

        result = _manual_entry(
            tmp_path,
            monkeypatch,
            self._density_must_not_run,
            structure_builder=builder,
        )

        assert result.status == "ok"
        assert result.retryable is False
        assert result.reason_codes == []
        assert result.error == ""
        assert result.n_metals == 0
        assert result.n_bonds == 0
        assert result.n_candidates == 0
        assert result.no_metals is True

    def test_zero_occupancy_metal_is_not_an_analyzable_site(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        builder = helpers.StructureBuilder()
        builder.add_metal("ZN", 1, chain="A", occupancy=0.0)

        result = _manual_entry(
            tmp_path,
            monkeypatch,
            self._density_must_not_run,
            structure_builder=builder,
        )

        assert result.status == "ok"
        assert result.retryable is False
        assert result.reason_codes == []
        assert result.n_metals == 0
        assert result.n_bonds == 0
        assert result.n_candidates == 0
        assert result.no_metals is True
        assert "zero_occupancy_atoms" in result.warning_codes


# Hand-written because the behaviours under test are raw ``.``/``?`` occupancy
# tokens and >3-character component ids, neither of which gemmi would write.
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


def _cif_atom(
    serial: int | str,
    element: str,
    atom_name: str,
    comp_id: str,
    label_asym: str,
    entity: int,
    label_seq: int | str,
    xyz: tuple[float, float, float],
    occupancy: str | None,
    auth_seq: int,
    auth_asym: str,
    group: str = "ATOM",
    with_occupancy: bool = True,
) -> str:
    x, y, z = xyz
    fields = [
        group,
        str(serial),
        element,
        atom_name,
        ".",
        comp_id,
        label_asym,
        str(entity),
        str(label_seq),
        "?",
        f"{x}",
        f"{y}",
        f"{z}",
    ]
    if with_occupancy:
        # ``occupancy`` is None only where the caller also drops the column, so
        # the token appended here is always a string.
        fields.append(cast(str, occupancy))
    fields += ["20.0", str(auth_seq), auth_asym, "1"]
    return " ".join(fields)


def _write_cif(path: Path, atom_lines: Sequence[str], header: str = _CIF_HEADER) -> str:
    text = header + "".join(line + "\n" for line in atom_lines)
    with open(path, "w") as handle:
        handle.write(text)
    return str(path)


def _pdb_atom_lines(path: str | Path) -> list[str]:
    with open(path) as handle:
        return [
            line for line in handle if line[:6].strip().upper() in ("ATOM", "HETATM")
        ]


def _occupancy_field(line: str) -> str:
    """PDB columns 55-60, exactly as written."""
    return line[54:60]


def _element_field(line: str) -> str:
    """PDB columns 77-78, exactly as written."""
    return line[76:78].strip()


def _simple_structure(
    residues: Sequence[tuple[str, int, str, Sequence[tuple[str, str]]]],
) -> gemmi.Structure:
    """Build a one-model gemmi structure from (chain, seqid, name, atoms).

    gemmi's ``add_chain``/``add_residue`` copy their argument, so each chain is
    fully populated before it is attached.
    """
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain_order: list[str] = []
    chains: dict[str, gemmi.Chain] = {}
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


class TestPositiveInt:
    """``positive_int`` is the argparse gate for --workers/--max-pdbs/etc."""

    # ``value`` is typed ``Any`` throughout: argparse only ever hands it text,
    # and the cases below deliberately include the ints and ``None`` a
    # programmatic caller can pass, which the declared parameter type excludes.
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", 1),
            ("2", 2),
            ("28", 28),
            ("  7  ", 7),  # int() tolerates surrounding whitespace
            ("+3", 3),
            ("1000000", 1000000),
            (5, 5),  # already an int (programmatic callers)
        ],
    )
    def test_accepts_integers_of_at_least_one(self, value: Any, expected: int) -> None:
        assert cli.positive_int(value) == expected
        assert isinstance(cli.positive_int(value), int)

    @pytest.mark.parametrize("value", ["0", "-1", "-28", 0, -3])
    def test_rejects_zero_and_negative(self, value: Any) -> None:
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            cli.positive_int(value)
        assert "at least 1" in str(excinfo.value)

    @pytest.mark.parametrize(
        "value",
        [
            "1.5",
            "2.0",
            "abc",
            "",
            " ",
            "1e3",
            "0x2",
            None,
            "nan",
            "inf",
            "1,000",
        ],
    )
    def test_rejects_non_integer_text(self, value: Any) -> None:
        with pytest.raises(argparse.ArgumentTypeError) as excinfo:
            cli.positive_int(value)
        assert "positive integer" in str(excinfo.value)

    def test_boundary_is_one_not_zero(self) -> None:
        assert cli.positive_int("1") == 1
        with pytest.raises(argparse.ArgumentTypeError):
            cli.positive_int("0")


class TestLoadDone:
    """Which manifest rows count as finished work that --resume may skip."""

    @pytest.mark.parametrize(
        "status,retryable,expected_done",
        [
            ("ok", "False", True),
            ("ok", "True", True),  # status ok is terminal regardless
            ("ok", "", True),
            ("partial", "False", True),  # terminal partial: nothing left to do
            ("partial", "0", True),
            ("partial", "no", True),
            ("partial", "True", False),
            ("partial", "", False),
            ("error", "True", False),
            # An error is never done: a resumed run may have been handed a
            # repaired input, and skipping would drop work already fixed.
            ("error", "False", False),
            ("skip", "False", False),
            ("skip", "True", False),
            ("", "", False),
        ],
    )
    def test_terminality_by_status_and_retryable(
        self, tmp_path: Path, status: str, retryable: str, expected_done: bool
    ) -> None:
        """Only ok and non-retryable partial rows are skippable."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": status,
                    "retryable": retryable,
                    "n_bonds": "3",
                    "n_candidates": "5",
                },
            ],
        )
        assert ("109m" in _manifest_ids(path)) is expected_done

    def test_an_error_is_retried_whatever_its_reason_code_claims(
        self, tmp_path: Path
    ) -> None:
        """A deterministic-looking failure is still offered to the next resume.

        The reason code says the failure recurs on the same inputs, but a
        resumed run may name a repaired file, so resume cannot treat the claim
        as licence to skip.
        """
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "error",
                    "retryable": "False",
                    "reason_codes": "deterministic_processing_error",
                },
            ],
        )
        assert _manifest_ids(path) == set()

    def test_ids_are_normalized_to_lowercase(self, tmp_path: Path) -> None:
        """Manifest IDs join against the driver's lowercased selection list."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": " 1CLL ", "status": "ok", "retryable": "False"},
            ],
        )
        assert _manifest_ids(path) == {"1cll"}

    def test_explicit_terminal_retry_unprotects_only_selected_partials(
        self, tmp_path: Path
    ) -> None:
        """Forced resume never turns an explicitly selected ``ok`` into work."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "1ok1",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "1",
                    "n_candidates": "1",
                },
                {
                    "pdbID": "1par",
                    "status": "partial",
                    "retryable": "False",
                    "n_bonds": "1",
                    "n_candidates": "1",
                },
                {
                    "pdbID": "2par",
                    "status": "partial",
                    "retryable": "False",
                    "n_bonds": "1",
                    "n_candidates": "1",
                },
                {
                    "pdbID": "1err",
                    "status": "error",
                    "retryable": "True",
                    "n_bonds": "",
                    "n_candidates": "",
                },
            ],
        )

        assert _manifest_ids(path, retry_partial_ids={"1OK1", "1PAR"}) == {
            "1ok1",
            "2par",
        }

    def test_missing_manifest_is_an_empty_done_set(self, tmp_path: Path) -> None:
        assert _manifest_ids(tmp_path / "absent.csv") == set()

    def test_manifest_without_the_required_columns_is_ignored(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "manifest.csv"
        with open(path, "w", newline="") as handle:
            csv.writer(handle).writerows([["id", "state"], ["109m", "ok"]])
        assert _manifest_ids(path) == set()

    def test_empty_manifest_file_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.csv"
        path.write_text("")
        assert _manifest_ids(path) == set()

    def test_truncated_final_row_is_retried_without_losing_valid_rows(
        self, tmp_path: Path
    ) -> None:
        """An interrupted writer may leave only the first cells of its row."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok", "retryable": "False"}],
        )
        with open(path, "a", newline="") as handle:
            csv.writer(handle).writerow(["1cll", "ok"])

        assert _manifest_ids(path, bonds_required=False) == {"109m"}

    def test_row_without_a_pdb_id_is_not_done(self, tmp_path: Path) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "0",
                    "n_candidates": "0",
                },
            ],
        )
        assert _manifest_ids(path) == set()

    @pytest.mark.parametrize(
        "n_bonds,n_candidates,expected_done",
        [
            ("0", "0", True),  # measured zero: the stage ran and found none
            ("4", "9", True),
            ("", "0", False),  # blank: the stage never ran
            ("0", "", False),
            ("", "", False),
            ("   ", "0", False),  # whitespace is still blank
        ],
    )
    def test_blank_counts_mean_the_bond_stage_never_ran(
        self, tmp_path: Path, n_bonds: str, n_candidates: str, expected_done: bool
    ) -> None:
        """README: blank counts = not run, 0 = ran and found nothing."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": n_bonds,
                    "n_candidates": n_candidates,
                },
            ],
        )
        done = _manifest_ids(path, bonds_required=True)
        assert ("109m" in done) is expected_done

    def test_blank_counts_are_irrelevant_when_bonds_are_not_required(
        self, tmp_path: Path
    ) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "",
                    "n_candidates": "",
                },
            ],
        )
        assert _manifest_ids(path, bonds_required=False) == {"109m"}

    def test_indeterminate_metal_presence_needs_no_bond_stage(
        self, tmp_path: Path
    ) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "partial",
                    "retryable": "False",
                    "reason_codes": "metal_presence_indeterminate",
                    "n_bonds": "",
                    "n_candidates": "",
                }
            ],
        )

        assert _manifest_ids(path, bonds_required=True) == {"109m"}
        assert (
            resume.load_done(
                path,
                bonds_required=True,
                bond_output_present=False,
                candidate_output_present=True,
            )
            == set()
        )

    @pytest.mark.parametrize(
        "bond_present,candidate_present,expected_done",
        [
            (True, True, True),
            (False, True, False),
            (True, False, False),
            (False, False, False),
        ],
    )
    def test_absent_bond_outputs_make_the_result_incomplete(
        self,
        tmp_path: Path,
        bond_present: bool,
        candidate_present: bool,
        expected_done: bool,
    ) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "4",
                    "n_candidates": "9",
                },
            ],
        )
        done = resume.load_done(
            path,
            bonds_required=True,
            bond_output_present=bond_present,
            candidate_output_present=candidate_present,
        )
        assert ("109m" in done) is expected_done

    def test_selects_only_the_terminal_rows_of_a_mixed_manifest(
        self, tmp_path: Path
    ) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {
                    "pdbID": "109m",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "6",
                    "n_candidates": "8",
                },
                {
                    "pdbID": "1cll",
                    "status": "partial",
                    "retryable": "False",
                    "n_bonds": "0",
                    "n_candidates": "0",
                },
                {
                    "pdbID": "1blu",
                    "status": "partial",
                    "retryable": "True",
                    "n_bonds": "1",
                    "n_candidates": "2",
                },
                {"pdbID": "2fha", "status": "error", "retryable": "True"},
                {"pdbID": "2cyp", "status": "skip", "retryable": "False"},
                {
                    "pdbID": "100d",
                    "status": "ok",
                    "retryable": "False",
                    "n_bonds": "",
                    "n_candidates": "",
                },
            ],
        )
        assert _manifest_ids(path, bonds_required=True) == {"109m", "1cll"}
        assert _manifest_ids(path, bonds_required=False) == {"109m", "1cll", "100d"}


class TestManifestValuesById:
    """Reading one prior manifest column for the resume carry-forward."""

    def test_returns_the_column_keyed_by_normalized_id(self, tmp_path: Path) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "109M", "status": "ok", "n_bonds": "7"},
                {"pdbID": " 1cll", "status": "ok", "n_bonds": "0"},
                {"pdbID": "1blu", "status": "error", "n_bonds": ""},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {
            "109m": "7",
            "1cll": "0",
            "1blu": "",
        }

    def test_missing_and_empty_files_yield_no_values(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        assert resume.manifest_values_by_id(str(tmp_path / "gone.csv"), "n_bonds") == {}
        assert resume.manifest_values_by_id(str(empty), "n_bonds") == {}

    def test_unknown_column_yields_blanks_not_an_error(self, tmp_path: Path) -> None:
        """An older manifest lacking the column must degrade to "not run"."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok"}],
            columns=["pdbID", "status"],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": ""}

    def test_truncated_rows_do_not_supply_carry_forward_values(
        self, tmp_path: Path
    ) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [{"pdbID": "109m", "status": "ok", "n_bonds": "7"}],
        )
        with open(path, "a", newline="") as handle:
            csv.writer(handle).writerow(["1cll", "ok"])

        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "7"}

    def test_blank_ids_are_dropped(self, tmp_path: Path) -> None:
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "", "status": "ok", "n_bonds": "9"},
                {"pdbID": "109m", "status": "ok", "n_bonds": "1"},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "1"}

    def test_a_later_row_supersedes_an_earlier_one(self, tmp_path: Path) -> None:
        """Resume appends, so the last row for an ID is the current one."""
        path = _write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": "109m", "status": "error", "n_bonds": ""},
                {"pdbID": "109m", "status": "ok", "n_bonds": "5"},
            ],
        )
        assert resume.manifest_values_by_id(path, "n_bonds") == {"109m": "5"}


class TestInitialResult:
    """The per-entry skeleton that guarantees a complete manifest row."""

    def test_seeds_bond_counts_blank_not_zero(self) -> None:
        """``0`` is a measured result, so an unrun bond stage must stay blank."""
        result = worker.initial_result("109m", CFG, None)
        assert result.n_bonds is None
        assert result.n_candidates is None
        assert result.n_bonds != 0
        assert result.n_candidates != 0

    def test_every_non_derived_manifest_column_is_present_up_front(self) -> None:
        """A failure at any stage still projects onto a complete row."""
        result = worker.initial_result("109m", CFG, None)
        required = set(MANIFEST_COLUMNS) - DERIVED_MANIFEST_COLUMNS
        supplied = {
            column for column, name in MANIFEST_FIELDS.items() if hasattr(result, name)
        }
        assert required.issubset(supplied)

    def test_supplies_the_fields_manifest_row_reads_directly(self) -> None:
        """``manifest_row`` reads these without a default; they must exist."""
        result = worker.initial_result("109m", CFG, None)
        for name in ("n_metals", "runtime_s", "n_bonds", "n_candidates", "pdb_id"):
            assert hasattr(result, name)

    def test_defaults_to_a_retryable_error(self) -> None:
        result = worker.initial_result("109m", CFG, None)
        assert result.status == "error"
        assert result.retryable is True

    def test_carries_the_reference_data_identity(self) -> None:
        """Two runs whose z-scores used different reference distances must be
        distinguishable in the output.
        """
        result = worker.initial_result("109m", CFG, None)
        row = manifest_row(result, False, True, {}, {})

        assert result.reference_data_id == CFG.reference_data_id
        assert row["reference_data_id"] == CFG.reference_data_id

    def test_carries_run_provenance_from_the_config(self) -> None:
        result = worker.initial_result("109m", CFG, None)
        assert result.alchemy_version == pool.ALCHEMY_VERSION
        assert result.alchemy_commit == CFG.alchemy_commit
        assert result.gemmi_version == CFG.gemmi_version
        assert result.ccp4_version == CFG.ccp4_version
        assert result.model_policy == worker_contracts.MODEL_POLICY
        assert result.altloc_policy == worker_contracts.ALTLOC_POLICY
        assert result.symmetry_contact_policy == worker_contracts.SYMMETRY_POLICY

    @pytest.mark.parametrize(
        "manual_inputs,expected",
        [
            (None, "final"),
            ({}, "final"),
            ({"pdb_file": "/x.pdb"}, "manual"),
        ],
    )
    def test_refinement_state_reflects_manual_inputs(
        self, manual_inputs: dict[str, str | None] | None, expected: str
    ) -> None:
        """Manual coordinate/MTZ input is not a PDB-REDO final re-refinement."""
        result = worker.initial_result("109m", CFG, manual_inputs)
        assert result.refinement_state == expected

    def test_row_lists_are_independent_between_entries(self) -> None:
        first = worker.initial_result("109m", CFG, None)
        second = worker.initial_result("1cll", CFG, None)
        first.rows.append(MetalStatsRow.from_output_fields("109m", "metal", [1]))
        first.reason_codes.append("boom")
        assert second.rows == []
        assert second.reason_codes == []

    def test_a_misspelled_field_cannot_be_assigned(self) -> None:
        """``slots=True`` makes a misspelled assignment the error, not a field
        nobody reads and a stale value in the manifest.
        """
        result = worker.initial_result("109m", CFG, None)

        with pytest.raises(AttributeError, match="retryble"):
            result.retryble = True  # type: ignore[attr-defined]

        assert result.retryable is True, "the real field must be untouched"

    def test_unmeasured_fields_render_blank_not_none(self) -> None:
        """A ``None`` reaching a CSV as ``"None"`` reads back to the next
        ``--resume`` as a completed stage.
        """
        result = worker.initial_result("109m", CFG, None)
        row = manifest_row(result, False, True, {}, {})

        for column in ("n_bonds", "n_candidates", "input_model_count"):
            assert row[column] == ""
        assert "None" not in set(map(str, row.values()))


class TestManifestRow:
    """Projection of a worker result onto the manifest schema."""

    def test_projects_exactly_the_manifest_columns(self) -> None:
        row = manifest_row(_result(), False, True, {}, {})
        assert set(row) == set(MANIFEST_COLUMNS)
        assert "rows" not in row
        assert "bond_rows" not in row
        assert "timings" not in row

    def test_renames_and_joins_the_derived_columns(self) -> None:
        """n/runtime become n_metals/runtime_s; code lists become pipe text."""
        result = _result(
            n_metals=3,
            runtime_s=12.5,
            n_bonds=7,
            n_candidates=11,
            reason_codes=["a", "b"],
            warning_codes=["w"],
        )
        row = manifest_row(result, False, True, {}, {})
        assert row["n_metals"] == 3
        assert row["runtime_s"] == 12.5
        assert row["n_bonds"] == 7
        assert row["n_candidates"] == 11
        assert row["reason_codes"] == "a|b"
        assert row["warning_codes"] == "w"

    def test_empty_code_lists_render_blank(self) -> None:
        """No codes must not become a spurious separator or literal '[]'."""
        row = manifest_row(_result(), False, True, {}, {})
        assert row["reason_codes"] == ""
        assert row["warning_codes"] == ""

    def test_bonds_enabled_reports_the_measured_zero(self) -> None:
        row = manifest_row(_result(n_bonds=0, n_candidates=0), False, True, {}, {})
        assert row["n_bonds"] == 0
        assert row["n_candidates"] == 0

    def test_fresh_no_bonds_run_writes_blank_counts(self) -> None:
        """Without --resume there is no prior stage to carry forward."""
        row = manifest_row(
            _result(n_bonds=4, n_candidates=6),
            False,
            False,
            {"109m": "99"},
            {"109m": "98"},
        )
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_resume_no_bonds_carries_the_prior_counts_forward(self) -> None:
        row = manifest_row(_result("109M"), True, False, {"109m": "6"}, {"109m": "8"})
        assert row["n_bonds"] == "6"
        assert row["n_candidates"] == "8"

    def test_resume_no_bonds_carry_forward_preserves_a_prior_zero(self) -> None:
        row = manifest_row(_result(), True, False, {"109m": "0"}, {"109m": "0"})
        assert row["n_bonds"] == "0"
        assert row["n_candidates"] == "0"

    def test_resume_no_bonds_without_a_prior_row_stays_blank(self) -> None:
        row = manifest_row(_result("1cll"), True, False, {"109m": "6"}, {"109m": "8"})
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_prior_blank_counts_are_not_upgraded(self) -> None:
        row = manifest_row(_result(), True, False, {"109m": ""}, {"109m": ""})
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""


class TestUnrunBondStageChain:
    """An unrun bond stage must never look complete.

    What matters is the chain, not the seed value:
      1. an entry fails before the bond stage,
      2. its manifest row is carried forward by ``--resume --no-bonds``,
      3. a bond-enabled ``--resume`` reads that row.
    A ``0`` anywhere along it makes step 3 skip the entry permanently while the
    bond CSVs hold no rows for it.
    """

    @staticmethod
    def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_failed_bond_enabled_entry_is_not_marked_bond_complete(
        self, tmp_path: Path
    ) -> None:
        """Step 1: a pre-bond failure writes blank counts, so it is retried."""
        result = _result(
            "109m", status="error", retryable=True, error="density stage failed"
        )
        row = manifest_row(result, False, True, {}, {})
        manifest = tmp_path / "manifest.csv"
        self._write_rows(manifest, [row])

        assert resume.load_done(str(manifest), bonds_required=True) == set()
        assert resume.load_done(str(manifest), bonds_required=False) == set()
        assert row["n_bonds"] == ""
        assert row["n_candidates"] == ""

    def test_resume_no_bonds_recovery_does_not_fake_a_completed_bond_stage(
        self, tmp_path: Path
    ) -> None:
        """Run 1 fails before the bond stage, run 2 recovers it under
        ``--no-bonds``, and run 3 must still schedule it: no bond analysis has
        ever run for the entry.
        """
        manifest = tmp_path / "manifest.csv"

        failed = manifest_row(
            _result("109m", status="error", retryable=True, error="edstats failed"),
            resume=False,
            bonds_enabled=True,
            prior_bond_counts={},
            prior_candidate_counts={},
        )
        self._write_rows(manifest, [failed])

        assert resume.load_done(str(manifest), bonds_required=False) == set()
        prior_bonds = resume.manifest_values_by_id(str(manifest), "n_bonds")
        prior_candidates = resume.manifest_values_by_id(str(manifest), "n_candidates")

        recovered = manifest_row(
            _result("109m", status="ok", retryable=False, n_metals=2),
            resume=True,
            bonds_enabled=False,
            prior_bond_counts=prior_bonds,
            prior_candidate_counts=prior_candidates,
        )
        assert recovered["status"] == "ok"
        assert recovered["n_metals"] == 2
        self._write_rows(manifest, [recovered])

        assert resume.load_done(str(manifest), bonds_required=False) == {"109m"}
        assert resume.load_done(str(manifest), bonds_required=True) == set()

        # Only a genuine measured zero may let a later resume skip the entry.
        completed = manifest_row(
            _result(
                "109m",
                status="ok",
                retryable=False,
                n_metals=2,
                n_bonds=0,
                n_candidates=0,
            ),
            resume=True,
            bonds_enabled=True,
            prior_bond_counts={},
            prior_candidate_counts={},
        )
        self._write_rows(manifest, [completed])
        assert resume.load_done(str(manifest), bonds_required=True) == {"109m"}


class TestResumeReplacementSucceeded:
    """Only a terminal retry may replace the rows it is retrying."""

    @pytest.mark.parametrize(
        "status,retryable,expected",
        [
            ("ok", False, True),
            ("ok", True, True),
            ("partial", False, True),
            ("partial", True, False),
            ("error", False, False),
            ("error", True, False),
            ("skip", False, False),
        ],
    )
    def test_terminality(self, status: str, retryable: bool, expected: bool) -> None:
        """A retryable or failed retry leaves the previous rows in place."""
        result = _result(status=status, retryable=retryable)
        assert resume.resume_replacement_succeeded(result) is expected

    @pytest.mark.parametrize("status", ["OK", " ok ", ""])
    def test_an_invalid_internal_status_is_rejected(self, status: str) -> None:
        with pytest.raises(ValueError):
            EntryStatus(status)

    def test_an_untouched_partial_is_assumed_unfinished(self) -> None:
        """``retryable`` defaults to true, so a partial nothing cleared cannot
        replace.
        """
        assert resume.resume_replacement_succeeded(_result(status="partial")) is False


class TestResumeStaging:
    """Staged retries replace rows only on a completed, terminal batch."""

    TARGET_NAMES = (
        "manifest.csv",
        "metal_stats_all.csv",
        "metal_bonds_all.csv",
        "metal_candidates_all.csv",
    )

    def _outputs(self, output_dir: Path, confidence: bool = False) -> tuple[str, ...]:
        """Create the four (or five) output CSVs with two entries' rows."""
        names: list[str] = list(self.TARGET_NAMES)
        if confidence:
            names.append("confidence_inputs_all.csv")
        paths: list[str] = []
        for name in names:
            path = output_dir / name
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value"])
                writer.writerow(["109m", f"old-109m-{name}"])
                writer.writerow(["1cll", f"old-1cll-{name}"])
            paths.append(str(path))
        return tuple(paths)

    def _stage(
        self,
        staging: resume.ResumeStaging,
        names_and_rows: Mapping[int, Sequence[Sequence[str]]],
    ) -> None:
        """Write staged rows for the given staged-file indices."""
        for index, rows in names_and_rows.items():
            with open(staging.staged[index], "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["pdbID", "value"])
                for row in rows:
                    writer.writerow(row)

    def test_staged_paths_mirror_the_targets_inside_the_output_dir(
        self, tmp_path: Path
    ) -> None:
        """Staging is a sibling temp dir so a merge is a same-filesystem move."""
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            assert os.path.isdir(staging.dir)
            assert os.path.dirname(staging.dir) == str(tmp_path)
            assert [os.path.basename(p) for p in staging.staged] == [
                os.path.basename(p) for p in targets
            ]
            assert all(os.path.dirname(p) == staging.dir for p in staging.staged)
            assert staging.replacement_ids == set()
            for target in targets:
                assert _read_csv(target)[1][1].startswith("old-109m")
        finally:
            staging.discard()

    def test_commit_without_replacement_ids_leaves_outputs_byte_identical(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        before = {p: open(p, "rb").read() for p in targets}
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {0: [["109m", "new"]]})
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        assert {p: open(p, "rb").read() for p in targets} == before

    def test_an_interrupted_batch_commits_the_entries_it_finished(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The halted-run path is reached from ``process_entries`` itself.

        The loss lived in the wiring: staging was discarded whenever the batch
        failed to run to completion. A test that calls the commit helper
        directly still passes with that defect in place, so this one drives the
        real function and interrupts it mid-batch.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        layout = driver_pool.OutputLayout(str(output_dir))
        headers = (
            (layout.manifest, MANIFEST_COLUMNS),
            (layout.stats, STATS_COLUMNS),
            (layout.bonds, coordination_schema.BOND_COLUMNS),
            (layout.candidates, coordination_schema.CANDIDATE_COLUMNS),
        )
        for path, columns in headers:
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
        prior = {name: "" for name in MANIFEST_COLUMNS}
        prior.update(pdbID="aaaa", status="ok", retryable="False")
        with open(layout.manifest, "a", newline="") as handle:
            csv.writer(handle).writerow([prior[name] for name in MANIFEST_COLUMNS])

        def _dispatch(
            args: RunConfig,
            ids: Sequence[str],
            cfg: worker_contracts.WorkerConfig,
            workers: int,
            writers: OutputWriters,
            plan: driver_pool.ConfidencePlan,
            staging: resume.ResumeStaging,
            counts: tuple[dict[str, str], dict[str, str]],
            prior: set[str],
            run_log: RunLog,
            memory_plan: driver_pool.MemoryPlan,
        ) -> None:
            del memory_plan
            for pdb_id in ids:
                row = {name: "" for name in MANIFEST_COLUMNS}
                row.update(pdbID=pdb_id, status="ok", retryable="False")
                writers.write_manifest_row(row)
                staging.replacement_ids.add(pdb_id)
            raise KeyboardInterrupt

        monkeypatch.setattr(driver_pool, "_dispatch_entries", _dispatch)
        args = _run_config(resume=True, bonds=True, output_dir=str(output_dir))
        run_log = RunLog(args, "pytest")

        with pytest.raises(KeyboardInterrupt):
            driver_pool.process_entries(
                args,
                ["bbbb", "cccc"],
                # ``None`` proves the halted-batch path never reaches the worker
                # configuration: ``_dispatch_entries`` is replaced above.
                cast(worker_contracts.WorkerConfig, None),
                1,
                layout,
                driver_pool.ConfidencePlan(),
                run_log,
                driver_pool.MemoryPlan(
                    [
                        resources.EntryMemoryEstimate("bbbb", 1, "test"),
                        resources.EntryMemoryEstimate("cccc", 1, "test"),
                    ],
                    None,
                    None,
                ),
            )

        assert [row[0] for row in _read_csv(layout.manifest)[1:]] == [
            "aaaa",
            "bbbb",
            "cccc",
        ]
        assert run_log.summary["resume_entries_committed_after_interrupt"] == 2

    def _interrupt(
        self,
        staging: resume.ResumeStaging,
        tmp_path: Path,
        *,
        bonds: bool = True,
        confidence: bool = False,
    ) -> dict[str, Any]:
        """Run the halted-resume path and return the run-log summary."""
        args = _run_config(bonds=bonds, output_dir=str(tmp_path))
        plan = driver_pool.ConfidencePlan()
        # ``enabled`` is derived from the mode, which is what a scoring run sets.
        plan.mode = "reference" if confidence else None
        run_log = RunLog(args, "pytest")
        driver_pool.keep_completed_staging(staging, args, plan, run_log)
        return run_log.summary

    def test_an_interrupted_resume_keeps_the_entries_it_completed(
        self, tmp_path: Path
    ) -> None:
        """A halted batch commits finished entries instead of destroying them.

        Discarding lost every entry the run had completed, including entries
        with no previous manifest row that staging never existed to protect,
        while the interrupt message promised they had been kept.
        """
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(
            staging,
            {
                index: [["8new", f"new-8new-{name}"]]
                for index, name in enumerate(self.TARGET_NAMES)
            },
        )
        staging.replacement_ids.add("8new")

        summary = self._interrupt(staging, tmp_path)

        for target in targets:
            ids = [row[0] for row in _read_csv(target)[1:]]
            assert ids == ["109m", "1cll", "8new"], target
        assert summary["resume_entries_committed_after_interrupt"] == 1
        assert not os.path.isdir(staging.dir)

    def test_an_interrupted_resume_drops_an_entry_that_never_completed(
        self, tmp_path: Path
    ) -> None:
        """Rows without a manifest row are not promoted, so the entry retries.

        ``write_entry`` adds the id only after the manifest row, so an entry
        interrupted mid-write is absent from ``replacement_ids``.
        """
        targets = self._outputs(tmp_path)
        before = {path: open(path, "rb").read() for path in targets}
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(staging, {1: [["8new", "half-written"]]})

        summary = self._interrupt(staging, tmp_path)

        assert {path: open(path, "rb").read() for path in targets} == before
        assert "resume_entries_committed_after_interrupt" not in summary
        assert not os.path.isdir(staging.dir)

    def test_a_failed_interrupt_merge_leaves_the_staged_rows_on_disk(
        self, tmp_path: Path
    ) -> None:
        """A merge failure must not delete the only copy of the work."""
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(staging, {0: [["8new", "new"]]})
        staging.replacement_ids.add("8new")
        # A staged file whose header disagrees makes ``commit`` raise.
        with open(staging.staged[1], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["different", "header"])
            writer.writerow(["8new", "new"])

        try:
            summary = self._interrupt(staging, tmp_path)

            assert os.path.isdir(staging.dir)
            assert summary["resume_staging_recovery_dir"] == staging.dir
            assert "staged CSV schema" in summary["resume_staging_commit_error"]
        finally:
            staging.discard()

    def test_commit_replaces_only_the_retried_ids(self, tmp_path: Path) -> None:
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(
                staging,
                {
                    index: [["109m", f"new-109m-{name}"]]
                    for index, name in enumerate(self.TARGET_NAMES)
                },
            )
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
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
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
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        bond_paths = targets[2:4]
        before = {p: open(p, "rb").read() for p in bond_paths}
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(
                staging,
                {
                    0: [["109m", "new-manifest"]],
                    1: [["109m", "new-stats"]],
                    2: [["109m", "SHOULD-NOT-APPEAR"]],
                    3: [["109m", "SHOULD-NOT-APPEAR"]],
                },
            )
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=False)
        finally:
            staging.discard()
        assert {p: open(p, "rb").read() for p in bond_paths} == before
        manifest_values = {row[0]: row[1] for row in _read_csv(targets[0])[1:]}
        assert manifest_values == {
            "109m": "new-manifest",
            "1cll": "old-1cll-manifest.csv",
        }

    def test_commit_replaces_confidence_rows_only_when_enabled(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path, confidence=True)
        confidence_path = targets[4]
        before = open(confidence_path, "rb").read()

        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {index: [["109m", "new"]] for index in range(5)})
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=False)
        finally:
            staging.discard()
        assert open(confidence_path, "rb").read() == before

        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {index: [["109m", "new"]] for index in range(5)})
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=True)
        finally:
            staging.discard()
        values = {row[0]: row[1] for row in _read_csv(confidence_path)[1:]}
        assert values == {"109m": "new", "1cll": ("old-1cll-confidence_inputs_all.csv")}

    def test_commit_replaces_every_enabled_confidence_output(
        self, tmp_path: Path
    ) -> None:
        """Scored and input confidence rows participate in one staged commit."""
        targets = list(self._outputs(tmp_path, confidence=True))
        scores = tmp_path / "confidence_scores_all.csv"
        with open(scores, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(
                [
                    ["pdbID", "value"],
                    ["109m", "old-109m-scores"],
                    ["1cll", "old-1cll-scores"],
                ]
            )
        targets.append(str(scores))
        staging = resume.ResumeStaging(str(tmp_path), tuple(targets))
        try:
            self._stage(
                staging,
                {index: [["109m", f"new-{index}"]] for index in range(len(targets))},
            )
            staging.replacement_ids.add("109m")
            staging.commit(bonds_enabled=True, confidence_enabled=True)
        finally:
            staging.discard()

        for index in (4, 5):
            values = {row[0]: row[1] for row in _read_csv(targets[index])[1:]}
            expected_old = (
                "old-1cll-confidence_inputs_all.csv"
                if index == 4
                else "old-1cll-scores"
            )
            assert values == {"109m": f"new-{index}", "1cll": expected_old}

    def test_discard_leaves_previous_rows_intact(self, tmp_path: Path) -> None:
        targets = self._outputs(tmp_path)
        before = {p: open(p, "rb").read() for p in targets}
        staging = resume.ResumeStaging(str(tmp_path), targets)
        self._stage(staging, {index: [["109m", "new"]] for index in range(4)})
        staging.replacement_ids.add("109m")
        staging.discard()
        assert not os.path.exists(staging.dir)
        assert {p: open(p, "rb").read() for p in targets} == before

    def test_discard_is_idempotent(self, tmp_path: Path) -> None:
        """Cleanup runs in a finally block and may be reached twice."""
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        staging.discard()
        staging.discard()
        assert not os.path.exists(staging.dir)

    def test_replacement_ids_are_matched_case_insensitively(
        self, tmp_path: Path
    ) -> None:
        targets = self._outputs(tmp_path)
        staging = resume.ResumeStaging(str(tmp_path), targets)
        try:
            self._stage(staging, {index: [["109M", "new"]] for index in range(4)})
            staging.replacement_ids.add("109M")
            staging.commit(bonds_enabled=True)
        finally:
            staging.discard()
        values = {row[0].lower(): row[1] for row in _read_csv(targets[0])[1:]}
        assert values["109m"] == "new"
        assert len(values) == 2

    def test_a_staged_schema_mismatch_aborts_without_touching_the_target(
        self, tmp_path: Path
    ) -> None:
        """A header disagreement must fail loudly, not silently misalign rows."""
        targets = self._outputs(tmp_path)
        before = open(targets[0], "rb").read()
        staging = resume.ResumeStaging(str(tmp_path), targets)
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
        leftovers = [
            name for name in os.listdir(tmp_path) if name.startswith(".manifest.csv.")
        ]
        assert leftovers == []


class _Handles(NamedTuple):
    """The open output files, in ``OutputWriters`` argument order."""

    manifest: Any
    stats: Any
    bonds: Any
    candidates: Any
    confidence: Any


class TestOutputWriters:
    """The streamed CSVs: headers on creation, running counts, schema guards."""

    @staticmethod
    def _handles(
        tmp_path: Path,
        bonds: bool = True,
        candidates: bool = True,
        confidence: bool | None = None,
    ) -> _Handles:
        return _Handles(
            manifest=open(tmp_path / "manifest.csv", "w", newline=""),
            stats=open(tmp_path / "stats.csv", "w", newline=""),
            bonds=open(tmp_path / "bonds.csv", "w", newline="") if bonds else None,
            candidates=(
                open(tmp_path / "candidates.csv", "w", newline="")
                if candidates
                else None
            ),
            confidence=(
                open(tmp_path / "confidence.csv", "w", newline="")
                if confidence
                else None
            ),
        )

    @staticmethod
    def _close(handles: _Handles) -> None:
        for handle in handles:
            if handle is not None:
                handle.close()

    def test_headers_survive_a_run_that_produced_no_rows(self, tmp_path: Path) -> None:
        """README: the CSVs keep their headers when nothing was found."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        writers.write_stats_rows([])
        writers.write_bond_rows([])
        writers.write_candidate_rows([])
        self._close(handles)

        assert _read_csv(tmp_path / "manifest.csv") == [MANIFEST_COLUMNS]
        assert _read_csv(tmp_path / "stats.csv") == [list(STATS_COLUMNS)]
        assert _read_csv(tmp_path / "bonds.csv") == [
            list(coordination_schema.BOND_COLUMNS)
        ]
        assert _read_csv(tmp_path / "candidates.csv") == [
            list(coordination_schema.CANDIDATE_COLUMNS)
        ]
        assert (writers.n_rows, writers.n_bonds, writers.n_candidates) == (0, 0, 0)

    def test_running_counts_track_the_rows_actually_written(
        self, tmp_path: Path
    ) -> None:
        """The end-of-run totals come from these counters, not from re-reading."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        stats_rows = [
            MetalStatsRow.from_output_fields(
                "109m", "metal", ["x"] * (len(STATS_COLUMNS) - 2)
            )
            for _ in range(3)
        ]
        bond_rows = [
            coordination_schema.BondRow(
                dict.fromkeys(coordination_schema.BOND_COLUMNS, "")
            )
            for _ in range(4)
        ]
        candidate_rows = [
            coordination_schema.CandidateRow(
                dict.fromkeys(coordination_schema.CANDIDATE_COLUMNS, "")
            )
            for _ in range(2)
        ]
        writers.write_stats_rows(stats_rows)
        writers.write_bond_rows(bond_rows)
        writers.write_candidate_rows(candidate_rows)
        writers.write_stats_rows(stats_rows)
        self._close(handles)

        assert writers.n_rows == 6
        assert writers.n_bonds == 4
        assert writers.n_candidates == 2
        assert len(_read_csv(tmp_path / "stats.csv")) == 7
        assert len(_read_csv(tmp_path / "bonds.csv")) == 5
        assert len(_read_csv(tmp_path / "candidates.csv")) == 3

    def test_stats_rows_are_projected_onto_the_fixed_schema(
        self, tmp_path: Path
    ) -> None:
        """id and category lead each row; the EDSTATS block follows verbatim."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        fields = [str(i) for i in range(len(STATS_COLUMNS) - 2)]
        writers.write_stats_rows(
            [MetalStatsRow.from_output_fields("109m", "cofactor", fields)]
        )
        self._close(handles)
        rows = _read_csv(tmp_path / "stats.csv")
        assert rows[1] == ["109m", "cofactor"] + fields
        assert len(rows[1]) == len(STATS_COLUMNS)

    def test_stats_batch_is_validated_before_any_rows_are_written(
        self, tmp_path: Path
    ) -> None:
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        fields = [""] * (len(STATS_COLUMNS) - 2)
        rows = [
            MetalStatsRow.from_output_fields("109m", "metal", fields),
            MetalStatsRow.from_output_fields("1cll", "metal", fields[:-1]),
        ]
        try:
            with pytest.raises(ValueError):
                writers.write_stats_rows(rows)
        finally:
            self._close(handles)
        assert writers.n_rows == 0
        assert _read_csv(tmp_path / "stats.csv") == [STATS_COLUMNS]

    def test_bond_rows_are_written_in_schema_order(self, tmp_path: Path) -> None:
        """Columns are positional, so the projection order is load-bearing."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        row = {column: f"v-{column}" for column in coordination_schema.BOND_COLUMNS}
        shuffled = {key: row[key] for key in reversed(list(row))}
        writers.write_bond_rows([coordination_schema.BondRow(shuffled)])
        self._close(handles)
        written = _read_csv(tmp_path / "bonds.csv")[1]
        assert written == [f"v-{column}" for column in coordination_schema.BOND_COLUMNS]

    def test_disabled_bond_outputs_are_a_no_op_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        """--no-bonds passes None handles; writes must be silently skipped."""
        handles = self._handles(tmp_path, bonds=False, candidates=False)
        writers = OutputWriters(*handles)
        writers.write_bond_rows(
            [
                coordination_schema.BondRow(
                    dict.fromkeys(coordination_schema.BOND_COLUMNS, "")
                )
            ]
        )
        writers.write_candidate_rows(
            [
                coordination_schema.CandidateRow(
                    dict.fromkeys(coordination_schema.CANDIDATE_COLUMNS, "")
                )
            ]
        )
        self._close(handles)
        assert writers.n_bonds == 0
        assert writers.n_candidates == 0
        assert not (tmp_path / "bonds.csv").exists()
        assert not (tmp_path / "candidates.csv").exists()

    @pytest.mark.parametrize(
        "mutate,columns_name",
        [
            ("drop", "BOND_COLUMNS"),
            ("add", "BOND_COLUMNS"),
        ],
    )
    def test_bond_row_schema_drift_fails_loudly(
        self, tmp_path: Path, mutate: str, columns_name: str
    ) -> None:
        """A silently dropped or ignored column would corrupt every later row."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        row = dict.fromkeys(getattr(coordination_schema, columns_name), "")
        if mutate == "drop":
            drifted = next(iter(row))
            row.pop(drifted)
            expected_clause = f"missing {drifted}"
        else:
            drifted = "unexpected_column"
            row[drifted] = ""
            expected_clause = f"unexpected {drifted}"
        try:
            with pytest.raises(RuntimeError) as excinfo:
                coordination_schema.BondRow(row)
        finally:
            self._close(handles)
        message = str(excinfo.value)
        assert "metal_bonds_all.csv" in message
        # The guard must name the column: "the schema drifted" alone sends a
        # maintainer through 90 columns by hand.
        assert expected_clause in message
        assert writers.n_bonds == 0

    def test_candidate_row_schema_drift_fails_loudly(self, tmp_path: Path) -> None:
        """Same guard on the candidate stream, named for its own file."""
        row = dict.fromkeys(coordination_schema.CANDIDATE_COLUMNS, "")
        row["bogus"] = ""
        with pytest.raises(RuntimeError) as excinfo:
            coordination_schema.CandidateRow(row)
        assert "metal_candidates_all.csv" in str(excinfo.value)
        assert "unexpected bogus" in str(excinfo.value)

    def test_confidence_output_requires_its_columns(self, tmp_path: Path) -> None:
        handles = self._handles(tmp_path, confidence=True)
        try:
            with pytest.raises(ValueError):
                OutputWriters(
                    handles.manifest,
                    handles.stats,
                    handles.bonds,
                    handles.candidates,
                    confidence_fh=handles.confidence,
                    confidence_columns=None,
                )
        finally:
            self._close(handles)

    def test_confidence_header_and_counts(self, tmp_path: Path) -> None:
        columns = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
        handles = self._handles(tmp_path, confidence=True)
        writers = OutputWriters(
            handles.manifest,
            handles.stats,
            handles.bonds,
            handles.candidates,
            confidence_fh=handles.confidence,
            confidence_columns=columns,
        )
        writers.write_confidence_rows([])
        assert writers.n_confidence == 0
        writers.write_confidence_rows(
            [dict.fromkeys(columns, ""), dict.fromkeys(columns, "")]
        )
        self._close(handles)
        rows = _read_csv(tmp_path / "confidence.csv")
        assert rows[0] == columns
        assert writers.n_confidence == 2
        assert len(rows) == 3

    def test_scored_confidence_stream_projects_synchronized_inputs(
        self, tmp_path: Path
    ) -> None:
        """A targeted scored resume updates its reusable input rows as well."""
        scored_columns = [
            *confidence_score.CONFIDENCE_INPUT_COLUMNS,
            *confidence_score.ANALYSIS_COLUMNS,
        ]
        handles = self._handles(tmp_path, confidence=True)
        inputs_handle = open(tmp_path / "confidence_inputs.csv", "w", newline="")
        try:
            writers = OutputWriters(
                handles.manifest,
                handles.stats,
                handles.bonds,
                handles.candidates,
                confidence_fh=handles.confidence,
                confidence_columns=scored_columns,
                confidence_inputs_fh=inputs_handle,
            )
            row = {column: f"value-{column}" for column in scored_columns}
            writers.write_confidence_rows([row])
        finally:
            self._close(handles)
            inputs_handle.close()

        scored = _read_csv(tmp_path / "confidence.csv")
        inputs = _read_csv(tmp_path / "confidence_inputs.csv")
        assert scored[0] == scored_columns
        assert inputs[0] == list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
        assert inputs[1] == [
            row[column] for column in confidence_score.CONFIDENCE_INPUT_COLUMNS
        ]

    def test_confidence_row_schema_mismatch_is_rejected(self, tmp_path: Path) -> None:
        columns = list(confidence_score.CONFIDENCE_INPUT_COLUMNS)
        handles = self._handles(tmp_path, confidence=True)
        writers = OutputWriters(
            handles.manifest,
            handles.stats,
            handles.bonds,
            handles.candidates,
            confidence_fh=handles.confidence,
            confidence_columns=columns,
        )
        valid: dict[str, Any] = dict.fromkeys(columns, "")
        malformed = valid.copy()
        malformed.pop(columns[0])
        try:
            with pytest.raises(RuntimeError):
                writers.write_confidence_rows([valid, malformed])
        finally:
            self._close(handles)
        assert writers.n_confidence == 0
        assert _read_csv(tmp_path / "confidence.csv") == [columns]

    def test_manifest_rows_round_trip_through_the_real_projection(
        self, tmp_path: Path
    ) -> None:
        """A written manifest is readable by load_done without reinterpretation."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        writers.write_manifest_row(
            manifest_row(
                _result(
                    "109m",
                    status="ok",
                    retryable=False,
                    n_metals=1,
                    n_bonds=0,
                    n_candidates=0,
                    runtime_s=1.0,
                ),
                False,
                True,
                {},
                {},
            )
        )
        writers.write_manifest_row(
            manifest_row(
                _result("1cll", status="error", retryable=True), False, True, {}, {}
            )
        )
        self._close(handles)
        path = str(tmp_path / "manifest.csv")
        assert resume.load_done(path, bonds_required=True) == {"109m"}
        assert resume.manifest_values_by_id(path, "n_bonds") == {
            "109m": "0",
            "1cll": "",
        }

    def test_each_stream_is_flushed_before_the_manifest_marker(
        self, tmp_path: Path
    ) -> None:
        """An interrupted batch must retain the rows of completed entries."""
        handles = self._handles(tmp_path)
        writers = OutputWriters(*handles)
        writers.write_stats_rows(
            [
                MetalStatsRow.from_output_fields(
                    "109m", "metal", ["x"] * (len(STATS_COLUMNS) - 2)
                )
            ]
        )
        writers.write_bond_rows(
            [
                coordination_schema.BondRow(
                    dict.fromkeys(coordination_schema.BOND_COLUMNS, "")
                )
            ]
        )
        writers.write_manifest_row(
            manifest_row(_result(status="ok"), False, True, {}, {})
        )
        # Read before ``_close``: the rows must already be on disk.
        assert len(_read_csv(tmp_path / "stats.csv")) == 2
        assert len(_read_csv(tmp_path / "bonds.csv")) == 2
        assert len(_read_csv(tmp_path / "manifest.csv")) == 2
        self._close(handles)


class TestScheduleEntries:
    """The work list is reduced by resume first and capped by --max-pdbs after.

    Capping first would hand a resumed run the same already-finished prefix on
    every attempt, filter all of it away, and schedule nothing.
    """

    @staticmethod
    def _args(tmp_path: Path, ids: Sequence[str], **overrides: Any) -> RunConfig:
        id_file = tmp_path / "ids.txt"
        id_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
        fields: dict[str, Any] = {
            "id": None,
            "id_file": str(id_file),
            "pdb_file": None,
            "mtz_file": None,
            "cif_file": None,
            "data_json": None,
            "pdb_redo_root": str(tmp_path),
            "resume": False,
            "retry_partials": False,
            "bonds": True,
            "max_pdbs": None,
        }
        fields.update(overrides)
        return _run_config(**fields)

    @staticmethod
    def _manifest(tmp_path: Path, done_ids: Sequence[str]) -> Path:
        path = tmp_path / "manifest.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for pdb_id in done_ids:
                row = dict.fromkeys(MANIFEST_COLUMNS, "")
                row.update(pdbID=pdb_id, status="ok", n_bonds="0", n_candidates="0")
                writer.writerow(row)
        return path

    def _schedule(self, tmp_path: Path, args: RunConfig) -> tuple[list[str], RunLog]:
        run_log = RunLog(args, "pytest")
        layout = pool.OutputLayout(str(tmp_path))
        ids, _root, _manual = pool.schedule_entries(
            args, layout, str(tmp_path), run_log
        )
        return ids, run_log

    def test_a_capped_resume_reaches_entries_past_the_finished_prefix(
        self, tmp_path: Path
    ) -> None:
        """Two already-done entries must not consume the --max-pdbs budget."""
        all_ids = ["1abc", "2abc", "3abc", "4abc", "5abc"]
        self._manifest(tmp_path, ["1abc", "2abc"])
        (tmp_path / "metal_bonds_all.csv").write_text("", encoding="utf-8")
        (tmp_path / "metal_candidates_all.csv").write_text("", encoding="utf-8")
        args = self._args(tmp_path, all_ids, resume=True, max_pdbs=2)

        ids, run_log = self._schedule(tmp_path, args)
        assert ids == ["3abc", "4abc"]
        assert run_log.details["entries_selected_before_resume"] == 5
        assert run_log.details["entries_scheduled"] == 2

    def test_without_resume_the_cap_takes_the_first_entries(
        self, tmp_path: Path
    ) -> None:
        args = self._args(tmp_path, ["1abc", "2abc", "3abc"], max_pdbs=2)
        ids, _run_log = self._schedule(tmp_path, args)
        assert ids == ["1abc", "2abc"]


class TestWriteEntry:
    """One entry's rows, with the manifest row written last.

    The manifest is the completion marker ``--resume`` reads, so writing it
    first would let an interruption mark an entry whose statistics never
    reached disk as finished.
    """

    class _RecordingWriters:
        """Records the writer calls in order, which is what is under test.

        The real ``OutputWriters`` writes files and remembers nothing, so the
        casts below carry this recorder into the precisely-typed parameter.
        """

        def __init__(self) -> None:
            self.calls: list[str] = []

        def __getattr__(self, name: str) -> Callable[..., None]:
            def record(*args: Any) -> None:
                self.calls.append(name)

            return record

    @staticmethod
    def _args(**overrides: Any) -> RunConfig:
        fields: dict[str, Any] = {"resume": False, "bonds": True}
        fields.update(overrides)
        return _run_config(**fields)

    def test_the_manifest_row_is_written_after_every_data_row(self) -> None:
        writers = self._RecordingWriters()
        plan = pool.ConfidencePlan()
        pool.write_entry(
            _result(),
            self._args(),
            plan,
            cast(OutputWriters, writers),
            None,
            ({}, {}),
        )
        assert writers.calls[-1] == "write_manifest_row"
        assert set(writers.calls[:-1]) == {
            "write_stats_rows",
            "write_bond_rows",
            "write_candidate_rows",
        }

    def test_a_staged_entry_is_registered_for_replacement(self) -> None:
        """Staging replaces rows by id, so every written entry must be listed."""
        # A real ``ResumeStaging`` would create a scratch directory to hold
        # rows nothing here writes; registration touches only the id set.
        staging = cast(resume.ResumeStaging, SimpleNamespace(replacement_ids=set()))
        pool.write_entry(
            _result(),
            self._args(resume=True),
            pool.ConfidencePlan(),
            cast(OutputWriters, self._RecordingWriters()),
            staging,
            ({}, {}),
        )
        assert staging.replacement_ids == {"109m"}


class TestProgressReporter:
    """The heartbeat is throttled differently for a terminal and a log file."""

    class _Stream(io.StringIO):
        def __init__(self, terminal: bool) -> None:
            super().__init__()
            self._terminal = terminal

        # No @override: typing.override arrived in 3.12 and this project still
        # supports 3.11, where importing it would fail at runtime.
        def isatty(self) -> bool:  # type: ignore[explicit-override]
            return self._terminal

    @staticmethod
    def _counts() -> dict[str, int]:
        return {"ok": 1, "partial": 0, "skip": 0, "error": 0}

    def _reporter(
        self, terminal: bool, clock: Callable[[], float]
    ) -> tuple[ProgressReporter, TestProgressReporter._Stream]:
        stream = self._Stream(terminal)
        return ProgressReporter(total=10, stream=stream, clock=clock), stream

    def test_a_redirected_run_renders_far_less_often_than_a_terminal(self) -> None:
        """A run redirected to a file must not grow it by a line per second.

        The counts follow from stepping the clock by ``TERMINAL_INTERVAL_S``:
        a terminal renders every step, a log renders once.
        """
        now = [1000.0]
        for terminal, expected in ((True, 3), (False, 1)):
            reporter, stream = self._reporter(terminal, lambda: now[0])
            for _ in range(3):
                reporter.render(1, self._counts(), 0)
                now[0] += ProgressReporter.TERMINAL_INTERVAL_S
            assert stream.getvalue().count("elapsed=") == expected, (
                f"terminal={terminal} rendered the wrong number of lines"
            )

    def test_force_renders_regardless_of_the_interval(self) -> None:
        reporter, stream = self._reporter(False, lambda: 1000.0)
        reporter.render(1, self._counts(), 0)
        reporter.render(2, self._counts(), 0)
        assert stream.getvalue().count("elapsed=") == 1
        reporter.render(10, self._counts(), 0, force=True, final=True)
        assert stream.getvalue().count("elapsed=") == 2
        assert "[10/10 100.0%]" in stream.getvalue()


class TestRunLog:
    """The written log is the only record of how a finished run behaved."""

    def test_the_log_goes_to_a_subdirectory_of_the_output_by_default(
        self, tmp_path: Path
    ) -> None:
        """One log per invocation accumulates; the four result CSVs do not."""
        args = _run_config(output_dir=str(tmp_path), log_dir=None, workers=1)

        path = RunLog(args, "pytest").write(0)

        assert os.path.dirname(path) == str(tmp_path / runlog.DEFAULT_LOG_DIRNAME)
        assert sorted(os.listdir(tmp_path)) == [runlog.DEFAULT_LOG_DIRNAME]

    def test_an_explicit_log_dir_is_used_as_given(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "shared-logs"
        args = _run_config(
            output_dir=str(tmp_path / "out"), log_dir=str(elsewhere), workers=1
        )

        path = RunLog(args, "pytest").write(0)

        assert os.path.dirname(path) == str(elsewhere)
        assert not (tmp_path / "out").exists(), "the output dir is not created here"

    @staticmethod
    def _log(tmp_path: Path, runtimes: Sequence[float]) -> RunLog:
        run_log = RunLog(
            _run_config(output_dir=str(tmp_path), log_dir=None, workers=1),
            "pytest",
        )
        for index, runtime in enumerate(runtimes):
            run_log.record_entry(
                _result(
                    f"e{index}",
                    status="ok",
                    runtime_s=runtime,
                    n_metals=1,
                    n_bonds=0,
                    n_candidates=0,
                )
            )
        return run_log

    def test_the_slowest_entries_table_is_ordered_slowest_first(
        self, tmp_path: Path
    ) -> None:
        """Reversed, the table still looks plausible while naming the entries
        that mattered least.
        """
        text = open(self._log(tmp_path, [0.5, 9.0, 3.0]).write(0)).read()
        section = text.split("Slowest entries")[1].split("Per-entry results")[0]
        listed = [
            line.split(" | ")[0]
            for line in section.splitlines()
            if line.startswith("e")
        ]
        assert listed == ["e1", "e2", "e0"]

    def test_an_existing_log_is_never_overwritten(self, tmp_path: Path) -> None:
        first = self._log(tmp_path, [1.0]).write(0)
        second = self._log(tmp_path, [2.0]).write(1)
        assert first != second
        assert os.path.isfile(first) and os.path.isfile(second)
        assert "Exit code: 0" in open(first).read()
        assert "Exit code: 1" in open(second).read()


class TestCifToPdb:
    """mmCIF -> analysis PDB conversion must not lose provenance."""

    @staticmethod
    def _standard_cif(tmp_path: Path, name: str = "in.cif") -> str:
        return _write_cif(
            tmp_path / name,
            [
                _cif_atom(
                    1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A"
                ),
                _cif_atom(
                    2, "C", "CA", "GLY", "A", 1, 1, (21.5, 20.0, 20.0), "?", 1, "A"
                ),
                _cif_atom(
                    3, "C", "C", "GLY", "A", 1, 1, (22.0, 21.4, 20.0), ".", 1, "A"
                ),
                _cif_atom(
                    4,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "0.75",
                    1,
                    "B",
                    group="HETATM",
                ),
                _cif_atom(
                    5,
                    "FE",
                    "FE1",
                    "SF4X",
                    "C",
                    3,
                    ".",
                    (5.0, 0.0, 0.0),
                    "1.00",
                    2,
                    "B",
                    group="HETATM",
                ),
            ],
        )

    def test_missing_occupancies_are_blank_not_one(self, tmp_path: Path) -> None:
        """README: '.' and '?' become blank PDB occupancy, never 1.00."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        lines = _pdb_atom_lines(out)
        assert len(lines) == 5
        occupancies = [_occupancy_field(line) for line in lines]
        assert occupancies[1].strip() == ""  # '?'
        assert occupancies[2].strip() == ""  # '.'
        assert occupancies[0].strip() == "1.00"
        assert float(occupancies[3]) == 0.75
        assert float(occupancies[4]) == 1.00

    def test_blanking_occupancy_does_not_shift_the_other_columns(
        self, tmp_path: Path
    ) -> None:
        """Only columns 55-60 change; coordinates, B and element stay put."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        blanked = _pdb_atom_lines(out)[1]
        assert blanked[30:38].strip() == "21.500"
        assert blanked[60:66].strip() == "20.00"
        assert _element_field(blanked) == "C"
        assert blanked[17:20].strip() == "GLY"
        assert len(blanked.rstrip("\n")) >= 78

    def test_absent_occupancy_column_uses_dictionary_default_with_provenance(
        self, tmp_path: Path
    ) -> None:
        cif = _write_cif(
            tmp_path / "noocc.cif",
            [
                _cif_atom(
                    1,
                    "N",
                    "N",
                    "GLY",
                    "A",
                    1,
                    1,
                    (20.0, 20.0, 20.0),
                    None,
                    1,
                    "A",
                    with_occupancy=False,
                ),
                _cif_atom(
                    2,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    None,
                    1,
                    "B",
                    group="HETATM",
                    with_occupancy=False,
                ),
            ],
            header=_CIF_HEADER_NO_OCCUPANCY,
        )
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        for line in _pdb_atom_lines(out):
            assert _occupancy_field(line).strip() == "1.00"

        context = structure_analysis.load_structure("test", out)
        assert context.occupancy_validation_failed is False
        assert context.defaulted_occupancy_atom_count == 2
        assert "occupancy_dictionary_default_applied" in context.warning_codes
        assert structure_analysis.count_deposited_ni(context) == approx(2.0)

    def test_type_symbol_is_written_into_the_pdb_element_field(
        self, tmp_path: Path
    ) -> None:
        """Alchemy reads the element column, never guesses from atom names."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        elements = [_element_field(line) for line in _pdb_atom_lines(out)]
        assert elements == ["N", "C", "C", "ZN", "FE"]

    def test_long_component_ids_are_truncated_but_recorded(
        self, tmp_path: Path
    ) -> None:
        """A >3-character CCD id cannot fit the legacy field, so map it."""
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        with open(out) as handle:
            remarks = [
                line.split()
                for line in handle
                if line.startswith(RESNAME_REMARK_PREFIX)
            ]
        assert len(remarks) == 1
        fields = remarks[0]
        # REMARK 950 ALCHEMY RESNAME <model> <chain> <resnum> <written> <source>
        assert fields[4:9] == ["1", "B", "2", "SF4", "SF4X"]
        written_names = {line[17:20].strip() for line in _pdb_atom_lines(out)}
        assert "SF4" in written_names
        assert "SF4X" not in written_names

    def test_no_mapping_remark_when_every_name_fits(self, tmp_path: Path) -> None:
        cif = _write_cif(
            tmp_path / "short.cif",
            [
                _cif_atom(
                    1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A"
                ),
                _cif_atom(
                    2,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "1.00",
                    1,
                    "B",
                    group="HETATM",
                ),
            ],
        )
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        with open(out) as handle:
            assert not any(line.startswith(RESNAME_REMARK_PREFIX) for line in handle)

    def test_conversion_round_trips_through_load_structure(
        self, tmp_path: Path
    ) -> None:
        """The mapping is reversible: the truncated name is what EDSTATS sees,
        while Alchemy's own output keeps the mmCIF identity.
        """
        cif = self._standard_cif(tmp_path)
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        context = structure_analysis.load_structure("test", out)
        by_atom = {site.atom_name: site for site in context.source_atoms}

        assert by_atom["FE1"].residue_name == "SF4X"
        assert by_atom["FE1"].coordinate_residue_name == "SF4"
        assert by_atom["FE1"].element == "FE"
        assert by_atom["ZN"].residue_name == "ZN"

        assert by_atom["CA"].occupancy_valid is False
        assert by_atom["C"].occupancy_valid is False
        assert by_atom["N"].occupancy == approx(1.0)
        assert by_atom["ZN"].occupancy == approx(0.75)
        assert context.defaulted_occupancy_atom_count == 0
        assert "occupancy_dictionary_default_applied" not in context.warning_codes

    def test_more_than_62_chains_use_reversible_pdb_safe_residue_ids(
        self, tmp_path: Path
    ) -> None:
        """Every source site survives when one-character chain ids run out."""
        source_chains = [f"C{index:02d}" for index in range(62)]
        source_chains.insert(20, "A")
        cif = _write_cif(
            tmp_path / "many-chains.cif",
            [
                _cif_atom(
                    index + 1,
                    "ZN",
                    "ZN",
                    "ZN",
                    chain,
                    index + 1,
                    ".",
                    (float(index), 0.0, 0.0),
                    "1.00",
                    1,
                    chain,
                    group="HETATM",
                )
                for index, chain in enumerate(source_chains)
            ],
        )

        out = conversion.cif_to_pdb(cif, str(tmp_path / "many-chains.pdb"))
        atom_lines = _pdb_atom_lines(out)
        assert len(atom_lines) == 63
        assert {line[21:22] for line in atom_lines} == {"A"}
        with open(out) as handle:
            identity_remarks = [
                line for line in handle if line.startswith(RESIDUE_REMARK_PREFIX)
            ]
        assert len(identity_remarks) == 63

        context = structure_analysis.load_structure("test", out)
        metals = context.metal_atoms({"ZN"}, canonical=True)
        assert len(metals) == 63
        assert {metal.chain_id for metal in metals} == set(source_chains)
        assert {metal.coordinate_chain_id for metal in metals} == {"A"}
        assert {metal.coordinate_resnum for metal in metals} == {
            str(index) for index in range(1, 64)
        }
        assert {metal.chain_index for metal in metals} == {0}
        assert [metal.output_chain_index for metal in metals] == list(range(63))
        assert {metal.output_residue_index for metal in metals} == {0}
        assert context.raw_occupancy_mapping_failed is False
        assert "legacy_pdb_identifiers_packed" in context.warning_codes
        # A source key can coincide with a different residue's packed key, so
        # the two namespaces are indexed separately.
        assert len(context.residues_for_author("ZN", "A", "1")) == 2
        (source_a,) = context.residues_for_source_author("ZN", "A", "1")
        (coordinate_a1,) = context.residues_for_coordinate_author("ZN", "A", "1")
        assert source_a.chain_id == "A"
        assert coordinate_a1.chain_id == source_chains[0]

        # EDSTATS sees the packed identities, while aggregate rows restore the
        # original mmCIF author identifiers.
        stats_path = helpers.write_edstats_for_structure(
            tmp_path / "stats.out", context, metrics={"ZDm": 2.0}
        )
        rows, _ = metal_identification.extract_metal_statistics(
            "test", stats_path, {"ZN"}, set(), structure=context
        )
        assert len(rows) == 63
        assert {row["chain"] for row in rows} == set(source_chains)
        assert {row["fields"][1] for row in rows} == set(source_chains)
        assert {row["resnum"] for row in rows} == {"1"}

        # Source struct_conn partners resolve through the same provenance,
        # without reproducing the packed numbering.
        source = gemmi.read_structure(cif)
        cra = next(item for item in source[0].all() if item.chain.name == "A")
        resolved = declared_connections.analysis_atom_for_partner(context, cra, {})
        assert resolved is not None
        assert resolved.chain_id == "A"
        assert resolved.resnum == "1"

    def test_missing_input_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            conversion.cif_to_pdb(str(tmp_path / "gone.cif"), str(tmp_path / "out.pdb"))

    def test_duplicate_atom_site_id_is_rejected(self, tmp_path: Path) -> None:
        """Serials key the occupancy restoration; duplicates make it ambiguous."""
        cif = _write_cif(
            tmp_path / "dup.cif",
            [
                _cif_atom(
                    1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A"
                ),
                _cif_atom(
                    1,
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "?",
                    1,
                    "B",
                    group="HETATM",
                ),
            ],
        )
        with pytest.raises(ValueError, match="duplicate mmCIF atom_site id"):
            conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))

    def test_code_atom_site_ids_are_mapped_to_generated_pdb_serials(
        self, tmp_path: Path
    ) -> None:
        cif = _write_cif(
            tmp_path / "code-ids.cif",
            [
                _cif_atom(
                    "A1", "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "0.25", 1, "A"
                ),
                _cif_atom(
                    "metal-B",
                    "ZN",
                    "ZN",
                    "ZN",
                    "B",
                    2,
                    ".",
                    (0.0, 0.0, 0.0),
                    "0.75",
                    1,
                    "B",
                    group="HETATM",
                ),
                _cif_atom(
                    "atom-two",
                    "C",
                    "C",
                    "ALA",
                    "A",
                    1,
                    2,
                    (21.5, 20.0, 20.0),
                    "?",
                    2,
                    "A",
                ),
            ],
        )
        out = conversion.cif_to_pdb(cif, str(tmp_path / "out.pdb"))
        lines = _pdb_atom_lines(out)

        # Gemmi groups the two A-chain rows even though the B-chain row was
        # interleaved in atom_site. Generated serials preserve the source row
        # association through that reorder.
        assert [line[12:16].strip() for line in lines] == ["N", "C", "ZN"]
        assert [int(line[6:11]) for line in lines] == [1, 2, 4]
        assert float(_occupancy_field(lines[0])) == approx(0.25)
        assert _occupancy_field(lines[1]).strip() == ""
        assert float(_occupancy_field(lines[2])) == approx(0.75)

    def test_multiple_atom_site_blocks_are_rejected(self, tmp_path: Path) -> None:
        atoms = [
            _cif_atom(1, "N", "N", "GLY", "A", 1, 1, (20.0, 20.0, 20.0), "1.00", 1, "A")
        ]
        text = (
            _CIF_HEADER
            + "".join(line + "\n" for line in atoms)
            + "\n"
            + _CIF_HEADER.replace("data_TEST", "data_SECOND")
            + "".join(line + "\n" for line in atoms)
        )
        path = tmp_path / "two.cif"
        path.write_text(text)
        with pytest.raises(ValueError, match="exactly one block"):
            conversion.cif_to_pdb(str(path), str(tmp_path / "out.pdb"))

    def test_cif_without_atom_records_is_rejected(self, tmp_path: Path) -> None:
        """A metadata-only mmCIF is not a coordinate file."""
        path = tmp_path / "meta.cif"
        path.write_text("data_TEST\n_cell.length_a 60.0\n")
        with pytest.raises(ValueError, match="exactly one block"):
            conversion.cif_to_pdb(str(path), str(tmp_path / "out.pdb"))

    def test_creates_the_destination_directory(self, tmp_path: Path) -> None:
        cif = self._standard_cif(tmp_path)
        dst = tmp_path / "nested" / "deeper" / "out.pdb"
        out = conversion.cif_to_pdb(cif, str(dst))
        assert out == str(dst)
        assert os.path.isfile(out)


class TestResidueConversionRecords:
    """The identity assertions ``cif_to_pdb`` enforces on gemmi's writer."""

    def test_identical_structures_produce_no_records(self) -> None:
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(_BASE_RESIDUES)
        assert conversion.residue_conversion_records(source, converted) == []

    def test_a_renamed_residue_is_recorded_with_its_author_identity(self) -> None:
        """The record must locate the residue the way the PDB reader will."""
        source = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("B", 7, "SF4X", [("FE1", "FE")]),
            ]
        )
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("B", 7, "SF4", [("FE1", "FE")]),
            ]
        )
        assert conversion.residue_conversion_records(source, converted) == [
            (1, "B", "7", "SF4", "SF4X")
        ]

    def test_reordering_is_rejected(self) -> None:
        """EDSTATS joins by position-independent identity only if order holds."""
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(list(reversed(_BASE_RESIDUES)))
        with pytest.raises(ValueError, match="changed residue ordering"):
            conversion.residue_conversion_records(source, converted)

    def test_changed_author_identifiers_are_rejected(self) -> None:
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N"), ("CA", "C")]),
                ("B", 99, "ZN", [("ZN", "ZN")]),
            ]
        )
        with pytest.raises(ValueError, match="ordering|author identifiers"):
            conversion.residue_conversion_records(source, converted)

    def test_changed_atom_membership_is_rejected(self) -> None:
        source = _simple_structure(_BASE_RESIDUES)
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("B", 2, "ZN", [("ZN", "ZN")]),
            ]
        )
        with pytest.raises(ValueError, match="atom membership"):
            conversion.residue_conversion_records(source, converted)

    def test_changed_duplicate_multiplicity_is_rejected(self) -> None:
        source = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("A", 1, "ALA", [("N", "N")]),
            ]
        )
        converted = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
            ]
        )
        with pytest.raises(ValueError, match="ordering|multiplicity"):
            conversion.residue_conversion_records(source, converted)

    def test_the_index_keys_on_model_chain_and_author_resnum(self) -> None:
        """Residues are located by the identifiers EDSTATS also reports."""
        structure = _simple_structure(_BASE_RESIDUES)
        index, order = conversion.residue_index_by_author(structure, "mmCIF")
        assert order == [(0, "A", "1"), (0, "B", "2")]
        assert index[(0, "B", "2")] == [("ZN", (("ZN", "Zn"),))]

    def test_duplicate_author_ids_are_indexed_together_in_order(self) -> None:
        structure = _simple_structure(
            [
                ("A", 1, "GLY", [("N", "N")]),
                ("A", 1, "ALA", [("N", "N")]),
            ]
        )
        index, order = conversion.residue_index_by_author(structure, "mmCIF")
        assert order == [(0, "A", "1"), (0, "A", "1")]
        assert [name for name, _ in index[(0, "A", "1")]] == ["GLY", "ALA"]


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

    def test_single_model_file_is_used_in_place(self, tmp_path: Path) -> None:
        builder = helpers.simple_metal_site(
            "ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)]
        )
        source = builder.write_pdb(tmp_path / "site.pdb")
        dst = tmp_path / "first.pdb"
        path, count = conversion.first_model_pdb(source, str(dst))
        assert path == source
        assert count == 1
        assert not dst.exists()

    def test_extracts_only_the_first_model(self, tmp_path: Path) -> None:
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        path, count = conversion.first_model_pdb(str(source), str(dst))
        assert path == str(dst)
        assert count == 2

        text = dst.read_text()
        atom_lines = _pdb_atom_lines(dst)
        assert len(atom_lines) == 2
        assert [line[30:38].strip() for line in atom_lines] == ["20.000", "21.500"]
        assert "30.000" not in text

    def test_model_wrappers_are_removed(self, tmp_path: Path) -> None:
        """EDSTATS emits a synthetic separator residue for any MODEL wrapper."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        for line in dst.read_text().splitlines():
            assert line[:6].strip().upper() not in ("MODEL", "ENDMDL")
        assert gemmi.read_structure(str(dst)).__len__() == 1

    def test_nummdl_is_dropped_but_crystallographic_header_is_kept(
        self, tmp_path: Path
    ) -> None:
        """NUMMDL would lie about a one-model file; CRYST1 must survive."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        lines = dst.read_text().splitlines()
        assert not any(line.startswith("NUMMDL") for line in lines)
        assert any(line.startswith("CRYST1") for line in lines)
        assert any(line.startswith("HEADER") for line in lines)
        assert any(line.startswith("REMARK   3") for line in lines)

    def test_trailing_records_after_the_first_model_are_dropped(
        self, tmp_path: Path
    ) -> None:
        """Bookkeeping records describe the full ensemble, not this extract."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        lines = dst.read_text().splitlines()
        assert not any(line.startswith("MASTER") for line in lines)
        assert lines[-1] == "END"

    def test_the_extract_preserves_records_byte_for_byte(self, tmp_path: Path) -> None:
        """Extraction is textual so occupancies and identifiers are untouched."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "first.pdb"
        conversion.first_model_pdb(str(source), str(dst))
        source_atom_lines = [
            line for line in _MULTI_MODEL_PDB.splitlines() if line.startswith("ATOM")
        ]
        assert [
            line.rstrip("\n") for line in _pdb_atom_lines(dst)
        ] == source_atom_lines[:2]

    def test_creates_the_destination_directory(self, tmp_path: Path) -> None:
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        dst = tmp_path / "nested" / "first.pdb"
        path, _ = conversion.first_model_pdb(str(source), str(dst))
        assert os.path.isfile(path)

    def test_reported_model_count_is_the_source_ensemble_size(
        self, tmp_path: Path
    ) -> None:
        """The manifest's input_model_count describes the deposited file."""
        source = tmp_path / "multi.pdb"
        source.write_text(_MULTI_MODEL_PDB)
        _, count = conversion.first_model_pdb(str(source), str(tmp_path / "first.pdb"))
        assert count == 2
        assert len(gemmi.read_structure(str(source))) == count


class TestOutputDirectoryLock:
    def test_windows_backend_uses_a_nonblocking_lock_outside_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, int, int]] = []

        class FakeMsvcrt:
            LK_UNLCK = 0
            LK_NBLCK = 2

            @staticmethod
            def locking(descriptor: int, mode: int, size: int) -> None:
                position = os.lseek(descriptor, 0, os.SEEK_CUR)
                calls.append((mode, position, size))

        path = tmp_path / "lock"
        path.write_text("metadata", encoding="utf-8")
        monkeypatch.setattr(output_lock, "_IS_WINDOWS", True)
        monkeypatch.setattr(output_lock, "_platform_lock", FakeMsvcrt)

        with open(path, "r+", encoding="utf-8") as handle:
            output_lock.acquire_file_lock(handle)
            assert handle.tell() == 0
            output_lock.release_file_lock(handle)
            assert handle.tell() == 0

        assert calls == [
            (FakeMsvcrt.LK_NBLCK, output_lock.WINDOWS_LOCK_OFFSET, 1),
            (FakeMsvcrt.LK_UNLCK, output_lock.WINDOWS_LOCK_OFFSET, 1),
        ]

    def test_lock_path_symlink_is_refused_without_touching_its_target(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        target = tmp_path / "important.txt"
        target.write_text("important data\n", encoding="utf-8")
        lock_path = output_dir / output_lock.LOCK_FILENAME
        try:
            lock_path.symlink_to(target)
        except OSError as exc:
            pytest.skip(f"this Windows account cannot create symlinks: {exc}")

        with pytest.raises(
            output_lock.OutputDirectoryLockError,
            match="symbolic link|unsafe lock paths are refused",
        ):
            with output_lock.OutputDirectoryLock(str(output_dir), "unsafe link"):
                pass

        assert lock_path.is_symlink()
        assert target.read_text(encoding="utf-8") == "important data\n"

    def test_multiply_linked_lock_is_refused_without_touching_its_target(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        target = tmp_path / "important.txt"
        target.write_text("important data\n", encoding="utf-8")
        lock_path = output_dir / output_lock.LOCK_FILENAME
        os.link(target, lock_path)

        with pytest.raises(
            output_lock.OutputDirectoryLockError, match="multiple hard links"
        ):
            with output_lock.OutputDirectoryLock(str(output_dir), "unsafe hard link"):
                pass

        assert target.read_text(encoding="utf-8") == "important data\n"

    def test_second_owner_fails_with_first_owners_details(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with output_lock.OutputDirectoryLock(str(output_dir), "alchemy first"):
            with pytest.raises(output_lock.OutputDirectoryBusyError) as caught:
                with output_lock.OutputDirectoryLock(str(output_dir), "alchemy second"):
                    pass

        message = str(caught.value)
        assert str(output_dir) in message
        assert f"pid={os.getpid()}" in message
        assert "command=alchemy first" in message
        assert (output_dir / output_lock.LOCK_FILENAME).is_file()

        # The stable file remains, but the kernel lease is reusable.
        with output_lock.OutputDirectoryLock(str(output_dir), "alchemy third"):
            pass

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
    def test_process_exit_releases_the_lock_without_deleting_its_file(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        ready_read, ready_write = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(ready_read)
            with output_lock.OutputDirectoryLock(str(output_dir), "crashing owner"):
                os.write(ready_write, b"1")
                os._exit(0)
        os.close(ready_write)
        try:
            assert os.read(ready_read, 1) == b"1"
        finally:
            os.close(ready_read)
            os.waitpid(pid, 0)

        with output_lock.OutputDirectoryLock(str(output_dir), "recovery run"):
            pass

    def test_rejected_run_does_not_sweep_or_replace_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            pool, "resolve_ccp4_environment", _resolved_ccp4_environment
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest = output_dir / "manifest.csv"
        manifest.write_bytes(b"existing manifest\n")
        scratch = output_lock.create_owned_scratch_directory(
            str(output_dir), prefix=".alchemy-109m-", kind="entry"
        )
        id_file = tmp_path / "ids.txt"
        id_file.write_text("109m\n", encoding="utf-8")

        with output_lock.OutputDirectoryLock(str(output_dir), "active batch"):
            exit_code = main.main(
                [
                    "--id-file",
                    str(id_file),
                    "--output-dir",
                    str(output_dir),
                    "--pdb-redo-root",
                    str(tmp_path / "absent-mirror"),
                    "--pdb-redo-cache",
                    str(tmp_path / "cache"),
                ]
            )

        assert exit_code == 1
        assert manifest.read_bytes() == b"existing manifest\n"
        assert os.path.isdir(scratch)
        error = capsys.readouterr().err
        assert "already in use by another Alchemy run" in error
        assert "active batch" in error


class TestLeakedWorkDirectorySweep:
    """The startup sweep removes only disposable scratch owned by Alchemy."""

    def test_removes_per_entry_and_staging_directories(self, tmp_path: Path) -> None:
        """Both scratch shapes are swept, with their contents.

        A per-entry directory is otherwise removed only on the normal
        completion path, and holds that entry's maps.
        """
        entry: str | Path = output_lock.create_owned_scratch_directory(
            str(tmp_path), prefix=".alchemy-109m-", kind="entry"
        )
        entry = tmp_path / os.path.basename(entry)
        (entry / "2mFo-DFc.map").write_text("stale", encoding="utf-8")
        staging: str | Path = output_lock.create_owned_scratch_directory(
            str(tmp_path), prefix=".alchemy-resume-", kind="resume"
        )
        staging = tmp_path / os.path.basename(staging)
        (staging / "manifest.csv").write_text("stale", encoding="utf-8")

        removed = pool.sweep_owned_scratch_dirs(str(tmp_path))

        assert removed == 2
        assert sorted(os.listdir(tmp_path)) == []

    def test_leaves_real_output_alone(self, tmp_path: Path) -> None:
        """Names alone never authorize deleting output-directory contents.

        The unmarked directory has the exact shape of legacy entry scratch;
        it still belongs to the user as far as the conservative sweep knows.
        """
        (tmp_path / "manifest.csv").write_text("keep", encoding="utf-8")
        (tmp_path / "alchemy_run_20260101.log").write_text("keep", encoding="utf-8")
        (tmp_path / "109m").mkdir()  # a user directory named for an id
        (tmp_path / ".alchemyrc").write_text("keep", encoding="utf-8")
        (tmp_path / ".alchemy-109m-unmarked").mkdir()

        assert pool.sweep_owned_scratch_dirs(str(tmp_path)) == 0
        assert sorted(os.listdir(tmp_path)) == [
            ".alchemy-109m-unmarked",
            ".alchemyrc",
            "109m",
            "alchemy_run_20260101.log",
            "manifest.csv",
        ]

    def test_preserved_scratch_is_not_swept(self, tmp_path: Path) -> None:
        kept = output_lock.create_owned_scratch_directory(
            str(tmp_path),
            prefix=".alchemy-109m-",
            kind="entry",
            preserve=True,
        )

        assert pool.sweep_owned_scratch_dirs(str(tmp_path)) == 0
        assert os.path.isdir(kept)

    def test_symlink_is_not_followed_even_if_its_target_is_marked(
        self, tmp_path: Path
    ) -> None:
        target_root = tmp_path / "elsewhere"
        target_root.mkdir()
        target = output_lock.create_owned_scratch_directory(
            str(target_root), prefix=".alchemy-109m-", kind="entry"
        )
        link = tmp_path / ".alchemy-109m-link"
        link.symlink_to(target, target_is_directory=True)

        assert pool.sweep_owned_scratch_dirs(str(tmp_path)) == 0
        assert link.is_symlink()
        assert os.path.isdir(target)

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert pool.sweep_owned_scratch_dirs(str(tmp_path / "absent")) == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions required")
def test_unwritable_output_dir_exits_cleanly_naming_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A read-only destination exits like every other unusable input.

    CCP4 resolution is stubbed because it runs before the directory is created,
    and would otherwise fail on a machine with no CCP4 installed.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")

    monkeypatch.setattr(pool, "resolve_ccp4_environment", _resolved_ccp4_environment)
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    output_dir = parent / "output"
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    try:
        exit_code = main.main(
            [
                "--id-file",
                str(id_file),
                "--output-dir",
                str(output_dir),
                "--pdb-redo-root",
                str(tmp_path / "absent-mirror"),
                "--pdb-redo-cache",
                str(tmp_path / "cache"),
            ]
        )
    finally:
        parent.chmod(0o700)

    # 1 is the driver's code for a fixable failure; argparse owns 2.
    assert exit_code == 1
    message = capsys.readouterr().err
    assert str(output_dir) in message, message
    assert "Traceback" not in message


@pytest.mark.skipif(os.name != "posix", reason="uses a POSIX-only stub env")
def test_a_run_sweeps_leaked_scratch_before_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep is wired into the driver, not merely available to it.

    Driven through ``main.main`` so that deleting the call site fails the test.
    The run itself fails for want of a mirror: sweeping happens at startup, so
    even a failing run must leave the directory clean.
    """
    monkeypatch.setattr(pool, "resolve_ccp4_environment", _resolved_ccp4_environment)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    leaked: str | Path = output_lock.create_owned_scratch_directory(
        str(output_dir), prefix=".alchemy-109m-", kind="entry"
    )
    leaked = output_dir / os.path.basename(leaked)
    (leaked / "2mFo-DFc.map").write_text("stale map bytes", encoding="utf-8")
    id_file = tmp_path / "ids.txt"
    id_file.write_text("109m\n", encoding="utf-8")

    main.main(
        [
            "--id-file",
            str(id_file),
            "--output-dir",
            str(output_dir),
            "--pdb-redo-root",
            str(tmp_path / "absent-mirror"),
            "--pdb-redo-cache",
            str(tmp_path / "cache"),
        ]
    )

    leftovers = sorted(
        name for name in os.listdir(output_dir) if name.startswith(".alchemy-")
    )
    assert leftovers == [], f"the run left scratch behind: {leftovers}"


class TestDensityResultReachesTheResult:
    """The worker must report the density stage's own answers, not its request."""

    @staticmethod
    def density_result(**overrides: Any) -> density.DensityResult:
        fields: dict[str, Any] = {
            "stats_out": "/nonexistent/stats.out",
            "rszd": "/nonexistent/rszd.pdb",
            "fo_map": "/nonexistent/fo.map",
            "df_map": "/nonexistent/df.map",
            "mtz_for_maps": "/nonexistent/entry.mtz",
            "mtzfix_log": "/nonexistent/mtzfix.log",
            "mtzfix_applied": False,
            "timings": {"edstats_s": 12.5},
            "twin_coefficient_normalization_applied": False,
            "twin_coefficient_normalization": None,
            "density_map_scope_requested": "model-envelope",
            "density_map_scope_used": "model-envelope",
            "full_map_bytes": 8192,
            "edstats_map_bytes": 2048,
        }
        fields.update(overrides)
        return density.DensityResult(**fields)

    def _entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: density.DensityResult,
        **cfg_overrides: Any,
    ) -> worker_contracts.EntryResult:
        monkeypatch.setattr(worker, "extract_metal_statistics", _empty_metal_statistics)

        def density_stage(*_args: Any, **_kwargs: Any) -> density.DensityResult:
            return result

        return _manual_entry(tmp_path, monkeypatch, density_stage, **cfg_overrides)

    def test_the_scope_reported_is_the_one_achieved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crop that could not be applied must not be reported as applied.

        Requested and achieved scope are separate fields because they disagree:
        an envelope larger than the map, or one past the cell edge, falls back
        to the full map.
        """
        result = self._entry(
            tmp_path,
            monkeypatch,
            self.density_result(
                density_map_scope_requested="model-envelope",
                density_map_scope_used="full-extent-fallback",
            ),
        )

        assert result.density_map_scope_used == "full-extent-fallback"
        assert result.density_full_map_bytes == 8192
        assert result.density_edstats_map_bytes == 2048

    def test_density_timings_reach_the_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """They are how a run log attributes an entry's cost to a CCP4 program."""
        result = self._entry(
            tmp_path, monkeypatch, self.density_result(timings={"edstats_s": 12.5})
        )

        assert result.timings["edstats_s"] == 12.5
        assert "density_total_s" in result.timings

    def test_a_normalized_twin_entry_is_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The normalization is a real modification and must stay visible."""
        result = self._entry(
            tmp_path,
            monkeypatch,
            self.density_result(
                twin_coefficient_normalization_applied=True,
                twin_coefficient_normalization={"usable_reflections": 10},
            ),
        )

        assert "twin_refmac_coefficients_normalized" in result.warning_codes

    @pytest.mark.parametrize("keep", [False, True])
    def test_keep_intermediates_decides_the_scratch_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: bool
    ) -> None:
        """The scratch directory holds the maps, so only the flag may keep it."""
        self._entry(tmp_path, monkeypatch, self.density_result(), keep=keep)

        scratch = [
            name for name in os.listdir(tmp_path) if name.startswith(".alchemy-")
        ]

        assert bool(scratch) is keep
        if keep:
            marker = tmp_path / scratch[0] / output_lock.SCRATCH_MARKER_FILENAME
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            assert metadata["preserve"] is True


def test_a_loaded_structure_fills_in_the_model_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model fields start unmeasured, like the bond counts, so an entry that
    failed before ``load_structure`` cannot claim a model count; a successful
    load must fill them in.
    """
    monkeypatch.setattr(worker, "extract_metal_statistics", _empty_metal_statistics)

    def density_stage(*_args: Any, **_kwargs: Any) -> density.DensityResult:
        return TestDensityResultReachesTheResult.density_result()

    result = _manual_entry(
        tmp_path,
        monkeypatch,
        density_stage,
    )

    assert result.input_model_count == 1
    assert result.model_analyzed == 1
    assert result.multi_model_structure is False

    row = manifest_row(result, False, True, {}, {})
    assert row["input_model_count"] == 1
    assert row["model_analyzed"] == 1


class TestCcp4TimeoutOutcome:
    """A stalled CCP4 program must be a named, retryable entry outcome.

    The generic ``except Exception`` handler would report
    ``unexpected_processing_error``, hiding a stalled installation behind an
    error code shared with real bugs.
    """

    @staticmethod
    def _entry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> worker_contracts.EntryResult:
        """Run one manual-input entry whose density stage raises ``failure``."""

        def fake_density(*args: Any, **kwargs: Any) -> None:
            raise failure

        return _manual_entry(tmp_path, monkeypatch, fake_density)

    def test_timeout_is_reported_as_a_named_retryable_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="edstats",
                timeout_s=900,
                elapsed_s=900.4,
                log_path=str(tmp_path / "edstats.log"),
                timings={"edstats_s": 900.4},
            ),
        )

        assert result.reason_codes == ["ccp4_tool_timeout"]
        assert result.retryable is True, (
            "the program was killed for running too long and reported nothing "
            "about the entry, so it must stay eligible for retry"
        )
        assert result.status == "partial"
        assert "edstats" in result.error
        assert result.confidence_inputs_missing_reason == "ccp4_tool_timeout"
        # The abandoned attempt still cost time, and the run log reports it.
        assert result.timings.get("edstats_s") == 900.4

    def test_the_partial_log_survives_scratch_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``Ccp4ToolTimeoutError`` names a partial log, but ``process`` deletes
        the scratch directory unless --keep-intermediates is given, so the log
        alone is copied out: the maps beside it can be hundreds of megabytes.
        """
        scratch_log = tmp_path / "scratch_edstats.log"
        scratch_log.write_text("edstats output before the stall\n", encoding="utf-8")

        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="edstats",
                timeout_s=900,
                elapsed_s=900.4,
                log_path=str(scratch_log),
                timings={"edstats_s": 900.4},
            ),
        )

        kept = result.ccp4_timeout_log_path
        assert kept, "the timeout log path must be recorded on the result"
        assert os.path.isfile(kept), (
            f"the retained log is missing at {kept}; the path named in the "
            "timeout message must outlive the scratch directory"
        )
        assert worker.TIMEOUT_LOG_DIRNAME in kept
        with open(kept, encoding="utf-8") as handle:
            assert "before the stall" in handle.read()

    def test_only_the_log_is_kept_not_the_scratch_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "edstats.log").write_text("stalled\n", encoding="utf-8")
        huge_map = scratch / "1abc_fo.map"
        huge_map.write_bytes(b"0" * 4096)

        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="edstats",
                timeout_s=900,
                elapsed_s=900.4,
                log_path=str(scratch / "edstats.log"),
                timings={},
            ),
        )

        kept_dir = os.path.dirname(result.ccp4_timeout_log_path)
        assert os.listdir(kept_dir) == ["1abc_edstats_timeout.log"], (
            "only the log belongs in the retained directory"
        )

    def test_a_missing_log_is_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._entry(
            tmp_path,
            monkeypatch,
            density.Ccp4ToolTimeoutError(
                tool="fft",
                timeout_s=900,
                elapsed_s=900.1,
                log_path=str(tmp_path / "never_written.log"),
                timings={},
            ),
        )

        assert result.ccp4_timeout_log_path == ""
        assert result.reason_codes == ["ccp4_tool_timeout"]
        assert result.retryable is True

    def test_a_validation_failure_remains_terminal_and_distinct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The neighbouring handler keeps its own, non-retryable meaning."""
        result = self._entry(
            tmp_path,
            monkeypatch,
            density.MtzfixValidationError("bad coefficients", timings={}),
        )

        assert result.reason_codes == ["mtzfix_validation_failure"]
        assert result.retryable is False
