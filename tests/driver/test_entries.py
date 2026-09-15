"""Test entry selection: id files, cache and mirror roots, and scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from helpers import run_config, write_manifest

import cli
import inputs
from driver import entries as driver_entries, errors, layout as driver_layout
from driver.runlog import RunLog
from run_config import RunConfig


def test_an_id_file_accepts_mixed_separators_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text(
        "9myr, 6NLR\n# a comment line\n\n9nxl 1abc   # trailing comment\n",
        encoding="utf-8",
    )

    assert driver_entries.load_ids_from_file(str(path)) == [
        "9myr",
        "6nlr",
        "9nxl",
        "1abc",
    ]


def test_an_id_file_reports_the_line_of_a_bad_id(tmp_path: Path) -> None:
    """A typo in a long ID list must name where it is, not just that it exists."""
    path = tmp_path / "ids.txt"
    path.write_text("9myr\n6nlr\nnot-an-id\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid PDB id .*ids\.txt:3"):
        driver_entries.load_ids_from_file(str(path))


def test_a_missing_id_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="id file not found"):
        driver_entries.load_ids_from_file(str(tmp_path / "absent.txt"))


def test_an_id_file_with_a_byte_order_mark_is_read(tmp_path: Path) -> None:
    """A BOM belongs to the encoding, not to the first id.

    Windows editors and spreadsheet exports add one routinely, and under plain
    utf-8 it was glued to the first token, failing validation with a message
    that blamed the id.
    """
    path = tmp_path / "ids.txt"
    path.write_text("9myr, 6nlr\n", encoding="utf-8-sig")

    assert driver_entries.load_ids_from_file(str(path)) == ["9myr", "6nlr"]


def test_mirror_batch_requires_an_explicit_root(tmp_path: Path) -> None:
    args = cli.parse_args([])
    assert args.pdb_redo_root is None
    with pytest.raises(errors.DriverError, match="Supply --pdb-redo-root"):
        driver_entries.select_entry_ids(args, str(tmp_path / "cache"))


@pytest.mark.parametrize("cached", [False, True])
def test_single_id_uses_cache_without_a_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    cache = str(tmp_path / "cache")
    downloads: list[str] = []

    def populate(pdb_id: str, cache_root: str) -> None:
        entry = Path(inputs.entry_dir_for(cache_root, pdb_id))
        entry.mkdir(parents=True)
        (entry / f"{pdb_id}_final.pdb").write_text("END\n")
        (entry / f"{pdb_id}_final.mtz").write_bytes(b"MTZ ")

    def download(pdb_id: str, cache_root: str) -> None:
        downloads.append(pdb_id)
        populate(pdb_id, cache_root)

    if cached:
        populate("9myr", cache)
    monkeypatch.setattr(inputs, "download_entry_to_cache", download)
    args = cli.parse_args(["--id", "9myr"])
    assert driver_entries.select_entry_ids(args, cache) == (["9myr"], cache, None)
    assert downloads == ([] if cached else ["9myr"])


def test_id_list_uses_cache_as_root_without_a_mirror(tmp_path: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("9myr\n109m\n")
    args = cli.parse_args(["--id-file", str(ids)])
    cache = str(tmp_path / "cache")
    assert driver_entries.select_entry_ids(args, cache) == (
        ["9myr", "109m"],
        cache,
        None,
    )


def test_an_unwritable_cache_is_reported_as_a_driver_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local failure is not a missing entry, and is not a traceback either.

    ``ensure_entry_available`` creates directories and writes files, so an
    unwritable ``--pdb-redo-cache`` raised ``PermissionError`` straight out of
    the driver's pre-flight, past a handler that caught only
    ``FileNotFoundError``.
    """

    def unwritable(pdb_id: str, mirror_root: str, cache_root: str) -> str:
        raise PermissionError(13, "Permission denied", str(cache_root))

    monkeypatch.setattr(driver_entries, "ensure_entry_available", unwritable)
    args = cli.parse_args(["--id", "9myr", "--pdb-redo-root", str(tmp_path / "mirror")])

    with pytest.raises(errors.DriverError) as excinfo:
        driver_entries.select_entry_ids(args, str(tmp_path / "cache"))

    message = str(excinfo.value)
    assert "PermissionError" in message
    assert "not found" not in message, (
        "a local failure must not read as a missing entry"
    )


def test_manual_run_rejects_invalid_explicit_data_json_before_scheduling(
    tmp_path: Path,
) -> None:
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

    with pytest.raises(errors.DriverError, match=r"Invalid --data-json:.*not found"):
        driver_entries.select_entry_ids(args, str(tmp_path / "cache"))


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
        return run_config(**fields)

    def _schedule(self, tmp_path: Path, args: RunConfig) -> tuple[list[str], RunLog]:
        run_log = RunLog(args, "pytest")
        layout = driver_layout.OutputLayout(str(tmp_path))
        ids, _root, _manual = driver_entries.schedule_entries(
            args, layout, str(tmp_path), run_log
        )
        return ids, run_log

    def test_a_capped_resume_reaches_entries_past_the_finished_prefix(
        self, tmp_path: Path
    ) -> None:
        """Two already-done entries must not consume the --max-pdbs budget."""
        all_ids = ["1abc", "2abc", "3abc", "4abc", "5abc"]
        write_manifest(
            tmp_path / "manifest.csv",
            [
                {"pdbID": pdb_id, "status": "ok", "n_bonds": "0", "n_candidates": "0"}
                for pdb_id in ("1abc", "2abc")
            ],
        )
        (tmp_path / "metal_bonds_all.csv").write_text("", encoding="utf-8")
        (tmp_path / "metal_contact_candidates_all.csv").write_text("", encoding="utf-8")
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
