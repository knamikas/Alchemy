"""Test the owned-scratch sweep that removes leaked work directories."""

from __future__ import annotations

import os
from pathlib import Path

import scratch


class TestLeakedWorkDirectorySweep:
    """The startup sweep removes only disposable scratch_dir owned by Alchemy."""

    def test_removes_per_entry_and_staging_directories(self, tmp_path: Path) -> None:
        """Both scratch shapes are swept, with their contents.

        A per-entry directory is otherwise removed only on the normal
        completion path, and holds that entry's maps.
        """
        entry: str | Path = scratch.create_owned_scratch_directory(
            str(tmp_path), prefix=".alchemy-109m-", kind="entry"
        )
        entry = tmp_path / os.path.basename(entry)
        (entry / "2mFo-DFc.map").write_text("stale", encoding="utf-8")
        staging: str | Path = scratch.create_owned_scratch_directory(
            str(tmp_path), prefix=".alchemy-resume-", kind="resume"
        )
        staging = tmp_path / os.path.basename(staging)
        (staging / "manifest.csv").write_text("stale", encoding="utf-8")

        removed = scratch.sweep_owned_scratch_directories(str(tmp_path))

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

        assert scratch.sweep_owned_scratch_directories(str(tmp_path)) == 0
        assert sorted(os.listdir(tmp_path)) == [
            ".alchemy-109m-unmarked",
            ".alchemyrc",
            "109m",
            "alchemy_run_20260101.log",
            "manifest.csv",
        ]

    def test_preserved_scratch_is_not_swept(self, tmp_path: Path) -> None:
        kept = scratch.create_owned_scratch_directory(
            str(tmp_path),
            prefix=".alchemy-109m-",
            kind="entry",
            preserve=True,
        )

        assert scratch.sweep_owned_scratch_directories(str(tmp_path)) == 0
        assert os.path.isdir(kept)

    def test_symlink_is_not_followed_even_if_its_target_is_marked(
        self, tmp_path: Path
    ) -> None:
        target_root = tmp_path / "elsewhere"
        target_root.mkdir()
        target = scratch.create_owned_scratch_directory(
            str(target_root), prefix=".alchemy-109m-", kind="entry"
        )
        link = tmp_path / ".alchemy-109m-link"
        link.symlink_to(target, target_is_directory=True)

        assert scratch.sweep_owned_scratch_directories(str(tmp_path)) == 0
        assert link.is_symlink()
        assert os.path.isdir(target)

    def test_marker_of_the_wrong_shape_is_refused_not_fatal(
        self, tmp_path: Path
    ) -> None:
        """The sweep runs before any output is read, so it must never raise."""
        stray = tmp_path / ".alchemy-109m-stray"
        stray.mkdir()
        (stray / scratch.SCRATCH_MARKER_FILENAME).write_text("[1]\n", encoding="utf-8")

        assert scratch.sweep_owned_scratch_directories(str(tmp_path)) == 0
        assert stray.is_dir()

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert scratch.sweep_owned_scratch_directories(str(tmp_path / "absent")) == 0
