"""Test the output-directory lock and how a rejected run behaves."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import resolved_ccp4_environment

import cli
import scratch
from driver import environment, output_lock


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

        with (
            pytest.raises(
                output_lock.OutputDirectoryLockError,
                match="symbolic link|unsafe lock paths are refused",
            ),
            output_lock.OutputDirectoryLock(str(output_dir), "unsafe link"),
        ):
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

        with (
            pytest.raises(
                output_lock.OutputDirectoryLockError, match="multiple hard links"
            ),
            output_lock.OutputDirectoryLock(str(output_dir), "unsafe hard link"),
        ):
            pass

        assert target.read_text(encoding="utf-8") == "important data\n"

    def test_second_owner_fails_with_first_owners_details(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with (
            output_lock.OutputDirectoryLock(str(output_dir), "alchemy first"),
            pytest.raises(output_lock.OutputDirectoryBusyError) as caught,
            output_lock.OutputDirectoryLock(str(output_dir), "alchemy second"),
        ):
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
            environment, "resolve_ccp4_environment", resolved_ccp4_environment
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest = output_dir / "manifest.csv"
        manifest.write_bytes(b"existing manifest\n")
        scratch_dir = scratch.create_owned_scratch_directory(
            str(output_dir), prefix=".alchemy-109m-", kind="entry"
        )
        id_file = tmp_path / "ids.txt"
        id_file.write_text("109m\n", encoding="utf-8")

        with output_lock.OutputDirectoryLock(str(output_dir), "active batch"):
            exit_code = cli.main(
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
        assert os.path.isdir(scratch_dir)
        error = capsys.readouterr().err
        assert "already in use by another Alchemy run" in error
        assert "active batch" in error
