"""Guard the text encoding of every file Alchemy reads or writes.

Output CSVs, resume readers, and the review queue carry crystallization text
with non-ASCII characters such as the ångström sign. A text-mode open without
an explicit encoding follows the locale, so a Windows or C-locale run would
fail an entry write mid-batch, or read a resumed file back with a different
codec than the one that wrote it.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tools")


def _text_mode_opens_without_encoding(path: Path) -> list[str]:
    """Locations of ``open``/``os.fdopen`` calls that let the locale pick a codec."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_open = isinstance(func, ast.Name) and func.id == "open"
        is_fdopen = (
            isinstance(func, ast.Attribute)
            and func.attr == "fdopen"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        )
        if not (is_open or is_fdopen):
            continue
        mode: ast.expr | None = node.args[1] if len(node.args) > 1 else None
        if mode is None:
            mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
        binary = (
            isinstance(mode, ast.Constant)
            and isinstance(mode.value, str)
            and "b" in mode.value
        )
        if binary or any(kw.arg == "encoding" for kw in node.keywords):
            continue
        violations.append(f"{path.relative_to(REPO_DIR)}:{node.lineno}")
    return violations


def test_every_text_mode_open_names_its_encoding() -> None:
    violations = [
        violation
        for directory in SOURCE_DIRS
        for source in sorted((REPO_DIR / directory).rglob("*.py"))
        for violation in _text_mode_opens_without_encoding(source)
    ]
    assert violations == [], (
        "text-mode opens that follow the locale instead of naming an encoding:\n"
        + "\n".join(violations)
    )


# Runs in a child interpreter whose locale encoding is ASCII, exercising the
# driver's output writer, the resume readers and merge, and the review queue
# with text that only round-trips under an explicit UTF-8 encoding.
_ASCII_LOCALE_SCRIPT = r"""
import contextlib
import csv
import locale
import sys

src, output_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
if "utf" in locale.getencoding().lower().replace("-", ""):
    print("LOCALE_IS_UTF8")
    raise SystemExit(0)

from crystallization_conditions import SUMMARY_COLUMNS, write_review_queue
from driver import pool, resume
from driver.writers import MANIFEST_COLUMNS

layout = pool.OutputLayout(output_dir)
targets = pool.output_targets_for_run(layout, pool.ConfidencePlan())
# Escapes keep the script ASCII: the C-locale child cannot decode argv otherwise.
detail = "resolution 1.8 \u00c5, pH 7.5 \u00b1 0.2"

with contextlib.ExitStack() as handles:
    writers = pool._open_writers(handles, targets, bonds=True, confidence_columns=None)
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(pdbID="1abc", status="ok", retryable="False", status_detail=detail)
    writers.write_manifest_row(row)
assert resume.manifest_values_by_id(layout.manifest, "status_detail") == {"1abc": detail}
assert resume.load_done(layout.manifest) == {"1abc"}

staging = resume.ResumeStaging(output_dir, targets)
retried = detail + " (retried)"
with open(staging.staged.manifest, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(MANIFEST_COLUMNS)
    row["status_detail"] = retried
    writer.writerow([row[name] for name in MANIFEST_COLUMNS])
staging.replacement_ids.add("1abc")
staging.commit(bonds_enabled=True)
staging.discard()
assert resume.manifest_values_by_id(layout.manifest, "status_detail") == {"1abc": retried}

columns = ("pdbID", "metal_element", "alchemy_level")
with open(layout.confidence_scores, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(columns)
    writer.writerow(["1abc", "ZN", "REVIEW"])
with open(layout.crystallization_summary, "w", newline="", encoding="utf-8") as handle:
    summary_writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
    summary_writer.writeheader()
    summary = dict.fromkeys(SUMMARY_COLUMNS, "")
    summary.update(
        pdbID="1abc",
        crystallization_data_status="available",
        crystallization_raw_text=detail,
    )
    summary_writer.writerow(summary)
count = write_review_queue(
    layout.confidence_scores,
    layout.crystallization_summary,
    layout.review_queue,
    columns,
)
assert count == 1, count
with open(layout.review_queue, newline="", encoding="utf-8") as handle:
    queue_rows = list(csv.DictReader(handle))
assert queue_rows[0]["crystallization_raw_text"] == detail, queue_rows
print("OK")
"""


@pytest.mark.skipif(
    sys.platform == "win32", reason="LC_ALL cannot force a non-UTF-8 locale on Windows"
)
def test_outputs_round_trip_non_ascii_text_under_a_non_utf8_locale(
    tmp_path: Path,
) -> None:
    """Every output and resume path must name UTF-8 rather than trust the locale."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LC_", "LANG", "PYTHON"))
    }
    env.update(
        LC_ALL="C",
        LANG="C",
        PYTHONUTF8="0",
        PYTHONCOERCECLOCALE="0",
        PYTHONIOENCODING="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _ASCII_LOCALE_SCRIPT,
            str(REPO_DIR / "src"),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    if "LOCALE_IS_UTF8" in completed.stdout:
        pytest.skip("this interpreter cannot be placed in a non-UTF-8 locale")
    assert completed.stdout.strip().endswith("OK"), completed.stdout
