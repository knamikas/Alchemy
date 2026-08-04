"""The detailed run log written once, at the end of every run.

``_RunLog`` accumulates compact per-entry diagnostics during the batch -- status,
reason codes, stage timings, map sizes -- and renders them into one
human-readable file. It deliberately keeps its own structured state rather than
being a logging handler: the log is a diagnostic artifact of the run, written
whole and atomically, not a stream of records.
"""

import os
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone

from driver.resources import available_cpu_count, available_memory_bytes


from worker import blank_if_unmeasured

#: Where run logs go when ``--log-dir`` is not given. A subdirectory rather
#: than the output directory itself: one log per invocation accumulates, and a
#: directory holding four result CSVs plus a year of logs is one a user has to
#: filter by hand every time they look at it. Also the reason the startup
#: sweep can be narrow -- nothing it might delete lives beside the results.
DEFAULT_LOG_DIRNAME = "logs"


def log_dir_for(args):
    """Return the directory this run writes its log to."""
    return args.log_dir or os.path.join(args.output_dir, DEFAULT_LOG_DIRNAME)


class _RunLog:
    """Collect compact run diagnostics and write one human-readable log."""

    def __init__(self, args, command):
        self.args = args
        self.command = command
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.details = {
            "initial_available_memory_bytes": available_memory_bytes(),
        }
        self.summary = {}
        self.entries = []
        self.driver_error = ""

    def record_entry(self, result):
        """Retain diagnostic fields without keeping large result-row payloads."""
        self.entries.append(
            {
                "pdbID": result.pdb_id,
                "status": result.status,
                "retryable": bool(result.retryable),
                "no_metals": bool(result.no_metals),
                "n_metals": result.n_metals,
                "n_bonds": blank_if_unmeasured(result.n_bonds),
                "n_candidates": blank_if_unmeasured(result.n_candidates),
                "runtime_s": float(result.runtime_s),
                "timings": dict(result.timings),
                "reason_codes": list(result.reason_codes),
                "warning_codes": list(result.warning_codes),
                "error": str(result.error),
                "density_map_scope_used": result.density_map_scope_used,
                "density_full_map_bytes": result.density_full_map_bytes,
                "density_edstats_map_bytes": result.density_edstats_map_bytes,
            }
        )

    @staticmethod
    def _clean(value):
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _counter_text(counter):
        if not counter:
            return "none"
        return ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        )

    def _render(self, exit_code, finished_at, elapsed_s):
        lines = [
            "Alchemy detailed run log",
            "========================",
            f"Started (UTC): {self.started_at.isoformat()}",
            f"Finished (UTC): {finished_at.isoformat()}",
            f"Wall time: {elapsed_s:.3f} s",
            f"Exit code: {exit_code}",
            f"Command: {self.command}",
            "",
            "System",
            "------",
            f"Platform: {platform.platform()}",
            f"Python: {platform.python_version()}",
            f"Available CPUs at startup: {available_cpu_count()}",
        ]
        initial_memory = self.details.get("initial_available_memory_bytes")
        if initial_memory is None:
            lines.append("Available memory at startup: unknown")
        else:
            lines.append(
                f"Available memory at startup: {initial_memory / (1024**3):.2f} GiB"
            )
        final_memory = available_memory_bytes()
        lines.append(
            "Available memory at finish: "
            + (
                f"{final_memory / (1024**3):.2f} GiB"
                if final_memory is not None
                else "unknown"
            )
        )

        lines.extend(["", "Configuration", "-------------"])
        for name, value in sorted(vars(self.args).items()):
            lines.append(f"{name}: {self._clean(value)}")
        for name, value in sorted(self.details.items()):
            if name == "initial_available_memory_bytes":
                continue
            lines.append(f"{name}: {self._clean(value)}")

        status_counts = Counter(entry["status"] for entry in self.entries)
        reason_counts = Counter(
            reason for entry in self.entries for reason in entry["reason_codes"]
        )
        warning_counts = Counter(
            warning for entry in self.entries for warning in entry["warning_codes"]
        )
        retryable_count = sum(entry["retryable"] for entry in self.entries)
        no_metal_count = sum(entry["no_metals"] for entry in self.entries)
        map_scope_counts = Counter(
            entry["density_map_scope_used"]
            for entry in self.entries
            if entry["density_map_scope_used"]
        )
        total_entry_s = sum(entry["runtime_s"] for entry in self.entries)
        throughput = len(self.entries) * 60.0 / elapsed_s if elapsed_s > 0 else 0.0

        lines.extend(
            [
                "",
                "Summary",
                "-------",
                f"Entries returned: {len(self.entries)}",
                f"Status counts: {self._counter_text(status_counts)}",
                f"Retryable entries: {retryable_count}",
                f"Metal-free entries: {no_metal_count}",
                f"Summed entry runtime: {total_entry_s:.3f} s",
                f"Throughput: {throughput:.2f} entries/minute",
                f"Reason codes: {self._counter_text(reason_counts)}",
                f"Warning codes: {self._counter_text(warning_counts)}",
                f"Density map scopes used: {self._counter_text(map_scope_counts)}",
            ]
        )
        for name, value in sorted(self.summary.items()):
            lines.append(f"{name}: {self._clean(value)}")
        if self.driver_error:
            lines.append(f"Driver error: {self._clean(self.driver_error)}")

        stage_values: dict[str, list[float]] = {}
        for entry in self.entries:
            for name, value in entry["timings"].items():
                try:
                    stage_values.setdefault(name, []).append(float(value))
                except (TypeError, ValueError):
                    continue
        lines.extend(["", "Stage timing", "------------"])
        if not stage_values:
            lines.append("No completed stage timings were recorded.")
        else:
            lines.append(
                "Totals sum per-entry measurements; density_total_s contains "
                "its subprocess stages, and parallel totals are not wall time."
            )
            lines.append("stage | entries | total_s | mean_s | max_s | max_entry")
            for name in sorted(stage_values):
                values = stage_values[name]
                stage_entries = [
                    entry for entry in self.entries if name in entry["timings"]
                ]
                max_entry = max(
                    stage_entries, key=lambda entry: float(entry["timings"][name])
                )
                lines.append(
                    f"{name} | {len(values)} | {sum(values):.3f} | "
                    f"{sum(values) / len(values):.3f} | {max(values):.3f} | "
                    f"{max_entry['pdbID']}"
                )

        incomplete_entries = [
            entry for entry in self.entries if entry["status"] != "ok"
        ]
        lines.extend(["", "Incomplete entries", "------------------"])
        if not incomplete_entries:
            lines.append("None.")
        else:
            lines.append("pdbID | status | retryable | reasons | error")
            for entry in incomplete_entries:
                lines.append(
                    f"{entry['pdbID']} | {entry['status']} | "
                    f"{self._clean(entry['retryable'])} | "
                    f"{'|'.join(entry['reason_codes']) or '-'} | "
                    f"{self._clean(entry['error']) or '-'}"
                )

        lines.extend(["", "Slowest entries", "---------------"])
        if not self.entries:
            lines.append("No entries were processed.")
        else:
            lines.append(
                "pdbID | status | runtime_s | metals | bonds | candidates | reasons"
            )
            for entry in sorted(
                self.entries, key=lambda item: item["runtime_s"], reverse=True
            )[:20]:
                lines.append(
                    f"{entry['pdbID']} | {entry['status']} | "
                    f"{entry['runtime_s']:.2f} | {entry['n_metals']} | "
                    f"{entry['n_bonds']} | {entry['n_candidates']} | "
                    f"{'|'.join(entry['reason_codes']) or '-'}"
                )

        lines.extend(["", "Per-entry results", "-----------------"])
        if not self.entries:
            lines.append("No entries were processed.")
        for entry in self.entries:
            timing_text = (
                ",".join(
                    f"{name}={float(value):.3f}"
                    for name, value in sorted(entry["timings"].items())
                )
                or "-"
            )
            lines.append(
                f"{entry['pdbID']} | status={entry['status']} | "
                f"retryable={self._clean(entry['retryable'])} | "
                f"no_metals={self._clean(entry['no_metals'])} | "
                f"runtime_s={entry['runtime_s']:.2f} | "
                f"metals={entry['n_metals']} | bonds={entry['n_bonds']} | "
                f"candidates={entry['n_candidates']} | timings={timing_text} | "
                f"density_map_scope={entry['density_map_scope_used'] or '-'} | "
                f"full_map_bytes={entry['density_full_map_bytes']} | "
                f"edstats_map_bytes={entry['density_edstats_map_bytes']} | "
                f"reasons={'|'.join(entry['reason_codes']) or '-'} | "
                f"warnings={'|'.join(entry['warning_codes']) or '-'} | "
                f"error={self._clean(entry['error']) or '-'}"
            )
        lines.append("")
        return "\n".join(lines)

    def write(self, exit_code):
        """Atomically write the final timestamped log and return its path."""
        directory = log_dir_for(self.args)
        os.makedirs(directory, exist_ok=True)
        finished_at = datetime.now(timezone.utc)
        elapsed_s = time.monotonic() - self.started_monotonic
        run_date = self.started_at.strftime("%Y%m%d")
        log_stem = f"alchemy_run_{run_date}"
        path = os.path.join(directory, f"{log_stem}.log")
        suffix = 2
        while os.path.lexists(path):
            path = os.path.join(directory, f"{log_stem}_{suffix}.log")
            suffix += 1
        handle, temporary_path = tempfile.mkstemp(
            prefix=".alchemy-run-log-", dir=directory, text=True
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as log:
                log.write(self._render(exit_code, finished_at, elapsed_s))
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
        return path
