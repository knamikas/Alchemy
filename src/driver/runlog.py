"""The run report written once, at the end of every run.

``RunLog`` accumulates compact per-entry diagnostics during the batch and
publishes a concise human-readable log alongside a complete CSV diagnostics
table. Matching names are reserved together so concurrent runs cannot split a
report across different suffixes or overwrite an earlier run.
"""

import csv
import os
import platform
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any, TextIO, cast

from codes import ReasonCode
from driver.resources import available_cpu_count, available_memory_bytes
from run_config import RunConfig
from worker_contracts import (
    ALTLOC_POLICY,
    MAX_ANALYZED_METAL_SITES,
    MODEL_POLICY,
    SYMMETRY_POLICY,
    EntryResult,
    blank_if_unmeasured,
)

# A subdirectory rather than the output directory itself: one log accumulates
# per invocation, and the startup sweep never sees them beside the result CSVs.
DEFAULT_LOG_DIRNAME = "logs"

ENTRY_DIAGNOSTIC_BASE_COLUMNS = (
    "pdbID",
    "status",
    "retryable",
    "no_metals",
    "metal_site_limit_exceeded",
    "runtime_s",
    "n_metals",
    "n_bonds",
    "n_candidates",
)

PREFERRED_TIMING_COLUMNS = (
    "input_structure_s",
    "mtzfix_s",
    "twin_coefficient_normalization_s",
    "fft_2fofc_s",
    "mapmask_2fofc_s",
    "fft_fofc_s",
    "mapmask_fofc_s",
    "edstats_s",
    "density_total_s",
    "statistics_extraction_s",
    "bond_analysis_s",
    "cleanup_s",
)

ENTRY_DIAGNOSTIC_TRAILING_COLUMNS = (
    "density_map_scope",
    "full_map_bytes",
    "edstats_map_bytes",
    "memory_estimate_bytes",
    "reason_codes",
    "warning_codes",
    "status_detail",
)

PROVENANCE_DETAIL_KEYS = (
    "alchemy_version",
    "alchemy_commit",
    "gemmi_version",
    "ccp4_version",
    "reference_data_id",
    "analysis_config_id",
    "metal_distances_info_sha256",
    "metallocofactors_id_sha256",
)


def log_dir_for(args: RunConfig) -> str:
    """Return the directory this run writes its log to."""
    log_dir: str | None = args.log_dir
    output_dir: str = args.output_dir
    return log_dir or os.path.join(output_dir, DEFAULT_LOG_DIRNAME)


def _copy_log_exclusively(source_path: str, destination_path: str) -> None:
    """Copy a complete report artifact without overwriting an existing file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(destination_path, flags, 0o600)
    try:
        destination = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(destination_path)
        except OSError:
            pass
        raise
    try:
        with open(source_path, "rb") as source, destination:
            shutil.copyfileobj(source, destination)
    except BaseException:
        try:
            os.unlink(destination_path)
        except OSError:
            pass
        raise
    os.unlink(source_path)


def claim_report_paths(directory: str, stem: str) -> tuple[str, str, str]:
    """Reserve matching names for a log and its entry-diagnostics table.

    The hidden claim prevents concurrent runs from choosing the same suffix
    before either finished artifact exists. Existing files are checked while
    the claim is held so an older standalone log or orphaned diagnostics table
    is never overwritten.
    """
    suffix = 1
    while True:
        base = stem if suffix == 1 else f"{stem}_{suffix}"
        log_path = os.path.join(directory, f"{base}.log")
        diagnostics_path = os.path.join(directory, f"{base}_entries.csv")
        claim_path = os.path.join(directory, f".{base}.claim")
        try:
            descriptor = os.open(
                claim_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except FileExistsError:
            suffix += 1
            continue
        os.close(descriptor)
        if os.path.lexists(log_path) or os.path.lexists(diagnostics_path):
            os.unlink(claim_path)
            suffix += 1
            continue
        return log_path, diagnostics_path, claim_path


def _format_duration(seconds: float) -> str:
    exact = f"{seconds:.3f} s"
    if seconds < 60:
        return exact
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d} ({exact})"


def _format_bytes(value: int) -> str:
    units = ("bytes", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for candidate in units[1:]:
        if abs(amount) < 1024.0:
            break
        amount /= 1024.0
        unit = candidate
    if unit == "bytes":
        return f"{value} bytes"
    return f"{amount:.2f} {unit} ({value} bytes)"


class RunLog:
    """Collect run diagnostics and publish the report pair at completion."""

    def __init__(self, args: RunConfig, command: str) -> None:
        self.args = args
        self.command = command
        self.started_at = datetime.now(UTC)
        self.started_monotonic = time.monotonic()
        # The driver records whatever a stage learned about itself here, so the
        # values are as heterogeneous as the stages that supply them.
        self.details: dict[str, Any] = {
            "initial_available_memory_bytes": available_memory_bytes(),
        }
        self.summary: dict[str, Any] = {}
        self.entries: list[dict[str, Any]] = []
        self.driver_error = ""

    def record_entry(
        self, result: EntryResult, memory_estimate_bytes: int | None = None
    ) -> None:
        """Retain diagnostic fields without keeping large result-row payloads."""
        self.entries.append(
            {
                "pdbID": result.pdb_id,
                "status": result.status,
                "retryable": bool(result.retryable),
                "no_metals": bool(result.no_metals),
                "metal_site_limit_exceeded": bool(result.metal_site_limit_exceeded),
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
                "memory_estimate_bytes": memory_estimate_bytes,
            }
        )

    @staticmethod
    def _clean(value: object) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _counter_text(counter: Mapping[str, int]) -> str:
        if not counter:
            return "none"
        return ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                counter.items(), key=lambda item: (-item[1], item[0])
            )
        )

    @staticmethod
    def _detail_value(name: str, value: object) -> str:
        if name.endswith("_bytes") and isinstance(value, int):
            return _format_bytes(value)
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            counter: dict[str, int] = {}
            for key, count in mapping.items():
                if not isinstance(key, str) or not isinstance(count, int):
                    return RunLog._clean(mapping)
                counter[key] = count
            return RunLog._counter_text(counter)
        return RunLog._clean(value)

    def _timing_columns(self) -> tuple[str, ...]:
        present = {name for entry in self.entries for name in entry["timings"].keys()}
        preferred = tuple(name for name in PREFERRED_TIMING_COLUMNS if name in present)
        return (*preferred, *sorted(present - set(preferred)))

    def _write_entry_diagnostics(self, handle: TextIO) -> None:
        timing_columns = self._timing_columns()
        columns = (
            *ENTRY_DIAGNOSTIC_BASE_COLUMNS,
            *timing_columns,
            *ENTRY_DIAGNOSTIC_TRAILING_COLUMNS,
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry in sorted(self.entries, key=lambda item: item["pdbID"].lower()):
            row: dict[str, object] = {
                "pdbID": entry["pdbID"],
                "status": entry["status"],
                "retryable": self._clean(entry["retryable"]),
                "no_metals": self._clean(entry["no_metals"]),
                "metal_site_limit_exceeded": self._clean(
                    entry["metal_site_limit_exceeded"]
                ),
                "runtime_s": f"{entry['runtime_s']:.3f}",
                "n_metals": entry["n_metals"],
                "n_bonds": entry["n_bonds"],
                "n_candidates": entry["n_candidates"],
                "density_map_scope": entry["density_map_scope_used"],
                "full_map_bytes": entry["density_full_map_bytes"],
                "edstats_map_bytes": entry["density_edstats_map_bytes"],
                "memory_estimate_bytes": (
                    ""
                    if entry["memory_estimate_bytes"] is None
                    else entry["memory_estimate_bytes"]
                ),
                "reason_codes": "|".join(entry["reason_codes"]),
                "warning_codes": "|".join(entry["warning_codes"]),
                "status_detail": self._clean(entry["error"]),
            }
            row.update(
                {
                    name: (
                        f"{float(entry['timings'][name]):.3f}"
                        if name in entry["timings"]
                        else ""
                    )
                    for name in timing_columns
                }
            )
            writer.writerow(row)

    def _render(
        self,
        exit_code: int,
        finished_at: datetime,
        elapsed_s: float,
        diagnostics_path: str,
    ) -> str:
        lines = [
            "Alchemy run report",
            "==================",
            f"Started (UTC): {self.started_at.isoformat()}",
            f"Finished (UTC): {finished_at.isoformat()}",
            f"Elapsed: {_format_duration(elapsed_s)}",
            f"Exit code: {exit_code}",
            f"Command: {self.command}",
            f"Entry diagnostics: {diagnostics_path}",
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

        lines.extend(
            ["", "Provenance and analysis policy", "------------------------------"]
        )
        provenance_labels = {
            "alchemy_version": "Alchemy version",
            "alchemy_commit": "Alchemy commit",
            "gemmi_version": "Gemmi version",
            "ccp4_version": "CCP4 version",
            "reference_data_id": "Reference data ID",
            "analysis_config_id": "Analysis configuration ID",
            "metal_distances_info_sha256": "Metal-distance table SHA-256",
            "metallocofactors_id_sha256": "Metal-cofactor catalog SHA-256",
        }
        for name in PROVENANCE_DETAIL_KEYS:
            lines.append(
                f"{provenance_labels[name]}: "
                f"{self._clean(self.details.get(name, 'unknown'))}"
            )
        lines.extend(
            [
                f"Maximum selected metal sites per entry: {MAX_ANALYZED_METAL_SITES}",
                f"Model policy: {MODEL_POLICY}",
                f"Alternate-conformer policy: {ALTLOC_POLICY}",
                f"Symmetry-contact policy: {SYMMETRY_POLICY}",
                f"Bond analysis enabled: {self._clean(self.args.bonds)}",
                f"Density-map scope requested: {self.args.density_map_scope}",
            ]
        )

        lines.extend(
            [
                "",
                "Input and execution configuration",
                "---------------------------------",
                "Invocation options:",
            ]
        )
        configuration = (
            (field.name, getattr(self.args, field.name))
            for field in fields(self.args)
            if field.name not in {"bonds", "density_map_scope"}
        )
        for name, value in sorted(configuration):
            lines.append(f"  {name}: {self._clean(value)}")
        lines.append("Resolved execution:")
        for name, value in sorted(self.details.items()):
            if (
                name == "initial_available_memory_bytes"
                or name in PROVENANCE_DETAIL_KEYS
            ):
                continue
            lines.append(f"  {name}: {self._detail_value(name, value)}")

        status_counts = Counter(entry["status"] for entry in self.entries)
        reason_counts = Counter(
            reason for entry in self.entries for reason in entry["reason_codes"]
        )
        warning_counts = Counter(
            warning for entry in self.entries for warning in entry["warning_codes"]
        )
        retryable_count = sum(entry["retryable"] for entry in self.entries)
        no_metal_count = sum(entry["no_metals"] for entry in self.entries)
        metal_site_limit_exceeded_count = sum(
            entry["metal_site_limit_exceeded"] for entry in self.entries
        )
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
                "Outcome summary",
                "---------------",
                f"Entries completed: {len(self.entries)}",
                "Status counts: "
                + " ".join(
                    f"{name}={status_counts[name]}"
                    for name in ("ok", "partial", "skip", "error")
                ),
                f"Retryable entries: {retryable_count}",
                f"Metal-free entries: {no_metal_count}",
                f"Policy-excluded entries: {metal_site_limit_exceeded_count}",
                f"Summed entry runtime: {_format_duration(total_entry_s)}",
                f"Throughput: {throughput:.2f} entries/minute",
                f"Reason codes: {self._counter_text(reason_counts)}",
                f"Warning codes: {self._counter_text(warning_counts)}",
                f"Density map scopes used: {self._counter_text(map_scope_counts)}",
            ]
        )
        if self.driver_error:
            lines.append(f"Driver error: {self._clean(self.driver_error)}")

        summary = dict(self.summary)
        if "confidence_rows" not in summary and "confidence_rows_written" in summary:
            summary["confidence_rows"] = summary.pop("confidence_rows_written")
        if summary.get("confidence_rows") == summary.get("confidence_rows_written"):
            summary.pop("confidence_rows_written", None)
        lines.extend(["", "Output files", "------------"])
        output_specs = (
            ("Manifest", "manifest_path", None),
            ("Metal sites", "metal_sites_path", "metal_rows_written"),
            ("Bonds", "metal_bonds_path", "bond_rows_written"),
            (
                "Contact candidates",
                "metal_contact_candidates_path",
                "candidate_rows_written",
            ),
            (
                "Crystallization conditions",
                "crystallization_conditions_path",
                "crystallization_condition_rows_written",
            ),
            (
                "Crystallization summary",
                "crystallization_summary_path",
                "crystallization_summary_rows_written",
            ),
            (
                "Density context",
                "density_context_path",
                "density_context_rows_written",
            ),
            ("Confidence scores", "confidence_scores_path", "confidence_rows"),
            ("Confidence reference", "confidence_reference_path", None),
            ("Review queue", "review_queue_path", "review_queue_rows"),
        )
        any_output = False
        for label, path_key, count_key in output_specs:
            if path_key not in summary:
                continue
            any_output = True
            path = summary.pop(path_key)
            count = summary.pop(count_key, None) if count_key else None
            count_text = f" ({count} rows)" if count is not None else ""
            lines.append(f"{label}: {self._clean(path)}{count_text}")
        if not any_output:
            lines.append("No output files were completed.")
        if "confidence_status" in summary:
            lines.append(
                f"Confidence status: {self._clean(summary.pop('confidence_status'))}"
            )
        if "confidence_scored_rows" in summary:
            lines.append(
                "Confidence rows scored: "
                f"{self._clean(summary.pop('confidence_scored_rows'))}"
            )
        if "confidence_reference_cohort" in summary:
            lines.append(
                "Confidence reference cohort: "
                f"{self._clean(summary.pop('confidence_reference_cohort'))}"
            )
        if summary:
            lines.append("Additional completion details:")
            for name, value in sorted(summary.items()):
                lines.append(f"  {name}: {self._detail_value(name, value)}")

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

        lines.extend(["", "Exceptions and exclusions", "-------------------------"])
        excluded_entries = [
            entry for entry in self.entries if entry["metal_site_limit_exceeded"]
        ]
        lines.append(
            f"Policy exclusions above {MAX_ANALYZED_METAL_SITES} metal sites: "
            f"{len(excluded_entries)}"
        )
        if excluded_entries:
            lines.append("pdbID | detected_metal_sites")
            for entry in sorted(
                excluded_entries,
                key=lambda item: (-int(item["n_metals"]), item["pdbID"].lower()),
            ):
                lines.append(f"{entry['pdbID']} | {entry['n_metals']}")

        non_ok_entries = [entry for entry in self.entries if entry["status"] != "ok"]
        lines.append(f"Partial, skipped, or failed entries: {len(non_ok_entries)}")
        common_partial_entries = [
            entry
            for entry in non_ok_entries
            if entry["status"] == "partial"
            and not entry["retryable"]
            and set(entry["reason_codes"])
            == {ReasonCode.MISSING_FIRST_SPHERE_REFERENCE}
        ]
        if common_partial_entries:
            lines.append(
                "Terminal partials caused only by missing first-sphere references: "
                f"{len(common_partial_entries)} (IDs are in the entry diagnostics)"
            )
        notable_entries = [
            entry for entry in non_ok_entries if entry not in common_partial_entries
        ]
        if not notable_entries:
            lines.append("Other partial, skipped, failed, or retryable entries: none")
        else:
            lines.append("Other partial, skipped, failed, or retryable entries:")
            lines.append("pdbID | status | retryable | reasons | status_detail")
            for entry in sorted(
                notable_entries, key=lambda item: item["pdbID"].lower()
            ):
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

        lines.extend(["", "Entry diagnostics", "-----------------"])
        lines.append(
            "Complete per-entry outcomes, timings, map sizes, memory estimates, "
            f"reasons, warnings, and status details: {diagnostics_path}"
        )
        lines.append("")
        return "\n".join(lines)

    def write(self, exit_code: int) -> str:
        """Write the timestamped run report without overwriting either artifact."""
        directory = log_dir_for(self.args)
        os.makedirs(directory, exist_ok=True)
        finished_at = datetime.now(UTC)
        elapsed_s = time.monotonic() - self.started_monotonic
        run_date = self.started_at.strftime("%Y%m%d")
        log_stem = f"alchemy_run_{run_date}"
        log_path, diagnostics_path, claim_path = claim_report_paths(directory, log_stem)
        temporary_log = ""
        temporary_diagnostics = ""
        diagnostics_published = False
        try:
            diagnostics_handle, temporary_diagnostics = tempfile.mkstemp(
                prefix=".alchemy-run-entries-", dir=directory, text=True
            )
            with os.fdopen(
                diagnostics_handle, "w", encoding="utf-8", newline=""
            ) as diagnostics:
                self._write_entry_diagnostics(diagnostics)

            log_handle, temporary_log = tempfile.mkstemp(
                prefix=".alchemy-run-log-", dir=directory, text=True
            )
            with os.fdopen(log_handle, "w", encoding="utf-8", newline="\n") as log:
                log.write(
                    self._render(exit_code, finished_at, elapsed_s, diagnostics_path)
                )

            # Publishing the diagnostics first ensures a visible log never
            # points readers at a companion table that has not been written.
            _copy_log_exclusively(temporary_diagnostics, diagnostics_path)
            temporary_diagnostics = ""
            diagnostics_published = True
            try:
                _copy_log_exclusively(temporary_log, log_path)
                temporary_log = ""
            except BaseException:
                try:
                    os.unlink(diagnostics_path)
                except OSError:
                    pass
                diagnostics_published = False
                raise
            return log_path
        finally:
            for temporary_path in (temporary_log, temporary_diagnostics):
                if not temporary_path:
                    continue
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
            if diagnostics_published and not os.path.lexists(log_path):
                try:
                    os.unlink(diagnostics_path)
                except OSError:
                    pass
            try:
                os.unlink(claim_path)
            except OSError:
                pass
