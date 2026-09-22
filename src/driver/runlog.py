"""Collect entry diagnostics and write a run report with a companion CSV.

Reserve matching output names together to prevent concurrent runs from
overwriting files or splitting the pair across different suffixes.
"""

from __future__ import annotations

import contextlib
import csv
import os
import platform
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, TextIO, cast

from analysis_config import (
    ALTLOC_POLICY,
    MAX_ANALYZED_METAL_SITES,
    MODEL_POLICY,
    SYMMETRY_POLICY,
)
from codes import EntryStatus, ReasonCode
from driver.resources import available_cpu_count, available_memory_bytes
from output_rows import blank_if_unmeasured
from run_config import RunConfig
from worker.contracts import EntryResult

# Keep per-run logs separate from result CSVs and startup cleanup.
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
        with contextlib.suppress(OSError):
            os.unlink(destination_path)
        raise
    try:
        with open(source_path, "rb") as source, destination:
            shutil.copyfileobj(source, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(destination_path)
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


@dataclass(frozen=True)
class EntryDiagnostic:
    """What the run report keeps of one entry once its result rows are written."""

    pdb_id: str
    status: EntryStatus
    retryable: bool
    no_metals: bool
    metal_site_limit_exceeded: bool
    n_metals: int
    #: Blank when the bond stage did not run, so the CSV cannot show a count.
    n_bonds: int | str
    n_candidates: int | str
    runtime_s: float
    timings: Mapping[str, float]
    reason_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]
    status_detail: str
    density_map_scope_used: str
    density_full_map_bytes: int
    density_edstats_map_bytes: int
    memory_estimate_bytes: int | None

    @classmethod
    def of(
        cls, result: EntryResult, memory_estimate_bytes: int | None
    ) -> EntryDiagnostic:
        """Retain diagnostic fields without keeping large result-row payloads."""
        return cls(
            pdb_id=result.pdb_id,
            status=result.status,
            retryable=bool(result.retryable),
            no_metals=bool(result.no_metals),
            metal_site_limit_exceeded=bool(result.metal_site_limit_exceeded),
            n_metals=result.n_metals,
            n_bonds=blank_if_unmeasured(result.n_bonds),
            n_candidates=blank_if_unmeasured(result.n_candidates),
            runtime_s=float(result.runtime_s),
            timings=dict(result.timings),
            reason_codes=tuple(result.reason_codes),
            warning_codes=tuple(result.warning_codes),
            status_detail=str(result.status_detail),
            density_map_scope_used=result.density.density_map_scope_used,
            density_full_map_bytes=result.density.density_full_map_bytes,
            density_edstats_map_bytes=result.density.density_edstats_map_bytes,
            memory_estimate_bytes=memory_estimate_bytes,
        )


@dataclass
class RunSummary:
    """What the batch completed, for the report's "Output files" section.

    Every field is ``None`` until the driver records it. A path field and
    its row count are recorded together when an output stream is closed.
    """

    manifest_path: str | None = None
    metal_sites_path: str | None = None
    metal_rows_written: int | None = None
    #: ``"disabled"`` rather than a path when the bond stage did not run.
    metal_bonds_path: str | None = None
    bond_rows_written: int | None = None
    metal_contact_candidates_path: str | None = None
    candidate_rows_written: int | None = None
    crystallization_conditions_path: str | None = None
    crystallization_condition_rows_written: int | None = None
    crystallization_summary_path: str | None = None
    crystallization_summary_rows_written: int | None = None
    density_context_path: str | None = None
    density_context_rows_written: int | None = None
    #: Rows this run streamed to its score output, whichever file that is.
    score_rows_written: int | None = None
    #: Rows in the finalized scores file, which a resumed database run
    #: inherits from earlier runs; only database finalization records it.
    score_rows: int | None = None
    scores_path: str | None = None
    score_reference_path: str | None = None
    score_status: str | None = None
    scored_rows: int | None = None
    score_reference_cohort: int | None = None
    score_recoverable_entries: int | None = None
    review_queue_path: str | None = None
    review_queue_rows: int | None = None
    resume_staging_recovery_dir: str | None = None
    resume_staging_commit_error: str | None = None
    resume_entries_committed_after_interrupt: int | None = None
    memory_scheduler_worker_overhead_bytes: int | None = None
    memory_scheduler_peak_reserved_bytes: int | None = None
    memory_scheduler_max_active_entries: int | None = None
    memory_scheduler_oversized_entries: int | None = None
    memory_scheduler_pressure_pauses: int | None = None
    memory_scheduler_budget_backoffs: int | None = None
    memory_scheduler_budget_recoveries: int | None = None
    #: ``"unavailable"`` when the host's memory could not be measured.
    memory_scheduler_final_budget_bytes: int | str | None = None
    worker_log_listener_abandoned: bool | None = None
    worker_pool_forced_shutdown: bool | None = None

    #: Fields the report renders on their own labelled lines; every other
    #: recorded field is listed under "Additional completion details".
    RENDERED_BY_NAME = frozenset(
        {
            "manifest_path",
            "metal_sites_path",
            "metal_rows_written",
            "metal_bonds_path",
            "bond_rows_written",
            "metal_contact_candidates_path",
            "candidate_rows_written",
            "crystallization_conditions_path",
            "crystallization_condition_rows_written",
            "crystallization_summary_path",
            "crystallization_summary_rows_written",
            "density_context_path",
            "density_context_rows_written",
            "score_rows_written",
            "score_rows",
            "scores_path",
            "score_reference_path",
            "score_status",
            "scored_rows",
            "score_reference_cohort",
            "review_queue_path",
            "review_queue_rows",
        }
    )

    def score_row_count(self) -> int | None:
        """The row count shown for the scores file.

        The finalized total is authoritative when a database run recorded
        one; otherwise the stream count is all there is.
        """
        if self.score_rows is not None:
            return self.score_rows
        return self.score_rows_written

    def completion_details(self) -> dict[str, object]:
        """Recorded fields that have no labelled line of their own, by name."""
        details: dict[str, object] = {
            field.name: value
            for field in fields(self)
            if field.name not in self.RENDERED_BY_NAME
            and (value := getattr(self, field.name)) is not None
        }
        # A score row count belongs beside its scores file; without one it
        # is reported here so the rows are still accounted for.
        row_count = self.score_row_count()
        if self.scores_path is None and row_count is not None:
            details["score_rows"] = row_count
        # A finalized total that differs from this run's stream count means a
        # resumed run inherited rows; show both so the difference is visible.
        if self.score_rows is not None and self.score_rows_written not in (
            None,
            self.score_rows,
        ):
            details["score_rows_written"] = self.score_rows_written
        return details


class RunLog:
    """Collect run diagnostics and publish the report pair at completion."""

    def __init__(self, args: RunConfig, command: str) -> None:
        """Initialize diagnostics and timing for one driver invocation."""
        self.args = args
        self.command = command
        self.started_at = datetime.now(UTC)
        self.started_monotonic = time.monotonic()
        # Stage-specific diagnostics may contain different value types.
        self.details: dict[str, Any] = {
            "initial_available_memory_bytes": available_memory_bytes(),
        }
        self.summary = RunSummary()
        self.entries: list[EntryDiagnostic] = []
        self.driver_error = ""

    def record_entry(
        self, result: EntryResult, memory_estimate_bytes: int | None = None
    ) -> None:
        """Retain diagnostic fields without keeping large result-row payloads."""
        self.entries.append(EntryDiagnostic.of(result, memory_estimate_bytes))

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
        present = {name for entry in self.entries for name in entry.timings}
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
        for entry in sorted(self.entries, key=lambda item: item.pdb_id.lower()):
            writer.writerow(self._diagnostic_row(entry, timing_columns))

    def _diagnostic_row(
        self, entry: EntryDiagnostic, timing_columns: tuple[str, ...]
    ) -> dict[str, object]:
        """Project one entry onto the diagnostics columns."""
        row: dict[str, object] = {
            "pdbID": entry.pdb_id,
            "status": entry.status,
            "retryable": self._clean(entry.retryable),
            "no_metals": self._clean(entry.no_metals),
            "metal_site_limit_exceeded": self._clean(entry.metal_site_limit_exceeded),
            "runtime_s": f"{entry.runtime_s:.3f}",
            "n_metals": entry.n_metals,
            "n_bonds": entry.n_bonds,
            "n_candidates": entry.n_candidates,
            "density_map_scope": entry.density_map_scope_used,
            "full_map_bytes": entry.density_full_map_bytes,
            "edstats_map_bytes": entry.density_edstats_map_bytes,
            "memory_estimate_bytes": (
                ""
                if entry.memory_estimate_bytes is None
                else entry.memory_estimate_bytes
            ),
            "reason_codes": "|".join(entry.reason_codes),
            "warning_codes": "|".join(entry.warning_codes),
            "status_detail": self._clean(entry.status_detail),
        }
        row.update(
            {
                name: (
                    f"{float(entry.timings[name]):.3f}" if name in entry.timings else ""
                )
                for name in timing_columns
            }
        )
        return row

    def _render(
        self,
        exit_code: int,
        finished_at: datetime,
        elapsed_s: float,
        diagnostics_path: str,
    ) -> str:
        sections = (
            self._header_lines(exit_code, finished_at, elapsed_s, diagnostics_path),
            self._provenance_lines(),
            self._configuration_lines(),
            self._outcome_lines(elapsed_s),
            self._output_lines(),
            self._stage_timing_lines(),
            self._exception_lines(),
            self._slowest_entry_lines(),
            self._diagnostics_lines(diagnostics_path),
        )
        return "\n".join(line for section in sections for line in section)

    def _header_lines(
        self,
        exit_code: int,
        finished_at: datetime,
        elapsed_s: float,
        diagnostics_path: str,
    ) -> list[str]:
        """Run identity, timing, and the host it ran on."""
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
        return lines

    def _provenance_lines(self) -> list[str]:
        """Software versions, reference identities, and the analysis policy."""
        lines: list[str] = []
        lines.extend(
            ["", "Provenance and analysis policy", "------------------------------"]
        )
        provenance_labels = {
            "alchemy_version": "Alchemy version",
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
        return lines

    def _configuration_lines(self) -> list[str]:
        """Invocation options and the execution choices resolved from them."""
        lines: list[str] = []
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
        return lines

    def _outcome_lines(self, elapsed_s: float) -> list[str]:
        """Entry counts, status and code tallies, and throughput."""
        lines: list[str] = []
        status_counts = Counter(entry.status for entry in self.entries)
        reason_counts = Counter(
            reason for entry in self.entries for reason in entry.reason_codes
        )
        warning_counts = Counter(
            warning for entry in self.entries for warning in entry.warning_codes
        )
        retryable_count = sum(entry.retryable for entry in self.entries)
        no_metal_count = sum(entry.no_metals for entry in self.entries)
        metal_site_limit_exceeded_count = sum(
            entry.metal_site_limit_exceeded for entry in self.entries
        )
        map_scope_counts = Counter(
            entry.density_map_scope_used
            for entry in self.entries
            if entry.density_map_scope_used
        )
        total_entry_s = sum(entry.runtime_s for entry in self.entries)
        throughput = len(self.entries) * 60.0 / elapsed_s if elapsed_s > 0 else 0.0

        lines.extend(
            [
                "",
                "Outcome summary",
                "---------------",
                f"Entries completed: {len(self.entries)}",
                "Status counts: "
                + " ".join(
                    f"{status}={status_counts[status]}" for status in EntryStatus
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
        return lines

    def _output_lines(self) -> list[str]:
        """Output files with row counts, scoring status, and leftover details."""
        summary = self.summary
        lines: list[str] = []
        lines.extend(["", "Output files", "------------"])
        outputs: tuple[tuple[str, str | None, int | None], ...] = (
            ("Manifest", summary.manifest_path, None),
            ("Metal sites", summary.metal_sites_path, summary.metal_rows_written),
            ("Bonds", summary.metal_bonds_path, summary.bond_rows_written),
            (
                "Contact candidates",
                summary.metal_contact_candidates_path,
                summary.candidate_rows_written,
            ),
            (
                "Crystallization conditions",
                summary.crystallization_conditions_path,
                summary.crystallization_condition_rows_written,
            ),
            (
                "Crystallization summary",
                summary.crystallization_summary_path,
                summary.crystallization_summary_rows_written,
            ),
            (
                "Density context",
                summary.density_context_path,
                summary.density_context_rows_written,
            ),
            (
                "Scores",
                summary.scores_path,
                summary.score_row_count(),
            ),
            ("Score reference", summary.score_reference_path, None),
            ("Review queue", summary.review_queue_path, summary.review_queue_rows),
        )
        any_output = False
        for label, path, count in outputs:
            if path is None:
                continue
            any_output = True
            count_text = f" ({count} rows)" if count is not None else ""
            lines.append(f"{label}: {self._clean(path)}{count_text}")
        if not any_output:
            lines.append("No output files were completed.")
        if summary.score_status is not None:
            lines.append(f"Scoring status: {self._clean(summary.score_status)}")
        if summary.scored_rows is not None:
            lines.append(f"Rows scored: {self._clean(summary.scored_rows)}")
        if summary.score_reference_cohort is not None:
            lines.append(
                f"Score reference cohort: {self._clean(summary.score_reference_cohort)}"
            )
        details = summary.completion_details()
        if details:
            lines.append("Additional completion details:")
            for name, value in sorted(details.items()):
                lines.append(f"  {name}: {self._detail_value(name, value)}")
        return lines

    def _stage_timing_lines(self) -> list[str]:
        """Per-stage totals summed over entries."""
        lines: list[str] = []
        stage_values: dict[str, list[float]] = {}
        for entry in self.entries:
            for name, value in entry.timings.items():
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
                    entry for entry in self.entries if name in entry.timings
                ]
                max_entry = max(
                    stage_entries, key=lambda entry: float(entry.timings[name])
                )
                lines.append(
                    f"{name} | {len(values)} | {sum(values):.3f} | "
                    f"{sum(values) / len(values):.3f} | {max(values):.3f} | "
                    f"{max_entry.pdb_id}"
                )
        return lines

    def _exception_lines(self) -> list[str]:
        """Policy exclusions and every entry that did not finish ok."""
        lines: list[str] = []
        lines.extend(["", "Exceptions and exclusions", "-------------------------"])
        excluded_entries = [
            entry for entry in self.entries if entry.metal_site_limit_exceeded
        ]
        lines.append(
            f"Policy exclusions above {MAX_ANALYZED_METAL_SITES} metal sites: "
            f"{len(excluded_entries)}"
        )
        if excluded_entries:
            lines.append("pdbID | detected_metal_sites")
            for entry in sorted(
                excluded_entries,
                key=lambda item: (-int(item.n_metals), item.pdb_id.lower()),
            ):
                lines.append(f"{entry.pdb_id} | {entry.n_metals}")

        non_ok_entries = [
            entry for entry in self.entries if entry.status != EntryStatus.OK
        ]
        lines.append(f"Partial, skipped, or failed entries: {len(non_ok_entries)}")
        common_partial_entries = [
            entry
            for entry in non_ok_entries
            if entry.status == EntryStatus.PARTIAL
            and not entry.retryable
            and set(entry.reason_codes) == {ReasonCode.MISSING_FIRST_SPHERE_REFERENCE}
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
            for entry in sorted(notable_entries, key=lambda item: item.pdb_id.lower()):
                lines.append(
                    f"{entry.pdb_id} | {entry.status} | "
                    f"{self._clean(entry.retryable)} | "
                    f"{'|'.join(entry.reason_codes) or '-'} | "
                    f"{self._clean(entry.status_detail) or '-'}"
                )
        return lines

    def _slowest_entry_lines(self) -> list[str]:
        """The twenty longest-running entries."""
        lines: list[str] = []
        lines.extend(["", "Slowest entries", "---------------"])
        if not self.entries:
            lines.append("No entries were processed.")
        else:
            lines.append(
                "pdbID | status | runtime_s | metals | bonds | candidates | reasons"
            )
            for entry in sorted(
                self.entries, key=lambda item: item.runtime_s, reverse=True
            )[:20]:
                lines.append(
                    f"{entry.pdb_id} | {entry.status} | "
                    f"{entry.runtime_s:.2f} | {entry.n_metals} | "
                    f"{entry.n_bonds} | {entry.n_candidates} | "
                    f"{'|'.join(entry.reason_codes) or '-'}"
                )
        return lines

    def _diagnostics_lines(self, diagnostics_path: str) -> list[str]:
        """Where the complete per-entry diagnostics live."""
        lines: list[str] = []
        lines.extend(["", "Entry diagnostics", "-----------------"])
        lines.append(
            "Complete per-entry outcomes, timings, map sizes, memory estimates, "
            f"reasons, warnings, and status details: {diagnostics_path}"
        )
        lines.append("")
        return lines

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
                with contextlib.suppress(OSError):
                    os.unlink(diagnostics_path)
                diagnostics_published = False
                raise
            return log_path
        finally:
            for temporary_path in (temporary_log, temporary_diagnostics):
                if not temporary_path:
                    continue
                with contextlib.suppress(OSError):
                    os.unlink(temporary_path)
            if diagnostics_published and not os.path.lexists(log_path):
                with contextlib.suppress(OSError):
                    os.unlink(diagnostics_path)
            with contextlib.suppress(OSError):
                os.unlink(claim_path)
