"""Decide whether and how a run scores confidence, and finalize its outputs."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, Literal

from ccp4_setup import REPO_DIR
from confidence_score import (
    ANALYSIS_COLUMNS as CONFIDENCE_ANALYSIS_COLUMNS,
)
from confidence_score import (
    CONFIDENCE_INPUT_COLUMNS,
    REFERENCE_METADATA_FILE,
    ConfidenceReference,
    classify_without_reference,
    complete_confidence_site_count,
    finalize_database_confidence,
    prepare_result_confidence_inputs,
    score_against_reference,
)
from confidence_score import (
    load_reference as load_confidence_reference,
)
from crystallization_conditions import (
    write_review_queue,
)
from driver.errors import DriverError
from driver.layout import (
    OutputLayout,
)
from driver.runlog import RunLog
from driver.writers import STATS_COLUMNS
from run_config import RunConfig
from run_logging import logger_for
from worker_contracts import EntryResult

logger = logger_for(__name__)


DEFAULT_CONFIDENCE_REFERENCE_DIR = os.path.join(
    REPO_DIR, "src", "data", "confidence_reference"
)


def resolve_confidence_reference_dir(
    output_dir: str, configured_dir: str | None = None
) -> tuple[str | None, tuple[str, ...]]:
    """Find a frozen confidence reference, honoring an explicit override."""
    candidates: tuple[str, ...]
    if configured_dir is not None:
        candidates = (configured_dir,)
    else:
        candidates = (
            os.path.join(output_dir, "confidence_reference"),
            DEFAULT_CONFIDENCE_REFERENCE_DIR,
        )
    for candidate in candidates:
        metadata_path = os.path.join(candidate, REFERENCE_METADATA_FILE)
        if os.path.isfile(metadata_path):
            return candidate, candidates
    return None, candidates


ConfidenceMode = Literal["database", "reference", "classification"]


class ConfidencePlan:
    """Whether this run scores confidence, and against what.

    ``database`` streams the inputs an uncapped full-database run finalizes
    into a new reference; ``reference`` scores against one that already exists;
    ``classification`` emits raw-threshold verdicts without empirical ranking;
    ``None`` means confidence analysis is off because bonds are disabled.
    """

    def __init__(self) -> None:
        """Initialize a disabled confidence-analysis plan."""
        self.mode: ConfidenceMode | None = None
        self.reference: ConfidenceReference | None = None
        self.stream_path: str | None = None
        self.columns: Sequence[str] | None = None
        # A resumed ``reference`` run also extends the inputs stream a prior
        # database run left behind, so the two stay row-for-row aligned.
        self.synchronize_inputs: bool = False

    @property
    def enabled(self) -> bool:
        """Return whether confidence analysis is enabled."""
        return self.mode is not None

    @property
    def output_path(self) -> str:
        """The confidence stream this run writes; set whenever a mode is."""
        if self.stream_path is None:
            raise RuntimeError("confidence output path is not configured")
        return self.stream_path

    @property
    def scored_reference(self) -> ConfidenceReference:
        """The frozen reference a ``reference`` run scores against."""
        if self.reference is None:
            raise RuntimeError("confidence reference is not configured")
        return self.reference


def classify_run(args: RunConfig) -> tuple[str, bool]:
    """Return the run mode and whether this is an uncapped full-database run.

    Only the full-database mode can build a new confidence reference.
    """
    manual_requested = bool(args.pdb_file or args.mtz_file or args.cif_file)
    database_run = (
        not args.id
        and not args.id_file
        and not manual_requested
        and args.max_pdbs is None
    )
    run_mode = (
        "manual"
        if manual_requested
        else "single"
        if args.id
        else "id_file"
        if args.id_file
        else "database"
        if database_run
        else "capped_database"
    )
    return run_mode, database_run


def plan_confidence(
    args: RunConfig,
    layout: OutputLayout,
    database_run: bool,
    run_log: RunLog,
) -> ConfidencePlan:
    """Decide this run's confidence mode before any entry is processed."""
    plan = ConfidencePlan()
    if not args.bonds:
        return plan
    if database_run:
        # A full-database run builds its own reference. Warn about an existing-reference
        # option without rejecting commands shared with smaller runs.
        if args.confidence_reference_dir:
            logger.warning(
                "--confidence-reference-dir is ignored on an uncapped "
                "full-database run: that run builds the reference later runs "
                "are scored against, so it cannot be scored against an "
                "existing one. Cap the run with --max-pdbs to use %s.",
                args.confidence_reference_dir,
            )
        plan.mode = "database"
        plan.stream_path = layout.confidence_inputs
        plan.columns = CONFIDENCE_INPUT_COLUMNS
        return plan

    reference_dir, searched_dirs = resolve_confidence_reference_dir(
        args.output_dir, args.confidence_reference_dir
    )
    if reference_dir is None:
        logger.info(
            "no frozen confidence reference is installed, so Alchemy will "
            "emit authoritative PASS/REVIEW/SUSPECT classifications without "
            "empirical ranking scores. Complete an uncapped full-database run "
            "or pass --confidence-reference-dir to add rankings. (searched: %s)",
            ", ".join(searched_dirs),
        )
        plan.mode = "classification"
        plan.stream_path = layout.confidence_scores
        plan.columns = (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS)
        return plan

    try:
        plan.reference = load_confidence_reference(reference_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError(f"Invalid confidence reference: {exc}") from None
    run_log.details["confidence_reference_dir"] = reference_dir
    plan.mode = "reference"
    plan.stream_path = layout.confidence_scores
    plan.columns = (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS)
    plan.synchronize_inputs = bool(args.resume) and os.path.isfile(
        layout.confidence_inputs
    )
    return plan


def finalize_confidence_reference(layout: OutputLayout) -> tuple[int, int, int]:
    """Score the streamed inputs and freeze the database reference."""
    try:
        return finalize_database_confidence(
            layout.confidence_inputs,
            layout.confidence_scores,
            layout.reference_dir,
            manifest_path=layout.manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError(f"Confidence finalization failed: {exc}") from None


def confidence_rows_for(
    result: EntryResult, plan: ConfidencePlan
) -> list[dict[str, Any]]:
    """Prepare one entry's confidence rows, scoring against a reference if available."""
    if result.metal_site_limit_exceeded:
        return []
    rows = prepare_result_confidence_inputs(
        result.rows, result.bond_rows, STATS_COLUMNS
    )
    rows = complete_confidence_site_count(
        rows,
        result.pdb_id,
        result.n_metals,
        result.confidence_inputs_missing_reason,
    )
    if plan.reference is not None:
        rows = score_against_reference(rows, plan.reference)
    elif plan.mode == "classification":
        rows = classify_without_reference(rows)
    return rows


def finalize_review_queue(layout: OutputLayout) -> int:
    """Regenerate the derived triage view from canonical completed outputs."""
    try:
        return write_review_queue(
            layout.confidence_scores,
            layout.crystallization_summary,
            layout.review_queue,
            (*CONFIDENCE_INPUT_COLUMNS, *CONFIDENCE_ANALYSIS_COLUMNS),
        )
    except (OSError, ValueError) as exc:
        raise DriverError(f"Review queue finalization failed: {exc}") from None
