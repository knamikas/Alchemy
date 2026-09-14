"""Decide whether and how a run scores confidence, and finalize its outputs."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

from typing_extensions import override

from ccp4_setup import REPO_DIR
from codes import RunMode
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
    validate_scored_reference,
)
from confidence_score import (
    load_reference as load_confidence_reference,
)
from driver.errors import DriverError
from driver.layout import (
    OutputLayout,
)
from driver.review_queue import write_review_queue
from driver.runlog import RunLog
from driver.writers import STATS_COLUMNS
from run_config import RunConfig
from run_logging import logger_for
from worker_contracts import EntryResult

if TYPE_CHECKING:
    from driver.dispatch import BatchTally

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

#: Columns of a scores file: the inputs plus the verdict and ranking columns.
SCORED_CONFIDENCE_COLUMNS = (
    *CONFIDENCE_INPUT_COLUMNS,
    *CONFIDENCE_ANALYSIS_COLUMNS,
)


class FinalizedReference(NamedTuple):
    """What finalizing a database reference produced."""

    rows: int
    scored_rows: int
    cohort: int


def reference_marker(layout: OutputLayout) -> str:
    """The reference metadata file, which is the reference's completion marker."""
    return os.path.join(layout.reference_dir, REFERENCE_METADATA_FILE)


class ConfidencePlan:
    """Whether this run scores confidence, and against what.

    This base plan is the disabled one: bonds are off, so there is no
    evidence to score. ``plan_confidence`` returns one of the enabled
    subclasses otherwise, and each step of the driver that depends on the
    mode asks the plan rather than comparing mode names.
    """

    #: ``None`` means confidence analysis is off.
    mode: ClassVar[ConfidenceMode | None] = None

    @property
    def enabled(self) -> bool:
        """Return whether confidence analysis is enabled."""
        return self.mode is not None

    @property
    def builds_reference(self) -> bool:
        """Whether this run builds the database reference.

        Such a run is judged complete despite deterministic exclusions,
        which its reference metadata records.
        """
        return False

    @property
    def stream_path(self) -> str | None:
        """The confidence stream this run writes; ``None`` when disabled."""
        return None

    @property
    def columns(self) -> tuple[str, ...] | None:
        """The header of ``stream_path``; ``None`` when disabled."""
        return None

    @property
    def synchronize_inputs(self) -> bool:
        """Whether the run also extends a prior database run's inputs stream."""
        return False

    def stale_outputs(self, layout: OutputLayout) -> tuple[str, ...]:
        """Confidence outputs a fresh run in this mode must not leave behind."""
        return (
            layout.confidence_inputs,
            layout.confidence_scores,
            reference_marker(layout),
        )

    def validate_resumed_output(self) -> None:
        """Refuse to extend a confidence stream this plan cannot continue."""
        return None

    def score_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add this mode's verdicts to prepared confidence-input rows."""
        return rows

    def finalize(
        self,
        layout: OutputLayout,
        tally: BatchTally,
        run_log: RunLog,
        *,
        confidence_rows_written: int,
    ) -> list[str]:
        """Complete the confidence outputs after the batch.

        Returns the lines the operator is shown; the caller prints them.
        """
        del layout, tally, run_log, confidence_rows_written
        return []

    def finalize_resumed_inputs(
        self, layout: OutputLayout, run_log: RunLog, *, resume: bool
    ) -> FinalizedReference | None:
        """Finalize the reference a resumed run's retained inputs describe.

        ``None`` unless this is a resumed database run whose inputs stream
        already exists: every entry is done, so nothing remains but to score
        the stream and freeze the reference.
        """
        del layout, run_log, resume
        return None


class DatabasePlan(ConfidencePlan):
    """Stream compact inputs; finalize them into a new reference when complete."""

    mode: ClassVar[ConfidenceMode | None] = "database"

    def __init__(self, layout: OutputLayout) -> None:
        """Stream to the inputs file the finalization reads back."""
        self._stream_path = layout.confidence_inputs

    @property
    @override
    def builds_reference(self) -> bool:
        return True

    @property
    @override
    def stream_path(self) -> str:
        return self._stream_path

    @property
    @override
    def columns(self) -> tuple[str, ...]:
        return CONFIDENCE_INPUT_COLUMNS

    @override
    def stale_outputs(self, layout: OutputLayout) -> tuple[str, ...]:
        return (layout.confidence_scores, reference_marker(layout))

    @override
    def finalize(
        self,
        layout: OutputLayout,
        tally: BatchTally,
        run_log: RunLog,
        *,
        confidence_rows_written: int,
    ) -> list[str]:
        del confidence_rows_written
        # Finalize when no recoverable entries remain. Known deterministic exclusions
        # are recorded in reference metadata and do not prevent completion.
        unfinished = tally.recoverable_incompleteness()
        if unfinished:
            run_log.summary.confidence_status = "not_finalized_incomplete_run"
            run_log.summary.confidence_recoverable_entries = unfinished
            return [
                f"      confidence inputs were retained, but the database "
                f"reference was not finalized: {unfinished} entr"
                f"{'y' if unfinished == 1 else 'ies'} could still be added by "
                f"--resume (missing inputs, lost workers, or retryable "
                f"processing failures)."
            ]
        permanent = tally.terminal_errors
        if permanent:
            logger.warning(
                "finalizing the confidence reference with %d permanently "
                "failed entr%s: no retry can add them, and their absence "
                "is recorded in the reference metadata",
                permanent,
                "y" if permanent == 1 else "ies",
            )
        finalized = finalize_database_reference(layout, run_log)
        return [
            f"      {finalized.rows} confidence rows ({finalized.scored_rows} "
            f"scored; reference cohort {finalized.cohort}) -> "
            f"{layout.confidence_scores}",
            f"      confidence reference -> {layout.reference_dir}",
        ]

    @override
    def finalize_resumed_inputs(
        self, layout: OutputLayout, run_log: RunLog, *, resume: bool
    ) -> FinalizedReference | None:
        if not (resume and os.path.isfile(layout.confidence_inputs)):
            return None
        return finalize_database_reference(layout, run_log)


class ReferencePlan(ConfidencePlan):
    """Score each entry against a frozen reference as it completes."""

    mode: ClassVar[ConfidenceMode | None] = "reference"

    def __init__(
        self,
        layout: OutputLayout,
        reference: ConfidenceReference,
        *,
        synchronize_inputs: bool,
    ) -> None:
        """Score into the scores file, optionally extending the inputs stream too."""
        self._stream_path = layout.confidence_scores
        self.reference = reference
        # A resumed ``reference`` run also extends the inputs stream a prior
        # database run left behind, so the two stay row-for-row aligned.
        self._synchronize_inputs = synchronize_inputs

    @property
    @override
    def stream_path(self) -> str:
        return self._stream_path

    @property
    @override
    def columns(self) -> tuple[str, ...]:
        return SCORED_CONFIDENCE_COLUMNS

    @property
    @override
    def synchronize_inputs(self) -> bool:
        return self._synchronize_inputs

    @override
    def stale_outputs(self, layout: OutputLayout) -> tuple[str, ...]:
        return (layout.confidence_inputs,)

    @override
    def validate_resumed_output(self) -> None:
        if not os.path.isfile(self._stream_path):
            return
        try:
            validate_scored_reference(self._stream_path, self.reference)
        except (OSError, ValueError) as exc:
            raise DriverError(f"Cannot resume confidence output: {exc}") from None

    @override
    def score_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return score_against_reference(rows, self.reference)

    @override
    def finalize(
        self,
        layout: OutputLayout,
        tally: BatchTally,
        run_log: RunLog,
        *,
        confidence_rows_written: int,
    ) -> list[str]:
        del tally
        cohort_size = self.reference.cohort_size
        run_log.summary.confidence_status = "scored_against_reference"
        run_log.summary.confidence_reference_cohort = cohort_size
        run_log.summary.confidence_scores_path = layout.confidence_scores
        return [
            f"      {confidence_rows_written} confidence rows compared with "
            f"database cohort {cohort_size} -> "
            f"{layout.confidence_scores}"
        ]


class ClassificationPlan(ConfidencePlan):
    """Emit raw-threshold verdicts without empirical ranking: no reference exists."""

    mode: ClassVar[ConfidenceMode | None] = "classification"

    def __init__(self, layout: OutputLayout) -> None:
        """Write classifications straight into the scores file."""
        self._stream_path = layout.confidence_scores

    @property
    @override
    def stream_path(self) -> str:
        return self._stream_path

    @property
    @override
    def columns(self) -> tuple[str, ...]:
        return SCORED_CONFIDENCE_COLUMNS

    @override
    def score_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return classify_without_reference(rows)

    @override
    def finalize(
        self,
        layout: OutputLayout,
        tally: BatchTally,
        run_log: RunLog,
        *,
        confidence_rows_written: int,
    ) -> list[str]:
        del tally
        run_log.summary.confidence_status = "classified_without_reference"
        run_log.summary.confidence_scores_path = layout.confidence_scores
        return [
            f"      {confidence_rows_written} confidence classifications "
            f"(empirical ranking unavailable) -> {layout.confidence_scores}"
        ]


def classify_run(args: RunConfig) -> RunMode:
    """Return how the run chose its entries.

    Only the uncapped database mode can build a new confidence reference.
    """
    if args.pdb_file or args.mtz_file or args.cif_file:
        return RunMode.MANUAL
    if args.id:
        return RunMode.SINGLE
    if args.id_file:
        return RunMode.ID_FILE
    if args.max_pdbs is None:
        return RunMode.DATABASE
    return RunMode.CAPPED_DATABASE


def plan_confidence(
    args: RunConfig,
    layout: OutputLayout,
    run_mode: RunMode,
    run_log: RunLog,
) -> ConfidencePlan:
    """Decide this run's confidence mode before any entry is processed."""
    if not args.bonds:
        return ConfidencePlan()
    if run_mode is RunMode.DATABASE:
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
        return DatabasePlan(layout)

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
        return ClassificationPlan(layout)

    try:
        reference = load_confidence_reference(reference_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DriverError(f"Invalid confidence reference: {exc}") from None
    run_log.details["confidence_reference_dir"] = reference_dir
    return ReferencePlan(
        layout,
        reference,
        synchronize_inputs=bool(args.resume)
        and os.path.isfile(layout.confidence_inputs),
    )


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


def finalize_database_reference(
    layout: OutputLayout, run_log: RunLog
) -> FinalizedReference:
    """Freeze the database reference and record the outcome in the run report.

    The one finalization path, whether the batch just completed or a resumed
    run found nothing left to retry.
    """
    finalized = FinalizedReference(*finalize_confidence_reference(layout))
    summary = run_log.summary
    summary.confidence_status = "finalized"
    summary.confidence_rows = finalized.rows
    summary.confidence_scored_rows = finalized.scored_rows
    summary.confidence_reference_cohort = finalized.cohort
    summary.confidence_scores_path = layout.confidence_scores
    summary.confidence_reference_path = layout.reference_dir
    return finalized


def confidence_rows_for(
    result: EntryResult, plan: ConfidencePlan
) -> list[dict[str, Any]]:
    """Prepare one entry's confidence rows with the plan's verdicts added."""
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
    return plan.score_rows(rows)


def finalize_review_queue(layout: OutputLayout, run_log: RunLog) -> int:
    """Regenerate the derived triage view from canonical completed outputs."""
    try:
        rows = write_review_queue(
            layout.confidence_scores,
            layout.crystallization_summary,
            layout.review_queue,
            SCORED_CONFIDENCE_COLUMNS,
        )
    except (OSError, ValueError) as exc:
        raise DriverError(f"Review queue finalization failed: {exc}") from None
    run_log.summary.review_queue_rows = rows
    run_log.summary.review_queue_path = layout.review_queue
    return rows
