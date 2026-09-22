"""Decide whether and how a run calculates scores, and finalize its outputs."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

from typing_extensions import override

from codes import RunMode
from driver.errors import DriverError
from driver.layout import OutputLayout
from driver.review_queue import write_review_queue
from driver.runlog import RunLog
from driver.writers import STATS_COLUMNS
from run_config import RunConfig
from run_logging import logger_for
from score import (
    ANALYSIS_COLUMNS as SCORE_ANALYSIS_COLUMNS,
    REFERENCE_METADATA_FILE,
    SCORE_INPUT_COLUMNS,
    ScoreReference,
    classify_without_reference,
    complete_score_site_count,
    finalize_database_score,
    load_reference as load_score_reference,
    prepare_result_score_inputs,
    score_against_reference,
    validate_scored_reference,
)
from score.reference import (
    DEFAULT_SCORE_REFERENCE_DIR as DEFAULT_SCORE_REFERENCE_DIR,
)
from worker.contracts import EntryResult

if TYPE_CHECKING:
    from driver.dispatch import BatchTally

logger = logger_for(__name__)


def resolve_score_reference_dir(
    layout: OutputLayout, configured_dir: str | None = None
) -> tuple[str | None, tuple[str, ...]]:
    """Find a frozen score reference, honoring an explicit override."""
    candidates: tuple[str, ...]
    if configured_dir is not None:
        candidates = (configured_dir,)
    else:
        candidates = (
            layout.reference_dir,
            DEFAULT_SCORE_REFERENCE_DIR,
        )
    for candidate in candidates:
        metadata_path = os.path.join(candidate, REFERENCE_METADATA_FILE)
        if os.path.isfile(metadata_path):
            return candidate, candidates
    return None, candidates


ScoreMode = Literal["database", "reference", "classification"]

#: Columns of a scores file: the inputs plus the verdict and ranking columns.
SCORED_COLUMNS = (
    *SCORE_INPUT_COLUMNS,
    *SCORE_ANALYSIS_COLUMNS,
)


class FinalizedReference(NamedTuple):
    """What finalizing a database reference produced."""

    rows: int
    scored_rows: int
    cohort: int


def reference_marker(layout: OutputLayout) -> str:
    """The reference metadata file, which is the reference's completion marker."""
    return os.path.join(layout.reference_dir, REFERENCE_METADATA_FILE)


class ScorePlan:
    """Whether this run calculates scores, and against what.

    This base plan is the disabled one: bonds are off, so there is no
    evidence to score. ``plan_score`` returns one of the enabled
    subclasses otherwise, and each step of the driver that depends on the
    mode asks the plan rather than comparing mode names.
    """

    #: ``None`` means scoring is off.
    mode: ClassVar[ScoreMode | None] = None

    @property
    def enabled(self) -> bool:
        """Return whether scoring is enabled."""
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
        """The score stream this run writes; ``None`` when disabled."""
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
        """Score outputs a fresh run in this mode must not leave behind."""
        return (
            layout.score_inputs,
            layout.scores,
            reference_marker(layout),
        )

    def validate_resumed_output(self) -> None:
        """Refuse to extend a score stream this plan cannot continue."""
        return None

    def score_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add this mode's verdicts to prepared score-input rows."""
        return rows

    def finalize(
        self,
        layout: OutputLayout,
        tally: BatchTally,
        run_log: RunLog,
        *,
        score_rows_written: int,
    ) -> list[str]:
        """Complete the score outputs after the batch.

        Returns the lines the operator is shown; the caller prints and
        indents them.
        """
        del layout, tally, run_log, score_rows_written
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


class DatabasePlan(ScorePlan):
    """Stream compact inputs; finalize them into a new reference when complete."""

    mode: ClassVar[ScoreMode | None] = "database"

    def __init__(self, layout: OutputLayout) -> None:
        """Stream to the inputs file the finalization reads back."""
        self._stream_path = layout.score_inputs

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
        return SCORE_INPUT_COLUMNS

    @override
    def stale_outputs(self, layout: OutputLayout) -> tuple[str, ...]:
        return (
            layout.scores,
            reference_marker(layout),
        )

    @override
    def finalize(
        self,
        layout: OutputLayout,
        tally: BatchTally,
        run_log: RunLog,
        *,
        score_rows_written: int,
    ) -> list[str]:
        del score_rows_written
        # Finalize when no recoverable entries remain. Known deterministic exclusions
        # are recorded in reference metadata and do not prevent completion.
        unfinished = tally.recoverable_incompleteness()
        if unfinished:
            run_log.summary.score_status = "not_finalized_incomplete_run"
            run_log.summary.score_recoverable_entries = unfinished
            return [
                f"score inputs were retained, but the database "
                f"reference was not finalized: {unfinished} entr"
                f"{'y' if unfinished == 1 else 'ies'} could still be added by "
                f"--resume (missing inputs, lost workers, or retryable "
                f"processing failures)."
            ]
        permanent = tally.terminal_errors
        if permanent:
            logger.warning(
                "finalizing the score reference with %d permanently "
                "failed entr%s: no retry can add them, and their absence "
                "is recorded in the reference metadata",
                permanent,
                "y" if permanent == 1 else "ies",
            )
        finalized = finalize_database_reference(layout, run_log)
        return [
            f"{finalized.rows} score rows ({finalized.scored_rows} "
            f"scored; reference cohort {finalized.cohort}) -> "
            f"{layout.scores}",
            f"score reference -> {layout.reference_dir}",
        ]

    @override
    def finalize_resumed_inputs(
        self, layout: OutputLayout, run_log: RunLog, *, resume: bool
    ) -> FinalizedReference | None:
        if not (resume and os.path.isfile(layout.score_inputs)):
            return None
        return finalize_database_reference(layout, run_log)


class ReferencePlan(ScorePlan):
    """Score each entry against a frozen reference as it completes."""

    mode: ClassVar[ScoreMode | None] = "reference"

    def __init__(
        self,
        layout: OutputLayout,
        reference: ScoreReference,
        *,
        synchronize_inputs: bool,
    ) -> None:
        """Score into the scores file, optionally extending the inputs stream too."""
        self._stream_path = layout.scores
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
        return SCORED_COLUMNS

    @property
    @override
    def synchronize_inputs(self) -> bool:
        return self._synchronize_inputs

    @override
    def stale_outputs(self, layout: OutputLayout) -> tuple[str, ...]:
        return (layout.score_inputs,)

    @override
    def validate_resumed_output(self) -> None:
        if not os.path.isfile(self._stream_path):
            return
        try:
            validate_scored_reference(self._stream_path, self.reference)
        except (OSError, ValueError, csv.Error) as exc:
            raise DriverError(f"Cannot resume score output: {exc}") from None

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
        score_rows_written: int,
    ) -> list[str]:
        del tally
        cohort_size = self.reference.cohort_size
        run_log.summary.score_status = "scored_against_reference"
        run_log.summary.score_reference_cohort = cohort_size
        run_log.summary.scores_path = layout.scores
        return [
            f"{score_rows_written} score rows compared with "
            f"database cohort {cohort_size} -> "
            f"{layout.scores}"
        ]


class ClassificationPlan(ScorePlan):
    """Emit raw-threshold verdicts without empirical ranking: no reference exists."""

    mode: ClassVar[ScoreMode | None] = "classification"

    def __init__(self, layout: OutputLayout) -> None:
        """Write classifications straight into the scores file."""
        self._stream_path = layout.scores

    @property
    @override
    def stream_path(self) -> str:
        return self._stream_path

    @property
    @override
    def columns(self) -> tuple[str, ...]:
        return SCORED_COLUMNS

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
        score_rows_written: int,
    ) -> list[str]:
        del tally
        run_log.summary.score_status = "classified_without_reference"
        run_log.summary.scores_path = layout.scores
        return [
            f"{score_rows_written} classifications "
            f"(empirical ranking unavailable) -> {layout.scores}"
        ]


def classify_run(args: RunConfig) -> RunMode:
    """Return how the run chose its entries.

    Only the uncapped database mode can build a new score reference.
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


def plan_score(
    args: RunConfig,
    layout: OutputLayout,
    run_mode: RunMode,
    run_log: RunLog,
) -> ScorePlan:
    """Decide this run's scoring mode before any entry is processed."""
    if not args.bonds:
        return ScorePlan()
    if run_mode is RunMode.DATABASE:
        # A full-database run builds its own reference. Warn about an existing-reference
        # option without rejecting commands shared with smaller runs.
        if args.score_reference_dir:
            logger.warning(
                "--score-reference-dir is ignored on an uncapped "
                "full-database run: that run builds the reference later runs "
                "are scored against, so it cannot be scored against an "
                "existing one. Cap the run with --max-pdbs to use %s.",
                args.score_reference_dir,
            )
        return DatabasePlan(layout)

    reference_dir, searched_dirs = resolve_score_reference_dir(
        layout, args.score_reference_dir
    )
    if reference_dir is None:
        logger.info(
            "no frozen score reference is installed, so Alchemy will "
            "emit authoritative PASS/REVIEW/SUSPECT classifications without "
            "empirical ranking scores. Complete an uncapped full-database run "
            "or pass --score-reference-dir to add rankings. (searched: %s)",
            ", ".join(searched_dirs),
        )
        return ClassificationPlan(layout)

    try:
        reference = load_score_reference(reference_dir)
    except (OSError, ValueError, csv.Error) as exc:
        raise DriverError(f"Invalid score reference: {exc}") from None
    run_log.details["score_reference_dir"] = reference_dir
    return ReferencePlan(
        layout,
        reference,
        synchronize_inputs=args.resume and os.path.isfile(layout.score_inputs),
    )


def finalize_database_reference(
    layout: OutputLayout, run_log: RunLog
) -> FinalizedReference:
    """Score the streamed inputs, freeze the reference, and record the outcome.

    The one finalization path, whether the batch just completed or a resumed
    run found nothing left to retry.
    """
    try:
        result = finalize_database_score(
            layout.score_inputs,
            layout.scores,
            layout.reference_dir,
            manifest_path=layout.manifest,
        )
    except (OSError, ValueError, csv.Error) as exc:
        raise DriverError(f"Score finalization failed: {exc}") from None
    finalized = FinalizedReference(
        rows=result.rows,
        scored_rows=result.scored_rows,
        cohort=result.cohort_size,
    )
    summary = run_log.summary
    summary.score_status = "finalized"
    summary.score_rows = finalized.rows
    summary.scored_rows = finalized.scored_rows
    summary.score_reference_cohort = finalized.cohort
    summary.scores_path = layout.scores
    summary.score_reference_path = layout.reference_dir
    return finalized


def score_rows_for(result: EntryResult, plan: ScorePlan) -> list[dict[str, Any]]:
    """Prepare one entry's score rows with the plan's verdicts added."""
    if result.metal_site_limit_exceeded:
        return []
    rows = prepare_result_score_inputs(result.rows, result.bond_rows, STATS_COLUMNS)
    rows = complete_score_site_count(
        rows,
        result.pdb_id,
        result.n_metals,
        result.score_inputs_missing_reason,
    )
    return plan.score_rows(rows)


def finalize_review_queue(layout: OutputLayout, run_log: RunLog) -> int:
    """Regenerate the derived triage view from canonical completed outputs."""
    try:
        rows = write_review_queue(
            layout.scores,
            layout.crystallization_summary,
            layout.review_queue,
            SCORED_COLUMNS,
        )
    except (OSError, ValueError, csv.Error) as exc:
        raise DriverError(f"Review queue finalization failed: {exc}") from None
    run_log.summary.review_queue_rows = rows
    run_log.summary.review_queue_path = layout.review_queue
    return rows
