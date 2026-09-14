"""Report the batch outcome to the operator and the run log."""

from __future__ import annotations

import os

from driver import confidence
from driver.confidence import (
    ConfidencePlan,
)
from driver.dispatch import BatchTally
from driver.layout import (
    OutputLayout,
)
from driver.runlog import RunLog
from driver.writers import OutputWriters
from run_config import RunConfig
from run_logging import logger_for

logger = logger_for(__name__)


def report_batch(
    args: RunConfig,
    layout: OutputLayout,
    plan: ConfidencePlan,
    tally: BatchTally,
    writers: OutputWriters,
    run_log: RunLog,
) -> int:
    """Report batch results, finalize eligible confidence outputs, and return the exit code.

    Build a database reference only when no recoverable entries remain.
    """
    print(
        f"Done. ok={tally.counts['ok']} partial={tally.counts['partial']} "
        f"skip={tally.counts['skip']} error={tally.counts['error']} "
        f"no_metals={tally.no_metals} "
        f"metal_site_limit_exceeded={tally.metal_site_limit_exceeded}; "
        f"{writers.n_rows} metal/cofactor rows -> {layout.stats}",
        flush=True,
    )
    if args.bonds:
        print(f"      {writers.n_bonds} bond rows -> {layout.bonds}", flush=True)
        print(
            f"      {writers.n_candidates} candidate rows -> {layout.candidates}",
            flush=True,
        )
    print(
        f"      {writers.n_crystallization_conditions} crystallization condition "
        f"rows -> {layout.crystallization_conditions}",
        flush=True,
    )
    print(
        f"      {writers.n_crystallization_summaries} crystallization summary "
        f"rows -> {layout.crystallization_summary}",
        flush=True,
    )
    print(
        f"      {writers.n_density_contexts} density context rows -> "
        f"{layout.density_context}",
        flush=True,
    )
    database_run = plan.mode == "database"
    exit_code = tally.exit_code(database_run=database_run)
    if plan.mode == "database":
        # Finalize when no recoverable entries remain. Known deterministic exclusions
        # are recorded in reference metadata and do not prevent completion.
        unfinished = tally.recoverable_incompleteness()
        if unfinished == 0:
            permanent = tally.terminal_errors
            if permanent:
                logger.warning(
                    "finalizing the confidence reference with %d permanently "
                    "failed entr%s: no retry can add them, and their absence "
                    "is recorded in the reference metadata",
                    permanent,
                    "y" if permanent == 1 else "ies",
                )
            total, scored, cohort = confidence.finalize_confidence_reference(layout)
            run_log.summary.update(
                confidence_status="finalized",
                confidence_rows=total,
                confidence_scored_rows=scored,
                confidence_reference_cohort=cohort,
                confidence_scores_path=layout.confidence_scores,
                confidence_reference_path=layout.reference_dir,
            )
            print(
                f"      {total} confidence rows ({scored} scored; "
                f"reference cohort {cohort}) -> {layout.confidence_scores}",
                flush=True,
            )
            print(f"      confidence reference -> {layout.reference_dir}", flush=True)
        else:
            run_log.summary["confidence_status"] = "not_finalized_incomplete_run"
            run_log.summary["confidence_recoverable_entries"] = unfinished
            print(
                f"      confidence inputs were retained, but the database "
                f"reference was not finalized: {unfinished} entr"
                f"{'y' if unfinished == 1 else 'ies'} could still be added by "
                f"--resume (missing inputs, lost workers, or retryable "
                f"processing failures).",
                flush=True,
            )
    elif plan.mode == "reference":
        cohort_size = plan.scored_reference.cohort_size
        print(
            f"      {writers.n_confidence} confidence rows compared with "
            f"database cohort {cohort_size} -> "
            f"{layout.confidence_scores}",
            flush=True,
        )
        run_log.summary.update(
            confidence_status="scored_against_reference",
            confidence_reference_cohort=cohort_size,
            confidence_scores_path=layout.confidence_scores,
        )
    elif plan.mode == "classification":
        print(
            f"      {writers.n_confidence} confidence classifications "
            f"(empirical ranking unavailable) -> {layout.confidence_scores}",
            flush=True,
        )
        run_log.summary.update(
            confidence_status="classified_without_reference",
            confidence_rows=writers.n_confidence,
            confidence_scores_path=layout.confidence_scores,
        )
    if plan.enabled and os.path.isfile(layout.confidence_scores):
        review_rows = confidence.finalize_review_queue(layout)
        run_log.summary.update(
            review_queue_rows=review_rows,
            review_queue_path=layout.review_queue,
        )
        print(
            f"      {review_rows} REVIEW/SUSPECT rows -> {layout.review_queue}",
            flush=True,
        )
    if exit_code:
        logger.warning(
            "completed with incomplete entries: errors=%d "
            "(recoverable=%d, terminal=%d), skips=%d, retryable_partials=%d",
            tally.counts["error"],
            tally.recoverable_errors,
            tally.terminal_errors,
            tally.counts["skip"],
            tally.retryable_partials,
        )
    return exit_code
