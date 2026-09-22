"""Report the batch outcome to the operator and the run log."""

from __future__ import annotations

import os

from codes import EntryStatus
from driver import scoring
from driver.dispatch import BatchTally
from driver.layout import OutputLayout
from driver.runlog import RunLog
from driver.writers import OutputWriters
from run_config import RunConfig
from run_logging import logger_for

logger = logger_for(__name__)


def report_batch(
    args: RunConfig,
    layout: OutputLayout,
    plan: scoring.ScorePlan,
    tally: BatchTally,
    writers: OutputWriters,
    run_log: RunLog,
) -> int:
    """Report batch results, finalize eligible score outputs, and return the exit code.

    The plan decides what finalizing means for its mode; this function is the
    one place that prints.
    """
    print(
        f"Done. ok={tally.counts['ok']} partial={tally.counts['partial']} "
        f"skip={tally.counts['skip']} error={tally.counts['error']} "
        f"no_metals={tally.no_metals} "
        f"metal_site_limit_exceeded={tally.metal_site_limit_exceeded}; "
        f"{writers.n_sites} metal/cofactor rows -> {layout.stats}",
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
    exit_code = tally.exit_code(database_run=plan.builds_reference)
    for line in plan.finalize(
        layout, tally, run_log, score_rows_written=writers.n_score
    ):
        print(f"      {line}", flush=True)
    if plan.enabled and os.path.isfile(layout.scores):
        review_rows = scoring.finalize_review_queue(layout, run_log)
        print(
            f"      {review_rows} REVIEW/SUSPECT rows -> {layout.review_queue}",
            flush=True,
        )
    if exit_code:
        logger.warning(
            "completed with incomplete entries: errors=%d "
            "(recoverable=%d, terminal=%d), skips=%d, retryable_partials=%d",
            tally.counts[EntryStatus.ERROR],
            tally.recoverable_errors,
            tally.terminal_errors,
            tally.counts[EntryStatus.SKIP],
            tally.retryable_partials,
        )
    return exit_code
