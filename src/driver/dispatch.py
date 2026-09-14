"""Run a batch of entries across a worker pool and collect their results.

Assign entries to worker processes as memory permits, hand each result to
the caller as it lands, and tally the batch outcome.

Handle worker crashes and limit shutdown waits so failed workers do not
leave the batch stuck.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from collections.abc import Callable, Collection, Sequence
from multiprocessing import Pool, SimpleQueue
from typing import TYPE_CHECKING, Any

from codes import EntryStatus, ReasonCode
from driver.memory_admission import MemoryAdmission
from driver.progress import ProgressReporter
from driver.resources import (
    GIB,
    WORKER_FIXED_OVERHEAD_BYTES,
    EntryMemoryEstimate,
    MemoryPlan,
    available_memory_bytes,
)
from driver.runlog import RunLog
from run_logging import (
    create_worker_log_queue,
    logger_for,
    start_worker_log_listener,
)
from worker import (
    initialize_worker,
    is_worker_death_result,
    process,
    worker_death_result,
)
from worker_contracts import EntryResult, WorkerConfig

if TYPE_CHECKING:
    # Import the actual Pool and Queue classes for type annotations.
    from logging.handlers import QueueListener
    from multiprocessing.pool import AsyncResult
    from multiprocessing.pool import Pool as WorkerPool
    from multiprocessing.queues import Queue as WorkerLogQueue

# If a worker dies and its entry is unknown, wait this many seconds
# without any new results before marking all unfinished entries as failed.
# Those entries can be retried on a later run.
WORKER_STALL_GRACE_S = 600.0
# Bound clean shutdown before forcefully stopping remaining workers.
WORKER_SHUTDOWN_GRACE_S = 5.0

ResultSink = Callable[[EntryResult], None]
"""Receives each entry's result, in completion order, before it is tallied."""

logger = logger_for(__name__)


def drain_inflight(
    inflight: SimpleQueue[tuple[str, int, str]], assignments: dict[int, str]
) -> None:
    """Apply pending worker notifications to the pid -> entry assignment map."""
    while True:
        try:
            if inflight.empty():
                return
            state, pid, pdb_id = inflight.get()
        except (OSError, EOFError):  # pragma: no cover - pipe torn down
            return
        if state == "start":
            assignments[pid] = pdb_id
        else:
            assignments.pop(pid, None)


def _pool_children(pool: WorkerPool) -> list[Any]:
    """The pool's live worker processes.

    ``Pool`` exposes no public roster; ``_pool`` has been the worker list since
    Python 2 and is read defensively so a change disables death detection
    rather than crashing the batch.
    """
    return [
        child
        for child in getattr(pool, "_pool", ()) or ()
        if getattr(child, "pid", None)
    ]


def dead_worker_pids(pool: WorkerPool, known_pids: set[int]) -> set[int]:
    """Return worker pids that have disappeared since the last check."""
    current = {child.pid for child in _pool_children(pool)}
    if not current:
        return set()
    dead = known_pids - current
    known_pids.clear()
    known_pids.update(current)
    return dead


def _signal_worker_process_group(pid: int, sig: int) -> None:
    """Signal a worker group even if its leader has already exited.

    CCP4 children can remain in that group after the worker is reaped, so a
    leader-existence check would skip required cleanup. A reused process-group
    ID could target an unrelated group.
    """
    if os.name != "posix" or not hasattr(os, "killpg") or not pid:
        return
    with contextlib.suppress(OSError, ValueError):
        os.killpg(pid, sig)


def stop_log_listener(listener: QueueListener, queue: WorkerLogQueue[Any]) -> bool:
    """Stop the logging listener within a deadline and close its queue.

    A killed worker may leave the write lock held, blocking the stop sentinel.
    Return True if the daemon listener had to be abandoned.
    """
    stopper = threading.Thread(target=listener.stop, daemon=True)
    stopper.start()
    stopper.join(WORKER_SHUTDOWN_GRACE_S)
    abandoned = stopper.is_alive()
    # Do not join the queue feeder; it may be blocked on the same lock.
    with contextlib.suppress(Exception):
        queue.close()
    return abandoned


def _shutdown_pool(pool: WorkerPool) -> bool:
    """Close a worker pool, killing its children if a clean shutdown hangs.

    A worker killed while blocked in the task queue's ``get()`` never releases
    that queue's lock, and ``Pool.terminate`` then blocks acquiring it, so the
    clean shutdown runs on a deadline. Returns ``True`` when it had to be
    forced.
    """
    children = _pool_children(pool)
    closer = threading.Thread(target=pool.terminate, daemon=True)
    closer.start()
    closer.join(WORKER_SHUTDOWN_GRACE_S)
    # Terminate workers before killing surviving CCP4 groups to avoid task-queue races.
    children_by_pid = {child.pid: child for child in [*children, *_pool_children(pool)]}
    if os.name == "posix":
        for child in children_by_pid.values():
            _signal_worker_process_group(child.pid, signal.SIGKILL)
    if not closer.is_alive():
        return False

    # Cancel the at-exit finalizer, which would repeat the same blocked wait.
    finalizer = getattr(pool, "_terminate", None)
    if finalizer is not None:
        with contextlib.suppress(Exception):  # best effort; shutdown must proceed
            finalizer.cancel()
    for child in children_by_pid.values():
        with contextlib.suppress(OSError, ValueError, AttributeError):
            # Process.kill is SIGKILL on POSIX and TerminateProcess on Windows,
            # where signal.SIGKILL does not exist.
            child.kill()
    # A daemon closer can be abandoned if a dead worker left its lock held.
    return True


def batch_exit_code(incomplete_entry_count: int) -> int:
    """Return failure exactly when one or more entries remain recoverable."""
    return 1 if incomplete_entry_count else 0


class BatchTally:
    """Running totals for the batch and the exit code they imply."""

    def __init__(self) -> None:
        """Start every count at zero."""
        self.counts = {"ok": 0, "partial": 0, "skip": 0, "error": 0}
        self.no_metals = 0
        self.metal_site_limit_exceeded = 0
        self.retryable_partials = 0
        self.terminal_errors = 0
        self.recoverable_errors = 0

    def record(self, result: EntryResult) -> None:
        """Add one entry's outcome to the totals."""
        status = result.status
        self.counts[status] = self.counts.get(status, 0) + 1
        if result.no_metals:
            self.no_metals += 1
        if result.metal_site_limit_exceeded:
            self.metal_site_limit_exceeded += 1
        if status == EntryStatus.PARTIAL and result.retryable:
            self.retryable_partials += 1
        if status == EntryStatus.ERROR:
            if self._terminal_error(result):
                self.terminal_errors += 1
            else:
                self.recoverable_errors += 1

    @staticmethod
    def _terminal_error(result: EntryResult) -> bool:
        """Whether identical inputs will reproduce this entry failure.

        Errors remain eligible for ``--resume`` because the operator may repair
        an input or update a tool. For completion of the current database
        snapshot, however, only an explicitly deterministic reason is a known
        terminal exclusion. Unknown, mixed, worker, and unexpected errors are
        recoverable so a new failure mode cannot silently enter a reference.
        """
        return bool(result.reason_codes) and all(
            code == ReasonCode.DETERMINISTIC_PROCESSING_ERROR
            for code in result.reason_codes
        )

    def exit_code(self, *, database_run: bool = False) -> int:
        """Return success for a complete run under the requested run policy.

        A full database may complete with documented deterministic exclusions.
        A targeted run remains strict: asking for one entry and receiving an
        error still exits nonzero even when that error will recur.
        """
        incomplete = (
            self.recoverable_incompleteness()
            if database_run
            else self.counts.get("error", 0)
            + self.counts.get("skip", 0)
            + self.retryable_partials
        )
        return batch_exit_code(incomplete)

    def recoverable_incompleteness(self) -> int:
        """Return the count of entries that may succeed on a later attempt.

        Only explicitly deterministic failures are terminal for database completion.
        """
        return (
            self.counts.get("skip", 0)
            + self.retryable_partials
            + self.recoverable_errors
        )


class _WorkerDeathWatch:
    """Track dead workers and synthesize results for their unfinished entries.

    Use worker notifications to identify lost entries. Resolve unattributed
    deaths only after the grace period and stalled-work checks. Only entries
    reported through ``track_submitted_entry`` can ever be blamed on a death,
    so work the admission controller never started is never recorded as lost.
    """

    def __init__(
        self,
        pool: WorkerPool,
        inflight: SimpleQueue[tuple[str, int, str]],
        cfg: WorkerConfig,
    ) -> None:
        self._pool = pool
        self._inflight = inflight
        self._ids: list[str] = []
        self._submitted_ids: set[str] = set()
        self._cfg = cfg
        self._assignments: dict[int, str] = {}
        self._worker_pids: set[int] = set()
        self._lost_ids: set[str] = set()
        self._unattributed_deaths = 0
        # Snapshot workers before submitting tasks so an early death cannot go unnoticed.
        dead_worker_pids(self._pool, self._worker_pids)

    def track_submitted_entry(self, pdb_id: str) -> None:
        if pdb_id not in self._submitted_ids:
            self._submitted_ids.add(pdb_id)
            self._ids.append(pdb_id)

    def poll(self) -> list[EntryResult]:
        """Results for the entries whose worker has died since the last call."""
        drain_inflight(self._inflight, self._assignments)
        losses: list[EntryResult] = []
        for dead_pid in dead_worker_pids(self._pool, self._worker_pids):
            if os.name == "posix":
                _signal_worker_process_group(dead_pid, signal.SIGKILL)
            dead_id = self._assignments.pop(dead_pid, None)
            if dead_id is None:
                self._unattributed_deaths += 1
            elif dead_id not in self._lost_ids:
                losses.append(self._lose(dead_id, dead_pid))
        return losses

    def stalled_losses(
        self, completed_ids: set[str], stalled_for: float, remaining: int
    ) -> list[EntryResult] | None:
        """Results for the outstanding entries an unattributed death held.

        ``None`` until every unassigned outstanding entry can be accounted for
        by a death that named no entry and the run has been quiet for the grace
        period. Entries assigned to pids still in the pool are known to be
        alive and cannot be blamed on an older, unrelated death.
        """
        if not (
            self._unattributed_deaths
            and stalled_for > WORKER_STALL_GRACE_S
            and remaining <= self._unattributed_deaths
        ):
            return None
        live_ids = {
            pdb_id
            for pid, pdb_id in self._assignments.items()
            if pid in self._worker_pids
        }
        # Only unassigned entries that never returned a result can still be
        # held by a process that died before naming its entry. A live assigned
        # entry may legitimately spend longer than the fallback grace period
        # inside CCP4, while a completed one already has its real output row.
        losses: list[EntryResult] = []
        for stuck_id in self._ids:
            if (
                stuck_id in self._lost_ids
                or stuck_id in completed_ids
                or stuck_id in live_ids
            ):
                continue
            losses.append(self._lose(stuck_id, 0))
            if len(losses) >= remaining:
                break
        return losses or None

    def superseded(self, result: EntryResult, completed_ids: Collection[str]) -> bool:
        """Ignore delayed real or death results rather than write an entry twice."""
        if result.pdb_id in completed_ids:
            return True
        return result.pdb_id in self._lost_ids and not is_worker_death_result(result)

    def _lose(self, pdb_id: str, pid: int) -> EntryResult:
        self._lost_ids.add(pdb_id)
        return worker_death_result(pdb_id, self._cfg, pid)


def pop_admissible_estimate(
    pending: list[EntryMemoryEstimate],
    reserved_bytes: int,
    budget_bytes: int | None,
    active_estimates: Collection[EntryMemoryEstimate],
    *,
    workers: int = 0,
) -> EntryMemoryEstimate | None:
    """Admit the first pending entry whose estimated memory fits the budget.

    Scan past oversized entries to use available capacity. Live pressure checks
    handle underestimated peaks.
    """
    if not pending:
        return None
    if not active_estimates:
        # Nothing is running, so the head is admitted even when it is oversized:
        # refusing an entry larger than the whole budget would deadlock the batch.
        return pending.pop(0)
    # The candidate occupies one idle worker; its estimate already includes
    # that worker's overhead. Other idle processes remain resident as well.
    idle_bytes = (
        max(0, workers - len(active_estimates) - 1) * WORKER_FIXED_OVERHEAD_BYTES
    )
    for index, estimate in enumerate(pending):
        if (
            budget_bytes is not None
            and reserved_bytes + estimate.bytes + idle_bytes > budget_bytes
        ):
            continue
        return pending.pop(index)
    return None


def guarded_available_memory(
    memory_plan: MemoryPlan, current_available_bytes: int | None
) -> int | None:
    """Measure remaining headroom against host and explicit limits.

    ``available_memory_bytes`` already tracks changing host/cgroup allowance.
    For a smaller explicit limit, approximate this run's consumption from the
    availability observed immediately before the pool started.
    """
    if current_available_bytes is None:
        return None
    if (
        memory_plan.configured_limit_bytes is None
        or memory_plan.initial_available_bytes is None
    ):
        return current_available_bytes
    consumed = max(0, memory_plan.initial_available_bytes - current_available_bytes)
    configured_remaining = max(0, memory_plan.configured_limit_bytes - consumed)
    return min(current_available_bytes, configured_remaining)


class _AdmissionLedger:
    """The entries running in the pool, their reserved bytes, and the run's peaks.

    Every worker process stays resident whether or not it holds an entry, so
    resident memory counts the fixed overhead of each idle worker on top of
    the estimates reserved for the active entries.
    """

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.active: dict[
            str, tuple[AsyncResult[EntryResult], EntryMemoryEstimate]
        ] = {}
        self.reserved_bytes = 0
        self.peak_resident_bytes = 0
        self.max_active = 0
        self.oversized_entries: set[str] = set()

    @property
    def resident_bytes(self) -> int:
        """Reserved entry bytes plus the overhead of every idle worker."""
        idle_workers = self.workers - len(self.active)
        return self.reserved_bytes + idle_workers * WORKER_FIXED_OVERHEAD_BYTES

    def resident_bytes_with(self, estimate: EntryMemoryEstimate) -> int:
        """Resident bytes once ``estimate`` occupies one of the idle workers."""
        idle_workers = self.workers - len(self.active) - 1
        return (
            self.reserved_bytes
            + estimate.bytes
            + idle_workers * WORKER_FIXED_OVERHEAD_BYTES
        )

    def has_idle_worker(self) -> bool:
        return len(self.active) < self.workers

    def runs_oversized_alone(self, budget_bytes: int | None) -> bool:
        """Whether the single active entry already exceeds the byte budget."""
        return (
            len(self.active) == 1
            and budget_bytes is not None
            and self.resident_bytes > budget_bytes
        )

    def active_estimates(self) -> list[EntryMemoryEstimate]:
        return [estimate for _task, estimate in self.active.values()]

    def admit(
        self, task: AsyncResult[EntryResult], estimate: EntryMemoryEstimate
    ) -> None:
        """Record an entry that was just handed to a worker."""
        self.active[estimate.pdb_id] = (task, estimate)
        self.reserved_bytes += estimate.bytes
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        self.max_active = max(self.max_active, len(self.active))

    def release(self, pdb_id: str) -> None:
        """Forget an entry whose worker returned or died; unknown ids are ignored."""
        item = self.active.pop(pdb_id, None)
        if item is not None:
            self.reserved_bytes -= item[1].bytes

    def collect_ready(self) -> list[EntryResult]:
        """Take every finished result out of the ledger."""
        results: list[EntryResult] = []
        for pdb_id, (task, _estimate) in tuple(self.active.items()):
            if task.ready():
                results.append(task.get())
                self.release(pdb_id)
        return results


def _log_admission_changes(
    admission: MemoryAdmission,
    old_budget: int | None,
    old_pauses: int,
    current_available: int | None,
    reserve_bytes: int | None,
) -> None:
    """Warn once per budget change or new pause; the controller is otherwise quiet."""
    new_budget = admission.budget
    if new_budget != old_budget and old_budget is not None and new_budget is not None:
        logger.warning(
            "%s memory admission from %.2f GiB to %.2f GiB",
            "recovering" if new_budget > old_budget else "reducing",
            old_budget / GIB,
            new_budget / GIB,
        )
    if (
        admission.pauses != old_pauses
        and current_available is not None
        and reserve_bytes is not None
    ):
        logger.warning(
            "pausing new entries: %.2f GiB available has reached the "
            "%.2f GiB protected reserve",
            current_available / GIB,
            reserve_bytes / GIB,
        )


def _admit_pending(
    pending: list[EntryMemoryEstimate],
    ledger: _AdmissionLedger,
    admission: MemoryAdmission,
    memory_plan: MemoryPlan,
    deaths: _WorkerDeathWatch,
    pool: WorkerPool,
) -> None:
    """Start pending entries on idle workers until memory or the budget says stop."""
    while pending and ledger.has_idle_worker():
        current_available = guarded_available_memory(
            memory_plan, available_memory_bytes()
        )
        if (
            ledger.active
            and memory_plan.reserve_bytes is not None
            and current_available is not None
            and current_available <= memory_plan.reserve_bytes
        ):
            break
        estimate = pop_admissible_estimate(
            pending,
            ledger.reserved_bytes,
            admission.budget,
            ledger.active_estimates(),
            workers=ledger.workers,
        )
        if estimate is None:
            break
        required_bytes = ledger.resident_bytes_with(estimate)
        if admission.budget is not None and required_bytes > admission.budget:
            ledger.oversized_entries.add(estimate.pdb_id)
            logger.warning(
                "%s needs an estimated %.2f GiB including resident workers, "
                "above the %.2f GiB worker budget; admitting it alone",
                estimate.pdb_id,
                required_bytes / GIB,
                admission.budget / GIB,
            )
        deaths.track_submitted_entry(estimate.pdb_id)
        ledger.admit(pool.apply_async(process, (estimate.pdb_id,)), estimate)


def _release_workers(
    pool: WorkerPool,
    log_listener: QueueListener,
    log_queue: WorkerLogQueue[Any],
    run_log: RunLog,
) -> None:
    """Stop the pool and its log listener, recording any forced shutdown."""
    forced = _shutdown_pool(pool)
    # Keep forwarding logs until worker shutdown completes.
    if stop_log_listener(log_listener, log_queue):
        run_log.summary["worker_log_listener_abandoned"] = True
        logger.warning(
            "the worker log listener did not stop within %gs and was "
            "abandoned; some worker records may be missing from this log",
            WORKER_SHUTDOWN_GRACE_S,
        )
    if forced:
        run_log.summary["worker_pool_forced_shutdown"] = True
        logger.warning(
            "a worker pool shutdown had to be forced after a worker died "
            "holding the task-queue lock; results above are complete"
        )


def dispatch_entries(
    ids: Sequence[str],
    cfg: WorkerConfig,
    workers: int,
    memory_plan: MemoryPlan,
    run_log: RunLog,
    sink: ResultSink,
) -> BatchTally:
    """Run every entry across a worker pool, passing results to ``sink`` as they land.

    The loop polls rather than iterating the pool's results, because waiting on
    the pool alone would hang forever on an entry a killed worker was holding.
    """
    tally = BatchTally()
    progress = ProgressReporter(len(ids))
    inflight: SimpleQueue[tuple[str, int, str]] = SimpleQueue()
    completed_ids: set[str] = set()
    completed = 0
    last_progress = time.monotonic()
    # Use Python's default start method for platform compatibility.
    # Avoid Pool's context manager: its unbounded terminate can hang after a
    # worker dies with a queue lock held. The finally block bounds shutdown.
    # Start the log listener after the pool to avoid forking with active threads.
    log_queue = create_worker_log_queue()
    pool = Pool(
        workers, initializer=initialize_worker, initargs=(cfg, inflight, log_queue)
    )
    log_listener = start_worker_log_listener(log_queue)
    deaths = _WorkerDeathWatch(pool, inflight, cfg)

    def render(*, force: bool = False, final: bool = False) -> None:
        progress.render(
            completed,
            tally.counts,
            tally.no_metals,
            tally.metal_site_limit_exceeded,
            force=force,
            final=final,
        )

    try:
        pending = list(memory_plan.estimates)
        estimates_by_id = {
            estimate.pdb_id: estimate for estimate in memory_plan.estimates
        }
        ledger = _AdmissionLedger(workers)
        admission = MemoryAdmission(
            memory_plan.budget_bytes,
            WORKER_FIXED_OVERHEAD_BYTES,
            memory_plan.reserve_bytes,
        )
        render(force=True)
        while completed < len(ids):
            current_available = guarded_available_memory(
                memory_plan, available_memory_bytes()
            )
            old_budget = admission.budget
            old_pauses = admission.pauses
            pause = admission.observe(
                current_available,
                ledger.resident_bytes,
                time.monotonic(),
                active=bool(ledger.active),
                oversized=ledger.runs_oversized_alone(memory_plan.budget_bytes),
            )
            _log_admission_changes(
                admission,
                old_budget,
                old_pauses,
                current_available,
                memory_plan.reserve_bytes,
            )
            if not pause:
                _admit_pending(pending, ledger, admission, memory_plan, deaths, pool)

            batch = ledger.collect_ready()
            batch.extend(deaths.poll())
            if not batch:
                stalled = deaths.stalled_losses(
                    completed_ids,
                    time.monotonic() - last_progress,
                    len(ledger.active),
                )
                if stalled is None:
                    render()
                    time.sleep(0.05)
                    continue
                batch = stalled
            for result in batch:
                if is_worker_death_result(result):
                    ledger.release(result.pdb_id)
            if any(is_worker_death_result(result) for result in batch):
                old_budget = admission.budget
                admission.back_off(time.monotonic())
                new_budget = admission.budget
                if (
                    new_budget != old_budget
                    and old_budget is not None
                    and new_budget is not None
                ):
                    logger.warning(
                        "reducing future memory admission from %.2f GiB to "
                        "%.2f GiB after a worker process died",
                        old_budget / GIB,
                        new_budget / GIB,
                    )
            last_progress = time.monotonic()
            for result in batch:
                if deaths.superseded(result, completed_ids):
                    continue
                completed += 1
                completed_ids.add(result.pdb_id)
                run_log.record_entry(
                    result,
                    memory_estimate_bytes=estimates_by_id[result.pdb_id].bytes,
                )
                sink(result)
                tally.record(result)
                finished = completed == len(ids)
                render(force=progress.terminal or finished, final=finished)
        run_log.summary.update(
            memory_scheduler_worker_overhead_bytes=workers
            * WORKER_FIXED_OVERHEAD_BYTES,
            memory_scheduler_peak_reserved_bytes=ledger.peak_resident_bytes,
            memory_scheduler_max_active_entries=ledger.max_active,
            memory_scheduler_oversized_entries=len(ledger.oversized_entries),
            memory_scheduler_pressure_pauses=admission.pauses,
            memory_scheduler_budget_backoffs=admission.backoffs,
            memory_scheduler_budget_recoveries=admission.recoveries,
            memory_scheduler_final_budget_bytes=(
                admission.budget if admission.budget is not None else "unavailable"
            ),
        )
    finally:
        _release_workers(pool, log_listener, log_queue, run_log)
        progress.close()
    return tally
