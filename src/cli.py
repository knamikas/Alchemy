"""Alchemy's command-line interface.

Parse and validate arguments, start the pipeline, write the run report,
and handle termination signals.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import shlex
import signal
import sys
from collections.abc import Generator, Sequence
from types import FrameType, TracebackType

from ccp4_setup import REPO_DIR
from density_analysis import (
    CCP4_TOOL_TIMEOUT_S,
    DENSITY_MAP_SCOPES,
    MODEL_ENVELOPE_BORDER_ANGSTROM,
)
from driver.pool import run
from driver.runlog import RunLog
from run_config import RunConfig
from run_logging import configure_driver_logging, logger_for

logger = logger_for(__name__)


def parse_pdb_id(value: str) -> str:
    """Parse and normalize a four-character PDB identifier."""
    if not re.fullmatch(r"[A-Za-z0-9]{4}", value):
        raise argparse.ArgumentTypeError(
            "PDB ID must contain exactly four alphanumeric characters"
        )
    return value.lower()


def positive_int(value: str) -> int:
    """Argparse type for integer options that must be at least one."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def memory_size_bytes(value: str) -> int:
    """Argparse type for byte sizes such as ``8G`` or ``16GiB``."""
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?I?B?)?", value.upper())
    if match is None:
        raise argparse.ArgumentTypeError(
            "must be a positive byte size such as 8G or 16GiB"
        )
    number, suffix = match.groups()
    normalized = (suffix or "B").removesuffix("B").removesuffix("I")
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }[normalized]
    numeric_value = float(number)
    if not math.isfinite(numeric_value):
        raise argparse.ArgumentTypeError("must be a finite byte size")
    scaled_value = numeric_value * multiplier
    if not math.isfinite(scaled_value):
        raise argparse.ArgumentTypeError("byte size is too large")
    parsed = int(scaled_value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def utilization_fraction(value: str) -> float:
    """Argparse type for a fraction in the interval ``(0, 1]``."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "must be a number greater than 0 and at most 1"
        ) from exc
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Declare every command-line option; validation lives in ``parse_args``."""
    ap = argparse.ArgumentParser(
        description="Batch Alchemy core pipeline over PDB-REDO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--id", type=parse_pdb_id, help="process a single PDB id (else batch the root)"
    )
    ap.add_argument(
        "--id-file",
        help="path to a file of PDB ids (comma-, whitespace-, or newline-separated)",
    )
    ap.add_argument("--pdb-file", help="path to a local PDB file for manual input mode")
    ap.add_argument("--mtz-file", help="path to a local MTZ file for manual input mode")
    ap.add_argument(
        "--cif-file", help="path to a local mmCIF file for manual input mode"
    )
    ap.add_argument(
        "--data-json", help="optional path to a local data.json for manual input mode"
    )
    ap.add_argument(
        "--pdb-redo-root",
        help="path to a local PDB-REDO mirror; required when no IDs or manual files are supplied",
    )
    ap.add_argument(
        "--pdb-redo-cache",
        default=os.path.join(REPO_DIR, "pdb-redo-cache"),
        help="root of local cache for auto-downloaded PDB-REDO entries",
    )
    ap.add_argument(
        "--pdb-metadata-cache",
        default=os.path.join(REPO_DIR, "pdb-metadata-cache"),
        help=(
            "persistent cache for original-PDB crystallization metadata "
            "retrieved from the RCSB Data API"
        ),
    )
    ap.add_argument(
        "--no-crystallization-download",
        dest="crystallization_download",
        action="store_false",
        help=(
            "do not fetch missing original-PDB crystallization metadata; use "
            "only the metadata cache and coordinate-file records"
        ),
    )
    ap.set_defaults(crystallization_download=True)
    ap.add_argument(
        "--max-pdbs",
        type=positive_int,
        default=None,
        help="process only the first N entries (minimum: 1)",
    )
    ap.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help=(
            "worker-process ceiling (minimum: 1); memory-aware admission "
            "still limits simultaneously active entries"
        ),
    )
    ap.add_argument(
        "--memory-limit",
        type=memory_size_bytes,
        default=None,
        help=(
            "memory available to Alchemy (for example 8G or 16GiB); "
            "default: auto-detect the host, container, or scheduler limit"
        ),
    )
    ap.add_argument(
        "--memory-utilization",
        type=utilization_fraction,
        default=0.80,
        help=(
            "maximum fraction of detected or configured memory used for "
            "worker estimates; the protected 4 GiB reserve still applies"
        ),
    )
    ap.add_argument("--output-dir", default=os.path.join(REPO_DIR, "output"))
    ap.add_argument(
        "--density-map-scope",
        choices=DENSITY_MAP_SCOPES,
        default="model-envelope",
        help=(
            "map extent supplied to EDSTATS; model-envelope retains every "
            f"coordinate plus a {MODEL_ENVELOPE_BORDER_ANGSTROM} Angstrom "
            "border and falls back to full when cropping would be unsafe or "
            "larger"
        ),
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "increase diagnostic detail; -v adds per-entry and per-CCP4-program "
            "records from inside the workers"
        ),
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="report warnings and errors only, suppressing the run narrative",
    )
    ap.add_argument(
        "--log-dir",
        default=None,
        help=(
            "directory for run reports (default: <output-dir>/logs/). Each "
            "invocation writes an immutable log and entry-diagnostics CSV there"
        ),
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help=(
            "also write full debug-level diagnostics to this file, whatever "
            "the console verbosity"
        ),
    )
    ap.add_argument(
        "--ccp4-timeout",
        type=positive_int,
        default=CCP4_TOOL_TIMEOUT_S,
        help=(
            "per-program wall-clock budget in seconds for each CCP4 step "
            "(mtzfix, fft, mapmask, edstats); raise it for exceptionally "
            "large structures"
        ),
    )
    ap.add_argument(
        "--confidence-reference-dir",
        default=None,
        help=(
            "explicit frozen full-database confidence reference for single, "
            "ID-file, manual, and capped runs; otherwise Alchemy searches "
            "the output directory and repository default"
        ),
    )
    ap.add_argument(
        "--ccp4-setup",
        default=None,
        help="optional CCP4 setup script override (e.g. .../bin/ccp4.setup-sh)",
    )
    ap.add_argument(
        "--configure-ccp4",
        default=None,
        help="save a CCP4 setup script path for future runs",
    )
    ap.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="keep per-entry maps/logs (default: delete after extract)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip terminal ok/partial results; retry retryable incomplete ids",
    )
    ap.add_argument(
        "--retry-partials",
        action="store_true",
        help=(
            "with --resume, reprocess non-retryable partial entries from "
            "the manifest while still skipping successful entries; --id "
            "or --id-file may restrict the retry set"
        ),
    )
    ap.add_argument(
        "--no-bonds",
        dest="bonds",
        action="store_false",
        help="skip the metal-ligand bond-distance stage (edstats "
        "stats only); bond analysis is enabled by default "
        "(bonds=%(default)s)",
    )
    ap.set_defaults(bonds=True)
    return ap


def _validate_arguments(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject option combinations that argparse alone cannot express."""
    if args.id and args.id_file:
        ap.error("use either --id or --id-file, not both")
    if args.retry_partials and not args.resume:
        ap.error("--retry-partials requires --resume")
    if args.retry_partials and (args.pdb_file or args.mtz_file or args.cif_file):
        ap.error("--retry-partials cannot be used with manual structure inputs")
    if (args.pdb_file or args.cif_file) and not args.mtz_file:
        ap.error("manual structure input requires --mtz-file")
    if args.mtz_file and not (args.pdb_file or args.cif_file):
        ap.error("--mtz-file requires --pdb-file or --cif-file")
    if args.pdb_file and args.cif_file:
        ap.error("use either --pdb-file or --cif-file, not both")
    manual_requested = bool(args.pdb_file or args.mtz_file or args.cif_file)
    if manual_requested and args.id_file:
        ap.error(
            "--id-file cannot be combined with manual structure inputs; "
            "use --id to identify the single manual structure"
        )
    if args.data_json and not manual_requested:
        ap.error(
            "--data-json requires manual structure inputs: --mtz-file with "
            "either --pdb-file or --cif-file"
        )


def _run_config(args: argparse.Namespace) -> RunConfig:
    """Freeze the parsed namespace into the immutable run configuration."""
    return RunConfig(
        id=args.id,
        id_file=args.id_file,
        pdb_file=args.pdb_file,
        mtz_file=args.mtz_file,
        cif_file=args.cif_file,
        data_json=args.data_json,
        pdb_redo_root=args.pdb_redo_root,
        pdb_redo_cache=args.pdb_redo_cache,
        pdb_metadata_cache=args.pdb_metadata_cache,
        crystallization_download=args.crystallization_download,
        max_pdbs=args.max_pdbs,
        workers=args.workers,
        memory_limit=args.memory_limit,
        memory_utilization=args.memory_utilization,
        output_dir=args.output_dir,
        density_map_scope=args.density_map_scope,
        verbose=args.verbose,
        quiet=args.quiet,
        log_dir=args.log_dir,
        log_file=args.log_file,
        ccp4_timeout=args.ccp4_timeout,
        confidence_reference_dir=args.confidence_reference_dir,
        ccp4_setup=args.ccp4_setup,
        configure_ccp4=args.configure_ccp4,
        keep_intermediates=args.keep_intermediates,
        resume=args.resume,
        retry_partials=args.retry_partials,
        bonds=args.bonds,
    )


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    """Parse command-line arguments into an immutable run configuration."""
    ap = build_parser()
    args = ap.parse_args(argv)
    _validate_arguments(ap, args)
    return _run_config(args)


@contextlib.contextmanager
def _sigterm_as_keyboard_interrupt() -> Generator[None]:
    """Handle SIGTERM like Ctrl-C so cleanup can stop workers and CCP4 processes.

    The previous handler is restored on exit. Where signal handling is
    unavailable (off the main thread, or on a platform that rejects the call)
    the body runs without a handler, exactly as before.
    """

    def _raise_interrupt(signum: int, frame: FrameType | None) -> None:
        del signum, frame  # fixed by the signal-handler signature
        raise KeyboardInterrupt

    try:
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            with contextlib.suppress(AttributeError, OSError, ValueError):
                signal.signal(signal.SIGTERM, previous)


class _Reporting:
    """Write the run report when the run ends, however it ends.

    ``exit_code`` starts at 1 so that a run which dies of an unexpected
    exception is still reported as a failure; the body stores the run's own
    code. An interruption is converted here: it is recorded in the report,
    announced on stderr, swallowed, and reported as exit code 130.
    """

    def __init__(self, run_log: RunLog) -> None:
        self.run_log = run_log
        self.exit_code = 1

    def __enter__(self) -> _Reporting:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        interrupted = isinstance(exc, KeyboardInterrupt)
        if interrupted:
            # run() has already stopped the workers; report the interruption.
            self.run_log.driver_error = "interrupted before completion"
            print(
                "\nInterrupted: workers stopped; rows already flushed are kept "
                "and --resume will continue.",
                file=sys.stderr,
                flush=True,
            )
            self.exit_code = 130
        elif exc is not None:
            self.run_log.driver_error = f"{type(exc).__name__}: {exc}"
        try:
            log_path = self.run_log.write(self.exit_code)
        except OSError as write_error:
            logger.error("could not write run report: %s", write_error)
        else:
            logger.info("run report -> %s", log_path)
        return interrupted


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the pipeline, and attempt to write its run report."""
    raw_args = None if argv is None else list(argv)
    args = parse_args(raw_args)
    try:
        configure_driver_logging(level=args.log_level, log_file=args.log_file)
    except OSError as exc:
        print(
            f"Cannot write --log-file {args.log_file}: {exc.strerror or exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    command_parts = list(sys.argv) if raw_args is None else [sys.argv[0], *raw_args]
    run_log = RunLog(args, shlex.join(command_parts))
    # The signal handler is the inner context so that it is restored before
    # the report is written, as the original try/finally did.
    with _Reporting(run_log) as report, _sigterm_as_keyboard_interrupt():
        report.exit_code = run(args, run_log)
    return report.exit_code
