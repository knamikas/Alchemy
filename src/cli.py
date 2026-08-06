"""The command line: arguments in, exit code out.

Argument parsing and its validation, plus the two things that bracket every
run: the detailed run log, written whatever happens, and the SIGTERM handler
that makes a scheduler stop unwind the same way Ctrl-C does.
"""

import argparse
import contextlib
import os
import re
import shlex
import signal
import sys
from collections.abc import Callable, Sequence
from types import FrameType

from density_analysis import (
    CCP4_TOOL_TIMEOUT_S,
    DENSITY_MAP_SCOPES,
    MODEL_ENVELOPE_BORDER_ANGSTROM,
)
from run_logging import (
    configure_driver_logging,
    level_for_verbosity,
    logger_for,
)
from ccp4_setup import REPO_DIR
from driver.pool import DEFAULT_ROOT, run
from driver.runlog import RunLog
from run_config import RunConfig


logger = logger_for(__name__)


def parse_pdb_id(value: str) -> str:
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


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    ap = argparse.ArgumentParser(
        description="Batch Alchemy core pipeline over PDB-REDO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--id", type=parse_pdb_id, help="process a single PDB id (else batch the root)"
    )
    ap.add_argument(
        "--id-file", help="path to a file of PDB ids (comma- and/or newline-separated)"
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
        "--pdb-redo-root", default=DEFAULT_ROOT, help="root of the PDB-REDO mirror"
    )
    ap.add_argument(
        "--pdb-redo-cache",
        default=os.path.join(REPO_DIR, "pdb-redo-cache"),
        help="root of local cache for auto-downloaded PDB-REDO entries",
    )
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
            "directory for run logs (default: <output-dir>/logs/). Each "
            "invocation writes one immutable, timestamped log there"
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
    # ArgumentDefaultsHelpFormatter would append ``bonds``' default, rendering
    # "(default: True)" against --no-bonds; naming %(default)s in the help
    # string suppresses that append.
    ap.add_argument(
        "--no-bonds",
        dest="bonds",
        action="store_false",
        help="skip the metal-ligand bond-distance stage (edstats "
        "stats only); bond analysis is enabled by default "
        "(bonds=%(default)s)",
    )
    ap.set_defaults(bonds=True)

    args = ap.parse_args(argv)
    if args.id and args.id_file:
        ap.error("use either --id or --id-file, not both")
    if args.retry_partials and not args.resume:
        ap.error("--retry-partials requires --resume")
    if args.retry_partials and (args.pdb_file or args.mtz_file or args.cif_file):
        ap.error("--retry-partials cannot be used with manual structure inputs")
    # Manual mode needs reflections and coordinates. Caught here rather than in
    # a worker, where the omission surfaces as an unexpected processing error
    # instead of a usage mistake.
    if (args.pdb_file or args.cif_file) and not args.mtz_file:
        ap.error("manual structure input requires --mtz-file")
    if args.mtz_file and not (args.pdb_file or args.cif_file):
        ap.error("--mtz-file requires --pdb-file or --cif-file")
    # Both name the coordinates; unrejected, the cif silently wins.
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
    return RunConfig(
        id=args.id,
        id_file=args.id_file,
        pdb_file=args.pdb_file,
        mtz_file=args.mtz_file,
        cif_file=args.cif_file,
        data_json=args.data_json,
        pdb_redo_root=args.pdb_redo_root,
        pdb_redo_cache=args.pdb_redo_cache,
        max_pdbs=args.max_pdbs,
        workers=args.workers,
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


def _install_termination_handler() -> (
    Callable[[int, FrameType | None], object] | int | None
):
    """Route SIGTERM through the same unwind path as Ctrl-C.

    SIGTERM's default disposition kills the process without running any
    ``finally``, so the pool is never shut down and its children are reparented
    to init still driving CCP4 subprocesses and holding scratch directories
    open. Returns the previous handler, or ``None`` where SIGTERM cannot be
    trapped.
    """

    def _raise_interrupt(  # noqa: ARG001 - signal API
        signum: int, frame: FrameType | None
    ) -> None:
        raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGTERM, _raise_interrupt)
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute the driver, and always emit a run log."""
    raw_args = None if argv is None else list(argv)
    args = parse_args(raw_args)
    try:
        configure_driver_logging(
            level=level_for_verbosity(args.verbose, args.quiet),
            log_file=args.log_file,
        )
    except OSError as exc:
        # The one failure that cannot be logged, because it is the logging.
        print(
            f"Cannot write --log-file {args.log_file}: {exc.strerror or exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    command_parts = list(sys.argv) if raw_args is None else [sys.argv[0], *raw_args]
    run_log = RunLog(args, shlex.join(command_parts))
    exit_code = 1
    previous_term = _install_termination_handler()
    try:
        exit_code = run(args, run_log)
        return exit_code
    except KeyboardInterrupt:
        # ``run``'s own finally has already shut the pool down, so this only
        # decides how the interrupt is reported.
        run_log.driver_error = "interrupted before completion"
        print(
            "\nInterrupted: workers stopped; rows already flushed are kept "
            "and --resume will continue.",
            file=sys.stderr,
            flush=True,
        )
        exit_code = 130
        return exit_code
    except BaseException as exc:
        run_log.driver_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if previous_term is not None:
            with contextlib.suppress(AttributeError, OSError, ValueError):
                signal.signal(signal.SIGTERM, previous_term)
        try:
            log_path = run_log.write(exit_code)
        except OSError as exc:
            logger.error("could not write detailed run log: %s", exc)
        else:
            logger.info("detailed run log -> %s", log_path)
