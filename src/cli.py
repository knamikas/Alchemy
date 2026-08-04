"""The command line: arguments in, exit code out.

Everything a run needs before the driver can start, and everything it owes the
user afterwards. Argument parsing and its validation live here, as do the two
things that bracket every run: the detailed run log, which is written whatever
happens, and the SIGTERM handler that makes a scheduler stop unwind the same
way Ctrl-C does.

``src/main.py`` is a shim over this module. It stays because it is the
documented entry point -- the README and this package's own docstring both say
``python src/main.py`` -- and moving a public script is a different thing from
moving an implementation.
"""

import argparse
import contextlib
import os
import re
import shlex
import signal
import sys

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
from driver.pool import DEFAULT_ROOT, _run
from driver.runlog import _RunLog


logger = logger_for(__name__)


def parse_pdb_id(value):
    if not re.fullmatch(r"[A-Za-z0-9]{4}", value):
        raise argparse.ArgumentTypeError(
            "PDB ID must contain exactly four alphanumeric characters"
        )
    return value.lower()


def positive_int(value):
    """Argparse type for integer options that must be at least one."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv=None):
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
            "number of worker processes (minimum: 1); by default Alchemy "
            "uses the lower CPU or available-memory limit"
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
    # ArgumentDefaultsHelpFormatter appends the default of ``bonds``, not of
    # the flag, so an unqualified help string renders "(default: True)" -- the
    # negation of what --no-bonds does. Naming %(default)s explicitly suppresses
    # that append and lets the value be labelled with the setting it belongs to.
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
        raise SystemExit("use either --id or --id-file, not both")
    if args.retry_partials and not args.resume:
        ap.error("--retry-partials requires --resume")
    if args.retry_partials and (args.pdb_file or args.mtz_file or args.cif_file):
        ap.error("--retry-partials cannot be used with manual structure inputs")
    return args


def _install_termination_handler():
    """Route SIGTERM through the same unwind path as Ctrl-C.

    SIGTERM's default disposition kills the process immediately: no ``finally``
    runs, so the worker pool is never shut down and its children -- which are
    daemonic but not yet signalled -- are reparented to init and keep working,
    driving CCP4 subprocesses and holding their scratch directories open.
    Raising ``KeyboardInterrupt`` instead makes a scheduler stop or a plain
    ``kill`` unwind exactly like an interactive interrupt, so the pool is
    terminated and the run log is still written.

    Returns the previous handler, or ``None`` where SIGTERM cannot be trapped
    (a non-main thread, or a platform without it).
    """

    def _raise_interrupt(signum, frame):  # noqa: ARG001 - signal API
        raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGTERM, _raise_interrupt)
    except (AttributeError, OSError, ValueError):  # pragma: no cover
        return None


def main(argv=None):
    """Parse arguments, execute the driver, and always emit a run log."""
    raw_args = None if argv is None else list(argv)
    args = parse_args(raw_args)
    # Handlers are attached once, here, and nowhere else: the driver is the
    # only process that writes them, and workers reach them over a queue.
    configure_driver_logging(
        level=level_for_verbosity(args.verbose, args.quiet),
        log_file=args.log_file,
    )
    command_parts = list(sys.argv) if raw_args is None else [sys.argv[0], *raw_args]
    run_log = _RunLog(args, shlex.join(command_parts))
    exit_code = 1
    previous_term = _install_termination_handler()
    try:
        exit_code = _run(args, run_log)
        return exit_code
    except KeyboardInterrupt:
        # Ctrl-C or SIGTERM. The pool has already been shut down by _run's
        # own finally, so this only decides how the driver reports it: a
        # conventional interrupt status and one line, rather than a traceback
        # that looks like a crash.
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
