"""Standalone ``finalize`` and ``score`` subcommands for prepared inputs."""

import argparse
import csv
import sys
from collections.abc import Sequence

from confidence_score.reference import (
    finalize_database_confidence,
    load_reference,
    score_file_against_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PYTHONPATH=src python3 -m confidence_score",
        description="Finalize or apply Alchemy confidence scores.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    finalize = commands.add_parser(
        "finalize", help="finalize a streamed complete-database cohort"
    )
    finalize.add_argument(
        "--input", required=True, help="streamed database confidence-input CSV"
    )
    finalize.add_argument(
        "--output", required=True, help="output database confidence-score CSV"
    )
    finalize.add_argument(
        "--reference-dir",
        required=True,
        help="output frozen database reference directory",
    )
    finalize.add_argument(
        "--manifest",
        help="optional completed run manifest to record as cohort provenance",
    )

    score = commands.add_parser(
        "score", help="score inputs against a frozen database reference"
    )
    score.add_argument("--input", required=True, help="prepared confidence-input CSV")
    score.add_argument(
        "--output", required=True, help="output confidence-score CSV path"
    )
    score.add_argument(
        "--reference-dir", required=True, help="frozen database reference directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the confidence-reference CLI and return an exit status."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "finalize":
            total, scored, cohort = finalize_database_confidence(
                args.input,
                args.output,
                args.reference_dir,
                manifest_path=args.manifest,
            )
            print(
                f"finalized {total} rows ({scored} scored; database cohort "
                f"{cohort}) to {args.output}"
            )
        elif args.command == "score":
            reference = load_reference(args.reference_dir)
            total, scored = score_file_against_reference(
                args.input, args.output, reference
            )
            print(f"wrote {total} rows ({scored} scored) to {args.output}")
        else:  # pragma: no cover - the parser rejects unknown subcommands
            raise AssertionError(f"unhandled confidence command: {args.command}")
    except (OSError, ValueError, csv.Error) as exc:
        print(f"confidence {args.command} failed: {exc}", file=sys.stderr)
        return 1
    return 0
