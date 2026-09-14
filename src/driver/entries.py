"""Select and schedule the entries a run analyzes."""

from __future__ import annotations

import os
import re

from driver.errors import DriverError
from driver.layout import (
    OutputLayout,
)
from driver.resume import (
    load_done,
)
from driver.runlog import RunLog
from inputs import (
    ensure_entry_available,
    enumerate_entries,
    infer_pdb_id_from_path,
    read_data_json_properties,
)
from run_config import RunConfig
from run_logging import logger_for

logger = logger_for(__name__)


def load_ids_from_file(path: str) -> list[str]:
    """Read PDB IDs from comma- or newline-separated text.

    Use utf-8-sig to accept byte-order marks from Windows editors and exports.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"id file not found: {path}")
    ids: list[str] = []
    with open(path, encoding="utf-8-sig") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in re.split(r"[,\s]+", line):
                if not token:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9]{4}", token):
                    raise ValueError(f"invalid PDB id {token!r} at {path}:{lineno}")
                ids.append(token.lower())
    return list(dict.fromkeys(ids))


def select_entry_ids(
    args: RunConfig, cache_root: str
) -> tuple[list[str], str, dict[str, str | None] | None]:
    """Resolve the run's work list, returning ``(ids, root, manual_inputs)``."""
    root = args.pdb_redo_root or cache_root
    if args.pdb_file or args.mtz_file or args.cif_file:
        pdb_id = (
            args.id
            or infer_pdb_id_from_path(args.cif_file)
            or infer_pdb_id_from_path(args.pdb_file)
            or infer_pdb_id_from_path(args.mtz_file)
        )
        if not pdb_id:
            raise DriverError(
                "Manual input mode requires --id or a file name that contains "
                "a 4-character PDB id."
            )
        if args.data_json:
            try:
                read_data_json_properties(args.data_json)
            except ValueError as exc:
                raise DriverError(f"Invalid --data-json: {exc}") from None
        return (
            [pdb_id],
            root,
            {
                "pdb_file": args.pdb_file,
                "mtz_file": args.mtz_file,
                "cif_file": args.cif_file,
                "data_json": args.data_json,
            },
        )

    if args.id:
        try:
            used_root = ensure_entry_available(args.id, args.pdb_redo_root, cache_root)
        except FileNotFoundError:
            raise DriverError(
                f"Entry {args.id} not found locally and download failed."
            ) from None
        except OSError as exc:
            # Report cache and filesystem failures as usage errors without mislabeling
            # them as missing entries.
            raise DriverError(
                f"Entry {args.id} could not be prepared: {type(exc).__name__}: {exc}"
            ) from None
        if used_root != args.pdb_redo_root:
            logger.info("auto-downloaded %s into cache at %s", args.id, cache_root)
        return [args.id], used_root, None

    if args.id_file:
        try:
            ids = load_ids_from_file(args.id_file)
        except (FileNotFoundError, ValueError) as exc:
            raise DriverError(str(exc)) from None
        logger.info("loaded %d IDs from %s", len(ids), args.id_file)
        return ids, root, None

    if not args.pdb_redo_root:
        raise DriverError(
            "Supply --pdb-redo-root to process a local PDB-REDO mirror, "
            "or choose entries with --id, --id-file, or manual input files."
        )
    logger.info("enumerating final PDB-REDO entries under %s", root)
    # Resume subtracts finished entries from the full set, so enumeration can
    # stop early only when not resuming.
    limit = args.max_pdbs if (args.max_pdbs and not args.resume) else None
    return enumerate_entries(root, limit=limit), root, None


def schedule_entries(
    args: RunConfig,
    layout: OutputLayout,
    cache_root: str,
    run_log: RunLog,
) -> tuple[list[str], str, dict[str, str | None] | None]:
    """Return ``(ids, root, manual_inputs)`` for the entries this run will do.

    ``--resume`` removes finished entries before ``--max-pdbs`` caps what is
    left; capping first would re-offer the same finished prefix forever.
    """
    ids, root, manual_inputs = select_entry_ids(args, cache_root)
    run_log.details["entries_selected_before_resume"] = len(ids)
    run_log.details["resolved_input_root"] = root

    if args.resume:
        bonds_required = bool(args.bonds)
        bond_output_present = os.path.isfile(layout.bonds)
        candidate_output_present = os.path.isfile(layout.candidates)
        normally_done = load_done(
            layout.manifest,
            bonds_required=bonds_required,
            bond_output_present=bond_output_present,
            candidate_output_present=candidate_output_present,
        )
        if args.retry_partials:
            done = load_done(
                layout.manifest,
                bonds_required=bonds_required,
                bond_output_present=bond_output_present,
                candidate_output_present=candidate_output_present,
                retry_partial_ids=ids,
            )
            reselected = normally_done - done
            run_log.details["terminal_partials_reselected"] = len(reselected)
            logger.info(
                "selected %d terminal partial entr%s for retry",
                len(reselected),
                "y" if len(reselected) == 1 else "ies",
            )
        else:
            done = normally_done
        # Normalize IDs for manifest comparison while preserving on-disk path spelling.
        ids = [i for i in ids if i.lower() not in done]
    if args.max_pdbs is not None:
        ids = ids[: args.max_pdbs]
    run_log.details["entries_scheduled"] = len(ids)
    return ids, root, manual_inputs
