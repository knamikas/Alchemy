# Design: Auto-download into cache and mirror layout

Overview
--------

Introduce a small downloader that fetches PDB-REDO files for a single PDB id
into a repo-local cache directory using the same hash-layout expected by the
pipeline: `<cache_root>/<id[1:3]>/<id>/...`.

Behavior
--------
- New CLI flag: `--pdb-redo-cache` (default: `pdb-redo-cache/` inside repo).
- New CLI flag: `--id-file <path>` for running a list of PDB IDs from a
  comma-separated and/or newline-separated text file.
- New CLI flags for manual input mode: `--pdb-file`, `--mtz-file`, and
  optional `--cif-file` / `--data-json` for using user-supplied local files.
- When `--id <pdb>` or `--id-file <path>` is used, the program checks in order:
  1. `--pdb-redo-root` (full local mirror)
  2. `--pdb-redo-cache` (local cache)
  3. attempt to download files from `https://pdb-redo.eu/db/<pdb>/` into
     `--pdb-redo-cache` and re-check.
- Downloaded files mirror the names expected by the pipeline. For `final` we
  prefer uncompressed files but accept `.gz` and uncompress them into the
  cache so `prepare_inputs()` continues to work unchanged.
- Manual input mode bypasses the mirror layout and uses the supplied files
  directly, which is ideal for one-off tests or comparing user-downloaded data.

Failure modes
-------------
- Download failures for a single `--id` or each entry from `--id-file` cause the
  run to abort with a clear error message for that entry.
- Manual input mode fails fast when the user provides an invalid or missing file.
