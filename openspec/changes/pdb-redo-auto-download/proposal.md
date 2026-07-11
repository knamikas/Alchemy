# PDB-REDO Auto-download and Cache

What & Why
---------

Improve the user-facing input workflow for the Alchemy pipeline so that users can
run analyses in several ways without requiring a full local PDB-REDO mirror.
This makes the tool easier to use for quick tests, curated structure lists, and
manual local-file workflows while preserving the existing mirror-based batch mode.

The change now supports:
- `--id 1cbs` to run a single PDB ID, automatically using a local mirror if
  available or downloading needed files into a local cache otherwise.
- `--id-file ids.txt` to run a user-provided list of PDB IDs from a simple
  comma-separated and/or newline-separated text file.
- `--pdb-file`, `--mtz-file`, and optionally `--cif-file` / `--data-json` to
  run from manually supplied local files instead of requiring the mirror layout.

Key outcomes:
- A basic user can run the pipeline with a plain PDB ID and need not understand
  the mirror/cache internals.
- Advanced users can still point to a full local mirror with `--pdb-redo-root`.
- Curated structure lists can be provided directly through `--id-file`.
- Users who already have local structure and reflection files can run the pipeline
  without downloading or reorganizing them into a mirror directory.
- Downloads are stored in a mirror-like layout under `pdb-redo-cache/` so the
  rest of the pipeline stays consistent.
