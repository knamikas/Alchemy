# Tasks: Implementation steps

1. Add CLI flag `--pdb-redo-cache` and wire default path.
2. Implement `download_entry_to_cache(pdbID, cache_root, state)` to fetch files
   from `https://pdb-redo.eu/db/<pdb>/` and place them in the mirror layout.
3. Implement `ensure_entry_available()` that checks mirror -> cache -> downloads
   and returns the root to use for subsequent pipeline steps.
4. Integrate `ensure_entry_available()` into `main()` for `--id` and `--id-file`.
5. Add `--id-file` parsing for comma-separated and/or newline-separated ID lists.
6. Add manual input mode with `--pdb-file`, `--mtz-file`, and optional
   `--cif-file` / `--data-json` support.
7. Add basic network and file error handling with clear user-facing messages.
8. Add openspec artifacts (proposal.md, design.md, tasks.md) and mark change ready.
