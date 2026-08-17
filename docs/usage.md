# Usage

How to point Alchemy at coordinates and what the options do. For what it then
computes, see [method.md](method.md); for running batches and reading the
outputs, [operations.md](operations.md).

## Input modes

### Local PDB-REDO mirror

Without `--id` or `--id-file`, the pipeline enumerates entries under
`--pdb-redo-root` (default `/datasets/bioinfo/pdb-redo/`). The expected layout is:

```text
<root>/<middle-two-id-characters>/<pdb-id>/
```

### Requested PDB IDs

With `--id` or `--id-file`, the pipeline checks the configured mirror first. If
an entry is unavailable there, it downloads the required PDB-REDO files into
`--pdb-redo-cache` (default `./pdb-redo-cache/`).

### Manual files

Use `--mtz-file` with either `--pdb-file` or `--cif-file`. `--data-json` can
provide optional PDB-REDO metadata for the DPI calculation. Supply `--id` if a
four-character PDB ID cannot be inferred from the filenames. Manual mode
processes exactly one structure, so it cannot be combined with `--id-file`.
`--data-json` is accepted only with manual coordinate and MTZ inputs; automatic
mirror and download modes discover their own entry metadata.

Without `--data-json` there is no reflection count, so DPI and every value
derived from it are unavailable. Bond geometry is still measured and emitted,
and the omission is reported as `missing_dpi_metadata_source` rather than as a
calculation failure.

When `--data-json` is supplied explicitly, it must name a readable, valid JSON
file containing a top-level `properties` object. Invalid explicit metadata is
an input error rather than a request to use the no-metadata fallbacks.

## Important options

| Option | Purpose |
| --- | --- |
| `--id <pdbid>` | Process one PDB ID. |
| `--id-file <path>` | Process IDs from a comma-, whitespace-, or newline-separated file. |
| `--pdb-file`, `--cif-file`, `--mtz-file` | Process manually supplied structure data. |
| `--pdb-redo-root <path>` | Set the local mirror root. |
| `--pdb-redo-cache <path>` | Set the cache for downloaded entries. |
| `--pdb-metadata-cache <path>` | Set the persistent per-entry cache for original-PDB crystallization records retrieved from the RCSB Data API. |
| `--no-crystallization-download` | Run without fetching missing original-PDB metadata; valid cache entries and coordinate-file fallbacks are still used. |
| `--output-dir <path>` | Set the result directory. |
| `--density-map-scope {model-envelope,full}` | Set the map extent passed to EDSTATS. The default model envelope retains every coordinate plus a 10 Angstrom border; `full` selects the legacy complete-map path. |
| `--ccp4-timeout <s>` | Per-program wall-clock budget for each CCP4 step (`mtzfix`, `fft`, `mapmask`, `edstats`), in seconds; default 900. The budget applies to each program separately, not to the entry as a whole. A program that exceeds it is killed and the entry becomes a retryable `partial` with reason `ccp4_tool_timeout`; its partial log is copied to `<output-dir>/ccp4_timeout_logs/`. Raise it for exceptionally large structures. |
| `--workers <n>` | Set the worker-process ceiling; must be at least 1. The default leaves two detected CPUs free and also caps the pool from the usable memory budget. An explicit count does not disable byte-budget admission or the live memory guard. |
| `--memory-limit <size>` | Override the memory capacity used for scheduling, for example `240G` or `16GiB`. Auto-detection remains the default. A tighter detected host, cgroup, container, or scheduler allowance still wins. |
| `--memory-utilization <fraction>` | Set the maximum fraction of detected or configured memory available to worker estimates; default `0.8`. Alchemy still protects at least 4 GiB where the allocation is large enough. All entries share the resulting byte budget, with no fixed high-memory percentage. Live pressure or a worker death reduces future admission automatically. |
| `--max-pdbs <n>` | Limit a run for testing. |
| `--resume` | Skip `ok` and terminal `partial` outcomes; retry `skip`, `error`, and retryable `partial` outcomes without duplicating their previous rows. |
| `--retry-partials` | With `--resume`, retry all `partial` entries recorded in the manifest. Successful `ok` entries remain skipped; `--id` or `--id-file` may restrict the retry set. |
| `--log-dir` | Directory for run reports and their companion entry-diagnostics tables. Defaults to `<output-dir>/logs/`, keeping one immutable report pair per invocation out of the directory holding the scientific result CSVs. |
| `--no-bonds` | Skip bond-distance analysis. A fresh run removes previous `metal_bonds_all.csv` and `metal_contact_candidates_all.csv` files in the output directory. |
| `--confidence-reference-dir <path>` | Score confidence against an existing frozen reference instead of the one in the output directory. Applies to single, ID-file, manual, and capped runs; an uncapped full-database run builds its own reference and ignores this. See [method.md](method.md) for what the score claims. |
| `-v`, `--verbose` | Increase diagnostic detail. The default reports the run narrative; `-v` adds per-entry and per-CCP4-program records from inside the worker processes. |
| `--quiet` | Report warnings and errors only. |
| `--log-file <path>` | Also write full debug-level diagnostics to a file, whatever the console verbosity. Independent of the per-run report in `alchemy_run_*.log`. |
| `--keep-intermediates` | Retain per-entry maps and logs. |
| `--ccp4-setup <path>` | Source and verify a CCP4 setup script for this run. |
| `--configure-ccp4 <path>` | Save a CCP4 setup script path for later runs. |
