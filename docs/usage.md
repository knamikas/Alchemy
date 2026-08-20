# Usage

Choose an input mode, then configure the options that apply to your run. For
the calculations that Alchemy performs, see the [method reference](method.md).
For batch execution and recovery, see the [operations guide](operations.md).

## Input modes

### Process a local PDB-REDO mirror

If you omit `--id` and `--id-file`, Alchemy enumerates the entries under
`--pdb-redo-root`. The default root is `/datasets/bioinfo/pdb-redo/`, with this
layout:

```text
ROOT/MIDDLE_TWO_ID_CHARACTERS/PDB_ID/
```

For example, run a capped batch from a local mirror with:

```bash
./alchemy \
    --pdb-redo-root /datasets/bioinfo/pdb-redo \
    --max-pdbs 20 \
    --ccp4-setup CCP4_SETUP
```

Replace `CCP4_SETUP` with the absolute path to the CCP4 setup script.

### Process requested PDB IDs

Use `--id` for one entry or `--id-file` for a list. Alchemy first checks the
configured mirror. If it doesn't find an entry there, it downloads the required
PDB-REDO files into `--pdb-redo-cache`, which defaults to `pdb-redo-cache/` in
the checkout.

```bash
./alchemy --id 9myr --ccp4-setup CCP4_SETUP
./alchemy --id-file PDB_ID_FILE --ccp4-setup CCP4_SETUP
```

The ID file can contain comma-, whitespace-, or newline-separated PDB IDs.

### Process manual files

Use `--mtz-file` with either `--pdb-file` or `--cif-file`. If you provide both
coordinate formats, Alchemy uses the mmCIF file. Manual mode processes one
structure, so don't combine it with `--id-file`.

For an mmCIF input, replace `PDB_ID` with the four-character entry ID and run:

```bash
./alchemy \
    --id PDB_ID \
    --cif-file /data/PDB_ID.cif \
    --mtz-file /data/PDB_ID.mtz \
    --data-json /data/PDB_ID_data.json \
    --ccp4-setup CCP4_SETUP
```

Omit `--id` if Alchemy can infer a four-character PDB ID from the filenames.
The optional `--data-json` file must contain a top-level `properties` object.
Alchemy uses its PDB-REDO metadata to calculate the diffraction precision index
(DPI). If you omit the file, Alchemy still measures and emits contact distances,
but DPI and derived z-scores remain unavailable. The manifest reports
`missing_dpi_metadata_source` instead of a calculation failure.

Use `--data-json` only with manual coordinate and MTZ inputs. Mirror and
download modes discover their own entry metadata.

If you explicitly provide an unreadable or invalid `--data-json` file, Alchemy
reports an input error. It doesn't fall back to the no-metadata behavior.

## Important options

| Option | Purpose |
| --- | --- |
| `--id PDB_ID` | Process one PDB ID. |
| `--id-file ID_FILE` | Process IDs from a comma-, whitespace-, or newline-separated file. |
| `--pdb-file PDB_FILE`, `--cif-file CIF_FILE`, `--mtz-file MTZ_FILE` | Process manually supplied structure data. |
| `--data-json DATA_JSON` | Supply optional PDB-REDO metadata for a manual run. |
| `--pdb-redo-root ROOT` | Set the local mirror root. |
| `--pdb-redo-cache CACHE_DIR` | Set the cache for downloaded entries. |
| `--pdb-metadata-cache CACHE_DIR` | Set the persistent cache for original-PDB crystallization records retrieved from the RCSB Data API. |
| `--no-crystallization-download` | Don't fetch missing original-PDB metadata. Alchemy still uses valid cache entries and coordinate-file fallbacks. |
| `--output-dir OUTPUT_DIR` | Set the result directory. |
| `--density-map-scope {model-envelope,full}` | Set the map extent passed to EDSTATS. The default model envelope retains every coordinate plus a 10 ångström border; `full` selects the complete-map path. |
| `--ccp4-timeout SECONDS` | Set the wall-clock limit for each CCP4 program; the default is 900 seconds per program. A timeout produces a retryable `partial` result and a log under `OUTPUT_DIR/ccp4_timeout_logs/`. |
| `--workers COUNT` | Set the worker-process ceiling; the value must be at least 1. Memory-aware admission can lower the active count. |
| `--memory-limit SIZE` | Override the memory capacity used for scheduling, such as `240G` or `16GiB`. A tighter host, cgroup, container, or scheduler limit still takes precedence. |
| `--memory-utilization FRACTION` | Set the maximum fraction of detected or configured memory used for worker estimates. The default is `0.8`, with a protected reserve where capacity permits. |
| `--max-pdbs COUNT` | Limit a run for testing. |
| `--resume` | Skip `ok` and terminal `partial` outcomes. Retry `skip`, `error`, and retryable `partial` outcomes without duplicating their previous rows. |
| `--retry-partials` | With `--resume`, also retry non-retryable `partial` entries. `--id` or `--id-file` can restrict the retry set. |
| `--log-dir LOG_DIR` | Set the run-report directory. The default is `OUTPUT_DIR/logs/`. |
| `--no-bonds` | Skip bond-distance analysis. A fresh run removes existing bond and candidate CSV files from the output directory. |
| `--confidence-reference-dir REFERENCE_DIR` | Score a single, ID-file, manual, or capped run against an existing frozen reference. An uncapped database run builds its own reference and ignores this option. |
| `-v`, `--verbose` | Add per-entry and per-CCP4-program diagnostics. |
| `--quiet` | Report only warnings and errors. |
| `--log-file LOG_FILE` | Also write full debug diagnostics to a file. This file is separate from the per-run report. |
| `--keep-intermediates` | Retain per-entry maps and logs. |
| `--ccp4-setup CCP4_SETUP` | Source and verify a CCP4 setup script for this run. |
| `--configure-ccp4 CCP4_SETUP` | Save a CCP4 setup-script path for later runs. |

Run `./alchemy --help` for the authoritative command-line defaults. See the
[confidence-scoring method](method.md#database-referenced-confidence-scoring)
before comparing site classifications or empirical scores.
