# Usage

Choose an input mode, then configure the options that apply to your run. For the
calculations that Alchemy performs, see the [method reference](method.md). For
batch execution and recovery, see the [operations guide](operations.md).

The examples run `./alchemy` from the checkout. On Windows, run `python alchemy`
instead.

## Input modes

### Process a local PDB-REDO mirror

If you omit `--id` and `--id-file`, Alchemy enumerates the entries under
`--pdb-redo-root`. You must supply the path to your local mirror; there is no
default mirror location. The mirror must use this layout:

```text
ROOT/MIDDLE_TWO_ID_CHARACTERS/PDB_ID/
```

Each entry directory must contain `PDB_ID_final.mtz` and either
`PDB_ID_final.cif` or `PDB_ID_final.pdb`, each optionally gzipped. Alchemy reads
the entry's `data.json` for DPI metadata. Directories without the final model
files are skipped without a manifest row.

For example, run a capped batch from a local mirror with:

```bash
./alchemy \
    --pdb-redo-root /path/to/pdb-redo \
    --max-pdbs 20
```

Replace `/path/to/pdb-redo` with your mirror directory.

### Process requested PDB IDs

Use `--id` for one entry or `--id-file` for a list. A local mirror is optional.
Alchemy checks the mirror if `--pdb-redo-root` is supplied, then checks
`--pdb-redo-cache` and downloads any missing PDB-REDO files into that cache. The
cache defaults to `pdb-redo-cache/` in the checkout.

```bash
./alchemy --id 9myr
./alchemy --id-file PDB_ID_FILE
```

The ID file can contain comma-, whitespace-, or newline-separated PDB IDs.

### Process manual files

Use `--mtz-file` with either `--pdb-file` or `--cif-file`. The command rejects
`--pdb-file` and `--cif-file` together. Manual mode processes one structure, so
don't combine it with `--id-file`.

For an mmCIF input, replace `PDB_ID` with the four-character entry ID and run:

```bash
./alchemy \
    --id PDB_ID \
    --cif-file /data/PDB_ID.cif \
    --mtz-file /data/PDB_ID.mtz \
    --data-json /data/PDB_ID_data.json
```

Omit `--id` if Alchemy can infer a four-character PDB ID from the filenames.
Inference works when a file name starts with the ID, alone or followed by an
underscore, such as `1abc.cif` or `1abc_final.cif`. The optional `--data-json`
file must contain a top-level `properties` object. Alchemy uses its PDB-REDO
metadata to calculate the diffraction precision index (DPI). If you omit the
file, Alchemy still measures and emits contact distances, but DPI and derived
z-scores remain unavailable. The entry then ends with `status=partial` and the
reason code `missing_dpi_metadata_source` instead of a calculation failure. An
ordinary `--resume` doesn't retry that `partial` outcome.

Use `--data-json` only with manual coordinate and MTZ inputs. Mirror and
download modes discover their own entry metadata.

If you explicitly provide an unreadable or invalid `--data-json` file, Alchemy
reports an input error. It doesn't fall back to the no-metadata behavior.

## Locate CCP4

No CCP4 option is needed when Alchemy can find CCP4 on its own. It looks in
this order:

1. The setup script given by `--ccp4-setup SETUP_SCRIPT`, for this run only.
2. The CCP4 programs already on `PATH`, for example in a shell where CCP4 has
   been set up.
3. The setup script named by the `CCP4_SETUP` environment variable.
4. The setup script saved by `./alchemy --configure-ccp4 SETUP_SCRIPT`. That
   command verifies the script, saves its path to `~/.config/alchemy/ccp4.json`,
   and exits without running an analysis.
5. A setup script in a common CCP4 install location.

If none of these finds CCP4, the run stops with a message that the required
tools weren't found; name a setup script then, or when you want a different
installation. On Linux and macOS, use the Bourne-shell script
`ccp4.setup-sh` from the CCP4 installation, often in its `bin` directory; the
csh variant isn't supported. On Windows, use `ccp4.setup.bat` or
`ccp4.setup.cmd`.

## Important options

| Option | Purpose |
| --- | --- |
| `--id PDB_ID` | Process one PDB ID. |
| `--id-file ID_FILE` | Process IDs from a comma-, whitespace-, or newline-separated file; `#` starts a comment. |
| `--pdb-file PDB_FILE`, `--cif-file CIF_FILE`, `--mtz-file MTZ_FILE` | Process manually supplied structure data. |
| `--data-json DATA_JSON` | Supply optional PDB-REDO metadata for a manual run. |
| `--pdb-redo-root ROOT` | Set the local mirror root. |
| `--pdb-redo-cache CACHE_DIR` | Set the cache for downloaded entries. |
| `--pdb-metadata-cache CACHE_DIR` | Set the persistent cache for original-PDB crystallization records retrieved from the RCSB Data API. |
| `--no-crystallization-download` | Don't fetch missing original-PDB metadata. Alchemy still uses valid cache entries and coordinate-file fallbacks. |
| `--output-dir OUTPUT_DIR` | Set the result directory. The default is `output/` in the checkout. A run without `--resume` replaces the results already in this directory. |
| `--density-map-scope {model-envelope,full}` | Set the map extent passed to EDSTATS. The default model envelope retains every coordinate plus a 10 ångström border, and falls back to the full map when the crop would not be smaller or safe; `full` selects the complete-map path directly. |
| `--ccp4-timeout SECONDS` | Set the wall-clock limit for each CCP4 program; the default is 900 seconds per program. A timeout produces a retryable `partial` result and a log under `OUTPUT_DIR/ccp4_timeout_logs/`. |
| `--workers COUNT` | Set the worker-process ceiling; the value must be at least 1. By default, Alchemy chooses the count from the available CPUs and memory. Memory-aware admission can lower the active count. |
| `--memory-limit SIZE` | Override the memory capacity used for scheduling, such as `8G` or `16GiB`. A tighter host, cgroup, container, or scheduler limit still takes precedence. |
| `--memory-utilization FRACTION` | Set the maximum fraction of detected or configured memory used for worker estimates. The default is `0.8`, with a protected reserve where capacity permits. |
| `--max-pdbs COUNT` | Limit a run for testing. |
| `--resume` | Skip `ok` and terminal `partial` outcomes. Retry `skip`, `error`, and retryable `partial` outcomes without duplicating their previous rows. Resuming requires the existing output headers and `analysis_config_id` to match the current run; use a new output directory when they differ. |
| `--retry-partials` | With `--resume`, also retry non-retryable `partial` entries. `--id` or `--id-file` can restrict the retry set. |
| `--log-dir LOG_DIR` | Set the run-report directory. The default is `OUTPUT_DIR/logs/`. |
| `--no-bonds` | Skip bond-distance analysis. The run produces no scores or review queue. A fresh run removes existing bond and candidate CSV files from the output directory. |
| `--score-reference-dir DIR` | Score a single, ID-file, manual, or capped run against an existing frozen reference. If `DIR` has no usable `metadata.json`, the run stops with an error and exit code 1. An uncapped database run builds its own reference and ignores this option. |
| `-v`, `--verbose` | Add per-entry and per-CCP4-program diagnostics. |
| `--quiet` | Report only warnings and errors. |
| `--log-file LOG_FILE` | Also write full debug diagnostics to a file. This file is separate from the per-run report. |
| `--keep-intermediates` | Retain per-entry maps and logs. |
| `--ccp4-setup SETUP_SCRIPT` | Source and verify a CCP4 setup script for this run. |
| `--configure-ccp4 SETUP_SCRIPT` | Verify a CCP4 setup script and save its path to `~/.config/alchemy/ccp4.json` for later runs, then exit without running an analysis. |

Run `./alchemy --help` for the authoritative command-line defaults. See the
[scoring method](method.md#database-referenced-scoring)
before comparing site classifications or empirical scores.
