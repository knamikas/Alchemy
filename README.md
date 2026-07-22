# Alchemy Pipeline

Alchemy evaluates how well metal atoms in protein crystal structures are
supported by experimental electron density. For each PDB entry, it calculates
2mFo-DFc and mFo-DFc maps, extracts per-residue real-space statistics, identifies
metal ions and metal-containing cofactors, and measures metal-ligand bond
geometry against literature reference distances.

`src/main.py` is the maintained batch entry point. It can read a local PDB-REDO
mirror, download explicitly requested entries into a local cache, or process
manually supplied coordinate and MTZ files.

## Quick start

```bash
# Smoke test: 109m = myoglobin, expect an FE row
conda run -n metal python src/main.py --id 109m \
    --ccp4-setup <path/to/ccp4.setup-sh>

# Additional examples: 300d has six metals; 100d is a no-metal DNA control
conda run -n metal python src/main.py --id 300d --ccp4-setup <…>
conda run -n metal python src/main.py --id 100d --ccp4-setup <…>

# Run IDs from a comma-, whitespace-, or newline-separated file
conda run -n metal python src/main.py \
    --id-file path/to/pdb_ids.txt --ccp4-setup <…>
```

Results are written to:

- `output/metal_stats_all.csv` — real-space statistics for metals and
  metal-containing cofactors.
- `output/metal_bonds_all.csv` — metal-ligand distances and reference-based
  z-scores.
- `output/manifest.csv` — per-entry status, counts, runtime, and errors.

## Dependencies

- **CCP4** with `fft` and `edstats` on `PATH`. Pass
  `--ccp4-setup <path/to/ccp4.setup-sh>` or configure the path once with
  `--configure-ccp4`.
- **Python packages:** `requests`, `gemmi`, and `biopython` (`Bio.PDB`).

The `metal`, `biotools`, and `vinda` Conda environments used by this project
already contain gemmi and Biopython. There is currently no `requirements.txt`.

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
four-character PDB ID cannot be inferred from the filenames.

## Pipeline flow

### 1. Input preparation — `src/main.py`

The driver selects the requested refinement state:

- `final`: uncompressed `{id}_final.mtz` and `{id}_final.pdb`.
- `besttls`: compressed best-TLS MTZ and PDB files.
- `0cyc`: compressed zero-cycle MTZ and mmCIF files; gemmi converts the mmCIF
  coordinates to PDB format in the work directory.

Resolution limits come from PDB-REDO `data.json` when available, with an MTZ
fallback through gemmi.

### 2. Maps and real-space statistics — `src/density_analysis.py`

`run_density_analysis()` runs:

1. CCP4 `fft` with `FWT/PHWT` to produce a 2mFo-DFc map.
2. CCP4 `fft` with `DELFWT/PHDELWT` to produce an mFo-DFc difference map.
3. CCP4 `edstats` to produce per-residue real-space statistics and an RSZD
   coordinate file.

The MTZ input must contain `FWT`, `PHWT`, `DELFWT`, and `PHDELWT` columns.

### 3. Metal and cofactor identification — `src/metal_identification.py`

`extract_metal_statistics()` parses the edstats table. Plain metal ions are identified from
their actual atom elements in the parsed structure. Metal-containing cofactors
are matched against the Chemical Component Dictionary list maintained by
`src/build_metallocofactor_catalog.py`.

### 4. Bond-distance analysis — `src/bond_analysis.py`

`run_bond_analysis()` uses Biopython to find configured metal elements and their
N/O/S ligand contacts within 4 Å. Where a literature reference is available, it
calculates:

```text
z = (d_observed - mu) / sqrt(DPI^2 + sigma_lit^2)
```

The DPI is calculated from PDB-REDO reflection and R-free metadata, the
asymmetric-unit volume, and occupancy-weighted atom counts. Contacts without a
reference distance or complete DPI inputs are still emitted with their measured
geometry and NaN derived values.

## Important options

| Option | Purpose |
| --- | --- |
| `--id <pdbid>` | Process one PDB ID. |
| `--id-file <path>` | Process IDs from a comma-, whitespace-, or newline-separated file. |
| `--pdb-file`, `--cif-file`, `--mtz-file` | Process manually supplied structure data. |
| `--refine-state {final,0cyc,besttls}` | Choose the PDB-REDO refinement state. |
| `--pdb-redo-root <path>` | Set the local mirror root. |
| `--pdb-redo-cache <path>` | Set the cache for downloaded entries. |
| `--output-dir <path>` | Set the result directory. |
| `--workers <n>` | Set multiprocessing parallelism. |
| `--max-pdbs <n>` | Limit a run for testing. |
| `--resume` | Skip entries completed with `ok`; retry `partial`, `skip`, and `error` entries without duplicating their existing rows. |
| `--no-bonds` | Skip bond-distance analysis. |
| `--keep-intermediates` | Retain per-entry maps and logs. |
| `--refresh-cofactors` | Force a fresh wwPDB cofactor-list build. |
| `--ccp4-setup <path>` | Source and verify a CCP4 setup script for this run. |
| `--configure-ccp4 <path>` | Save a CCP4 setup script path for later runs. |

## Cofactor reference maintenance — `src/build_metallocofactor_catalog.py`

Before processing entries, the pipeline checks its metal-containing cofactor
list. A current list in the user cache is reused; a missing or stale list is
rebuilt from the wwPDB Chemical Component Dictionary. Refreshed files are stored
under the user cache (normally `~/.cache/alchemy/`) so normal runs do not modify
the repository. If refreshing fails, the pipeline falls back to an available
cached or committed list.

## Supporting utility — `src/scripts/extract_metalpdb_ids.py`

This standalone data-preparation utility extracts four-character PDB IDs from a
manually downloaded MetalPDB TSV report. It is not imported by the analysis
pipeline.

## Reference data

- `src/data/metallocofactors_id.txt` — committed fallback list of
  metal-containing Chemical Component Dictionary IDs.
- `src/data/metallocofactors_id.meta.json` — generation metadata for the
  committed cofactor list.
- `src/data/metal_distances_info.txt` — reference metal-ligand distances and
  standard deviations.

## Operational notes

- Per-entry maps and logs are removed after their rows are extracted unless
  `--keep-intermediates` is supplied.
- Output CSV handles are flushed after each processed entry so interrupted batch
  runs retain completed results.
- A failure in bond analysis does not discard real-space-statistics rows already
  calculated for that entry; the manifest records the entry as `partial` with
  the bond-stage error.
