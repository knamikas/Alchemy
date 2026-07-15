# Alchemy Pipeline

A collection of research scripts that evaluate **how well metal atoms in protein crystal
structures are supported by experimental electron density**.

For a list of PDB IDs, the pipeline downloads (or reads from a local mirror) re-refined data
from [PDB-REDO](https://pdb-redo.eu/), computes electron-density maps and per-atom real-space
statistics (RSZD / RSR via CCP4 `edstats`), isolates the lines for metals and metal-containing
cofactors, optionally renders map images, and computes metal–ligand bond-length deviations
(z-scores) against literature reference distances.

> Originated in the BioXFEL project; the `*_kn.py` files are Python ports of earlier scripts.
> There is **no package, build system, or test suite** — scripts are run directly with
> `python <path/to/script>.py`, and `src/main.py` is the batch entry point.

---

## Quick start

The fastest path is the batch driver `src/main.py`, which runs the core pipeline over the local
PDB-REDO mirror without downloading anything:

```bash
# Smoke test: 109m = myoglobin, expect an FE row
conda run -n metal python src/main.py --id 109m --ccp4-setup <path/to/ccp4.setup-sh>

# A handful of structures with 6 metals (300d) and a no-metal DNA control (100d)
conda run -n metal python src/main.py --id 300d --ccp4-setup <…>
conda run -n metal python src/main.py --id 100d --ccp4-setup <…>

# Run a selected list of structures from a file
conda run -n metal python src/main.py --id-file src/data/sample_ids_list.txt --ccp4-setup <…>
```

Results stream to `output/metal_stats_all.csv` (real-space stats),
`output/metal_bonds_all.csv` (metal–ligand bond z-scores), and `output/manifest.csv`.

---

## Dependencies

- **CCP4 suite** on `PATH` — `fft`, `edstats` (core pipeline), plus `mtzdump` / `ccp4mg`
  (legacy / image rendering). The core pipeline is unusable without CCP4. Pass
  `--ccp4-setup <…/ccp4.setup-sh>` to `src/main.py` (it is sourced in a subshell and verified at
  startup) or pre-source it.
- **Python packages**: `requests`, `sh` (for `gunzip`), `gemmi`, `biopython` (`Bio.PDB`),
  `numpy`, `pandas`, `seaborn`, `matplotlib`. The `metal`, `biotools`, and `vinda` conda envs
  already have gemmi + Biopython.

There is no `requirements.txt`.

---

## Batch pipeline — `src/main.py`

`src/main.py` is the primary entry point for running the pipeline (stages 2 → 3, plus the
bond-distance stage) across a
local PDB-REDO mirror (default `/datasets/bioinfo/pdb-redo/`, layout
`<root>/<id[1:3]>/<id>/`, ~24.6k entries). It does not download anything — it reads each
entry's files in place and writes only under `--output-dir` (default `./output/`).

**Per entry it:**

1. Selects the refinement state (`--refine-state {final,0cyc,besttls}`, default `final` =
   `{id}_final.{mtz,pdb}`, uncompressed; `0cyc` / `besttls` are gunzipped, and `0cyc` is
   CIF→PDB-converted via gemmi into the work dir).
2. Reads resolution from the entry's `data.json` (`DATARESL` / `DATARESH`, with a gemmi MTZ
   fallback).
3. Calls `run_alchemy`, then `run_analysis`, then `run_bond_analysis` (unless `--no-bonds`).

**Key options:**

| Option | Purpose |
| --- | --- |
| `--id <pdbid>` | Run a single structure. |
| `--id-file <path>` | Run a list of structures from a comma- and/or newline-separated file. |
| `--refine-state {final,0cyc,besttls}` | Which refinement files to use (default `final`). |
| `--output-dir <path>` | Output directory (default `./output/`). |
| `--workers <n>` | Parallelism via `multiprocessing.Pool` (default `cpu_count() - 2`). |
| `--max-pdbs <n>` | Cap the run (early-stop enumeration) for quick tests. |
| `--resume` | Skip IDs already present in the manifest. |
| `--no-bonds` | Skip the bond-distance stage (write edstats stats only). |
| `--keep-intermediates` | Retain per-entry maps / logs (otherwise the per-entry work dir is deleted after extraction, so a full 24k run stays small on disk). |
| `--ccp4-setup <path>` | Source a `ccp4.setup-sh` in a subshell (fail-fast verified at startup). |

**Outputs** (all flushed per entry, so long runs survive interruption):

- `output/metal_stats_all.csv` — `pdbID` + edstats columns, one row per metal / cofactor atom.
- `output/metal_bonds_all.csv` — one row per metal–ligand bond: distance, literature reference
  distance / stdev, resolution-aware z-score, DPI, the `edstats` sigmas, and bond metadata.
- `output/manifest.csv` — `pdbID,status,n_metals,n_bonds,runtime_s,error`.

---

## Pipeline stages

The scripts run in sequence; each consumes files written by the previous one into a single
shared working directory.

### 1. `legacy/Assistant_kn.py` — fetch
Downloads `{pdb}_0cyc.mtz(.gz)` and `{pdb}_0cyc.pdb(.gz)` from PDB-REDO and gunzips them.
*(Superseded by `src/main.py` for the local-mirror workflow, which reads files in place.)*

### 2. `src/Alchemy_kn.py` — maps & real-space stats
Exposes `run_alchemy(pdbID, mtz_path, pdb_path, out_dir, reslo, reshi, env)`: runs CCP4 `fft`
(Fo and difference maps) + `edstats`, producing `{pdb}_stats.out` / `_rszd.pdb` / `_qq.out` /
logs in `out_dir`. The MTZ must have `FWT / PHWT / DELFWT / PHDELWT` columns.

> The old `mtzdump` resolution-scrape, the `7cup` special case, and the anomalous-map branch
> were dropped from the core path; resolution is now passed in by the caller.

### 3. `src/Analysisv2_kn.py` — parse metals & cofactors
Exposes `load_cofactors()` and `run_analysis(pdbID, stats_out, metals_set, cofactor_set)`,
which parses `{pdb}_stats.out` and returns structured rows for metal ions and metallocofactors.
Matching is on the parsed `fields[0]` token (fixes the old `startswith(f"{x:<4}")`
4-char / 5-char-CCD bug). A `__main__` block reproduces the legacy per-structure file dump.

### 4. `legacy/Autoplotv3_kn.py` — render map images
Reads the `*_Data` files, finds metal coordinates in `{pdb}_rszd.pdb`, fills a `default.mgpic`
template with map contour levels (from `_edstats.log`), and calls `ccp4mg` to render images.

> ⚠️ Image rendering is reported as **broken on modern Ubuntu** (works on Ubuntu 16).

### 5. `src/bond_analysis.py` — bond-distance analysis (integrated into `src/main.py`)
Exposes `run_bond_analysis(pdbID, pdb_path, entry_dir, stats_rows, dpi_inputs)`: uses Biopython
to find **all** metal atoms, runs a 4 Å neighbor search, and for each metal–ligand contact to a
coordinating residue/water computes the bond length and a resolution-aware z-score against
`src/data/metal_distances_info.txt`:

```
z = (d_observed − μ) / sqrt(DPI² + σ_lit²)
```

The structure's DPI (Blow 2002 eq. 7 — the per-atom coordinate uncertainty) is reconstructed
from PDB-REDO `data.json` (`NREFCNT`, `RFFIN`) and the asymmetric-unit volume computed directly
from the crystal cell ÷ symmetry operations via gemmi (the Matthews coefficient the legacy code
scraped is absent from PDB-REDO files). edstats sigmas are joined in-process from the
`run_analysis` rows. Bonds with no literature reference (e.g. Ni, uncommon metals) or a missing
DPI input still emit the measured distance with NaN in the derived columns.

> Generalized port of the legacy **`legacy/fe_biopython_analysis_dpi_final.py`** (Fe-only, multi-source
> literature file, header-scraped DPI), which remains in the repo for reference but is no longer
> used. The port also fixes a key-matching bug that had silently dropped all His-N / Cys-S bonds.

---

## Supporting / one-off scripts

- **`scripts/Alloy_kn.py`** — scans the PDB Chemical Component Dictionary (`components.cif`, fetched via
  gemmi) for any component whose formula or atoms contain a metal, regenerating
  `src/data/metallocofactors_id.txt`. Run this to refresh the cofactor list (the CCD updates weekly).
- **`scripts/metal_pdb_read.py`** — extracts PDB-ID lists from [MetalPDB](https://metalweb.cerm.unifi.it/)
  `.tsv` exports (search MetalPDB by a metal of interest, export, then extract the IDs).

---

## Reference data files (committed)

- **`src/data/metallocofactors_id.txt`** — `{CCD_id}\t{formula}` for all metal-containing CCD components
  (current as of 2025-06-16). Consumed by `src/Analysisv2_kn.py`; regenerated by `scripts/Alloy_kn.py`.
- **`src/data/metal_distances_info.txt`** — `residue atom metal avg_bond_dist st_dev` reference bond
  lengths (Harding 2006). Consumed by the bond-distance analysis.

---

## Critical conventions before running anything

- **Hardcoded absolute paths.** The standalone `*_kn.py` scripts have machine-specific paths
  near the top (and inside `src/Alchemy_kn.py`'s `edstats` calls and `find_dist` / `get_sigma` in
  the bond script). These must be edited for the current machine — the standalone scripts will
  not run unmodified. `src/main.py` uses repo-relative paths and is the maintained path.
- **`debug` flag at the top of most files** toggles hardcoded inputs vs. interactive `input()`
  prompts, and in `src/Alchemy_kn.py` / `legacy/Autoplotv3_kn.py` toggles hardcoded test data or disables
  image generation. Check it before running a standalone script.
- **Standalone input** is a comma-separated list of 4-char PDB IDs in a text file (default
  `alchemyTest.txt`), read and split on `,`. (`src/main.py` instead takes `--id` / `--id-file`.)
- **Standalone scripts append (`'a'`)** to output files, and several `os.makedirs` / `os.mkdir`
  calls fail if the target already exists — delete prior `{pdb}metals/`, `{pdb}images/`, and
  stale output files before re-running, or you will get duplicated rows / errors. `src/main.py`
  manages its own work dirs and supports `--resume`.
- **`edstats` column layout matters.** Metal sigma values are read by fixed column index from
  `*_stats.out` / `*_Data` lines (fields `[12]`, `[13]`, `[14]` for magnitude / neg / pos;
  field `[12]` is the sort key). Don't assume whitespace-tolerant parsing elsewhere.

---

## Known issues / rough edges (in the committed standalone scripts)

These standalone scripts contain bugs that prevent execution as-is (treat fixing them as
expected work):

- **`legacy/Assistant_kn.py`** — unterminated f-strings (mismatched `'` / `"` on the URL lines),
  `if debug = 1:` (assignment, not `==`), and mis-indented `with` blocks.
- **`legacy/fe_biopython_analysis_dpi_final.py`** — stray `(` in
  `elif (neighbor.get_name().startswith("O"):`, and references to undefined names
  (`directory_use`, `metal_directory_use`, `directory_old`, hardcoded
  `start_from = pdbList.index('7die')`).
- **`src/Analysisv2_kn.py`** — legacy cofactor matching used `startswith` on the first 4 chars, so
  newer 5-char CCD IDs were mishandled (fixed in the `run_analysis` token match used by
  `src/main.py`).
