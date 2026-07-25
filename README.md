# Alchemy Pipeline

Alchemy evaluates how well metal atoms in protein crystal structures are
supported by experimental electron density. For each PDB entry, it calculates
2mFo-DFc and mFo-DFc maps, extracts per-residue real-space statistics, identifies
metal ions and metal-containing cofactors, and measures first-sphere
metal-coordination geometry against literature reference distances. Generated
contacts can arise from crystallographic symmetry, strict noncrystallographic
symmetry (NCS), or both. They describe the deposited structure model and do not
by themselves establish that a metal or ligand is biologically functional.

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

- `output/metal_stats_all.csv` — one row per selected metal site, with
  real-space statistics, DPI and validation provenance, and explicit-only
  versus image-inclusive contact summaries. A cofactor containing multiple
  selected metal sites repeats its residue-level EDSTATS values once per site.
  Diagnostic rows for cofactors that cannot be matched to a coordinate site may
  have blank site-specific fields; `coordinate_mapping_status` and
  `selected_metal_site_status` distinguish failed joins from cofactors without
  a selected metal. These permanent limitations are also reported as terminal
  `partial` manifest outcomes and are not counted as metal sites.
- `output/metal_bonds_all.csv` — one row per retained first-sphere contact,
  including distance, reference-based z-score, conformer selection, and
  separate crystallographic-symmetry and strict-NCS provenance.
- `output/manifest.csv` — per-entry status and reason, runtime, input
  provenance, and relevant software and analysis-policy versions. Its
  `n_metals` value counts distinct selected coordinate-model sites, not
  diagnostic or repeated EDSTATS rows.

## Dependencies

- **CCP4** with `mtzfix`, `fft`, and `edstats` on `PATH`. Pass
  `--ccp4-setup <path/to/ccp4.setup-sh>` or configure the path once with
  `--configure-ccp4`.
- **Python package:** `gemmi>=0.7.0`, as declared in `pyproject.toml`.

Install the Python dependencies in the environment used to run Alchemy:

```bash
python -m pip install "gemmi>=0.7.0"
```

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

The automatic PDB-REDO workflow analyzes only the final re-refined and rebuilt
model. It prefers `{id}_final.cif` as the authoritative coordinate source and
converts it to an EDSTATS-compatible analysis PDB with gemmi. The
`{id}_final.pdb` compatibility export is used only when mmCIF is unavailable.
Alternative coordinate/MTZ pairs can still be supplied through the manual-file
options; when both coordinate formats are supplied, mmCIF takes precedence.

During mmCIF conversion, `.` and `?` occupancy values are written as blank PDB
occupancies rather than being replaced by `1.00`. Alchemy also embeds a
reversible mapping for component identifiers that exceed the three-character
legacy PDB residue-name field. The original CCD identifier is restored after
EDSTATS so cofactor catalog matching and output retain the mmCIF identity.

The overall diffraction resolution comes from PDB-REDO `data.json` when
available, with an MTZ fallback through gemmi. EDSTATS instead receives the
common finite resolution range of `FWT`, `PHWT`, `DELFWT`, and `PHDELWT`, which
are the columns used to calculate its two maps.

### 2. Maps and real-space statistics — `src/density_analysis.py`

`run_density_analysis()` runs:

1. CCP4 `mtzfix` to check and, when needed, correct the Fourier map
   coefficients, including their centric and acentric consistency. If the input
   passes, MTZFIX intentionally writes no replacement and the original MTZ is
   used.
2. CCP4 `fft` with `FWT/PHWT` from that validated MTZ to produce a 2mFo-DFc map.
3. CCP4 `fft` with `DELFWT/PHDELWT` from that validated MTZ to produce an
   mFo-DFc difference map.
4. CCP4 `edstats` to produce per-residue real-space statistics and an RSZD
   coordinate file.

The MTZ input must contain `FWT`, `PHWT`, `DELFWT`, and `PHDELWT` columns.

### 3. Metal and cofactor identification — `src/metal_identification.py`

`extract_metal_statistics()` parses the edstats table. Alchemy reads the
deposited PDB element field directly; blank or invalid fields are marked unknown
rather than inferred from atom names. During mmCIF conversion, the explicit
`_atom_site.type_symbol` is written into that PDB field. Metal-containing
cofactors are matched against the Chemical Component Dictionary list maintained
by `src/build_metallocofactor_catalog.py`. A structure with unknown elements
does not receive a DPI because its non-hydrogen atom count is indeterminate.
EDSTATS' `_` marker is normalized to a blank chain identifier before coordinate
matching.

### 4. Bond-distance analysis — `src/bond_analysis.py`

Gemmi parses the same coordinate representation supplied to EDSTATS. For a
multi-model structure, Alchemy analyzes the first model only for metal
identification, DPI, and contact searching; model count and the selected-model
policy are recorded with the results. Before map statistics are calculated,
Alchemy writes a first-model-only PDB and supplies that exact coordinate file
to both EDSTATS and Gemmi. Atoms and density statistics from different models
are never combined.

Alternative conformations are selected coherently per residue. Blank-altloc
atoms are shared, while the named conformer with the highest mean valid atomic
occupancy is selected (ties are resolved by altloc label). This avoids creating
an artificial residue by choosing A/B alternatives independently for each atom.
Every neighboring residue is considered independently, and the selected and
available alternatives are recorded.

`run_bond_analysis()` searches broadly for positive-occupancy N/O/S atoms no
more than 4 Å from a configured metal, outside the metal's own residue, in a
recognized amino acid or water. It then retains only first-coordination-sphere
contacts. Following [Harding's coordination-group
definition](https://doi.org/10.1107/S0907444904004081), the upper limit is the
target metal-donor distance plus 0.75 Å. If the exact residue-specific reference
is absent, the largest target for the same metal and donor element is used only
for sphere membership. A pair with no such target is discarded and reported as
`missing_first_sphere_reference`; DPI never expands the chemical cutoff. Atoms
belonging to the metal's own cofactor residue remain excluded, so these rows
describe external first-sphere contacts rather than a complete cofactor
coordination number.

Alchemy reports first-sphere contacts to atoms explicitly present in the
analyzed model and contacts generated by crystallographic symmetry, strict NCS,
or a combination of the two. Image-inclusive geometry is the primary result,
while explicit-only counts and geometry are retained separately. Generated
rows independently record whether an NCS transform and a crystallographic
operation contributed. They also record the strict-NCS operation identifier,
Gemmi image index, symmetry code, and unit-cell translation. Near-coincident
images of the same deposited atom within Gemmi's 0.8 Å special-position cutoff
are collapsed, while the stricter 0.001 Å tolerance remains reserved for
conflicting duplicate coordinate records.

Where a literature reference is available, Alchemy calculates:

```text
z = (d_observed - mu) / sqrt(DPI^2 + sigma_lit^2)
```

Following the method used in the in-preparation Alchemy manuscript,
reference-covered contacts with `|Zbond| >= 6` are geometry outliers.
First-sphere contacts admitted by a same-element fallback, or without complete
DPI inputs, are still emitted with their measured geometry and NaN derived
values. The `geometry_outlier` and `geometry_consistent` columns are nullable
booleans: a blank value means that geometry was not assessed, not that the
contact passed or failed the cutoff.

The DPI is calculated from PDB-REDO reflection and R-free metadata, the
asymmetric-unit volume, and `Ni`, the sum of occupancies for all non-hydrogen and
non-deuterium atoms in the complete first-model asymmetric unit. Alternate
positions contribute separately to this global sum. If non-given strict-NCS
operations generate copies that are not explicitly deposited, each copy is
included in `Ni`; NCS operations marked as already given are not counted again.
The deposited count, strict-NCS multiplier, and resulting complete count are all
reported. A missing, non-finite, negative, or greater-than-one occupancy makes
DPI unavailable rather than being silently repaired; contact distances that do
not require DPI are retained. Zero occupancy is valid for `Ni` but is not
accepted as evidence for a candidate contact.
For PDB input, raw occupancy records are matched to Gemmi atoms by chain,
residue number and insertion code, residue and atom names, alternate location,
and atom serial rather than parser traversal order.

## Important options

| Option | Purpose |
| --- | --- |
| `--id <pdbid>` | Process one PDB ID. |
| `--id-file <path>` | Process IDs from a comma-, whitespace-, or newline-separated file. |
| `--pdb-file`, `--cif-file`, `--mtz-file` | Process manually supplied structure data. |
| `--pdb-redo-root <path>` | Set the local mirror root. |
| `--pdb-redo-cache <path>` | Set the cache for downloaded entries. |
| `--output-dir <path>` | Set the result directory. |
| `--workers <n>` | Set multiprocessing parallelism; must be at least 1. |
| `--max-pdbs <n>` | Limit a run for testing. |
| `--resume` | Skip `ok` and terminal `partial` outcomes; retry `skip`, `error`, and retryable `partial` outcomes without duplicating their previous rows. |
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

## Reference data

- `src/data/metallocofactors_id.txt` — committed fallback list of
  metal-containing Chemical Component Dictionary IDs.
- `src/data/metallocofactors_id.meta.json` — generation metadata for the
  committed cofactor list.
- `src/data/metal_distances_info.txt` — reference metal-ligand distances and
  standard deviations.

## Operational notes

- Per-entry maps and logs are written to uniquely created working directories
  and removed after their rows are extracted unless `--keep-intermediates` is
  supplied. Cleanup never targets a pre-existing `<output-dir>/<pdb-id>`
  directory.
- Output CSV handles are flushed after each processed entry so interrupted batch
  runs retain completed results.
- Statistics and bond CSV files retain their column headers when a completed run
  finds no metals or no first-sphere contacts.
- Resume retries are staged separately. Existing rows are replaced only after a
  retry produces a terminal result and the retry batch completes; failed or
  interrupted retries leave the previous rows intact. `--resume --no-bonds`
  preserves existing bond rows.
- A failure in bond analysis does not discard real-space-statistics rows already
  calculated for that entry; the manifest records the entry as `partial` with
  the bond-stage error.
- The command exits nonzero when any entry ends as `error`, `skip`, or a
  retryable `partial`. Completed `ok` and terminal `partial` results exit
  successfully.
- `partial` describes usable but incomplete scientific output; it does not by
  itself mean that rerunning can repair the entry. Deterministic limitations,
  such as invalid deposited occupancy or unavailable symmetry metadata, are
  recorded as `partial` with `retryable=false`. Transient processing failures
  are recorded with `retryable=true`. `--resume` uses that field so terminal
  entries do not run forever.
- The Gemmi migration expands all three CSV schemas. `--resume` refuses to mix
  new rows with incompatible pre-migration headers; use a new `--output-dir`
  for the first Gemmi-based run.
