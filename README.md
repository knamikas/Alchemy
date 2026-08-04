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
# Smoke test: 9myr has two Cys3-His zinc sites
python src/main.py --id 9myr \
    --ccp4-setup <path/to/ccp4.setup-sh>

# Additional examples: 6nlr is a multi-element stress case; 9nxl is metal-free
python src/main.py --id 6nlr --ccp4-setup <…>
python src/main.py --id 9nxl --ccp4-setup <…>

# Run IDs from a comma-, whitespace-, or newline-separated file
python src/main.py \
    --id-file path/to/pdb_ids.txt --ccp4-setup <…>
```

Results are written to (the column lists and row builders live in
`src/bond_schema.py`):

- `output/metal_stats_all.csv` — one row per selected metal site, with
  real-space statistics, DPI and validation provenance, and explicit-only
  versus image-inclusive contact summaries. A cofactor containing multiple
  selected metal sites repeats its residue-level EDSTATS values once per site.
  These rows share a `density_observation_id`; `density_scope`,
  `density_shared_site_count`, and `density_is_shared` make the repetition
  explicit. Statistical analyses of density measurements should deduplicate on
  `density_observation_id` rather than counting each metal-site row as an
  independent density observation.
  Diagnostic rows for cofactors that cannot be matched to a coordinate site may
  have blank site-specific fields; `coordinate_mapping_status` and
  `selected_metal_site_status` distinguish failed joins from cofactors without
  a selected metal. These permanent limitations are also reported as terminal
  `partial` manifest outcomes and are not counted as metal sites.
- `output/metal_bonds_all.csv` — one row per inferred or declared contact,
  including distance, reference-based z-score, conformer selection, and
  separate crystallographic-symmetry and strict-NCS provenance.
- `output/metal_candidates_all.csv` — one row per donor-like atom found by the
  broad 4 Å search or supplied by a source `_struct_conn`/`LINK` declaration.
  Each row records whether it is first-sphere eligible, outside the applicable
  cutoff, or missing an assignment reference, together with connection, cutoff,
  and reference provenance. These rows are retained for audit and are not all
  treated as bonds.
- `output/confidence_inputs_all.csv` — compact site-level confidence evidence
  streamed only during an uncapped full-database run. On successful completion
  it is finalized into `confidence_scores_all.csv` and
  `output/confidence_reference/`. Later small runs write confidence scores
  directly when a compatible frozen database reference is installed.
- `output/manifest.csv` — per-entry status and reason, runtime, input
  provenance, and relevant software and analysis-policy versions.
  `reference_data_id` identifies the bundled catalog and distance table the row
  was measured against; rows are comparable only if it matches, and the run log
  records the two file checksums it is composed from. Adding this column means
  a manifest written by an earlier build cannot be resumed into — start a new
  `--output-dir`. Its
  `n_metals` value counts distinct selected coordinate-model sites, not
  diagnostic or repeated EDSTATS rows; `n_bonds` and `n_candidates` distinguish
  assigned-output size from candidate-evidence size.
- `output/logs/alchemy_run_YYYYMMDD.log` — one detailed, immutable log for
  each invocation, or `--log-dir` when one is given. Additional runs on the
  same UTC date receive a numeric suffix.
  The log records the command, configuration, software and system provenance,
  worker limits, output and confidence summaries, grouped reasons and warnings,
  slowest entries, every per-entry outcome, and timings for input preparation,
  `mtzfix`, both FFT calculations, model-envelope cropping, `edstats`,
  statistics extraction, bond analysis, and cleanup. It also records the
  full-map and EDSTATS-map sizes for each density-scored entry. Resume runs
  create another log rather than replacing the original run record.

## Dependencies

- **CCP4** with `mtzfix`, `fft`, `mapmask`, and `edstats` on `PATH`. Pass
  `--ccp4-setup <path/to/ccp4.setup-sh>` or configure the path once with
  `--configure-ccp4`.
- **Python:** 3.11 or newer.
- **Python packages:** `gemmi>=0.7.0` and `numpy>=1.17`. Both are required:
  `gemmi` does not install `numpy`. The authoritative dependency list and
  version constraints are in `pyproject.toml`.

Install these packages in the Python environment used to run Alchemy, using
your preferred system or environment package manager. If you use pip, run:

```bash
python -m pip install .
```

The pip command reads the requirements from `pyproject.toml`, installs them and
Alchemy's distribution metadata, but deliberately installs no importable
Alchemy modules. Alchemy is run from a clone of this repository, not imported
as a library. Pip and conda are optional; a system-wide Python installation is
also suitable as long as it provides the required packages.

## Documentation

The README is the overview and the quick start. Everything else lives in
[`docs/`](docs/):

- [docs/usage.md](docs/usage.md) — input modes and the options that matter.
- [docs/method.md](docs/method.md) — what each stage computes, which contacts
  become bonds, which donors the reference table covers, and what the
  confidence score claims.
- [docs/operations.md](docs/operations.md) — batch runs, resuming, retries, and
  reading the outputs.
- [docs/maintenance.md](docs/maintenance.md) — the bundled cofactor catalog and
  distance table: what they are, how to rebuild them, and how a change is
  recorded.
- [tests/README.md](tests/README.md) — running the suite.
