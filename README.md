# Alchemy

Alchemy evaluates how well experimental electron density supports metal atoms
in protein crystal structures. For each PDB entry, it calculates 2mFo-DFc and
mFo-DFc maps, extracts per-residue real-space statistics, identifies metal ions
and metal-containing cofactors, and compares first-sphere coordination geometry
with literature reference distances.

Generated contacts can arise from crystallographic symmetry, strict
noncrystallographic symmetry (NCS), or both. They describe the deposited
structure model. They don't establish by themselves that a metal or ligand is
biologically functional.

Run Alchemy from a source checkout with `./alchemy`. The command can read a
local PDB-REDO mirror, download explicitly requested entries into a local
cache, or process coordinate and MTZ files that you provide.

## Requirements

- CCP4 with `mtzfix`, `fft`, `mapmask`, and `edstats`. If these programs aren't
  already on `PATH`, pass a setup script with `--ccp4-setup` or save its path
  with `--configure-ccp4`. On Windows, use the CCP4 `ccp4.setup.bat` or
  `ccp4.setup.cmd` launcher.
- Python 3.11 or later.
- The Python dependencies declared in `pyproject.toml`, including
  `gemmi>=0.7.0`, `numpy>=1.17`, and `typing_extensions>=4.6`.

Install the Python dependencies in the environment that you use to run
Alchemy:

```bash
python -m pip install .
```

This command installs the dependencies and Alchemy's distribution metadata. It
doesn't install importable Alchemy modules. Run Alchemy from the checkout
instead of importing it as a library.

## Run your first entry

The following walkthrough processes `9myr`, which contains two Cys3-His zinc
sites.

1. Confirm that Python and the launcher are available:

   ```bash
   python --version
   ./alchemy --help
   ```

2. Run the entry. Replace `CCP4_SETUP` with the absolute path to your CCP4
   setup script:

   ```bash
   ./alchemy --id 9myr --ccp4-setup CCP4_SETUP
   ```

3. Check the run outcome in `output/manifest.csv`. A completed analysis has
   `status=ok`, `n_metals=2`, and `no_metals=false` for `9myr`.

4. Open `output/metal_sites_all.csv` for the two site rows. Join contacts from
   `output/metal_bonds_all.csv` by `metal_site_id`. For guidance on verdicts,
   missing values, and joins, see [Interpret a result](docs/output-schema.md#interpret-a-result).

The run also writes an immutable report under `output/logs/`. If the command
doesn't complete, start with the entry's `reason_codes` and `status_detail` in
the manifest, then see the
[manifest reason-code reference](docs/operations.md#manifest-reason-codes).

## Run other input modes

Process a multi-element example or a metal-free control:

```bash
./alchemy --id 6nlr --ccp4-setup CCP4_SETUP
./alchemy --id 9nxl --ccp4-setup CCP4_SETUP
```

Process IDs from a comma-, whitespace-, or newline-separated file:

```bash
./alchemy --id-file PDB_ID_FILE --ccp4-setup CCP4_SETUP
```

For local mirrors and manual coordinate and MTZ inputs, see
[Choose an input mode](docs/usage.md#input-modes).

## Outputs

Alchemy writes results to `output/` by default:

- `manifest.csv` records one outcome per entry, including status, row counts,
  input provenance, software versions, and analysis-policy identities.
- `metal_sites_all.csv` records selected metal sites and their density,
  precision, validation, and contact summaries.
- `metal_bonds_all.csv` records assigned inferred or declared contacts.
- `metal_contact_candidates_all.csv` records the broader candidate evidence;
  not every candidate is a bond.
- `density_context_all.csv` records the non-target residue-level density
  distribution for each processed entry.
- `crystallization_conditions_all.csv` and
  `crystallization_summary_all.csv` record deposited experimental context.
- `confidence_scores_all.csv` records site classifications. An uncapped
  database run also writes `confidence_inputs_all.csv` and
  `confidence_reference/`.
- `review_queue_all.csv` contains only `REVIEW` and `SUSPECT` confidence rows
  with their crystallization context. That context doesn't affect scoring.
- `logs/alchemy_run_YYYYMMDD.log` and its matching `_entries.csv` file record
  the run configuration, summary, per-entry outcomes, timing, and resource
  diagnostics.

For row grain, units, null handling, and every output column, see the
[output schema](docs/output-schema.md). For batch behavior, retries, and
failure handling, see the [operations guide](docs/operations.md).

## Documentation

- [Usage guide](docs/usage.md): Choose an input mode and configure the command.
- [Method reference](docs/method.md): Understand each analysis stage, contact
  assignment, reference coverage, and confidence classification.
- [Output schema](docs/output-schema.md): Interpret results and find the data
  dictionary for each output.
- [Operations guide](docs/operations.md): Run batches, resume work, and diagnose
  failures.
- [Reference-data maintenance](docs/maintenance.md): Rebuild and verify the
  bundled cofactor catalog and distance table.
- [Test guide](tests/README.md): Run the offline and CCP4-backed test lanes.
