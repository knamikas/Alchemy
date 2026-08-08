# Alchemy Pipeline

Alchemy evaluates how well metal atoms in protein crystal structures are
supported by experimental electron density. For each PDB entry, it calculates
2mFo-DFc and mFo-DFc maps, extracts per-residue real-space statistics, identifies
metal ions and metal-containing cofactors, and measures first-sphere
metal-coordination geometry against literature reference distances. Generated
contacts can arise from crystallographic symmetry, strict noncrystallographic
symmetry (NCS), or both. They describe the deposited structure model and do not
by themselves establish that a metal or ligand is biologically functional.

`./alchemy` is the command-line entry point for a source checkout. It can read a
local PDB-REDO mirror, download explicitly requested entries into a local cache,
or process manually supplied coordinate and MTZ files. `src/main.py` remains the
maintained Python entry point used by the launcher.

## Quick start

```bash
# Smoke test: 9myr has two Cys3-His zinc sites
./alchemy --id 9myr \
    --ccp4-setup <path/to/ccp4.setup-sh>

# Additional examples: 6nlr is a multi-element stress case; 9nxl is metal-free
./alchemy --id 6nlr --ccp4-setup <…>
./alchemy --id 9nxl --ccp4-setup <…>

# Run IDs from a comma-, whitespace-, or newline-separated file
./alchemy \
    --id-file path/to/pdb_ids.txt --ccp4-setup <…>
```

Results are written to (the column lists and row builders live in
`src/coordination/schema.py`, `src/confidence_score.py`, and
`src/crystallization_conditions.py`):

- `output/metal_sites_all.csv` — one row per selected metal site, with
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
- `output/metal_contact_candidates_all.csv` — one row per donor-like atom found by the
  broad 4 Å search or supplied by a source `_struct_conn`/`LINK` declaration.
  Each row records whether it is first-sphere eligible, outside the applicable
  cutoff, or missing an assignment reference, together with connection, cutoff,
  and reference provenance. These rows are retained for audit and are not all
  treated as bonds.
- `output/crystallization_conditions_all.csv` — deposited crystallization
  records extracted once per PDB entry. Normal runs prefer original-PDB
  `_exptl_crystal_grow` records cached from the RCSB Data API and fall back to
  the PDB-REDO coordinate file; manual runs prefer their supplied coordinate
  file. Raw text is preserved alongside reported pH, temperature, method, and
  retrieval/revision provenance.
- `output/crystallization_summary_all.csv` — one row per processed PDB entry
  with data-availability status, pH and temperature ranges, detected metals,
  and contextual chemical flags. Blank flags mean the condition was
  unavailable; they do not mean experimental absence.
- `output/confidence_inputs_all.csv` — compact site-level confidence evidence
  streamed only during an uncapped full-database run. On successful completion
  it is finalized into `confidence_scores_all.csv` and
  `output/confidence_reference/`. Later small runs write confidence scores
  directly. PASS/REVIEW/SUSPECT classifications use the raw final thresholds;
  a compatible frozen database reference adds optional empirical ranking
  scores.
- `output/review_queue_all.csv` — only REVIEW and SUSPECT confidence rows,
  joined to their entry-level crystallization summary. This is a triage view:
  crystallization metadata never changes a component level, `alchemy_level`,
  or any empirical support score.

- `output/manifest.csv` — per-entry status, machine-readable reasons, a bounded
  `status_detail`, millisecond runtime, counts, input provenance, and relevant
  software and analysis-policy versions. Explicit `no_metals` and
  `metal_site_limit_exceeded` fields keep successful negative results and
  policy exclusions distinguishable from ordinary analyzed entries without
  parsing `reason_codes`. PDB-REDO inputs record the source-relative coordinate
  path plus the `VERSION` and `TIME` values from `data.json`; manual inputs
  retain the path supplied by the user.
  `reference_data_id` identifies the bundled catalog and distance table the row
  was measured against; rows are comparable only if it matches, and the run log
  records the two file checksums it is composed from. Adding this column means
  a manifest written by an earlier build cannot be resumed into — start a new
  `--output-dir`. Its
  `n_metals` value counts distinct selected coordinate-model sites, not
  diagnostic or repeated EDSTATS rows; `n_bonds` and `n_candidates` distinguish
  assigned-output size from candidate-evidence size.

Entries with more than 100 selected canonical metal sites are recorded in the
manifest with `metal_site_limit_exceeded=true` and excluded before CCP4
processing. This keeps exceptionally metal-dense, highly correlated assemblies
from dominating the standard database cohort; their detected `n_metals` count
remains available for audit.

- `output/logs/alchemy_run_YYYYMMDD.log` — one concise, immutable run report
  for each invocation, or `--log-dir` when one is given. It records the
  command, configuration, software and reference-data provenance, analysis
  policies, worker limits, output and confidence summaries, grouped exceptions,
  stage aggregates, and slowest entries.
- `output/logs/alchemy_run_YYYYMMDD_entries.csv` — the report's machine-readable
  companion, containing every per-entry outcome, reason, warning, runtime,
  stage timing, map size, and memory estimate. The two files receive the same
  numeric suffix when there are additional runs on a UTC date. Resume runs
  create a new pair rather than replacing the original record.

## Dependencies

- **CCP4** with `mtzfix`, `fft`, `mapmask`, and `edstats` on `PATH`. Pass
  `--ccp4-setup <path/to/ccp4.setup-sh>` or configure the path once with
  `--configure-ccp4`. On Windows, pass the CCP4 `ccp4.setup.bat` or
  `ccp4.setup.cmd` launcher instead.
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
- [docs/output-schema.md](docs/output-schema.md) — row grain, join identifiers,
  serialization conventions, units, and the scientific CSV data dictionary.
- [docs/operations.md](docs/operations.md) — batch runs, resuming, retries, and
  reading the outputs.
- [docs/maintenance.md](docs/maintenance.md) — the bundled cofactor catalog and
  distance table: what they are, how to rebuild them, and how a change is
  recorded.
- [tests/README.md](tests/README.md) — running the suite.
