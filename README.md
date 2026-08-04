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

Results are written to:

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
  provenance, and relevant software and analysis-policy versions. Its
  `n_metals` value counts distinct selected coordinate-model sites, not
  diagnostic or repeated EDSTATS rows; `n_bonds` and `n_candidates` distinguish
  assigned-output size from candidate-evidence size.
- `output/alchemy_run_YYYYMMDD.log` — one detailed, immutable log for each
  invocation. Additional runs on the same UTC date receive a numeric suffix.
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

Without `--data-json` there is no reflection count, so DPI and every value
derived from it are unavailable. Bond geometry is still measured and emitted,
and the omission is reported as `missing_dpi_metadata_source` rather than as a
calculation failure.

## Pipeline flow

### 1. Input preparation — `src/inputs.py` and `src/coordinate_conversion.py`

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

If a model has more chains than the one-character PDB namespace can represent,
Alchemy packs its residues into synthetic one-character chains with unique
four-column sequence numbers. ``REMARK 950 ALCHEMY RESIDUE`` records preserve
the original component, chain, sequence number, insertion code, source traversal
indices, and polymer-terminal position. EDSTATS and Gemmi analyze the same
packed coordinates, then statistics, contacts, declarations, and CSV identifiers
are mapped back to the source mmCIF identities. Conversion validates the atom
and residue membership before analysis, so an oversized structure cannot
silently lose sites at the legacy-PDB boundary.

The overall diffraction resolution comes from PDB-REDO `data.json` when
available, with an MTZ fallback through gemmi. EDSTATS instead receives the
common finite resolution range of `FWT`, `PHWT`, `DELFWT`, and `PHDELWT`, which
are the columns used to calculate its two maps.

### 2. Maps and real-space statistics — `src/density_analysis.py`

`run_density_analysis()` runs:

1. CCP4 `mtzfix` to check and, when needed, correct the Fourier map
   coefficients, including their centric and acentric consistency. If the input
   passes, MTZFIX intentionally writes no replacement and the original MTZ is
   used. If its consistency re-test fails for an entry whose PDB-REDO
   `data.json` explicitly has `properties.ISTWIN=true`, Alchemy can instead
   normalize recognizable Refmac composite coefficients on a temporary MTZ.
   This guarded path requires Refmac provenance, the complete expected column
   schema, and reflection-by-reflection agreement with Refmac's raw coefficient
   identity; its output is independently checked against the convention
   EDSTATS consumes. The source MTZ is never modified. Successful use is
   recorded as the `twin_refmac_coefficients_normalized` warning.
2. CCP4 `fft` with `FWT/PHWT` from that validated MTZ to produce a 2mFo-DFc map.
3. By default, CCP4 `mapmask` limits that full FFT map to the complete deposited
   coordinate-model envelope plus a 10 Angstrom border. The same crop is then
   applied to the mFo-DFc difference map calculated from `DELFWT/PHDELWT`.
   This retains every modeled atom while avoiding EDSTATS work over distant
   empty unit-cell volume. If the envelope would not be smaller, Alchemy uses
   the original full maps automatically. It also uses full maps when a
   translated model would produce a grid extent known to be incompatible with
   EDSTATS coordinate lookup.
4. CCP4 `edstats` uses the resulting map pair to produce per-residue real-space
   statistics and an RSZD coordinate file.

Use `--density-map-scope full` to select the legacy behavior in which EDSTATS
receives both complete unit-cell maps without model-envelope cropping.

The MTZ input must contain `FWT`, `PHWT`, `DELFWT`, and `PHDELWT` columns.

### 3. Metal and cofactor identification — `src/metal_identification.py`

`extract_metal_statistics()` parses the edstats table. Alchemy reads the
deposited PDB element field directly; blank or invalid fields are marked unknown
rather than inferred from atom names. During mmCIF conversion, the explicit
`_atom_site.type_symbol` is written into that PDB field. Metal-containing
cofactors are matched against the Chemical Component Dictionary list maintained
by `tools/build_metallocofactor_catalog.py`. A structure with unknown elements
does not receive a DPI because its non-hydrogen atom count is indeterminate.
EDSTATS' missing-chain markers are normalized before coordinate matching. When
EDSTATS omits its empty trailing chain field for a blank-chain residue, Alchemy
restores that field before validating the standard 42-column schema. Other row
width mismatches still fail. Decimal and hybrid-36 PDB residue numbers are
decoded to the same canonical integer representation used by Gemmi before raw
PDB atoms and EDSTATS rows are joined. The table must contain finite numeric
statistics or the documented `n/a` marker and a row for every selected metal or
cofactor residue. Empty, malformed, incomplete, or wrong-model output fails the
entry instead of being written to the aggregate CSV.

### 4. Bond-distance analysis — `src/bond_analysis.py`

Gemmi parses the same coordinate representation supplied to EDSTATS. For a
multi-model structure, Alchemy analyzes the first model only for metal
identification, DPI, and contact searching; model count and the selected-model
policy are recorded with the results. Before map statistics are calculated,
Alchemy writes a wrapper-free, first-model-only PDB and supplies that exact
coordinate file to both EDSTATS and Gemmi. Removing explicit MODEL/ENDMDL
records prevents EDSTATS from adding a synthetic separator residue. Atoms and
density statistics from different models are never combined.

Alternative conformations are selected coherently per residue. Blank-altloc
atoms are shared, while the named conformer with the highest mean valid atomic
occupancy is selected (ties are resolved by altloc label). This avoids creating
an artificial residue by choosing A/B alternatives independently for each atom.
Every neighboring residue is considered independently, and the selected and
available alternatives are recorded.

`run_bond_analysis()` uses a 4 Å search only to discover broad
positive-occupancy N/O/S candidates around a configured metal, outside the
metal's own residue, in a recognized amino acid or water. Discovery does not
assign a candidate as a bond. A separate eligibility stage identifies likely
first-coordination-sphere candidates for the current bond output, and an
atom-level chemical rule determines which candidates Alchemy may infer as
bonds. Following [Harding's
coordination-group
definition](https://doi.org/10.1107/S0907444904004081), the upper limit is the
target metal-donor distance plus 0.75 Å. If the exact residue-specific reference
is absent, the largest target for the same metal and donor element is used only
for sphere membership. A pair with no such target is retained as candidate
evidence but is not inferred as a contact; the entry reports
`missing_first_sphere_reference`. DPI never expands the chemical cutoff. Atoms
belonging to the metal's own cofactor residue remain excluded, so these rows
describe external coordination rather than a complete cofactor coordination
number.

The geometry-inference donor table covers all 20 standard amino acids. Backbone
carbonyl `O` is allowed for every residue. Typical side-chain donors are ASN
OD1; ASP OD1/OD2; CYS SG; GLN OE1; GLU OE1/OE2; HIS ND1/NE2; LYS NZ; MET SD;
SER OG; THR OG1; and TYR OH. Water oxygen is allowed. Polymer N-terminal `N`
and C-terminal `OXT`/`OT1`/`OT2` are allowed only when Gemmi identifies the
residue at the corresponding polymer boundary. Other proximal N/O/S atoms are
retained in `metal_candidates_all.csv`, marked
`inferred_donor_allowed=false`, and cannot become geometry-inferred bonds.
This includes internal peptide N, ASN/GLN amide N, TRP pyrrole N, and ARG
guanidinium N. A declaration can still establish such an atom as a declared
bond; `donor_rule_override=declared_connection` makes that exception explicit.

### Reference coverage of the donor table

Inferring a contact and scoring one are separate questions. The donor table
above governs inference; scoring additionally requires a literature reference
distance in `src/data/metal_distances_info.txt`, and that reference does not
cover every donor Alchemy will infer:

- **ASN, GLN, LYS and MET have no reference entry.** Harding (2006) tabulates
  water `O`, ASP/GLU carboxylate `O`, backbone carbonyl `O`, HIS `N` and CYS
  `S` only. Contacts to these four side chains are therefore discovered,
  reported and measured, but never receive a Zbond: they carry the
  same-element fallback and NaN derived values. This is a limitation of the
  bundled reference data, not of the geometry.
- **SER, THR and TYR values are approximations** derived from statements in
  Harding (2006) rather than from its tables, so their `sigma_lit` is not an
  empirical spread. Treat z-scores for these three donors as indicative.
- **Nucleic acids, modified residues and other ligands have no reference at
  all.** A metal coordinated by, say, a DNA phosphate oxygen is real
  coordination, but no bundled distance can assess it.

A declared contact to a donor class with no reference is retained in
`metal_candidates_all.csv` with its measured distance and full connection
provenance, and the entry records
`declared_donor_outside_supported_classes`. It is deliberately **not** promoted
to a bond row: doing so would raise the site's coordination count and enlarge
the confidence geometry-coverage (`QG`) denominator on the strength of a
contact that nothing in the reference data can evaluate. The distinction that
matters for a consumer is that "no reference for this donor class" and "this
metal has no coordination" are now different, visibly, in the output.

Alchemy separately parses `_struct_conn` records from the authoritative source
mmCIF and `LINK` records from a source PDB. A declared metal–donor contact is
merged with a matching proximity candidate when present and added independently
when it lies outside 4 Å. Declared contacts remain bonds even when they fail the
distance-based first-sphere eligibility rule; their measured distance and Zbond
are still calculated whenever the required literature reference and DPI exist.
The output labels these as model declarations rather than treating them as
geometric proof of coordination.

Alchemy reports assigned contacts to atoms explicitly present in the
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

A bond length depends on two atomic positions, but a single DPI enters the
denominator rather than the `sqrt(2) * DPI` an independent-error treatment of
both atoms would give. This assumes the metal is well enough ordered that its
positional uncertainty is negligible beside that of the lighter donor atom, so
the DPI term stands for the donor alone. Using two atoms' worth of uncertainty
would divide every z-score by up to `sqrt(2)` and reclassify borderline geometry
outliers.

Reference-covered contacts with `|Zbond| >= 6` are geometry outliers.
First-sphere contacts admitted by a same-element fallback, or without complete
DPI inputs, are still emitted with their measured geometry and NaN derived
values. The `geometry_outlier` and `geometry_consistent` columns are nullable
booleans: a blank value means that geometry was not assessed, not that the
contact passed or failed the cutoff.

Assigned contacts are also grouped by metal and donor-residue image, with no
upper limit on the number of contacts in a group. Because backbone atoms carry
the same residue identity in the coordinate model, this grouping automatically
includes backbone–side-chain combinations without a separate backbone rule.
Every bond retains its measured distance, Zbond, and ordinary geometry flag.
Groups with two or more donors set `multi_donor_detected=true` and record their
full contact count. Every assessable member contributes to aggregate scoring
normally. If any member is an outlier, every member of the group records
`multi_donor_geometry_status=suspect` and
`multi_donor_contains_suspect_bond=true`; the particular unusual bonds retain
`geometry_outlier=true`. This makes possible multidentate context conspicuous
without weakening or excluding the result. A group with unavailable Zbond
values and no detected outlier is labeled `indeterminate`; only the individual
unassessable bonds are omitted from scoring.

`context_warning` is a binary interpretive flag and does not alter the numerical
confidence calculation. Machine-readable `context_warning_reasons` explain the
trigger. Bond rows are flagged for declared non-typical donors and for membership
in a multi-donor group containing a suspect bond. Candidate rows additionally
flag every proximal atom outside the typical donor table. At site level, the
flag summarizes coordination-relevant cases: a non-typical atom satisfying the
first-sphere distance rule, a declared donor-rule override, or a suspect
multi-donor group. A distant non-typical atom found only by the broad 4 Å search
does not by itself place the complete metal site under warning.

### Database-referenced confidence scoring

Confidence inputs are collected as part of an uncapped full-database run: no
`--id`, `--id-file`, manual coordinate arguments, or `--max-pdbs`. As each
worker result reaches the driver, Alchemy combines its already in-memory density
and assigned-bond evidence and streams one compact row per selected metal site
to `confidence_inputs_all.csv`. It never rereads the complete statistics and
bond tables to reconstruct those inputs.

The compact row records the magnitude of metal-site `ZDm` as `rszd_magnitude`,
the largest absolute Zbond among assigned contacts, the responsible contact,
and geometry coverage. Geometry coverage is the number of assigned contacts
with an exact reference distance divided by the total number of assigned
contacts, matching the manuscript definition of `QG`. The finite-Zbond count is
recorded separately so missing DPI cannot be mistaken for usable geometry
evidence. Rejected broad-search candidates do not enter the denominator.
Missing density, absent bonds, partial coverage, diagnostic EDSTATS rows, and
shared-cofactor density provenance remain explicit. The streamed file retains
exactly one row per manifest-counted selected metal; a site with no recoverable
density or bond identity is represented by an unresolved, unscorable placeholder
rather than silently disappearing from the cohort denominator.

Only after the database run completes without operationally incomplete entries
does Alchemy finalize confidence. It scans the compact input—not the raw
analysis outputs—to write `confidence_scores_all.csv` and a reusable
`confidence_reference/` directory containing policy metadata and the empirical
score distribution. An interrupted run retains its compact inputs for
`--resume` but does not publish a completed reference.

The provisional June 2026 fixed score transforms absolute RSZD and maximum
absolute Zbond through piecewise-linear severity anchors and calculates:

```text
confidence = 100 * (1 - 0.50*SR - 0.35*QG*SG
                       - 0.15*QG*sqrt(SR*SG))
```

Alchemy also reports an average-rank percentile relative to the scorable full
database cohort, its size, and a policy identifier. The fixed-formula score and
database percentile are visibly distinct. A deterministic
`confidence_reference_id` identifies the exact frozen policy and score
distribution and prevents resumed output from mixing database snapshots.
`context_warning` is carried into the result as an interpretive annotation and
does not subtract points.

**No confidence reference is distributed with Alchemy, and the score is not
finalized.** A fresh clone therefore produces every other output but reports
that confidence scoring is not enabled; this is expected behaviour, not a
misconfiguration. Enabling it requires either completing an uncapped
full-database run, which writes a reference of your own, or supplying one
explicitly with `--confidence-reference-dir`. Treat the score described in this
section as provisional and subject to change until it is released.

For later single-entry, ID-file, manual, or capped runs, Alchemy first looks for
the reference produced under the current output directory's
`confidence_reference/`, then in the repository's `confidence_reference/`.
`--confidence-reference-dir` selects an explicit copy instead. Alchemy loads
that reference once, derives each new site's compact inputs while its normal
result is still in memory, and writes `confidence_scores_all.csv` directly. These runs
are compared with the frozen database and never generate percentiles from their
own small cohort. If no compatible reference is installed, the ordinary
Alchemy outputs are still produced and confidence scoring is explicitly
disabled.

`src/confidence_score.py` retains `finalize` and `score` subcommands for recovery
and reproducibility using already compact confidence-input CSVs; neither command
reconstructs inputs by rescanning `metal_stats_all.csv` or
`metal_bonds_all.csv`.

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
| `--density-map-scope {model-envelope,full}` | Set the map extent passed to EDSTATS. The default model envelope retains every coordinate plus a 10 Angstrom border; `full` selects the legacy complete-map path. |
| `--ccp4-timeout <s>` | Per-program wall-clock budget for each CCP4 step (`mtzfix`, `fft`, `mapmask`, `edstats`), in seconds; default 900. The budget applies to each program separately, not to the entry as a whole. A program that exceeds it is killed and the entry becomes a retryable `partial` with reason `ccp4_tool_timeout`; its partial log is copied to `<output-dir>/ccp4_timeout_logs/`. Raise it for exceptionally large structures. |
| `--workers <n>` | Override multiprocessing parallelism; must be at least 1. By default, Alchemy uses the lower CPU or available-memory limit, budgeting 1.25 GiB per worker. |
| `--max-pdbs <n>` | Limit a run for testing. |
| `--resume` | Skip `ok` and terminal `partial` outcomes; retry `skip`, `error`, and retryable `partial` outcomes without duplicating their previous rows. |
| `--retry-partials` | With `--resume`, retry all `partial` entries recorded in the manifest. Successful `ok` entries remain skipped; `--id` or `--id-file` may restrict the retry set. |
| `--no-bonds` | Skip bond-distance analysis. A fresh run removes previous `metal_bonds_all.csv` and `metal_candidates_all.csv` files in the output directory. |
| `-v`, `--verbose` | Increase diagnostic detail. The default reports the run narrative; `-v` adds per-entry and per-CCP4-program records from inside the worker processes. |
| `--quiet` | Report warnings and errors only. |
| `--log-file <path>` | Also write full debug-level diagnostics to a file, whatever the console verbosity. Independent of the per-run report in `alchemy_run_*.log`. |
| `--keep-intermediates` | Retain per-entry maps and logs. |
| `--ccp4-setup <path>` | Source and verify a CCP4 setup script for this run. |
| `--configure-ccp4 <path>` | Save a CCP4 setup script path for later runs. |

## Cofactor reference maintenance

Normal analysis always loads the fixed catalog bundled in `src/data`. It never
checks the catalog's age, accesses the network, selects a user cache, or
rebuilds the catalog.

Catalog updates are an explicit developer maintenance operation:

```console
python tools/build_metallocofactor_catalog.py
```

The isolated builder downloads the wwPDB Chemical Component Dictionary and
replaces the bundled catalog and its metadata only after a successful build,
using atomic replacement for each file. A local CCD snapshot can be supplied
with `--ccd`. Use `--status` to report the generation time, entry count,
checksum, and integrity of the currently bundled catalog
without downloading anything. Catalog changes should be reviewed and committed
as part of a software release before an analysis run.

Cluster and heme classes are derived from CCD atom connectivity, not formula
stoichiometry. Clusters require a bridging sulfur or selenium, while hemes
require an Fe-bound porphyrinoid core. Run a full rebuild whenever the component
list or classification rules change.

## Reference data

- `src/data/metallocofactors_id.txt` — fixed bundled list of metal-containing
  Chemical Component Dictionary IDs used by every analysis run.
- `src/data/metallocofactors_id.meta.json` — generation metadata for the
  committed cofactor list.
- `src/data/metal_distances_info.txt` — reference metal-ligand distances and
  standard deviations, keyed by donor residue, donor element and metal. Values
  are from Harding (2006), [Acta Cryst. D62,
  678-682](https://doi.org/10.1107/S0907444906014594), except NI, which is from
  Zheng et al. (2008), and SER/THR/TYR, which are approximated from statements
  in Harding (2006) rather than tabulated. See "Reference coverage of the donor
  table" for which donors this file does and does not cover.

## Operational notes

- Alchemy separates three kinds of output. The progress line and the final
  result summary go to **stdout**, so a redirected stdout remains a usable
  record of what was produced. Diagnostics go to **stderr** as log records,
  controlled by `-v`/`--quiet`/`--log-file`. The per-run report in
  `output/alchemy_run_*.log` is written separately and is unaffected by
  verbosity: it is a structured artifact, not a transcript. Worker processes
  emit records through a queue that the driver re-emits, so per-entry
  diagnostics from parallel workers never interleave mid-line.
- Interactive runs redraw a single progress line after every completed
  structure. While waiting on a slow structure, the elapsed time refreshes
  approximately once per second. Redirected output is limited to one progress
  line every 30 seconds to avoid producing oversized logs.
- Per-entry maps and logs are written to uniquely created working directories
  and removed after their rows are extracted unless `--keep-intermediates` is
  supplied. Cleanup never targets a pre-existing `<output-dir>/<pdb-id>`
  directory.
- Model-envelope mode still calculates each complete FFT map before cropping,
  so map values come from the same Fourier calculation as legacy full-map mode.
  Full temporary maps are deleted as soon as they are no longer needed unless
  `--keep-intermediates` is supplied.
- Output CSV handles are flushed after each processed entry so interrupted batch
  runs retain completed results.
- In the manifest, blank `n_bonds` and `n_candidates` values mean bond analysis
  was not run; `0` means it ran successfully but found no rows of that type.
  Resume uses this distinction to add bond-stage results after an earlier
  `--no-bonds` run. Missing bond or candidate CSVs make bond-enabled results
  incomplete.
- Statistics, bond, and candidate CSV files retain their column headers when a
  completed run finds no metals, contacts, or proximal candidates.
- The statistics header is a fixed schema rather than a copy of whatever EDSTATS
  emitted, because `extract_metal_statistics` already requires the EDSTATS
  residue table to match the standard column set and order.
- Resume retries are staged separately. Existing rows are replaced only after a
  retry produces a terminal result and the retry batch completes; failed or
  interrupted retries leave the previous rows intact. `--resume --no-bonds`
  preserves existing bond and candidate rows and their manifest counts. Entries
  originating from a bond-disabled run retain blank `n_bonds` and
  `n_candidates`, so a later bond-enabled resume will process them.
- `--resume --retry-partials` reprocesses non-retryable `partial` entries from
  the manifest after a processing improvement while continuing to protect
  `ok` entries. Optional `--id` or `--id-file` selectors restrict that set. Errors,
  skips, and retryable partials already follow ordinary resume behavior. The
  same staged replacement rules apply, so an interrupted or retryably failed
  attempt does not discard the previous terminal result. When a frozen
  reference scores a targeted resume inside an existing database output,
  `confidence_scores_all.csv` and `confidence_inputs_all.csv` are replaced
  together so their per-entry evidence cannot diverge.
- A fresh `--no-bonds` run removes pre-existing `metal_bonds_all.csv` and
  `metal_candidates_all.csv` files before replacing the manifest and
  statistics, so old bond-stage rows cannot be mistaken for current output.
- A failure in bond analysis does not discard real-space-statistics rows already
  calculated for that entry; the manifest records the entry as `partial` with
  the bond-stage error.
- If `mtzfix` cannot make an MTZ's Fourier coefficients pass its consistency
  re-test, Alchemy does not use the rejected maps or retry indefinitely. An
  explicitly twin-refined PDB-REDO entry is eligible for the guarded Refmac
  coefficient normalization described above. Every other entry, and any twin
  entry that fails a provenance, schema, or coefficient-identity check, is a
  terminal `partial` with `mtzfix_validation_failure`; coordinate-based bond
  analysis still runs, while its metal sites remain explicitly unscorable by
  confidence because RSZD is unavailable.
- After canonical model and conformer selection, structures with no recognized
  metal atoms finish with `n_metals=0` without running `mtzfix`, either FFT, or
  `edstats`. These stages cannot produce metal-site output for such entries.
  Progress and completion summaries report these successful negative results as
  `no_metals`; this is an informational subset of `ok`, whereas `skip` remains
  reserved for entries that could not be processed operationally.
- The command exits nonzero when any entry ends as `error`, `skip`, or a
  retryable `partial`. Completed `ok` and terminal `partial` results exit
  successfully.
- A CCP4 program that exceeds `--ccp4-timeout` is killed and its entry recorded
  as a retryable `partial` with reason `ccp4_tool_timeout`, distinct from a
  program that failed with an error. A killed program reported nothing about the
  entry, so retrying it is meaningful. Its partial log is copied to
  `<output-dir>/ccp4_timeout_logs/<id>_<tool>_timeout.log` before the entry's
  scratch directory is removed; the maps beside it are not retained.
- `partial` describes usable but incomplete scientific output; it does not by
  itself mean that rerunning can repair the entry. Deterministic limitations,
  such as invalid deposited occupancy or unavailable symmetry metadata, are
  recorded as `partial` with `retryable=false`. Transient processing failures
  are recorded with `retryable=true`. `--resume` uses that field so terminal
  entries do not run forever.
- The candidate-output migration expands all four CSV schemas. `--resume`
  refuses to mix new rows with incompatible pre-migration headers; use a new `--output-dir`
  for the first run. All four headers are compared in full,
  including the EDSTATS block of `metal_stats_all.csv`, so appended rows cannot
  be silently misaligned by output from a different EDSTATS build.
