# Architecture

Alchemy runs through one launcher, `./alchemy`. The Python files under `src/`
provide functions and shared data structures used by that run; they are not a
sequence of standalone commands. The driver coordinates the batch, worker
processes analyze individual entries, and the driver writes the combined
outputs.

This page maps execution and ownership; [architecture.png](architecture.png)
renders the same lanes as one picture. For scientific rules, see the
[method reference](method.md); for CSV fields, see the
[output schema](output-schema.md); for retries and resource controls, see
the [operations guide](operations.md).

## Execution and data flow

The driver prepares the batch and schedules entries across worker processes.
Each worker prepares inputs, loads the structure and crystallization context,
checks whether metals can be analyzed, then runs density, identification, and
coordination analysis in order. Entries without analyzable metals or above the
site limit return early; `--no-bonds` skips coordination analysis. The driver
collects results, prepares confidence scores, writes outputs, finalizes the
batch, builds the review queue, and writes the run report. Several entries can
run concurrently, while the stages within each entry run in order.

Inputs come from the mirror, downloaded cache, or manual files.
[rcsb_metadata_cache.py](../src/rcsb_metadata_cache.py) warms the original-PDB
metadata cache before workers start, and
[crystallization_conditions.py](../src/crystallization_conditions.py) reads it
for crystallization context. `reference_data.py` and
`src/data/` provide the cofactor catalog and distance table used in batch
preparation, metal identification, and coordination analysis. Identification
and coordination share the loaded structure context; confidence scoring uses a
frozen reference when applicable.

### Startup and scheduling

1. [The launcher](../alchemy) adds `src/` to the import path and calls
   [main.py](../src/main.py), which delegates to
   [cli.py](../src/cli.py). The CLI validates arguments into an immutable
   `RunConfig`, configures diagnostics, and creates the run report object.
2. [driver/pool.py](../src/driver/pool.py) orchestrates the batch. Its
   collaborators are [driver/layout.py](../src/driver/layout.py) for output
   paths, [driver/entries.py](../src/driver/entries.py) for entry selection,
   [driver/confidence.py](../src/driver/confidence.py) for the confidence
   plan, and [driver/report.py](../src/driver/report.py) for the batch
   summary and confidence finalization. The pool loads the bundled cofactor
   catalog and resolves the CCP4 environment through
   [driver/environment.py](../src/driver/environment.py), which also records
   the Alchemy, Gemmi, and CCP4 versions for provenance.
   [ccp4_setup.py](../src/ccp4_setup.py) locates the setup script for it:
   the `--ccp4-setup` option, the `CCP4_SETUP` environment variable, the path
   saved by `--configure-ccp4`, then common install locations.
   `--configure-ccp4` saves the setup path and exits before analysis. A
   `DriverError` from [driver/errors.py](../src/driver/errors.py) at any
   startup step ends the run with exit code 1.
3. The driver creates the output directory if needed, then acquires the
   output-directory lock before reading or writing any run output. It determines the confidence mode, checks resume compatibility,
   and selects entries from the requested input mode. Completed entries may
   be excluded by resume policy.
4. Crystallization metadata is prefetched before expensive analysis. Manual
   input mode uses coordinate records and existing cache entries without
   downloading original-PDB metadata.
5. [driver/resources.py](../src/driver/resources.py) estimates entry memory
   from each entry's metadata and chooses a worker-process ceiling within the
   memory budget. [driver/memory_admission.py](../src/driver/memory_admission.py)
   then controls how many entries are active at once, backing off under memory
   pressure and recovering afterwards; the worker count alone does not
   determine concurrency.

### One entry

[worker.py](../src/worker.py) owns the entry lifecycle and its temporary
directory. The pool initializer installs `WorkerConfig` and logging once in
each process; subsequent tasks call `process()` with a PDB ID. Input
resolution lives in [worker_inputs.py](../src/worker_inputs.py) and the
analysis stages in [worker_stages.py](../src/worker_stages.py); `worker.py`
folds their outcomes into the `EntryResult`.

[worker_inputs.py](../src/worker_inputs.py) uses [inputs.py](../src/inputs.py)
to locate or retrieve files and read reflection limits and PDB-REDO metadata.
[coordinate_conversion.py](../src/coordinate_conversion.py) handles coordinate
conversion and first-model extraction, recording source-residue provenance in
`REMARK 950` records whose format [pdb_remarks.py](../src/pdb_remarks.py) owns,
writer and parser alike. Both EDSTATS and
[structure_analysis.py](../src/structure_analysis.py) use that prepared model.
`structure_analysis.py` is the facade for structure loading: the analyzed-model
types live in [structure_model.py](../src/structure_model.py), the loading
steps in [structure_loading.py](../src/structure_loading.py), raw PDB record
fields in [pdb_records.py](../src/pdb_records.py), and conformer choice in
[conformer_selection.py](../src/conformer_selection.py).
The original coordinate path is retained for deposited connection records,
crystallization context, and provenance.

After loading the structure and extracting crystallization context, the worker
checks whether analysis can proceed (the early-exit, density, and bond stages
are the functions of `worker_stages.py`). Entries without selected metals return
early; unknown element symbols can make metal absence indeterminate. Entries
above `MAX_ANALYZED_METAL_SITES` also return early with an explicit reason.
These paths avoid map generation and contact analysis.

For analyzable entries, the worker runs density analysis, extracts metal
statistics, then evaluates contacts. It passes the same `StructureContext`
to identification and coordination analysis so their atom selection and
coordinate provenance agree. Finally, it merges site summaries into the
statistics rows and returns an `EntryResult` containing rows, status, counts,
timings, warnings, and provenance.

## External CCP4 execution

[density_analysis.py](../src/density_analysis.py) invokes CCP4 as subprocesses.
Map generation and density extraction proceed as follows:

1. `mtzfix` validates or corrects the input MTZ map coefficients. The pipeline
   selects the original, corrected, or guarded twin-normalized coefficients.
2. With the default model-envelope scope, `fft` generates a full 2mFo-DFc map
   and `mapmask` crops it around the model. If the crop is smaller and safe,
   `fft` generates the mFo-DFc map and `mapmask` applies a matching model crop.
   Otherwise, the pipeline retains the full 2mFo-DFc map and generates a full
   mFo-DFc map. With full-map scope, `fft` generates both full maps directly.
3. `edstats` combines the prepared model and both maps to produce residue
   statistics.
4. `edstats_statistics.py` extracts metal-site rows and density context from
   those statistics.

Twin normalization is a guarded recovery path after MTZFIX validation fails
for an entry explicitly marked twinned in PDB-REDO metadata. Each CCP4 invocation
has its own timeout. Maps and intermediate files belong to the worker's scratch
directory and are normally deleted after extraction; `--keep-intermediates`
preserves them.

## Coordination module relationships

[coordination/analysis.py](../src/coordination/analysis.py) orchestrates contact
analysis and returns bond rows, candidate rows, site summaries, and stage
metadata. Its collaborators have distinct responsibilities:

| Module | Responsibility |
| --- | --- |
| [structure_analysis.py](../src/structure_analysis.py) | Load the analyzed model; its types (`AtomSite`, `StructureContext`, `ContactImage`) and neighbor searches including symmetry images live in [structure_model.py](../src/structure_model.py). |
| [policy.py](../src/coordination/policy.py) | Search radii and scoring thresholds shared by every coordination stage: the 4 Å search, the 0.75 Å first-sphere tolerance, the 0.8 Å special-position cutoff, the 6 Å nearby-metal radius, and the |z| >= 6 outlier cutoff. |
| [candidates.py](../src/coordination/candidates.py) | Discover donor-like atom images around a metal, collapse near-coincident special-position images, and merge proximity with declaration provenance. |
| [eligibility.py](../src/coordination/eligibility.py) | Apply the donor rule and the literature-distance rule to decide which candidates are first-sphere contacts. |
| [geometry.py](../src/coordination/geometry.py) | Score assigned contacts with the DPI-aware z-score and group contacts that share one donor-residue image. |
| [site_environment.py](../src/coordination/site_environment.py) | Contact-independent per-metal context: entry model statistics, neighbouring metals, crystallographic site symmetry, and the metal's parent component type. |
| [site_summary.py](../src/coordination/site_summary.py) | Define the typed `SiteSummary`, the analysis's share of the site columns, and assemble it from the assessed contacts of both search scopes. |
| [density_zscores.py](../src/coordination/density_zscores.py) | Index one entry's EDSTATS rows by metal site or author identity to attach the RSZD triple to bond rows. |
| [declared_connections.py](../src/coordination/declared_connections.py) | Resolve deposited `LINK` and mmCIF connection records into contact candidates. |
| [donor_chemistry.py](../src/coordination/donor_chemistry.py) | Determine which donor chemistries permit inferred contacts. |
| [dpi.py](../src/coordination/dpi.py) | Calculate coordinate-precision components used in geometry assessment. |
| [contact_record.py](../src/coordination/contact_record.py) | Carry candidate provenance, eligibility, geometry, and multi-donor assessments. |
| [reference_data.py](../src/reference_data.py) | Load and verify reference distances and cofactor classifications. |
| [edstats_statistics.py](../src/edstats_statistics.py) | Validate the EDSTATS residue table, join its rows to coordinate residues, and aggregate the density-context row; contact analysis builds its per-site density z-score index from those rows. |
| [schema.py](../src/coordination/schema.py) | Define contact/site output columns and serialize their values. |

Proximity candidates and declared connections are merged before assessment.
Candidate evidence is broader than assigned bonds. Geometry summaries feed
both the metal-site output and downstream confidence preparation.

## Outputs, confidence, and recovery

Workers return data to the driver; they do not append to the combined CSVs.
[driver/writers.py](../src/driver/writers.py) writes site, bond, candidate,
crystallization, density-context, and optional confidence rows. The manifest row
is written last as the entry's completion marker. Entries are collected as they
finish, so output order is not guaranteed to match input order.

[confidence_score/](../src/confidence_score/) runs in the driver and uses
the returned site and bond evidence:

| Run mode | Confidence behavior |
| --- | --- |
| Single entry, ID file, manual input, or capped run | Score each completed entry against an explicit, output-directory, or bundled frozen reference, in that search order. Without a reference, emit classifications without empirical rankings. |
| Uncapped database run | Stream compact confidence inputs, then finalize scores and a reusable reference when the batch has no recoverable unfinished entries. |
| `--no-bonds` | Skip contact analysis and disable confidence output. |

When confidence scores are available,
[driver/review_queue.py](../src/driver/review_queue.py) builds the review queue
by joining `REVIEW`/`SUSPECT` sites to the crystallization summary.
Crystallization metadata does not participate in scoring.

Recovery spans several layers:

- The worker preserves geometry analysis after explicitly handled density
  timeouts or MTZFIX validation failures. A bond-stage failure preserves density
  rows already produced. Other entry exceptions become entry outcomes rather
  than stopping the whole batch.
- [driver/dispatch.py](../src/driver/dispatch.py) runs the worker pool,
  admits entries as memory permits, monitors worker deaths, and records
  retryable failures for tasks that cannot return a result. It manages worker
  and CCP4 process shutdown.
- [driver/resume.py](../src/driver/resume.py) validates existing outputs and
  stages replacements; an unsuccessful retry does not overwrite a protected
  previous result.
- [driver/output_lock.py](../src/driver/output_lock.py) provides exclusive
  output ownership. [scratch.py](../src/scratch.py) creates the marked scratch
  directories workers and resume staging use, and sweeps the ones an earlier
  run left behind; it lives outside the driver so the worker does not import
  driver code.
- [run_logging.py](../src/run_logging.py) carries worker diagnostics to driver
  logging. [driver/runlog.py](../src/driver/runlog.py) writes the run report
  file and its per-entry diagnostics CSV through the CLI's cleanup path,
  including interrupted or failed runs.

## Shared contracts and separate commands

| Module | Shared role |
| --- | --- |
| [run_config.py](../src/run_config.py) | Validated command-line configuration. |
| [worker_contracts.py](../src/worker_contracts.py) | Worker configuration, entry results, and input provenance records. |
| [output_rows.py](../src/output_rows.py) | Typed site rows and CSV value formatting. |
| [codes.py](../src/codes.py) | Status, reason, warning, and contact vocabulary. |
| [analysis_config.py](../src/analysis_config.py) | Analysis-policy identity and compatibility, including the 100-site entry limit. |
| [ccp4_setup.py](../src/ccp4_setup.py) | Locate the CCP4 setup script and prepare the process environment. |
| [driver/errors.py](../src/driver/errors.py) | The driver's fatal-error type; raising it ends the run with exit code 1. |
| [driver/resources.py](../src/driver/resources.py) | Per-entry memory estimates and the worker ceiling for a memory budget. |
| [driver/memory_admission.py](../src/driver/memory_admission.py) | Admission of entries under memory pressure, with backoff and delayed recovery. |
| [metal_elements.py](../src/metal_elements.py) | Recognized metal elements. |
| [pdb_remarks.py](../src/pdb_remarks.py) | The `REMARK 950 ALCHEMY` provenance records of converted PDB files: record layouts, writer, and parser. |
| [gemmi_typing.py](../src/gemmi_typing.py) | Typed views of Gemmi members its stub leaves untyped. |
| [driver/progress.py](../src/driver/progress.py) | Batch progress reporting. |
| [worker_memory.py](../src/worker_memory.py) | Release idle memory after an entry's analysis frame is gone. |
| [worker_inputs.py](../src/worker_inputs.py) | Resolve one entry's inputs into the first-model PDB, structure, and provenance. |
| [worker_stages.py](../src/worker_stages.py) | The per-entry early-exit, density, and bond stages and the outcomes they return. |
| [_version.py](../src/_version.py) | Software version used in provenance. |

The maintenance tools are separate entry points, never automatic pipeline
stages. [build_metallocofactor_catalog.py](../tools/build_metallocofactor_catalog.py)
rebuilds the bundled cofactor catalog; [stamp_distance_table.py](../tools/stamp_distance_table.py)
updates or checks distance-table metadata. Normal runs verify and read their
committed artifacts through `reference_data.py`. See
[reference-data maintenance](maintenance.md) before changing those artifacts.

The `confidence_score` package also exposes standalone `finalize` and `score`
subcommands for prepared confidence-input files, run as
`PYTHONPATH=src python3 -m confidence_score`. Its modules separate the column
vocabulary, input preparation, classification and ranking, reference
persistence, and the command line. Normal analysis calls its functions directly
from the driver.
