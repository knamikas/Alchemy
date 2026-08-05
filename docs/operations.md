# Operations

Running batches, resuming them, and reading what a run wrote. See
[usage.md](usage.md) for the options themselves and
[maintenance.md](maintenance.md) for the bundled reference data.

## Operational notes

- Alchemy separates three kinds of output. The progress line and the final
  result summary go to **stdout**, so a redirected stdout remains a usable
  record of what was produced. Diagnostics go to **stderr** as log records,
  controlled by `-v`/`--quiet`/`--log-file`. The per-run report in
  `<log-dir>/alchemy_run_*.log` -- `<output-dir>/logs/` unless `--log-dir`
  says otherwise -- is written separately and is unaffected by
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
- Before appending on resume, Alchemy verifies that every terminal manifest
  entry is backed by the enabled output files and that its selected-statistics,
  bond, candidate, and confidence row counts agree with the manifest. Duplicate
  complete manifest IDs and duplicate selected-site keys are refused. Rows
  written before an interrupted entry reached its manifest row remain eligible
  for staged replacement and do not invalidate the resume.
- The statistics header is a fixed schema rather than a copy of whatever EDSTATS
  emitted, because `extract_metal_statistics` already requires the EDSTATS
  residue table to match the standard column set and order.
- Resume retries are staged separately. Existing rows are replaced only after a
  retry produces a terminal result and the retry batch completes; failed or
  interrupted retries leave the previous rows intact. `--resume --no-bonds`
  preserves existing bond and candidate rows and their manifest counts. Entries
  originating from a bond-disabled run retain blank `n_bonds` and
  `n_candidates`, so a later bond-enabled resume will process them.
- `--resume --retry-partials` reprocesses non-retryable `partial` and `error`
  entries from the manifest after a processing improvement while continuing to
  protect `ok` entries. Optional `--id` or `--id-file` selectors restrict that
  set. Skips, retryable errors, and retryable partials already follow ordinary
  resume behavior. The
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
  positive-occupancy metal sites and no unknown-element atoms finish with
  `n_metals=0` without
  running `mtzfix`, either FFT, or `edstats`. These stages cannot produce
  metal-site output for such entries. Progress and completion summaries report
  these successful negative results as `no_metals`; valid zero-occupancy metal
  records remain visible through the `zero_occupancy_atoms` warning but are not
  counted as sites. `no_metals` is an informational subset of `ok`, whereas
  `skip` remains reserved for entries that could not be processed
  operationally. If any atom has a missing or invalid deposited
  element, metal absence cannot be established under the no-inference policy;
  the entry instead finishes as terminal `partial` with
  `metal_presence_indeterminate` and is not counted as `no_metals`.
- The command exits nonzero when any entry ends as `error`, `skip`, or a
  retryable `partial`. Completed `ok` and terminal `partial` results exit
  successfully.
- One Alchemy process on one machine owns an output directory at a time. The
  lease is an advisory `flock`, which the kernel enforces between processes on
  the same host. Across a network filesystem it is only as reliable as that
  filesystem's lock support, so two cluster nodes pointed at one `--output-dir`
  may both believe they own it and both truncate the result CSVs. Give
  concurrent runs separate output directories rather than relying on the lease
  to arbitrate between hosts. A run takes a
  non-blocking advisory lease on `<output-dir>/.alchemy.lock` before it reads,
  replaces, or resumes any result file. A second run fails immediately and
  reports the current owner's process, host, start time, and command instead of
  touching those results. The lock file intentionally remains after exit; the
  operating-system lease, not the file's presence, determines whether the
  directory is busy, and the lease is released automatically if the process
  exits or crashes. Alchemy refuses a lock path that is a symbolic link,
  non-regular file, foreign-owned file, or an inode with multiple hard links,
  so recording lease metadata cannot overwrite another file through that path.
- Startup cleanup removes only Alchemy scratch directories carrying a valid
  disposable ownership marker. Unmarked directories, symlinks, malformed
  markers, and intermediates retained by `--keep-intermediates` are left alone.
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
- An unanticipated exception ends the entry as `error`, and its type is recorded
  so a large run can be triaged. A parse, lookup, arithmetic, or type error
  describes the entry's data or Alchemy's own code and will recur while both
  stay the same, so it is reported as `deterministic_processing_error`. Anything
  else — an `OSError`, a `MemoryError`, a `RuntimeError` from a CCP4 program —
  may describe the machine rather than the entry, and is reported as
  `unexpected_processing_error`. The distinction is advisory and does not change
  what `--resume` does: every `error` entry is retried either way, because a
  resumed run may have been given a repaired input file or a re-downloaded
  mirror entry, and Alchemy does not checksum its inputs to tell. Skipping an
  entry the operator had just fixed would be worse than repeating one.
- The manifest's `error` column holds one truncated line naming the exception.
  The traceback is written to the debug log, so `--log-file`, or `-v` on the
  console, is what locates an unanticipated failure without rerunning the entry
  by hand under `--keep-intermediates`.
- The candidate-output migration expands all four CSV schemas. `--resume`
  refuses to mix new rows with incompatible pre-migration headers; use a new `--output-dir`
  for the first run. All four headers are compared in full,
  including the EDSTATS block of `metal_stats_all.csv`, so appended rows cannot
  be silently misaligned by output from a different EDSTATS build.

## Manifest reason codes

`reason_codes` is a `|`-separated list explaining why an entry is not a plain
`ok`. The vocabulary is defined once as `ReasonCode` in `src/codes.py`, and
`tests/test_documentation.py` fails if a member is missing from the table below,
so a code cannot be added or renamed without this list following it.

| Code | Meaning |
| --- | --- |
| `worker_process_died` | The worker holding the entry died before returning a result; the driver recorded the loss on its behalf. |
| `missing_input` | A coordinate or reflection file named on the command line, or expected in the mirror, was absent. |
| `ccp4_tool_timeout` | A CCP4 program was killed at `--ccp4-timeout`. It reported nothing about the entry, so this is retryable. |
| `mtzfix_validation_failure` | `mtzfix` failed its consistency re-test, and the entry is not a PDB-REDO-declared twin. |
| `unexpected_processing_error` | An unanticipated exception whose type leaves a retry meaningful, such as an `OSError`. |
| `deterministic_processing_error` | An unanticipated exception that will recur identically on the same inputs, such as a parse or lookup error. Terminal. |
| `metal_presence_indeterminate` | An atom's deposited element could not be trusted, so metal absence cannot be established and no site is analysable. |
| `bond_stage_failure` | The geometry stage raised, so its rows are not legitimate density-only evidence. |
| `cofactor_coordinate_join_failed` | An EDSTATS row for a catalog cofactor matched no coordinate residue. |
| `ambiguous_coordinate_residue_join` | An EDSTATS row matched more than one coordinate residue. |
| `cofactor_without_selected_metal` | A matched cofactor contains no configured metal site to select. |
| `metal_site_without_density` | A selected coordinate metal site is absent from the statistics table, so it has no density evidence; it remains included in the coordinate-site total `n_metals`. This is detected with or without bond analysis and can happen when a metal sits inside a multi-atom residue absent from the bundled cofactor catalog, which is a fixed snapshot: see [maintenance.md](maintenance.md) for rebuilding it. |
| `declared_connection_resolution_incomplete` | A source `_struct_conn` or `LINK` record named an atom that could not be resolved in the coordinate model. |
| `symmetry_search_unavailable` | The structure has no usable cell or space group, so only explicit contacts could be found. |
| `missing_first_sphere_reference` | No bundled reference distance covers a donor class present at the site, so those contacts cannot be z-scored. |
| `missing_dpi_metadata_source` | Manual input without `--data-json`: the reflection count has no source, which differs from a calculation that ran and failed. |
| `invalid_dpi_metadata` | The reflection count, R-free, or asymmetric-unit volume was present but not numeric. |
| `invalid_occupancy` | Deposited occupancies could not be read, or overfull alternates exceeded the tolerance, leaving `Ni` unusable. |
| `missing_or_invalid_reflection_count` | `NREFCNT` was absent or non-positive. |
| `missing_or_invalid_rfree` | `RFFIN` was absent or non-positive, and no R-free could be read from the coordinate file. |
| `missing_or_invalid_asu_volume` | The asymmetric-unit volume could not be derived from the cell and space group. |
| `invalid_dpi_atom_count` | `Ni` came out non-positive with every other DPI input valid. |
| `dpi_calculation_failed` | The DPI calculation raised; the entry keeps its contact distances, which do not require a DPI. |
