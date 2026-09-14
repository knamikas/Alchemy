# Release notes

## 1.0.0 — Unreleased

Initial release of Alchemy for crystallographic validation of modeled metal
sites. Run Alchemy from a source checkout using the
[installation and first-entry instructions](README.md#requirements).

### Features

- Analyze individual PDB-REDO entries, ID lists, local mirrors, or manually
  supplied coordinate and MTZ files.
- Combine local electron-density evidence and first-sphere coordination
  geometry into PASS, REVIEW, and SUSPECT classifications and empirical
  support scores.
- Report metal sites, assigned contacts, candidate contacts, crystallization
  context, confidence scores, and run provenance in documented CSV outputs.
  Crystallization conditions provide context and do not affect scoring.
- Support memory-aware worker scheduling, per-program CCP4 timeouts, and
  resumable batches with validated outputs.
- Include the fixed cofactor catalog, literature distance table, and frozen
  manuscript confidence reference with their provenance.

### Manuscript confidence reference

The reference in
[`src/data/confidence_reference/`](src/data/confidence_reference/README.md)
is copied unchanged from the finalized August 17, 2026
[manuscript dataset](https://doi.org/10.5281/zenodo.22032936). Its cohort contains
330,978 sites from 76,954 entries, and its reference ID is
`alchemy-confidence-8ba6808c816791ffbb87`.

Single-entry, ID-file, manual, and capped runs use it automatically when their
output directory has no reference of its own. `--confidence-reference-dir`
selects an explicit reference. Uncapped full-database runs build their own
reference under the output directory. Runs with `--no-bonds` do not produce
confidence scores.

### Fixes included

- Site rows now carry the `suspect_multi_donor_group` context warning that
  the method reference documents; previously only bond rows recorded it.
- Crystallization metal detection no longer reads mass units such as
  `PROTEIN 5 MG/ML` as magnesium, which affected the detected-metal columns
  and review-queue context flags.
- Legacy PDB remark temperatures stated in Celsius are converted to kelvin,
  and ambiguous unitless values are left blank instead of being recorded as
  kelvin.
- The manifest `retryable` field is derived once from the entry status and
  reason codes instead of being set by each stage in turn.
- A bond-enabled resume can complete entries previously run with `--no-bonds`
  without requiring confidence rows from the unfinished stage. Integrity
  checks remain in place for completed results.
- The memory-admission regression test selects its simulated Linux environment
  explicitly on Windows and macOS.
- Integration tests handle documented blank z-scores and verify actual map
  cropping before comparing numerical outputs.

### Tested software and platforms

Real CCP4 integration checks were run on Linux with this environment:

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| Gemmi | 0.7.5 |
| NumPy | 2.5.2 |
| typing_extensions | 4.16.0 |
| CCP4 suite | 9.0.015 |

The integration fixtures are checksum-pinned inputs for `9myr`, `6nlr`, and
`9nxl`. A fresh `9myr` run also verified automatic use of the bundled manuscript
reference and numerical scores for both zinc sites.

The [automated checks](https://github.com/knamikas/Alchemy/actions/runs/34645238753)
passed on Linux with Python 3.11 and 3.12, and on Windows and macOS with Python
3.12. These CI jobs exercise the offline suite; Linux also checks lint,
formatting, types, and the 86% coverage minimum. CI does not provision CCP4 or
run the full crystallographic pipeline on those platforms. See the
[test guide](tests/README.md) for the separate offline and CCP4 test commands.

Python 3.11 or later and the dependencies in `pyproject.toml` are required.
CCP4 must provide `mtzfix`, `fft`, `mapmask`, and `edstats`.

### Known limitations

- The bundled distance table covers a limited set of metal–donor environments.
  Unsupported contacts cannot receive geometry z-scores, and SER, THR, and TYR
  reference values are approximations. See
  [reference coverage](docs/method.md#reference-coverage-of-the-donor-table).
- Missing density or geometry evidence can leave component scores blank.
  Numerical scores rank sites against the reference cohort; classifications
  follow the documented thresholds and combination rule. Generated symmetry
  contacts do not establish biological function.
- Entries with more than 100 selected metal sites are excluded from detailed
  analysis. Their detected counts and exclusion reason remain in the manifest.
- Manual-file runs need suitable PDB-REDO metadata through `--data-json` for
  DPI and derived geometry z-scores. Measured distances remain available
  without that metadata. See [manual inputs](docs/usage.md#process-manual-files).
- CCP4 tool limits and unsuitable reflection data can prevent an entry from
  completing. The manifest and run logs record the outcome and reason; see
  [failure diagnosis](docs/operations.md#diagnose-failures).

For interpretation, units, missing values, and joins, use the
[output schema](docs/output-schema.md). The full calculation definitions are
in the [method reference](docs/method.md).
