# Manuscript confidence reference

These are the unchanged frozen reference files from the finalized
[Alchemy PDB-REDO dataset, August 17, 2026](https://doi.org/10.5281/zenodo.22032936).
They are the reference used for the manuscript, not the September rerun or the
earlier, uncorrected August output.

- Reference ID: `alchemy-confidence-8ba6808c816791ffbb87`
- Cohort ID: `alchemy-cohort-2e97cf013eefa9d8e0b4`
- Cohort: 330,978 sites from 76,954 entries
- Density reference: 330,887 observations
- Geometry reference: 275,870 observations

`metadata.json` records the scoring policy, source provenance, and cohort
identity. `component_distributions.csv` contains the empirical distributions
used to rank new sites. Both files are copied byte for byte from
`output/confidence_reference/` in the dataset's
`Alchemy-PDB-REDO-20260817-part1-sites-confidence.tar.gz` archive.

Single-entry, ID-file, manual, and capped runs use this reference automatically
when the output directory has no reference of its own. No extra download is
needed. `--confidence-reference-dir` overrides automatic selection. An uncapped
full-database run creates its own reference under the output directory.

The numerical scores rank available density and geometry evidence against this
cohort. Missing evidence can leave component scores blank; classifications
continue to follow the documented thresholds and combination rule.

Do not regenerate or replace these files during routine analysis. Their
SHA-256 checksums are:

```text
92ca1c704db005172eee9111d1e0d7cad907d736bf20e7b54c160814a1314b4e  component_distributions.csv
b62ceaf812d4512c77740290d5b7dcf9d986df1d385d711e6492c4e0ba9c0b4e  metadata.json
```
