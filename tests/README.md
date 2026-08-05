# Tests

The unit tests use small structures built in `tests/helpers.py`. The integration
tests run the full pipeline with CCP4 and three PDB-REDO entries.

Install the test dependency and run the suite with:

```bash
python3 -m pip install '.[test]'
python3 -m pytest
```

Alchemy supports Python 3.11 and newer and requires gemmi 0.7 or newer.

## Common commands

```bash
# Fast tests, with no CCP4 or network access
python3 -m pytest --no-ccp4 --no-network --skip-slow

# Full PDB-REDO integration tests
python3 -m pytest tests/test_pipeline_integration.py \
    --require-ccp4 --require-entry-data -v

# Coverage, as CI measures it
python3 -m pytest --no-ccp4 --no-network --skip-slow \
    --cov=src --cov-report=term-missing
```

CCP4 must provide `mtzfix`, `fft`, `mapmask`, and `edstats`. If they are not
already on `PATH`, source the CCP4 setup script first:

```bash
. /opt/xtal/ccp4-9/bin/ccp4.setup-sh
```

## What CI runs, and what it does not

CI runs the offline lane — `--no-ccp4 --no-network --skip-slow` — on Linux,
Windows and macOS, with Linux additionally enforcing `ruff check`,
`ruff format --check`, Mypy, and the coverage floor. Provisioning CCP4 and the pinned entry data in CI
is deliberately not done: the setup cost is not considered worth it.

The consequence is worth stating plainly, because it is not visible from a green
run. Nothing in CI executes `mtzfix`, `fft`, `mapmask`, or `edstats`, and nothing
runs the pipeline end to end. Every `slow` test is also `ccp4`-marked, so
`--skip-slow` removes nothing that `--no-ccp4` had not already removed. A defect
in how Alchemy invokes a CCP4 program, or parses what one writes, can pass CI
with the whole suite green.

That is not hypothetical. Commit `e0429d0` read the EDSTATS `NR` column as an
ordinal over the whole model when EDSTATS restarts it per chain, which produced
zero output for every multi-chain entry — most of the PDB. All 1142 offline
tests passed, because the synthetic `stats.out` in `helpers.py` numbered `NR` the
same incorrect way. Only a real `edstats` run distinguished them.

So run the full lane locally before merging changes to CCP4 invocation, to the
parsing of any CCP4 program's output (`metal_identification.py`,
`density_analysis.py`), or to the worker's stage sequence:

```bash
python3 -m pytest --require-ccp4 --require-entry-data
```

Use those `--require-*` options rather than a plain `pytest`. Without them a
missing capability skips silently, which looks identical to passing. With them
the run fails instead, so an unexercised lane cannot be mistaken for a verified
one. Keep synthetic fixtures honest about the real format they stand in for —
where a fixture and the code it feeds share an assumption, the offline suite
cannot tell you the assumption is wrong.

## Integration entries

The end-to-end tests use checksum-pinned PDB-REDO files for:

- `9myr`: two Cys3-His zinc sites
- `6nlr`: the Snell-group multi-site Mn, Co, Fe, and Ca case
- `9nxl`: a metal-free control

These entries are regression fixtures, not a representative sample of the PDB.
If no cache is configured, pytest downloads them into its temporary directory.
Set a persistent cache to avoid downloading them again:

```bash
ALCHEMY_TESTS_CACHE="$HOME/.cache/alchemy-test-entries" python3 -m pytest
```

The older `ALCHEMY_TEST_CACHE` name is still accepted. Without a warm cache or
network access, entry-backed tests skip.

## Markers and options

- `ccp4`: requires the CCP4 programs listed above
- `network`: may contact PDB-REDO
- `entry_data`: requires the pinned integration entries
- `slow`: runs the end-to-end pipeline

`--no-ccp4` and `--no-network` disable those capabilities. The corresponding
`--require-ccp4` and `--require-network` options make a missing capability fail
the run. `--require-entry-data` ensures the integration tests did not all skip.

## Known defects

`test_known_limitations.py` is the ledger for validated-but-unfixed defects. It
currently holds none: every finding it once pinned has been fixed and its test
moved to the module that owns the behaviour, so the file collects no tests and
documents the protocol for the next one.

That protocol is: assert the behaviour Alchemy *should* have, mark it
`xfail(strict=True)` with a source anchor in the reason, and move it out when it
turns green. Strictness is the point -- an unexpected pass fails the suite, so a
fix cannot land unnoticed. Defects that can only be reproduced by killing a
process or exhausting memory are recorded as explicit skips with the manual
procedure in their docstrings.

## Adding tests

- Use `tmp_path` or `tmp_path_factory` for generated files and output.
- Prefer the synthetic builders in `tests/helpers.py` for unit tests.
- Mark tests that require CCP4, network access, entry data, or a slow run.
- Assert observable behavior rather than internal calls.
- Put unfixed source defects in `test_known_limitations.py`; test changes should
  not silently patch `src/`.
