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
