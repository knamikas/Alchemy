# Alchemy — Production Readiness Code Review

**Date:** 2026-07-31
**Scope:** Architecture, code organization, naming, readability. Not a bug hunt or security audit — but one correctness defect surfaced while verifying a related claim; it is recorded as 1.8 and has since been fixed.
**Reviewed:** `src/` (7,929 lines), `tests/` (14,031 lines), `tools/`, `pyproject.toml`, `.github/workflows/tests.yml`, `README.md`.
**Distribution model:** users obtain Alchemy by cloning the repository and running `python src/main.py`. It is not a package and is not installed, which shapes several findings below.

---

## Summary

This is well above average scientific code. The domain reasoning is careful, the failure semantics are deliberate, the comments explain *why* rather than *what*, and the test investment is serious (478 tests, 1,844 assertions, 1.77 lines of test per line of source, full offline suite green in 24 seconds on a bare checkout).

The gap to production is mostly **structure, setup documentation, and CI**, with one genuine correctness defect. Six things stand out:

1. **Invalid geometry coverage contaminated the frozen confidence reference** (1.8) — **fixed in `36e5dbf`**. A fix that landed in the scoring path had never reached the reference-building path, so a corrupt site was reported `unscorable` in the output and simultaneously scored into the cohort every percentile is measured against. Reproduced, fixed, and verified byte-identical on the existing 24,365-row dataset.
2. **A fresh clone cannot be set up from the README** (1.9). The documented `pip install "gemmi>=0.7.0"` omits numpy, so a new user's first command is a traceback — verified. Every Quick Start example also invokes `conda run -n metal`, a personal environment name that appears nowhere outside the docs. Since the repo is the artifact, this is the first thing every user hits.
3. **`pyproject.toml` claims to be a distribution and isn't one.** Installing it publishes eight generic top-level modules into site-packages. Alchemy is not meant to be installed, so the fix is to declare that explicitly rather than to build a package.
4. **CI never runs the pipeline.** The single workflow job excludes 26 tests — every end-to-end run and the entire real-CCP4 density arm. Those tests exist and are good; nothing executes them on push.
5. **Two files carry the system.** `main.py` (3,126 lines) and `bond_analysis.py` (1,915 lines) are pipelines, not modules — each holds five or six independently-changeable responsibilities, and `_run` alone is 534 lines.
6. **The codebase has two standards.** `structure_analysis.py` is fully typed, dataclass-based, and snake_case. Every other module is untyped, dict-based, and camelCase in places. The good standard already exists in-repo; it just was not propagated.

Nothing here requires re-deriving the science. It is all mechanical, testable, and can be sequenced behind the existing suite.

**How to read this document.** Tier 1 is what blocks a production release. Tier 2 is architecture — large, low-risk, and worth doing before the next major feature. Tier 3 is readability and naming. Tier 4 is test coverage. Every claim is anchored to a file and line, and the empirical ones — packaging discovery and wheel build, marker selection, type-hint and key counts, the timeout audit, and the 1.8 reproduction — were verified by running them.

---

## Tier 1 — Fix before production

### 1.1 `pip install .` produces a namespace-polluting install

`src/` has no `__init__.py`, so setuptools src-layout auto-discovery resolves to:

```
packages   : ['data']
py_modules : ['density_analysis', 'confidence_score', 'bond_analysis', 'metal_elements',
              'main', 'ccp4_setup', 'structure_analysis', 'metal_identification']
package_dir: {'': 'src'}
```

*(Verified by running setuptools' `ConfigDiscovery` against a copy of the tree.)*

**The install is functional, not broken** — verified by `pip install --target`, which produces the eight modules plus `data/`, and `python -m main --help` runs correctly. The problem is what it claims and how fragile it is:

- Installing Alchemy claims the top-level names `main` and `data` in site-packages. Any other distribution — or any user script named `main.py` — collides with or shadows the pipeline.
- `[tool.setuptools.package-data] data = [...]` in [pyproject.toml:20-25](pyproject.toml#L20-L25) silently depends on `data` being the accidental package name. Add one `__init__.py` under `src/` and the bundled catalogs stop shipping, with no error.
- Building against the repository leaves untracked `build/` and `src/alchemy.egg-info/` in the checkout, because `*.egg-info/` is not in `.gitignore` (4.5).
- CI hides all of this: [.github/workflows/tests.yml](.github/workflows/tests.yml) runs `pip install -e '.[test]'`, and the tests pass only because [tests/conftest.py:28](tests/conftest.py#L28) inserts `src/` onto `sys.path`. The install step is effectively just a dependency installer.
- The same `sys.path` surgery is repeated in [tests/helpers.py:40-41](tests/helpers.py#L40-L41) and [tools/build_metallocofactor_catalog.py:22-27](tools/build_metallocofactor_catalog.py#L22-L27) — three places compensating for the missing package.

**Fix — declare nothing, deliberately.**

Alchemy is run as a script (`python src/main.py`), not imported as a library, and there is no intention to distribute it as a package. The defect is therefore not "it should be a package" — it is that `pyproject.toml` currently *claims* to be a distribution and silently publishes eight generic top-level names as a side effect. Making that claim explicitly empty resolves it:

```toml
# Alchemy is run as a script, not imported as a library. Declaring no modules
# keeps `pip install` a dependency-only operation instead of silently
# publishing `main`, `data`, and six other generic names into site-packages.
[tool.setuptools]
packages = []
py-modules = []
```

This keeps `[project.dependencies]` and `[project.optional-dependencies]` doing their job, so CI's `pip install -e '.[test]'` still resolves gemmi, numpy and pytest — it just stops installing source modules.

*(Verified: with this configuration `pip install --target` installs zero modules, the dependency metadata is unchanged, the offline suite still passes 853, and `import main` fails from outside the repository. The editable variant could not be exercised here — PEP 668 blocked it in this environment — so confirm `pip install -e '.[test]'` once on a real runner.)*

The three `sys.path` blocks then stay, because they are the mechanism rather than a workaround. They cannot be consolidated into a single insertion point — `tools/build_metallocofactor_catalog.py` runs standalone, without pytest, so it needs its own bootstrap. What *is* redundant is `helpers.py`'s copy, since `conftest.py` always runs first under pytest and helpers is imported nowhere outside `tests/`. The three currently disagree: [tests/conftest.py:28](tests/conftest.py#L28) adds both `tests/` and `src/`, [tests/helpers.py:40-41](tests/helpers.py#L40-L41) adds only `src/`, and [tools/build_metallocofactor_catalog.py:22-27](tools/build_metallocofactor_catalog.py#L22-L27) does its own — three divergent copies of one rule.

Two knock-on consequences of this decision, both acceptable but worth stating:

- **`importlib.resources` is unavailable**, so 2.4's data-loading fix becomes a single shared `DATA_DIR` constant in one module rather than a resource accessor. Same consolidation, simpler mechanism.
- **The splits in 2.1 and 2.3 add more flat top-level modules** — `inputs`, `worker`, `dpi`, `bond_schema` and so on — each of which is a generic name living on `sys.path`. That is fine inside a repository nobody installs, but it does mean the `sys.path` convention becomes load-bearing for a dozen modules instead of eight. Consolidating to one insertion point (above) is what keeps that manageable.

### 1.2 Confidence scoring weights are duplicated in three places

> **Deferred — not part of this plan.** Confidence scoring is not finalized, so no commit below acts on this. Recorded because whoever finalizes the score needs it: the weights are the thing most likely to be edited, and this is where that edit goes wrong.

The core scoring constants `0.50 / 0.35 / 0.15` appear in:

1. [src/confidence_score.py:368-371](src/confidence_score.py#L368-L371) — `score_site()`, the actual computation
2. [src/confidence_score.py:423-426](src/confidence_score.py#L423-L426) — `_scoring_metadata()`, which feeds `_reference_identifier()` and the compatibility gate in `load_reference()`
3. [README.md:378-379](README.md#L378-L379) — the published formula

The anchors are correctly single-sourced (`_scoring_metadata` references the `DENSITY_ANCHORS`/`GEOMETRY_ANCHORS` constants), which makes the weights an oversight rather than a choice.

**Why this is Tier 1:** copy 2 is the hash input for `reference_id`. Change a weight in `score_site` and forget `_scoring_metadata`, and the reference identifier is unchanged — so `load_reference` accepts a frozen distribution computed under *different* weights. The "refuse an incompatible reference" mechanism, which is the module's central safety property, has a hole exactly where a maintainer is most likely to edit.

**Fix:** `SCORE_WEIGHTS = {"density": 0.50, "geometry": 0.35, "interaction": 0.15}` at module scope, read by both call sites; generate the README formula from it or add a test asserting the README matches.

### 1.3 No timeout on any external process

No `subprocess.run` call in `src/` passes `timeout=`:

| Call site | Process |
|---|---|
| [src/density_analysis.py:266](src/density_analysis.py#L266) | every CCP4 binary (`mtzfix`, `fft`, `mapmask`, `edstats`) |
| [src/main.py:226](src/main.py#L226) | Windows CCP4 setup via `cmd` |
| [src/main.py:264](src/main.py#L264) | POSIX CCP4 setup via `bash` |
| [src/main.py:1092](src/main.py#L1092), [:1096](src/main.py#L1096) | `git rev-parse`, `git status` |

The only timeouts in the codebase are the HTTP download ([main.py:923](src/main.py#L923)) and the 1-second result poll ([main.py:2893](src/main.py#L2893)). A hung `edstats` or `fft` pins a worker slot indefinitely in a database-scale run. The driver has elaborate machinery for detecting *dead* workers ([`_dead_worker_pids`](src/main.py#L1201), [`_worker_death_result`](src/main.py#L1318), `WORKER_STALL_GRACE_S`) but nothing for *hung* ones.

**Fix:** a `CCP4_TOOL_TIMEOUT_S` constant applied in `density_analysis._run`, surfaced as a retryable reason code.

### 1.4 No logging framework

Zero uses of `logging` across `src/`, `tests/`, and `tools/`. Diagnostics are 42 `print()` calls in `main.py`, 3 in `confidence_score.py`, 1 in `density_analysis.py`, plus per-entry CCP4 log files and a bespoke [`_RunLog`](src/main.py#L2311) renderer.

For production this means: no severity levels, no way to raise or lower verbosity, no way to route diagnostics to a file/syslog/aggregator separately from user-facing progress output, and no structured fields for log processing. The unexplained truncations — `[:500]` at [density_analysis.py:271](src/density_analysis.py#L271), `[:300]` at [main.py:1534](src/main.py#L1534) and five other sites — are doing a job a logger should do.

**Fix:** a `logging` logger per module; keep `print` only for the progress line and the final human summary. `_RunLog` can stay — it is a genuinely good artifact — but it should be fed by a handler, not by parallel bookkeeping.

### 1.5 No LICENSE, and no lint/format/type gate

- No `LICENSE`, `CONTRIBUTING.md`, or `CHANGELOG.md` in the repository root. For a tool intended for publication or external use, the missing license is a blocker.
- No `ruff`, `black`, `flake8`, `mypy`, `pre-commit`, `setup.cfg`, `tox.ini`, or `.editorconfig`, and no lint configuration in `pyproject.toml`. Style is currently maintained by hand — mostly well, but with observable drift: 12 lines over 88 characters in `main.py`, 65 over 79, trailing whitespace at [main.py:2016](src/main.py#L2016), and `pathlib` mixed with `os.path` inside the 128-line [ccp4_setup.py](src/ccp4_setup.py).
- No coverage tooling (`pytest-cov`/`coverage` not installed, not declared, not run in CI). With 853 tests there is almost certainly good coverage, but nothing measures or defends it.

**Fix:** add `LICENSE`; add `ruff` (lint + format) and `mypy` configured to check `structure_analysis.py` strictly and the rest permissively, tightening as annotations land; add `pytest-cov` with a floor in CI.

### 1.6 Version string duplicated

`1.0.0` is declared in both [pyproject.toml:7](pyproject.toml#L7) and [src/main.py:121](src/main.py#L121) (`ALCHEMY_VERSION`), and the latter is written into every manifest row as run provenance. These can drift silently, which corrupts the provenance record.

**Fix:** a `src/_version.py` holding `__version__`, read by `pyproject.toml` via `dynamic = ["version"]`. Note that `version` cannot simply be dropped — `[project]` requires it — and `importlib.metadata` is not a substitute: the dist-info *is* installed even with `packages = []` (verified), but a bare clone never installs, so a source-checkout fallback is mandatory.

### 1.7 CI has one lane, and it excludes the entire end-to-end pipeline

[.github/workflows/tests.yml:26](.github/workflows/tests.yml#L26) runs only `pytest --no-ccp4 --no-network --skip-slow`. There is no second job. Measured marker selection:

```
-m "ccp4"                  → 25 tests
-m "slow"                  → 24 tests
-m "network"               → 24 tests
-m "entry_data"            → 23 tests
-m "ccp4 or slow or network" → 26 tests   ← exactly the 26 the offline run skips
-m "entry_data and not ccp4" →  0 tests
```

So **26 tests never execute on any push** — every `main.main()` end-to-end run, the real CCP4 density arm, and the `_MEASURED_RSZD` oracle at [test_pipeline_integration.py:106-127](tests/test_pipeline_integration.py#L106-L127), described in-file as "the only genuine oracle" for the density arm. The integration suite's true pass/fail state is whatever a developer last saw locally.

This also means the whole strict-lane apparatus in `conftest.py` is wired to nothing: `--require-ccp4` / `--require-network` / `--require-entry-data` ([conftest.py:104-116](tests/conftest.py#L104-L116)), the probe-vs-require reconciliation, the empty-selection guard, and the ran-nothing-so-fail hooks — roughly 70 lines built specifically to prevent a silently-all-skipped capability lane.

> **Accepted, not addressed in this plan.** Running these tests automatically needs a machine that is always on — a self-hosted runner only picks up jobs while it is powered and connected, and there is no always-on host available. The finding stands; the gap is being carried knowingly rather than closed. Two consequences worth holding onto: the 25 tests remain the only end-to-end check that exists, so **run them by hand before any significant merge or release** (`pytest -m "ccp4 or slow" --no-network --require-ccp4 --require-entry-data` — 25 passed, 854 deselected, ~2.5 minutes on a CCP4 machine); and because CI will not catch a break in the density or bond path, the offline coverage commits become load-bearing rather than optional. If a lab server or a licence-permitted container image becomes available later, this is a two-line change.

**Fix — and note that the obvious cheap fix does not work.** A cached `entry_data` lane is *not* achievable without CCP4: `entry_data and not ccp4` selects zero tests, because every `entry_data` test is also `ccp4` and `slow` (the block beginning at [test_pipeline_integration.py:424](tests/test_pipeline_integration.py#L424)). Running `pytest -m entry_data --require-entry-data` on a CCP4-less runner collects 23 cases, skips all 23, and exits 1.

So there are only two real options:

1. **Provide a CCP4-capable runner** — a container image with the four binaries, or a self-hosted/nightly job — and run `--require-ccp4`. This is the option that actually restores coverage.

   **This needs a trust boundary before it is switched on.** [.github/workflows/tests.yml:3-5](.github/workflows/tests.yml#L3-L5) triggers on `push` *and* `pull_request`. A persistent self-hosted runner executing checked-out pull-request code is arbitrary code execution on your machine, with access to whatever else that machine holds — including, on this box, a CCP4 install and any PDB-REDO mirror credentials. Either use an ephemeral isolated runner, or restrict the CCP4 job to trusted triggers (`push` to your own branches, `schedule`, `workflow_dispatch`) and leave `pull_request` on the hosted offline lane.

   **Pick the flags deliberately.** A cached, deterministic lane should run `--no-network --require-ccp4 --require-entry-data`, which selects 25 tests (`ccp4 or slow`). Reaching all 26 needs `--require-network` for the one live `pdb-redo.eu` smoke test, which imports an external dependency into your signal. The honest total for a deterministic lane is 878, not 879. Also decide the cold-cache behaviour: on a cache miss the snapshot must download, so a first run needs network even when steady-state runs do not.
2. **Write genuinely CCP4-free entry-data tests** — integrity checks over the pinned snapshot, marked `entry_data` but not `ccp4`. **Tried and reverted.** Six such tests were written and did run, but keeping them honest meant provisioning the snapshot in CI: fetching from pdb-redo.eu on every run makes the lane fail whenever upstream is slow or revised, and a project-controlled archive means publishing and maintaining one. The infrastructure outweighed what the tests checked, so this was dropped by decision rather than oversight. The metadata readers involved stay covered by unit tests elsewhere.

Option 1 is the one that matters. Option 2 is a useful supplement, not a substitute.

---

### 1.8 Invalid geometry coverage contaminated the frozen confidence reference — FIXED

> **Fixed in `36e5dbf`** (*Apply one coverage-validity rule to scoring and reference building*). The analysis below is retained because it is what a finalization pass needs to know; the defect itself is closed.
>
> `coverage_is_valid()` is now a shared public helper called by both `_score_prepared_row` and `finalize_database_confidence`, so a site scoring refuses can no longer enter the cohort it is measured against. `COVERAGE_POLICY` was added to `_scoring_metadata()` and to the `load_reference` compatibility gate, so a reference built under the old rule is refused rather than silently accepted.
>
> Verified three ways: the cohort now excludes damaged rows (out-of-range, blank, non-finite, negative, non-numeric — five parametrised cases, each confirmed to fail against the pre-fix loop); re-finalizing the real 24,365-row input reproduces a **byte-identical** distribution, so the fix changed nothing it should not; and `reference_id` moved from `…1f3b3916c29dfbb40c52` to `…0fcf30bc2ff40fba1551`, with the production reference re-finalized accordingly.
>
> An earlier draft of this review called the defect "latent." That was wrong, and the correction is worth keeping: `confidence_mode` is set to `"database"` automatically for any uncapped run with bonds enabled ([main.py:2603-2604](src/main.py#L2603-L2604)) and finalization then runs unconditionally, so nothing had to opt in for the defect to be exercised.

**This is the most severe item in this document, and the only outright correctness defect in it.**

`_score_prepared_row` ([confidence_score.py:522-541](src/confidence_score.py#L522-L541)) was fixed in `f62af87` to refuse a site whose geometry coverage is damaged, and its comment states exactly why:

> A blank, non-numeric or out-of-range coverage is damaged input, not evidence that geometry is irrelevant. Coercing it to zero removed the two geometry terms from the score, so a corrupt cell scored *higher* than the same site with real geometry evidence.

The **reference-building** loop inside `finalize_database_confidence` ([confidence_score.py:600-608](src/confidence_score.py#L600-L608)) never got that fix:

```python
coverage = _finite_float(row.get("geometry_coverage", ""))
if not math.isfinite(coverage):
    coverage = 0.0
result = score_site(rszd, zbond, coverage)
if result is not None:
    score_counts[result["confidence_score"]] += 1
```

Two distinct defects in five lines. Non-finite coverage is coerced to `0.0` — the precise behaviour the scoring path was fixed to stop. And there is no range check at all, so an out-of-range value passes `math.isfinite` and is silently clamped by `score_site`'s `min(1.0, max(0.0, …))`.

**Reproduced** with a two-row input (one valid site at coverage `1.0`, one corrupt at coverage `7.5`):

```
finalize_database_confidence -> total=2  scored=1  cohort=2
reference distribution        -> confidence_score,count
                                 100.0,2
output row 2bbb               -> status=unscorable  reason=geometry_coverage_invalid
```

The corrupt row is correctly reported `unscorable` in `confidence_scores_all.csv` — and simultaneously scored as `100.0` into the frozen distribution, and counted in the cohort denominator. Every percentile derived from that reference is contaminated, and `reference_id` records the contamination as legitimate policy.

**Fix:** the reference loop must use the same validation as the scoring path. Extract the `coverage_valid` test from `_score_prepared_row` into a shared helper and call it from both, so a site excluded from scoring is also excluded from the cohort. Add a regression test asserting that a corrupt row changes neither `cohort_size` nor the distribution.

**This also reverses a conclusion elsewhere in this document** — see 4.3.

---

### 1.9 A fresh clone cannot be set up by following the README

Alchemy is distributed by cloning the repository, so the README *is* the installer. It currently does not work.

**Following [README.md:87](README.md#L87) gives you a broken environment.** The stated setup is:

```bash
python -m pip install "gemmi>=0.7.0"
```

but [pyproject.toml:11-14](pyproject.toml#L11-L14) correctly declares `numpy>=1.17` as well, and [density_analysis.py:23](src/density_analysis.py#L23) imports it at module scope — which `main.py` reaches unconditionally through its own import block. Verified by blocking numpy in a fresh clone:

```
=> FAILS without numpy: ImportError: No module named 'numpy'
```

So a new user's first command after following the documented setup is an immediate traceback. [README.md:82](README.md#L82) has the same omission ("**Python package:** `gemmi>=0.7.0`").

*(numpy is genuinely required, not a stale declaration: gemmi 0.7.5 declares no dependencies of its own, so it does not pull numpy in; `density_analysis.py` uses it at ~20 sites for complex map coefficients and reflection masking; and `read_map_column_resolution` — numpy-based — is called from `process()` at [main.py:1473](src/main.py#L1473) on every entry of every run.)*

**Every Quick Start example bakes in a personal conda environment name.** [README.md:20-28](README.md#L20-L28) uses `conda run -n metal python src/main.py …` four times, and the same appears in the module docstring at [main.py:29-31](src/main.py#L29-L31). But `metal` occurs in exactly those six documentation lines and nowhere else — not in the code, the tests, or CI. Nothing in Alchemy requires conda at all.

So the defect is not a missing `environment.yml`; it is that the documented invocation is the author's local environment leaking into public instructions. A cloner is told to run a command referencing an environment that only exists on your machine, with no way to create it, while the adjacent setup section tells them to use pip.

**Confidence scoring is deliberately not shipped, but the documentation and the runtime message both imply it should be.** No reference is bundled — that is intentional, and consistent with [README.md:374](README.md#L374) describing "the provisional June 2026 fixed score." The problem is that nothing says so. [README.md:390-392](README.md#L390-L392) tells users Alchemy falls back to "the repository's `confidence_reference/`", and `DEFAULT_CONFIDENCE_REFERENCE_DIR` ([main.py:111](src/main.py#L111)) points at a directory that will never exist. A fresh clone running `--id 9myr` then prints:

```
No frozen confidence reference found in <dir1>, <dir2>; confidence scoring
disabled for this non-database run.
```

Naming the directories it searched invites the reader to conclude they have misconfigured something, when in fact they have hit intended behaviour. A user cannot tell "this feature is not released yet" from "you installed this wrong."

**Fix:** state in the README that no reference ships by default, that confidence scoring requires either an uncapped full-database run or an externally supplied `--confidence-reference-dir`, and that the score is provisional and not yet released. Reword the runtime message so it reads as expected behaviour rather than a missing file. Both are a few lines and neither commits you to publishing the score.

**The default mirror root is an institutional path.** `DEFAULT_ROOT = "/datasets/bioinfo/pdb-redo"` ([main.py:110](src/main.py#L110)) is the default for `--pdb-redo-root`, so the no-arguments invocation enumerates a directory that exists on your machines and nobody else's. The download path via `--id` covers the common case, but the bare `python src/main.py` a new user is most likely to try first fails in a way that looks like a bug rather than a configuration gap.

**Fix:** drop `conda run -n metal` from all six example lines — plain `python src/main.py …` is correct and imposes nothing — and state the real requirements plainly: Python ≥3.11, `gemmi>=0.7.0`, `numpy>=1.17`. An `environment.yml` is worth offering as a *convenience* under a neutral name, but it should not be the documented invocation, and it should not enshrine `metal`. Separately, either make the mirror root's absence a clear diagnostic or document it as site-specific where it appears.

This is small work, but under a clone-and-run model it is the first thing every user encounters, which makes it a release blocker rather than a documentation nicety.

*(The underlying mechanics are sound — see "What is genuinely good" for what a fresh clone does get right.)*

---

## Tier 2 — Architecture

### 2.1 `main.py` (3,126 lines) is five modules

The file's own section banners name the split:

| Banner | Lines | Size | Actually contains |
|---|---|---|---|
| CCP4 environment | 174–357 | ~184 | env capture (POSIX + Windows), tool verification, config resolution |
| Per-entry input preparation | 358–1,064 | ~707 | gunzip, mmCIF→PDB conversion, residue-identity provenance, legacy PDB packing, resolution/twin metadata, HTTP download + cache, entry enumeration |
| Worker | 1,065–1,612 | ~548 | provenance capture, worker globals, in-flight protocol, dead-worker recovery, signal handling, pool shutdown, scratch sweeping, `process()` |
| Driver | 1,613–3,126 | ~1,514 | resume/manifest logic, CSV merge, resource probing, CLI parsing, staging, writers, progress, run log, orchestration |

The banners are an accurate decomposition that was never carried out. Splitting along them yields four ~200–700 line modules plus a thin `main.py`.

The heaviest single offender is [`_run`](src/main.py#L2551) at **534 lines**, which alone does: catalog loading, CCP4 resolution, output-directory setup, scratch sweeping, path construction, run-mode classification, confidence-mode selection, resume schema validation, entry selection, stale-output cleanup, worker sizing, worker-config assembly, staging setup, file opening, the pool loop, staging commit, summary printing, confidence finalization, and exit-code computation. Nineteen responsibilities, no return until line 3,084.

**Fix (suggested layout):**

```
src/ccp4_setup.py    ← absorbs main.py's env resolution (see 2.2)
src/inputs.py        ← entry resolution, download/cache, conversion
src/coordinate_conversion.py  ← mmCIF↔PDB identity records, legacy packing
src/worker.py        ← process(), worker config, in-flight protocol
src/driver/          ← pool.py, resume.py, writers.py, progress.py, runlog.py
src/cli.py           ← parse_args + main()
```

Target for `_run`: a ~60-line orchestrator over named phase functions.

`src/driver/` needs an `__init__.py` to be importable as `driver.pool`, but that makes it a package *directory*, not an installable distribution — the two are unrelated, and 1.1's `packages = []` keeps it uninstallable either way. If you would rather avoid even that, flatten to `driver_pool.py`, `driver_resume.py` and so on; the split is the point, not the nesting.

### 2.2 CCP4 handling is split across two modules along an incoherent seam

`ccp4_setup.py` owns discovery; `main.py:247-359` owns resolution. The seam is not a responsibility boundary, and it has produced four concrete defects:

- **Two definitions of "the default config files."** [ccp4_setup.py:10-13](src/ccp4_setup.py#L10-L13) lists two paths; [main.py:286-292](src/main.py#L286-L292) lists those two *plus* a repo-local `.alchemy/ccp4.json`. Production always uses main.py's; `ccp4_setup.DEFAULT_CONFIG_FILES` is reached only by [tests/helpers.py:836](tests/helpers.py#L836). The test harness and the application disagree about where configuration lives — and `load_ccp4_setup_config`'s own docstring documents a past bug caused by exactly this class of drift.
- **`verify_ccp4` re-implements `ccp4_tools_available`.** [main.py:275-284](src/main.py#L275-L284) and [ccp4_setup.py:123-128](src/ccp4_setup.py#L123-L128) run the identical `shutil.which(tool, path=env.get("PATH"))` sweep. One returns a bool, one raises.
- **`find_ccp4_setup`'s `explicit_setup` parameter is dead in production.** [main.py:340-345](src/main.py#L340-L345) always passes `None`, because main.py handles `--ccp4-setup` itself at [main.py:324-335](src/main.py#L324-L335) with stronger semantics. Two implementations of one concept; only one runs.
- **Library functions raise `SystemExit`.** [main.py:258, 266, 280](src/main.py#L258). This is why [tests/helpers.py:846](tests/helpers.py#L846) must catch `(Exception, SystemExit)` — direct evidence of the smell — and it means CCP4 resolution can never be reused outside a CLI process.

**Fix:** move `resolve_env`, `_resolve_env_windows`, `_normalize_path_key`, `_parse_windows_set_output`, and the verification predicate into `ccp4_setup.py`, raising a `Ccp4SetupError`. Leave `resolve_ccp4_environment(args)` in the CLI layer as the only `SystemExit` raiser. One `DEFAULT_CONFIG_FILES`. Delete the dead `explicit_setup` branch.

### 2.3 `bond_analysis.py` (1,915 lines) is six subsystems

| Responsibility | Lines | Size |
|---|---|---|
| Reference-data file parsing | 107–135, 260–286 | ~55 |
| DPI / crystallographic metadata | 292–409 | ~120 |
| EDSTATS density-sigma join | 466–490 | ~25 |
| `struct_conn`/`LINK` declaration resolution | 795–1,140 | ~350 |
| Bond geometry & coordination chemistry | 415–793, 1,143–1,400 | ~600 |
| CSV schema + row serialization | 140–257, 1,544–1,769 | ~290 |

The DPI block computes a whole-structure precision index and never touches a bond. The sigma join is EDSTATS parsing — and [metal_identification.py:39-54](src/metal_identification.py#L39-L54) already owns `EDSTATS_COLUMNS`, so EDSTATS knowledge is split across two modules. The declaration-resolution block is a self-contained mmCIF provenance subsystem with its own vocabulary that shares nothing with z-scores.

Worst single function: [`_collect_declared_candidates`](src/bond_analysis.py#L982) — **159 lines, nesting depth 6**, seven interacting booleans, and six sequential `continue` guards.

**One caveat for whoever extracts this.** The metal-partner check that appears twice ([:1025-1027](src/bond_analysis.py#L1025-L1027), then widened by `or` at [:1033-1035](src/bond_analysis.py#L1033-L1035)) is *deliberate*, not redundant, and must survive the refactor. The first call runs before `find_cra`, which can raise, so the metal determination — and its audit trail — survives a failed CRA resolution; the second uses the resolved atom's element, which is the preferred evidence. The rationale is documented in [`_declared_partner_is_metal`'s docstring](src/bond_analysis.py#L902) ("Prefer the element of the atom resolved in the source model… fall back only to unambiguous component/atom identifiers"), and the behaviour is protected by [test_declared_connections.py:1221](tests/test_declared_connections.py#L1221). Extract the loop body, but keep both phases.

**Fix:** extract `dpi.py`, `declared_connections.py`, `reference_data.py`, and `bond_schema.py`; move the sigma helpers into `metal_identification.py`. Leaves ~600 lines of genuine bond chemistry.

### 2.4 Bundled data has no owner

Three path constants, three loaders, two of them at import time, one catalog parsed twice by two different parsers:

- [metal_identification.py:16-20](src/metal_identification.py#L16-L20) defines `COFACTOR_CATALOG_PATH`; [bond_analysis.py:42-43](src/bond_analysis.py#L42-L43) independently recomputes `BASE_DIR`/`DATA_DIR`; [density_analysis.py:26](src/density_analysis.py#L26) computes a third `BASE_DIR`.
- `bond_analysis` imports `COFACTOR_CATALOG_PATH` *from the EDSTATS parsing module* and re-parses the same file with a **stricter** parser ([bond_analysis.py:107-132](src/bond_analysis.py#L107-L132) requires three tab fields and non-empty class sets; [metal_identification.py:23-33](src/metal_identification.py#L23-L33) accepts any single column). A legacy one-column catalog loads fine in one module and hard-fails at import in the other.
- **Import-time file I/O and import-time exceptions.** `import bond_analysis` reads two files ([:135](src/bond_analysis.py#L135), [:281](src/bond_analysis.py#L281)) and can raise `ValueError` from two places. A worker spawn, a test collection pass, and `--help` all pay this and can all die at import with no useful context.
- Meanwhile the cofactor ID set follows the *opposite* policy — explicitly loaded in [main.py:2554](src/main.py#L2554) and threaded through `cfg["cofactors"]`. Two contradictory strategies for one directory.

The import-time results are also left mutable and leak loop variables into the module namespace (verified):

```
module-namespace leaks from top-level loop: ['donor', 'metal_element', 'target', 'key']
LIT                  : dict, 79 entries      (mutable, process-wide)
FIRST_SPHERE_TARGETS : dict, 22 entries      (mutable, process-wide)
AA                   : set                    (mutable)
CLUSTER              : frozenset              (correct)
```

The file knows the right idiom — `CLUSTER`, `HEMES`, and `INFERRED_DONOR_ATOMS`' values are all correctly frozen — and applies it inconsistently.

**Fix:** one `reference_data.py` owning a single `DATA_DIR` constant and exposing `@lru_cache` accessors returning validated, immutable objects. No import-time I/O. Build `FIRST_SPHERE_TARGETS` inside a function.

### 2.5 Reference data is not verified at runtime

[src/data/metallocofactors_id.meta.json](src/data/metallocofactors_id.meta.json) records `catalog_sha256`, the CCD source, ETag, and counts — and **nothing in `src/` reads it**. Only `tools/build_metallocofactor_catalog.py --status` does. The pipeline will run happily against a hand-edited catalog, and the output CSVs carry no catalog identity.

`metal_distances_info.txt` is worse off: no checksum, no metadata sidecar, no in-band citation (provenance lives only in [README.md:466-478](README.md#L466-L478)), trailing whitespace on four lines, inconsistent precision, and a loader that silently `continue`s past unparseable rows ([bond_analysis.py:268-277](src/bond_analysis.py#L268-L277)) — so a typo in a reference distance silently disables the z-score for that donor rather than failing.

There is also a real trap in the format: column 1 `CA` means "backbone carbonyl pseudo-residue" ([bond_analysis.py:425-427](src/bond_analysis.py#L425-L427)), while column 3 `CA` means calcium. Undocumented in-band.

**Fix:** verify `catalog_sha256` at load; stamp catalog identity into `manifest.csv` alongside `alchemy_commit`; give `metal_distances_info.txt` a metadata sidecar and a strict loader; document both formats in a header the loader skips.

### 2.6 Untyped mutable dicts as the primary data structure

**The two contracts most worth typing are the cross-process ones in `main.py`,** because a pickling boundary makes a shape error remote and hard to attribute:

- **`EntryResult`** — the **35-key** dict built by [`_initial_result`](src/main.py#L1148), mutated by `process()` across every pipeline stage, pickled back from the worker, and then read by the manifest writer, the CSV writers, the confidence stage, and `_RunLog`. Its own docstring already carries load-bearing invariants (why `n_bonds` starts blank rather than zero) that a dataclass would state structurally.
- **`WorkerConfig`** — the **14-key** dict assembled at [main.py:2801-2810](src/main.py#L2801-L2810), pickled into every worker via `_init_worker`, and read through `cfg["…"]` at a dozen sites in `process()`. A frozen dataclass here is nearly free and removes a whole class of typo.

The candidate record in `bond_analysis` is a weaker case than it first appears, but still real. It starts at **13 keys** in [`_collect_proximal_candidates:775-789`](src/bond_analysis.py#L775-L789) — not the ~25 an earlier draft of this review claimed — and both branches of `_identify_first_sphere_candidates` add the *same* nine keys, so the branch asymmetry is not a defect either. What remains is genuine: the record is grown by four more functions at four later stages, nothing declares the shape, and the code compensates with defensive `.get()` calls that each invent a default:

```python
supported     = candidate.get("donor_class_supported", True)     # :638
donor_allowed = candidate.get("inferred_donor_allowed", True)    # :664
```

Two incompatible shapes do flow through the same functions: declared candidates carry `"metal"` and `"donor_class_supported"`; proximity candidates carry neither. `_merge_candidates` unions them without reconciling.

The same pattern recurs elsewhere:

- `run_density_analysis` returns a bare 14-key dict ([density_analysis.py:457-467](src/density_analysis.py#L457-L467)). `main.py` consumes it with a *mix* of required and optional access in one block — `res["density_map_scope_used"]` beside `res.get("timings", {})` — which is the maintainer signalling uncertainty about the contract.
- `_bond_summary` ([confidence_score.py:164-178](src/confidence_score.py#L164-L178)) returns 12 keys **mixing types**: `assigned_contact_count` is an `int` while `geometry_coverage` is already stringified. Callers do arithmetic on some and treat others as display strings.
- `finalize_database_confidence` returns a bare 3-tuple, unpacked positionally at two call sites.
- [metal_identification.py:338-356 and 376-393](src/metal_identification.py#L338-L393) are two near-identical 15-key `rows.append({...})` literals differing in three keys, each re-deriving the same two values above them.

`structure_analysis.py`'s dataclasses and `confidence_score.ConfidenceReference` show the author already knows how to do this well — the rest just was not brought along.

### 2.7 The output schema is maintained in three to four parallel places

`BOND_COLUMNS` (87 entries), `CANDIDATE_COLUMNS` (64), `STATS_EXTRA_COLUMNS` (86) — 237 hand-maintained strings with 53 shared between the first two and 15 shared across all three. Each is duplicated again in the row builders (`_bond_row`, `_candidate_row`, `stats_extra_values`), which repeat identical blocks verbatim (~40 duplicated key/value lines). [`_scope_summary`](src/bond_analysis.py#L1336) adds a *third* vocabulary — 14 short keys that [`_site_summary:1507-1531`](src/bond_analysis.py#L1507-L1531) hand-maps to the long CSV names one by one, with the NaN branch writing all 14 a second time.

Adding one metric requires edits in three places, and the drift check lives in `main.py` (`_check_row_schema`) — the module that *owns* the schema does not enforce it.

`stats_extra_values` compounds this by writing its result **three times** ([bond_analysis.py:1548-1601](src/bond_analysis.py#L1548-L1601)): an explicit dict, then a `setdefault` pass, then an `update` pass that silently *overrides* three of the explicit values (`strict_ncs_operation_count`, `crystallographic_operation_count`, `dpi_atom_count_multiplier`). The precedence — summary beats structure — is real behaviour that nothing states.

---

## Tier 3 — Readability, naming, consistency

### 3.1 Type hints are bimodal

| Module | Functions | Return-annotated | Any arg annotated |
|---|---:|---:|---:|
| `structure_analysis.py` | 45 | **45** | **29** |
| `main.py` | 93 | 4 | 3 |
| `bond_analysis.py` | 48 | 1 | 0 |
| `confidence_score.py` | 28 | 0 | 0 |
| `density_analysis.py` | 11 | 0 | 0 |
| `metal_identification.py` | 9 | 0 | 0 |
| `ccp4_setup.py` | 5 | 0 | 0 |

`structure_analysis.py` is written to a completely different standard from everything else. `confidence_score.py` even imports `Any, Dict` ([:20](src/confidence_score.py#L20)) and uses them for exactly two *local variable* annotations — not one signature in 706 lines. That makes the omission look accidental rather than chosen.

Note also a mixed style *within* the good module: `typing.Dict/List/Tuple` ([:21](src/structure_analysis.py#L21)) alongside builtin generics `frozenset[str]` ([:302](src/structure_analysis.py#L302)), on `requires-python = ">=3.11"`.

**Fix:** annotate public surfaces first — `run_bond_analysis`, `extract_metal_statistics`, `run_density_analysis`, `score_site`, `prepare_confidence_inputs`, `find_ccp4_setup`. ~20 signatures. Standardize on PEP 585 builtins. Then turn on `mypy`, which makes the dataclass work in 2.6 self-enforcing.

### 3.2 `pdbID` vs `pdb_id`

| Module | `pdbID` | `pdb_id` |
|---|---:|---:|
| `main.py` | 99 | 13 |
| `density_analysis.py` | 26 | 0 |
| `bond_analysis.py` | 8 | 4 |
| `metal_identification.py` | 7 | 2 |
| `confidence_score.py` | 5 | 3 |
| `structure_analysis.py` | **0** | **7** |

The convention split maps exactly onto the module split. `bond_analysis` bridges the two mid-call: `run_bond_analysis(pdbID, …)` → `_bond_row(pdb_id, …)` → `{"pdbID": pdb_id}`.

**Fix:** `pdb_id` for every Python identifier. The CSV *column* `pdbID` is an external contract — keep it, map once at the writer, and comment why.

### 3.3 Duplicated helpers with divergent semantics

- **`NAN = float("nan")`** defined twice — [structure_analysis.py:26](src/structure_analysis.py#L26) and [bond_analysis.py:105](src/bond_analysis.py#L105) — and `main.py` imports it from `bond_analysis`.
- **`_format_number`** defined twice with **different missing-value output**: [structure_analysis.py:136](src/structure_analysis.py#L136) returns `"NA"` and formats `.6g`; [confidence_score.py:122](src/confidence_score.py#L122) returns `""` and formats `.6f` with stripping. Different modules therefore represent a missing number differently in the same CSV family.
- **`load_structure` is re-exported.** `main.py` imports it from `bond_analysis` ([main.py:85](src/main.py#L85)) — a pure pass-through — while importing other names from `structure_analysis` directly ([main.py:75](src/main.py#L75)). The layering is ambiguous to a newcomer.

### 3.4 Magic values and near-miss inconsistencies

- **Two epsilons for the same boundary.** `SEARCH_EPSILON = 1e-6` ([bond_analysis.py:90](src/bond_analysis.py#L90)) governs the search radius and eligibility, but [:771](src/bond_analysis.py#L771) hardcodes `1e-9` for the same 4 Å cutoff, 48 lines later.
- **Magic strings encoding a constant's value.** [:696](src/bond_analysis.py#L696), [:698](src/bond_analysis.py#L698) emit `"distance_within_target_plus_0.75"` — change `FIRST_SPHERE_TOLERANCE` at [:95](src/bond_analysis.py#L95) and the reason codes silently lie.
- **`GRID SAMP=5`** ([density_analysis.py:332](src/density_analysis.py#L332)) is an unexplained bare number affecting map sampling, inside an inline CCP4 keyword f-string. `MODEL_ENVELOPE_BORDER_ANGSTROM` in the same file shows the right treatment.
- **`_format_number(digits=6)`** ([confidence_score.py:122](src/confidence_score.py#L122)) is the precision of every emitted number in the pipeline, as a default argument.
- **Truncation constants** `[:300]`, `[:500]` appear at seven sites unexplained.

### 3.5 Two overlapping status vocabularies, one with a space

`_scope_summary` emits `"insufficient data"` ([:1377](src/bond_analysis.py#L1377)) — the only value in the file containing a space — alongside `"suspect"` and `"plausible"`. `multi_donor_geometry_status` uses a *different* set that reuses one term: `"single_donor"`/`"consistent"`/`"suspect"`/`"indeterminate"`. Two `*_status` columns both containing `"suspect"` with different meanings.

This is part of a broader pattern: status and reason codes are a shared vocabulary with no shared definition. [structure_analysis.py:1016-1034](src/structure_analysis.py#L1016-L1034) appends nine literal warning codes whose message table lives in [main.py:139-144](src/main.py#L139-L144). Occupancy statuses are written as literals in two places and compared as strings in a third. A typo anywhere is a silent behaviour change no test or type checker catches.

**Fix:** a `codes.py` of `StrEnum`s shared by producers and consumers.

### 3.6 Long functions

| Function | Lines | Location |
|---|---:|---|
| `_run` | 534 | [main.py:2551](src/main.py#L2551) |
| `load_structure` | 258 | [structure_analysis.py:843](src/structure_analysis.py#L843) |
| `run_density_analysis` | 254 | [density_analysis.py:214](src/density_analysis.py#L214) |
| `extract_metal_statistics` | 198 | [metal_identification.py:220](src/metal_identification.py#L220) |
| `process` | 180 | [main.py:1431](src/main.py#L1431) |
| `_collect_declared_candidates` | 159 | [bond_analysis.py:982](src/bond_analysis.py#L982) |
| `_RunLog._render` | 157 | [main.py:2365](src/main.py#L2365) |
| `run_bond_analysis` | 144 | [bond_analysis.py:1772](src/bond_analysis.py#L1772) |
| `normalize_refmac_twin_coefficients` | 134 | [density_analysis.py:78](src/density_analysis.py#L78) |
| `_site_summary` | 114 (9 params, no docstring) | [bond_analysis.py:1428](src/bond_analysis.py#L1428) |

`run_density_analysis` additionally contains five closures. Two of them — `_map_size` and `_map_extent_requires_full_map` — close over nothing but `pdbID` (used only in error text) and are trivially liftable to module level, where they become unit-testable. The latter is 35 lines of binary CCP4-header `struct.unpack` parsing buried four levels deep inside an orchestration function.

### 3.7 Long parameter lists

`_site_summary` (9), `_bond_row` (8), `_OutputWriters.__init__` (8), `run_density_analysis` (10), `run_bond_analysis` (7), `find_ccp4_setup` (5 on a 118-character line in a file that otherwise wraps at 79).

Several parameters are also redundantly derived: `_bonding_key(neighbor, nb_res, metal_el)` — both call sites pass `neighbor.residue_name`, derivable from `neighbor`; `_parent_type(structure, metal, metal_res, metal_el)` — its sole call site passes `metal.residue_name, metal.element`. Four parameters where two suffice.

### 3.8 Function-local `import gemmi`

Eight occurrences in `main.py`, five in `bond_analysis.py`, while `structure_analysis.py` and `density_analysis.py` import it at module level. Since both files unconditionally import `structure_analysis`, gemmi is already resident — the deferred imports buy nothing and hide the dependency. Similarly, `import argparse` is top-level in `confidence_score.py` but function-local in `density_analysis.py`.

### 3.9 Broad exception handling, unevenly disciplined

`main.py` has 10 broad `except` clauses and annotates nearly all of them with `# noqa: BLE001` plus a rationale — good practice. `bond_analysis.py` has 6 with none. The worst is [bond_analysis.py:408](src/bond_analysis.py#L408):

```python
except Exception:
    return NAN, resolution, "dpi_calculation_failed"
```

A `TypeError` or `AttributeError` introduced by a refactor becomes a silent `"dpi_calculation_failed"` in a CSV column rather than a crash. Similarly, [ccp4_setup.py:75-76](src/ccp4_setup.py#L75-L76) swallows every `OSError`/`JSONDecodeError` with a bare `continue` — so a corrupt config file produces zero diagnostics, which is precisely the failure mode ("configuration appeared to succeed and was then silently ignored") the function's own docstring says it was written to fix.

### 3.10 Inconsistent error signalling in the driver

Three mechanisms coexist:

| Mechanism | Example | Exit code |
|---|---|---|
| `raise SystemExit(msg)` | [main.py:2570](src/main.py#L2570), [:2019](src/main.py#L2019) | 1 |
| `return 1` after `print` | [main.py:2559](src/main.py#L2559), [:2620](src/main.py#L2620), [:2644](src/main.py#L2644) … | 1 |
| `_DriverError` | [main.py:2027](src/main.py#L2027), caught at [:2677](src/main.py#L2677) | 1 |
| `ap.error(...)` | [main.py:2021](src/main.py#L2021), [:2023](src/main.py#L2023) | **2** |

The last is a real user-visible inconsistency: `--retry-partials` without `--resume` exits 2, while `--id` with `--id-file` exits 1, for the same class of mistake in the same function.

**Fix:** one `DriverError`, one handler, one exit-code policy. Use `ap.error` for *all* argument validation (exit 2 is the conventional argparse behaviour) or `SystemExit` for all of it — not both.

### 3.11 Smaller naming notes

- `ap` for the argument parser ([main.py:1957](src/main.py#L1957)) — vague.
- `LIT` ([bond_analysis.py:281](src/bond_analysis.py#L281)) — uppercase implies constant, but it is a mutable dict; the abbreviation is opaque.
- `complete_confidence_site_count` ([confidence_score.py:307](src/confidence_score.py#L307)) counts nothing; it pads a row list. `pad_confidence_rows_to_site_count` says what it does.
- `severity` ([confidence_score.py:342](src/confidence_score.py#L342)) is a very generic name for a module-level export.
- `confidence_score.main` in a package that also has a `main.py` module — `cli_main` disambiguates.
- No `__all__` anywhere. `bond_analysis` exposes 38 public names, of which `main.py` uses 7, and 15 underscore-private functions are reached into directly by tests (`_analysis_atom_for_partner` 10 refs, `_zscore` 9, `_asu_volume` 9, `_calculate_dpi_details` 7 …). These are live code, not test-only cruft — the fact that they each need direct unit tests is the clearest evidence they want to be *public* API of the separate modules proposed in 2.3.
- `structure=None` ([metal_identification.py:220](src/metal_identification.py#L220)) is an optional parameter that immediately raises when omitted. Make it required.
- `find_ccp4_setup` returns `None` for two different meanings — "tools already on PATH, no setup needed" and "nothing found". [main.py:346](src/main.py#L346) treats it as fatal; [tests/helpers.py:838-841](tests/helpers.py#L838-L841) treats it as a soft skip.
- Falsy-vs-`None` defaults block injection: `env or os.environ.copy()`, `config or load_ccp4_setup_config(...)`, `config_files or DEFAULT_CONFIG_FILES` ([ccp4_setup.py:61, 81, 100, 101](src/ccp4_setup.py#L61)). An intentionally empty `{}` or `[]` silently gets the global default, so "no config" and "empty environment" cannot be injected in a test.

### 3.12 Dead code

- `_write_csv` ([confidence_score.py:136](src/confidence_score.py#L136)) — defined, never called anywhere in `src/`, `tests/`, or `tools/`.
- `find_ccp4_setup`'s `common_candidates` parameter — never passed by any caller.
- `COMMON_METALS` / `UNCOMMON_METALS` ([metal_elements.py:4-16](src/metal_elements.py#L4-L16)) — used only to build `METAL_ELEMENTS` on the next line. No consumer anywhere, no defined criterion for the split, no provenance for the 88 symbols, no test.
- `_bonded_to`'s `is_water=False` default ([bond_analysis.py:445](src/bond_analysis.py#L445)) — the one call site always supplies it. The function emits `"P"`, an undocumented abbreviation, into a column nothing in `src/` reads.
- Defensive branches that cannot fire: `getattr(structure, "residues_for_coordinate_author", structure.residues_for_author)` ([metal_identification.py:300-302](src/metal_identification.py#L300-L302)) duck-types against a class in the same repo that always has the method — and does so *inside a per-row loop*, evaluating the fallback eagerly every iteration. `if indices is None:  # defensive` ([:278-279](src/metal_identification.py#L278-L279)) is unreachable given line 272.

### 3.13 Documentation style

- `density_analysis.py` and `metal_identification.py` open with `#` comment banners rather than module docstrings, so `__doc__` is `None` and the content is invisible to `help()` and doc tooling. `structure_analysis.py` and `confidence_score.py` do it correctly.
- `metal_identification.py` opens with a version-history label, `"Analysis v2:"`, and `extract_metal_statistics`'s docstring narrates history ("as before").
- Stale docstring: `_calculate_dpi_details` ([bond_analysis.py:353](src/bond_analysis.py#L353)) says it returns a 3-tuple in the summary line and a 2-tuple six lines later.
- The `bond_analysis` module docstring names its caller ("`main.py` calls `run_bond_analysis`…") — useful today, a lie the day a second caller appears.
- Twelve functions lack docstrings, including the two largest undocumented ones, `_site_summary` (114 lines) and `_bond_row` (97).

### 3.14 Debug entry point writes into the source tree

`BASE_DIR` ([density_analysis.py:26](src/density_analysis.py#L26)) exists solely as the `--out-dir` default for the module's self-described "manual testing" CLI ([:470-491](src/density_analysis.py#L470-L491)) — and its value is `src/`, so that CLI writes `.map` files into the source directory by default. It is untested, undocumented in the README, and re-declares defaults that can drift from `main.py`'s. Promote it or delete it.

*(By contrast, the `confidence_score.py` CLI is **not** redundant: `main.py` exposes no confidence subcommands, the recovery path is documented at [README.md:401-405](README.md#L401-L405), and it is exercised at [tests/test_pipeline_integration.py:1345](tests/test_pipeline_integration.py#L1345). Keep it.)*

---

## Tier 4 — Test suite

The suite is a genuine asset and the best-maintained part of the repository: **853 passed, 26 skipped in 24.5 seconds** offline — 478 test functions, 879 collected items, 1,844 assertions, 468 of 478 tests carrying docstrings. 97% of it is meaningful on a bare checkout with no CCP4 and no network, which is rare for a crystallography pipeline with a hard binary dependency and is clearly deliberate. See "What is genuinely good" below for specifics worth protecting.

The gaps are coverage-shaped, not quality-shaped.

### 4.1 `density_analysis.py` is the thinnest-covered module, and its *default* path has no offline coverage

491 source lines against **170 test lines / 6 tests** — a 0.35:1 ratio in a suite that runs 1.0–2.0:1 everywhere else. Five of the six tests concern the twin-normalization feature added last week in `cb6738a`.

`run_density_analysis` is 254 lines. The **default** map scope is `model-envelope` ([main.py:1979-1982](src/main.py#L1979-L1982)), including mapmask cropping and the documented "falls back to full when cropping would be unsafe or larger" rule. Every `map_scope=` in the unit tests is `"full"` ([test_density_analysis.py:141, 146, 170](tests/test_density_analysis.py#L141)). The only `model-envelope` exercise is [test_pipeline_integration.py:1031](tests/test_pipeline_integration.py#L1031), which is `ccp4`+`slow` and therefore never runs in CI (1.7). `keep_full_maps` has no test at all.

**Fix:** `_fake_ccp4_run_factory` ([test_density_analysis.py:97-114](tests/test_density_analysis.py#L97-L114)) already stubs `mtzfix`/`fft`/`edstats` convincingly. Extend it to `mapmask` and drive `model-envelope` through the crop-succeeds, crop-unsafe, and crop-larger-than-full branches offline. Roughly 80 lines closes the largest single coverage hole in the repository.

### 4.2 A band of user-facing `main.py` input handling is untested by name

Zero references *by name* anywhere under `tests/` for: `parse_pdb_id`, `load_ids_from_file`, `enumerate_entries`, `resolve_manual_inputs`, `prepare_inputs`, `validate_resume_schemas`, `_batch_exit_code`, `automatic_worker_limits`, `available_cpu_count`, `available_memory_bytes`, `verify_ccp4`, `remove_stale_disabled_bond_outputs`, `read_resolution`, `has_final_files`. The `--keep-intermediates` flag has **0 references** in the entire test tree.

**These are not all untested.** Roughly eight of the fourteen do execute indirectly in the offline lane, reached through `main.main()` and the CLI-level tests, so this is a *directness* gap rather than a coverage void — a regression in them surfaces as a confusing failure somewhere downstream, if it surfaces at all. The rest have no offline exercise at all.

They are also the wrong set to leave indirect: argument parsers, resume-safety validators, exit-code contracts and worker autoscaling are exactly the surfaces that decide whether a batch run silently does the wrong thing. `validate_resume_schemas` and `_batch_exit_code` deserve direct unit tests most — both are close to pure, and both govern data-destructive behaviour.

### 4.3 The defect ledger is empty, but the defect it names is still real

The file is 52 lines of comments with no executable content — `pytest` collects **0 tests** from it. Its docstring states that one open finding "is pinned elsewhere: corrupt confidence coverage, in `test_confidence_score.py`." **There is no `xfail` marker anywhere in `tests/`** — the only matches for the string are this file's own prose and [tests/README.md:68](tests/README.md#L68). [tests/README.md:26](tests/README.md#L26) additionally instructs users to run `pytest tests/test_known_limitations.py -rxs`, which collects nothing.

**Do not resolve this by deleting the entry.** The obvious reading — that `f62af87` fixed the defect and the ledger simply went stale — is wrong. That commit fixed the *scoring* path only; the *reference-building* path still coerces invalid coverage and still contaminates the frozen cohort (1.8, reproduced there). The ledger is not describing a closed finding; it is describing an open one that lost its pin.

**Fix.** The finding this ledger names is now closed: 1.8 was fixed in `36e5dbf` and is covered by five parametrised regression tests in `test_confidence_score.py`, which is exactly the lifecycle [test_known_limitations.py:11](tests/test_known_limitations.py#L11) prescribes — pin it, fix it, move the test to the module that owns it. What remains is bookkeeping: correct the two stale sentences in `tests/README.md`, and either delete `test_known_limitations.py` or reduce it to an explicitly empty protocol stub, since it currently documents an `xfail` mechanism with no instances.

The wider lesson is that a ledger whose contents cannot be executed provides no protection: nothing failed when the pin disappeared, and the finding survived a release as a comment.

### 4.4 `tools/build_metallocofactor_catalog.py` has no tests at all

No reference to it anywhere under `tests/`, despite it producing the data that gates cofactor classification for every run. `classify_component`, `element_counts`, and `_biconnected_components` are pure functions, trivially testable with a fixture CIF block. Its only guard, `_verify_canonical_classes`, runs solely during a network rebuild. `_biconnected_components` also uses unbounded recursion — fine for CCD-sized components, but worth a note or an iterative form.

### 4.5 Smaller test-suite notes

- **File names don't name their subject for the two largest modules.** `bond_analysis.py` is covered by `test_bond_geometry.py` + `test_declared_connections.py` + `test_symmetry_provenance.py`; `main.py` by `test_driver_manifest.py` + `test_worker_recovery.py` + `test_cli_and_config.py` + `test_pipeline_integration.py`. The split is genuinely principled — every module docstring opens with a Scope paragraph, and several state what is "out of scope here (owned elsewhere)" — but nothing in the filenames says so and there is no index, while four other modules *are* 1:1 named. The convention is half-applied. Two concrete misplacements: [test_pipeline_integration.py:1178](tests/test_pipeline_integration.py#L1178) is a pure argparse test whose exact sibling lives in `test_cli_and_config.py`, and [test_pipeline_integration.py:1399](tests/test_pipeline_integration.py#L1399) is a confidence-policy invariant that belongs with `test_confidence_score.py`.
- **Two structural conventions coexist.** `test_driver_manifest.py` is the only file using test classes (13 of them, 97 of its 99 tests inside). The other 11 files use `# ---` banner comments. Both work; the class form gives failure output a namespace. Pick one.
- **Conventions are eroding at the newest edge.** `test_density_analysis.py` (the newest file) has **0/6 docstrings** where every other file is at or near 100%, no `from __future__ import annotations`, no section banners, and tuple-form `parametrize` argnames where the rest of the suite uses the string form.
- **`helpers.py` is two modules.** Lines 37–811 are synthetic-input builders (cohesive, well-documented, honest `__all__`). Lines 814–899 are environment capability probes — a different responsibility, and the only reason [conftest.py:33](tests/conftest.py#L33) imports helpers at all. Splitting the probes into `tests/capabilities.py` would let conftest do its collection-time work without importing the entire gemmi structure-building stack.
- **The bond-analysis harness is duplicated three times** — `_analyze` in `test_bond_geometry.py:124` and `test_symmetry_provenance.py:118`, `analyze` in `test_declared_connections.py:98` — each re-deriving the same `load_structure` → `dpi_inputs` → `run_bond_analysis` call with different knobs. A signature change requires three coordinated edits.
- **`slow` carries no independent selection power.** `ccp4` selects 25 tests, `slow` 24, `ccp4 and not slow` exactly 1. The two markers are effectively the same set. Meanwhile `test_worker_recovery.py` spends ~19 s of the 24.5 s offline run (including a `time.sleep(30)` at line 917) and carries no marker at all. Either drop `slow` or apply it to the genuinely slow offline tests.
- **Warnings are not gated.** `filterwarnings = ["error::pytest.PytestUnknownMarkWarning"]` is redundant — `--strict-markers`, already in `addopts`, errors on unknown marks first. There is no `"error"` default, so gemmi and numpy `DeprecationWarning`s pass silently. For a suite pinned to `gemmi>=0.7.0` that is exactly the early-warning channel you want.
- **The declared dependency floor is never exercised.** gemmi always resolves to latest in CI, so the `gemmi>=0.7.0` floor — enforced by 35 lines of careful machinery at [conftest.py:36-71](tests/conftest.py#L36-L71) — is never actually tested. The matrix is also 3.11/3.12 against `requires-python = ">=3.11"`, leaving 3.13/3.14 untested.
- **134 message-less asserts inside loops** (43 in `test_bond_geometry.py`, 38 in `test_pipeline_integration.py`, 26 in `test_confidence_score.py`). pytest's rewrite shows the value but not which loop item produced it; appending the loop variable costs nothing and halves diagnosis time.
- **`*.egg-info/` is not in `.gitignore`**, so the documented local workflow ([tests/README.md:9](tests/README.md#L9)) leaves build metadata inside the checkout — contradicting the suite's own release-critical promise asserted at [test_smoke.py:76-101](tests/test_smoke.py#L76-L101) that "a test run puts nothing inside the checkout."
- **No coverage measurement** anywhere (see 1.5), and no pip cache or lint step in CI.

---

## Docs and repository hygiene

- **README** is excellent in content — precise, scientifically careful, honest about limitations — but it is 554 lines mixing a user guide, method documentation, an operations runbook, and maintenance procedures. Split into `docs/` (usage, method, operations, maintenance) and keep the README as an overview plus quick start. Note that it instructs users to `pip install "gemmi>=0.7.0"` manually rather than `pip install .`, which is an honest workaround for 1.1.
- **Run logs are written into `--output-dir` alongside the result CSVs** (`output/alchemy_run_20260731_2.log`, 3.6 MB in one case). A `logs/` subdirectory or a separate `--log-dir` keeps machine-readable results separate from diagnostics.
- **`tests/README.md` contains two stale claims** (4.3) and should be corrected alongside the ledger.

---

## What is genuinely good

Worth protecting through any refactor:

**Comments state constraints and foreclose wrong "fixes."** This is the highest-value kind of comment and it is everywhere. `_zscore` explicitly tells a future maintainer *not* to change the denominator to `2 * dpi ** 2` and says what breaks if they do. `_is_placeholder_cell` explains why the check is narrow and refuses to widen it into "a cutoff nobody derived." `_collect_declared_candidates` explains why atom serials cannot carry the partner join, because Gemmi's PDB writer emits TER records that consume serial numbers — a hard-won fact that would otherwise be rediscovered by a bug. `confidence_score.py:526-530` explains why invalid coverage must not coerce to zero, including the specific perverse outcome it caused. `main.py:315-322` explains why `--ccp4-setup` beats ambient `PATH`. These read like they were each written after a real incident.

**Constants are named, grouped, and cited.** `FIRST_SPHERE_TOLERANCE` cites Harding 2004 with a DOI. `SPECIAL_POSITION_DEDUP_CUTOFF` notes that it matches Gemmi's own `ContactSearch` default. The `CUTOFF` comment pre-empts the most likely misreading: "This must not be treated as a bond cutoff."

**Failure semantics are deliberate and consistent.** `MtzfixValidationError` carries `timings` so partial cost is recorded even on failure. `_calculate_dpi_details` is declared "Never raises" and returns a machine-readable reason code rather than a bare NaN, and those codes propagate into `partial_reason_codes` instead of vanishing. The retryable/terminal distinction is carefully maintained — `n_bonds` starting blank rather than zero, so a `--resume` cannot mistake "never ran" for "ran and found nothing," is exactly the kind of detail that separates a tool that survives a database-scale run from one that does not.

**Crash and interrupt handling is unusually thorough.** `_install_termination_handler`, `_shutdown_pool`'s bounded terminate with a documented rationale for why the lock cannot be released, `_sweep_leaked_work_dirs`, the in-flight queue protocol for attributing OOM-killed workers to their entry, and staged resume replacement that leaves prior rows intact on a failed retry. Each carries a docstring explaining the specific failure it was written for.

**Atomic writes are correct and consistent.** `.tmp` + `os.replace` throughout `confidence_score.py` and `tools/`, with cleanup on exception. Deleting `metadata.json` *first* in `finalize_database_confidence`, so a crashed finalization cannot leave a stale reference looking current, is exactly right — and the comment says why.

**Determinism is designed in, not hoped for.** `_contact_sort_key`, `_special_position_preference`, and the `sorted(by_source)` in `_deduplicate_special_position_contacts`, with a docstring stating that the sort makes the result independent of Gemmi's neighbour-search order.

**`structure_analysis.py` is the model to converge on.** Frozen typed dataclasses with derived `@property` join keys, full annotations, a clear module docstring explaining the `source_atoms`/`contact_atoms` split, and the right instinct — keeping conformer selection explicit rather than trusting a parser's implicit altloc pick. `confidence_score.ConfidenceReference` is the same quality: validates length, positivity, and strict monotonicity up front, precomputes the cumulative array once, and implements average-rank ECDF cleanly via `bisect`.

**`tools/` separation is right.** The catalog builder is not importable from the pipeline, does its network work explicitly, writes atomically, emits a checksum-bearing metadata sidecar, and asserts that canonical cofactors survive a rule change. Deriving cluster/heme classes from CCD connectivity rather than a hand-maintained list in the analysis code is the correct call, and it is explained where it is made.

**The candidate/bond conceptual split is real and correctly held.** The `CANDIDATE_COLUMNS` header comment — that failing first-sphere eligibility "does not establish that an atom is chemically nonbonded" — and the deliberate absence of z-score columns from the candidate schema is exactly the right modelling for a validation tool.

**The clone-and-run mechanics are right, which validates not making this a package.** A fresh `git clone` is 33 tracked files and 1.1 MB, and `python src/main.py --help` works immediately with no build, no install, and no path configuration — verified against a clean clone. Paths are `__file__`-anchored rather than cwd-relative, so the pipeline runs from any working directory. The bundled reference data travels with the checkout by construction, and `_alchemy_commit()` records the short hash plus a `+dirty` flag, so under this distribution model the git commit *is* the version and every manifest row carries it. That is a coherent design for a scientific tool people obtain by cloning; 1.9 is a documentation gap sitting on top of working machinery, not a structural problem.

### In the test suite specifically

**Test naming is the best part of the suite.** 478 tests, average name length 53 characters, essentially all behavioural rather than implementational: `test_dpi_never_widens_the_chemical_cutoff`, `test_negative_max_pdbs_does_not_silently_drop_entries_from_the_end`, `test_geometry_coverage_with_no_assigned_contacts_is_zero_not_an_error`, `test_worker_death_reason_codes_discriminate_synthesized_from_real`. Names state the invariant being defended, not the function being called, so a failure line is readable without opening the file.

**Regression tests spell out the original wrong behaviour.** 468 of 478 tests have docstrings, the good ones cite their source of truth (a README sentence or a source anchor), and regressions are labelled `Regression:` with the defect they prevent described explicitly. This is the single most valuable documentation in the repository.

**`conftest.py`'s capability handling is unusually careful.** `pytest_collection_modifyitems` is `@pytest.hookimpl(trylast=True)` specifically so `-m`/`-k` deselection happens *before* any probe fires; probes run at most once per session and only when a surviving item carries the marker. Better still, there is a test that *proves* it — [test_smoke.py:621-669](tests/test_smoke.py#L621-L669) spawns a nested pytest with `helpers.network_available` replaced by a raiser and asserts collection still succeeds. A real invariant, tested at the right level.

**Oracle independence is thought about.** `EDSTATS_HEADER` ([helpers.py:670-679](tests/helpers.py#L670-L679)) is transcribed independently of the production constant, with a comment explaining that sharing it would let one reorder silently update both the input and its expected schema. The same instinct appears in the restated scoring anchors and the donor table transcribed from the README.

**Magic literals are anchored rather than floating.** The reference-table row count is pinned *with an explanation of why a dropped row would be invisible at runtime*; the 13 measured RSZD triples are documented as the density arm's only genuine oracle with the 0.3 tolerance justified quantitatively against EDSTATS' one-decimal printing.

**Test size and fixture discipline are healthy.** Median 15 lines, p90 31, max 84, 3.9 asserts per test, zero tests over 100 lines. Only 6 conftest fixtures and 6 module-local ones for 478 tests, correctly scoped throughout — session for the expensive caches, function for `work_dir` (which uses `monkeypatch.chdir`, so cwd is restored even on failure). `entry_cache` correctly distinguishes "capability missing" from "code broken": a dead socket skips, but a download that returns without the files raises. That distinction is where most integration suites go soft, and this one does not.

---

## Suggested sequence

This is the roadmap. The commit-level breakdown that follows is what you actually work from.

| # | Change | Ref | Effort | Unblocks |
|---|---|---|---|---|
| 1 | Make setup honest: correct the README's dependency list, drop the personal conda env from the examples, document confidence scoring as unreleased, declare an empty distribution, drop the redundant `sys.path` block, one shared `DATA_DIR` | 1.1, 1.9, 4.5 | S | first run |
| 2 | `LICENSE`; `ruff` + `mypy` + `pytest-cov` in `pyproject.toml` and CI | 1.5 | S | drift control |
| 3 | One definition of the version | 1.6 | S | — |
| 4 | Separate timeout budgets for CCP4 tools, setup shells and provenance commands | 1.3 | S | — |
| 5 | ~~CCP4-free `entry_data` integrity tests~~ — **dropped**; see 1.7. Kept in place so later step numbers stay stable. | 1.7 | — | — |
| 6 | `logging` per module; `print` only for progress and final summary | 1.4 | M | ops |
| 7 | Offline `model-envelope` coverage via an extended `_fake_ccp4_run_factory` | 4.1 | M | 13 |
| 8 | Direct unit tests for `validate_resume_schemas`, `_batch_exit_code`, and the indirectly-covered CLI surfaces | 4.2 | M | 13 |
| 9 | `pdb_id` rename; annotate public surfaces; delete dead code | 3.1, 3.2, 3.12 | M | 13, 14 |
| 10 | Merge CCP4 env handling into one module; one `DEFAULT_CONFIG_FILES`; `Ccp4SetupError` | 2.2 | M | 13 |
| 11 | Split `main.py` along its own four banners; decompose `_run` into phase functions | 2.1 | L | 15 |
| 12 | Extract `dpi.py`, `declared_connections.py`, `reference_data.py`, `bond_schema.py` from `bond_analysis.py` (preserving the two-phase partner check) | 2.3, 2.4 | L | 15 |
| 13 | `WorkerConfig` and `EntryResult` dataclasses first, then the density result and candidate record; `codes.py` of `StrEnum`s | 2.6, 3.5 | L | typing |
| 14 | Tests for `tools/build_metallocofactor_catalog.py`; verify `catalog_sha256` at load and stamp it into the manifest | 2.5, 4.4 | M | reproducibility |
| 15 | Split README into `docs/`; move run logs out of `--output-dir`; correct `tests/README.md` | Docs | S | — |

**No CI safety net.** The CCP4 lane is knowingly skipped (1.7), so every step below is defended by the offline suite alone — it never executes the real density or bond path. That made steps 7 and 8 prerequisites rather than improvements; both have now landed, taking `density_analysis` to 83% and `main.py` to 70%. The restructuring steps still carry more risk than they otherwise would: step 4 modified CCP4 subprocess handling, and steps 11–13 move the density and bond code, none of which CI exercises end to end.

**Two caveats.** The Effort column is a guess, not an estimate — step 1 was written as "M" before it was measured and turned out to be a config file. And step 2 touches the `sys.path` wiring the test harness itself depends on, so it changes the tests and the code together — which is why it is split into three small commits below.

**Nothing here requires a database re-run, and nothing here touches confidence scoring.** The two confidence findings (1.2, 1.8) are deferred until the score is finalized; the only confidence-related work in the plan is the documentation fix in commit 1. Your existing outputs, reference and `reference_id` are therefore untouched by every commit below.

Step 14 separately adds a manifest column, which makes the existing output directory unresumable, so defer it until you are planning a run anyway.

**Nothing here makes Alchemy an installable package.**

---

## Commit-level plan

Each commit is a coherent unit of work, sized to be reviewed in one sitting — roughly 100–700 lines, with related tasks grouped rather than split. They are individually reviewable and leave both test lanes green, but they are *not* all independently revertible once downstream commits land; see the Rhythm note at the end. Messages follow the repo's existing style — imperative mood, describing the behaviour change rather than the mechanism.

*Numbering note: commits are 1–26 and are unrelated to the roadmap's steps 1–15. Roadmap step 5 was dropped but kept in place, so its number is retired rather than reused.*

## Progress

**Phases A–D are complete, and Phase E is under way** — the production-readiness gate, the coverage work the restructuring depends on, the mechanical cleanups, and the first three `main.py` extractions. Thirteen of 26 commits.

| Commit | Landed as | Notes |
|---|---|---|
| 1 · `Fix the setup documentation` | `02d3261` | numpy added to README and `main.py`; `conda run -n metal` removed from all six lines; confidence documented as unreleased and its runtime message reworded. Five tests, each verified to fail on the original defect. |
| 2 · `Make the repository configuration honest` | `10009fa` | `packages = []` / `py-modules = []`; inert `package-data` table deleted; `*.egg-info/` and `build/` ignored; redundant `sys.path` block removed; `DATA_DIR` single-sourced. Two tests assert the built wheel carries only `alchemy-*.dist-info/` members. |
| — · 1.8 coverage defect | `36e5dbf` | Out of plan by exception; see 1.8. Production reference re-finalized to `…0fcf30bc2ff40fba1551`, distribution byte-identical. |
| 3 · `Add project governance files and gates` | `878f334` | `ruff` (E/W/F/B at line-length 88) and `pytest-cov` with a 75% floor, both enforced in CI; version single-sourced to `src/_version.py` via `dynamic = ["version"]`. **The licence is still outstanding** — the only part of commit 3 not done. |
| 4 · `Apply ruff format` | `9e64b1f` | Whole tree formatted, 24 files. Verified semantics-preserving by comparing the AST of every file before and after. Ruff pinned to `0.16.1` in CI, with `ruff format --check` enforcing it. |
| 5 · `Bound every external process with a timeout` | `7888153` | Three budgets: CCP4 900 s (per program, `--ccp4-timeout`), setup shell 30 s, provenance 1 s — each with its own outcome. A stalled program becomes a retryable `ccp4_tool_timeout`, its partial log copied to `<output-dir>/ccp4_timeout_logs/`. |
| — · `Annotate confidence scoring return values` | `bbad75f` | Landed outside the plan. |
| 6 · `Route diagnostics through logging` | `6bda639` | `src/run_logging.py`; 35 of 46 prints converted, the rest being the progress line, final summary and interrupt message. Workers log over a queue the driver re-emits. `-v` / `--quiet` / `--log-file`. Scattered `[:300]` slices replaced by two named bounds. |
| 7 · `Cover the model-envelope map path offline` | `e5d1a6c` | The default map scope, previously exercised only in the `ccp4`+`slow` lane. All three fallback outcomes plus the 63/64 boundary, verified to catch a `>=` → `>` weakening. `density_analysis` 60% → 83%. |
| 8 · `Unit-test the highest-risk CLI and resume surfaces` | `de8a36d` | 59 tests in `tests/test_driver_surfaces.py` covering resume-schema validation, the exit-code contract, entry selection, input preparation including the legacy PDB fallback, `read_resolution`, and worker autoscaling. Eleven mutations confirmed caught. `main.py` 66% → 70%. |
| 9 · `Rename pdbID to pdb_id throughout` | `337310f` | Identifiers only, across five files; the `pdbID` CSV columns are untouched, so no output schema moved. |
| 10 · `Remove dead code and annotate the public surfaces` | `81ddd05` | `_write_csv`, `COMMON_METALS`/`UNCOMMON_METALS` and `find_ccp4_setup`'s `common_candidates` deleted; the unreachable `metal_identification` branches removed by making the header/index pair one `Optional` state; PEP 585 annotations on the public functions. |
| 11 · `Move CCP4 environment resolution into ccp4_setup` | `52732ab` | `resolve_env`, its two Windows helpers, `_normalize_path_key` and `verify_ccp4` moved; one `DEFAULT_CONFIG_FILES` the driver no longer shadows; `ccp4_tools_available` and `verify_ccp4` reduced to one `missing_ccp4_tools` probe; `Ccp4SetupError` replaces `SystemExit` throughout the library, leaving `resolve_ccp4_environment` the only exit. Three mutations confirmed caught. |
| 12 · `Separate input preparation and coordinate conversion` | `55c7a1a` | 24 functions and 3 constants out of `main.py`, verified AST-identical to the originals: `coordinate_conversion.py` (mmCIF→legacy-PDB conversion and the residue-identity remarks that make it reversible) and `inputs.py` (mirror/cache resolution, decompression, resolution metadata). `main.py` 1,580 → 1,193 statements. Named `coordinate_conversion.py` rather than the plan's `provenance.py`: the repo already uses "provenance" for run, coordinate-format and `image_provenance` lineage, none of which this module holds. |
| 13 · `Extract the worker entry point` | *pending* | 13 functions, 7 constants and the two per-worker globals out of `main.py`, verified AST-identical to the originals except one token: `_initial_result` now stamps `_version.__version__` directly rather than reaching for the driver's `ALCHEMY_VERSION`. `worker.py` holds the whole per-entry pipeline plus the worker half of the liveness protocol; the driver keeps its half (`_drain_inflight`, `_dead_worker_pids`, `_shutdown_pool`) and the run-provenance stampers, which run once before the pool exists. `main.py` 1,193 → 991 statements. Four mutations tried, three confirmed caught; the fourth is noted below. |

Suite: **991 passed, 26 skipped** offline, coverage **84.3%** (853 and untracked at review time). Lint, format and coverage are gates in CI.

**Two decisions resolved since the review.** 1.8 was fixed rather than gated, because full-database runs finalize confidence automatically (see 1.8). The CCP4 integration lane is knowingly skipped for want of an always-on runner (see 1.7), which is why Phase C is a prerequisite rather than an improvement: nothing in CI exercises the density or bond path that Phases E–G restructure.

**Still open:** the licence, and whether editing `metal_distances_info.txt` should invalidate frozen references (2.5).

Next up is the rest of Phase E: commits 14–16 take the driver's output machinery, resume handling and CLI.

**Two things commit 13 surfaced rather than caused.** The mutation that makes `_coordinate_provenance` always report a conversion survives the offline suite: the mirror branch is exercised only by the entry-data lane. And the entry-data lane itself had been erroring at fixture setup since commit 12, which left `tests/conftest`-adjacent `main.download_entry_to_cache` pointing at a name `main` no longer re-exports; commit 13 repairs the reference. With the lane running again, seven of its end-to-end tests fail on stdout assertions for messages commit 6 moved from `print` to `logger` — identical failures on `HEAD` before this commit, so they are a separate repair.

---

**There is no CI safety net.** The CCP4 lane is knowingly skipped (1.7), so every commit below is defended by the offline suite alone. Commits 7 and 8 close the two coverage holes that sit directly under the code Phases E–G restructure; they are prerequisites, not improvements. And the 25 end-to-end tests should be run by hand before any significant merge — they are the only check that exercises real maps.* Where a phase replaces a roadmap step, the heading says so.*

### Phase A — Make setup honest

**On confidence scoring.** The score is not finalized, so the plan does not tune it. Two exceptions were made deliberately: the correctness defect 1.8 was fixed (`36e5dbf`) because full-database runs finalize confidence automatically, so deferring it would have meant knowingly rebuilding the reference through a defective path; and commit 1 documents the feature as unreleased, which supports the decision rather than pre-empting it. **1.2 remains deferred** — duplicated scoring weights only bite when a weight is edited, and that edit belongs to the finalization pass.

Commit 1 is the release blocker: it is what every new user hits before anything else. Both commits in this phase have landed — see Progress above.

| # | Commit | Contents | ~Size |
|---|---|---|---|
| 1 | `Fix the setup documentation` | Replace the hand-listed dependencies at README:82/:87 with `python -m pip install .`, which after commit 2 installs dependencies and no modules — so the README stops duplicating a list that can drift. Replace `conda run -n metal python src/main.py` with plain `python src/main.py` in the four README examples and the two in `main.py`'s docstring. State that no confidence reference ships and the score is unreleased, and reword the "No frozen confidence reference found" message so it reads as expected behaviour (1.9). **The regression test must exercise the documented command in a clean environment** — importing `main` with the *declared* dependencies would pass even today, since numpy is already in `pyproject.toml`; it is the README that is wrong. | 190 |
| 2 | `Make the repository configuration honest` | `packages = []` / `py-modules = []` with a comment saying why; **delete the now-inert `[tool.setuptools.package-data] data = [...]` table** ([pyproject.toml:21-25](pyproject.toml#L21-L25)), which becomes misleading once nothing is installed, and verify the built wheel contains metadata and dependencies but no source modules or catalogs; `*.egg-info/` and `build/` in `.gitignore`; delete `helpers.py`'s redundant `sys.path` block (a single shared insertion point is impossible — `tools/` runs without pytest); unify `metal_identification.COFACTOR_CATALOG_PATH` and `bond_analysis.DATA_DIR`, leaving `density_analysis.BASE_DIR` alone since it is the debug CLI's `--out-dir` default, not a data path. | 110 |

### Phase B — Tier 1 hygiene

| # | Commit | Contents | ~Size |
|---|---|---|---|
| 3 | `Add project governance files and gates` | `LICENSE` plus the `license` field; `ruff` + `mypy` (strict on `structure_analysis`, permissive elsewhere) + `pytest-cov` with a CI floor; a `src/_version.py` read by `pyproject.toml` via `dynamic = ["version"]`. Note `version` cannot simply be dropped — `[project]` requires it — and `importlib.metadata` is not a substitute, since a bare clone never installs. The `ruff format` pass lands as its own follow-up commit immediately after this one — kept separate so a large whitespace-only diff never hides a logic change. | 150 |
| 4 | `Apply ruff format` | Whitespace and formatting only, no logic. Separate so a large mechanical diff never hides a real change. Also clears the 12 over-88-character lines and the trailing whitespace at `main.py:2016`. | 400 (mechanical) |
| 5 | `Bound every external process with a timeout` | **Three separate configurable budgets**, because one value cannot fit all three: `CCP4_TOOL_TIMEOUT_S` (minutes — FFT and EDSTATS on a large structure), `SETUP_SHELL_TIMEOUT_S` (seconds — sourcing `ccp4.setup-sh`), `PROVENANCE_COMMAND_TIMEOUT_S` (~1 s — the `git` probes at `main.py:1092`/`:1096`). A single constant either makes hung-git detection uselessly slow or kills valid crystallographic work. **Each needs its own outcome and its own test**, because they differ: a CCP4 timeout is a retryable entry failure, a setup-shell timeout is a startup failure that should abort the run, and a provenance timeout is bounded degradation (record `unknown` and continue). One shared reason code would conflate all three. | 140 |
| 6 | `Route diagnostics through logging` | A `logging` logger per module; `print` reserved for the progress line and the final human summary. Three constraints: **keep the explicit `[:300]`/`[:500]` bounds** (or add a bounded-output filter) — levels control severity, not message size, so they are not a substitute; **specify `QueueHandler`/`QueueListener`**, since workers are separate processes and their handlers would otherwise interleave mid-record; and **leave `_RunLog` as structured state**, not a logging handler — it is a deliberate diagnostic artifact, not a log sink. Roadmap step 6. | 260 |

### Phase C — Close the coverage holes *before* restructuring

| # | Commit | Contents | ~Size |
|---|---|---|---|
| 7 | `Cover the model-envelope map path offline` | Extend `_fake_ccp4_run_factory` to `mapmask` and drive `model-envelope` through crop-accepted, crop-larger-than-original and crop-past-the-cell-edge — the largest single coverage hole (4.1), and the *default* map scope, previously exercised only in the `ccp4`+`slow` lane that never runs. | 190 |
| 8 | `Unit-test the highest-risk CLI and resume surfaces` | A deliberately selected subset of 4.2, not all fourteen: `validate_resume_schemas` and `_batch_exit_code` first (both govern data-destructive behaviour), then `parse_pdb_id`, `load_ids_from_file`, `enumerate_entries`, `automatic_worker_limits`, `remove_stale_disabled_bond_outputs`, `--keep-intermediates`. **Also `verify_ccp4`, `prepare_inputs`, `read_resolution` and `has_final_files`, because Phase F moves all four** — commit 11 relocates `verify_ccp4`, commit 12 relocates the other three — and moving untested code is how a regression gets attributed to the wrong commit. Only `resolve_manual_inputs`, `available_cpu_count` and `available_memory_bytes` stay indirectly covered, none of which Phase F touches. | 330 |

### Phase D — Mechanical cleanups *before* the splits

Doing these first keeps the split diffs pure moves. Doing them after means renames interleave with structural changes and neither is reviewable.

| # | Commit | Contents | ~Size |
|---|---|---|---|
| 9 | `Rename pdbID to pdb_id throughout` | Identifiers only — the CSV column stays `pdbID`, mapped once at the writer with a comment. Huge diff, trivial review; kept alone for exactly that reason. | 300 |
| 10 | `Remove dead code and annotate the public surfaces` | Delete `_write_csv`, `COMMON_METALS`/`UNCOMMON_METALS`, `find_ccp4_setup`'s `common_candidates`, and the unreachable defensive branches in `metal_identification`. Then return and argument types on the ~20 public functions, standardising on PEP 585 builtins. | 260 |

### Phase E — Split `main.py` (replaces roadmap steps 10–11)

Extract outward-in, leaving `_run` for last: every extraction shrinks it, so by commit 15 the phase boundaries are obvious instead of guessed.

| # | Commit | Moves out of `main.py` | ~Size |
|---|---|---|---|
| 11 | `Move CCP4 environment resolution into ccp4_setup` | `resolve_env`, `_resolve_env_windows`, `_normalize_path_key`, `_parse_windows_set_output`, `verify_ccp4`. One `DEFAULT_CONFIG_FILES`; add `Ccp4SetupError`; delete the dead `explicit_setup` branch. Roadmap step 10. | 250 |
| 12 | `Extract input preparation and conversion provenance` | → `inputs.py`: `entry_dir_for`, `prepare_inputs`, `_gunzip_to`, download/cache, `enumerate_entries`, `read_resolution`, `read_map_column_resolution`. → `coordinate_conversion.py`: residue-identity records, legacy PDB packing, `_cif_to_pdb`, `_first_model_pdb`. Both are "getting an entry ready for analysis". | 700 |
| 13 | `Extract the worker entry point` | → `worker.py`: `process`, `_initial_result`, `_init_worker`, `_announce_inflight`, `_finalize_result`. | 300 |
| 14 | `Extract the driver's output machinery` | → `driver/writers.py`, `driver/progress.py`, `driver/runlog.py`. | 450 |
| 15 | `Extract resume handling and decompose the driver` | → `driver/resume.py`: `load_done`, `_ResumeStaging`, `_merge_csv_replacements`, `validate_resume_schemas`. Then `_run` — by now ~250 lines — becomes a ~60-line orchestrator over named phases. The decomposition is the only part here that is not a pure move. | 550 |
| 16 | `Extract the command-line interface` | → `cli.py`: `parse_args`, `main`, signal handling. **`src/main.py` stays** as a thin delegating shim — it is the documented entry point (commit 2) and 8 test files import it directly. Move the implementation, not the public script. | 160 |

### Phase F — Split `bond_analysis.py` (replaces roadmap step 12)

| # | Commit | Contents | ~Size |
|---|---|---|---|
| 17 | `Extract the DPI calculation and EDSTATS sigma helpers` | → `dpi.py`, which touches no bond logic and is the safest first move. `_sigma_index`, `_zd_indices` and `_sigma_for` go to `metal_identification`, so EDSTATS knowledge lives in one module. Both are pure moves. | 210 |
| 18 | `Load reference data on demand instead of at import` | → `reference_data.py`: `lru_cache` accessors, frozen results, one catalog parser shared with `metal_identification`. Removes the import-time I/O and the module-namespace leak. Behavioural, so kept separate from the moves. | 200 |
| 19 | `Extract declared-connection resolution` | → `declared_connections.py`. **Preserve both phases of the metal-partner check** (2.3): the first call runs before `find_cra` so the audit trail survives a failed resolution, the second uses the resolved element. | 400 |
| 20 | `Extract the bond output schema and row builders` | → `bond_schema.py`; move `_check_row_schema` in beside them so the module that owns the schema enforces it. | 350 |

### Phase G — Typed contracts (replaces roadmap step 13)

Cross-process dicts first — a shape error there is remote and hard to attribute.

| # | Commit | Contents | ~Size |
|---|---|---|---|
| 21 | `Introduce typed worker and density contracts` | Frozen `WorkerConfig` for the 14-key `cfg` dict, and a dataclass for the 14-key `run_density_analysis` return, which removes the mixed `[...]`/`.get()` access. The two lower-risk contracts, grouped. | 240 |
| 22 | `Introduce a typed entry result` | `EntryResult` for the 35-key result dict, carrying its blank-vs-zero invariant structurally. The largest behavioural risk in the plan; kept alone. | 350 |
| 23 | `Introduce a typed candidate record and a shared code vocabulary` | Dataclass for the 13-key candidate with explicitly optional post-annotation fields, plus `codes.py` of `StrEnum`s shared by producers and consumers — which resolves the two `"suspect"` vocabularies and `"insufficient data"`. | 450 |

### Phase H — Remaining

| # | Commit | Contents | ~Size |
|---|---|---|---|
| 24 | `Test and checksum the bundled reference data` | Unit tests for `classify_component`, `element_counts` and `_biconnected_components` against a fixture CIF block — `tools/` currently has none. Plus verifying `catalog_sha256` at load and giving `metal_distances_info.txt` the same treatment. | 300 |
| 25 | `Record the reference-data identity in the manifest` | Stamp **both** bundled datasets — the cofactor catalog and `metal_distances_info.txt`, which directly sets bond assignment and z-scores — as two hashes or one composite `reference_data_id`. Adds a manifest column, so it **breaks resume compatibility**; land it alongside a planned database run. Kept alone for that reason. | 70 |
| 26 | `Tidy driver reporting and documentation` | One `DriverError`, one handler, one exit-code policy (3.10); a separate `--log-dir` defaulting to `<output-dir>/logs/`; and the README split into `docs/` (usage, method, operations, maintenance), correcting the two stale claims in `tests/README.md`. | 760 |

**Rhythm.** Phases A–B are six commits and are the production-readiness gate; commit 1 alone fixes the first-run experience and is worth landing today. Phase C is two test commits that must land before E and F. Phases E–G are thirteen commits averaging ~330 lines.

**On revertibility.** Calling these "pure moves, individually revertible" would be too strong. Genuinely pure moves are commits 11–14, 16, 17, 19 and 20. The rest change behaviour: the `_run` decomposition in 15, lazy reference loading and parser unification in 18, and every dataclass and enum commit in Phase H. And once downstream commits land, none reverts cleanly in isolation — they are dependency-ordered. Treat them as **staged behavioural refactors with focused regression gates**: each needs its own test before the next lands.
