# AGENTS.md

Cross-tool entry point for AI coding agents (Codex, Cursor, Aider, Claude, ...). The dense, decisions-and-rationale project memory lives in [`CLAUDE.md`](CLAUDE.md); read it before non-trivial work.

## TL;DR

`landau.py` — thermodynamic equilibria and phase diagrams in the (semi-)grand ensemble. Dataclass-heavy, plug-in strategies (`Phase`, `Interpolator`, `Refiner`, `AbstractPolyMethod`).

## Commands

```bash
pip install -e .[test,constraints,fast-tsp,python-tsp]   # install with all extras used in CI
pytest                                                   # full suite
pytest tests/unit/test_calculate.py                      # one file
pytest tests/unit/test_calculate.py::test_foo            # one test
pytest -k pattern                                        # by name
ruff check .                                             # lint (line length 120, configured in pyproject.toml)
sphinx-build -b html docs docs/_build/html               # docs (needs .[docs])
python tests/integration/testplots.py --only 2d_basics   # render visual-review plots to tests/integration/_plots/
```

Python `>=3.11,<3.14`. Extras: `test`, `constraints`, `fast-tsp`, `python-tsp`, `ase`, `docs`.

## Repo layout

| Path | What's there |
|------|--------------|
| `landau/calculate.py` | `calc_phase_diagram`, `refine_phase_diagram`, `guess_mu_range`, `get_transitions`, clustering helpers |
| `landau/phases/` | `Phase` ABC, `AbstractLinePhase`, `LinePhase`, `TemperatureDependentLinePhase` (alias `TemperatureDepandantLinePhase` kept for back-compat), `IdealSolution`, `RegularSolution`, `InterpolatingPhase`, `SlowInterpolatingPhase`, `FastInterpolatingPhase` (the default solution-phase choice; `SlowInterpolatingPhase` stays as the per-scalar `scipy.optimize.brute` reference oracle). Sibling modules: `pointdefects.py` (`AbstractPointDefect`, `AbstractPointDefectSublattice`, `ConstantPointDefect`, `PointDefectSublattice`, `LowTemperatureExpansionSublattice`, `PointDefectedPhase`; the pre-split public API — `AbstractPointDefect` as a plain alias, `ConstantPointDefect`/`PointDefectSublattice`/`PointDefectedPhase` behind `@deprecate` shims — is re-exported through `phases/__init__.py`; import `AbstractPointDefectSublattice`/`LowTemperatureExpansionSublattice` from `landau.phases.pointdefects` directly) and `asewrapper.py` (`AsePhase`). User-facing classes are re-exported from `landau/__init__.py` |
| `landau/interpolate/` | `Interpolator` strategies: `PolyFit`, `SplineFit`, `SGTE`, `RedlichKister`, `StitchedFit`, `SoftplusFit`, `WhitneyTemperatureInterpolator` (the sklearn-style `WhitneyRBFInterpolator` it wraps is not re-exported). Central contract: `Interpolator.fit(x, y) → Interpolation`; `Interpolation.deriv() → Interpolation` returns `f'` (analytic for `PolyFit` / `SGTE` / `RedlichKister`, numerical-default for closure-wrapped fits) — `FastInterpolatingPhase` needs the analytic first derivative |
| `landau/refine.py` | `Refiner` ABC + `ScanRefiner`, `DelaunayLineRefiner`, `DelaunayTripleRefiner`, `ClausiusClapeyronRefiner`, `MiscibilityGapRefiner` |
| `landau/plot.py` | `plot_phase_diagram`, `plot_mu_phase_diagram`, 1d variants, `plot_excess_free_energy`, `get_polygons` |
| `landau/poly.py` | Point-cloud → polygon: `Concave`, `Segments`, optional `PythonTsp` / `FastTsp` / segment variants |
| `landau/resample.py` | `resample_borders`, `RandomlyShiftedPhase` — bootstrap-style border resampling |
| `landau/features.py` | `Locus` enum (`INTERIOR`/`BOUNDARY`/`TRIPLE`); imported as `from landau.features import Locus` (not re-exported from package root) |
| `benchmarks/` | Scripts backing any number cited in a PR body (see Working style); committed alongside the PR that introduces the number |
| `tests/unit/` | One subdirectory per module; filenames mirror what's under test |
| `tests/regression/` | Bug pins — names contain issue numbers or descriptive labels |
| `tests/integration/test_border_coverage.py` | Polygon-coverage smoke test across every `poly_method` × axis pair |
| `tests/integration/testplots.py` | Render script (NOT a pytest test); produces PNGs for visual review |
| `notebooks/` | Sphinx-included examples; commit with executed outputs |

## Conventions (hard constraints)

- **Pandas 2/3 compat**. Every `groupby().apply()` must pass `include_groups=False`. Do not drop pandas 2 support.
- **`shapely>=2.1`** required — `AbstractPolyMethod.make` uses `make_valid(method="structure")`.
- **Module names: no underscores.** `asewrapper`, not `ase_wrapper`.
- **Frozen dataclass + ABC** for `Phase` and `Interpolator` subclasses (immutable, structurally hashable). `Refiner` subclasses are plain mutable; `AbstractPolyMethod` is non-frozen.
- **Optional-dep gating** uses `pyiron_snippets.import_alarm.ImportAlarm`. In tests use module-wide `pytestmark = pytest.mark.skipif(ImportAlarm(...).message is not None, ...)` (whole-file dep) or per-test `@pytest.mark.skipif` against a `try/except`-set flag (mixed deps). Bare `try/except ImportError` is not enough.
- **`distance_threshold`** is a required kwarg on `cluster_T_c` / `cluster_T_c_mu` / `cluster`; user-facing `cluster_phase` and `get_polygons` default it to `0.5`.

## Working style (project house rules)

- Match a terse, imperative tone in commits, PRs, and comments. No marketing language, no apologies, no `**bold**` headline numbers in PR bodies, no `[Fix this →]` action links. Plain technical reports.
- **Evidence over claims.** Cite a commit hash, file path, test count, or command output for any claim. Numbers in a PR body come with a script in `benchmarks/` in the same PR. For physics/numerics, prefer saying nothing over saying something unverified.
- **Comments and docstrings reflect only the current state.** No "old vs new", no rejected alternatives, no narration of the PR's evolution.
- **One purpose per PR.** Split notebooks, benchmarks, refactors, and unrelated fixes into their own PRs. If a PR grows a second concept, propose splitting before pushing more.
- **Minimal change.** Reuse existing helpers (numpy, scipy, shapely, matplotlib.testing, pyiron_snippets) before writing a new one. Don't introduce abstractions for hypothetical future requirements.
- **Tests assert tight conditions** that a degenerate or constant solution would fail. Loose `atol=0.05` "it ran" tests get rejected. Use Hypothesis for round-trip recovery on fits.
- **Git: rebase, never merge.** Merge commits are disabled on GitHub. Rebase onto `origin/main` and force-push to keep history linear.
- **Conventional Commits drive releases.** `release-please` (`.github/workflows/release-please.yml`, PR #225) reads conventional-commit messages on `main` to open release PRs; non-conforming messages are ignored by the release tooling. Use `feat:`/`fix:`/`docs:`/`test:`/`chore:`/`refactor:`, `!` or `BREAKING CHANGE:` for breaks.
- **Do not commit** `.hypothesis/`, `_version.py`, stray top-level scripts, duplicate exploratory files.
- **Keep PR body in sync with the diff.** When review feedback invalidates a claim in the body, edit the body — never leave a stale claim standing.
- **Notebooks** are committed with executed outputs only.

## Deeper context

[`CLAUDE.md`](CLAUDE.md) carries: architecture rationale, module-level design notes, the full open-scope / closed-decisions cheat sheet keyed by issue and PR number, and the testplot label/comment workflow. Read it before changing public API, refiner/poly-method behaviour, or pandas-touching code.
