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
| `landau/calculate.py` | `calc_phase_diagram` (its `refine=` accepts `True` / `False` / `Sequence[Refiner]`; every output row carries a `locus` column populated from `landau.features.Locus`), `refine_phase_diagram`, `guess_mu_range`, `get_transitions`, clustering helpers |
| `landau/phases/` | `Phase` ABC, `AbstractLinePhase`, `LinePhase`, `TemperatureDependentLinePhase` (typo alias `TemperatureDepandantLinePhase` kept behind `@deprecate`, removal at 2.0), `IdealSolution`, `RegularSolution`, `InterpolatingPhase`, `SlowInterpolatingPhase` (per-scalar `scipy.optimize.brute` reference oracle), `FastInterpolatingPhase` (vectorised subclass — the default user-facing solution phase), `Surface2DInterpolatingPhase` (fits one `f(T, c)` surface globally, slices per-T). Sibling modules: `pointdefects.py` (`AbstractPointDefect`, `AbstractPointDefectSublattice`, `ConstantPointDefect`, `PointDefectSublattice`, `LowTemperatureExpansionSublattice`, `PointDefectedPhase`) and `asewrapper.py` (`AsePhase`). `phases/__init__.py` re-exports the pre-split point-defect names — `AbstractPointDefect` as a plain alias, `ConstantPointDefect`/`PointDefectSublattice`/`PointDefectedPhase` behind `@deprecate` shims. Post-split names (`AbstractPointDefectSublattice`, `LowTemperatureExpansionSublattice`) are **not** re-exported — import from `landau.phases.pointdefects` directly. User-facing classes are also re-exported from `landau/__init__.py` |
| `landau/interpolate/` | 1-D `Interpolator` strategies (`Interpolator.fit(x, y) → Interpolation`; `Interpolation.deriv() → Interpolation`, analytic for `PolyFit`/`SGTE`/`RedlichKister`, `NumericalDerivative` central difference otherwise — `FastInterpolatingPhase` uses the first derivative): `PolyFit`, `SplineFit`, `SGTE`, `RedlichKister`, `StitchedFit`, `SoftplusFit`, `WhitneyTemperatureInterpolator` (the sklearn-style `WhitneyRBFInterpolator` it wraps is not re-exported). 2-D surface strategies (`SurfaceInterpolator.fit(T, c, H) → FittedSurface`; `FittedSurface.slice_at(T) → Interpolation`; consumed by `Surface2DInterpolatingPhase(surface_interpolator=...)`, required kwarg): `CalphadSurface2DInterpolator` (SGTE terminals + poly-in-T Redlich-Kister — needs c=0/c=1 terminal phases), `SoftplusSurface2DInterpolator` (softplus-link amplitudes, provably convex fixed-T slices — no terminal requirement, good for intermetallics), `WhitneySurface2DInterpolator` (Whitney-extended RBF — no terminal requirement); paired `CalphadFittedSurface`/`SoftplusFittedSurface`/`WhitneyFittedSurface` helpers. The `_scalarize(x)` 0-d → Python scalar helper also lives here (in `basic.py`) so `landau.phases` and `SplineFit` can share it without a cycle |
| `landau/refine.py` | `Refiner` ABC + `ScanRefiner`, `DelaunayLineRefiner`, `DelaunayTripleRefiner`, `ClausiusClapeyronRefiner`, `MiscibilityGapRefiner`; the last two share `_CCBase` (keyword-only ctors: `dT_max=5.0, dT_min=1.0, dc_max=0.01, dc_min=0.0, max_steps=500`). Delaunay refiners' `propose()` yields candidate objects (`_SimplexCandidate` / `_TripleCandidate`) wrapping a numpy-backed `_Simplex` view (`.unique_phases()`, `.centroids()`); `_delaunay_simplices(df) → list[tuple[_Simplex, int]]` memoises tessellation on the most recent frame via weakref-identity keying (module-level `_simplex_cache`), so the default 2-D refiners share one Delaunay per refine pass. `_CCBase._trace` aborts as soon as an emitted point is `_dominated` (pair crossed a triple point into the metastable tail). `default_refiners(df)`: 2-D grid → `DelaunayTripleRefiner` + `ClausiusClapeyronRefiner` + `MiscibilityGapRefiner`; 1-D scan → matching `ScanRefiner`(s). `DelaunayLineRefiner` is opt-in. `RefinedPoint` / `RefinedMiscibilityGap` carry `boundary_id: int` — refined rows are tagged by coexistence line |
| `landau/plot.py` | `plot_phase_diagram`, `plot_mu_phase_diagram`, 1d variants, `plot_excess_free_energy`, `get_polygons` |
| `landau/poly.py` | Point-cloud → polygon: `Concave`, `Segments`, optional `PythonTsp` / `FastTsp` / segment variants |
| `landau/resample.py` | `resample_borders`, `RandomlyShiftedPhase` — bootstrap-style border resampling |
| `landau/features.py` | `Locus` `StrEnum` (`INTERIOR`/`BOUNDARY`/`TRIPLE`) backing the `locus` column of `calc_phase_diagram` output; imported as `from landau.features import Locus` (not re-exported from package root) |
| `benchmarks/` | Scripts backing any number cited in a PR body (see Working style); committed alongside the PR that introduces the number |
| `tests/unit/` | Filenames mirror what's under test; `interpolate/`, `phases/`, `plot/` are subdirectories, the rest are flat (`test_calculate.py`, `test_refine.py`, `test_poly.py`, `test_resample.py`, `test_softplus.py`, `test_whitney.py`, `test_phases.py`) |
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

## Design themes

Brief map of open scope; the exhaustive cheat sheet keyed by issue+PR lives in [`CLAUDE.md`](CLAUDE.md). Fetch the issue before re-litigating.

**Active**

- **#116 refactor umbrella** — long-running sweep splitting big functions and pinning private helpers with direct unit tests. Sub-issues open at the time of writing: #376 (`SGTEInterpolation.deriv()`), #377 (`_scalarize`), #386 (`_rbf_gradient` in `interpolate/whitney.py`), #388 (`_semigrand_average_concentration` in `calculate.py`), #389 (`_curve_obstacles` in `plot.py`), #390 (`ConstantPointDefect.excess_free_energy` + `AbstractPointDefectSublattice._get_zes`), #398 (`_CallableInterpolation` in `interpolate/basic.py`), #399 (dedup the empty-`tdf` branch of `get_transitions` in `calculate.py`), #400 (`_delaunay_simplices` cache flip on a fresh frame). PR #397 tracks a matching pin for `_scipy_at_least` without a numbered sub-issue.

- **#137 `phases/__init__.py` split** — `pointdefects.py` and `asewrapper.py` already split out; further splits (line vs solution vs interpolating) are the open direction. `phases/__init__.py` is still ~980 lines.

- **#332 phase free-energy parametrization is indirect** — both `InterpolatingPhase` and `Surface2DInterpolatingPhase` reach `f(c[, T])` through a sample-and-refit round-trip. Open direction: an optional "just give me a callable" builder path so a `.tdb` importer (#138) or bespoke model can skip it. Parked with #137.

- **#138 TDB file import** — scoping; feeds #332.

- **#34, #60 plot/calc API 2.0 refactor** — axes-as-arg for `plot_{mu,}_phase_diagram` + broader rearrangement for 2.0.

- **#39 entropy/enthalpy methods on phases** — blocks the diagnostic + EEC transfer in #339 and item 5 of the mapping notes in #337.

- **#337, #339 CALPHAD-side design notes** — mapping/ZPF-line ideas (Sundman/Dupin/Hallstedt Calphad 75 2021) and equal-entropy criterion (Sundman et al. Calphad 68 2020). Design conversations, no code planned.

- **#344 collapse CEF Au-Cu into one partitioning fcc phase with `MiscibilityGapRefiner`** — blocked on `MiscibilityGapRefiner`'s per-step scan cost.

- **#81 analytic SRO models** — prototype `QuasiChemicalPhase` (PR #123) was closed without merging; still open.

- **#33 fast Legendre transforms**, **#59 autodiff for `concentration`** — stretch goals; no owner.

- **#62 flat → `src/` layout**, **#70 hypothesis strategies for polygon tests are weak** — long-standing.

**Out of scope**

- Dropping pandas 2 support (PR #93, #113).

- Speculative "might be useful" work — parked, not landed (PR #129).

- Underscored module names — use `asewrapper`, not `ase_wrapper` (PR #68).

## Deeper context

[`CLAUDE.md`](CLAUDE.md) carries: architecture rationale, module-level design notes, the full open-scope / closed-decisions cheat sheet keyed by issue and PR number, and the testplot label/comment workflow. Read it before changing public API, refiner/poly-method behaviour, or pandas-touching code.
