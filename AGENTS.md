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

Python `>=3.11,<3.14`. Extras: `test`, `constraints`, `fast-tsp`, `python-tsp`, `ase`, `phonopy`, `docs`, `test-fleche` (fleche digest-hook tests, own CI job: `pytest -m fleche`).

## Repo layout

| Path | What's there |
|------|--------------|
| `landau/calculate.py` | `calc_phase_diagram` (its `refine=` accepts `True` / `False` / `Sequence[Refiner]`; every output row carries a `locus` column populated from `landau.features.Locus`), `refine_phase_diagram`, `guess_mu_range`, `get_transitions`, clustering helpers |
| `landau/phases/` | `Phase` ABC, `AbstractLinePhase`, `LinePhase`, `TemperatureDependentLinePhase` (typo alias `TemperatureDepandantLinePhase` kept behind `@deprecate`, removal at 2.0), `IdealSolution`, `RegularSolution`, `InterpolatingPhase`, `SlowInterpolatingPhase` (per-scalar `scipy.optimize.brute` reference oracle), `FastInterpolatingPhase` (vectorised subclass — the default user-facing solution phase), `Surface2DInterpolatingPhase` (fits one `f(T, c)` surface globally, slices per-T). Sibling modules: `pointdefects.py` (`AbstractPointDefect`, `AbstractPointDefectSublattice`, `ConstantPointDefect`, `PointDefectSublattice`, `LowTemperatureExpansionSublattice`, `PointDefectedPhase`), `asewrapper.py` (`AsePhase`) and `quasiharmonic.py` (`PhonopyQuasiHarmonicPhase` — evaluates `min_V [E(V) + F_vib(V, T)]` from phonopy `ThermalProperties` at any `T` instead of fitting a sampled `F(T)`; `phonopy` extra, PR #440). `phases/__init__.py` re-exports the pre-split point-defect names — `AbstractPointDefect` as a plain alias, `ConstantPointDefect`/`PointDefectSublattice`/`PointDefectedPhase` behind `@deprecate` shims. Post-split names (`AbstractPointDefectSublattice`, `LowTemperatureExpansionSublattice`) are **not** re-exported — import from `landau.phases.pointdefects` directly. User-facing classes are also re-exported from `landau/__init__.py` |
| `landau/interpolate/` | 1-D `Interpolator` strategies (`Interpolator.fit(x, y) → Interpolation`; `Interpolation.deriv() → Interpolation`, analytic for `PolyFit`/`SGTE`/`RedlichKister`, `NumericalDerivative` central difference otherwise — `FastInterpolatingPhase` uses the first derivative): `PolyFit`, `SplineFit`, `SGTE`, `RedlichKister`, `StitchedFit`, `SoftplusFit`, `WhitneyTemperatureInterpolator` (the sklearn-style `WhitneyRBFInterpolator` it wraps is not re-exported). 2-D surface strategies (`SurfaceInterpolator.fit(T, c, H) → FittedSurface`; `FittedSurface.slice_at(T) → Interpolation`; consumed by `Surface2DInterpolatingPhase(surface_interpolator=...)`, required kwarg): `CalphadSurface2DInterpolator` (SGTE terminals + poly-in-T Redlich-Kister — needs c=0/c=1 terminal phases), `SoftplusSurface2DInterpolator` (softplus-link amplitudes, provably convex fixed-T slices — no terminal requirement, good for intermetallics), `WhitneySurface2DInterpolator` (Whitney-extended RBF — no terminal requirement); paired `CalphadFittedSurface`/`SoftplusFittedSurface`/`WhitneyFittedSurface` helpers. `CalphadSurface2DInterpolator`'s terminal T-model is swappable via `terminal_interpolator=` (default `SGTE(4)`; `terminal_sgte_order` deprecated, removal at 2.0 — PR #446; pass `WhitneyTemperatureInterpolator()` when the surface must extrapolate below its fitted T-window, since SGTE's unbounded low-T entropy fabricates spurious low-T liquid fields). The `_scalarize(x)` 0-d → Python scalar helper also lives here (in `basic.py`) so `landau.phases` and `SplineFit` can share it without a cycle. Scalar-in/scalar-out and shape preservation are enforced suite-wide: one Hypothesis property in `tests/unit/interpolate/test_shape_contract.py` runs every 1-D interpolator plus its `deriv()` — add new interpolators to its `CASES` dict (PR #432) — and `tests/regression/test_issue_428.py` fails when a `TemperatureInterpolator` is exported without `calc_phase_diagram` coverage (PR #430) |
| `landau/refine.py` | `Refiner` ABC + `ScanRefiner`, `DelaunayLineRefiner`, `DelaunayTripleRefiner`, `ClausiusClapeyronRefiner`, `MiscibilityGapRefiner`; the last two share `_CCBase` (keyword-only ctors: `dT_max=5.0, dT_min=1.0, dc_max=0.01, dc_min=0.0, max_steps=500`). Delaunay refiners' `propose()` yields candidate objects (`_SimplexCandidate` / `_TripleCandidate`) wrapping a numpy-backed `_Simplex` view (`.unique_phases()`, `.centroids()`); `_delaunay_simplices(df) → list[tuple[_Simplex, int]]` memoises tessellation on the most recent frame via weakref-identity keying (module-level `_simplex_cache`), so the default 2-D refiners share one Delaunay per refine pass. `_CCBase._trace` aborts as soon as an emitted point is `_dominated` (pair crossed a triple point into the metastable tail). `_dominated` (PR #345): drops a transition when an own phase is absent (`phi = +inf`), when an outside phase undercuts `max(own_phi)` (deterministic — no set-iteration order), or — triple points only — when the three potentials spread wider than `_TRIPLE_COEXIST_TOL = 1e-4` eV (two-phase boundaries are exempt: a first-order boundary carries a real spread). `default_refiners(df)`: 2-D grid → `DelaunayTripleRefiner` + `ClausiusClapeyronRefiner` + `MiscibilityGapRefiner`; 1-D scan → matching `ScanRefiner`(s). `DelaunayLineRefiner` is opt-in. `RefinedPoint` / `RefinedMiscibilityGap` carry `boundary_id: int` — refined rows are tagged by coexistence line |
| `landau/plot.py` | `plot_phase_diagram`, `plot_mu_phase_diagram`, 1d variants, `plot_excess_free_energy`, `get_polygons` |
| `landau/poly.py` | Point-cloud → polygon: `Concave`, `Segments`, optional `PythonTsp` / `FastTsp` / segment variants |
| `landau/resample.py` | `resample_borders`, `RandomlyShiftedPhase` — bootstrap-style border resampling |
| `landau/features.py` | `Locus` `StrEnum` (`INTERIOR`/`BOUNDARY`/`TRIPLE`) backing the `locus` column of `calc_phase_diagram` output; imported as `from landau.features import Locus` (not re-exported from package root) |
| `landau/fleche.py` | fleche digest hooks (`digest_hooks`), loaded only by fleche's entry-point machinery — landau never imports fleche (PRs #441 + #450, refs #438). Hooks cover only types fleche rejects outright: `Refiner`, `FittedSurface`, the plain `Interpolation` classes, `AsePhase`, `PhonopyQuasiHarmonicPhase`, `WhitneyRBFInterpolator`, and raw scipy `UnivariateSpline` (a stopgap, #449). Closure-backed fits (`SplineFit`/`StitchedFit`/`SoftplusFit`/Whitney interpolations) still raise `Indigestible`. Needs `fleche>=0.22.1` (PR #448) |
| `benchmarks/` | Scripts backing any number cited in a PR body (see Working style); committed alongside the PR that introduces the number |
| `scripts/` | `cc_demo.py` (+ rendered PNG) — standalone Clausius-Clapeyron tracing demo on an analytic sub-regular solution, compared against brute-force isothermal refinement; not part of the test suite |
| `tests/unit/` | Filenames mirror what's under test; `interpolate/`, `phases/`, `plot/` are subdirectories, the rest are flat (`test_calculate.py`, `test_refine.py`, `test_poly.py`, `test_resample.py`, `test_softplus.py`, `test_whitney.py`, `test_phases.py`, `test_fleche.py` — the `fleche`-marked digest-hook tests, with the `flecheprobe.py` helper module beside them) |
| `tests/regression/` | Bug pins — names contain issue numbers or descriptive labels |
| `tests/integration/test_border_coverage.py` | Polygon-coverage smoke test across every `poly_method` × axis pair |
| `tests/integration/test_qha_vs_phonopy.py` | Reproduces the QHA fit-error mechanism from #427 on fcc-Cu/EMT (needs `ase` + `phonopy`) |
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

- **#116 refactor umbrella** — long-running sweep splitting big functions and pinning private helpers with direct unit tests. Sub-issues open at the time of writing (open PRs in flight noted; check before duplicating):

  - #388 (`_semigrand_average_concentration` in `calculate.py`, PR #391)

  - #424 (`SoftplusSurface2DInterpolator._unpack` / `_n_params` / `_const_init` in `interpolate/softplus.py`)

  - #425 (`_label_fits` in `plot.py`)

  - #435 (`Surface2DInterpolatingPhase._gather_training_data` in `phases/__init__.py`)

  - #436 (`FastInterpolatingPhase._find_phi_c_cached` in `phases/__init__.py`)

  - #437 (`SoftplusSurface2DInterpolator._solver_kwargs` / `_vandermonde` in `interpolate/softplus.py`)

  PR #422 (direct tests for `_fit_softplus` / `_fit_slice`) merged 2026-08-30 without a numbered sub-issue; #423 closed 2026-09-01 by PR #434; #390 closed 2026-09-02 by PR #394 and #413 by PR #414. Closed sub-issues are recorded one cohort per line in [`CLAUDE.md`](CLAUDE.md)'s #116 section — check there before re-picking one.

- **#137 `phases/__init__.py` split** — `pointdefects.py` and `asewrapper.py` already split out; further splits (line vs solution vs interpolating) are the open direction. `phases/__init__.py` is still ~980 lines.

- **#332 phase free-energy parametrization is indirect** — both `InterpolatingPhase` and `Surface2DInterpolatingPhase` reach `f(c[, T])` through a sample-and-refit round-trip. Open direction: an optional "just give me a callable" builder path so a `.tdb` importer (#138) or bespoke model can skip it. Parked with #137.

- **#138 TDB file import** — scoping; feeds #332.

- **#34, #60 plot/calc API 2.0 refactor** — axes-as-arg for `plot_{mu,}_phase_diagram` + broader rearrangement for 2.0.

- **#39 entropy/enthalpy methods on phases** — blocks the diagnostic + EEC transfer in #339 and item 5 of the mapping notes in #337.

- **#337, #339 CALPHAD-side design notes** — mapping/ZPF-line ideas (Sundman/Dupin/Hallstedt Calphad 75 2021) and equal-entropy criterion (Sundman et al. Calphad 68 2020). Design conversations, no code planned.

- **#344 collapse CEF Au-Cu into one partitioning fcc phase with `MiscibilityGapRefiner`** — blocked on `MiscibilityGapRefiner`'s per-step scan cost.

- **#427 SGTE is the wrong default for QHA/phonon `F(T)`** — the `T·ln(T)` term makes `S = -dG/dT` diverge as `T → 0`, so quantum-harmonic data (which includes `T = 0`) fits badly enough to shift a melting point ~+10 K with the default `SGTE(3)`; `PolyFit(8)` is the workaround. Still open for the upstream half: a solid-state QHA `ThermoChem` consumed via `AsePhase` (validated `QuasiHarmonicCrystalThermo` prototype in the issue, headed for ASE) and a docs note / `T = 0` warning. The phonopy-direct route **landed as PR #440** (merged 2026-09-02, released in 1.12.0): `PhonopyQuasiHarmonicPhase` (`landau/phases/quasiharmonic.py`, new `phonopy` extra) — an `AbstractLinePhase` evaluating `min_V [E(V) + F_vib(V, T)]` from phonopy `ThermalProperties` at any `T` instead of fitting a sampled `F(T)`. `eos` defaults to `PolyFit(8)` (capped at one fewer parameter than stable volumes; the strings `"vinet"`/`"birch_murnaghan"`/`"murnaghan"` select phonopy closed forms); volumes with modes below `min_frequency = -0.05` THz are dropped with `DynamicalInstabilityWarning`; past the sampled volumes the minimisation is *constrained* to the sampled range (clamped, with `EosExtrapolationWarning`) — `extrapolate=True` follows the fit out but needs a closed-form eos. `check_equation_of_state(T, plot_error=...)` is the diagnostic. Also raised the `ase` floor to 3.21.0.

- **#438 `AbstractLinePhase.__hash__` is process-local** — closed 2026-09-02 by **PR #441** (released in 1.12.0): fleche digest hooks ship under fleche's entry-point group (`landau/fleche.py`, `test-fleche` extra with its own CI job; landau never imports fleche), and the `_hash` dataclass *field* was dropped (now a plain attribute derived in `__post_init__`, so a dataclass-walking digest no longer folds the per-interpreter-salted `hash(bytes)` value in — `__hash__`/`__eq__` unchanged). **PR #450** (released in 1.13.0) swept every remaining refused object and added three hooks: raw scipy `UnivariateSpline` (`_eval_args` + `ext`; a stopgap under **#449** — drop it once fleche digests scipy splines itself), `PhonopyQuasiHarmonicPhase` (its own `_key`; its `_hash` field also became a plain attribute), `WhitneyRBFInterpolator` (training data + hyperparameters — makes `WhitneyFittedSurface` digestible too). Closure-backed fits still raise `Indigestible` since two fits of one interpolator share a code object — the proper fix (fit provenance on `Interpolation`) remains a possible follow-up. Upstream caveat: fleche's destructuring *storage* bypasses digest hooks (fleche#918), so a hooked `AsePhase` passed as a `@fleche` argument digests fine but can fail on save. `test-fleche` floor is `fleche>=0.22.1` (PR #448 — 0.22.1's captured-state function digest turned the old closure-collision test into an `Indigestible` raise).

- **#443 nuclear quantum correction to classical-MD solid free energies via the QHA** (idea, filed 2026-09-02) — `F = F_calphy − F_QHA,cl + F_QHA,qu`: anharmonicity from MD, quantum statistics from the harmonic model, the classical harmonic term cancels the double-counting; phonopy returns both statistics from one mesh. Which correction dominates flips by element (Al: quantum ~5× anharmonic; Ca/Si: reversed), 1 meV/atom ≈ 10 K of melting point. Solid-only (the liquid has no harmonic reference); reference data for Al/Ca/Si on a GRACE MLIP attached to the issue. No code planned yet — a possible helper where `PhonopyQuasiHarmonicPhase` and calphy-style free energies meet.

- **#81 analytic SRO models** — prototype `QuasiChemicalPhase` (PR #123) was closed without merging; still open.

- **#33 fast Legendre transforms**, **#59 autodiff for `concentration`** — stretch goals; no owner.

- **#62 flat → `src/` layout** — long-standing. #70 (weak Hypothesis strategies for polygon tests) closed 2026-09-02 by PR #395: `poly_dataframe` now correlates `c`/`T`/`mu` per row instead of drawing them independently.

- **Open PRs in flight** (check before duplicating): #452 (`transition_temperatures` option on the 2d plotters, opened 2026-09-03), #391 (tests for #388), the CEF stack #324/#326/#346 (+ parked #334) behind #344, and long-open design prototypes #306 (`IntermetallicPhase`), #250 (`PhaseDiagram` object interface), #249 (`BufferedSegments`).

**Out of scope**

- Dropping pandas 2 support (PR #93, #113).

- Speculative "might be useful" work — parked, not landed (PR #129).

- Underscored module names — use `asewrapper`, not `ase_wrapper` (PR #68).

## Deeper context

[`CLAUDE.md`](CLAUDE.md) carries: architecture rationale, module-level design notes, the full open-scope / closed-decisions cheat sheet keyed by issue and PR number, and the testplot label/comment workflow. Read it before changing public API, refiner/poly-method behaviour, or pandas-touching code.
