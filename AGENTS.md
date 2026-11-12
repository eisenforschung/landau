# AGENTS.md

Reference for AI coding agents working on landau (Claude loads it through `CLAUDE.md`).
It describes the code as it is now.
History — which PR did what, when, and what it replaced — lives in git and on the issue tracker, not here.

When editing this file, write one sentence per line, and one `;`-clause per line.
Never hard-wrap or join lines.
Many PRs touch this file at once, and line-per-sentence keeps their diffs small and mergeable.
`python .github/scripts/reflow_agent_docs.py` applies the rule, and `--check` reports violations.

**landau.py** calculates thermodynamic equilibria and plots phase diagrams in the (semi-)grand ensemble, mostly for binary alloys.
Dataclass-heavy, with plug-in strategies: `Phase`, `Interpolator`, `Refiner`, `AbstractPolyMethod`.

## Working style (read this first)

Distilled from maintainer feedback;
several of these have been violated often enough to repeat.

### Tone in PRs, commits, and comments

- Terse, imperative.
  No "Happy to...", no apologies, no enthusiasm filler.
- PR bodies are plain technical reports: no **bold** headline numbers, no "Conclusion" sections, no `[Fix this →]` action links, no marketing claims ("dramatic speedup").
- Do not invoke the maintainer's name or authority in PR bodies or comments.
- Keep the PR description in sync with the diff;
  when review or later commits invalidate a claim in the body, edit the body.
- When naming another PR's issue in a commit message or PR body, avoid closing keywords ("closes #N") unless closure is intended — GitHub applies them on push to `main`.

### Comments and docstrings

- Reflect only the current state of the code.
  Historical motivation is fine (why a threshold exists);
  intermediate steps, rejected alternatives and "old vs new" comparisons belong in the PR description.
- Never reference thresholds, approaches or behaviour that no longer exist.

### Evidence over claims

- Cite the commit hash, file path, test count or command output behind any claim.
  A number in a PR body comes with the script that produced it, committed under `benchmarks/` in the same PR.
- For physics/numerics, rather say nothing than something unverified.
- If asked to verify, actually verify — read the docs, run the code.
  Do not re-assert.

### Scope and abstraction

- Minimal change: "rather merge less features now and extend later than break API".
- Before writing a helper, check the codebase and existing deps (numpy, scipy, shapely, matplotlib.testing, pyiron_snippets) for one that already does the job.
- One purpose per PR.
  Notebooks, benchmarks, refactors and unrelated fixes go in separate PRs;
  when a PR grows a second concept, propose splitting before pushing more.
- Speculative or "might be useful" work gets parked, not landed.

### Tests

- Assert tight conditions a degenerate/constant solution would fail.
  Loose `atol=0.05` "it ran" tests get rejected.
- Hypothesis for round-trip recovery (random coeffs in → `fit()` → coeffs out) where applicable.
- Share one `atol` constant across related tests instead of scattering magic numbers.

### Debugging numerics

- When diagnosing a fit/solver/physics issue, plot the relevant landscape (loss vs parameter, etc.), attach it inline to the PR comment, and add a walkthrough section to a notebook so the maintainer can step through it.

### Git

- Merge commits are disabled on GitHub.
  Never `git merge` to update a branch: `git reset` off any stray merge, `git rebase origin/main`, force-push.
- Conventional Commits drive `release-please` (`feat:`/`fix:`/`docs:`/`test:`/`chore:`/`refactor:`, `!` or `BREAKING CHANGE:` for breaks);
  non-conforming messages are ignored by the release tooling.
- Do not commit `.hypothesis/`, `_version.py`, stray top-level scripts, or duplicates of files you are adding.
- Notebooks are committed with executed outputs only.

## Hard constraints

- **Pandas 2 and 3 are both supported**, so every `groupby().apply()` passes `include_groups=False`.
  Do not drop pandas 2 support.
  The single-group `apply` workaround lives in `calculate._apply_series(grouped, fn, name)` — reuse it.
  Pinned by `tests/regression/test_single_group_apply.py` and `test_cluster_phase_pandas3.py`.
- **`shapely>=2.1`**: `AbstractPolyMethod.make` repairs self-intersecting rings with `make_valid(method="structure", keep_collapsed=False)`, which returns the enclosed polygon instead of a `GeometryCollection`.
- **Lower bounds are tested**: `test-minimum-deps.yml` installs every direct dep at its floor (`uv pip install --resolution lowest-direct`) in a Python 3.11 venv;
  raising a feature's requirement means raising the floor in `pyproject.toml`.
- **Module names have no underscores**: `asewrapper`, not `ase_wrapper`.
  Promote a module to a subpackage once it grows.
- **Optional deps** gate behind `pyiron_snippets.import_alarm.ImportAlarm` (import works, instantiation raises).
  In tests: a module-level `pytestmark = pytest.mark.skipif(ImportAlarm(...).message is not None, ...)` when a whole file needs one extra (`tests/unit/phases/test_ase.py`), per-test `skipif` on `try/except`-set flags when a file mixes extras (`tests/unit/test_poly.py`).
  A bare `try/except ImportError` is not a gate.
- **Deprecations** use `pyiron_snippets` `@deprecate(..., version="2.0")` for functions/kwargs.
  Classes can't be wrapped without breaking `isinstance`, so deprecated classes warn from `__post_init__` (`_warn_deprecated_phase`, `stacklevel=4`).

## Commands

```bash
pip install -e .[test,constraints,fast-tsp,python-tsp]   # CI extras; also: ase, phonopy, docs, test-fleche
pytest                                                   # full suite (pytest -k pattern, file::test as usual)
pytest -m fleche                                         # fleche digest-hook tests, need .[test-fleche]; own CI job
ruff check .                                             # line length 120
sphinx-build -b html docs docs/_build/html               # docs, needs .[docs]
python tests/integration/testplots.py --only 2d_basics 2d_toy --poly-method fasttsp   # PNGs → tests/integration/_plots/
```

Python `>=3.11,<3.14`.

## Architecture

The semi-grand potential `phi(T, dmu)` is the central quantity;
equilibrium is `argmin` over phases, and `c = -∂phi/∂dmu`.
`Phase` and `Interpolator` subclasses are frozen dataclasses (immutable, structurally hashable);
`Refiner` subclasses are plain mutable classes;
`AbstractPolyMethod` is a non-frozen dataclass.
Every phase obeys a scalar-in/scalar-out, shape-preserving contract;
0-d numpy results collapse to Python scalars through the single helper `_scalarize` (`interpolate/basic.py`).

### `landau/phases/`

- `__init__.py`: the `Phase` ABC,
  `AbstractLinePhase`,
  `LinePhase`,
  `TemperatureDependentLinePhase`,
  `IdealSolution`, the interpolating phases, the ideal-mixing entropy `S(c)` and `check_concentration_interpolation`.
  User-facing classes are re-exported from `landau/__init__.py`;
  `Phase`/`AbstractLinePhase` stay under `landau.phases`.
- Deprecated, removal at 2.0: `RegularSolution` and `InterpolatingPhase` (use `FastInterpolatingPhase`, with `interpolator=RedlichKister(n)` for a regular solution), the typo alias `TemperatureDepandantLinePhase`, and the solution phases' `check_interpolation` methods, which forward to `check_concentration_interpolation`.
  `TemperatureDependentLinePhase.check_interpolation` is not deprecated.
  The suite, testplots and notebooks still construct the deprecated classes — new code should not.
- `FastInterpolatingPhase` is the default solution phase;
  `SlowInterpolatingPhase` (per-scalar `scipy.optimize.brute`) is the reference oracle it is tested against.
  Fast: per `T`, evaluate `f(c) = fe(c) - T·S(c)` on a 201-point grid (`_n_grid`), one vectorized `argmin` over `dmu` picks the global basin (handles miscibility gaps), then 6 Newton steps (`_n_newton`) in the logit `u = log(c/(1-c))`, where the entropy term is linear so the polish is well-conditioned from dilute to saturated.
  First derivative from `fe.deriv()`, second by finite difference at `_fd2 = 1e-3`;
  `c = -∂phi/∂dmu` holds to ~1e-6.
  `_find_phi_c_cached` (`lru_cache` keyed on `(shape, tobytes())`) lets `semigrand_potential` and `concentration` share one solve.
  Known limitation (#563): within ~1e-9 eV of a gap's coexistence `mu*`, `concentration` can return a grid node that is neither branch's minimum.
  `concentration_range` and `maximum_extrapolation` are mutually exclusive (`ValueError` if both set).
- `Surface2DInterpolatingPhase(surface_interpolator=...)` (required kwarg) fits one `f(T, c)` surface globally from line phases sampled over `temperature_range`, then reuses the Fast solver on `FittedSurface.slice_at(T)`.
- `pointdefects.py`: `AbstractPointDefect`, `ConstantPointDefect`, `PointDefectedPhase` (host line phase + sublattices, with `site_fractions(T, dmu)`), and the sublattice models.
  `AbstractPointDefectSublattice` is the ABC, with hooks `_site_log_partition(s)` → `L` and `_site_normalization(s)` → `D`, coupled by `D = 1/L'`.
  `PointDefectedPhase` clamps `c` to `[0, 1]` and continues `phi` as the tangent line past each saturation point (`_clamp_fixed_T`, window from `_saturation_window`), so the unbounded LTE model can't drive `phi → -∞` and spuriously win `argmin`;
  a no-op for the bounded models.
  Block exclusion needs no class: a `PointDefectSublattice` on `sublattice_fraction = η·m` with each defect's `excess_entropy` raised by `kB·ln(1/m)`.
  Only the pre-split names are re-exported from `landau.phases` (`AbstractPointDefect` as alias, `ConstantPointDefect`/`PointDefectSublattice`/`PointDefectedPhase` behind deprecation shims);
  import everything else from `landau.phases.pointdefects`.
  The sublattice ABC's three concrete models are siblings, not a subclass chain:
  - `PointDefectSublattice` — exact, `L = ln(1+s)`.
  - `LowTemperatureExpansionSublattice` — dilute, `L = s`, unnormalised site fractions.
  - `CappedPointDefectSublattice` — exact below `s* = m/(1-m)` (`m = max_site_fraction`), a line phase at total site fraction `m` above, and `phi_capped ≥ phi_exact` everywhere.
- `asewrapper.py`: `AsePhase(AbstractLinePhase)` wrapping `ase.thermochemistry.ThermoChem`;
  Helmholtz, falling back to Gibbs at 1 atm;
  `atoms_per_formula` normalises per atom;
  equality/hash by `pickle.dumps(thermochem)`.
- `quasiharmonic.py`: `PhonopyQuasiHarmonicPhase(AbstractLinePhase)` evaluates `min_V [E(V) + F_vib(V, T)]` from phonopy `ThermalProperties` at any `T` instead of fitting a sampled `F(T)`.
  Mutates the `temperatures` of the phonopy objects it is given.
  `eos` is an `Interpolator` (default `PolyFit(8)`, fitted in a centred scaled volume coordinate, fixed parameter count — per-T selection would invent steps in `F(T)`) or one of `"vinet"`/`"birch_murnaghan"`/`"murnaghan"`.
  Volumes with a mode below `min_frequency = -0.05` THz are dropped (`DynamicalInstabilityWarning`);
  beyond the sampled volumes the minimisation clamps (`EosExtrapolationWarning`) unless `extrapolate=True` with a closed-form eos.
  Volume selection is the pure helper `_select_stable_volumes`, testable without phonopy.

### `landau/interpolate/`

- `basic.py`: `Interpolator` (factory, `fit(x, y) → Interpolation`) and `Interpolation` (callable fitted curve with `deriv() → Interpolation`);
  type-only subclasses `TemperatureInterpolator` / `ConcentrationInterpolator`.
  `deriv()` is analytic for `PolyFit` (`PolynomialInterpolation`, `coefficients` in ascending order), `SGTE` (`SGTEInterpolation`) and `RedlichKister` (`RedlichKisterInterpolation`);
  everything else is closure-backed (`_CallableInterpolation`) and falls back to `NumericalDerivative`.
  Also `SplineFit` (wraps `UnivariateSpline`, `smoothing=0` interpolates, `degree` clamped to `len(x)-1`), `StitchedFit` (sorts its input, `edge` clamped), `G_calphad`, `_scalarize`.
  `RedlichKister.fit` is one `numpy.linalg.lstsq` on the mixing basis (under-determined → minimum norm, terminals only → the chord).
- 2-D: `SurfaceInterpolator.fit(T, c, H) → FittedSurface`, `FittedSurface.slice_at(T) → Interpolation`.
  `CalphadSurface2DInterpolator` (in `basic.py`) is poly-in-T Redlich–Kister in c and needs c=0 and c=1 terminal phases.
  Its terminal T-model is `terminal_interpolator=`, default `SGTE(4)`.
  Pass `WhitneyTemperatureInterpolator()` when evaluating below the fitted T-window: SGTE's entropy is unbounded as `T → 0` and fabricates spurious low-T liquid fields.
- `softplus.py`: `SoftplusFit` (1-D) and `SoftplusSurface2DInterpolator`, fitting `H(T,c) = off(T) + Σ softplus(α_i(T))·softplus(b_i(T)·(cn + c_i(T)))` — non-negative amplitudes make every fixed-T slice convex (no spurious miscibility gaps);
  the slope `b_i` is a polynomial in 1/T because wells sharpen on cooling.
  The coupled fit is multimodal: two per-slice seeds (coldest and central T, knee placed at the curvature peak after removing a best-fit line) race through a short polish and only the faster-converging one is finished.
  Solver: `lm` on scipy ≥ 1.16, trust-region with `x_scale='jac'` otherwise (older `lm` disabled MINPACK's variable scaling);
  robust `loss` or `knee_bounds` force trust-region, `method='lm'` with either raises.
  `knee_bounds` requires `c_order=0`.
- `whitney.py`: `WhitneyTemperatureInterpolator` (exported, constant-entropy continuation outside the data, an upper bound on a concave `F(T)`), the internal `WhitneyRBFInterpolator`, and `WhitneySurface2DInterpolator` (proposed for deprecation, #531 — slow, scale-dependent in T, invents negative curvature in c).
- Every new 1-D interpolator gets a `CASES` entry in `tests/unit/interpolate/test_shape_contract.py`;
  a new `TemperatureInterpolator` must also be covered in `tests/regression/test_issue_428.py`, whose guard test fails otherwise.

### `landau/calculate.py`

- `calc_phase_diagram(phases, Ts, mu=..., refine=True, keep_unstable=False, ...)` builds the frame `(T, mu, phase, c, phi, stable, border, refined, locus, boundary_id, …)`.
  `refine` is `True` (defaults), `False`, or a `Sequence[Refiner]` (an empty list refines nothing).
  `locus` (`landau.features.Locus`: `interior`/`boundary`/`triple`/`congruent`) says what part of the diagram a row lies on;
  grid samples and the synthetic `±inf` frame-edge rows stay `interior`.
  With `keep_unstable=True` the frame also has `dphi = phi - min(phi)` over rows sharing `(T, mu)`.
  `f_excess` is `f` minus the chord between the stable pure-component free energies at the row's own `T`, evaluated `_TERMINAL_KT_FAR = 40` kT past the sampled μ window;
  the column is omitted when the phases don't reach both c=0 and c=1 (then `plot_excess_free_energy` raises).
- `refine_phase_diagram(df, phases, refiners=None)` runs the refiners;
  `_drop_terminal_triples` drops a terminal transition that coincides with a triple point of the same phases (one invariant, not two).
- `guess_mu_range(phases, T, samples, tolerance=1e-2)`: scans `c(mu)` on `(-10, 10)` with `scipy.optimize.brute(..., finish=None)` and inverts it onto an even `c` grid.
  `finish=None` matters: the default `fmin` polish is unconstrained and walks far outside the bracket because `c(mu)` saturates only asymptotically.
- `get_transitions(df)` pairs border rows at one `(mu, T)` into two-phase tie-lines (`_tie_lines`);
  select invariants by `locus == "triple"`, not by phase-label shape.
- `cluster_T_c` / `cluster_T_c_mu` / `cluster`: agglomerative clustering;
  `distance_threshold` is required here.
  The user-facing `cluster_phase` / `get_polygons` default it to `0.2` (0.5 merged distinct same-phase fields, #456).

### `landau/refine.py`

- `Refiner` ABC: `propose(df)` → candidates, `solve(cand, phases)` → `RefinedPoint` / `RefinedMiscibilityGap` (both `RefinedBase`: `phase_names`, `to_rows`, `emitted_concentrations`).
  `run()` drops `_dominated` and negative-T results and assigns `boundary_id` (one per coexistence line).
- `_dominated(pt, phases)` drops a point when an own phase is absent (`phi = +inf`), an outside phase undercuts `max(own_phi)`, or the own potentials spread more than `_COEXIST_TOL = 1e-5` eV.
  Applies to two-phase points too: `phi` is continuous across a first-order boundary (`c` jumps), and every two-phase refiner root-finds `phi_1 - phi_2 = 0`.
- Refiners:
  - `ScanRefiner(by=...)` — 1-D bisection;
    recurses into both sub-brackets when a third phase dominates the crossing, so a stable window narrower than the grid still yields its two transitions.
  - `DelaunayLineRefiner` (opt-in) and `DelaunayTripleRefiner` work on `_Simplex` numpy views from `_delaunay_simplices(df)`, which memoises the tessellation on the most recent frame (weakref identity, assumes the frame isn't mutated in place).
    A triple point is emitted only by its owning simplex (`_simplex_containment` over the candidate's siblings), so `solve` stays pure.
  - `ClausiusClapeyronRefiner` / `MiscibilityGapRefiner` share `_CCBase`, a predictor-corrector trace of a coexistence line in both T directions (keyword-only `dT_max=5.0, dT_min=1.0, dc_max=0.01, dc_min=0.0, max_steps=500`, `dc_max` is applied last and wins).
    The bootstrap bracket widens by the Clausius–Clapeyron seed slope (`_seed_slope`) for steep lines.
    The trace aborts the moment a point is `_dominated`;
    at that abort `_dominated_node` (overridden by `MiscibilityGapRefiner`) solves the monotectic/monotectoid/syntectic invariant, which names the gap phase twice and which `DelaunayTripleRefiner` (counting distinct phases per simplex) can never propose.
    `RefinedPoint.concentrations` keeps the two branches of one phase distinct.
    Per line,
    `_tag_features` tags closures: `ClausiusClapeyronRefiner(congruent_tol=0.1)` finds where `|c1 - c2| ≤ tol · max(gap)` (relative — closures have no absolute scale) and solves the congruent point exactly in `(T, mu)`;
    a closure that belongs to a pure component (`_terminal_closure`, decided by reach, not composition) is left to `TerminalRefiner`.
    `MiscibilityGapRefiner` bisects the consolute point to `closure_xtol=1e-3` K using `_gap_jump` and emits it as a congruent `RefinedMiscibilityGap`;
    `_lines` splits one phase's separate domes.
  - `TerminalRefiner` — T-scan at c = 0 and c = 1 at effectively infinite μ (`_KT_FAR = 40` kT past the window, only phases within `_TERMINAL_C_TOL = 1e-6` of the terminal compete);
    emits `congruent` rows at `mu = ±inf` with `phi = NaN`.
- `default_refiners(df)`: 2-D grid → `DelaunayTripleRefiner`, `ClausiusClapeyronRefiner`, `MiscibilityGapRefiner`, `TerminalRefiner`;
  1-D scan → the matching `ScanRefiner`.

### `landau/plot/`

- `__init__.py`: `plot_phase_diagram` (c–T) and `plot_mu_phase_diagram` (μ–T) share `_plot_phase_diagram`;
  `plot_1d_mu_phase_diagram` / `plot_1d_T_phase_diagram` share `_plot_1d_phase_diagram(df, scan_col, ...)`;
  `plot_excess_free_energy` (needs a `keep_unstable=True` frame, one axes for a single T, a FacetGrid otherwise);
  `get_polygons`, `plot_polygons`, `get_phase_colors`, `cluster_phase`.
- `labels.py`: every text-placement helper.
  Imports go one way — `labels` never imports from `plot`.
- `style.py`: `theme`, a `Theme` dataclass with `rc` (a validating `mpl.RcParams` subclass) and one dict per drawn element (`phase_boundary`, `metastable`, `tie_line`, `transition_guide`, `transition_tick`, `invariant_marker`, `reference_line`, `region_label`, `curve_label`, `value_label`, `label_outline`) carrying colour, style and weight only;
  sizes come from `lines.linewidth`/`lines.markersize`/`font.size`.
  Values derived from fields and rcParams come from `Theme` methods (`stability_dashes`, `reference_line_kwargs`, `line_phase_markersize`);
  `docs/api/plot.md` tables every element field with its default, and a test keeps the table in sync with `Theme()`.
  `styled()` is a context manager and decorator applying `theme.rc`;
  `_plot_phase_diagram`, `_plot_1d_phase_diagram` and `plot_excess_free_energy` carry `@styled()`, so `theme.rc` wins over the caller's rcParams for its keys.
  Call sites read element styles at draw time and merge them last over appearance kwargs (`color`, `zorder`), never over placement kwargs.
  `theme.rc` carries only keys that differ from matplotlib's defaults, since a default-valued key would only override the caller.
  On style conflicts the excess-free-energy plot follows the 1d/2d look.
- 2d options: `legend`/`inline_legend` (inline labels at the polygon's pole of inaccessibility, with rotate/offset fallbacks),
  `triplepoints=None` (follows `transition_temperatures` unless set, `tielines=` is a deprecated alias),
  `transition_temperatures=False` (labels `triple`/`congruent` rows — reads the `locus` tags, so unrefined frames draw nothing),
  `color_by=<column>` with `cmap`/`norm`/`colorbar` (each polygon filled from its own phase's stable rows only, rasterized in vector output, with `inline_legend=False, legend=True` it raises).
- 1d: segments split at each `stable` flip (`_assign_segment_ids`), dashed branches bridged to the exact transition rows;
  `reference_phase=` subtracts a phase's `phi` and raises unless that phase is sampled at every grid point (needs `keep_unstable=True`).
- Colours are keyed by **sorted** phase name via `get_phase_colors` (cycles the palette past 10 phases), so a phase keeps its hue across plots;
  row order is not a usable key.
- Axis and colorbar text comes from the `_column_label(column, element)` table.

### Other modules

- `poly.py`: point cloud → polygon per phase region.
  `Concave`, `Segments`, and with the TSP extras `PythonTsp`/`SegmentPythonTsp`/`FastTsp`/`SegmentFastTsp`;
  `handle_poly_method` falls back `segment-fasttsp → segment-tsp → fasttsp → tsp → concave` by what's installed.
  `make` returns a region's exact outline (a `LineString` for a line phase);
  `apply` widens only regions narrower than `min_c_width` everywhere (`_thicken`) and then subtracts neighbours' exact outlines (`_trim_overlaps`), so boundaries sit on the data and eutectic apexes stay sharp.
  `_segment_tsp_polygon` rotates the tour off an intra-segment edge before walking it (else a valid tour yields a self-intersecting ring).
  fast-tsp ≥ 0.2 rejects distances above 65535, hence `_integer_distances`.
- `resample.py`: `resample_borders`, `RandomlyShiftedPhase` (bootstrap border resampling, draws from numpy's global RNG, #516).
- `features.py`: `Locus` `StrEnum`.
  Deliberately not re-exported from `landau`.
- `fleche.py`: fleche digest hooks, registered as the `fleche` entry point in `pyproject.toml`;
  landau never imports fleche.
  Hooks only cover types fleche would otherwise reject;
  hooking a type fleche already digests would make digests depend on entry-point load order.
  Closure-backed fits raise `Indigestible` (two fits share one code object).
  Per-instance `_hash` values are plain attributes, not dataclass fields, so dataclass-walking digests don't fold in Python's per-process hash salt.

## Tests

- `tests/unit/` mirrors the package (`interpolate/`, `phases/`, `plot/` subdirectories);
  `tests/regression/` pins bugs by issue number;
  `tests/conftest.py` holds the shared fixtures (`two_phase_ideal`, `three_phase_regular_solution`, `eutectic_phases`, `eutectic_diagram`).
- `tests/unit/phases/test_broadcast_contract.py` pins the broadcast contract of every `Phase` (`semigrand_potential`, `concentration`, `free_energy`, `line_free_energy`);
  a new phase gets a `CASES` entry there, and per-class tests check values, not shapes.
- `tests/integration/test_border_coverage.py` is the smoke test for "a polygon amputated a phase region" (every `poly_method` × c-T/μ-T).
- `tests/integration/testplots.py` is **not** a pytest test: it renders PNGs for visual review.
  The `testplot` PR label or an `@testplot ...` comment renders them in CI (`.github/scripts/testplot_args.py` maps free text to flags with Haiku) and posts them on the PR.
  Inline-attach a render when a PR changes plotting.
- `.hypothesis/` is local state;
  never commit it.

## Docs

Sphinx in `docs/` (myst-nb, furo), built by ReadTheDocs on Python 3.12;
notebooks under `notebooks/` are included and must carry executed outputs.

## Open design directions

Fetch the issue before re-litigating;
check open PRs before starting anything adjacent.

- #116 — umbrella for small refactors and direct tests of private helpers;
  check its open sub-issues before filing new ones.
- #137 — split `phases/__init__.py` (line vs solution vs interpolating phases).
- #332, #138 — let a phase take its free energy as a callable instead of a sample-and-refit round-trip;
  TDB import builds on it.
- #34, #60 — 2.0 plotting/calculate API rearrangement (axes as arguments, general 1d cuts).
- #39 — entropy/enthalpy methods on phases;
  blocks the CALPHAD-side ideas in #337/#339 (lever-rule plots, equal-entropy criterion).
- #344 — Au–Cu CEF as one partitioning phase;
  blocked on `MiscibilityGapRefiner` scan cost.
- #427 — SGTE is wrong for QHA/phonon `F(T)` (`S` diverges as `T → 0`);
  use `PolyFit(8)` or `PhonopyQuasiHarmonicPhase`.
- #534 — anchor classical `F(T)` at the relaxed 0 K energy;
  maintainer prefers it as `SGTE` behaviour when a `T = 0` sample is present, not a new class.
- #443 — nuclear quantum correction for classical-MD solids via the QHA (idea only).
- #531 — deprecate `WhitneySurface2DInterpolator`.
- #463 — transition-temperature labels can drift too far from their invariant to attribute.
- #81 (analytic SRO models), #33 (fast Legendre transforms), #59 (autodiff `concentration`), #62 (`src/` layout).

## Non-goals and settled decisions

- Dropping pandas 2.
- Underscored module names.
- Boundary-Voronoi polygons were rejected in favour of trimming against neighbours' outlines.
