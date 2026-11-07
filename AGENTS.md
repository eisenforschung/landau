# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, Aider, ...) working **on**
landau. If you are *using* landau to compute a phase diagram, read
[`skills/landau/SKILL.md`](skills/landau/SKILL.md) instead.

This file records conventions and contracts, not events. Release notes live in
[`CHANGELOG.md`](CHANGELOG.md), history in `git log`, and open work in the issue
tracker. Check those first (`gh issue list`, `gh pr list`, `gh issue view N`);
never rely on a copy of their state in this file.

## What landau is

A library for thermodynamic equilibria and phase diagrams of binary systems in
the (semi-)grand-canonical ensemble. Each phase provides a semi-grand potential
`φ(T, Δμ)` and a concentration `c(T, Δμ) = −∂φ/∂Δμ`.

`calc_phase_diagram` samples a `(T, Δμ)` grid, marks the phase with the lowest
`φ` as stable, and then refines (root-finds) the boundaries and invariants. It
returns one pandas DataFrame, and every plotting and analysis function reads it.

Units are eV/atom, K and eV. `c` is the fraction of the second species.

The design is dataclass-heavy with pluggable strategies: `Phase`, `Interpolator`
(1-D), `SurfaceInterpolator` (2-D), `Refiner`, and `AbstractPolyMethod` (point
cloud to polygon).

## Commands

```bash
pip install -e .[test,phonopy,constraints,fast-tsp,python-tsp]   # extras used in CI
pytest                                                   # full suite
pytest tests/unit/test_calculate.py::test_foo            # one test; -k pattern by name
pytest -m fleche                                         # digest-hook tests; needs .[test-fleche]
ruff check <files you touched>                           # lint, line length 120 (not run in CI)
sphinx-build -b html docs docs/_build/html               # docs; needs .[docs]
python tests/integration/testplots.py --only 2d_basics   # render review PNGs to tests/integration/_plots/
```

- Python `>=3.11,<3.14`.
- CI runs `pytest -m "not fleche"` and, in a separate job, `pytest -m fleche`.
  Ruff is configured but not enforced. Lint what you touch and do not reformat
  untouched code.
- Extras: `test`, `constraints`, `fast-tsp`, `python-tsp`, `ase`, `phonopy`,
  `docs`, `test-fleche`.
- A minimum-dependency CI job installs every direct dependency at its declared
  floor. If you raise or lower a floor, that job is what checks it.

## Repo map

| path | contents |
|---|---|
| `landau/phases/__init__.py` | `Phase`, `AbstractLinePhase`, `LinePhase`, `TemperatureDependentLinePhase`, `IdealSolution`, `SlowInterpolatingPhase`, `FastInterpolatingPhase`, `Surface2DInterpolatingPhase`, the deprecated `RegularSolution`/`InterpolatingPhase`, `S(c)`, `kB` |
| `landau/phases/{pointdefects,asewrapper,quasiharmonic}.py` | point-defect sublattices and `PointDefectedPhase`; `AsePhase`; `PhonopyQuasiHarmonicPhase` |
| `landau/interpolate/` | `basic.py` (ABCs, `PolyFit`, `SplineFit`, `SGTE`, `RedlichKister`, `StitchedFit`, `CalphadSurface2DInterpolator`), `softplus.py`, `whitney.py` |
| `landau/calculate.py` | `calc_phase_diagram`, `refine_phase_diagram`, `guess_mu_range`, `get_transitions`, clustering helpers, the `f_excess` reference |
| `landau/refine.py` | `Refiner` ABC; `ScanRefiner`, `DelaunayLineRefiner`, `DelaunayTripleRefiner`, `ClausiusClapeyronRefiner`, `MiscibilityGapRefiner`, `TerminalRefiner`; `default_refiners`; `RefinedPoint`/`RefinedMiscibilityGap` |
| `landau/plot/` | `__init__.py`: public plotters and polygon/colour helpers; `labels.py`: all label placement (it never imports from `plot`) |
| `landau/poly.py` | `Concave`, `Segments`, optional TSP-based methods; `handle_poly_method` |
| `landau/features.py` | `Locus` StrEnum (`interior`/`boundary`/`triple`/`congruent`), not re-exported from the root |
| `landau/resample.py` | `resample_borders`, `RandomlyShiftedPhase` (bootstrap error bars) |
| `landau/fleche.py` | digest hooks loaded by fleche's entry point; landau never imports fleche |
| `tests/unit/`, `tests/regression/`, `tests/integration/` | unit tests mirror module paths; regression tests are named after the issue they pin; `integration/testplots.py` is a render script, not a test |
| `benchmarks/` | the script behind every number quoted in a PR |
| `notebooks/` | Sphinx-rendered examples, committed with executed outputs |

## Contracts (break these only on purpose, as a breaking change)

**Phases**
- A `Phase` is a `@dataclass(frozen=True)` with an ABC base. It implements
  `semigrand_potential(T, dmu)` and `concentration(T, dmu)`. Line phases
  implement `line_concentration` (a property) and `line_free_energy(T)`.
- Scalar in gives scalar out, and arrays broadcast. Collapse 0-d results with
  `_scalarize` (in `interpolate/basic.py`).
- `c` must equal `−∂φ/∂Δμ`. Tests should check this numerically.
- Solution phases: `add_entropy=False` means the input free energies already
  contain the ideal mixing entropy. It is stripped (`+T·S(c)`) before the fit and
  restored afterwards. `add_entropy=True` means the inputs lack it and landau
  adds it. The name reads backwards to users, so keep docstrings explicit about it.
- `calc_phase_diagram` keys phases by `name`.

**Interpolators**
- An `Interpolator` is a frozen dataclass: `fit(x, y) → Interpolation`.
- An `Interpolation` is callable and has `deriv()`. The default `deriv()` is a
  numerical central difference; closed forms override it. `FastInterpolatingPhase`
  uses the first derivative.
- Every 1-D interpolator must be in the `CASES` of
  `tests/unit/interpolate/test_shape_contract.py`. Every exported
  `TemperatureInterpolator` must be exercised through `calc_phase_diagram`;
  `tests/regression/test_issue_428.py` fails otherwise.
- A `SurfaceInterpolator` provides `fit(T, c, H) → FittedSurface`, and a
  `FittedSurface` provides `slice_at(T) → Interpolation`. `H` is
  entropy-stripped in the same way as for solution phases.

**Refiners**
- `propose(df)` yields candidates, and `solve(cand, phases)` returns a
  `RefinedPoint` or `RefinedMiscibilityGap` (or nothing).
- `solve` must be a pure function of its arguments.
- `Refiner.run()` drops points that `_dominated` rejects: an own phase absent,
  an outside phase lower than the highest own `φ`, or own potentials spread wider
  than `_COEXIST_TOL = 1e-5` eV. `φ` is continuous across a first-order boundary;
  only `c` jumps.
- `default_refiners(df)`:
  - 2-D grid: `DelaunayTripleRefiner`, `ClausiusClapeyronRefiner`,
    `MiscibilityGapRefiner`, `TerminalRefiner`.
  - 1-D scan: the matching `ScanRefiner`.
  - Adding to the defaults changes every user's output, so it needs a strong
    reason and a benchmark.
- The tracer constructors (`_CCBase` subclasses) are keyword-only.

**Output frame**
- Columns: `T, phase, phi, mu, c, stable, locus, border, refined, boundary_id, f`,
  plus `f_excess` when the phases reach c = 0 and c = 1, plus `dphi` with
  `keep_unstable=True`.
- Users cache these frames and filter on the columns. Changing a column's
  meaning, or which rows carry a given `locus`, is a user-visible change: say so
  in the commit (`feat!`/`fix!` or a `BREAKING CHANGE:` footer when it is one)
  so it reaches the changelog.
- `calc_phase_diagram(mu=...)`: an iterable is used as given; a **non-zero int**
  means "guess that many samples with `guess_mu_range` at `max(Ts)`"; any other
  scalar is a fixed Δμ. Keep this documented wherever `mu` is described.

**Public API and deprecation**
- User-facing classes are re-exported from `landau/__init__.py`.
- `Locus` is deliberately not re-exported from the root, to keep
  `landau.features` as a namespace that can grow. `WhitneyTemperatureInterpolator`
  and `StitchedFit` are not re-exported either; whether they should be is open.
- Deprecate functions and keywords with `pyiron_snippets` `@deprecate`. A class
  cannot use the decorator without breaking `isinstance`, so it warns from
  `__post_init__` (see `_warn_deprecated_phase`).
- Everything deprecated is removed at 2.0. Do not remove anything earlier.

**Dependencies and environment**
- pandas 2 and 3 are both supported (dropping pandas 2 is out of scope). Every
  `groupby().apply()` passes `include_groups=False`; single-group cases go
  through `calculate._apply_series`.
- `shapely>=2.1` is required for `make_valid(method="structure")` in
  `AbstractPolyMethod.make`.
- Optional dependencies are gated with `pyiron_snippets.import_alarm.ImportAlarm`:
  - Tests for a whole file use
    `pytestmark = pytest.mark.skipif(ImportAlarm(...).message is not None, ...)`.
  - Files that mix extras use per-test `skipif` on a flag set in `try/except`.
  - A bare `try/except ImportError` is not an acceptable gate.
- landau never imports fleche. Hooks live in `landau/fleche.py`, which is loaded
  only through the `fleche` entry point. Closure-backed fits (Whitney, Stitched,
  Softplus, Spline interpolations) are intentionally indigestible.
- Module names have no underscores (`asewrapper`, not `ase_wrapper`). Promote a
  module to a subpackage when it grows.

## Numerics worth knowing before you touch them

The docstrings are canonical; this is the map.

- `FastInterpolatingPhase` finds the basin on a 201-point grid, then takes 6
  Newton steps in `u = log(c/(1−c))`. The ideal-entropy term is linear in `u`,
  so the steps are well conditioned. It is accurate to ~1e-6 in `c`, except
  within ~1e-9 eV of a miscibility gap's coexistence Δμ (#563).
  `SlowInterpolatingPhase` (brute force) is the reference oracle.
- `f_excess` is referenced against the most stable pure component at each row's
  `T`, evaluated `40 kT` past the sampled Δμ window (`_TERMINAL_KT_FAR`).
  `TerminalRefiner` shares that convention. The column is omitted when the
  phases don't reach both ends.
- `guess_mu_range` scans `(-10, 10)` eV with `brute(..., finish=None)`. The
  `finish=None` is load-bearing: the default polish walks out of the bracket.
- The Clausius-Clapeyron tracer step is clipped to `[dT_min, dT_max]`, then the
  `dc_min` floor applies, then the `dc_max` cap applies last. The order matters.
- Delaunay tessellations are memoised on the most recent frame (weakref
  identity), which assumes frames are not mutated in place within a refine pass.
- `WhitneyTemperatureInterpolator` continues at constant entropy outside the
  data, which is an upper bound on F. Its RBF solve is dense, O(N²) in memory.
  `SGTE`'s entropy diverges as T → 0: that is wrong for QHA data (#427) and
  extrapolates badly for classical data (#534).
- `TemperatureDependentLinePhase` stores a precomputed `_hash` as a plain
  attribute, *not* a dataclass field, so dataclass-walking digests stay
  process-independent.
- Polygons: self-intersecting rings are repaired with
  `make_valid(method="structure")`. `distance_threshold=0.2` (in `cluster_phase`
  and `get_polygons`) sits in a safe band of about 0.01–0.3 on the fixtures;
  see `tests/regression/test_issue_456.py`.

## Working style (maintainer's house rules)

- **Tone.** Terse and imperative in commits, PRs and comments. No marketing,
  apologies, bold headline numbers, "Conclusion" sections or action links.
  Do not invoke the maintainer's name or authority.
- **Evidence over claims.**
  - Cite a commit, path, test or command output for every claim.
  - A number in a PR body comes with its script in `benchmarks/` in the same PR.
  - For physics or numerics, say nothing rather than something unverified.
  - When asked to verify, run or read; do not restate.
- **Comments and docstrings describe the current code only.** Rationale is
  welcome; the history of the PR (rejected alternatives, old-versus-new numbers)
  belongs in the PR body.
- **Minimal change, one purpose per PR.** Notebooks, benchmarks, refactors and
  fixes go in separate PRs. Reuse numpy, scipy, shapely, matplotlib.testing and
  pyiron_snippets before writing a helper. Park speculative work rather than
  landing it. Propose a split before a PR grows a second concept.
- **Tests assert tight conditions** that a degenerate or constant answer would
  fail; an `atol=0.05` "it ran" test is rejected.
  - Use Hypothesis round-trips (coefficients in, `fit()`, coefficients out) for
    fits.
  - Share one tolerance constant across related tests.
  - Pin each fixed bug in `tests/regression/`.
  - Mutation-check new direct tests: break the code and confirm the test fails.
- **Debug numerics visually.** Plot the landscape (loss against a parameter, a
  free-energy curve) and attach the PNG to the PR. For plotting changes, render
  `testplots.py` before and after.
- **Keep the PR body in sync with the diff.** Edit it when review invalidates a
  claim.
- **When referencing an issue** that the PR does not close, avoid the closing
  keywords (`closes`, `fixes`).

## Git and releases

- **Rebase onto `origin/main`, never merge.** Merge commits are disabled.
  Force-push the rebased branch.
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `chore:`,
  `refactor:`, `perf:`, with `!` or `BREAKING CHANGE:` for breaks) drive
  release-please. Messages that don't conform are invisible to the changelog.
- **Do not commit** `.hypothesis/`, `landau/_version.py`, stray top-level
  scripts, or duplicates of exploratory files. Notebooks are committed with
  executed outputs only.
- **Visual review:** the `testplot` label, or an `@testplot ...` comment, renders
  diagrams on a PR (`.github/workflows/testplot*.yml`). Read
  `.github/scripts/testplot_args.py` before changing that pipeline: PR-controlled
  code is sandboxed away from secrets there on purpose.

## Settled decisions (do not reopen without new evidence)

- Keep pandas 2 support (#93, #113).
- No underscored module names (#68).
- Speculative "might be useful" features are parked, not merged (#129).
- Polygon overlap: plain symmetric buffer trimming (#127). Boundary-Voronoi
  (#129) and apex clean-up (#130) were rejected; the eutectic-apex sliver is a
  known cosmetic issue.
- `FastInterpolatingPhase` replaces `SlowInterpolatingPhase` for users. Slow
  stays as the oracle, and `RegularSolution`/`InterpolatingPhase` go at 2.0 (#503).
- The fleche integration is entry-point hooks only. landau-side fleche helpers
  were declined (#459).
- Closed without merging: the CEF prototype stack (#324/#326/#346/#334, issue
  #344 still open), `IntermetallicPhase` (#306), the `PhaseDiagram` object
  interface (#250) and `QuasiChemicalPhase` (#123, issue #81 open).

## Long-running directions

Use these as pointers. Check each issue's current state before starting work.

- #116 refactor umbrella: split large functions and pin private helpers with
  direct tests. Sub-issues are filed in weekly cohorts.
- #137: further split `phases/__init__.py`.
- #62: `src/` layout.
- #34, #60: the 2.0 plot/calculate API (axes as an argument). The `Axes` refiner
  groundwork for pressure as a diagram axis builds toward it.
- #138, #332: TDB import/export, and a direct "callable" path for phase free
  energies.
- #39: entropy and enthalpy methods on phases.
- #427, #534, #443: low-T free energies (QHA, anchoring classical F(0) = E0,
  quantum corrections).
- #531: deprecate `WhitneySurface2DInterpolator`.
- #33, #59: fast Legendre transforms and autodiff. Stretch goals.

## Common tasks

- **New phase type.**
  1. Write a frozen dataclass subclassing `Phase` or `AbstractLinePhase`.
  2. Add it to `phases/__init__.py:__all__` (or put it in a new module under
     `landau/phases/`) and re-export it from `landau/__init__.py` if users need it.
  3. Test the scalar/array shape contract, a tight value check, and
     `c = −∂φ/∂Δμ`.
- **New interpolator.**
  1. Subclass `TemperatureInterpolator` or `ConcentrationInterpolator`, and
     return an `Interpolation` (override `deriv()` if a closed form exists).
  2. Export it from `interpolate/__init__.py`.
  3. Add a Hypothesis round-trip test, a `CASES` entry in the shape contract,
     and `calc_phase_diagram` coverage for T-interpolators.
- **New refiner.**
  1. Subclass `Refiner` with `propose` and `solve`, and add it to `__all__`.
  2. Leave it opt-in unless it is cheap and broadly right.
  3. Test it against a small fixture with a known transition, asserting exact
     rows (see `tests/conftest.py` for `eutectic_diagram` and friends).
- **Plot change.**
  - Label placement belongs in `plot/labels.py`.
  - Colours go through `get_phase_colors`, keyed by sorted phase name.
  - Render `testplots.py` and attach the before/after images.

## Keeping this file useful

- Record a convention or contract in the same PR that introduces or changes it.
- Do **not** record events here: merges, releases, open-PR lists, audit dates,
  test counts, file line counts, or per-PR histories. They go stale within days
  and are already in `git log`, `CHANGELOG.md` and the tracker.
- If a scheduled audit runs against this file, it should check that the
  contracts and commands above still hold against the code: run the commands,
  grep the named symbols. It should not mirror tracker state.
- `CLAUDE.md` only imports this file. Do not let the two diverge.
