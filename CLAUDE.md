# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**landau.py** is a Python library for calculating thermodynamic equilibria and plotting phase diagrams in the (semi-)grand ensemble. It enables computational thermodynamics research, particularly for alloy systems.

## Working Style (read this first)

These are project-specific preferences distilled from past PR/issue feedback. Some restate harness defaults but are repeated here because they have been violated often enough to matter.

### Tone in PRs, commits, and comments
- Match a terse, imperative tone. No "Happy to...", no apologies, no enthusiasm filler. (Restates the harness default — keep it in mind anyway.)
- PR bodies are plain technical reports. No **bold** "headline numbers", no "Conclusion" sections, no `[Fix this →]` action links, no marketing claims ("dramatic speedup", "significantly improves").
- Do not invoke the maintainer's name or authority in PR bodies or comments (PR #94: "don't use my name in vain").

### Evidence over claims
- Cite the specific commit hash, file path, test count, or command output backing any claim. If a number appears in the PR body, the script that produced it goes in `benchmarks/` in the same PR (PR #94).
- For physics/numerics claims, prefer saying nothing over saying something unverified (PR #112: "Rather say nothing than something wrong").
- If asked to verify, actually verify — read the docs, run the code. Do not re-assert. (PR #113: "I'm not entirely sure I believe you. Read the docs and get back to me.")

### Scope and abstraction
- Prefer the minimal change. "I'd rather merge less features now and extend later than break API" (PR #68).
- Before writing a new helper, check the codebase and existing deps (numpy, scipy, shapely, matplotlib.testing, pyiron_snippets) for one that already does the job. Reuse over reimplement (PR #115).
- One purpose per PR. Notebooks, benchmarks, refactors, and unrelated fixes go in their own PRs (PR #68, #82, #85). Notebooks are committed with executed outputs only.
- When a PR grows a second concept, propose splitting it before pushing more.
- Speculative or "might be useful" work gets parked, not landed (PR #129 closed, #130 parked).

### Test quality bar
(Expands on the Testing section below.)
- Tests must assert tight conditions a degenerate/constant solution would fail. Loose `atol=0.05` "it ran" tests get rejected (PR #80, #101).
- Use Hypothesis for round-trip recovery (random coeffs in → `fit()` → coeffs out) where applicable (PR #82).
- Share a single `atol` constant across related tests rather than scattering magic numbers (PR #82).
- Use the module-wide pytest skip marker for optional-dep tests (PR #111).

### Debugging numerics visually
- When diagnosing a fit/solver/physics issue, generate a plot of the relevant landscape (loss vs parameter, etc.), attach it inline to the PR comment, and add a walkthrough section to a notebook so the maintainer can step through interactively (PR #82, #115).

### Naming and layout
- Module names: no underscores. `asewrapper`, not `ase_wrapper` (PR #68).
- Promote single-file modules to subpackages once they grow (as was done for `interpolate/`).

### Do not commit
- `.hypothesis/`, `_version.py`, stray top-level scripts, duplicate exploratory files (PR #71, #94). Check for duplicates of what you are about to add before pushing.

### Pandas 2/3 compatibility (hard constraint)
- All `groupby().apply()` calls must pass `include_groups=False`.
- Code must work on both pandas 2 and 3. Do not drop pandas 2 (PR #93, #113).
- The fix in commit `945f850` replaced deprecated `include_groups=True`; regressions are pinned in `tests/regression/test_single_group_apply.py` and `test_cluster_phase_pandas3.py`.

## Development Setup

- Python `>=3.11,<3.14` (`pyproject.toml`).
- Install: `pip install -e .[test,constraints,fast-tsp,python-tsp]`. Optional extras: `ase`, `docs`, `python-tsp`, `fast-tsp`, `constraints` (`polyfit`).
- Lint: Ruff, line length 120 (`pyproject.toml [tool.ruff]`).
- TSP / polyfit / ASE deps gate behind `pyiron_snippets.import_alarm.ImportAlarm`, so importing the relevant module without the extra raises only at instantiation.

## Testing

```bash
pytest                                          # full suite
pytest tests/unit/test_calculate.py             # one file
pytest tests/unit/test_calculate.py::test_foo   # one test
pytest -k pattern                               # by name
```

Layout:
- `tests/unit/` — per-module: `test_calculate.py`, `test_phases.py`, `test_poly.py`, `test_refine.py`, `test_resample.py`, `test_softplus.py`, `test_whitney.py`; subdirs `interpolate/` (`test_g_calphad.py`, `test_polyfit.py`, `test_redlich_kister.py`, `test_sgte.py`, `test_softplus_fit.py`, `test_stitched_fit.py`), `phases/` (`test_ase.py`, `test_vectorize.py`), `plot/` (`test_axis.py`, `test_colors.py`, `test_excess_free_energy.py`, `test_polygons.py`).
- `tests/regression/` — `test_issue_1.py`, `test_issue_51.py`, `test_pandas_groupby.py`, `test_single_group_apply.py`, `test_cluster_phase_pandas3.py`.
- `tests/integration/testplots.py` — **not a pytest test**; a render script that writes phase-diagram PNGs to `tests/integration/_plots/`. Triggered by the `testplot` label (`.github/workflows/testplot.yml`; Haiku also auto-selects a plot subset from changed files — PR #152) or by `@testplot ...` comments (`testplot-mention.yml`, parser uses Haiku to map free-text to `--only`/`--poly-method`/`--tielines`/`diff` flags against an allow-list frozen on `main`; see PR #149 for the parse/render/publish security split, PR #150 for diff-against-main rendering). Images are pushed to the `testplots` gallery branch and inlined into a PR comment.
- `tests/conftest.py` provides shared `two_phase_ideal` / `three_phase_regular_solution` phase fixtures.
- Uses **Hypothesis** for property-based tests. Optional-dep tests guard the import with `with ImportAlarm() as alarm:` then a module-level `pytestmark = pytest.mark.skipif(alarm.message is not None, ...)` when a whole file depends on one extra (see `tests/unit/phases/test_ase.py`); files mixing independent extras use per-test `@pytest.mark.skipif` against `try/except`-set flags instead (see `tests/unit/test_poly.py`). Do not use `try/except ImportError` as the gate by itself (PR #111).

## Architecture

### Module layout

**`landau/phases/`** — phase definitions (subpackage; split from `phases.py` per PR #111). `landau/__init__.py` re-exports only the user-facing subset: `LinePhase`, `TemperatureDepandantLinePhase`/`TemperatureDependentLinePhase`, `IdealSolution`, `RegularSolution`, `InterpolatingPhase`, `SlowInterpolatingPhase`, `AsePhase` (the `Phase`/`AbstractLinePhase` ABCs and the point-defect classes are subpackage-only).
- `phases/__init__.py` — `Phase` (ABC), `AbstractLinePhase`, `LinePhase`, `TemperatureDependentLinePhase` (alias `TemperatureDepandantLinePhase` kept for back-compat), `IdealSolution`, `RegularSolution`, `InterpolatingPhase`, `SlowInterpolatingPhase`, `AbstractPointDefect`, `ConstantPointDefect`, `PointDefectSublattice`, `PointDefectedPhase`.
- `phases/asewrapper.py` — `AsePhase(AbstractLinePhase)` wrapping `ase.thermochemistry.ThermoChem`. Compares/hashes by a `_key()` tuple `(name, fixed_concentration, pressure, atoms_per_formula, pickle.dumps(thermochem))` so two instances from equivalent inputs are equal. Falls back from `get_helmholtz_energy` to `get_gibbs_energy` (1 atm default). `atoms_per_formula` divides ASE's energy so the result is per atom. Splitting this further is open (#137).

**`landau/interpolate/`** — interpolation strategies.
- `basic.py` — `Interpolator` (ABC), `TemperatureInterpolator`, `ConcentrationInterpolator`, `PolyFit`, `SGTE`, `RedlichKister` (+ `RedlichKisterInterpolation` helper), `StitchedFit`, `G_calphad` standalone fn.
- `softplus.py` — `SoftplusFit` smooth-step.
- `whitney.py` — `WhitneyRBFInterpolator` (sklearn-style estimator) and `WhitneyTemperatureInterpolator`. Used for Whitney-extension RBF interpolation that handles convex-hull projection.

**`landau/calculate.py`** — core thermodynamic calculations.
- `calc_phase_diagram(phases, Ts, mu=..., refine=True, keep_unstable=False, ...)` — main entry point that builds the raw `(T, mu, phase, c, phi)` dataframe. `keep_unstable=True` retains unstable-phase rows (required by `plot_excess_free_energy`).
- `refine_phase_diagram(pdf, phases, min_c, max_c)` — orchestrator over a sequence of `Refiner` instances (defaults via `landau.refine.default_refiners`).
- `guess_mu_range(phases, T, samples, tolerance=1e-2)` — autodetect the μ window with two `scipy.optimize.brute` scans (μ at min/max concentration), then `interp1d`-invert c(μ) onto `samples` points; returns `(mu_array, c0, c1)`.
- `cluster_T_c` / `cluster_T_c_mu` / `cluster` — agglomerative clustering for collapsing co-located transitions; `distance_threshold` is a **required** kwarg here. User-facing `cluster_phase` (`plot.py`) and `get_polygons` accept it with `default=0.5`.
- `get_transitions(df)` — extract phase-boundary rows.
- Private `_join_phase_unit` / `_split_phase_unit` shared with `poly.py` / `plot.py` (PR #132).

**`landau/refine.py`** — phase-boundary refinement strategies (PR #115 introduced the `Refiner` API).
- `Refiner` ABC: subclasses implement `propose(df)` + `solve(candidate, phases)` → `RefinedPoint` / `RefinedMiscibilityGap` (the module's exported result dataclasses). Base `run(df, phases)` drops dominated/negative-T rows and packages output rows.
- Shipped refiners: `ScanRefiner` (1-D bisection between disagreeing samples), `DelaunayLineRefiner` (one transition per two-phase simplex), `DelaunayTripleRefiner` (triple points from three-phase simplices; dedups via enclosing-simplex extents, PR #144), `ClausiusClapeyronRefiner` (predictor-corrector trace of a two-phase coexistence line, both T directions), `MiscibilityGapRefiner` (same idea for intra-phase miscibility splits). The last two share `_CCBase`.
- `default_refiners(df)` picks the set based on which of `mu`/`T` is sampled. For 2-D grids (both axes sampled) the default is `DelaunayTripleRefiner` + `ClausiusClapeyronRefiner` + `MiscibilityGapRefiner` — `DelaunayLineRefiner` and `ScanRefiner` are opt-in. For 1-D scans the default is just the matching `ScanRefiner`.
- A `boundary_id` column tagging refined rows by the line they belong to is in flight (PR #163, closes #124).

**`landau/plot.py`** — phase-diagram plotting.
- `plot_phase_diagram` (c–T), `plot_mu_phase_diagram` (μ–T), `plot_1d_mu_phase_diagram`, `plot_1d_T_phase_diagram`. The mu/c variants share `_plot_phase_diagram` + the `_set_axis_for` helper (PR #122). Generalising the 1D/2D split is open (#34, #60).
- `plot_excess_free_energy(df, col_wrap=3, convex_hull=True, ...)` — common-tangent / excess-free-energy plot, one facet per T, via seaborn `relplot`; needs a `calc_phase_diagram(..., keep_unstable=True)` frame. `convex_hull=True` draws stable phases as solid curves, metastable as faded, and overlays black common-tangent segments (one per `mu` group). Self-contained — no module-level helpers. Re-exported from `landau/__init__.py`.
- `get_polygons(df, poly_method=..., distance_threshold=0.5)` / `plot_polygons` / `get_phase_colors(phase_names, override=None)`.
- `cluster_phase(df)` is the pandas-groupby site that has historically broken on pandas 3 — keep `include_groups=False` (PR #93, #113).

**`landau/poly.py`** — point-cloud → matplotlib polygon conversion (per phase region).
- `AbstractPolyMethod` (dataclass, `min_c_width=0.01`); concrete: `Concave`, `Segments`. With optional `python-tsp`/`fast-tsp` extras, `PythonTsp` / `SegmentPythonTsp` / `FastTsp` / `SegmentFastTsp` register themselves.
- `handle_poly_method(poly_method, **kwargs)` resolves a string or `AbstractPolyMethod` against the active registry; the default falls back through `segment-fasttsp → segment-tsp → fasttsp → tsp → concave` based on what's installed.
- `_trim_overlaps` (called from `AbstractPolyMethod.apply`) lives here too — symmetric buffer overlap trimming, guarded against shapely `GEOSException`. (The eutectic-apex sliver is a known cosmetic issue — see Design themes for the rejected fixes.)
- `_greedy_stitch` — module-level helper extracted from `Segments._sort_segments` (a7ca752); the greedy nearest-endpoint stitch, pending the `_segment_tsp_polygon` rewire (#162).

**`landau/resample.py`** — bootstrap-style border resampling (`resample_borders`, `RandomlyShiftedPhase`).

### Key design patterns

- **Frozen-dataclass + ABC** for all `Phase` / `Interpolator` / `Refiner` types — instances are immutable, structurally hashable.
- **Semi-grand potential** is the central thermodynamic quantity; equilibrium = argmin over phases.
- **Interpolator strategy** plugged into `InterpolatingPhase` / `SlowInterpolatingPhase` decouples the analytical model (SGTE, RedlichKister, PolyFit, Softplus, Whitney) from the phase wrapper.
- **Refiner strategy** plugged into `refine_phase_diagram` decouples "where might a transition live?" from "find the exact location".

## Documentation

Sphinx in `docs/`: `index.md`, `installation.md`, `api.md` + per-module `api/{calculate,interpolate,phases,plot,poly,resample}.md`, `notebooks.rst`, `conf.py`, and `docs/notebooks` (a **symlink** to `../notebooks`). Built by ReadTheDocs on Python 3.12. Extensions: `myst-nb`, `sphinx-autodoc-typehints`, `furo`. Notebooks under `notebooks/` (`Basics`, `IdealSolution`, `Intermetallics`, `ClausiusClapeyron`, `ExcessFreeEnergy`, `PointDefects`, `Toy`, `ASE/{EMT_CuAg,FCC_BCC_Fe,Hydrate}`, `MgCa/MgCa`); commit notebooks **with executed outputs only**.

Local build:
```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

## Design themes / open scope (issue tracker)

Cheat sheet of what's been considered. Fetch the issue before re-litigating.

**Active design themes**
- **Phase subpackage cleanup** (#137) — `phases/__init__.py` lumps everything together; split is desired.
- **Refiner extensions** (#124 boundary_id — PR #163 open, not yet merged; #142 closed via #144 triple-point dedup). `ClausiusClapeyronRefiner` and `MiscibilityGapRefiner` are on by default for 2-D grids; `DelaunayLineRefiner` and the 1-D `ScanRefiner` remain opt-in. Open bug: not all triple points are caught with `tielines=True` (#32, reproducer in the Intermetallics notebook).
- **Polygon plotting robustness** (#125 decided in favour of plain symmetric subtract — PR #127 landed; both the boundary-Voronoi alternative #129 and apex-cleanup follow-up #130 were rejected. #38 closed by #147). Hypothesis strategies for polygon tests are weak (#70). Recent sub-issues under the #116 Refactor Opportunities umbrella: rewire `Segments._sort_segments` through `_segment_tsp_polygon` so the no-extras fallback no longer fails when a phase is stable at an axis edge (#162); add direct unit tests for the `_pca_sort_segment` / `_segments_from_labels` / `_segment_tsp_polygon` helpers (#161); pin pandas-3 single-group `groupby.apply` shape for `get_transitions`, `AbstractPolyMethod.apply`, and the `f_excess` `.T`-workaround branch in `calc_phase_diagram` (#160).
- **Excess-free-energy / common-tangent plot** (#136) — `plot_excess_free_energy` (self-contained, seaborn `relplot`) landed in `landau/plot.py` with the `ExcessFreeEnergy` notebook and `tests/unit/plot/test_excess_free_energy.py`; re-exported from `landau/__init__.py`. Bug #155 (`f_excess` reference picks the wrong endpoint phase when phases overlap near c=0/c=1) is still open; fix proposed in PR #166 (per-phase endpoint reference, not yet merged).
- **Pandas 2/3 compat** — hard constraint. Every `groupby().apply()` needs `include_groups=False`. Recent regressions covered in `tests/regression/test_single_group_apply.py` (PR #143).
- **API generalisation** (#34, #60) — plot_{mu,}_phase_diagram axes-as-arg refactor and the broader 2.0 plotting/calculate rearrangement.
- **Performance** (#23) — `SlowInterpolatingPhase` needs a precomputed interpolation pass; (#33) fast Legendre transforms; autodiff for concentration methods is a stretch goal (#59).
- **Repo layout** (#62) — flat → `src/` layout is wanted.
- **Phase extras** — analytic SRO models still open (#81; PR #123 prototype `QuasiChemicalPhase` was closed without merging). Entropy/enthalpy methods on phases still open (#39). TDB-file import scoping in progress (#138).

**Decisions already landed (don't redo)**
- `landau/phases.py` split into a `landau/phases/` subpackage with `asewrapper.py` (PR #111, squashed replacement of #68).
- Refiner strategy (PR #115), `_set_axis_for` extraction (PR #122), `_join_phase_unit` DRY (PR #132).
- Plain symmetric buffer trimming landed (PR #127); both the boundary-Voronoi alternative (PR #129) and apex-cleanup follow-up (PR #130) were **rejected**.
- `testplot` label + `@testplot` mention workflow (PR #103, #148, #149, #150, #152). PR-controlled code is sandboxed away from repo write secrets; label workflow Haiku-picks the plot subset from changed files (#152); mention parser supports `diff` against main (#150).

**Out-of-scope / explicit non-goals**
- Dropping pandas 2 support — *do not* (PR #93, #113).
- Speculative / "might be useful" work parks rather than lands (PR #129).
- Underscored module names; use `asewrapper`, not `ase_wrapper` (PR #68).

## Common Tasks

### Add a new phase type
1. Subclass `Phase` (or `AbstractLinePhase`) as a `@dataclass(frozen=True)` in `landau/phases/__init__.py` (or a new module under `landau/phases/`).
2. Implement `semigrand_potential(T, dmu)` and `concentration(T, dmu)`.
3. Add to `landau/phases/__init__.py:__all__` *and* re-export from `landau/__init__.py` if it's user-facing.
4. Tests: at minimum a `tests/unit/test_phases.py` case checking scalar/array input shapes plus a tight numerical assertion (loose `atol=0.05` "it ran" tests get rejected; see Working Style).

### Add an interpolation method
1. New class in `landau/interpolate/basic.py` (or a new module) inheriting `Interpolator` / `TemperatureInterpolator` / `ConcentrationInterpolator`.
2. Implement the interpolation interface.
3. Export from `landau/interpolate/__init__.py:__all__`.
4. Add Hypothesis round-trip tests (random coeffs in → `fit()` → coeffs out) under `tests/unit/interpolate/` (PR #82).

### Add a refiner
1. New `Refiner` subclass in `landau/refine.py` with `propose` + `solve`.
2. Add to `landau/refine.py:__all__`.
3. Decide whether it belongs in `default_refiners` (most don't; opt-in unless cheap).
4. Tests in `tests/unit/test_refine.py` against a small known-transition fixture; assert exact rows, not just non-empty.

### Render the visual-review plots
```bash
python tests/integration/testplots.py --only 2d_basics 2d_toy --poly-method fasttsp
```
Output: `tests/integration/_plots/*.png`. Inline-attach the PNG to a PR comment when diagnosing a plotting change.

## Citation

```
@article{poul2025automated,
    title = {Automated generation of structure datasets for machine learning potentials and alloys},
    journal = {npj Computational Materials},
    author={Poul, Marvin and Huber, Liam and Neugebauer, Jörg},
    volume = {11}, number = {1}, pages = {174}, year={2025},
    doi = {10.1038/s41524-025-01669-4},
}
```
