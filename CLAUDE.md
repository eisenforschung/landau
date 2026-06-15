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
- Keep the PR description (the OP) in sync with the diff. When review feedback or later commits invalidate something in the body, edit the body — never leave a stale claim standing (e.g. a dropped export still described as exported, an outdated test count).

### Comments and docstrings
- Reflect only the current state of the code. Historical motivation is acceptable (why a threshold exists, why an approach was chosen), but never narrate the intermediate steps, rejected alternatives, or "old 10% vs new 1%" comparisons that arose during a PR. Those belong in the PR description, not in the source.
- Do not reference thresholds, approaches, or behaviour that no longer exist in the code.

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

### Git workflow
- Merge commits are disabled on GitHub; never bring a branch up to date with `git merge`. Rebase onto `origin/main` instead (`git reset` off any stray merge, then `git rebase origin/main`) and force-push, so history stays linear.

### Do not commit
- `.hypothesis/`, `_version.py`, stray top-level scripts, duplicate exploratory files (PR #71, #94). Check for duplicates of what you are about to add before pushing.

### Pandas 2/3 compatibility (hard constraint)
- All `groupby().apply()` calls must pass `include_groups=False`.
- Code must work on both pandas 2 and 3. Do not drop pandas 2 (PR #93, #113).
- The fix in commit `945f850` replaced deprecated `include_groups=True`; regressions are pinned in `tests/regression/test_single_group_apply.py` and `test_cluster_phase_pandas3.py`.

## Development Setup

- Python `>=3.11,<3.14` (`pyproject.toml`).
- Install: `pip install -e .[test,constraints,fast-tsp,python-tsp]`. Optional extras: `ase`, `docs`, `python-tsp`, `fast-tsp`, `constraints` (`polyfit`).
- Lint: Ruff, line length 120 (`pyproject.toml [tool.ruff]`). There is no `.flake8` in the repo currently.
- TSP / polyfit / ASE deps gate behind `pyiron_snippets.import_alarm.ImportAlarm`, so importing the relevant module without the extra raises only at instantiation.
- `shapely>=2.1` is required because `AbstractPolyMethod.make` calls `make_valid(method="structure", keep_collapsed=False)` to recover the enclosed polygon from a self-intersecting ring (PR #221). The minimum-dependency CI (`.github/workflows/test-minimum-deps.yml`, PR #222) installs every direct dep at its declared floor via `uv pip install --resolution lowest-direct` and runs the suite, so the lower bounds are exercised on every PR.

## Testing

```bash
pytest                                          # full suite
pytest tests/unit/test_calculate.py             # one file
pytest tests/unit/test_calculate.py::test_foo   # one test
pytest -k pattern                               # by name
```

Layout:
- `tests/unit/` — per-module: `test_calculate.py`, `test_phases.py`, `test_poly.py`, `test_refine.py`, `test_resample.py`, `test_softplus.py`, `test_whitney.py`; subdirs `interpolate/` (`test_g_calphad.py`, `test_polyfit.py`, `test_redlich_kister.py`, `test_sgte.py`, `test_softplus_fit.py`, `test_stitched_fit.py`), `phases/` (`test_ase.py`, `test_vectorize.py`), `plot/` (`test_1d_plots.py`, `test_axis.py`, `test_colors.py`, `test_excess_free_energy.py`, `test_polygons.py`).
- `tests/regression/` — `test_issue_1.py`, `test_issue_51.py`, `test_issue_155.py`, `test_pandas_groupby.py`, `test_single_group_apply.py` (also pins the single-T `f_excess` branch in `calc_phase_diagram` after #160 was closed by `dac7d73`), `test_cluster_phase_pandas3.py`, `test_excess_endpoint_reference.py` (pins the tangent-based endpoint reference per PR #231: stable terminal line phase has `f_excess = 0` to `atol = 1e-12`; metastable one matches the true gap).
- `tests/integration/test_border_coverage.py` — full-pipeline check that every finite stable border point from `calc_phase_diagram` lies within 2% of the axis ranges of its phase's polygon outline; parametrized over every `poly_method` × c-T/μ-T axis pair (PR #221). `Segments × μ-T` is xfailed (min-x head-heuristic limitation noted in `Segments._sort_segments`). This is the smoke test for "polygon doesn't amputate a phase region".
- `tests/integration/testplots.py` — **not a pytest test**; a render script that writes phase-diagram PNGs to `tests/integration/_plots/`. Triggered by the `testplot` label (`.github/workflows/testplot.yml`; Haiku also auto-selects a plot subset from changed files — PR #152) or by `@testplot ...` comments (`testplot-mention.yml`, parser uses Haiku to map free-text to `--only`/`--poly-method`/`--tielines`/`diff` flags against an allow-list frozen on `main`; see PR #149 for the parse/render/publish security split, PR #150 for diff-against-main rendering). The Haiku call sites and shared helpers (plot-list loader, JSON validation, `run_haiku`) all live in `.github/scripts/testplot_args.py` (subcommands `parse` for mentions and `select` for the label workflow). The Haiku CLI runs as a pure text→JSON transform (PR #212: `--system-prompt replace`, `--tools ""`, isolated cwd) and the prompt's plot list is built from `testplots.py` docstrings (`PLOT_LIST`) so keys never go stale. `run_haiku` retries once and falls back to "render everything" on a final JSON parse failure. Images are pushed to the `testplots` gallery branch and inlined into a PR comment.
- `tests/conftest.py` provides shared `two_phase_ideal` / `three_phase_regular_solution` phase fixtures.
- Uses **Hypothesis** for property-based tests. Optional-dep tests use a module-level `pytestmark = pytest.mark.skipif(ImportAlarm(...).message is not None, ...)` when a whole file depends on one extra (see `tests/unit/phases/test_ase.py`); files mixing independent extras use per-test `@pytest.mark.skipif` against `try/except`-set flags instead (see `tests/unit/test_poly.py`). Do not use `try/except ImportError` as the gate by itself (PR #111).

## Architecture

### Module layout

**`landau/phases/`** — phase definitions (subpackage; split from `phases.py` per PR #111). Re-exported from `landau/__init__.py`.
- `phases/__init__.py` — `Phase` (ABC), `AbstractLinePhase`, `LinePhase`, `TemperatureDependentLinePhase` (alias `TemperatureDepandantLinePhase` kept for back-compat), `IdealSolution`, `RegularSolution`, `InterpolatingPhase`, `SlowInterpolatingPhase`, `AbstractPointDefect`, `ConstantPointDefect`, `PointDefectSublattice`, `PointDefectedPhase`. Private `_scalarize(x)` helper (PR #208) is the single site that collapses 0-d numpy results back to Python scalars so the scalar-in/scalar-out contract holds across `AbstractLinePhase.concentration`, `IdealSolution.semigrand_potential`, `RegularSolution.semigrand_potential`, and `(Slow)InterpolatingPhase._find_phi_c` — five sites that previously hand-rolled the same `if ndim == 0` / `if shape == ()` / `isinstance(x, np.ndarray)` check and had drifted.
- `phases/asewrapper.py` — `AsePhase(AbstractLinePhase)` wrapping `ase.thermochemistry.ThermoChem`. Compares/hashes by `pickle.dumps(thermochem)` so two instances from equivalent inputs are equal. Falls back from `get_helmholtz_energy` to `get_gibbs_energy` (1 atm default). `atoms_per_formula` divides ASE's energy so the result is per atom. Splitting this further is open (#137).

**`landau/interpolate/`** — interpolation strategies.
- `basic.py` — `Interpolator` (ABC), `TemperatureInterpolator`, `ConcentrationInterpolator`, `PolyFit`, `SGTE`, `RedlichKister` (+ `RedlichKisterInterpolation` helper), `StitchedFit`, `G_calphad` standalone fn.
- `softplus.py` — `SoftplusFit` smooth-step.
- `whitney.py` — `WhitneyRBFInterpolator` (sklearn-style estimator) and `WhitneyTemperatureInterpolator`. Used for Whitney-extension RBF interpolation that handles convex-hull projection.

**`landau/calculate.py`** — core thermodynamic calculations.
- `calc_phase_diagram(phases, Ts, mu=..., refine=True, ...)` — main entry point that builds the raw `(T, mu, phase, c, phi)` dataframe. The per-phase `f_excess` endpoint reference compares candidate phases by the tangent-line value at the pure concentrations (`phi` at c=0, `phi + mu` at c=1) instead of `f` at the extreme sample (PR #231, follow-up to #155/#166) — `f(c)` is convex with slope `mu`, so a phase ending at `c = 1 - eps` lies below its own tangent value at `c = 1` by `~mu*eps` and would otherwise steal the reference from a line phase sitting exactly at `c = 1`, lifting that phase's `f_excess` off zero. Line phases located exactly at an endpoint are unchanged (both reduce to `f`).
- `refine_phase_diagram(pdf, phases, min_c, max_c)` — orchestrator over a sequence of `Refiner` instances (defaults via `landau.refine.default_refiners`).
- `guess_mu_range(phases, T, samples, tolerance=1e-2)` — autodetect mu sampling window via `scipy.optimize.brute` grid scan plus local refinement.
- `cluster_T_c` / `cluster_T_c_mu` / `cluster` — agglomerative clustering for collapsing co-located transitions; `distance_threshold` is a **required** kwarg here. User-facing `cluster_phase` (`plot.py`) and `get_polygons` accept it with `default=0.5`.
- `get_transitions(df)` — extract phase-boundary rows.
- Private `_join_phase_unit` / `_split_phase_unit` shared with `poly.py` / `plot.py` (PR #132).

**`landau/refine.py`** — phase-boundary refinement strategies (PR #115 introduced the `Refiner` API).
- `Refiner` ABC: subclasses implement `propose(df, phases)` + `solve(candidate)` → `RefinedPoint` / `RefinedMiscibilityGap`. Base `run()` drops dominated/negative-T rows and packages output rows.
- Shipped refiners: `ScanRefiner` (1-D bisection between disagreeing samples), `DelaunayLineRefiner` (one transition per two-phase simplex), `DelaunayTripleRefiner` (triple points from three-phase simplices; dedups via enclosing-simplex extents, PR #144), `ClausiusClapeyronRefiner` (predictor-corrector trace of a two-phase coexistence line, both T directions), `MiscibilityGapRefiner` (same idea for intra-phase miscibility splits). The last two share `_CCBase`.
- `default_refiners(df)` picks the set based on which of `mu`/`T` is sampled. For 2-D grids (both axes sampled) the default is `DelaunayTripleRefiner` + `ClausiusClapeyronRefiner` + `MiscibilityGapRefiner` — `DelaunayLineRefiner` and `ScanRefiner` are opt-in. For 1-D scans the default is just the matching `ScanRefiner`.
- A `boundary_id` int column tags refined rows by the coexistence line they belong to; `_CCBase.run()` assigns one id per `_pair_key`, base `Refiner.run()` assigns one id per non-empty solve call. Landed in PR #163, closes #124.

**`landau/plot.py`** — phase-diagram plotting.
- `plot_phase_diagram` (c–T), `plot_mu_phase_diagram` (μ–T), `plot_1d_mu_phase_diagram`, `plot_1d_T_phase_diagram`. The mu/c variants share `_plot_phase_diagram` + the `_set_axis_for` helper (PR #122). Generalising the 1D/2D split is open (#34, #60). 2d variants take `inline_legend=True` (PR #199): when set, the legend box is dropped and each polygon is labelled in place at the pole of inaccessibility (`shapely.ops.polylabel`, coordinates pre-normalised by the axis data-ranges) with a white-outlined black text via the shared `_text_with_background` helper; the same helper renders the 1d top-spine labels. `_add_inline_polygon_labels` falls back in two pixel-space steps when the horizontal label box does not fit its polygon (PR #228): rotate by 90° at the pole, then offset horizontally beside the polygon (left-flipped at the right axis edge, clamped into the axes both ways) so terminal line phases at c=0/c=1 stay visible. Off-polygon labels are no longer constrained to their own polygon, so a final pass via `_group_overlapping_intervals` + `_spread_labels` fans labels with overlapping horizontal extents apart vertically within the axes (PR #230). Both 1d functions assign per-segment `_seg_id`s via `_assign_segment_ids(df, scan_col)`: ordering by the cut axis (`'mu'` / `'T'`), a phase starts a new segment at each `stable` flip, so a phase unstable in two disjoint windows draws as separate dashed branches instead of one line through the stable region. Threshold-free and grid-density independent (PR #179, closed #177). `scan_col` is a parameter so it extends to generalised 1d cuts (#34, #60). Before drawing, `_bridge_unstable_segments(df, scan_col)` duplicates each refined transition (`border` + `stable`) row onto the adjacent unstable `_seg_id`s so a phase's dashed branch runs flush to the exact transition instead of starting at the next (possibly coarse) sample; the duplicate feeds `sns.lineplot` only, annotations still use the un-bridged frame. Both 1d functions also accept `reference_phase=None` (PR #189) — when set, subtracts that phase's `phi` along the cut via `_subtract_reference_phase` and updates the y-label. The seaborn hue/style legend is replaced by `_add_1d_phase_legend` (top-spine ticks at transitions + right-end labels per phase, PR #190); colours are stashed on `ax._landau_phase_colors`. `top_labels`/`side_labels` flags toggle each element.
- `plot_excess_free_energy(df, col_wrap=3, convex_hull=True, inline_legend=True, ...)` — common-tangent / excess-free-energy plot; needs a `calc_phase_diagram(..., keep_unstable=True)` frame. **Multi-T** fans out into a seaborn `relplot` FacetGrid (one facet per T; `height`/`aspect`/`col_wrap` apply only here, and `col_wrap` is clamped to the number of temperatures — PR #186, closed #183). **Single-T** draws a plain `sns.lineplot` onto `plt.gca()` and returns that axes (PR #226) — the caller controls figure allocation, `height`/`aspect` are ignored, no title is set. `convex_hull=True` draws stable phases as solid curves, metastable as faded, and overlays black common-tangent segments (one per `mu` group). `inline_legend=True` (PR #206) drops the figure legend and labels each curve in place via `_add_inline_curve_labels`: anchors solution-phase labels above the largest continuous stable c-region, line-phase labels below their dot, reuses `_text_with_outline` + `_spread_labels` + `_bold_math` so the mathtext subscripts come out bold and overlapping labels fan apart. `inline_legend=False` restores the previous figure legend. Re-exported from `landau/__init__.py`.
- `get_polygons(df, poly_method=..., distance_threshold=0.5)` / `plot_polygons` / `get_phase_colors(phase_names, override=None)`.
- `cluster_phase(df)` is the pandas-groupby site that has historically broken on pandas 3 — keep `include_groups=False` (PR #93, #113).

**`landau/poly.py`** — point-cloud → matplotlib polygon conversion (per phase region).
- `AbstractPolyMethod` (dataclass, `min_c_width=0.01`); concrete: `Concave`, `Segments`. With optional `python-tsp`/`fast-tsp` extras, `PythonTsp` / `SegmentPythonTsp` / `FastTsp` / `SegmentFastTsp` register themselves.
- `handle_poly_method(poly_method, **kwargs)` resolves a string or `AbstractPolyMethod` against the active registry; the default falls back through `segment-fasttsp → segment-tsp → fasttsp → tsp → concave` based on what's installed (PR #224 added the pinning tests, closed #173).
- `AbstractPolyMethod.make` validates the returned shape and, on `is_valid=False`, repairs via `shapely.make_valid(shape, method="structure", keep_collapsed=False)` (PR #221, requires `shapely>=2.1`) — the `"structure"` algorithm returns the enclosed polygon directly instead of the `GeometryCollection` that the default `"linework"` algorithm emits for self-intersecting rings (which the old repair branch dropped, hiding solid phases — Intermetallics c-T diagram).
- `AbstractPolyMethod._trim_overlaps` (called from `.apply`) does symmetric buffer overlap trimming (PR #127), guarded against shapely `GEOSException` (PR #147). The apex-residual cleanup follow-up (PR #130) and the boundary-Voronoi alternative (PR #129) were both closed without merging — the eutectic-apex sliver is a known cosmetic issue.
- `_segment_tsp_polygon(segments, solve_tour)` (shared by `SegmentPythonTsp` / `SegmentFastTsp`) rotates the cyclic tour off its intra-segment edge before walking it — without this, the solver was free to return a rotation whose ends fall inside a segment, and the first-encounter reconstruction would walk that segment the wrong way and produce a self-intersecting ring out of an optimal tour (PR #227, closing a follow-up to #221). The exact-DP fallback added to `SegmentPythonTsp` in #221 was removed by #227 since the bug was reconstruction-side, not solver-side. `PythonTsp` keeps its `retries` ladder (re-solves with a 10× larger iteration budget when the resulting polygon is invalid) since that *is* genuine solver underconvergence.

**`landau/resample.py`** — bootstrap-style border resampling (`resample_borders`, `RandomlyShiftedPhase`).

### Key design patterns

- **Frozen-dataclass + ABC** for `Phase` and concrete `Interpolator` subclasses (`PolyFit`, `SGTE`, `RedlichKister`, `StitchedFit`, `SoftplusFit`) — instances are immutable, structurally hashable. `Refiner` concrete subclasses are plain mutable classes (they accumulate state across `propose`/`solve` calls); `AbstractPolyMethod` is a non-frozen dataclass.
- **Semi-grand potential** is the central thermodynamic quantity; equilibrium = argmin over phases.
- **Interpolator strategy** plugged into `InterpolatingPhase` / `SlowInterpolatingPhase` decouples the analytical model (SGTE, RedlichKister, PolyFit, Softplus, Whitney) from the phase wrapper.
- **Refiner strategy** plugged into `refine_phase_diagram` decouples "where might a transition live?" from "find the exact location".

## Documentation

Sphinx in `docs/` (`api/`, `notebooks/`, `index.md`, `installation.md`). Built by ReadTheDocs on Python 3.12. Extensions: `myst-nb`, `sphinx-autodoc-typehints`, `furo`. Notebooks under `notebooks/` (`Basics`, `IdealSolution`, `Intermetallics`, `ClausiusClapeyron`, `ExcessFreeEnergy`, `PointDefects`, `Toy`, `ASE/{EMT_CuAg,FCC_BCC_Fe,Hydrate}`, `MgCa/...`); commit notebooks **with executed outputs only**.

Local build:
```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

## Design themes / open scope (issue tracker)

Cheat sheet of what's been considered. Fetch the issue before re-litigating.

**Active design themes**
- **Phase subpackage cleanup** (#137) — `phases/__init__.py` lumps everything together; split is desired.
- **Refiner extensions** — `ClausiusClapeyronRefiner` and `MiscibilityGapRefiner` are on by default for 2-D grids; `DelaunayLineRefiner` and the 1-D `ScanRefiner` remain opt-in. Open bug: not all triple points are caught with `tielines=True` (#32, reproducer in the Intermetallics notebook).
- **Polygon plotting robustness** (#125 decided in favour of plain symmetric subtract — PR #127 landed; both the boundary-Voronoi alternative #129 and apex-cleanup follow-up #130 were rejected. #38 closed by #147). Hypothesis strategies for polygon tests are weak (#70).
- **#116 Refactor Opportunities umbrella** — open sub-issues:
  - #210: move `DelaunayTripleRefiner`'s `_found` dedup state out of `solve()` into a thin `run()` override — every other refiner keeps `solve()` pure, so the current shape silently turns into a no-op on a second call with the same candidate and trips up subclassers who only override `solve()`.
  - #211: direct unit tests for `_plot_tielines` refined vs unrefined branches (still carries `# FIXME` / `# TODO` from the maintainer; coverage is only incidental via `plot_phase_diagram(tielines=True)`).
  - #162 (`Segments._sort_segments` rewire) closed deferred — next major release is expected to consolidate and drop several poly-methods, so the rework was parked.
  - #160 closed by `dac7d73`; #173 closed by PR #224 (see landed list).
- **Pandas 2/3 compat** — hard constraint. Every `groupby().apply()` needs `include_groups=False`. Recent regressions covered in `tests/regression/test_single_group_apply.py` (PR #143).
- **API generalisation** (#34, #60) — plot_{mu,}_phase_diagram axes-as-arg refactor and the broader 2.0 plotting/calculate rearrangement.
- **Performance** (#23) — `SlowInterpolatingPhase` needs a precomputed interpolation pass; (#33) fast Legendre transforms.
- **Repo layout** (#62) — flat → `src/` layout is wanted.
- **Phase extras** — analytic SRO models still open (#81; PR #123 prototype `QuasiChemicalPhase` was closed without merging). Entropy/enthalpy methods on phases still open (#39). TDB-file import scoping in progress (#138).

**Decisions already landed (don't redo)**
- `landau/phases.py` split into a `landau/phases/` subpackage with `asewrapper.py` (PR #111, squashed replacement of #68).
- Refiner strategy (PR #115), `_set_axis_for` extraction (PR #122), `_join_phase_unit` DRY (PR #132).
- `boundary_id` int column tagging refined rows by coexistence line (PR #163, closed #124). `_CCBase.run()` assigns one id per `_pair_key`, base `Refiner.run()` one id per non-empty `solve()` call. `RefinedPoint`/`RefinedMiscibilityGap` carry `boundary_id: int = 0`.
- Plain symmetric buffer trimming landed (PR #127); both the boundary-Voronoi alternative (PR #129) and apex-cleanup follow-up (PR #130) were **rejected**.
- `testplot` label + `@testplot` mention workflow (PR #103, #148, #149, #150, #152). PR-controlled code is sandboxed away from repo write secrets; label workflow Haiku-picks the plot subset from changed files (#152); mention parser supports `diff` against main (#150). Parse/select and their shared helpers are all in `.github/scripts/testplot_args.py`. Haiku robustness rebuilt in two passes: PR #181 added an AST-based plot-name reader + retry/JSON-fallback, and PR #212 finally closed #180 by running the Haiku CLI as a pure text→JSON transform (`--system-prompt replace`, `--tools ""`, isolated cwd) and deriving the prompt's plot list from `testplots.py` docstrings so keys can never drift.
- 1d phase-diagram segmentation by stability flips, not distance clustering (PR #179, closed #177). `_assign_segment_ids(df, scan_col)` is threshold-free and grid-density independent; `_auto_threshold` / `_max_normed_gap` removed.
- 1d plot legend rewritten as top-spine ticks + right-end phase labels (PR #190, closed #188), with `reference_phase` y-axis subtraction (PR #189, closed #187) and `top_labels`/`side_labels` toggles (both default `True`). `plot_excess_free_energy` legend now appears when only line phases are stable (PR #191, closed #182); `col_wrap` clamped to len(temperatures) (PR #186, closed #183).
- `plot_excess_free_energy` `f_excess` reference now picks the most stable endpoint phase per group (PR #166, closed #155); per-phase reference selection with 1% concentration-span tolerance.
- `cluster_T_c` / `cluster_T_c_mu` / `cluster` no longer take the unused `eps` kwarg (closed #171); `distance_threshold` is the only live control.
- `handle_poly_method('concave', ratio=…)` no longer raises `TypeError`; the resolver pops `ratio` then falls back to `alpha` then `Concave.ratio` (PR #174, closed #172). Precedence is **ratio over alpha**.
- Direct unit tests for `_split_stable`, `_border_edges`, `reduce`, and the `cluster` dispatcher landed in `tests/unit/test_calculate.py` (PR #170).
- Direct unit tests for `_pca_sort_segment` / `_segments_from_labels` / `_segment_tsp_polygon` landed (PR #194, closed #161).
- 2d phase-diagram inline polygon labels via `polylabel`, sharing the white-stroke text helper with 1d top-spine labels (PR #199).
- `plot_excess_free_energy` inline curve labels (`inline_legend=True`, default-on) via `_add_inline_curve_labels`, reusing `_text_with_outline` / `_spread_labels` / `_bold_math` from the 1d/2d work (PR #206, with a follow-up placement cleanup in PR #213).
- `_scalarize(x)` private helper in `landau/phases/__init__.py` deduplicates five drifted copies of the 0-d numpy → Python scalar collapse (PR #208).
- Dead COM-angle pre-sort removed from `Segments._sort_segments`; `_greedy_stitch`'s `min(x_col)` head pick is the only live ordering (PR #214, closed #209). Live heuristic pinned by a new direct test in `tests/unit/test_poly.py`.
- `testplot` Haiku selector emits JSON, not prose (PR #212, closed #180). Stops the select stage from falling back to "render all plots"; plot list derived from `testplots.py` docstrings.
- Polygon repair recovers self-intersecting rings via `make_valid(method="structure", keep_collapsed=False)` (PR #221) — fixes the missing solid phase in the Intermetallics c-T diagram. Minimum `shapely` bumped to 2.1. `tests/integration/test_border_coverage.py` smoke-tests every `poly_method` × cT/μT pair.
- Segment-TSP polygons no longer self-intersect when the solver returns a rotation cut through an intra-segment edge (PR #227, follow-up to #221). `_segment_tsp_polygon` rotates the tour off its intra-segment edge before walking it; the exact-DP fallback added in #221 was removed in #227 along with `SegmentPythonTsp.retries`/`exact_node_limit`.
- `plot_excess_free_energy` single-T path draws onto `plt.gca()` via plain `sns.lineplot` and returns that axes; multi-T still uses `relplot`/FacetGrid (PR #226). `height`/`aspect` apply to the facet grid only; caller owns figure allocation for the single-T case.
- `handle_poly_method(None, ...)` default fall-through chain pinned by tests in `tests/unit/test_poly.py` (PR #224, closed #173). Each test monkeypatches `poly.__all__` to strip TSP registrations and verifies `SegmentFastTsp → SegmentPythonTsp → Concave` selection independently of installed extras.
- Single-T `f_excess` branch in `calc_phase_diagram` switched from `len(Ts) > 1` + `.T` workaround to the same `isinstance(fex, pd.DataFrame)` + `stack().reset_index()` shape used by `get_transitions` (`dac7d73`, closes #160). Regression pinned in `tests/regression/test_single_group_apply.py`.
- Minimum-dependency CI (`.github/workflows/test-minimum-deps.yml`, PR #222) installs every direct dep at its declared floor via `uv pip install --resolution lowest-direct -e ".[test,constraints,fast-tsp,python-tsp]"` and runs the suite, so the lower bounds are exercised on every PR. `uv pip install` is used instead of `uv sync` because the universal lock would mask real minimums by pulling deps up through cross-extra constraints.
- 2d inline polygon labels rotate to 90° and offset off the polygon when the horizontal pole label does not fit (PR #228) — covers thin line-phase regions and their terminal-axis edge cases. The `make_valid` repair in `_largest_inscribed_circle_center` was factored into `_shapely_polygon` so the pixel-space fit test reuses it.
- Off-polygon inline labels fan apart vertically when their horizontal extents overlap, via new `_group_overlapping_intervals` + the existing `_spread_labels` (PR #230, follow-up to #228). Single-stack clamping into the axes vertically falls out of the same call.
- `calc_phase_diagram` `f_excess` endpoint reference compares by tangent value at the pure concentrations (`phi`, `phi + mu`) instead of raw `f` at the extreme sample (PR #231, closes follow-up to #155/#166). Pinned in `tests/regression/test_excess_endpoint_reference.py`.

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
