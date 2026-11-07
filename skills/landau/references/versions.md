# landau versions: what changed between 1.11 and 1.18

Check `landau.__version__` before you trust an example. Releases are cut from
conventional commits, and the upstream `CHANGELOG.md` has every entry. These are
the changes that alter results or the API a user touches.

| version (date) | change that matters to users |
|---|---|
| 1.11.0 (2026-07) | `calc_phase_diagram(refine=[...])` takes an explicit refiner list; `Surface2DInterpolatingPhase` and the surface interpolators added |
| 1.11.1 (2026-08) | `WhitneyTemperatureInterpolator` returns scalars for scalar input. Before this it broke `calc_phase_diagram` ("inhomogeneous shape" / errors inside `brentq`). Also faster refiners |
| 1.12.0 (2026-09) | `PhonopyQuasiHarmonicPhase` (`phonopy` extra); digest hooks for the `fleche` caching library |
| 1.13.0 (2026-09) | `CalphadSurface2DInterpolator(terminal_interpolator=...)`; refined points that are not truly stable are dropped more robustly |
| 1.14.0 (2026-09) | `transition_temperatures=` on the 2-D plotters; `Locus.CONGRUENT` tagged by the refiner |
| 1.14.1 (2026-09) | more than 10 phases no longer crash the colour map; disjoint fields of one phase are no longer bridged (`distance_threshold` 0.5 → 0.2) |
| 1.15.0 (2026-09) | `dphi` column with `keep_unstable=True`; `RegularSolution` and `InterpolatingPhase` **deprecated**; `RedlichKister` fits by linear least squares (previously `curve_fit`); better Clausius-Clapeyron start bracket |
| 1.16.0 (2026-10) | `StitchedFit` sorts its input before taking edge windows (unsorted input gave wrong extrapolation); `SoftplusSurface2DInterpolator(knee_bounds=)` |
| 1.17.0 (2026-10/11) | `color_by=` on the 2-D plotters; fix for `fast-tsp` distance overflow (fields cut by the concave fallback) |
| 1.18.0 (2026-11) | **refiner overhaul**: congruent points solved exactly; pure-component transitions as `congruent` rows at `mu=±inf` (`TerminalRefiner`); monotectic-type invariants and miscibility-gap closures tagged; refined two-phase points must share φ within 1e-5 eV. **`f_excess` redefined** against the most stable pure components at each row's T. `get_transitions` returns two-phase tie-lines only (no `A-B-C` labels). Colours keyed by sorted phase name, so the same phase has the same colour across plots |

An A/B comparison of 1.14.0 and 1.18.0 on 31 computed binaries gave the same
stable phase at every grid point. Invariants moved by at most 1.8 K. 1.18
removed one spurious congruent tag and named new monotectic invariants. So the
numerics are stable across these versions, while the frame's annotations
(`locus`, `f_excess`, `get_transitions` labels) are not.

## Practice

- **Pin the version** in your environment, and put `landau.__version__` in any
  cache key for diagram frames.
- **Upgrade with an A/B.** Recompute a few diagrams from the same cached
  free-energy arrays under both versions and compare the stable phase per grid
  point and the invariant temperatures. Then bump your cache version.
- If you install a newer landau with pip over a conda or pixi environment,
  record it in the environment spec. A later solver run can silently put back
  the old version, or leave a mixed install where the dist-info and the code
  disagree. `landau.__file__` and `__version__` tell you which you are running.
- Deprecated names to replace before 2.0: `TemperatureDepandantLinePhase`,
  `RegularSolution`, `InterpolatingPhase`, `plot_phase_diagram(tielines=, alpha=,
  min_c_width=)`, `show=` on the 1-D plotters,
  `CalphadSurface2DInterpolator(terminal_sgte_order=)`, and the point-defect
  classes imported from `landau.phases` (import from `landau.phases.pointdefects`).
- `landau.calculate.find_one_point(p1, p2, potential, bracket)` is still
  importable, but only as a re-export of the private `refine._find_one_point` (a
  `root_scalar` on the difference of two potentials). It is not public API.
  Prefer `calc_phase_diagram(..., mu=0.0)` for a same-composition crossing,
  which also checks that no third phase undercuts the pair.
