# Benchmarks

Microbenchmarks comparing the wall-time and fit quality of the
interpolators in `landau.interpolate` on a small set of representative
1-D datasets.

## Running

```bash
pip install -e .[test,constraints]
python benchmarks/bench_interpolators.py
```

The script prints a Markdown-friendly table of median wall-times (3
reps after a warm-up) and the training RMS for each `(case, method)`
pair.  All cases share `N_POINTS` (top of `bench_interpolators.py`)
so timings are directly comparable across datasets.  Methods that
don't apply to a case (e.g. `RedlichKister` on a temperature axis)
are caught and reported as skipped rather than aborting the run.
Add cases by editing `case_*` factories.

## `bench_fast_interpolating_phase.py`

```bash
python benchmarks/bench_fast_interpolating_phase.py
```

Compares `FastInterpolatingPhase` against `SlowInterpolatingPhase` on a
`calc_phase_diagram`-style driver (scalar-`T` loop over a `dmu` array),
reporting wall-time, speedup, and each solver's error against a dense-grid
true minimum.  Fast is ~2 orders of magnitude faster and at least as
accurate (the brute reference's coarse 20-point grid can miss deep wells).

## `cc_sampling_density.py`

```bash
python benchmarks/cc_sampling_density.py [--plot out.png]
```

Isolates the `ClausiusClapeyronRefiner` sampling-density regime that the
`dc_max` cap targets: a coexistence boundary that is flat in `mu`
(`dmu/dT -> 0`, where `_dT_adapt` saturates at `dT_max`) but whose plotted
concentration still sweeps.  Reports the point count and worst per-step
concentration jump for the old defaults (`dT_max=50`, no drift cap) vs the
new defaults (`dT_max=5`, `dc_max=0.01`); `--plot` writes the c-T scatter.
