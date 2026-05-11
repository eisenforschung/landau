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
aren't available in the installed version (e.g. `SoftplusFit.global_fit`,
introduced in #82) are reported as skipped rather than aborting the
run.  Add cases by editing `case_*` factories.
