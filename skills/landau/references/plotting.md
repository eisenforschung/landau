# Plotting with landau

All plotters live in `landau.plot`. `plot_phase_diagram` and
`plot_excess_free_energy` are also exported from the package root. The c–T,
Δμ–T and 1-D plotters draw into `ax=` if you pass it, and otherwise into the
current axes. `plot_excess_free_energy` takes no `ax=`: for one temperature it
draws into the current axes, so create the figure first; for several it builds
its own seaborn `FacetGrid`.

| function | draws | returns |
|---|---|---|
| `plot_phase_diagram(df, ...)` | c–T diagram (phase fields as polygons) | `None` |
| `plot_mu_phase_diagram(df, ...)` | Δμ–T diagram | `None` |
| `plot_1d_T_phase_diagram(df, ax=None, reference_phase=None, ...)` | φ along a fixed-Δμ T scan (e.g. unary) | axes |
| `plot_1d_mu_phase_diagram(df, ax=None, reference_phase=None, element=None, ...)` | φ along an isothermal Δμ scan | axes |
| `plot_excess_free_energy(df, col_wrap=3, convex_hull=True, inline_legend=True, element=None, ...)` | `f_excess(c)` per phase plus common tangents | axes (one T) or a seaborn `FacetGrid` (several T) |

## Keyword reference for the 2-D plotters (1.18)

`plot_phase_diagram(df, element=None, color_override={}, triplepoints=None,
transition_temperatures=False, poly_method=None, inline_legend=True, legend=True,
ax=None, color_by=None, cmap=None, norm=None, colorbar=True)`

- `triplepoints=True` draws an isotherm at each three-phase invariant.
  `transition_temperatures=True` writes the temperature at every triple and
  congruent invariant (1.14+). Both need a **refined** frame, because they read
  the `locus` column. A frame cached under landau < 1.14 has no `congruent` rows.
- `element="Mg"` labels the axes for the second species.
- `color_by="dphi"` (or any numeric column, 1.17+) fills each field with that
  quantity.
- `color_override={"liquid": "#rrggbbaa"}`: fills are opaque, so give a
  translucent RGBA when you want a field tinted.
- `poly_method`: `"segment-fasttsp"` is the default when `fast-tsp` is
  installed, then `"segment-tsp"` / `"fasttsp"` / `"tsp"` / `"concave"`.
  `"concave"` can cut into narrow or re-entrant fields. Install `fast-tsp`.
- `alpha=` and `min_c_width=` are deprecated. Pass a `landau.poly` method
  object instead.
- `tielines=` is deprecated; use `triplepoints=`.

## Traps

- **The plotters return `None`.** Do not assign the result and expect a figure.
  Use `fig, ax = plt.subplots(); plot_phase_diagram(df, ax=ax)`.
- **`ylim` is set to the frame's T range.** Set your own limits *after* the call,
  and before adding overlays, so the overlays don't move them.
- **`legend=False` also suppresses the inline labels.** The logic is
  `if legend: (inline labels if inline_legend else legend box)`. For in-place
  labels without a box, keep `legend=True, inline_legend=True`.
- **`plot_excess_free_energy` needs `keep_unstable=True`** and the `f_excess`
  column. That column is absent when the phases do not reach both c = 0 and c = 1.
- **Slow multi-temperature excess plots.** On one system (2 temperatures,
  matplotlib 3.11, seaborn 0.13.2) the FacetGrid took 30 s without inline labels
  and several minutes with them, because each inline label triggers a full
  figure redraw. One temperature per call took 0.1 s. Loop over temperatures,
  or pass `inline_legend=False`.
- **1-D plots with `reference_phase=`** need the reference phase sampled
  everywhere, i.e. `keep_unstable=True`. Otherwise you get a `ValueError` naming
  the missing rows.
- **A phase "missing" from a plot** usually just means it is never stable. Check
  with `keep_unstable=True` and the `dphi` column before suspecting a bug.
- **Duplicate labels.** A pure-component transition and an invariant at nearly
  the same T can both be labelled. Older versions also double-tagged congruent
  points; 1.18 fixed most cases. De-duplicate `ax.texts` if needed.
- **Old-version bugs worth recognising:**
  - More than 10 stable phases raised `KeyError` in the colour map (fixed in 1.14.1).
  - Disjoint fields of one phase were bridged into a single polygon (fixed in 1.14.1).
  - Nearly coincident boundary points overflowed `fast-tsp`'s distance limit,
    and the concave-hull fallback then cut a field (fixed in 1.17.0).

## Show what the diagram rests on

A computed diagram looks equally confident where it is data and where it is
extrapolation, so draw the data over it. The pattern below uses matplotlib only:

```python
fig, ax = plt.subplots()
plot_phase_diagram(df, ax=ax, triplepoints=True, transition_temperatures=True)
ax.set_xlim(0, 1); ax.set_ylim(T_lo, T_hi)                 # fix limits first

# each free-energy run is a vertical bar at its composition over its sampled T window
for i, (c, t0, t1, kind) in enumerate(runs):               # e.g. (0.25, 900, 1600, "liquid")
    t0, t1 = max(t0, T_lo), min(t1, T_hi)
    col = "tab:red" if kind == "liquid" else "tab:blue"
    dx = 0.0 if kind == "liquid" else (0.012 if c < 0.5 else -0.012)   # solid and liquid share terminals
    for lw, a in ((9, 0.3), (1, 0.85)):                    # translucent halo + thin core
        ln, = ax.plot([c + dx] * 2, [t0, t1], color=col, lw=lw, alpha=a,
                      solid_capstyle="butt", zorder=9)
        ln.set_gid(f"data-{kind}-{i}")                     # becomes <g id=...> in SVG output

# hatch what the model does not describe (e.g. above the highest data)
ax.axhspan(T_valid_max, T_hi, facecolor="0.85", hatch="///", edgecolor="0.55",
           alpha=0.6, zorder=8)
```

- **Mark data; don't veil non-data.** A translucent veil over unsupported regions
  washed the diagram out. Thin bars at the run compositions say the same with
  less ink.
- **zorder.** landau gives each field `zorder = 1/area`, so values are much
  smaller than 1. Overlays at zorder ≥ 2 sit above every field. Keep labels
  readable with bars at 9–10 and text at 11.
- **Find the polygons landau added**: take `n = len(ax.patches)` before the call;
  `ax.patches[n:]` afterwards are the fields. Each is labelled with its phase
  name, primed (`liquid'`) when a phase has several fields.
- **Reference data** (an assessed CALPHAD diagram, experiments) go on as dashed
  lines in one colour. Draw invariants as dashed horizontals spanning the three
  compositions.
- **Unary 1-D plots**: draw each phase's data window as a thin bar just below
  the curves, so the extrapolated parts are obvious.
- **`set_gid`** makes matplotlib write an SVG group id, which a web page can
  toggle with CSS. Upstream PR #543 (open at 1.18) proposes a
  `landau.plot.svg.save_layered_svg` that does this for every element.
- A per-phase free-energy panel is the fastest way to spot a bad fit: raw samples
  against `line_free_energy(T)`, with the data window shaded and the residual in
  meV/atom underneath.
