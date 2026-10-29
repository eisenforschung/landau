# landau.plot

```{eval-rst}
.. automodule:: landau.plot
   :members:
   :undoc-members:
   :show-inheritance:
```

## Style

`landau.plot.style.theme` sets the look of every landau plot.
Its `rc` field holds a few rcParams: line width, tick and legend font sizes, the
legend frame, figure size and resolution.
Its other fields each style one kind of element.
They hold matplotlib keyword arguments for colour, line style and font weight.
Line widths, marker sizes and font sizes come from `lines.linewidth`,
`lines.markersize` and `font.size`.

| Field | Default | Styles |
|---|---|---|
| `phase_boundary` | `{"edgecolor": "k"}` | The outline of a phase field in a 2d diagram. |
| `metastable` | `{"linestyle": "--"}` | A phase's curve where another phase is more stable, in 1d cuts and excess plots. |
| `tie_line` | `{"color": "k", "alpha": 0.5, "linestyle": "-"}` | A line between coexisting states: the three-phase isotherm of a c-T diagram, a common tangent of an excess plot. |
| `transition_guide` | `{"color": "k", "alpha": 0.5, "linestyle": ":"}` | The full-height line at a transition along a 1d cut. |
| `transition_tick` | `{"direction": "in", "length": 6}` | The tick on the top spine of a 1d cut at a transition. |
| `invariant_marker` | `{"color": "k", "marker": "o"}` | The dot at a coexisting state: a transition of a 1d cut, a triple point of a μ-T diagram, a tangent vertex. |
| `line_phase_marker_scale` | `1.5` | The size of a line-phase dot in an excess plot, in units of `lines.markersize`. |
| `reference_line` | `{"color": ".5", "linestyle": "-"}` | The zero line of an excess plot. It is `axes.linewidth` wide unless the field sets `linewidth`. |
| `region_label` | `{"fontsize": "small", "fontweight": "bold"}` | A phase name on the region where the phase is stable: a 2d field, a window along the top of a 1d cut. |
| `curve_label` | `{"fontsize": "small"}` | A phase name on its curve: the side labels of a 1d cut, the labels of an excess plot. |
| `value_label` | `{"fontsize": "small"}` | A number read off the plot: a transition temperature or chemical potential. |
| `label_outline` | `{"linewidth": 3, "foreground": "white"}` | The stroke behind every label. |

Every plot function applies `theme.rc` while it runs.
Your rcParams set everything else, fonts included.
To change the look, change the theme:

```python
from landau.plot.style import theme

theme.rc["lines.linewidth"] = 1.5      # validated like rcParams
del theme.rc["figure.figsize"]         # your rcParams set the figure size again
theme.tie_line["color"] = "tab:red"
theme.reset()                          # back to the defaults
```

`styled()` applies `theme.rc` to your own plots too:

```python
from landau.plot.style import styled

with styled():
    plt.plot(x, y)
    plt.savefig("plot.png")            # the theme's resolution and tight bounding box
```

```{eval-rst}
.. automodule:: landau.plot.style
   :members:
```

## Layered SVG export

{func}`landau.plot.save_layered_svg` writes a figure as SVG with every phase
field, label, two-phase region, phase boundary, invariant, the legend and the
axes parts in their own `<g>` groups, each with a stable `id`, a `class` and
`data-*` attributes naming the phases, so a web page or an editor can address
them. Overlays drawn on top of a diagram join their own layer through
{func}`landau.plot.svg.tag`.

```python
import matplotlib.pyplot as plt
from landau.plot import plot_phase_diagram, save_layered_svg
from landau.plot.svg import tag

fig, ax = plt.subplots()
plot_phase_diagram(df, ax=ax, transition_temperatures=True)
tag(ax.plot([0.3, 0.3], [600, 900], color="red"), "calphy-liquid", c=0.3)
save_layered_svg(fig, "diagram.svg", id_prefix="CaMg_")
```

In a page that inlines the file, the groups are selected by class and data
attribute:

```css
.landau-phase:hover path { filter: brightness(0.9); }
.landau-two-phase:hover path { fill-opacity: 0.3; }
.landau-boundary[data-phase="liquid"] path { stroke: red; stroke-opacity: 1; }
.hide-labels [data-layer="labels"].landau-layer { display: none; }
```

```js
svg.querySelectorAll(".landau-phase").forEach((g) =>
  g.addEventListener("click", () => showPhase(g.dataset.phase)));
```

```{eval-rst}
.. automodule:: landau.plot.svg
   :members: tag
```
