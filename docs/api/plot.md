# landau.plot

```{eval-rst}
.. automodule:: landau.plot
   :members:
   :undoc-members:
   :show-inheritance:
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
