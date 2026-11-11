# landau.plot

```{eval-rst}
.. automodule:: landau.plot
   :members:
   :undoc-members:
   :show-inheritance:
```

## Style

Every plot sizes its lines, markers and text from `lines.linewidth`,
`lines.markersize` and `font.size`, and draws each kind of element (metastable
branch, tie line, transition guide, phase label) the same way.

`landau.plot.style.theme` holds a few rcParams (line width, tick and legend font
sizes, legend frame, figure size) that every plot function applies for the
duration of its call, whatever your rcParams hold for those keys. Every other
rcParam, fonts included, is yours. Change the theme instead of `rcParams`:

```python
from landau.plot.style import theme

theme.update({"lines.linewidth": 1.5})   # validated like rcParams
del theme["figure.figsize"]              # this key follows rcParams again
theme.clear()                            # all of them do
theme.reset()                            # back to landau's values
```

Inside the call the theme reaches what the call creates: lines, markers, text,
and the figure when no `ax` is passed. The same values ship as the style sheet
`landau.plot.landau`; load it globally to also cover axes and figures you create
yourself, and `savefig`:

```python
import matplotlib.pyplot as plt
plt.style.use("landau.plot.landau")
```

```{eval-rst}
.. automodule:: landau.plot.style
   :members:
```
