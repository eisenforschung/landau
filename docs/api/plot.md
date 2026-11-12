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
Its other fields each style one kind of element: metastable branches, tie lines,
transition guides, invariant markers and labels.

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
