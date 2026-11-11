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
branch, tie line, transition guide, phase label) the same way. The figure-level
defaults ship as a matplotlib style sheet:

```python
import matplotlib.pyplot as plt
plt.style.use("landau.plot.landau")
```

The plot functions never apply it themselves; a caller's own style is left alone.

```{eval-rst}
.. automodule:: landau.plot.style
   :members:
```
