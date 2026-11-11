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
branch, tie line, transition guide, phase label) the same way. The defaults ship
as a matplotlib style sheet, `landau.plot.landau`, which every plot function
applies during its call to each rcParam still at matplotlib's built-in default;
whatever you have set (by `plt.style.use`, a `matplotlibrc` or `rcParams`) wins.
A value set back to exactly matplotlib's default can't be told apart from an
untouched one and is overridden.

Inside the call the sheet reaches what the call creates: lines, markers, text,
and the figure when no `ax` is passed. To also cover axes and figures you create
yourself, and `savefig`, load it globally:

```python
import matplotlib.pyplot as plt
plt.style.use("landau.plot.landau")
```

```{eval-rst}
.. automodule:: landau.plot.style
   :members:
```
