# Caching with fleche

[fleche](https://github.com/pmrv/fleche) caches function calls on disk, keyed by
the content of their arguments. Installing `landau` makes its phases,
interpolators and refiners valid arguments; there is nothing to import or
register:

```python
from fleche import fleche

from landau.calculate import calc_phase_diagram


@fleche
def diagram(phases, Ts, mu):
    return calc_phase_diagram(phases, Ts, mu)
```

Calling `diagram` again with phases rebuilt from the same data hits the cache,
in this interpreter and in the next one. Phases and fitted interpolations can
also be cached as results.

`AsePhase` needs [fleche-ase](https://pypi.org/project/fleche-ase/) as well
(`pip install fleche-ase`); without it, a call taking an `AsePhase` runs
uncached.
