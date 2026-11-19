# Installation

## From PyPI

You can install `landau` using pip:

```bash
pip install landau
```

## From Source

To install the latest version from source, clone the repository and install it in editable mode:

```bash
git clone https://github.com/eisenforschung/landau.git
cd landau
pip install -e .
```

## Building packages

If you want to build the package yourself:

```bash
pip install build
python -m build --sdist
```

## Caching landau objects with fleche

[fleche](https://github.com/pmrv/fleche) keys its cache on a content digest of
every argument. Installing `landau` registers digest hooks for its types under
fleche's `fleche` entry point group, so landau objects can be passed to cached
functions with no further setup — no imports, no `add_hook` calls:

```python
from fleche import fleche

from landau.calculate import calc_phase_diagram


@fleche
def diagram(phases, Ts, mu):
    return calc_phase_diagram(phases, Ts, mu)
```

Calling `diagram` again with phases rebuilt from the same data hits the cache,
in this interpreter and in the next one. `landau` does not depend on `fleche`;
the hook module is imported only by fleche's entry point loader.

Fitted interpolations — what `SplineFit`, `StitchedFit`, `SoftplusFit` and the
Whitney interpolators return — are digested by their fitted state, and phases
and fits pickle with plain `pickle`, `dill` or `cloudpickle`. A fleche value
store that destructures a phase into its fields rebuilds it without running
`__post_init__` or `__setstate__`, so it cannot check which landau wrote the
entry; put `landau.__version__` into the `version=` of a cached function whose
result holds landau objects:

```python
import landau


@fleche(version=f"1.{landau.__version__}")
def phases(path):
    ...
```
