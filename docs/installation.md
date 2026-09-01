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

One case is deliberately refused rather than digested: a fitted
`Interpolation` returned by an interpolator that builds it from a closure
(`SplineFit`, `StitchedFit`, `SoftplusFit`, the Whitney interpolators). fleche
digests a function from its code object alone, and two fits of one interpolator
share that code object, so caching on them would return one result for two
different curves. Pass the `Interpolator` and its samples into the cached
function and fit there instead:

```python
@fleche
def free_energy(interpolator, temperatures, energies, T):
    return interpolator.fit(temperatures, energies)(T)
```
