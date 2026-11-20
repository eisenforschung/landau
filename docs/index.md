# landau.py

A library to calculate thermodynamic equilibria and plot phase diagrams of
binary alloys in the (semi-)grand ensemble.

You supply a free energy model for each phase — analytic toy models,
CALPHAD-style parametrizations, or interpolations of computed free energy data —
and landau finds the stable phases, refines the phase boundaries and draws the
diagram.

```{toctree}
:maxdepth: 2
:hidden:

self
installation
fleche
api
notebooks
publications
```

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Mg–Ca worked example
:link: notebooks/MgCa/MgCa
:link-type: doc
:img-top: images/mgca_phase_diagram.png
:img-alt: Mg-Ca phase diagram with labelled transition temperatures

A real system, from free energies of a machine learning potential.
:::

:::{grid-item-card} Transition temperatures
:link: notebooks/TransitionTemperatures
:link-type: doc
:img-top: images/transition_temperatures.png
:img-alt: Phase diagram with a liquid miscibility gap over a line compound, invariant temperatures labelled

Three-phase reactions and congruent points, solved and labelled.
:::

:::{grid-item-card} Point defects
:link: notebooks/PointDefects
:link-type: doc
:img-top: images/point_defects.png
:img-alt: Phase diagram of an ordered compound with a finite solubility range

Off-stoichiometry of an ordered compound from point defect energies.
:::

:::{grid-item-card} Excess free energy
:link: notebooks/ExcessFreeEnergy
:link-type: doc
:img-top: images/excess_free_energy.png
:img-alt: Free energy curves of four phases with their common tangents

The free energy curves and common tangents behind a diagram.
:::

:::{grid-item-card} Basics
:link: notebooks/Basics
:link-type: doc
:img-top: images/scan_1d.png
:img-alt: Semi-grand potential of three phases against temperature, transitions marked

Phases compete through their semi-grand potential.
:::

:::{grid-item-card} Plot gallery
:link: notebooks/PlotGallery
:link-type: doc
:img-top: images/mu_phase_diagram.png
:img-alt: Phase diagram in chemical potential and temperature

Every plot function, including diagrams in chemical potential.
:::
::::

## Installation

```bash
pip install landau
```

[Installation](installation.md) covers installing from source.

## Quickstart

Phases are defined by their free energies.
Two ideal solutions — a solid and a liquid — already give a full binary phase diagram:

```python
import numpy as np
from landau import LinePhase, IdealSolution, plot_phase_diagram
from landau.calculate import calc_phase_diagram
from landau.phases import kB

solid = IdealSolution("solid",
    LinePhase("A",    fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB),
    LinePhase("B",    fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB))
liquid = IdealSolution("liquid",
    LinePhase("A(l)", fixed_concentration=0, line_energy=-1.9, line_entropy=2.5 * kB),
    LinePhase("B(l)", fixed_concentration=1, line_energy=-2.9, line_entropy=2.2 * kB))

df = calc_phase_diagram([solid, liquid], Ts=np.linspace(400, 1800, 100), mu=100)
plot_phase_diagram(df)
```

```{image} images/quickstart_phase_diagram.png
:alt: Concentration-temperature phase diagram of a solid and a liquid ideal solution
:width: 75%
:align: center
```

`calc_phase_diagram` returns a plain pandas dataframe of `(T, mu, phase, c, phi)`
samples that the plotting functions consume, so intermediate results stay easy
to inspect and post-process.

## Where Next

- [Examples](notebooks.rst): all worked notebooks, from the basic concepts to phases built on ASE calculators.
- [API Reference](api.md): phases, interpolators, refiners and plotting.
- [Caching with fleche](fleche.md): reuse expensive diagrams across sessions.
- [Publications](publications.md): papers that use landau.

## Citation

This code is part of a [publication](https://doi.org/10.1038/s41524-025-01669-4), please cite it accordingly if you use this package in your work

```
@article{poul2025automated,
    title = {Automated generation of structure datasets for machine learning potentials and alloys},
    journal = {npj Computational Materials},
    author = {Poul, Marvin and Huber, Liam and Neugebauer, J{\"o}rg},
    volume = {11},
    number = {1},
    pages = {174},
    year = {2025},
    doi = {10.1038/s41524-025-01669-4},
}
```
