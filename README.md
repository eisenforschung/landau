[![PyPI](https://img.shields.io/pypi/v/landau)](https://pypi.org/project/landau/)
[![DOI](https://zenodo.org/badge/931240296.svg)](https://doi.org/10.5281/zenodo.15513439)
[![Documentation Status](https://readthedocs.org/projects/landau/badge/?version=latest)](https://landau.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/eisenforschung/landau)

# landau.py

A library to calculate thermodynamic equilibria and plot phase diagrams of
binary alloys in the (semi-)grand ensemble.

You supply a free energy model for each phase — analytic toy models,
CALPHAD-style parametrizations, or interpolations of computed free energy data —
and landau finds the stable phases, refines the phase boundaries and draws the
diagram.

## Installation

```bash
pip install landau
```

## Quickstart

Phases are defined by their free energies.  Combining end member phases into
(ideal) solutions and scanning over temperature and chemical potential
difference gives a full binary phase diagram in a few lines:

```python
import numpy as np
from landau import LinePhase, IdealSolution
from landau.phases import kB
from landau.calculate import calc_phase_diagram
from landau.plot import plot_phase_diagram

hcp = IdealSolution("hcp",
    LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * kB),
    LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95,  line_entropy=1.1 * kB))
fcc = IdealSolution("fcc",
    LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * kB),
    LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * kB))
liquid = IdealSolution("liquid",
    LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * kB),
    LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * kB))

df = calc_phase_diagram([hcp, fcc, liquid], Ts=np.linspace(100, 1000, 100), mu=100)
plot_phase_diagram(df, tielines=True)
```

![Concentration-temperature phase diagram of three ideal solution phases](https://raw.githubusercontent.com/eisenforschung/landau/main/docs/images/quickstart_phase_diagram.png)

`calc_phase_diagram` returns a plain pandas dataframe of `(T, mu, phase, c, phi)`
samples that the plotting functions consume, so intermediate results stay easy
to inspect and post-process.

## Real Systems

The same interface works with computed free energies.  This Mg–Ca phase
diagram is calculated from free energies of an Atomic Cluster Expansion
potential, obtained by thermodynamic integration with
[calphy](https://calphy.org):

![Mg-Ca phase diagram from machine learning potential data](https://raw.githubusercontent.com/eisenforschung/landau/main/docs/images/mgca_phase_diagram.png)

The [Mg–Ca worked example](https://landau.readthedocs.io/en/latest/notebooks/MgCa/MgCa.html)
goes from raw free energy samples to this diagram.

## Looking Under the Hood

`plot_excess_free_energy` shows the free energy curves behind a diagram,
including metastable phases and the common tangent constructions that
determine the stable regions:

![Excess free energy curves with common tangent constructions at three temperatures](https://raw.githubusercontent.com/eisenforschung/landau/main/docs/images/excess_free_energy.png)

## More Examples

The [documentation](https://landau.readthedocs.io) walks through the full API,
including
[the basic concepts](https://landau.readthedocs.io/en/latest/notebooks/Basics.html),
[congruent and non-congruent melting](https://landau.readthedocs.io/en/latest/notebooks/IdealSolution.html),
[stoichiometric line compounds](https://landau.readthedocs.io/en/latest/notebooks/Intermetallics.html) and
[point defects](https://landau.readthedocs.io/en/latest/notebooks/PointDefects.html).

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
