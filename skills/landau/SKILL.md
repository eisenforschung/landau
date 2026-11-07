---
name: landau
description: >-
  Use the Python package landau (github.com/eisenforschung/landau, docs at
  landau.readthedocs.io) for binary-alloy thermodynamics: phase diagrams, phase
  boundaries, melting and boiling points, solidus, liquidus and solvus lines,
  miscibility gaps, eutectic, peritectic and congruent invariants,
  common-tangent constructions and semi-grand-canonical equilibria. Also use it
  to turn free energies F(T) or F(T, c) into a diagram, whether they come from
  MD thermodynamic integration (calphy and similar), quasi-harmonic or phonon
  calculations, DFT, CALPHAD-style parametrisations or toy models. Read this
  BEFORE you hand-roll a crossing or tangent solve, a melting-point bisection, an
  F(T) fit or a phase-boundary tracer, and before trusting a landau diagram built
  from computed data. Triggers: "landau", "phase diagram", "melting point",
  "liquidus", "solidus", "solvus", "miscibility gap", "common tangent",
  "semi-grand", "semigrand potential", "free energy curve", "where do these
  phases cross", "calc_phase_diagram", "TemperatureDependentLinePhase",
  "FastInterpolatingPhase", "plot_phase_diagram".
---

# landau: phase diagrams from free energies

landau computes equilibria of a **binary** system in the (semi-)grand-canonical
ensemble. You give each phase a free-energy model. landau finds the stable phase
at every `(T, Δμ)`, root-finds the exact phase boundaries and invariants, and
draws the diagram. If the question is "which phase is stable here?" or "where do
these two curves cross?", use landau rather than writing your own tangent or
crossing solver: its refiners are exact for the model you give it.

`pip install landau` (Python `>=3.11,<3.14`; it is also on conda-forge). Optional
extras: `fast-tsp` (better phase polygons, recommended), `ase`, `phonopy`.
Check `landau.__version__` first, since behaviour changed between 1.11 and 1.18
(see `references/versions.md`).

## Mental model

1. **Line phases** have a fixed composition and a free energy `F(T)`. Your
   computed data enter here.
2. **Solution phases** combine several line phases into `f(c)` at each `T`,
   with ideal mixing entropy handled explicitly.
3. **`calc_phase_diagram`** samples a `(T, Δμ)` grid, marks the phase with the
   lowest semi-grand potential `φ = f − Δμ·c` as stable, and then **refines**
   (root-finds) the boundaries. It returns a pandas DataFrame, and every plot
   and analysis step reads that frame.

**Units:** free energy in **eV/atom**, `T` in **K**, `Δμ` in **eV**. `c` is the
fraction of the *second* species (`c = 0` is pure A). `from landau.phases import kB`
is Boltzmann's constant in eV/K. Everything is per atom: per-formula-unit or
per-cell energies give a diagram that looks plausible and is wrong.

## Core workflow (computed F(T) data)

```python
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")          # set before numpy is imported (see pitfalls)

import numpy as np
import matplotlib.pyplot as plt
from landau import TemperatureDependentLinePhase, FastInterpolatingPhase, LinePhase
from landau.interpolate import WhitneyTemperatureInterpolator
from landau.calculate import calc_phase_diagram, guess_mu_range
from landau.plot import plot_phase_diagram

W = WhitneyTemperatureInterpolator()       # exact inside the data, constant entropy outside

def line(name, c, T, F, n=240):
    """One composition's F(T) in eV/atom, thinned to <= n points (Whitney is O(N^2) in memory)."""
    T, F = np.asarray(T, float), np.asarray(F, float)
    i = np.unique(np.linspace(0, len(T) - 1, min(n, len(T))).astype(int))
    return TemperatureDependentLinePhase(name, fixed_concentration=c,
                                         temperatures=T[i], free_energies=F[i], interpolator=W)

# MD free energies of disordered cells already contain the ideal mixing entropy -> add_entropy=False
liquid = FastInterpolatingPhase("liquid", [line(f"L{c}", c, T, F) for c, T, F in liquid_runs])
# 0 K / ordered-cell energies lack it -> add_entropy=True
fcc = FastInterpolatingPhase("fcc", [line(f"fcc{c}", c, T, F) for c, T, F in fcc_runs], add_entropy=True)
compound = line("AB2", 2/3, T_ab2, F_ab2)  # stoichiometric compound: a bare line phase
phases = [liquid, fcc, compound]

# A Δμ window that is valid at BOTH ends of the T range (an int mu guesses only at max(Ts))
Ts = np.linspace(300, 1800, 120)
w = np.concatenate([guess_mu_range(phases, T, 100)[0] for T in (Ts.min(), Ts.max())])
mu = np.linspace(w.min(), w.max(), 250)

df = calc_phase_diagram(phases, Ts=Ts, mu=mu)            # refine=True by default
fig, ax = plt.subplots()
plot_phase_diagram(df, ax=ax, triplepoints=True, transition_temperatures=True)  # returns None
```

Before you believe the diagram, run the checks under **Verification** below.

## Choosing a phase class

| class | use for |
|---|---|
| `LinePhase(name, fixed_concentration, line_energy, line_entropy=0)` | analytic `F = E − T·S`; toy models, quick tests |
| `TemperatureDependentLinePhase(name, fixed_concentration, temperatures, free_energies, interpolator=SGTE(3))` | **sampled `F(T)`**, the workhorse. Always pass `interpolator=` explicitly; the default is `SGTE(3)` |
| `FastInterpolatingPhase(name, phases, add_entropy=False, maximum_extrapolation=0, concentration_range=(0,1), interpolator=None)` | **the solution phase**: fits `f(c)` through line phases at each T. Handles miscibility gaps. Default `interpolator` is `RedlichKister(min(5, n))` when c=0 and c=1 are present, otherwise `PolyFit(min(4, n))` |
| `IdealSolution(name, phase1, phase2)` | exact ideal mixing between two terminals (c=0 and c=1 required) |
| `Surface2DInterpolatingPhase(..., surface_interpolator=...)` | one global `f(T, c)` fit; read `references/free-energy-models.md` before using it |
| `AsePhase(name, fixed_concentration, thermochem, pressure=None, atoms_per_formula=1)` | ASE `ThermoChem` objects (ideal gas, harmonic); needs `ase` |
| `PhonopyQuasiHarmonicPhase(...)` | QHA straight from phonopy `ThermalProperties` (≥ 1.12, `phonopy` extra); avoids fitting a sampled `F(T)` |
| `landau.phases.pointdefects.PointDefectedPhase` | line phase plus point-defect sublattices |

`RegularSolution` and `InterpolatingPhase` are deprecated (since 1.15, removal in
2.0). Use `FastInterpolatingPhase` with `interpolator=RedlichKister(n)`.
`SlowInterpolatingPhase` is the brute-force reference, useful for cross-checking.
`TemperatureDepandantLinePhase` (sic) is a deprecated alias.

**`add_entropy` reads backwards, so check it every time.** `add_entropy=False`
(the default) means *your inputs already contain the ideal mixing entropy*: landau
strips `−T·S_id(c)` before fitting and adds it back afterwards, so the inputs
are reproduced. Use it for MD/TI free energies of disordered cells and for
alchemical or composition-scaling data. `add_entropy=True` means *your inputs
lack it* (0 K energies, ordered cells, toy enthalpies), and landau adds
`−T·S_id(c)` on top. A wrong choice shifts every interior composition by about
`T·S_id`, which is 60 meV/atom at 1000 K and c = 0.5. Terminals are unaffected.

## `calc_phase_diagram(phases, Ts, mu, refine=True, keep_unstable=False)`

- `Ts`: an array, or a scalar for a single isotherm.
- `mu` is overloaded:
  - **an array** is used as given. Prefer this for production diagrams.
  - **a non-zero int `n`** asks `guess_mu_range` for `n` samples at
    **`max(Ts)` only**. At lower T the window can stop short, so phases near
    c = 0 or 1 go missing. Build the window from both ends, as in the workflow above.
  - **a float or `0`** fixes Δμ and gives a T-only scan. Use `mu=0.0` for a
    **unary** system: every phase at the same `fixed_concentration`, where the
    refined transition is the melting or boiling point. A non-zero int on a
    unary system raises `ValueError` ("Cannot guess chemical potential range…").
- `refine`: `True` picks refiners from the sampled axes. On a 2-D grid that is
  Delaunay triple points, Clausius-Clapeyron tracing, miscibility gaps and
  terminal transitions; a 1-D scan gets a scan refiner. `False` leaves raw grid
  resolution. A list of `landau.refine.Refiner` objects uses exactly those.
- `keep_unstable=True` keeps every phase at every point and adds `dphi`, the
  height above the stable phase. You need it for `plot_excess_free_energy`,
  for `plot_1d_*` with `reference_phase=`, and for asking why a phase never wins.

`guess_mu_range(phases, T, samples)` returns a **3-tuple** `(mu_array, c0, c1)`.
Take `[0]`.

**Output columns:** `T, phase, phi, mu, c, stable, locus, border, refined,
boundary_id, f`, plus `f_excess` when the phases reach both c = 0 and c = 1, plus
`dphi` with `keep_unstable=True`. The meaning of each column and how to pull
boundaries and invariants out of the frame are in `references/api.md`. In short:
- Invariants are the stable rows with `locus` in `{"triple", "congruent"}`
  (`from landau.features import Locus`).
- Refined rows carry the refiner's name in `refined` (`"no"` marks grid
  samples, `NaN` marks the synthetic `mu = ±inf` edge rows).
- **`border` is not "is a transition".** It also marks the edges of the sampled
  hull, including synthetic `mu = ±inf` rows.
- `get_transitions(df)` (in `landau.calculate`) returns tidy two-phase tie-line
  rows. For a unary scan, `get_transitions(df)["T"]` is the transition temperature.

## Pitfalls (each one has cost real time)

1. **The fit is the result.** The refiner finds the crossing of your *fitted*
   curves exactly. Moving `F` by δ moves a transition by about `δ/ΔS`, which is
   roughly 10–15 K per meV/atom for melting. A default `SGTE(3)` that missed its
   own samples by 4.6 meV/atom moved a melting point by 56 K. Pass the
   interpolator explicitly and look at its residuals.
2. **Extrapolation decides the low- and high-T parts of the diagram.**
   - `PolyFit(n)` diverges outside the data. In one case it reached −2·10⁷
     eV/atom, and the symptom was a confusing `ValueError` from `guess_mu_range`.
   - `SGTE`'s entropy diverges as T → 0, so an extrapolated liquid can sink under
     the solids and create a spurious low-T liquid field.
   - `WhitneyTemperatureInterpolator` continues at constant entropy. That is the
     safest default, but it gives an **upper bound** on F, so phases whose data
     start at higher T are penalised. One compound on the 0 K hull showed up
     only above 391 K.
   - Constant-entropy continuation also puts kinks in liquidus lines at the edge
     of each data window.

   Know each phase's data window. Make the data cover the diagram, and for
   classical-MD solids anchor `F(0) = E0` (see `references/free-energy-models.md`).
3. **A melting point needs data on both sides.** If the solid's sweep stops below
   Tm, the crossing sits in extrapolation. Run solid sweeps into the superheated
   range.
4. **`add_entropy` backwards** (see above). One team's notes had it inverted for
   two months.
5. **Δμ window too narrow at low T** when `mu` is an int (see above). The symptom
   is a diagram that never reaches high c below some temperature.
6. **Reading boundaries from the frame naively.** `groupby("T").c.min()` and
   filtering on `border` pick up grid interiors and hull edges. Use refined
   rows, `locus` or `get_transitions`. `df.query("not stable")` is empty unless
   you passed `keep_unstable=True`.
7. **Redlich–Kister interpolates exactly; it does not smooth.**
   - The fit caps `nparam` at the number of interior compositions, and the
     default is `min(5, n_phases)`. With up to five interior compositions the
     curve therefore goes exactly through every input, noise included.
   - Between the outermost interior point and a terminal it extrapolates, and
     can bend into `d²f/dc² < 0`, which is a spurious miscibility gap.
   - The basis is powers of **`(2c − 1)`**, not CALPHAD's `(x_A − x_B) = 1 − 2c`,
     so odd coefficients flip sign.
   - Check the curvature, and place interior runs near both ends.
8. **Whitney cost is O(N²) memory and O(N³) time.** A raw MD sweep of 50k points
   asked for 3.5 GB. Thin it to about 200–300 points.
9. **Unpinned BLAS threads.** On a many-core machine, small RBF solves ran three
   to four orders of magnitude slower with default threading (12 s instead of
   2 ms). Set
   `OMP_NUM_THREADS=1` and the matching OpenBLAS/MKL variables before importing
   numpy, and never quote a timing taken without them.
10. **Diagram caches go stale across versions.** The frame's meaning changed:
    `congruent` loci arrived in 1.14, `f_excess` was redefined in 1.18, and the
    refiners changed in 1.15–1.18. Put `landau.__version__` in any cache key.
    Cache on input arrays, not phase objects: closure-backed fits such as
    Whitney cannot be content-hashed reliably.
11. **Plotting:**
    - Plotters draw into `ax` (or the current axes) and return `None`.
    - `plot_phase_diagram` sets `ylim` to the data T range. Set your limits
      after calling it.
    - `legend=False` also removes the inline phase labels.
    - Without `fast-tsp` installed, phase polygons fall back to slower or cruder
      methods.
    - More than 10 phases raised `KeyError` before 1.14.1.
    - Near-coincident points broke `fast-tsp` polygons before 1.17.
    - Details are in `references/plotting.md`.
12. **`resample_borders`** (bootstrap error bars) defaults to `cores=20`. Its
    workers share numpy's global RNG and repeat shifts (upstream #516), which
    understates the spread. Pass `cores=` explicitly and treat the error bars as a
    lower bound.

## Verification (do this before reporting any number)

```python
p.check_interpolation(Tl=0.9, Tu=1.1)              # each TemperatureDependentLinePhase: fit vs samples
p.check_interpolation(plot_error=True)             # residuals at the samples (meV matter, see pitfall 1)
sol.check_concentration_interpolation(T=1000, plot_excess=True)   # each solution phase: f(c) vs inputs
sol.check_concentration_interpolation(T=1000, plot_error=True)
```

- Record each phase's sampled T window, `p.temperatures.min()` and `.max()`, and
  say which parts of the diagram are extrapolated. A Whitney residual is zero by
  construction, so it tells you nothing about extrapolation.
- Check that refinement fired: `df.refined.unique()` should hold more than `"no"`.
- List the invariants (stable rows with `locus` in triple or congruent) and
  sanity-check them against known data or an assessed diagram.
- Look at `plot_excess_free_energy` on a `keep_unstable=True` frame at a few
  temperatures. A wrong boundary is almost always visible there as a bad `f(c)`.
- For a claim like "the boundary is at X ± Y K", estimate how far it moves under
  the free-energy uncertainty (`resample_borders`, or shift one phase by its
  error bar and recompute).
- Do the work in a notebook, so the check plots stay with the result.

## More

- `references/api.md`: signatures verified against 1.18 and main, the output
  frame, refiners, extracting boundaries and invariants, and custom phases.
- `references/free-energy-models.md`: interpolators, feeding MD/QHA/DFT data,
  F(0)=E0 anchoring, liquid models across composition, and surfaces.
- `references/plotting.md`: plot functions, keyword traps, and provenance
  overlays (show where the data are).
- `references/versions.md`: what changed from 1.11 to 1.18, and upgrade
  practice.
- Upstream: README, the `notebooks/` gallery (Basics, IdealSolution,
  Intermetallics, ExcessFreeEnergy, TransitionTemperatures, MgCa) and the
  issue tracker.
