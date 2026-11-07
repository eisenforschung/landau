# landau API reference (checked against 1.18.0 and main at b387f70)

The signatures below were read from the source. Names not re-exported from the
package root must be imported from their submodule.

## Imports

```python
from landau import (LinePhase, TemperatureDependentLinePhase, IdealSolution,
                    FastInterpolatingPhase, SlowInterpolatingPhase,
                    Surface2DInterpolatingPhase, AsePhase, PhonopyQuasiHarmonicPhase,
                    SGTE, PolyFit, SplineFit, RedlichKister, SoftplusFit,
                    CalphadSurface2DInterpolator, SoftplusSurface2DInterpolator,
                    plot_phase_diagram, plot_excess_free_energy)
from landau.phases import kB, S          # Boltzmann (eV/K); ideal mixing entropy S(c) in eV/K/atom
from landau.interpolate import WhitneyTemperatureInterpolator, StitchedFit   # NOT in the package root
from landau.calculate import calc_phase_diagram, guess_mu_range, get_transitions, refine_phase_diagram
from landau.plot import (plot_mu_phase_diagram, plot_1d_T_phase_diagram,
                         plot_1d_mu_phase_diagram, get_polygons, get_phase_colors)
from landau.features import Locus        # NOT in the package root
from landau.refine import (ScanRefiner, DelaunayLineRefiner, DelaunayTripleRefiner,
                           ClausiusClapeyronRefiner, MiscibilityGapRefiner, TerminalRefiner,
                           default_refiners)
from landau.resample import resample_borders
from landau.phases.pointdefects import PointDefectedPhase, PointDefectSublattice  # (newer sublattice models live here only)
```

## Phases

Every phase is a frozen dataclass with a `name` (unique within one diagram;
`calc_phase_diagram` keys phases by name, so a duplicate name silently replaces
the earlier phase) and two methods: `semigrand_potential(T, dmu)` and
`concentration(T, dmu)`. Both broadcast over arrays and return scalars for scalar input.

Line phases (`AbstractLinePhase` subclasses) also have `line_concentration`
(property) and `line_free_energy(T)`. Solution phases have `free_energy(T, c)`.

| class | fields (defaults) |
|---|---|
| `LinePhase` | `name, fixed_concentration, line_energy, line_entropy=0` |
| `TemperatureDependentLinePhase` | `name, fixed_concentration, temperatures, free_energies, interpolator=SGTE(3)`; arrays become read-only; `check_interpolation(Tl=0.9, Tu=1.1, samples=50, plot_error=False)` (Tl/Tu are *fractions* of the sampled min/max T) |
| `IdealSolution` | `name, phase1, phase2`; asserts terminals at exactly c=0 and c=1 and sorts them by concentration itself |
| `SlowInterpolatingPhase` / `FastInterpolatingPhase` | `name, phases, add_entropy=False, maximum_extrapolation=0, concentration_range=(0., 1.), interpolator=None`; `concentration_range` and `maximum_extrapolation` are mutually exclusive; without c=0 and c=1 terminals the range defaults to the span of the inputs (extend it with `maximum_extrapolation`), and `RedlichKister` is refused; `check_concentration_interpolation(T=1000, samples=50, plot_excess=False, plot_error=False)` |
| `Surface2DInterpolatingPhase` | as Fast, plus `surface_interpolator` (**required**), `num_temperature_samples=40`, `temperature_range=None` (defaults to the union of the inputs' T windows) |
| `AsePhase` | `name, fixed_concentration, thermochem, pressure=None, atoms_per_formula=1`; uses Helmholtz if available, else Gibbs at 1 atm by default |
| `PhonopyQuasiHarmonicPhase` | `name, fixed_concentration, thermal_properties, volumes, energies, atoms_per_cell, atoms_per_primitive_cell, eos=PolyFit(8), min_frequency=-0.05, extrapolate=False`; volumes with imaginary modes are dropped with a warning; the volume is clamped to the sampled range unless `extrapolate=True` (closed-form eos only); `check_equation_of_state(T, ...)` |

`FastInterpolatingPhase` finds `φ = min_c [f(c) − Δμ·c]` on a 201-point grid plus
a Newton polish, so `c = −∂φ/∂Δμ` holds to about 1e-6. Within ~1e-9 eV of a
miscibility gap's coexistence Δμ, `concentration` can return a grid node
(upstream #563). Away from that point it is irrelevant.

### Writing your own phase

If your model can be evaluated directly, for example from a density of states or
a closed-form expression, subclass `AbstractLinePhase` instead of sampling and
refitting:

```python
from dataclasses import dataclass
from landau.phases import AbstractLinePhase

@dataclass(frozen=True)
class MyLinePhase(AbstractLinePhase):
    fixed_concentration: float
    model: object                     # anything hashable/comparable; frozen dataclass rules apply
    @property
    def line_concentration(self):
        return self.fixed_concentration
    def line_free_energy(self, T):    # eV/atom; must broadcast over arrays
        return self.model.free_energy(T)
```

Return `NaN` where the model is not trustworthy rather than a plausible number.
Then restrict the T grid, or sample the trusted window into a
`TemperatureDependentLinePhase`. Keep the name unique and the energy per atom.

## `calc_phase_diagram(phases, Ts, mu, refine=True, keep_unstable=False)`

See SKILL.md for the meaning of `mu`. The frame has these columns:

| column | meaning |
|---|---|
| `T`, `mu`, `phase`, `c`, `phi` | sample point, phase, its composition and semi-grand potential (eV/atom) |
| `stable` | lowest `phi` at that `(T, mu)` |
| `f` | `phi + mu*c`, the free energy |
| `f_excess` | `f` minus the chord between the most stable pure components at that row's T (1.18+ definition; absent if the phases do not reach both c=0 and c=1) |
| `locus` | `Locus` StrEnum: `interior`, `boundary`, `triple`, `congruent` (`congruent` since 1.14, set by the refiner) |
| `border` | on the hull of the sampled region or a refined boundary; includes synthetic `mu=±inf` rows that close polygons |
| `refined` | `"no"` for grid points, `NaN` on the synthetic `mu=±inf` edge rows, otherwise the refiner that produced the row (e.g. `"clausius-clapeyron"`, `"delaunay-triple"`, `"terminal"`, `"T"` for a 1-D T scan) |
| `boundary_id` | integer id of the coexistence line a refined row belongs to |
| `dphi` | only with `keep_unstable=True`: `phi − min(phi)` at the same `(T, mu)` |

### Extracting results

```python
st = df[df.stable]

# invariants (eutectic, peritectic, monotectic, congruent melting, pure-component transitions)
inv = st[st.locus.astype(str).isin(["triple", "congruent"])]
for T, g in inv.groupby(inv["T"].round(1)):
    print(T, sorted(set(g.phase)), sorted(set(g.c.round(3))))

# refined two-phase boundary points, by coexistence line
bd = st[(st.refined != "no") & (st.locus.astype(str) == "boundary")]
for bid, g in bd.groupby("boundary_id"):
    ...                                  # g has both phases' c at each T

# tidy tie-lines: columns mu, T, transition ("A-B"), c, phase, transition_unit, border_segment
tr = get_transitions(df)

# unary: transition temperature(s) of a mu=0.0 scan
Tm = get_transitions(df_unary)["T"].unique()
```

Since 1.18 `get_transitions` pairs border rows into two-phase tie-lines. To find
three-phase invariants, select on `locus == "triple"`; there are no `A-B-C`
labels any more.

Terminal (pure-component) transitions appear as `congruent` rows at
`mu = ±inf` with `c` exactly 0 or 1 and `phi = NaN` (1.18+).

## Refiners (`landau.refine`)

`default_refiners(df)`:
- On a 2-D grid: `DelaunayTripleRefiner`, `ClausiusClapeyronRefiner`,
  `MiscibilityGapRefiner` and `TerminalRefiner`.
- On a 1-D scan: `ScanRefiner(by="T")` or `ScanRefiner(by="mu")`.
- `DelaunayLineRefiner` is opt-in.

The tracers take keyword-only `dT_max=5.0, dT_min=1.0, dc_max=0.01, dc_min=0.0,
max_steps=500`. `ClausiusClapeyronRefiner` also takes `congruent_tol=0.1`.
Pass refiners explicitly when the defaults miss a feature, or for a denser trace:

```python
df = calc_phase_diagram(phases, Ts, mu, refine=[DelaunayTripleRefiner(),
                                                ClausiusClapeyronRefiner(dT_max=2.0),
                                                MiscibilityGapRefiner(), TerminalRefiner()])
```

The refined boundaries are exact solutions for *your model*. Their accuracy is
the accuracy of the free energies (SKILL.md, pitfall 1), not the grid spacing.
A coarse grid (tens of T and Δμ points) is usually enough. A feature narrower
than the grid can be missed entirely, so if a known phase field is absent,
refine the grid locally before blaming the model.

## Bootstrap error bars

```python
from landau.resample import resample_borders
b = resample_borders(phases, Ts, mu_array, noise_norm=1e-3, n=100, cores=4)  # eV/atom noise per phase
```

Each run shifts every phase's `φ` by one Gaussian draw of width `noise_norm` and
recomputes the refined borders. Up to 1.18, workers share numpy's global RNG,
so draws repeat across workers (#516). Treat the spread as a lower bound, or
run with `cores=1`.

## Helpers worth knowing

- `S(c)` is the ideal mixing entropy per atom in eV/K
  (`-kB(c ln c + (1-c) ln(1-c))`). Use it to add the mixing entropy to data
  that lack it, or to remove it from data that have it.
- `phase.line_free_energy(T)` and `phase.free_energy(T, c)` evaluate your fitted
  model anywhere. Use them, not `np.interp` on the raw samples (which clamps at
  the ends), when comparing curves.
- `phase._interpolation.deriv()` (private) returns `dF/dT` as an `Interpolation`
  object. It has no unary minus, so negate the values: `S = -d(T)`.
