# Free-energy models: getting computed data into landau

A landau diagram is exactly as good as the `F(T)` and `f(c)` models behind it.
The refiners are exact; the models are not. This file covers choosing and
checking those models.

## Interpolators

There are two independent axes. `TemperatureInterpolator` fits `F(T)` for one
line phase. `ConcentrationInterpolator` fits `f(c)` across line phases at a
fixed T. A third kind, `SurfaceInterpolator`, fits `f(T, c)` at once.

| interpolator | axis | inside the data | outside the data | use when |
|---|---|---|---|---|
| `WhitneyTemperatureInterpolator()` (`landau.interpolate`) | T | interpolates exactly (thin-plate RBF, `smoothing=0`) | **linear in T** = constant entropy, an upper bound on F | default for sampled MD/TI or QHA `F(T)`; thin to ≤ ~300 points |
| `SGTE(n)` | T | least squares to `a·T ln T + Σ p_i T^i` | CALPHAD-like; **entropy diverges as T → 0** | smooth data that already spans the needed range; check residuals |
| `PolyFit(n)` | T or c | least squares | **diverges** | only inside the data, e.g. dense QHA on a wide window |
| `StitchedFit(interpolating=SGTE(4), low=None, upp=PolyFit(2), edge=10)` | T | inner model | separate edge fits (default above: a line, i.e. constant entropy) | control extrapolation of a non-Whitney fit |
| `SoftplusFit`, `SplineFit` | c (Softplus also T) | smooth fits | — | special cases |
| `RedlichKister(n)` | c | basis `c(1−c)·(2c−1)^k`, linear least squares, terminals required | — | the default for solution phases with both terminals |

`TemperatureDependentLinePhase` defaults to `SGTE(3)`. That is a CALPHAD
convention, not a recommendation for computed data, so always pass `interpolator=`.

### What went wrong in practice, and the rule each case gives

- **SGTE(3) residuals of 2–5 meV/atom moved melting points by 50–190 K.**
  `PolyFit(8)` on the same dense QHA data stayed within a few K. A transition
  shifts by about `δF/ΔS`. Look at `check_interpolation(plot_error=True)` and
  compare residuals with your error budget, not with "looks fine".
- **PolyFit(8) on short QHA windows** went to −2·10⁷ eV/atom and +16.7 eV/atom
  a few hundred K outside the data. The first visible symptom was
  `ValueError: chemical potential range degenerate` from `guess_mu_range`, which
  names neither the phase nor the fit. When that error appears with phases at
  different compositions, evaluate every `line_free_energy` over the full
  T grid first.
- **SGTE-extrapolated liquids** sank under the solids at low T and created a
  spurious low-T liquid field in six binaries. With constant-entropy
  continuation the liquid kept its entropy and the artefact disappeared.
- **Whitney's constant-entropy extension is an upper bound on F.** F is concave
  in T, so the tangent lies above it. Phases sampled only from a higher T get
  penalised at low T:
  - 12.6 meV/atom at 300 K below a 500 K anchor produced spurious re-entrant
    triple points.
  - A compound that sits on the 0 K hull appeared stable only above 391 K.
- **Window-edge kinks.** Outside the data `C_p = 0`. Two phases with different
  windows then imply an excess heat capacity that jumps at each window edge, and
  that shows up as wiggles (dc/dT reversals) in a liquidus. Extrapolate the
  *excess* over the terminal chord rather than F itself (see the liquid section
  below), or extend the data.
- **Whitney with a T = 0 point** oscillates across the gap (±5–9 meV/atom at
  50–250 K). Do not mix a 0 K point into a Whitney fit; use the anchored SGTE below.
- **Never splice two data sources** (e.g. QHA below and MD above) into one
  `F(T)` array. That produced entropy jumps of up to 0.28 kB. Pick one source
  per phase, or anchor one to the other properly.

### Classical-MD solids: anchor F(0) to the relaxed 0 K energy

For free energies from **classical** MD (no zero-point energy), the relaxed 0 K
energy `E0` from the same potential is an exact data point: `F(0) = E0`. If your
phases' data windows start at different temperatures, fit SGTE with the constant
term held at `E0` and continue at constant entropy above the data. The upstream
proposal is eisenforschung/landau#534 (not merged as of 1.18), so here is a
self-contained version that uses only public API:

```python
from dataclasses import dataclass
import numpy as np, scipy.optimize as so
from landau.interpolate import TemperatureInterpolator, Interpolation, SGTEInterpolation, G_calphad

class _Anchored(Interpolation):
    def __init__(self, sgte, tmax):
        self.sgte, self.tmax = sgte, tmax
        self.ftop, self.stop = sgte(tmax), sgte.deriv()(tmax)
    def __call__(self, T):
        T = np.asarray(T, float)
        v = np.where(T > self.tmax, self.ftop + self.stop * (T - self.tmax),
                     self.sgte(np.minimum(T, self.tmax)))
        return v.item() if v.ndim == 0 else v

@dataclass(frozen=True)
class AnchoredSGTE(TemperatureInterpolator):
    """SGTE with F(0) = E0 held exactly; constant entropy above the data. Pass T including one T=0 sample."""
    nparam: int = 5
    def fit(self, T, F):
        T, F = np.asarray(T, float), np.asarray(F, float)
        z = np.isclose(T, 0.0); assert z.sum() == 1, "need exactly one T=0 sample (the relaxed E0)"
        e0, t, f = float(F[z][0]), T[~z], F[~z]
        p, *_ = so.curve_fit(lambda x, a, *q: G_calphad(x, a, e0, *q), t, f, p0=[0.0] * (self.nparam - 1))
        return _Anchored(SGTEInterpolation((p[0], e0, *p[1:])), t.max())

phase = TemperatureDependentLinePhase("AB2", 2/3, temperatures=np.r_[0.0, T],
                                      free_energies=np.r_[E0, F], interpolator=AnchoredSGTE(5))
```

- **Energy-zero check.** The fitted `T ln T` coefficient `a` should be about
  `−3 kB` per atom, the classical harmonic limit. If `−a/(3 kB)` is far from 1,
  `E0` and `F` do not share an energy zero (different potential, cell, or per-atom
  normalisation).
- **Classical data only.** Quantum (QHA/phonon) free energies have
  `F(0) = E0 + ZPE` and `S → 0` as T → 0. For those, use
  `PhonopyQuasiHarmonicPhase`, or a polynomial fit that stays inside its data
  (upstream #427).
- Do not add `(0, E0)` as an ordinary sample to a free SGTE or Whitney fit.
  Least squares effectively ignores it, and Whitney oscillates.

## Melting and boiling points (unary)

```python
df = calc_phase_diagram([solid, liquid], Ts=np.linspace(Tlo, Thi, 100), mu=0.0)
Tm = get_transitions(df)["T"].unique()
```

- Both phases need data at Tm. Run solid sweeps into the superheated range
  (about 1.05 Tm, with whatever solid-stability check your MD tool has); run
  liquid sweeps into the undercooled range.
- A crossing that falls in one phase's extrapolation is that interpolator's
  opinion, not a result. Say so when you report it.
- With several polymorphs, put all of them in the list. landau picks the lowest
  at each T, and you get the polymorphic transitions for free.

## Solution phases from a composition grid

```python
liq = FastInterpolatingPhase("liquid", [line(f"L{c}", c, T_c, F_c) for c in grid])   # add_entropy=False
```

- `add_entropy`: see SKILL.md. MD of a disordered alloy cell, and alchemical or
  composition-scaling free energies, already contain the configurational
  entropy, so use `False`. Ordered-cell or 0 K energies do not, so use `True`.
- **The default `RedlichKister` interpolates exactly** whenever there are at most
  five interior compositions. There is no smoothing, and the noise of each run
  goes straight into `f(c)` and its curvature.
  - With a single interior run you get only `L0` (a symmetric excess).
  - To smooth, pass fewer coefficients (`interpolator=RedlichKister(2)`), which
    becomes a least-squares fit, and check the residuals.
- **Spurious miscibility gaps** come from the polynomial bending between the last
  interior run and a terminal. Check `d²f/dc² > 0` across the range on the
  `check_concentration_interpolation(plot_excess=True)` plot, and place interior
  runs near both ends.
- Odd RK coefficients have the opposite sign to the CALPHAD `(x_A − x_B)` convention.
- Leave-one-out cross-validation over interpolator order (drop one composition,
  predict it) is a cheap way to choose `n`. In one alloy it chose
  `RedlichKister(3)`; higher orders oscillated.

### Liquids whose compositions have different T windows

Liquid alloy runs often cover different temperature ranges. Fitting each
composition's `F(T)` separately and then interpolating in c makes the excess
heat capacity jump wherever a window ends.

A pattern that reproduced every input exactly at every T:
1. Fit the two terminals with Whitney.
2. For each alloy run, fit the **excess over the terminal chord**,
   `Δ(T) = F(T) − [(1−c) F_A(T) + c F_B(T)]`, with Whitney. This is a
   Neumann–Kopp continuation: the excess, not F, is held at constant entropy
   outside its window.
3. At each T, solve the Redlich–Kister coefficients from the `Δ_i(T)` (with
   fixed compositions this is one precomputable linear map,
   `L = V⁻¹ y` with `V_ik = (2c_i − 1)^k` and `y_i = Δ_i / (c_i(1−c_i))`).
4. Implement it as a `FastInterpolatingPhase` subclass that overrides
   `_get_interpolation(T)` and returns a `RedlichKisterInterpolation(df, f0, L)`
   built from the entropy-stripped values. This is private API, so pin the landau
   version, but landau keeps the analytic c-derivative this way.

### `Surface2DInterpolatingPhase` (one global f(T, c) fit)

- **`CalphadSurface2DInterpolator(num_coeffs=5, coeff_poly_order=2,
  terminal_interpolator=SGTE(4))`.** It fits RK coefficients per sampled T, then a
  **quadratic** in T for each coefficient over `temperature_range`. Everything
  hinges on that range:
  - **Narrow** (where all inputs have data): the quadratic extrapolates outside
    it, and the terminals are reproduced only inside it. Pure-element melting
    points came out 30–40 K off.
  - **Wide** (the whole diagram's range): the quadratic is dominated by
    extrapolated samples and missed interior alloy runs by 2–9 meV/atom, which
    moved a eutectic by about 100 K.
  - Pass `terminal_interpolator=WhitneyTemperatureInterpolator()` (1.13+) when
    the surface is evaluated below the terminals' data.
  - `num_temperature_samples=40` is coarse.
- **`WhitneySurface2DInterpolator`: avoid** (deprecation proposed upstream, #531).
  - It is an isotropic RBF on raw `(T in K, c)`.
  - Its default `smoothing=1e-3` does not interpolate.
  - It gave spurious `d²f/dc² < 0` inside the data.
  - Diagrams ran 8–19× slower.
  - Rescaling T moved invariants by 4–10 K.
- **`SoftplusSurface2DInterpolator`** gives convex fixed-T slices by
  construction, with no terminals required. It is suited to intermetallics with a
  narrow well; `knee_bounds=` needs 1.16+.
- After any surface fit, check that it reproduces the terminals and **each alloy
  input** at their sampled T. Checking only the terminals once let a change pass
  that moved a eutectic by 100 K.

## Feeding typical sources

- **MD thermodynamic integration and reversible scaling** (calphy and similar)
  gives one `F(T)` per phase and composition. Thin it, use Whitney or anchored
  SGTE, and keep the run's T window as metadata.
- **QHA/phonons** (`F(0) = E0 + ZPE`): prefer `PhonopyQuasiHarmonicPhase`,
  which evaluates `min_V[E(V) + F_vib(V,T)]` exactly. Otherwise use a polynomial
  inside the data. Avoid SGTE, whose divergent low-T entropy violates the third law.
- **0 K DFT or MLIP energies of ordered compounds**: use a `LinePhase` with
  `line_entropy=0` for a ground-state hull only. Add vibrational free energy
  before claiming any temperature.
- **Ideal gas**: use `AsePhase` wrapping `ase.thermochemistry.IdealGasThermo`,
  with `atoms_per_formula` for molecules. Line-phase names you give are used
  verbatim in the frame.
- **Your own model**: subclass `AbstractLinePhase` (see `api.md`) instead of
  sampling and refitting.

## Uncertainty you should carry

- About 1 meV/atom of free-energy error is about 10–15 K on a melting point
  (`ΔT ≈ δF/ΔS_fus`, with `ΔS_fus` of roughly 1 kB per atom). For shallow
  solid–solid transitions, where `ΔS` is small, it can be 40+ K per meV/atom.
- Report which parts of the diagram rest on extrapolation. A crossing between
  two extrapolated curves is a model assumption.
