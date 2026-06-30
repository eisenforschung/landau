"""Unit tests for refiners in landau.refine."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from landau.features import Locus
from landau.phases import LinePhase, Phase, TemperatureDependentLinePhase
from landau.interpolate import SGTE
from landau.interpolate.basic import G_calphad
from landau.refine import (
    ClausiusClapeyronRefiner,
    MiscibilityGapRefiner,
    DelaunayLineRefiner,
    DelaunayTripleRefiner,
    RefinedPoint,
    RefinedMiscibilityGap,
    ScanRefiner,
)
from landau.refine import (
    _point_on_line,
    _simplex_straddles,
    _dominated,
    _InterCandidate,
    _delaunay_simplices,
    _simplex_containment,
)


def _two_phase_diagram_df(phases):
    """Coarse sampled (T, mu) grid with each point tagged by its stable phase."""
    Ts = np.linspace(400.0, 1200.0, 9)
    mus = np.linspace(-0.05, 0.05, 11)
    rows = []
    for T in Ts:
        for mu in mus:
            phis = [(p.name, float(p.semigrand_potential(T, mu))) for p in phases]
            name, phi = min(phis, key=lambda kv: kv[1])
            rows.append({"T": T, "mu": mu, "phi": phi,
                         "c": float({p.name: p for p in phases}[name].concentration(T, mu)),
                         "phase": name, "stable": True})
    return pd.DataFrame(rows)


@pytest.fixture
def two_phase_system():
    T_grid = np.linspace(300.0, 1300.0, 25)
    a = TemperatureDependentLinePhase(
        name="A", fixed_concentration=0.2,
        temperatures=T_grid, free_energies=G_calphad(T_grid, 1e-4, -2.0, 5e-4),
        interpolator=SGTE(3),
    )
    b = TemperatureDependentLinePhase(
        name="B", fixed_concentration=0.8,
        temperatures=T_grid, free_energies=G_calphad(T_grid, 1e-4, -1.9, 4e-4),
        interpolator=SGTE(3),
    )
    phases = [a, b]
    return phases, {p.name: p for p in phases}


def test_clausius_clapeyron_refiner_traces_coexistence(two_phase_system):
    phases, mapping = two_phase_system
    df = _two_phase_diagram_df(phases)
    # Sanity: both phases should appear stable somewhere
    assert set(df.phase.unique()) == {"A", "B"}

    refiner = ClausiusClapeyronRefiner(dT_max=100.0)
    out = refiner.run(df, mapping)

    # Each RefinedPoint is expanded to one row per phase, so coexistence
    # points come in pairs.
    assert not out.empty
    assert (out["refined"] == "clausius-clapeyron").all()
    assert out["stable"].all() and out["border"].all()
    # locus is comparable both by enum member and by plain string value.
    assert (out["locus"] == Locus.BOUNDARY).all()
    assert (out["locus"] == "boundary").all()

    # Pair up coexistence rows by (T, mu); each unique location should carry
    # both phase names.
    grouped = out.groupby(["T", "mu"])["phase"].agg(lambda s: tuple(sorted(set(s))))
    assert (grouped == ("A", "B")).all()

    # The refiner should span much of the input T range with many more points
    # than DelaunayLineRefiner would produce on the same df.
    Ts = grouped.index.get_level_values("T").to_numpy()
    assert Ts.min() < 500.0 and Ts.max() > 1100.0
    assert len(Ts) > 5

    # Cross-check accuracy against direct root-finding at a few Ts.
    a_phase, b_phase = mapping["A"], mapping["B"]
    import scipy.optimize as so
    for T_check in [600.0, 900.0, 1100.0]:
        mu_true = so.brentq(
            lambda mu: (a_phase.semigrand_potential(T_check, mu)
                        - b_phase.semigrand_potential(T_check, mu)),
            -0.1, 0.1, xtol=1e-12,
        )
        nearest = Ts[np.argmin(np.abs(Ts - T_check))]
        mu_refined = grouped.index.get_level_values("mu")[
            np.argmin(np.abs(Ts - T_check))
        ]
        # The tracer hits its own T grid, not these specific Ts, but at any
        # converged point the residual should be near bisection tolerance.
        rows = out[(out["T"] == nearest)]
        assert len(rows) == 2
        phi_a = a_phase.semigrand_potential(nearest, rows["mu"].iloc[0])
        phi_b = b_phase.semigrand_potential(nearest, rows["mu"].iloc[0])
        assert abs(phi_a - phi_b) < 1e-7


def test_clausius_clapeyron_refiner_skips_straddling_simplices(two_phase_system):
    """Many Delaunay simplices straddle the same coexistence line; the
    refiner should trace it once and skip the rest."""
    phases, mapping = two_phase_system
    df = _two_phase_diagram_df(phases)
    refiner = ClausiusClapeyronRefiner()
    cands = list(refiner.propose(df))
    # The grid does have many two-phase simplices for the same pair...
    pairs = [frozenset((c.phase1, c.phase2)) for c in cands]
    assert pairs.count(frozenset(("A", "B"))) > 1
    # ...but run() should skip retraces and emit roughly one line's worth:
    # the point count tracks a single T-sweep (span / dT_max), not ~one
    # full trace per straddling simplex stacked on the same line.
    out = refiner.run(df, mapping)
    n_points = out.groupby(["T", "mu"]).ngroups
    T_span = df["T"].max() - df["T"].min()
    assert n_points < 3 * T_span / refiner.dT_max


def test_clausius_clapeyron_refiner_respects_dT_min(two_phase_system):
    phases, mapping = two_phase_system
    df = _two_phase_diagram_df(phases)
    refiner = ClausiusClapeyronRefiner(dT_min=20.0, dT_max=50.0)
    out = refiner.run(df, mapping)
    Ts = np.sort(out["T"].unique())
    dTs = np.diff(Ts)
    # Median step honors dT_min; the only exceptions are the truncation
    # at each trace's boundary toward T_min / T_max.
    assert np.median(dTs) >= 20.0
    assert np.median(dTs) <= 50.0


def test_clausius_clapeyron_refiner_label():
    assert ClausiusClapeyronRefiner.label == "clausius-clapeyron"


# -- dc_max concentration-drift cap ------------------------------------------

# Physical drift slope: c sweeps ~0.8 -> 0.1 across the sampled T range.
_DRIFT_SLOPE = -0.7 / 1100.0
# Tight enough that the cap binds below the default dT_min = 1.0 K, so the
# "dT_min yields to dc_max" path is exercised.
_DC_TIGHT = 5e-4


@dataclass(frozen=True)
class _DriftLinePhase(Phase):
    """Line-like phase whose plotted composition drifts linearly with T.

    ``phi(T, mu) = e - mu * c(T)`` so ``c = -dphi/dmu = c(T)`` exactly and
    sweeps with T at fixed mu.  Two of these with equal ``e`` coexist at
    ``mu = 0`` for every T — a boundary that is exactly flat in mu — which
    isolates the ``dc_max`` density floor: ``_dT_adapt`` saturates at
    ``dT_max`` because ``dmu/dT = 0``, yet ``c`` still moves, so only the
    concentration cap limits the step.
    """

    e: float = 0.0
    c0: float = 0.5
    slope: float = 0.0
    T0: float = 300.0

    def _c(self, T):
        return self.c0 + self.slope * (np.asarray(T, float) - self.T0)

    def semigrand_potential(self, T, mu):
        return self.e - np.asarray(mu, float) * self._c(T)

    def concentration(self, T, mu):
        return self._c(T) + 0.0 * np.asarray(mu, float)


def _drift_candidate(T_min=300.0, T_max=1400.0):
    T_seed = (T_min + T_max) / 2.0
    return _InterCandidate(
        phase1="P", phase2="Q", T_seed=T_seed,
        mu_bracket=(-0.05, 0.05), T_bracket=(T_seed - 50.0, T_seed + 50.0),
        T_min=T_min, T_max=T_max,
        proj_p1=(T_seed - 50.0, -0.1), proj_p2=(T_seed + 50.0, 0.1),
    )


def _solve_drift(dc_max, p_slope, T_min=300.0, T_max=1400.0):
    P = _DriftLinePhase(name="P", e=-2.0, c0=0.8, slope=p_slope)
    Q = _DriftLinePhase(name="Q", e=-2.0, c0=0.05, slope=0.0)
    pts = ClausiusClapeyronRefiner(dc_max=dc_max).solve(
        _drift_candidate(T_min, T_max), {"P": P, "Q": Q})
    return P, sorted(pts, key=lambda p: p.T)


def test_cc_refiner_dc_max_bounds_concentration_drift():
    """On a boundary flat in mu but sweeping in c, dc_max caps every
    per-step concentration jump, and dT_min yields so the cap holds."""
    # A short T window keeps c physical while dc_max binds below dT_min.
    P, pts = _solve_drift(_DC_TIGHT, p_slope=_DRIFT_SLOPE, T_min=800.0, T_max=900.0)
    # Regime check: the located boundary is exactly flat in mu, so the
    # mu-drift step heuristic alone would saturate at dT_max.
    assert max(abs(p.mu) for p in pts) < 1e-9
    # The fine stepping does not truncate the walk: it spans the window
    # in both directions.
    Ts = np.array([p.T for p in pts])
    assert Ts.min() < 805.0 and Ts.max() > 895.0
    # Every consecutive step keeps the plotted concentration drift under
    # the cap (priming from the seed leaves no coarse bootstrap step).
    cs = np.array([float(P.concentration(p.T, p.mu)) for p in pts])
    assert np.abs(np.diff(cs)).max() <= _DC_TIGHT + 1e-9
    # dT_min (default 1.0 K) gives way to the drift target: steps go finer.
    assert np.diff(Ts).min() < 1.0


def test_cc_refiner_dc_max_densifies_curved_boundary():
    """Tightening dc_max adds samples on a curved-in-c boundary that the
    mu-drift heuristic alone leaves coarse."""
    _, loose = _solve_drift(1e9, p_slope=_DRIFT_SLOPE, T_min=800.0, T_max=900.0)
    _, tight = _solve_drift(_DC_TIGHT, p_slope=_DRIFT_SLOPE, T_min=800.0, T_max=900.0)
    assert len(tight) > len(loose)


def test_cc_refiner_dc_max_noop_on_constant_c_boundary():
    """A boundary straight in c (dc/dT ~ 0) must not be over-sampled:
    dc_max never engages, so the point count is identical whether the cap
    is tight or effectively off."""
    _, tight = _solve_drift(1e-4, p_slope=0.0)
    _, loose = _solve_drift(1e9, p_slope=0.0)
    assert len(loose) > 3  # genuinely traced, not just the seed
    assert len(tight) == len(loose)


# -- dc_min concentration-drift floor ----------------------------------------


@dataclass(frozen=True)
class _SteepMuFlatCPhase(Phase):
    """Toy phase with decoupled boundary slope and plotted concentration.

    ``phi(T, mu) = -mu * a - k * (T - T0)`` so a coexistence pair with
    different ``a`` locates ``mu* = -(k1 - k2)/(a1 - a2) * (T - T0)`` — a
    boundary whose mu-slope is set by ``k`` alone. ``concentration`` is
    defined independently as a near-flat ``c0 + cslope * (T - T0)``; the toy
    need not satisfy ``c = -dphi/dmu``, so the boundary can be steep in mu
    (``_dT_adapt`` pins the step at ``dT_min``) while c barely drifts —
    exactly the regime the ``dc_min`` density ceiling targets.
    """

    a: float = 0.0
    k: float = 0.0
    c0: float = 0.5
    cslope: float = 0.0
    T0: float = 850.0

    def semigrand_potential(self, T, mu):
        return -np.asarray(mu, float) * self.a - self.k * (np.asarray(T, float) - self.T0)

    def concentration(self, T, mu):
        return self.c0 + self.cslope * (np.asarray(T, float) - self.T0) + 0.0 * np.asarray(mu, float)


def _steep_candidate(T_min=840.0, T_max=860.0, T_c=850.0):
    return _InterCandidate(
        phase1="A", phase2="B", T_seed=T_c,
        mu_bracket=(-0.05, 0.05), T_bracket=(T_c - 5.0, T_c + 5.0),
        T_min=T_min, T_max=T_max,
        proj_p1=(T_c, -0.1), proj_p2=(T_c, 0.1),
    )


def _solve_steep(dc_min, cslope=1e-3, K=0.025):
    # mu* = K * (T - 850); |dmu/dT| = K, so _dT_adapt = half_width / K = 2 K,
    # finer than dT_max = 5 K (the boundary is over-sampled) yet the dT_min
    # bootstrap shift K * 1 = 0.025 stays inside the half-width bracket.
    A = _SteepMuFlatCPhase(name="A", a=1.0, k=0.0, c0=0.5, cslope=cslope)
    B = _SteepMuFlatCPhase(name="B", a=0.0, k=K, c0=0.2, cslope=0.0)
    pts = ClausiusClapeyronRefiner(dc_min=dc_min).solve(
        _steep_candidate(), {"A": A, "B": B})
    return A, sorted(pts, key=lambda p: p.T)


def test_cc_refiner_dc_min_floors_concentration_drift():
    """On a steep-in-mu, flat-in-c boundary the floor grows each steady step
    until the plotted concentration drifts exactly dc_min (capped at dT_max)."""
    A, pts = _solve_steep(dc_min=4e-3)
    assert len(pts) > 4  # genuinely traced
    order = np.argsort([p.T for p in pts])
    Ts = np.array([p.T for p in pts])[order]
    cs = np.array([float(A.concentration(p.T, p.mu)) for p in pts])[order]
    # Steady steps sit on the floor: dT = dc_min / cslope = 4 K, dc = dc_min.
    # Nothing drifts more (dc_max is slack, dT_max = 5 K is not reached); the
    # bootstrap and the truncated end steps drift less.
    assert np.abs(np.diff(cs)).max() == pytest.approx(4e-3, abs=1e-9)
    assert np.diff(Ts).max() == pytest.approx(4.0, abs=1e-9)


def test_cc_refiner_dc_min_thins_oversampled_boundary():
    """Without the floor the over-sampled steep-in-mu boundary steps at the
    bare _dT_adapt size; the floor coarsens it to fewer points."""
    _, dense = _solve_steep(dc_min=0.0)
    _, thin = _solve_steep(dc_min=4e-3)
    # Regime check: with the floor off every step is the 2 K _dT_adapt size.
    assert np.diff(sorted(p.T for p in dense)).max() == pytest.approx(2.0, abs=1e-9)
    assert len(thin) < len(dense)


def test_cc_refiner_dc_min_noop_when_drift_already_met():
    """A boundary whose bare step already drifts past dc_min never engages the
    floor: identical point count whether it is set or off."""
    _, off = _solve_steep(dc_min=0.0, cslope=3e-3)
    _, on = _solve_steep(dc_min=4e-3, cslope=3e-3)
    assert len(off) > 4  # genuinely traced
    assert len(on) == len(off)


def test_simplex_straddles_segment_crossing():
    """A simplex with no traced vertex inside still gets skipped if a
    segment of the traced line crosses its bounding box."""
    # Trace: two points T=100→500, mu sweeping -1 → +1.
    cand = _InterCandidate(
        phase1="A", phase2="B",
        T_seed=300.0,
        mu_bracket=(-0.1, 0.1),
        T_bracket=(250.0, 350.0),  # neither traced T is inside
        T_min=0.0, T_max=1000.0,
        proj_p1=(250.0, -0.1), proj_p2=(350.0, 0.1),
    )
    traces = (((100.0, -1.0), (500.0, 1.0)),)
    assert _simplex_straddles(cand, traces)
    # Shift the bbox below the line; no longer crosses.
    cand2 = replace_candidate(cand, mu_bracket=(-2.0, -1.5))
    assert not _simplex_straddles(cand2, traces)


def _make_inter_cand(*, mu_bracket, T_bracket=(250.0, 350.0)):
    """Helper: minimal _InterCandidate for straddle-only tests."""
    return _InterCandidate(
        phase1="A", phase2="B", T_seed=sum(T_bracket) / 2,
        mu_bracket=mu_bracket, T_bracket=T_bracket,
        T_min=0.0, T_max=1000.0,
        proj_p1=(T_bracket[0], mu_bracket[0]),
        proj_p2=(T_bracket[1], mu_bracket[1]),
    )


def test_simplex_straddles_single_point_trace_inside_bbox():
    """A previous trace that emitted only its seed point should still
    block any later simplex whose inflated bbox contains it."""
    cand = _make_inter_cand(mu_bracket=(0.10, 0.15))
    seed_inside = (((300.0, 0.12),),)
    assert _simplex_straddles(cand, seed_inside)


def test_simplex_straddles_single_point_trace_outside_bbox():
    """A single-point trace far away in mu must not block a candidate
    in an unrelated region of mu."""
    cand = _make_inter_cand(mu_bracket=(0.10, 0.15))  # inflated to [0.05, 0.20]
    seed_far = (((300.0, 2.0),),)
    assert not _simplex_straddles(cand, seed_far)


def test_simplex_straddles_disjoint_seed_traces_do_not_fake_a_line():
    """Two single-point traces at widely separated mus must NOT be
    treated as a connected line — otherwise a simplex sitting between
    them would get spuriously blocked.

    Regression test for the kink near the s3 wedge in the toy 2d_toy_mu
    plot: separate seed-only traces at T~906 K were being concatenated
    into a fake polyline that swept the whole mu range at that T,
    blocking the real left-side liquid/solid trace.
    """
    cand = _make_inter_cand(mu_bracket=(0.45, 0.55))  # mid-range
    # Two disjoint seeds at the same T, far from the bbox:
    traces = (
        ((300.0, -0.30),),  # to the left of cand
        ((300.0, +2.10),),  # to the right of cand
    )
    assert not _simplex_straddles(cand, traces)


def replace_candidate(cand, **kw):
    from dataclasses import replace
    return replace(cand, **kw)


def test_point_on_line_helper():
    traces = (((100.0, 0.0), (200.0, 1.0), (300.0, 2.0)),)
    assert _point_on_line(150.0, 0.5, traces, tol_mu=0.01)
    assert _point_on_line(250.0, 1.5, traces, tol_mu=0.01)
    # Outside the trace's T range -- distance to nearest endpoint
    # dominates and exceeds tol.
    assert not _point_on_line(50.0, 0.0, traces, tol_mu=0.1)
    assert not _point_on_line(400.0, 2.0, traces, tol_mu=0.1)
    # Too far in mu
    assert not _point_on_line(150.0, 5.0, traces, tol_mu=0.1)
    # Two disjoint traces: hit on the second one only.
    traces2 = (((0.0, 0.0),), ((100.0, 10.0), (200.0, 10.0)))
    assert _point_on_line(150.0, 10.0, traces2, tol_mu=0.01)
    assert not _point_on_line(150.0, 0.0, traces2, tol_mu=0.01)


def test_clausius_clapeyron_refiner_interphase_idealsolution():
    """Inter-phase (solid/liquid) boundary in an ideal-solution binary,
    lifted from notebooks/IdealSolution.ipynb. Refiner should trace
    the coexistence line with both phase names present at every point."""
    from scipy.constants import Boltzmann, eV
    from landau.phases import LinePhase, IdealSolution
    from landau.calculate import calc_phase_diagram, refine_phase_diagram

    kB = Boltzmann / eV
    solid_a = LinePhase('A',    fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB)
    solid_b = LinePhase('B',    fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB)
    liquid_a = LinePhase('A(l)', fixed_concentration=0, line_energy=-1.9, line_entropy=2.5 * kB)
    liquid_b = LinePhase('B(l)', fixed_concentration=1, line_energy=-2.9, line_entropy=2.2 * kB)
    solid  = IdealSolution('solid',  solid_a,  solid_b)
    liquid = IdealSolution('liquid', liquid_a, liquid_b)

    Ts = np.linspace(200, 1800, 25)
    mus = np.linspace(-0.3, 0.3, 21)
    coarse = calc_phase_diagram(
        [solid, liquid], Ts=Ts, mu=mus, refine=False, keep_unstable=False)
    out = refine_phase_diagram(
        coarse, {'solid': solid, 'liquid': liquid},
        refiners=[ClausiusClapeyronRefiner()])
    cc = out[out['refined'] == 'clausius-clapeyron']

    assert len(cc) > 0
    pairs = cc.groupby(['T', 'mu'])['phase'].apply(
        lambda s: tuple(sorted(s.unique())))
    # Every refined coexistence point carries one row per phase.
    assert (pairs == ('liquid', 'solid')).all()
    assert (cc['refined'] == 'clausius-clapeyron').all()
    # Trace should span a substantial part of the supplied T range.
    assert cc['T'].max() - cc['T'].min() > 500


def test_miscibility_gap_refiner_regular_solution():
    """Intra-phase gap of a regular solution with repulsive
    interaction: refiner should trace mu*(T) ≈ 0 across the gap."""
    from scipy.constants import Boltzmann, eV
    from landau.phases import RegularSolution
    from landau.calculate import calc_phase_diagram, refine_phase_diagram

    kB = Boltzmann / eV
    L0 = 0.1
    T_c = L0 / (2 * kB)
    left  = LinePhase(name='left',  fixed_concentration=0.0,
                      line_energy=0.0, line_entropy=0.0)
    mid   = LinePhase(name='mid',   fixed_concentration=0.5,
                      line_energy=L0 / 4, line_entropy=0.0)
    right = LinePhase(name='right', fixed_concentration=1.0,
                      line_energy=0.0, line_entropy=0.0)
    sol = RegularSolution(name='sol', phases=[left, mid, right],
                          num_coeffs=1, add_entropy=True)
    Ts = np.linspace(150.0, T_c - 20.0, 15)
    mus = np.linspace(-0.05, 0.05, 13)
    coarse = calc_phase_diagram([sol], Ts=Ts, mu=mus,
                                refine=False, keep_unstable=True)
    out = refine_phase_diagram(coarse, {'sol': sol},
                               refiners=[MiscibilityGapRefiner()])
    cc = out[out['refined'] == 'miscibility-gap']
    assert len(cc) >= 5
    # Symmetric regular solution: mu* = 0 exactly. The argmax-of-dcs
    # localizer is data-driven (no midpoint assumption), so we allow a
    # few-meV slack across the trace.
    assert np.median(np.abs(cc['mu'])) < 5e-3
    # All emitted rows are tagged with the single phase name.
    assert (cc['phase'] == 'sol').all()


def test_miscibility_gap_refiner_asymmetric_subregular():
    """Sub-regular solution f_mix = c(1-c)(L0 + L1(2c-1)) with L1 != 0
    breaks c <-> 1-c symmetry, so mu*(T) != 0 and c_left + c_right != 1.
    Builds the phase as a FastInterpolatingPhase fitted to control
    points whose line energies sample the analytical f_mix; the
    refiner should trace the asymmetric binodal without complaint."""
    from scipy.constants import Boltzmann, eV
    from landau.phases import FastInterpolatingPhase
    from landau.interpolate import RedlichKister
    from landau.calculate import calc_phase_diagram, refine_phase_diagram

    kB = Boltzmann / eV
    L0, L1 = 0.10, 0.04  # cc_demo.py parameters

    def f_mix(c):
        return c * (1 - c) * (L0 + L1 * (2 * c - 1))

    control_cs = (0.0, 0.25, 0.5, 0.75, 1.0)
    line_phases = [
        LinePhase(name=f'p{i}', fixed_concentration=c,
                  line_energy=f_mix(c), line_entropy=0.0)
        for i, c in enumerate(control_cs)
    ]
    sub = FastInterpolatingPhase(
        name='sub', phases=line_phases,
        add_entropy=True, interpolator=RedlichKister(2),
    )

    # Analytical T_c ~ 723 K, c_c ~ 0.68 for this L0/L1 combination
    # (cc_demo.py); sample below that.
    Ts = np.linspace(150, 700, 18)
    mus = np.linspace(-0.05, 0.05, 21)
    coarse = calc_phase_diagram([sub], Ts=Ts, mu=mus,
                                refine=False, keep_unstable=True)
    out = refine_phase_diagram(
        coarse, {'sub': sub}, refiners=[MiscibilityGapRefiner()])
    cc = out[out['refined'] == 'miscibility-gap']
    assert len(cc) >= 10

    # The trace should be visibly asymmetric in mu and in c.
    assert np.median(np.abs(cc['mu'])) > 1e-3, \
        "mu*(T) should be nonzero for an asymmetric gap"

    pairs = cc.groupby(['T', 'mu'])['c'].agg(
        lambda s: (float(min(s)), float(max(s))))
    sums = np.array([cl + cr for cl, cr in pairs])
    # Symmetric would give sums == 1; asymmetric drifts away from it.
    assert (sums > 1.05).any(), \
        "c_left + c_right should depart from 1 for an asymmetric gap"
    # All pairs are physically ordered.
    for cl, cr in pairs:
        assert 0.0 <= cl < cr <= 1.0


def test_miscibility_gap_refiner_auto_stops_above_T_c():
    """When the supplied Ts range crosses T_c the trace must stop on its
    own (gap_close / gap_share_min) instead of running to T_max.

    Regression test for the earlier overshoot, where the trace kept
    walking 100+ K past the critical point and the only thing stopping
    it was the data boundary.
    """
    from scipy.constants import Boltzmann, eV
    from landau.phases import RegularSolution
    from landau.calculate import calc_phase_diagram, refine_phase_diagram

    kB = Boltzmann / eV
    L0 = 0.1
    T_c = L0 / (2 * kB)  # ~580 K
    left  = LinePhase(name='left',  fixed_concentration=0.0,
                      line_energy=0.0, line_entropy=0.0)
    mid   = LinePhase(name='mid',   fixed_concentration=0.5,
                      line_energy=L0 / 4, line_entropy=0.0)
    right = LinePhase(name='right', fixed_concentration=1.0,
                      line_energy=0.0, line_entropy=0.0)
    sol = RegularSolution(name='sol', phases=[left, mid, right],
                          num_coeffs=1, add_entropy=True)
    Ts_max = T_c + 250.0  # well above T_c
    Ts = np.linspace(150.0, Ts_max, 25)
    mus = np.linspace(-0.05, 0.05, 21)
    coarse = calc_phase_diagram([sol], Ts=Ts, mu=mus,
                                refine=False, keep_unstable=True)
    out = refine_phase_diagram(coarse, {'sol': sol},
                               refiners=[MiscibilityGapRefiner()])
    cc = out[out['refined'] == 'miscibility-gap']
    assert not cc.empty
    # Trace must stop well before T_max.
    assert cc['T'].max() < Ts_max - 50.0


def test_refined_miscibility_gap_emits_two_rows_with_equal_mu():
    """RefinedMiscibilityGap expands to two rows that share an exact mu
    value and carry the c_left / c_right values straight from the
    scan, without re-querying the phase (whose concentration() may
    quantise or collapse to one branch for sharp gaps)."""

    class StickyPhase:
        """Always returns the same c regardless of mu — would yield a
        single-branch result if to_rows re-queried instead of using
        the stored scan values."""
        name = "p"

        def semigrand_potential(self, T, mu):
            return 0.0

        def concentration(self, T, mu):
            return 0.5

    pt = RefinedMiscibilityGap(
        T=400.0, mu=0.0, phase="p", c_left=0.1, c_right=0.9)
    rows = pt.to_rows({"p": StickyPhase()})
    assert len(rows) == 2
    assert rows[0]["mu"] == rows[1]["mu"] == 0.0
    # Even though the phase's concentration() collapses to 0.5, the
    # row's c column reflects the c_left / c_right stored on the dataclass.
    assert {round(rows[0]["c"], 3), round(rows[1]["c"], 3)} == {0.1, 0.9}


def test_clausius_clapeyron_refiner_no_two_phase_simplex():
    """Empty result when the input has no two-phase coexistence."""
    p = LinePhase(name="solo", fixed_concentration=0.5,
                  line_energy=-1.0, line_entropy=0.0)
    rows = []
    for T in np.linspace(300, 1000, 5):
        for mu in np.linspace(-0.1, 0.1, 5):
            rows.append({"T": T, "mu": mu,
                         "phi": float(p.semigrand_potential(T, mu)),
                         "c": 0.5, "phase": "solo", "stable": True})
    df = pd.DataFrame(rows)
    refiner = ClausiusClapeyronRefiner()
    out = refiner.run(df, {"solo": p})
    assert out.empty
    # MiscibilityGapRefiner is the right tool here but the simplices
    # have no c-spread (all c=0.5), so it also yields nothing.
    gap_out = MiscibilityGapRefiner().run(df, {"solo": p})
    assert gap_out.empty


def _three_phase_system():
    """Three LinePhases with a single triple point at (T=300 K, mu=0.2 eV).

    Coexistence curves (derived from equal-phi conditions):
      A-B: mu = 0.002*T - 0.4
      A-C: mu = 0.003*T - 0.7
      B-C: mu = 0.004*T - 1.0
    All three meet at T=300, mu=0.2.
    """
    ph_a = LinePhase(name='A', fixed_concentration=0.0,
                     line_energy=-1.0, line_entropy=0.004)
    ph_b = LinePhase(name='B', fixed_concentration=0.5,
                     line_energy=-1.2, line_entropy=0.003)
    ph_c = LinePhase(name='C', fixed_concentration=1.0,
                     line_energy=-1.7, line_entropy=0.001)
    return {'A': ph_a, 'B': ph_b, 'C': ph_c}


def _coarse_df(phases, Ts, mus):
    rows = []
    for T in Ts:
        for mu in mus:
            phis = {n: float(p.semigrand_potential(T, mu))
                    for n, p in phases.items()}
            name = min(phis, key=phis.get)
            rows.append({"T": T, "mu": mu, "phi": phis[name],
                         "c": float(phases[name].concentration(T, mu)),
                         "phase": name, "stable": True})
    return pd.DataFrame(rows)


def test_delaunay_triple_refiner_deduplicates():
    """Triple refiner emits each triple point exactly once even when
    multiple three-phase Delaunay simplices independently detect it."""
    phases = _three_phase_system()
    # Coarse grid: step ~50 K × ~0.1 eV, triple point (300, 0.2) lies
    # between grid lines so several adjacent simplices are three-phase.
    Ts = np.linspace(220.0, 480.0, 6)
    mus = np.linspace(-0.05, 0.55, 7)
    df = _coarse_df(phases, Ts, mus)

    n_triple = sum(1 for _, n in _delaunay_simplices(df) if n == 3)
    assert n_triple > 1, "grid should produce multiple three-phase simplices"

    out = DelaunayTripleRefiner().run(df, phases)

    assert not out.empty
    assert (out["refined"] == "delaunay-triple").all()
    assert (out["locus"] == Locus.TRIPLE).all()
    # Exactly one triple point → 3 rows (one per phase).
    assert len(out) == 3
    assert set(out["phase"]) == {"A", "B", "C"}
    assert np.allclose(out["T"], 300.0, atol=10.0)
    assert np.allclose(out["mu"], 0.2, atol=0.05)


def test_delaunay_triple_solve_is_pure_and_simplex_owned():
    """``solve`` only emits from the simplex that owns the triple point, and is
    a pure function of its candidate (no dedup state)."""
    phases = _three_phase_system()
    Ts = np.linspace(220.0, 480.0, 6)
    mus = np.linspace(-0.05, 0.55, 7)
    df = _coarse_df(phases, Ts, mus)

    refiner = DelaunayTripleRefiner()
    cands = list(refiner.propose(df))
    assert len(cands) > 1, "grid should produce multiple three-phase simplices"

    emitting = [c for c in cands if refiner.solve(c, phases)]
    # The point is attributed to exactly one owning simplex.
    assert len(emitting) == 1

    # Pure: re-solving the same candidates yields the same partition; the
    # previously-emitting simplex still emits (old self._found would mute it).
    assert [c for c in cands if refiner.solve(c, phases)] == emitting

    pt = refiner.solve(emitting[0], phases)[0]
    assert np.isclose(pt.T, 300.0, atol=10.0)
    assert np.isclose(pt.mu, 0.2, atol=0.05)


def test_simplex_containment_scores_ownership():
    """``_simplex_containment`` is the affine-invariant ownership score: ``>= 0``
    when the point is inside, negative outside, and largest for the simplex the
    point is least far outside of — the fallback that attributes a triple point
    landing just past every three-phase simplex to a single owner."""
    inside = pd.DataFrame({"T": [0.0, 2.0, 0.0], "mu": [0.0, 0.0, 2.0]})
    # Centroid is strictly inside, edge midpoint sits on the boundary.
    assert _simplex_containment((0.5, 0.5), inside) > 0
    assert np.isclose(_simplex_containment((1.0, 0.0), inside), 0.0)

    # A point past the hypotenuse is outside; the simplex it is least far
    # outside of wins ``max``.
    near = pd.DataFrame({"T": [0.0, 2.0, 0.0], "mu": [0.0, 0.0, 2.0]})
    far = pd.DataFrame({"T": [0.0, -2.0, 0.0], "mu": [0.0, 0.0, -2.0]})
    point = (1.1, 1.1)
    assert _simplex_containment(point, near) < 0
    assert _simplex_containment(point, far) < _simplex_containment(point, near)
    assert max((near, far), key=lambda s: _simplex_containment(point, s)) is near

    # Degenerate (collinear) simplex never wins ownership.
    line = pd.DataFrame({"T": [0.0, 1.0, 2.0], "mu": [0.0, 1.0, 2.0]})
    assert _simplex_containment((0.5, 0.5), line) == float("-inf")


def test_boundary_id_cc_refiner_single_line(two_phase_system):
    """All rows from a single two-phase trace share one boundary_id."""
    phases, mapping = two_phase_system
    df = _two_phase_diagram_df(phases)
    out = ClausiusClapeyronRefiner(dT_max=100.0).run(df, mapping)

    assert "boundary_id" in out.columns
    # Two-phase system → one coexistence line → all rows share the same id.
    assert out["boundary_id"].nunique() == 1


def test_boundary_id_cc_refiner_two_lines():
    """Rows from different coexistence lines get distinct boundary_ids."""
    # Three-phase system has A-B and A-C (and possibly B-C) coexistence lines.
    phases = _three_phase_system()
    Ts = np.linspace(220.0, 480.0, 12)
    mus = np.linspace(-0.05, 0.55, 15)
    df = _coarse_df(phases, Ts, mus)
    out = ClausiusClapeyronRefiner(dT_max=100.0).run(df, phases)

    assert not out.empty
    assert "boundary_id" in out.columns
    # Three-phase system → at least two distinct coexistence lines.
    assert out["boundary_id"].nunique() >= 2
    # Each boundary_id group should contain rows from at most two phases.
    for _bid, group in out.groupby("boundary_id"):
        assert group["phase"].nunique() <= 2


def test_boundary_id_miscibility_gap_refiner():
    """MiscibilityGapRefiner assigns a single boundary_id to all gap rows."""
    from scipy.constants import Boltzmann, eV
    from landau.phases import RegularSolution
    from landau.calculate import calc_phase_diagram, refine_phase_diagram

    kB = Boltzmann / eV
    L0 = 0.1
    T_c = L0 / (2 * kB)
    left = LinePhase(name="left", fixed_concentration=0.0, line_energy=0.0, line_entropy=0.0)
    mid = LinePhase(name="mid", fixed_concentration=0.5, line_energy=L0 / 4, line_entropy=0.0)
    right = LinePhase(name="right", fixed_concentration=1.0, line_energy=0.0, line_entropy=0.0)
    sol = RegularSolution(name="sol", phases=[left, mid, right], num_coeffs=1, add_entropy=True)
    Ts = np.linspace(150.0, T_c - 20.0, 15)
    mus = np.linspace(-0.05, 0.05, 13)
    coarse = calc_phase_diagram([sol], Ts=Ts, mu=mus, refine=False, keep_unstable=True)
    out = refine_phase_diagram(coarse, {"sol": sol}, refiners=[MiscibilityGapRefiner()])
    cc = out[out["refined"] == "miscibility-gap"]

    assert not cc.empty
    assert "boundary_id" in cc.columns
    # One miscibility gap → one boundary_id.
    assert cc["boundary_id"].nunique() == 1
    # Gap branches are coexistence points, not triple points.
    assert (cc["locus"] == Locus.BOUNDARY).all()


def test_boundary_id_refined_point_to_rows():
    """RefinedPoint.to_rows propagates boundary_id into every emitted row."""
    ph = LinePhase(name="x", fixed_concentration=0.3, line_energy=-1.0, line_entropy=0.0)
    pt = RefinedPoint(T=500.0, mu=0.05, phases=("x",), boundary_id=7)
    rows = pt.to_rows({"x": ph})
    assert all(row["boundary_id"] == 7 for row in rows)


def test_boundary_id_refined_miscibility_gap_to_rows():
    """RefinedMiscibilityGap.to_rows propagates boundary_id into both rows."""
    class _Phase:
        name = "p"

        def semigrand_potential(self, T, mu):
            return 0.0

    pt = RefinedMiscibilityGap(T=400.0, mu=0.0, phase="p",
                               c_left=0.1, c_right=0.9, boundary_id=3)
    rows = pt.to_rows({"p": _Phase()})
    assert len(rows) == 2
    assert all(row["boundary_id"] == 3 for row in rows)


def test_locus_refined_point_to_rows():
    """RefinedPoint.to_rows tags rows by phase count: two coexisting phases
    make a boundary point, three a triple point."""
    mapping = {
        n: LinePhase(name=n, fixed_concentration=c, line_energy=-1.0, line_entropy=0.0)
        for n, c in [("x", 0.1), ("y", 0.5), ("z", 0.9)]
    }
    pair = RefinedPoint(T=500.0, mu=0.05, phases=("x", "y"))
    assert all(row["locus"] is Locus.BOUNDARY for row in pair.to_rows(mapping))
    triple = RefinedPoint(T=500.0, mu=0.05, phases=("x", "y", "z"))
    assert all(row["locus"] is Locus.TRIPLE for row in triple.to_rows(mapping))


def test_locus_refined_miscibility_gap_to_rows():
    """RefinedMiscibilityGap.to_rows tags both branch rows as boundary."""
    class _Phase:
        name = "p"

        def semigrand_potential(self, T, mu):
            return 0.0

    pt = RefinedMiscibilityGap(T=400.0, mu=0.0, phase="p", c_left=0.1, c_right=0.9)
    rows = pt.to_rows({"p": _Phase()})
    assert all(row["locus"] is Locus.BOUNDARY for row in rows)


def test_boundary_id_delaunay_triple_rows_share_id():
    """DelaunayTripleRefiner emits boundary_id; all rows of one triple share it."""
    phases = _three_phase_system()
    Ts = np.linspace(220.0, 480.0, 6)
    mus = np.linspace(-0.05, 0.55, 7)
    df = _coarse_df(phases, Ts, mus)
    out = DelaunayTripleRefiner().run(df, phases)

    assert "boundary_id" in out.columns
    # One triple point (3 rows) → all rows share the same boundary_id.
    assert out["boundary_id"].nunique() == 1


# -- ScanRefiner --------------------------------------------------------------

SCAN_ATOL = 1e-6  # xtol of _find_one_point


def _narrow_window_system():
    """Three LinePhases where B is stable only in mu = (2.4, 2.6).

    With phi = E - mu*c the stable phase along mu at any T is A below 2.4,
    B inside (2.4, 2.6), and C above 2.6.  On an integer mu grid no sample
    ever sees B stable, so a scan only shows an A→C change between mu=2 and
    mu=3 while the metastable A-C crossing at mu=2.5 is dominated by B.
    """
    ph_a = LinePhase(name="A", fixed_concentration=0.0, line_energy=0.0)
    ph_b = LinePhase(name="B", fixed_concentration=0.5, line_energy=1.2)
    ph_c = LinePhase(name="C", fixed_concentration=1.0, line_energy=2.5)
    return {"A": ph_a, "B": ph_b, "C": ph_c}


def test_scan_refiner_locates_pairwise_transition():
    """Two-phase scan: the exact crossing is found within root-finder tolerance."""
    phases = _narrow_window_system()
    del phases["B"]
    df = _coarse_df(phases, [300.0], np.linspace(0.0, 4.0, 5))
    out = ScanRefiner("mu").run(df, phases)
    # A-C crossing at phi_A = phi_C: mu = 2.5; one point, one row per phase.
    assert sorted(out["phase"]) == ["A", "C"]
    np.testing.assert_allclose(out["mu"], 2.5, atol=SCAN_ATOL)
    assert out["stable"].all() and out["border"].all()


def test_scan_refiner_splits_dominated_crossing():
    """A stable window narrower than the grid spacing yields both real transitions.

    The A-C crossing at mu=2.5 is dominated by B, so the refiner must recurse
    and return the A-B and B-C transitions instead of dropping the candidate
    (which left no border row at all between two stably-sampled phases).
    """
    phases = _narrow_window_system()
    df = _coarse_df(phases, [300.0], np.linspace(0.0, 4.0, 5))
    assert set(df["phase"]) == {"A", "C"}, "grid must not sample B stable"
    out = ScanRefiner("mu").run(df, phases)
    by_mu = out.groupby("mu")["phase"].agg(lambda s: tuple(sorted(s)))
    assert len(by_mu) == 2
    # phi_A = phi_B at mu = 2*1.2; phi_B = phi_C at mu = 2*(2.5 - 1.2).
    np.testing.assert_allclose(by_mu.index, [2.4, 2.6], atol=SCAN_ATOL)
    assert by_mu.tolist() == [("A", "B"), ("B", "C")]


def test_scan_refiner_splits_dominated_crossing_T_scan():
    """Same recursion along the T axis: entropy opens a narrow B window in T."""
    # phi_A = 0, phi_B = 0.49 - 0.001*T, phi_C = 1 - 0.002*T at mu=0:
    # B is stable only for T in (490, 510), inside the (350, 550) grid gap.
    phases = {
        "A": LinePhase(name="A", fixed_concentration=0.0, line_energy=0.0),
        "B": LinePhase(name="B", fixed_concentration=0.5, line_energy=0.49, line_entropy=0.001),
        "C": LinePhase(name="C", fixed_concentration=1.0, line_energy=1.0, line_entropy=0.002),
    }
    df = _coarse_df(phases, np.linspace(150.0, 950.0, 5), [0.0])
    assert set(df["phase"]) == {"A", "C"}, "grid must not sample B stable"
    out = ScanRefiner("T").run(df, phases)
    by_T = out.groupby("T")["phase"].agg(lambda s: tuple(sorted(s)))
    assert len(by_T) == 2
    np.testing.assert_allclose(by_T.index, [490.0, 510.0], atol=SCAN_ATOL)
    assert by_T.tolist() == [("A", "B"), ("B", "C")]


# -- _dominated ---------------------------------------------------------------
#
# `_dominated(pt, phases)` is the predicate every refiner's run() uses to drop
# a refined transition whose phases are not globally stable at (pt.T, pt.mu):
# True iff some phase outside pt.phase_names() has a strictly lower phi there.
# The cases below cover each branch of that contract orthogonally: empty rival
# set, lower / equal / higher rival, the strictness of "<", the size-3 own set,
# and the own-set filter that hides supposedly-dominating coexisting phases.


def _dom_phases(*specs):
    """Build a name -> LinePhase mapping. Each spec is (name, c, E)."""
    return {name: LinePhase(name=name, fixed_concentration=c, line_energy=E)
            for name, c, E in specs}


def test_dominated_no_rivals_returns_false():
    """Two phases coexist, the mapping carries only those two: nothing else
    can dominate. The generator is empty and any() returns False."""
    phases = _dom_phases(("A", 0.0, 0.0), ("B", 1.0, 0.0))
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is False


def test_dominated_rival_with_lower_phi_returns_true():
    """phi_A = 0, phi_B = -mu = 0, phi_C = E_C - 0.5*mu = -0.1 at (T, mu=0).
    C is outside pt.phase_names() and strictly lower, so it dominates."""
    phases = _dom_phases(("A", 0.0, 0.0), ("B", 1.0, 0.0), ("C", 0.5, -0.1))
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is True


def test_dominated_rival_with_higher_phi_returns_false():
    """phi_C = +0.5 > phi_A = phi_B = 0 at mu = 0: no dominator."""
    phases = _dom_phases(("A", 0.0, 0.0), ("B", 1.0, 0.0), ("C", 0.5, 0.5))
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is False


def test_dominated_rival_equal_phi_returns_false():
    """phi_C == own_phi exactly: the comparison is strict "<", so a
    degenerate rival is not treated as dominating. At a true triple point
    we want this so the refined point survives instead of getting dropped."""
    phases = _dom_phases(("A", 0.0, 0.0), ("B", 1.0, 0.0), ("C", 0.5, 0.0))
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is False


def test_dominated_skips_phases_in_own_set():
    """Triple-point candidate. D would dominate but it is *in* ``own``, so
    the ``if p.name not in own`` filter excludes it and no rival remains."""
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, 0.0), ("D", 0.5, -1.0),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B", "D"))
    assert _dominated(pt, phases) is False


def test_dominated_triple_with_outside_dominator_returns_true():
    """Three phases coexist; a fourth phase outside ``own`` dominates."""
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, 0.0), ("C", 0.5, 0.0), ("X", 0.25, -0.5),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B", "C"))
    assert _dominated(pt, phases) is True


def test_dominated_picks_any_lower_rival_among_many():
    """Two outside rivals: one higher, one lower. The lower one alone is
    enough to make the candidate dominated."""
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, 0.0),
        ("hi", 0.25, +0.5), ("lo", 0.75, -0.2),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is True
