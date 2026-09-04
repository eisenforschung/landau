"""Unit tests for refiners in landau.refine."""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
import shapely
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from landau.features import Locus
from landau.calculate import calc_phase_diagram
from landau.phases import (
    IdealSolution,
    LinePhase,
    Phase,
    TemperatureDependentLinePhase,
    kB,
)
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
    _TRIPLE_COEXIST_TOL,
    _point_on_line,
    _simplex_brackets,
    _simplex_straddles,
    _dominated,
    _InterCandidate,
    _delaunay_simplices,
    _simplex_containment,
    _trace_geom,
    _state_row,
    _Simplex,
    _StepResult,
)


def _mk_simplex(*, T, mu, phase=None, c=None):
    """Build a numpy-backed _Simplex for the direct helper tests."""
    T = np.asarray(T, float)
    mu = np.asarray(mu, float)
    if phase is None:
        phase = np.array(["x"] * len(T))
    if c is None:
        c = np.zeros(len(T))
    return _Simplex(T=T, mu=mu, phase=np.asarray(phase), c=np.asarray(c, float))


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


def test_cc_refiner_trace_aborts_in_dominated_region():
    """The trace stops when the pair goes metastable past a triple point,
    instead of walking the whole T range and leaving run() to drop the
    dominated tail: every point solve() emits is globally stable.

    Regression — without the early abort solve() walks the A-B line across the
    triple point into C's region and emits dominated points (only filtered
    later by run())."""
    phases = _three_phase_system()  # triple point at (T=300, mu=0.2)
    Ts = np.linspace(220.0, 480.0, 12)
    mus = np.linspace(-0.05, 0.55, 15)
    df = _coarse_df(phases, Ts, mus)
    refiner = ClausiusClapeyronRefiner()
    cands = list(refiner.propose(df))
    # Across every coexistence pair, solve() emits only globally stable points:
    # the trace aborts on domination and a seed projected into a metastable
    # sliver is not emitted either. Without the abort the metastable tails are
    # emitted and only dropped later by run().
    for c in cands:
        for pt in refiner.solve(c, phases):
            assert not _dominated(pt, phases)

    # A-B is stable only for T > 300 here (phi_C - phi_A = 0.001*T - 0.3): the
    # stable side up toward T_max is traced, and the metastable tail below the
    # triple point is not (without the abort it would reach T_min = 220).
    ab = [pt for c in cands if {c.phase1, c.phase2} == {"A", "B"}
          for pt in refiner.solve(c, phases)]
    assert ab, "grid should contain an A-B two-phase simplex"
    assert max(pt.T for pt in ab) > 450.0
    assert min(pt.T for pt in ab) > 285.0


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


def _solve_steep_caps(dc_max, dc_min, cslope):
    A = _SteepMuFlatCPhase(name="A", a=1.0, k=0.0, c0=0.5, cslope=cslope)
    B = _SteepMuFlatCPhase(name="B", a=0.0, k=0.025, c0=0.2, cslope=0.0)
    pts = ClausiusClapeyronRefiner(dc_max=dc_max, dc_min=dc_min).solve(
        _steep_candidate(), {"A": A, "B": B})
    return A, sorted(pts, key=lambda p: p.T)


def test_cc_refiner_dc_max_takes_precedence_over_dc_min():
    """The max bound wins: a dc_min set above dc_max still cannot drive a step
    past the dc_max cap, so the per-step drift is pinned at dc_max."""
    # dc_min = 0.05 alone wants dT = 0.05 / 4e-3 = 12.5 K (clamped to dT_max),
    # but dc_max = 0.01 caps the drift at dT = 0.01 / 4e-3 = 2.5 K.
    A, pts = _solve_steep_caps(dc_max=0.01, dc_min=0.05, cslope=4e-3)
    assert len(pts) > 4
    cs = np.array([float(A.concentration(p.T, p.mu)) for p in pts])
    assert np.abs(np.diff(cs)).max() == pytest.approx(0.01, abs=1e-9)


def test_cc_refiner_dT_max_bounds_dc_min_floor():
    """The dc_min floor never oversteps dT_max even when the drift target
    would call for a larger step."""
    # dc_min = 0.05 wants dT = 50 K with cslope = 1e-3; dc_max = 1.0 is slack,
    # so only dT_max = 5 K bounds the step.
    _, pts = _solve_steep_caps(dc_max=1.0, dc_min=0.05, cslope=1e-3)
    assert len(pts) > 4
    Ts = np.sort([p.T for p in pts])
    assert np.diff(Ts).max() == pytest.approx(5.0, abs=1e-9)


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


def test_simplex_brackets_returns_mu_T_extents_and_T_centroid():
    """The (mu_lo, mu_hi, T_lo, T_hi, T_seed) tuple is the seed simplex's
    mu/T bounding box plus the centroid T that ``_CCBase._trace`` walks
    from — verified on a hand-built 3-row simplex."""
    simplex = _mk_simplex(T=[100.0, 200.0, 300.0], mu=[0.5, 0.2, 0.8], phase=["A", "B", "A"])
    mu_lo, mu_hi, T_lo, T_hi, T_seed = _simplex_brackets(simplex)
    assert (mu_lo, mu_hi) == (0.2, 0.8)
    assert (T_lo, T_hi) == (100.0, 300.0)
    assert T_seed == 200.0  # arithmetic mean of the T column


def test_simplex_brackets_row_order_agnostic():
    """The returned bracket is symmetric in row order; a shuffled simplex
    yields the same output."""
    T = [300.0, 100.0, 200.0]
    mu = [0.8, 0.5, 0.2]
    forward = _simplex_brackets(_mk_simplex(T=T, mu=mu))
    reversed_ = _simplex_brackets(_mk_simplex(T=T[::-1], mu=mu[::-1]))
    assert forward == reversed_


def test_simplex_brackets_returns_python_floats():
    """Every field is a plain ``float`` — the downstream refiners store
    these as dataclass fields and rely on scalar arithmetic, not numpy
    scalar types."""
    simplex = _mk_simplex(T=[100, 200], mu=[0.5, 0.7])
    for value in _simplex_brackets(simplex):
        assert type(value) is float


def test_trace_geom_single_point_returns_shapely_point():
    """A trace that emitted only its seed becomes a ``Point`` so
    ``_point_on_line`` / ``_simplex_straddles`` can test distance to it
    without inventing a spurious segment."""
    geom = _trace_geom([(100.0, 0.5)])
    assert isinstance(geom, shapely.Point)
    assert (geom.x, geom.y) == (100.0, 0.5)


def test_trace_geom_two_points_returns_linestring_with_input_coords():
    """A two-point trace becomes a ``LineString`` whose coordinates are
    exactly the input points, in input order."""
    pts = [(100.0, 0.5), (200.0, 0.8)]
    geom = _trace_geom(pts)
    assert isinstance(geom, shapely.LineString)
    assert list(geom.coords) == pts


def test_trace_geom_three_points_preserves_all_vertices():
    """Multi-point traces keep every vertex; ``LineString`` is not
    simplified into a segment between the endpoints."""
    pts = [(100.0, 0.5), (200.0, 0.8), (300.0, 1.1)]
    geom = _trace_geom(pts)
    assert isinstance(geom, shapely.LineString)
    assert list(geom.coords) == pts


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


def test_delaunay_simplices_yields_numpy_backed_vertices():
    """``_delaunay_simplices`` yields ``_Simplex`` views whose length-3 numpy
    arrays reproduce the source rows, with the right phase count per simplex."""
    phases = _three_phase_system()
    Ts = np.linspace(220.0, 480.0, 5)
    mus = np.linspace(-0.05, 0.55, 6)
    df = _coarse_df(phases, Ts, mus)
    rows = set(zip(df["T"], df["mu"], df["phase"], df["c"]))

    seen_counts = set()
    for simplex, n in _delaunay_simplices(df):
        assert isinstance(simplex, _Simplex)
        assert len(simplex.T) == len(simplex.mu) == len(simplex.phase) == len(simplex.c) == 3
        # every vertex is an actual (T, mu, phase, c) row of the input frame
        for t, m, p, c in zip(simplex.T, simplex.mu, simplex.phase, simplex.c):
            assert (t, m, p, c) in rows
        # phase count matches the distinct phase names on the vertices
        assert n == len(set(simplex.phase))
        assert list(simplex.unique_phases()) == list(dict.fromkeys(simplex.phase))
        seen_counts.add(n)
    # the grid straddles a triple point, so 1-, 2- and 3-phase simplices appear
    assert seen_counts == {1, 2, 3}


def test_delaunay_simplices_memoised_per_frame():
    """Repeated calls on the same frame reuse one tessellation (so the default
    refiners share it), while a different frame recomputes."""
    phases = _three_phase_system()
    Ts = np.linspace(220.0, 480.0, 5)
    mus = np.linspace(-0.05, 0.55, 6)
    df = _coarse_df(phases, Ts, mus)

    first = _delaunay_simplices(df)
    assert _delaunay_simplices(df) is first  # same frame -> cached list reused

    df2 = _coarse_df(phases, Ts, mus)  # equal content, different object
    other = _delaunay_simplices(df2)
    assert other is not first  # identity-keyed: not aliased to the old frame
    assert len(other) == len(first)


def test_delaunay_simplices_cache_flips_on_fresh_frame():
    """Once a second frame has taken the one-slot cache, revisiting the first
    frame must recompute rather than returning the stale cached list."""
    phases = _three_phase_system()
    Ts = np.linspace(220.0, 480.0, 5)
    mus = np.linspace(-0.05, 0.55, 6)
    df = _coarse_df(phases, Ts, mus)
    df2 = _coarse_df(phases, Ts, mus)  # equal content, different object

    first = _delaunay_simplices(df)
    _delaunay_simplices(df2)  # flips the cache onto df2

    third = _delaunay_simplices(df)
    assert third is not first  # cache now keyed on df2, so df misses and recomputes


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
    inside = _mk_simplex(T=[0.0, 2.0, 0.0], mu=[0.0, 0.0, 2.0])
    # Centroid is strictly inside, edge midpoint sits on the boundary.
    assert _simplex_containment((0.5, 0.5), inside) > 0
    assert np.isclose(_simplex_containment((1.0, 0.0), inside), 0.0)

    # A point past the hypotenuse is outside; the simplex it is least far
    # outside of wins ``max``.
    near = _mk_simplex(T=[0.0, 2.0, 0.0], mu=[0.0, 0.0, 2.0])
    far = _mk_simplex(T=[0.0, -2.0, 0.0], mu=[0.0, 0.0, -2.0])
    point = (1.1, 1.1)
    assert _simplex_containment(point, near) < 0
    assert _simplex_containment(point, far) < _simplex_containment(point, near)
    assert max((near, far), key=lambda s: _simplex_containment(point, s)) is near

    # Degenerate (collinear) simplex never wins ownership.
    line = _mk_simplex(T=[0.0, 1.0, 2.0], mu=[0.0, 1.0, 2.0])
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
# a refined transition that isn't a valid piece of the global phase boundary
# at (pt.T, pt.mu). The cases below cover each branch of that contract
# orthogonally: empty rival set, lower / equal / higher rival, the strictness
# of "<", an absent (phi = +inf) own phase, a triple whose own phases don't
# share one potential, an outside dominator in a triple, and multiple rivals.
# The spread branch is pinned on both sides of _TRIPLE_COEXIST_TOL, together
# with the two-phase exemption and the headroom a real refined triple leaves.


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


def test_dominated_outside_rival_between_own_potentials_returns_true():
    """Two-phase boundary with unequal own potentials (phi_A=0, phi_B=0.5;
    allowed since the coexistence-tolerance check only applies to triples).
    Rival X sits at 0.3 - strictly between the two own potentials, so it beats
    B but not A. The reference is ``max(own_phi)``, i.e. "does X beat the
    least-stable own phase": X beats B, so the own set is not the top-2 most
    stable and the point is dominated whichever way the ``own`` set happens to
    iterate."""
    phases = _dom_phases(("A", 0.0, 0.0), ("B", 1.0, 0.5), ("X", 0.5, 0.3))
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is True


def test_dominated_own_phase_below_others_returns_true():
    """A claimed triple point whose own phases don't share a potential is not a
    real coexistence: D lies far below A and B at ``(T, mu)``, so only D is
    stable there and the A+B+D point is spurious. The own-phase spread (1.0 eV)
    exceeds ``_TRIPLE_COEXIST_TOL``, so it is dropped regardless of set
    order."""
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, 0.0), ("D", 0.5, -1.0),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B", "D"))
    assert _dominated(pt, phases) is True


def test_dominated_absent_own_phase_returns_true():
    """A phase reported absent (``phi = +inf``) must not survive as a
    triple-point vertex. C is absent, so A+B+C is really just A+B and the
    triple point is dropped."""
    class _Absent:
        name = "C"

        def semigrand_potential(self, T, mu):
            return float("inf")

        def concentration(self, T, mu):
            return 0.0

    phases = {**_dom_phases(("A", 0.0, 0.0), ("B", 1.0, 0.0)), "C": _Absent()}
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B", "C"))
    assert _dominated(pt, phases) is True


def test_dominated_triple_with_outside_dominator_returns_true():
    """Three phases coexist; a fourth phase outside ``own`` dominates."""
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, 0.0), ("C", 0.5, 0.0), ("X", 0.25, -0.5),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B", "C"))
    assert _dominated(pt, phases) is True


def test_dominated_triple_spread_just_below_tolerance_survives():
    """`_TRIPLE_COEXIST_TOL` is a real threshold, not a formality: a triple whose
    own potentials span half of it still reads as one coexistence. At mu = 0 a
    LinePhase's potential is its line energy, so the spread here is exactly
    0.5 * _TRIPLE_COEXIST_TOL. No phase sits outside `own`, so the spread branch
    is the only one that can fire."""
    eps = 0.5 * _TRIPLE_COEXIST_TOL
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, eps), ("C", 0.5, 0.5 * eps),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B", "C"))
    assert _dominated(pt, phases) is False


def test_dominated_triple_spread_just_above_tolerance_is_dropped():
    """The other side of the same threshold: double the tolerance and the
    triple is dropped. Together with the test above this pins the cut to within
    a factor of four, where the 1.0 eV spread of
    test_dominated_own_phase_below_others_returns_true would pass for any
    tolerance below 1 eV."""
    eps = 2.0 * _TRIPLE_COEXIST_TOL
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, eps), ("C", 0.5, 0.5 * eps),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B", "C"))
    assert _dominated(pt, phases) is True


def test_dominated_two_phase_spread_far_above_tolerance_survives():
    """The tolerance is triple-only. Two phases whose potentials differ by 5000x
    _TRIPLE_COEXIST_TOL are a first-order boundary, where the stable phase jumps
    rather than crosses - a real boundary that must survive. Nothing sits outside
    `own`, so only the (skipped) spread branch could drop it."""
    phases = _dom_phases(("A", 0.0, 0.0), ("B", 1.0, 5000.0 * _TRIPLE_COEXIST_TOL))
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is False


def test_refined_triple_point_leaves_orders_of_magnitude_of_headroom(
    eutectic_phases, eutectic_diagram
):
    """The tolerance is a filter for spurious triples, not a bound the solver
    strains against: a converged eutectic invariant sits far inside it. Asserting
    a hard fraction of _TRIPLE_COEXIST_TOL rather than the tolerance itself keeps
    this a statement about refinement quality - the emitted rows are already past
    the `_dominated` gate, so comparing to the tolerance would be circular."""
    by_name = {p.name: p for p in eutectic_phases}
    triple = eutectic_diagram[eutectic_diagram["locus"] == Locus.TRIPLE]
    invariants = list(triple.groupby(["T", "mu"]))
    assert len(invariants) == 1, "eutectic system must yield exactly one invariant"

    for (T, mu), rows in invariants:
        names = sorted(set(rows["phase"]))
        assert len(names) == 3
        phi = [by_name[n].semigrand_potential(T, mu) for n in names]
        assert max(phi) - min(phi) < _TRIPLE_COEXIST_TOL / 100


def test_dominated_picks_any_lower_rival_among_many():
    """Two outside rivals: one higher, one lower. The lower one alone is
    enough to make the candidate dominated."""
    phases = _dom_phases(
        ("A", 0.0, 0.0), ("B", 1.0, 0.0),
        ("hi", 0.25, +0.5), ("lo", 0.75, -0.2),
    )
    pt = RefinedPoint(T=300.0, mu=0.0, phases=("A", "B"))
    assert _dominated(pt, phases) is True


# -- _state_row -----------------------------------------------------------------
#
# `_state_row(phase, T, mu)` (refine.py:183) projects one phase at one (T, mu)
# to the dict `RefinedPoint.to_rows` consumes.


def test_state_row_projects_exact_values():
    """The dict carries T, mu, phi, c, phase with the exact values the phase
    reports at that point."""
    phase = LinePhase(name="A", fixed_concentration=0.3, line_energy=0.1, line_entropy=1e-4)
    T, mu = 500.0, 0.02
    row = _state_row(phase, T, mu)
    assert row.keys() == {"T", "mu", "phi", "c", "phase"}
    assert row["T"] == T
    assert row["mu"] == mu
    assert row["phi"] == phase.semigrand_potential(T, mu)
    assert row["c"] == phase.concentration(T, mu)
    assert row["phase"] == phase.name


def test_state_row_passes_T_and_mu_through_unchanged():
    """T and mu are not rounded or cast; an int and a numpy scalar survive
    as the exact objects passed in."""
    phase = LinePhase(name="A", fixed_concentration=0.5, line_energy=0.0)
    T = 400
    mu = np.float64(0.0123456789)
    row = _state_row(phase, T, mu)
    assert row["T"] is T
    assert row["mu"] is mu


def test_state_row_two_phases_differ_only_in_projected_fields():
    """Two distinct phases at the same (T, mu) produce two dicts that agree
    on T and mu and differ only in phi, c, phase."""
    T, mu = 600.0, -0.01
    a = LinePhase(name="A", fixed_concentration=0.2, line_energy=0.05)
    b = LinePhase(name="B", fixed_concentration=0.8, line_energy=-0.05)
    row_a = _state_row(a, T, mu)
    row_b = _state_row(b, T, mu)
    assert row_a["T"] == row_b["T"] == T
    assert row_a["mu"] == row_b["mu"] == mu
    assert row_a["phi"] != row_b["phi"]
    assert row_a["c"] != row_b["c"]
    assert row_a["phase"] != row_b["phase"]


# -- _Simplex.centroids -----------------------------------------------------------
#
# `_Simplex.centroids()` maps each distinct phase to its vertices' (T, mu)
# centroid; a two-phase simplex yields the two seed-projection endpoints.


def test_simplex_centroids_arithmetic_mean_per_phase():
    """A 3-vertex two-phase simplex (A once, B twice) gives A's own vertex
    unchanged and B's the arithmetic mean of its two vertices."""
    simplex = _mk_simplex(
        phase=["A", "B", "B"], T=[300.0, 300.0, 320.0], mu=[0.0, 0.01, 0.02])
    cents = simplex.centroids()
    assert cents["A"] == pytest.approx((300.0, 0.0))
    assert cents["B"] == pytest.approx((310.0, 0.015))


def test_simplex_centroids_values_are_plain_float_tuples():
    """Values are (T, mu) tuples of Python floats, not numpy arrays, so
    downstream np.array(xy) construction works as expected."""
    simplex = _mk_simplex(
        phase=["A", "A", "B"], T=[100.0, 200.0, 400.0], mu=[0.0, 0.0, 0.0])
    cents = simplex.centroids()
    assert set(cents) == {"A", "B"}
    for xy in cents.values():
        assert isinstance(xy, tuple) and len(xy) == 2
        assert all(isinstance(v, float) for v in xy)


def test_simplex_centroids_ordering_follows_unique_phases():
    """Entries are in _Simplex.unique_phases order (vertex appearance), so the
    first-appearing phase comes first."""
    simplex = _mk_simplex(
        phase=["zeta", "alpha", "zeta"], T=[100.0, 500.0, 300.0], mu=[0.0, 0.0, 0.0])
    assert list(simplex.unique_phases()) == ["zeta", "alpha"]
    cents = simplex.centroids()
    assert list(cents) == ["zeta", "alpha"]
    assert cents["zeta"] == pytest.approx((200.0, 0.0))  # mean of two vertices
    assert cents["alpha"] == pytest.approx((500.0, 0.0))  # single vertex


# -- _CCBase._predict_mu / default _dT_adapt -----------------------------------
#
# Shared predictor and step scaler for both ClausiusClapeyronRefiner and
# MiscibilityGapRefiner (refine.py:819, refine.py:837). Only the override in
# MiscibilityGapRefiner is exercised by the miscibility-gap pipeline tests;
# these pin the base defaults directly via a concrete subclass instance.


def test_predict_mu_linear_extrapolation():
    """Default predictor is plain linear extrapolation: mu* + dmu/dT * dT."""
    refiner = ClausiusClapeyronRefiner()
    assert refiner._predict_mu(0.5, 0.02, 3.0) == pytest.approx(0.56)


def test_dT_adapt_default_half_width_over_slope():
    """Default step size is half_width / |dmu_dT|; the sign of dmu_dT does
    not matter since only its magnitude is used."""
    refiner = ClausiusClapeyronRefiner()
    step = _StepResult(mu_star=0.0, extra=None)
    assert refiner._dT_adapt(step, 0.02, half_width=0.1) == pytest.approx(5.0)
    assert refiner._dT_adapt(step, -0.02, half_width=0.1) == pytest.approx(5.0)


def test_dT_adapt_slope_floor_at_zero():
    """A flat coexistence line (dmu_dT == 0) does not divide by zero; the
    slope is floored at 1e-9 so the step saturates at half_width / 1e-9."""
    refiner = ClausiusClapeyronRefiner()
    step = _StepResult(mu_star=0.0, extra=None)
    assert refiner._dT_adapt(step, 0.0, half_width=0.1) == pytest.approx(1e8)


# -- _CCBase._emitted_concentrations -------------------------------------------
#
# The subclass-agnostic hook the dc_max/dc_min density caps in _trace read
# their concentration drift from (refine.py:870). Two branches: a
# RefinedMiscibilityGap plots its two pre-computed branch concentrations;
# a RefinedPoint plots one branch per coexisting phase, queried from `phases`.


def test_emitted_concentrations_gap_returns_stored_pair():
    """RefinedMiscibilityGap: the stored c_left/c_right pass through unchanged,
    not re-queried from the phase."""
    refiner = ClausiusClapeyronRefiner()
    gap = RefinedMiscibilityGap(T=400.0, mu=0.0, phase="p", c_left=0.1, c_right=0.9)
    assert refiner._emitted_concentrations(gap, {}) == (0.1, 0.9)


def test_emitted_concentrations_point_queries_each_coexisting_phase():
    """RefinedPoint: one concentration per phase in pt.phases order, matching
    a direct concentration(T, mu) call on each phase."""
    refiner = ClausiusClapeyronRefiner()
    phases = {
        "A": LinePhase(name="A", fixed_concentration=0.2, line_energy=0.0),
        "B": LinePhase(name="B", fixed_concentration=0.7, line_energy=1.0),
    }
    pt = RefinedPoint(T=500.0, mu=0.05, phases=("A", "B"))
    got = refiner._emitted_concentrations(pt, phases)
    expected = tuple(phases[n].concentration(pt.T, pt.mu) for n in pt.phases)
    assert got == pytest.approx(expected)
    assert got == pytest.approx((0.2, 0.7))


def test_emitted_concentrations_returns_plain_floats():
    """Both branches collapse to plain float, not numpy scalars, since the
    caller does arithmetic against dc_max."""
    refiner = ClausiusClapeyronRefiner()
    gap = RefinedMiscibilityGap(T=400.0, mu=0.0, phase="p", c_left=0.1, c_right=0.9)
    for c in refiner._emitted_concentrations(gap, {}):
        assert type(c) is float

    phases = {"A": LinePhase(name="A", fixed_concentration=0.2, line_energy=0.0)}
    pt = RefinedPoint(T=500.0, mu=0.05, phases=("A",))
    for c in refiner._emitted_concentrations(pt, phases):
        assert type(c) is float


# -- ClausiusClapeyronRefiner._tag_features (congruent points) -----------------
#
# A congruent transformation is one where both coexisting phases share a
# composition, so the line's concentration gap bottoms out there. The tag goes
# on the smallest gap among the points closing within `congruent_tol` of one
# composition, so one closure yields one tag and two closures a tolerance apart
# stay separate. These fixtures sample composition at `dc_max`, the drift the
# refiner steps by.

_DC = 0.01  # ClausiusClapeyronRefiner's default dc_max, hence the sample spacing
_GRID = np.round(np.arange(0.0, 1.0 + _DC / 2, _DC), 10)


def _gap_phases(gaps, shared):
    """Phases whose gap at mu = i is `gaps[i]`, centred on `shared[i]`.

    Both are keyed off mu so a RefinedPoint at mu=i reproduces the wanted
    numbers through the refiner's own `_emitted_concentrations`.
    """

    @dataclass(frozen=True)
    class _GapPhase:
        name: str
        sign: float

        def concentration(self, T, mu):
            i = int(round(mu))
            return shared[i] + self.sign * gaps[i] / 2

        def semigrand_potential(self, T, mu):
            return 0.0

    return {"lo": _GapPhase("lo", -1.0), "hi": _GapPhase("hi", +1.0)}


def _gap_points(n):
    return [RefinedPoint(T=300.0 + i, mu=float(i), phases=("lo", "hi")) for i in range(n)]


def _closes_at(gaps, shared=_GRID, **kwargs):
    """Compositions of the points tagged congruent along this line."""
    gaps, shared = list(gaps), list(shared)
    refiner = ClausiusClapeyronRefiner(**kwargs)
    out = refiner._tag_features(_gap_points(len(gaps)), _gap_phases(gaps, shared))
    return [shared[i] for i, pt in enumerate(out) if pt.congruent]


def test_tag_congruent_lens_closes_at_both_terminals():
    """An isomorphous lens closes at both pure components, however wide it is
    in between. Comparing the gap against the tolerance alone instead of
    against its neighbours tags the narrow lens once and the wide one twice."""
    for width in (0.025, 0.20):
        gaps = width * np.sin(np.pi * _GRID)
        assert _closes_at(gaps) == pytest.approx([0.0, 1.0]), f"width {width}"


def test_tag_congruent_ignores_a_line_that_never_closes():
    """A narrow two-phase field running between two triple points has no
    minimum near zero, however narrow it is."""
    assert _closes_at([0.04] * len(_GRID)) == []


def test_tag_congruent_interior_closure():
    """An intermediate phase melting congruently: the gap dips to zero where
    the other phase reaches its composition."""
    gaps = 0.10 * np.abs(_GRID - 0.4) + 0.001
    assert _closes_at(gaps) == pytest.approx([0.4])


def test_tag_congruent_one_sided_closure():
    """A line closing at one end only is tagged there and nowhere else."""
    assert _closes_at(np.linspace(0.20, 0.001, len(_GRID))) == pytest.approx([1.0])


def test_tag_congruent_separates_closures_far_apart_in_composition():
    """One boundary_id covers a whole phase pair, which a triple point can
    leave as two disjoint branches closing at c=0 and c=1 -- the case that
    T-ordering interleaves into one line and half-tags. Compositions a whole
    tolerance apart never share a window, so the two closures stay apart."""
    shared = [0.0, 0.005, 0.995, 1.0]
    assert _closes_at([0.001, 0.02, 0.02, 0.001], shared=shared) == pytest.approx([0.0, 1.0])


def test_tag_congruent_collapses_an_equal_gap_run_to_one_point():
    """Two pure components melting at the same temperature leave a run of
    points at exactly zero gap; it is one closure, so it gets one tag."""
    shared = [0.40, 0.41, 0.42, 0.43]
    assert _closes_at([0.0] * 4, shared=shared) == pytest.approx([0.40])


def test_tag_congruent_tolerance_follows_the_trace_step():
    """The trace lands within about a step of a closure, so the default
    tolerance follows dc_max rather than a constant."""
    assert ClausiusClapeyronRefiner().congruent_tol == pytest.approx(3 * 0.01)
    assert ClausiusClapeyronRefiner(dc_max=0.002).congruent_tol == pytest.approx(3 * 0.002)
    assert ClausiusClapeyronRefiner(congruent_tol=0.4).congruent_tol == 0.4


def test_tag_congruent_tolerance_is_configurable():
    gaps = 0.10 * np.abs(_GRID - 0.4) + 0.06
    assert _closes_at(gaps) == []
    assert _closes_at(gaps, congruent_tol=0.1) == pytest.approx([0.4])


def test_tag_congruent_leaves_other_emitted_types_alone():
    """MiscibilityGapRefiner emits a type with no congruent flag; the shared
    hook must not touch it."""
    refiner = ClausiusClapeyronRefiner()
    points = [
        RefinedMiscibilityGap(T=1.0, mu=0.0, phase="p", c_left=0.1, c_right=0.9),
        RefinedPoint(T=300.0, mu=0.0, phases=("lo", "hi")),
        RefinedPoint(T=301.0, mu=1.0, phases=("lo", "hi")),
    ]
    out = refiner._tag_features(points, _gap_phases([0.001, 0.02], [0.0, 0.01]))
    assert isinstance(out[0], RefinedMiscibilityGap)
    assert [pt.congruent for pt in out[1:]] == [True, False]


@given(
    closures=st.lists(st.floats(min_value=0.05, max_value=0.95), min_size=1, max_size=3),
    depth=st.floats(min_value=1e-6, max_value=0.005),
    ceiling=st.floats(min_value=0.006, max_value=0.5),
)
@settings(deadline=None, max_examples=50)
def test_tag_congruent_recovers_planted_closures(closures, depth, ceiling):
    """Plant closures at known compositions, read them back.

    The gap rises away from each planted closure towards `ceiling`, which is
    drawn from below as well as above the tolerance: a line that stays under it
    everywhere is exactly the case a rule comparing the gap against the
    tolerance alone gets wrong, tagging one closure where there are two.
    """
    closures = sorted(np.round(np.array(closures) / _DC) * _DC)
    # Closures nearer than the window would be one closure, not two.
    tol = ClausiusClapeyronRefiner().congruent_tol
    assume(len(closures) == 1 or np.diff(closures).min() >= 2 * tol)
    dist = np.min(np.abs(_GRID[:, None] - np.array(closures)[None, :]), axis=1)
    gaps = depth + dist * ceiling / (dist + ceiling)  # -> depth at a closure, -> ceiling away
    assert _closes_at(gaps) == pytest.approx(closures, abs=_DC)


@pytest.fixture(scope="module")
def split_line_diagram():
    """Refined diagram whose solid/liquid pair is split into two branches.

    Solid and liquid are symmetric ideal solutions, so both pure components
    melt at the same temperature, and an intermediate line phase at c=0.5 cuts
    the solid/liquid coexistence in two. One boundary_id therefore holds two
    disjoint branches that close at c=0 and c=1 at the *same* T -- in T-order
    they interleave, which is what a sequential scan gets wrong.
    """
    solid = IdealSolution(
        "solid",
        LinePhase("sA", fixed_concentration=0, line_energy=-3.0, line_entropy=1.0 * kB),
        LinePhase("sB", fixed_concentration=1, line_energy=-3.0, line_entropy=1.0 * kB),
    )
    liquid = IdealSolution(
        "liquid",
        LinePhase("lA", fixed_concentration=0, line_energy=-2.6, line_entropy=5.0 * kB),
        LinePhase("lB", fixed_concentration=1, line_energy=-2.6, line_entropy=5.0 * kB),
    )
    inter = LinePhase("AB", fixed_concentration=0.5, line_energy=-3.2, line_entropy=1.2 * kB)
    return calc_phase_diagram([solid, liquid, inter], np.linspace(400.0, 1600.0, 20),
                              mu=30, refine=True)


def test_both_branches_of_a_split_line_are_tagged(split_line_diagram):
    """Both pure-component melting points come back, not just one.

    The two branches of the solid/liquid pair share a boundary_id and reach
    their terminals at the same temperature; scanning the line in T-order tags
    one of them at best.
    """
    congruent = split_line_diagram[split_line_diagram["locus"] == Locus.CONGRUENT]
    invariants = [grp for _key, grp in congruent.groupby(["mu", "T"])
                  if set(grp["phase"]) == {"solid", "liquid"}]
    assert len(invariants) == 2, "one per pure component"

    shared = sorted(grp["c"].mean() for grp in invariants)
    assert shared[0] == pytest.approx(0.0, abs=0.01)
    assert shared[1] == pytest.approx(1.0, abs=0.01)
    # Symmetric end members, so the two melting points coincide.
    Ts = [grp["T"].iloc[0] for grp in invariants]
    assert Ts[0] == pytest.approx(Ts[1], abs=1.0)


def test_congruent_melting_of_an_intermediate_phase_is_tagged(split_line_diagram):
    """The line phase at c=0.5 melts congruently: liquid reaches its
    composition, so the gap closes away from either terminal."""
    congruent = split_line_diagram[split_line_diagram["locus"] == Locus.CONGRUENT]
    invariants = [grp for _key, grp in congruent.groupby(["mu", "T"])
                  if set(grp["phase"]) == {"AB", "liquid"}]
    assert len(invariants) == 1
    assert invariants[0]["c"].mean() == pytest.approx(0.5, abs=0.02)


def test_terminal_melting_points_are_tagged_end_to_end(eutectic_diagram):
    """The hcp/fcc/liquid fixture's pure-component transitions come back tagged,
    each with the two coexisting phases at (nearly) the same composition."""
    congruent = eutectic_diagram[eutectic_diagram["locus"] == Locus.CONGRUENT]
    groups = list(congruent.groupby(["mu", "T"]))
    assert len(groups) == 3, "one per pure-component transition in this system"
    for _key, grp in groups:
        assert len(grp) == 2  # exactly the two coexisting phases
        assert grp["c"].max() - grp["c"].min() < 0.05  # they share a composition
        assert min(grp["c"].min(), 1 - grp["c"].max()) < 0.05  # at a pure component
