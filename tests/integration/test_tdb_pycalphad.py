"""Integration check of the TDB export against pycalphad.

Each system below is defined with landau phases, solved with
``calc_phase_diagram``, written with ``to_tdb`` and read back by pycalphad,
which then has to reproduce landau's diagram from the file alone: the
tie-lines along every refined phase boundary, the temperature and compositions
of every triple point, and the congruent transition temperatures (terminal
melting points, a compound melting or forming at its own composition).
``comparison_figure`` draws the two diagrams side by side; running this module
as a script writes them as ``tests/integration/_plots/2d_tdb_pycalphad_<system>.png``
for visual review.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.optimize as so

import landau.calculate as ldc
import landau.interpolate as ldi
import landau.phases as ldp
from landau.features import Locus
from landau.plot import plot_phase_diagram
from landau.tdb import to_tdb

try:
    from pycalphad import Database, binplot, equilibrium
    from pycalphad import variables as v

    HAS_PYCALPHAD = True
except ImportError:
    HAS_PYCALPHAD = False

pytestmark = [
    pytest.mark.pycalphad,
    pytest.mark.skipif(not HAS_PYCALPHAD, reason="pycalphad is not installed"),
]

COMPS = ["A", "B", "VA"]
PRESSURE = 101325

T_ATOL = 0.05
"""K; pycalphad's transition temperature vs landau's (measured 0.004, the bisection resolution)."""
TIE_ATOL = 1e-4
"""composition; pycalphad's tie-line endpoints vs landau's refined boundary points (measured 1e-5)."""
INVARIANT_ATOL = 1e-3
"""composition; the two-phase side of a triple point, read T_OFFSET away from it (measured 1.4e-4)."""
T_BISECT = 0.01
"""K; bisection resolution for pycalphad's transition temperatures."""
T_OFFSET = 0.25
"""K; distance from a transition at which the phase sets on either side are read."""
T_EXCLUDE = 3.0
"""K; boundary rows this close to an invariant are not compared as tie-lines (the endpoints of a
tie-line next to a closure move by 1e-4 per 0.1 K)."""
MIN_GAP = 0.02
"""composition; narrower two-phase regions sit at a closure, where pycalphad's equilibrium may
return no phases at all (seen at a gap of 0.01 next to the compound's congruent maximum)."""
X_TERMINAL = 1e-6
"""composition pycalphad is asked for instead of a pure component."""
T_NUDGES = (0.0, 1e-3, -1e-3, 2e-3)
"""K; pycalphad's equilibrium returns no phases at isolated exact (T, x) inputs while converging at
every neighbour, so a failed probe is repeated this far away -- well below T_BISECT and TIE_ATOL."""
SAMPLES_PER_BOUNDARY = 8


# --------------------------------------------------------------------------- #
# systems
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class System:
    name: str
    phases: tuple
    Ts: np.ndarray
    mu: object
    """``mu`` argument of :func:`landau.calculate.calc_phase_diagram`."""
    congruent: tuple
    """``(x, low, high)``: at composition ``x`` phase ``low`` turns into ``high`` on heating,
    both at that composition, so the transition is where their free energies cross."""

    @property
    def tdb_names(self):
        return [p.name.upper() for p in self.phases]

    def phase(self, name):
        return next(p for p in self.phases if p.name == name)


def eutectic_system() -> System:
    """hcp / fcc / liquid ideal solutions (Basics.ipynb): one eutectic, two terminal melting points."""
    kB = ldp.kB
    fcc = ldp.IdealSolution(
        "fcc",
        ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * kB),
        ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * kB),
    )
    hcp = ldp.IdealSolution(
        "hcp",
        ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * kB),
        ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * kB),
    )
    liquid = ldp.IdealSolution(
        "liquid",
        ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * kB),
        ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * kB),
    )
    congruent = ((0.0, "hcp", "liquid"), (1.0, "fcc", "liquid"))
    return System("eutectic", (hcp, fcc, liquid), np.linspace(200.0, 1000.0, 25), 50, congruent)


def toy_system() -> System:
    """Redlich-Kister liquid over PolyFit line phases, SGTE ideal solid and a stoichiometric
    compound at c=0.4 (Toy.ipynb): every export path in one diagram, with two eutectics, the
    compound melting congruently and forming from the solid solution at its own composition."""
    T = [1, 750, 1000]
    l0 = ldp.TemperatureDependentLinePhase("l0", 0, T, [2.00, 1.80, 1.00], interpolator=ldi.PolyFit(3))
    l1 = ldp.TemperatureDependentLinePhase("l1", 1, T, [3.00, 2.80, 2.00], interpolator=ldi.PolyFit(3))
    l2 = ldp.TemperatureDependentLinePhase("l2", 0.5, T, [2.45, 2.00, 1.42], interpolator=ldi.PolyFit(3))
    s0 = ldp.TemperatureDependentLinePhase("s0", 0, T, [1.9, 1.6, 1.2], interpolator=ldi.SGTE(2))
    s1 = ldp.TemperatureDependentLinePhase("s1", 1, T, [2.9, 2.6, 2.2], interpolator=ldi.SGTE(2))
    s3 = ldp.TemperatureDependentLinePhase(
        "s3", 0.4, T, np.array([2.4, 1.85, 1.45]) - 0.05, interpolator=ldi.SGTE(3)
    )
    liquid = ldp.FastInterpolatingPhase("liquid", [l0, l2, l1])
    solid = ldp.IdealSolution("solid", s0, s1)
    c = np.linspace(0, 1, 75)[1:-1]
    mu = 1 + ldp.kB * 4000 * np.log(c / (1 - c))
    congruent = ((0.0, "solid", "liquid"), (1.0, "solid", "liquid"), (0.4, "solid", "s3"), (0.4, "s3", "liquid"))
    return System("toy", (liquid, solid, s3), np.linspace(500.0, 1000.0, 40), mu, congruent)


SYSTEMS = {s.name: s for s in (eutectic_system(), toy_system())}


def phase_diagram(system: System):
    return ldc.calc_phase_diagram(list(system.phases), system.Ts, mu=system.mu, refine=True)


def database(system: System):
    """pycalphad's view of the system: the exported TDB read back."""
    T = system.Ts
    text = to_tdb(system.phases, temperature_range=(0.5 * T.min(), 2 * T.max()))
    return Database.from_string(text, fmt="tdb")


def free_energy(phase, T, c):
    """``f(T, c)`` of any of the phases above, in eV/atom."""
    if isinstance(phase, ldp.IdealSolution):
        return (1 - c) * phase.phase1.line_free_energy(T) + c * phase.phase2.line_free_energy(T) - T * ldp.S(c)
    if isinstance(phase, ldp.AbstractLinePhase):
        assert c == phase.line_concentration
        return phase.line_free_energy(T)
    return float(phase.free_energy(T, c))


# --------------------------------------------------------------------------- #
# pycalphad probes
# --------------------------------------------------------------------------- #
def _tie_line(db, names, T, x) -> dict[str, float]:
    """``{phase: composition}`` of the phases pycalphad finds stable at ``(T, x)``."""
    x = min(max(x, X_TERMINAL), 1 - X_TERMINAL)
    for nudge in T_NUDGES:
        eq = equilibrium(db, COMPS, names, {v.X("B"): x, v.T: T + nudge, v.P: PRESSURE, v.N: 1})
        phases = eq.Phase.values.squeeze()
        comps = eq.X.sel(component="B").values.squeeze()
        found = {str(p): float(c) for p, c in zip(phases, comps) if p != ""}
        if found:
            return found
    raise RuntimeError(f"pycalphad found no stable phase at T={T}, x={x}")


def _stable_set(db, names, T, x) -> frozenset:
    return frozenset(_tie_line(db, names, T, x))


def _transition_temperature(db, names, x, T_lo, T_hi, high_set) -> float:
    """Bisect for the temperature at which the stable set at ``x`` turns into ``high_set``."""
    assert _stable_set(db, names, T_hi, x) == high_set
    assert _stable_set(db, names, T_lo, x) != high_set
    while T_hi - T_lo > T_BISECT:
        T = 0.5 * (T_lo + T_hi)
        if _stable_set(db, names, T, x) == high_set:
            T_hi = T
        else:
            T_lo = T
    return 0.5 * (T_lo + T_hi)


# --------------------------------------------------------------------------- #
# landau's diagram, read back
# --------------------------------------------------------------------------- #
def _triple_points(df):
    """``(T, {phase: c})`` for every triple point landau located."""
    rows = df[df.locus == Locus.TRIPLE]
    return [
        (float(T), dict(zip(group.phase.str.upper(), group.c.astype(float))))
        for (T, _mu), group in rows.groupby(["T", "mu"])
    ]


def _tie_lines(df):
    """``(boundary_id, T, {phase: c})`` for every refined two-phase point on landau's boundaries."""
    rows = df[df.locus.isin([Locus.BOUNDARY, Locus.CONGRUENT])]
    return [
        (int(group.boundary_id.iloc[0]), float(T), dict(zip(group.phase.str.upper(), group.c.astype(float))))
        for (T, _mu), group in rows.groupby(["T", "mu"])
        if len(group) == 2 and abs(group.c.iloc[0] - group.c.iloc[1]) >= MIN_GAP
    ]


def _congruent_temperatures(system: System):
    """Where the free energies of each of the system's congruent pairs cross, from landau's phases."""
    return [
        so.brentq(
            lambda T: free_energy(system.phase(high), T, x) - free_energy(system.phase(low), T, x),
            system.Ts.min(),
            system.Ts.max(),
        )
        for x, low, high in system.congruent
    ]


def _sampled_tie_lines(system: System, df):
    """Up to :data:`SAMPLES_PER_BOUNDARY` tie-lines per boundary, away from the invariants."""
    invariant_Ts = [T for T, _ in _triple_points(df)] + _congruent_temperatures(system)
    by_boundary = {}
    for boundary_id, T, comps in _tie_lines(df):
        if all(abs(T - Ti) > T_EXCLUDE for Ti in invariant_Ts):
            by_boundary.setdefault(boundary_id, []).append((T, comps))
    sampled = []
    for points in by_boundary.values():
        points.sort()
        picks = np.unique(np.linspace(0, len(points) - 1, SAMPLES_PER_BOUNDARY).round().astype(int))
        sampled += [points[i] for i in picks]
    return sampled


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def comparison_figure(system: System, df, db, x_step=0.02, T_step=5.0):
    """landau's diagram on the left, pycalphad's ``binplot`` of the exported TDB on the right."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_phase_diagram(df, ax=left)
    left.set_title("landau")
    T = system.Ts
    conditions = {v.X("B"): (0, 1, x_step), v.T: (T.min(), T.max(), T_step), v.P: PRESSURE, v.N: 1}
    binplot(db, COMPS, system.tdb_names, conditions, plot_kwargs={"ax": right})
    right.set_title("pycalphad, from to_tdb")
    right.set_ylim(left.get_ylim())
    fig.suptitle(system.name)
    return fig


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=list(SYSTEMS), ids=list(SYSTEMS))
def case(request):
    system = SYSTEMS[request.param]
    return system, phase_diagram(system), database(system)


def test_triple_points(case):
    """At the composition of the phase between the other two, pycalphad's stable set flips
    at landau's eutectic temperature from the outer pair to that phase alone, and the outer
    pair's compositions just below the eutectic are landau's."""
    system, df, db = case
    triples = _triple_points(df)
    assert triples, "landau located no triple point"
    for T, comps in triples:
        middle = sorted(comps, key=comps.get)[1]
        x = comps[middle]
        below = _tie_line(db, system.tdb_names, T - T_OFFSET, x)
        above = _tie_line(db, system.tdb_names, T + T_OFFSET, x)
        assert {frozenset(below), frozenset(above)} == {frozenset({middle}), frozenset(comps) - {middle}}
        T_pc = _transition_temperature(db, system.tdb_names, x, T - T_OFFSET, T + T_OFFSET, frozenset(above))
        assert T_pc == pytest.approx(T, abs=T_ATOL)
        outer = below if len(below) == 2 else above
        for phase, c in outer.items():
            assert c == pytest.approx(comps[phase], abs=INVARIANT_ATOL), phase


def test_congruent_points(case):
    """Where two phases of equal composition swap stability -- a pure component melting, the
    compound melting or forming from the solid solution -- pycalphad's transition sits at
    the temperature landau's free energies cross."""
    system, df, db = case
    for (x, low, high), T_landau in zip(system.congruent, _congruent_temperatures(system)):
        T_pc = _transition_temperature(
            db, system.tdb_names, x, T_landau - T_OFFSET, T_landau + T_OFFSET, frozenset({high.upper()})
        )
        assert T_pc == pytest.approx(T_landau, abs=T_ATOL), (x, low, high)


def test_tie_lines(case):
    """Inside every two-phase region landau refined, pycalphad finds the same pair of
    phases at the same compositions."""
    system, df, db = case
    sampled = _sampled_tie_lines(system, df)
    assert len(sampled) >= 3 * len(system.phases) - 3
    for T, comps in sampled:
        x = float(np.mean(list(comps.values())))
        found = _tie_line(db, system.tdb_names, T, x)
        assert set(found) == set(comps), (T, x, found, comps)
        for phase, c in comps.items():
            assert found[phase] == pytest.approx(c, abs=TIE_ATOL), (T, phase)


def test_comparison_figure(case, tmp_path):
    system, df, db = case
    fig = comparison_figure(system, df, db)
    path = tmp_path / f"2d_tdb_pycalphad_{system.name}.png"
    fig.savefig(path)
    plt.close(fig)
    assert path.stat().st_size > 0



def main():
    parser = argparse.ArgumentParser(description="Render landau's and pycalphad's diagrams side by side.")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "_plots", help="output directory for PNGs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for system in SYSTEMS.values():
        fig = comparison_figure(system, phase_diagram(system), database(system))
        path = args.out / f"2d_tdb_pycalphad_{system.name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
