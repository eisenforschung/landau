"""Render reference phase diagrams used for visual review of landau changes.

Each function below builds a minimal set of phases drawn from the example
notebooks and saves a single phase-diagram PNG into the output directory
(``tests/integration/_plots`` by default, override with ``--out``).

The script is intentionally not a pytest test: the "correctness" of a
landau plot is hard to formalise so we instead post the rendered figures
on PRs labeled ``testplot`` (or mentioned with ``@testplot``) and review
them by eye.

Known issue (UX trade-off):

Both the ``testplot`` label workflow and the ``@testplot`` mention
workflow are split into three jobs so PR code never runs in the same
context as the Haiku OAuth token or a write-scoped ``GITHUB_TOKEN``:
``select`` / ``parse`` checks out ``main`` (trusted, has the Claude
secret), ``render`` checks out the PR head (untrusted code, no secrets,
``contents: read``), and ``publish`` runs no PR code at all.

The cost of that split is that the Haiku allow-list of plot keys is
frozen to ``main``. A PR that adds a new plot here can't be exercised
via ``@testplot`` or the diff-aware label until the allow-lists on
``main`` are updated to know the new key. The workaround is to land a
small parser-only PR first, then the plot PR.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import landau.calculate as ldc
import landau.interpolate as ldi
import landau.phases as ldp
import landau.plot as lpl

POLY_METHODS = ["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"]
TIELINE_VALUES = ["on", "off"]


def _save(fig, out_dir: Path, name: str, suffix: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _title_suffix(poly_method: str | None = None, tielines: bool = True) -> str:
    parts = []
    if poly_method:
        parts.append(poly_method)
    if not tielines:
        parts.append("no tielines")
    return f" [{', '.join(parts)}]" if parts else ""


def _file_suffix(poly_method: str | None = None, tielines: bool = True) -> str:
    parts = []
    if poly_method:
        parts.append(poly_method)
    if not tielines:
        parts.append("notielines")
    return "_" + "_".join(parts) if parts else ""


def _calc_1d_T_bcc_fcc_liquid():
    """bcc/fcc/liquid TemperatureDependentLinePhase scan shared by the 1D T testplots."""
    T_s = np.array([100.0, 400.0, 700.0, 1000.0])
    # G(T) = a - b*T - c*T^2 (c > 0: concave-down, strictly decreasing).
    # bcc: shallow slope, stable at low T.
    # fcc: intermediate slope with more curvature, stable at mid T.
    # liquid: steepest slope, stable at high T.
    bcc = ldp.TemperatureDependentLinePhase(
        "bcc", fixed_concentration=0, temperatures=T_s,
        free_energies=-3.00 - 2e-4 * T_s - 1e-7 * T_s**2,
    )
    fcc = ldp.TemperatureDependentLinePhase(
        "fcc", fixed_concentration=0, temperatures=T_s,
        free_energies=-2.97 - 2.857e-4 * T_s - 2e-7 * T_s**2,
    )
    liquid = ldp.TemperatureDependentLinePhase(
        "liquid", fixed_concentration=0, temperatures=T_s,
        free_energies=-2.75 - 3.5e-4 * T_s - 5e-7 * T_s**2,
    )
    Ts = np.linspace(100, 1000, 50)
    return ldc.calc_phase_diagram([bcc, fcc, liquid], Ts, mu=0.0, refine=True, keep_unstable=True)


def _calc_1d_mu_hcp_fcc_liquid():
    """hcp/fcc/liquid IdealSolution mu scan shared by the 1D mu testplots."""
    fcca = ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.02, line_entropy=1.0 * ldp.kB)
    fccb = ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.02, line_entropy=1.1 * ldp.kB)
    hcpa = ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
    hcpb = ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB)
    lqda = ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.724, line_entropy=1.5 * ldp.kB)
    lqdb = ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-2.050, line_entropy=1.2 * ldp.kB)
    fcc = ldp.IdealSolution("fcc", fcca, fccb)
    hcp = ldp.IdealSolution("hcp", hcpa, hcpb)
    lqd = ldp.IdealSolution("liquid", lqda, lqdb)
    return ldc.calc_phase_diagram([hcp, fcc, lqd], Ts=1000.0, mu=100, keep_unstable=True)


def _calc_1d_mu_intermetallic():
    """A/B solid solution + AB2 line phase mu scan for the 1D intermetallic testplot.

    The subscripted phase name (``AB$_2$``) exercises the bold-mathtext top-spine
    label path: AB2 is the stable intermediate over a mu window, with the solid
    solution stable on the A-rich and B-rich sides.
    """
    sa = ldp.LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * ldp.kB)
    sb = ldp.LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * ldp.kB)
    solid = ldp.IdealSolution("solid", sa, sb)
    inter = ldp.LinePhase("AB$_2$", fixed_concentration=2 / 3, line_energy=-2.8, line_entropy=1.3 * ldp.kB)
    return ldc.calc_phase_diagram([solid, inter], Ts=600.0, mu=200, keep_unstable=True)


def plot_1d_T_three_stable(out_dir: Path, **_) -> Path:
    """1D T diagram: three stable phases with concave-down free energies.

    fcc is the intermediate phase, stable roughly 350–760 K; it is unstable
    in two disjoint T ranges at low and high T, exposing the segment-splitting
    fix in plot_1d_T_phase_diagram.
    """
    df = _calc_1d_T_bcc_fcc_liquid()
    fig, ax = plt.subplots(figsize=(6, 4))
    lpl.plot_1d_T_phase_diagram(df, ax=ax)
    ax.set_title("1D T phase diagram (curved G: bcc / fcc / liquid)")
    return _save(fig, out_dir, "1d_T_three_stable_phase_diagram")


def plot_1d_mu_three_stable(out_dir: Path, **_) -> Path:
    """1D mu phase diagram: three stable solution phases (hcp / fcc / liquid).

    hcp/fcc parameters are taken from the two-phase Basics.ipynb scenario; a
    liquid phase is added so that fcc is the intermediate stable phase.  fcc
    endpoint energies are shifted −0.02 eV relative to the Basics.ipynb values
    to open a stable window (~0.09 eV wide in µ, c ≈ 0.37–0.62 at T=1000 K).
    fcc is unstable in two disjoint µ ranges (below and above its stable
    window), exposing the segment-splitting fix in plot_1d_mu_phase_diagram.
    """
    df = _calc_1d_mu_hcp_fcc_liquid()
    fig, ax = plt.subplots(figsize=(6, 4))
    lpl.plot_1d_mu_phase_diagram(df, ax=ax)
    ax.set_title(r"1D $\mu$ phase diagram (hcp / fcc / liquid, T=1000K)")
    return _save(fig, out_dir, "1d_mu_three_stable_phase_diagram")


def plot_1d_T_reference_phase(out_dir: Path, **_) -> Path:
    """1D T diagram: bcc/fcc/liquid with reference_phase='fcc', showing relative semi-grand potential."""
    df = _calc_1d_T_bcc_fcc_liquid()
    fig, ax = plt.subplots(figsize=(6, 4))
    lpl.plot_1d_T_phase_diagram(df, ax=ax, reference_phase="fcc")
    ax.set_title("1D T phase diagram with reference_phase='fcc'")
    return _save(fig, out_dir, "1d_T_reference_phase_diagram")


def plot_1d_mu_reference_phase(out_dir: Path, **_) -> Path:
    """1D mu diagram: hcp/fcc/liquid with reference_phase='fcc', showing relative semi-grand potential."""
    df = _calc_1d_mu_hcp_fcc_liquid()
    fig, ax = plt.subplots(figsize=(6, 4))
    lpl.plot_1d_mu_phase_diagram(df, ax=ax, reference_phase="fcc")
    ax.set_title(r"1D $\mu$ phase diagram with reference_phase='fcc'")
    return _save(fig, out_dir, "1d_mu_reference_phase_diagram")


def plot_1d_mu_intermetallic(out_dir: Path, **_) -> Path:
    """1D mu diagram with a subscripted phase name (AB$_2$), exercising bold mathtext labels."""
    df = _calc_1d_mu_intermetallic()
    fig, ax = plt.subplots(figsize=(6, 4))
    lpl.plot_1d_mu_phase_diagram(df, ax=ax)
    ax.set_title(r"1D $\mu$ phase diagram (solid / AB$_2$ intermetallic, T=600K)")
    return _save(fig, out_dir, "1d_mu_intermetallic_phase_diagram")


def plot_2d_basics(out_dir: Path, poly_method: str | None = None, tielines: bool = True, **_) -> Path:
    """2D c-T phase diagram (hcp / fcc / liquid ideal solutions) from Basics.ipynb."""
    fcca = ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * ldp.kB)
    fccb = ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * ldp.kB)
    hcpa = ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
    hcpb = ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB)
    lqda = ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * ldp.kB)
    lqdb = ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * ldp.kB)
    fcc = ldp.IdealSolution("fcc", fcca, fccb)
    hcp = ldp.IdealSolution("hcp", hcpa, hcpb)
    lqd = ldp.IdealSolution("liquid", lqda, lqdb)

    Ts = np.linspace(200, 1000, 50)
    df = ldc.calc_phase_diagram([hcp, fcc, lqd], Ts, mu=100)

    fig, ax = plt.subplots(figsize=(6, 5))
    lpl.plot_phase_diagram(df, ax=ax, tielines=tielines, poly_method=poly_method)
    ax.set_title(f"2D c-T diagram (hcp / fcc / liquid ideal solutions){_title_suffix(poly_method, tielines)}")
    return _save(fig, out_dir, "2d_basics_phase_diagram", _file_suffix(poly_method, tielines))


def plot_2d_basics_mu(out_dir: Path, poly_method: str | None = None, **_) -> Path:
    """2D T-mu phase diagram (hcp / fcc / liquid ideal solutions) from Basics.ipynb."""
    fcca = ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * ldp.kB)
    fccb = ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * ldp.kB)
    hcpa = ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
    hcpb = ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB)
    lqda = ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * ldp.kB)
    lqdb = ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * ldp.kB)
    fcc = ldp.IdealSolution("fcc", fcca, fccb)
    hcp = ldp.IdealSolution("hcp", hcpa, hcpb)
    lqd = ldp.IdealSolution("liquid", lqda, lqdb)

    Ts = np.linspace(200, 1000, 100)
    mus = np.linspace(0.5, 1.5, 100)
    df = ldc.calc_phase_diagram([hcp, fcc, lqd], Ts, mu=mus)

    fig, ax = plt.subplots(figsize=(6, 5))
    lpl.plot_mu_phase_diagram(df, ax=ax, poly_method=poly_method)
    ax.set_title(rf"2D T-$\mu$ diagram (hcp / fcc / liquid ideal solutions){_title_suffix(poly_method)}")
    return _save(fig, out_dir, "2d_basics_mu_phase_diagram", _file_suffix(poly_method))


def plot_2d_toy(out_dir: Path, poly_method: str | None = None, tielines: bool = True, **_) -> Path:
    """2D c-T diagram with a regular-solution liquid + intermediate solid, from Toy.ipynb."""
    l1 = ldp.TemperatureDependentLinePhase(
        "l0", fixed_concentration=0, temperatures=[1, 750, 1000],
        free_energies=[2.00, 1.80, 1.00], interpolator=ldi.PolyFit(3),
    )
    l2 = ldp.TemperatureDependentLinePhase(
        "l1", fixed_concentration=1, temperatures=[1, 750, 1000],
        free_energies=[3.00, 2.80, 2.00], interpolator=ldi.PolyFit(3),
    )
    l3 = ldp.TemperatureDependentLinePhase(
        "l2", fixed_concentration=0.5, temperatures=[1, 750, 1000],
        free_energies=[2.45, 2.00, 1.42], interpolator=ldi.PolyFit(3),
    )
    s1 = ldp.TemperatureDependentLinePhase(
        "s0", fixed_concentration=0, temperatures=[1, 750, 1000],
        free_energies=[1.9, 1.6, 1.2], interpolator=ldi.SGTE(2),
    )
    s2 = ldp.TemperatureDependentLinePhase(
        "s1", fixed_concentration=1, temperatures=[1, 750, 1000],
        free_energies=[2.9, 2.6, 2.2], interpolator=ldi.SGTE(2),
    )
    s3 = ldp.TemperatureDependentLinePhase(
        "s3", fixed_concentration=0.4, temperatures=[1, 750, 1000],
        free_energies=np.array([2.4, 1.85, 1.45]) - 0.05, interpolator=ldi.SGTE(3),
    )
    rliq = ldp.RegularSolution("liquid", [l1, l3, l2])
    sol = ldp.IdealSolution("solid", s1, s2)

    c = np.linspace(0, 1, 75)[1:-1]
    mu = 1 + ldp.kB * 4000 * np.log(c / (1 - c))
    df = ldc.calc_phase_diagram([rliq, sol, s3], np.linspace(500, 1000, 40), mu, refine=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    lpl.plot_phase_diagram(df, ax=ax, tielines=tielines, poly_method=poly_method)
    ax.set_title(f"2D c-T diagram (regular-solution liquid + intermediate solid){_title_suffix(poly_method, tielines)}")
    return _save(fig, out_dir, "2d_toy_phase_diagram", _file_suffix(poly_method, tielines))


def plot_excess_free_energy(out_dir: Path, **_) -> Path:
    """Excess free energy vs concentration (Intermetallics example) from ExcessFreeEnergy.ipynb."""
    solid_a = ldp.LinePhase("A",    fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * ldp.kB)
    solid_b = ldp.LinePhase("B",    fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * ldp.kB)
    solid   = ldp.IdealSolution("solid", solid_a, solid_b)

    liquid_a = ldp.LinePhase("A(l)", fixed_concentration=0, line_energy=-1.9, line_entropy=2.5 * ldp.kB)
    liquid_b = ldp.LinePhase("B(l)", fixed_concentration=1, line_energy=-2.9, line_entropy=2.2 * ldp.kB)
    liquid   = ldp.IdealSolution("liquid", liquid_a, liquid_b)

    inter = ldp.LinePhase("AB$_2$", fixed_concentration=2/3, line_energy=-2.8, line_entropy=1.3 * ldp.kB)

    sigma_pts = [
        ldp.LinePhase("sig@0.30", fixed_concentration=0.30, line_energy=-2.30),
        ldp.LinePhase("sig@0.40", fixed_concentration=0.40, line_energy=-2.50),
        ldp.LinePhase("sig@0.50", fixed_concentration=0.50, line_energy=-2.60),
        ldp.LinePhase("sig@0.60", fixed_concentration=0.60, line_energy=-2.55),
        ldp.LinePhase("sig@0.70", fixed_concentration=0.70, line_energy=-2.40),
    ]
    sigma = ldp.SlowInterpolatingPhase(name="sigma", phases=sigma_pts)

    import pandas as pd
    df = pd.concat(
        [ldc.calc_phase_diagram([solid, liquid, inter, sigma], Ts=T, mu=200, keep_unstable=True)
         for T in [500, 1000, 1600]],
        ignore_index=True,
    )
    g = lpl.plot_excess_free_energy(df, convex_hull=True, height=4, aspect=1.1)
    return _save(g.fig, out_dir, "excess_free_energy")


def plot_excess_free_energy_line_phases(out_dir: Path, **_) -> Path:
    """Excess free energy with only line phases — legend fix for #182."""
    import pandas as pd
    e0 = ldp.LinePhase("A",  fixed_concentration=0,   line_energy=-2.0, line_entropy=1.0 * ldp.kB)
    e1 = ldp.LinePhase("B",  fixed_concentration=1,   line_energy=-3.0, line_entropy=1.5 * ldp.kB)
    inter = ldp.LinePhase("AB", fixed_concentration=0.5, line_energy=-2.8, line_entropy=1.3 * ldp.kB)
    df = pd.concat(
        [ldc.calc_phase_diagram([e0, e1, inter], Ts=T, mu=50, keep_unstable=True)
         for T in [500, 1000, 1500]],
        ignore_index=True,
    )
    g = lpl.plot_excess_free_energy(df, convex_hull=True, height=4, aspect=1.1)
    return _save(g.fig, out_dir, "excess_free_energy_line_phases")


def plot_2d_toy_mu(out_dir: Path, poly_method: str | None = None, **_) -> Path:
    """2D T-mu diagram with a regular-solution liquid + intermediate solid, from Toy.ipynb."""
    l1 = ldp.TemperatureDependentLinePhase(
        "l0", fixed_concentration=0, temperatures=[1, 750, 1000],
        free_energies=[2.00, 1.80, 1.00], interpolator=ldi.PolyFit(3),
    )
    l2 = ldp.TemperatureDependentLinePhase(
        "l1", fixed_concentration=1, temperatures=[1, 750, 1000],
        free_energies=[3.00, 2.80, 2.00], interpolator=ldi.PolyFit(3),
    )
    l3 = ldp.TemperatureDependentLinePhase(
        "l2", fixed_concentration=0.5, temperatures=[1, 750, 1000],
        free_energies=[2.45, 2.00, 1.42], interpolator=ldi.PolyFit(3),
    )
    s1 = ldp.TemperatureDependentLinePhase(
        "s0", fixed_concentration=0, temperatures=[1, 750, 1000],
        free_energies=[1.9, 1.6, 1.2], interpolator=ldi.SGTE(2),
    )
    s2 = ldp.TemperatureDependentLinePhase(
        "s1", fixed_concentration=1, temperatures=[1, 750, 1000],
        free_energies=[2.9, 2.6, 2.2], interpolator=ldi.SGTE(2),
    )
    s3 = ldp.TemperatureDependentLinePhase(
        "s3", fixed_concentration=0.4, temperatures=[1, 750, 1000],
        free_energies=np.array([2.4, 1.85, 1.45]) - 0.05, interpolator=ldi.SGTE(3),
    )
    rliq = ldp.RegularSolution("liquid", [l1, l3, l2])
    sol = ldp.IdealSolution("solid", s1, s2)

    c = np.linspace(0, 1, 75)[1:-1]
    mu = 1 + ldp.kB * 4000 * np.log(c / (1 - c))
    df = ldc.calc_phase_diagram([rliq, sol, s3], np.linspace(500, 1000, 40), mu, refine=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    lpl.plot_mu_phase_diagram(df, ax=ax, poly_method=poly_method)
    ax.set_title(rf"2D T-$\mu$ diagram (regular-solution liquid + intermediate solid){_title_suffix(poly_method)}")
    return _save(fig, out_dir, "2d_toy_mu_phase_diagram", _file_suffix(poly_method))


# Each plot maps to (function, kwargs it actually consumes). 1D plots ignore
# poly_method and tielines, so cross-product iteration over them dedupes
# automatically instead of re-rendering identical files.
PLOTS = {
    "1d_T_three_stable":      (plot_1d_T_three_stable,      ()),
    "1d_T_reference_phase":   (plot_1d_T_reference_phase,   ()),
    "1d_mu_three_stable":     (plot_1d_mu_three_stable,     ()),
    "1d_mu_reference_phase":  (plot_1d_mu_reference_phase,  ()),
    "1d_mu_intermetallic":    (plot_1d_mu_intermetallic,    ()),
    "2d_basics":              (plot_2d_basics,               ("poly_method", "tielines")),
    "2d_basics_mu":           (plot_2d_basics_mu,            ("poly_method",)),
    "2d_toy":                 (plot_2d_toy,                  ("poly_method", "tielines")),
    "2d_toy_mu":              (plot_2d_toy_mu,               ("poly_method",)),
    "excess_free_energy":             (plot_excess_free_energy,             ()),
    "excess_free_energy_line_phases": (plot_excess_free_energy_line_phases, ()),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "_plots",
        help="output directory for PNGs",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="PLOT",
        help=f"restrict to a subset of plots (default: all); choices: {', '.join(sorted(PLOTS))}",
    )
    parser.add_argument(
        "--poly-method",
        choices=POLY_METHODS,
        nargs="+",
        default=None,
        help="one or more polygon-construction methods to cross-product over the 2D plots "
             "(default: library default, single rendering)",
    )
    parser.add_argument(
        "--tielines",
        choices=TIELINE_VALUES,
        nargs="+",
        default=["on"],
        help="tieline modes to cross-product over 2D c-T plots: 'on', 'off', or both (default: on)",
    )
    args = parser.parse_args()

    if args.only:
        unknown = [n for n in args.only if n not in PLOTS]
        for u in unknown:
            print(f"warning: unknown plot name {u!r}; skipping", file=sys.stderr)
        names = [n for n in args.only if n in PLOTS]
        if not names:
            print("warning: no valid plot names given; rendering all plots", file=sys.stderr)
            names = list(PLOTS)
    else:
        names = list(PLOTS)
    poly_methods = args.poly_method if args.poly_method else [None]
    tielines = [v == "on" for v in args.tielines]
    axes = {"poly_method": poly_methods, "tielines": tielines}

    for name in names:
        fn, uses = PLOTS[name]
        for combo in itertools.product(*(axes[k] for k in uses)):
            call_kwargs = dict(zip(uses, combo))
            path = fn(args.out, **call_kwargs)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
