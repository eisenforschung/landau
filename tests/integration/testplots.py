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


def plot_1d_T(out_dir: Path, **_) -> Path:
    """1D T phase diagram from notebooks/Basics.ipynb (three pure-A phases)."""
    fcca = ldp.LinePhase("fcc", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * ldp.kB)
    hcpa = ldp.LinePhase("hcp", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
    lqda = ldp.LinePhase("liquid", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * ldp.kB)

    Ts = np.linspace(0, 1000, 25)
    df = ldc.calc_phase_diagram([fcca, hcpa, lqda], Ts, mu=0.0, refine=True, keep_unstable=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    lpl.plot_1d_T_phase_diagram(df, ax=ax)
    ax.set_title("1D T phase diagram (pure A: fcc / hcp / liquid)")
    return _save(fig, out_dir, "1d_T_phase_diagram")


def plot_1d_mu(out_dir: Path, **_) -> Path:
    """1D mu phase diagram from notebooks/Basics.ipynb (hcp vs fcc isothermal)."""
    fcca = ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * ldp.kB)
    fccb = ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * ldp.kB)
    hcpa = ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
    hcpb = ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB)
    fcc = ldp.IdealSolution("fcc", fcca, fccb)
    hcp = ldp.IdealSolution("hcp", hcpa, hcpb)

    df = ldc.calc_phase_diagram([hcp, fcc], Ts=1000.0, mu=100, keep_unstable=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    lpl.plot_1d_mu_phase_diagram(df, ax=ax)
    ax.set_title(r"1D $\mu$ phase diagram (hcp vs fcc, T=1000K)")
    return _save(fig, out_dir, "1d_mu_phase_diagram")


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
    "1d_T":         (plot_1d_T,         ()),
    "1d_mu":        (plot_1d_mu,        ()),
    "2d_basics":    (plot_2d_basics,    ("poly_method", "tielines")),
    "2d_basics_mu": (plot_2d_basics_mu, ("poly_method",)),
    "2d_toy":       (plot_2d_toy,       ("poly_method", "tielines")),
    "2d_toy_mu":    (plot_2d_toy_mu,    ("poly_method",)),
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
        choices=sorted(PLOTS),
        nargs="+",
        help="restrict to a subset of plots (default: all)",
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

    names = args.only or list(PLOTS)
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
