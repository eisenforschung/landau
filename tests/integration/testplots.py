"""Render reference phase diagrams used for visual review of landau changes.

Each function below builds a minimal set of phases drawn from the example
notebooks and saves a single phase-diagram PNG into the output directory
(``tests/integration/_plots`` by default, override with ``--out``).

The script is intentionally not a pytest test: the "correctness" of a
landau plot is hard to formalise so we instead post the rendered figures
on PRs labeled ``testplot`` and review them by eye.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import landau.calculate as ldc
import landau.interpolate as ldi
import landau.phases as ldp
import landau.plot as lpl


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_1d_T(out_dir: Path) -> Path:
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


def plot_1d_mu(out_dir: Path) -> Path:
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


def plot_2d_basics(out_dir: Path) -> Path:
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
    lpl.plot_phase_diagram(df, ax=ax, tielines=True)
    ax.set_title("2D c-T diagram (hcp / fcc / liquid ideal solutions)")
    return _save(fig, out_dir, "2d_basics_phase_diagram")


def plot_2d_basics_mu(out_dir: Path) -> Path:
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
    lpl.plot_mu_phase_diagram(df, ax=ax)
    ax.set_title(r"2D T-$\mu$ diagram (hcp / fcc / liquid ideal solutions)")
    return _save(fig, out_dir, "2d_basics_mu_phase_diagram")


def plot_2d_toy(out_dir: Path) -> Path:
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
    lpl.plot_phase_diagram(df, ax=ax, tielines=True)
    ax.set_title("2D c-T diagram (regular-solution liquid + intermediate solid)")
    return _save(fig, out_dir, "2d_toy_phase_diagram")


PLOTS = {
    "1d_T": plot_1d_T,
    "1d_mu": plot_1d_mu,
    "2d_basics": plot_2d_basics,
    "2d_basics_mu": plot_2d_basics_mu,
    "2d_toy": plot_2d_toy,
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
        nargs="*",
        help="restrict to a subset of plots",
    )
    args = parser.parse_args()

    names = args.only or list(PLOTS)
    for name in names:
        path = PLOTS[name](args.out)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
