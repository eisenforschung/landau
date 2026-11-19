"""
Render the figures shown in README.md and on the documentation landing page (docs/index.md).

The quickstart figure runs the README's quickstart code;
every gallery figure rebuilds the system of the notebook its card links to.

Usage
-----
    python scripts/readme_images.py
    → writes docs/images/{quickstart_phase_diagram,mgca_phase_diagram,transition_temperatures,
      point_defects,excess_free_energy,scan_1d,mu_phase_diagram}.png
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from landau import FastInterpolatingPhase, IdealSolution, LinePhase
from landau.calculate import calc_phase_diagram
from landau.interpolate import PolyFit
from landau.phases import kB
from landau.phases.pointdefects import ConstantPointDefect, PointDefectedPhase, PointDefectSublattice
from landau.plot import (
    plot_1d_T_phase_diagram,
    plot_excess_free_energy,
    plot_mu_phase_diagram,
    plot_phase_diagram,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "docs", "images")

# Gallery cards sit three to a row, so they share one small canvas.
CARD_FIGSIZE = (4.8, 3.6)
CARD_DPI = 170


def line(name, c, E, S):
    return LinePhase(name, fixed_concentration=c, line_energy=E, line_entropy=S * kB)


def ideal(name, E_A, S_A, E_B, S_B):
    return IdealSolution(name, line(name + "A", 0, E_A, S_A), line(name + "B", 1, E_B, S_B))


def card():
    return plt.subplots(figsize=CARD_FIGSIZE, layout="constrained")[1]


def save(name, **kwargs):
    plt.savefig(os.path.join(IMAGES, name + ".png"), **kwargs)
    plt.close()


def quickstart():
    """README.md, Quickstart."""
    solid = IdealSolution(
        "solid",
        LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB),
        LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB),
    )
    liquid = IdealSolution(
        "liquid",
        LinePhase("A(l)", fixed_concentration=0, line_energy=-1.9, line_entropy=2.5 * kB),
        LinePhase("B(l)", fixed_concentration=1, line_energy=-2.9, line_entropy=2.2 * kB),
    )
    df = calc_phase_diagram([solid, liquid], Ts=np.linspace(400, 1800, 100), mu=100)
    plot_phase_diagram(df)
    save("quickstart_phase_diagram", dpi=130, bbox_inches="tight")


def mgca():
    """notebooks/MgCa/MgCa.ipynb."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        phases = pd.read_pickle(os.path.join(ROOT, "notebooks", "MgCa", "MgCaPhases.pckl"))
        df = calc_phase_diagram(phases, Ts=np.linspace(100, 1500, 50), mu=150)
    plot_phase_diagram(df, transition_temperatures=True, ax=card())
    save("mgca_phase_diagram", dpi=CARD_DPI)


def transition_temperatures():
    """notebooks/TransitionTemperatures.ipynb, Syntectic."""
    E_A, S_A, E_B, S_B, L0 = -2.8, 3.0, -2.35, 3.0, 0.25
    liquid = FastInterpolatingPhase(
        "liquid",
        phases=[
            line(f"liquid{i}", c, (1 - c) * E_A + c * E_B + L0 * c * (1 - c), (1 - c) * S_A + c * S_B)
            for i, c in enumerate([0, 0.25, 0.5, 0.75, 1])
        ],
        add_entropy=True,
        interpolator=PolyFit(4),
    )
    phases = [line("α", 0, -3.0, 1.0), line("β", 1, -2.5, 1.0), line("γ", 0.5, -2.78, 1.2), liquid]
    df = calc_phase_diagram(phases, np.linspace(300, 1600, 80), mu=100)
    plot_phase_diagram(df, transition_temperatures=True, element="B", ax=card())
    save("transition_temperatures", dpi=CARD_DPI)


def point_defects():
    """notebooks/PointDefects.ipynb, exact model."""
    host = line("AB", 0.5, -0.40, 1.0)
    sublattices = [
        PointDefectSublattice(
            name="alpha",
            sublattice=0,
            sublattice_fraction=0.5,
            defects=[
                ConstantPointDefect("B_alpha", excess_energy=0.28, excess_entropy=0, excess_solutes=+1),
                ConstantPointDefect("V_alpha", excess_energy=0.45, excess_entropy=0, excess_solutes=+0.5),
            ],
        ),
        PointDefectSublattice(
            name="beta",
            sublattice=1,
            sublattice_fraction=0.5,
            defects=[
                ConstantPointDefect("A_beta", excess_energy=0.30, excess_entropy=0, excess_solutes=-1),
                ConstantPointDefect("V_beta", excess_energy=0.50, excess_entropy=0, excess_solutes=-0.5),
            ],
        ),
    ]
    phases = [
        ideal("solid", -0.33, 1.0, -0.30, 1.0),
        ideal("liquid", -0.10, 3.3, -0.13, 3.3),
        PointDefectedPhase(name="AB", line_phase=host, sublattices=sublattices),
    ]
    df = calc_phase_diagram(phases, Ts=np.linspace(200, 1400, 25), mu=np.linspace(-0.8, 0.8, 200))
    plot_phase_diagram(df, triplepoints=True, ax=card())
    save("point_defects", dpi=CARD_DPI)


def excess_free_energy():
    """notebooks/ExcessFreeEnergy.ipynb, at 1000 K."""
    sigma = FastInterpolatingPhase(
        name="sigma",
        phases=[
            LinePhase(name=f"sig@{c:.2f}", fixed_concentration=c, line_energy=E)
            for c, E in [(0.30, -2.30), (0.40, -2.50), (0.50, -2.60), (0.60, -2.55), (0.70, -2.40)]
        ],
    )
    phases = [
        ideal("solid", -2.0, 1.0, -3.0, 1.5),
        ideal("liquid", -1.9, 2.5, -2.9, 2.2),
        line("AB₂", 2 / 3, -2.8, 1.3),
        sigma,
    ]
    df = calc_phase_diagram(phases, Ts=1000, mu=200, keep_unstable=True)
    card()
    plot_excess_free_energy(df, convex_hull=True)
    save("excess_free_energy", dpi=CARD_DPI)


def scan_1d():
    """notebooks/Basics.ipynb, its solution phases along a temperature scan."""
    phases = [
        ideal("fcc", -3.00, 1.0, -2.00, 1.1),
        ideal("hcp", -2.975, 1.8, -1.95, 1.1),
        ideal("liquid", -2.75, 5.0, -1.75, 4.4),
    ]
    df = calc_phase_diagram(phases, np.linspace(100, 1000, 40), mu=0.0, keep_unstable=True)
    plot_1d_T_phase_diagram(df, ax=card())
    save("scan_1d", dpi=CARD_DPI)


def mu_phase_diagram():
    """notebooks/PlotGallery.ipynb, 2D chemical-potential-temperature diagram."""
    phases = [
        ideal("hcp", -3.00, 1.0, -1.93, 1.0),
        ideal("fcc", -2.95, 1.9, -1.95, 1.9),
        ideal("bcc", -2.90, 1.1, -2.00, 1.1),
        ideal("liquid", -2.76, 5.0, -1.69, 5.0),
    ]
    df = calc_phase_diagram(phases, np.linspace(200, 1000, 80), mu=np.linspace(0.45, 1.55, 80))
    plot_mu_phase_diagram(df, ax=card())
    save("mu_phase_diagram", dpi=CARD_DPI)


if __name__ == "__main__":
    plt.style.use("default")
    for render in (
        quickstart,
        mgca,
        transition_temperatures,
        point_defects,
        excess_free_energy,
        scan_1d,
        mu_phase_diagram,
    ):
        render()
