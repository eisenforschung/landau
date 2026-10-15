"""
Render the Mg-Ca phase diagram shown in README.md.

Phases come from the MgCa worked example (notebooks/MgCa/MgCaPhases.pckl); the
calc_phase_diagram call mirrors the one in notebooks/MgCa/MgCa.ipynb.

Usage
-----
    python scripts/readme_mgca.py
    → writes docs/images/mgca_phase_diagram.png
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import landau as ld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    phases = pd.read_pickle(os.path.join(ROOT, "notebooks", "MgCa", "MgCaPhases.pckl"))
    df = ld.calculate.calc_phase_diagram(phases, Ts=np.linspace(100, 1500, 50), mu=150, keep_unstable=True)

plt.style.use("default")
ld.plot.plot_phase_diagram(df, transition_temperatures=True)
plt.savefig(os.path.join(ROOT, "docs", "images", "mgca_phase_diagram.png"), dpi=130, bbox_inches="tight")
