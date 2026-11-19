"""Cost of ``get_transitions`` and of the tie-line pairing inside it.

``get_transitions`` pairs the border rows at each ``(mu, T)`` into tie-lines
(``_tie_lines``) and clusters them.  The segment poly methods call it once
per plot.  This times, on refined 2-D frames:

* ``pairing``: ``_tie_lines`` on the frame's border rows;
* ``per group``: the same pairing as one ``groupby.apply`` call per
  ``(mu, T)``, the form ``_tie_lines`` is vectorised from;
* ``get_transitions``: the whole call;
* ``plot``: ``plot_phase_diagram`` with the default poly method.

Run with ``python benchmarks/bench_get_transitions.py`` from the repo root.
"""

from __future__ import annotations

import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from landau.calculate import _tie_lines, calc_phase_diagram, get_transitions
from landau.phases import IdealSolution, LinePhase, kB
from landau.plot import plot_phase_diagram


def line(name, c, energy, entropy):
    return LinePhase(name, fixed_concentration=c, line_energy=energy, line_entropy=entropy * kB)


def ideal(name, energy_A, entropy_A, energy_B, entropy_B):
    return IdealSolution(name, line(name + "A", 0, energy_A, entropy_A), line(name + "B", 1, energy_B, entropy_B))


def eutectic():
    """Three solution phases, one invariant."""
    phases = [ideal("hcp", -2.975, 1.8, -1.95, 1.1), ideal("fcc", -3.00, 1.0, -2.00, 1.1),
              ideal("liquid", -2.75, 5.0, -1.75, 4.4)]
    return calc_phase_diagram(phases, np.linspace(200, 1000, 50), mu=100)


def compounds():
    """Two terminal solutions, three line compounds and a liquid."""
    phases = [ideal("alpha", -3.0, 1.0, -2.40, 1.0), ideal("beta", -2.92, 1.0, -2.5, 1.0),
              line("gamma", 0.5, -2.86, 1.0), line("delta", 0.25, -2.955, 1.0), line("epsilon", 0.75, -2.70, 1.0),
              ideal("liquid", -2.8, 3.0, -2.35, 3.0)]
    return calc_phase_diagram(phases, np.linspace(300, 1500, 60), mu=100)


def per_group(dd):
    c, phase = dd["c"].to_numpy(), dd["phase"].to_numpy()
    if len(dd) == 1:
        return pd.DataFrame({"transition": [phase[0]], "c": c, "phase": phase})
    rows = []
    for i in range(len(dd) - 1):
        name = f"{phase[i]}-{phase[i + 1]}"
        rows.append((name, c[i], phase[i]))
        rows.append((name, c[i + 1], phase[i + 1]))
    return pd.DataFrame(rows, columns=["transition", "c", "phase"])


def best(fn, reps=5):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times) * 1e3


def plot(df):
    fig, ax = plt.subplots()
    plot_phase_diagram(df, ax=ax)
    plt.close(fig)


def timings(df):
    bdf = df.query("border").sort_values("c")
    grouped = bdf.groupby(["mu", "T"])[["c", "phase"]]
    return len(bdf), (
        best(lambda: _tie_lines(bdf)),
        best(lambda: grouped.apply(per_group, include_groups=False)),
        best(lambda: get_transitions(df)),
        best(lambda: plot(df), reps=3),
    )


def main():
    warnings.simplefilter("ignore")
    print("| frame | border rows | pairing [ms] | per group [ms] | get_transitions [ms] | plot [ms] |")
    print("|---|---|---|---|---|---|")
    for make in (eutectic, compounds):
        rows, times = timings(make())
        print(f"| {make.__name__} | {rows} | " + " | ".join(f"{t:.1f}" for t in times) + " |")


if __name__ == "__main__":
    main()
