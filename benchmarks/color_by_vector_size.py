"""Size and save time of a ``color_by`` phase diagram in vector formats.

``_plot_scalar_field`` draws each phase field as a Gouraud ``pcolormesh`` on a 200x200
grid.  Written as vector geometry, the SVG backend emits every triangle as its own
``<path>``; rasterised, the field is embedded as one image.  This saves the README
quickstart diagram (``color_by="mu"``) with the meshes rasterised and not, next to the
plain ``color_by=None`` diagram.  Run from the repo root, about a minute and a half:

    python benchmarks/color_by_vector_size.py

Measured 2026-11-03 (matplotlib 3.11.2):

    color_by  rasterized  format     size [MB]   <path>   save [s]
    None      -           svg            0.044       27       0.03
    None      -           pdf            0.016        -       0.12
    mu        True        svg            0.124       35       0.35
    mu        True        pdf            0.069        -       0.52
    mu        False       svg          518.730  1267269      69.12
    mu        False       pdf            1.527        -       0.70
"""

import io
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh

from landau import IdealSolution, LinePhase
from landau.calculate import calc_phase_diagram
from landau.phases import kB
from landau.plot import plot_phase_diagram

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

print(f"{'color_by':<10}{'rasterized':<12}{'format':<8}{'size [MB]':>12}{'<path>':>9}{'save [s]':>11}")
for color_by, rasterized in [(None, None), ("mu", True), ("mu", False)]:
    fig, ax = plt.subplots()
    plot_phase_diagram(df, ax=ax, color_by=color_by)
    if rasterized is not None:
        for mesh in ax.collections:
            if isinstance(mesh, QuadMesh):
                mesh.set_rasterized(rasterized)
    for fmt in ("svg", "pdf"):
        buf = io.BytesIO()
        start = time.perf_counter()
        fig.savefig(buf, format=fmt)
        elapsed = time.perf_counter() - start
        out = buf.getvalue()
        paths = str(out.count(b"<path")) if fmt == "svg" else "-"
        flag = "-" if rasterized is None else str(rasterized)
        print(f"{color_by!s:<10}{flag:<12}{fmt:<8}{len(out) / 1e6:>12.3f}{paths:>9}{elapsed:>11.2f}")
    plt.close(fig)
