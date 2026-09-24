"""Stress test for the transition-temperature label placement (#463, PR #474).

Builds binary systems with many synthetic line compounds between an ideal
solid and an ideal liquid, so the refined diagram carries a eutectic between
every neighbouring pair of compounds, a congruent melting point per compound
and the two terminal melting points: some twenty invariants, many at similar
temperatures. Two variants: ``crowded`` gives every compound the same entropy,
so the eutectics and melting points pile up in a narrow temperature band;
``spread`` staggers the entropies and with them the melting points.

Each variant is drawn in c-T and mu-T with ``transition_temperatures=True``
and the placed labels are measured against the invariants they label:

* displacement of the label centre from its anchor, in label heights,
* pairs of label boxes that overlap,
* labels whose box crosses a phase-polygon outline (frame edges excluded),
* pairs of labels sharing horizontal extent whose vertical order disagrees
  with their temperature order,
* invariants that were coalesced into one label,
* wall time of the annotation.

Usage::

    python benchmarks/stress_transition_labels.py [out_dir]

writes ``stress_<variant>_<axes>.png`` into ``out_dir`` (default
``benchmarks/_plots``) and prints one metrics row per figure.
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely

from landau.calculate import calc_phase_diagram
from landau.features import Locus
from landau.phases import IdealSolution, LinePhase, kB
from landau.plot import get_polygons, plot_mu_phase_diagram, plot_phase_diagram
from landau.plot.labels import _LABEL_PAD, _annotate_transition_temperatures, _get_renderer, _shapely_polygon

N_COMPOUNDS = 8


def system(variant):
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
    rng = np.random.default_rng(0)
    compounds = []
    for k in range(1, N_COMPOUNDS + 1):
        c = k / (N_COMPOUNDS + 1)
        # Formation energies on a convex curve, so every compound lies on the
        # low-temperature hull; the depth puts the congruent melting points
        # near 1500 K, and the liquid's mixing entropy makes a eutectic between
        # every neighbouring pair. ``crowded`` gives every compound the same
        # entropy, so all those invariants land within about a hundred kelvin
        # of each other; ``spread`` staggers the melting points instead.
        dE = -0.12 + 0.15 * (c - 0.5) ** 2
        S = 1.3 if variant == "crowded" else rng.uniform(1.2, 1.7)
        E = (1 - c) * -2.0 + c * -3.0 + dE
        compounds.append(LinePhase(f"A{N_COMPOUNDS + 1 - k}B{k}", fixed_concentration=c, line_energy=E,
                                   line_entropy=S * kB))
    return [solid, liquid, *compounds]


def invariants(df, variables):
    """(x, T) anchor of every tagged invariant, as _annotate_transition_temperatures places them."""
    out = []
    triple = df[df["locus"] == Locus.TRIPLE]
    for (mu, T), grp in triple.groupby(["mu", "T"], sort=False):
        out.append((grp["c"].median() if variables[0] == "c" else mu, T))
    congruent = df[df["locus"] == Locus.CONGRUENT]
    for (mu, T), grp in congruent.groupby(["mu", "T"], sort=False):
        out.append((grp["c"].mean() if variables[0] == "c" else mu, T))
    return out


def measure(ax, df, polys, variables):
    renderer = _get_renderer(ax.figure)
    axbb = ax.get_window_extent(renderer)
    frame = shapely.box(axbb.x0, axbb.y0, axbb.x1, axbb.y1).exterior.buffer(_LABEL_PAD)
    outlines = []
    for patch in polys:
        region = _shapely_polygon(ax.transData.transform(patch.get_xy()))
        if region is not None:
            outlines.append(region.exterior.difference(frame))
    outline = shapely.union_all(outlines) if outlines else None

    anchors = [(ax.transData.transform((x, T)), T) for x, T in invariants(df, variables)]
    labels = []
    for text in ax.texts:
        if not text.get_text().endswith(" K"):
            continue
        box = text.get_window_extent(renderer)
        center = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
        T = float(text.get_text()[:-2])
        # nearest invariant that rounds to this label's text
        cands = [(np.hypot(*(center - a)), a) for a, aT in anchors if f"{aT:.0f}" == f"{T:.0f}"]
        dist, anchor = min(cands, key=lambda t: t[0])
        labels.append(dict(T=T, box=box, center=center, anchor=anchor, height=box.height,
                           disp=dist / box.height))
    overlaps = crossings = swaps = 0
    for i, a in enumerate(labels):
        ba = shapely.box(*a["box"].extents)
        if outline is not None and ba.intersects(outline):
            crossings += 1
        for b in labels[i + 1:]:
            bb = shapely.box(*b["box"].extents)
            if ba.intersects(bb):
                overlaps += 1
            shared = min(a["box"].x1, b["box"].x1) - max(a["box"].x0, b["box"].x0)
            if shared > 0 and a["T"] != b["T"] and (a["center"][1] - b["center"][1]) * (a["T"] - b["T"]) < 0:
                swaps += 1
    return dict(
        invariants=len(anchors), labels=len(labels),
        disp_max=max(label["disp"] for label in labels), disp_mean=np.mean([label["disp"] for label in labels]),
        overlaps=overlaps, crossings=crossings, swaps=swaps,
    )


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'figure':22s} {'inv':>4s} {'lab':>4s} {'disp_max':>8s} {'disp_mean':>9s} {'overlap':>7s} {'cross':>5s} {'swap':>4s} {'t_label':>7s}")
    for variant, T_max in (("crowded", 1700.0), ("spread", 2400.0)):
        phases = system(variant)
        df = calc_phase_diagram(phases, Ts=np.linspace(300.0, T_max, 100), mu=200)
        for variables, plotter in ((["c", "T"], plot_phase_diagram), (["mu", "T"], plot_mu_phase_diagram)):
            fig, ax = plt.subplots(figsize=(8, 6))
            plotter(df, ax=ax, transition_temperatures=False, triplepoints=True)
            polys = get_polygons(df, variables=variables)
            t0 = time.perf_counter()
            _annotate_transition_temperatures(df, polys, ax=ax, variables=variables)
            t_label = time.perf_counter() - t0
            m = measure(ax, df, polys, variables)
            name = f"stress_{variant}_{variables[0]}T"
            fig.savefig(out_dir / f"{name}.png", dpi=110)
            plt.close(fig)
            print(f"{name:22s} {m['invariants']:4d} {m['labels']:4d} {m['disp_max']:8.2f} {m['disp_mean']:9.2f} "
                  f"{m['overlaps']:7d} {m['crossings']:5d} {m['swaps']:4d} {t_label:6.2f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/_plots"))
