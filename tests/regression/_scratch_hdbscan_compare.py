"""Scratch script for issue #456: compare AgglomerativeClustering vs HDBSCAN on the
Y-Zn liquid fixture. Not part of the test suite -- generates comparison PNGs for the
issue discussion, then gets removed.
"""
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN

from landau import poly
from landau.calculate import _rescale_T, calc_phase_diagram
from landau.phases import kB
from landau import LinePhase, IdealSolution

DATA_PATH = Path(__file__).parent / "data" / "issue_456_yzn_liquid.csv.gz"
OUT_DIR = Path(__file__).parent.parent.parent / "_hdbscan_compare"
OUT_DIR.mkdir(exist_ok=True)


def load_yzn():
    with gzip.open(DATA_PATH, "rt") as f:
        df = pd.read_csv(f)
    return df.query("stable").copy()


def agglomerative_labels(dd, distance_threshold):
    t = _rescale_T(dd["T"])
    clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage="single")
    return clusterer.fit_predict(np.transpose([t, dd["c"]]))


def hdbscan_labels(dd, **kwargs):
    t = _rescale_T(dd["T"])
    clusterer = HDBSCAN(**kwargs)
    return clusterer.fit_predict(np.transpose([t, dd["c"]]))


def scatter_panel(ax, dd, labels, title):
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    for lab in sorted(set(labels)):
        sel = labels == lab
        if lab == -1:
            ax.scatter(dd["c"][sel], dd["T"][sel], s=4, color="lightgray", label="noise")
        else:
            ax.scatter(dd["c"][sel], dd["T"][sel], s=4, label=f"cluster {lab}")
    ax.set_xlabel("c")
    ax.set_ylabel("T [K]")
    ax.set_title(f"{title}\n{n_clusters} clusters, {n_noise} noise pts")
    ax.set_xlim(-0.02, 1.02)


def polygon_panel(ax, dd, labels, title):
    dd = dd.copy()
    dd["phase"] = "liquid"
    dd["phase_unit"] = labels
    dd = dd.query("phase_unit >= 0")
    pm = poly.handle_poly_method("concave")
    polys = pm.apply(dd, variables=["c", "T"])
    for (_phase, _unit), p in polys.items():
        xy = p.get_xy()
        ax.fill(xy[:, 0], xy[:, 1], alpha=0.5, edgecolor="black")
    ax.set_xlabel("c")
    ax.set_ylabel("T [K]")
    ax.set_title(f"{title}\n{len(polys)} polygon(s)")
    ax.set_xlim(-0.02, 1.02)


def main():
    dd = load_yzn()
    print("n stable liquid rows:", len(dd))

    scatter_configs = [
        ("agglomerative, distance_threshold=0.5 (old default)", agglomerative_labels(dd, 0.5)),
        ("agglomerative, distance_threshold=0.2 (new default)", agglomerative_labels(dd, 0.2)),
        ("HDBSCAN, min_cluster_size=5 (sklearn default)", hdbscan_labels(dd, min_cluster_size=5)),
        ("HDBSCAN, min_cluster_size=50", hdbscan_labels(dd, min_cluster_size=50)),
        ("HDBSCAN, min_cluster_size=100 (tuned to match)", hdbscan_labels(dd, min_cluster_size=100)),
        ("HDBSCAN, min_cluster_size=2000 (over-tuned)", hdbscan_labels(dd, min_cluster_size=2000)),
    ]
    for title, labels in scatter_configs:
        print(title, "->", sorted(set(labels)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (title, labels) in zip(axes.flat, scatter_configs):
        scatter_panel(ax, dd, labels, title)
    fig.suptitle("Y-Zn liquid fixture (9999 stable rows) -- clustering comparison")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_comparison.png", dpi=130)
    plt.close(fig)

    # Polygon-level comparison: agglomerative old/new default vs. the HDBSCAN
    # setting that happens to reproduce the right answer on this fixture.
    poly_configs = [
        ("agglomerative, distance_threshold=0.5 (old default)", agglomerative_labels(dd, 0.5)),
        ("agglomerative, distance_threshold=0.2 (new default)", agglomerative_labels(dd, 0.2)),
        ("HDBSCAN, min_cluster_size=100 (tuned to match)", hdbscan_labels(dd, min_cluster_size=100)),
    ]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, labels) in zip(axes2, poly_configs):
        polygon_panel(ax, dd, labels, title)
    fig2.suptitle("Y-Zn liquid polygons built from each clustering's labels (poly_method='concave')")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "polygon_comparison.png", dpi=130)
    plt.close(fig2)

    # Density sensitivity: the eutectic fixture from tests/conftest.py samples the
    # liquid phase ~25x sparser (400 stable rows vs. 9999) than the Y-Zn data. Show
    # that HDBSCAN's min_cluster_size=100 (tuned above) no longer separates anything
    # useful there, and needs its own retuning, while distance_threshold=0.2 (scale-free
    # on normalised coordinates) needs none.
    fcc = IdealSolution(
        "fcc",
        LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * kB),
        LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * kB),
    )
    hcp = IdealSolution(
        "hcp",
        LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * kB),
        LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * kB),
    )
    liquid = IdealSolution(
        "liquid",
        LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * kB),
        LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * kB),
    )
    eutectic_df = calc_phase_diagram([hcp, fcc, liquid], np.linspace(200.0, 1000.0, 25), mu=50, refine=True)
    stable_counts = eutectic_df.groupby("phase")["stable"].sum()
    print("eutectic fixture stable rows per phase:\n", stable_counts)

    eu_fcc = eutectic_df.query("stable and phase == 'fcc'").copy()
    eu_configs = [
        ("agglomerative, distance_threshold=0.2 (unchanged)", agglomerative_labels(eu_fcc, 0.2)),
        ("HDBSCAN, min_cluster_size=100 (same knob as Y-Zn)", hdbscan_labels(eu_fcc, min_cluster_size=100)),
        ("HDBSCAN, min_cluster_size=5 (re-tuned down for this grid)", hdbscan_labels(eu_fcc, min_cluster_size=5)),
    ]
    for title, labels in eu_configs:
        print("eutectic fcc:", title, "->", sorted(set(labels)))

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, labels) in zip(axes3, eu_configs):
        scatter_panel(ax, eu_fcc, labels, title)
    fig3.suptitle(f"eutectic fixture, single-region 'fcc' phase ({len(eu_fcc)} stable rows) -- same knobs as above")
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "density_sensitivity.png", dpi=130)
    plt.close(fig3)


if __name__ == "__main__":
    main()
