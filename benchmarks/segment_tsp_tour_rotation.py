"""Diagnosis benchmark for the segment-TSP tour rotation in `_segment_tsp_polygon`.

Builds the segment-endpoint TSP instances of the toy T-mu phase diagram
(`tests/integration/testplots.py::plot_2d_toy_mu`), then for each phase region
solves the tour repeatedly with `solve_tsp_record_to_record` and reports, per
trial:

* whether the tour length equals the exact (dynamic programming) optimum,
* whether the *naive* chunk reconstruction (first-encounter direction, no
  rotation) yields a valid polygon,
* whether rotating the tour off an intra-segment wrap-around edge — what
  `_segment_tsp_polygon` does — yields a valid polygon.

The heuristic finds the optimal tour length in every trial; invalid polygons
come exclusively from tours returned in a rotation that cuts through a
segment's zero-cost intra edge, and the rotation repairs every one of them.

Run: ``python benchmarks/segment_tsp_tour_rotation.py``
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import shapely
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_record_to_record
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

import landau.calculate as ldc
import landau.interpolate as ldi
import landau.phases as ldp
from landau.plot import cluster_phase
from landau.poly import SegmentPythonTsp, _segments_from_labels

TRIALS = 30
MAX_ITERATIONS = 10


def toy_mu_instances():
    """Segment lists and endpoint distance matrices per phase region of the
    toy T-mu diagram."""
    l1 = ldp.TemperatureDependentLinePhase(
        "l0", fixed_concentration=0, temperatures=[1, 750, 1000],
        free_energies=[2.00, 1.80, 1.00], interpolator=ldi.PolyFit(3))
    l2 = ldp.TemperatureDependentLinePhase(
        "l1", fixed_concentration=1, temperatures=[1, 750, 1000],
        free_energies=[3.00, 2.80, 2.00], interpolator=ldi.PolyFit(3))
    l3 = ldp.TemperatureDependentLinePhase(
        "l2", fixed_concentration=0.5, temperatures=[1, 750, 1000],
        free_energies=[2.45, 2.00, 1.42], interpolator=ldi.PolyFit(3))
    s1 = ldp.TemperatureDependentLinePhase(
        "s0", fixed_concentration=0, temperatures=[1, 750, 1000],
        free_energies=[1.9, 1.6, 1.2], interpolator=ldi.SGTE(2))
    s2 = ldp.TemperatureDependentLinePhase(
        "s1", fixed_concentration=1, temperatures=[1, 750, 1000],
        free_energies=[2.9, 2.6, 2.2], interpolator=ldi.SGTE(2))
    s3 = ldp.TemperatureDependentLinePhase(
        "s3", fixed_concentration=0.4, temperatures=[1, 750, 1000],
        free_energies=np.array([2.4, 1.85, 1.45]) - 0.05, interpolator=ldi.SGTE(3))
    rliq = ldp.RegularSolution("liquid", [l1, l3, l2])
    sol = ldp.IdealSolution("solid", s1, s2)

    c = np.linspace(0, 1, 75)[1:-1]
    mu = 1 + ldp.kB * 4000 * np.log(c / (1 - c))
    df = ldc.calc_phase_diagram([rliq, sol, s3], np.linspace(500, 1000, 40), mu, refine=True)
    df = df.query("stable").copy()
    df = cluster_phase(df)

    instances = {}
    prepared = SegmentPythonTsp().prepare(df)
    for (phase, _unit), dd in prepared.groupby(["phase", "phase_unit"]):
        dd = dd.drop(columns=["phase", "phase_unit"])
        seg = dd["border_segment"].to_numpy()
        idx = np.argsort(dd["mu"].to_numpy())
        pp = dd[["mu", "T"]].to_numpy()[idx]
        seg = seg[idx]
        mask = np.isfinite(pp).all(axis=-1)
        pp, seg = pp[mask], seg[mask]
        _, unique_idx = np.unique(pp, axis=0, return_index=True)
        unique_idx.sort()
        pp, seg = pp[unique_idx], seg[unique_idx]
        pp_scaled = StandardScaler().fit_transform(pp)
        instances[phase] = _segments_from_labels(pp_scaled, seg)
    return instances


def endpoint_dm_int(segments):
    n = len(segments)
    endpoints = np.array([[s[0], s[-1]] for s in segments]).reshape(2 * n, -1)
    dm = pairwise_distances(endpoints)
    for i in range(n):
        dm[2 * i, 2 * i + 1] = 0
        dm[2 * i + 1, 2 * i] = 0
    pos = dm[dm > 0]
    return (dm / pos.min()).round().astype(int)


def reconstruct(segments, tour):
    seen, chunks = set(), []
    for node in tour:
        seg = node // 2
        if seg in seen:
            continue
        seen.add(seg)
        chunks.append(segments[seg] if node % 2 == 0 else segments[seg][::-1])
    return shapely.Polygon(np.vstack(chunks))


def rotate_off_intra_edge(tour):
    tour = list(tour)
    while tour[0] // 2 == tour[-1] // 2:
        tour.append(tour.pop(0))
    return tour


def main():
    for phase, segments in toy_mu_instances().items():
        dm_int = endpoint_dm_int(segments)
        _opt_tour, opt_len = solve_tsp_dynamic_programming(dm_int)
        optimal = naive_valid = rotated_valid = 0
        for _ in range(TRIALS):
            tour, length = solve_tsp_record_to_record(dm_int, max_iterations=MAX_ITERATIONS)
            optimal += length == opt_len
            naive_valid += reconstruct(segments, tour).is_valid
            rotated_valid += reconstruct(segments, rotate_off_intra_edge(tour)).is_valid
        print(f"{phase}: {len(segments)} segments | tour length optimal {optimal}/{TRIALS} | "
              f"valid polygon: naive {naive_valid}/{TRIALS}, rotated {rotated_valid}/{TRIALS}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
