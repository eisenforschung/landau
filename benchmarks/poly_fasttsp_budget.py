"""Sweep fast_tsp compute budget vs polygon quality.

Builds the same synthetic ring-shaped phase region as ``poly_methods.py`` (true
shape: unit disc, area = pi). For each ``duration_seconds`` setting we call
``FastTsp._make`` and ``SegmentFastTsp._make`` over a few seeds, then measure:

  * IoU against the true unit disc
  * polygon validity (``shape.is_valid``)
  * wall-time per call

Run: ``python benchmarks/poly_fasttsp_budget.py``
"""

import time

import numpy as np
import shapely

from landau.poly import FastTsp, SegmentFastTsp

from poly_methods import make_problem


def iou(poly, ref):
    if poly is None or poly.is_empty:
        return 0.0
    if not poly.is_valid:
        poly = poly.buffer(0)
    inter = poly.intersection(ref).area
    union = poly.union(ref).area
    return inter / union if union > 0 else 0.0


REFERENCE = shapely.Point(0, 0).buffer(1.0, quad_segs=256)
BUDGETS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
CASES = [(20, 20), (40, 30), (80, 30)]
SEEDS = (0, 1, 2)


def run(method_cls, case):
    ns, ppc = case
    print(f"\n=== {method_cls.__name__} on {ns}seg x {ppc}pt ({ns*ppc} border pts) ===")
    print(f"{'budget_s':>10s} | {'mean IoU':>10s} | {'min IoU':>10s} | "
          f"{'valid':>6s} | {'mean ms':>10s}")
    print("-" * 60)
    for budget in BUDGETS:
        ious, valids, times = [], [], []
        for seed in SEEDS:
            pp, border, seg = make_problem(ns, ppc, seed=seed)
            sl = seg if method_cls is SegmentFastTsp else np.ones_like(seg)
            method = method_cls(duration_seconds=budget)
            t0 = time.perf_counter()
            shape = method._make(pp, border, sl)
            times.append(time.perf_counter() - t0)
            if shape is None:
                ious.append(0.0)
                valids.append(False)
                continue
            ious.append(iou(shape, REFERENCE))
            valids.append(shape.is_valid)
        print(f"{budget:10.3f} | {np.mean(ious):10.4f} | {min(ious):10.4f} | "
              f"{sum(valids)}/{len(valids):<4d} | {np.mean(times)*1000:10.1f}")


def main():
    for case in CASES:
        run(FastTsp, case)
    for case in CASES:
        run(SegmentFastTsp, case)


if __name__ == "__main__":
    main()
