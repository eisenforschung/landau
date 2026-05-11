"""Timing benchmark for landau.poly methods.

Builds a synthetic ring-shaped phase region from `n_segments` unit-circle arcs
plus a cloud of interior filler points, then calls each method's ``_make``
directly so we measure algorithm cost without the ``apply``/``prepare``
overhead. Best wall-time of 2 runs is reported per cell.

Run: ``python benchmarks/poly_methods.py``
"""

import time

import numpy as np

from landau.poly import (
    Concave,
    FastTsp,
    PythonTsp,
    SegmentFastTsp,
    SegmentPythonTsp,
    Segments,
)


def make_problem(n_segments, points_per_segment, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pp_list, lab_list = [], []
    for k in range(n_segments):
        t0 = 2 * np.pi * k / n_segments
        t1 = 2 * np.pi * (k + 1) / n_segments
        ts = np.linspace(t0, t1, points_per_segment, endpoint=False)
        xs = np.cos(ts) + rng.normal(0, noise, points_per_segment)
        ys = np.sin(ts) + rng.normal(0, noise, points_per_segment)
        pp_list.append(np.column_stack([xs, ys]))
        lab_list.append(np.full(points_per_segment, k))
    n_int = max(5, n_segments * points_per_segment // 4)
    rr = 0.9 * np.sqrt(rng.uniform(0, 1, n_int))
    tt = rng.uniform(0, 2 * np.pi, n_int)
    interior = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
    pp_list.append(interior)
    lab_list.append(np.full(n_int, -1))
    pp = np.vstack(pp_list)
    segment_label = np.concatenate(lab_list)
    border = segment_label >= 0
    return pp, border, segment_label


def time_call(fn, repeat=2):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


METHODS = {
    "Concave":          Concave(),
    "Segments":         Segments(),
    "PythonTsp":        PythonTsp(max_iterations=3),
    "FastTsp":          FastTsp(duration_seconds=0.1),
    "SegmentPythonTsp": SegmentPythonTsp(max_iterations=3),
    "SegmentFastTsp":   SegmentFastTsp(duration_seconds=0.1),
}

CASES = [(10, 10), (20, 20), (40, 30), (80, 30)]


def main():
    problems = [(ns, pp, make_problem(ns, pp)) for ns, pp in CASES]
    headers = [f"{ns}seg x {pp}pt" for ns, pp, _ in problems]
    print(f"{'method':18s} | " + " | ".join(f"{h:>15s}" for h in headers))
    print("-" * 90)

    for name, method in METHODS.items():
        row = [f"{name:18s}"]
        for ns, ppc, prob in problems:
            pp, border, seg = prob
            sl = seg if ("Segment" in name or name == "Segments") else np.ones_like(seg)
            # PythonTsp has quadratic memory and explodes on large point sets;
            # skip beyond a safe bound rather than wait minutes.
            if name == "PythonTsp" and ns * ppc > 800:
                row.append(f"{'(skip)':>15s}")
                continue
            try:
                t = time_call(lambda: method._make(pp, border, sl), repeat=2)
                row.append(f"{t * 1000:12.1f} ms")
            except Exception:
                row.append(f"{'ERR':>15s}")
        print(" | ".join(row))


if __name__ == "__main__":
    main()
