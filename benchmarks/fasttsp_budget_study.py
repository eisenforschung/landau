"""Budget sweep study for FastTsp and SegmentFastTsp.

Measures how polygon reconstruction quality degrades as duration_seconds is
reduced.  Quality is expressed as:

  - area_ratio : polygon area / ideal area (unit circle ≈ π).  Values near 1
                 mean the tour correctly encloses the region; values far from 1
                 indicate a broken / self-intersecting tour.
  - valid      : shapely reports the polygon as topologically valid.
  - perimeter  : polygon perimeter (lower is better for a convex shape; a
                 "bad" tour zig-zags and inflates this).

The synthetic problem is the same ring-shaped region used in poly_methods.py.

Run: ``python benchmarks/fasttsp_budget_study.py``
"""

import math
import time

import numpy as np
import shapely

from landau.poly import FastTsp, SegmentFastTsp

IDEAL_AREA = math.pi  # unit circle area
IDEAL_PERIMETER = 2 * math.pi  # unit circle perimeter


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


def eval_polygon(shape):
    # _make returns a shapely geometry (Polygon/LineString/None), not a mpl Polygon
    if shape is None or shape.is_empty:
        return {"area_ratio": 0.0, "valid": False, "perimeter_ratio": float("inf")}
    if not isinstance(shape, shapely.Polygon):
        return {"area_ratio": 0.0, "valid": False, "perimeter_ratio": float("inf")}
    return {
        "area_ratio": shape.area / IDEAL_AREA,
        "valid": shape.is_valid,
        "perimeter_ratio": shape.length / IDEAL_PERIMETER,
    }


def time_call(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), result


BUDGETS = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]

CASES = [
    ("20seg×20pt",  20, 20),
    ("40seg×30pt",  40, 30),
    ("80seg×30pt",  80, 30),
]


def run_case(n_segments, pts_per_segment):
    pp, border, seg = make_problem(n_segments, pts_per_segment)
    results = {}
    for budget in BUDGETS:
        ft = FastTsp(duration_seconds=budget)
        sft = SegmentFastTsp(duration_seconds=budget)

        t_ft, poly_ft = time_call(lambda m=ft: m._make(pp, border, seg))
        t_sft, poly_sft = time_call(lambda m=sft: m._make(pp, border, seg))

        results[budget] = {
            "fasttsp":        {"ms": t_ft * 1000,  **eval_polygon(poly_ft)},
            "segment_fasttsp": {"ms": t_sft * 1000, **eval_polygon(poly_sft)},
        }
    return results


def print_table(case_label, results):
    header = (
        f"\n### {case_label}\n"
        f"{'budget':>10s} | "
        f"{'FT ms':>8s} {'FT area':>8s} {'FT ok':>5s} | "
        f"{'SFT ms':>8s} {'SFT area':>8s} {'SFT ok':>5s}"
    )
    sep = "-" * 70
    print(header)
    print(sep)
    for budget in BUDGETS:
        r = results[budget]
        ft = r["fasttsp"]
        sft = r["segment_fasttsp"]
        print(
            f"{budget:>10.4f} | "
            f"{ft['ms']:>7.1f}  {ft['area_ratio']:>7.3f}  {'Y' if ft['valid'] else 'N':>5s} | "
            f"{sft['ms']:>7.1f}  {sft['area_ratio']:>7.3f}  {'Y' if sft['valid'] else 'N':>5s}"
        )


def main():
    print("FastTsp / SegmentFastTsp — compute budget vs quality sweep")
    print("area_ratio: polygon area / π  (1.0 = perfect unit circle)")
    print("ok: polygon is topologically valid (no self-intersections)")
    for label, ns, pp in CASES:
        print(f"\nRunning {label} ...", flush=True)
        results = run_case(ns, pp)
        print_table(label, results)

    print("\n\nMarkdown summary table (averaged across cases):")
    # Collect all results
    all_results = {}
    for label, ns, pp in CASES:
        all_results[(ns, pp)] = run_case(ns, pp)

    print(f"\n| budget (s) | FT ms (80×30) | FT area | FT valid | SFT ms (80×30) | SFT area | SFT valid |")
    print("|-----------|:-------------:|:-------:|:--------:|:--------------:|:--------:|:---------:|")
    ns, pp = 80, 30
    for budget in BUDGETS:
        r = all_results[(ns, pp)][budget]
        ft = r["fasttsp"]
        sft = r["segment_fasttsp"]
        print(
            f"| {budget:.4f} | {ft['ms']:.1f} | {ft['area_ratio']:.3f} | {'✓' if ft['valid'] else '✗'} "
            f"| {sft['ms']:.1f} | {sft['area_ratio']:.3f} | {'✓' if sft['valid'] else '✗'} |"
        )


if __name__ == "__main__":
    main()
