"""Compare interpolator wall-time and fit quality on a handful of
representative 1-D datasets.

Run with ``python benchmarks/bench_interpolators.py`` from the repository
root.  Requires the ``[test,constraints]`` extras to be installed.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from landau.interpolate import PolyFit, RedlichKister, SGTE, SoftplusFit


def _softplus(t):
    t = np.asarray(t, float)
    return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)


# ---------- datasets ----------

def case_smooth_alloy_mixing(n: int = 121):
    """RK-shaped mixing free energy plus a small linear term."""
    c = np.concatenate([[0.0], np.linspace(1e-3, 1 - 1e-3, n - 2), [1.0]])
    L = [-2.5, 0.6, -0.3]
    pre = c * (1 - c)
    f = pre * sum(Li * (2 * c - 1) ** i for i, Li in enumerate(L)) + 0.3 + 0.1 * c
    return c, f


def case_two_softpluses(n: int = 201):
    """The PR #82 historical pathological case for SoftplusFit."""
    x = np.linspace(-1.0, 1.0, n)
    p = [2.0, 2.25, 0.0, 2.0, -2.0, 1.0, 0.0]
    y = (
        p[-1]
        + p[0] * _softplus(p[1] * (x + p[2]))
        + p[3] * _softplus(p[4] * (x + p[5]))
    )
    return x, y


def case_sgte_like(n: int = 80):
    """Calphad-flavoured Gibbs energy on temperature."""
    T = np.linspace(300.0, 2000.0, n)
    G = -25.0 - 0.012 * T + 1e-5 * T ** 2 - T * np.log(T) * 1e-4
    return T, G


# ---------- benchmark machinery ----------

@dataclass
class Result:
    case: str
    method: str
    time_ms: float
    rms: float


def bench(fn: Callable[[], Callable], repeats: int = 3) -> tuple[float, Callable]:
    """Return median wall time (seconds) and an example output."""
    # warm-up
    out = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def rms(predict: Callable, x, y) -> float:
    try:
        return float(np.sqrt(np.mean((predict(x) - y) ** 2)))
    except Exception:
        return float("nan")


def run() -> list[Result]:
    cases = {
        "smooth_alloy_mixing (121 pts)": case_smooth_alloy_mixing(),
        "two_softpluses     (201 pts)": case_two_softpluses(),
        "sgte_like          (80 pts) ": case_sgte_like(),
    }
    results: list[Result] = []
    for name, (x, y) in cases.items():
        methods = [
            ("PolyFit(4)",                  lambda x=x, y=y: PolyFit(4).fit(x, y)),
            ("PolyFit('auto')",             lambda x=x, y=y: PolyFit("auto").fit(x, y)),
            ("SGTE(4)",                     lambda x=x, y=y: SGTE(4).fit(x, y)),
            ("RedlichKister(3)",            lambda x=x, y=y: RedlichKister(3).fit(x, y)),
            ("SoftplusFit(n=2).fit",        lambda x=x, y=y: SoftplusFit(n_softplus=2, max_nfev=200).fit(x, y)),
            ("SoftplusFit(n=2,linear).fit", lambda x=x, y=y: SoftplusFit(n_softplus=2, loss="linear", max_nfev=200).fit(x, y)),
            ("SoftplusFit(n=2).global_fit", lambda x=x, y=y: SoftplusFit(n_softplus=2, max_nfev=200).global_fit(x, y)),
        ]
        for label, fn in methods:
            try:
                t, pred = bench(fn)
            except Exception as exc:
                results.append(Result(name, label, float("nan"), float("nan")))
                print(f"  {label:40s}: skipped ({type(exc).__name__})")
                continue
            r = rms(pred, x, y)
            results.append(Result(name, label, t * 1e3, r))
    return results


def print_table(results: list[Result]) -> None:
    width = max(len(r.case) for r in results) + 2
    header = f"{'case':{width}s}{'method':40s}{'time (ms)':>12s}{'rms':>14s}"
    print(header)
    print("-" * len(header))
    last_case = None
    for r in results:
        case = r.case if r.case != last_case else ""
        last_case = r.case
        t = f"{r.time_ms:.2f}" if not np.isnan(r.time_ms) else "—"
        rms_s = f"{r.rms:.3e}" if not np.isnan(r.rms) else "—"
        print(f"{case:{width}s}{r.method:40s}{t:>12s}{rms_s:>14s}")


if __name__ == "__main__":
    results = run()
    print()
    print_table(results)
