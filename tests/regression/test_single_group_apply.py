"""Regression tests for pandas 2/3 single-group groupby.apply (issue #134, #160).

Sites tested:
  - get_transitions: calculate.py:309 ``isinstance(res, pd.DataFrame)`` branch
  - AbstractPolyMethod.apply: poly.py:99 single-group dropna semantics

Also pins that a single-temperature calc_phase_diagram populates f_excess, the
site that went through a single-group groupby.apply before f_excess was
referenced against the pure components.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon

from landau.calculate import calc_phase_diagram, get_transitions
from landau.phases import IdealSolution, LinePhase
from landau.poly import Concave


def test_get_transitions_single_transition():
    """All border rows sharing one transition exercises the single-group path.

    Removing the ``isinstance(res, pd.DataFrame)`` branch at calculate.py:310
    would raise ValueError on pandas 3 when this test runs.
    """
    mus = np.linspace(-0.1, 0.1, 6)
    Ts = np.linspace(500.0, 1000.0, 6)
    rows = []
    for mu, T in zip(mus, Ts):
        rows += [
            {"mu": mu, "T": T, "c": 0.1, "phase": "A", "border": True},
            {"mu": mu, "T": T, "c": 0.9, "phase": "B", "border": True},
        ]
    df = pd.DataFrame(rows)

    out = get_transitions(df)

    assert out["transition"].eq("A-B").all(), "single-group invariant broken"
    assert "transition_unit" in out.columns
    assert out["transition_unit"].notna().all()
    assert pd.api.types.is_integer_dtype(out["transition_unit"])


def test_poly_apply_single_group():
    """Concave.apply with a single (phase, phase_unit) must return a pd.Series.

    Pins single-group behaviour of poly.py:99 groupby.apply so that
    a pandas 3 regression that collapses the result to a DataFrame is caught.
    """
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    df = pd.DataFrame({
        "c": 0.5 + 0.3 * np.cos(theta),
        "T": 700.0 + 200.0 * np.sin(theta),
        "phase": "A",
        "phase_unit": 0,
        "border": True,
    })

    result = Concave().apply(df)

    assert isinstance(result, pd.Series), "apply must return pd.Series not DataFrame"
    assert len(result) == 1
    assert isinstance(result.iloc[0], MplPolygon)


def test_calc_phase_diagram_single_T_f_excess():
    """calc_phase_diagram with a single temperature populates f_excess correctly."""
    l1 = LinePhase("A", 0, 0, 0)
    l2 = LinePhase("B", 1, 0.1, 0)
    sol = IdealSolution("sol", l1, l2)
    mu = np.linspace(-0.15, 0.15, 8)

    df = calc_phase_diagram([l1, l2, sol], Ts=[700.0], mu=mu, refine=False, keep_unstable=True)

    assert "f_excess" in df.columns
    assert len(df["f_excess"]) == len(df)
    assert pd.api.types.is_float_dtype(df["f_excess"])
    assert df["f_excess"].notna().all()
