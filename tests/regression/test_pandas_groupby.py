# Regression tests for pandas 3.0 compatibility: groupby.apply with include_groups
# https://github.com/eisenforschung/landau/pull/64
import numpy as np
import pytest

from landau.phases import LinePhase, IdealSolution
from landau.calculate import calc_phase_diagram


@pytest.fixture
def two_phase_system():
    l1 = LinePhase("A", 0, 0, 0)
    l2 = LinePhase("B", 1, 0.1, 0)
    sol = IdealSolution("sol", l1, l2)
    return [l1, l2, sol]


def test_single_T_multiple_mu(two_phase_system):
    """calc_phase_diagram with a single T and multiple mu values must not raise ValueError.

    Previously crashed with pandas 3.0 due to include_groups=True in groupby.apply.
    """
    pdf = calc_phase_diagram(two_phase_system, Ts=1000.0, mu=np.linspace(-0.2, 0.2, 10))
    assert len(pdf) > 0
    assert "phase" in pdf.columns
    assert "c" in pdf.columns


def test_multiple_T_single_mu(two_phase_system):
    """calc_phase_diagram with multiple T values and a single mu must not raise ValueError.

    Previously crashed with pandas 3.0 due to include_groups=True in groupby.apply.
    """
    pdf = calc_phase_diagram(two_phase_system, Ts=np.linspace(100, 2000, 10), mu=0.05)
    assert len(pdf) > 0
    assert "phase" in pdf.columns
    assert "c" in pdf.columns
