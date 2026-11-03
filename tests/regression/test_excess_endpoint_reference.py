"""Regression tests for the f_excess endpoint reference with near-endpoint samples.

Follow-up to #155.  A solution phase whose last sample lands at c=1-eps (the mu
window does not push it all the way to saturation) must not take the c=1
reference from a line phase sitting exactly at c=1, and its own reference must
be its true pure-component free energy, not a value read off its last sample.

The reference is the lowest pure-component free energy among the phases,
evaluated from the phases themselves past the sampled mu window, so it does
not depend on how far the window reaches.

Setup: ideal-solution liquid between f_A=0 and f_B=0.05 eV at T=800 K, with a
mu window stopping the liquid near c=0.995, and a line phase at c=1 placed
either below the liquid's f(1) (stable) or above it (metastable).
"""

import numpy as np
import pytest
from landau.calculate import calc_phase_diagram
from landau.phases import LinePhase, IdealSolution

ATOL = 1e-12


@pytest.fixture(scope="module")
def liquid():
    liq_a = LinePhase("A(l)", fixed_concentration=0, line_energy=0.0, line_entropy=0)
    liq_b = LinePhase("B(l)", fixed_concentration=1, line_energy=0.05, line_entropy=0)
    return IdealSolution("liquid", liq_a, liq_b)


def diagram(liquid, bcc_energy):
    bcc = LinePhase("bcc", fixed_concentration=1.0, line_energy=bcc_energy, line_entropy=0)
    # window chosen so the liquid's largest sample is ~0.995, short of c=1
    mu = np.linspace(-0.5, 0.42, 60)
    return calc_phase_diagram([liquid, bcc], Ts=[800.0], mu=mu, keep_unstable=True, refine=False)


def test_stable_endpoint_line_phase_is_reference(liquid):
    """bcc at c=1 below the liquid's f(1) is the reference; its f_excess is exactly 0."""
    df = diagram(liquid, bcc_energy=0.048)
    liq_c_max = df.query("phase=='liquid'").c.max()
    assert 0.99 < liq_c_max < 1, "liquid must stop short of c=1 for this test"
    bcc = df.query("phase=='bcc'")
    assert bcc.stable.any()
    np.testing.assert_allclose(bcc.f_excess, 0, atol=ATOL)


def test_metastable_endpoint_line_phase_above_reference(liquid):
    """bcc at c=1 above the liquid's f(1)=0.05 gets f_excess of exactly its 0.01 eV gap."""
    df = diagram(liquid, bcc_energy=0.06)
    liq_c_max = df.query("phase=='liquid'").c.max()
    assert 0.99 < liq_c_max < 1, "liquid must stop short of c=1 for this test"
    np.testing.assert_allclose(df.query("phase=='bcc'").f_excess, 0.01, atol=ATOL)
