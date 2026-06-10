"""Regression tests for the f_excess endpoint reference with near-endpoint samples.

Follow-up to #155.  The per-phase endpoint selection introduced there compared
candidate reference phases by f at their extreme sampled concentration.  A
solution phase whose last sample lands at c=1-eps (inside the 1% qualification
band, e.g. because the mu window does not push it all the way to saturation)
has f below its own value at c=1 by ~mu*eps, since f is convex with slope mu.
Comparing raw f therefore let such a phase steal the c=1 reference from a line
phase sitting exactly at c=1, lifting the f_excess of the actually stable
endpoint phase off zero.

The fix compares tangent-line values at the pure concentrations instead: phi
is the c=0 value of the tangent at a sample (c, f) with slope mu, phi+mu the
c=1 value.  Both reduce to f for a line phase located exactly at an endpoint.

Setup: ideal-solution liquid between f_A=0 and f_B=0.05 eV at T=800 K, with a
mu window stopping the liquid near c=0.995, and a line phase at c=1 placed
between the liquid's last-sample f (~0.0477 eV) and its true endpoint value
(0.05 eV).  Before the fix the line phase had f_excess ~ +3e-4 despite being
the stable phase at c=1.
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
    # window chosen so the liquid's largest sample is ~0.995: inside the 1%
    # qualification band but short of c=1
    mu = np.linspace(-0.5, 0.42, 60)
    return calc_phase_diagram([liquid, bcc], Ts=[800.0], mu=mu, keep_unstable=True, refine=False)


def test_stable_endpoint_line_phase_is_reference(liquid):
    """bcc at c=1 below the liquid's f(1) is the reference; its f_excess is exactly 0."""
    df = diagram(liquid, bcc_energy=0.048)
    liq_c_max = df.query("phase=='liquid'").c.max()
    assert 0.99 < liq_c_max < 1, "liquid must stop inside the qualification band for this test"
    bcc = df.query("phase=='bcc'")
    assert bcc.stable.any()
    np.testing.assert_allclose(bcc.f_excess, 0, atol=ATOL)


def test_metastable_endpoint_line_phase_above_reference(liquid):
    """bcc at c=1 above the liquid's f(1)=0.05 gets f_excess of its 0.01 eV gap.

    The liquid reference is its tangent value at c=1 taken from the last
    sample, which underestimates f(1) by the convexity error, so the gap comes
    out slightly above 0.01 eV.
    """
    df = diagram(liquid, bcc_energy=0.06)
    fex = df.query("phase=='bcc'").f_excess
    assert (fex > 0.01).all()
    assert (fex < 0.011).all()
