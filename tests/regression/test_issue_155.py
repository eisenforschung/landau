"""Regression tests for #155.

When two solution phases both sample near c=0, the old code selected the phase
with the globally smallest sampled concentration as the f_excess reference, even
when a different phase has a lower free energy there.  The fix evaluates each
phase at its own endpoint concentration and takes the thermodynamically stable one
as the reference.

Fixture setup (T=500 K, no vibrational entropy):
  solid:  f_a=-2.0 eV, f_b=-3.2 eV  (Δf=-1.2 eV)  → solid is more stable at c=0
  liquid: f_a=-1.5 eV, f_b=-2.5 eV  (Δf=-1.0 eV)  → Δf_liquid > Δf_solid

Because |Δf_liquid| < |Δf_solid|, at any fixed negative mu liquid concentrates
more strongly toward c=0 and samples a smaller c than solid (original bug: c_liquid
≈ 2e-6 vs c_solid ≈ 5e-6).  The old code therefore picked liquid as the c=0
reference, giving solid a spurious f_excess ≈ -0.5 eV at c≈0 (thermodynamically
impossible for the reference phase).  The correct value is ≈ kT*(mixing entropy) ≈ 0.
"""

import numpy as np
import pytest
from landau.calculate import calc_phase_diagram
from landau.phases import LinePhase, IdealSolution


@pytest.fixture
def two_phase_system():
    """Solid + liquid where solid is stable at c=0 but liquid samples smaller c."""
    solid_a = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=0)
    solid_b = LinePhase("B", fixed_concentration=1, line_energy=-3.2, line_entropy=0)
    solid = IdealSolution("solid", solid_a, solid_b)

    liquid_a = LinePhase("A(l)", fixed_concentration=0, line_energy=-1.5, line_entropy=0)
    liquid_b = LinePhase("B(l)", fixed_concentration=1, line_energy=-2.5, line_entropy=0)
    liquid = IdealSolution("liquid", liquid_a, liquid_b)

    return solid, liquid


@pytest.fixture
def phase_diagram(two_phase_system):
    solid, liquid = two_phase_system
    mu_range = np.linspace(-2.0, 2.0, 50)
    df = calc_phase_diagram(
        [solid, liquid],
        Ts=[500.0, 501.0],
        mu=mu_range,
        keep_unstable=True,
        refine=False,
    )
    return df[df["T"] == 500.0]


def test_stable_phase_at_c0_has_near_zero_f_excess(phase_diagram):
    """solid is the stable reference at c=0; its f_excess should be ~0 there.

    The old bug gave solid f_excess ≈ -0.5 eV (solid was penalised against
    liquid which was wrongly chosen as the reference).  The correct value is
    kT*(mixing entropy) ≈ 0.
    """
    near_c0 = phase_diagram[phase_diagram["c"] < 0.01]
    solid_fex = near_c0[near_c0["phase"] == "solid"]["f_excess"]

    assert len(solid_fex) > 0, "solid should sample near c=0"
    # kT*mixing at c=0.01 ≈ -0.0024 eV; threshold -0.1 eV clearly separates
    # the fix (≈ 0) from the old bug (≈ -0.5 eV)
    assert solid_fex.min() > -0.1, (
        f"solid f_excess at c<0.01 should be ~0 (kT*mixing only), got {solid_fex.min():.4f} eV; "
        f"the old bug would give ≈ -0.5 eV"
    )


def test_metastable_phase_at_c0_has_positive_f_excess(phase_diagram):
    """liquid is less stable at c=0 than the reference (solid); f_excess must be > 0."""
    near_c0 = phase_diagram[phase_diagram["c"] < 0.01]
    liquid_fex = near_c0[near_c0["phase"] == "liquid"]["f_excess"]

    assert len(liquid_fex) > 0, "liquid should sample near c=0"
    # The energy gap is 0.5 eV; expect f_excess close to 0.5 eV there
    assert liquid_fex.min() > 0, (
        f"liquid is metastable at c≈0; its f_excess must be > 0, got {liquid_fex.min():.4f} eV"
    )


def test_metastable_phase_f_excess_exceeds_stable_at_c0(phase_diagram):
    """At c≈0, liquid (metastable) must have higher f_excess than solid (reference)."""
    near_c0 = phase_diagram[phase_diagram["c"] < 0.01]
    solid_fex = near_c0[near_c0["phase"] == "solid"]["f_excess"]
    liquid_fex = near_c0[near_c0["phase"] == "liquid"]["f_excess"]

    assert len(solid_fex) > 0 and len(liquid_fex) > 0
    assert liquid_fex.min() > solid_fex.max(), (
        f"liquid f_excess ({liquid_fex.min():.4f}) should exceed solid f_excess "
        f"({solid_fex.max():.4f}) near c=0"
    )


def test_line_phase_near_endpoint_not_used_as_reference():
    """A line phase fixed at c=0.08 must not corrupt the c=0 reference.

    The endpoint threshold is 1% of the full concentration span.  A LinePhase
    at c=0.08 has c_min=0.08 > lo_thr=0.01, so it is excluded as a c=0
    reference even when its line_energy=-5.0 is lower than the solution phase
    endpoint energy.  f_excess(solid) must therefore be ≈0 at c≈0.
    """
    solid_a = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=0)
    solid_b = LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=0)
    solid = IdealSolution("solid", solid_a, solid_b)

    im = LinePhase("IM", fixed_concentration=0.08, line_energy=-5.0, line_entropy=0)

    mu_range = np.linspace(-2.0, 2.0, 50)
    df = calc_phase_diagram(
        [solid, im],
        Ts=[500.0, 501.0],
        mu=mu_range,
        keep_unstable=True,
        refine=False,
    )
    df_T = df[df["T"] == 500.0]
    near_c0 = df_T[df_T["c"] < 0.01]
    solid_fex = near_c0[near_c0["phase"] == "solid"]["f_excess"]

    assert len(solid_fex) > 0, "solid should sample near c=0"
    assert solid_fex.min() < 0.1, (
        f"solid f_excess at c<0.01 should be ~0; got {solid_fex.min():.4f} eV — "
        f"the IM at c=0.08 may have been used as the c=0 reference (f_IM=-5 eV)"
    )
