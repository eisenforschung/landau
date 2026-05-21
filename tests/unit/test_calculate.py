import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from landau.calculate import (
    find_one_point,
    cluster_T_c,
    cluster_T_c_mu,
    _join_phase_unit,
    _split_phase_unit,
)
from unittest.mock import MagicMock

def test_find_one_point_success():
    """
    Test finding a root when it clearly exists within the bracket.
    """
    phase1 = MagicMock()
    phase2 = MagicMock()

    # Let potential(phase1, x) = 2x
    # Let potential(phase2, x) = x + 3
    # The root of potential(phase1, x) - potential(phase2, x) = 2x - (x + 3) = x - 3 = 0 is x = 3
    def mock_potential(phase, x):
        if phase is phase1:
            return 2 * x
        elif phase is phase2:
            return x + 3
        return 0

    root = find_one_point(phase1, phase2, mock_potential, (0, 10))
    assert np.isclose(root, 3.0, atol=1e-5)


def test_find_one_point_no_root():
    """
    Test the behavior when no root exists within the bracket.
    scipy.optimize.root_scalar should raise ValueError because f(a) and f(b) must have different signs.
    """
    phase1 = MagicMock()
    phase2 = MagicMock()

    # Let potential(phase1, x) = x
    # Let potential(phase2, x) = 0
    # The root of x = 0 is 0.
    def mock_potential(phase, x):
        if phase is phase1:
            return x
        elif phase is phase2:
            return 0
        return 0

    # Search in interval (1, 10) where there is no root and signs are the same (all positive)
    with pytest.raises(ValueError, match=r"f\(a\) and f\(b\) must have different signs"):
        find_one_point(phase1, phase2, mock_potential, (1, 10))


# --- cluster_T_c tests ---

def test_cluster_T_c_single_blob():
    # 10-point diagonal stripe: consecutive distance ~0.11 < distance_threshold=0.5
    # so single linkage chains all points into one cluster
    dd = pd.DataFrame({
        "T": np.linspace(300, 400, 10),
        "c": np.linspace(0.0, 0.2, 10),
    })
    labels = cluster_T_c(dd, distance_threshold=0.5)
    assert labels.nunique() == 1


def test_cluster_T_c_two_blobs():
    # Two diagonal stripes separated by a gap > 0.5 in c
    T = np.concatenate([np.linspace(300, 400, 10), np.linspace(300, 400, 10)])
    c = np.concatenate([np.linspace(0.0, 0.1, 10), np.linspace(0.7, 0.8, 10)])
    labels = cluster_T_c(pd.DataFrame({"T": T, "c": c}), distance_threshold=0.5)
    assert labels.nunique() == 2


def test_cluster_T_c_empty():
    dd = pd.DataFrame({"T": [], "c": []})
    labels = cluster_T_c(dd, distance_threshold=0.5)
    assert len(labels) == 0
    assert labels.dtype.kind == "i"


# --- cluster_T_c_mu tests ---

def test_cluster_T_c_mu_inf_get_own_labels():
    dd = pd.DataFrame({
        "T": [300, 400, 300, 400],
        "c": [0.0, 0.0, 1.0, 1.0],
        "mu": [-np.inf, -np.inf, np.inf, np.inf],
    })
    labels = cluster_T_c_mu(dd, distance_threshold=0.5)
    pos_label = labels.loc[dd["mu"] == +np.inf].iloc[0]
    neg_label = labels.loc[dd["mu"] == -np.inf].iloc[0]
    assert labels.loc[dd["mu"] == +np.inf].nunique() == 1
    assert labels.loc[dd["mu"] == -np.inf].nunique() == 1
    assert pos_label != neg_label


def test_cluster_T_c_mu_degenerate_T():
    dd = pd.DataFrame({
        "T": [300, 300, 300, 300],
        "c": [0.1, 0.2, 0.8, 0.9],
        "mu": [-0.1, 0.0, 0.0, 0.1],
    })
    labels = cluster_T_c_mu(dd, distance_threshold=0.5)
    assert len(labels) == 4


def test_cluster_T_c_mu_empty():
    dd = pd.DataFrame({"T": [], "c": [], "mu": []})
    labels = cluster_T_c_mu(dd, distance_threshold=0.5)
    assert len(labels) == 0
    assert labels.dtype.kind == "i"


# --- _join_phase_unit / _split_phase_unit tests ---


def test_join_phase_unit_basic():
    phase = pd.Series(["alpha", "beta", "gamma"])
    unit = pd.Series([0, 1, 2])
    out = _join_phase_unit(phase, unit)
    assert out.tolist() == ["alpha_0", "beta_1", "gamma_2"]


def test_split_phase_unit_basic():
    combined = pd.Series(["alpha_0", "beta_1", "gamma_2"])
    phase, unit = _split_phase_unit(combined)
    assert phase.tolist() == ["alpha", "beta", "gamma"]
    assert unit.tolist() == [0, 1, 2]
    assert unit.map(type).eq(int).all()


@given(
    phase=st.lists(
        st.text(alphabet=st.characters(blacklist_characters="_"), min_size=1, max_size=8),
        min_size=1,
        max_size=10,
    ),
    unit=st.lists(st.integers(min_value=0, max_value=99), min_size=1, max_size=10),
)
def test_phase_unit_round_trip(phase, unit):
    n = min(len(phase), len(unit))
    p = pd.Series(phase[:n])
    u = pd.Series(unit[:n])
    p_out, u_out = _split_phase_unit(_join_phase_unit(p, u))
    pd.testing.assert_series_equal(p_out, p, check_names=False)
    pd.testing.assert_series_equal(u_out, u, check_names=False)


def test_phase_unit_round_trip_underscore_in_name():
    # rsplit(n=1) preserves underscores inside the phase name; only the final
    # `_<int>` segment is interpreted as the unit.
    p = pd.Series(["foo_bar", "baz_42"])
    u = pd.Series([3, 7])
    p_out, u_out = _split_phase_unit(_join_phase_unit(p, u))
    assert p_out.tolist() == ["foo_bar", "baz_42"]
    assert u_out.tolist() == [3, 7]


def test_split_phase_unit_rejects_non_integer_unit():
    with pytest.raises(ValueError):
        _split_phase_unit(pd.Series(["alpha_not_an_int"]))
