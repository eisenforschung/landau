import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from landau.calculate import (
    calc_phase_diagram,
    find_one_point,
    cluster,
    cluster_T_c,
    cluster_T_c_mu,
    reduce,
    _apply_series,
    _border_edges,
    _join_phase_unit,
    _split_phase_unit,
    _split_stable,
    guess_mu_range,
)
from landau.features import Locus
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


# --- _apply_series tests ---


def test_apply_series_multi_group_returns_row_indexed_series():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 10.0, 20.0]})
    out = _apply_series(df.groupby("g", group_keys=False), lambda d: d["x"] * 2, "y")
    assert isinstance(out, pd.Series)
    pd.testing.assert_series_equal(
        out.sort_index(),
        pd.Series([2.0, 4.0, 20.0, 40.0], name="y"),
        check_dtype=False,
    )


def test_apply_series_single_group_returns_row_indexed_series():
    # The whole point of the helper: pandas 3 returns a one-row DataFrame
    # from groupby.apply when only one group exists and the callable returns
    # a Series.  The helper must yield a row-aligned Series regardless.
    df = pd.DataFrame({"g": ["a", "a", "a"], "x": [1.0, 2.0, 3.0]})
    out = _apply_series(df.groupby("g", group_keys=False), lambda d: d["x"] + 1, "y")
    assert isinstance(out, pd.Series)
    pd.testing.assert_series_equal(
        out.sort_index(),
        pd.Series([2.0, 3.0, 4.0], name="y"),
        check_dtype=False,
    )


def test_apply_series_aligns_to_assignment():
    # Round-trip via DataFrame column assignment is the primary call shape in
    # calc_phase_diagram / get_transitions; the returned Series must align by
    # the original row index.
    df = pd.DataFrame({"g": ["a", "b", "a", "b"], "x": [1, 2, 3, 4]}, index=[10, 20, 30, 40])
    df["y"] = _apply_series(df.groupby("g", group_keys=False), lambda d: d["x"] * 10, "y")
    assert df.loc[10, "y"] == 10
    assert df.loc[20, "y"] == 20
    assert df.loc[30, "y"] == 30
    assert df.loc[40, "y"] == 40


def test_apply_series_callable_does_not_see_group_column():
    # include_groups=False contract: the callable must not receive the group
    # key column.  Asserting this here is what lets the call sites stay terse.
    seen_columns = []

    def fn(d):
        seen_columns.append(list(d.columns))
        return d["x"]

    df = pd.DataFrame({"g": ["a", "a", "b"], "x": [1, 2, 3]})
    _apply_series(df.groupby("g", group_keys=False), fn, "y")
    for cols in seen_columns:
        assert "g" not in cols


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


# --- guess_mu_range tests ---

from landau.phases import LinePhase, IdealSolution


_GMR_SAMPLES = 50


def _ideal_phases():
    """Two terminal phases + IdealSolution; transition near dmu=0."""
    a = LinePhase("A", 0, 0, 0)
    b = LinePhase("B", 1, 0, 0)
    sol = IdealSolution("sol", a, b)
    return [a, b, sol]


def test_guess_mu_range_returns_samples():
    mus, c0, c1 = guess_mu_range(_ideal_phases(), T=1000, samples=_GMR_SAMPLES)
    assert len(mus) == _GMR_SAMPLES


def test_guess_mu_range_covers_concentration_space():
    mus, c0, c1 = guess_mu_range(_ideal_phases(), T=1000, samples=_GMR_SAMPLES)
    assert c0 < 0.1
    assert c1 > 0.9


def test_guess_mu_range_low_temperature():
    # At T=10 K, c(mu) is a step function.  Before the fix, BFGS starting
    # from mu=0 would not move and the function would raise or loop forever.
    mus, c0, c1 = guess_mu_range(_ideal_phases(), T=10, samples=_GMR_SAMPLES)
    assert len(mus) == _GMR_SAMPLES
    assert c0 < 0.5
    assert c1 > 0.5


def test_guess_mu_range_mu_not_centred_at_zero():
    # Phases whose transition lies far from mu=0; old x0=[0] seed would return
    # in a flat region of c(mu) at low T.
    shift = 5.0
    a = LinePhase("A", 0, -shift, 0)
    b = LinePhase("B", 1, 0, 0)
    sol = IdealSolution("sol", a, b)
    mus, c0, c1 = guess_mu_range([a, b, sol], T=100, samples=_GMR_SAMPLES)
    assert len(mus) == _GMR_SAMPLES
    assert c0 < 0.5
    assert c1 > 0.5


def test_guess_mu_range_degenerate_raises():
    # All-same-concentration phases should raise.
    a = LinePhase("A", 0.5, 0, 0)
    b = LinePhase("B", 0.5, 0.1, 0)
    with pytest.raises(ValueError):
        guess_mu_range([a, b], T=1000, samples=_GMR_SAMPLES)


# --- _split_stable tests ---


@pytest.fixture
def stable_unstable_frame():
    return pd.DataFrame(
        {
            "T": [300.0, 400.0, 500.0, 600.0],
            "mu": [0.0, 0.1, 0.2, 0.3],
            "c":  [0.1, 0.2, 0.3, 0.4],
            "phase": ["A", "B", "A", "B"],
            "stable": [True, False, True, False],
        }
    )


def test_split_stable_partitions_by_stable_flag(stable_unstable_frame):
    df = stable_unstable_frame
    sdf, udf = _split_stable(df)
    assert sdf["stable"].all()
    assert not udf["stable"].any()
    assert len(sdf) + len(udf) == len(df)


def test_split_stable_resets_index(stable_unstable_frame):
    df = stable_unstable_frame.iloc[[3, 2, 1, 0]]  # shuffle source index
    sdf, udf = _split_stable(df)
    assert list(sdf.index) == list(range(len(sdf)))
    assert list(udf.index) == list(range(len(udf)))


def test_split_stable_adds_border_and_refined_columns(stable_unstable_frame):
    df = stable_unstable_frame
    sdf, udf = _split_stable(df)
    # both halves get a border=False column
    assert (sdf["border"] == False).all()  # noqa: E712
    assert (udf["border"] == False).all()  # noqa: E712
    # only the stable half gets refined="no"
    assert (sdf["refined"] == "no").all()
    assert "refined" not in udf.columns
    # plain samples in both halves are interior points
    assert (sdf["locus"] == Locus.INTERIOR).all()
    assert (udf["locus"] == Locus.INTERIOR).all()


def test_split_stable_does_not_mutate_input(stable_unstable_frame):
    df = stable_unstable_frame
    before = df.copy()
    _split_stable(df)
    pd.testing.assert_frame_equal(df, before)


# --- _border_edges tests ---


@pytest.fixture
def grid_frame():
    """A small (T, mu) grid with two phases; mimics the post-_split_stable
    `sdf` that `refine_phase_diagram` feeds into `_border_edges`."""
    Ts = [300.0, 400.0, 500.0]
    mus = [-0.5, 0.0, 0.5]
    rows = []
    for T in Ts:
        for mu in mus:
            rows.append({"T": T, "mu": mu, "c": 0.5, "phase": "A", "border": False})
    return pd.DataFrame(rows)


def test_border_edges_marks_T_extremes_in_place(grid_frame):
    df = grid_frame
    _border_edges(df, min_c=0.0, max_c=1.0)
    assert df.loc[df["T"] == 300.0, "border"].all()
    assert df.loc[df["T"] == 500.0, "border"].all()
    assert not df.loc[df["T"] == 400.0, "border"].any()


def test_border_edges_left_right_use_extreme_mu_rows(grid_frame):
    df = grid_frame
    left, right = _border_edges(df, min_c=0.05, max_c=0.95)
    # one synthetic row per (phase, T) at each mu extreme — here one phase × 3 Ts
    assert len(left) == 3
    assert len(right) == 3
    assert (left["mu"] == -np.inf).all()
    assert (right["mu"] == +np.inf).all()
    assert (left["c"] == 0.05).all()
    assert (right["c"] == 0.95).all()
    assert (left["border"] == True).all()  # noqa: E712
    assert (right["border"] == True).all()  # noqa: E712
    assert (left["stable"] == True).all()  # noqa: E712
    assert (right["stable"] == True).all()  # noqa: E712
    # frame edges only close the sampling window, they are no transitions
    assert (left["locus"] == Locus.INTERIOR).all()
    assert (right["locus"] == Locus.INTERIOR).all()


def test_border_edges_preserves_T_and_phase_from_source(grid_frame):
    df = grid_frame
    left, right = _border_edges(df, min_c=0.0, max_c=1.0)
    # the source rows at the extreme mu values are at all three Ts, single phase A
    assert sorted(left["T"].tolist()) == [300.0, 400.0, 500.0]
    assert sorted(right["T"].tolist()) == [300.0, 400.0, 500.0]
    assert (left["phase"] == "A").all()
    assert (right["phase"] == "A").all()


# --- reduce tests ---


def test_reduce_joins_phase_names_sorted_by_c():
    # rows deliberately out of c order; transition string must reflect ascending-c sort
    dd = pd.DataFrame({"phase": ["liq", "fcc", "bcc"], "c": [0.7, 0.2, 0.5]})
    phases_by_c = dd.sort_values("c")["phase"].tolist()
    out = reduce(dd)
    assert out["transition"] == "-".join(phases_by_c)
    assert out["c"] == sorted(dd["c"].tolist())
    assert out["phase"] == phases_by_c


def test_reduce_single_phase_no_dash():
    dd = pd.DataFrame({"phase": ["fcc"], "c": [0.3]})
    out = reduce(dd)
    assert out["transition"] == "fcc"
    assert out["c"] == [0.3]


# --- cluster dispatcher tests ---


def test_cluster_use_mu_true_dispatches_to_cluster_T_c_mu():
    # mu=+inf gets its own label only in the cluster_T_c_mu branch
    dd = pd.DataFrame({"T": [300.0, 400.0], "c": [0.1, 0.2], "mu": [np.inf, 0.0]})
    labels = cluster(dd, distance_threshold=0.5)
    assert labels.loc[dd["mu"] == np.inf].nunique() == 1
    assert labels.loc[dd["mu"] == np.inf].iloc[0] != labels.loc[dd["mu"] == 0.0].iloc[0]


def test_cluster_use_mu_false_dispatches_to_cluster_T_c():
    # When use_mu=False, the function must work on a frame without a mu column
    dd = pd.DataFrame({
        "T": np.linspace(300, 400, 10),
        "c": np.linspace(0.0, 0.2, 10),
    })
    labels = cluster(dd, use_mu=False, distance_threshold=0.5)
    assert labels.nunique() == 1
    assert len(labels) == len(dd)


# --- locus column tests ---


@pytest.fixture
def triple_point_phases():
    """Three LinePhases with a single triple point at (T=300 K, mu=0.2 eV)."""
    return [
        LinePhase("A", 0.0, -1.0, 0.004),
        LinePhase("B", 0.5, -1.2, 0.003),
        LinePhase("C", 1.0, -1.7, 0.001),
    ]


def test_calc_phase_diagram_unrefined_locus_all_interior(two_phase_ideal):
    df = calc_phase_diagram(two_phase_ideal, Ts=np.linspace(300, 1000, 5),
                            mu=np.linspace(-0.5, 0.5, 5), refine=False, keep_unstable=True)
    assert (df["locus"] == Locus.INTERIOR).all()


def test_calc_phase_diagram_refined_locus(triple_point_phases):
    df = calc_phase_diagram(triple_point_phases, Ts=np.linspace(220, 480, 12),
                            mu=np.linspace(-0.05, 0.55, 15), keep_unstable=True)
    assert not df["locus"].isna().any()
    assert set(map(str, df["locus"])) <= {"interior", "boundary", "triple"}
    assert (df["locus"] == Locus.BOUNDARY).any()
    # refined rows are exactly the non-interior ones
    refined = df["refined"].fillna("no") != "no"
    assert (df.loc[refined, "locus"] != Locus.INTERIOR).all()
    assert (df.loc[~refined, "locus"] == Locus.INTERIOR).all()
    # frame edges are border=True but still interior
    edges = df["mu"].abs() == np.inf
    assert edges.any()
    assert df.loc[edges, "border"].all()
    assert (df.loc[edges, "locus"] == Locus.INTERIOR).all()


def test_calc_phase_diagram_locus_triple(triple_point_phases):
    df = calc_phase_diagram(triple_point_phases, Ts=np.linspace(220, 480, 12),
                            mu=np.linspace(-0.05, 0.55, 15))
    triple = df[df["locus"] == Locus.TRIPLE]
    assert not triple.empty
    # every triple point consists of three coexisting phases at one (T, mu)
    assert (triple.groupby(["T", "mu"])["phase"].nunique() == 3).all()
    assert np.allclose(triple["T"], 300.0, atol=10.0)
    assert np.allclose(triple["mu"], 0.2, atol=0.05)
