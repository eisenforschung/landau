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
    _f_excess_tangent_chord,
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


def test_cluster_T_c_mu_zero_finite_rows():
    # 0 finite rows: F.sum() == 0, clustering skipped entirely.
    # ids initialises to 0; +inf → m+1 = 1, -inf → m+2 = 2.
    dd = pd.DataFrame({"T": [300.0, 400.0], "c": [0.0, 1.0], "mu": [+np.inf, -np.inf]})
    labels = cluster_T_c_mu(dd, distance_threshold=0.5)
    assert labels.dtype.kind == "i"
    assert labels.loc[dd["mu"] == +np.inf].iloc[0] != labels.loc[dd["mu"] == -np.inf].iloc[0]


def test_cluster_T_c_mu_one_finite_row():
    # 1 finite row: F.sum() == 1 < 2, clustering still skipped.
    # finite row keeps label 0; +inf → 1, -inf → 2 — all three distinct.
    dd = pd.DataFrame({
        "T": [300.0, 400.0, 500.0],
        "c": [0.5, 0.0, 1.0],
        "mu": [0.0, +np.inf, -np.inf],
    })
    labels = cluster_T_c_mu(dd, distance_threshold=0.5)
    assert labels.dtype.kind == "i"
    assert labels.nunique() == 3


def test_cluster_T_c_mu_only_plus_inf_rows():
    # No -inf rows: the ids.loc[mu == -inf] = m+2 assignment is a no-op and must not raise.
    # All +inf rows share the same label (m+1 off the all-zero base).
    dd = pd.DataFrame({"T": [300.0, 400.0], "c": [0.0, 0.0], "mu": [+np.inf, +np.inf]})
    labels = cluster_T_c_mu(dd, distance_threshold=0.5)
    assert labels.dtype.kind == "i"
    assert labels.nunique() == 1


def test_cluster_T_c_mu_two_finite_rows():
    # F.sum() == 2: the clustering branch executes (the >= 2 guard is met).
    # Two far-apart points produce two distinct labels.
    dd = pd.DataFrame({"T": [300.0, 400.0], "c": [0.0, 1.0], "mu": [-1.0, 1.0]})
    labels = cluster_T_c_mu(dd, distance_threshold=0.5)
    assert labels.dtype.kind == "i"
    assert labels.nunique() == 2


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


# --- f_excess / sub tests (issue #245) ---

_SUB_ATOL = 1e-12


def test_sub_endpoint_line_phases_have_zero_f_excess():
    """Terminal line phases at c=0 and c=1 are their own references; f_excess must be 0."""
    a = LinePhase("A", fixed_concentration=0.0, line_energy=0.0)
    b = LinePhase("B", fixed_concentration=1.0, line_energy=0.3)
    df = calc_phase_diagram([a, b], Ts=np.array([300.0]), mu=np.array([0.3]), refine=False, keep_unstable=True)
    np.testing.assert_allclose(df.loc[df.phase == "A", "f_excess"].values, 0.0, atol=_SUB_ATOL)
    np.testing.assert_allclose(df.loc[df.phase == "B", "f_excess"].values, 0.0, atol=_SUB_ATOL)


def test_sub_tangent_wins_over_f_at_near_endpoint():
    """f_excess of line phase at c=1 is 0 even when a near-endpoint phase has lower raw f.

    sol at c=0.99 has f=0.500; bcc at c=1.0 has f=0.505.
    With mu=1.0:
      tangent1_sol = phi_sol + mu = (0.500 - 0.99) + 1.0 = 0.510
      tangent1_bcc = phi_bcc + mu = (0.505 - 1.00) + 1.0 = 0.505
    Tangent logic picks bcc (0.505 < 0.510) so f_excess_bcc = 0.
    f-based logic would pick sol (0.500 < 0.505), giving f_excess_bcc = 0.005.
    """
    a = LinePhase("A", fixed_concentration=0.0, line_energy=0.0)
    sol = LinePhase("sol", fixed_concentration=0.99, line_energy=0.500)
    bcc = LinePhase("bcc", fixed_concentration=1.0, line_energy=0.505)
    df = calc_phase_diagram([a, sol, bcc], Ts=np.array([300.0]), mu=np.array([1.0]), refine=False, keep_unstable=True)
    np.testing.assert_allclose(df.loc[df.phase == "bcc", "f_excess"].values, 0.0, atol=_SUB_ATOL)


def test_sub_f_excess_is_deviation_from_tangent_chord():
    """f_excess equals f minus the linear interpolation between tangent endpoint references.

    A(c=0, f=0), B(c=1, f=1), mid(c=0.5, f=0.3) with mu=0.5.
    References: f0 = tangent0[A] = phi_A = 0, f1 = tangent1[B] = phi_B + mu = 0.5 + 0.5 = 1.0.
    f_excess_mid = 0.3 - (0*(1-0.5) + 1.0*0.5) = 0.3 - 0.5 = -0.2.
    """
    a = LinePhase("A", fixed_concentration=0.0, line_energy=0.0)
    b = LinePhase("B", fixed_concentration=1.0, line_energy=1.0)
    mid = LinePhase("mid", fixed_concentration=0.5, line_energy=0.3)
    df = calc_phase_diagram([a, b, mid], Ts=np.array([300.0]), mu=np.array([0.5]), refine=False, keep_unstable=True)
    np.testing.assert_allclose(
        df.loc[df.phase == "mid", "f_excess"].values, 0.3 - 0.5, atol=_SUB_ATOL
    )


# --- _f_excess_tangent_chord direct tests ---


def _chord_input(rows):
    """Helper: build a per-T slice with the columns the helper reads."""
    df = pd.DataFrame(rows)
    df["f"] = df["phi"] + df["mu"] * df["c"]
    return df


def test_f_excess_tangent_chord_empty_input_returns_empty_series():
    """All rows drop out under the -inf<mu<inf filter: return an empty Series."""
    dd = _chord_input([
        {"phase": "A", "c": 0.0, "phi": 0.0, "mu": -np.inf},
        {"phase": "B", "c": 1.0, "phi": 0.0, "mu": +np.inf},
    ])
    out = _f_excess_tangent_chord(dd)
    assert isinstance(out, pd.Series)
    assert out.empty


def test_f_excess_tangent_chord_drops_infinite_mu_rows_from_output():
    """The +-inf rows are excluded from both the references and the result."""
    dd = _chord_input([
        {"phase": "A", "c": 0.0, "phi": 0.0, "mu": 0.0},
        {"phase": "B", "c": 1.0, "phi": 0.0, "mu": 0.0},
        {"phase": "A", "c": 0.0, "phi": 9.9, "mu": -np.inf},
    ])
    out = _f_excess_tangent_chord(dd)
    assert len(out) == 2
    assert -np.inf not in out.index.map(lambda i: dd.loc[i, "mu"]).tolist()


def test_f_excess_tangent_chord_zero_for_terminal_line_phases():
    """Two terminals at c=0 and c=1 are their own references; f_excess is exactly 0."""
    dd = _chord_input([
        {"phase": "A", "c": 0.0, "phi": -0.1, "mu": 0.3},
        {"phase": "B", "c": 1.0, "phi": +0.2, "mu": 0.3},
    ])
    out = _f_excess_tangent_chord(dd)
    np.testing.assert_allclose(out.values, 0.0, atol=1e-12)


def test_f_excess_tangent_chord_picks_lowest_endpoint_tangent():
    """When two phases reach an endpoint window, the smaller tangent value wins."""
    # Two phases reach c≈1 (within the 1% window of c_span=1):
    #   sol at c=0.99: tangent1 = phi + mu = -0.49 + 0.5 = 0.01
    #   bcc at c=1.00: tangent1 = phi + mu =  0.00 + 0.5 = 0.50
    # The minimum (sol, 0.01) wins.  An A phase at c=0 anchors f0 = phi = 0.
    dd = _chord_input([
        {"phase": "A",   "c": 0.0,  "phi":  0.00, "mu": 0.5},
        {"phase": "sol", "c": 0.99, "phi": -0.49, "mu": 0.5},
        {"phase": "bcc", "c": 1.00, "phi":  0.00, "mu": 0.5},
    ])
    out = _f_excess_tangent_chord(dd)
    chord = 0.00 * (1 - dd["c"]) + 0.01 * dd["c"]
    np.testing.assert_allclose(out.values, (dd["f"] - chord).values, atol=1e-12)


def test_f_excess_tangent_chord_endpoint_window_is_relative_to_sampled_extreme():
    """A phase qualifies for the endpoint reference if its c reaches within
    1% of the sampled c-span — not 1% of c=0 / c=1.

    Sampled c range [0.0, 1.0] (span 1) ⇒ window for c=0 is [0, 0.01].
    A phase whose minimum lands at c=0.02 is outside that window, so it cannot
    set the c=0 reference even though it would have if c-span were measured
    against the pure concentration.
    """
    dd = _chord_input([
        {"phase": "A",   "c": 0.00, "phi":  0.0, "mu": 0.1},  # owns c=0 ref
        {"phase": "sol", "c": 0.02, "phi": -0.5, "mu": 0.1},  # outside 1% window
        {"phase": "B",   "c": 1.00, "phi":  0.0, "mu": 0.1},  # owns c=1 ref
    ])
    out = _f_excess_tangent_chord(dd)
    # f0 = tangent0[A] = phi_A = 0.0 (sol does not qualify, so it does not lower f0)
    # f1 = tangent1[B] = phi_B + mu = 0.1
    chord = 0.0 * (1 - dd["c"]) + 0.1 * dd["c"]
    np.testing.assert_allclose(out.values, (dd["f"] - chord).values, atol=1e-12)


def test_f_excess_tangent_chord_preserves_input_index():
    """The returned Series aligns to the input row index, not a positional one."""
    dd = _chord_input([
        {"phase": "A", "c": 0.0, "phi": 0.0, "mu": 0.1},
        {"phase": "B", "c": 1.0, "phi": 0.0, "mu": 0.1},
        {"phase": "M", "c": 0.5, "phi": 0.0, "mu": 0.1},
    ])
    dd.index = [100, 200, 300]
    out = _f_excess_tangent_chord(dd)
    assert list(out.index) == [100, 200, 300]
