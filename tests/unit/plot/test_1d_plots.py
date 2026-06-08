"""Unit tests for _assign_segment_ids (scan-order segmentation) and 1d-plot functions.

A 1d phase diagram is a scan along one cut axis (mu, T, ... in future generalised
cuts).  ``_assign_segment_ids`` splits a phase into a new line segment at each
stability flip in scan order.  The tests below pin two properties that the prior
distance-threshold approaches got wrong:

* density independence — the segment count depends only on the stability pattern,
  never on how finely the cut is sampled (a constant threshold over-split coarse
  grids; a data-informed one merged disjoint branches on coarse grids);
* metric independence — disjoint runs are split even when their concentration (or
  any other derived quantity) coincides (a (T, c) clusterer would merge them).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import pytest
from collections import Counter
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib.colors import to_rgba

import landau.calculate as ldc
import landau.phases as ldp
from landau.plot import (
    _assign_segment_ids,
    _bold_math,
    _bridge_unstable_segments,
    _phase_visible_in_band,
    _place_transition_labels,
    _spread_labels,
    plot_1d_mu_phase_diagram,
    plot_1d_T_phase_diagram,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan_df(stable_pattern, phase="A", scan_col="mu"):
    """One phase sampled at evenly spaced scan points with a given stability pattern.

    ``stable_pattern`` is a sequence of bools, one per scan point in scan order.
    """
    n = len(stable_pattern)
    return pd.DataFrame(
        {scan_col: np.linspace(0.0, 1.0, n), "phase": phase, "stable": list(stable_pattern)}
    )


def _run_count(pattern):
    """Number of maximal constant-``stable`` runs = 1 + number of flips."""
    return 1 + sum(a != b for a, b in zip(pattern, pattern[1:]))


def _color_to_phase_map(ax):
    """Return {rgba_color: phase_name} from the axes, preferring the axes attribute."""
    white = to_rgba("w")
    gray2 = to_rgba(".2")
    # Prefer the color map stored by _add_1d_phase_legend (legend is removed).
    if hasattr(ax, "_landau_phase_colors") and ax._landau_phase_colors:
        return {to_rgba(v): k for k, v in ax._landau_phase_colors.items()}
    # Fallback: read from the seaborn legend (pre-legend-removal code paths).
    legend = ax.get_legend()
    if legend is None:
        return {}
    result = {}
    for handle, text in zip(legend.legend_handles, legend.texts):
        try:
            c = to_rgba(handle.get_color())
        except (AttributeError, ValueError):
            continue
        if c == white or c == gray2:
            continue
        result[c] = text.get_text()
    return result


def _lines_per_phase(ax):
    """Return {phase_name: line_count} for non-empty Line2D objects on *ax*."""
    color_to_phase = _color_to_phase_map(ax)
    counts = Counter()
    for line in ax.lines:
        if len(line.get_xdata()) == 0:
            continue
        try:
            c = to_rgba(line.get_color())
        except ValueError:
            continue
        phase = color_to_phase.get(c)
        if phase is not None:
            counts[phase] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Deterministic unit tests — scan-order semantics
# ---------------------------------------------------------------------------


def test_single_run_is_one_segment():
    ids = _assign_segment_ids(_scan_df([True] * 20), "mu")
    assert ids.nunique() == 1


def test_single_flip_two_segments():
    ids = _assign_segment_ids(_scan_df([True] * 5 + [False] * 5), "mu")
    assert ids.nunique() == 2


def test_middle_phase_three_segments():
    # unstable / stable / unstable along the scan -> 3 segments, 2 of them unstable
    df = _scan_df([False] * 4 + [True] * 4 + [False] * 4)
    ids = _assign_segment_ids(df, "mu")
    assert ids.nunique() == 3
    assert ids[~df["stable"]].nunique() == 2


@pytest.mark.parametrize("n", [3, 4, 5, 8, 11, 50, 500])
def test_contiguous_run_one_segment_at_any_density(n):
    """A single contiguous run stays one segment regardless of sampling density.

    The constant-0.1 threshold shattered this into one segment per point once the
    normalised spacing 1/(n-1) exceeded 0.1 (n <= 11).
    """
    assert _assign_segment_ids(_scan_df([True] * n), "mu").nunique() == 1


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 40])
def test_density_independent_three_runs(k):
    """unstable/stable/unstable is always 3 segments, however coarse or fine.

    A data-informed (1.5x spacing) threshold merged the two unstable branches on
    coarse grids, where the inter-branch gap and intra-run spacing are comparable.
    """
    df = _scan_df([False] * k + [True] * k + [False] * k)
    assert _assign_segment_ids(df, "mu").nunique() == 3


def test_disjoint_runs_split_with_constant_concentration():
    """Two unstable runs separated by a stable run split even at identical c.

    A (T, c) distance clusterer would merge all rows (one c value); scan-order
    splits on the stability flips instead.
    """
    df = _scan_df([False, False, True, True, False, False])
    df["c"] = 0.5
    ids = _assign_segment_ids(df, "mu")
    assert ids.nunique() == 3
    assert ids[~df["stable"]].nunique() == 2


def test_different_phases_never_share_id():
    a = _scan_df([True] * 10, phase="A")
    b = _scan_df([True] * 10, phase="B")
    df = pd.concat([a, b], ignore_index=True)
    ids = _assign_segment_ids(df, "mu")
    assert set(ids[df.phase == "A"]).isdisjoint(set(ids[df.phase == "B"]))


def test_stable_and_unstable_never_share_id():
    df = _scan_df([True] * 5 + [False] * 5)
    ids = _assign_segment_ids(df, "mu")
    assert set(ids[df.stable]).isdisjoint(set(ids[~df.stable]))


def test_unsorted_input_realigned_to_index():
    df = _scan_df([True, True, False, False])
    shuffled = df.iloc[[3, 1, 0, 2]]
    ids = _assign_segment_ids(shuffled, "mu")
    # result is realigned to the caller's (shuffled) index, so df[col] = ids works
    assert list(ids.index) == list(shuffled.index)
    # the two stable rows share one id; the two unstable rows share another
    assert ids.loc[0] == ids.loc[1]
    assert ids.loc[2] == ids.loc[3]
    assert ids.loc[1] != ids.loc[2]


# ---------------------------------------------------------------------------
# Hypothesis — the tight invariant: segments == stability runs
# ---------------------------------------------------------------------------


@given(pattern=st.lists(st.booleans(), min_size=1, max_size=40))
@settings(max_examples=300)
def test_segments_equal_number_of_stability_runs(pattern):
    """For one phase, the segment count equals its number of constant-stable runs."""
    df = _scan_df(pattern)
    assert _assign_segment_ids(df, "mu").nunique() == _run_count(pattern)


@given(
    pattern=st.lists(st.booleans(), min_size=1, max_size=20),
    n_per_point=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=200)
def test_segment_count_invariant_under_upsampling(pattern, n_per_point):
    """Refining each scan point into ``n_per_point`` copies keeps the segment count."""
    coarse = _assign_segment_ids(_scan_df(pattern), "mu").nunique()
    dense_pattern = [s for s in pattern for _ in range(n_per_point)]
    dense = _assign_segment_ids(_scan_df(dense_pattern), "mu").nunique()
    assert coarse == dense == _run_count(pattern)


@given(
    phases=st.lists(st.text(alphabet="ABCDE", min_size=1, max_size=2), min_size=2, max_size=4, unique=True),
)
@settings(max_examples=100)
def test_multiple_phases_ids_pairwise_disjoint(phases):
    df = pd.concat([_scan_df([True, False, True], phase=p) for p in phases], ignore_index=True)
    ids = _assign_segment_ids(df, "mu")
    id_sets = {p: set(ids[df.phase == p]) for p in phases}
    for i, p in enumerate(phases):
        for q in phases[i + 1:]:
            assert id_sets[p].isdisjoint(id_sets[q]), f"{p!r} and {q!r} share IDs"


# ---------------------------------------------------------------------------
# Three-phase physics fixtures (shared by segment-id and line-count tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def df_mu_three_stable():
    """Isothermal (T=1000K) 3-phase mu scan: hcp / fcc / liquid.

    fcc is the middle stable phase; it is unstable in two disjoint mu ranges
    (one below and one above its stable window).  liquid's stable window is a
    narrow c-band near 1, which inflated the c-distance threshold and merged
    fcc's two branches under the data-informed approach.
    """
    fcca = ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.02, line_entropy=1.0 * ldp.kB)
    fccb = ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.02, line_entropy=1.1 * ldp.kB)
    hcpa = ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
    hcpb = ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB)
    lqda = ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.724, line_entropy=1.5 * ldp.kB)
    lqdb = ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-2.050, line_entropy=1.2 * ldp.kB)
    fcc = ldp.IdealSolution("fcc", fcca, fccb)
    hcp = ldp.IdealSolution("hcp", hcpa, hcpb)
    lqd = ldp.IdealSolution("liquid", lqda, lqdb)
    return ldc.calc_phase_diagram([hcp, fcc, lqd], Ts=1000.0, mu=100, keep_unstable=True)


def _three_line_phases():
    T_s = np.array([100.0, 400.0, 700.0, 1000.0])
    bcc = ldp.TemperatureDependentLinePhase(
        "bcc", fixed_concentration=0, temperatures=T_s,
        free_energies=-3.00 - 2e-4 * T_s - 1e-7 * T_s**2,
    )
    fcc = ldp.TemperatureDependentLinePhase(
        "fcc", fixed_concentration=0, temperatures=T_s,
        free_energies=-2.97 - 2.857e-4 * T_s - 2e-7 * T_s**2,
    )
    liquid = ldp.TemperatureDependentLinePhase(
        "liquid", fixed_concentration=0, temperatures=T_s,
        free_energies=-2.75 - 3.5e-4 * T_s - 5e-7 * T_s**2,
    )
    return [bcc, fcc, liquid]


@pytest.fixture(scope="module")
def df_T_three_stable():
    """1D T scan (mu=0) on a fine 50-point grid: bcc / fcc / liquid line phases.

    fcc is the middle stable phase; it is unstable at both low and high T.
    """
    Ts = np.linspace(100, 1000, 50)
    return ldc.calc_phase_diagram(_three_line_phases(), Ts, mu=0.0, refine=True, keep_unstable=True)


@pytest.fixture(scope="module")
def df_T_three_stable_coarse():
    """Same three line phases on a coarse 8-point T grid.

    At this density the inter-branch T-gap and the intra-run spacing are
    comparable: constant-0.1 shatters every line, the data-informed threshold
    merges fcc's two unstable branches.  Scan-order is unaffected.
    """
    Ts = np.linspace(100, 1000, 8)
    return ldc.calc_phase_diagram(_three_line_phases(), Ts, mu=0.0, refine=True, keep_unstable=True)


# ---------------------------------------------------------------------------
# _assign_segment_ids on the physics fixtures
# ---------------------------------------------------------------------------


def test_assign_segment_ids_mu_middle_phase_three_segments(df_mu_three_stable):
    """fcc (middle stable phase) gets three distinct segment IDs in the mu scan."""
    df = df_mu_three_stable
    ids = _assign_segment_ids(df, "mu")
    fcc_ids = ids[df["phase"] == "fcc"]
    assert fcc_ids.nunique() == 3, f"Expected 3 fcc segments, got {fcc_ids.nunique()}: {fcc_ids.unique()}"
    assert ids[(df["phase"] == "fcc") & (~df["stable"])].nunique() == 2
    for phase in ("hcp", "liquid"):
        assert ids[df["phase"] == phase].nunique() == 2, f"Expected 2 segments for {phase}"


@pytest.mark.parametrize("fixture", ["df_T_three_stable", "df_T_three_stable_coarse"])
def test_assign_segment_ids_T_middle_phase_three_segments(fixture, request):
    """fcc gets three segment IDs in the T scan, at both fine and coarse density."""
    df = request.getfixturevalue(fixture)
    ids = _assign_segment_ids(df, "T")
    fcc_ids = ids[df["phase"] == "fcc"]
    assert fcc_ids.nunique() == 3, f"Expected 3 fcc segments, got {fcc_ids.nunique()}: {fcc_ids.unique()}"
    assert ids[(df["phase"] == "fcc") & (~df["stable"])].nunique() == 2


# ---------------------------------------------------------------------------
# _bridge_unstable_segments — dashed branch reaches the exact transition
# ---------------------------------------------------------------------------


def _two_phase_transition_df(scan_col="T"):
    """Two line phases sharing one refined transition at x=2, as keep_unstable output.

    A is stable below the transition and metastable above; B is the mirror.  The
    transition row (x=2) is marked ``border`` and ``stable`` for both phases, the
    way :func:`~landau.calculate.calc_phase_diagram` tags a refined coexistence
    point.  Both metastable branches resume only at the next sample (x=3 for A,
    x=1 for B), so without bridging each leaves a gap back to x=2.
    """
    df = pd.DataFrame(
        {
            "phase": ["A"] * 5 + ["B"] * 5,
            scan_col: [0, 1, 2, 3, 4] * 2,
            "phi": [0.0, -1.0, -2.0, -3.0, -4.0, 1.0, 0.0, -2.0, -3.5, -5.0],
            "stable": [True, True, True, False, False, False, False, True, True, True],
            "border": [False, False, True, False, False, False, False, True, False, False],
        }
    )
    df["_seg_id"] = _assign_segment_ids(df, scan_col)
    return df


def test_bridge_no_border_column_is_noop():
    df = _scan_df([True] * 3 + [False] * 3)
    df["_seg_id"] = _assign_segment_ids(df, "mu")
    pd.testing.assert_frame_equal(_bridge_unstable_segments(df, "mu"), df)


def test_bridge_adds_one_unstable_twin_per_metastable_side():
    df = _two_phase_transition_df("T")
    bridged = _bridge_unstable_segments(df, "T")
    # Two phases, one metastable side each -> two appended rows.
    assert len(bridged) == len(df) + 2
    added = bridged.iloc[len(df):]
    assert (~added["stable"]).all()
    assert (added["T"] == 2).all()
    # Twins carry the transition's phi, not an interpolated/edge value.
    assert sorted(added["phi"]) == [-2.0, -2.0]


def test_bridge_twin_shares_adjacent_unstable_segment_id():
    df = _two_phase_transition_df("T")
    bridged = _bridge_unstable_segments(df, "T")
    for phase, neighbour_x in (("A", 3), ("B", 1)):
        twin = bridged[(bridged.phase == phase) & (bridged["T"] == 2) & (~bridged.stable)]
        assert len(twin) == 1
        neighbour_seg = df.loc[(df.phase == phase) & (df["T"] == neighbour_x), "_seg_id"].iloc[0]
        assert twin["_seg_id"].iloc[0] == neighbour_seg


def test_bridge_closes_gap_metastable_branch_reaches_transition():
    df = _two_phase_transition_df("T")
    bridged = _bridge_unstable_segments(df, "T")
    # A's single metastable branch now starts exactly at the transition (x=2),
    # B's ends exactly at it -- no gap to the solid line on either side.
    a_unstable = bridged[(bridged.phase == "A") & (~bridged.stable)]
    b_unstable = bridged[(bridged.phase == "B") & (~bridged.stable)]
    assert a_unstable["T"].min() == 2
    assert b_unstable["T"].max() == 2
    # The original solid lines are untouched: A and B still reach x=2 while stable.
    assert ((df.phase == "A") & (df["T"] == 2) & df.stable).any()
    assert ((df.phase == "B") & (df["T"] == 2) & df.stable).any()


@pytest.mark.parametrize(
    "fixture,scan_col",
    [
        ("df_T_three_stable", "T"),
        ("df_T_three_stable_coarse", "T"),
        ("df_mu_three_stable", "mu"),
    ],
)
def test_bridge_every_interior_transition_reached_on_metastable_side(fixture, scan_col, request):
    """On the physics fixtures, every metastable branch touches its bounding transition.

    For each phase and each metastable segment, an endpoint that is not the global
    scan edge must coincide with a transition point -- i.e. the dashed branch runs
    flush to the solid line at the flip, with no grid-density-dependent gap.
    """
    df = request.getfixturevalue(fixture).copy()
    df["_seg_id"] = _assign_segment_ids(df, scan_col)
    bridged = _bridge_unstable_segments(df, scan_col)
    transitions = set(np.round(df.loc[df["border"], scan_col].unique(), 9))
    x_lo, x_hi = df[scan_col].min(), df[scan_col].max()
    for _, seg in bridged[~bridged["stable"]].groupby("_seg_id"):
        lo, hi = seg[scan_col].min(), seg[scan_col].max()
        for end in (lo, hi):
            if np.isclose(end, x_lo) or np.isclose(end, x_hi):
                continue
            assert round(end, 9) in transitions, (
                f"metastable segment endpoint {end} is neither a scan edge nor a transition"
            )


# ---------------------------------------------------------------------------
# plot_1d_mu / plot_1d_T — line-count tests
# ---------------------------------------------------------------------------


def test_plot_1d_mu_middle_phase_three_lines(df_mu_three_stable):
    """fcc appears as three distinct Line2D objects on the axis (stable + 2 unstable)."""
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df_mu_three_stable, ax=ax)
        counts = _lines_per_phase(ax)
        assert counts.get("fcc") == 3, f"Expected 3 fcc lines, got {counts.get('fcc')}"
        assert counts.get("hcp") == 2
        assert counts.get("liquid") == 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fixture", ["df_T_three_stable", "df_T_three_stable_coarse"])
def test_plot_1d_T_middle_phase_three_lines(fixture, request):
    """fcc appears as three Line2D objects at both fine and coarse T sampling."""
    df = request.getfixturevalue(fixture)
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax)
        counts = _lines_per_phase(ax)
        assert counts.get("fcc") == 3, f"Expected 3 fcc lines, got {counts.get('fcc')}"
        assert counts.get("bcc") == 2
        assert counts.get("liquid") == 2
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# reference_phase subtraction
# ---------------------------------------------------------------------------


def _ydata_for_phase(ax, phase_name):
    """Return a flat array of all y-values plotted for *phase_name* on *ax*."""
    color_to_phase = _color_to_phase_map(ax)
    ydata = []
    for line in ax.lines:
        yd = line.get_ydata()
        if len(yd) == 0:
            continue
        try:
            c = to_rgba(line.get_color())
        except ValueError:
            continue
        if color_to_phase.get(c) == phase_name:
            ydata.extend(yd)
    return np.asarray(ydata)


def test_plot_1d_mu_reference_phase_is_zero(df_mu_three_stable):
    """After subtracting the reference phase, its own phi is zero at every mu."""
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df_mu_three_stable, ax=ax, reference_phase="hcp")
        yd = _ydata_for_phase(ax, "hcp")
        assert len(yd) > 0
        np.testing.assert_allclose(yd, 0.0, atol=1e-12)
    finally:
        plt.close(fig)


def test_plot_1d_mu_reference_phase_differences_preserved(df_mu_three_stable):
    """phi(fcc) - phi(ref) in the plot matches fcc_phi - hcp_phi from the raw data."""
    df = df_mu_three_stable
    ref = "hcp"
    # Differences at mu values where BOTH fcc and hcp have exact rows.  At
    # border-only mu values (where the reference is absent from the refined data),
    # _subtract_reference_phase interpolates; those extra values are not tested here.
    shared_mu = set(df.loc[df["phase"] == ref, "mu"]) & set(df.loc[df["phase"] == "fcc", "mu"])
    pivot = df[df["mu"].isin(shared_mu)].pivot_table(index="mu", columns="phase", values="phi")
    expected = np.sort((pivot["fcc"] - pivot[ref]).dropna().values)

    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df, ax=ax, reference_phase=ref)
        yd_fcc = _ydata_for_phase(ax, "fcc")
        yd_valid = np.sort(yd_fcc[~np.isnan(yd_fcc)])
        assert len(yd_valid) >= len(expected)
        for v in expected:
            assert np.any(np.isclose(yd_valid, v, atol=1e-10)), (
                f"Expected fcc-hcp difference {v:.8f} not found in plotted fcc values"
            )
    finally:
        plt.close(fig)


def test_plot_1d_mu_reference_phase_invalid_raises(df_mu_three_stable):
    """A reference_phase not present in the data raises ValueError."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="reference_phase"):
            plot_1d_mu_phase_diagram(df_mu_three_stable, ax=ax, reference_phase="nonexistent")
    finally:
        plt.close(fig)


def test_plot_1d_T_reference_phase_is_zero(df_T_three_stable):
    """After subtracting the reference phase, its own phi is zero at every T."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, reference_phase="bcc")
        yd = _ydata_for_phase(ax, "bcc")
        assert len(yd) > 0
        np.testing.assert_allclose(yd, 0.0, atol=1e-12)
    finally:
        plt.close(fig)


def test_plot_1d_T_reference_phase_invalid_raises(df_T_three_stable):
    """A reference_phase not present in the data raises ValueError."""
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="reference_phase"):
            plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, reference_phase="nonexistent")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Transition-marker introspection
# ---------------------------------------------------------------------------


def _transition_marker_y_positions(ax):
    """Return the y-positions of all black scatter markers (transition dots) on *ax*.

    Transition markers are added via ``ax.scatter(..., c='k')``, producing a
    PathCollection with black facecolor.  Other artists (seaborn line plots) do
    not create black PathCollections, so this isolates the marker set.
    """
    ys = []
    for coll in ax.collections:
        try:
            fc = coll.get_facecolor()
        except AttributeError:
            continue
        if fc is None or len(fc) == 0:
            continue
        color = np.asarray(fc[0])
        if len(color) >= 3 and np.allclose(color[:3], [0.0, 0.0, 0.0], atol=0.01):
            offsets = np.asarray(coll.get_offsets())
            if len(offsets) > 0:
                ys.extend(offsets[:, 1])
    return np.asarray(ys)


def test_plot_1d_T_reference_phase_all_transitions_marked(df_T_three_stable):
    """All transition markers are present and have finite y when reference_phase is set.

    Before the np.interp fix, border points where the reference phase had no
    exact row got phi=NaN from the merge, making every marker after the first
    invisible (ax.scatter with y=NaN renders nothing).
    """
    fig0, ax0 = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax0)
        n_expected = len(_transition_marker_y_positions(ax0))
    finally:
        plt.close(fig0)

    assert n_expected >= 2, "fixture must have at least 2 transition markers"

    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, reference_phase="bcc")
        ys = _transition_marker_y_positions(ax)
        assert len(ys) == n_expected, (
            f"Expected {n_expected} transition markers with reference_phase='bcc', "
            f"got {len(ys)}. Likely NaN phi at a border point where bcc has no row."
        )
        assert np.all(np.isfinite(ys)), f"Some transition markers have non-finite y: {ys}"
    finally:
        plt.close(fig)


def _transition_text_artists(ax):
    """Return the rotated transition-marker text artists ($T=...$ / $\\Delta\\mu=...$)."""
    return [t for t in ax.texts if abs(t.get_rotation() - 90.0) < 1e-6]


@pytest.mark.parametrize(
    "plot_func, fixture, ylim",
    [
        (plot_1d_T_phase_diagram, "df_T_three_stable", (-3.3, -3.1)),
        (plot_1d_mu_phase_diagram, "df_mu_three_stable", (-3.3, -3.1)),
    ],
)
def test_transition_labels_stay_inside_axis_under_ylim(plot_func, fixture, ylim, request):
    """A zoomed ylim keeps the transition-marker text inside the axes.

    The labels are positioned in axes-fraction y, so they no longer track the crossing
    phi (which the zoom can place far outside the window).
    """
    df = request.getfixturevalue(fixture)
    fig, ax = plt.subplots()
    try:
        plot_func(df, ax=ax, ylim=ylim)
        renderer = _renderer(fig)
        axbb = ax.get_window_extent(renderer)
        labels = _transition_text_artists(ax)
        assert labels, "no transition-marker labels found"
        for t in labels:
            bb = t.get_window_extent(renderer)
            assert axbb.y0 - 0.5 <= bb.y0 and bb.y1 <= axbb.y1 + 0.5, (
                f"{t.get_text()!r} y-box [{bb.y0}, {bb.y1}] outside axes [{axbb.y0}, {axbb.y1}]"
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "plot_func, fixture",
    [
        (plot_1d_T_phase_diagram, "df_T_three_stable"),
        (plot_1d_mu_phase_diagram, "df_mu_three_stable"),
    ],
)
def test_transition_labels_have_white_outline(plot_func, fixture, request):
    """Transition-marker labels are stroked with a white outline (legible over tielines)."""
    df = request.getfixturevalue(fixture)
    fig, ax = plt.subplots()
    try:
        plot_func(df, ax=ax)
        labels = _transition_text_artists(ax)
        assert labels, "no transition-marker labels found"
        for t in labels:
            effects = t.get_path_effects()
            assert len(effects) == 1, f"{t.get_text()!r} has no path effect"
            assert to_rgba(effects[0]._gc["foreground"]) == to_rgba("white")
            assert t.get_zorder() >= 100, f"{t.get_text()!r} zorder too low"
    finally:
        plt.close(fig)


def _boxes_overlap_x(a, b, tol=0.5):
    """True when two pixel bounding boxes overlap along x (beyond ``tol`` slack)."""
    return not (a.x1 <= b.x0 + tol or b.x1 <= a.x0 + tol)


@pytest.mark.parametrize("side", ["left", "right"])
def test_place_transition_labels_spread_and_avoid_lines(side):
    """Crowded transition labels are fanned apart and kept off the dotted lines.

    Two transitions a hair apart would otherwise stack their vertical labels on top
    of one another and on the lines; the placement spreads them horizontally (via the
    same routine that stacks the side labels) and confines them to the free gaps.
    """
    fig, ax = plt.subplots()
    try:
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_xlim(0.0, 1.0)
        positions = [0.50, 0.515]
        for p in positions:
            ax.axvline(p)
        labels = [rf"$\Delta\mu = {p:.03f}$" for p in positions]
        texts = _place_transition_labels(ax, positions, labels, side=side)
        r = _renderer(fig)
        boxes = [t.get_window_extent(r) for t in texts]
        lines_px = [ax.transData.transform((p, 0.0))[0] for p in positions]

        for a, b in itertools.combinations(boxes, 2):
            assert not _boxes_overlap_x(a, b), "transition labels overlap horizontally"
        for bb in boxes:
            for L in lines_px:
                assert not (bb.x0 + 0.5 < L < bb.x1 - 0.5), "label box sits on a transition line"
        axbb = ax.get_window_extent(r)
        for bb in boxes:
            assert axbb.x0 - 0.5 <= bb.x0 and bb.x1 <= axbb.x1 + 0.5, "label pushed outside the axes"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "plot_func, fixture",
    [
        (plot_1d_T_phase_diagram, "df_T_three_stable"),
        (plot_1d_mu_phase_diagram, "df_mu_three_stable"),
    ],
)
def test_transition_labels_never_overlap(plot_func, fixture, request):
    """On the physics fixtures no two transition labels overlap horizontally."""
    df = request.getfixturevalue(fixture)
    fig, ax = plt.subplots()
    try:
        plot_func(df, ax=ax)
        r = _renderer(fig)
        boxes = [t.get_window_extent(r) for t in _transition_text_artists(ax)]
        assert len(boxes) >= 2, "fixture must have at least two transition labels"
        for a, b in itertools.combinations(boxes, 2):
            assert not _boxes_overlap_x(a, b), "transition labels overlap horizontally"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("side", ["left", "right"])
def test_place_transition_labels_dense_keep_order_no_overlap(side):
    """Three transitions tighter than a label is wide still never overlap and keep order.

    Confining each label to a free gap is impossible here; the guarantee is the
    weaker pair (no mutual overlap, preserved left-to-right order), line avoidance
    being best effort only.
    """
    fig, ax = plt.subplots()
    try:
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_xlim(0.0, 1.0)
        positions = [0.50, 0.51, 0.52]
        for p in positions:
            ax.axvline(p)
        labels = [rf"$\Delta\mu = {p:.03f}$" for p in positions]
        texts = _place_transition_labels(ax, positions, labels, side=side)
        r = _renderer(fig)
        boxes = [t.get_window_extent(r) for t in texts]
        for a, b in itertools.combinations(boxes, 2):
            assert not _boxes_overlap_x(a, b), "transition labels overlap horizontally"
        centers = [0.5 * (b.x0 + b.x1) for b in boxes]
        assert centers == sorted(centers), "label order not preserved"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Custom phase legend (_add_1d_phase_legend)
# ---------------------------------------------------------------------------


def _top_labels(ax):
    """Return text strings of top-interior phase-name annotations on *ax*."""
    texts = []
    for t in ax.texts:
        try:
            if t.get_va() == "top" and t.get_ha() == "center":
                texts.append(t.get_text())
        except Exception:
            pass
    return texts


def _top_label_artists(ax):
    """Return the Text artists for the top-interior phase-name annotations on *ax*."""
    return [t for t in ax.texts if t.get_va() == "top" and t.get_ha() == "center"]


def _right_annotations(ax):
    """Return text strings of right-end phase annotations placed in data coords."""
    texts = []
    for t in ax.texts:
        try:
            if t.get_ha() == "left" and t.get_va() == "center":
                texts.append(t.get_text())
        except Exception:
            pass
    return texts


def _legend_texts(ax):
    """Return the set of legend entry strings on *ax*, empty if no legend."""
    legend = ax.get_legend()
    return set() if legend is None else {t.get_text() for t in legend.texts}


def _side_label_artists(ax):
    """Return the Text artists for the right/left side-label stacks on *ax*.

    Both stacks use va='center'; the right stack is ha='left', the left ha='right'.
    The top-spine labels use va='top', so filtering on va='center' isolates them.
    """
    return [t for t in ax.texts if t.get_va() == "center" and t.get_ha() in ("left", "right")]


def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _synthetic_T_df(specs):
    """Build a 1d T-scan frame with controlled per-phase phi curves.

    ``specs`` maps a phase name to its phi values along the scan.  All rows are marked
    unstable so the side-label stack is exercised; no 'border' column is added so the
    plot function returns right after the legend (top labels and transition markers are
    skipped, keeping the test focused on the side stack).
    """
    n = len(next(iter(specs.values())))
    T = np.linspace(0.0, 1.0, n)
    frames = [
        pd.DataFrame({"T": T, "mu": 0.0, "phase": phase, "phi": np.asarray(phi, float), "stable": False})
        for phase, phi in specs.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_mu_phase_legend_removed(df_mu_three_stable):
    """The seaborn phase legend is replaced: no phase names appear in any legend."""
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df_mu_three_stable, ax=ax)
        phases = set(df_mu_three_stable["phase"].unique())
        assert phases.isdisjoint(_legend_texts(ax))
    finally:
        plt.close(fig)


def test_T_phase_legend_removed(df_T_three_stable):
    """The seaborn phase legend is replaced: no phase names appear in any legend."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        phases = set(df_T_three_stable["phase"].unique())
        assert phases.isdisjoint(_legend_texts(ax))
    finally:
        plt.close(fig)


def test_side_labels_false_keeps_seaborn_legend(df_T_three_stable):
    """side_labels=False keeps the default seaborn legend (phase names present)."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, side_labels=False)
        phases = set(df_T_three_stable["phase"].unique())
        assert phases <= _legend_texts(ax), "phase names missing from kept legend"
    finally:
        plt.close(fig)


def test_side_labels_false_no_right_annotations(df_T_three_stable):
    """side_labels=False draws no right-end annotations."""
    assert not df_T_three_stable["stable"].all(), "fixture must have unstable rows"
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, side_labels=False)
        assert _right_annotations(ax) == []
    finally:
        plt.close(fig)


def test_top_labels_false_no_top_labels(df_T_three_stable):
    """top_labels=False draws no top-spine phase labels."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, top_labels=False)
        assert _top_labels(ax) == []
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "plot_func, fixture",
    [
        (plot_1d_mu_phase_diagram, "df_mu_three_stable"),
        (plot_1d_T_phase_diagram, "df_T_three_stable"),
    ],
)
def test_flags_are_independent(plot_func, fixture, request):
    """top_labels and side_labels toggle their own annotations, in both 1d plots."""
    df = request.getfixturevalue(fixture)
    fig, ax = plt.subplots()
    try:
        plot_func(df, ax=ax, top_labels=True, side_labels=False)
        assert _top_labels(ax), "top labels missing with top_labels=True"
        assert _right_annotations(ax) == [], "side labels present with side_labels=False"
        assert ax.get_legend() is not None, "seaborn legend removed with side_labels=False"
    finally:
        plt.close(fig)


def test_mu_phase_colors_stored(df_mu_three_stable):
    """ax._landau_phase_colors is populated for all phases."""
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df_mu_three_stable, ax=ax)
        colors = ax._landau_phase_colors
        assert set(colors.keys()) == {"hcp", "fcc", "liquid"}
    finally:
        plt.close(fig)


def test_T_phase_colors_stored(df_T_three_stable):
    """ax._landau_phase_colors is populated for all phases."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        colors = ax._landau_phase_colors
        assert set(colors.keys()) == {"bcc", "fcc", "liquid"}
    finally:
        plt.close(fig)


def test_mu_top_spine_labels_all_stable_phases(df_mu_three_stable):
    """Each stable phase appears as a centered label near the top of the axis."""
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df_mu_three_stable, ax=ax)
        labels = _top_labels(ax)
        stable_phases = df_mu_three_stable.loc[df_mu_three_stable["stable"], "phase"].unique()
        for phase in stable_phases:
            assert phase in labels, f"{phase!r} not in top labels {labels}"
    finally:
        plt.close(fig)


def test_T_top_spine_labels_all_stable_phases(df_T_three_stable):
    """Each stable phase appears as a centered label near the top of the axis."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        labels = _top_labels(ax)
        stable_phases = df_T_three_stable.loc[df_T_three_stable["stable"], "phase"].unique()
        for phase in stable_phases:
            assert phase in labels, f"{phase!r} not in top labels {labels}"
    finally:
        plt.close(fig)


def test_T_top_spine_ticks_at_transitions(df_T_three_stable):
    """The secondary top axis carries one tick at each interior transition."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        # secondary_xaxis('top') registers itself as a child axis.
        assert ax.child_axes, "no secondary top axis created"
        ticks = sorted(ax.child_axes[0].get_xticks())
        x_min, x_max = df_T_three_stable["T"].min(), df_T_three_stable["T"].max()
        expected = sorted(
            t for t in df_T_three_stable.loc[df_T_three_stable["border"], "T"].unique()
            if x_min < t < x_max
        )
        assert len(expected) >= 2, "fixture must have at least two interior transitions"
        assert len(ticks) == len(expected), f"got {ticks}, expected {expected}"
        np.testing.assert_allclose(ticks, expected, atol=1e-6)
    finally:
        plt.close(fig)


def test_top_spine_labels_bold_with_white_outline(df_T_three_stable):
    """Top labels are bold and stroked with a white outline (legible over tielines)."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        artists = _top_label_artists(ax)
        assert artists, "no top-spine phase labels found"
        for t in artists:
            assert t.get_fontweight() == "bold", f"{t.get_text()!r} not bold"
            effects = t.get_path_effects()
            assert len(effects) == 1, f"{t.get_text()!r} has no path effect"
            assert to_rgba(effects[0]._gc["foreground"]) == to_rgba("white")
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Al", "Al"),                                       # no math: unchanged
        ("liquid", "liquid"),
        ("Ca (bcc)", "Ca (bcc)"),
        ("Al$_4$Ca$_2$", r"Al$\mathbf{_4}$Ca$\mathbf{_2}$"),
        ("Al$_{14}$Ca$_{13}$", r"Al$\mathbf{_{14}}$Ca$\mathbf{_{13}}$"),
        (r"$\frac{1}{2}$AB", r"$\mathbf{\frac{1}{2}}$AB"),  # arbitrary LaTeX preserved
        ("a$b$c$d", "a$b$c$d"),                             # unbalanced '$': untouched
        ("X$$Y", "X$$Y"),                                   # empty math segment: untouched
    ],
)
def test_bold_math_wraps_math_segments(label, expected):
    assert _bold_math(label) == expected


def _label_darkness(text):
    """Rendered ink (summed darkness) of a bold Text artist drawn on a fresh figure."""
    fig = plt.figure(figsize=(3, 1), dpi=200)
    try:
        fig.text(0.5, 0.5, text, fontsize=40, fontweight="bold", ha="center", va="center")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return int((255 - buf[..., 0]).sum())
    finally:
        plt.close(fig)


def test_bold_math_subscripts_render_bolder():
    """The \\mathbf wrap actually thickens the subscript; fontweight='bold' alone does not."""
    plain = _label_darkness("X$_2$")
    bolded = _label_darkness(_bold_math("X$_2$"))
    assert bolded > plain * 1.05, f"bolded subscript not thicker: {bolded} vs {plain}"


def test_mu_right_annotations_when_unstable(df_mu_three_stable):
    """With unstable phases present, every phase gets a right-end annotation inside the axis."""
    assert not df_mu_three_stable["stable"].all(), "fixture must have unstable rows"
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df_mu_three_stable, ax=ax)
        right = sorted(_right_annotations(ax))
        all_phases = sorted(df_mu_three_stable["phase"].unique())
        assert right == all_phases, f"expected right annotations {all_phases}, got {right}"
    finally:
        plt.close(fig)


def test_T_right_annotations_when_unstable(df_T_three_stable):
    """With unstable phases present, every phase gets a right-end annotation inside the axis."""
    assert not df_T_three_stable["stable"].all(), "fixture must have unstable rows"
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        right = sorted(_right_annotations(ax))
        all_phases = sorted(df_T_three_stable["phase"].unique())
        assert right == all_phases, f"expected right annotations {all_phases}, got {right}"
    finally:
        plt.close(fig)


def test_T_right_annotations_right_of_line_ends_inside_axis(df_T_three_stable):
    """Right-end labels sit past each line's end but stay within the axis x-limits."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        line_end = df_T_three_stable["T"].max()
        x_left, x_right = ax.get_xlim()
        artists = [t for t in ax.texts if t.get_ha() == "left" and t.get_va() == "center"]
        assert artists, "no right-end annotations found"
        for t in artists:
            x = t.get_position()[0]
            assert x > line_end, f"{t.get_text()!r} at x={x} not right of line end {line_end}"
            assert x_left < x < x_right, f"{t.get_text()!r} at x={x} outside axis [{x_left}, {x_right}]"
    finally:
        plt.close(fig)


def _stable_only_df(df):
    """Return a version of df with only stable rows."""
    return df.loc[df["stable"]].copy()


def test_mu_no_right_annotations_stable_only(df_mu_three_stable):
    """Stable-only data produces no right-edge annotations."""
    df = _stable_only_df(df_mu_three_stable)
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(df, ax=ax)
        assert _right_annotations(ax) == []
    finally:
        plt.close(fig)


def test_T_no_right_annotations_stable_only(df_T_three_stable):
    """Stable-only data produces no right-edge annotations."""
    df = _stable_only_df(df_T_three_stable)
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax)
        assert _right_annotations(ax) == []
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# _spread_labels — vertical overlap resolution
# ---------------------------------------------------------------------------


def test_spread_labels_separates_coincident_targets():
    """Three labels requesting the same center come out non-overlapping in [lo, hi]."""
    out = _spread_labels([5.0, 5.0, 5.0], [1.0, 1.0, 1.0], lo=0.0, hi=10.0)
    out = sorted(out)
    for a, b in zip(out, out[1:]):
        assert b - a >= 1.0 - 1e-9, f"labels overlap: {out}"
    assert out[0] >= 0.0 - 1e-9 and out[-1] <= 10.0 + 1e-9


def test_spread_labels_leaves_separated_targets_untouched():
    """Targets already clear of one another are returned unchanged."""
    out = _spread_labels([1.0, 5.0, 9.0], [1.0, 1.0, 1.0], lo=0.0, hi=10.0)
    np.testing.assert_allclose(out, [1.0, 5.0, 9.0])


def test_spread_labels_preserves_input_order():
    """The returned list is aligned with the input order, not the sorted order."""
    out = _spread_labels([9.0, 1.0, 5.0], [1.0, 1.0, 1.0], lo=0.0, hi=10.0)
    # largest target stays the largest output, smallest stays smallest
    assert out[0] == max(out)
    assert out[1] == min(out)


def test_spread_labels_respects_top_bound():
    """Targets piled near the top are pushed down to fit under hi."""
    out = _spread_labels([9.5, 9.6, 9.7], [1.0, 1.0, 1.0], lo=0.0, hi=10.0)
    assert max(out) <= 10.0 + 1e-9
    out_sorted = sorted(out)
    for a, b in zip(out_sorted, out_sorted[1:]):
        assert b - a >= 1.0 - 1e-9


# ---------------------------------------------------------------------------
# ylim keyword
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_func, fixture",
    [
        (plot_1d_mu_phase_diagram, "df_mu_three_stable"),
        (plot_1d_T_phase_diagram, "df_T_three_stable"),
    ],
)
def test_ylim_sets_axis_limits(plot_func, fixture, request):
    """The ylim kwarg sets the axes y-limits like the matplotlib builtin."""
    df = request.getfixturevalue(fixture)
    fig, ax = plt.subplots()
    try:
        plot_func(df, ax=ax, ylim=(-0.1, 0.2))
        np.testing.assert_allclose(ax.get_ylim(), (-0.1, 0.2))
    finally:
        plt.close(fig)


def test_ylim_scalar_sets_upper_bound_only(df_T_three_stable):
    """A scalar ylim sets only the top; the bottom stays at its autoscaled value."""
    fig0, ax0 = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax0)
        auto_bottom = ax0.get_ylim()[0]
    finally:
        plt.close(fig0)

    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, ylim=0.2)
        bottom, top = ax.get_ylim()
        assert top == pytest.approx(0.2)
        assert bottom == pytest.approx(auto_bottom)
    finally:
        plt.close(fig)


def test_ylim_clamps_side_labels_within_window():
    """Side labels stay within the ylim window (their y is clamped to it)."""
    df = _synthetic_T_df({"rising": np.linspace(-1.0, 3.0, 6), "flat": np.zeros(6)})
    ylim = (-2.0, 2.0)
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax, ylim=ylim, top_labels=False)
        labels = _side_label_artists(ax)
        assert {t.get_text() for t in labels} == {"rising", "flat"}
        for t in labels:
            y = t.get_position()[1]
            assert ylim[0] - 1e-9 <= y <= ylim[1] + 1e-9, f"{t.get_text()!r} at y={y} outside {ylim}"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Adaptive width reservation (rendered text size, not string length)
# ---------------------------------------------------------------------------


def test_side_labels_fit_inside_axis(df_T_three_stable):
    """Every side label's rendered box fits horizontally inside the axes."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        renderer = _renderer(fig)
        axbb = ax.get_window_extent(renderer)
        labels = _side_label_artists(ax)
        assert labels, "no side labels drawn"
        for t in labels:
            bb = t.get_window_extent(renderer)
            assert axbb.x0 - 0.5 <= bb.x0 and bb.x1 <= axbb.x1 + 0.5, (
                f"{t.get_text()!r} box [{bb.x0}, {bb.x1}] not within axes [{axbb.x0}, {axbb.x1}]"
            )
    finally:
        plt.close(fig)


def test_longer_labels_reserve_more_space(df_T_three_stable):
    """Wider rendered labels reserve more horizontal room than narrow ones.

    The reservation is read off the rendered text size, so renaming the phases to long
    strings widens the right x-limit beyond what the short names need — no string-length
    heuristic, just the measured box.
    """
    short = df_T_three_stable
    long = short.copy()
    long["phase"] = long["phase"].map(lambda p: p + "_" + "x" * 30)

    x_data_max = short["T"].max()

    def reserved(df):
        fig, ax = plt.subplots()
        try:
            plot_1d_T_phase_diagram(df, ax=ax)
            return ax.get_xlim()[1] - x_data_max
        finally:
            plt.close(fig)

    assert reserved(long) > reserved(short)


# ---------------------------------------------------------------------------
# Adaptive vertical avoidance
# ---------------------------------------------------------------------------


def test_side_labels_keep_terminal_order_when_clamped():
    """Spread labels keep their line-terminal vertical order, even clamped to the window.

    'a' ends higher (-8) than 'b' (-10); both terminals fall below the window so both
    are pulled to the bottom edge.  Pre-clamping tied them and fell back to alphabetical
    order (b above a); the fix keeps the terminal order (a above b).
    """
    df = _synthetic_T_df({
        "a": np.linspace(0.0, -8.0, 6),
        "b": np.linspace(0.0, -10.0, 6),
    })
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax, ylim=(-2.0, 2.0), top_labels=False)
        ypos = {t.get_text(): t.get_position()[1] for t in _side_label_artists(ax)}
        assert set(ypos) == {"a", "b"}
        assert ypos["a"] > ypos["b"], f"terminal order not preserved: {ypos}"
    finally:
        plt.close(fig)


def test_side_labels_stay_below_top_labels(df_T_three_stable):
    """Both side stacks are capped below the bold top stable-phase labels.

    With a ylim placing a right-end value near the top of the window, the side label
    would otherwise rise into the top-label band; it must be pushed below it.
    """
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax, reference_phase="fcc", ylim=(-0.16, 0.16))
        renderer = _renderer(fig)
        top_boxes = [t.get_window_extent(renderer) for t in _top_label_artists(ax)]
        side_boxes = [t.get_window_extent(renderer) for t in _side_label_artists(ax)]
        assert top_boxes, "no top labels present"
        assert side_boxes, "no side labels present"
        top_bottom = min(b.y0 for b in top_boxes)
        for b in side_boxes:
            assert b.y1 <= top_bottom + 0.5, "side label intrudes into the top-label band"
    finally:
        plt.close(fig)


def test_side_labels_do_not_overlap():
    """Side labels in a stack never overlap vertically once placed.

    Three phases whose right ends nearly coincide would collide without spreading.
    """
    df = _synthetic_T_df({
        "a": np.linspace(-0.8, 0.00, 6),
        "b": np.linspace(-0.7, 0.03, 6),
        "c": np.linspace(-0.9, -0.03, 6),
    })
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax, ylim=(-1.0, 1.0), top_labels=False)
        assert len(_side_label_artists(ax)) == 3
        renderer = _renderer(fig)
        # Each stack shares an x anchor; group by ha so the two columns are checked apart.
        for ha in ("left", "right"):
            boxes = sorted(
                (t.get_window_extent(renderer) for t in _side_label_artists(ax) if t.get_ha() == ha),
                key=lambda b: b.y0,
            )
            for lower, upper in zip(boxes, boxes[1:]):
                assert upper.y0 >= lower.y1 - 0.5, "side labels overlap vertically"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Left-hand stack for labels clipped off the top by ylim
# ---------------------------------------------------------------------------


def test_ylim_moves_clipped_label_to_left_stack():
    """A phase visible at its left end but clipped off the top labels on the left.

    'rising' is visible at low T but exits the top of the window before its right end,
    so it labels on the left (ha='right'); 'flat' stays in view and labels on the right.
    """
    df = _synthetic_T_df({"rising": np.linspace(-1.0, 3.0, 6), "flat": np.zeros(6)})
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax, ylim=(-2.0, 2.0), top_labels=False)
        left_labels = {t.get_text() for t in ax.texts if t.get_ha() == "right" and t.get_va() == "center"}
        right_labels = {t.get_text() for t in ax.texts if t.get_ha() == "left" and t.get_va() == "center"}
        assert left_labels == {"rising"}
        assert right_labels == {"flat"}
    finally:
        plt.close(fig)


def test_ylim_drops_fully_invisible_phase_labels():
    """A phase whose whole line the y-limit pushes out of view gets no label."""
    df = _synthetic_T_df({
        "rising": np.linspace(-1.0, 3.0, 6),  # visible at left, clipped top -> left stack
        "flat": np.zeros(6),                   # fully in view -> right stack
        "above": np.full(6, 5.0),              # entirely above the window -> dropped
        "below": np.full(6, -5.0),             # entirely below the window -> dropped
    })
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax, ylim=(-2.0, 2.0), top_labels=False)
        labelled = {t.get_text() for t in _side_label_artists(ax)}
        assert labelled == {"rising", "flat"}
    finally:
        plt.close(fig)


def test_left_stack_reserves_space_on_left():
    """When labels move to the left stack, the left x-limit is widened to fit them."""
    df = _synthetic_T_df({"rise1": np.linspace(-1.0, 3.0, 6), "rise2": np.linspace(-1.2, 4.0, 6)})
    x_data_min = df["T"].min()
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df, ax=ax, ylim=(-2.0, 2.0), top_labels=False)
        left_stack = [t for t in ax.texts if t.get_ha() == "right" and t.get_va() == "center"]
        assert {t.get_text() for t in left_stack} == {"rise1", "rise2"}
        # The left x-limit has been pushed below the leftmost data point to make room.
        assert ax.get_xlim()[0] < x_data_min
        # Every left label sits inside the axis.
        renderer = _renderer(fig)
        axbb = ax.get_window_extent(renderer)
        for t in left_stack:
            bb = t.get_window_extent(renderer)
            assert bb.x0 >= axbb.x0 - 0.5, f"{t.get_text()!r} clipped past the left spine"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# _phase_visible_in_band — line visibility within a y-window
# ---------------------------------------------------------------------------


def test_phase_visible_point_inside_band():
    assert _phase_visible_in_band([5.0, 6.0, 7.0], 5.5, 6.5) is True


def test_phase_visible_all_above_band():
    assert _phase_visible_in_band([5.0, 6.0], 0.0, 1.0) is False


def test_phase_visible_all_below_band():
    assert _phase_visible_in_band([-5.0, -6.0], 0.0, 1.0) is False


def test_phase_visible_segment_crosses_band():
    # No sampled point lies inside, but the segment straddles the whole window.
    assert _phase_visible_in_band([-5.0, 5.0], -1.0, 1.0) is True


def test_phase_visible_empty_or_nan():
    assert _phase_visible_in_band([np.nan, np.nan], 0.0, 1.0) is False
    assert _phase_visible_in_band([], 0.0, 1.0) is False
