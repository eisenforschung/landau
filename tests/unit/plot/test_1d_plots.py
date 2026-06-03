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


def test_top_spine_labels_bold_with_translucent_white_bbox(df_T_three_stable):
    """Top labels are bold and backed by a semi-transparent white box (legible over tielines)."""
    fig, ax = plt.subplots()
    try:
        plot_1d_T_phase_diagram(df_T_three_stable, ax=ax)
        artists = _top_label_artists(ax)
        assert artists, "no top-spine phase labels found"
        for t in artists:
            assert t.get_fontweight() == "bold", f"{t.get_text()!r} not bold"
            patch = t.get_bbox_patch()
            assert patch is not None, f"{t.get_text()!r} has no bbox patch"
            assert to_rgba(patch.get_facecolor())[:3] == to_rgba("white")[:3]
            assert 0 < patch.get_alpha() < 1, f"bbox alpha {patch.get_alpha()} not translucent"
    finally:
        plt.close(fig)


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
