# https://github.com/eisenforschung/landau/issues/456
"""Regression test for #456: disjoint stable fields of one phase drawn as one polygon.

``get_polygons`` (via ``cluster_phase``) uses single-linkage agglomerative
clustering in normalised (T, c) space to split a phase's stable rows into
connected components before handing each one to a ``poly_method``. With the
old default ``distance_threshold=0.5``, two stable fields of the same phase
that are far apart in c but present at the same (normalised) T got merged
into a single cluster, so every poly_method drew one polygon bridging the
gap — visibly for the TSP methods (which warn and "repair" a self-intersecting
loop into the bridge) and silently for ``concave``.

The fixture is real data attached to the issue: the ``liquid`` rows of a
``calc_phase_diagram`` result for a Y-Zn binary, whose stable region has a
genuine gap of ~0.39 in normalised composition (nothing stable for
0.60 <= c < 0.95) — not a narrow sliver ``min_c_width`` would catch.
"""
import gzip
from pathlib import Path

import pandas as pd
import pytest

from landau.plot import cluster_phase, get_polygons

DATA_PATH = Path(__file__).parent / "data" / "issue_456_yzn_liquid.csv.gz"


def _load():
    with gzip.open(DATA_PATH, "rt") as f:
        return pd.read_csv(f)


def test_cluster_phase_splits_disjoint_liquid_field():
    df = _load()
    stable = df.query("stable").copy()
    out = cluster_phase(stable)
    assert out["phase_unit"].nunique() == 2
    ranges = out.groupby("phase_unit")["c"].agg(["min", "max"]).sort_values("min")
    (_lo0, hi0), (lo1, _hi1) = ranges.to_numpy()
    # The two clusters are the low-c and high-c fields either side of the real
    # gap, not an arbitrary split of one continuous region.
    assert hi0 == pytest.approx(0.562579, abs=1e-5)
    assert lo1 == pytest.approx(0.956186, abs=1e-5)


def test_get_polygons_draws_two_disjoint_liquid_polygons():
    df = _load()
    polys = get_polygons(df, poly_method="concave")
    assert sorted(polys.index) == [("liquid", 0), ("liquid", 1)]
    low, high = (polys[("liquid", 0)], polys[("liquid", 1)])
    low_c = low.get_xy()[:, 0]
    high_c = high.get_xy()[:, 0]
    # Neither polygon straddles the gap; each stays within its own field.
    assert low_c.max() < 0.6
    assert high_c.min() > 0.9


@pytest.mark.parametrize("distance_threshold", [0.01, 0.1, 0.2, 0.3])
def test_cluster_phase_splits_across_the_safe_threshold_band(distance_threshold):
    """The whole empirically-checked safe band, not just the shipped default."""
    df = _load()
    stable = df.query("stable").copy()
    out = cluster_phase(stable, distance_threshold=distance_threshold)
    assert out["phase_unit"].nunique() == 2


def test_cluster_phase_old_default_bridged_the_gap():
    """Pins *why* 0.5 was too high: it merges the real ~0.39-wide gap here.

    Guards against silently reverting the tuned default in `cluster_phase` /
    `get_polygons` without noticing this is the fixture that motivated it.
    """
    df = _load()
    stable = df.query("stable").copy()
    out = cluster_phase(stable, distance_threshold=0.5)
    assert out["phase_unit"].nunique() == 1
