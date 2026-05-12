# Regression test for pandas 3.0 compatibility in landau.plot.cluster_phase.
# In pandas 3, groupby('phase').apply(cluster, include_groups=False) wraps
# a per-group Series whose index is the original row indices into a single-row
# DataFrame whose columns are the original indices.  Assigning that DataFrame
# back to df["phase_unit"] then raises:
#   ValueError: Cannot set a DataFrame with multiple columns to the single column phase_unit
import numpy as np
import pandas as pd

from landau.plot import cluster_phase


def _make_single_phase_df():
    n = 30
    T = np.tile(np.linspace(200.0, 800.0, 5), n // 5)
    mu = np.linspace(-0.05, 0.05, n)
    c = (mu - mu.min()) / (mu.max() - mu.min())
    df = pd.DataFrame(
        {"phase": ["A"] * n, "T": T, "c": c, "mu": mu, "stable": True, "border": True}
    )
    df_inf = pd.DataFrame(
        {
            "phase": ["A"] * 4,
            "T": [200.0, 800.0, 200.0, 800.0],
            "c": [0.0, 0.0, 1.0, 1.0],
            "mu": [-np.inf, -np.inf, np.inf, np.inf],
            "stable": True,
            "border": True,
        }
    )
    return pd.concat([df, df_inf], ignore_index=True)


def test_cluster_phase_single_phase():
    out = cluster_phase(_make_single_phase_df())
    assert "phase_unit" in out.columns
    assert "phase_id" in out.columns
    assert len(out["phase_unit"]) == len(out)
    assert (out["phase_id"].astype(str).str.startswith("A_")).all()


def test_cluster_phase_two_phases():
    a = _make_single_phase_df()
    b = _make_single_phase_df()
    b["phase"] = "B"
    b["c"] = 1.0 - b["c"]
    out = cluster_phase(pd.concat([a, b], ignore_index=True))
    assert "phase_unit" in out.columns
    assert "phase_id" in out.columns
    assert set(out["phase_id"].astype(str).str.split("_").str[0]) == {"A", "B"}
