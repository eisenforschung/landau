import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import seaborn as sns
from matplotlib.colors import to_rgb

import landau.calculate as ldc
from landau.plot import get_phase_colors, plot_1d_mu_phase_diagram, plot_excess_free_energy, plot_phase_diagram

def test_get_phase_colors_default():
    phase_names = ["Alpha", "Beta", "Gamma"]
    colors = get_phase_colors(phase_names, override={})

    pastel = sns.palettes.SEABORN_PALETTES["pastel"]
    assert colors["Alpha"] == pastel[0]
    assert colors["Beta"] == pastel[1]
    assert colors["Gamma"] == pastel[2]

def test_get_phase_colors_override_simple():
    phase_names = ["Alpha", "Beta"]
    override = {"Alpha": "red"}
    colors = get_phase_colors(phase_names, override=override)

    assert colors["Alpha"] == "red"
    assert colors["Beta"] == sns.palettes.SEABORN_PALETTES["pastel"][1]

def test_get_phase_colors_override_not_present():
    phase_names = ["Alpha"]
    override = {"Beta": "red"}
    colors = get_phase_colors(phase_names, override=override)

    assert "Beta" not in colors
    assert colors["Alpha"] == sns.palettes.SEABORN_PALETTES["pastel"][0]

def test_get_phase_colors_duplicate_handling():
    phase_names = ["Alpha", "Beta", "Gamma"]
    pastel = sns.palettes.SEABORN_PALETTES["pastel"]
    alpha_default = pastel[0]
    beta_default = pastel[1]

    override = {"Alpha": beta_default}
    colors = get_phase_colors(phase_names, override=override)

    assert colors["Alpha"] == beta_default
    assert colors["Beta"] == alpha_default
    assert colors["Gamma"] == pastel[2]

def test_get_phase_colors_none_override():
    phase_names = ["Alpha", "Beta"]
    colors = get_phase_colors(phase_names)

    pastel = sns.palettes.SEABORN_PALETTES["pastel"]
    assert colors["Alpha"] == pastel[0]
    assert colors["Beta"] == pastel[1]

def test_get_phase_colors_more_than_palette_size():
    pastel = sns.palettes.SEABORN_PALETTES["pastel"]
    phase_names = [f"phase{i}" for i in range(len(pastel) + 3)]

    colors = get_phase_colors(phase_names)

    assert set(phase_names) == set(colors)
    for i, name in enumerate(sorted(phase_names)):
        assert colors[name] == pastel[i % len(pastel)]

def test_get_phase_colors_multiple_duplicates():
    phase_names = ["A", "B", "C", "D"]
    pastel = sns.palettes.SEABORN_PALETTES["pastel"]

    # Let's say we override A to use B's color and C to use D's color
    override = {"A": pastel[1], "C": pastel[3]}
    colors = get_phase_colors(phase_names, override)

    # A uses B's color
    assert colors["A"] == pastel[1]
    # B uses A's default color
    assert colors["B"] == pastel[0]

    # C uses D's color
    assert colors["C"] == pastel[3]
    # D uses C's default color
    assert colors["D"] == pastel[2]


# ---------------------------------------------------------------------------
# One phase, one hue across plotters
# ---------------------------------------------------------------------------

PASTEL = sns.palettes.SEABORN_PALETTES["pastel"]
MUTED = sns.palettes.SEABORN_PALETTES["muted"]


def test_get_phase_colors_sorts_names():
    assert get_phase_colors(["Gamma", "Alpha", "Beta"]) == get_phase_colors(["Alpha", "Beta", "Gamma"])
    assert get_phase_colors(["Beta", "Alpha"])["Alpha"] == PASTEL[0]


def test_get_phase_colors_palette():
    colors = get_phase_colors(["Alpha", "Beta"], palette="muted")
    assert colors == {"Alpha": MUTED[0], "Beta": MUTED[1]}


@pytest.fixture(scope="module")
def eutectic_cut(eutectic_phases):
    """Isothermal cut above the eutectic; the liquid is stable throughout, so its rows
    come first and the frame's phase order (liquid, hcp, fcc) is neither the input
    order nor sorted."""
    return ldc.calc_phase_diagram(eutectic_phases, Ts=[900.0], mu=200, keep_unstable=True)


def test_phase_colors_shared_across_plots(eutectic_diagram, eutectic_cut):
    names = sorted(eutectic_diagram.phase.unique())
    assert list(eutectic_cut.phase.unique()) != names
    assert list(eutectic_diagram.query("stable").phase.unique()) != names

    fig, (ax2d, ax1d, axf) = plt.subplots(1, 3)
    try:
        plot_phase_diagram(eutectic_diagram, ax=ax2d)
        plot_1d_mu_phase_diagram(eutectic_cut, ax=ax1d)
        plt.sca(axf)
        plot_excess_free_energy(eutectic_cut)

        fills = {p.get_label(): to_rgb(p.get_facecolor()) for p in ax2d.patches}
        lines = {k: to_rgb(v) for k, v in ax1d._landau_phase_colors.items()}
        excess = {t.get_text(): to_rgb(t.get_color()) for t in axf.texts if t.get_text() in names}
        for i, name in enumerate(names):
            assert fills[name] == to_rgb(PASTEL[i]), name
            assert lines[name] == to_rgb(MUTED[i]), name
            assert excess[name] == to_rgb(MUTED[i]), name
    finally:
        plt.close(fig)


def test_1d_color_override(eutectic_cut):
    fig, ax = plt.subplots()
    try:
        plot_1d_mu_phase_diagram(eutectic_cut, ax=ax, color_override={"liquid": "red"})
        colors = {k: to_rgb(v) for k, v in ax._landau_phase_colors.items()}
        assert colors["liquid"] == to_rgb("red")
        assert colors["fcc"] == to_rgb(MUTED[0])
        assert colors["hcp"] == to_rgb(MUTED[1])
    finally:
        plt.close(fig)
