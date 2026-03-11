import pytest
import seaborn as sns
from landau.plot import get_phase_colors

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
