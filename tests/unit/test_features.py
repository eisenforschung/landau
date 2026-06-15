import numpy as np
import pandas as pd
import pytest

from landau.calculate import calc_phase_diagram
from landau.features import Coexistence, Domain, Locus, PhaseDiagram
from landau.phases import IdealSolution, LinePhase


@pytest.fixture
def regular_frame(three_phase_regular_solution):
    """A refined c-T frame for a three-LinePhase regular solution (A, B, C + sol)."""
    sol = three_phase_regular_solution
    phases = [*sol.phases, sol]
    df = calc_phase_diagram(phases, np.linspace(200, 1100, 10), 20, refine=True)
    return df, {p.name: p for p in phases}


def _names(coex):
    """Coexistence phase tuple as plain names regardless of how it was built."""
    return tuple(getattr(p, "name", p) for p in coex.phases)


def test_from_frame_requires_columns():
    with pytest.raises(ValueError, match="missing required column"):
        PhaseDiagram.from_frame(pd.DataFrame({"phase": ["A"]}))


def test_frame_is_escape_hatch(regular_frame):
    df, _ = regular_frame
    pd_ = PhaseDiagram.from_frame(df)
    assert pd_.frame is df


def test_phases_without_mapping_are_names(regular_frame):
    df, _ = regular_frame
    pd_ = PhaseDiagram.from_frame(df)
    # only stable phases appear in the frame; C is covered by the solution
    assert pd_.phases == set(df["phase"].unique())
    assert pd_.phases <= {"A", "B", "C", "sol"}


def test_phases_with_mapping_is_the_mapping(regular_frame):
    df, mapping = regular_frame
    pd_ = PhaseDiagram.from_frame(df, mapping)
    assert pd_.phases is mapping


def test_domains_are_single_connected_phase_regions(regular_frame):
    df, _ = regular_frame
    pd_ = PhaseDiagram.from_frame(df)
    domains = pd_.domains()
    assert domains, "expected at least one domain"
    assert all(isinstance(d, Domain) for d in domains)
    for d in domains:
        # a domain is exactly one phase, drawn from interior stable samples
        assert set(d.frame["phase"].unique()) == {d.phase}
        assert (d.frame["locus"] == Locus.INTERIOR).all()
        assert d.frame["stable"].all()
        assert len(d.frame) > 0
    # the three terminal/solution phases that are stable each own a domain
    assert {d.phase for d in domains} == set(df.query("stable")["phase"].unique())


def test_domains_resolve_to_phase_objects_with_mapping(regular_frame):
    df, mapping = regular_frame
    pd_ = PhaseDiagram.from_frame(df, mapping)
    for d in pd_.domains():
        assert d.phase is mapping[d.phase.name]


def test_disconnected_phase_yields_multiple_domains():
    """A solution stable in two disjoint windows must split into two domains."""
    l1 = LinePhase("A", 0, 0, 0)
    l2 = LinePhase("B", 1, 0.1, 0)
    sol = IdealSolution("sol", l1, l2)
    df = calc_phase_diagram([l1, l2, sol], np.linspace(100, 1200, 12), 20, refine=True)
    pd_ = PhaseDiagram.from_frame(df)
    sol_domains = [d for d in pd_.domains() if d.phase == "sol"]
    assert len(sol_domains) == 2
    # the split is geometric: the two windows occupy disjoint concentration ranges
    lo, hi = sorted(sol_domains, key=lambda d: d.frame["c"].mean())
    assert lo.frame["c"].max() < hi.frame["c"].min()


def test_coexistence_classification():
    """codimension and miscibility-gap classification off the phase multiset."""
    empty = pd.DataFrame({"phase": [], "locus": []})
    line = Coexistence(("A", "sol"), empty)
    gap = Coexistence(("sol", "sol"), empty)
    triple = Coexistence(("A", "B", "C"), empty)

    assert line.codimension == 1 and not line.is_miscibility_gap
    assert gap.codimension == 1 and gap.is_miscibility_gap
    assert triple.codimension == 2 and not triple.is_miscibility_gap


def _coex_row(phase, locus, boundary_id, refined, T, c):
    return {
        "phase": phase, "stable": True, "locus": locus, "border": True,
        "boundary_id": boundary_id, "refined": refined, "T": T, "mu": 0.1, "c": c,
    }


def test_miscibility_gap_parsed_as_doubled_phase():
    """One phase splitting into two branches parses to a (X, X) k=2 coexistence."""
    rows = [
        _coex_row("X", Locus.BOUNDARY, 0, "miscibility-gap", T, c)
        for T in (300.0, 350.0, 400.0)
        for c in (0.2, 0.8)
    ]
    pd_ = PhaseDiagram.from_frame(pd.DataFrame(rows))
    (gap,) = pd_.coexistences
    assert _names(gap) == ("X", "X")
    assert gap.codimension == 1
    assert gap.is_miscibility_gap


def test_triple_point_parsed_as_three_phases():
    rows = [
        _coex_row(ph, Locus.TRIPLE, 0, "delaunay-triple", T, c)
        for T in (300.0, 400.0)
        for ph, c in (("A", 0.1), ("B", 0.5), ("C", 0.9))
    ]
    pd_ = PhaseDiagram.from_frame(pd.DataFrame(rows))
    (triple,) = pd_.coexistences
    assert _names(triple) == ("A", "B", "C")
    assert triple.codimension == 2
    assert not triple.is_miscibility_gap


def test_boundary_id_collision_across_refiners_kept_separate():
    """boundary_id restarts per refiner, so (refined, boundary_id) keys a line."""
    rows = [
        _coex_row("A", Locus.BOUNDARY, 0, "clausius-clapeyron", 300.0, 0.1),
        _coex_row("sol", Locus.BOUNDARY, 0, "clausius-clapeyron", 300.0, 0.4),
        _coex_row("X", Locus.BOUNDARY, 0, "miscibility-gap", 300.0, 0.2),
        _coex_row("X", Locus.BOUNDARY, 0, "miscibility-gap", 300.0, 0.8),
    ]
    pd_ = PhaseDiagram.from_frame(pd.DataFrame(rows))
    cox = pd_.coexistences
    assert len(cox) == 2  # not merged into one despite the shared boundary_id
    assert {_names(c) for c in cox} == {("A", "sol"), ("X", "X")}


def test_coexistences_parse_boundary_lines(regular_frame):
    df, _ = regular_frame
    pd_ = PhaseDiagram.from_frame(df)
    cox = pd_.coexistences
    assert cox, "expected refined coexistence lines"
    assert all(isinstance(c, Coexistence) for c in cox)
    # the regular solution coexists with each terminal line phase
    by_names = {_names(c): c for c in cox}
    assert ("A", "sol") in by_names
    assert ("B", "sol") in by_names
    for c in cox:
        assert c.codimension == 1
        assert (c.frame["locus"] != Locus.INTERIOR).all()
        assert len(c.frame) > 0


def test_coexistence_phases_resolve_with_mapping(regular_frame):
    df, mapping = regular_frame
    pd_ = PhaseDiagram.from_frame(df, mapping)
    line = next(c for c in pd_.coexistences if _names(c) == ("A", "sol"))
    assert line.phases == (mapping["A"], mapping["sol"])


def test_no_coexistences_without_refinement(three_phase_regular_solution):
    sol = three_phase_regular_solution
    phases = [*sol.phases, sol]
    df = calc_phase_diagram(phases, np.linspace(200, 1100, 6), 15, refine=False)
    pd_ = PhaseDiagram.from_frame(df)
    assert pd_.coexistences == []
    # unrefined diagram still has single-phase domains
    assert pd_.domains()
