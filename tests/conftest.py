import os

import numpy as np
import pytest
from hypothesis import settings
from hypothesis.database import (
    DirectoryBasedExampleDatabase,
    GitHubArtifactDatabase,
    MultiplexedDatabase,
    ReadOnlyDatabase,
)

import landau.calculate as ldc
from landau.phases import LinePhase, IdealSolution, RegularSolution, kB


if os.environ.get("GITHUB_ACTIONS") == "true":
    _local = DirectoryBasedExampleDatabase(".hypothesis/examples")
    _shared = ReadOnlyDatabase(
        GitHubArtifactDatabase(
            "eisenforschung",
            "landau",
            artifact_name=os.environ.get(
                "HYPOTHESIS_ARTIFACT_NAME", "hypothesis-example-db"
            ),
        )
    )
    settings.register_profile("ci", database=MultiplexedDatabase(_local, _shared))
    settings.load_profile("ci")


@pytest.fixture
def two_phase_ideal():
    """Two terminal LinePhases bridged by an IdealSolution (A at c=0, B at c=1)."""
    l1 = LinePhase("A", 0, 0, 0)
    l2 = LinePhase("B", 1, 0.1, 0)
    sol = IdealSolution("sol", l1, l2)
    return [l1, l2, sol]


@pytest.fixture
def three_phase_regular_solution():
    """Three LinePhases (A, B, C) with a RegularSolution fitting through all three."""
    p1 = LinePhase("A", 0, 0)
    p2 = LinePhase("B", 1, 0)
    p3 = LinePhase("C", 0.5, 0)
    return RegularSolution(name="sol", phases=[p1, p2, p3])


@pytest.fixture(scope="module")
def eutectic_phases():
    """hcp / fcc / liquid ideal solutions (Basics.ipynb parameters) whose
    common tangent is a eutectic: one temperature where all three coexist."""
    fcc = IdealSolution(
        "fcc",
        LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * kB),
        LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * kB),
    )
    hcp = IdealSolution(
        "hcp",
        LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * kB),
        LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * kB),
    )
    liquid = IdealSolution(
        "liquid",
        LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * kB),
        LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * kB),
    )
    return [hcp, fcc, liquid]


@pytest.fixture(scope="module")
def eutectic_diagram(eutectic_phases):
    """Refined phase diagram of :func:`eutectic_phases`, carrying the
    ``Locus.TRIPLE`` rows of its eutectic invariant."""
    return ldc.calc_phase_diagram(eutectic_phases, np.linspace(200.0, 1000.0, 25), mu=50, refine=True)
