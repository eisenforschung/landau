import os

import pytest
from hypothesis import settings
from hypothesis.database import (
    DirectoryBasedExampleDatabase,
    GitHubArtifactDatabase,
    MultiplexedDatabase,
    ReadOnlyDatabase,
)

from landau.phases import LinePhase, IdealSolution, RegularSolution


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
