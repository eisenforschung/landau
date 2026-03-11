import pytest
import numpy as np
from landau.calculate import find_one_point
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
