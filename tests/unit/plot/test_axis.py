import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from landau.plot import _set_axis_for


def _ax():
    return MagicMock()


def _df(mus):
    return pd.DataFrame({"mu": mus})


class TestSetAxisForConcentration:
    def test_sets_xlim_zero_one(self):
        ax = _ax()
        _set_axis_for("c", pd.DataFrame(), None, ax)
        ax.set_xlim.assert_called_once_with(0, 1)

    def test_default_xlabel(self):
        ax = _ax()
        _set_axis_for("c", pd.DataFrame(), None, ax)
        ax.set_xlabel.assert_called_once_with("$c$")

    def test_with_element(self):
        ax = _ax()
        _set_axis_for("c", pd.DataFrame(), "Fe", ax)
        ax.set_xlabel.assert_called_once_with(r"$c_\mathrm{Fe}$")


class TestSetAxisForMu:
    def test_default_xlabel(self):
        ax = _ax()
        _set_axis_for("mu", _df([0.1, 0.2, 0.3]), None, ax)
        ax.set_xlabel.assert_called_once_with(r"$\Delta\mu$ [eV]")

    def test_with_element(self):
        ax = _ax()
        _set_axis_for("mu", _df([0.1, 0.2]), "Al", ax)
        ax.set_xlabel.assert_called_once_with(r"$\Delta\mu_\mathrm{Al}$ [eV]")

    def test_finite_mus_sets_xlim(self):
        ax = _ax()
        _set_axis_for("mu", _df([0.1, 0.5, 0.3]), None, ax)
        ax.set_xlim.assert_called_once_with(0.1, 0.5)

    def test_all_nonfinite_no_xlim(self):
        ax = _ax()
        _set_axis_for("mu", _df([np.inf, -np.inf, np.nan]), None, ax)
        ax.set_xlim.assert_not_called()

    def test_mixed_finite_nonfinite_xlim(self):
        ax = _ax()
        _set_axis_for("mu", _df([0.2, np.inf, 0.8, np.nan]), None, ax)
        ax.set_xlim.assert_called_once_with(0.2, 0.8)

    def test_element_with_nonfinite_mus(self):
        ax = _ax()
        _set_axis_for("mu", _df([np.inf, np.nan]), "Cu", ax)
        ax.set_xlim.assert_not_called()
        ax.set_xlabel.assert_called_once_with(r"$\Delta\mu_\mathrm{Cu}$ [eV]")


def test_unknown_axis_raises():
    ax = _ax()
    with pytest.raises(ValueError, match="Unknown coordinate system"):
        _set_axis_for("x", pd.DataFrame(), None, ax)
