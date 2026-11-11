"""Shared drawing conventions for every plot in :mod:`landau.plot`.

Two layers:

* :data:`theme` holds the rcParams of ``landau.mplstyle`` next to this module
  (line width, tick and legend font sizes, legend frame, figure size). Every
  plot function applies all of it for the duration of its call through
  :func:`styled`, whatever the caller's rcParams hold for those keys; every
  other rcParam is the caller's. Change the theme with ``theme.update(...)``,
  hand a key back to ``rcParams`` with ``del theme[key]`` (``theme.clear()``
  for all of them) and restore it with ``theme.reset()``. Inside a call the
  theme reaches only what the call creates (lines, markers, text, and the
  figure when no ``ax`` is passed); axes passed in keep their tick fonts, and
  ``savefig.*`` applies only when the sheet is loaded globally with
  ``plt.style.use("landau.plot.landau")``.
* The dictionaries below name each kind of element the plots draw and carry
  only what the element *means* -- colour, line style, weight. Sizes are left
  out so that ``lines.linewidth`` / ``lines.markersize`` / ``font.size`` govern
  every plot alike; :func:`line_phase_markersize` is the one size derived from
  them.

The same element looks the same in every plot: a metastable branch is dashed
in the 1d cuts and in the excess-free-energy plot, a tie line (the three-phase
isotherm of a c-T diagram, a common tangent) is a faded black line, a
transition of a 1d cut is marked by a faded dotted guide, and coexisting
states are black dots of one size. A phase name set in bold labels the region
where that phase is stable; in regular weight it names a curve.
"""

import functools
from importlib.resources import files

import matplotlib as mpl

__all__ = [
    "CURVE_LABEL",
    "INVARIANT_MARKER",
    "METASTABLE",
    "REGION_LABEL",
    "STYLE",
    "TIE_LINE",
    "TRANSITION_GUIDE",
    "VALUE_LABEL",
    "Theme",
    "line_phase_markersize",
    "reference_line",
    "stability_dashes",
    "styled",
    "theme",
]

#: Name to pass to :func:`matplotlib.pyplot.style.use`.
STYLE = "landau.plot.landau"

_SHEET = dict(mpl.rc_params_from_file(
    str(files(__package__) / "landau.mplstyle"), use_default_template=False,
))


class Theme(mpl.RcParams):
    """rcParams applied inside every landau plot call; values are validated like ``rcParams``."""

    def __init__(self):
        super().__init__()
        self.reset()

    def __delitem__(self, key):
        dict.__delitem__(self, key)

    def reset(self):
        """Restore the keys and values of ``landau.mplstyle``."""
        self.clear()
        self.update(_SHEET)


#: The rcParams every plot function applies; see the module docstring.
theme = Theme()


def styled(func):
    """Run ``func`` with :data:`theme` applied on top of the caller's rcParams."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(dict(theme)):
            return func(*args, **kwargs)
    return wrapper


#: A phase's curve where another phase is more stable.
METASTABLE = {"linestyle": "--"}

#: A tie line between coexisting states: the three-phase isotherm of a c-T
#: diagram, a common tangent of an excess-free-energy plot.
TIE_LINE = {"color": "k", "alpha": 0.5, "linestyle": "-"}

#: The full-height guide marking a transition along a 1d cut.
TRANSITION_GUIDE = {"color": "k", "alpha": 0.5, "linestyle": ":"}

#: The coexisting states on an invariant: transition points of a 1d cut, triple
#: points of a mu-T diagram, tangent vertices of an excess-free-energy plot.
INVARIANT_MARKER = {"color": "k", "marker": "o"}

#: A phase's name labelling the region where it is stable: a field of a 2d
#: diagram, a stable window along the top of a 1d cut.
REGION_LABEL = {"fontsize": "small", "fontweight": "bold"}

#: A phase's name labelling its curve: the side labels of a 1d cut, the curve
#: labels of an excess-free-energy plot.
CURVE_LABEL = {"fontsize": "small"}

#: A number read off the diagram (a transition temperature or chemical potential).
VALUE_LABEL = {"fontsize": "small"}


def stability_dashes() -> dict[bool, str | tuple[float, ...]]:
    """Seaborn ``dashes=`` mapping for a ``style="stable"`` lineplot.

    Stable rows are solid; metastable rows use matplotlib's own ``"--"``
    pattern (``lines.dashed_pattern``), so a seaborn-drawn metastable branch is
    dashed exactly like one drawn with :data:`METASTABLE`.
    """
    return {True: "", False: tuple(mpl.rcParams["lines.dashed_pattern"])}


def reference_line() -> dict:
    """Style of a zero line the data is measured against.

    Solid grey at the weight of the axes frame (``axes.linewidth``), so it reads
    as part of the axes and never as a dashed metastable curve.
    """
    return {"color": ".5", "linestyle": "-", "linewidth": mpl.rcParams["axes.linewidth"]}


def line_phase_markersize() -> float:
    """Marker size (points) of a line phase drawn as a single dot.

    A line phase in the excess-free-energy plot is a whole phase shown as one
    point, so it is drawn larger than the invariant markers that sit on it:
    1.5 times ``lines.markersize``.
    """
    return 1.5 * mpl.rcParams["lines.markersize"]
