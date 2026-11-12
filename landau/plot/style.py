"""The look of every plot in :mod:`landau.plot`.

:data:`theme` holds it. Its ``rc`` field holds rcParams: line width, tick and
legend font sizes, the legend frame, figure size and resolution. Its other
fields style one kind of plot element each, and set colour, line style and
font weight only. Line widths, marker sizes and font sizes come from the
rcParams ``lines.linewidth``, ``lines.markersize`` and ``font.size``.

Every plot function runs inside :func:`styled`, so it uses ``theme.rc``
instead of your rcParams for the keys in ``theme.rc``. Your rcParams set
everything else. Use ``styled`` on your own plots too::

    with styled():
        plt.plot(x, y)

Change the theme on the object itself, not in ``rcParams``::

    theme.rc["lines.linewidth"] = 1.5    # validated like rcParams
    del theme.rc["figure.figsize"]       # rcParams set the figure size again
    theme.rc.clear()                     # rcParams set everything again
    theme.tie_line["color"] = "tab:red"
    theme.reset()                        # back to the defaults

The same element looks the same in every plot. A metastable branch is dashed.
A tie line (the three-phase isotherm of a c-T diagram, the common tangent of an
excess-free-energy plot) is a faded black line. A faded dotted line marks a
transition along a 1d cut. Coexisting states are black dots of one size. A
phase name in bold labels the region where that phase is stable; in regular
weight it names a curve.
"""

import contextlib
from dataclasses import dataclass, field, fields

import matplotlib as mpl

__all__ = ["Theme", "styled", "theme"]


class _Rc(mpl.RcParams):
    """rcParams of a :class:`Theme`; keys and values are validated like ``rcParams``."""

    def __init__(self, params=()):
        super().__init__()
        self.update(params)

    def __delitem__(self, key):
        dict.__delitem__(self, key)


def _default_rc():
    return _Rc({
        "lines.linewidth": 2.0,
        "axes.titlesize": "medium",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "legend.fontsize": "small",
        "legend.frameon": False,
        "figure.figsize": [6.0, 4.5],
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


@dataclass
class Theme:
    """The look of the landau plots; see the module docstring."""

    #: rcParams applied inside :func:`styled`. Every key differs from matplotlib's default, because a key at its
    #: default would only override your rcParams.
    rc: _Rc = field(default_factory=_default_rc)
    #: The outline of a phase field in a 2d diagram.
    phase_boundary: dict = field(default_factory=lambda: {"edgecolor": "k"})
    #: A phase's curve where another phase is more stable.
    metastable: dict = field(default_factory=lambda: {"linestyle": "--"})
    #: A tie line: the three-phase isotherm of a c-T diagram, a common tangent of an excess-free-energy plot.
    tie_line: dict = field(default_factory=lambda: {"color": "k", "alpha": 0.5, "linestyle": "-"})
    #: The full-height line that marks a transition along a 1d cut.
    transition_guide: dict = field(default_factory=lambda: {"color": "k", "alpha": 0.5, "linestyle": ":"})
    #: A coexisting state: a transition point of a 1d cut, a triple point of a mu-T diagram, a tangent vertex.
    invariant_marker: dict = field(default_factory=lambda: {"color": "k", "marker": "o"})
    #: A line phase drawn as one dot in an excess-free-energy plot, this many times ``lines.markersize`` wide.
    line_phase_marker_scale: float = 1.5
    #: The zero line of an excess-free-energy plot; ``axes.linewidth`` wide unless it sets ``linewidth``.
    reference_line: dict = field(default_factory=lambda: {"color": ".5", "linestyle": "-"})
    #: A phase name on the region where the phase is stable: a 2d field, a stable window atop a 1d cut.
    region_label: dict = field(default_factory=lambda: {"fontsize": "small", "fontweight": "bold"})
    #: A phase name on its curve: the side labels of a 1d cut, the labels of an excess-free-energy plot.
    curve_label: dict = field(default_factory=lambda: {"fontsize": "small"})
    #: A number read off the plot: a transition temperature or chemical potential.
    value_label: dict = field(default_factory=lambda: {"fontsize": "small"})
    #: The stroke behind every label, which keeps it legible over fills and lines.
    label_outline: dict = field(default_factory=lambda: {"linewidth": 3, "foreground": "white"})
    #: The tick on the top spine of a 1d cut that marks a transition; keyword arguments of ``tick_params``.
    transition_tick: dict = field(default_factory=lambda: {"direction": "in", "length": 6})

    def reset(self):
        """Restore every field to its default."""
        default = Theme()
        for f in fields(self):
            setattr(self, f.name, getattr(default, f.name))


#: The theme every plot function uses.
theme = Theme()


@contextlib.contextmanager
def styled():
    """Apply ``theme.rc`` on top of your rcParams; use as ``with styled():`` or as the decorator ``@styled()``."""
    with mpl.rc_context(dict(theme.rc)):
        yield


_DASH_PATTERNS = {"--": "dashed", "dashed": "dashed", ":": "dotted", "dotted": "dotted", "-.": "dashdot",
                  "dashdot": "dashdot"}


def _stability_dashes() -> dict:
    """Seaborn ``dashes=`` mapping for a ``style="stable"`` lineplot, drawn like ``theme.metastable``."""
    linestyle = theme.metastable.get("linestyle", "-")
    if isinstance(linestyle, tuple):
        dashes = tuple(linestyle[1])
    elif linestyle in ("-", "solid"):
        dashes = ""
    else:
        dashes = tuple(mpl.rcParams[f"lines.{_DASH_PATTERNS[linestyle]}_pattern"])
    return {True: "", False: dashes}


def _reference_line() -> dict:
    """Line kwargs of a zero line, ``axes.linewidth`` wide unless ``theme.reference_line`` sets a width."""
    return {"linewidth": mpl.rcParams["axes.linewidth"], **theme.reference_line}


def _line_phase_markersize() -> float:
    return theme.line_phase_marker_scale * mpl.rcParams["lines.markersize"]
