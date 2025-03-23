from .phases import (
    LinePhase,
    TemperatureDepandantLinePhase,
    TemperatureDependentLinePhase,
    IdealSolution,
    RegularSolution,
    InterpolatingPhase,
)

from .plot import plot_phase_diagram

try:
    from ._version import __version__
except ImportError:
    __version__ = "dev"
