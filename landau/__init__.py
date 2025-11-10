from .phases import (
    LinePhase,
    TemperatureDepandantLinePhase,
    TemperatureDependentLinePhase,
    IdealSolution,
    RegularSolution,
    InterpolatingPhase,
    SlowInterpolatingPhase
)

from .interpolate import (
    ConcentrationInterpolator,
    TemperatureInterpolator,
    SGTE,
    PolyFit,
    RedlichKister,
    SoftplusFit,
)

from .plot import plot_phase_diagram

try:
    from ._version import __version__
except ImportError:
    __version__ = "dev"
