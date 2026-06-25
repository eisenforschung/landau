from .phases import (
    LinePhase,
    TemperatureDepandantLinePhase,
    TemperatureDependentLinePhase,
    IdealSolution,
    RegularSolution,
    InterpolatingPhase,
    SlowInterpolatingPhase,
    FastInterpolatingPhase,
    AsePhase,
)

from .interpolate import (
    ConcentrationInterpolator,
    TemperatureInterpolator,
    SGTE,
    PolyFit,
    SplineFit,
    RedlichKister,
    SoftplusFit,
)

from .plot import plot_phase_diagram, plot_excess_free_energy

from .features import Locus, Domain, Coexistence, PhaseDiagram

try:
    from ._version import __version__
except ImportError:
    __version__ = "dev"
