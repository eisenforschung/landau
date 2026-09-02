from .phases import (
    LinePhase,
    TemperatureDepandantLinePhase,
    TemperatureDependentLinePhase,
    IdealSolution,
    RegularSolution,
    InterpolatingPhase,
    SlowInterpolatingPhase,
    FastInterpolatingPhase,
    Surface2DInterpolatingPhase,
    CompoundEnergyPhase,
    AsePhase,
)

from .interpolate import (
    ConcentrationInterpolator,
    TemperatureInterpolator,
    SurfaceInterpolator,
    FittedSurface,
    SGTE,
    PolyFit,
    SplineFit,
    RedlichKister,
    SoftplusFit,
    CalphadSurface2DInterpolator,
    WhitneySurface2DInterpolator,
    SoftplusSurface2DInterpolator,
)

from .plot import plot_phase_diagram, plot_excess_free_energy

try:
    from ._version import __version__
except ImportError:
    __version__ = "dev"
