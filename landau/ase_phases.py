import numpy as np
from dataclasses import dataclass
from .phases import AbstractLinePhase

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm("ASE is required to use ASE phase wrappers. Install with pip install 'landau[ase]'") as ase_alarm:
    from ase.thermochemistry import ThermoChem

@dataclass(frozen=True)
class ASEThermoPhase(AbstractLinePhase):
    """
    Phase wrapper for ASE's ThermoChem classes.
    """
    fixed_concentration: float
    thermochem: 'ThermoChem'
    use_gibbs: bool = False
    pressure: float = 0.0

    @ase_alarm
    def __post_init__(self, *args, **kwargs):
        pass

    @property
    def line_concentration(self):
        return self.fixed_concentration

    def line_free_energy(self, T):
        if self.use_gibbs:
            func = np.vectorize(
                lambda t: self.thermochem.get_gibbs_energy(t, pressure=self.pressure, verbose=False),
                otypes=[float]
            )
        else:
            func = np.vectorize(
                lambda t: self.thermochem.get_helmholtz_energy(t, verbose=False),
                otypes=[float]
            )
        return func(T)
