import numpy as np
from dataclasses import dataclass
from .phases import AbstractLinePhase

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm("ASE is required to use ASE phase wrappers. Install with pip install 'landau[ase]'") as ase_alarm:
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo, CrystalThermo

    @dataclass(frozen=True)
    class ASEIdealGasPhase(AbstractLinePhase):
        """
        Phase wrapper for ASE's IdealGasThermo class.
        """
        fixed_concentration: float
        thermochem: 'IdealGasThermo'
        pressure: float = 1e5

        @property
        def line_concentration(self):
            return self.fixed_concentration

        def line_free_energy(self, T):
            func = np.vectorize(
                lambda t: self.thermochem.get_gibbs_energy(t, pressure=self.pressure, verbose=False),
                otypes=[float]
            )
            return func(T)


    @dataclass(frozen=True)
    class ASEHarmonicPhase(AbstractLinePhase):
        """
        Phase wrapper for ASE's HarmonicThermo class.
        """
        fixed_concentration: float
        thermochem: 'HarmonicThermo'

        @property
        def line_concentration(self):
            return self.fixed_concentration

        def line_free_energy(self, T):
            func = np.vectorize(
                lambda t: self.thermochem.get_helmholtz_energy(t, verbose=False),
                otypes=[float]
            )
            return func(T)


    @dataclass(frozen=True)
    class ASECrystalPhase(AbstractLinePhase):
        """
        Phase wrapper for ASE's CrystalThermo class.
        """
        fixed_concentration: float
        thermochem: 'CrystalThermo'

        @property
        def line_concentration(self):
            return self.fixed_concentration

        def line_free_energy(self, T):
            func = np.vectorize(
                lambda t: self.thermochem.get_helmholtz_energy(t, verbose=False),
                otypes=[float]
            )
            return func(T)
