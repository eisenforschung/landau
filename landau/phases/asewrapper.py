import pickle

import numpy as np
from dataclasses import dataclass

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm("ASE is required to use ASE phase wrappers. Install with pip install 'landau[ase]'") as ase_alarm:
    from ase.thermochemistry import ThermoChem

from . import AbstractLinePhase

@dataclass(frozen=True)
class AsePhase(AbstractLinePhase):
    """
    Phase wrapper for ASE's ThermoChem classes.

    Equality and hashing compare ``thermochem`` by its pickled bytes so two
    ``AsePhase`` instances built from equivalent inputs compare equal even
    though ASE's ``ThermoChem`` defaults to identity-based equality.
    """
    fixed_concentration: float
    thermochem: 'ThermoChem'
    pressure: float | None = None

    @ase_alarm
    def __post_init__(self, *args, **kwargs):
        pass

    @property
    def line_concentration(self):
        return self.fixed_concentration

    def line_free_energy(self, T):
        # ASE ThermoChem subclasses do not uniformly expose both energy methods:
        # HarmonicThermo/CrystalThermo only define get_helmholtz_energy, while
        # IdealGasThermo only defines get_gibbs_energy. Prefer Helmholtz when
        # available (landau has no pressure concept yet); fall back to Gibbs at
        # self.pressure, defaulting to 1 atm.
        if hasattr(self.thermochem, "get_helmholtz_energy"):
            func = np.vectorize(
                lambda t: self.thermochem.get_helmholtz_energy(t, verbose=False),
                otypes=[float],
            )
        elif hasattr(self.thermochem, "get_gibbs_energy"):
            pressure = self.pressure if self.pressure is not None else 101325.0
            func = np.vectorize(
                lambda t: self.thermochem.get_gibbs_energy(t, pressure=pressure, verbose=False),
                otypes=[float],
            )
        else:
            raise TypeError(
                f"{type(self.thermochem).__name__} exposes neither get_helmholtz_energy "
                "nor get_gibbs_energy; cannot compute a free energy."
            )

        res = func(T)
        if res.ndim == 0:
            return res.item()
        return res

    def _key(self):
        return (
            self.name,
            self.fixed_concentration,
            self.pressure,
            pickle.dumps(self.thermochem),
        )

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())
