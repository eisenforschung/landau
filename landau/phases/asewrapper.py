import pickle

import numpy as np
from dataclasses import dataclass

from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm("ASE is required to use ASE phase wrappers. Install with pip install 'landau[ase]'") as ase_alarm:
    try:
        from ase.thermochemistry import ThermoChem
    except ImportError:
        # ASE >=3.28 renamed the base class.
        from ase.thermochemistry import BaseThermoChem as ThermoChem

from . import AbstractLinePhase

@dataclass(frozen=True)
class AsePhase(AbstractLinePhase):
    """
    Phase wrapper for ASE's ThermoChem classes.

    Equality and hashing compare ``thermochem`` by its pickled bytes so two
    ``AsePhase`` instances built from equivalent inputs compare equal even
    though ASE's ``ThermoChem`` defaults to identity-based equality.

    ``atoms_per_formula`` divides the energy returned by ``thermochem`` so the
    result is per atom (landau's convention).  Use 2 for an ASE ``IdealGasThermo``
    built around H₂ or O₂, 3 for CO₂, etc.; the default of 1 is correct
    when the ASE object already represents one atom or one per-atom formula unit
    (most ``HarmonicThermo`` setups, monatomic ``IdealGasThermo``).
    """
    fixed_concentration: float
    thermochem: 'ThermoChem'
    pressure: float | None = None
    atoms_per_formula: int = 1

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

        res = func(T) / self.atoms_per_formula
        if res.ndim == 0:
            return res.item()
        return res

    def _key(self):
        return (
            self.name,
            self.fixed_concentration,
            self.pressure,
            self.atoms_per_formula,
            pickle.dumps(self.thermochem),
        )

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())
