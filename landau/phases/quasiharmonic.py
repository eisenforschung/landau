"""Quasi-harmonic line phases evaluated directly from phonon thermal properties.

The free energy is computed, not fitted: :class:`PhonopyQuasiHarmonicPhase` holds one
phonopy :class:`~phonopy.phonon.thermal_properties.ThermalProperties` per sampled volume
and evaluates

.. math::

    F(T) = \\min_V \\left[ E_\\mathrm{static}(V) + F_\\mathrm{vib}(V, T) \\right]

on demand at any temperature.  Both steps are cheap -- the vibrational part is a sum
over modes, the minimisation an equation-of-state fit through the sampled volumes -- so
there is no temperature grid to choose and no interpolator to fit.

Nothing here recomputes a mode sum.  ``ThermalProperties.temperatures`` is a plain
setter and ``run()`` recomputes, so the phase resets the temperature on the caller's own
object and reads the result back; the statistics, the cutoff convention and the physical
constants are phonopy's, chosen by the caller when they ran
:meth:`~phonopy.Phonopy.run_thermal_properties`.
"""

import pickle
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "phonopy is required to use quasi-harmonic phases. Install with pip install 'landau[phonopy]'"
) as phonopy_alarm:
    from phonopy.physical_units import get_physical_units
    from phonopy.qha.eos import fit_to_eos, get_eos

from ..interpolate.basic import _scalarize
from . import AbstractLinePhase

__all__ = [
    "DynamicalInstabilityWarning",
    "EosExtrapolationWarning",
    "PhonopyQuasiHarmonicPhase",
]


class DynamicalInstabilityWarning(UserWarning):
    """A sampled volume carries imaginary modes and was dropped from the fit."""


class EosExtrapolationWarning(UserWarning):
    """Thermal expansion carried the equilibrium volume outside the sampled range."""


@phonopy_alarm
def _lowest_frequency(thermal_properties):
    """Lowest mode of a ``ThermalProperties``, in THz, negative for an imaginary one.

    Reaching past phonopy's public API on purpose.  A ``ThermalProperties`` keeps the
    frequencies it was built from privately, in eV, and exposes no accessor for them --
    ``number_of_modes`` and ``number_of_integrated_modes`` only count how many survived
    the cutoff, which for a Gamma-centred mesh already excludes the three acoustic
    branches of a perfectly stable crystal.  Without this the dynamic-instability screen
    below cannot run at all, and an imaginary-mode volume joins the equation-of-state fit
    with a smooth plausible free energy that the mode sum produced by dropping exactly the
    modes that make it wrong.  The upper ``phonopy`` bound is pinned to the minor release
    so a new one arrives as a dependabot pull request and CI checks this against it;
    ``test_lowest_frequency_matches_the_mesh`` is what fails if the attribute moves.
    """
    return float(np.min(thermal_properties._frequencies)) / get_physical_units().THzToEv


@dataclass(frozen=True)
class PhonopyQuasiHarmonicPhase(AbstractLinePhase):
    """A fixed-concentration phase whose free energy is evaluated from phonon spectra.

    Give one phonopy :class:`~phonopy.phonon.thermal_properties.ThermalProperties` per
    sampled volume, together with the volume and static energy of the cell each was
    computed for, or build the phase from live :class:`~phonopy.Phonopy` objects with
    :meth:`from_phonopy`.  At every requested temperature the free energy of each volume
    comes from phonopy's mode sum, an equation of state is fitted through the results and
    its minimum is returned -- one number per temperature, computed rather than
    interpolated, at any temperature the caller asks for.

    Compared with sampling the same calculation onto a grid and handing that to
    :class:`~landau.phases.TemperatureDependentLinePhase`, this removes both the grid and
    the fit.  On quasi-harmonic data the default ``SGTE(3)`` leaves residuals of several
    meV/atom (issue #427), and a shallow solid-solid transition converts one meV/atom
    into tens of kelvin.

    Statistics (``classical``), the mode cutoff (``cutoff_frequency``) and the sampling
    mesh are the caller's, fixed when they ran
    :meth:`~phonopy.Phonopy.run_thermal_properties`; this class does not second-guess
    them.  It does reset ``temperatures`` on those objects and re-``run()`` them -- once at
    construction and again at every temperature asked of the phase -- so the
    ``ThermalProperties`` handed over are left holding whatever temperature was evaluated
    last.  Pass copies if the caller still needs their own results.

    **Primitive vs unit cell.**  ``volumes`` and ``energies`` describe whatever cell was
    actually relaxed, while phonopy normalises a free energy per *primitive* cell.  These
    differ whenever the unit cell is not primitive -- phonopy's
    ``primitive_matrix="auto"`` reduces a cubic fcc cell by 4 and hcp/bcc by 2 -- and
    pairing them without converting silently rescales every vibrational free energy by
    that factor.  Both counts are therefore required and each part is normalised by its
    own before they are added.

    Two ways a quasi-harmonic fit degrades without saying so, and what happens here:

    **Dynamically unstable volumes.**  Imaginary modes are dropped by the mode sum rather
    than raised on, so a volume with unstable branches still returns a smooth, plausible
    free energy that then corrupts the equation-of-state fit every other volume feeds.
    Volumes whose lowest mode falls below :attr:`min_frequency` are excluded and reported
    through :attr:`lowest_frequencies`, with a :class:`DynamicalInstabilityWarning` at
    construction.  See :func:`_lowest_frequency` for how that mode is recovered from a
    ``ThermalProperties``, which does not expose it.

    **Extrapolation past the sampled volumes.**  Once thermal expansion carries the
    equilibrium volume above the largest volume sampled, the equation of state is
    extrapolating and degrades without any sign of it.  :meth:`line_free_energy` returns
    NaN there rather than a number, which keeps the phase out of the ``idxmin`` in
    :func:`~landau.calculate.calc_phase_diagram` -- the phase is simply absent above its
    ceiling.  :meth:`max_temperature` reports where that ceiling is; sampling wider
    volumes is the only way to raise it.
    """

    fixed_concentration: float
    """The fixed concentration of the phase."""
    thermal_properties: tuple
    """One phonopy :class:`~phonopy.phonon.thermal_properties.ThermalProperties` per
    sampled volume, already run by :meth:`~phonopy.Phonopy.run_thermal_properties`.  At
    least four dynamically stable ones are needed to fit the four equation-of-state
    parameters."""
    volumes: tuple
    """Volume of the cell each entry of :attr:`energies` refers to, in cubic Angstrom."""
    energies: tuple
    """Static (electronic) energy of that same cell, in eV."""
    atoms_per_cell: int
    """Number of atoms in the cell :attr:`volumes` and :attr:`energies` refer to."""
    atoms_per_primitive_cell: int
    """Number of atoms in the primitive cell the phonons were computed on; phonopy
    reports a free energy per primitive cell."""
    eos: str = "vinet"
    """Equation of state to minimise over volume; one of ``"vinet"``,
    ``"birch_murnaghan"`` or ``"murnaghan"``, passed to :func:`phonopy.qha.eos.get_eos`."""
    min_frequency: float = -0.05
    """Volumes whose lowest mode falls below this, in THz, are treated as dynamically
    unstable and excluded from the fit.  Slightly negative rather than zero because the
    acoustic branches at Gamma are numerically noisy in both directions."""
    _key: tuple = field(default=(), init=False, repr=False)
    _hash: int = field(default=0, init=False, repr=False)

    @phonopy_alarm
    def __post_init__(self):
        for name in ("thermal_properties", "volumes", "energies"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "atoms_per_cell", int(self.atoms_per_cell))
        object.__setattr__(self, "atoms_per_primitive_cell", int(self.atoms_per_primitive_cell))
        lengths = {"volumes": len(self.volumes), "energies": len(self.energies)}
        if set(lengths.values()) != {len(self.thermal_properties)}:
            raise ValueError(
                f"thermal_properties, volumes and energies must be parallel sequences, but got "
                f"{len(self.thermal_properties)} thermal_properties against {lengths}"
            )
        if self.eos not in ("vinet", "birch_murnaghan", "murnaghan"):
            raise ValueError(f"eos must be one of 'vinet', 'birch_murnaghan', 'murnaghan', got {self.eos!r}")
        for name in ("atoms_per_cell", "atoms_per_primitive_cell"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if any(v <= 0 for v in self.volumes):
            raise ValueError(f"volumes must be positive, got {self.volumes}")
        per_atom = self.volumes_per_atom
        if len(set(per_atom)) != len(per_atom):
            raise ValueError(f"the sampled volumes must be distinct, got {per_atom} per atom")

        object.__setattr__(self, "_lowest", tuple(_lowest_frequency(tp) for tp in self.thermal_properties))
        unstable = tuple(i for i, f in enumerate(self._lowest) if f < self.min_frequency)
        stable = tuple(sorted((i for i in range(len(per_atom)) if i not in unstable), key=lambda i: per_atom[i]))
        if len(stable) < 4:
            raise ValueError(
                f"{len(stable)} dynamically stable volume(s) out of {len(per_atom)}, but fitting "
                f"{self.eos} needs at least four; lowest frequencies (THz) were {list(self._lowest)}"
            )
        if unstable:
            warnings.warn(
                f"{self.name}: dropped {len(unstable)} of {len(per_atom)} volumes carrying modes below "
                f"{self.min_frequency} THz (volumes per atom {[per_atom[i] for i in unstable]}, lowest "
                f"frequencies {[self._lowest[i] for i in unstable]}); the equation of state is "
                f"fitted through the remaining {len(stable)}",
                DynamicalInstabilityWarning,
                stacklevel=2,
            )
        object.__setattr__(self, "_stable", stable)
        object.__setattr__(self, "_unstable", unstable)
        object.__setattr__(self, "_key", self._content_key())
        object.__setattr__(self, "_hash", hash(self._key))

    @classmethod
    def from_phonopy(cls, name, fixed_concentration, phonopys, energies, **kwargs):
        """Build a phase from live :class:`~phonopy.Phonopy` objects, one per volume.

        Each object must already carry thermal properties, i.e. the caller has run
        :meth:`~phonopy.Phonopy.run_thermal_properties` and so has chosen the mesh, the
        statistics and the mode cutoff.  The volume, both atom counts and the lowest
        sampled frequency are read off the objects.

        Args:
            name (str): name of the phase
            fixed_concentration (float): concentration of the phase
            phonopys (iterable of phonopy.Phonopy): one per sampled volume
            energies (iterable of float): static energy of each object's unit cell, in eV
            **kwargs: forwarded to the constructor (``eos``, ``min_frequency``, ...)

        Returns:
            PhonopyQuasiHarmonicPhase
        """
        phonopys = tuple(phonopys)
        missing = [i for i, p in enumerate(phonopys) if p.thermal_properties is None]
        if missing:
            raise ValueError(
                f"phonopy objects at positions {missing} carry no thermal properties; call "
                "run_thermal_properties(temperatures=...) on each one first"
            )
        atom_counts = {(len(p.unitcell), len(p.primitive)) for p in phonopys}
        if len(atom_counts) > 1:
            raise ValueError(
                "all phonopy objects must describe the same structure, but their "
                f"(atoms per unit cell, atoms per primitive cell) differ: {sorted(atom_counts)}"
            )
        atoms_per_cell, atoms_per_primitive_cell = atom_counts.pop()
        return cls(
            name,
            fixed_concentration,
            thermal_properties=tuple(p.thermal_properties for p in phonopys),
            volumes=tuple(p.unitcell.volume for p in phonopys),
            energies=tuple(energies),
            atoms_per_cell=atoms_per_cell,
            atoms_per_primitive_cell=atoms_per_primitive_cell,
            **kwargs,
        )

    def _content_key(self):
        """Everything that makes this phase the phase it is, computed once at construction.

        A :class:`~phonopy.phonon.thermal_properties.ThermalProperties` compares by
        identity, so it goes into the key as its pickled bytes -- the same route
        :class:`~landau.phases.AsePhase` takes for its ``ThermoChem``.  Those bytes carry
        the temperature the object was last run at, though, and this class resets that on
        every evaluation.  Running each one at 0 K first reduces the key to the
        frequencies, weights, cutoff and statistics, and freezing it here keeps it from
        drifting as the phase is used.
        """
        pickled = []
        for tp in self.thermal_properties:
            tp.temperatures = np.array([0.0])
            with np.errstate(over="ignore", invalid="ignore"):
                tp.run()
            pickled.append(pickle.dumps(tp))
        return (
            self.name,
            self.fixed_concentration,
            self.volumes,
            self.energies,
            self.atoms_per_cell,
            self.atoms_per_primitive_cell,
            self.eos,
            self.min_frequency,
            tuple(pickled),
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self._key == other._key

    @property
    def line_concentration(self):
        return self.fixed_concentration

    @property
    def volumes_per_atom(self):
        """:attr:`volumes` divided by :attr:`atoms_per_cell`, in the order given."""
        return tuple(v / self.atoms_per_cell for v in self.volumes)

    @property
    def energies_per_atom(self):
        """:attr:`energies` divided by :attr:`atoms_per_cell`, in the order given."""
        return tuple(e / self.atoms_per_cell for e in self.energies)

    @property
    def lowest_frequencies(self):
        """Lowest mode of each sampled volume, in THz, in the order given.

        Negative where a volume carries an imaginary branch; those below
        :attr:`min_frequency` are the ones excluded from the fit.
        """
        return self._lowest

    @property
    def sampled_volumes(self):
        """Volumes per atom the equation of state is fitted through, ascending."""
        return np.array([self.volumes_per_atom[i] for i in self._stable])

    @property
    def unstable_volumes(self):
        """Volumes per atom excluded as dynamically unstable, in the order given."""
        return np.array([self.volumes_per_atom[i] for i in self._unstable])

    def volume_free_energies(self, T):
        """Free energy per atom of each of :attr:`sampled_volumes`, in eV, at one temperature.

        Static plus vibrational, each normalised by its own atom count.  The vibrational
        part is phonopy's own mode sum, obtained by resetting the temperature on the
        stored :class:`~phonopy.phonon.thermal_properties.ThermalProperties` and running
        it again.

        Args:
            T (float): temperature in K, non-negative

        Returns:
            numpy.ndarray: one free energy per stable volume, ascending in volume
        """
        T = float(T)
        if T < 0:
            # phonopy's temperatures setter silently drops negative entries, which would
            # misalign the result against the requested temperatures
            raise ValueError(f"temperatures must be non-negative, got {T}")
        to_ev = 1 / get_physical_units().EvTokJmol
        static = self.energies_per_atom
        out = []
        for i in self._stable:
            tp = self.thermal_properties[i]
            tp.temperatures = np.array([T])
            with np.errstate(over="ignore", invalid="ignore"):
                tp.run()
            vibrational = tp.thermal_properties[1][0] * to_ev / self.atoms_per_primitive_cell
            out.append(static[i] + vibrational)
        return np.array(out)

    @lru_cache(maxsize=1024)
    def eos_parameters(self, T):
        """Equation-of-state parameters at one temperature.

        Fits ``self.eos`` through the per-atom free energies of :attr:`sampled_volumes`.

        Args:
            T (float): temperature in K

        Returns:
            tuple of float: ``(E_0, B_0, B'_0, V_0)`` -- the minimum free energy in eV per
            atom, the bulk modulus in eV per cubic Angstrom, its pressure derivative, and
            the equilibrium volume in cubic Angstrom per atom.  All four are NaN if the
            fit fails.
        """
        free_energies = self.volume_free_energies(T)
        # get_eos returns a closure that phonopy 3 calls as eos(v, *p) and phonopy 4 as
        # eos(v, p); handing it straight to fit_to_eos and never calling it keeps that
        # difference out of here
        try:
            parameters = fit_to_eos(self.sampled_volumes, free_energies, get_eos(self.eos))
        except (RuntimeError, TypeError) as error:
            warnings.warn(
                f"{self.name}: fitting {self.eos} to the {len(self._stable)} sampled volumes failed at "
                f"T = {T} K ({error}); the free energy is NaN there",
                RuntimeWarning,
                stacklevel=2,
            )
            return (np.nan,) * 4
        return tuple(float(p) for p in parameters)

    def _minimum(self, T):
        """``(F, V)`` per atom at one temperature, NaN outside the sampled volumes."""
        energy, _, _, volume = self.eos_parameters(T)
        if not np.isfinite(volume):
            # the fit failed and has already warned; do not blame extrapolation for it
            return np.nan, np.nan
        volumes = self.sampled_volumes
        if not volumes[0] <= volume <= volumes[-1]:
            warnings.warn(
                f"{self.name}: thermal expansion carried the equilibrium volume outside the sampled range "
                f"[{volumes[0]}, {volumes[-1]}] A^3/atom, where the equation of state only extrapolates; "
                "the free energy is NaN there. Call max_temperature() for the ceiling, or sample wider "
                "volumes to raise it",
                EosExtrapolationWarning,
                stacklevel=3,
            )
            return np.nan, np.nan
        return energy, volume

    def line_free_energy(self, T):
        """Free energy per atom in eV, minimised over volume.

        Args:
            T (float or array of float): temperature in K

        Returns:
            float or array of float: NaN wherever the equilibrium volume falls outside the
            sampled range
        """
        T = np.asarray(T, dtype=float)
        out = np.array([self._minimum(float(t))[0] for t in T.flat], dtype=float)
        return _scalarize(out.reshape(T.shape))

    def equilibrium_volume(self, T):
        """Volume per atom in cubic Angstrom that minimises the free energy.

        Args:
            T (float or array of float): temperature in K

        Returns:
            float or array of float: NaN wherever the fit would extrapolate
        """
        T = np.asarray(T, dtype=float)
        out = np.array([self._minimum(float(t))[1] for t in T.flat], dtype=float)
        return _scalarize(out.reshape(T.shape))

    def max_temperature(self, upper=5000.0, tolerance=1.0):
        """Highest temperature at which the equation of state still interpolates.

        Thermal expansion is monotonic in temperature for a stable solid, so the
        equilibrium volume leaves the sampled range once and does not come back; this
        bisects for that crossing.

        Args:
            upper (float): temperature in K to bracket the search from above; returned
                unchanged if the fit still interpolates there
            tolerance (float): bracket width in K at which to stop

        Returns:
            float: the ceiling in K, or NaN if the fit already extrapolates at ``T = 0``
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EosExtrapolationWarning)
            if np.isfinite(self._minimum(upper)[0]):
                return float(upper)
            lo, hi = 0.0, float(upper)
            if not np.isfinite(self._minimum(lo)[0]):
                return np.nan
            while hi - lo > tolerance:
                mid = 0.5 * (lo + hi)
                if np.isfinite(self._minimum(mid)[0]):
                    lo = mid
                else:
                    hi = mid
        return lo
