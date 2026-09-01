"""Quasi-harmonic line phases evaluated directly from phonon spectra.

The free energy is computed, not fitted: :class:`PhonopyQuasiHarmonicPhase` holds one
phonon spectrum per sampled volume and evaluates

.. math::

    F(T) = \\min_V \\left[ E_\\mathrm{static}(V) + F_\\mathrm{vib}(V, T) \\right]

on demand at any temperature.  Both steps are cheap -- the vibrational part is a sum
over modes, the minimisation an equation-of-state fit through the sampled volumes -- so
there is no temperature grid to choose and no interpolator to fit.

The mode sum itself is phonopy's, not a reimplementation: a spectrum is presented to
:class:`phonopy.phonon.thermal_properties.ThermalProperties` through the four
attributes it reads off a q-point mesh, so the statistics, the cutoff convention and the
physical constants are phonopy's own.
"""

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from pyiron_snippets.import_alarm import ImportAlarm

with ImportAlarm(
    "phonopy is required to use quasi-harmonic phases. Install with pip install 'landau[phonopy]'"
) as phonopy_alarm:
    from phonopy.phonon.thermal_properties import ThermalProperties
    from phonopy.physical_units import get_physical_units
    from phonopy.qha.eos import fit_to_eos, get_eos

from ..interpolate.basic import _scalarize
from . import AbstractLinePhase

__all__ = [
    "DynamicalInstabilityWarning",
    "EosExtrapolationWarning",
    "PhononSpectrum",
    "PhonopyQuasiHarmonicPhase",
]


class DynamicalInstabilityWarning(UserWarning):
    """A sampled volume carries imaginary modes and was dropped from the fit."""


class EosExtrapolationWarning(UserWarning):
    """Thermal expansion carried the equilibrium volume outside the sampled range."""


class _MeshShim:
    """The whole of the q-point mesh interface that ``ThermalProperties`` consumes.

    ``ThermalPropertiesBase.__init__`` reads exactly ``mesh.frequencies``,
    ``mesh.weights``, ``mesh.eigenvectors`` (phonopy 3 only, and only stored) and
    ``mesh.dynamical_matrix.primitive.Z``.  Presenting stored arrays through those four
    names drives phonopy's own kernel without a live
    :class:`~phonopy.Phonopy` object, so the free energy landau reports is phonopy's
    number rather than a second implementation of it.

    ``Z`` -- formula units per primitive cell -- reaches nothing on the thermal-property
    path (it is kept for yaml output only), so it is fixed at 1.
    """

    def __init__(self, frequencies, weights):
        self.frequencies = np.array(frequencies, dtype="double", order="C")
        self.weights = np.array(weights, dtype="int_")
        self.eigenvectors = None
        self.dynamical_matrix = SimpleNamespace(primitive=SimpleNamespace(Z=1))


@dataclass(frozen=True)
class PhononSpectrum:
    """One sampled volume of a quasi-harmonic calculation.

    Everything needed to evaluate the harmonic free energy at that volume: phonon
    frequencies on a q-point mesh, the mesh weights, and the static (electronic) energy
    of the same structure.  Plain arrays are the primitive form on purpose -- a pickled
    :class:`~phonopy.Phonopy` runs to hundreds of megabytes where the spectrum is a few
    kilobytes, so the spectrum is the only form that can be cached per volume.
    :meth:`from_phonopy` builds one from a live phonopy object.

    ``frequencies`` and ``weights`` come from a **primitive** cell, while ``volume`` and
    ``energy`` describe whatever cell was actually relaxed.  These differ whenever the
    unit cell is not primitive -- phonopy's ``primitive_matrix="auto"`` reduces a cubic
    fcc cell by 4 and hcp/bcc by 2 -- and pairing them without converting silently
    rescales every vibrational free energy by that factor.  The two are therefore
    normalised separately, each by its own atom count, before they are added:
    ``atoms_per_primitive_cell`` is derived from the band count
    (``frequencies.shape[1] // 3``) and so cannot be got wrong, while ``atoms_per_cell``
    must be stated.
    """

    frequencies: np.ndarray
    """Phonon frequencies in THz, shape ``(n_qpoints, 3 * atoms_per_primitive_cell)``.
    Imaginary modes are negative, as phonopy reports them."""
    weights: np.ndarray
    """Multiplicity of each q point, shape ``(n_qpoints,)``; integers."""
    volume: float
    """Volume of the cell ``energy`` refers to, in cubic Angstrom."""
    energy: float
    """Static (electronic) energy of that same cell, in eV."""
    atoms_per_cell: int
    """Number of atoms in the cell ``volume`` and ``energy`` refer to."""
    _hash: int = field(default=0, init=False, repr=False)

    @phonopy_alarm
    def __post_init__(self):
        def to_ro_numpy(a, dtype):
            a = np.array(a, dtype=dtype)
            a.flags.writeable = False
            return a

        weights = np.asarray(self.weights)
        if not np.all(weights == np.floor(np.asarray(weights, dtype=float))):
            raise ValueError("weights are q-point multiplicities and must be whole numbers")
        object.__setattr__(self, "frequencies", to_ro_numpy(self.frequencies, "double"))
        object.__setattr__(self, "weights", to_ro_numpy(weights, "int_"))
        object.__setattr__(self, "volume", float(self.volume))
        object.__setattr__(self, "energy", float(self.energy))
        object.__setattr__(self, "atoms_per_cell", int(self.atoms_per_cell))
        if self.frequencies.ndim != 2:
            raise ValueError(f"frequencies must be 2d (n_qpoints, n_bands), got shape {self.frequencies.shape}")
        if self.frequencies.shape[1] % 3 != 0:
            raise ValueError(
                f"band count {self.frequencies.shape[1]} is not a multiple of three, so it cannot be "
                "3 * atoms_per_primitive_cell; frequencies must be shaped (n_qpoints, n_bands)"
            )
        if self.weights.shape != self.frequencies.shape[:1]:
            raise ValueError(
                f"weights shape {self.weights.shape} does not match the q points in frequencies "
                f"{self.frequencies.shape[:1]}"
            )
        if self.atoms_per_cell < 1:
            raise ValueError(f"atoms_per_cell must be positive, got {self.atoms_per_cell}")
        if self.volume <= 0:
            raise ValueError(f"volume must be positive, got {self.volume}")
        # precompute: hashing the arrays on every cache lookup is too expensive, and the
        # dataclass advertises as frozen anyway
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.frequencies.tobytes(),
                    self.weights.tobytes(),
                    self.volume,
                    self.energy,
                    self.atoms_per_cell,
                )
            ),
        )

    @classmethod
    def from_phonopy(cls, phonopy, energy, atoms_per_cell=None, mesh=None):
        """Build a spectrum from a live :class:`phonopy.Phonopy` object.

        Args:
            phonopy (phonopy.Phonopy): with force constants already produced
            energy (float): static energy of ``phonopy.unitcell``, in eV
            atoms_per_cell (int): atoms ``energy`` refers to; defaults to
                ``len(phonopy.unitcell)``, which is the cell ``phonopy.unitcell.volume``
                is taken from
            mesh: sampling mesh to pass to :meth:`~phonopy.Phonopy.run_mesh`.  The
                default runs phonopy's own default mesh if none has been run yet, and
                reuses the existing one if one has.

        Returns:
            PhononSpectrum
        """
        if mesh is not None:
            phonopy.run_mesh(mesh)
        elif phonopy.mesh is None:
            phonopy.run_mesh()
        if atoms_per_cell is None:
            atoms_per_cell = len(phonopy.unitcell)
        return cls(
            frequencies=phonopy.mesh.frequencies,
            weights=phonopy.mesh.weights,
            volume=phonopy.unitcell.volume,
            energy=energy,
            atoms_per_cell=atoms_per_cell,
        )

    @property
    def atoms_per_primitive_cell(self):
        """Atoms in the cell ``frequencies`` refer to, from the band count."""
        return self.frequencies.shape[1] // 3

    @property
    def min_frequency(self):
        """Lowest frequency on the mesh, in THz; negative means dynamically unstable."""
        return float(self.frequencies.min())

    @property
    def volume_per_atom(self):
        """Cell volume divided by :attr:`atoms_per_cell`, in cubic Angstrom."""
        return self.volume / self.atoms_per_cell

    @property
    def static_energy_per_atom(self):
        """Static energy divided by :attr:`atoms_per_cell`, in eV."""
        return self.energy / self.atoms_per_cell

    @lru_cache(maxsize=64)
    def _thermal(self, classical, cutoff_frequency):
        """phonopy's ``ThermalProperties`` over this spectrum, built once and reused."""
        return ThermalProperties(
            _MeshShim(self.frequencies, self.weights),
            cutoff_frequency=cutoff_frequency,
            classical=classical,
        )

    def thermal_properties(self, T, classical=False, cutoff_frequency=0.0):
        """Harmonic thermal properties per atom, from phonopy's own kernel.

        .. math::

            F_\\mathrm{quantum}   &= \\sum_i \\left[ \\tfrac{1}{2}\\hbar\\omega_i
                + k_BT \\ln\\left(1 - e^{-\\hbar\\omega_i / k_BT}\\right) \\right] \\\\
            F_\\mathrm{classical} &= \\sum_i k_BT \\ln\\left(\\hbar\\omega_i / k_BT\\right)

        weighted by the q-point multiplicities, plus the matching entropy and heat
        capacity.  ``T`` may be any list of temperatures in any order -- there is no grid
        and nothing is interpolated.

        At ``T = 0`` the quantum free energy is the zero-point energy while the classical
        one is zero: the classical expression has no zero-temperature limit, and zero is
        what phonopy reports.  Entropy and heat capacity are numerically unreliable for
        the first few kelvin of the quantum branch, where phonopy's own expressions
        overflow before they cancel.

        Args:
            T (float or array of float): temperature in K, non-negative
            classical (bool): use Boltzmann rather than Bose-Einstein statistics
            cutoff_frequency (float): drop modes at or below this frequency, in THz

        Returns:
            tuple: free energy in eV per atom, entropy in eV/K per atom and heat capacity
            in eV/K per atom, each shaped like ``T``
        """
        T = np.asarray(T, dtype=float)
        if np.any(T < 0):
            # phonopy's temperatures setter silently drops negative entries, which would
            # misalign the result against the requested temperatures
            raise ValueError(f"temperatures must be non-negative, got {T[T < 0]}")
        tp = self._thermal(bool(classical), float(cutoff_frequency))
        tp.temperatures = np.atleast_1d(T).ravel()
        with np.errstate(over="ignore", invalid="ignore"):
            tp.run()
        _, free_energy, entropy, heat_capacity = tp.thermal_properties
        units = get_physical_units()
        per_atom = self.atoms_per_primitive_cell
        return tuple(
            _scalarize(np.asarray(a).reshape(T.shape) * scale / per_atom)
            for a, scale in (
                (free_energy, 1 / units.EvTokJmol),
                (entropy, 1e-3 / units.EvTokJmol),
                (heat_capacity, 1e-3 / units.EvTokJmol),
            )
        )

    def vibrational_free_energy(self, T, classical=False, cutoff_frequency=0.0):
        """Harmonic free energy per atom at this volume, in eV.

        The free energy of :meth:`thermal_properties`; args are the same.
        """
        return self.thermal_properties(T, classical=classical, cutoff_frequency=cutoff_frequency)[0]

    def free_energy(self, T, classical=False, cutoff_frequency=0.0):
        """Static plus vibrational free energy per atom at this volume, in eV.

        Args are as for :meth:`thermal_properties`.
        """
        return self.static_energy_per_atom + self.vibrational_free_energy(
            T, classical=classical, cutoff_frequency=cutoff_frequency
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.volume == other.volume
            and self.energy == other.energy
            and self.atoms_per_cell == other.atoms_per_cell
            and np.array_equal(self.frequencies, other.frequencies)
            and np.array_equal(self.weights, other.weights)
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(volume={self.volume}, energy={self.energy}, "
            f"atoms_per_cell={self.atoms_per_cell}, n_qpoints={self.frequencies.shape[0]}, "
            f"atoms_per_primitive_cell={self.atoms_per_primitive_cell})"
        )


@dataclass(frozen=True)
class PhonopyQuasiHarmonicPhase(AbstractLinePhase):
    """A fixed-concentration phase whose free energy is evaluated from phonon spectra.

    Give one :class:`PhononSpectrum` per sampled volume, or build the phase from live
    :class:`~phonopy.Phonopy` objects with :meth:`from_phonopy`.  At every requested
    temperature the free energy of each volume comes from phonopy's mode sum, an equation
    of state is fitted through the results and its minimum is returned -- one number per
    temperature, computed rather than interpolated, at any temperature the caller asks
    for.

    Compared with sampling the same calculation onto a grid and handing that to
    :class:`~landau.phases.TemperatureDependentLinePhase`, this removes both the grid and
    the fit.  On quasi-harmonic data the default ``SGTE(3)`` leaves residuals of several
    meV/atom (issue #427), and a shallow solid-solid transition converts one meV/atom
    into tens of kelvin.

    Two things this refuses to do quietly:

    **Dynamically unstable volumes.**  Imaginary modes are dropped by the mode sum rather
    than raised on, so a volume with unstable branches still returns a smooth, plausible
    free energy that then corrupts the equation-of-state fit every other volume feeds.
    Spectra whose lowest mode falls below ``min_frequency`` are excluded and reported
    through :attr:`unstable_spectra`, with a :class:`DynamicalInstabilityWarning` at
    construction.

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
    spectra: tuple
    """One :class:`PhononSpectrum` per sampled volume; at least four are needed to fit
    the four equation-of-state parameters."""
    classical: bool = False
    """Use Boltzmann rather than Bose-Einstein statistics for the mode occupation.  The
    classical free energy has no zero-point energy and no zero-temperature limit; pick it
    to compare against classical molecular dynamics."""
    eos: str = "vinet"
    """Equation of state to minimise over volume; one of ``"vinet"``,
    ``"birch_murnaghan"`` or ``"murnaghan"``, passed to :func:`phonopy.qha.eos.get_eos`."""
    cutoff_frequency: float = 0.0
    """Modes at or below this frequency, in THz, are dropped from the mode sum.  Zero is
    phonopy's own convention, but it keeps the acoustic branches at Gamma, which the sum
    rule puts at zero and which come back as numerical noise of either sign.  The
    positive ones then have :math:`k_BT\\ln(\\hbar\\omega/k_BT)` evaluated on a frequency
    that should be zero: for fcc Ca that is -1.0 meV/atom at 2400 K, and it does not
    cancel in a free-energy difference because it depends on the temperature and on where
    the noise happens to land at each volume.  A cutoff of 1e-3 excludes no physical
    mode."""
    min_frequency: float = -0.05
    """Spectra whose lowest mode falls below this frequency, in THz, are treated as
    dynamically unstable and excluded from the fit.  Slightly negative rather than zero
    because the acoustic branches at Gamma are numerically noisy in both directions."""
    _hash: int = field(default=0, init=False, repr=False)

    @phonopy_alarm
    def __post_init__(self):
        object.__setattr__(self, "spectra", tuple(self.spectra))
        if not all(isinstance(s, PhononSpectrum) for s in self.spectra):
            raise TypeError("spectra must all be PhononSpectrum instances")
        if self.eos not in ("vinet", "birch_murnaghan", "murnaghan"):
            raise ValueError(f"eos must be one of 'vinet', 'birch_murnaghan', 'murnaghan', got {self.eos!r}")

        atom_counts = {(s.atoms_per_cell, s.atoms_per_primitive_cell) for s in self.spectra}
        if len(atom_counts) > 1:
            raise ValueError(
                "all spectra must describe the same structure, but their "
                f"(atoms_per_cell, atoms_per_primitive_cell) differ: {sorted(atom_counts)}"
            )
        volumes = [s.volume_per_atom for s in self.spectra]
        if len(set(volumes)) != len(volumes):
            raise ValueError(f"spectra must sample distinct volumes, got {volumes}")

        unstable = tuple(s for s in self.spectra if s.min_frequency < self.min_frequency)
        stable = tuple(sorted((s for s in self.spectra if s not in unstable), key=lambda s: s.volume_per_atom))
        if len(stable) < 4:
            raise ValueError(
                f"{len(stable)} dynamically stable volume(s) out of {len(self.spectra)}, but fitting "
                f"{self.eos} needs at least four; lowest frequencies (THz) were "
                f"{[s.min_frequency for s in self.spectra]}"
            )
        if unstable:
            warnings.warn(
                f"{self.name}: dropped {len(unstable)} of {len(self.spectra)} volumes carrying modes below "
                f"{self.min_frequency} THz (volumes per atom {[s.volume_per_atom for s in unstable]}, lowest "
                f"frequencies {[s.min_frequency for s in unstable]}); the equation of state is fitted "
                f"through the remaining {len(stable)}",
                DynamicalInstabilityWarning,
                stacklevel=2,
            )
        object.__setattr__(self, "_stable", stable)
        object.__setattr__(self, "_unstable", unstable)
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    self.name,
                    self.fixed_concentration,
                    self.spectra,
                    self.classical,
                    self.eos,
                    self.cutoff_frequency,
                    self.min_frequency,
                )
            ),
        )

    @classmethod
    def from_phonopy(cls, name, fixed_concentration, spectra, *, mesh=None, atoms_per_cell=None, **kwargs):
        """Build a phase from live :class:`~phonopy.Phonopy` objects, one per volume.

        Args:
            name (str): name of the phase
            fixed_concentration (float): concentration of the phase
            spectra (iterable of tuple): ``(phonopy, energy)`` pairs, where ``energy`` is
                the static energy of that object's unit cell in eV
            mesh: sampling mesh, passed on to :meth:`PhononSpectrum.from_phonopy`
            atoms_per_cell (int): atoms the energies refer to, passed on to
                :meth:`PhononSpectrum.from_phonopy`
            **kwargs: forwarded to the constructor (``classical``, ``eos``, ...)

        Returns:
            PhonopyQuasiHarmonicPhase
        """
        return cls(
            name,
            fixed_concentration,
            spectra=tuple(
                PhononSpectrum.from_phonopy(p, energy, atoms_per_cell=atoms_per_cell, mesh=mesh)
                for p, energy in spectra
            ),
            **kwargs,
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.name == other.name
            and self.fixed_concentration == other.fixed_concentration
            and self.spectra == other.spectra
            and self.classical == other.classical
            and self.eos == other.eos
            and self.cutoff_frequency == other.cutoff_frequency
            and self.min_frequency == other.min_frequency
        )

    @property
    def line_concentration(self):
        return self.fixed_concentration

    @property
    def stable_spectra(self):
        """The spectra the equation of state is fitted through, sorted by volume."""
        return self._stable

    @property
    def unstable_spectra(self):
        """The spectra excluded as dynamically unstable, in the order given."""
        return self._unstable

    @property
    def sampled_volumes(self):
        """Volumes per atom of :attr:`stable_spectra`, ascending, in cubic Angstrom."""
        return np.array([s.volume_per_atom for s in self._stable])

    @lru_cache(maxsize=1024)
    def eos_parameters(self, T):
        """Equation-of-state parameters at one temperature.

        Fits ``self.eos`` through the per-atom free energies of :attr:`stable_spectra`.

        Args:
            T (float): temperature in K

        Returns:
            tuple of float: ``(E_0, B_0, B'_0, V_0)`` -- the minimum free energy in eV per
            atom, the bulk modulus in eV per cubic Angstrom, its pressure derivative, and
            the equilibrium volume in cubic Angstrom per atom.  All four are NaN if the
            fit fails.
        """
        energies = np.array(
            [s.free_energy(T, classical=self.classical, cutoff_frequency=self.cutoff_frequency) for s in self._stable]
        )
        # get_eos returns a closure that phonopy 3 calls as eos(v, *p) and phonopy 4 as
        # eos(v, p); handing it straight to fit_to_eos and never calling it keeps that
        # difference out of here
        try:
            parameters = fit_to_eos(self.sampled_volumes, energies, get_eos(self.eos))
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
