"""Export landau phases to a Thermo-Calc database (TDB) file.

A TDB stores CALPHAD models: one closed-form Gibbs energy ``G(T)`` per phase
end-member plus Redlich-Kister interaction parameters, each a polynomial in
``T`` with an optional ``T*LN(T)`` term.  Only landau phases whose free energy
already has that form are exported; every other phase raises :exc:`TypeError`.

- :class:`~landau.phases.LinePhase` and :class:`~landau.phases.TemperatureDependentLinePhase`
  (fitted with :class:`~landau.interpolate.SGTE` or :class:`~landau.interpolate.PolyFit`)
  become stoichiometric phases ``(A)_{1-c}(B)_c`` -- one sublattice at the terminals --
  carrying their ``G(T)``.
- :class:`~landau.phases.IdealSolution` becomes an ``(A,B)`` solution phase carrying the
  terminals' ``G(T)`` and no interaction parameter.
- :class:`~landau.phases.RegularSolution`, :class:`~landau.phases.InterpolatingPhase`,
  :class:`~landau.phases.SlowInterpolatingPhase` and
  :class:`~landau.phases.FastInterpolatingPhase` interpolating with
  :class:`~landau.interpolate.RedlichKister` become ``(A,B)`` phases with interaction
  parameters ``L_v(T)``.  The fit is linear in the line phases' free energies, so each
  ``L_v(T)`` is an exact combination of their closed forms.
- :class:`~landau.phases.Surface2DInterpolatingPhase` over a
  :class:`~landau.interpolate.CalphadSurface2DInterpolator` becomes an ``(A,B)`` phase
  from the fitted terminal and interaction models.

Energies are converted from eV/atom to J/mol.  Site ratios sum to one, so a
formula unit is one mole of atoms and every ``G`` is per atom as in landau; the
ideal mixing entropy of a solution phase is supplied by the TDB reader, exactly
as landau's ``-T S(c)``.  Reference states are absolute (no ``GHSER``), and the
``ELEMENT`` records carry no mass or reference data.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.constants import Avogadro, eV

from .interpolate.basic import (
    CalphadSurface2DInterpolator,
    Interpolation,
    PolynomialInterpolation,
    RedlichKister,
    SGTEInterpolation,
)
from .phases import (
    AbstractLinePhase,
    IdealSolution,
    InterpolatingPhase,
    LinePhase,
    Phase,
    RegularSolution,
    S,
    SlowInterpolatingPhase,
    Surface2DInterpolatingPhase,
    TemperatureDependentLinePhase,
)

__all__ = ["to_tdb", "write_tdb"]

_J_PER_MOL = eV * Avogadro
"""eV/atom -> J/mol."""

_LINE_WIDTH = 78
"""Thermo-Calc reads at most this many characters per line."""

#: One term of a TDB temperature function: ``(power of T, multiplied by ln T)``.
_Term = tuple[int, bool]


def _number(x: float) -> str:
    """Shortest round-tripping scientific notation, always signed: ``+1.5E+03``."""
    s = np.format_float_scientific(float(x), unique=True, sign=True).upper()
    return s.replace(".E", ".0E")


def _plain(x: float) -> str:
    """Unsigned number for site ratios and temperature limits."""
    return f"{float(x):.16g}".upper()


def _monomial(term: _Term) -> str:
    power, log = term
    if log:
        return "*T*LN(T)"
    if power == 0:
        return ""
    if power == 1:
        return "*T"
    if power < 0:
        return f"*T**({power})"
    return f"*T**{power}"


@dataclass(frozen=True)
class _GFunction:
    """A closed-form ``G(T) = sum_i a_i T^i + b T ln T`` in eV/atom."""

    terms: Mapping[_Term, float]

    def __add__(self, other: "_GFunction") -> "_GFunction":
        terms = dict(self.terms)
        for term, coeff in other.terms.items():
            terms[term] = terms.get(term, 0.0) + coeff
        return _GFunction(terms)

    def __mul__(self, scalar: float) -> "_GFunction":
        return _GFunction({term: scalar * coeff for term, coeff in self.terms.items()})

    __rmul__ = __mul__

    def tdb(self) -> list[str]:
        """The J/mol expression as TDB terms, one signed token each."""
        tokens = [
            _number(coeff * _J_PER_MOL) + _monomial(term)
            for term, coeff in sorted(self.terms.items())
            if coeff != 0
        ]
        return tokens or [_number(0.0)]


_ZERO = _GFunction({})


def _describe(phase: Phase) -> str:
    return f'{type(phase).__name__}("{phase.name}")'


def _from_interpolation(interpolation: Interpolation, owner: Phase) -> _GFunction:
    """The closed form behind a fitted temperature interpolation, if it has one."""
    if isinstance(interpolation, SGTEInterpolation):
        log_coeff, *poly = interpolation.parameters
        terms = {(i, False): float(p) for i, p in enumerate(poly)}
        terms[(1, True)] = float(log_coeff)
        return _GFunction(terms)
    if isinstance(interpolation, PolynomialInterpolation):
        return _GFunction({(i, False): float(p) for i, p in enumerate(interpolation.coefficients)})
    raise TypeError(
        f"{_describe(owner)}: a {type(interpolation).__name__} has no closed form in T; "
        "only SGTE and PolyFit fits can be written to a TDB"
    )


def _line_g(phase: AbstractLinePhase) -> _GFunction:
    """``G(T)`` of a line phase in eV/atom."""
    if isinstance(phase, LinePhase):
        return _GFunction({(0, False): float(phase.line_energy), (1, False): -float(phase.line_entropy)})
    if isinstance(phase, TemperatureDependentLinePhase):
        return _from_interpolation(phase._interpolation, phase)
    raise TypeError(
        f"{_describe(phase)}: only LinePhase and TemperatureDependentLinePhase have a closed-form free energy; "
        f"{type(phase).__name__} cannot be written to a TDB"
    )


def _member_g(line_phase: AbstractLinePhase, owner: Phase) -> _GFunction:
    """``G(T)`` of one of a solution phase's line phases, naming the solution phase on failure."""
    try:
        return _line_g(line_phase)
    except TypeError as error:
        raise TypeError(f"{_describe(owner)}: {error}") from None


@dataclass
class _TdbPhase:
    """One ``PHASE`` block: sublattices, their constituents and the ``PARAMETER`` records."""

    name: str
    site_ratios: tuple[float, ...]
    constituents: tuple[tuple[str, ...], ...]
    parameters: list[tuple[str, _GFunction]]


def _stoichiometric(name: str, phase: AbstractLinePhase, elements: tuple[str, str]) -> _TdbPhase:
    c = float(phase.line_concentration)
    if not 0 <= c <= 1:
        raise ValueError(f"{_describe(phase)}: concentration {c} is outside [0, 1]")
    first, second = elements
    g = _line_g(phase)
    if c == 0:
        return _TdbPhase(name, (1.0,), ((first,),), [(f"G({name},{first};0)", g)])
    if c == 1:
        return _TdbPhase(name, (1.0,), ((second,),), [(f"G({name},{second};0)", g)])
    return _TdbPhase(name, (1 - c, c), ((first,), (second,)), [(f"G({name},{first}:{second};0)", g)])


def _solution(
    name: str, g_first: _GFunction, g_second: _GFunction, interactions: list[_GFunction], elements: tuple[str, str]
) -> _TdbPhase:
    """A single-sublattice ``(A,B)`` phase with Redlich-Kister interactions.

    ``interactions`` are landau's coefficients of ``c(1-c) (2c-1)^v`` with ``c`` the
    fraction of ``second``.  A TDB reader takes ``L(PHASE,a,b;v)`` as the coefficient
    of ``x_a x_b (x_a - x_b)^v`` and pycalphad sorts ``a, b`` alphabetically before
    reading the sign off the order, so the parameter is written in sorted order and
    the odd orders flip sign when that puts ``first`` in front.
    """
    first = elements[0]
    a, b = sorted(elements)
    sign = -1 if a == first else 1
    parameters = [(f"G({name},{a};0)", g_first if a == first else g_second)]
    parameters.append((f"G({name},{b};0)", g_second if a == first else g_first))
    parameters += [(f"L({name},{a},{b};{v})", L * sign**v) for v, L in enumerate(interactions)]
    return _TdbPhase(name, (1.0,), ((a, b),), parameters)


def _rk_interpolator(phase: Phase) -> RedlichKister:
    """The concentration interpolator a solution phase refits at every temperature."""
    if isinstance(phase, (RegularSolution, InterpolatingPhase)):
        interpolator = phase._concentration_interpolator
    else:
        interpolator = phase.interpolator
    if not isinstance(interpolator, RedlichKister):
        raise TypeError(
            f"{_describe(phase)}: interpolates in concentration with {interpolator!r}, but only a RedlichKister "
            "fit through terminal line phases at c=0 and c=1 maps onto TDB interaction parameters"
        )
    return interpolator


def _from_redlich_kister(name: str, phase: Phase, elements: tuple[str, str]) -> _TdbPhase:
    interpolator = _rk_interpolator(phase)
    concentrations = np.array([p.line_concentration for p in phase.phases], dtype=float)
    if not (np.isclose(concentrations.min(), 0) and np.isclose(concentrations.max(), 1)):
        raise TypeError(f"{_describe(phase)}: needs terminal line phases at c=0 and c=1 to be written to a TDB")
    # The reader adds the ideal mixing entropy itself, so the parameters carry
    # the entropy-removed H = f + T S(c), as the phase fits it.
    samples = []
    for line_phase, c in zip(phase.phases, concentrations):
        h = _member_g(line_phase, phase)
        if not phase.add_entropy:
            h = h + _GFunction({(1, False): float(S(c))})
        samples.append(h)
    # RedlichKister.fit is linear in the sampled free energies: fitting the unit
    # vectors yields (f0, df, L_v) as weights on the samples, and the same weights
    # combine the samples' closed forms into closed-form parameters.
    fits = [interpolator.fit(concentrations, unit) for unit in np.eye(len(concentrations))]

    def combine(weights):
        return sum((w * h for w, h in zip(weights, samples)), _ZERO)

    g_first = combine([fit.f0 for fit in fits])
    g_second = combine([fit.f0 + fit.df for fit in fits])
    interactions = [
        combine([fit.rk_parameters[v] for fit in fits]) for v in range(len(fits[0].rk_parameters))
    ]
    return _solution(name, g_first, g_second, interactions, elements)


def _from_surface(name: str, phase: Surface2DInterpolatingPhase, elements: tuple[str, str]) -> _TdbPhase:
    if not isinstance(phase.surface_interpolator, CalphadSurface2DInterpolator):
        raise TypeError(
            f"{_describe(phase)}: a {type(phase.surface_interpolator).__name__} surface has no Redlich-Kister form; "
            "only CalphadSurface2DInterpolator can be written to a TDB"
        )
    surface = phase._fitted_surface
    g_first = _from_interpolation(surface._f0_model, phase)
    g_second = g_first + _from_interpolation(surface._df_model, phase)
    interactions = [_from_interpolation(model, phase) for model in surface._L_models]
    return _solution(name, g_first, g_second, interactions, elements)


def _tdb_name(phase: Phase) -> str:
    """Upper-case the name and squash anything that is not alphanumeric to ``_``."""
    name = re.sub(r"[^A-Z0-9]+", "_", phase.name.upper()).strip("_")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise ValueError(f"cannot derive a TDB phase name from {phase.name!r}; rename the phase")
    return name


def _convert(phase: Phase, elements: tuple[str, str]) -> _TdbPhase:
    name = _tdb_name(phase)
    if isinstance(phase, AbstractLinePhase):
        return _stoichiometric(name, phase, elements)
    if isinstance(phase, IdealSolution):
        return _solution(name, _member_g(phase.phase1, phase), _member_g(phase.phase2, phase), [], elements)
    if isinstance(phase, Surface2DInterpolatingPhase):
        return _from_surface(name, phase, elements)
    if isinstance(phase, (RegularSolution, InterpolatingPhase, SlowInterpolatingPhase)):
        return _from_redlich_kister(name, phase, elements)
    raise TypeError(f"{_describe(phase)}: {type(phase).__name__} has no CALPHAD form and cannot be written to a TDB")


def _check_elements(elements) -> tuple[str, str]:
    elements = tuple(str(e).upper() for e in elements)
    if len(elements) != 2 or len(set(elements)) != 2:
        raise ValueError(f"elements must name two distinct components, got {elements}")
    for element in elements:
        if not re.fullmatch(r"[A-Z][A-Z]?", element):
            raise ValueError(f"element names are one or two letters, got {element!r}")
    return elements


def _wrap(head: str, tokens: list[str], tail: str) -> str:
    """Join ``head``, the expression tokens and ``tail`` into lines of at most :data:`_LINE_WIDTH`."""
    lines = [head]
    for token in tokens + [tail]:
        if len(lines[-1]) + 1 + len(token) > _LINE_WIDTH:
            lines.append("  " + token)
        else:
            lines[-1] += " " + token
    return "\n".join(lines)


def _render(phase: _TdbPhase, low: str, high: str) -> str:
    ratios = " ".join(_plain(r) for r in phase.site_ratios)
    constituents = ":".join(",".join(sublattice) for sublattice in phase.constituents)
    lines = [
        f"PHASE {phase.name} % {len(phase.site_ratios)} {ratios} !",
        f"CONSTITUENT {phase.name} :{constituents}: !",
    ]
    for parameter, g in phase.parameters:
        *terms, last = g.tdb()
        lines.append(_wrap(f"PARAMETER {parameter} {low}", terms + [last + ";"], f"{high} N !"))
    return "\n".join(lines)


def to_tdb(
    phases: Iterable[Phase],
    elements: tuple[str, str] = ("A", "B"),
    temperature_range: tuple[float, float] = (298.15, 6000.0),
) -> str:
    """Write phases as a Thermo-Calc database (TDB).

    See the module docstring for which phases can be represented and how.

    Args:
        phases: phases to export; each becomes one ``PHASE`` record named after the
            phase (upper-cased, non-alphanumeric characters replaced by ``_``).
        elements: TDB element names of the components at ``c=0`` and ``c=1``, in
            that order; one or two letters each.
        temperature_range: ``(low, high)`` validity limits in K stamped on every
            parameter.  Readers treat the functions as undefined outside (pycalphad
            evaluates them to zero), so the range should cover every temperature
            the database will be used at.

    Returns:
        the TDB file contents.

    Raises:
        TypeError: a phase has no closed-form CALPHAD representation.
        ValueError: element names, temperature limits or a phase name are unusable,
            or two phases map onto the same TDB name.
    """
    from . import __version__

    elements = _check_elements(elements)
    low, high = map(float, temperature_range)
    if not 0 < low < high:
        raise ValueError(f"temperature_range must satisfy 0 < low < high, got {temperature_range}")

    phases = list(phases)
    converted = [_convert(phase, elements) for phase in phases]
    duplicates = [name for name, count in Counter(p.name for p in converted).items() if count > 1]
    if duplicates:
        clashes = [phase.name for phase, tdb in zip(phases, converted) if tdb.name in duplicates]
        raise ValueError(f"phases {clashes} map onto the same TDB name(s) {duplicates}; rename them")

    header = [
        f"$ Thermodynamic database written by landau {__version__}",
        "$ Energies in J/mol of atoms, converted from eV/atom; site ratios sum to one",
        "$ so that a formula unit is one mole of atoms.  Reference states are absolute.",
        "",
        "ELEMENT /- ELECTRON_GAS 0.0 0.0 0.0 !",
        "ELEMENT VA VACUUM 0.0 0.0 0.0 !",
    ]
    header += [f"ELEMENT {element} BLANK 0.0 0.0 0.0 !" for element in sorted(elements)]
    header += ["", "TYPE_DEFINITION % SEQ * !"]
    blocks = [_render(phase, _plain(low), _plain(high)) for phase in converted]
    return "\n".join(header) + "\n\n" + "\n\n".join(blocks) + "\n"


def write_tdb(
    phases: Iterable[Phase],
    path,
    elements: tuple[str, str] = ("A", "B"),
    temperature_range: tuple[float, float] = (298.15, 6000.0),
) -> None:
    """Write :func:`to_tdb` output to ``path``; arguments as there."""
    Path(path).write_text(to_tdb(phases, elements=elements, temperature_range=temperature_range))
