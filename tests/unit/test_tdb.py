"""Tests for the TDB export in ``landau.tdb``.

Every ``PARAMETER`` record is read back by a small CALPHAD evaluator in this
file -- the emitted ``G(T)`` expressions are evaluated as written and solution
phases summed with the Redlich-Kister convention of a TDB reader -- and compared
against the landau phase it came from.  The pycalphad round trip at the end
repeats the comparison through a real reader.
"""

import re
import warnings

import numpy as np
import pytest
from hypothesis import given, strategies as st
from scipy.constants import Avogadro, Boltzmann, eV

from landau import (
    CalphadSurface2DInterpolator,
    FastInterpolatingPhase,
    IdealSolution,
    InterpolatingPhase,
    LinePhase,
    PolyFit,
    RedlichKister,
    RegularSolution,
    SGTE,
    SlowInterpolatingPhase,
    SoftplusSurface2DInterpolator,
    Surface2DInterpolatingPhase,
    TemperatureDependentLinePhase,
    to_tdb,
    write_tdb,
)
from landau.interpolate import StitchedFit, WhitneyTemperatureInterpolator
from landau.phases import AbstractLinePhase, Phase, S, kB
from landau.phases.pointdefects import ConstantPointDefect, PointDefectedPhase, PointDefectSublattice
from landau.tdb import _number

try:
    from pycalphad import Database, calculate
    from pycalphad import variables as v

    HAS_PYCALPHAD = True
except ImportError:
    HAS_PYCALPHAD = False

J_PER_MOL = eV * Avogadro
R = Boltzmann * Avogadro
"""Not scipy's ``R``: scipy 1.11 tabulates it rounded to 8.314462618."""
ATOL = 1e-6
"""J/mol; the emitted numbers round-trip exactly, so this is evaluation round-off (measured ~1e-9)."""

TS = (300.0, 1150.0, 2000.0)
CS = np.linspace(0.02, 0.98, 25)


# --------------------------------------------------------------------------- #
# fixtures: line phases with the closed forms a TDB can hold
# --------------------------------------------------------------------------- #
def _sgte_phase(name, c, a, b, log, d):
    """A ``TemperatureDependentLinePhase`` whose SGTE(4) fit recovers ``a + bT + T ln T log + dT^2``."""
    T = np.linspace(300.0, 2000.0, 50)
    return TemperatureDependentLinePhase(name, c, T, a + b * T + log * T * np.log(T) + d * T**2, interpolator=SGTE(4))


@pytest.fixture(scope="module")
def terminals():
    return (
        _sgte_phase("fccA", 0.0, -3.0, 2e-4, -3e-4, 1e-8),
        _sgte_phase("fccB", 1.0, -2.5, 1e-4, -2.5e-4, 2e-8),
    )


@pytest.fixture(scope="module")
def line_phases(terminals):
    """Terminals plus three interior line phases; more samples than Redlich-Kister
    orders below, so the fit is a least-squares one, not an interpolation."""
    A, B = terminals
    return (
        A,
        _sgte_phase("m1", 0.25, -2.95, 1.5e-4, -2.8e-4, 1e-8),
        LinePhase("m2", 0.5, -2.9, 1.2 * kB),
        _sgte_phase("m3", 0.7, -2.8, 1.2e-4, -2.6e-4, 1.5e-8),
        B,
    )


def _rk_phases(line_phases, add_entropy=False):
    """The four Redlich-Kister solution classes over the same line phases, two orders each."""
    return [
        RegularSolution("reg", line_phases, num_coeffs=2, add_entropy=add_entropy),
        InterpolatingPhase("interp", line_phases, num_coeffs=4, add_entropy=add_entropy),
        SlowInterpolatingPhase("slow", line_phases, add_entropy=add_entropy, interpolator=RedlichKister(2)),
        FastInterpolatingPhase("fast", line_phases, add_entropy=add_entropy, interpolator=RedlichKister(2)),
    ]


def _surface_phase(line_phases):
    return Surface2DInterpolatingPhase(
        "surf",
        line_phases,
        surface_interpolator=CalphadSurface2DInterpolator(num_coeffs=2, coeff_poly_order=1),
        temperature_range=(300.0, 2000.0),
    )


# --------------------------------------------------------------------------- #
# a reader for the written records
# --------------------------------------------------------------------------- #
def _records(text):
    """Every ``!``-terminated record with comments stripped and continuation lines joined."""
    body = " ".join(line for line in text.splitlines() if not line.startswith("$"))
    return [record.strip() for record in body.split("!") if record.strip()]


def _parameters(text):
    """``{name: (low, high, G)}`` for every PARAMETER record, ``G(T)`` evaluating the emitted expression."""
    out = {}
    for record in _records(text):
        m = re.fullmatch(r"PARAMETER\s+(\S+)\s+(\S+)\s+(.*);\s*(\S+)\s+N", record)
        if m is None:
            continue
        name, low, expression, high = m.groups()
        code = expression.replace("LN(", "np.log(")
        out[name] = (float(low), float(high), lambda T, code=code: eval(code, {"np": np}, {"T": np.asarray(T, float)}))
    return out


def _solution_free_energy(text, name, elements, T, c):
    """``G(T, c)`` in J/mol as a TDB reader forms it from an ``(A,B)`` phase block:
    end-members, ideal mixing, and ``L(name,a,b;v)`` as the coefficient of
    ``x_a x_b (x_a - x_b)^v`` with ``a, b`` in the order written."""
    params = _parameters(text)
    first, second = elements
    x = {first: 1 - c, second: c}
    G = sum(x[e] * params[f"G({name},{e};0)"][2](T) for e in elements)
    G = G + R * T * sum(xe * np.log(xe) for xe in x.values())
    for key in params:
        m = re.fullmatch(rf"L\({name},(\w+),(\w+);(\d+)\)", key)
        if m is None:
            continue
        a, b, order = m.group(1), m.group(2), int(m.group(3))
        G = G + x[a] * x[b] * (x[a] - x[b]) ** order * params[key][2](T)
    return G


def _interactions(text, name):
    return sorted(key for key in _parameters(text) if key.startswith(f"L({name},"))


# --------------------------------------------------------------------------- #
# number formatting
# --------------------------------------------------------------------------- #
@given(st.floats(allow_nan=False, allow_infinity=False))
def test_number_round_trips(x):
    """Signed, upper-case scientific notation with digits either side of the point that reads back exactly."""
    s = _number(x)
    assert re.fullmatch(r"[+-]\d\.\d+E[+-]\d+", s)
    assert float(s) == x


# --------------------------------------------------------------------------- #
# file layout
# --------------------------------------------------------------------------- #
def test_header_and_line_width(line_phases):
    text = to_tdb(line_phases + tuple(_rk_phases(line_phases)))
    records = _records(text)
    assert text.startswith("$ Thermodynamic database written by landau")
    assert records[:4] == [
        "ELEMENT /- ELECTRON_GAS 0.0 0.0 0.0",
        "ELEMENT VA VACUUM 0.0 0.0 0.0",
        "ELEMENT A BLANK 0.0 0.0 0.0",
        "ELEMENT B BLANK 0.0 0.0 0.0",
    ]
    assert records[4] == "TYPE_DEFINITION % SEQ *"
    assert max(len(line) for line in text.splitlines()) <= 78


def test_write_tdb_writes_to_tdb_output(tmp_path, terminals):
    path = tmp_path / "out.tdb"
    write_tdb(terminals, path, elements=("Mg", "Ca"), temperature_range=(1.0, 3000.0))
    assert path.read_text() == to_tdb(terminals, elements=("Mg", "Ca"), temperature_range=(1.0, 3000.0))


def test_temperature_range_is_stamped_on_every_parameter(line_phases):
    text = to_tdb(_rk_phases(line_phases), temperature_range=(1.0, 3000.0))
    params = _parameters(text)
    assert len(params) == 4 * 4  # two end-members and two interactions per phase
    assert all((low, high) == (1.0, 3000.0) for low, high, _ in params.values())


@pytest.mark.parametrize("temperature_range", [(0.0, 1000.0), (1000.0, 500.0), (-1.0, 1000.0)])
def test_temperature_range_must_be_positive_and_ordered(terminals, temperature_range):
    with pytest.raises(ValueError, match="temperature_range"):
        to_tdb(terminals, temperature_range=temperature_range)


# --------------------------------------------------------------------------- #
# elements and names
# --------------------------------------------------------------------------- #
def test_elements_are_upper_cased_and_sorted_in_header(terminals):
    records = _records(to_tdb(terminals, elements=("mg", "ca")))
    assert records[2:4] == ["ELEMENT CA BLANK 0.0 0.0 0.0", "ELEMENT MG BLANK 0.0 0.0 0.0"]
    assert "PHASE FCCA % 1 1" in records and "CONSTITUENT FCCA :MG:" in records
    assert "PHASE FCCB % 1 1" in records and "CONSTITUENT FCCB :CA:" in records


@pytest.mark.parametrize("elements", [("ABC", "B"), ("A", "A"), ("A",), ("A", "B", "C"), ("A1", "B")])
def test_elements_must_be_two_distinct_symbols(terminals, elements):
    with pytest.raises(ValueError, match="element"):
        to_tdb(terminals, elements=elements)


@pytest.mark.parametrize("name, tdb_name", [("Mg2Ca-C14", "MG2CA_C14"), ("L1_2 (ordered)", "L1_2_ORDERED"), ("fcc", "FCC")])
def test_phase_names_are_upper_cased_alphanumeric(name, tdb_name):
    records = _records(to_tdb([LinePhase(name, 0.5, -1.0)]))
    assert f"PHASE {tdb_name} % 2 0.5 0.5" in records
    assert f"CONSTITUENT {tdb_name} :A:B:" in records


@pytest.mark.parametrize("name", ["α", "2H", "", "-"])
def test_unusable_phase_names_raise(name):
    with pytest.raises(ValueError, match="TDB phase name"):
        to_tdb([LinePhase(name, 0.5, -1.0)])


def test_colliding_phase_names_raise():
    with pytest.raises(ValueError, match=r"\['fcc', 'FCC'\].*same TDB name"):
        to_tdb([LinePhase("fcc", 0.0, -1.0), LinePhase("FCC", 1.0, -1.0)])


# --------------------------------------------------------------------------- #
# stoichiometric phases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "c, phase_record, constituent_record, parameter",
    [
        (0.0, "PHASE X % 1 1", "CONSTITUENT X :A:", "G(X,A;0)"),
        (1.0, "PHASE X % 1 1", "CONSTITUENT X :B:", "G(X,B;0)"),
        (1 / 3, "PHASE X % 2 0.6666666666666667 0.3333333333333333", "CONSTITUENT X :A:B:", "G(X,A:B;0)"),
    ],
)
def test_line_phase_sublattices(c, phase_record, constituent_record, parameter):
    """Terminals are one-sublattice, interior line phases two, with site ratios ``1-c`` and ``c``."""
    phase = LinePhase("x", c, -2.9, 1.2 * kB)
    text = to_tdb([phase])
    records = _records(text)
    assert phase_record in records and constituent_record in records
    params = _parameters(text)
    assert list(params) == [parameter]
    T = np.array(TS)
    np.testing.assert_allclose(params[parameter][2](T), phase.line_free_energy(T) * J_PER_MOL, atol=ATOL)


def test_line_phase_entropy_of_one_kb_is_minus_gas_constant():
    """The unit conversion maps ``kB`` eV/K per atom onto ``R`` J/mol/K."""
    text = to_tdb([LinePhase("x", 0.0, 0.0, kB)])
    m = re.search(r"PARAMETER G\(X,A;0\) 298.15 (\S+)\*T; 6000 N !", text)
    assert float(m.group(1)) == pytest.approx(-R, rel=1e-14)


def test_line_phase_concentration_outside_unit_interval_raises():
    with pytest.raises(ValueError, match="outside"):
        to_tdb([LinePhase("x", 1.5, -1.0)])


@pytest.mark.parametrize("interpolator", [SGTE(4), SGTE(2), PolyFit(3), PolyFit(1)])
def test_temperature_dependent_line_phase(interpolator):
    """The written ``G(T)`` is the phase's own fit, whatever closed form it uses."""
    T = np.linspace(300.0, 2000.0, 50)
    phase = TemperatureDependentLinePhase("x", 0.25, T, -3.0 + 2e-4 * T - 3e-4 * T * np.log(T), interpolator=interpolator)
    G = _parameters(to_tdb([phase]))["G(X,A:B;0)"][2]
    Tq = np.linspace(250.0, 2100.0, 7)
    np.testing.assert_allclose(G(Tq), phase.line_free_energy(Tq) * J_PER_MOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# solution phases
# --------------------------------------------------------------------------- #
def test_ideal_solution(terminals):
    A, B = terminals
    phase = IdealSolution("sol", B, A)
    text = to_tdb([phase])
    records = _records(text)
    assert "PHASE SOL % 1 1" in records and "CONSTITUENT SOL :A,B:" in records
    assert sorted(_parameters(text)) == ["G(SOL,A;0)", "G(SOL,B;0)"]
    for T in TS:
        expected = (1 - CS) * A.line_free_energy(T) + CS * B.line_free_energy(T) - T * S(CS)
        np.testing.assert_allclose(_solution_free_energy(text, "SOL", ("A", "B"), T, CS), expected * J_PER_MOL, atol=ATOL)


@pytest.mark.parametrize("add_entropy", [False, True])
@pytest.mark.parametrize("index", range(4), ids=["regular", "interpolating", "slow", "fast"])
def test_redlich_kister_phases(line_phases, index, add_entropy):
    """Each L_v(T) is the fit's linear combination of the line phases' closed forms, so the
    written phase reproduces ``free_energy`` at every (T, c), least-squares fit included."""
    phase = _rk_phases(line_phases, add_entropy)[index]
    text = to_tdb([phase])
    name = phase.name.upper()
    assert _interactions(text, name) == [f"L({name},A,B;0)", f"L({name},A,B;1)"]
    for T in TS:
        np.testing.assert_allclose(
            _solution_free_energy(text, name, ("A", "B"), T, CS), phase.free_energy(T, CS) * J_PER_MOL, atol=ATOL
        )


def test_odd_interactions_flip_sign_for_unsorted_elements(line_phases):
    """Parameters are written with alphabetically sorted constituents, which for
    ``elements=("MG", "CA")`` puts the ``c=1`` component first and negates odd orders."""
    phase = _rk_phases(line_phases)[3]
    text = to_tdb([phase], elements=("MG", "CA"))
    assert _interactions(text, "FAST") == ["L(FAST,CA,MG;0)", "L(FAST,CA,MG;1)"]
    reference = _parameters(to_tdb([phase]))
    written = _parameters(text)
    T = np.array(TS)
    np.testing.assert_allclose(written["L(FAST,CA,MG;0)"][2](T), reference["L(FAST,A,B;0)"][2](T), atol=ATOL)
    np.testing.assert_allclose(written["L(FAST,CA,MG;1)"][2](T), -reference["L(FAST,A,B;1)"][2](T), atol=ATOL)
    for T in TS:
        np.testing.assert_allclose(
            _solution_free_energy(text, "FAST", ("MG", "CA"), T, CS), phase.free_energy(T, CS) * J_PER_MOL, atol=ATOL
        )


def test_surface_phase(line_phases):
    phase = _surface_phase(line_phases)
    text = to_tdb([phase])
    assert _interactions(text, "SURF") == ["L(SURF,A,B;0)", "L(SURF,A,B;1)"]
    for T in TS:
        np.testing.assert_allclose(
            _solution_free_energy(text, "SURF", ("A", "B"), T, CS), phase.free_energy(T, CS) * J_PER_MOL, atol=ATOL
        )


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
class _OpaqueLinePhase(AbstractLinePhase):
    line_concentration = 0.5

    def line_free_energy(self, T):
        return -1.0 + 0 * np.asarray(T)


class _OpaquePhase(Phase):
    def semigrand_potential(self, T, dmu):
        return -1.0 + 0 * np.asarray(T)

    def concentration(self, T, dmu):
        return 0.5 + 0 * np.asarray(T)


def _stitched(name, c):
    T = np.linspace(300.0, 2000.0, 50)
    return TemperatureDependentLinePhase(name, c, T, -3.0 - 1e-3 * T, interpolator=StitchedFit())


def _whitney(name, c):
    T = np.linspace(300.0, 2000.0, 50)
    return TemperatureDependentLinePhase(name, c, T, -3.0 - 1e-3 * T, interpolator=WhitneyTemperatureInterpolator())


def _point_defected():
    host = LinePhase("host", 0.0, -3.0)
    sublattice = PointDefectSublattice(
        name="vac",
        sublattice=0,
        sublattice_fraction=1.0,
        defects=[ConstantPointDefect("v", excess_energy=1.0, excess_entropy=0.0, excess_solutes=1)],
    )
    return PointDefectedPhase(name="x", line_phase=host, sublattices=[sublattice])


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda A, B, mid: _stitched("x", 0.5), "no closed form in T"),
        (lambda A, B, mid: _whitney("x", 0.5), "no closed form in T"),
        (lambda A, B, mid: IdealSolution("x", _stitched("s", 0.0), B), "no closed form in T"),
        (lambda A, B, mid: _OpaqueLinePhase("x"), "closed-form free energy"),
        (lambda A, B, mid: _OpaquePhase("x"), "no CALPHAD form"),
        (lambda A, B, mid: _point_defected(), "no CALPHAD form"),
        (lambda A, B, mid: FastInterpolatingPhase("x", [A, mid, B], interpolator=PolyFit(3)), "only a RedlichKister"),
        (lambda A, B, mid: FastInterpolatingPhase("x", [A, mid]), "only a RedlichKister"),
        (lambda A, B, mid: InterpolatingPhase("x", [mid, A, B]), "only a RedlichKister"),
        (lambda A, B, mid: RegularSolution("x", [A, _stitched("s", 0.5), B]), "no closed form in T"),
        (
            lambda A, B, mid: Surface2DInterpolatingPhase(
                "x", [A, mid, B], surface_interpolator=SoftplusSurface2DInterpolator(), temperature_range=(300.0, 2000.0)
            ),
            "only CalphadSurface2DInterpolator",
        ),
        (
            lambda A, B, mid: Surface2DInterpolatingPhase(
                "x",
                [A, mid, B],
                surface_interpolator=CalphadSurface2DInterpolator(
                    num_coeffs=1, terminal_interpolator=WhitneyTemperatureInterpolator()
                ),
                temperature_range=(300.0, 2000.0),
            ),
            "no closed form in T",
        ),
    ],
    ids=[
        "stitched",
        "whitney",
        "ideal-over-stitched",
        "opaque-line-phase",
        "opaque-phase",
        "point-defected",
        "polyfit-in-c",
        "missing-terminal",
        "terminals-not-first-and-last",
        "rk-over-stitched",
        "softplus-surface",
        "whitney-terminals",
    ],
)
def test_phases_without_a_calphad_form_raise(line_phases, build, match):
    """Every refusal is a ``TypeError`` naming the offending phase."""
    A, mid, B = line_phases[0], line_phases[1], line_phases[-1]
    phase = build(A, B, mid)
    with pytest.raises(TypeError, match=match) as info:
        to_tdb([phase])
    assert '"x"' in str(info.value)


# --------------------------------------------------------------------------- #
# pycalphad round trip
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_PYCALPHAD, reason="pycalphad is not installed")
def test_pycalphad_reads_back_the_same_free_energies(line_phases):
    """A real TDB reader recovers every phase's free energy, including the sign of the
    odd interaction under non-alphabetical element order and a two-sublattice compound."""
    A, B = line_phases[0], line_phases[-1]
    phases = list(line_phases) + [IdealSolution("ideal", A, B)] + _rk_phases(line_phases) + [_surface_phase(line_phases)]
    text = to_tdb(phases, elements=("MG", "CA"), temperature_range=(1.0, 3000.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", DeprecationWarning)  # pyparsing API deprecations inside pycalphad
        db = Database.from_string(text, fmt="tdb")
    assert sorted(db.phases) == sorted(p.name.upper() for p in phases)
    comps = ["MG", "CA", "VA"]
    T = np.array(TS)
    for phase in line_phases:
        GM = calculate(db, comps, phase.name.upper(), T=T, P=101325, N=1).GM.values.squeeze()
        np.testing.assert_allclose(GM, phase.line_free_energy(T) * J_PER_MOL, atol=ATOL)
    # pycalphad's ideal-mixing term uses a rounded gas constant; swap it in for the comparison
    entropy_scale = J_PER_MOL - float(v.R) / kB
    points = np.column_stack([CS, 1 - CS])  # site fractions in sorted constituent order: CA, MG
    for phase in phases[len(line_phases):]:
        for Ti in TS:
            result = calculate(db, comps, phase.name.upper(), T=Ti, P=101325, N=1, points=points)
            np.testing.assert_allclose(result.X.sel(component="CA").values.squeeze(), CS)
            if isinstance(phase, IdealSolution):
                f = (1 - CS) * A.line_free_energy(Ti) + CS * B.line_free_energy(Ti) - Ti * S(CS)
            else:
                f = phase.free_energy(Ti, CS)
            expected = f * J_PER_MOL + Ti * S(CS) * entropy_scale
            np.testing.assert_allclose(result.GM.values.squeeze(), expected, atol=ATOL)
