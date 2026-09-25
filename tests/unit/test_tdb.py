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
from pyiron_snippets.import_alarm import ImportAlarm
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

# ImportAlarm keeps its message only when the import fails, so it needs one to report the failure.
with ImportAlarm("pycalphad is not installed; pip install 'landau[test-pycalphad]'") as pycalphad_alarm:
    from pycalphad import Database, calculate
    from pycalphad import variables as v

needs_pycalphad = pytest.mark.skipif(pycalphad_alarm.message is not None, reason="pycalphad is not installed")

J_PER_MOL = eV * Avogadro
R = Boltzmann * Avogadro
"""Not scipy's ``R``: scipy 1.11 tabulates it rounded to 8.314462618."""
ATOL = 1e-6
"""J/mol, absolute: every comparison passes ``rtol=0``.  Closed forms are written to full
precision, so this is evaluation round-off (measured 1.5e-7 at worst)."""
FIT_ATOL = 1e-3
"""J/mol; landau's iterative Redlich-Kister fit against the least-squares solution the export
writes, on the ``line_phases`` fixture (measured 1.7e-4)."""

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


def _least_squares_free_energy(line_phases, n_orders, T, c, add_entropy=False):
    """landau's Redlich-Kister model solved exactly, in eV/atom: the chord through the
    terminals plus ``L_v`` from a linear least-squares fit to the entropy-removed rest."""
    cs = np.array([p.line_concentration for p in line_phases], dtype=float)
    h = np.array([p.line_free_energy(T) for p in line_phases], dtype=float)
    if not add_entropy:
        h = h + T * S(cs)
    order = cs.argsort()
    cs, h = cs[order], h[order]

    def mix(x):
        return (x * (1 - x))[:, None] * np.vander(2 * x - 1, n_orders, increasing=True)

    L = np.linalg.lstsq(mix(cs), h - h[0] - (h[-1] - h[0]) * cs, rcond=None)[0]
    c = np.asarray(c, dtype=float)
    return h[0] + (h[-1] - h[0]) * c + mix(c) @ L - T * S(c)


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


@pytest.mark.parametrize("elements", [("ABC", "B"), ("A", "A"), ("A",), ("A", "B", "C"), ("A1", "B"), ("VA", "B")])
def test_elements_must_be_two_distinct_symbols(terminals, elements):
    with pytest.raises(ValueError, match="element"):
        to_tdb(terminals, elements=elements)


@pytest.mark.parametrize(
    "name, tdb_name", [("Mg2Ca-C14", "MG2CA_C14"), ("L1_2 (ordered)", "L1_2_ORDERED"), ("fcc", "FCC")]
)
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


def test_phase_names_longer_than_24_characters_raise():
    with pytest.raises(ValueError, match="longer than 24"):
        to_tdb([LinePhase("x" * 25, 0.5, -1.0)])


def test_24_character_names_keep_every_line_within_78_characters(line_phases):
    """The longest names, two-letter elements and a two-sublattice compound with full-precision site ratios."""
    phases = [LinePhase("C" * 24, 1 / 3, -2.9, kB), RegularSolution("R" * 24, line_phases, num_coeffs=2)]
    text = to_tdb(phases, elements=("MG", "CA"), temperature_range=(298.15, 6000.0))
    assert f"PHASE {'C' * 24} % 2 0.6666666666666667 0.3333333333333333 !" in text.splitlines()
    assert max(len(line) for line in text.splitlines()) <= 78


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
    np.testing.assert_allclose(params[parameter][2](T), phase.line_free_energy(T) * J_PER_MOL, rtol=0, atol=ATOL)


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
    G_sampled = -3.0 + 2e-4 * T - 3e-4 * T * np.log(T)
    phase = TemperatureDependentLinePhase("x", 0.25, T, G_sampled, interpolator=interpolator)
    G = _parameters(to_tdb([phase]))["G(X,A:B;0)"][2]
    Tq = np.linspace(250.0, 2100.0, 7)
    np.testing.assert_allclose(G(Tq), phase.line_free_energy(Tq) * J_PER_MOL, rtol=0, atol=ATOL)


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
        written = _solution_free_energy(text, "SOL", ("A", "B"), T, CS)
        np.testing.assert_allclose(written, expected * J_PER_MOL, rtol=0, atol=ATOL)


@pytest.mark.parametrize("add_entropy", [False, True])
@pytest.mark.parametrize("index", range(4), ids=["regular", "interpolating", "slow", "fast"])
def test_redlich_kister_phases(line_phases, index, add_entropy):
    """The written phase is the least-squares Redlich-Kister fit through the line phases at
    every (T, c), and so matches ``free_energy`` to landau's own fit convergence."""
    phase = _rk_phases(line_phases, add_entropy)[index]
    text = to_tdb([phase])
    name = phase.name.upper()
    assert _interactions(text, name) == [f"L({name},A,B;0)", f"L({name},A,B;1)"]
    for T in TS:
        written = _solution_free_energy(text, name, ("A", "B"), T, CS)
        exact = _least_squares_free_energy(line_phases, 2, T, CS, add_entropy)
        np.testing.assert_allclose(written, exact * J_PER_MOL, rtol=0, atol=ATOL)
        np.testing.assert_allclose(written, phase.free_energy(T, CS) * J_PER_MOL, rtol=0, atol=FIT_ATOL)


def test_redlich_kister_export_is_the_least_squares_fit_when_ill_conditioned():
    """Eight line phases crowded between c=0.6 and 0.9 under the default RedlichKister(5):
    the least-squares problem is badly conditioned, and the written phase is still its solution."""
    cs = [0.0, 0.595, 0.619, 0.697, 0.751, 0.808, 0.888, 1.0]
    energies = [-2.815, -3.108, -3.137, -2.994, -2.857, -2.897, -3.531, -2.857]
    entropies = [0.873, 1.536, 0.705, 2.099, 2.529, 1.183, 1.826, 0.956]
    line_phases = [LinePhase(f"p{i}", c, e, s * kB) for i, (c, e, s) in enumerate(zip(cs, energies, entropies))]
    phase = FastInterpolatingPhase("liq", line_phases)
    assert phase.interpolator == RedlichKister(5)
    text = to_tdb([phase])
    assert len(_interactions(text, "LIQ")) == 5
    for T in TS:
        exact = _least_squares_free_energy(line_phases, 5, T, CS)
        np.testing.assert_allclose(
            _solution_free_energy(text, "LIQ", ("A", "B"), T, CS), exact * J_PER_MOL, rtol=0, atol=ATOL
        )


@pytest.mark.parametrize(
    "build, order",
    [
        (lambda phases: RegularSolution("x", phases, num_coeffs=2), (4, 1, 0, 2, 3)),
        (lambda phases: FastInterpolatingPhase("x", phases, interpolator=RedlichKister(2)), (4, 1, 0, 2, 3)),
        (lambda phases: FastInterpolatingPhase("x", phases, interpolator=RedlichKister(2)), (1, 5, 0, 2, 3, 4)),
        (lambda phases: FastInterpolatingPhase("x", phases, interpolator=RedlichKister(2)), (5, 1, 0, 2, 3, 4)),
    ],
    ids=["regular-shuffled", "fast-shuffled", "fast-second-terminal-first", "fast-second-terminal-last"],
)
def test_redlich_kister_terminals_anywhere_in_phases(line_phases, build, order):
    """The terminals are taken where the fit takes them, not from the ends of ``phases``;
    with a second line phase at c=0 (index 5 below; RegularSolution refuses one), whichever
    the fit picks."""
    candidates = (*line_phases, LinePhase("fccA2", 0.0, -3.05, 1.1 * kB))
    phases = [candidates[i] for i in order]
    text = to_tdb([build(phases)])
    for T in TS:
        exact = _least_squares_free_energy(phases, 2, T, CS)
        written = _solution_free_energy(text, "X", ("A", "B"), T, CS)
        np.testing.assert_allclose(written, exact * J_PER_MOL, rtol=0, atol=ATOL)


def test_interior_concentration_close_to_a_terminal_counts_as_interior():
    """c = 0.999995 is within ``np.isclose`` of 1 but is a line phase of its own, the second of
    the two interior concentrations two orders need."""
    phases = [
        LinePhase("a", 0.0, -3.0, kB),
        LinePhase("m", 0.5, -3.1, 1.5 * kB),
        LinePhase("n", 0.999995, -2.6, kB),
        LinePhase("b", 1.0, -2.5, kB),
    ]
    text = to_tdb([RegularSolution("x", phases, num_coeffs=2, add_entropy=True)])
    assert _interactions(text, "X") == ["L(X,A,B;0)", "L(X,A,B;1)"]


def test_surface_phase_with_a_near_zero_terminal(line_phases):
    """A terminal at c = 1e-12 leaves ``concentration_range = (1e-12, 1)``, which the surface
    fit accepts as the full axis, so the export does too."""
    phases = [LinePhase("a", 1e-12, -3.0, kB), LinePhase("m", 0.5, -3.1, 1.5 * kB), LinePhase("b", 1.0, -2.5, kB)]
    phase = Surface2DInterpolatingPhase(
        "x",
        phases,
        surface_interpolator=CalphadSurface2DInterpolator(num_coeffs=1, coeff_poly_order=1),
        temperature_range=(300.0, 2000.0),
    )
    assert phase.concentration_range == (1e-12, 1)
    text = to_tdb([phase])
    for T in TS:
        np.testing.assert_allclose(
            _solution_free_energy(text, "X", ("A", "B"), T, CS),
            phase.free_energy(T, CS) * J_PER_MOL,
            rtol=0,
            atol=ATOL,
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda A, m, twin, B: RegularSolution("x", [A, m, twin, B], num_coeffs=2),
        lambda A, m, twin, B: FastInterpolatingPhase("x", [A, m, twin, B], interpolator=RedlichKister(2)),
    ],
    ids=["regular", "fast"],
)
def test_redlich_kister_fit_that_is_not_unique_raises(line_phases, build):
    """Two orders fitted through a single interior concentration leave one ``L_v`` free."""
    A, m, B = line_phases[0], line_phases[1], line_phases[-1]
    twin = LinePhase("twin", m.line_concentration, -2.9, kB)
    with pytest.raises(ValueError, match=r'"x".*got 1; the fit is not unique'):
        to_tdb([build(A, m, twin, B)])


@pytest.mark.parametrize("cls", [InterpolatingPhase, FastInterpolatingPhase])
def test_redlich_kister_phase_over_the_terminals_alone_raises(terminals, cls):
    with pytest.raises(ValueError, match=r'"x".*needs a line phase between the terminals'):
        to_tdb([cls("x", terminals)])


def test_odd_interactions_flip_sign_for_unsorted_elements(line_phases):
    """Parameters are written with alphabetically sorted constituents, which for
    ``elements=("MG", "CA")`` puts the ``c=1`` component first and negates odd orders."""
    phase = _rk_phases(line_phases)[3]
    text = to_tdb([phase], elements=("MG", "CA"))
    assert _interactions(text, "FAST") == ["L(FAST,CA,MG;0)", "L(FAST,CA,MG;1)"]
    reference = _parameters(to_tdb([phase]))
    written = _parameters(text)
    T = np.array(TS)
    np.testing.assert_allclose(written["L(FAST,CA,MG;0)"][2](T), reference["L(FAST,A,B;0)"][2](T), rtol=0, atol=ATOL)
    np.testing.assert_allclose(written["L(FAST,CA,MG;1)"][2](T), -reference["L(FAST,A,B;1)"][2](T), rtol=0, atol=ATOL)
    for T in TS:
        exact = _least_squares_free_energy(line_phases, 2, T, CS)
        np.testing.assert_allclose(
            _solution_free_energy(text, "FAST", ("MG", "CA"), T, CS), exact * J_PER_MOL, rtol=0, atol=ATOL
        )


def test_surface_phase(line_phases):
    phase = _surface_phase(line_phases)
    text = to_tdb([phase])
    assert _interactions(text, "SURF") == ["L(SURF,A,B;0)", "L(SURF,A,B;1)"]
    for T in TS:
        np.testing.assert_allclose(
            _solution_free_energy(text, "SURF", ("A", "B"), T, CS),
            phase.free_energy(T, CS) * J_PER_MOL,
            rtol=0,
            atol=ATOL,
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
        (lambda A, B, mid: FastInterpolatingPhase("x", [A, mid]), "confined to concentration_range"),
        (lambda A, B, mid: InterpolatingPhase("x", [mid, A, B]), "only a RedlichKister"),
        (lambda A, B, mid: RegularSolution("x", [A, _stitched("s", 0.5), B]), "no closed form in T"),
        (
            lambda A, B, mid: Surface2DInterpolatingPhase(
                "x",
                [A, mid, B],
                surface_interpolator=SoftplusSurface2DInterpolator(),
                temperature_range=(300.0, 2000.0),
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
        (
            lambda A, B, mid: Surface2DInterpolatingPhase(
                "x",
                [A, mid, B],
                surface_interpolator=CalphadSurface2DInterpolator(num_coeffs=1),
                temperature_range=(300.0, 2000.0),
                concentration_range=(0.2, 0.8),
            ),
            "confined to concentration_range",
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
        "restricted-concentration-range",
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
@pytest.mark.pycalphad
@needs_pycalphad
def test_pycalphad_reads_back_the_same_free_energies(line_phases):
    """A real TDB reader forms every phase's free energy as this file's evaluator does,
    including the sign of the odd interactions under non-alphabetical element order and a
    two-sublattice compound."""
    A, B = line_phases[0], line_phases[-1]
    phases = [*line_phases, IdealSolution("ideal", A, B), *_rk_phases(line_phases), _surface_phase(line_phases)]
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
        np.testing.assert_allclose(GM, phase.line_free_energy(T) * J_PER_MOL, rtol=0, atol=ATOL)
    # pycalphad's ideal-mixing term uses a rounded gas constant; swap it in for the comparison
    mixing = CS * np.log(CS) + (1 - CS) * np.log(1 - CS)
    points = np.column_stack([CS, 1 - CS])  # site fractions in sorted constituent order: CA, MG
    for phase in phases[len(line_phases):]:
        name = phase.name.upper()
        for Ti in TS:
            result = calculate(db, comps, name, T=Ti, P=101325, N=1, points=points)
            np.testing.assert_allclose(result.X.sel(component="CA").values.squeeze(), CS)
            expected = _solution_free_energy(text, name, ("MG", "CA"), Ti, CS) + (float(v.R) - R) * Ti * mixing
            np.testing.assert_allclose(result.GM.values.squeeze(), expected, rtol=0, atol=ATOL)


@pytest.mark.pycalphad
@needs_pycalphad
def test_pycalphad_evaluates_past_the_temperature_range(terminals):
    """What ``to_tdb`` documents for ``temperature_range``: pycalphad ignores the limits."""
    A, _ = terminals
    db = Database.from_string(to_tdb([A], temperature_range=(500.0, 1000.0)), fmt="tdb")
    T = np.array([300.0, 1500.0])
    GM = calculate(db, ["A", "B", "VA"], "FCCA", T=T, P=101325, N=1).GM.values.squeeze()
    np.testing.assert_allclose(GM, A.line_free_energy(T) * J_PER_MOL, rtol=0, atol=ATOL)
