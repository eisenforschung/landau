"""Frame of the tests/integration/test_border_coverage.py system. Run on claude/cc-mu-window (#600)."""
import sys
import numpy as np
import landau.calculate as ldc
import landau.phases as ldp


def ideal(name, E_A, S_A, E_B, S_B):
    return ldp.IdealSolution(
        name,
        ldp.LinePhase(name + "A", fixed_concentration=0, line_energy=E_A, line_entropy=S_A * ldp.kB),
        ldp.LinePhase(name + "B", fixed_concentration=1, line_energy=E_B, line_entropy=S_B * ldp.kB),
    )


phases = [ideal("hcp", -2.975, 1.8, -1.95, 1.1), ideal("fcc", -3.00, 1.0, -2.00, 1.1), ideal("liquid", -2.75, 5.0, -1.75, 4.4)]
df = ldc.calc_phase_diagram(phases, np.linspace(200, 1000, 50), mu=np.linspace(0.5, 1.5, 50))
df.to_pickle(sys.argv[1])
