"""
Error propagation for pedestrians.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from .phases import Phase


@dataclass(frozen=True)
class RandomlyShiftedPhase(Phase):
    """
    Add a constant noise to an arbitrary phase.  Purely for error estimation via bootstrapping.
    """

    name: str
    phase: Phase
    noise: float = 1e-3
    shift: float = field(init=False)

    def __post_init__(self, *args, **kwargs):
        object.__setattr__(self, "shift", np.random.normal(loc=0, scale=self.noise))

    def semigrand_potential(self, T, mu):
        return self.phase.semigrand_potential(T, mu) + self.shift

    def concentration(self, T, mu):
        return self.phase.concentration(T, mu)


def _resample_borders_once(phases, Ts, dmus, noise_norm, run=0):
    noise_phases = [RandomlyShiftedPhase(p.name, p, noise=noise_norm) for p in phases]
    df = calc_phase_diagram(noise_phases, Ts, dmus, refine=True).query("border and -inf<mu<inf")
    df["run"] = run
    return df


def resample_borders(phases, Ts, dmus, noise_norm=1e-3, n=100, cores=20):
    with ProcessPoolExecutor(max_workers=cores) as pool:
        return pd.concat(
            list(pool.map(_resample_borders_once, [phases] * n, [Ts] * n, [dmus] * n, [noise_norm] * n, range(n))),
            ignore_index=True,
        )
