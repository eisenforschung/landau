"""Domain vocabulary for features of a phase diagram."""

from enum import StrEnum

__all__ = ["Locus"]


class Locus(StrEnum):
    """Classifies what part of the phase diagram a sampled point belongs to.

    Backs the ``locus`` column of the dataframes returned by
    :func:`landau.calculate.calc_phase_diagram`.  Members are strings, so
    rows can be selected either by member or by plain value::

        df.query("locus == 'triple'")
        df[df["locus"] == Locus.TRIPLE]

    INTERIOR
        A regular sample inside a single-phase region, including the
        synthetic frame points emitted at the edges of the sampling window.
    BOUNDARY
        A refined point on a two-phase coexistence line, including the two
        branches of a miscibility gap.
    TRIPLE
        A refined point at which three phases coexist.
    """

    INTERIOR = "interior"
    BOUNDARY = "boundary"
    TRIPLE = "triple"
