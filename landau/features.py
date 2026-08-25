"""Object interface over the dataframe returned by :func:`landau.calculate.calc_phase_diagram`.

The free functions in :mod:`landau.calculate` and :mod:`landau.plot` operate
directly on the ``(T, mu, phase, c, phi, ...)`` dataframe.  This module wraps
that same frame in a small read-only object layer so callers can ask for the
*features* of a diagram by name instead of re-deriving them from the columns:

* a :class:`Domain` is a connected single-phase region (one phase, ``k = 1``),
* a :class:`Coexistence` is a region where ``k >= 2`` phases coexist; its
  :attr:`~Coexistence.codimension` is ``k - 1``, so a two-phase boundary line
  and a triple point are the same kind of object at different ``k``.

The frame stays the source of truth and is always reachable via
:attr:`PhaseDiagram.frame`; the objects only parse it.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Mapping

import pandas as pd

if TYPE_CHECKING:
    from .phases import Phase

__all__ = ["Locus", "Domain", "Coexistence", "PhaseDiagram"]


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


@dataclass(frozen=True, eq=False)
class Domain:
    """A connected single-phase region of a phase diagram (``k = 1``).

    One phase is stable across a single connected window of the sampled
    variables.  A phase that is stable in two disjoint windows (a solid
    solution interrupted by an intermetallic, say) yields two separate
    ``Domain`` objects.

    Attributes
    ----------
    phase : str or Phase
        The stable phase.  A :class:`~landau.phases.Phase` when
        :meth:`PhaseDiagram.from_frame` was given a phase mapping, otherwise
        the phase name.
    frame : pandas.DataFrame
        The sub-frame of interior sample rows that make up this region.
    """

    phase: "str | Phase"
    frame: pd.DataFrame = field(repr=False)

    def __repr__(self) -> str:
        return f"Domain(phase={self.phase!r}, points={len(self.frame)})"


@dataclass(frozen=True, eq=False)
class Coexistence:
    """A region where ``k >= 2`` phases coexist.

    Covers what are geometrically a two-phase boundary line (``k = 2``) and a
    triple point (``k = 3``) as a single kind of object, distinguished only by
    :attr:`codimension`.  A miscibility gap is the ``k = 2`` case where both
    coexisting phases are the same phase (two composition branches), see
    :attr:`is_miscibility_gap`.

    Attributes
    ----------
    phases : tuple of (str or Phase)
        The coexisting phases as a multiset of length ``k``: two equal entries
        for a miscibility gap, ``k`` distinct entries otherwise.
        :class:`~landau.phases.Phase` instances when
        :meth:`PhaseDiagram.from_frame` was given a phase mapping, otherwise
        names.
    frame : pandas.DataFrame
        The refined boundary rows that trace this coexistence (one
        ``boundary_id`` group).
    """

    phases: tuple
    frame: pd.DataFrame = field(repr=False)

    @property
    def codimension(self) -> int:
        """``k - 1``: 1 for a coexistence line, 2 for a triple point."""
        return len(self.phases) - 1

    @property
    def is_miscibility_gap(self) -> bool:
        """True when the two coexisting phases are the same phase."""
        return len(self.phases) == 2 and self.phases[0] == self.phases[1]

    def __repr__(self) -> str:
        return f"Coexistence(phases={tuple(self.phases)!r}, points={len(self.frame)})"


_REQUIRED_COLUMNS = ("phase", "stable", "locus")


@dataclass(frozen=True, eq=False)
class PhaseDiagram:
    """Object view over a :func:`~landau.calculate.calc_phase_diagram` frame.

    Build with :meth:`from_frame`.  The wrapped :attr:`frame` stays the source
    of truth; :attr:`domains` and :attr:`coexistences` only parse it.
    """

    frame: pd.DataFrame = field(repr=False)
    phase_map: "Mapping[str, Phase] | None" = None

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, phases: "Mapping[str, Phase] | object | None" = None
    ) -> "PhaseDiagram":
        """Wrap a phase-diagram dataframe.

        Args:
            frame: a dataframe as returned by
                :func:`landau.calculate.calc_phase_diagram` (needs the
                ``phase``, ``stable`` and ``locus`` columns).
            phases: optional mapping of name to :class:`~landau.phases.Phase`,
                or any iterable of phases.  When given, :attr:`Domain.phase`
                and :attr:`Coexistence.phases` are resolved to the phase
                objects instead of bare names.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"frame is missing required column(s) {missing}; pass a frame from "
                "calc_phase_diagram"
            )
        if phases is not None and not isinstance(phases, Mapping):
            phases = {p.name: p for p in phases}
        return cls(frame=frame, phase_map=phases)

    @property
    def phases(self):
        """The phases of the diagram.

        The ``{name: Phase}`` mapping if one was passed to :meth:`from_frame`,
        otherwise the set of phase names present in :attr:`frame`.
        """
        if self.phase_map is not None:
            return self.phase_map
        return set(self.frame["phase"].unique())

    def _resolve(self, name: str):
        if self.phase_map is not None:
            return self.phase_map[name]
        return name

    def domains(self, distance_threshold: float = 0.5) -> list[Domain]:
        """The single-phase regions, one :class:`Domain` per connected window.

        Args:
            distance_threshold: passed to
                :func:`landau.plot.cluster_phase`; the agglomerative-clustering
                cut-off in normalised ``(T, c)`` space that decides which
                interior samples belong to the same connected region.  Lower it
                when two stable windows of one phase are wrongly merged.
        """
        from .plot import cluster_phase

        interior = self.frame.query("stable and locus == 'interior'").copy()
        if interior.empty:
            return []
        interior = cluster_phase(interior, distance_threshold=distance_threshold)
        # cluster_phase marks points it could not assign with phase_unit == -1
        # (mirrors get_polygons, which drops them).
        interior = interior[interior["phase_unit"] >= 0]
        helper = ["phase_unit", "phase_id"]
        return [
            Domain(phase=self._resolve(g["phase"].iloc[0]), frame=g.drop(columns=helper))
            for _, g in interior.groupby("phase_id", sort=False)
        ]

    @property
    def coexistences(self) -> list[Coexistence]:
        """The coexistence features (``k >= 2``), one per ``boundary_id`` line.

        ``boundary_id`` only counts up from zero within a single refiner, so
        the ``(refined, boundary_id)`` pair is what identifies one coexistence
        line across the whole frame.
        """
        boundary = self.frame[self.frame["locus"] != Locus.INTERIOR]
        if boundary.empty or "boundary_id" not in boundary.columns:
            return []
        out = []
        for _, g in boundary.groupby(["refined", "boundary_id"], sort=False):
            names = tuple(sorted(g["phase"].unique()))
            k = 3 if (g["locus"] == Locus.TRIPLE).any() else 2
            if len(names) < k:
                # a miscibility gap is one phase splitting into k branches
                names = names * k
            phases = tuple(self._resolve(n) for n in names)
            out.append(Coexistence(phases=phases, frame=g))
        return out

    def __repr__(self) -> str:
        phases = self.phase_map if self.phase_map is not None else self.phases
        return f"PhaseDiagram(phases={len(phases)}, rows={len(self.frame)})"
