"""Translate an ``@testplot`` PR-comment mention into testplots.py CLI args.

Reads the comment body from ``$COMMENT_BODY`` and pipes it through the
``claude`` CLI (Claude Code), which must respond with a single JSON
object of the form::

    {"args": ["--only", "1d_T_three_stable", "--poly-method", "fasttsp"],
     "note": "...", "diff": false}

The ``args`` list is shell-splattered into ``python tests/integration/testplots.py``
by the calling workflow. ``note`` is a short human-readable summary that gets
echoed back in the PR comment. ``diff`` is a boolean: when true the workflow
also renders the same args against ``main`` and posts a side-by-side
comparison.

Authenticated via ``$CLAUDE_CODE_OAUTH_TOKEN`` (same secret the existing
.github/workflows/claude.yml uses). The script is invoked by
``.github/workflows/testplot-mention.yml``.
"""
from __future__ import annotations

import json
import os
import sys

from testplot_common import PLOT_NAMES, POLY_METHODS, run_haiku

SYSTEM = f"""You translate informal `@testplot` mentions in GitHub PR comments
into arguments for `tests/integration/testplots.py`.

The script renders reference phase diagrams. All multi-valued flags
cross-product, so `--poly-method a b --tielines on off` against four 2D
plots renders 4 x 2 x 2 = 16 figures. Be generous: if the user says
"everything" or names a plural ("methods", "tielines on and off"), pass
every matching value rather than picking one.

Flags:

  --only {{{','.join(PLOT_NAMES)}}} [...]
        restrict to a subset of plots. Plot keys:
          1d_T_three_stable : 1D temperature diagram (pure-A fcc/hcp/liquid line phases)
          1d_mu             : 1D chemical-potential diagram (hcp vs fcc isothermal)
          2d_basics         : 2D c-T diagram (ideal-solution hcp/fcc/liquid)
          2d_basics_mu      : 2D T-mu diagram (same phases as 2d_basics)
          2d_toy            : 2D c-T diagram (regular-solution liquid + intermediate solid)
          2d_toy_mu         : 2D T-mu diagram (same phases as 2d_toy)
          excess_free_energy: excess free energy vs concentration (Intermetallics example)

  --poly-method {{{','.join(POLY_METHODS)}}} [...]
        one or more polygon-construction methods, cross-producted over the
        2D plots. Omit entirely to use the library default.

  --tielines {{on,off}} [...]
        one or more tieline modes, cross-producted over 2D c-T plots
        (only affects 2d_basics / 2d_toy). Default: on.

Mapping conventions:
  - "1d T" / "1D temperature" -> --only 1d_T_three_stable
  - "1d mu" / "1D chemical potential" -> --only 1d_mu
  - "1d" alone -> --only 1d_T_three_stable 1d_mu
  - "2d" alone -> all four 2D c-T/mu plots
  - "2d basics" -> 2d_basics 2d_basics_mu ; "2d basics c" -> 2d_basics
  - "2d toy" -> 2d_toy 2d_toy_mu ; "2d toy mu" -> 2d_toy_mu
  - "excess" / "excess free energy" -> --only excess_free_energy
  - "fasttsp" / "fast tsp" -> --poly-method fasttsp
  - "all tsp methods" / "tsp methods" -> --poly-method tsp fasttsp segment-tsp segment-fasttsp
  - "segment methods" / "segment tsp methods" -> --poly-method segment-tsp segment-fasttsp
  - "all poly methods" / "every poly-method" -> --poly-method concave segments fasttsp tsp segment-fasttsp segment-tsp
  - "tielines off" / "no tielines" -> --tielines off
  - "with and without tielines" / "both tieline modes" -> --tielines on off
  - "everything" / "all plots" / bare `@testplot` -> no extra args (renders every plot once)

Diff mode:
  Set `"diff": true` when the user wants a side-by-side comparison against
  `main`. Triggers: "diff", "diff main", "vs main", "compare with main",
  "before/after", "side by side", "regression". Otherwise omit the field
  or set it to false.

Respond with a single JSON object and NOTHING else (no prose, no code fences):

  {{"args": ["--only", "1d_T_three_stable"], "note": "1D T diagram", "diff": false}}

`args` is a flat list of strings to be passed to argparse. `note` is a short
human-readable summary (under 80 chars). Do not include `--out`; the workflow
sets that. If the request is ambiguous, pick the closest reasonable mapping
rather than refusing.
"""

_FALLBACK = {"args": [], "note": "Haiku parse failed; rendering all plots", "diff": False}


def main() -> None:
    body = os.environ.get("COMMENT_BODY", "").strip()
    if not body:
        print(json.dumps({"args": [], "note": "empty mention; rendering all plots", "diff": False}))
        return

    run_haiku(body, SYSTEM, _FALLBACK)


if __name__ == "__main__":
    main()
