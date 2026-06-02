"""Pick testplots.py args for a PR based on which files it changed.

Reads ``$CHANGED_FILES`` (newline-separated repo-relative paths) and pipes
it through the ``claude`` CLI to choose which plots and poly methods are
worth rendering. Output is the same JSON shape consumed by
.github/workflows/testplot.yml's render step::

    {"args": ["--only", "2d_basics", "--poly-method", "fasttsp"],
     "note": "...", "diff": false}

The validator is intentionally identical to parse_testplot_mention.py's
allow-list: same plot keys, same poly methods, same tieline values. If
the model emits anything else the script errors out instead of letting
arbitrary tokens reach argparse.

Authenticated via ``$CLAUDE_CODE_OAUTH_TOKEN``.
"""
from __future__ import annotations

import json
import os
import sys

from testplot_common import PLOT_NAMES, POLY_METHODS, run_haiku

SYSTEM = f"""You pick which phase-diagram tests to render for a PR.

The script tests/integration/testplots.py renders these plots:
  1d_T_three_stable : 1D temperature diagram (pure-A line phases)
  1d_mu             : 1D chemical-potential diagram (ideal solutions)
  2d_basics         : 2D c-T diagram (ideal-solution hcp/fcc/liquid)
  2d_basics_mu      : 2D T-mu diagram (same phases as 2d_basics)
  2d_toy            : 2D c-T diagram (regular-solution liquid + intermediate solid)
  2d_toy_mu         : 2D T-mu diagram (same phases as 2d_toy)
  excess_free_energy: excess free energy vs concentration (Intermetallics example)

Flags you can emit:
  --only <plot> [...]                     subset of plots (omit = all {len(PLOT_NAMES)})
  --poly-method <method> [...]            cross-product over the 2D plots;
                                          choices: {','.join(POLY_METHODS)}
  --tielines on|off [...]                 tieline modes on 2D c-T plots

Heuristics from changed file paths:
  - landau/poly.py touched
      → render all four 2D c-T/mu plots
      → --poly-method fasttsp tsp segment-fasttsp segment-tsp
        (skip "concave" / "segments" unless the change explicitly affects them)
  - landau/plot.py touched
      → all plots (no extra flags)
  - landau/phases.py or landau/calculate.py touched
      → all plots (no extra flags)
  - landau/interpolate/** touched
      → --only 2d_toy 2d_toy_mu (toy uses interpolators; basics doesn't)
  - tests/integration/testplots.py touched
      → all plots (the test itself changed)
  - only docs/, *.md, *.cfg, pyproject.toml, .github/** touched
      → minimal smoke test: --only 1d_T_three_stable 2d_basics
  - mix of the above
      → take the union; prefer the broader set when a specific rule and an
        "all plots" rule both apply

Respond with a single JSON object and NOTHING else (no prose, no code fences):

  {{"args": ["--only", "2d_basics", "2d_basics_mu", "2d_toy", "2d_toy_mu",
            "--poly-method", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"],
   "note": "poly.py touched: all 2D × tsp variants", "diff": false}}

`args` is flat list of argparse tokens. `note` is a short one-line summary
(under 100 chars) explaining what was selected and why. Do not include
`--out`. `diff` should stay false in this script (the label-driven flow
doesn't do side-by-side renders yet); always include the key.
"""

_FALLBACK = {"args": [], "note": "Haiku select failed; rendering all plots", "diff": False}


def main() -> None:
    changed = os.environ.get("CHANGED_FILES", "").strip()
    if not changed:
        print(json.dumps({"args": [], "note": "no changed files reported; rendering all plots", "diff": False}))
        return

    run_haiku("Changed files in this PR:\n" + changed, SYSTEM, _FALLBACK)


if __name__ == "__main__":
    main()
