"""Unified CLI for testplots.py argument selection — two subcommands.

  parse   — translate an ``@testplot`` PR-comment mention into CLI args.
            Reads the comment body from ``$COMMENT_BODY``.

  select  — pick args based on which files a PR changed.
            Reads a newline-separated list of paths from ``$CHANGED_FILES``.

Both subcommands emit a single JSON object to stdout::

    {"args": ["--only", "2d_basics", "--poly-method", "fasttsp"],
     "note": "...", "diff": false}

Authenticated via ``$CLAUDE_CODE_OAUTH_TOKEN``.
Called by ``.github/workflows/testplot-mention.yml`` (parse) and
``.github/workflows/testplot.yml`` (select).
"""
from __future__ import annotations

import argparse
import json
import os

from testplot_common import PLOT_NAMES, POLY_METHODS, run_haiku

# ---------------------------------------------------------------------------
# parse subcommand
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = f"""You translate informal `@testplot` mentions in GitHub PR comments
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

_PARSE_FALLBACK = {"args": [], "note": "Haiku parse failed; rendering all plots", "diff": False}


def _cmd_parse() -> None:
    body = os.environ.get("COMMENT_BODY", "").strip()
    if not body:
        print(json.dumps({"args": [], "note": "empty mention; rendering all plots", "diff": False}))
        return
    run_haiku(body, _PARSE_SYSTEM, _PARSE_FALLBACK)


# ---------------------------------------------------------------------------
# select subcommand
# ---------------------------------------------------------------------------

_SELECT_SYSTEM = f"""You pick which phase-diagram tests to render for a PR.

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

_SELECT_FALLBACK = {"args": [], "note": "Haiku select failed; rendering all plots", "diff": False}


def _cmd_select() -> None:
    changed = os.environ.get("CHANGED_FILES", "").strip()
    if not changed:
        print(json.dumps({"args": [], "note": "no changed files reported; rendering all plots", "diff": False}))
        return
    run_haiku("Changed files in this PR:\n" + changed, _SELECT_SYSTEM, _SELECT_FALLBACK)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select testplots.py args via Haiku.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("parse", help="translate @testplot mention ($COMMENT_BODY) into args")
    sub.add_parser("select", help="pick args from changed files ($CHANGED_FILES)")
    args = parser.parse_args()
    if args.command == "parse":
        _cmd_parse()
    else:
        _cmd_select()


if __name__ == "__main__":
    main()
