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

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

POLY_METHODS = ["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"]


def _read_plot_names() -> list[str]:
    """Read available plot names from tests/integration/testplots.py via AST.

    Parses the PLOTS dict keys without importing landau, so it works in
    the trusted select job before the package is installed. Always stays in
    sync with whatever is on the main checkout.
    """
    script = Path(__file__).parents[2] / "tests" / "integration" / "testplots.py"
    try:
        tree = ast.parse(script.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "PLOTS" for t in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                return [
                    k.value
                    for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                ]
    except Exception as exc:
        print(f"warning: could not read PLOTS from testplots.py ({exc}); using fallback list", file=sys.stderr)
    return ["1d_T_three_stable", "1d_mu", "2d_basics", "2d_basics_mu", "2d_toy", "2d_toy_mu", "excess_free_energy"]


PLOT_NAMES = _read_plot_names()

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

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK = {"args": [], "note": "Haiku select failed; rendering all plots", "diff": False}


def _extract_json(text: str) -> dict:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object in model output: {text!r}")
    return json.loads(match.group(0))


def _validate(payload: dict) -> dict:
    args = payload.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError(f"args must be a list of strings, got {args!r}")
    allowed_values = (
        set(PLOT_NAMES)
        | set(POLY_METHODS)
        | {"on", "off"}
        | {"--only", "--poly-method", "--tielines"}
    )
    for a in args:
        if a.startswith("--"):
            if a not in allowed_values:
                raise ValueError(f"disallowed flag in args: {a!r}")
        else:
            if a not in allowed_values:
                raise ValueError(f"disallowed value in args: {a!r}")
    note = payload.get("note", "")
    if not isinstance(note, str):
        raise ValueError(f"note must be a string, got {note!r}")
    diff = payload.get("diff", False)
    if not isinstance(diff, bool):
        raise ValueError(f"diff must be a bool, got {diff!r}")
    return {"args": args, "note": note[:200], "diff": diff}


def _call_claude(changed_files: str) -> str:
    user_msg = "Changed files in this PR:\n" + changed_files
    result = subprocess.run(
        [
            "claude",
            "--model", "claude-haiku-4-5-20251001",
            "--append-system-prompt", SYSTEM,
            "--output-format", "text",
            "--permission-mode", "plan",
            "-p", user_msg,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return result.stdout


def main() -> None:
    changed = os.environ.get("CHANGED_FILES", "").strip()
    if not changed:
        print(json.dumps({"args": [], "note": "no changed files reported; rendering all plots", "diff": False}))
        return

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            text = _call_claude(changed)
            payload = _validate(_extract_json(text))
            json.dump(payload, sys.stdout)
            return
        except Exception as exc:
            last_exc = exc
            print(f"attempt {attempt + 1} failed: {exc}", file=sys.stderr)

    print(f"all attempts failed ({last_exc}); falling back to all plots", file=sys.stderr)
    json.dump(_FALLBACK, sys.stdout)


if __name__ == "__main__":
    main()
