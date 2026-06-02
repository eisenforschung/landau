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
    the trusted parse job before the package is installed. Always stays in
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

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK = {"args": [], "note": "Haiku parse failed; rendering all plots", "diff": False}


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


def _call_claude(body: str) -> str:
    result = subprocess.run(
        [
            "claude",
            "--model", "claude-haiku-4-5-20251001",
            "--append-system-prompt", SYSTEM,
            "--output-format", "text",
            "--permission-mode", "plan",
            "-p", body,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return result.stdout


def main() -> None:
    body = os.environ.get("COMMENT_BODY", "").strip()
    if not body:
        print(json.dumps({"args": [], "note": "empty mention; rendering all plots", "diff": False}))
        return

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            text = _call_claude(body)
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
