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
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# plot list + Haiku transform helpers
# ---------------------------------------------------------------------------

POLY_METHODS = ["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"]


_FALLBACK_PLOTS = [
    ("1d_T_three_stable", "1D temperature diagram"),
    ("1d_mu_three_stable", "1D chemical-potential diagram"),
    ("2d_basics", "2D c-T diagram (ideal solutions)"),
    ("2d_basics_mu", "2D T-mu diagram (ideal solutions)"),
    ("2d_toy", "2D c-T diagram (regular solution)"),
    ("2d_toy_mu", "2D T-mu diagram (regular solution)"),
    ("excess_free_energy", "excess free energy vs concentration"),
]


def _read_plots() -> list[tuple[str, str]]:
    """Read (plot name, one-line description) pairs from testplots.py via AST.

    Parses the PLOTS dict keys and the first docstring line of each mapped
    plot function, without importing landau, so it works in the trusted
    parse/select job before the package is installed. The list always stays
    in sync with whatever plots exist on the main checkout, so the Haiku
    prompts never reference a renamed or removed plot.
    """
    script = Path(__file__).parents[2] / "tests" / "integration" / "testplots.py"
    try:
        tree = ast.parse(script.read_text())
        summaries = {
            node.name: next(iter((ast.get_docstring(node) or "").strip().splitlines()), "").strip()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "PLOTS" for t in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                plots = []
                for key, value in zip(node.value.keys, node.value.values):
                    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                        continue
                    fn = value.elts[0] if isinstance(value, ast.Tuple) and value.elts else None
                    summary = summaries.get(fn.id, "") if isinstance(fn, ast.Name) else ""
                    plots.append((key.value, summary))
                if plots:
                    return plots
    except Exception as exc:
        print(f"warning: could not read PLOTS from testplots.py ({exc}); using fallback list", file=sys.stderr)
    return list(_FALLBACK_PLOTS)


PLOTS = _read_plots()
PLOT_NAMES = [name for name, _ in PLOTS]
# Markdown bullet list of "  name : description" lines for the Haiku prompts.
PLOT_LIST = "\n".join(f"  {name:<22}: {desc}" if desc else f"  {name}" for name, desc in PLOTS)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def _call_claude(prompt: str, system: str) -> str:
    # Run Haiku as a pure text->JSON transform, not an interactive agent:
    #   --system-prompt   replaces the default agentic prompt (instead of
    #                     appending to it), so the model follows our JSON
    #                     instructions rather than asking the user what to do;
    #   --tools ""        disables every tool, so there is no plan-mode dance
    #                     and the model can only emit text;
    #   cwd=<tmp>         isolates the call from the repo so CLAUDE.md is not
    #                     auto-discovered and read as a PR-review task.
    # Without these, Haiku loaded the landau CLAUDE.md and replied with prose
    # like "What would you like me to do with these changes?" instead of JSON.
    with tempfile.TemporaryDirectory() as workdir:
        result = subprocess.run(
            [
                "claude",
                "--model", "claude-haiku-4-5-20251001",
                "--system-prompt", system,
                "--tools", "",
                "--output-format", "text",
                "-p", prompt,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
            cwd=workdir,
        )
    return result.stdout


def run_haiku(prompt: str, system: str, fallback: dict) -> None:
    """Call Haiku with retry, validate the JSON response, and write it to stdout."""
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            text = _call_claude(prompt, system)
            payload = _validate(_extract_json(text))
            json.dump(payload, sys.stdout)
            return
        except Exception as exc:
            last_exc = exc
            print(f"attempt {attempt + 1} failed: {exc}", file=sys.stderr)

    print(f"all attempts failed ({last_exc}); falling back to all plots", file=sys.stderr)
    json.dump(fallback, sys.stdout)


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

  --only <plot> [...]
        restrict to a subset of plots. Available plot keys (the ONLY valid
        values for --only — never invent or guess a key that is not listed):
{PLOT_LIST}

  --poly-method {{{','.join(POLY_METHODS)}}} [...]
        one or more polygon-construction methods, cross-producted over the
        2D plots. Omit entirely to use the library default.

  --tielines {{on,off}} [...]
        one or more tieline modes, cross-producted over 2D c-T plots
        (only affects 2d_* c-T plots). Default: on.

Mapping conventions (match against the plot keys listed above; if several
keys share a prefix, e.g. multiple `1d_mu_*`, include all of them):
  - "1d T" / "1D temperature" -> every 1d_T_* key
  - "1d mu" / "1D chemical potential" -> every 1d_mu_* key
  - "1d" alone -> every 1d_* key
  - "2d" alone -> every 2d_* key
  - "2d basics" -> every 2d_basics* key ; "2d basics c" -> 2d_basics
  - "2d toy" -> every 2d_toy* key ; "2d toy mu" -> 2d_toy_mu
  - "excess" / "excess free energy" -> every excess_free_energy* key
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

The script tests/integration/testplots.py renders these plots. These keys are
the ONLY valid values for --only; never invent or guess a key not listed here:
{PLOT_LIST}

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
