"""Shared helpers for testplot_args.py (parse and select subcommands)."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

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
