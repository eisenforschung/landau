"""Shared helpers for parse_testplot_mention.py and select_testplot_args_from_diff.py."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

POLY_METHODS = ["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"]


def _read_plot_names() -> list[str]:
    """Read available plot names from tests/integration/testplots.py via AST.

    Parses the PLOTS dict keys without importing landau, so it works in
    the trusted parse/select job before the package is installed. Always stays
    in sync with whatever is on the main checkout.
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
    result = subprocess.run(
        [
            "claude",
            "--model", "claude-haiku-4-5-20251001",
            "--append-system-prompt", system,
            "--output-format", "text",
            "--permission-mode", "plan",
            "-p", prompt,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
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
