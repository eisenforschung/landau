"""Translate an ``@testplot`` PR-comment mention into testplots.py CLI args.

Reads the comment body from ``$COMMENT_BODY`` and pipes it through the
``claude`` CLI (Claude Code), which must respond with a single JSON
object of the form::

    {"args": ["--only", "1d_T", "--poly-method", "fasttsp"], "note": "..."}

The ``args`` list is shell-splattered into ``python tests/integration/testplots.py``
by the calling workflow. ``note`` is a short human-readable summary that gets
echoed back in the PR comment.

Authenticated via ``$CLAUDE_CODE_OAUTH_TOKEN`` (same secret the existing
.github/workflows/claude.yml uses). The script is invoked by
``.github/workflows/testplot-mention.yml``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

PLOT_NAMES = ["1d_T", "1d_mu", "2d_basics", "2d_basics_mu", "2d_toy", "2d_toy_mu"]
POLY_METHODS = ["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"]

SYSTEM = f"""You translate informal `@testplot` mentions in GitHub PR comments
into arguments for `tests/integration/testplots.py`.

The script renders reference phase diagrams. Its relevant flags are:

  --only {{{','.join(PLOT_NAMES)}}} ...
        restrict to a subset of plots. Plot keys:
          1d_T         : 1D temperature diagram (pure-A fcc/hcp/liquid line phases)
          1d_mu        : 1D chemical-potential diagram (hcp vs fcc isothermal)
          2d_basics    : 2D c-T diagram (ideal-solution hcp/fcc/liquid)
          2d_basics_mu : 2D T-mu diagram (same phases as 2d_basics)
          2d_toy       : 2D c-T diagram (regular-solution liquid + intermediate solid)
          2d_toy_mu    : 2D T-mu diagram (same phases as 2d_toy)

  --poly-method {{{','.join(POLY_METHODS)}}}
        polygon-construction method for 2D plots (default: library default).

  --tielines / --no-tielines
        draw tielines on 2D c-T diagrams (default: on; only affects 2d_basics / 2d_toy).

Mapping conventions:
  - "1d T" / "1D temperature" -> --only 1d_T
  - "1d mu" / "1D chemical potential" -> --only 1d_mu
  - "1d" alone -> --only 1d_T 1d_mu
  - "2d" alone -> all four 2D plots
  - "2d basics" -> 2d_basics 2d_basics_mu ; "2d basics c" -> 2d_basics
  - "2d toy" -> 2d_toy 2d_toy_mu ; "2d toy mu" -> 2d_toy_mu
  - "fasttsp" / "fast tsp" -> --poly-method fasttsp
  - "segment fasttsp" / "segment-fasttsp" -> --poly-method segment-fasttsp
  - "tielines off" / "no tielines" -> --no-tielines
  - bare `@testplot` with no other directives -> no extra args (renders everything)

Respond with a single JSON object and NOTHING else (no prose, no code fences):

  {{"args": ["--only", "1d_T"], "note": "1D T diagram"}}

`args` is a flat list of strings to be passed to argparse. `note` is a short
human-readable summary (under 80 chars). Do not include `--out`; the workflow
sets that. If the request is ambiguous, pick the closest reasonable mapping
rather than refusing.
"""

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
    allowed_values = set(PLOT_NAMES) | set(POLY_METHODS) | {"--only", "--poly-method", "--tielines", "--no-tielines"}
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
    return {"args": args, "note": note[:200]}


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
        print(json.dumps({"args": [], "note": "empty mention; rendering all plots"}))
        return

    text = _call_claude(body)
    payload = _validate(_extract_json(text))
    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
