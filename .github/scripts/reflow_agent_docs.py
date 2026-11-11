"""Reflow the agent guides to one sentence per line (semantic line breaks).

Usage (from anywhere in the repo):

    python .github/scripts/reflow_agent_docs.py          # rewrite AGENTS.md and CLAUDE.md in place
    python .github/scripts/reflow_agent_docs.py --check  # list files that need reflowing; exit 1 if any
    python .github/scripts/reflow_agent_docs.py FILE...  # only these files

Copied from pmrv/fleche; keep the two in step.

Every sentence and every ``;``-separated clause starts its own line, and long
```a`, `b`, `c``` enumerations break before each item.
Code fences, tables, headings and HTML are left alone.
The rewrite is whitespace-only: it refuses to write a file whose word sequence
would change, and running it twice is a no-op.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "etc.", "cf.", "approx.", "resp.")
LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+\.)\s+")
ENUM_LIMIT = 200  # only split ``, `item`` enumerations on lines longer than this
UNTOUCHED = ("#", "|", "<", "---")


def _skip_code_span(text: str, i: int) -> int:
    """Return the index just past the code span opening at ``text[i]``."""
    j = i
    while j < len(text) and text[j] == "`":
        j += 1
    close = text.find(text[i:j], j)
    return close + (j - i) if close != -1 else j


def split_sentences(text: str) -> list[str]:
    """Split after ``.``/``;``/``?``/``!`` followed by a space, outside code spans."""
    parts, start, i = [], 0, 0
    while i < len(text):
        c = text[i]
        if c == "`":
            i = _skip_code_span(text, i)
            continue
        if c in ".;?!" and text[i + 1 : i + 2] == " " and text[i + 1 :].strip():
            head = text[start : i + 1]
            word = head.split()[-1] if head.strip() else ""
            abbreviation = word.lower().endswith(ABBREVIATIONS)
            # "1. " or a lone initial like "A. "
            ordinal = c == "." and len(word) <= 2
            if not (abbreviation or ordinal):
                parts.append(head.strip())
                start = i + 1
        i += 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def split_enumeration(text: str) -> list[str]:
    """Break a long line before each top-level ``, `item`` entry."""
    if len(text) <= ENUM_LIMIT:
        return [text]
    parts, start, depth, i = [], 0, 0, 0
    while i < len(text):
        c = text[i]
        if c == "`":
            i = _skip_code_span(text, i)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0 and text.startswith(" `", i + 1):
            parts.append(text[start : i + 1].strip())
            start = i + 1
        i += 1
    parts.append(text[start:].strip())
    return parts


def _emit(prefix: str, indent: str, text: str) -> list[str]:
    parts = [e for s in split_sentences(text) for e in split_enumeration(s)]
    return [prefix + parts[0]] + [indent + p for p in parts[1:]]


def _is_continuation(line: str, also_stop: tuple[str, ...] = ()) -> bool:
    s = line.lstrip()
    return bool(s) and not LIST_ITEM.match(line) and not s.startswith(("```", "|", "#") + also_stop)


def reflow(src: str) -> str:
    lines = src.split("\n")
    out: list[str] = []
    i, in_fence = 0, False
    while i < len(lines):
        line = lines[i]
        s = line.lstrip()
        if s.startswith("```"):
            in_fence = not in_fence
        if in_fence or s.startswith("```") or not s or s.startswith(UNTOUCHED):
            out.append(line)
            i += 1
        elif s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip())
                i += 1
            out += _emit("> ", "> ", " ".join(buf))
        elif m := LIST_ITEM.match(line):
            prefix = m.group(0)
            buf = [line[len(prefix) :].strip()]
            i += 1
            while i < len(lines) and _is_continuation(lines[i]):
                buf.append(lines[i].strip())
                i += 1
            out += _emit(prefix, " " * len(prefix), " ".join(buf))
        else:
            lead = line[: len(line) - len(s)]
            buf = []
            while i < len(lines) and _is_continuation(lines[i], also_stop=(">",)):
                buf.append(lines[i].strip())
                i += 1
            out += _emit(lead, lead, " ".join(buf))
    return "\n".join(out)


def _words(text: str) -> list[str]:
    return [w for w in text.split() if w != ">"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path, help="default: AGENTS.md and CLAUDE.md")
    parser.add_argument("--check", action="store_true", help="report files that need reflowing; don't write")
    args = parser.parse_args(argv)

    files = args.files or [ROOT / "AGENTS.md", ROOT / "CLAUDE.md"]
    dirty = []
    for path in files:
        old = path.read_text()
        new = reflow(old)
        if _words(old) != _words(new):
            print(f"{path}: reflow would change words, not just line breaks — skipped", file=sys.stderr)
            return 2
        if new != old:
            dirty.append(path)
            if not args.check:
                path.write_text(new)
    for path in dirty:
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"{'needs reflow' if args.check else 'reflowed'}: {shown}")
    return 1 if args.check and dirty else 0


if __name__ == "__main__":
    sys.exit(main())
