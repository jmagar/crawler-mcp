"""
Utilities to parse and apply GitHub-style suggestion blocks from review comments.

Assumptions:
- Suggestion blocks are fenced like ```suggestion\n...\n```.
- We apply suggestions to the file/path and line range implied by metadata
  from our PR items (start_line..line) when available; otherwise to a single line.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Suggestion:
    content: str


def extract_suggestions(markdown: str) -> list[Suggestion]:
    """Extract GitHub suggestion blocks from markdown text."""
    suggestions: list[Suggestion] = []
    if not markdown:
        return suggestions
    try:
        lines = markdown.splitlines()
        in_block = False
        buf: list[str] = []
        for ln in lines:
            if not in_block:
                if ln.strip().lower().startswith("```suggestion"):
                    in_block = True
                    buf = []
                continue
            # inside block
            if ln.strip().startswith("```"):
                # end block
                suggestions.append(Suggestion("\n".join(buf)))
                in_block = False
                buf = []
            else:
                buf.append(ln)
    except Exception:
        return suggestions
    return suggestions


def _apply_to_lines(
    src_lines: list[str], start: int, end: int, new_text: str
) -> list[str]:
    # start/end are 1-based inclusive; clamp to bounds
    n = len(src_lines)
    s = max(1, int(start))
    e = max(s, int(end))
    if s > n + 1:
        s = n + 1
    if e > n + 1:
        e = n + 1
    # replace lines s..e with new_text splitlines
    prefix = src_lines[: s - 1]
    suffix = src_lines[e:]
    repl = new_text.splitlines()
    return prefix + repl + suffix


def apply_suggestion(
    original_text: str,
    *,
    start_line: int,
    end_line: int | None,
    suggestion_text: str,
) -> str:
    """Return modified text with suggestion applied."""
    lines = original_text.splitlines()
    end = end_line if end_line and end_line >= start_line else start_line
    out = _apply_to_lines(lines, start_line, end, suggestion_text)
    return "\n".join(out) + ("\n" if original_text.endswith("\n") else "")


def unified_diff(path: str, before: str, after: str) -> str:
    import difflib

    a_lines = before.splitlines(keepends=True)
    b_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        a_lines, b_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
    )
    return "".join(diff)
