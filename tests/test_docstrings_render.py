"""Guard against docstring markup that mkdocstrings cannot render.

mkdocstrings renders Google-style docstrings as Markdown. Two classes of markup
silently break in the rendered API docs (no build warning is emitted):

1. **reStructuredText markup** — Sphinx directives (``.. math::``, ``.. warning::``,
   ``.. code-block::``), cross-reference roles (``:class:`X```, ``:func:`X```), and
   ``::`` literal-block markers all render *literally* instead of as math / admonitions
   / code, because the Markdown renderer does not understand RST.
2. **Run-on examples** — griffe only recognises the **plural** ``Examples:`` section
   (not singular ``Example:``), and inside it only ``>>>`` doctest lines or fenced
   ```` ``` ```` blocks render as code. A singular ``Example:`` becomes a generic
   admonition, and a plain (un-fenced, non-doctest) body collapses into one run-on
   paragraph.

This test scans every source file so a regression is caught at test time rather than
discovered by eye on the published site. See PR discussion in the docs overhaul.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent.parent / "src" / "gauss_flows"
PY_FILES = sorted(SRC.rglob("*.py"))

# RST artefacts that render literally under mkdocstrings/Markdown.
RST_DIRECTIVE = re.compile(r"\.\. [A-Za-z-]+::")
RST_ROLE = re.compile(r":(?:class|func|meth|mod|obj|ref|attr|data|exc|term):`")
RST_LITERAL = re.compile(r"::\s*$")
SINGULAR_EXAMPLE = re.compile(r"^\s*Example:\s*$")
EXAMPLES_HEADER = re.compile(r"^(\s*)Examples:\s*$")
FENCE = re.compile(r"^\s*```")


def _rel(path: Path) -> str:
    return str(path.relative_to(SRC.parent.parent))


def _code_region_mask(lines: list[str]) -> list[bool]:
    """Mark lines that live inside a fenced block or a ``>>>`` doctest block.

    RST-artefact checks must skip these — a code example may legitimately contain
    ``::`` (e.g. ``std::``) or directive-like text without it being broken markup.
    """
    mask = [False] * len(lines)
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if FENCE.match(line):
            in_fence = not in_fence
            mask[i] = True
            continue
        if in_fence or stripped.startswith(">>>") or stripped.startswith("..."):
            mask[i] = True
    return mask


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_no_rst_markup(path: Path) -> None:
    """Docstrings must use Markdown + ``$...$`` math, not reStructuredText."""
    lines = path.read_text().split("\n")
    in_code = _code_region_mask(lines)
    bad: list[str] = []
    for i, line in enumerate(lines, start=1):
        if in_code[i - 1] or "http" in line:
            continue
        if RST_DIRECTIVE.search(line):
            bad.append(f"  {_rel(path)}:{i}: RST directive → {line.strip()[:60]}")
        if RST_ROLE.search(line):
            bad.append(f"  {_rel(path)}:{i}: RST role → {line.strip()[:60]}")
        if RST_LITERAL.search(line):
            bad.append(
                f"  {_rel(path)}:{i}: RST '::' literal block → {line.strip()[:60]}"
            )
    assert not bad, (
        "reStructuredText markup does not render in mkdocstrings:\n" + "\n".join(bad)
    )


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_examples_section_renders(path: Path) -> None:
    """Example sections must be plural ``Examples:`` with a code body.

    griffe only recognises ``Examples:`` (plural), and only renders ``>>>`` doctest
    lines or fenced code as code; anything else collapses into a run-on paragraph.
    """
    lines = path.read_text().split("\n")
    bad: list[str] = []
    for i, line in enumerate(lines):
        if SINGULAR_EXAMPLE.match(line):
            bad.append(
                f"  {_rel(path)}:{i + 1}: singular 'Example:' is not recognised — "
                "use plural 'Examples:'"
            )
        m = EXAMPLES_HEADER.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        body: list[str] = []
        for nxt in lines[i + 1 :]:
            if not nxt.strip():
                body.append(nxt)
                continue
            if len(nxt) - len(nxt.lstrip()) <= indent:
                break
            body.append(nxt)
        body_text = "\n".join(body)
        if ">>>" not in body_text and "```" not in body_text:
            bad.append(
                f"  {_rel(path)}:{i + 1}: 'Examples:' body has no '>>>' or fenced "
                "block — it will render as a run-on paragraph"
            )
    assert not bad, "Example sections will not render as code:\n" + "\n".join(bad)
