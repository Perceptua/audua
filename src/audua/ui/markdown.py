"""A small markdown renderer for the documents audua writes.

Not a general markdown implementation, and deliberately not a dependency. It
covers what ``summary.md`` and ``transcript_digest.md`` actually contain --
headings, emphasis, code spans, links, lists, quotes, rules, pipe tables, and
the numbered footnotes the summary format is built on -- and it does one thing
a general renderer would not: a relative link to a clip becomes an *in-app*
action rather than a navigation, so clicking `[audio]` in a citation plays the
clip and clicking `[transcript]` opens it beside the prose.

The footnote numbers are clip indexes, not counters, so the reference list is
an ``<ol>`` of explicitly-valued items -- ``[^14]`` renders as 14 whether or
not it is the fourteenth citation on the page.
"""

from __future__ import annotations

import re

from ..config import AUDIO_EXTENSIONS

# Documents that open in the reading pane rather than downloading.
DOC_EXTENSIONS = frozenset({".md", ".txt", ".json"})

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBER_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*```+\s*(\w*)\s*$")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

_FOOTNOTE_DEF_RE = re.compile(r"^\[\^(\d+)\]:\s?(.*)$")
_FOOTNOTE_REF_RE = re.compile(r"\[\^(\d+)\](?!:)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+)\s*\)")
_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<![\w*`])[*_](?!\s)(.+?)(?<!\s)[*_](?![\w*`])")

# Marks spans that must not be reprocessed: code, links, footnote markers.
_SHELF_RE = re.compile("\x00(\\d+)\x00")

_NEWLINE = "\n"


def escape(text: str) -> str:
    """HTML-escape. Applied before anything else, so markup can only come from us."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --------------------------------------------------------------------------
# inline
# --------------------------------------------------------------------------

def _link_html(label: str, target: str) -> str:
    """Render one link, turning clip references into in-app actions.

    A relative link is a file in the same run folder -- that is the whole point
    of the citation format. Audio opens in the player, a transcript opens in
    the reading pane, and anything absolute stays an ordinary outbound link.
    """
    if target.startswith(("http://", "https://", "mailto:")):
        return f'<a href="{escape(target)}" rel="noopener noreferrer" target="_blank">{label}</a>'
    if target.startswith("#"):
        return f'<a href="{escape(target)}">{label}</a>'

    name = target.split("#")[0].split("?")[0]
    suffix = ("." + name.rsplit(".", 1)[-1]).lower() if "." in name else ""

    if suffix in AUDIO_EXTENSIONS:
        return f'<a href="#" class="ln-audio" data-audio="{escape(name)}">{label}</a>'
    if suffix in DOC_EXTENSIONS:
        return f'<a href="#" class="ln-doc" data-doc="{escape(name)}">{label}</a>'
    return f'<span class="ln-dead" title="{escape(name)}">{label}</span>'


def _footnote_ref(number: str) -> str:
    return (
        f'<sup class="fn-ref" id="fnref-{number}">'
        f'<a href="#fn-{number}" data-fn="{number}">{number}</a></sup>'
    )


def inline(text: str) -> str:
    """Render inline markup for one line of raw source text."""
    shelf: list[str] = []

    def stash(html: str) -> str:
        shelf.append(html)
        return f"\x00{len(shelf) - 1}\x00"

    text = escape(text)
    # Stashed in this order so emphasis cannot reach inside a code span, a URL,
    # or a footnote marker.
    text = _CODE_RE.sub(lambda m: stash(f"<code>{m.group(1)}</code>"), text)
    text = _LINK_RE.sub(lambda m: stash(_link_html(m.group(1), m.group(2))), text)
    text = _FOOTNOTE_REF_RE.sub(lambda m: stash(_footnote_ref(m.group(1))), text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return _SHELF_RE.sub(lambda m: shelf[int(m.group(1))], text)


def strip_inline(text: str) -> str:
    """Plain text: markup removed, citations dropped, whitespace collapsed.

    Used for the one-line summaries in the output list, where the point is to
    read the sentence rather than render it.
    """
    text = _FOOTNOTE_REF_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Heading anchor, so the reading pane can grow a contents list."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", strip_inline(text).lower()).strip("-")
    return cleaned or "section"


def _starts_block(line: str) -> bool:
    """True when a line opens a block, so a paragraph must stop before it."""
    return bool(
        _HEADING_RE.match(line)
        or _RULE_RE.match(line)
        or _QUOTE_RE.match(line)
        or _BULLET_RE.match(line)
        or _NUMBER_RE.match(line)
        or _FENCE_RE.match(line)
    )


def _table(rows: list[str]) -> str:
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    out = ["<table><thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in cells(rows[0])]
    out.append("</tr></thead><tbody>")
    for line in rows[2:]:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(line)) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render(text: str) -> str:
    """Render a markdown document to an HTML fragment."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Footnote definitions are pulled out first and rendered last, wherever
    # they happened to sit in the source.
    body: list[str] = []
    notes: list[tuple[int, str]] = []
    for line in lines:
        match = _FOOTNOTE_DEF_RE.match(line)
        if match:
            notes.append((int(match.group(1)), match.group(2)))
        elif notes and line.startswith(("    ", "\t")) and line.strip():
            number, existing = notes[-1]
            notes[-1] = (number, f"{existing} {line.strip()}")
        else:
            body.append(line)

    out: list[str] = []
    index, total = 0, len(body)

    while index < total:
        line = body[index]

        if not line.strip():
            index += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            index += 1
            block = []
            while index < total and not _FENCE_RE.match(body[index]):
                block.append(body[index])
                index += 1
            index += 1  # the closing fence, or the end of the document
            language = f' class="lang-{fence.group(1)}"' if fence.group(1) else ""
            out.append(f"<pre><code{language}>{escape(_NEWLINE.join(block))}</code></pre>")
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            level, content = len(heading.group(1)), heading.group(2)
            out.append(
                f'<h{level} id="{_slug(content)}" class="doc-h{level}">'
                f"{inline(content)}</h{level}>"
            )
            index += 1
            continue

        if _RULE_RE.match(line):
            out.append("<hr>")
            index += 1
            continue

        quote = _QUOTE_RE.match(line)
        if quote:
            block = []
            while index < total and (match := _QUOTE_RE.match(body[index])):
                block.append(match.group(1))
                index += 1
            out.append(f"<blockquote>{render(_NEWLINE.join(block))}</blockquote>")
            continue

        if "|" in line and index + 1 < total and _TABLE_RULE_RE.match(body[index + 1]):
            block = [line, body[index + 1]]
            index += 2
            while index < total and "|" in body[index] and body[index].strip():
                block.append(body[index])
                index += 1
            out.append(_table(block))
            continue

        if _BULLET_RE.match(line) or _NUMBER_RE.match(line):
            ordered = _NUMBER_RE.match(line) is not None
            pattern = _NUMBER_RE if ordered else _BULLET_RE
            group = 2 if ordered else 1
            items: list[str] = []
            while index < total:
                match = pattern.match(body[index])
                if match:
                    items.append(match.group(group))
                elif items and body[index].startswith(("  ", "\t")) and body[index].strip():
                    # A wrapped continuation line belongs to the item above it.
                    items[-1] += " " + body[index].strip()
                else:
                    break
                index += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(i)}</li>" for i in items) + f"</{tag}>")
            continue

        paragraph = []
        while index < total and body[index].strip() and not _starts_block(body[index]):
            paragraph.append(body[index].strip())
            index += 1
        out.append(f"<p>{inline(' '.join(paragraph))}</p>")

    if notes:
        out.append('<section class="footnotes"><h2 class="doc-h2">References</h2><ol>')
        for number, content in sorted(notes):
            out.append(
                f'<li id="fn-{number}" value="{number}">{inline(content)} '
                f'<a class="fn-back" href="#fnref-{number}" '
                f'title="back to the citation">&#8617;</a></li>'
            )
        out.append("</ol></section>")

    return "\n".join(out)
