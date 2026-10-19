"""PDF → Document with sections and paragraphs.

PDF text extraction gives visual lines, not paragraphs. Heuristics used:
* de-hyphenate words broken across lines ("infor-\\nmation");
* a line is a section heading if it looks like "3 Method", "3.2 Training
  Details", "Abstract" or an ALL-CAPS short line;
* a paragraph ends at a short line that ends a sentence (the last line of a
  paragraph is usually shorter than the column width);
* stop at References / Bibliography.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from ..types import Document, Paragraph

_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,:&/()\-]{2,60}$")
_NUMBERED = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z]")
_KNOWN = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "analysis",
    "conclusion",
    "conclusions",
    "evaluation",
    "datasets",
    "dataset",
    "limitations",
    "future work",
}
_STOP = re.compile(r"^(?:\d+\.?\s*)?(references|bibliography|acknowledg(e)?ments?)\s*$", re.IGNORECASE)


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 70 or s.endswith((".", ",", ";")):
        return False
    bare = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", s).lower()
    if bare in _KNOWN:
        return True
    if _NUMBERED.match(s) and _HEADING.match(s) and len(s.split()) <= 8:
        return True
    return s.isupper() and 1 <= len(s.split()) <= 6 and s.replace(" ", "").isalpha()


def text_to_document(text: str, paper_id: str, title: str = "", source: str = "") -> Document:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate
    lines = [ln.rstrip() for ln in text.splitlines()]
    lengths = [len(ln) for ln in lines if len(ln) > 20]
    width = statistics.median(lengths) if lengths else 80

    section = "Body"
    abstract_parts: list[str] = []
    paragraphs: list[Paragraph] = []
    cur: list[str] = []

    def flush() -> None:
        if cur:
            para = re.sub(r"\s+", " ", " ".join(cur)).strip()
            if len(para.split()) >= 5:
                (abstract_parts if section == "Abstract" else paragraphs).append(
                    para if section == "Abstract" else Paragraph(section, para)
                )
            cur.clear()

    for ln in lines:
        s = ln.strip()
        if not s:
            flush()
            continue
        if _STOP.match(s):
            break
        if is_heading(s):
            flush()
            section = (
                re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", s).strip().title()
                if s.isupper()
                else re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", s).strip()
            )
            if section.lower() == "abstract":
                section = "Abstract"
            continue
        cur.append(s)
        if s.endswith((".", "?", "!")) and len(s) < 0.7 * width:
            flush()
    flush()
    if not title:
        first = next((ln.strip() for ln in lines if len(ln.strip()) > 10), paper_id)
        title = first[:200]
    return Document(paper_id, title, " ".join(abstract_parts), paragraphs, source)


def pdf_to_document(path: Path, paper_id: str | None = None, title: str = "") -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    return text_to_document(text, paper_id or path.stem, title or meta_title, str(path))
