"""Polite arXiv client: search via the Atom API, download PDFs, parse them.

arXiv asks API users to wait ~3 seconds between requests and to identify
themselves; both are honoured here.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..types import Document
from .pdf import pdf_to_document

API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str


def parse_feed(xml_text: str) -> list[ArxivEntry]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", _NS):
        raw_id = (e.findtext("a:id", default="", namespaces=_NS) or "").rsplit("/abs/", 1)[-1]
        pdf = next(
            (ln.get("href") for ln in e.findall("a:link", _NS) if ln.get("title") == "pdf"),
            f"https://arxiv.org/pdf/{raw_id}",
        )
        out.append(
            ArxivEntry(
                arxiv_id=raw_id,
                title=" ".join((e.findtext("a:title", default="", namespaces=_NS) or "").split()),
                abstract=" ".join((e.findtext("a:summary", default="", namespaces=_NS) or "").split()),
                pdf_url=pdf,
            )
        )
    return out


class ArxivClient:
    def __init__(self, user_agent: str, delay_s: float = 3.0, client: httpx.Client | None = None) -> None:
        self.delay_s = delay_s
        self.http = client or httpx.Client(headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True)
        self._last = 0.0

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.delay_s:
            time.sleep(self.delay_s - gap)
        self._last = time.monotonic()

    def search(self, query: str, max_results: int = 10) -> list[ArxivEntry]:
        self._wait()
        r = self.http.get(API, params={"search_query": query, "max_results": max_results, "sortBy": "relevance"})
        r.raise_for_status()
        return parse_feed(r.text)

    def fetch(self, entry: ArxivEntry, pdf_dir: Path) -> Document:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        path = pdf_dir / f"{entry.arxiv_id.replace('/', '_')}.pdf"
        if not path.exists():
            self._wait()
            r = self.http.get(entry.pdf_url)
            r.raise_for_status()
            path.write_bytes(r.content)
        doc = pdf_to_document(path, paper_id=f"arxiv:{entry.arxiv_id}", title=entry.title)
        # The API abstract is cleaner than one parsed from the PDF.
        doc.abstract = entry.abstract
        doc.source = f"https://arxiv.org/abs/{entry.arxiv_id}"
        return doc
