"""Load QASPER (parquet export on the HF hub) into Documents + eval questions.

Each question has one or more annotator answers. An answer is extractive
spans, free-form text, yes/no, or "unanswerable", plus the evidence paragraphs
the annotator used. We keep answerable questions with non-figure/table
evidence, because those can be scored against retrieved text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..types import Document, Paragraph


@dataclass
class QAItem:
    question_id: str
    paper_id: str
    question: str
    answers: list[str]  # gold answer strings (one per annotator, non-empty)
    evidence: list[set[str]] = field(default_factory=list)  # per annotator: evidence paragraph texts


def _norm(t: str) -> str:
    return " ".join(t.split())


def _answer_text(a: dict) -> str | None:
    if a.get("unanswerable"):
        return None
    if a.get("extractive_spans"):
        return ", ".join(a["extractive_spans"])
    if a.get("free_form_answer"):
        return a["free_form_answer"]
    if a.get("yes_no") is not None:
        return "Yes" if a["yes_no"] else "No"
    return None


def load_qasper(path: Path, max_papers: int | None = None) -> tuple[list[Document], list[QAItem]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if max_papers:
        rows = rows[:max_papers]
    docs: list[Document] = []
    items: list[QAItem] = []
    for r in rows:
        ft = r["full_text"]
        paragraphs = [
            Paragraph(sec or "Body", _norm(p))
            for sec, paras in zip(ft["section_name"], ft["paragraphs"], strict=True)
            for p in paras
            if p and p.strip()
        ]
        docs.append(Document(r["id"], r["title"], _norm(r["abstract"] or ""), paragraphs, f"qasper:{r['id']}"))
        qas = r["qas"]
        for q, qid, ans in zip(qas["question"], qas["question_id"], qas["answers"], strict=True):
            golds, evidence = [], []
            for a in ans["answer"]:
                text = _answer_text(a)
                ev = {
                    _norm(e)
                    for e in (a.get("evidence") or [])
                    if e and not e.startswith(("FLOAT SELECTED", "TABLE", "FIGURE"))
                }
                if text and ev:
                    golds.append(text)
                    evidence.append(ev)
            if golds:
                items.append(QAItem(qid, r["id"], q, golds, evidence))
    return docs, items
