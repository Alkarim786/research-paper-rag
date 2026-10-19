"""Paragraph-aware chunking.

Paragraphs are the natural unit of meaning in papers, so chunks never cross a
section boundary. Short paragraphs in the same section are merged up to
``max_words``; long paragraphs are split into overlapping word windows on
sentence boundaries. Each chunk records which paragraphs it covers, which is
what lets the evaluation map gold evidence paragraphs to chunks exactly.
"""

from __future__ import annotations

from .text import split_sentences, word_count
from .types import Chunk, Document

MIN_MERGE_WORDS = 40


def chunk_document(doc: Document, max_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    if max_words <= overlap_words:
        raise ValueError("max_words must be greater than overlap_words")
    chunks: list[Chunk] = []

    def emit(section: str, text: str, pids: tuple[int, ...]) -> None:
        chunks.append(
            Chunk(
                chunk_id=f"{doc.paper_id}#{len(chunks)}",
                paper_id=doc.paper_id,
                title=doc.title,
                section=section,
                text=text,
                paragraph_ids=pids,
            )
        )

    if doc.abstract.strip():
        emit("Abstract", doc.abstract.strip(), (-1,))

    buf: list[str] = []
    buf_ids: list[int] = []
    buf_section = ""

    def flush() -> None:
        nonlocal buf, buf_ids
        if buf:
            emit(buf_section, " ".join(buf), tuple(buf_ids))
        buf, buf_ids = [], []

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        n = word_count(text)
        if para.section != buf_section:
            flush()
            buf_section = para.section
        if n > max_words:
            flush()
            for window in _split_long(text, max_words, overlap_words):
                emit(para.section, window, (i,))
            continue
        if buf and (sum(map(word_count, buf)) + n > max_words or n >= MIN_MERGE_WORDS):
            flush()
        buf.append(text)
        buf_ids.append(i)
        if n >= MIN_MERGE_WORDS:
            flush()
    flush()
    return chunks


def _split_long(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Greedy sentence packing with a word-level overlap between windows."""
    sentences = split_sentences(text) or [text]
    windows: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sentences:
        sw = word_count(s)
        if cur and cur_words + sw > max_words:
            windows.append(" ".join(cur))
            # Carry the tail of the previous window forward as overlap.
            tail: list[str] = []
            tail_words = 0
            for prev in reversed(cur):
                if tail_words + word_count(prev) > overlap_words:
                    break
                tail.insert(0, prev)
                tail_words += word_count(prev)
            cur, cur_words = tail, tail_words
        if sw > max_words:  # a single giant "sentence": hard-split by words
            words = s.split()
            step = max_words - overlap_words
            for j in range(0, len(words), step):
                windows.append(" ".join(words[j : j + max_words]))
            cur, cur_words = [], 0
            continue
        cur.append(s)
        cur_words += sw
    if cur:
        windows.append(" ".join(cur))
    return windows
