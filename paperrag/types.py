"""Core data types shared across ingestion, retrieval, answering and eval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Paragraph:
    section: str
    text: str


@dataclass
class Document:
    paper_id: str
    title: str
    abstract: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)
    source: str = ""  # e.g. arxiv URL or file path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<paper_id>#<n>"
    paper_id: str
    title: str
    section: str
    text: str
    # Indices into Document.paragraphs that this chunk covers (for gold mapping).
    paragraph_ids: tuple[int, ...]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    # Rank and score from each stage, for debugging and the API.
    stages: dict[str, float] = field(default_factory=dict)


@dataclass
class Citation:
    n: int  # 1-based source number shown in the answer
    chunk_id: str
    paper_id: str
    title: str
    section: str
    quote: str  # the supporting text from the chunk
    support: float  # 0..1, how well the chunk supports the cited sentence
    verified: bool


@dataclass
class AnswerSentence:
    text: str
    citations: list[int]  # citation numbers


@dataclass
class Answer:
    question: str
    sentences: list[AnswerSentence]
    citations: list[Citation]
    groundedness: float  # share of sentences whose citations all verified
    generator: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def plain_text(self) -> str:
        """Answer without citation markers (used for answer-F1 scoring)."""
        return " ".join(s.text for s in self.sentences)

    @property
    def text(self) -> str:
        parts = []
        for s in self.sentences:
            refs = "".join(f"[{n}]" for n in s.citations)
            parts.append(f"{s.text} {refs}".strip())
        return " ".join(parts)
