"""Okapi BM25 over an in-memory inverted index (no external search library).

score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

with the non-negative Lucene IDF  idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)).
k1 controls term-frequency saturation; b controls document-length
normalisation.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        # term -> list of (doc_idx, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self._idf: dict[str, float] = {}

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def avgdl(self) -> float:
        return sum(self.doc_len) / self.n_docs if self.n_docs else 0.0

    def add(self, texts: Iterable[str]) -> None:
        for text in texts:
            idx = len(self.doc_len)
            counts = Counter(tokenize(text))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((idx, tf))
        self._idf = {t: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def idf(self, term: str) -> float:
        return self._idf.get(term, 0.0)

    def scores(self, query: str, allowed: set[int] | None = None) -> dict[int, float]:
        """Sparse scores for every document containing at least one query term."""
        out: dict[int, float] = defaultdict(float)
        avgdl = self.avgdl or 1.0
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc, tf in self.postings[term]:
                if allowed is not None and doc not in allowed:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / avgdl)
                out[doc] += idf * tf * (self.k1 + 1) / norm
        return out

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        s = self.scores(query, allowed)
        return heapq.nlargest(k, s.items(), key=lambda kv: (kv[1], -kv[0]))

    # Persistence -----------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"k1": self.k1, "b": self.b, "doc_len": self.doc_len, "postings": self.postings}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        raw = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(raw["k1"], raw["b"])
        idx.doc_len = raw["doc_len"]
        idx.postings = defaultdict(list, {t: [tuple(p) for p in ps] for t, ps in raw["postings"].items()})
        n = idx.n_docs
        idx._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in idx.postings.items()}
        return idx

    @classmethod
    def build(cls, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        idx = cls(k1, b)
        idx.add(texts)
        return idx
