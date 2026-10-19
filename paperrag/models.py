"""Embedding and reranking models behind small protocols.

Production uses sentence-transformers models. Tests and CI use the
deterministic, download-free ``HashingEmbedder`` and ``LexicalReranker``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (len(texts), dim)."""
        ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Relevance scores (higher is better), one per passage."""
        ...


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        emb = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, device="cpu")

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self._model.predict([(query, p) for p in passages], batch_size=32), dtype=np.float32)


class HashingEmbedder:
    """Feature-hashing bag of words (+ bigrams). Deterministic and fast; good
    enough to exercise the dense path in tests."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _bucket(self, tok: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for tok in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]:
                j, sign = self._bucket(tok)
                out[i, j] += sign
        return _normalise(out)


class LexicalReranker:
    """Query-term coverage weighted by passage-local term frequency."""

    name = "lexical"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        q = set(tokenize(query))
        scores = []
        for p in passages:
            toks = tokenize(p)
            if not q or not toks:
                scores.append(0.0)
                continue
            covered = q.intersection(toks)
            scores.append(len(covered) / len(q) + 0.01 * sum(toks.count(t) for t in covered) / len(toks))
        return np.asarray(scores, dtype=np.float32)


class NoReranker:
    name = "none"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        # Preserve incoming order: strictly decreasing scores.
        return -np.arange(len(passages), dtype=np.float32)


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)


def make_reranker(kind: str, model_name: str) -> Reranker:
    if kind == "lexical":
        return LexicalReranker()
    if kind == "none":
        return NoReranker()
    return CrossEncoderReranker(model_name)
