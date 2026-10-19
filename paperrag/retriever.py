"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion, then
cross-encoder reranking."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .models import Embedder, Reranker
from .types import Chunk, Hit


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """RRF(d) = sum_r 1 / (k + rank_r(d)), ranks 1-based. Robust to score
    scales that differ wildly (BM25 is unbounded, cosine is in [-1, 1])."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class DenseIndex:
    """Exact cosine search over a normalised embedding matrix. Exact search is
    the right call up to ~10^5-10^6 chunks on CPU; past that, swap in FAISS/HNSW
    behind the same interface."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)

    def search(self, qvec: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        sims = self.vectors @ qvec
        if allowed is not None:
            masked = np.full_like(sims, -np.inf)
            masked[allowed] = sims[allowed]
            sims = masked
        k = min(k, int(np.isfinite(sims).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(i), float(sims[i])) for i in top]


class HybridRetriever:
    MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index,
        dense: DenseIndex,
        embedder: Embedder,
        reranker: Reranker,
        *,
        candidates: int = 50,
        rerank_top_n: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_top_n = rerank_top_n
        self.rrf_k = rrf_k
        self._by_paper: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            self._by_paper[c.paper_id].append(i)

    @classmethod
    def build(
        cls, chunks: list[Chunk], embedder: Embedder, reranker: Reranker, *, k1=1.5, b=0.75, **kw
    ) -> HybridRetriever:
        texts = [_index_text(c) for c in chunks]
        bm25 = BM25Index.build(texts, k1=k1, b=b)
        vectors = embedder.encode(texts) if texts else np.zeros((0, embedder.dim), np.float32)
        return cls(chunks, bm25, DenseIndex(vectors), embedder, reranker, **kw)

    def paper_ids(self) -> list[str]:
        return list(self._by_paper)

    def search(
        self, query: str, k: int = 5, mode: str = "hybrid_rerank", paper_ids: Sequence[str] | None = None
    ) -> tuple[list[Hit], dict[str, float]]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        timings: dict[str, float] = {}
        allowed_list: list[int] | None = None
        if paper_ids:
            allowed_list = [i for p in paper_ids for i in self._by_paper.get(p, [])]
            if not allowed_list:
                return [], timings
        allowed_set = set(allowed_list) if allowed_list is not None else None
        allowed_arr = np.asarray(allowed_list, dtype=np.int64) if allowed_list is not None else None
        n = self.candidates if mode in ("hybrid", "hybrid_rerank") else max(k, self.rerank_top_n)
        stages: dict[int, dict[str, float]] = defaultdict(dict)

        bm25_hits: list[tuple[int, float]] = []
        dense_hits: list[tuple[int, float]] = []
        if mode != "dense":
            t0 = time.perf_counter()
            bm25_hits = self.bm25.search(query, n, allowed_set)
            timings["bm25_ms"] = (time.perf_counter() - t0) * 1e3
            for r, (i, s) in enumerate(bm25_hits, 1):
                stages[i].update(bm25=s, bm25_rank=r)
        if mode != "bm25":
            t0 = time.perf_counter()
            qvec = self.embedder.encode([query])[0]
            dense_hits = self.dense.search(qvec, n, allowed_arr)
            timings["dense_ms"] = (time.perf_counter() - t0) * 1e3
            for r, (i, s) in enumerate(dense_hits, 1):
                stages[i].update(dense=s, dense_rank=r)

        if mode == "bm25":
            ranked = bm25_hits
        elif mode == "dense":
            ranked = dense_hits
        else:
            t0 = time.perf_counter()
            ranked = reciprocal_rank_fusion([[i for i, _ in bm25_hits], [i for i, _ in dense_hits]], self.rrf_k)
            timings["fusion_ms"] = (time.perf_counter() - t0) * 1e3
            for r, (i, s) in enumerate(ranked, 1):
                stages[i].update(rrf=s, rrf_rank=r)

        if mode == "hybrid_rerank" and ranked:
            t0 = time.perf_counter()
            pool = [i for i, _ in ranked[: self.rerank_top_n]]
            scores = self.reranker.score(query, [self.chunks[i].text for i in pool])
            order = np.argsort(-scores, kind="stable")
            ranked = [(pool[j], float(scores[j])) for j in order]
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1e3
            for i, s in ranked:
                stages[i]["rerank"] = s

        hits = [Hit(self.chunks[i], s, dict(stages[i])) for i, s in ranked[:k]]
        return hits, timings

    # Persistence -----------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        self.bm25.save(directory / "bm25.json")
        np.save(directory / "embeddings.npy", self.dense.vectors)
        (directory / "meta.json").write_text(
            json.dumps({"embedder": self.embedder.name, "n_chunks": len(self.chunks)}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, reranker: Reranker, **kw) -> HybridRetriever:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(f"index was built with {meta['embedder']}, not {embedder.name}; rebuild it")
        chunks = []
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                d["paragraph_ids"] = tuple(d["paragraph_ids"])
                chunks.append(Chunk(**d))
        bm25 = BM25Index.load(directory / "bm25.json")
        vectors = np.load(directory / "embeddings.npy")
        return cls(chunks, bm25, DenseIndex(vectors), embedder, reranker, **kw)

    def add(self, new_chunks: list[Chunk]) -> None:
        """Append chunks (e.g. a freshly ingested paper). BM25 IDF is recomputed."""
        if not new_chunks:
            return
        texts = [_index_text(c) for c in new_chunks]
        start = len(self.chunks)
        self.chunks.extend(new_chunks)
        self.bm25.add(texts)
        self.dense.vectors = np.vstack([self.dense.vectors, self.embedder.encode(texts)])
        for i, c in enumerate(new_chunks, start):
            self._by_paper[c.paper_id].append(i)


def _index_text(c: Chunk) -> str:
    # Section titles carry strong signal ("Experimental Setup", "Datasets").
    return f"{c.section}. {c.text}" if c.section else c.text
