"""High-level RAG pipeline: build/load an index, search, answer."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from .answer import AnswerSentence, AnthropicGenerator, ExtractiveGenerator, Generator, answer_question
from .chunking import chunk_document
from .config import Settings
from .models import Embedder, Reranker, make_embedder, make_reranker
from .retriever import HybridRetriever
from .types import Answer, Chunk, Document, Hit


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, reranker: Reranker, generator: Generator) -> None:
        self.s = settings
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator
        self.retriever: HybridRetriever | None = None

    @classmethod
    def from_settings(cls, s: Settings) -> RAGPipeline:
        embedder = make_embedder(s.embedder, s.embed_model)
        reranker = make_reranker(s.reranker, s.rerank_model)
        generator: Generator
        if s.generator == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            generator = AnthropicGenerator(s.anthropic_model)
        else:
            generator = ExtractiveGenerator(reranker, s.max_answer_sentences)
        return cls(s, embedder, reranker, generator)

    # Index lifecycle ---------------------------------------------------------
    def chunk(self, docs: Iterable[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for d in docs:
            out.extend(chunk_document(d, self.s.chunk_max_words, self.s.chunk_overlap_words))
        return out

    def build(self, docs: Iterable[Document]) -> int:
        chunks = self.chunk(docs)
        self.retriever = HybridRetriever.build(
            chunks,
            self.embedder,
            self.reranker,
            k1=self.s.bm25_k1,
            b=self.s.bm25_b,
            candidates=self.s.candidates_per_retriever,
            rerank_top_n=self.s.rerank_top_n,
            rrf_k=self.s.rrf_k,
        )
        return len(chunks)

    def add(self, docs: Iterable[Document]) -> int:
        chunks = self.chunk(docs)
        if self.retriever is None:
            return self.build_from_chunks(chunks)
        existing = {c.paper_id for c in self.retriever.chunks}
        chunks = [c for c in chunks if c.paper_id not in existing]
        self.retriever.add(chunks)
        return len(chunks)

    def build_from_chunks(self, chunks: list[Chunk]) -> int:
        self.retriever = HybridRetriever.build(
            chunks,
            self.embedder,
            self.reranker,
            k1=self.s.bm25_k1,
            b=self.s.bm25_b,
            candidates=self.s.candidates_per_retriever,
            rerank_top_n=self.s.rerank_top_n,
            rrf_k=self.s.rrf_k,
        )
        return len(chunks)

    def save(self, directory: Path | None = None) -> None:
        self._require().save(directory or self.s.index_dir)

    def load(self, directory: Path | None = None) -> bool:
        d = directory or self.s.index_dir
        if not (d / "meta.json").exists():
            return False
        self.retriever = HybridRetriever.load(
            d,
            self.embedder,
            self.reranker,
            candidates=self.s.candidates_per_retriever,
            rerank_top_n=self.s.rerank_top_n,
            rrf_k=self.s.rrf_k,
        )
        return True

    # Query -------------------------------------------------------------------
    def search(
        self, query: str, k: int | None = None, mode: str = "hybrid_rerank", paper_ids: Sequence[str] | None = None
    ) -> tuple[list[Hit], dict[str, float]]:
        return self._require().search(query, k or self.s.top_k, mode, paper_ids)

    def ask(self, question: str, k: int | None = None, paper_ids: Sequence[str] | None = None) -> Answer:
        t0 = time.perf_counter()
        hits, timings = self.search(question, k, "hybrid_rerank", paper_ids)
        t1 = time.perf_counter()
        if not hits:
            ans = Answer(
                question, [AnswerSentence("No relevant passages were found.", [])], [], 0.0, self.generator.name
            )
        else:
            ans = answer_question(question, hits, self.generator, self.s.support_threshold)
        timings["retrieval_ms"] = (t1 - t0) * 1e3
        timings["generation_ms"] = (time.perf_counter() - t1) * 1e3
        ans.timings_ms = {k: round(v, 2) for k, v in timings.items()}
        return ans

    def stats(self) -> dict:
        r = self.retriever
        if r is None:
            return {"papers": 0, "chunks": 0}
        return {
            "papers": len(r.paper_ids()),
            "chunks": len(r.chunks),
            "embedder": self.embedder.name,
            "reranker": self.reranker.name,
            "generator": self.generator.name,
            "vocabulary": len(r.bm25.postings),
        }

    def _require(self) -> HybridRetriever:
        if self.retriever is None:
            raise RuntimeError("index is empty; ingest documents first")
        return self.retriever
