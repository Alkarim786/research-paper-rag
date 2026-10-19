"""Runtime settings, read from environment variables (prefix ``PAPERRAG_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PAPERRAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")

    # Models (downloaded from the Hugging Face hub on first use).
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # "hashing" / "lexical" give deterministic, download-free models for tests.
    embedder: str = "sentence-transformers"  # or "hashing"
    reranker: str = "cross-encoder"  # or "lexical" / "none"

    # Chunking
    chunk_max_words: int = 180
    chunk_overlap_words: int = 30

    # Retrieval
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    candidates_per_retriever: int = 50
    rerank_top_n: int = 20
    rrf_k: int = 60
    top_k: int = 5

    # Answering
    generator: str = "extractive"  # or "anthropic"
    anthropic_model: str = "claude-sonnet-5-5"
    max_answer_sentences: int = 3
    support_threshold: float = 0.5

    # Ingestion
    arxiv_delay_s: float = 3.0
    user_agent: str = "PaperRAG/0.1 (research project; github.com/Alkarim786/research-paper-rag)"


@lru_cache
def get_settings() -> Settings:
    return Settings()
