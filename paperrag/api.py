"""FastAPI service.

Run:  uvicorn paperrag.api:app --port 8000
"""

from __future__ import annotations

import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .ingest.arxiv import ArxivClient
from .ingest.pdf import pdf_to_document
from .pipeline import RAGPipeline

MAX_PDF_BYTES = 30 * 1024 * 1024


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    k: int = Field(5, ge=1, le=50)
    mode: str = Field("hybrid_rerank", pattern="^(bm25|dense|hybrid|hybrid_rerank)$")
    paper_ids: list[str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(5, ge=1, le=20)
    paper_ids: list[str] | None = None


class ArxivRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300, examples=["ti:retrieval augmented generation"])
    max_results: int = Field(5, ge=1, le=25)


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    s = settings or get_settings()
    lock = threading.Lock()  # index writes are rare; reads don't take it

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            app.state.pipeline = RAGPipeline.from_settings(s)
            app.state.pipeline.load()
        yield

    app = FastAPI(title="PaperRAG", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline

    def pipe() -> RAGPipeline:
        p: RAGPipeline = app.state.pipeline
        if p.retriever is None:
            raise HTTPException(409, "index is empty; ingest papers first (POST /ingest/arxiv or /ingest/pdf)")
        return p

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict:
        return app.state.pipeline.stats()

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        hits, timings = pipe().search(req.query, req.k, req.mode, req.paper_ids)
        return {
            "query": req.query,
            "mode": req.mode,
            "hits": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "paper_id": h.chunk.paper_id,
                    "title": h.chunk.title,
                    "section": h.chunk.section,
                    "score": round(h.score, 5),
                    "stages": {k: round(v, 5) for k, v in h.stages.items()},
                    "text": h.chunk.text,
                }
                for h in hits
            ],
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        ans = pipe().ask(req.question, req.k, req.paper_ids)
        return {
            "question": ans.question,
            "answer": ans.text,
            "sentences": [asdict(x) for x in ans.sentences],
            "citations": [asdict(c) for c in ans.citations],
            "groundedness": ans.groundedness,
            "generator": ans.generator,
            "timings_ms": ans.timings_ms,
        }

    @app.post("/ingest/arxiv")
    def ingest_arxiv(req: ArxivRequest) -> dict:
        client = ArxivClient(s.user_agent, s.arxiv_delay_s)
        entries = client.search(req.query, req.max_results)
        docs, failed = [], []
        for e in entries:
            try:
                docs.append(client.fetch(e, s.data_dir / "pdfs"))
            except Exception as exc:  # one bad PDF shouldn't fail the batch
                failed.append({"arxiv_id": e.arxiv_id, "error": str(exc)[:200]})
        with lock:
            added = app.state.pipeline.add(docs)
            app.state.pipeline.save()
        return {"papers": [d.paper_id for d in docs], "chunks_added": added, "failed": failed}

    @app.post("/ingest/pdf")
    async def ingest_pdf(file: UploadFile = File(...)) -> dict:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(415, "only .pdf files are accepted")
        data = await file.read()
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(413, "PDF too large")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.pdf"
            path.write_bytes(data)
            try:
                doc = pdf_to_document(path, paper_id=f"upload:{Path(file.filename).stem}")
            except Exception as exc:
                raise HTTPException(422, f"could not parse PDF: {exc}") from exc
        with lock:
            added = app.state.pipeline.add([doc])
            app.state.pipeline.save()
        return {"paper_id": doc.paper_id, "title": doc.title, "chunks_added": added}

    return app


app = create_app()
