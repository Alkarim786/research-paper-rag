import httpx
import pytest
from fastapi.testclient import TestClient

from paperrag.api import create_app
from paperrag.ingest.arxiv import ArxivClient, parse_feed
from paperrag.ingest.pdf import is_heading, pdf_to_document, text_to_document

from .conftest import make_pdf

PAPER_LINES = [
    "Sparse Retrieval Still Matters",
    "Abstract",
    "We revisit lexical retrieval for scientific question answering.",
    "1 Introduction",
    "Neural retrievers dominate leaderboards, yet BM25 remains a strong base-",
    "line on out-of-domain scientific text.",
    "3.2 Training Details",
    "We train for 3 epochs with a learning rate of 2e-5 on 8 GPUs.",
    "References",
    "[1] Robertson and Zaragoza. The probabilistic relevance framework.",
]


def test_heading_detection():
    assert is_heading("1 Introduction") and is_heading("3.2 Training Details") and is_heading("ABSTRACT")
    assert not is_heading("We train for 3 epochs with a learning rate of 2e-5.")
    assert not is_heading("Neural retrievers dominate leaderboards, yet BM25")


def test_text_to_document_sections_hyphenation_and_references():
    doc = text_to_document("\n".join(PAPER_LINES), "x")
    assert doc.abstract.startswith("We revisit lexical retrieval")
    sections = [p.section for p in doc.paragraphs]
    assert sections == ["Introduction", "Training Details"]
    assert "baseline on out-of-domain" in doc.paragraphs[0].text  # de-hyphenated
    assert all("Robertson" not in p.text for p in doc.paragraphs)  # stopped at References


def test_pdf_roundtrip(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf(PAPER_LINES))
    doc = pdf_to_document(path, paper_id="paper")
    assert doc.title == "Sparse Retrieval Still Matters"
    assert any("3 epochs" in p.text for p in doc.paragraphs)


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>Large pre-trained language models store factual knowledge.</summary>
    <link title="pdf" href="http://arxiv.org/pdf/2005.11401v4" rel="related" type="application/pdf"/>
  </entry>
</feed>"""


def test_arxiv_feed_parsing_and_fetch(tmp_path):
    (e,) = parse_feed(ATOM)
    assert e.arxiv_id == "2005.11401v4" and e.title.startswith("Retrieval-Augmented Generation for Knowledge")

    def handler(req: httpx.Request) -> httpx.Response:
        if "export.arxiv.org" in str(req.url):
            return httpx.Response(200, text=ATOM)
        return httpx.Response(200, content=make_pdf(PAPER_LINES))

    client = ArxivClient("test", delay_s=0, client=httpx.Client(transport=httpx.MockTransport(handler)))
    (entry,) = client.search("ti:rag", 1)
    doc = client.fetch(entry, tmp_path)
    assert doc.paper_id == "arxiv:2005.11401v4"
    assert doc.abstract == "Large pre-trained language models store factual knowledge."
    assert doc.source == "https://arxiv.org/abs/2005.11401v4"


@pytest.fixture
def client(settings, pipeline):
    return TestClient(create_app(settings, pipeline))


def test_api_search_ask_stats(client):
    r = client.post("/search", json={"query": "dense retriever BM25 accuracy", "k": 2, "mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["paper_id"] == "p2" and "rrf" in body["hits"][0]["stages"]

    r = client.post("/ask", json={"question": "How much better is the dense retriever than BM25?"})
    assert r.status_code == 200
    a = r.json()
    assert a["citations"] and a["groundedness"] == 1.0 and "[1]" in a["answer"]

    assert client.get("/stats").json()["papers"] == 2
    assert client.get("/health").json() == {"status": "ok"}


def test_api_validation(client):
    assert client.post("/search", json={"query": "x"}).status_code == 422
    assert client.post("/search", json={"query": "bert", "mode": "magic"}).status_code == 422
    assert client.post("/ingest/pdf", files={"file": ("a.txt", b"hello")}).status_code == 415


def test_api_pdf_upload_is_searchable(client):
    files = {"file": ("sparse.pdf", make_pdf(PAPER_LINES), "application/pdf")}
    r = client.post("/ingest/pdf", files=files)
    assert r.status_code == 200 and r.json()["chunks_added"] >= 2
    hits = client.post("/search", json={"query": "learning rate epochs GPUs", "k": 1}).json()["hits"]
    assert hits[0]["paper_id"] == "upload:sparse"
