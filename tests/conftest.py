from __future__ import annotations

import pytest

from paperrag.answer import ExtractiveGenerator
from paperrag.config import Settings
from paperrag.models import HashingEmbedder, LexicalReranker
from paperrag.pipeline import RAGPipeline
from paperrag.types import Document, Paragraph


def make_pdf(lines: list[str]) -> bytes:
    """Build a minimal valid one-page PDF with the given text lines (Helvetica).
    Hand-rolled so tests need no PDF-writing dependency."""
    esc = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in lines]
    ops = ["BT", "/F1 10 Tf", "12 TL", "50 780 Td"] + [f"({ln}) Tj T*" for ln in esc] + ["ET"]
    stream = "\n".join(ops).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(
            "p1",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "We introduce BERT, a language representation model pre-trained on unlabeled text.",
            [
                Paragraph("Introduction", "Language model pre-training has been shown to improve many NLP tasks."),
                Paragraph(
                    "Pre-training Data",
                    "For the pre-training corpus we use the BooksCorpus with 800M words and English Wikipedia "
                    "with 2,500M words. We extract only the text passages and ignore lists and tables.",
                ),
                Paragraph(
                    "Experiments",
                    "BERT obtains new state-of-the-art results on eleven natural language processing tasks, "
                    "including pushing the GLUE score to 80.5%.",
                ),
            ],
        ),
        Document(
            "p2",
            "Dense Passage Retrieval for Open-Domain Question Answering",
            "We show that retrieval can be practically implemented using dense representations alone.",
            [
                Paragraph(
                    "Method",
                    "The dense passage retriever uses two independent BERT encoders and the dot product of their "
                    "outputs as the similarity. Negatives come from in-batch training examples.",
                ),
                Paragraph(
                    "Results",
                    "Our dense retriever outperforms a strong Lucene BM25 system by 9 to 19 points in top-20 "
                    "passage retrieval accuracy.",
                ),
            ],
        ),
    ]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        index_dir=tmp_path / "index",
        embedder="hashing",
        reranker="lexical",
        generator="extractive",
        candidates_per_retriever=10,
        rerank_top_n=10,
        top_k=3,
    )


@pytest.fixture
def pipeline(settings, docs) -> RAGPipeline:
    rr = LexicalReranker()
    p = RAGPipeline(settings, HashingEmbedder(), rr, ExtractiveGenerator(rr, max_sentences=2))
    p.build(docs)
    return p
