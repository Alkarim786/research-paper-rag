import math

import numpy as np
import pytest

from paperrag.bm25 import BM25Index
from paperrag.models import HashingEmbedder, LexicalReranker
from paperrag.retriever import DenseIndex, HybridRetriever, reciprocal_rank_fusion


def test_bm25_matches_hand_computed_score():
    docs = ["cat sat mat", "dog sat log dog", "bird flew"]
    idx = BM25Index.build(docs, k1=1.5, b=0.75)
    # query "dog": df=1, N=3 -> idf = ln(1 + (3-1+.5)/(1+.5)) = ln(1 + 2.5/1.5)
    idf = math.log(1 + 2.5 / 1.5)
    avgdl = (3 + 4 + 2) / 3
    tf, dl = 2, 4
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    ((doc, score),) = idx.search("dog", k=5)
    assert doc == 1 and score == pytest.approx(expected)


def test_bm25_rare_terms_outweigh_common_ones():
    docs = ["common common rare", "common common common", "common other"]
    idx = BM25Index.build(docs)
    assert idx.search("common rare", k=1)[0][0] == 0
    assert idx.idf("rare") > idx.idf("common")


def test_bm25_save_load_roundtrip(tmp_path):
    idx = BM25Index.build(["alpha beta", "beta gamma", "gamma delta"])
    idx.save(tmp_path / "b.json")
    loaded = BM25Index.load(tmp_path / "b.json")
    assert loaded.search("gamma", 3) == idx.search("gamma", 3)


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    order = [d for d, _ in fused]
    assert order[0] == 1  # ranks 1 and 2
    assert order.index(3) < order.index(2)  # 3 appears in both lists
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 62)


def test_dense_index_topk_and_mask():
    vecs = np.eye(4, dtype=np.float32)
    idx = DenseIndex(vecs)
    q = np.array([0.1, 0.9, 0.0, 0.2], dtype=np.float32)
    assert [i for i, _ in idx.search(q, 2)] == [1, 3]
    assert [i for i, _ in idx.search(q, 2, allowed=np.array([0, 2]))] == [0, 2]


@pytest.mark.parametrize("mode", HybridRetriever.MODES)
def test_all_modes_find_the_obvious_passage(pipeline, mode):
    hits, timings = pipeline.search("BooksCorpus Wikipedia pre-training corpus", k=3, mode=mode)
    assert hits[0].chunk.paper_id == "p1"
    assert "BooksCorpus" in hits[0].chunk.text
    assert timings  # per-stage latency is reported


def test_paper_filter(pipeline):
    hits, _ = pipeline.search("BERT encoders", k=10, paper_ids=["p2"])
    assert hits and {h.chunk.paper_id for h in hits} == {"p2"}
    assert pipeline.search("BERT", paper_ids=["missing"])[0] == []


def test_index_persistence(pipeline, settings, tmp_path):
    pipeline.save(tmp_path / "ix")
    r2 = HybridRetriever.load(tmp_path / "ix", HashingEmbedder(), LexicalReranker())
    a, _ = pipeline.retriever.search("dense retriever BM25", 3)
    b, _ = r2.search("dense retriever BM25", 3)
    assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]


def test_incremental_add(pipeline, docs):
    from paperrag.types import Document, Paragraph

    before = pipeline.stats()["chunks"]
    added = pipeline.add(
        [
            Document(
                "p3",
                "Attention Is All You Need",
                "",
                [Paragraph("Model", "The Transformer uses multi-head self-attention only.")],
            )
        ]
    )
    assert added == 1 and pipeline.stats()["chunks"] == before + 1  # no abstract -> one chunk
    assert pipeline.add(docs) == 0  # already indexed papers are skipped
    hits, _ = pipeline.search("multi-head self-attention Transformer", k=1)
    assert hits[0].chunk.paper_id == "p3"
