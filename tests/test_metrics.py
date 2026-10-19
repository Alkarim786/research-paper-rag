import math

import pytest

from paperrag.eval.metrics import hit_at_k, max_f1, mrr_at_k, ndcg_at_k, recall_at_k, token_f1


def test_ranking_metrics():
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d"}
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert hit_at_k(ranked, rel, 1) == 0.0 and hit_at_k(ranked, rel, 2) == 1.0
    assert mrr_at_k(ranked, rel) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, rel, 10) == pytest.approx(expected)
    assert ndcg_at_k(["b", "d"], rel) == pytest.approx(1.0)


def test_token_f1_squad_style():
    assert token_f1("The BooksCorpus!", "bookscorpus") == 1.0
    assert token_f1("BooksCorpus and Wikipedia", "Wikipedia") == pytest.approx(2 * (1 / 3) * 1 / (1 / 3 + 1))
    assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0
    assert max_f1("wikipedia", ["books", "english wikipedia"]) == pytest.approx(2 / 3)
