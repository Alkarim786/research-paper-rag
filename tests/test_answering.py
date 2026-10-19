import pytest

from paperrag.answer import AnswerSentence, answer_question, parse_llm_answer
from paperrag.grounding import numbers, support


def test_support_scores_and_numbers():
    chunk = "We use the BooksCorpus (800M words) and English Wikipedia (2,500M words)."
    good = support("They pre-train on BooksCorpus and English Wikipedia.", chunk)
    assert good.score >= 0.75 and good.numbers_ok and good.verified(0.5)
    wrong_number = support("BooksCorpus has 900M words.", chunk)
    assert not wrong_number.numbers_ok and not wrong_number.verified(0.5)
    unrelated = support("The model is trained with reinforcement learning on Atari.", chunk)
    assert unrelated.score < 0.3
    assert numbers("GLUE 80.5% and 2,500M") == {"80.5%", "2500"}


def test_extractive_answer_is_quoted_and_fully_grounded(pipeline):
    ans = pipeline.ask("What corpus is used for pre-training?")
    assert ans.generator == "extractive"
    assert ans.groundedness == 1.0
    assert ans.citations and all(c.verified for c in ans.citations)
    assert any("BooksCorpus" in s.text for s in ans.sentences)
    # citations are renumbered 1..n in order of first use
    assert [c.n for c in ans.citations] == list(range(1, len(ans.citations) + 1))
    assert "retrieval_ms" in ans.timings_ms


class FakeLLM:
    """Simulates an abstractive generator that gets one claim wrong."""

    name = "fake-llm"

    def draft(self, question, hits):
        books = next(n for n, h in enumerate(hits, 1) if "BooksCorpus" in h.chunk.text)
        return [
            AnswerSentence("BERT is pre-trained on BooksCorpus and English Wikipedia.", [books]),
            AnswerSentence("BooksCorpus contains 900M words.", [books]),  # hallucinated number
            AnswerSentence("It was trained on a cluster of GPUs.", []),  # uncited
        ]


def test_verifier_flags_hallucinated_numbers_and_uncited_claims(pipeline):
    hits, _ = pipeline.search("BERT pre-training corpus BooksCorpus", k=3)
    ans = answer_question("What data?", hits, FakeLLM())
    assert ans.groundedness == pytest.approx(1 / 3, abs=1e-4)
    # the shared citation is unverified because one sentence citing it is wrong
    assert len(ans.citations) == 1 and not ans.citations[0].verified


def test_parse_llm_answer_tolerates_fences_and_bad_refs():
    raw = 'Here you go:\n```json\n{"sentences": [{"text": "A.", "citations": [1, 9]}, {"text": "B.", "citations": ["2"]}]}\n```'
    out = parse_llm_answer(raw, n_sources=3)
    assert [(s.text, s.citations) for s in out] == [("A.", [1]), ("B.", [2])]
    assert parse_llm_answer("not json", 3)[0].citations == []


def test_no_hits_answer(pipeline):
    ans = pipeline.ask("anything", paper_ids=["nope"])
    assert ans.citations == [] and ans.groundedness == 0.0
