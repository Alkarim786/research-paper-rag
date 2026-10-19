from paperrag.chunking import chunk_document
from paperrag.text import split_sentences, tokenize
from paperrag.types import Document, Paragraph


def test_tokenize_drops_stopwords_and_keeps_hyphenated_terms():
    assert tokenize("The state-of-the-art model uses BM25!") == ["state-of-the-art", "model", "uses", "bm25"]
    assert "the" in tokenize("the model", keep_stopwords=True)


def test_split_sentences_handles_abbreviations():
    text = "We follow Devlin et al. (2019) closely. Results are in Fig. 3 below. See e.g. the appendix. Done."
    assert split_sentences(text) == [
        "We follow Devlin et al. (2019) closely.",
        "Results are in Fig. 3 below.",
        "See e.g. the appendix.",
        "Done.",
    ]


def _doc(paragraphs):
    return Document("d", "T", "An abstract.", [Paragraph(s, t) for s, t in paragraphs])


def test_short_paragraphs_merge_within_a_section_only():
    doc = _doc([("A", "one two three."), ("A", "four five six."), ("B", "seven eight nine.")])
    chunks = chunk_document(doc, max_words=50, overlap_words=5)
    assert [c.section for c in chunks] == ["Abstract", "A", "B"]
    assert chunks[1].text == "one two three. four five six."
    assert chunks[1].paragraph_ids == (0, 1)
    assert chunks[2].paragraph_ids == (2,)


def test_long_paragraph_split_with_overlap():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(30)]
    doc = _doc([("Body", " ".join(sentences))])
    chunks = chunk_document(doc, max_words=50, overlap_words=10)[1:]  # skip abstract
    assert len(chunks) > 3
    assert all(len(c.text.split()) <= 50 for c in chunks)
    assert all(c.paragraph_ids == (0,) for c in chunks)
    # consecutive windows share their boundary sentence(s)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert split_sentences(a.text)[-1] == split_sentences(b.text)[0]
    # every sentence is covered
    covered = {s for c in chunks for s in split_sentences(c.text)}
    assert covered == set(sentences)


def test_chunk_ids_are_stable_and_unique():
    doc = _doc([("A", "x " * 60), ("B", "y " * 60)])
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert len(ids) == len(set(ids)) and ids[0] == "d#0"
