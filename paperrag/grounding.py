"""Citation verification.

A citation is *verified* when the cited chunk actually supports the sentence:
1. Content-word support: the best-matching sentence in the chunk must cover at
   least ``threshold`` of the answer sentence's content words (stopwords
   removed; light stemming so "models"/"model" match).
2. Numbers must not be invented: every number in the answer sentence must
   appear in the cited chunk. Hallucinated figures are the most damaging RAG
   failure in technical domains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import split_sentences, tokenize

_NUM = re.compile(r"\d+(?:[.,]\d+)?%?")


def _stem(tok: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def content_terms(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if not t.isdigit()}


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


@dataclass
class Support:
    score: float
    quote: str
    numbers_ok: bool

    def verified(self, threshold: float) -> bool:
        return self.numbers_ok and self.score >= threshold


def support(sentence: str, chunk_text: str) -> Support:
    terms = content_terms(sentence)
    chunk_sents = split_sentences(chunk_text) or [chunk_text]
    best_score, best_quote = 0.0, chunk_sents[0]
    if terms:
        for cs in chunk_sents:
            s = len(terms & content_terms(cs)) / len(terms)
            if s > best_score:
                best_score, best_quote = s, cs
        # Allow support spread across adjacent sentences of the chunk.
        whole = len(terms & content_terms(chunk_text)) / len(terms)
        best_score = max(best_score, 0.9 * whole)
    else:
        best_score = 1.0
    nums_ok = numbers(sentence) <= numbers(chunk_text)
    return Support(round(best_score, 4), best_quote, nums_ok)
