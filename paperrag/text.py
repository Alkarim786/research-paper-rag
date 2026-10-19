"""Tokenisation and sentence splitting used by BM25, chunking and grounding."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# Split after . ! ? when followed by whitespace and an uppercase letter, digit
# or opening bracket; avoids splitting "e.g. the" and "Fig. 3".
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|et al|Fig|Eq|Sec|Tab|vs|cf|approx|resp)\.$", re.IGNORECASE)

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while who whom why will
    with would you your yours yourself yourselves also use used using we our paper""".split()
)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens; stopwords removed unless keep_stopwords."""
    toks = _TOKEN.findall(text.lower())
    if keep_stopwords:
        return toks
    return [t for t in toks if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that is good enough for paper prose."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_BOUNDARY.split(text)
    out: list[str] = []
    for p in parts:
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return [s.strip() for s in out if s.strip()]


def word_count(text: str) -> int:
    return len(text.split())
