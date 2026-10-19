"""Answer generation with citations, followed by citation verification.

* ``ExtractiveGenerator`` (default) selects the most relevant sentences from
  the retrieved chunks using the reranker, so every claim is a verbatim quote
  and the system works with no API key.
* ``AnthropicGenerator`` writes an abstractive answer with Claude, constrained
  to cite numbered sources; its citations go through the same verifier and
  unsupported ones are marked unverified.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Protocol

from .grounding import support
from .models import Reranker
from .text import split_sentences, tokenize
from .types import Answer, AnswerSentence, Citation, Hit


class Generator(Protocol):
    name: str

    def draft(self, question: str, hits: Sequence[Hit]) -> list[AnswerSentence]:
        """Return answer sentences whose citations are 1-based indices into hits."""
        ...


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, reranker: Reranker, max_sentences: int = 3, redundancy: float = 0.6) -> None:
        self.reranker = reranker
        self.max_sentences = max_sentences
        self.redundancy = redundancy

    def draft(self, question: str, hits: Sequence[Hit]) -> list[AnswerSentence]:
        cands: list[tuple[str, int]] = []
        for n, h in enumerate(hits, start=1):
            for s in split_sentences(h.chunk.text):
                if len(s.split()) >= 4:
                    cands.append((s, n))
        if not cands:
            return []
        scores = self.reranker.score(question, [s for s, _ in cands])
        # Small prior for chunks the retriever ranked higher.
        ranked = sorted(range(len(cands)), key=lambda i: -(float(scores[i]) - 0.05 * (cands[i][1] - 1)))
        chosen: list[tuple[str, int]] = []
        for i in ranked:
            s, n = cands[i]
            if any(_jaccard(s, c) > self.redundancy for c, _ in chosen):
                continue
            chosen.append((s, n))
            if len(chosen) == self.max_sentences:
                break
        return [AnswerSentence(s, [n]) for s, n in chosen]


_PROMPT = """You answer questions about research papers using ONLY the numbered sources.
Rules:
- Every sentence must cite at least one source like [2]. Cite only sources that state the claim.
- Do not use outside knowledge. Do not invent numbers.
- If the sources do not contain the answer, reply with a single sentence saying so and cite nothing.
Return JSON only: {{"sentences": [{{"text": "...", "citations": [1]}}]}}

Question: {question}

Sources:
{sources}"""


class AnthropicGenerator:
    name = "anthropic"

    def __init__(self, model: str, client=None, max_tokens: int = 600) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def draft(self, question: str, hits: Sequence[Hit]) -> list[AnswerSentence]:
        sources = "\n\n".join(
            f"[{n}] ({h.chunk.title} | {h.chunk.section})\n{h.chunk.text}" for n, h in enumerate(hits, start=1)
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": _PROMPT.format(question=question, sources=sources)}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return parse_llm_answer(text, len(hits))


def parse_llm_answer(text: str, n_sources: int) -> list[AnswerSentence]:
    """Parse the model's JSON, tolerating code fences; drop out-of-range citations."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return [AnswerSentence(text.strip(), [])] if text.strip() else []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [AnswerSentence(text.strip(), [])]
    out = []
    for s in data.get("sentences", []):
        cites = [int(c) for c in s.get("citations", []) if str(c).isdigit() and 1 <= int(c) <= n_sources]
        if str(s.get("text", "")).strip():
            out.append(AnswerSentence(str(s["text"]).strip(), sorted(set(cites))))
    return out


def answer_question(question: str, hits: Sequence[Hit], generator: Generator, threshold: float = 0.5) -> Answer:
    """Draft an answer and verify each citation against its chunk.

    Citation numbers are renumbered so the answer cites [1], [2], ... in order
    of first use, and each citation carries its supporting quote."""
    drafted = generator.draft(question, hits)
    renumber: dict[int, int] = {}
    citations: list[Citation] = []
    sentences: list[AnswerSentence] = []
    grounded = 0
    for s in drafted:
        new_refs = []
        ok = bool(s.citations)
        for src in s.citations:
            hit = hits[src - 1]
            sup = support(s.text, hit.chunk.text)
            verified = sup.verified(threshold)
            ok = ok and verified
            if src not in renumber:
                renumber[src] = len(renumber) + 1
                citations.append(
                    Citation(
                        n=renumber[src],
                        chunk_id=hit.chunk.chunk_id,
                        paper_id=hit.chunk.paper_id,
                        title=hit.chunk.title,
                        section=hit.chunk.section,
                        quote=sup.quote,
                        support=sup.score,
                        verified=verified,
                    )
                )
            else:
                # A source cited by several sentences is verified only if it
                # supports all of them.
                c = citations[renumber[src] - 1]
                c.verified = c.verified and verified
                c.support = min(c.support, sup.score)
            new_refs.append(renumber[src])
        grounded += ok
        sentences.append(AnswerSentence(s.text, new_refs))
    groundedness = grounded / len(sentences) if sentences else 0.0
    return Answer(question, sentences, citations, round(groundedness, 4), generator.name)
