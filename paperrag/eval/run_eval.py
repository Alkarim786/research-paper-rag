"""Evaluate retrieval and answering on QASPER.

Retrieval is evaluated *within each question's paper* (the standard QASPER
setting: the reader knows which paper the question is about). A chunk is
relevant if it contains any gold evidence paragraph from any annotator.

    python -m paperrag.eval.run_eval --data data/qasper/validation.parquet --papers 100

Writes results/eval_<timestamp>.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from ..answer import ExtractiveGenerator, answer_question
from ..config import Settings
from ..ingest.qasper import load_qasper
from ..models import make_embedder, make_reranker
from ..pipeline import RAGPipeline
from ..types import Chunk
from .metrics import hit_at_k, max_f1, mean, mrr_at_k, ndcg_at_k, percentile, recall_at_k

MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def relevant_chunks(chunks: list[Chunk], doc_paragraphs: dict[str, list[str]], evidence: list[set[str]]) -> set[str]:
    """Map gold evidence paragraph texts to the chunk ids that contain them."""
    wanted = set().union(*evidence) if evidence else set()
    rel = set()
    for c in chunks:
        paras = doc_paragraphs[c.paper_id]
        for pid in c.paragraph_ids:
            text = paras[pid] if pid >= 0 else None
            if text is not None and text in wanted:
                rel.add(c.chunk_id)
                break
    return rel


def cpu_name() -> str:
    try:
        import subprocess

        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/qasper/validation.parquet"))
    ap.add_argument("--papers", type=int, default=100, help="number of papers (0 = all)")
    ap.add_argument("--embedder", default="sentence-transformers")
    ap.add_argument("--reranker", default="cross-encoder")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip-answers", action="store_true")
    args = ap.parse_args()

    s = Settings(embedder=args.embedder, reranker=args.reranker)
    docs, items = load_qasper(args.data, args.papers or None)
    doc_paragraphs = {d.paper_id: [p.text for p in d.paragraphs] for d in docs}

    embedder = make_embedder(s.embedder, s.embed_model)
    reranker = make_reranker(s.reranker, s.rerank_model)
    pipe = RAGPipeline(s, embedder, reranker, ExtractiveGenerator(reranker, s.max_answer_sentences))

    t0 = time.perf_counter()
    n_chunks = pipe.build(docs)
    build_s = time.perf_counter() - t0
    chunks = pipe.retriever.chunks
    print(f"indexed {len(docs)} papers -> {n_chunks} chunks in {build_s:.1f}s; {len(items)} questions")

    per_mode: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in MODES}
    answer_stats: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    for it in items:
        rel = relevant_chunks(chunks, doc_paragraphs, it.evidence)
        if not rel:
            skipped += 1  # evidence paragraph not found in parsed text (rare)
            continue
        for mode in MODES:
            t = time.perf_counter()
            hits, _ = pipe.search(it.question, k=10, mode=mode, paper_ids=[it.paper_id])
            per_mode[mode]["latency_ms"].append((time.perf_counter() - t) * 1e3)
            ranked = [h.chunk.chunk_id for h in hits]
            m = per_mode[mode]
            m["recall@1"].append(recall_at_k(ranked, rel, 1))
            m["recall@5"].append(recall_at_k(ranked, rel, 5))
            m["hit@5"].append(hit_at_k(ranked, rel, 5))
            m["mrr@10"].append(mrr_at_k(ranked, rel, 10))
            m["ndcg@10"].append(ndcg_at_k(ranked, rel, 10))
            if mode == "hybrid_rerank" and not args.skip_answers:
                t = time.perf_counter()
                ans = answer_question(it.question, hits[:5], pipe.generator, s.support_threshold)
                answer_stats["gen_latency_ms"].append((time.perf_counter() - t) * 1e3)
                answer_stats["answer_f1"].append(max_f1(ans.plain_text, it.answers))
                cited = {c.chunk_id for c in ans.citations}
                answer_stats["citation_precision"].append(len(cited & rel) / len(cited) if cited else 0.0)
                answer_stats["groundedness"].append(ans.groundedness)
    n_eval = len(items) - skipped

    summary = {
        "dataset": f"QASPER {args.data.stem}",
        "papers": len(docs),
        "chunks": n_chunks,
        "questions_evaluated": n_eval,
        "questions_skipped_no_evidence_match": skipped,
        "index_build_s": round(build_s, 1),
        "embedder": embedder.name,
        "reranker": reranker.name,
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "retrieval": {
            mode: {k: round(mean(v), 4) for k, v in m.items() if k != "latency_ms"}
            | {
                "p50_ms": round(percentile(m["latency_ms"], 0.5), 1),
                "p95_ms": round(percentile(m["latency_ms"], 0.95), 1),
            }
            for mode, m in per_mode.items()
        },
    }
    if answer_stats:
        summary["answering"] = {
            "generator": "extractive",
            "answer_token_f1": round(mean(answer_stats["answer_f1"]), 4),
            "citation_precision": round(mean(answer_stats["citation_precision"]), 4),
            "groundedness": round(mean(answer_stats["groundedness"]), 4),
            "gen_p50_ms": round(percentile(answer_stats["gen_latency_ms"], 0.5), 1),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print()
    print("| Mode | Recall@1 | Recall@5 | Hit@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for mode, r in summary["retrieval"].items():
        print(
            f"| {mode} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | {r['hit@5']:.3f} | {r['mrr@10']:.3f} "
            f"| {r['ndcg@10']:.3f} | {r['p50_ms']} | {r['p95_ms']} |"
        )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
