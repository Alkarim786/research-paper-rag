# Notes for Karim: PaperRAG

## Resume bullets
- Built a **hybrid RAG system** over research papers: BM25 written from scratch plus MiniLM embeddings, fused with RRF and reranked by a cross-encoder, raising **nDCG@10 from 0.401 to 0.497 (+24%)** and Hit@5 from 0.586 to 0.715 on QASPER (888 questions, 12k chunks).
- Added a **citation verifier** that rejects unsupported claims and invented numbers, and reported groundedness and citation precision per answer.
- Shipped a FastAPI service with arXiv/PDF ingestion and 31 tests, using deterministic fake models in CI, plus Docker.

## Likely questions
1. **Why hybrid?** BM25 wins on rare exact terms (dataset names, acronyms); dense wins on paraphrase. RRF merges the rankings without calibrating scores.
2. **Why a cross-encoder only at the end?** It reads query and passage together: accurate, but O(N) model calls. So rerank only the fused top 20.
3. **How do you evaluate RAG?** Separately: retrieval (recall@k, MRR, nDCG against gold evidence) and answers (F1, citation precision, groundedness). Watch for leakage.
4. **How do you stop hallucinated citations?** Verify support between each answer sentence and its cited passage, check numbers, and flag or drop what fails.
5. **How would you scale it?** ANN index (HNSW/FAISS), GPU or ONNX reranker, a cache for frequent queries, async ingestion workers.

## Understand before interviews
- BM25 formula in `bm25.py` (k1 saturation, b length normalisation).
- RRF in `retriever.py`.
- `grounding.support` (content-word recall plus number subset).
- Why the extractive F1 is low: answer length mismatch, not wrong content.
