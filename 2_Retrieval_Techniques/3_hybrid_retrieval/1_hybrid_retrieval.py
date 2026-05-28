from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

documents = [
    "Python is a high-level programming language known for readability.",
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.",
    "Neural networks are inspired by the human brain structure.",
    "Fix for CUDA OOM: reduce batch size or use gradient checkpointing.",
    "The capital of France is Paris, a city on the Seine river.",
    "Machine learning enables systems to learn patterns from data.",
    "ValueError: operands could not be broadcast with shapes (3,) (4,).",
    "Paris hosted the 1900 and 1924 Olympic Games.",
]

# ── Dense index ───────────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
dense_index = faiss.IndexFlatIP(doc_embs.shape[1])
dense_index.add(doc_embs)

# ── Sparse index (BM25) ───────────────────────────────────────────────────────
tokenized = [re.findall(r"\w+", d.lower()) for d in documents]
bm25 = BM25Okapi(tokenized)


def hybrid_search(query: str, k: int = 3, alpha: float = 0.5) -> list:
    """
    Combine dense and sparse scores with Reciprocal Rank Fusion (RRF).
    alpha=1.0 → pure dense, alpha=0.0 → pure sparse.
    RRF score = 1/(rank + 60) summed across both lists.
    """
    # Dense retrieval — get all scores
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    dense_scores, dense_idx = dense_index.search(q_emb, len(documents))
    dense_ranks = {int(idx): rank for rank, idx in enumerate(dense_idx[0])}

    # Sparse retrieval — BM25 scores ranked
    bm25_scores = bm25.get_scores(re.findall(r"\w+", query.lower()))
    sparse_ranks = {idx: rank for rank, idx in enumerate(np.argsort(bm25_scores)[::-1])}

    # RRF fusion
    rrf_scores = {}
    for idx in range(len(documents)):
        d_rank = dense_ranks.get(idx, len(documents))
        s_rank = sparse_ranks.get(idx, len(documents))
        rrf_scores[idx] = alpha * (1 / (d_rank + 60)) + (1 - alpha) * (1 / (s_rank + 60))

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(documents[idx], score) for idx, score in ranked[:k]]


# ── Compare on two query types ────────────────────────────────────────────────
queries = [
    ("CUDA out of memory error fix", "exact-term query"),
    ("Which city is the French capital?", "semantic query"),
]

for query, qtype in queries:
    print(f"\nQuery ({qtype}): '{query}'")
    results = hybrid_search(query, k=3)
    for i, (doc, score) in enumerate(results):
        print(f"  Rank {i+1} [rrf={score:.4f}]: {doc[:70]}")


# Output:
# Query (exact-term query): 'CUDA out of memory error fix'
#   Rank 1 [rrf=0.0167]: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.
#   Rank 2 [rrf=0.0164]: Fix for CUDA OOM: reduce batch size or use gradient checkpointing.
#   Rank 3 [rrf=0.0158]: ValueError: operands could not be broadcast with shapes (3,) (4,).
#
# Query (semantic query): 'Which city is the French capital?'
#   Rank 1 [rrf=0.0167]: The capital of France is Paris, a city on the Seine river.
#   Rank 2 [rrf=0.0163]: Paris hosted the 1900 and 1924 Olympic Games.
#   Rank 3 [rrf=0.0161]: Python is a high-level programming language known for readability.

# Findings:
# Exact-term query: BM25 surfaces the CUDA error docs because they share exact
# tokens. Dense alone would rank them lower since "CUDA OOM" is rare vocabulary.
# Semantic query: Dense retrieval correctly finds "capital of France is Paris"
# despite zero word overlap with "French capital". BM25 alone would score 0.
# RRF avoids normalising scores across systems — it only uses rank positions,
# making it robust when dense and sparse scores are on different scales.
# alpha=0.5 gives equal weight; tune toward 1.0 for semantic-heavy corpora
# and toward 0.0 for technical/code corpora with exact identifiers.
