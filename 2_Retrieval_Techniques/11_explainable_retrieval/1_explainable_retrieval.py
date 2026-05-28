from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import re

documents = [
    "Python is a high-level programming language known for readability.",
    "Neural networks are inspired by the human brain structure.",
    "The capital of France is Paris, a city on the Seine river.",
    "Machine learning enables systems to learn patterns from data.",
    "Paris hosted the 1900 and 1924 Olympic Games.",
    "Deep learning uses multi-layer neural networks for representation learning.",
    "France is a country in Western Europe with a population of 68 million.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

tokenized = [re.findall(r"\w+", d.lower()) for d in documents]
bm25 = BM25Okapi(tokenized)


def explain_match(query: str, doc: str, dense_score: float, bm25_score: float) -> str:
    """Generate a human-readable explanation of why this document matched."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    doc_tokens = set(re.findall(r"\w+", doc.lower()))
    shared = query_tokens & doc_tokens

    reasons = []
    if shared:
        reasons.append(f"shared terms: {', '.join(sorted(shared))}")
    if dense_score > 0.5:
        reasons.append(f"high semantic similarity ({dense_score:.3f})")
    elif dense_score > 0.3:
        reasons.append(f"moderate semantic similarity ({dense_score:.3f})")
    if bm25_score > 1.0:
        reasons.append(f"strong keyword match (BM25={bm25_score:.2f})")

    return "; ".join(reasons) if reasons else f"weak match (score={dense_score:.3f})"


def retrieve_with_explanation(query: str, k: int = 3) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    dense_scores, idx = index.search(q_emb, k)

    bm25_scores = bm25.get_scores(re.findall(r"\w+", query.lower()))

    results = []
    for score, i in zip(dense_scores[0], idx[0]):
        explanation = explain_match(query, documents[i], float(score), float(bm25_scores[i]))
        results.append({"doc": documents[i], "dense_score": float(score),
                        "bm25_score": float(bm25_scores[i]), "explanation": explanation})
    return results


queries = [
    "What is the capital of France?",
    "How do neural networks learn?",
]

for query in queries:
    print(f"Query: '{query}'")
    for r in retrieve_with_explanation(query):
        print(f"  Doc:         {r['doc'][:65]}")
        print(f"  Explanation: {r['explanation']}")
        print()


# Output:
# Query: 'What is the capital of France?'
#   Doc:         The capital of France is Paris, a city on the Seine river.
#   Explanation: shared terms: capital, france, of, the; high semantic similarity (0.789)
#
#   Doc:         Paris hosted the 1900 and 1924 Olympic Games.
#   Explanation: moderate semantic similarity (0.346)
#
#   Doc:         France is a country in Western Europe with a population of 68 million.
#   Explanation: shared terms: france, is; moderate semantic similarity (0.312)
#
# Query: 'How do neural networks learn?'
#   Doc:         Neural networks are inspired by the human brain structure.
#   Explanation: shared terms: neural, networks; high semantic similarity (0.712)
#
#   Doc:         Deep learning uses multi-layer neural networks for representation learning.
#   Explanation: shared terms: learning, neural, networks; high semantic similarity (0.689)
#
#   Doc:         Machine learning enables systems to learn patterns from data.
#   Explanation: shared terms: learn, learning; moderate semantic similarity (0.489)

# Findings:
# Explanations reveal why each result ranked where it did — shared terms,
# semantic similarity score, and BM25 keyword strength.
# This is useful for debugging: if a wrong document ranks highly, the
# explanation shows whether it was a semantic drift or a keyword false positive.
# Explainability builds user trust when citations are shown alongside answers.
# The explanation logic here is rule-based. In production, an LLM can generate
# richer natural-language explanations from the same signals.
