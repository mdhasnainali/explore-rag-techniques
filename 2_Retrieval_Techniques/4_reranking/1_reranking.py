from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

documents = [
    "Python is a high-level programming language known for readability.",
    "Neural networks are inspired by the human brain structure.",
    "The capital of France is Paris, a city on the Seine river.",
    "Machine learning enables systems to learn patterns from data.",
    "Paris hosted the 1900 and 1924 Olympic Games.",
    "Deep learning uses multi-layer neural networks for representation learning.",
    "France is a country in Western Europe with a population of 68 million.",
    "The Eiffel Tower is located in Paris and was built in 1889.",
]

# ── Stage 1: dense retrieval (fast, broad) ────────────────────────────────────
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = bi_encoder.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

# ── Stage 2: cross-encoder reranker (slow, precise) ──────────────────────────
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "What is the capital city of France?"

# Retrieve top-6 candidates with bi-encoder
q_emb = bi_encoder.encode([query]).astype("float32")
faiss.normalize_L2(q_emb)
_, candidate_idx = index.search(q_emb, 6)
candidates = [documents[i] for i in candidate_idx[0]]

# Rerank with cross-encoder
pairs = [[query, doc] for doc in candidates]
ce_scores = cross_encoder.predict(pairs)
reranked = sorted(zip(ce_scores, candidates), reverse=True)

print(f"Query: '{query}'\n")
print("After bi-encoder (initial order):")
for i, doc in enumerate(candidates):
    print(f"  {i+1}. {doc[:70]}")

print("\nAfter cross-encoder reranking:")
for i, (score, doc) in enumerate(reranked[:3]):
    print(f"  {i+1} [score={score:.3f}]: {doc[:70]}")


# Output:
# Query: 'What is the capital city of France?'
#
# After bi-encoder (initial order):
#   1. The capital of France is Paris, a city on the Seine river.
#   2. France is a country in Western Europe with a population of 68 million.
#   3. Paris hosted the 1900 and 1924 Olympic Games.
#   4. The Eiffel Tower is located in Paris and was built in 1889.
#   5. Python is a high-level programming language known for readability.
#   6. Machine learning enables systems to learn patterns from data.
#
# After cross-encoder reranking:
#   1 [score=9.234]: The capital of France is Paris, a city on the Seine river.
#   2 [score=4.112]: France is a country in Western Europe with a population of 68 million.
#   3 [score=2.891]: The Eiffel Tower is located in Paris and was built in 1889.

# Findings:
# The bi-encoder already ranks the correct answer first here, but the
# cross-encoder assigns a much higher score gap (9.2 vs 4.1) — making the
# top result more confidently separated from the rest.
# Cross-encoders read query and document together, capturing fine-grained
# relevance that bi-encoders miss (bi-encoders embed independently).
# The trade-off: cross-encoder is ~10-50x slower than bi-encoder similarity
# search, so it is only applied to the top-k candidates, not the full corpus.
# Typical pipeline: retrieve top-20 with bi-encoder, rerank to top-5 with
# cross-encoder, send top-5 to LLM.
