from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
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

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
dense_index = faiss.IndexFlatIP(doc_embs.shape[1])
dense_index.add(doc_embs)

tokenized = [re.findall(r"\w+", d.lower()) for d in documents]
bm25 = BM25Okapi(tokenized)


# ── Rule-based query classifier ───────────────────────────────────────────────
# In production, replace with an LLM classifier for more nuanced routing.
EXACT_SIGNALS = {"error", "exception", "traceback", "runtimeerror", "valueerror",
                 "typeerror", "oom", "cuda", "sku", "id", "code", "fix"}

def classify_query(query: str) -> str:
    tokens = set(re.findall(r"\w+", query.lower()))
    if tokens & EXACT_SIGNALS:
        return "exact"       # → sparse / BM25
    return "semantic"        # → dense / embedding


def retrieve(query: str, k: int = 3) -> list:
    qtype = classify_query(query)

    if qtype == "exact":
        tokens = re.findall(r"\w+", query.lower())
        scores = bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return qtype, [(documents[i], float(scores[i])) for i in top_idx]

    else:
        q_emb = model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, idx = dense_index.search(q_emb, k)
        return qtype, [(documents[i], float(s)) for i, s in zip(idx[0], scores[0])]


queries = [
    "CUDA out of memory RuntimeError fix",
    "Which city is the French capital?",
    "ValueError broadcast shapes",
    "How does machine learning work?",
]

for query in queries:
    qtype, results = retrieve(query)
    print(f"Query: '{query}'")
    print(f"  → Routed to: {qtype} retrieval")
    for doc, score in results:
        print(f"  [score={score:.3f}] {doc[:70]}")
    print()


# Output:
# Query: 'CUDA out of memory RuntimeError fix'
#   → Routed to: exact retrieval
#   [score=3.891] RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.
#   [score=2.134] Fix for CUDA OOM: reduce batch size or use gradient checkpointing.
#   [score=0.000] Python is a high-level programming language known for readability.
#
# Query: 'Which city is the French capital?'
#   → Routed to: semantic retrieval
#   [score=0.789] The capital of France is Paris, a city on the Seine river.
#   [score=0.346] Paris hosted the 1900 and 1924 Olympic Games.
#   [score=0.201] Machine learning enables systems to learn patterns from data.
#
# Query: 'ValueError broadcast shapes'
#   → Routed to: exact retrieval
#   [score=2.891] ValueError: operands could not be broadcast with shapes (3,) (4,).
#   [score=0.000] Python is a high-level programming language known for readability.
#   [score=0.000] Neural networks are inspired by the human brain structure.
#
# Query: 'How does machine learning work?'
#   → Routed to: semantic retrieval
#   [score=0.712] Machine learning enables systems to learn patterns from data.
#   [score=0.489] Neural networks are inspired by the human brain structure.
#   [score=0.401] Python is a high-level programming language known for readability.

# Findings:
# Error/code queries are routed to BM25 which matches exact tokens like
# "RuntimeError", "CUDA", "ValueError" — terms that embedding models may
# not distinguish well from similar-sounding terms.
# Conceptual queries are routed to dense retrieval which handles paraphrase
# ("French capital" → "capital of France") without shared vocabulary.
# The classifier is the weakest link — a wrong route hides good documents.
# In production, use an LLM to classify into: Factual, Analytical, Exact, or
# Contextual, and route each to the appropriate retrieval strategy.
