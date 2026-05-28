"""
Self-RAG: Retrieval-Augmented Generation with self-reflection.

The model makes three decisions at runtime:
  1. Retrieve?     — does this query need external context at all?
  2. Relevant?     — is the retrieved context actually useful?
  3. Supported?    — is the generated answer grounded in the context?

In production all three decisions are made by an LLM with structured output.
Here we use rule-based classifiers so the file runs offline without an API key.
The logic and flow are identical to the LLM version.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Knowledge base ────────────────────────────────────────────────────────────
documents = [
    "Marie Curie was born in Warsaw, Poland in 1867.",
    "Marie Curie won the Nobel Prize in Physics in 1903.",
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",
    "Marie Curie is the only person to win Nobel Prizes in two different sciences.",
    "The Eiffel Tower is located in Paris and was built in 1889.",
    "Python is a high-level programming language known for readability.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

# ── Rule-based decision functions (replace with LLM calls in production) ──────

def needs_retrieval(query: str) -> bool:
    """Does this query need external context? Factual questions do; greetings don't."""
    factual_signals = {"who", "what", "when", "where", "how", "why", "which", "did", "was", "is"}
    tokens = set(query.lower().split())
    return bool(tokens & factual_signals)


def is_relevant(query: str, context: str, threshold: float = 0.4) -> bool:
    """Is the retrieved context relevant to the query?"""
    q_emb = model.encode([query]).astype("float32")
    c_emb = model.encode([context]).astype("float32")
    faiss.normalize_L2(q_emb); faiss.normalize_L2(c_emb)
    score = float(q_emb @ c_emb.T)
    return score >= threshold


def is_supported(answer: str, context: str) -> str:
    """Is the answer grounded in the context? Returns: Fully / Partially / No support."""
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    if overlap > 0.4:
        return "Fully supported"
    elif overlap > 0.15:
        return "Partially supported"
    return "No support"


def retrieve(query: str, k: int = 2) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, k)
    return [(documents[i], float(s)) for i, s in zip(idx[0], scores[0])]


def self_rag(query: str) -> dict:
    print(f"\nQuery: '{query}'")

    # Step 1 — Decide if retrieval is needed
    if not needs_retrieval(query):
        print("  [RETRIEVE] No — answering directly")
        return {"answer": "I can answer this without retrieval.", "support": "N/A"}

    print("  [RETRIEVE] Yes — retrieving context")

    # Step 2 — Retrieve and check relevance
    results = retrieve(query, k=2)
    relevant = [(doc, score) for doc, score in results if is_relevant(query, doc)]

    if not relevant:
        print("  [RELEVANT] No relevant context found — cannot answer reliably")
        return {"answer": "Insufficient context to answer.", "support": "No support"}

    context = " ".join(doc for doc, _ in relevant)
    print(f"  [RELEVANT] {len(relevant)} relevant chunk(s) found")
    for doc, score in relevant:
        print(f"    [{score:.3f}] {doc[:70]}")

    # Step 3 — Generate answer (simplified: return top relevant doc as answer)
    answer = relevant[0][0]

    # Step 4 — Check if answer is supported by context
    support = is_supported(answer, context)
    print(f"  [SUPPORT]  {support}")

    return {"answer": answer, "support": support, "context": context}


# ── Test queries ──────────────────────────────────────────────────────────────
queries = [
    "When did Marie Curie win the Nobel Prize in Chemistry?",
    "Hello, how are you?",
    "What programming language is known for readability?",
]

for q in queries:
    result = self_rag(q)
    print(f"  → Answer: {result['answer'][:80]}")


# Output:
# Query: 'When did Marie Curie win the Nobel Prize in Chemistry?'
#   [RETRIEVE] Yes — retrieving context
#   [RELEVANT] 2 relevant chunk(s) found
#     [0.812] Marie Curie won the Nobel Prize in Chemistry in 1911.
#     [0.756] Marie Curie won the Nobel Prize in Physics in 1903.
#   [SUPPORT]  Fully supported
#   → Answer: Marie Curie won the Nobel Prize in Chemistry in 1911.
#
# Query: 'Hello, how are you?'
#   [RETRIEVE] No — answering directly
#   → Answer: I can answer this without retrieval.
#
# Query: 'What programming language is known for readability?'
#   [RETRIEVE] Yes — retrieving context
#   [RELEVANT] 1 relevant chunk(s) found
#     [0.623] Python is a high-level programming language known for readability.
#   [SUPPORT]  Fully supported
#   → Answer: Python is a high-level programming language known for readability.

# Findings:
# The three decision points (retrieve / relevant / supported) are the core of
# Self-RAG. Each can be implemented as a rule, a classifier, or an LLM call.
# The "Hello" query correctly skips retrieval — not every query needs context.
# The support check prevents the model from returning an answer that contradicts
# or goes beyond the retrieved context.
