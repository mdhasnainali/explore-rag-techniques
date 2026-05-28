"""
Corrective RAG (CRAG): Grade retrieved documents and correct course if retrieval is weak.

Three outcomes after grading:
  CORRECT   (score >= 0.60) → use retrieved docs directly
  AMBIGUOUS (0.35–0.60)     → use retrieved docs + supplement with web search
  INCORRECT (score < 0.35)  → discard, rewrite query, use web search only

In production the grader is an LLM. Here cosine similarity is used as a proxy
so the file runs offline without an API key.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Marie Curie won the Nobel Prize in Physics in 1903.",
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",
    "The Eiffel Tower is located in Paris and was built in 1889.",
    "Python is a high-level programming language known for readability.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

LOWER = 0.35
UPPER = 0.60


def retrieve(query: str, k: int = 2) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, k)
    return [(documents[i], float(s)) for i, s in zip(idx[0], scores[0])]


def grade(query: str, docs: list) -> tuple:
    best = max(s for _, s in docs)
    if best >= UPPER:   return "CORRECT",   best
    if best >= LOWER:   return "AMBIGUOUS", best
    return "INCORRECT", best


def web_search(query: str) -> list:
    """Simulated web search (use DuckDuckGo / Tavily in production)."""
    return [f"[Web] General information about: {query}"]


def crag(query: str):
    print(f"\nQuery: '{query}'")
    docs = retrieve(query)
    verdict, score = grade(query, docs)
    print(f"  [GRADE] {verdict} (score={score:.3f})")

    if verdict == "CORRECT":
        context = [d for d, _ in docs]
        print("  [ACTION] Using retrieved docs")

    elif verdict == "AMBIGUOUS":
        context = [d for d, _ in docs] + web_search(query)
        print("  [ACTION] Supplementing with web search")

    else:
        rewritten = f"{query} overview"
        context = web_search(rewritten)
        print(f"  [ACTION] Discarding docs. Rewritten: '{rewritten}'")

    for c in context:
        print(f"    - {c[:80]}")


crag("When did Marie Curie win the Nobel Prize in Chemistry?")
crag("What is the population of France?")
crag("Tell me about Paris")


# Output:
# Query: 'When did Marie Curie win the Nobel Prize in Chemistry?'
#   [GRADE] CORRECT (score=0.812)
#   [ACTION] Using retrieved docs
#     - Marie Curie won the Nobel Prize in Chemistry in 1911.
#     - Marie Curie won the Nobel Prize in Physics in 1903.
#
# Query: 'What is the population of France?'
#   [GRADE] INCORRECT (score=0.201)
#   [ACTION] Discarding docs. Rewritten: 'What is the population of France? overview'
#     - [Web] General information about: What is the population of France? overview
#
# Query: 'Tell me about Paris'
#   [GRADE] AMBIGUOUS (score=0.489)
#   [ACTION] Supplementing with web search
#     - The Eiffel Tower is located in Paris and was built in 1889.
#     - Python is a high-level programming language known for readability.
#     - [Web] General information about: Tell me about Paris

# Findings:
# CRAG makes retrieval failure visible and actionable. Instead of silently
# passing weak context to the LLM, it detects the failure and corrects course.
# The two thresholds (LOWER=0.35, UPPER=0.60) control sensitivity — tune per domain.
# In production: grader = LLM relevance scorer, web search = DuckDuckGo/Tavily.
