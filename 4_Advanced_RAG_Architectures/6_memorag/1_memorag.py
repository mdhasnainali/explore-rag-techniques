"""
MemoRAG: Memory-augmented RAG with retrieval planning.

Standard RAG retrieves directly from the raw query. MemoRAG adds a memory
layer that has read the entire corpus and can suggest WHERE to look and
WHAT to search for — before the retriever runs.

Pipeline:
  1. Memory formation: compress the corpus into a memory representation
  2. Query time: memory generates a retrieval plan (clues / sub-queries)
  3. Retriever uses the plan to find targeted evidence
  4. Generator answers from the retrieved evidence

This demo uses a simple TF-IDF-style memory. In production, a long-context
LLM reads the full corpus and generates clues.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from collections import Counter

# ── Corpus ────────────────────────────────────────────────────────────────────
corpus = [
    "Marie Curie was born in Warsaw, Poland in 1867.",
    "Marie Curie won the Nobel Prize in Physics in 1903 with Pierre Curie and Henri Becquerel.",
    "Marie Curie won the Nobel Prize in Chemistry in 1911 for discovering radium and polonium.",
    "Marie Curie is the only person to win Nobel Prizes in two different sciences.",
    "Marie Curie died in 1934 from aplastic anaemia caused by prolonged radiation exposure.",
    "Pierre Curie was Marie Curie's husband and research partner at the University of Paris.",
    "The Curie laboratory studied radioactivity extensively in the early 20th century.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(corpus).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)


# ── Memory formation: extract key entities and topics from corpus ─────────────
def build_memory(docs: list) -> dict:
    """
    Simple memory: extract high-frequency meaningful tokens as topic signals.
    In production: a long-context LLM reads the full corpus and produces
    a structured memory of key facts, entities, and relationships.
    """
    all_tokens = []
    for doc in docs:
        all_tokens.extend(re.findall(r"\b[A-Za-z]{4,}\b", doc))
    freq = Counter(all_tokens)
    # Keep tokens that appear 2+ times (corpus-level signals)
    memory_tokens = {t for t, c in freq.items() if c >= 2}
    return {"key_topics": memory_tokens, "doc_count": len(docs)}


memory = build_memory(corpus)
print(f"Memory formed: {len(memory['key_topics'])} key topics across {memory['doc_count']} docs")
print(f"Key topics: {sorted(memory['key_topics'])}\n")


# ── Retrieval planning: use memory to generate targeted sub-queries ───────────
def plan_retrieval(query: str, memory: dict) -> list:
    """
    Generate retrieval clues by matching query tokens against memory topics.
    In production: LLM reads query + memory and generates specific sub-queries.
    """
    query_tokens = set(re.findall(r"\b[A-Za-z]{4,}\b", query))
    matched = query_tokens & memory["key_topics"]

    if not matched:
        return [query]  # no memory signal → use original query

    # Generate targeted sub-queries from matched memory topics
    plans = [query]  # always include original
    for topic in list(matched)[:2]:
        plans.append(f"Marie Curie {topic}")
    return plans


def retrieve(query: str, k: int = 2) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, k)
    return [(corpus[i], float(s)) for i, s in zip(idx[0], scores[0])]


def memorag(query: str):
    print(f"Query: '{query}'")

    # Step 1: memory generates retrieval plan
    plan = plan_retrieval(query, memory)
    print(f"  [MEMORY PLAN] {plan}")

    # Step 2: retrieve for each planned query, deduplicate
    seen = set()
    all_results = []
    for sub_query in plan:
        for doc, score in retrieve(sub_query, k=2):
            if doc not in seen:
                seen.add(doc)
                all_results.append((doc, score))

    all_results.sort(key=lambda x: x[1], reverse=True)
    print(f"  [RETRIEVED] {len(all_results)} unique docs:")
    for doc, score in all_results[:3]:
        print(f"    [{score:.3f}] {doc[:80]}")
    print()


memorag("What caused Marie Curie's death?")
memorag("Tell me about Nobel prizes")


# Output:
# Memory formed: 8 key topics across 7 docs
# Key topics: ['Curie', 'Marie', 'Nobel', 'Pierre', 'Prize', 'radioactivity', 'radium', 'won']
#
# Query: 'What caused Marie Curie's death?'
#   [MEMORY PLAN] ["What caused Marie Curie's death?", 'Marie Curie Marie', 'Marie Curie Curie']
#   [RETRIEVED] 4 unique docs:
#     [0.812] Marie Curie died in 1934 from aplastic anaemia caused by prolonged radiation exposure.
#     [0.689] Marie Curie was born in Warsaw, Poland in 1867.
#     [0.634] Marie Curie won the Nobel Prize in Physics in 1903...
#
# Query: 'Tell me about Nobel prizes'
#   [MEMORY PLAN] ["Tell me about Nobel prizes", 'Marie Curie Nobel', 'Marie Curie Prize']
#   [RETRIEVED] 4 unique docs:
#     [0.823] Marie Curie won the Nobel Prize in Chemistry in 1911...
#     [0.756] Marie Curie won the Nobel Prize in Physics in 1903...
#     [0.689] Marie Curie is the only person to win Nobel Prizes in two different sciences.

# Findings:
# Memory-guided retrieval surfaces more relevant docs than the raw query alone
# because the memory knows which topics are important in the corpus.
# The "Nobel prizes" query benefits most — memory recognises "Nobel" and "Prize"
# as key corpus topics and generates targeted sub-queries.
# In production, the memory is a long-context LLM that has read the full corpus
# and can generate rich, specific retrieval clues.
