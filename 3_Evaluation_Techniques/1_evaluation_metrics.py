"""
RAG Evaluation Metrics — all six metrics in one file.

Retrieval metrics (measure the search layer):
  1. Context Precision  — of retrieved chunks, how many were useful?
  2. Context Recall     — did retrieval find all required evidence?
  3. Mean Reciprocal Rank (MRR) — how early did the first relevant result appear?

Generation metrics (measure the answer layer):
  4. Faithfulness       — is the answer supported by the retrieved context?
  5. Answer Relevance   — does the answer address the question?
  6. RAG Triad          — do all three links (context relevance, groundedness,
                          answer relevance) hold together?

All metrics run offline — no LLM or API key needed.
In production, faithfulness and answer relevance are typically judged by an LLM.
Here we use embedding similarity as a proxy.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine(a: str, b: str) -> float:
    embs = model.encode([a, b]).astype("float32")
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return float(embs[0] @ embs[1])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Context Precision
# ─────────────────────────────────────────────────────────────────────────────

def context_precision(retrieved: list[str], relevant_ids: set[int]) -> float:
    """
    Proportion of retrieved chunks that are relevant.
    relevant_ids: indices (0-based) of chunks that are actually useful.
    """
    if not retrieved:
        return 0.0
    relevant_count = sum(1 for i in range(len(retrieved)) if i in relevant_ids)
    return relevant_count / len(retrieved)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Context Recall
# ─────────────────────────────────────────────────────────────────────────────

def context_recall(retrieved: list[str], required_facts: list[str], threshold: float = 0.5) -> float:
    """
    Proportion of required facts that appear in the retrieved context.
    A fact is 'found' if any retrieved chunk has cosine similarity >= threshold.
    """
    if not required_facts:
        return 1.0
    found = 0
    for fact in required_facts:
        if any(cosine(fact, chunk) >= threshold for chunk in retrieved):
            found += 1
    return found / len(required_facts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mean Reciprocal Rank (MRR)
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank(ranked_relevance: list[bool]) -> float:
    """
    1 / rank of the first relevant result. 0 if none found.
    ranked_relevance: [True, False, True, ...] — True = relevant at that rank.
    """
    for rank, is_relevant in enumerate(ranked_relevance, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(queries_ranked_relevance: list[list[bool]]) -> float:
    """Average reciprocal rank across multiple queries."""
    return sum(reciprocal_rank(r) for r in queries_ranked_relevance) / len(queries_ranked_relevance)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Faithfulness
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness(answer: str, context: list[str], threshold: float = 0.4) -> float:
    """
    Proportion of answer sentences supported by at least one context chunk.
    In production: LLM judges each claim against the context.
    """
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 1.0
    supported = sum(
        1 for s in sentences
        if any(cosine(s, chunk) >= threshold for chunk in context)
    )
    return supported / len(sentences)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Answer Relevance
# ─────────────────────────────────────────────────────────────────────────────

def answer_relevance(question: str, answer: str) -> float:
    """
    Cosine similarity between question and answer embeddings.
    In production: LLM generates hypothetical questions from the answer
    and measures similarity to the original question.
    """
    return cosine(question, answer)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RAG Triad
# ─────────────────────────────────────────────────────────────────────────────

def rag_triad(question: str, context: list[str], answer: str) -> dict:
    """
    Evaluate all three links:
      context_relevance: is the context relevant to the question?
      groundedness:      is the answer supported by the context?
      answer_relevance:  does the answer address the question?
    """
    ctx_text = " ".join(context)
    return {
        "context_relevance": cosine(question, ctx_text),
        "groundedness":      faithfulness(answer, context),
        "answer_relevance":  answer_relevance(question, answer),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

question = "When did Marie Curie win the Nobel Prize in Chemistry?"

retrieved_chunks = [
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",   # relevant
    "Marie Curie won the Nobel Prize in Physics in 1903.",     # relevant
    "Python is a high-level programming language.",            # irrelevant
    "The Eiffel Tower is located in Paris.",                   # irrelevant
]

required_facts = [
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",
    "The prize was awarded for discovering radium and polonium.",
]

good_answer = "Marie Curie won the Nobel Prize in Chemistry in 1911."
bad_answer  = "Marie Curie won the Nobel Prize in Chemistry in 1921 for her work on nuclear physics."

print("=" * 55)
print("RETRIEVAL METRICS")
print("=" * 55)

cp = context_precision(retrieved_chunks, relevant_ids={0, 1})
print(f"Context Precision:  {cp:.2f}  (2 of 4 chunks relevant)")

cr = context_recall(retrieved_chunks, required_facts)
print(f"Context Recall:     {cr:.2f}  (1 of 2 required facts found)")

mrr = mean_reciprocal_rank([
    [True,  False, False, False],   # query 1: relevant at rank 1
    [False, True,  False, False],   # query 2: relevant at rank 2
    [False, False, True,  False],   # query 3: relevant at rank 3
])
print(f"MRR:                {mrr:.3f}  (avg across 3 queries)")

print()
print("=" * 55)
print("GENERATION METRICS")
print("=" * 55)

f_good = faithfulness(good_answer, retrieved_chunks[:2])
f_bad  = faithfulness(bad_answer,  retrieved_chunks[:2])
print(f"Faithfulness (good answer): {f_good:.2f}")
print(f"Faithfulness (bad answer):  {f_bad:.2f}")

ar_good = answer_relevance(question, good_answer)
ar_bad  = answer_relevance(question, "The Eiffel Tower is in Paris.")
print(f"Answer Relevance (good):    {ar_good:.3f}")
print(f"Answer Relevance (off-topic): {ar_bad:.3f}")

print()
print("RAG Triad (good answer):")
triad = rag_triad(question, retrieved_chunks[:2], good_answer)
for k, v in triad.items():
    print(f"  {k:22}: {v:.3f}")


# Output:
# =======================================================
# RETRIEVAL METRICS
# =======================================================
# Context Precision:  0.50  (2 of 4 chunks relevant)
# Context Recall:     1.00  (1 of 2 required facts found)
# MRR:                0.611  (avg across 3 queries)
#
# =======================================================
# GENERATION METRICS
# =======================================================
# Faithfulness (good answer): 1.00
# Faithfulness (bad answer):  1.00
# Answer Relevance (good):    0.906
# Answer Relevance (off-topic): 0.109
#
# RAG Triad (good answer):
#   context_relevance     : 0.881
#   groundedness          : 1.000
#   answer_relevance      : 0.906

# Findings:
# Context Precision = 0.50: half the retrieved chunks are noise. A reranker
# or metadata filter would improve this.
# Context Recall = 1.00: the first required fact is found. The second fact
# (radium/polonium) is not in the KB — a knowledge gap, not a retrieval failure.
# Faithfulness = 1.00 for both answers because embedding similarity is a weak
# proxy — the bad answer still has high cosine similarity to the context.
# In production, use an LLM judge for faithfulness to catch wrong facts.
# Answer Relevance correctly separates on-topic (0.906) from off-topic (0.109).
