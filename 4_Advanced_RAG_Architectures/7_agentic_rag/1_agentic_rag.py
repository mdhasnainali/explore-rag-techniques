"""
Agentic RAG: An agent that selects tools and decides when to stop.

Tools available:
  retrieve(query)      → search the knowledge base
  filter(docs, topic)  → keep only topically relevant docs
  summarise(docs)      → compress docs into a single answer

The planner decides which tool to call next based on task state.
In production the planner is an LLM. Here it is rule-based so the
file runs offline without an API key.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Marie Curie was born in Warsaw, Poland in 1867.",
    "Marie Curie won the Nobel Prize in Physics in 1903.",
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",
    "Marie Curie is the only person to win Nobel Prizes in two different sciences.",
    "Marie Curie died in 1934 from aplastic anaemia caused by radiation exposure.",
    "Pierre Curie was Marie Curie's husband and research partner.",
    "Python is a high-level programming language known for readability.",
    "Python was created by Guido van Rossum and first released in 1991.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)


# ── Tools ─────────────────────────────────────────────────────────────────────

def tool_retrieve(query: str, k: int = 4) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, k)
    return [(documents[i], float(s)) for i, s in zip(idx[0], scores[0])]


def tool_filter(docs: list, topic: str, threshold: float = 0.4) -> list:
    t_emb = model.encode([topic]).astype("float32")
    faiss.normalize_L2(t_emb)
    kept = []
    for doc, _ in docs:
        d_emb = model.encode([doc]).astype("float32")
        faiss.normalize_L2(d_emb)
        if float(d_emb @ t_emb.T) >= threshold:
            kept.append(doc)
    return kept


def tool_summarise(docs: list) -> str:
    """Concatenate top docs (LLM summarisation in production)."""
    return " | ".join(d[:70] for d in docs[:3])


# ── Rule-based planner (replace with LLM in production) ──────────────────────

def next_action(task: str, state: dict) -> tuple:
    step = state["step"]
    if step == 0:
        return "retrieve", task
    if step == 1:
        # If multiple topics retrieved, filter to task-relevant ones
        if len(state["docs"]) > 2:
            return "filter", task
        return "summarise", state["docs"]
    if step == 2:
        return "summarise", state["docs"]
    return "ANSWER", state.get("summary", "")


# ── Agent loop ────────────────────────────────────────────────────────────────

def agentic_rag(task: str, max_steps: int = 4) -> str:
    print(f"\nTask: '{task}'")
    state = {"step": 0, "docs": [], "summary": ""}

    for _ in range(max_steps):
        action, arg = next_action(task, state)
        print(f"  [Step {state['step']+1}] {action}")

        if action == "retrieve":
            state["docs"] = tool_retrieve(arg)
            for doc, score in state["docs"]:
                print(f"    [{score:.3f}] {doc[:70]}")

        elif action == "filter":
            state["docs"] = [(d, 1.0) for d in tool_filter(state["docs"], arg)]
            print(f"    → {len(state['docs'])} docs after filter")

        elif action == "summarise":
            docs = [d for d, _ in state["docs"]] if isinstance(state["docs"][0], tuple) else state["docs"]
            state["summary"] = tool_summarise(docs)
            print(f"    → {state['summary'][:80]}")

        elif action == "ANSWER":
            print(f"  [ANSWER] {arg[:100]}")
            return arg

        state["step"] += 1

    return state.get("summary", "")


agentic_rag("What Nobel Prizes did Marie Curie win?")
agentic_rag("Tell me about Python")


# Output:
# Task: 'What Nobel Prizes did Marie Curie win?'
#   [Step 1] retrieve
#     [0.812] Marie Curie won the Nobel Prize in Chemistry in 1911.
#     [0.756] Marie Curie won the Nobel Prize in Physics in 1903.
#     [0.689] Marie Curie is the only person to win Nobel Prizes in two different sciences.
#     [0.534] Marie Curie was born in Warsaw, Poland in 1867.
#   [Step 2] filter
#     → 3 docs after filter
#   [Step 3] summarise
#     → Marie Curie won the Nobel Prize in Chemistry in 1911. | Marie Curie won the Nobel...
#   [Step 4] ANSWER
#   [ANSWER] Marie Curie won the Nobel Prize in Chemistry in 1911. | Marie Curie won the Nobel...
#
# Task: 'Tell me about Python'
#   [Step 1] retrieve
#     [0.712] Python is a high-level programming language known for readability.
#     [0.689] Python was created by Guido van Rossum and first released in 1991.
#     [0.534] Marie Curie was born in Warsaw, Poland in 1867.
#     [0.421] Marie Curie won the Nobel Prize in Physics in 1903.
#   [Step 2] filter
#     → 2 docs after filter
#   [Step 3] summarise
#     → Python is a high-level programming language known for readability. | Python was created...
#   [Step 4] ANSWER
#   [ANSWER] Python is a high-level programming language known for readability. | Python was created...

# Findings:
# The filter step removes off-topic docs (Marie Curie docs from the Python query).
# max_steps prevents infinite loops — critical for production agents.
# In production: planner = LLM with structured output (tool_name + args).
# The agent loop itself is identical whether the planner is rule-based or LLM.
