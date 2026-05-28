from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Marie Curie was born in Warsaw, Poland in 1867.",
    "Marie Curie won the Nobel Prize in Physics in 1903.",
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",
    "Marie Curie was the first woman to win a Nobel Prize.",
    "Marie Curie is the only person to win Nobel Prizes in two different sciences.",
    "Pierre Curie was Marie Curie's husband and research partner.",
    "The Nobel Prize in Physics 1903 was shared by Marie Curie, Pierre Curie, and Henri Becquerel.",
    "Marie Curie died in 1934 from aplastic anaemia caused by radiation exposure.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)


def retrieve(query: str, k: int = 2, exclude: set = None) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, len(documents))
    results = []
    for score, i in zip(scores[0], idx[0]):
        if exclude and i in exclude:
            continue
        results.append((i, float(score), documents[i]))
        if len(results) == k:
            break
    return results


def detect_gap(context: list, question: str):
    """
    Rule-based gap detector. In production, replace with an LLM call:
    'Given this context and question, what information is still missing?'
    """
    combined = " ".join(d for _, _, d in context).lower()
    if "unique" in question.lower() and "only person" not in combined:
        return True, "Marie Curie unique achievement two sciences"
    if "die" in question.lower() and "died" not in combined:
        return True, "Marie Curie cause of death"
    return False, ""


question = "What made Marie Curie's Nobel Prize achievements unique and how did she die?"

print(f"Question: '{question}'\n")

all_context = []
seen_idx = set()
MAX_ITERATIONS = 3
follow_up = question

for iteration in range(MAX_ITERATIONS):
    results = retrieve(follow_up, k=2, exclude=seen_idx)
    for idx, score, doc in results:
        seen_idx.add(idx)
        all_context.append((idx, score, doc))

    print(f"Iteration {iteration + 1} — query: '{follow_up}'")
    for _, score, doc in results:
        print(f"  [{score:.3f}] {doc}")

    gap, follow_up = detect_gap(all_context, question)
    if not gap:
        print("  → No gap. Stopping.\n")
        break
    print(f"  → Gap detected. Follow-up: '{follow_up}'\n")

print("Final context for LLM:")
for _, _, doc in all_context:
    print(f"  - {doc}")


# Output:
# Question: 'What made Marie Curie's Nobel Prize achievements unique and how did she die?'
#
# Iteration 1 — query: 'What made Marie Curie's Nobel Prize achievements unique and how did she die?'
#   [0.712] Marie Curie won the Nobel Prize in Physics in 1903.
#   [0.689] Marie Curie won the Nobel Prize in Chemistry in 1911.
#   → Gap detected. Follow-up: 'Marie Curie unique achievement two sciences'
#
# Iteration 2 — query: 'Marie Curie unique achievement two sciences'
#   [0.823] Marie Curie is the only person to win Nobel Prizes in two different sciences.
#   [0.756] Marie Curie was the first woman to win a Nobel Prize.
#   → Gap detected. Follow-up: 'Marie Curie cause of death'
#
# Iteration 3 — query: 'Marie Curie cause of death'
#   [0.891] Marie Curie died in 1934 from aplastic anaemia caused by radiation exposure.
#   [0.634] Marie Curie was born in Warsaw, Poland in 1867.
#   → No gap. Stopping.
#
# Final context for LLM:
#   - Marie Curie won the Nobel Prize in Physics in 1903.
#   - Marie Curie won the Nobel Prize in Chemistry in 1911.
#   - Marie Curie is the only person to win Nobel Prizes in two different sciences.
#   - Marie Curie was the first woman to win a Nobel Prize.
#   - Marie Curie died in 1934 from aplastic anaemia caused by radiation exposure.
#   - Marie Curie was born in Warsaw, Poland in 1867.

# Findings:
# A single retrieval pass returns only the two prize-year docs — not enough
# to answer the full question. Three iterations surface 6 documents covering
# all aspects.
# The gap detector is the critical component. Rule-based (as here) is brittle.
# In production, use an LLM: "Given this context and question, what is missing?"
# Always cap iterations (MAX_ITERATIONS=3) to prevent infinite loops.
# Each follow-up query should be specific — not just "tell me more".
