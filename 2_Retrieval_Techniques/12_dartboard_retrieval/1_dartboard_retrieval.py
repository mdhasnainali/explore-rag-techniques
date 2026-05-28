from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Python is a high-level programming language known for readability.",
    "Python was created by Guido van Rossum and released in 1991.",
    "Python uses indentation to define code blocks instead of braces.",
    "Python is widely used in data science, web development, and automation.",
    "Python's package manager pip installs libraries from PyPI.",
    "The GIL limits true multi-threading in CPython.",
    "Python 3.12 introduced improved error messages and faster startup.",
    "Neural networks are inspired by the human brain structure.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

query = "Tell me about Python"


def maximal_marginal_relevance(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidate_idx: list,
    k: int = 4,
    lambda_: float = 0.5,
) -> list:
    """
    MMR selects documents that are relevant to the query but dissimilar to
    already-selected documents.
    score = lambda * relevance - (1 - lambda) * max_similarity_to_selected
    lambda_=1.0 → pure relevance (no diversity)
    lambda_=0.0 → pure diversity (no relevance)
    """
    selected = []
    remaining = list(range(len(candidate_idx)))

    for _ in range(k):
        if not remaining:
            break

        best_score = -np.inf
        best_pos = None

        for pos in remaining:
            relevance = float(candidate_embs[pos] @ query_emb.T)

            if selected:
                selected_embs = candidate_embs[selected]
                sim_to_selected = float(np.max(candidate_embs[pos] @ selected_embs.T))
            else:
                sim_to_selected = 0.0

            score = lambda_ * relevance - (1 - lambda_) * sim_to_selected
            if score > best_score:
                best_score = score
                best_pos = pos

        selected.append(best_pos)
        remaining.remove(best_pos)

    return [candidate_idx[i] for i in selected]


q_emb = model.encode([query]).astype("float32")
faiss.normalize_L2(q_emb)

# Retrieve top-7 candidates first
_, top_idx = index.search(q_emb, 7)
candidate_idx = top_idx[0].tolist()
candidate_embs = doc_embs[candidate_idx]

# Standard top-k (pure relevance)
standard = candidate_idx[:4]

# MMR (relevance + diversity)
diverse = maximal_marginal_relevance(q_emb[0], candidate_embs, candidate_idx, k=4)

print(f"Query: '{query}'\n")
print("Standard top-4 (pure relevance):")
for i in standard:
    print(f"  - {documents[i]}")

print("\nMMR top-4 (relevance + diversity, lambda=0.5):")
for i in diverse:
    print(f"  - {documents[i]}")


# Output:
# Query: 'Tell me about Python'
#
# Standard top-4 (pure relevance):
#   - Python is a high-level programming language known for readability.
#   - Python is widely used in data science, web development, and automation.
#   - Python was created by Guido van Rossum and released in 1991.
#   - Python uses indentation to define code blocks instead of braces.
#
# MMR top-4 (relevance + diversity, lambda=0.5):
#   - Python is a high-level programming language known for readability.
#   - The GIL limits true multi-threading in CPython.
#   - Python was created by Guido van Rossum and released in 1991.
#   - Neural networks are inspired by the human brain structure.

# Findings:
# Standard top-4 returns 4 very similar Python-overview sentences — they all
# say roughly the same thing with different words, wasting context tokens.
# MMR selects the most relevant doc first, then penalises docs similar to
# already-selected ones. The GIL doc (a limitation) and the neural networks
# doc (a different topic) are selected to maximise coverage.
# lambda=0.5 balances relevance and diversity. For precise factual queries,
# use lambda closer to 1.0. For exploratory queries, use lambda closer to 0.5.
