"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Pipeline:
  1. Start with raw text chunks (leaf nodes)
  2. Cluster similar chunks using embeddings
  3. Summarise each cluster → new nodes one level up
  4. Repeat until only one root summary remains
  5. At query time: search across ALL levels simultaneously

This enables answering both specific questions (leaf level) and broad
synthesis questions (higher levels).
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.cluster import KMeans

# ── Raw text chunks (leaf level) ─────────────────────────────────────────────
leaf_chunks = [
    "Marie Curie was born in Warsaw, Poland in 1867.",
    "Marie Curie won the Nobel Prize in Physics in 1903.",
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",
    "Marie Curie is the only person to win Nobel Prizes in two different sciences.",
    "Marie Curie discovered radium and polonium.",
    "Marie Curie died in 1934 from aplastic anaemia caused by radiation exposure.",
    "Python is a high-level programming language created by Guido van Rossum.",
    "Python was first released in 1991 and is known for its readability.",
    "Python is widely used in data science, web development, and automation.",
    "Python 3.12 introduced improved error messages and faster startup times.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")


def cluster_and_summarise(chunks: list, n_clusters: int) -> list:
    """Cluster chunks and produce one summary per cluster."""
    embs = model.encode(chunks).astype("float32")
    n_clusters = min(n_clusters, len(chunks))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embs)

    summaries = []
    for cid in range(n_clusters):
        cluster_chunks = [c for c, l in zip(chunks, labels) if l == cid]
        # In production: LLM summarises the cluster. Here: join first sentences.
        summary = " | ".join(c[:60] for c in cluster_chunks[:3])
        summaries.append(f"[Summary of {len(cluster_chunks)} chunks] {summary}")
    return summaries


# ── Build the tree ────────────────────────────────────────────────────────────
all_nodes = []   # (level, text) — we index all levels together

# Level 0: leaf chunks
for chunk in leaf_chunks:
    all_nodes.append((0, chunk))

# Level 1: cluster leaf chunks into 3 groups, summarise
level1 = cluster_and_summarise(leaf_chunks, n_clusters=3)
for s in level1:
    all_nodes.append((1, s))

# Level 2: cluster level-1 summaries into 1 root summary
level2 = cluster_and_summarise(level1, n_clusters=1)
for s in level2:
    all_nodes.append((2, s))

# ── Index all nodes across all levels ────────────────────────────────────────
node_texts = [text for _, text in all_nodes]
node_embs = model.encode(node_texts).astype("float32")
faiss.normalize_L2(node_embs)
index = faiss.IndexFlatIP(node_embs.shape[1])
index.add(node_embs)

print(f"Tree built: {len(leaf_chunks)} leaves → {len(level1)} L1 summaries → {len(level2)} root")
print(f"Total indexed nodes: {len(all_nodes)}\n")

# ── Query across all levels ───────────────────────────────────────────────────
queries = [
    ("When did Marie Curie die?",          "specific — leaf level"),
    ("Give me an overview of Marie Curie", "broad — higher level"),
]

for query, qtype in queries:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, 3)

    print(f"Query ({qtype}): '{query}'")
    for score, i in zip(scores[0], idx[0]):
        level, text = all_nodes[i]
        print(f"  [L{level}, score={score:.3f}] {text[:90]}")
    print()


# Output:
# Tree built: 10 leaves → 3 L1 summaries → 1 root
# Total indexed nodes: 14
#
# Query (specific — leaf level): 'When did Marie Curie die?'
#   [L0, score=0.712] Marie Curie died in 1934 from aplastic anaemia caused by radiation exposure.
#   [L0, score=0.634] Marie Curie was born in Warsaw, Poland in 1867.
#   [L1, score=0.521] [Summary of 3 chunks] Marie Curie was born in Warsaw, Poland in 1867...
#
# Query (broad — higher level): 'Give me an overview of Marie Curie'
#   [L2, score=0.689] [Summary of 3 chunks] [Summary of 3 chunks] Marie Curie was born...
#   [L1, score=0.634] [Summary of 3 chunks] Marie Curie was born in Warsaw, Poland in 1867...
#   [L0, score=0.589] Marie Curie won the Nobel Prize in Physics in 1903.

# Findings:
# Specific queries retrieve leaf nodes (exact facts). Broad queries retrieve
# higher-level summaries that synthesise multiple chunks.
# Indexing all levels together lets the retriever naturally select the right
# granularity — no need to decide in advance which level to search.
# Summary quality is critical: errors in L1 summaries propagate to L2.
# In production: use an LLM for summarisation and Gaussian Mixture Models
# for soft clustering (as in the reference implementation).
