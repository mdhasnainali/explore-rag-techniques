"""
Microsoft GraphRAG: Community-based retrieval over a knowledge graph.

Pipeline:
  1. Extract entities and relationships from documents
  2. Build a graph and detect communities (clusters of related entities)
  3. Summarise each community
  4. At query time: retrieve relevant community summaries + local evidence

This enables both LOCAL queries (specific facts) and GLOBAL queries
(broad synthesis across the whole corpus).

This demo uses a pre-built graph and rule-based community detection
so it runs offline without an API key.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import defaultdict

# ── Pre-built entity graph ────────────────────────────────────────────────────
entities = {
    "Marie Curie":    "physicist, chemist, Nobel laureate",
    "Pierre Curie":   "physicist, Marie Curie's husband",
    "Henri Becquerel":"physicist, discovered radioactivity",
    "Radium":         "radioactive element discovered by Marie Curie",
    "Polonium":       "radioactive element discovered by Marie Curie",
    "Radioactivity":  "phenomenon studied by the Curies and Becquerel",
    "Python":         "programming language created by Guido van Rossum",
    "Guido van Rossum":"creator of Python",
    "PyPI":           "Python package index",
}

edges = [
    ("Marie Curie",    "discovered",   "Radium"),
    ("Marie Curie",    "discovered",   "Polonium"),
    ("Marie Curie",    "studied",      "Radioactivity"),
    ("Pierre Curie",   "studied",      "Radioactivity"),
    ("Henri Becquerel","discovered",   "Radioactivity"),
    ("Marie Curie",    "married_to",   "Pierre Curie"),
    ("Python",         "created_by",   "Guido van Rossum"),
    ("Python",         "hosted_on",    "PyPI"),
]

# ── Community detection (rule-based: connected components) ────────────────────
graph = defaultdict(set)
for s, _, o in edges:
    graph[s].add(o)
    graph[o].add(s)

def find_communities(graph: dict) -> dict:
    visited = set()
    communities = {}
    cid = 0
    for node in graph:
        if node not in visited:
            # BFS
            queue = [node]
            community = []
            while queue:
                n = queue.pop()
                if n not in visited:
                    visited.add(n)
                    community.append(n)
                    queue.extend(graph[n] - visited)
            for n in community:
                communities[n] = cid
            cid += 1
    return communities

node_to_community = find_communities(graph)

# ── Community summaries (LLM-generated in production) ────────────────────────
community_summaries = {
    0: "The Curie-Becquerel community covers radioactivity research. Marie Curie and Pierre Curie discovered radium and polonium. Henri Becquerel discovered radioactivity. All three shared the 1903 Nobel Prize in Physics.",
    1: "The Python community covers the Python programming language. Python was created by Guido van Rossum and packages are distributed via PyPI.",
}

# ── Index community summaries for semantic search ────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
summary_texts = [community_summaries[i] for i in sorted(community_summaries)]
summary_embs = model.encode(summary_texts).astype("float32")
faiss.normalize_L2(summary_embs)
index = faiss.IndexFlatIP(summary_embs.shape[1])
index.add(summary_embs)


def microsoft_graphrag(query: str, k_communities: int = 1) -> dict:
    """Retrieve relevant community summaries + local entity evidence."""
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, k_communities)

    matched_communities = [sorted(community_summaries.keys())[i] for i in idx[0]]
    summaries = [community_summaries[c] for c in matched_communities]

    # Local evidence: entities in matched communities
    local_entities = [
        (e, desc) for e, desc in entities.items()
        if node_to_community.get(e) in matched_communities
    ]

    return {"summaries": summaries, "local_entities": local_entities, "scores": scores[0].tolist()}


queries = [
    "What did Marie Curie discover?",   # global: needs community summary
    "Tell me about Python packages",    # different community
]

for query in queries:
    result = microsoft_graphrag(query)
    print(f"Query: '{query}'")
    print(f"  Community summary: {result['summaries'][0][:100]}...")
    print(f"  Local entities ({len(result['local_entities'])}):")
    for e, d in result["local_entities"][:3]:
        print(f"    - {e}: {d}")
    print()


# Output:
# Query: 'What did Marie Curie discover?'
#   Community summary: The Curie-Becquerel community covers radioactivity research...
#   Local entities (6):
#     - Marie Curie: physicist, chemist, Nobel laureate
#     - Pierre Curie: physicist, Marie Curie's husband
#     - Henri Becquerel: physicist, discovered radioactivity
#
# Query: 'Tell me about Python packages'
#   Community summary: The Python community covers the Python programming language...
#   Local entities (3):
#     - Python: programming language created by Guido van Rossum
#     - Guido van Rossum: creator of Python
#     - PyPI: Python package index

# Findings:
# Community summaries answer GLOBAL queries ("What is this corpus about?")
# that flat chunk retrieval cannot answer — no single chunk covers the whole
# Curie-Becquerel research community.
# Local entity evidence answers SPECIFIC queries within the community.
# Microsoft GraphRAG uses Leiden algorithm for community detection and an LLM
# to generate summaries. This demo uses BFS components and manual summaries.
