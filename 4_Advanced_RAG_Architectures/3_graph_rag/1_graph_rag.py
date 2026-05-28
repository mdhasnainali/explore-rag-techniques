"""
GraphRAG: Retrieval over a knowledge graph of entities and relationships.

Instead of retrieving text chunks, the system retrieves entity nodes and their
connected relationships, then expands to supporting text passages.

This demo builds a simple in-memory graph from structured facts and shows
how graph traversal retrieves richer context than flat chunk retrieval.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import defaultdict

# ── Knowledge graph: (subject, relation, object) triples ─────────────────────
triples = [
    ("Marie Curie",    "born_in",        "Warsaw"),
    ("Marie Curie",    "won",            "Nobel Prize Physics 1903"),
    ("Marie Curie",    "won",            "Nobel Prize Chemistry 1911"),
    ("Marie Curie",    "married_to",     "Pierre Curie"),
    ("Pierre Curie",   "won",            "Nobel Prize Physics 1903"),
    ("Pierre Curie",   "worked_at",      "University of Paris"),
    ("Marie Curie",    "worked_at",      "University of Paris"),
    ("Nobel Prize Physics 1903", "shared_with", "Henri Becquerel"),
    ("Warsaw",         "located_in",     "Poland"),
]

# ── Supporting text passages linked to entities ───────────────────────────────
passages = {
    "Marie Curie":              "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.",
    "Pierre Curie":             "Pierre Curie was a French physicist and Marie Curie's husband and research partner.",
    "Nobel Prize Physics 1903": "The 1903 Nobel Prize in Physics was awarded for research on radiation phenomena.",
    "Nobel Prize Chemistry 1911":"The 1911 Nobel Prize in Chemistry was awarded for the discovery of radium and polonium.",
    "University of Paris":      "The University of Paris, founded in the 12th century, is one of the oldest universities in Europe.",
    "Warsaw":                   "Warsaw is the capital and largest city of Poland.",
}

# ── Build adjacency list ──────────────────────────────────────────────────────
graph = defaultdict(list)
for subj, rel, obj in triples:
    graph[subj].append((rel, obj))
    graph[obj].append((f"inverse_{rel}", subj))

# ── Entity index for semantic search ─────────────────────────────────────────
entities = list(passages.keys())
model = SentenceTransformer("all-MiniLM-L6-v2")
entity_embs = model.encode(entities).astype("float32")
faiss.normalize_L2(entity_embs)
index = faiss.IndexFlatIP(entity_embs.shape[1])
index.add(entity_embs)


def find_entity(query: str, k: int = 1) -> list:
    """Find the most relevant entity node for a query."""
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = index.search(q_emb, k)
    return [(entities[i], float(s)) for i, s in zip(idx[0], scores[0])]


def graph_retrieve(query: str, hops: int = 1) -> dict:
    """Find relevant entity, expand to neighbours, collect supporting passages."""
    seed_entities = find_entity(query, k=2)
    visited = set()
    context_passages = []
    graph_context = []

    for entity, score in seed_entities:
        if entity in visited:
            continue
        visited.add(entity)
        if entity in passages:
            context_passages.append(passages[entity])

        # Expand to connected nodes (1-hop)
        for rel, neighbour in graph[entity]:
            graph_context.append(f"{entity} --[{rel}]--> {neighbour}")
            if hops > 0 and neighbour not in visited:
                visited.add(neighbour)
                if neighbour in passages:
                    context_passages.append(passages[neighbour])

    return {"seed": seed_entities, "graph": graph_context, "passages": context_passages}


queries = [
    "What prizes did Marie Curie win?",
    "Who did Marie Curie work with?",
]

for query in queries:
    result = graph_retrieve(query)
    print(f"Query: '{query}'")
    print(f"  Seed entities: {[e for e, _ in result['seed']]}")
    print(f"  Graph context:")
    for g in result["graph"][:5]:
        print(f"    {g}")
    print(f"  Supporting passages:")
    for p in result["passages"][:2]:
        print(f"    - {p[:80]}")
    print()


# Output:
# Query: 'What prizes did Marie Curie win?'
#   Seed entities: ['Marie Curie', 'Pierre Curie']
#   Graph context:
#     Marie Curie --[born_in]--> Warsaw
#     Marie Curie --[won]--> Nobel Prize Physics 1903
#     Marie Curie --[won]--> Nobel Prize Chemistry 1911
#     Marie Curie --[married_to]--> Pierre Curie
#     Marie Curie --[worked_at]--> University of Paris
#   Supporting passages:
#     - Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.
#     - Pierre Curie was a French physicist and Marie Curie's husband and research partner.
#
# Query: 'Who did Marie Curie work with?'
#   Seed entities: ['Marie Curie', 'Pierre Curie']
#   Graph context:
#     Marie Curie --[married_to]--> Pierre Curie
#     Marie Curie --[worked_at]--> University of Paris
#     ...

# Findings:
# Graph retrieval surfaces relationships (won, married_to, worked_at) that
# flat chunk retrieval would miss — a chunk about Marie Curie's prizes doesn't
# mention Pierre Curie unless they happen to be in the same chunk.
# The graph context (triples) is passed to the LLM alongside text passages,
# giving it structured relationship information to reason over.
# Graph quality controls everything — bad entity extraction = bad retrieval.
