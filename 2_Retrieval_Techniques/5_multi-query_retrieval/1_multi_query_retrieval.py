from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Python supports object-oriented, functional, and procedural programming.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python's package manager pip installs libraries from PyPI.",
    "Python uses indentation to define code blocks instead of braces.",
    "Python is widely used in data science, web development, and automation.",
    "The GIL (Global Interpreter Lock) limits true multi-threading in CPython.",
    "Python 3.12 introduced improved error messages and faster startup times.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embs = model.encode(documents).astype("float32")
faiss.normalize_L2(doc_embs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

original_query = "Tell me about Python programming language"

# ── Query variants (in production, generate these with an LLM) ───────────────
query_variants = [
    "Tell me about Python programming language",    # original
    "Who created Python and when was it released?", # decomposed — history
    "What is Python used for?",                     # decomposed — use cases
    "What are the limitations of Python?",          # decomposed — weaknesses
]


def retrieve(query: str, k: int = 2) -> list:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, idx = index.search(q_emb, k)
    return idx[0].tolist()


# ── Retrieve for each variant, merge with deduplication ──────────────────────
seen = set()
merged = []
for variant in query_variants:
    for idx in retrieve(variant, k=2):
        if idx not in seen:
            seen.add(idx)
            merged.append(idx)

print(f"Original query: '{original_query}'\n")
print("Query variants:")
for v in query_variants:
    print(f"  - {v}")

print(f"\nMerged results ({len(merged)} unique docs):")
for rank, idx in enumerate(merged):
    print(f"  {rank+1}. {documents[idx]}")


# Output:
# Original query: 'Tell me about Python programming language'
#
# Query variants:
#   - Tell me about Python programming language
#   - Who created Python and when was it released?
#   - What is Python used for?
#   - What are the limitations of Python?
#
# Merged results (7 unique docs):
#   1. Python supports object-oriented, functional, and procedural programming.
#   2. Python is widely used in data science, web development, and automation.
#   3. Python was created by Guido van Rossum and first released in 1991.
#   4. Python uses indentation to define code blocks instead of braces.
#   5. Python's package manager pip installs libraries from PyPI.
#   6. The GIL (Global Interpreter Lock) limits true multi-threading in CPython.
#   7. Python 3.12 introduced improved error messages and faster startup times.

# Findings:
# The original query alone retrieves 2 docs. Four variants together surface
# all 7 — a significant recall improvement.
# The "limitations" variant specifically surfaces the GIL document, which
# the original broad query would not retrieve.
# Deduplication is essential — without it, the same document appears multiple
# times and wastes LLM context window tokens.
# In production, use an LLM to generate variants automatically with a prompt
# asking for: rewrite, step-back (broader), and decomposition (narrower).
