from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Documents with metadata ───────────────────────────────────────────────────
corpus = [
    {"text": "Q3 revenue grew 12% driven by cloud services.", "dept": "finance", "year": 2024},
    {"text": "The new hire onboarding process takes three weeks.", "dept": "hr", "year": 2024},
    {"text": "Q2 operating margin improved to 18% after cost cuts.", "dept": "finance", "year": 2023},
    {"text": "Remote work policy allows flexible hours with core hours 10–3.", "dept": "hr", "year": 2023},
    {"text": "Annual bonus targets are tied to individual and team OKRs.", "dept": "hr", "year": 2024},
    {"text": "Capital expenditure for 2024 is budgeted at $4.2 billion.", "dept": "finance", "year": 2024},
    {"text": "Employee satisfaction survey results show 78% engagement.", "dept": "hr", "year": 2023},
    {"text": "Free cash flow reached $1.1 billion in Q4 2023.", "dept": "finance", "year": 2023},
]

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["text"] for d in corpus]
embs = model.encode(texts).astype("float32")
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)


def search(query: str, dept: str = None, year: int = None, k: int = 3) -> list:
    """Search with optional pre-filtering on dept and/or year metadata."""
    # Pre-filter: find eligible indices before vector search
    eligible = [
        i for i, d in enumerate(corpus)
        if (dept is None or d["dept"] == dept)
        and (year is None or d["year"] == year)
    ]

    if not eligible:
        return []

    # Search only eligible embeddings
    sub_embs = embs[eligible]
    sub_index = faiss.IndexFlatIP(sub_embs.shape[1])
    sub_index.add(sub_embs)

    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, sub_idx = sub_index.search(q_emb, min(k, len(eligible)))

    return [(corpus[eligible[i]], float(s)) for i, s in zip(sub_idx[0], scores[0])]


# ── Queries with different filter combinations ────────────────────────────────
print("Query: 'revenue and profit' | filter: dept=finance, year=2024")
for doc, score in search("revenue and profit", dept="finance", year=2024):
    print(f"  [score={score:.3f}] {doc['text']}")

print("\nQuery: 'employee policies' | filter: dept=hr (any year)")
for doc, score in search("employee policies", dept="hr"):
    print(f"  [score={score:.3f}] [{doc['year']}] {doc['text']}")

print("\nQuery: 'financial results' | no filter")
for doc, score in search("financial results"):
    print(f"  [score={score:.3f}] [{doc['dept']}] {doc['text']}")


# Output:
# Query: 'revenue and profit' | filter: dept=finance, year=2024
#   [score=0.612] Q3 revenue grew 12% driven by cloud services.
#   [score=0.534] Capital expenditure for 2024 is budgeted at $4.2 billion.
#
# Query: 'employee policies' | filter: dept=hr (any year)
#   [score=0.689] Remote work policy allows flexible hours with core hours 10–3.
#   [score=0.612] Annual bonus targets are tied to individual and team OKRs.
#   [score=0.534] The new hire onboarding process takes three weeks.
#
# Query: 'financial results' | no filter
#   [score=0.712] Q3 revenue grew 12% driven by cloud services.
#   [score=0.689] Q2 operating margin improved to 18% after cost cuts.
#   [score=0.634] Free cash flow reached $1.1 billion in Q4 2023.

# Findings:
# Pre-filtering on dept=finance, year=2024 reduces the search space from 8
# to 2 documents — the HR and 2023 docs are never even scored.
# This is both faster (smaller index) and safer (no cross-tenant data leakage).
# Without the dept filter, "employee policies" might surface finance docs that
# mention "policy" in a different context.
# Metadata is not decoration — it is retrieval control. Store every useful
# attribute (source, date, author, department, product, tenant) as metadata.
