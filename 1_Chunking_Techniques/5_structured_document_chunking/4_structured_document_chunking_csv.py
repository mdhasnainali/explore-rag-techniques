import csv
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Sample CSV data ───────────────────────────────────────────────────────────
CSV_DATA = """name,department,salary,start_date,status
Alice Chen,Engineering,95000,2021-03-15,active
Bob Kumar,Marketing,72000,2022-07-01,active
Carol White,Engineering,105000,2019-11-20,active
David Park,Marketing,68000,2023-01-10,active
Eve Torres,Engineering,112000,2018-05-30,active
Frank Lee,HR,61000,2022-09-15,inactive
"""

# ── Step 1: Read CSV rows ─────────────────────────────────────────────────────
reader = csv.DictReader(io.StringIO(CSV_DATA))
rows = list(reader)


# ── Step 2: Convert each row to labeled text ──────────────────────────────────
# Include column names so the embedding model understands what each value means.
def row_to_text(row: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in row.items() if value)


texts = [row_to_text(row) for row in rows]

print("Sample indexed text (row 1):")
print(texts[0])
print()

# ── Step 3: Embed and index ───────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ── Step 4: Query ─────────────────────────────────────────────────────────────
queries = [
    "Who are the engineers and what do they earn?",
    "Which employees are inactive?",
]

for query in queries:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 3)

    print(f"Query: '{query}'")
    for score, idx in zip(scores[0], indices[0]):
        print(f"  [score={score:.3f}] {rows[idx]['name']} — {rows[idx]['department']}")
    print()


# Output:
# Sample indexed text (row 1):
# name: Alice Chen
# department: Engineering
# salary: 95000
# start_date: 2021-03-15
# status: active
#
# Query: 'Who are the engineers and what do they earn?'
#   [score=0.408] Alice Chen — Engineering
#   [score=0.384] Carol White — Engineering
#   [score=0.360] Eve Torres — Engineering
#
# Query: 'Which employees are inactive?'
#   [score=0.380] Frank Lee — HR
#   [score=0.351] Alice Chen — Engineering
#   [score=0.310] Carol White — Engineering

# Findings:
# Including column names in the indexed text ("department: Engineering") is
# essential — indexing raw values alone ("Engineering") loses the meaning.
# All three engineers are retrieved for the engineering query because the
# word "Engineering" appears in each row's indexed text.
# The inactive query correctly surfaces Frank Lee (status: inactive) at rank 1.
# For exact-value filtering (e.g., status == "active"), use metadata filters
# alongside semantic search rather than relying on embedding similarity alone.
