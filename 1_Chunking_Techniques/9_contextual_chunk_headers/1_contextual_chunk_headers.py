from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Sample document with clear section structure ──────────────────────────────
DOCUMENT_TITLE = "Nike 2023 Annual Report"

document = """
Executive Summary
Nike delivered strong results in fiscal 2023, with revenues of $51.2 billion,
up 10 percent on a reported basis. Direct sales grew 14 percent, reflecting
continued momentum in our digital and owned-store channels.

Risk Factors
Supply chain disruptions remain a key risk. Geopolitical tensions and port
congestion have increased lead times. We continue to diversify our supplier base
to reduce concentration risk in any single region.

Climate Change Impact
Nike is committed to reducing its carbon footprint by 70 percent by 2025.
Climate-related risks include increased raw material costs and disruption to
manufacturing facilities in climate-vulnerable regions.

Financial Performance
Gross margin was 43.5 percent, down 250 basis points due to higher product
costs and elevated freight expenses. Operating income was $5.5 billion.
Diluted earnings per share were $3.23.
"""

# ── Split into chunks ─────────────────────────────────────────────────────────
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_text(document)

# ── Add contextual header to each chunk ──────────────────────────────────────
# The header prepends the document title so every chunk is self-contained.
# In production, you would also include the section heading (parsed from the doc).
def add_header(chunk: str, title: str) -> str:
    return f"Document: {title}\n\n{chunk}"

chunks_with_headers = [add_header(c, DOCUMENT_TITLE) for c in chunks]

# ── Embed both versions ───────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(texts):
    embs = model.encode(texts).astype("float32")
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

index_plain   = build_index(chunks)
index_headers = build_index(chunks_with_headers)

# ── Compare retrieval scores on a query that needs document context ───────────
query = "Nike climate change impact"
q_emb = model.encode([query]).astype("float32")
faiss.normalize_L2(q_emb)

scores_plain,   idx_plain   = index_plain.search(q_emb, 1)
scores_headers, idx_headers = index_headers.search(q_emb, 1)

print(f"Query: '{query}'\n")
print("Without headers:")
print(f"  Score: {scores_plain[0][0]:.3f}")
print(f"  Chunk: {chunks[idx_plain[0][0]][:120]}\n")

print("With headers:")
print(f"  Score: {scores_headers[0][0]:.3f}")
print(f"  Chunk: {chunks_with_headers[idx_headers[0][0]][:160]}\n")

# ── Show all chunks with headers ──────────────────────────────────────────────
print("All chunks with contextual headers:")
for i, chunk in enumerate(chunks_with_headers):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk[:200])


# Output:
# Query: 'Nike climate change impact'
#
# Without headers:
#   Score: 0.837
#   Chunk: Climate Change Impact
# Nike is committed to reducing its carbon footprint by 70 percent by 2025.
# Climate-related risks in...
#
# With headers:
#   Score: 0.834
#   Chunk: Document: Nike 2023 Annual Report
#
# Climate Change Impact
# Nike is committed to reducing its carbon footprint by 70 percent by 2025.
# Climate-related risks include...
#
# All chunks with contextual headers:
# --- Chunk 1 ---
# Document: Nike 2023 Annual Report
# Executive Summary
# Nike delivered strong results in fiscal 2023...
# ...

# Findings:
# In this example the score difference is tiny (0.837 vs 0.834) because the
# chunk already contains "Nike" explicitly. Headers have the biggest impact
# when chunks use implicit references ("the company", "they", "it") without
# naming the subject — a common pattern in long corporate documents.
# The reference notebook (contextual_chunk_headers.ipynb) shows a real case
# where a chunk about "climate-related risks" with no company name scores 0.1
# without a header and 0.92 with the document title prepended.
# Headers are cheap: no LLM call needed if the document title is known.
# For maximum benefit, include the section heading in the header, not just
# the document title (e.g., "Nike 2023 Annual Report > Climate Change Impact").
# The KITE benchmark shows CCH improves average RAG score from 4.72 to 6.04
# (+27.9%) across AI papers, financial 10-Ks, handbooks, and legal opinions.
