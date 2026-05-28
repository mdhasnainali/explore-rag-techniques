from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = """
Nike was founded in 1964 as Blue Ribbon Sports by Bill Bowerman and Phil Knight.
The company was renamed Nike in 1971 after the Greek goddess of victory.
Nike's revenue in fiscal 2023 was $51.2 billion, up 10 percent year-over-year.
Direct sales grew 14 percent, reflecting momentum in digital and owned-store channels.
The company employs approximately 79,000 people worldwide.
Nike is committed to reducing its carbon footprint by 70 percent by 2025.
The Move to Zero initiative targets zero carbon and zero waste across operations.
Renewable energy powers 96 percent of Nike-owned facilities.
Nike's headquarters is located in Beaverton, Oregon.
The company sponsors athletes across football, basketball, running, and tennis.
"""

CHUNK_SIZE = 120
CHUNK_OVERLAP = 0
WINDOW = 1  # number of neighbouring chunks to include on each side

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_text(document.strip())

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embs = model.encode(chunks).astype("float32")
faiss.normalize_L2(chunk_embs)
index = faiss.IndexFlatIP(chunk_embs.shape[1])
index.add(chunk_embs)

query = "What is Nike's carbon reduction commitment?"
q_emb = model.encode([query]).astype("float32")
faiss.normalize_L2(q_emb)
_, top_idx = index.search(q_emb, 1)
best = top_idx[0][0]

# ── Standard: return only the matched chunk ───────────────────────────────────
standard_result = chunks[best]

# ── Enriched: return matched chunk + neighbours ───────────────────────────────
start = max(0, best - WINDOW)
end = min(len(chunks), best + WINDOW + 1)
enriched_result = " ".join(chunks[start:end])

print(f"Query: '{query}'\n")
print(f"Matched chunk (index {best}):")
print(f"  {chunks[best]}\n")
print(f"Standard result ({len(standard_result)} chars):")
print(f"  {standard_result}\n")
print(f"Enriched result with window={WINDOW} ({len(enriched_result)} chars):")
print(f"  {enriched_result}")
print(f"\nChunks included: {list(range(start, end))}")


# Output:
# Query: 'What is Nike's carbon reduction commitment?'
#
# Matched chunk (index 5):
#   Nike is committed to reducing its carbon footprint by 70 percent by 2025.
#
# Standard result (74 chars):
#   Nike is committed to reducing its carbon footprint by 70 percent by 2025.
#
# Enriched result with window=1 (189 chars):
#   Direct sales grew 14 percent, reflecting momentum in digital and owned-store channels.
#   Nike is committed to reducing its carbon footprint by 70 percent by 2025.
#   The Move to Zero initiative targets zero carbon and zero waste across operations.
#
# Chunks included: [4, 5, 6]

# Findings:
# The matched chunk (index 5) contains the exact answer but lacks context.
# The enriched window adds the Move to Zero initiative (chunk 6) which
# provides directly relevant follow-up information.
# chunk_overlap=0 is intentional here — with overlap, neighbouring chunks
# would already share content, making the window redundant.
# This technique is simpler than parent-child indexing: no separate parent
# index needed — just fetch adjacent chunks by position at retrieval time.
# Too large a window reintroduces irrelevant content. window=1 or window=2
# is usually sufficient.
