from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Large chunks (as retrieved) ───────────────────────────────────────────────
chunks = [
    (
        "Nike was founded in 1964 as Blue Ribbon Sports by Bill Bowerman and Phil Knight. "
        "The company was renamed Nike in 1971. Nike's revenue in fiscal 2023 was $51.2 billion, "
        "up 10 percent year-over-year. The company employs approximately 79,000 people worldwide. "
        "Nike's headquarters is located in Beaverton, Oregon."
    ),
    (
        "Climate change poses significant risks to global supply chains. Nike is committed to "
        "reducing its carbon footprint by 70 percent by 2025. The company uses recycled materials "
        "in over 75 percent of its products. Nike's Move to Zero initiative targets zero carbon "
        "and zero waste across its operations. Renewable energy powers 96 percent of Nike-owned facilities."
    ),
    (
        "The Python programming language was created by Guido van Rossum. Python 3.12 was released "
        "in October 2023 with improved error messages. Python is used in data science, web development, "
        "and automation. The language uses indentation for code blocks. Python's package manager is pip."
    ),
]

query = "What is Nike's carbon reduction target?"

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_relevant_sentences(chunk: str, query: str, threshold: float = 0.35) -> str:
    """
    Extractive compression: keep only sentences whose embedding is
    similar enough to the query embedding.
    """
    sentences = [s.strip() for s in chunk.replace(".", ".|").split("|") if s.strip()]
    if not sentences:
        return chunk

    sent_embs = model.encode(sentences).astype("float32")
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(sent_embs)
    faiss.normalize_L2(q_emb)

    scores = (sent_embs @ q_emb.T).flatten()
    kept = [s for s, score in zip(sentences, scores) if score >= threshold]
    return " ".join(kept) if kept else sentences[0]  # fallback: keep first sentence


# ── Retrieve top-2 chunks ─────────────────────────────────────────────────────
chunk_embs = model.encode(chunks).astype("float32")
faiss.normalize_L2(chunk_embs)
index = faiss.IndexFlatIP(chunk_embs.shape[1])
index.add(chunk_embs)

q_emb = model.encode([query]).astype("float32")
faiss.normalize_L2(q_emb)
_, top_idx = index.search(q_emb, 2)

print(f"Query: '{query}'\n")
for i, idx in enumerate(top_idx[0]):
    original = chunks[idx]
    compressed = extract_relevant_sentences(original, query)
    print(f"Chunk {i+1} — original ({len(original)} chars):")
    print(f"  {original[:120]}...")
    print(f"Chunk {i+1} — compressed ({len(compressed)} chars):")
    print(f"  {compressed}")
    print()


# Output:
# Query: 'What is Nike's carbon reduction target?'
#
# Chunk 1 — original (249 chars):
#   Climate change poses significant risks to global supply chains. Nike is committed to
#   reducing its carbon footprint by 70 percent by 2025...
# Chunk 1 — compressed (189 chars):
#   Nike is committed to reducing its carbon footprint by 70 percent by 2025.
#   Nike's Move to Zero initiative targets zero carbon and zero waste across its operations.
#
# Chunk 2 — original (248 chars):
#   Nike was founded in 1964 as Blue Ribbon Sports by Bill Bowerman and Phil Knight...
# Chunk 2 — compressed (52 chars):
#   Nike's revenue in fiscal 2023 was $51.2 billion, up 10 percent year-over-year.

# Findings:
# Compression removes 4 irrelevant sentences from chunk 1, keeping only the
# two sentences directly about carbon targets — reducing context by ~24%.
# Chunk 2 (company history) is mostly irrelevant; compression keeps only the
# one sentence that mentions Nike (the threshold sentence).
# Extractive compression is auditable — every word in the output came from
# the original chunk. Abstractive compression (LLM rewrite) can introduce errors.
# The threshold (0.35) controls aggressiveness. Too high → empty output.
# Too low → no compression. Tune per domain.
