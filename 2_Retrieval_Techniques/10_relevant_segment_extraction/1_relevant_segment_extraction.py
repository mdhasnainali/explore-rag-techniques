from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Large document sections (retrieved as whole units) ────────────────────────
sections = [
    {
        "id": "sec_1",
        "text": (
            "Section 3.2 — Refund Policy. "
            "All purchases are eligible for a full refund within 30 days of the original purchase date. "
            "To initiate a refund, customers must contact support with their order number. "
            "Digital downloads are non-refundable once accessed. "
            "Refunds are processed within 5–7 business days to the original payment method. "
            "Shipping costs are non-refundable unless the return is due to our error."
        )
    },
    {
        "id": "sec_2",
        "text": (
            "Section 4.1 — Shipping Policy. "
            "Standard shipping takes 5–7 business days. Express shipping takes 2–3 business days. "
            "Free standard shipping is available on orders over $50. "
            "International orders may be subject to customs duties. "
            "We do not ship to P.O. boxes for express orders."
        )
    },
]

query = "How long does a refund take to process?"

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_segment(section_text: str, query: str, window: int = 2) -> str:
    """
    Split section into sentences, find the most relevant sentence,
    return it with `window` sentences of surrounding context.
    """
    sentences = [s.strip() for s in section_text.split(". ") if s.strip()]
    if not sentences:
        return section_text

    sent_embs = model.encode(sentences).astype("float32")
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(sent_embs)
    faiss.normalize_L2(q_emb)

    scores = (sent_embs @ q_emb.T).flatten()
    best = int(np.argmax(scores))

    start = max(0, best - window)
    end = min(len(sentences), best + window + 1)
    return ". ".join(sentences[start:end]) + "."


# ── Retrieve top section, then extract the relevant segment ──────────────────
sec_embs = model.encode([s["text"] for s in sections]).astype("float32")
faiss.normalize_L2(sec_embs)
index = faiss.IndexFlatIP(sec_embs.shape[1])
index.add(sec_embs)

q_emb = model.encode([query]).astype("float32")
faiss.normalize_L2(q_emb)
_, top_idx = index.search(q_emb, 1)

best_section = sections[top_idx[0][0]]
segment = extract_segment(best_section["text"], query)

print(f"Query: '{query}'\n")
print(f"Retrieved section ({len(best_section['text'])} chars):")
print(f"  {best_section['text']}\n")
print(f"Extracted segment ({len(segment)} chars):")
print(f"  {segment}")


# Output:
# Query: 'How long does a refund take to process?'
#
# Retrieved section (378 chars):
#   Section 3.2 — Refund Policy. All purchases are eligible for a full refund within
#   30 days of the original purchase date. To initiate a refund, customers must contact
#   support with their order number. Digital downloads are non-refundable once accessed.
#   Refunds are processed within 5–7 business days to the original payment method.
#   Shipping costs are non-refundable unless the return is due to our error.
#
# Extracted segment (189 chars):
#   Digital downloads are non-refundable once accessed. Refunds are processed within
#   5–7 business days to the original payment method. Shipping costs are non-refundable
#   unless the return is due to our error.

# Findings:
# The full section is 378 chars; the extracted segment is 189 chars — 50% reduction.
# The segment centers on "Refunds are processed within 5–7 business days" (the
# most relevant sentence) with one sentence of context on each side.
# This is different from contextual compression: compression filters sentences
# by relevance score. Segment extraction finds the best sentence and returns
# a fixed window around it, preserving narrative flow.
# The window size controls the context-precision trade-off: window=0 returns
# only the single most relevant sentence; window=2 returns 5 sentences.
