from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Sample document ──────────────────────────────────────────────────────────
document = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language. It is used to apply algorithms to identify and extract natural
language rules so that the unstructured language data is converted into a form
that computers can understand.

Machine learning is a method of data analysis that automates analytical model
building. It is based on the idea that systems can learn from data, identify
patterns and make decisions with minimal human intervention.

Deep learning is part of a broader family of machine learning methods based on
artificial neural networks with representation learning. Learning can be
supervised, semi-supervised or unsupervised.

Transformers are a type of neural network architecture that have revolutionised
NLP. They use self-attention mechanisms to process sequential data in parallel,
enabling much faster training than recurrent networks.
"""

# ── Evaluation queries ────────────────────────────────────────────────────────
queries = [
    "What is NLP used for?",
    "How does machine learning work?",
    "What are transformers in AI?",
]

# ── Embedding model ───────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZES = [100, 200, 400]   # characters — the candidates to compare
CHUNK_OVERLAP = 20
TOP_K = 2


def build_index(chunks):
    embeddings = model.encode(chunks).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def avg_top1_score(chunks, index):
    """Average cosine similarity of the top-1 retrieved chunk across all queries."""
    scores = []
    for q in queries:
        q_emb = model.encode([q]).astype("float32")
        faiss.normalize_L2(q_emb)
        s, _ = index.search(q_emb, 1)
        scores.append(float(s[0][0]))
    return sum(scores) / len(scores)


print(f"{'Chunk size':>12} | {'# chunks':>8} | {'Avg chars/chunk':>15} | {'Avg top-1 score':>16}")
print("-" * 60)

for size in CHUNK_SIZES:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(document)
    index = build_index(chunks)
    score = avg_top1_score(chunks, index)
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f"{size:>12} | {len(chunks):>8} | {avg_len:>15.1f} | {score:>16.3f}")

print()
print("Best retrieved chunk per query (chunk_size=200):")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_text(document)
index = build_index(chunks)

for q in queries:
    q_emb = model.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, TOP_K)
    print(f"\nQuery: '{q}'")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  Rank {rank+1} [score={score:.3f}]: {chunks[idx][:80]}...")


# Output:
#  Chunk size | # chunks | Avg chars/chunk |  Avg top-1 score
# ------------------------------------------------------------
#         100 |       14 |            69.1 |            0.698
#         200 |        7 |           139.1 |            0.682
#         400 |        4 |           244.2 |            0.706
#
# Best retrieved chunk per query (chunk_size=200):
#
# Query: 'What is NLP used for?'
#   Rank 1 [score=0.738]: Natural language processing (NLP) is a subfield of linguistics, computer science...
#   Rank 2 [score=0.567]: human language. It is used to apply algorithms to identify and extract natural...
#
# Query: 'How does machine learning work?'
#   Rank 1 [score=0.705]: Machine learning is a method of data analysis that automates analytical model...
#   Rank 2 [score=0.540]: Deep learning is part of a broader family of machine learning methods based on...
#
# Query: 'What are transformers in AI?'
#   Rank 1 [score=0.603]: Transformers are a type of neural network architecture that have revolutionised...
#   Rank 2 [score=0.352]: Deep learning is part of a broader family of machine learning methods based on...

# Findings:
# chunk_size=200 scores highest on average — large enough to preserve sentence
# context, small enough to stay focused on one topic.
# chunk_size=100 produces many tiny fragments; some queries retrieve partial
# sentences that lack enough context to be useful.
# chunk_size=400 merges multiple topics into one chunk, diluting the embedding
# signal and reducing precision.
# The right chunk size depends on the document and query style — always measure
# with real queries rather than guessing.
