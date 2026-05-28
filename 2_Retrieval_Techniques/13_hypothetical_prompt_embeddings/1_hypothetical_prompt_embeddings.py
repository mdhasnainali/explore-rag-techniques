from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Knowledge base ────────────────────────────────────────────────────────────
chunks = [
    "Photosynthesis converts sunlight, CO2, and water into glucose and oxygen.",
    "The mitochondria produce ATP through cellular respiration.",
    "DNA replication occurs during the S phase of the cell cycle.",
    "Neurons transmit signals via electrochemical impulses across synapses.",
    "The immune system uses antibodies to neutralise pathogens.",
]

# ── Hypothetical prompts per chunk (in production, generate with an LLM) ─────
# HyPE generates questions a chunk could answer, then indexes those questions.
# At query time, the user's question is matched against these indexed questions.
hypothetical_prompts = {
    0: ["How do plants produce food?", "What is the equation for photosynthesis?",
        "What do plants need to make glucose?"],
    1: ["Where is ATP produced in the cell?", "What is the role of mitochondria?",
        "How does cellular respiration work?"],
    2: ["When does DNA replication happen?", "What phase of the cell cycle copies DNA?"],
    3: ["How do neurons communicate?", "What is a synapse?",
        "How are nerve signals transmitted?"],
    4: ["How does the immune system fight infection?", "What are antibodies?",
        "How does the body neutralise pathogens?"],
}

model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Build prompt index: embed all hypothetical prompts, link back to chunk ────
prompt_texts = []
prompt_to_chunk = []
for chunk_idx, prompts in hypothetical_prompts.items():
    for p in prompts:
        prompt_texts.append(p)
        prompt_to_chunk.append(chunk_idx)

prompt_embs = model.encode(prompt_texts).astype("float32")
faiss.normalize_L2(prompt_embs)
prompt_index = faiss.IndexFlatIP(prompt_embs.shape[1])
prompt_index.add(prompt_embs)

# ── Also build standard chunk index for comparison ───────────────────────────
chunk_embs = model.encode(chunks).astype("float32")
faiss.normalize_L2(chunk_embs)
chunk_index = faiss.IndexFlatIP(chunk_embs.shape[1])
chunk_index.add(chunk_embs)

queries = [
    "What happens during photosynthesis?",
    "How do brain cells send messages?",
]

for query in queries:
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    # Standard: match query against chunk embeddings
    std_scores, std_idx = chunk_index.search(q_emb, 1)

    # HyPE: match query against prompt embeddings, return linked chunk
    hype_scores, hype_idx = prompt_index.search(q_emb, 1)
    hype_chunk_idx = prompt_to_chunk[hype_idx[0][0]]

    print(f"Query: '{query}'")
    print(f"  Standard → [{std_scores[0][0]:.3f}] {chunks[std_idx[0][0]]}")
    print(f"  HyPE     → [{hype_scores[0][0]:.3f}] matched prompt: '{prompt_texts[hype_idx[0][0]]}'")
    print(f"             → chunk: {chunks[hype_chunk_idx]}")
    print()


# Output:
# Query: 'What happens during photosynthesis?'
#   Standard → [0.712] Photosynthesis converts sunlight, CO2, and water into glucose and oxygen.
#   HyPE     → [0.891] matched prompt: 'What is the equation for photosynthesis?'
#              → chunk: Photosynthesis converts sunlight, CO2, and water into glucose and oxygen.
#
# Query: 'How do brain cells send messages?'
#   Standard → [0.534] Neurons transmit signals via electrochemical impulses across synapses.
#   HyPE     → [0.923] matched prompt: 'How do neurons communicate?'
#              → chunk: Neurons transmit signals via electrochemical impulses across synapses.

# Findings:
# HyPE scores are higher (0.89 vs 0.71) because the user's question matches
# a generated question more closely than it matches the answer text.
# "How do brain cells send messages?" matches "How do neurons communicate?"
# better than it matches the chunk text about "electrochemical impulses".
# The cost: generating hypothetical prompts at indexing time requires one LLM
# call per chunk. For 10,000 chunks, that is 10,000 LLM calls.
# HyDE (query time) vs HyPE (index time): HyDE is cheaper per-query but
# adds latency. HyPE is expensive at indexing but fast at query time.
