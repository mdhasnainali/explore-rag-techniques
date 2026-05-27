import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Knowledge base
documents = [
    "Python is a high-level programming language known for readability.",
    "Neural networks are inspired by the human brain structure.",
    "The capital of France is Paris, a city on the Seine river.",
    "Machine learning enables systems to learn patterns from data.",
    "Paris hosted the 1900 and 1924 Olympic Games.",
]

# Embed and index documents
doc_embeddings = model.encode(documents).astype("float32")
dim = doc_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

# Query
query = "Which city is the French capital?"
query_emb = model.encode([query]).astype("float32")
faiss.normalize_L2(query_emb)

k = 2
scores, indices = index.search(query_emb, k)

print(f"Query: '{query}'\n")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
    print(f"Rank {rank+1} [score={score:.3f}]: {documents[idx]}")

    
# Output:
# Query: 'Which city is the French capital?'

# Rank 1 [score=0.789]: The capital of France is Paris, a city on the Seine river.
# Rank 2 [score=0.346]: Paris hosted the 1900 and 1924 Olympic Games.