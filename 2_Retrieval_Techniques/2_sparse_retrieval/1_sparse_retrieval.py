from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

# Knowledge base
documents = [
    "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "ValueError: operands could not be broadcast together with shapes (3,) (4,)",
    "TypeError: unsupported operand type(s) for +: int and str",
    "Fix for CUDA OOM: reduce batch size or use gradient checkpointing",
    "Memory error solutions: torch.cuda.empty_cache() clears GPU memory",
]

tokenized_docs = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Exact term query
query = "CUDA out of memory error fix"
tokenized_query = tokenize(query)
scores = bm25.get_scores(tokenized_query)

ranked = sorted(zip(scores, documents), reverse=True)
print(f"Query: '{query}'\n")
for score, doc in ranked[:3]:
    print(f"Score {score:.3f}: {doc}")


# Output:
# Score 2.697: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
# Score 1.933: Memory error solutions: torch.cuda.empty_cache() clears GPU memory
# Score 1.311: Fix for CUDA OOM: reduce batch size or use gradient checkpointing