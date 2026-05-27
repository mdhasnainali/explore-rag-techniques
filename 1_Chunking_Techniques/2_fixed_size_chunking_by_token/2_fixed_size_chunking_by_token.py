from transformers import AutoTokenizer

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. It is concerned with the interactions between computers 
and human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

CHUNK_SIZE = 30    # tokens per chunk
CHUNK_OVERLAP = 5  # tokens of overlap between chunks


def sliding_window(token_ids, chunk_size, overlap):
    """Yield (start, end) slices of token_ids using a sliding window."""
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        yield start, end
        if end == len(token_ids):
            break
        start += chunk_size - overlap

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
token_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"Total tokens: {len(token_ids)}\n")

for i, (s, e) in enumerate(sliding_window(token_ids, CHUNK_SIZE, CHUNK_OVERLAP)):
    chunk_text = tokenizer.decode(token_ids[s:e], skip_special_tokens=True)
    print(f"Chunk {i + 1} [tokens {s}–{e - 1}, {e - s} tokens]: {chunk_text!r}")


# Output:
# Total tokens: 50

# Chunk 1 [tokens 0–29, 30 tokens]: 'natural language processing ( nlp ) is a subfield of linguistics, computer science, and artificial intelligence. it is concerned with the interactions between computers and'
# Chunk 2 [tokens 25–49, 25 tokens]: 'the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.'
