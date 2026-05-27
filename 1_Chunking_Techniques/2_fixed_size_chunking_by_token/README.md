# Fixed-Size Chunking by Token

## Algorithm

Uses `CharacterTextSplitter.from_tiktoken_encoder()` to split text into chunks of a fixed number of **tokens** (not characters), measured by the `cl100k_base` tokenizer (OpenAI's encoding for GPT-4 / text-embedding-3-*).

### Logic

1. The text is encoded into token IDs using `tiktoken`.
2. Token sequences are split into groups of `chunk_size` tokens.
3. Each token group is decoded back into text.
4. The `chunk_overlap` controls how many tokens overlap between consecutive chunks (subtracted from the start position of the next window).

Unlike character-level splitting, token-level splitting ensures each chunk has a predictable **token count** for LLM context windows, even though the character count varies across chunks.

## Output

```
Chunk 1 [99 chars]: Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial in
Chunk 2 [100 chars]: telligence. It is concerned with the interactions between computers and human language, in particul
Chunk 3 [91 chars]: ar how to program computers to process and analyze large amounts of natural language data.

Chunk 1 token count: 21
Chunk 2 token count: 18
Chunk 3 token count: 17
```

## Key Findings

- **Character count varies** across chunks (99, 100, 91 chars) even though all chunks target the same number of **tokens**. This is because different tokens encode different amounts of text — common words use fewer tokens per character.
- **Token-aware chunking** guarantees each chunk fits within an LLM's context window, avoiding subtle truncation issues that character-based chunking can cause.
- With `chunk_overlap=0`, the token window slides cleanly without overlap, producing adjacent non-overlapping token sequences.
- `chunk_size=100` tokens is quite small — typically 75-150 tokens per chunk is used for dense retrieval scenarios.
- **Alternatives**: spaCy, SentenceTransformers, NLTK, and KoNLPy offer different tokenization strategies with varying language support and granularity.
