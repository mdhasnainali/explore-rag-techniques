# Fixed-Size Chunking by Character

## Algorithm

Uses `langchain_text_splitters.CharacterTextSplitter` to split text into chunks of a fixed number of characters.

### Logic

1. **Split phase**: The text is divided by a `separator` (default `""`, meaning every character is a split point).
2. **Merge phase**: Adjacent splits are merged back together targeting `chunk_size` characters.
3. **Overlap**: When a chunk boundary is hit, the splitter pops elements from the **front** of the accumulated splits while `total > chunk_overlap`. The remaining tail elements form the start of the next chunk, producing overlap.

**Key parameter behavior:**
- `separator=""` (blind cut): Splits at exactly `chunk_size` characters regardless of word boundaries. Words can be chopped mid-character.
- `separator="\n"` (smart cut): Splits only at newline boundaries. Pieces between newlines are atomic — if a line exceeds `chunk_size`, it becomes its own oversized chunk and no overlap is applied (since popping the single line element leaves nothing to carry forward).

## Output

```
Chunk 1 [99 chars]: Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial in
Chunk 2 [100 chars]: d artificial intelligence. It is concerned with the interactions between computers and human langua
Chunk 3 [100 chars]: nd human language, in particular how to program computers to process and analyze large amounts of n
Chunk 4 [36 chars]: ge amounts of natural language data.
```

## Key Findings

- **Blind cutting** (`separator=""`) splits words arbitrarily (e.g., `artificial in` / `d artificial`), which can degrade retrieval quality since tokens/words are fractured.
- **Smart cutting** (`separator="\n"`) preserves whole lines but can produce oversized chunks if a single line exceeds `chunk_size`. Overlap may not apply because each line is a single atomic piece.
- The `chunk_overlap` parameter works by dropping front elements from a multi-piece chunk's accumulator. It only produces visible overlap when a chunk contains **multiple small splits** that were merged together.
- For character-level chunking, overlap is most visible when the separator produces many small pieces (e.g., `separator=" "` with short words, or `separator=""` with individual characters).
