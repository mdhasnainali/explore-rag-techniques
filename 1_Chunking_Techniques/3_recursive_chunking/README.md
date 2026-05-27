# Recursive Chunking

## Algorithm

Uses `langchain_text_splitters.RecursiveCharacterTextSplitter` to split text by trying multiple separators **hierarchically**, from coarsest to finest.

### Logic

The splitter maintains an ordered list of separators (default: `["\n\n", "\n", " ", ""]`):

1. Start with the first separator (`\n\n` — paragraph breaks).
2. If a split piece is **larger than** `chunk_size`, recursively split it using the **next** separator in the list.
3. If splitting produces pieces **smaller than** `chunk_size`, merge adjacent pieces back together (same overlap logic as `CharacterTextSplitter`).
4. If no separator fits within `chunk_size`, fall through to `""` (character-level split) as the last resort.

This ensures the splitter prefers breaking at natural boundaries (paragraphs → lines → words → characters) and only falls back to arbitrary cuts when necessary.

## Output

```
Chunk 1 [96 chars]: Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial
Chunk 2 [84 chars]: and artificial intelligence. It is concerned with the interactions between computers
Chunk 3 [81 chars]: and human language, in particular how to program computers to process and analyze
Chunk 4 [39 chars]: large amounts of natural language data.
```

## Key Findings

- **Overlap is visible here** (unlike the simple character splitter with `separator="\n"`). Chunk 1 ends with `artificial` and Chunk 2 starts with `and artificial`, showing 15 characters of overlap at the boundary (`and artificial`).
- The overlap works because the text is first broken by `\n` into 3 lines, then each line is further broken by `" "` into words. A chunk can contain multiple words, and the front words get popped while tail words carry forward as overlap.
- Recursive chunking produces the **cleanest boundaries** among fixed-size approaches — it splits at sentence/word boundaries whenever possible.
- The character counts (96, 84, 81, 39) are closer to `chunk_size=100` than the naive character splitter (which produced a 167-char oversized chunk), because the recursive approach breaks oversized pieces further.
- **Trade-off**: More computation than simple character splitting, but produces more semantically coherent chunks.
