# Recursive Chunking

## Algorithm

Uses `langchain_text_splitters.RecursiveCharacterTextSplitter` to split text by trying multiple separators **hierarchically**, from coarsest to finest. Defined in `_split_text()` in `character.py`.

### Step-by-Step Logic

**Step 1 — Select the best separator**

Given an ordered list of separators (default: `["\n\n", "\n", " ", ""]`), scan from left to right to find the first separator that exists in the current text segment. This becomes the primary splitter for this level. The remaining separators become `new_separators` for recursive descent.

```
separators = ["\n\n", "\n", " ", ""]
text = "Line 1\nLine 2\n\nParagraph 2\nLine 4"

Check "\n\n" → found at "Paragraph 2"
→ separator = "\n\n", new_separators = ["\n", " ", ""]
```

**Step 2 — Split using the selected separator**

Divide the text using the selected separator via regex. Each resulting piece is checked against `chunk_size`.

```
splits = text.split("\n\n")
→ ["Line 1\nLine 2", "Paragraph 2\nLine 4"]
```

**Step 3 — Process each split recursively**

For each split piece:
- If `len(piece) < chunk_size`: Add to `good_splits` accumulator (these will be merged later).
- If `len(piece) >= chunk_size`: 
  - First merge and save any accumulated `good_splits` into chunks.
  - Then, if there are `new_separators` remaining, **recursively** call `_split_text(piece, new_separators)`.
  - If no separators remain (already at `""`), save the piece as-is.

```
piece "Line 1\nLine 2" (14 chars) < 100 → good_splits = ["Line 1\nLine 2"]
piece "Paragraph 2\nLine 4" (21 chars) < 100 → good_splits = ["Line 1\nLine 2", "Paragraph 2\nLine 4"]

Now merge good_splits into chunks of ~100 chars using _merge_splits
```

**Step 4 — Merge accumulated splits**

When recursion returns or the loop ends, the accumulated `good_splits` are merged using the same `_merge_splits` logic from `CharacterTextSplitter` — accumulating pieces, finalizing at `chunk_size` boundaries, and applying overlap by popping front elements.

### Full Trace with Example

```
Text:  "\nNatural language processing...\nand human language...\nlarge amounts..."
       chunk_size=100, chunk_overlap=15

1. Separator "\n\n" not found in text (no paragraph breaks).
2. Separator "\n" found. new_separators = [" ", ""]
3. Split by "\n": 3 pieces → ["Natural... (167 chars)", "and human... (81 chars)", "large amounts... (39 chars)"]

4. Process piece 1 (167 chars >= 100):
   → Recursively split with separators [" ", ""]
   → Separator " " found → split into words (~20 words)
   → Each word < 100 → accumulate in good_splits
   → Merge good_splits into chunks of ~100 chars:
     → Chunk A: words[0..N] totaling 96 chars → "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial"
     → Overlap: pop front words until total ≤ 15
       Remaining: last ~1-2 words ("and artificial", 15 chars)
     → Chunk B: overlap tail + remaining words
       → "and artificial intelligence. It is concerned with the interactions between computers"
       → 84 chars

5. Process piece 2 (81 chars < 100):
   → Accumulate in good_splits

6. Process piece 3 (39 chars < 100):
   → Accumulate in good_splits

7. Merge good_splits [piece2, piece3]:
   → 81 + 1 + 39 = 121 > 100
   → Chunk C: piece2 (81 chars) → "and human language, in particular how to program computers to process and analyze"
   → Overlap: pop front words → no room for 15 chars overlap from 81-char single piece
   → Chunk D: piece3 (39 chars) → "large amounts of natural language data."
```

### Mermaid Diagram

```mermaid
flowchart TD
    A[Input Text] --> B[Select first separator\nthat exists in text]
    B --> C[Split text by separator]
    C --> D{For each split piece}
    D --> E{len(piece) < chunk_size?}
    E -- Yes --> F[Add to good_splits]
    E -- No --> G[Save good_splits as chunks\nvia _merge_splits]
    G --> H{new_separators remain?}
    H -- Yes --> I[Recurse with\npiece + new_separators]
    H -- No --> J[Save piece as-is]
    I --> K[Return sub-chunks]
    J --> K
    F --> D
    K --> D
    D --> L[End of splits?]
    L -- No --> D
    L -- Yes --> M[Save remaining\ngood_splits as chunks]
    M --> N[Return all chunks]
```

### Separator Hierarchy

```mermaid
flowchart LR
    subgraph "Default Separator Order"
        S1["\\n\\n (paragraphs)"] --> S2["\\n (lines)"]
        S2 --> S3["  (words)"]
        S3 --> S4["\"\" (characters)\nlast resort"]
    end
    subgraph "Why This Order?"
        O1["Preserves largest\nsemantic units first"]
        O2["Falls back to smaller\nunits only when needed"]
        O3["Avoids mid-word cuts\nunless unavoidable"]
    end
```

## Output

```
Chunk 1 [96 chars]: Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial
Chunk 2 [84 chars]: and artificial intelligence. It is concerned with the interactions between computers
Chunk 3 [81 chars]: and human language, in particular how to program computers to process and analyze
Chunk 4 [39 chars]: large amounts of natural language data.
```

## Key Findings

- **Overlap is visible**: Chunk 1 ends with `artificial` and Chunk 2 starts with `and artificial` — 15 characters of overlap. This works because the first line was recursively split by `" "` into words, allowing the overlap mechanism to pop front words and keep tail words as overlap.
- **Character counts are closer to `chunk_size`** (96, 84, 81, 39) compared to the naive character splitter which produced a 167-char oversized chunk. The recursive approach breaks oversized pieces further.
- **Cleanest boundaries**: Splits at sentence/word boundaries whenever possible, only falling back to character-level cuts as a last resort.
- **Trade-off**: More computation than `CharacterTextSplitter` (recursive calls for each oversized piece), but produces more semantically coherent chunks.
- The overlap is most impactful when the finest separator (`" "` or `""`) is reached, creating many small pieces that can be selectively retained.
