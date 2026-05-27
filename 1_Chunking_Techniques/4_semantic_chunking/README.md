# Semantic Chunking

## Origin

First proposed by **Greg Kamradt**, later implemented in LangChain as `SemanticChunker`. Unlike fixed-size approaches that split at arbitrary character/token positions, semantic chunking splits text at **natural topic boundaries** using embedding similarity.

## Algorithm

Uses `langchain_experimental.text_splitter.SemanticChunker` with sentence embeddings to split text at natural semantic boundaries.

### Logic

1. **Embed sentences**: Each sentence is converted to a vector embedding using a sentence transformer model (e.g., `all-MiniLM-L6-v2` or OpenAI embeddings).
2. **Compute similarity gaps**: The cosine similarity between consecutive sentence embeddings is calculated. A sudden drop in similarity signals a topic shift.
3. **Identify breakpoints** using one of four threshold strategies:
   - `percentile` (used at 20%): Split at differences greater than the Nth percentile of all gaps.
   - `standard_deviation`: Split at differences greater than N standard deviations from the mean gap.
   - `interquartile`: Split at differences exceeding the interquartile distance (IQR-based).
   - `gradient`: Split at peaks in the rate of similarity change.
4. **Form chunks**: Sentences between breakpoints form a single chunk — all topically coherent.

### Typical Pipeline in Practice

```
PDF Document → read_pdf_to_string() → SemanticChunker → FAISS Vector Store → Retriever (top-k)
```

- PDF is extracted to plain text.
- Text is split into semantic chunks using OpenAI embeddings (or any embedding model).
- Chunks are embedded and indexed in a **FAISS** vector store for efficient similarity search.
- A **retriever** fetches the top-k most relevant chunks for a given query (e.g., `top_k=2`).

## Output

```
Chunk 1 [80 chars]: The Eiffel Tower is located in Paris. It was built in 1889 for the World's Fair.

Chunk 2 [45 chars]: Photosynthesis converts sunlight into energy.
```

## Comparison Diagram

![Semantic vs Regular Chunking](semantic_chunking_diagram.svg)

The diagram illustrates how regular chunking splits topics across arbitrary boundaries (Abstract/Intro/Methods mixed), while semantic chunking preserves entire topical sections (Methods stays together, Results stays together). For queries like *"What were the methods used to measure blood pressure?"*, semantic chunking retrieves the complete Methods section in one coherent chunk, whereas regular chunking requires stitching fragments from multiple chunks.

## Key Findings

- **Semantic coherence**: Related sentences stay together (Eiffel Tower facts grouped; photosynthesis kept separate). A fixed-size splitter would merge or split these across arbitrary boundaries.
- **Variable chunk sizes**: Lengths are determined by content, not a fixed limit — the defining difference from all other techniques.
- **No overlap needed**: Chunks are semantically complete units, making overlap unnecessary.
- **Embedding cost**: Requires model inference over every sentence — the most computationally expensive approach. For large documents, this is significant.
- **Threshold tuning** is critical: `breakpoint_threshold_amount` is a sensitive hyperparameter. Too high → too many tiny chunks; too low → fails to split unrelated topics.
- **Best for**: Documents with clear topical shifts — scientific papers, legal documents, comprehensive reports, FAQs. Less useful for homogeneous text where semantic boundaries are subtle.
