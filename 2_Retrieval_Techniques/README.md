# Retrieval Techniques

Retrieval decides which information reaches the generator. A strong answer is impossible if the right evidence never enters the prompt.

| # | Technique | Core idea |
|---|---|---|
| 1 | [Dense retrieval](1_dense_retrieval/README.md) | Search by semantic vector similarity. |
| 2 | [Sparse retrieval](2_sparse_retrieval/README.md) | Search by exact token statistics such as BM25. |
| 3 | [Hybrid retrieval](3_hybrid_retrieval/README.md) | Combine dense semantic recall with sparse exact-match recall. |
| 4 | [Reranking](4_reranking/README.md) | Reorder retrieved candidates with a stronger scoring model. |
| 5 | [Multi-query retrieval](5_multi-query_retrieval/README.md) | Ask several rewritten versions of the same question. |
| 6 | [HyDE](6_hypothetical_document_embeddings/README.md) | Generate a hypothetical answer document and retrieve against it. |
| 7 | [Adaptive retrieval](7_adaptive_retrieval/README.md) | Route each query to the retrieval strategy it needs. |
| 8 | [Metadata filtering](8_metadata_filtering/README.md) | Restrict search by structured fields before or after ranking. |
| 9 | [Contextual compression](9_contextual_compression/README.md) | Keep only the parts of retrieved chunks that matter. |
| 10 | [Relevant segment extraction](10_relevant_segment_extraction/README.md) | Extract answer-bearing spans from larger documents. |
| 11 | [Explainable retrieval](11_explainable_retrieval/README.md) | Show why a result was retrieved. |
| 12 | [Dartboard retrieval](12_dartboard_retrieval/README.md) | Balance relevance and diversity. |
| 13 | [HyPE](13_hypothetical_prompt_embeddings/README.md) | Generate likely questions or prompts and retrieve with those embeddings. |

```mermaid
flowchart LR
    Q[Query] --> A[Candidate retrieval]
    A --> B[Filter]
    B --> C[Rerank]
    C --> D[Compress]
    D --> E[Context for LLM]
```
