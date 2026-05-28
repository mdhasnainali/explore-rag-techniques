# Advanced RAG Architectures

These patterns extend basic RAG with loops, graphs, agents, memory, or multimodal understanding.

| # | Architecture | Core idea |
|---|---|---|
| 1 | [Self-RAG](1_self_rag/README.md) | The model decides when to retrieve and critique its own answer. |
| 2 | [Corrective RAG](2_corrective_rag/README.md) | Detect poor retrieval and repair the context path. |
| 3 | [GraphRAG](3_graph_rag/README.md) | Retrieve through entities and relationships. |
| 4 | [Microsoft GraphRAG](4_microsoft_graphrag/README.md) | Build community summaries over a knowledge graph. |
| 5 | [RAPTOR](5_raptor/README.md) | Build a tree of clustered summaries for multi-level retrieval. |
| 6 | [MemoRAG](6_memorag/README.md) | Add memory as a retrieval planning layer. |
| 7 | [Agentic RAG](7_agentic_rag/README.md) | Let an agent choose tools, retrieval steps, and checks. |
| 8 | [Multimodal RAG](8_multimodal_rag/README.md) | Retrieve over text, images, tables, and page layouts. |

> Retrieval Feedback Loop has moved to [`2_Retrieval_Techniques/15_retrieval_feedback_loop/`](../2_Retrieval_Techniques/15_retrieval_feedback_loop/README.md) — it is a retrieval strategy, not an architecture.
