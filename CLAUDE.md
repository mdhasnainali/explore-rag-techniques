# CLAUDE.md

Instructions for Claude AI agents working on this repository. Read AGENTS.md first — this file adds Claude-specific guidance on top of those shared rules.

---

## Project Context

This is a RAG learning repository. The audience is developers learning from scratch. Every file you touch must remain readable by a beginner. Prioritise clarity over cleverness in both code and documentation.

---

## What You Are Allowed to Do

- Add new technique folders following the numbering convention in AGENTS.md
- Write or update Python implementation files
- Write or update README files for technique folders
- Add research paper summaries to module-level READMEs
- Add new dependencies to `pyproject.toml` when a technique requires them

---

## What You Must Not Do

- Do not renumber existing folders or files
- Do not modify existing `# Output:` comments unless you have actually re-run the code
- Do not add utility modules, base classes, or shared helpers — every file is self-contained
- Do not add techniques unrelated to RAG (no general ML, no unrelated NLP tasks)
- Do not use `pip install` — use `uv add <package>` for new dependencies

---

## Code Generation Rules

When writing Python files:

- Scripts run top-to-bottom — no `if __name__ == "__main__"` wrappers
- All parameters are inline constants with comments explaining their purpose
- Imports at the top, no inline imports
- Always end with a `# Output:` block containing the real output from running the file
- Always end with a `# Findings:` block noting non-obvious behaviours

```python
# Correct pattern
CHUNK_SIZE = 100      # characters per chunk
CHUNK_OVERLAP = 15    # characters of overlap between chunks

splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} [{len(chunk)} chars]: {chunk}")

# Output:
# Chunk 1 [99 chars]: Natural language processing...

# Findings:
# separator="" cuts mid-word; separator=" " preserves word boundaries.
```

---

## README Section Order

Follow this order exactly — do not skip or reorder sections:

1. **Title**
2. **Feynman Explanation** — concrete analogy, no jargon, accessible to a 12-year-old
3. **Algorithm** — step-by-step, references library internals where relevant
4. **Worked Example** — uses the real input/output from the Python file
5. **Mermaid Diagram(s)** — at least one `flowchart TD` or `flowchart LR`
6. **Key Findings** — bullet list of non-obvious behaviours
7. **Pros, Cons & When to Use** — table + "Suitable for:" / "Not suitable for:" lists

---

## Mermaid Rules

Claude tends to over-engineer Mermaid diagrams. Keep them simple:

- Maximum 15 nodes per diagram — split into multiple diagrams if needed
- No special characters inside node labels: no `(`, `)`, `"`, `{`, `}`
- Node text under 40 characters — use `\n` for line breaks
- Use `flowchart TD` for pipelines, `flowchart LR` for comparisons, `sequenceDiagram` for request/response

Prefer simple and correct over complex and broken. A diagram that renders is worth more than a detailed one that doesn't.

---

## Feynman Explanation Rules

The Feynman explanation is the most important section. Claude tends to write explanations that are still too technical. Check against these criteria:

- Could a developer who has never heard of RAG understand this?
- Is there a concrete real-world analogy (not a technical one)?
- Are there zero unexplained acronyms in the first paragraph?
- Does it explain *why* this technique exists, not just *what* it does?

Bad: "Semantic chunking uses cosine similarity between sentence embeddings to detect topic boundaries."
Good: "Imagine reading a book and noticing when the author switches subjects. You don't need a ruler — you just feel the shift. Semantic chunking automates that feeling using embeddings."

---

## Research Paper Format

When adding a paper to a module README:

```markdown
### Paper N: Short description

*"Full Paper Title"*
Authors et al. (Year) — [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

| Metric | Baseline | This method |
|---|---|---|
| Accuracy | 0.78 | 0.89 |

Key findings:
1. Finding one — specific and quantified where possible.
2. Finding two.
```

Always include the full paper title and URL. Never cite by arXiv ID alone.

---

## Running Code

Always verify code runs before presenting results:

```bash
uv run python path/to/technique.py
```

Capture the actual output and put it in the `# Output:` comment. Do not invent output. If the code requires an API key and cannot be run, state that clearly and mark the output as expected rather than verified.
