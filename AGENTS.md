# AGENTS.md

Instructions for AI coding agents working on this repository.

---

## Project Purpose

This is a **learning repository** for Retrieval-Augmented Generation (RAG). The primary audience is developers learning RAG from scratch. Every addition must serve that learning goal — clarity and correctness matter more than brevity.

---

## Repository Conventions

### Numbering

Every folder and file is prefixed with a number. Numbers define the learning order and must be respected:

```
1_Chunking_Techniques/
  1_fixed_size_chunking_by_character/
    1_fixed_size_chunking_by_character.py
  2_fixed_size_chunking_by_token/
    1_fixed_size_chunking_by_token.py
  ...
2_Retrieval_Techniques/
  1_dense_retrieval/
  2_sparse_retrieval/
3_Evaluation_Techniques/
```

When adding a new technique, use the next available number at the correct level. Never renumber existing files.

### Python Files

- One file per technique. Keep it minimal — only the code needed to demonstrate the technique.
- Always include a commented `# Output:` block at the bottom showing the actual output from running the file.
- Always include a `# Findings:` block noting non-obvious behaviours.
- Use the same sample text or domain across techniques where possible (the NLP paragraph, the policy document, the Marie Curie passage) so learners can compare outputs directly.
- No `if __name__ == "__main__"` wrappers — scripts run top-to-bottom.

### README Files

Every technique folder must have a `README.md`. It must contain all of the following sections in this order:

1. **Title** — technique name
2. **Feynman Explanation** — explain the idea to a 12-year-old with a concrete analogy. No jargon.
3. **Algorithm** — step-by-step breakdown of how the code actually works internally. Reference the source library internals where relevant.
4. **Worked Example** — trace the actual input from the Python file through the algorithm to the actual output. Use the real output from the code, not invented examples.
5. **Mermaid Diagram(s)** — at least one flowchart. Use `flowchart TD` or `flowchart LR`. Keep node labels short.
6. **Key Findings** — bullet list of non-obvious behaviours discovered from running the code.
7. **Pros, Cons & When to Use** — a table with ✅/❌ rows, followed by "Suitable for:" and "Not suitable for:" bullet lists.

### Mermaid Diagrams

- Use `flowchart TD` (top-down) for pipelines and algorithms.
- Use `flowchart LR` (left-right) for comparisons.
- Use `sequenceDiagram` for request/response flows.
- Keep node text under 40 characters. Use `\n` for line breaks inside nodes.
- Avoid special characters in node labels that break Mermaid parsing: `(`, `)`, `{`, `}`, `"` inside labels must be escaped or avoided.
- Always test that diagrams render — malformed Mermaid silently fails in GitHub.

---

## Adding a New Technique

1. Create the numbered folder: `N_technique_name/`
2. Write the Python file: `N_technique_name.py`
3. Run it and capture the real output
4. Write `README.md` following the structure above
5. Update the parent folder's `README.md` table to include the new technique
6. If a new dependency is needed, add it to `pyproject.toml` with a pinned minimum version

---

## Adding Research Papers

Research findings go in the module-level `README.md` (e.g., `1_Chunking_Techniques/README.md`), not in individual technique READMEs.

Format:
- Section header: `### Paper N: [Short description]`
- Italic title + authors + year + arXiv link on the next line
- Performance tables using Markdown table syntax
- Key findings as a numbered list at the bottom of the section

Every paper citation in the body must include the full paper title and URL — not just an arXiv ID.

---

## Code Style

- Python 3.12+
- No type annotations required in example scripts (they add noise for learners)
- `uv` for dependency management — never `pip install` directly
- Keep imports at the top of the file
- No external configuration files — all parameters are inline constants with comments

---

## What Not to Do

- Do not add abstractions, base classes, or utility modules — every file must be self-contained and readable in isolation
- Do not add tests — this is a learning repo, not a production library
- Do not change existing output comments unless you have re-run the code and the output has actually changed
- Do not add techniques that are not RAG-related (no general ML, no unrelated NLP)
- Do not skip the Feynman explanation — it is the most important section for the target audience
