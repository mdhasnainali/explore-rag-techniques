# GEMINI.md

Instructions for Gemini AI agents working on this repository. Read AGENTS.md first — this file adds Gemini-specific guidance on top of those shared rules.

---

## Project Context

This is a RAG learning repository. The audience is developers learning from scratch. Every file you touch must remain readable by a beginner. Do not optimise for brevity at the expense of clarity.

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

```python
# Correct — inline constants, no wrappers, output comment at bottom
CHUNK_SIZE = 100
CHUNK_OVERLAP = 15

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} [{len(chunk)} chars]: {chunk}")

# Output:
# Chunk 1 [96 chars]: Natural language processing...

# Findings:
# The default separators are ["\n\n", "\n", " ", ""]
```

When writing README files, follow the section order in AGENTS.md exactly:
1. Title
2. Feynman Explanation
3. Algorithm
4. Worked Example
5. Mermaid Diagram(s)
6. Key Findings
7. Pros, Cons & When to Use

---

## Mermaid Rules

Gemini tends to generate Mermaid with special characters that break rendering. Follow these rules strictly:

- No `(` or `)` inside node labels — use `\n` for line breaks instead
- No `"` inside node labels — use single quotes or rephrase
- No `{` or `}` inside node labels — use `decision?` phrasing instead
- Keep every node label under 40 characters
- Test mentally: would this render on GitHub?

Good:
```
flowchart TD
    A[Input Text] --> B[Split by separator]
    B --> C{Size exceeds limit?}
    C -- Yes --> D[Save chunk]
    C -- No --> E[Accumulate]
```

Bad:
```
flowchart TD
    A["Input Text (raw)"] --> B["Split by separator (\n\n)"]
```

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

Capture the actual output and put it in the `# Output:` comment. Do not invent output.
