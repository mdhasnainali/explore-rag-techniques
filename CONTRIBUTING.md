# Contributing to Explore RAG Techniques

Thank you for your interest in contributing. This repository is a learning resource — every contribution must make it easier for a developer to understand RAG, not just add more content.

---

## Repository

```
https://github.com/mdhasnainali/explore-rag-techniques.git
```

---

## What We Welcome

- New RAG technique implementations (chunking, retrieval, evaluation)
- New research paper summaries added to module-level READMEs
- Corrections to existing documentation or code output comments
- Improvements to Feynman explanations or Mermaid diagrams
- Bug fixes in existing Python files

## What We Do Not Accept

- Techniques unrelated to RAG (general ML, unrelated NLP)
- Abstractions, base classes, or shared utility modules
- Changes that renumber existing folders or files
- Output comments that were not produced by actually running the code

---

## How to Contribute

### 1. Fork and clone

```bash
git clone https://github.com/mdhasnainali/explore-rag-techniques.git
cd explore-rag-techniques
```

### 2. Set up the environment

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

### 3. Create a branch

```bash
git checkout -b add/technique-name
# or
git checkout -b fix/description-of-fix
```

### 4. Make your changes

Follow the conventions in [AGENTS.md](AGENTS.md):

- Use the next available number for new folders and files
- Every technique needs both a `.py` file and a `README.md`
- Run the code and capture the real output before writing the `# Output:` comment
- README sections must follow the order: Title → Feynman Explanation → Algorithm → Worked Example → Mermaid Diagram(s) → Key Findings → Pros, Cons & When to Use

### 5. Verify your code runs

```bash
uv run python path/to/your/technique.py
```

Do not submit a technique whose output you have not verified.

### 6. Open a pull request

Push your branch and open a PR against `main`:

```bash
git push -u origin your-branch-name
```

PR title format: `add: <technique name>` or `fix: <short description>`

In the PR description, include:
- What technique or fix this adds
- The real output from running the code (paste it)
- Any new dependencies added and why

---

## Adding a Research Paper

Papers go in the module-level `README.md` (e.g., `1_Chunking_Techniques/README.md`), not in individual technique READMEs.

Use this format:

```markdown
### Paper N: Short description

*"Full Paper Title"*
Authors et al. (Year) — [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

| Metric | Baseline | This method |
|---|---|---|
| Accuracy | 0.78 | 0.89 |

Key findings:
1. Specific, quantified finding.
2. Another finding.
```

Always include the full paper title and URL.

---

## Questions

Open an issue on [GitHub](https://github.com/mdhasnainali/explore-rag-techniques/issues) if you are unsure whether a contribution fits the project before spending time on it.
