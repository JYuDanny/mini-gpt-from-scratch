# AGENTS.md — Mini-GPT from Scratch

> Instructions for AI coding assistants working on this project.
> This file is loaded by opencode as project-level instructions.
> All rules herein are mandatory and non-negotiable.

---

## Rule 1: Ask Before Acting — No Unilateral Decisions

**Before any high-risk operation or when information is unclear, you MUST ask for explicit confirmation.**

High-risk operations include (but are not limited to):
- Changing the model architecture (e.g., modifying attention, FFN, embedding layers)
- Modifying the training loop in ways that affect convergence
- Deleting files, checkpoints, or data
- Force-pushing to any branch
- Changing the conda environment or installing new packages
- Modifying `configs/mini_gpt.yaml` defaults
- Any operation that could corrupt or delete training artifacts

When information is unclear or ambiguous:
- **DO NOT** guess or assume — ask for clarification first
- State what you understand, what is unclear, and what you propose to do
- Wait for explicit confirmation before proceeding

**Principle:** The project owner has final authority. Your role is to *suggest*, *analyze*, and *execute under direction*, not to make autonomous architectural decisions.

---

## Rule 2: Verify Before Committing — No Broken Code

**Every code change MUST pass verification before commit and push.**

Verification workflow (mandatory):
1. Run the relevant self-tests embedded in the model files:
   - `python src/model.py` (base GPT architecture)
   - `python src/model_kvcache.py` (KV-cache correctness)
   - `python src/model_rope.py` (RoPE + KV-cache correctness)
2. If the change touches training logic, run `python train.py` for at least 1 evaluation cycle to verify no runtime errors.
3. If the change touches data pipeline, run `python src/prepare_data.py` and `python src/dataset.py` to verify.
4. Run `python src/tokenizer.py` to verify tokenizer roundtrip.
5. **Security check:** Before every commit, inspect `git diff --staged` to verify:
   - No API keys, tokens, or credentials in the diff
   - No personal information (real names, emails, file paths with usernames)
   - No hardcoded secrets or configuration values that should be externalized
   - No large binary files or checkpoint files accidentally staged

**Only after ALL checks pass AND the security review is clean may you proceed to commit and push.**

---

## Rule 3: Commit and Push After Every Change — No Lingering Work

**After each logical unit of work, commit and push immediately.**

Commit workflow:
1. Verify all checks pass (Rule 2)
2. Stage related changes together: `git add <files>`
3. Write a concise, descriptive commit message following the repo convention
4. Commit: `git commit -m "<message>"`
5. Push: `git push origin feat/restructure-and-optimize`

**Never** accumulate multiple unrelated changes into a single commit.
**Never** leave uncommitted changes at the end of a session.

---

## Environment

- **Conda environment:** `mini-gpt` (Python 3.11)
- **Activation:** `conda activate mini-gpt` before any Python commands
- **Hardware target:** RTX 4060 8GB + 16GB RAM
- **OS:** Windows 11
- **Python commands:** Use `python` (not `python3`)

Do NOT create new conda environments or install packages without explicit confirmation.

---

## Code Style Conventions

- Follow existing patterns: the codebase uses Chinese comments with English variable/class names
- **Teaching comment standard:** All code must have complete, clear Chinese teaching annotations explaining the principles and data flow of each component. Key technical terms should include their English equivalents in parentheses, e.g., `# 词嵌入 (Token Embedding)`. Variable and class names remain in English.
- `GPTConfig` is the configuration dataclass — add new model params here, not as globals
- Model files in `src/` are independently runnable with built-in self-tests
- Keep self-tests at the bottom of each file under `if __name__ == "__main__":`
- Training config priority: CLI args > YAML file > hard-coded defaults
