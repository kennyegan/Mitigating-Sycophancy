# Anonymization Record

Goal: ensure the supplement contains no personally or institutionally
identifying information.  Reviewers should be able to read the entire
supplement without learning the authors' identities.

This document is the human-readable record.  The automated check is
`tools/check_anonymization.py`.

## What was removed or rewritten

| Item | Source location | Action taken |
|------|-----------------|--------------|
| Cluster account name (`pi_larsonj_wit_edu`) | `slurm/*.sh`, `scripts/generate_figures_neurips.py` | All `slurm/*.sh` files were excluded from the supplement.  The figure script's hard-coded `Path(...)` was replaced with `Path(__file__).resolve().parent.parent`. |
| Cluster username (`egank2_wit_edu`) | absolute paths | Same as above. |
| Project absolute path (`/work/pi_larsonj_wit_edu/egank2_wit_edu/Mitigating-Sycophancy`) | scripts/slurm | All references rewritten to relative paths. |
| Author / collaborator names | manuscript files | Manuscript files (`paper.tex*`, `paper.pdf`, `paper.bbl`, `paper.aux`, `paper.log`, `paper.out`, `paper.md*`, `references.bib`, `neurips_2026.sty`, `checklist.tex*`) **excluded entirely**. |
| Acknowledgments section | `paper.tex` | Excluded with the manuscript. |
| Internal planning / handoff / proposal docs | `Research_Proposal.md`, `SESSION_HANDOFF.md`, `paper_todo.md`, `context.md`, `research.md`, `sycophancy-mech-interp-research.md`, `sycophancy-mechinterp-research.md`, `neurips-execution-plan.md`, `neurips-plan.md`, `notebooks/01_baseline_colab.ipynb`, `outputs/*.md`, `docs/*.md`, `PROJECT_OVERVIEW.md`, `QUICKSTART.md`, top-level `README.md` | All excluded. |
| WandB dependency | `requirements.txt` | Removed.  No code in the supplement imports `wandb`. |
| WandB run URLs / IDs | none detected in copied files | n/a |
| GitHub usernames / org URLs | none detected in copied files | n/a |
| API keys, tokens, credentials | none detected in any candidate file | n/a |
| `.git/`, `.cache/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info` | metadata caches | excluded |
| `git_hash` field in cached result JSONs | every result file's `metadata.git_hash` | overwritten with `null`.  The result numbers themselves are unchanged. |
| Tokenizer / chat-template / model-weight files | `results/*/*.safetensors`, `tokenizer*.json`, `chat_template.jinja`, `special_tokens_map.json` | excluded.  Reviewers should download upstream models from HuggingFace. |

## What was preserved

These are kept because they are necessary for verification and contain no
personal information:

- All cached numerical result JSON / CSV files needed to verify paper claims.
- `metadata.environment` blocks (Python / Torch / CUDA versions).
- Per-experiment manifests in `results/manifests/`.
- Pipeline scripts (with absolute paths replaced).
- The `src/` package.
- Public dataset prompts (Anthropic model-written-evals, GSM8k, TruthfulQA,
  MMLU references).  Only loader code is included; raw upstream datasets are
  pulled from their public sources.

## Known safe assumptions

- The figures (`figures/*.pdf|png`) were inspected visually for embedded
  metadata; matplotlib does not embed user paths in PDF/PNG output by default.
  The `tools/check_anonymization.py` scanner additionally parses the textual
  content of every file.
- All result JSONs were syntactically reparsed and re-serialized after
  `git_hash` redaction; this also strips any non-conforming whitespace or
  non-JSON debug artifacts that could carry identifying state.

## Remaining residual risks

| Risk | Mitigation | Status |
|------|-----------|--------|
| A reviewer infers institution from the model choices, dataset choices, or methodology | Inherent to the work; no further action possible. | Accepted. |
| Free-form audit sample contains transcripts in which the model self-identifies as Llama / Mistral / Qwen | The scorer's task is precisely to evaluate these; identity-of-the-model strings are not author-identifying. | Accepted. |
| Cached result file timestamps fall in a narrow window | Date stamps are stage-relative and not author-identifying.  No further action. | Accepted. |
| The `metadata.timestamp` in the JSONs reveals approximate working hours | Generic time stamps from research compute jobs; no author-locating signal. | Accepted. |

## How to re-verify

```
python tools/check_anonymization.py
```

The script returns exit code 0 only if every check passes.  See `tools/check_anonymization.py`
for the full pattern list and configurable blocklist at the top of the file.
