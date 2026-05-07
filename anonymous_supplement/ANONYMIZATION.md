# Anonymization Notes

This supplement was assembled in conservative "do-no-harm" mode. The goal was a
small, low-risk, reviewer-friendly documentation bundle. Where in doubt, we
preferred excluding material to including it.

## Categories of identifying information that were excluded

| Category | Excluded? | How |
|---|---|---|
| Author names, affiliations, ORCIDs | yes | not present anywhere in the supplement |
| Email addresses | yes | not present anywhere in the supplement |
| Code-host usernames or repo handles | yes | not present anywhere in the supplement |
| Institution, school, lab, or advisor names | yes | not present anywhere in the supplement |
| Cluster / shared-filesystem absolute paths from the working environment | yes | only relative paths within the working repository (e.g., `results/...`) appear in `provenance` strings; no absolute filesystem paths from the working environment |
| API keys, tokens, secrets | yes | not present; the judge configuration documents the rubric and decoding settings only |
| Experiment-tracker run URLs, project names, run IDs | yes | excluded |
| `.git` directory, `git config`, commit author lines | yes | not present (only short eight-character `git_hash` strings inside the cached JSONs were preserved as evaluation provenance, with no identifying metadata) |
| Per-prompt model transcripts and judge raw outputs | yes | excluded; only aggregate bootstrap statistics are surfaced |
| LoRA adapter weights / model checkpoints | yes | excluded |
| Training-pair JSONs (DPO and SFT) | yes | excluded |
| Raw caches and large logs | yes | excluded |
| `.pytest_cache`, `__pycache__`, build artifacts | yes | excluded |

## What scanning was performed

Before copying any file into `anonymous_supplement/`, the following pattern
checks were run against candidate sources:

- Common author-name and institutional substrings (author surname, lab PI
  surname, the institution domain, common email-handle variants, and
  `gmail`-style free-mail substrings).
- Cluster-path prefixes typical of the working environment (long absolute
  paths under shared-filesystem roots).
- Generic API-key patterns (e.g., long alphanumeric secret-key strings).
- Hosted-service URLs that would identify accounts (e.g., experiment-tracker
  hosts).

No identifying matches were found in the files copied into this supplement.

## Files that were inspected and intentionally **not** copied

| Source | Reason |
|---|---|
| `results/dpo_model*/` (LoRA adapter weights, tokenizers, checkpoints) | model artifacts; out of scope for a documentation supplement |
| `results/freeform/llama3_*_transcripts.jsonl`, `results/freeform/llama3_*_scores.jsonl` | raw per-prompt transcripts and per-prompt judge scores; only the aggregate `comparison_summary.json` is surfaced |
| `results/freeform/audit_sample.jsonl` | per-prompt audit file with full transcripts |
| `results/full_rerun_manifest.json`, `SESSION_HANDOFF.md`, internal notes (`paper_todo.md`, `notes/*.md`) | working notes and pipeline manifests with paths and task state from the original environment |
| `results/manifests/*.json`, `results/mistral/manifests/*.json` | run manifests with timestamps, runtime metadata |
| `multiautoresearch/`, `neurips2026_anonymous_supplement/` (a separate, larger supplement) | out of scope for this minimal package |
| `data/processed/master_sycophancy*.jsonl` | derived dataset built from public sources by the working repo's data preparation pipeline; reviewers can rebuild it from the public upstreams after code release |

## Files that were copied verbatim

| File | What it is |
|---|---|
| `prompts/freeform_*.jsonl` (5 files) | the 150 free-form benchmark prompt seeds, authored for this work, no identifiers |
| `prompts/FREEFORM_README.md` | the working repository's README for the free-form benchmark; no identifiers |
| `rubrics/freeform_judge_rubric.json` | the rubric file the judge sees; content-only |

## Files that were derived for this supplement

| File | What it is |
|---|---|
| `results/table1_llama_baseline.json` | aggregate values copied from `results/baseline_llama3_summary.json` and `results/probe_control_balanced_results.json`, with per-prompt rows and environment metadata stripped |
| `results/table3_dpo_sft.json` | aggregate values copied from `results/dpo_seed_summary.json` and per-seed eval files, with environment metadata stripped |
| `results/table4_generality_bounds.json` | mix of FOUND (Mistral, Qwen, OOD A/B, free-form) and SUMMARY_ONLY (Qwen ablation, Mistral baseline opinion); discrepancies surfaced rather than silently resolved |
| `results/patching_bootstrap.json` | aggregate Jaccard, layer-frequency, and top-3 head-frequency summary from the cached patching bootstrap; per-head recovery distributions omitted |
| `results/mistral_dpo_eval_results.json` | aggregate values copied from `results/mistral/dpo_eval_results.json`, with environment metadata stripped |
| `results/qwen_eval_results.json` | aggregate baseline values copied from `results/stronger/baseline_summary.json`; ablation deltas marked SUMMARY_ONLY |
| `results/freeform_comparison_summary.json` | wraps a verbatim copy of `results/freeform/comparison_summary.json` with a provenance block |

The `provenance` field on each result JSON identifies the source artifact in
the working repository's directory layout (e.g., `results/baseline_llama3_summary.json`).
These are paths within the working repository's `results/` tree and do not
expose the absolute filesystem path of the original environment.

## Residual risks

- **`git_hash` short strings.** Cached evaluation outputs were generated with a
  short eight-character commit hash recorded in their metadata (e.g.,
  `326a8b5a`). These were preserved on the result JSONs in the working
  repository and are mirrored here for evaluation provenance. They identify a
  commit only within the (yet-unreleased) project repository and do not link
  to a public host. We deemed this acceptable; if a reviewer wishes them
  removed entirely, the values can be replaced with `"<redacted>"` without
  affecting the reported metrics.
- **Stylistic fingerprinting.** The README, REPRODUCE, COMPUTE, and MANIFEST
  files were written de novo for this supplement; no working-repository
  documentation files were copied. The free-form benchmark README in
  `prompts/FREEFORM_README.md` is a verbatim copy from the working
  repository's `data/freeform/README.md`; it contains no identifiers but does
  describe the benchmark in the authors' own technical phrasing.
- **Public dataset persona names.** The OOD prompts (not included verbatim in
  this supplement, only summarized) contain fictional personas
  (e.g., "Sue", "John Doe", "Carissa Sanchez") that come from
  `Anthropic/model-written-evals`. These are upstream public dataset content
  and are not authors of this work.

## Files omitted entirely for anonymity

None of the files in `anonymous_supplement/` were redacted from a more
sensitive original. The boundary was drawn at the source-file level: files
that could not be safely included as a whole were not copied at all, and
their content was either summarized (with status `SUMMARY_ONLY` or
`PARTIAL`) or omitted with a note.

## Conclusion

This supplement was assembled to be safe to submit. The package contains:

- 5 Markdown documentation files
- 7 aggregate result JSONs in `results/`
- 1 rubric JSON + 1 rubric summary in `rubrics/`
- 5 free-form benchmark prompt JSONLs + 1 prompt summary + 1 judge prompt
  summary + 1 free-form README in `prompts/`

No author identifiers, no API keys, no experiment-tracker links, no cluster paths, no
checkpoints, no per-prompt transcripts, no training-pair JSONs.
