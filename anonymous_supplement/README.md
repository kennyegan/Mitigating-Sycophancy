# Anonymous Supplement — Mechanisms and Limits of Sycophancy Mitigation in Instruction-Tuned LLMs

This is a **minimal, documentation-only anonymous supplement** for the NeurIPS 2026
submission *"Mechanisms and Limits of Sycophancy Mitigation in Instruction-Tuned
LLMs."*

## What this supplement is

- A small bundle of aggregate result summaries, prompt and rubric files, compute
  documentation, and a manifest mapping each paper claim to a supporting artifact.
- All files are designed to be opened and read directly. **Reviewers are not
  expected to run any code.**

## What this supplement is **not**

- Not a runnable code release. Training scripts, evaluation scripts, model
  checkpoints, LoRA adapter weights, raw transcripts, training-pair JSONs, and
  WandB artifacts are intentionally excluded.
- Not a full reproduction package. Code, data preparation, and trained adapters
  will be released after acceptance per the paper's reproducibility statement.

## Layout

```
anonymous_supplement/
  README.md             — this file
  REPRODUCE.md          — what reviewers can verify here vs. what needs the full release
  COMPUTE.md            — compute requirements (fast doc verification vs. full rerun)
  ANONYMIZATION.md      — what was excluded for anonymity and why
  MANIFEST.md           — paper-claim → supplement-file → source-file table
  results/              — aggregate result JSONs, each with explicit provenance
  prompts/              — free-form benchmark prompts, judge system prompt, OOD summary
  rubrics/              — judge scoring rubric
```

## What each result JSON supports

| File | Paper claim | Status |
|---|---|---|
| `results/table1_llama_baseline.json` | Claim 1 (Llama-3 baseline rates) and Claim 2 (probe decomposition at Layer 1) | FOUND for rates and probe decomposition; SUMMARY_ONLY for the SC:BC bootstrap CI |
| `results/table3_dpo_sft.json` | Claim 4 (DPO vs. SFT, three seeds) | FOUND from cached multi-seed aggregate |
| `results/table4_generality_bounds.json` | Claims 5 and 6 (generality bounds across Mistral, Qwen, OOD, free-form) | PARTIAL — mix of FOUND and SUMMARY_ONLY |
| `results/patching_bootstrap.json` | Claim 3 (patching-to-ablation dissociation, top-3 Jaccard 0.09, ablation null) | FOUND aggregates from cached bootstrap; per-resample ablation deltas SUMMARY_ONLY |
| `results/mistral_dpo_eval_results.json` | Claim 5 (Mistral DPO failure case) | FOUND |
| `results/qwen_eval_results.json` | Claim 5 (Qwen baseline; ablation increases sycophancy) | PARTIAL — baseline FOUND, ablation SUMMARY_ONLY |
| `results/freeform_comparison_summary.json` | Claim 6 (free-form judge-scored result) | FOUND from cached 5,000-iteration bootstrap |

Every result JSON carries the same metadata block:

```json
{
  "artifact_type": "aggregate_result_summary",
  "raw_artifact_status": "FOUND | SUMMARY_ONLY | PARTIAL | MISSING",
  "provenance": "...",
  "paper_claim_supported": "...",
  "metrics": { ... },
  "notes": [ ... ],
  "not_raw_experimental_output": false
}
```

`not_raw_experimental_output` is `false` when values were copied from a cached
output that is itself a real evaluation summary, and would be `true` if a file
were generated from paper-reported numbers alone. None of the files in this
supplement are paper-only fabrications; SUMMARY_ONLY values within a FOUND or
PARTIAL file are explicitly flagged inline.

## Why some artifacts are SUMMARY_ONLY or MISSING

- **Mistral baseline opinion sycophancy.** The paper reports `82.5% -> 50.8%`
  for the Mistral DPO opinion delta. The cached
  `results/mistral/baseline_summary.json` artifact in the working repository
  reports `50.8%` for the baseline as well, so the `82.5%` baseline value used
  in the paper paragraph is not present as a standalone JSON in this repo.
  Both `results/mistral_dpo_eval_results.json` and
  `results/table4_generality_bounds.json` surface this discrepancy explicitly
  rather than silently picking one value.
- **Qwen top-3 ablation deltas (`+20.3 pp`, etc.).** The Qwen ablation
  evaluation log is parsed by `scripts/parse_qwen_ablation_log.py` in the
  working repository, but its parsed JSON output was not located in `results/`.
  The deltas are reproduced from the paper text and flagged
  `raw_artifact_status: "SUMMARY_ONLY"`.
- **Per-resample ablation deltas under bootstrap-resampled top-3 head sets.**
  The paper reports that the ablation null holds for every resample; the
  per-resample numbers are not present as a standalone JSON.
- **SC:BC ratio bootstrap CI `[1.4:1, 2.1:1]`.** The point estimate (1.79)
  comes directly from `results/probe_control_balanced_results.json`, but the
  raw 1,000-iteration bootstrap distribution itself is not present as a
  standalone file.

In every case where a value is SUMMARY_ONLY, the surrounding JSON says so.

## Anonymity statement

The authors and affiliations are intentionally omitted from this package, in
keeping with the NeurIPS double-blind review process. No filesystem paths from
the original working environment, no email addresses, no usernames, no
institutional cluster paths, no API keys, no experiment-tracker links, and no per-prompt
transcripts are included. See `ANONYMIZATION.md` for the full anonymization
checklist and remaining residual risks.

## Code and data release

Code, data preparation scripts, training and evaluation scripts, and
trained adapters will be released after acceptance, per the paper's
reproducibility statement. The full Llama-3 pipeline manifest in the working
repository validates that the cached artifacts referenced here were produced
by the same pipeline used in the paper.
