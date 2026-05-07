# OOD Sycophancy Benchmarks

Out-of-distribution sycophancy evaluation data from published benchmarks.

## Source

**Dataset:** [Anthropic/model-written-evals](https://huggingface.co/datasets/Anthropic/model-written-evals)
**Citation:** Perez et al. (2022), "Discovering Language Model Behaviors with Model-Written Evaluations" ([arXiv:2212.09251](https://arxiv.org/abs/2212.09251))

All data comes from the `sycophancy/` directory of the HuggingFace repository.

## Domains

| File | Source File | Samples | Description |
|------|-----------|---------|-------------|
| `nlp_survey.jsonl` | `sycophancy/sycophancy_on_nlp_survey.jsonl` | 500 | NLP researcher opinion questions from an NLP community survey |
| `political_typology.jsonl` | `sycophancy/sycophancy_on_political_typology_quiz.jsonl` | 500 | Political opinion questions from the Pew Political Typology Quiz |

### Note on PhilPapers 2020

The upstream repository lists a third file (`sycophancy_on_philpapers2020.jsonl`) intended
to contain philosophy survey questions from the PhilPapers 2020 survey. However, this file
is a duplicate of the NLP survey file — both point to the same blob hash
(`480ff822079406be617c21979a3848fb29e5c0e2`). This appears to be an upstream data issue in
the Anthropic repository. Only the two distinct datasets are included here.

## Why These Are OOD

DPO training used 400 pairs (seed=100) from the **generic** `sycophancy` category of
`Anthropic/model-written-evals`. The in-distribution evaluation uses 500 samples (seed=42)
from the same category.

The two subcategories above are **distinct files** in the repository, covering different
topic domains (NLP research opinions, political opinions) that were never seen during
training. They share the same biased/neutral forced-choice format (OOD-SAME-FORMAT), making
them directly comparable to our in-distribution evaluation.

## OOD Classification

- **OOD-SAME-FORMAT:** nlp_survey, political_typology
  Different topic domains, same biased (biography + opinion) vs. neutral (question only)
  forced-choice `(A)`/`(B)` structure.
- **OOD-DIFFERENT-FORMAT (excluded):** `are_you_sure.jsonl` from Sharma et al. (2024, ICLR,
  arXiv:2310.13548) uses multi-turn challenge-induced sycophancy. This format requires
  free-form generation and cannot be adapted to our forced-choice log-probability framework.
  Noted as future work.
- **Duplicate (excluded):** `sycophancy_on_philpapers2020.jsonl` — identical to NLP survey
  file in the upstream repository.

## Format Adaptations

Each raw Anthropic sample has:
- `question`: User biography expressing an opinion, followed by a question with `(A)`/`(B)` choices
- `answer_matching_behavior`: The sycophantic answer token (e.g., `" (A)"`)
- `answer_not_matching_behavior`: The honest answer token

Adaptations applied by `scripts/prepare_ood_benchmarks.py`:
1. **Neutral prompt extraction:** The biography/opinion preamble is stripped, keeping only the
   question stem and choice lines. For NLP survey, the question is on a separate line after
   the biography. For political typology, the question is embedded at the end of the biography
   line and is extracted using pattern matching.
2. **Llama-3 chat template:** Both neutral and biased prompts are wrapped in the
   `<|begin_of_text|>...<|eot_id|>` Llama-3 Instruct template.
3. **Single-token validation:** Only samples with single-token targets (e.g., `" (A)"`, `" (B)"`)
   are included.
4. **Deterministic sampling:** 500 samples per domain, selected with seed=42.

## Output Schema

Each line in the JSONL files:
```json
{
  "neutral_prompt": "<|begin_of_text|>...<|eot_id|>...",
  "biased_prompt": "<|begin_of_text|>...<|eot_id|>...",
  "sycophantic_target": " (A)",
  "honest_target": " (B)",
  "sample_id": "nlp_survey-a1b2c3d4e5f6",
  "metadata": {
    "source": "anthropic_nlp_survey",
    "category": "nlp_survey",
    "dataset": "Anthropic/model-written-evals",
    "ood_type": "ood_same_format"
  }
}
```

## Reproduction

```bash
# Prepare OOD benchmark data (requires internet for HuggingFace download)
python scripts/prepare_ood_benchmarks.py

# Run evaluation (requires GPU)
python scripts/09_ood_eval.py --adapter-path results/dpo_model
```

## Additional Files

- `metadata.json`: Processing statistics (raw counts, valid counts, skipped counts per domain)
