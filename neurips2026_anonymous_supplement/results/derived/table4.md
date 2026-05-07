# Table 4 — Cross-model generality bounds

Top-k zero-ablation column reports the change in overall sycophancy after
ablating the patching-identified top-k heads (all-to-zero), in percentage points.

| model | baseline overall | baseline opinion | top-k zero-ablation Δ (pp) | DPO opinion after | notes |
|-------|------------------|------------------|---------------------------|-------------------|-------|
| Llama-3-8B-Instruct | 0.280 | 0.825 | +0.47 | 0.571 | top-k = 10; DPO mean of 3 seeds |
| Mistral-7B-Instruct-v0.1 | 0.503 | 0.508 | +1.00 | 0.508 | DPO factual sycophancy → 1.000; GSM8k → 0.0 |
| Qwen-2.5-14B-Instruct | 0.282 | 0.753 | +20.33 | — | top-k = 3; ablation INCREASES sycophancy |

## OOD opinion retention vs. in-distribution DPO Δ

| condition | in-dist Δ (pp) | condition Δ (pp) | retention |
|-----------|----------------|------------------|-----------|
| condition_1 (New Anthropic samples (same template, different questions)) | -23.8 | -19.0 | 0.80 |
| condition_2 (Rephrased templates (different format, Anthropic questions)) | -23.8 | -20.5 | 0.86 |
| condition_3 (Manual diverse opinions (different format + different topics)) | -23.8 | -6.0 | 0.25 |
| all_ood (All OOD conditions combined) | -23.8 | -18.2 | 0.76 |
