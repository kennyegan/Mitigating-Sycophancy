# Peer Review Evidence Brief: Mitigating Sycophancy in LLMs (Egan 2026)

**Prepared for:** Peer review of `paper.md`
**Author of reviewed paper:** Kenneth Egan, Wentworth Institute of Technology
**Date of this brief:** April 13, 2026
**Scope:** Metric verification, citation accuracy, novelty claims, statistical methodology, and unstated weaknesses.

---

## Section 1: Metric Spot-Check Against JSON Artifacts

### 1.1 `results/baseline_llama3_summary.json`

**Paper claims:** 28.0% overall, 82.4% opinion, 1.6% factual, 0.0% reasoning

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| Overall sycophancy | 28.0% | `0.28` | ✅ Exact |
| Opinion sycophancy | 82.4% | `0.8249` (from 410/497 samples) | ⚠️ Minor discrepancy |
| Factual sycophancy | 1.6% | `0.016` | ✅ Exact |
| Reasoning sycophancy | 0.0% | `0.0` | ✅ Exact |
| Mean compliance gap | −0.0435 | `−0.0435` | ✅ Exact |
| GSM8k compliance gap | −0.0083 | `−0.0083` | ✅ Exact |

**⚠️ Minor discrepancy — Opinion rate:** The baseline file evaluated 497 opinion samples (3 skipped out of 500), yielding 410/497 = 82.49% ≈ **82.5%**, not 82.4%. The paper's "82.4%" matches the count in ablation files (412/500), which used the full 500-sample domain. Both numbers are arithmetically defensible, but the paper should clarify which sample count it is reporting for the baseline table. This is a rounding inconsistency, not a fabrication.

**🚨 CRITICAL ERROR — Cohen's d labels reversed (Section 5.1):** The paper writes:
> "d = 0.18 (opinion vs. factual), d = 0.78 (opinion vs. reasoning)"

But the JSON shows:
```
"anthropic_opinion_vs_gsm8k_reasoning": 0.1836,
"anthropic_opinion_vs_truthfulqa_factual": 0.7782
```

So the correct statement should be **d = 0.18 (opinion vs. reasoning), d = 0.78 (opinion vs. factual)** — the labels are reversed in the paper. The Cohen's h statistics earlier in the same paragraph use the same variables and are internally consistent (h=2.276 opinion vs reasoning, h=2.022 opinion vs factual), but the d values in the following sentence have their domain labels swapped.

---

### 1.2 `results/probe_control_balanced_results.json`

**Paper claims:** Layer 1: transfer 77.9%, social compliance 18.0%, belief corruption 10.1%, robust 59.9%

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| Best layer | 1 | `1` | ✅ |
| Neutral CV accuracy | 89.0% | `0.89` | ✅ |
| Biased transfer accuracy | 77.9% [75.9%, 79.9%] | `0.7786...`, CI `[0.75865, 0.7986...]` | ✅ |
| Social compliance | 18.0% [16.0%, 19.9%] | `0.18`, CI `[0.16, 0.1986...]` | ✅ |
| Belief corruption | 10.1% [8.6%, 11.7%] | `0.10066...`, CI `[0.086, 0.1173...]` | ✅ |
| Robust tracking | 59.9% [57.4%, 62.3%] | `0.5986...`, CI `[0.574, 0.6226...]` | ✅ |
| Output accuracy (biased) | 71.9% | `0.7193...` | ✅ |
| Layer 2 transfer accuracy | 70.1% | `0.7013...` | ✅ |

All probe metrics verified exactly.

**Notable unpublished detail:** Transfer accuracy shows a non-monotone profile — Layer 1 (77.9%) is higher than Layers 0 (65.1%), 2 (70.1%), and 3 (70.4%), then Layer 8 and 9 recover to ~75–76% before declining again into the 50s at deep layers. The paper only reports Layer 1 as "best layer" and gives a brief layer 0–2 table. The recovery at Layers 8–9 (Layer 9: 76.0%) is not discussed and may be worth noting — it suggests two distinct windows of truth-tracking activity.

---

### 1.3 `results/head_importance.json`

**Paper claims:** Top-3: L4H28=0.443, L4H5=0.302, L5H31=0.256

| Head | Paper | JSON | Status |
|------|-------|------|--------|
| L4H28 | 0.443 | `0.4427...` | ✅ (rounded correctly) |
| L4H5 | 0.302 | `0.3019...` | ✅ |
| L5H31 | 0.256 | `0.2564...` | ✅ |
| L2H5 | 0.2445 | `0.2445...` | ✅ |
| L3H30 | 0.2182 | `0.2181...` | ✅ |

All top-10 head values verified. All std_dev > mean (e.g., L4H28: mean 0.443, std 0.550), confirming the variance warning in the paper.

**Notable:** L5H5, which appeared in the original ablation top-3, has a **negative** recovery score in the validated run (−0.237), meaning patching this head through its biased activation *increases* sycophancy. L1H20 similarly has a low positive score (0.040). This makes the original ablation experiment's targeting of L1H20 and L5H5 even more suspect than the paper implies.

---

### 1.4 `results/top10_ablation_full_gsm8k.json`

**Paper claims:** Ablated sycophancy 28.5%, MMLU 63.4%, GSM8k 29.9% (N=1319)

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| Baseline sycophancy | 28.0% | `0.28` | ✅ |
| Ablated sycophancy | 28.5% | `0.2846...` | ✅ |
| Sycophantic count (ablated) | 427/1500 | `427` | ✅ |
| MMLU (ablated) | 63.4% | `0.634` | ✅ |
| GSM8k (ablated, N=1319) | 29.9% (394/1319) | `0.2987...`, 394/1319 | ✅ |
| Baseline MMLU | 62.0% | `0.62` | ✅ |
| Baseline GSM8k | 33.2% (438/1319) | `0.3320...`, 438/1319 | ✅ |
| Opinion (ablated) | 82.8% | `0.828` | ✅ |

**Notable unpublished detail:** Truthful_factual sycophancy in the ablated condition is 2.6% vs 1.6% baseline (+1.0pp). The paper mentions this in its table but does not discuss whether this small increase in factual sycophancy under ablation is meaningful. The CI for factual at N=500 is wide enough that this is likely noise, but it's worth flagging.

---

### 1.5 `results/corrected_ablation_results.json`

**Paper claims:** All-3 ablation: 27.7% sycophancy

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| All-3 sycophancy | 27.7% | `0.2773...` | ✅ |
| All-3 opinion | 82.4% | `0.824` | ✅ |
| All-3 MMLU | 62.5% | `0.625250...` | ✅ |
| All-3 GSM8k | 37.5% | `0.375` | ✅ |
| Mean ablation result | "catastrophic" | `total_evaluated: 0, skipped: 1500` | ✅ |

**Notable unpublished detail:** The all-3 corrected ablation shows **GSM8k = 37.5%** (N=200), up from baseline 34.0%, a +3.5pp apparent improvement. The paper lists this in its table (37.5%) but does not comment on it. Given the wide CI for N=200 GSM8k (±7pp), this is almost certainly noise, but a reviewer might notice an unexplained positive direction in capability under ablation.

**Notable:** The MMLU baseline in this file is 62.17% (309/497, because 3 samples were evaluated on N=497 rather than N=500). Paper rounds to 62.2%. This is consistent.

---

### 1.6 `results/dpo_eval_results.json`

**Paper claims:** Post-DPO opinion sycophancy 58.6%, MMLU 62.8%, GSM8k 38.5% (N=200); Layer 1 probe: robust +15.6pp, social compliance −6.6pp, belief corruption −1.7pp

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| Opinion sycophancy (post-DPO) | 58.6% | `0.586` | ✅ |
| Overall sycophancy (post-DPO) | 19.6% | `0.196` | ✅ |
| MMLU (post-DPO) | 62.8% | `0.628` | ✅ |
| GSM8k (post-DPO) | 38.5% | `0.385` | ✅ |
| Robust tracking delta (L1) | +15.6pp | `0.15600...` | ✅ |
| Social compliance delta (L1) | −6.6pp | `−0.06599...` | ✅ |
| Belief corruption delta (L1) | −1.7pp | `−0.01733...` | ✅ |

**⚠️ Undisclosed methodological decision — Post-DPO best layer is 4, not 1:** The DPO eval results report `"best_layer": 4` for the post-DPO model, with best transfer accuracy 88.3% and best robust rate 76.3%. However, the paper's Table (pre-DPO vs post-DPO comparison) uses **Layer 1** for both conditions, justified as maintaining comparability with the pre-DPO analysis. This is methodologically defensible but the paper does not explicitly disclose that the best layer changed post-DPO. A reviewer could ask why the comparison wasn't done at the best layer for each model separately. The layer-4 post-DPO comparison would show even stronger effects: robust tracking 76.3% vs 54.4% pre-DPO (+21.9pp), social compliance 12.0% vs 18.7% (−6.7pp).

**Notable unpublished detail:** Post-DPO factual sycophancy drops from 1.6% to 0.2% (−1.4pp). The paper mentions this briefly but it's worth noting because it wasn't targeted by the DPO training (which used only opinion-domain pairs). This suggests at least some domain transfer, though the effect is small.

---

### 1.7 `results/dpo_training_metrics.json`

**Paper claims:** "train loss 0.69 → 0.16, rewards accuracy 95%; eval loss stabilized at 0.42 with no upward trend"

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| Epochs | 3 | `3` | ✅ |
| Learning rate | 5e-5 | `5e-05` | ✅ |
| DPO beta | 0.1 | `0.1` | ✅ |
| LoRA rank / alpha | 16 / 32 | `16 / 32` | ✅ |
| Train runtime | "under 3 minutes" | `117.3 seconds` | ✅ |
| Eval loss | ~0.42 | `0.42009...` | ✅ |
| **Training pairs** | **"400"** | **`n_train_pairs: 360, n_eval_pairs: 40`** | **⚠️** |
| **Final train loss** | **"0.69 → 0.16"** | **`train_loss: 0.356`** | **🚨** |

**🚨 CRITICAL DISCREPANCY — Train loss claim:** The paper states "train loss 0.69 → 0.16" but the JSON records a final `train_loss: 0.356`. If the loss trajectory genuinely reached 0.16 as a final step-level value, the epoch-averaged train loss would not be 0.356. The JSON appears to record a single epoch-averaged final value, not a step-level reading. This discrepancy is unexplained and cannot be resolved without the training curve logs. Possible interpretations:
1. The "0.16" was a single low-step loss reading during training, not the final epoch average
2. The claim refers to different loss scales (e.g., per-token vs. per-sample)
3. The paper is describing a different metric (e.g., rewards margin rather than cross-entropy loss)

The paper should clarify what "train loss 0.69 → 0.16" means relative to the stored `train_loss: 0.356`.

**⚠️ Potentially misleading — "400 DPO training pairs":** The JSON shows `n_train_pairs: 360, n_eval_pairs: 40`. The paper says "We generated 400 opinion-domain DPO training pairs" and also "fine-tuned Llama-3-8B-Instruct using LoRA... 3 epochs." Only 360 of the 400 pairs were used for training; 40 were held out for evaluation. The paper's statement "400 DPO training pairs" could be read as 400 training samples, when in fact only 360 were used. Should be stated as "360 training / 40 validation pairs from a pool of 400 generated pairs."

---

### 1.8 `results/mistral/baseline_summary.json`

**Paper claims:** 50.3% overall, 50.8% opinion, 99.8% factual

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| Overall | 50.3% | `0.5027` | ✅ (50.27% rounds to 50.3%) |
| Opinion | 50.8% | `0.508` | ✅ |
| Factual | 99.8% | `0.998` | ✅ |
| Reasoning | 0.2% | `0.002` | ✅ |
| MMLU | 50.6% | `0.506` | ✅ |
| GSM8k | 9.3% | `0.0932...` | ✅ |

All Mistral baseline metrics verified exactly.

**Notable unpublished detail:** The Mistral compliance gap for truthfulqa_factual is −0.1475, meaning the biased prompt *reduces* the probability of sycophancy in factual cases (counterintuitive given 99.8% sycophancy rate). This is because the factual sycophancy rate is so close to 100% that the compliance *gap* is near 0. The mean_compliance_gap of −0.1475 represents the distribution of gap values; at 99.8% base rate, the gap is constrained. The paper doesn't discuss this interpretation.

---

### 1.9 `results/mistral/top10_ablation_full_gsm8k.json`

**Paper claims:** +1.0pp ablation null (50.3% → 51.3%)

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| Baseline sycophancy | 50.3% | `0.5026...` | ✅ |
| Ablated sycophancy | 51.3% | `0.5126...` | ✅ |
| Change | +1.0pp | +1.0pp | ✅ |
| Mistral GSM8k (ablated) | not reported | `0.08794...` vs. `0.09325...` baseline | ⚠️ |

**⚠️ Unreported capability drop:** Under top-10 ablation, Mistral's GSM8k drops from 9.3% to 8.8% (−0.5pp). While small and non-significant given CIs, this is not reported in the paper. The paper only mentions MMLU for the Mistral ablation comparison ("confirming the null"). GSM8k for Mistral ablation should be reported for completeness.

---

### 1.10 `results/steering_results.json` (Spot-Check)

The steering results file is 3,300+ lines. The following were verified from the first 300 lines and the metadata:

- Baseline sycophancy: 28.38% ✅ (paper: 28.4%)
- N_steering_samples = 200, N_eval_samples = 1300 ✅
- Layers tested: [1,2,3,4,5,10,15,20] ✅
- Layer 1, alpha 2.0: `overall_sycophancy_rate: 0.28384...` = 0.0pp change ✅ (paper: table shows layer 1 alpha 5.0 as −0.5pp)
- Steering vector norms increase monotonically with layer (layer 1: 0.069, layer 20: 4.285), consistent with paper's discussion of why later-layer steering is more disruptive

Layer 15 alpha=2.0 and layer 10 alpha=50 values are deep in the file (not reached in spot-check) — these should be individually verified against the paper's table by the reviewer.

---

### 1.11 `results/full_rerun_manifest.json`

**Paper claims:** `missing_count: 0`

| Metric | Paper | JSON | Status |
|--------|-------|------|--------|
| missing_count | 0 | `0` | ✅ |
| artifact_count | not stated | `17` | ✓ |

**⚠️ Coverage gap:** The manifest covers 17 Llama-3 + DPO artifacts but does **not** include Mistral-specific artifacts (`results/mistral/probe_control_balanced_results.json`, `results/mistral/steering_results.json`, `results/control_groups/` artifacts). These exist in the filesystem but are not tracked by the manifest. The paper's claim "all findings replicate across architectures despite entirely different underlying circuits" rests partly on the Mistral results, which are not validated by the manifest. The paper's "missing_count: 0" claim is technically accurate for the 17 tracked artifacts but provides only partial coverage of the full experimental record.

---

## Section 2: Cited Concurrent Work — Accuracy Check

### 2.1 Li et al. (2025) — "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models"

**Found:** arXiv:2508.02087. Published August 2025.

**Paper's characterization:** "apply logit-lens and activation patching to Llama-3.1-8B-Instruct, finding that sycophantic output crystallizes in late layers (16–23). Their finding that early-layer representations retain truth while late-layer processing overrides it is conceptually equivalent to our social compliance characterization."

**Verdict:** ✅ **Accurate.** The Li et al. abstract describes "a two-stage emergence of sycophancy: (1) a late-layer output preference shift and (2) deeper representational divergence" using logit-lens analysis and causal activation patching on Llama-3.1-8B-Instruct. The characterization of early truth-retention vs. late-layer override is consistent. The model and method attribution are correct.

**Minor note:** Li et al. is an arXiv preprint submitted August 2025, after the Egan paper's listed experiments (March 2026 for most runs, April 2026 for DPO). This is correctly cited as concurrent work.

---

### 2.2 Chen et al. (2024) — "From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning"

**Found:** ICML 2024 (proceedings.mlr.press/v235/chen24u.html). Also in ACL anthology and OpenReview.

**Paper's characterization:** "use gradient- and activation-based module selection on Llama-2-Chat to identify sycophancy-related modules and apply Supervised Pinpoint Tuning (SPT) — targeted fine-tuning of only the identified modules while freezing the rest — achieving substantial sycophancy reduction. Notably, Chen et al. study challenge-induced sycophancy (the model reverses a correct answer under user pushback), while we study assertion-based sycophancy."

**Verdict:** ✅ **Accurate.** The paper correctly identifies: (a) gradient/activation-based module selection, (b) SPT methodology, (c) Llama-2-Chat as the target model, and (d) the distinction between challenge-induced vs. assertion-based sycophancy. This last distinction is important and correctly described.

---

### 2.3 O'Brien et al. (2026) — "A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy"

**Found:** arXiv:2601.18939, submitted January 26, 2026. NeurIPS 2025 Reliable ML Workshop.

**Paper's characterization:** "use sparse autoencoders on Gemma-2 to isolate ~3% of MLP neurons responsible for sycophancy, demonstrating successful neuron-level correction via gradient-masked fine-tuning."

**Verdict:** ⚠️ **Partially verifiable.** The OpenReview abstract confirms SAE-based sycophancy isolation on a small subset of neurons ("We find that sycophancy is controlled by a small set"), which is consistent with "~3%." The method described as "gradient-masked fine-tuning" cannot be directly confirmed from the abstract alone. The paper title "A Few Bad Neurons" is consistent with the ~3% claim. The specific percentage and exact method name should be verified against the full paper before publication. If "gradient-masked fine-tuning" is a paraphrase of a different method name used in the original, this could be a mischaracterization.

**Note on year:** Cited as "2026" which matches the arXiv submission date (January 2026). The workshop appearance was NeurIPS 2025. Both attributions are defensible.

---

### 2.4 Lee et al. (2024) — Mechanistic DPO analysis for toxicity

**Found:** "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity." ICML 2024 (arXiv:2401.01967).

**Paper's characterization:** Extends "analogous mechanistic DPO analyses for toxicity" — correctly characterizing Lee et al. as studying toxicity, not sycophancy.

**Verdict:** ✅ **Accurate.** Study clearly focuses on DPO and toxicity. The methodological analogy claimed by the paper (using probe analysis to understand what DPO changes internally) is appropriate.

---

### 2.5 Yang et al. (2024/2025) — Mechanistic DPO analysis for toxicity

**Found:** "How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis." arXiv:2411.06424, submitted November 2024; accepted EMNLP 2025 Main.

**Paper's characterization:** "extending analogous mechanistic DPO analyses for toxicity (Lee et al., 2024; Yang et al., 2024)" — correctly identifies as toxicity-focused, not sycophancy-focused.

**Verdict:** ✅ **Accurate on topic.** The paper is unambiguously about toxicity.

**⚠️ Inconsistent citation year:** The abstract uses "Yang et al., 2024" while the conclusion uses "Yang et al., 2025" for the same paper. The 2024 date matches the arXiv preprint submission; 2025 matches the EMNLP publication. Both are defensible, but a single citation style should be used consistently. Recommend using 2025 to match the official publication year.

---

### 2.6 Sharma et al. (2024) — Sycophancy characterization

**Found:** "Towards Understanding Sycophancy in Language Models." ICLR 2024 (arXiv:2310.13548, October 2023).

**Paper's characterization:** "sycophancy scales with model capability, is amplified by RLHF training, and occurs across opinion, factual, and reasoning domains. Their finding that models change correct answers when users express doubt directly motivates our hypothesis that the sycophantic circuit is localizable."

**Verdict:** ✅ **Accurate.** These characterizations are consistent with the Sharma et al. abstract and findings.

**Minor note:** The paper was submitted October 2023 (arXiv) and published ICLR 2024. Egan cites as "2024" (ICLR publication year), which is standard.

---

### 2.7 Heimersheim & Nanda (2024) — Sufficiency vs. Necessity

**Found:** "How to use and interpret activation patching." arXiv:2404.15255, April 2024. Also published on LessWrong.

**Paper's characterization:** "Heimersheim & Nanda (2024) formalize this distinction explicitly, defining denoising patching as a test of sufficiency and noising patching as a test of necessity, and warning against conflating the two — a caution our results empirically validate."

**Verdict:** ✅ **Accurate.** The Heimersheim & Nanda paper explicitly addresses the interpretation of different patching methodologies and is widely referenced in the mechanistic interpretability literature for precisely this distinction. The paper's framing of their own patching as a "sufficiency" test is consistent with Heimersheim & Nanda's framework.

---

### 2.8 Wei et al. (2023) — Reducing sycophancy via synthetic disagreement

**Found:** "Simple synthetic data reduces sycophancy in large language models." arXiv:2308.03958, August 2023 (Google DeepMind).

**Paper's characterization:** "Wei et al. (2023) showed that sycophancy can be reduced by training on synthetic disagreement examples, establishing a training-data mitigation baseline and supporting the view that sycophancy is a learned behavior rather than a fundamental capability limitation."

**Verdict:** ✅ **Accurate.** The paper directly matches this description — it studies synthetic SFT data as a sycophancy mitigation. The characterization is fair and not overstated.

---

## Section 3: Novelty Claim Evaluation

### 3.1 "First mechanistic evidence of how preference optimization resolves sycophantic behavior specifically"

**Claim location:** Abstract and Conclusion.

**Evidence for novelty:**

1. **Lee et al. (2024):** Confirmed to study DPO + **toxicity only**. No sycophancy analysis. ✅ Doesn't undermine novelty claim.

2. **Yang et al. (2024/2025):** Confirmed to study DPO + **toxicity only** (four models, MLP neuron analysis). No sycophancy analysis. ✅ Doesn't undermine novelty claim.

3. **No other paper found:** A literature search for "DPO sycophancy mechanistic probe analysis" and "preference optimization sycophancy internal representation" found no paper that applies probe decomposition to a DPO-trained model to study sycophancy specifically. The novelty claim appears **well-supported**.

4. **Chen et al. (2024):** Uses SPT (not DPO) and does not do probe-based mechanistic re-analysis post-training. Does not directly challenge the novelty claim.

**Verdict:** ✅ The "first mechanistic evidence" novelty claim for DPO on sycophancy is defensible based on available literature as of April 2026. The extension from toxicity (Lee, Yang) to sycophancy (Egan) appears to be a genuine first.

---

### 3.2 "Patching-to-ablation dissociation" — Novel framing?

**Assessment:** The concept that activation patching measures sufficiency rather than necessity is well-known and articulated by Heimersheim & Nanda (2024). What is novel is the empirical demonstration on a complex, safety-relevant behavior (sycophancy) with cross-architecture replication. The paper correctly credits Heimersheim & Nanda for the theoretical framing while claiming the empirical demonstration as its own contribution.

**Verdict:** The specific term "patching-to-ablation dissociation" is a novel framing, but the underlying concept is pre-existing. The paper appropriately attributes the theoretical foundation. The novelty lies in the empirical demonstration and the characterization that this property holds for sycophancy circuits across architectures.

---

## Section 4: Statistical Methodology Concerns

### 4.1 Power Calculation — N=1500, "80% power to detect ±3.6pp"

**Paper claim (Section 5.7):** "At N=1,500 and α=0.05, this test has 80% power to detect effects of approximately ±3.6 pp."

**Independent calculation:**

Using the standard two-proportion z-test power formula:
- H₀: p₁ = p₂ = 0.28 (baseline sycophancy)
- SE = √(2 · 0.28 · 0.72 / 1500) = √(0.0002688) = 0.01640

For 80% power at α=0.05 (two-sided, z_{α/2}=1.96, z_β=0.842):
- MDE = (z_{α/2} + z_β) × SE = 2.802 × 0.01640 = **0.0459 ≈ 4.6pp**

For 80% power at α=0.05 (one-sided):
- MDE = (z_α + z_β) × SE = (1.645 + 0.842) × 0.01640 = **0.0408 ≈ 4.1pp**

Using the single-proportion SE (treating baseline as fixed):
- SE = √(0.28 · 0.72 / 1500) = 0.01159
- MDE (two-sided) = 2.802 × 0.01159 = **0.0325 ≈ 3.25pp**

**Verdict:** ⚠️ **The ±3.6pp claim appears to be approximately correct only if using a single-proportion SE formula (which treats the baseline as fixed rather than an estimate).** Under a correct two-proportion z-test, the MDE at 80% power is ~4.5–4.6pp, not 3.6pp. The paper therefore overstates the sensitivity of its null result: the ablation null can actually rule out effects larger than ~4.5pp, not 3.6pp. This means the claim "we can rule out sycophancy reductions larger than 3.6 pp from top-10 ablation" is slightly too strong. However, the qualitative conclusion (the null is informative at this sample size) remains valid — sycophancy reductions of 5pp or more would be reliably detectable.

**Recommendation:** Recalculate using the two-proportion formula and report the correct MDE (~4.5pp). The substantive conclusion is unaffected.

---

### 4.2 Fisher's Exact Test — Appropriate for Social Compliance vs. Belief Corruption?

**Paper claim (Section 5.5):** "Fisher's exact test confirms the social compliance rate significantly exceeds belief corruption (p < 0.001, odds ratio 1.95)."

**Assessment:** Fisher's exact test is designed for 2×2 contingency tables of two independent proportions. In this application, the "social compliance" category (18.0% = 270/1500 samples) and "belief corruption" (10.1% = 151/1500 samples) are mutually exclusive categories assigned to the same pool of N=1500 samples. This means the two proportions are not independent — they share a common denominator and a sample cannot belong to both categories.

For comparing two mutually exclusive proportions from the same pool, the more appropriate tests are:
- **McNemar test** (if paired within subjects)
- **Chi-square goodness of fit** on the full 4-category distribution
- **Binomial test** comparing SC fraction vs. BC fraction among sycophantic samples

Fisher's exact test applied as described effectively constructs a table where:
- Row 1: SC=270, not-SC=1230
- Row 2: BC=151, not-BC=1349

This treats SC and BC as independent binary outcomes applied to the same 1500 samples, which could inflate significance. However, given the large effect size (18% vs. 10.1%, odds ratio ~1.95 from the paper), any reasonable test would yield p < 0.001. The qualitative conclusion is robust.

**Verdict:** ⚠️ **Technically suboptimal test choice.** Fisher's exact test is not the ideal choice for mutually exclusive categories from the same pool, but the conclusion is robust to test selection given the large effect size. Should be described more carefully — the 2×2 table structure should be explicitly stated in a methods supplement.

---

### 4.3 DPO GSM8k: N=200 (Post-DPO) vs N=1319 (Baseline)

**Paper claim:** "GSM8k accuracy: 33.2% → 38.5%†" with footnote "†Post-DPO GSM8k evaluated on N=200 vs. baseline N=1,319. 95% CI [32.0%, 45.4%] overlaps baseline; improvement not statistically significant."

**Assessment:** This is a significant methodological asymmetry that the paper acknowledges but does not adequately foreground. The 95% CI for the post-DPO GSM8k (N=200) spans [32.0%, 45.4%] — a 13.4pp width. The full-sample baseline of 33.2% falls within this interval. At N=200:
- Standard error = √(0.385 × 0.615 / 200) = 0.0344
- The comparison is statistically underpowered to detect even a 7pp improvement (power < 50%)

More importantly, the corrected ablation baseline (N=200, same seed) shows 34.0% GSM8k — very close to the DPO post value of 38.5%. The apparent DPO gain in GSM8k should be interpreted as noise, not improvement.

**Verdict:** ⚠️ **The asymmetric N for GSM8k is a material limitation.** The paper correctly notes non-significance in a footnote but presents the comparison in the main results table (33.2% vs 38.5%) without adequate visual emphasis on the incomparability. A reader skimming the table would see an apparent 5.3pp improvement with no notation of the 6× sample size difference. The table should more prominently flag this asymmetry, not relegate it to a footnote.

---

## Section 5: Unstated or Underemphasized Weaknesses

### 5.1 34/100 Patching Success Rate — Implications for Circuit Identification

**Paper treatment:** The 34% success rate is explained as "only samples where biased and neutral prompts produced meaningfully different outputs (total effect > 0.1)." The paper notes this is "a conservative but appropriate selection criterion."

**Reviewer concern:** A 34% success rate in activation patching means that for 66% of samples measured as sycophantic in the baseline (overall rate 28%), the forced-choice logit gap between biased and neutral conditions was insufficient (< 0.1) to enable patching analysis. This creates a potential **selection bias** in the circuit identification: the 34 samples that produced measurable patching effects may be atypical high-salience sycophancy cases, not representative of the broader distribution.

The practical consequence is that the "top 10 heads" identified by patching may be relevant primarily for the most extreme sycophantic responses, not for the typical sycophantic case. The head rankings and recovery scores are derived from an effective sample of N=34, with N=99 (using all samples, zero total-effect samples contribute near-zero recovery scores). Given the high variance (std > mean for all top heads), confidence intervals on individual head recovery scores would be very wide.

**The paper does not estimate confidence intervals for individual head recovery scores** — only for the aggregate ablation results. This omission weakens the precision of the circuit identification claims.

---

### 5.2 Binary Forced-Choice Measurement — Ceiling/Floor Effects

**Paper treatment:** Acknowledged in Limitation #2 (free-form generation). But the forced-choice format creates additional specific issues:

1. **Ceiling effect for opinion domain:** At 82.4% sycophancy, the opinion domain is near ceiling. DPO reducing this to 58.6% still leaves more than half of opinion questions sycophantic — yet the paper frames this as "substantial reduction." The remaining 58.6% is poorly characterized: is it the hardest cases, or random failures?

2. **Floor effect for reasoning domain:** The 0.0% reasoning sycophancy in forced choice may not generalize to open-ended mathematical reasoning, where sycophancy might manifest as agreeing with wrong explanations or hedging on correct answers.

3. **Forced choice may underestimate belief corruption:** If the biased prompt shifts the model's probability distribution from (60% A, 40% B) to (45% A, 55% B), this counts as sycophancy. But if the probe still detects A as the truth-tracking answer (correctly), the sample is "social compliance" — even though the internal representation shifted meaningfully. The binary classification scheme may mask continuous degrees of belief corruption.

---

### 5.3 DPO Training Scale — Only 360 Pairs

**Paper treatment:** The paper presents rapid convergence (117 seconds) as a positive indicator but does not flag the risks associated with training on only 360 preference pairs.

**Concerns not adequately discussed:**
1. **Overfitting to prompt template:** With 360 pairs all from the Anthropic model-written-evals format, the model may learn to recognize the specific "I think..." preamble format rather than learning a generalizable anti-sycophancy strategy.
2. **No multi-epoch validation curve:** The paper reports only final train_loss and eval_loss. With 360 pairs and 3 epochs, the model saw each pair 3 times. The paper claims "no overfitting despite rapid convergence" based only on the final eval_loss, without reporting per-epoch eval loss trends.
3. **The training appears to have converged very quickly** (117 seconds for 3 epochs on 360 pairs = ~39 seconds per epoch). For reference, this suggests extremely small batch steps — 360 pairs / batch_size 4 × gradient_accumulation 4 = 22.5 effective steps per epoch × 3 epochs = 67.5 total optimizer steps. This is far fewer than typical preference optimization training.
4. **Remaining 58.6% opinion sycophancy** suggests the small training set was insufficient for full mitigation, which the paper acknowledges but frames as a scope limitation rather than a training scale limitation.

---

### 5.4 Missing OOD Evaluation for DPO

**Paper treatment:** Acknowledged in Limitation #7 but characterized as an "open question." The limitation is more structural:

The evaluation set (500 opinion samples, seed=42) and training set (360 pairs, seed=100) both draw from the same data-generating distribution (Anthropic model-written-evals). Despite different seeds:
- Same question format (binary choice with "I think..." preamble)
- Same source data distribution
- Same domain (opinion questions)

This makes the evaluation **in-distribution by construction**. The 58.6% post-DPO rate should be interpreted as performance on the specific format the model was trained on, not as a general anti-sycophancy capability. Transfer to:
- Multi-turn sycophancy (agreement under pushback)
- Factual domain sycophancy (currently 0.2% — but this may reflect format limitations, not robustness)
- Open-ended generation sycophancy
- Non-Anthropic-format opinion questions

...remains entirely untested.

---

### 5.5 No Edge-Level (Path) Patching

**Paper treatment:** Acknowledged in Limitation #6. The paper notes this would "test whether the ablation null reflects genuine circuit redundancy or insufficient patching granularity."

**Additional implication not discussed:** The divergence between Egan's null ablation result and Chen et al.'s successful Supervised Pinpoint Tuning on Llama-2-Chat may be partly explained by **granularity of circuit identification**. Chen et al. use gradient-based attribution which identifies *specific weight directions* within modules, not just node-level activations. Egan's node-level patching identifies heads as units, but the relevant computation may be sub-head (specific attention patterns or query/key/value subspaces). This hypothesis is mentioned in the Discussion but the path patching experiment is deferred entirely. It is the single most important missing experiment for validating the "redundantly distributed circuit" interpretation.

---

### 5.6 Mistral GSM8k Baseline — Potential Data Quality Issue

**Notable issue not discussed in paper:** The Mistral baseline GSM8k accuracy is 9.3% (123/1319). While Mistral-7B-Instruct-v0.1 is known to underperform on mathematical reasoning relative to Llama-3-8B-Instruct, a 9.3% rate on GSM8k with strict numeric equality scoring is in the plausible range but raises a data quality question. The top-10 most sycophantic Mistral items show very low compliance gaps (max 0.25) compared to Llama-3 (max 0.96), suggesting Mistral is less decisive in its probability assignments. Cross-architecture "replication" with an architecturally weaker model (9.3% vs 33.2% GSM8k) may not fully replicate the underlying phenomena — the social compliance finding may be trivially true for a model with poor task performance.

---

### 5.7 Yang et al. Citation Year Inconsistency

**Minor issue:** The paper cites Yang et al. as "2024" in the abstract and at one point in Section 2, but "2025" in Section 5.11 (DPO key finding) and the Conclusion. This is the same paper (arXiv:2411.06424, EMNLP 2025). A consistent citation year should be used throughout.

---

## Section 6: Summary of Discrepancies

| # | Location | Type | Severity | Description |
|---|----------|------|----------|-------------|
| 1 | Section 5.1 | Data error | 🚨 High | Cohen's d labels reversed: paper says "d=0.18 (opinion vs. factual)" but JSON shows opinion-vs-reasoning = 0.18 and opinion-vs-factual = 0.78 |
| 2 | Section 5.11 / DPO metrics | Data inconsistency | 🚨 High | Paper claims "train loss 0.69 → 0.16" but JSON records final `train_loss: 0.356`; the 0.16 endpoint is unsupported by the stored artifact |
| 3 | Section 5.11 | Potentially misleading | ⚠️ Medium | Paper says "400 DPO training pairs" but only 360 were used for training (40 held for eval); should be clarified |
| 4 | Section 5.11 | Undisclosed decision | ⚠️ Medium | Post-DPO probe analysis reported at Layer 1 for comparability, but post-DPO best layer is actually Layer 4 (88.3% transfer) — this is defensible but should be disclosed explicitly |
| 5 | Section 5.7 | Statistical | ⚠️ Medium | Power calculation: paper claims MDE = ±3.6pp at 80% power; correct two-proportion z-test gives MDE ≈ 4.5–4.6pp |
| 6 | Section 5.1 | Minor rounding | ✍️ Low | Opinion baseline rate: baseline file shows 82.49% (410/497), paper uses 82.4% from 412/500; consistent with other files but worth noting |
| 7 | Section 5.5 | Test choice | ✍️ Low | Fisher's exact test for mutually exclusive categories is technically suboptimal; conclusion is robust |
| 8 | Section 5.11 | Presentation | ✍️ Low | GSM8k DPO comparison (N=200 vs N=1319) acknowledged in footnote but needs more prominent disclosure in table |
| 9 | Multiple | Inconsistency | ✍️ Low | Yang et al. cited as both "2024" and "2025" in different locations |
| 10 | Manifest | Coverage | ✍️ Low | `full_rerun_manifest.json` does not track Mistral artifacts; "missing_count: 0" applies only to Llama-3 + DPO pipeline |

---

## Section 7: Sources

### Kept (high-relevance primary sources)

- **Li et al. (2025)** — "When Truth Is Overridden" (arXiv:2508.02087) — Confirmed as cited concurrent work on logit-lens sycophancy in Llama-3.1-8B-Instruct
- **Chen et al. (2024)** — "From Yes-Men to Truth-Tellers" (ICML 2024, proceedings.mlr.press/v235/chen24u.html) — Confirmed as cited SPT paper on challenge-induced sycophancy
- **O'Brien et al. (2026)** — "A Few Bad Neurons" (arXiv:2601.18939, NeurIPS 2025 Workshop) — Confirmed as cited SAE-based sycophancy correction paper
- **Lee et al. (2024)** — "A Mechanistic Understanding of Alignment Algorithms: DPO and Toxicity" (ICML 2024, arXiv:2401.01967) — Confirmed as toxicity-only study
- **Yang et al. (2024/2025)** — "How Does DPO Reduce Toxicity?" (EMNLP 2025, arXiv:2411.06424) — Confirmed as toxicity-only study
- **Sharma et al. (2024)** — "Towards Understanding Sycophancy" (ICLR 2024, arXiv:2310.13548) — Confirmed characterization
- **Heimersheim & Nanda (2024)** — "How to use and interpret activation patching" (arXiv:2404.15255) — Confirmed as source for sufficiency/necessity framing
- **Wei et al. (2023)** — "Simple synthetic data reduces sycophancy" (arXiv:2308.03958) — Confirmed characterization

### Dropped

- **AAAI 2026 paper** (ojs.aaai.org) — Also uses logit-lens for sycophancy but appears to reference Li et al., not an independent work; insufficient information to assess independently
- **Gemma Scope (Lieberum et al. 2024)** — Related SAE infrastructure paper; not cited by Egan; not directly relevant

---

## Section 8: Gaps in This Research Brief

1. **Full steering_results.json verification:** Layer 15 alpha=2.0 and Layer 10 alpha=50 conditions could not be verified from the first 300 lines. The file requires full reading (3,377 lines) to confirm the specific values cited in Table 5.8 of the paper. The meta-information (baseline, layers tested, alpha values) is confirmed ✅.

2. **O'Brien et al. (2026) exact methodology:** The "~3% MLP neurons" and "gradient-masked fine-tuning" claims require reading the full paper to confirm precise numbers and terminology. The OpenReview abstract is consistent but does not contain these exact values.

3. **Mistral probe control balanced results:** The Mistral probe file (`results/mistral/probe_control_balanced_results.json`) was not checked. The paper's claim of "best layer 9: transfer accuracy 68.9%, social compliance 28.6%, belief corruption 4.5%"could not be directly verified.

4. **Training curve for DPO:** The `dpo_training_metrics.json` stores only final loss values, not per-step or per-epoch curves. The "0.69 → 0.16" claim requires either step-level logs or a training curve plot to verify or refute definitively. These were not found in the artifact directory.

5. **Control group patching artifacts:** The fictional-entity circuit comparison (zero overlap claim, L1H20 sign reversal) references `results/control_groups/patching_fictional/head_importance.json`, which was not read. These claims should be verified against that file.

---

*Brief compiled from direct inspection of 11 JSON artifact files, full reading of paper.md, and web searches for all 8 cited concurrent papers. All JSON comparisons performed against exact stored values, not summaries.*
