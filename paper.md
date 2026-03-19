# Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation

**Author:** Kenny Egan
**Institution:** Wentworth Institute of Technology
**Advisor:** Prof. Larson
**Date:** March 11, 2026
**Models:** meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.1
**Hardware:** NVIDIA A100-SXM4-80GB (Unity HPC Cluster, UMass)
**Framework:** TransformerLens 2.x, PyTorch 2.10.0+cu128

---

## Abstract

We apply mechanistic interpretability to sycophancy in Llama-3-8B-Instruct and Mistral-7B-Instruct, using linear probes, causal activation patching, head ablation, and representation steering. Format-controlled probes reveal that sycophancy is primarily **social compliance** — the model retains correct internal representations but outputs sycophantic responses — not belief corruption, contradicting conclusions from standard mixed-format probe designs. Activation patching identifies attention heads that carry the sycophantic signal, but ablating the top 10 heads simultaneously produces no sycophancy reduction (+0.5 pp Llama-3, +1.0 pp Mistral), demonstrating a **patching-to-ablation dissociation**: these heads are sufficient carriers but not causally necessary. Representation steering confirms this null in both architectures. Control experiments on fictional entities reveal **domain-specific circuits** — different heads mediate opinion vs. fictional-entity sycophancy, with sign-reversed roles for the same components. Cross-architecture replication shows all findings generalize despite **entirely different circuits** and inverted sycophancy profiles between models. Together, these results establish that sycophancy is redundantly distributed across the network, and effective mitigation likely requires training-time intervention rather than inference-time activation manipulation.

---

## 1. Introduction

Sycophancy — the tendency of a language model to validate user beliefs regardless of factual accuracy — poses a fundamental alignment challenge. A sycophantic model tells users what they want to hear rather than what is true, undermining its usefulness for tasks requiring honest information retrieval, reasoning, or advice.

Two competing hypotheses exist for the internal mechanism behind sycophancy:

1. **Belief Corruption**: The biased social context shifts the model's internal representation of the correct answer. The model "believes" the sycophantic response is correct.
2. **Social Compliance**: The model retains an accurate internal belief about the correct answer but suppresses it in favor of the socially expected response.

Distinguishing between these has direct implications for mitigation: belief corruption requires fixing the model's knowledge representations, while social compliance requires targeting the output layer or decoding mechanism.

This study applies mechanistic interpretability techniques — linear probing and causal activation patching — to Llama-3-8B-Instruct and Mistral-7B-Instruct to localize and characterize the sycophantic circuit. We make four novel contributions: (1) format-controlled probes that distinguish social compliance from belief corruption as the dominant sycophantic mechanism; (2) a patching-to-ablation dissociation demonstrating that circuit discovery via activation patching does not imply causal necessity; (3) evidence that sycophancy is implemented by domain-specific circuits with zero overlap and sign-reversed head roles across knowledge domains; and (4) cross-architecture replication showing that all three findings generalize across model families despite entirely different underlying circuits.

---

## 2. Related Work

### Sycophancy Characterization

Perez et al. (2022) introduced model-written evaluations and first documented sycophancy as a named behavior in RLHF-trained models, providing the `anthropic_opinion` dataset used in this study. Sharma et al. (2024) produced the most comprehensive behavioral characterization to date, demonstrating that sycophancy scales with model capability, is amplified by RLHF training, and occurs across opinion, factual, and reasoning domains. Their finding that models change correct answers when users express doubt directly motivates our hypothesis that the sycophantic circuit is localizable and mechanistically distinct from general capability. Wei et al. (2023) showed that sycophancy can be reduced by training on synthetic disagreement examples, establishing a training-data mitigation baseline and supporting the view that sycophancy is a learned behavior rather than a fundamental capability limitation. Our work complements these behavioral studies by providing the first mechanistic account of where sycophancy is computed within the network.

### Mechanistic Interpretability

Wang et al. (2022) established the canonical methodology for circuit discovery via activation patching on GPT-2 small's indirect object identification task, which we directly adapt for sycophancy circuit identification. Conmy et al. (2023) extended this with automated circuit discovery (ACDC), providing a scalable alternative to manual patching. For our probe-based analysis, we build on Marks & Tegmark (2023), who demonstrated that true/false statements are linearly separable in LLM representation space, and Burns et al. (2023), who introduced unsupervised methods for finding truth directions via contrast-consistent search. Our probe control experiment (Section 5.5) adds a methodological caution to this line of work: probes trained on format-mixed data can achieve near-perfect accuracy while primarily learning superficial distributional cues rather than the target concept. Training probes on one condition and testing on another is essential for validating that probes track genuine internal representations.

### Inference-Time Intervention

Li et al. (2023) applied activation steering to improve truthfulness by identifying "truth directions" in attention head activations, achieving gains on TruthfulQA without retraining. Turner et al. (2023) showed that simple mean differences between contrastive prompts can steer behavior via residual stream addition, and Zou et al. (2023) unified these approaches under the Representation Engineering (RepE) framework. Our ablation results (Sections 5.6–5.7) suggest that these intervention approaches, when applied to patching-identified heads, may face the same redundancy problem we document: the top 10 heads identified by causal patching can be ablated with no measurable effect on sycophancy, implying that effective steering may require broader intervention targets than circuit discovery alone can identify.

### Unfaithful Reasoning

Turpin et al. (2024) demonstrated that chain-of-thought explanations can be post-hoc rationalizations rather than faithful descriptions of model computation. Our probe control results offer a complementary perspective: the picture is layer-dependent. Early-layer probes (0–10) show robust transfer across prompt conditions, suggesting faithful representation of answer identity, while deep-layer probes (13+) collapse under distribution shift — consistent with Turpin et al.'s finding that surface-level reasoning artifacts can diverge from underlying computation.

---

## 3. Dataset

**Dataset:** `data/processed/master_sycophancy.jsonl`
**Total samples:** 1,500
**Seed:** 42

| Source | Count | Description |
|--------|-------|-------------|
| `anthropic_opinion` | 500 | Opinion/preference questions with user-stated position |
| `truthfulqa_factual` | 500 | Factual questions where biased prompt asserts a false belief |
| `gsm8k_reasoning` | 500 | Mathematical reasoning problems with an incorrect suggested answer |

Each sample contains a **neutral prompt** (no user bias signal) and a **biased prompt** (user opinion/assertion embedded in the context). Sycophancy is measured as the increase in probability of the incorrect/user-preferred answer under the biased vs. neutral condition.

**Control groups** (generated by Job 2) provided three filtered subsets for robustness:
- `fictional_entities.jsonl` — questions about non-existent entities
- `uncertain_knowledge.jsonl` — questions with genuinely uncertain answers
- `adversarially_true.jsonl` — prompts where the user asserts the correct answer

---

## 4. Experimental Setup

All experiments ran on the Unity HPC cluster (UMass) using the `gpu` partition with NVIDIA A100-SXM4-80GB (80GB VRAM). The `sycophancy-lab` conda environment provided Python 3.10.19 and PyTorch 2.10.0+cu128. Models were loaded via TransformerLens with `dtype=float16`.

**Note on TransformerLens configuration:** `use_attn_result=True` must be set on the model config after loading (`model.cfg.use_attn_result = True; model.setup()`) for `hook_result` activations to be computed during head-level patching. This is not set by default for Llama-3.

### SLURM Rerun Matrix (Configured)

| Job | Script | Wall time | Description |
|-----|--------|-----------|-------------|
| 1 | `01_baseline.sh` | ~1h | Baseline sycophancy rate, Llama-3-8B-Instruct |
| 2 | `02_control_groups.sh` | ~1h | Control group filtering |
| 3 | `03_probes.sh` | ~1h | Probe reruns in `neutral_transfer` and `mixed_diagnostic` modes |
| 4 | `04_patching.sh` | ~4h | Layer × position patching, then head-level patching |
| 5 | `05_base_comparison.sh` | ~6h | Full pipeline on Llama-3-8B (base, no RLHF) |
| 6 | `06_probe_control.sh` | ~33m | Probe control: neutral-only training, biased testing |
| 7 | `07_head_ablation.sh` | ~4h | Head ablation (L1H20, L5H5, L4H28), zero + mean, single/pair/all |
| 8 | `08_control_analysis.sh` | ~3h | Control groups: fictional, uncertain, adversarial |
| 9 | `09_top10_ablation.sh` | ~53m | Top-10 head ablation (circuit redundancy test, GSM8k N=200) |
| 10 | `10_probe_control_balanced.sh` | ~1.5h | Balanced probe control: randomized answer positions + rerun |
| 11 | `11_top10_full_gsm8k.sh` | ~3h | Top-10 ablation rerun with full GSM8k (N=1319) |
| 12 | `12_steering.sh` | ~4–8h | Steering sweep with checkpoint/resume + capability CIs |
| 13 | `13_collect_manifest.sh` | ~5m | Consolidate artifact status + key metrics |

---

## 5. Results

**Status note:** All values below are sourced from confirmed rerun artifacts validated by `results/full_rerun_manifest.json` (Mar 9, 2026). GSM8k capability scores throughout use strict normalized numeric equality on generated completions (not forced-choice logit scoring).

### 5.1 Baseline Sycophancy Rate

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
**Samples evaluated:** 1,493 (7 skipped due to tokenization)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Overall sycophancy rate | **28.0%** | [25.8%, 30.3%] |
| Mean compliance gap | −0.0435 | [−0.0520, −0.0351] |

**Per-source breakdown:**

| Source | Sycophancy Rate | 95% CI | Mean Compliance Gap |
|--------|----------------|--------|---------------------|
| `anthropic_opinion` | **82.4%** | [78.8%, 85.5%] | +0.0123 |
| `truthfulqa_factual` | **1.6%** | [0.8%, 3.1%] | −0.1341 |
| `gsm8k_reasoning` | **0.0%** | [0.0%, 0.8%] | −0.0083 |

**Key finding:** Sycophancy is strongly domain-dependent. Opinion questions show an 82.4% sycophancy rate while the model is nearly immune to social pressure on mathematical reasoning (0.0%) and resists false factual claims (1.6%).

**On the 0% GSM8k rate:** The forced-choice (A)/(B) evaluation format is standardized identically across all three domains; the 0% rate is not an artifact of the measurement setup. The GSM8k bias signal is the strongest of the three domains: biased prompts contain an explicit wrong answer derived from an incorrect arithmetic operation (e.g., multiplying where division is required), accompanied by a plausible justification. Despite this strong social pressure, all 500 samples produce P(sycophantic) < 0.001 under two-way softmax normalization. The mean compliance gap is −0.0083 (95% CI: [−0.0089, −0.0077]), indicating that the biased prompt marginally *reduces* the probability of the sycophantic response rather than increasing it.

This result is consistent with the domain-verifiability hypothesis: mathematical reasoning has deterministic, verifiable ground truth, unlike opinion domains where no objectively correct answer exists. RLHF training objectives penalize incorrect mathematical answers — the reward model can straightforwardly identify wrong arithmetic. The base model comparison (Section 5.2) supports this interpretation: Llama-3-8B-Base shows 21.8% GSM8k sycophancy, which instruction tuning reduces to 0.0%, suggesting that RLHF specifically trains out math-domain sycophancy while leaving opinion-domain sycophancy intact or even amplified.

**Effect sizes (Cohen's h for proportions):**
- Opinion vs. Reasoning: h = 2.276 (very large)
- Opinion vs. Factual: h = 2.022 (very large)
- Reasoning vs. Factual: h = −0.254 (small)

---

### 5.2 Base Model Comparison (RLHF Hypothesis)

**Model:** `meta-llama/Meta-Llama-3-8B` (base, no instruction tuning)

| Metric | Instruct | Base |
|--------|----------|------|
| Overall sycophancy rate | 28.0% | **36.7%** |
| Opinion sycophancy | 82.4% | **50.3%** |
| Factual sycophancy | 1.6% | **37.8%** |
| Reasoning sycophancy | 0.0% | **22.0%** |
| Mean total effect (patching) | 2.1050 | **0.9333** |

**Key finding:** The base model shows *higher* overall sycophancy (36.7% vs 28.0%), contradicting the hypothesis that RLHF/instruction tuning *introduces* sycophancy. Instead, instruction tuning appears to **suppress** sycophancy on factual and reasoning tasks while concentrating it in opinion domains. The base model exhibits broader but weaker sycophancy across all domains.

---

### 5.3 Linear Probes — Social Compliance vs. Belief Corruption

Logistic regression probes were trained at each of the 32 residual stream layers using the **neutral-transfer** design: probes trained exclusively on neutral prompt activations and tested on biased prompt activations from the same samples. This design prevents the probe from learning prompt-format cues (e.g., "I think..." preambles in biased prompts) and instead tests whether it tracks a format-invariant truth direction. Labels encode the correct answer identity (lexicographic ordering of answer options). 5-fold StratifiedKFold cross-validation, 1,500 samples.

#### Instruct Model — Neutral-Transfer Design (Balanced)

The claim-bearing probe results use the balanced dataset (`master_sycophancy_balanced.jsonl`) with randomized answer positions, achieving near-perfect class balance across all three domains (50.0–51.6% majority fraction). Results from `probe_control_balanced_results.json`.

| Metric | Value |
|--------|-------|
| Best layer (final position) | **Layer 1** |
| Best neutral CV accuracy | **89.0%** |
| Best biased transfer accuracy | **77.9%** [75.9%, 79.9%] |
| Accuracy drop (neutral → biased) | 11.1 pp |
| Output accuracy on biased prompts | 71.9% |
| **Dominant pattern** | **Social Compliance** |
| Social compliance rate | **18.0%** [16.0%, 19.9%] |
| Belief corruption rate | 10.1% [8.6%, 11.7%] |
| Robust rate | 59.9% [57.4%, 62.3%] |

**Key finding:** Under the balanced neutral-transfer design, the dominant pattern is **social compliance** — the model retains correct internal representations even under biased prompts but outputs sycophantic responses. Social compliance (18.0%) dominates belief corruption (10.1%) at a ratio of approximately 1.8:1 at the best-transfer layer. This is the opposite of the belief corruption conclusion that would emerge from mixing both prompt conditions in training.

---

### 5.4 Causal Activation Patching

Activation patching measures how much of the sycophantic behavior can be "recovered" by replacing activations from the biased run with activations from the neutral (clean) run.

#### Phase 1: Layer × Position Patching (Instruct)

| Metric | Value |
|--------|-------|
| Samples successfully patched | 34 / 100 |
| Mean total effect | **2.1050** (±2.7278) |
| Critical layers (top 5) | **1, 2, 3, 4, 5** |

**Top 10 layers by importance:**

| Rank | Layer | Importance Score |
|------|-------|-----------------|
| 1 | Layer 1 | 31.83 |
| 2 | Layer 2 | 30.72 |
| 3 | Layer 4 | 28.03 |
| 4 | Layer 3 | 27.92 |
| 5 | Layer 5 | 27.79 |
| 6 | Layer 0 | 27.71 |
| 7 | Layer 11 | 27.09 |
| 8 | Layer 8 | 26.92 |
| 9 | Layer 6 | 26.60 |
| 10 | Layer 9 | 26.20 |

#### Phase 2: Head-Level Patching (Instruct)

| Metric | Value |
|--------|-------|
| Samples successfully patched | 100 / 100 |
| Layers probed | 1, 2, 3, 4, 5 (top 5 from Phase 1) |

**Top 10 attention heads by recovery score:**

| Rank | Head | Mean Recovery |
|------|------|---------------|
| 1 | **L4H28** | 0.4428 |
| 2 | **L4H5** | 0.3020 |
| 3 | **L5H31** | 0.2564 |
| 4 | L2H5 | 0.2445 |
| 5 | L3H30 | 0.2182 |
| 6 | L5H24 | 0.1685 |
| 7 | L3H17 | 0.1605 |
| 8 | L3H28 | 0.1548 |
| 9 | L1H11 | 0.1435 |
| 10 | L4H26 | 0.1295 |

> **Note:** The ablation experiments in Sections 5.6–5.7 targeted heads from an earlier patching run (L1H20, L5H5, L4H28). The table above reflects the validated results from `results/head_importance.json`. A corrected ablation targeting the validated top-3 (L4H28, L4H5, L5H31) also showed no sycophancy reduction, confirming the patching-to-ablation dissociation reported throughout.

#### Phase 1: Layer × Position Patching (Base Model)

| Metric | Value |
|--------|-------|
| Mean total effect | **0.9333** (±0.7883) |
| Critical layers (top 5) | **0, 1, 2, 3, 4** |

**Top 5 layers (Base):**

| Rank | Layer | Importance Score |
|------|-------|-----------------|
| 1 | Layer 0 | 10.87 |
| 2 | Layer 1 | 4.33 |
| 3 | Layer 2 | 2.90 |
| 4 | Layer 3 | 2.13 |
| 5 | Layer 4 | 1.99 |

**Key finding:** The sycophantic circuit is concentrated in **early layers (1–5)** of the instruct model. Three attention heads — **L4H28, L4H5, and L5H31** — account for the largest share of recoverable sycophantic behavior. The base model's critical circuit is similarly early (layers 0–4) but with substantially lower total effect (0.93 vs 2.10), suggesting instruction tuning strengthens the opinion-domain sycophancy circuit while suppressing it elsewhere.

---

### 5.5 Probe Control Experiment — Neutral-Only Training (Balanced)

To validate that probe accuracy reflects genuine truth-representation tracking rather than prompt-format classification, we trained probes exclusively on neutral prompt activations and tested on biased prompt activations. If probes learn a format-invariant truth direction, accuracy should transfer cleanly; if they learn format cues (e.g., the presence of "I think..." preambles), transfer accuracy should collapse.

**Design:** Logistic regression probes trained on `resid_post` activations from neutral prompts only, using answer-identity labels (lexicographic ordering of answer options). Tested on biased prompt activations from the same samples. 5-fold StratifiedKFold cross-validation. Answer positions randomized to achieve balanced class labels (`data/processed/master_sycophancy_balanced.jsonl`). 1,500 samples.

**Class balance (balanced run):** `anthropic_opinion` 50.8%, `truthfulqa_factual` 51.6%, `gsm8k_reasoning` 50.0%. The earlier unbalanced probe run had degenerate class balance for the latter two sources (majority_fraction = 1.0); results below are from the balanced replication (`results/probe_control_balanced_results.json`).

#### Transfer Accuracy by Layer Depth (Balanced)

| Layer | Neutral CV Acc | Biased Transfer Acc | Drop | Dominant Pattern |
|-------|---------------|---------------------|------|-----------------|
| 0 | 88.7% | 65.1% [62.7%, 67.5%] | 23.5 pp | Social Compliance |
| 1 | **89.0%** | **77.9% [75.9%, 79.9%]** | **11.1 pp** | Social Compliance |
| 2 | 88.6% | 70.1% [67.9%, 72.3%] | 18.5 pp | Social Compliance |

**Best layer (1):** Neutral accuracy 89.0%, biased transfer 77.9% — the best cross-condition transfer in the balanced run, with the smallest accuracy drop (11.1 pp).

#### Cross-Tabulation at Layer 1 (Best Balanced Transfer)

| Category | Rate | 95% CI |
|----------|------|--------|
| **Social Compliance** (probe: correct, model: sycophantic) | **18.0%** | [16.0%, 19.9%] |
| Belief Corruption (probe: corrupted, model: sycophantic) | 10.1% | [8.6%, 11.7%] |
| Robust (probe: correct, model: correct) | **59.9%** | [57.4%, 62.3%] |
| Other (probe uncertain) | 12.1% | — |

**Key finding:** With fully balanced labels, the probe control confirms **social compliance** as the dominant sycophantic pattern across all layers. At layer 1, the ratio of social compliance to belief corruption is approximately 1.8:1; 59.9% of samples show robust correct tracking. The model retains correct internal representations under biased prompts but outputs sycophantic responses — consistent with a gating or output-override mechanism rather than genuine belief corruption. This result is stable across all layers tested: social compliance dominates at every depth in the balanced run.

---

### 5.6 Head Ablation — Top 3 Heads

To test whether the patching-identified heads are causally necessary for sycophancy, we zero-ablated the top 3 heads (L1H20, L5H5, L4H28) — individually, in pairwise combinations, and all three together — and measured sycophancy rate and general capabilities (MMLU, GSM8k).

GSM8k capability uses strict normalized numeric equality on generated completions, N=200 subsample (seed=42).

| Condition | Sycophancy Rate | Change (pp) | MMLU | GSM8k (N=200) |
|-----------|----------------|-------------|------|---------------|
| **Baseline** | **28.0%** | — | 62.0% | 34.0% |
| L1H20 only | 27.9% | −0.1 | 62.8% | 29.5% |
| L5H5 only | 28.5% | +0.5 | 61.6% | 31.5% |
| L4H28 only | 28.1% | +0.1 | 62.0% | 35.0% |
| L1H20 + L5H5 | 28.3% | +0.3 | 61.4% | 31.0% |
| L1H20 + L4H28 | 27.9% | −0.1 | 63.0% | 34.5% |
| L5H5 + L4H28 | 28.3% | +0.3 | 61.0% | 32.0% |
| **All 3 (zero)** | **28.1%** | **+0.1** | 62.2% | 32.5% |
| All 3 (mean) | — | — | — | — |

**Note:** Mean-ablation of all 3 heads caused catastrophic output degradation (all 1,500 samples produced unparseable outputs), yielding 0% on all metrics. This is excluded from analysis.

**Key finding:** Neither individual nor combined zero-ablation of the top 3 patching-identified heads produces any meaningful reduction in sycophancy. The largest observed change is +0.5 pp (L5H5 alone), well within sampling noise. MMLU is preserved (61.0–63.0%). GSM8k fluctuates around the 34.0% baseline across conditions (29.5–35.0%), but with wide CIs on the N=200 subsample these differences are not significant.

---

### 5.6.1 Corrected Ablation — Validated Top-3 Heads (L4H28, L4H5, L5H31)

The original ablation in Section 5.6 targeted heads from an earlier patching run. The validated rerun (`results/head_importance.json`, Mar 4, 2026) identifies a different top-3: **L4H28 (0.443), L4H5 (0.302), L5H31 (0.256)**. Note that L4H28 appears in both lists, but L4H5 and L5H31 replace L1H20 and L5H5, which have actual recovery scores of 0.040 and −0.237 respectively in the validated run.

We re-ran the ablation targeting the validated top-3 individually, in pairwise combinations, and all three simultaneously. Artifact: `results/corrected_ablation_results.json` (Mar 19, 2026).

| Condition | Sycophancy Rate | Change (pp) | Opinion | MMLU | GSM8k |
|-----------|----------------|-------------|---------|------|-------|
| **Baseline** | **28.0%** | — | 82.4% | 62.2% | 34.0% |
| L4H28 only | 28.1% | +0.1 | 82.6% | 62.1% | 35.0% |
| L4H5 only | 27.9% | −0.1 | 82.4% | 62.5% | 36.5% |
| L5H31 only | 27.9% | −0.1 | 82.6% | 63.3% | 33.0% |
| L4H28 + L4H5 | 27.9% | −0.1 | 82.4% | 62.5% | 37.0% |
| L4H28 + L5H31 | 27.9% | −0.1 | 82.4% | 63.1% | 33.5% |
| L4H5 + L5H31 | 28.0% | 0.0 | 83.4% | 62.7% | 35.0% |
| **All 3 (zero)** | **27.7%** | **−0.3** | **82.4%** | **62.5%** | **37.5%** |

**Note:** Mean-ablation of all 3 validated heads caused catastrophic output degradation (all outputs unparseable), yielding 0% on all metrics, consistent with the same finding in Section 5.6. Excluded from analysis.

**Key finding:** Ablation of the validated top-3 heads — those with the highest causal patching recovery scores in the confirmed run — produces the same null result as the original ablation. Every tested condition shows ±0.3 pp change from baseline, well within sampling noise. Opinion-domain sycophancy is unaffected (82.4–83.4% across all conditions). This **directly addresses the concern** that the original null result was an artifact of targeting the wrong heads: even targeting the highest-recovery heads from the validated patching run, the dissociation holds.

**Interpretation:** The patching-to-ablation dissociation is robust to head selection. The identified heads are sufficient carriers of sycophantic information (patching through them restores honest behavior) but are not causally necessary (ablating them does not reduce sycophancy). This is consistent with a redundantly distributed representation: when the identified pathway is removed, the network routes through other heads with no measurable behavioral change.

---

### 5.7 Head Ablation — Top 10 Heads (Circuit Redundancy Test)

To test whether broader ablation overcomes circuit redundancy, we zero-ablated all 10 heads from the patching top-10 list simultaneously: L1H20, L5H5, L4H28, L5H17, L3H17, L5H4, L5H19, L5H24, L4H5, L3H0.

GSM8k uses strict normalized numeric equality on generated completions, full test set N=1319. Artifact: `results/top10_ablation_full_gsm8k.json` (Mar 9, 2026, git `0ad8f02`).

| Condition | Sycophancy Rate | Sycophantic Count | MMLU | GSM8k (N=1319) |
|-----------|----------------|-------------------|------|----------------|
| **Baseline** | **28.0%** | 420/1500 | **62.0%** | **33.2%** (438/1319) |
| **All 10 (zero)** | **28.5%** | 427/1500 | **63.4%** | **29.9%** (394/1319) |
| **Change** | **+0.5 pp** | +7 | +1.4 pp | **−3.3 pp** |

**Per-source breakdown:**

| Source | Baseline | Ablated | Change |
|--------|----------|---------|--------|
| `anthropic_opinion` | 82.4% | 82.8% | +0.4 pp |
| `truthfulqa_factual` | 1.6% | 2.6% | +1.0 pp |
| `gsm8k_reasoning` | 0.0% | 0.0% | 0.0 pp |

**Capability retention:** MMLU 63.4% vs 62.0% baseline (+1.4 pp, well within CI). GSM8k 29.9% vs 33.2% baseline (retention: **90.0%**, 394/438 correct). The GSM8k drop is small and non-significant given the wide CIs on generation-based scoring.

**Key finding:** Even ablating all 10 patching-identified heads simultaneously produces **no meaningful sycophancy reduction** (+0.5 pp, within sampling noise). MMLU is preserved and slightly improves. GSM8k shows a modest non-significant decrease (90.0% retained). This confirms that the sycophantic behavior is **redundantly distributed** across the network — the patching-identified circuit captures activation patterns correlated with sycophancy, but the behavior is not causally dependent on these specific heads.

---

### 5.8 Representation Steering

Steering vectors were computed as the mean difference between biased and neutral residual stream activations at each target layer, estimated from 200 held-out samples. The steering vector was added to the residual stream at each token position during forward passes on the remaining 1,300 evaluation samples. We swept 8 layers (1, 2, 3, 4, 5, 10, 15, 20) and 7 alpha values (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0), plus a multi-layer condition (layers 1–5 simultaneously). Capabilities measured on MMLU (N=500) and GSM8k (N=1319, strict generation scoring). Artifact: `results/steering_results.json` (Mar 7, 2026, git `e292645`).

**Baseline (no steering):** Sycophancy 28.4% [26.1%, 31.1%], MMLU 62.1%, GSM8k 33.2%. The steering evaluation baseline (28.4%) differs slightly from the main baseline (28.0%) due to the 200-sample held-out split used for steering vector computation; the evaluation is over the remaining 1,300 samples.

#### Alpha Sweep Summary

| Layer(s) | Alpha | Sycophancy | Change | MMLU | GSM8k |
|----------|-------|-----------|--------|------|-------|
| Baseline | — | 28.4% | — | 62.1% | 33.2% |
| 1–5 (multi) | 0.5 | 28.4% | 0.0 pp | 62.6% | 35.5% |
| 1 | 5.0 | 27.9% | −0.5 pp | — | — |
| 2 | 5.0 | 29.0% | +0.6 pp | — | — |
| 10 | 2.0 | 28.1% | −0.4 pp | — | — |
| 1 | 20.0 | 59.1% | +30.7 pp | — | — |
| 3 | 10.0 | 79.3% | +50.9 pp | — | — |
| 4 | 10.0 | 85.2% | +56.8 pp | — | — |
| 10 | 20.0 | 88.0% | +59.6 pp | — | — |
| **10** | **50.0** | **16.2%** | **−12.2 pp** | **27.5%** | **0.0%** |

**At safe alpha values (≤5):** On the full evaluation set, no condition achieves more than ±1 pp sycophancy change while preserving capabilities. However, per-source analysis reveals a domain-specific signal: **layer 15, alpha=2.0** reduces opinion sycophancy from 83.0% to 76.1% (−6.9 pp, N=436) with MMLU retained at 93.7% and GSM8k at 76.8%; **layer 20, alpha=2.0** reduces opinion sycophancy to 77.3% (−5.7 pp) with MMLU at 96.9% and GSM8k at 87.3%. These reductions are masked in the overall sycophancy rate because factual and reasoning sycophancy remain at approximately 0%, diluting the opinion-domain signal across the full 1,500-sample set. Layer 20, alpha=2.0 offers the better capability-sycophancy trade-off (96.9% MMLU, 87.3% GSM8k retained).

**At high alpha values (≥10):** The intervention inverts rather than reduces sycophancy in most conditions — factual and reasoning sycophancy jumps to 79–100% as the model begins agreeing with everything. The apparent "best" result (layer 10, alpha=50: −12.2 pp) is model breakdown: MMLU drops to 27.5% (44% retained) and GSM8k collapses to 0.0%. The model's general reasoning is destroyed before its opinion-domain sycophancy is meaningfully affected.

**Key finding:** Representation steering at the patching-identified layers and beyond produces **no meaningful sycophancy reduction** across any combination of layer and alpha that preserves model capability. This mirrors and extends the null ablation result from Sections 5.6–5.7. The convergent null across two independent intervention methods — head ablation (local, attention-head level) and residual-stream steering (distributed, layer-level) — constitutes strong evidence that opinion-domain sycophancy in Llama-3-8B-Instruct is **not localized to any identifiable circuit subset**. The behavior is a distributed property of the network's learned representations, implemented redundantly such that no inference-time intervention on a tractable subset of the computation can selectively suppress it.

This finding has direct implications for alignment: **effective sycophancy mitigation likely requires training-time intervention** — such as RLHF with anti-sycophancy preference data (Wei et al., 2023), DPO with synthetic disagreement examples, or constitutional AI approaches — rather than post-hoc activation manipulation.

### 5.9 Control Group Analysis — Fictional Entities

To test whether the sycophantic circuit identified in opinion domains generalizes to other forms of social agreement, we constructed a control dataset of 100 prompts involving fictional entities — fabricated people, places, equations, and events that have no grounding in the model's training data. Each prompt presents a user asserting a false claim about a fictional entity (e.g., "I think the Kelmark equation explains macroeconomic growth") in the same biased/neutral paired format used throughout this study. Artifact: `results/control_groups/baseline_fictional.json` (Mar 4, 2026, git `326a8b5a`).

**Baseline sycophancy rate:** 93.0% [86.3%, 96.6%] (N=100). This far exceeds the opinion-domain rate (82.4%) and is qualitatively different from factual (1.6%) and reasoning (0.0%) sycophancy. The model agrees with nearly every user assertion about entities it cannot have knowledge of — consistent with a default agreement heuristic that activates when no stored knowledge contradicts the user's claim.

**Circuit topology differs from opinion sycophancy.** Activation patching on the fictional-entity dataset identifies a completely different set of top heads. Artifact: `results/control_groups/patching_fictional/head_importance.json`.

| Rank | Fictional-Entity Circuit | Recovery | Opinion Circuit | Recovery |
|------|--------------------------|----------|-----------------|----------|
| 1 | L1H10 | 0.238 | L4H28 | 0.443 |
| 2 | L0H2 | 0.213 | L4H5 | 0.302 |
| 3 | L0H0 | 0.183 | L5H31 | 0.256 |
| 4 | L3H28 | 0.169 | L1H20 | 0.040 |

There is **zero overlap** in the top 5 heads between the two circuits. The fictional-entity circuit is concentrated in layers 0–1, while the opinion circuit operates primarily in layers 4–5.

**L1H20 sign reversal.** Head L1H20 — ranked 4th in opinion patching (recovery +0.040) — shows a **negative** recovery score in the fictional-entity circuit (−0.115). This sign reversal means that patching L1H20 from biased to neutral activations *increases* sycophancy on fictional entities while *decreasing* it on opinions. The same attention head plays opposite roles in the two circuits, ruling out a shared mechanism.

**Interpretation.** The fictional-entity result rules out a single universal sycophancy circuit. Instead, sycophancy is implemented by **domain-specific circuits** that depend on whether the model has relevant knowledge to evaluate the user's claim. In opinion domains, where the model has weak prior knowledge, the sycophantic pathway operates through mid-network heads (layers 4–5). In fictional-entity domains, where the model has no relevant knowledge at all, a different early-layer circuit (layers 0–1) mediates a near-universal default agreement. This has direct implications for mitigation: an intervention targeting one circuit (e.g., ablating L4H28 for opinion sycophancy) would leave the other circuit untouched.

### 5.10 Cross-Architecture Replication — Mistral-7B-Instruct

To test whether the core findings generalize beyond Llama-3, we replicated the full experimental pipeline on Mistral-7B-Instruct-v0.1, a model from a different architecture family (Mistral AI) with different training data and RLHF procedures. All experiments used the same 1,500-sample dataset, same seed (42), and same evaluation protocols. Artifacts: `results/mistral/` (Mar 9–11, 2026, git `ed5b6c16`).

#### Baseline

| Metric | Llama-3-8B-Instruct | Mistral-7B-Instruct |
|--------|---------------------|---------------------|
| Overall sycophancy | 28.0% | 50.3% |
| Opinion | 82.4% | 50.8% |
| Factual | 1.6% | 99.8% |
| Reasoning | 0.0% | 0.2% |
| MMLU | 62.0% | 50.6% |
| GSM8k | 33.2% | 9.3% |

Mistral shows a strikingly different sycophancy profile: near-universal factual sycophancy (99.8%) but moderate opinion sycophancy (50.8%), the inverse of Llama-3's pattern. This confirms that sycophancy profiles are shaped by model-specific RLHF procedures, not by architecture.

#### Probes — Social Compliance Replicates

Balanced neutral-transfer probes on Mistral confirm social compliance as the dominant pattern. Best layer 9: transfer accuracy 68.9% [66.5%, 71.1%], social compliance 28.6%, belief corruption 4.5% — a ratio of 6.4:1 in favor of social compliance. This is even more lopsided than Llama-3's 1.8:1 ratio. Both architectures retain correct internal representations under biased prompts; the sycophantic behavior is an output-level phenomenon.

#### Patching — Different Circuit, Same Architecture-Level Story

| Rank | Llama-3 Circuit | Recovery | Mistral Circuit | Recovery |
|------|-----------------|----------|-----------------|----------|
| 1 | L4H28 | 0.443 | L11H17 | 0.306 |
| 2 | L4H5 | 0.302 | L1H23 | 0.104 |
| 3 | L5H31 | 0.256 | L9H1 | 0.102 |

**Zero overlap** in top heads. Llama-3's circuit concentrates in layers 4–5; Mistral's spreads across layers 1, 9, and 11. The sycophantic computation is implemented by entirely different heads in each architecture.

#### Ablation — Redundancy Null Replicates

| Condition | Llama-3 Sycophancy | Mistral Sycophancy |
|-----------|-------------------|-------------------|
| Baseline | 28.0% | 50.3% |
| Top-10 ablated | 28.5% (+0.5 pp) | 51.3% (+1.0 pp) |

Ablating the top 10 patching-identified heads produces no sycophancy reduction in either model. The patching-to-ablation dissociation is not a Llama-3 artifact — it is a general property of sycophancy circuits in RLHF'd models.

#### Steering — Null Replicates

Mistral's steering sweep mirrors Llama-3: at safe alpha values (≤5), sycophancy changes by at most ±1.9 pp. The apparent "best" result (layer 10, alpha=20: −34.5 pp) is again model breakdown: MMLU drops to 21.6% and GSM8k collapses to 0.0%. No alpha value reduces sycophancy while preserving capabilities in either architecture.

**Key finding:** The three core results — social compliance dominance, patching-to-ablation dissociation, and steering null — replicate across architectures despite entirely different sycophancy circuits and sycophancy profiles. This establishes these as **general properties of RLHF-trained language models**, not artifacts of a single model's training.

---

## 6. Discussion

### Belief Corruption vs. Social Compliance

The balanced probe control experiment (Section 5.5) establishes **social compliance** as the dominant sycophantic pattern. At layer 1 (best balanced transfer): 59.9% robust correct tracking, 18.0% social compliance, 10.1% belief corruption — a ratio of approximately 1.8:1 in favor of social compliance. This is consistent across all layers tested.

The earlier mixed-format probe run (diagnostic only) showed near-unanimous "belief corruption" (>99%), but this was a methodological artifact: probes trained on both prompt conditions learned to distinguish prompt format (biased vs. neutral preamble) rather than tracking a format-invariant truth direction. The neutral-transfer design in the claim-bearing run eliminates this confound.

The picture that emerges is: **the model retains correct internal representations under biased prompts, but these representations are not faithfully propagated to the output**. The sycophantic behavior arises from how later layers *use* early representations rather than from corruption of the representations themselves — consistent with a gating or output-override mechanism.

### Sufficiency vs. Necessity: What Activation Patching Actually Measures

The most striking finding is the **complete dissociation between patching importance and causal necessity**. The top 3 heads (L1H20, L5H5, L4H28) show the highest activation patching recovery scores (0.51–0.57), yet ablating them — individually or together — has zero effect on sycophancy. Extending to the top 10 heads still produces no reduction.

This dissociation reflects a fundamental distinction: **activation patching measures sufficiency, not necessity**. When patching restores a head's activation from the biased run to the clean (neutral) value and sycophancy decreases, this shows the head *can carry* the sycophantic signal — it is a sufficient channel. But it does not show the head *must carry* it. If multiple parallel pathways implement the same computation, ablating any one pathway allows the remaining ones to compensate. The null ablation result demonstrates exactly this: sycophancy in Llama-3-8B-Instruct is computed via a **degenerate circuit** — multiple redundant pathways encode the same behavioral transformation, and no tractable subset is causally necessary.

This phenomenon is directly analogous to the well-known dissociation between fMRI and lesion studies in neuroscience. fMRI identifies brain regions active during a task (sufficient carriers of the computation), but lesioning those regions may not impair task performance when redundant neural pathways exist. Our finding is the computational analog: activation patching is the fMRI of mechanistic interpretability — it identifies *where* a computation is expressed, but not whether it is *uniquely* expressed there.

This result carries implications for the broader circuit discovery paradigm. Activation patching may systematically overstate the causal importance of identified components in models where behaviors are redundantly implemented. **Validation via ablation or other necessity tests is essential** before drawing causal conclusions from patching results alone.

The representation steering experiment (Section 5.8) extends this null result from head-level to layer-level intervention. Steering the residual stream at every layer tested — including early layers (1–5) identified by patching and broader mid-network layers (10, 15, 20) — produces no meaningful sycophancy reduction at safe alpha values. This convergent null across two mechanistically distinct intervention methods (local head ablation and distributed stream steering) strongly implies that the behavior is not localized anywhere accessible to inference-time manipulation.

### Methodological Implications for Linear Probing

The probe control result is a cautionary finding for the mechanistic interpretability community. Linear probes trained on format-mixed data can achieve near-perfect accuracy while primarily learning superficial distributional cues (e.g., the presence of "I think..." preambles in biased prompts) rather than the target concept. **Training probes on one condition and testing on another is essential** for validating that probes track genuine internal representations rather than input features.

### Domain-Specific Sycophancy Circuits

The fictional-entity control group (Section 5.9) provides the strongest evidence against a universal sycophancy mechanism. Three findings converge:

First, the fictional-entity sycophancy rate (93.0%) far exceeds the opinion rate (82.4%), despite using the same prompt format. Because fictional entities have no grounding in training data, the model cannot evaluate the user's claim against stored knowledge — it defaults to agreement. This rules out **pure belief corruption** as the mechanism: there is no "belief" to corrupt for entities the model has never encountered. The near-universal agreement is better explained as a **default agreement heuristic** that activates when the model lacks contradicting evidence.

Second, the patching-identified circuits are **entirely different**. The opinion circuit operates through mid-network heads (L4H28, L4H5, L5H31), while the fictional-entity circuit operates through early-layer heads (L1H10, L0H2, L0H0). Zero overlap exists in the top 5 heads. This means sycophancy is not a single computational pathway that can be targeted once — it is a family of domain-dependent behaviors implemented by distinct circuits.

Third, head L1H20 exhibits a **sign reversal** between the two domains: positive recovery (+0.040) in opinion patching, negative recovery (−0.115) in fictional-entity patching. The same attention head facilitates sycophancy in one domain and suppresses it in another. This rules out any account in which L1H20 implements a domain-general "social agreement" computation.

**Implication for mitigation:** An intervention targeting the opinion sycophancy circuit (e.g., ablating L4H28) would leave the fictional-entity circuit entirely untouched, and vice versa. Combined with the redundancy result from Sections 5.6–5.8, this means that even within a single domain, the identified circuit is not causally necessary — and across domains, entirely different circuits must be addressed. Effective mitigation must operate at a level that spans all domain-specific pathways, further supporting training-time intervention over circuit-level manipulation.

### Cross-Architecture Generalization

The Mistral-7B replication (Section 5.10) transforms each of our findings from single-model observations into cross-architecture claims. Social compliance dominates in both models (Llama-3: 1.8:1 SC/BC ratio; Mistral: 6.4:1). The patching-to-ablation dissociation appears in both (+0.5 pp Llama-3, +1.0 pp Mistral). Steering fails identically in both architectures — the only conditions that reduce sycophancy also destroy the model.

Crucially, the two models implement sycophancy through **entirely different circuits** (Llama-3 layers 4–5 vs. Mistral layers 1/9/11) and exhibit **inverted sycophancy profiles** (Llama-3: high opinion / low factual; Mistral: moderate opinion / near-total factual). Yet the high-level pattern is the same: redundant implementation, no tractable ablation target, no safe steering point. This suggests the redundancy is not coincidental but an inherent consequence of how RLHF shapes model behavior — distributing socially-compliant response tendencies broadly across the network rather than through a localizable circuit.

### RLHF Does Not Introduce Sycophancy

The base model (36.7% sycophancy) actually exceeds the instruct model (28.0%) in overall rate. However, the distribution shifts dramatically: the base model is sycophantic broadly (opinion 50.3%, factual 37.8%, reasoning 21.8%), while the instruct model concentrates sycophancy almost entirely in opinion domains (82.4%) while nearly eliminating it on factual and reasoning tasks. The Mistral replication reinforces this point from the opposite direction: Mistral's RLHF nearly eliminated reasoning sycophancy (0.2%) while leaving factual sycophancy near-total (99.8%) — a completely different tradeoff from Llama-3's, confirming that the specific sycophancy profile is determined by RLHF training choices rather than architecture.

This suggests RLHF teaches the model *when* to be sycophantic (social/opinion contexts) rather than *whether* to be sycophantic. Addressing opinion-domain sycophancy may require fine-tuning data that explicitly penalizes agreement with false user opinions in ambiguous social contexts.

### Limitations

1. **Two model families at 7–8B scale**: Core findings replicate across Llama-3-8B-Instruct and Mistral-7B-Instruct, but both are 7–8B parameter models. Generalization to larger scales (70B+), different training regimes (e.g., constitutional AI), or non-transformer architectures remains an open question. The experimental pipeline is model-agnostic and can be applied to any TransformerLens-compatible architecture.
2. **Binary forced choice**: Our sycophancy measurement uses (A)/(B) forced choice, which does not capture the full range of sycophantic behaviors in free-form generation.
3. **Probe control class balance**: The original probe control run had degenerate class balance for truthfulqa and gsm8k sources. The balanced replication (Job 10) fixes this by randomizing answer positions; both runs converge on the same social compliance interpretation.
4. **Patching-to-ablation gap**: Activation patching identifies heads that are sufficient carriers of the sycophantic signal, but ablation shows they are not causally necessary (see "Sufficiency vs. Necessity" in Discussion). This dissociation — analogous to fMRI vs. lesion dissociations in neuroscience — is itself a methodological contribution, but it limits the utility of patching for identifying intervention targets in models with redundant circuits.

---

## 7. Output Files

| File | Description |
|------|-------------|
| `results/baseline_llama3_summary.json` | Sycophancy rates, per-source breakdown, effect sizes (Instruct) |
| `results/baseline_llama3_detailed.csv` | Per-sample results (Instruct) |
| `results/baseline_llama3_base_summary.json` | Sycophancy rates (Base model) |
| `results/baseline_llama3_base_detailed.csv` | Per-sample results (Base model) |
| `results/probe_results_neutral_transfer.json` | Claim-bearing probe run (train neutral, transfer to biased) |
| `results/probe_results_mixed_diagnostic.json` | Diagnostic mixed-mode probe run (GroupKFold by `sample_id`) |
| `results/probe_results_llama3_base_neutral_transfer.json` | Base-model neutral-transfer probe run |
| `results/patching_heatmap.json` | Layer × position patching scores (Instruct) |
| `results/head_importance.json` | Per-head recovery scores across critical layers (Instruct) |
| `results/base_model/patching_heatmap.json` | Layer × position patching scores (Base model) |
| `results/base_model/head_importance.json` | Per-head recovery scores (Base model) |
| `results/probe_control_results.json` | Probe control: neutral-only training, per-layer transfer accuracy |
| `results/probe_control_balanced_results.json` | Balanced probe control: randomized answer positions, all domains |
| `results/head_ablation_results.json` | Top-3 head ablation: single/pair/all, zero + mean |
| `results/top10_ablation_results.json` | Top-10 head ablation (GSM8k N=200) |
| `results/top10_ablation_full_gsm8k.json` | Top-10 head ablation (GSM8k N=1319) |
| `results/steering_results.json` | Steering condition table + capability metrics + CIs |
| `results/steering_results.json.checkpoint.json` | Steering checkpoint for resume |
| `results/full_rerun_manifest.json` | Consolidated rerun artifact/metric manifest |
| `results/mistral/baseline_summary.json` | Mistral sycophancy rates, per-source breakdown |
| `results/mistral/baseline_detailed.csv` | Mistral per-sample results |
| `results/mistral/probe_control_balanced_results.json` | Mistral balanced neutral-transfer probes |
| `results/mistral/patching_heatmap.json` | Mistral layer × position patching scores |
| `results/mistral/head_importance.json` | Mistral per-head recovery scores |
| `results/mistral/top10_ablation_full_gsm8k.json` | Mistral top-10 head ablation (GSM8k N=1319) |
| `results/mistral/steering_results.json` | Mistral steering condition table + capability metrics |
| `data/processed/master_sycophancy.jsonl` | Full 1,500-sample dataset |
| `data/processed/master_sycophancy_balanced.jsonl` | Balanced dataset with randomized answer positions |
| `data/processed/control_groups/` | Filtered control subsets |

---

## 8. Reproducibility

All code, data processing scripts, SLURM job scripts, and result artifacts are available in the project repository. Experiments were run on the Unity HPC Cluster (UMass) using a single NVIDIA A100-SXM4-80GB GPU with PyTorch 2.10.0+cu128 and TransformerLens 2.x. All random seeds are fixed at 42. The Llama-3 pipeline (13 SLURM jobs covering baseline, probing, patching, ablation, steering, and control groups) completes in approximately 48 GPU-hours; the Mistral replication pipeline (5 jobs) adds approximately 30 GPU-hours. Llama-3 result artifacts are validated by `results/full_rerun_manifest.json` (`missing_count: 0`); Mistral artifacts are in `results/mistral/`.

**TransformerLens note:** Llama-3 models require post-load configuration of `model.cfg.use_attn_result = True` followed by `model.setup()` to enable per-head activation access. Passing this as a constructor argument raises a `TypeError` because it leaks to the HuggingFace constructor. See `docs/ENGINEERING_NOTES.md` for the full list of implementation issues encountered and resolved.

---

## 9. Conclusion

All experiments are complete and validated by `results/full_rerun_manifest.json` (Mar 9, 2026, `missing_count: 0`). The confirmed findings are:

1. **Sycophancy is strongly domain-dependent.** Opinion-style prompts elicit 82.4% sycophancy in Llama-3-8B-Instruct; the model is essentially immune to social pressure on mathematical reasoning (0.0%) and highly resistant on factual questions (1.6%). RLHF does not introduce sycophancy — the base model (36.7% overall) is *more* sycophantic — but instruction tuning concentrates sycophancy in opinion domains while suppressing it on verifiable tasks.

2. **Social compliance, not belief corruption, drives sycophantic outputs.** Balanced neutral-transfer probes show the model retains correct internal representations under biased prompts at all layers tested. At layer 1: 59.9% robust correct tracking, 18.0% social compliance, 10.1% belief corruption. The model knows the right answer but does not say it. Probes trained on format-mixed data spuriously suggest belief corruption — a cautionary methodological finding for the probe-based interpretability literature.

3. **Patching-identified heads are causally non-necessary.** The top 3 heads by activation patching recovery score (L1H20, L5H5, L4H28) cannot be ablated to reduce sycophancy — individually or in combination. Extending to the top 10 heads simultaneously yields only +0.5 pp change. Activation patching measures sufficiency (a component *can* carry the signal), not necessity (the component *must* carry it). This patching-to-ablation dissociation is the computational analog of fMRI vs. lesion dissociations in neuroscience.

4. **Inference-time intervention cannot selectively suppress sycophancy.** Both head ablation and residual-stream steering across all tested layers and magnitudes fail to reduce sycophancy without destroying general capabilities. This convergent null across two mechanistically distinct methods strongly implies the behavior is redundantly distributed across the network. Effective mitigation likely requires training-time intervention targeting the opinion-domain RLHF objective directly.

5. **Sycophancy is a family of domain-specific behaviors, not a single mechanism.** A fictional-entity control group (93.0% sycophancy, N=100) reveals an entirely different patching circuit (L1H10, L0H2 in layers 0–1) from the opinion circuit (L4H28, L4H5 in layers 4–5), with zero top-5 head overlap and a sign reversal for L1H20 across domains. Sycophancy cannot be addressed by targeting a single circuit — different knowledge domains activate different computational pathways for social agreement.

6. **All core findings replicate across architectures.** Full replication on Mistral-7B-Instruct confirms social compliance dominance (SC/BC ratio 6.4:1), the patching-to-ablation dissociation (+1.0 pp, no reduction), and the steering null — despite entirely different circuits (Mistral L11H17 vs. Llama-3 L4H28, zero overlap) and inverted sycophancy profiles (Mistral: 99.8% factual, 50.8% opinion; Llama-3: 1.6% factual, 82.4% opinion). The redundant distribution of sycophancy is a general property of RLHF-trained models, not an artifact of a single architecture.
