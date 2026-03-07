# Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation

**Author:** Kenny Egan
**Institution:** Wentworth Institute of Technology
**Advisor:** Prof. Larson
**Date:** March 3, 2026
**Model:** meta-llama/Meta-Llama-3-8B-Instruct
**Hardware:** NVIDIA A100-SXM4-80GB (Unity HPC Cluster, UMass)
**Framework:** TransformerLens 2.x, PyTorch 2.10.0+cu128

---

## Abstract

This paper reports a mechanistic sycophancy pipeline on Llama-3-8B-Instruct and Llama-3-8B-Base, including baseline evaluation, probe analysis, activation patching, ablation, and representation steering. As of March 3, 2026, the implementation has been upgraded to remove known confounds (probe mode split, leakage-safe grouping, answer-position randomization, length-normalized confidence, and steering checkpoint/resume), but **full corrected SLURM reruns are still in progress**. Therefore, quantitative claims below should be treated as **provisional** unless explicitly marked confirmed by rerun-backed artifacts. The central hypotheses under current evaluation are: (1) whether neutral-transfer probes support social-compliance interpretation in early layers, (2) whether patching-identified heads remain non-causal under upgraded capability scoring, and (3) whether representation steering can reduce sycophancy while preserving MMLU/GSM8k under strict scoring and CI reporting.

## Rerun Status (March 3, 2026)

| Experiment Track | Status | Artifact Contract |
|---|---|---|
| Data regeneration (`--randomize-positions`) | Ready | `data/processed/master_sycophancy_balanced.jsonl` + metadata |
| Baseline rerun (instruct/base) | Pending SLURM execution | `results/baseline_*_summary.json` |
| Probes (`neutral_transfer`, `mixed_diagnostic`) | Pending SLURM execution | `results/probe_results_neutral_transfer.json`, `results/probe_results_mixed_diagnostic.json` |
| Probe-control balanced rerun | Pending SLURM execution | `results/probe_control_balanced_results.json` |
| Patching/head ranking rerun | Pending SLURM execution | `results/patching_heatmap.json`, `results/head_importance.json` |
| Ablation rerun (top heads + top10) | Pending SLURM execution | `results/head_ablation_results.json`, `results/top10_ablation_*.json` |
| Steering rerun (checkpoint/resume) | Pending SLURM execution | `results/steering_results.json` + checkpoint JSON |
| Consolidated manifest | Pending post-rerun | `results/full_rerun_manifest.json` |

---

## 1. Introduction

Sycophancy — the tendency of a language model to validate user beliefs regardless of factual accuracy — poses a fundamental alignment challenge. A sycophantic model tells users what they want to hear rather than what is true, undermining its usefulness for tasks requiring honest information retrieval, reasoning, or advice.

Two competing hypotheses exist for the internal mechanism behind sycophancy:

1. **Belief Corruption**: The biased social context shifts the model's internal representation of the correct answer. The model "believes" the sycophantic response is correct.
2. **Social Compliance**: The model retains an accurate internal belief about the correct answer but suppresses it in favor of the socially expected response.

Distinguishing between these has direct implications for mitigation: belief corruption requires fixing the model's knowledge representations, while social compliance requires targeting the output layer or decoding mechanism.

This study applies mechanistic interpretability techniques — linear probing and causal activation patching — to Llama-3-8B-Instruct to localize and characterize the sycophantic circuit.

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

**Status note:** The subsection values below reflect the latest pre-upgrade or partially upgraded runs and are retained for transparency. Final claim-bearing numbers will be locked only after the corrected rerun matrix (Section "Rerun Status") completes and `results/full_rerun_manifest.json` validates all required artifacts.

### 5.1 Baseline Sycophancy Rate

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
**Samples evaluated:** 1,500

| Metric | Value | 95% CI |
|--------|-------|--------|
| Overall sycophancy rate | **28.0%** | [25.8%, 30.3%] |
| Mean compliance gap | −0.0434 | [−0.0518, −0.0349] |

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
| Opinion sycophancy | 82.4% | **50.4%** |
| Factual sycophancy | 1.6% | **37.8%** |
| Reasoning sycophancy | 0.0% | **21.8%** |
| Mean total effect (patching) | 2.1050 | **0.9333** |

**Key finding:** The base model shows *higher* overall sycophancy (36.7% vs 28.0%), contradicting the hypothesis that RLHF/instruction tuning *introduces* sycophancy. Instead, instruction tuning appears to **suppress** sycophancy on factual and reasoning tasks while concentrating it in opinion domains. The base model exhibits broader but weaker sycophancy across all domains.

---

### 5.3 Linear Probes — Social Compliance vs. Belief Corruption

Logistic regression and ridge regression probes were trained at each of the 32 residual stream layers to classify whether a prompt was biased or neutral, using 5-fold cross-validation on 1,500 samples.

#### Instruct Model (Logistic)

| Metric | Value |
|--------|-------|
| Best layer (final position) | Layer 6 |
| Best probe accuracy | **99.47%** |
| Output accuracy on biased prompts | 72.0% |
| **Dominant pattern** | **Belief Corruption** |
| Belief corruption rate | 99.76% |
| Social compliance rate | 0.24% |

#### Instruct Model (Ridge)

| Metric | Value |
|--------|-------|
| Best layer (final position) | Layer 2 |
| Best probe accuracy | **99.60%** |
| **Dominant pattern** | **Belief Corruption** |
| Belief corruption rate | 99.29% |

#### Base Model (Logistic)

| Metric | Value |
|--------|-------|
| Best layer (final position) | Layer 4 |
| Best probe accuracy | **99.30%** |
| **Dominant pattern** | **Belief Corruption** |
| Belief corruption rate | 99.46% |
| Social compliance rate | 0.00% |

**Key finding:** Both logistic and ridge probes converge on the same conclusion when trained on mixed (neutral + biased) data: sycophancy correlates with belief corruption (>99%) rather than social compliance. However, these probes were trained on activations from both prompt conditions, meaning they could learn to distinguish prompt *format* (biased vs. neutral preamble) rather than tracking genuine belief shifts.

**Important caveat:** The probe control experiment (Section 5.5) reveals that this near-perfect accuracy is **partially confounded by prompt format** at deep layers (13+). Probes trained only on neutral prompts drop from 94.6% to 60.3% accuracy when tested on biased prompts at layer 14 — indicating format sensitivity, not robust truth tracking. Early-layer probes (0–10) transfer cleanly (1–4 pp drop), and at these layers the dominant pattern is social compliance (~27.1%), not belief corruption (~0.9%). The belief corruption vs. social compliance interpretation therefore depends critically on which layers are examined and whether probes are trained on format-mixed data. See Section 5.5 for full analysis.

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

| Rank | Head | Mean Recovery | Std |
|------|------|---------------|-----|
| 1 | **L1H20** | 0.5690 | ±1.2114 |
| 2 | **L5H5** | 0.5669 | ±0.6947 |
| 3 | **L4H28** | 0.5062 | ±0.6719 |
| 4 | L5H17 | 0.2704 | ±0.7006 |
| 5 | L3H17 | 0.2192 | ±0.3889 |
| 6 | L5H4 | 0.1944 | ±0.3640 |
| 7 | L5H19 | 0.1864 | ±0.6869 |
| 8 | L5H24 | 0.1858 | ±0.3304 |
| 9 | L4H5 | 0.1835 | ±0.5108 |
| 10 | L3H0 | 0.1744 | ±0.3150 |

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

**Key finding:** The sycophantic circuit is concentrated in **early layers (1–5)** of the instruct model. Three attention heads — **L1H20, L5H5, and L4H28** — account for the largest share of recoverable sycophantic behavior. The base model's critical circuit is similarly early (layers 0–4) but with substantially lower total effect (0.93 vs 2.10), suggesting instruction tuning strengthens the opinion-domain sycophancy circuit while suppressing it elsewhere.

---

### 5.5 Probe Control Experiment — Neutral-Only Training

To validate that probe accuracy reflects genuine truth-representation tracking rather than prompt-format classification, we trained probes exclusively on neutral prompt activations and tested on biased prompt activations. If probes learn a format-invariant truth direction, accuracy should transfer; if they learn format cues (e.g., the presence of "I think..." preambles), transfer accuracy should collapse.

**Design:** Logistic regression probes trained on `resid_post` activations from neutral prompts only, using answer-identity labels (lexicographic ordering of answer options). Tested on biased prompt activations from the same samples. 5-fold cross-validation. 1,500 samples.

**Class balance (balanced replication):** The original run suffered from degenerate class balance in two sources — `truthfulqa_factual` and `gsm8k_reasoning` both mapped 100% to one label class due to hardcoded answer positions. We reran with randomized (A)/(B) answer positions, achieving near-perfect balance across all three domains: `anthropic_opinion` 50.8%, `truthfulqa_factual` 51.6%, `gsm8k_reasoning` 50.0%. Results below are from the balanced replication.

#### Transfer Accuracy by Layer Depth

| Layer Range | Neutral CV Acc | Biased CV Acc | Mean Drop | Interpretation |
|-------------|---------------|---------------|-----------|----------------|
| 0–10 | 88–90% | 65–78% | 5–27 pp | **Early layers show partial transfer** |
| 11–13 | 90–94% | 67–75% | 17–19 pp | Transitional — increasing format sensitivity |
| 14–31 | 91–95% | 50–59% | 33–41 pp | **Format-sensitive** — probes learn prompt structure |

**Best layer (14):** Neutral accuracy 94.6%, biased accuracy 60.3% (drop: **34.3 pp**). At this layer, the probe achieves near-ceiling accuracy on neutral prompts but drops substantially on biased prompts.

#### Cross-Tabulation at Layer 14 (Best Neutral Accuracy)

| Category | Count | Rate |
|----------|-------|------|
| Social Compliance (probe: true, model: sycophantic) | 407 | 27.1% |
| Belief Corruption (probe: corrupted, model: sycophantic) | 14 | 0.9% |
| Robust (probe: true, model: correct) | 562 | 37.5% |
| Other (probe confusion) | 517 | 34.5% |

#### Cross-Tabulation at Layer 4 (Best Early-Layer Transfer)

| Category | Count | Rate |
|----------|-------|------|
| Social Compliance | 403 | 26.9% |
| Belief Corruption | 18 | 1.2% |
| Robust | 851 | 56.7% |
| Other | 228 | 15.2% |

**Key finding:** With balanced labels across all three domains, the probe control confirms the **social compliance** interpretation. At every layer examined, social compliance (27%) dominates belief corruption (~1%) by a ratio of approximately 30:1. The model retains correct internal representations even under biased prompts but outputs sycophantic responses — consistent with a gating or output-override mechanism rather than genuine belief corruption. The `was_rebalanced: false` flag confirms the balanced data required no artificial rebalancing.

---

### 5.6 Head Ablation — Top 3 Heads

To test whether the patching-identified heads are causally necessary for sycophancy, we zero-ablated the top 3 heads (L1H20, L5H5, L4H28) — individually, in pairwise combinations, and all three together — and measured sycophancy rate and general capabilities (MMLU, GSM8k).

| Condition | Sycophancy Rate | Change (pp) | MMLU | GSM8k |
|-----------|----------------|-------------|------|-------|
| **Baseline** | **28.0%** | — | 62.4% | 12.5% |
| L1H20 only | 27.9% | −0.1 | 61.2% | 11.5% |
| L5H5 only | 28.5% | +0.5 | 62.0% | 10.5% |
| L4H28 only | 28.1% | +0.1 | 62.6% | 10.5% |
| L1H20 + L5H5 | 28.3% | +0.3 | 61.6% | 11.5% |
| L1H20 + L4H28 | 27.9% | −0.1 | 60.8% | 12.5% |
| L5H5 + L4H28 | 28.3% | +0.3 | 62.4% | 11.0% |
| **All 3 (zero)** | **28.1%** | **+0.1** | 62.4% | 11.5% |
| All 3 (mean) | — | — | — | — |

**Note:** Mean-ablation of all 3 heads caused catastrophic output degradation (all 1,500 samples produced unparseable outputs), yielding 0% on all metrics. This is excluded from analysis.

**Key finding:** Neither individual nor combined zero-ablation of the top 3 patching-identified heads produces any meaningful reduction in sycophancy. The largest observed change is +0.5 pp (L5H5 alone), well within sampling noise. MMLU is preserved (60.8–62.6%), while GSM8k shows minor fluctuation on the small N=200 subsample.

---

### 5.7 Head Ablation — Top 10 Heads (Circuit Redundancy Test)

To test whether broader ablation overcomes circuit redundancy, we zero-ablated all 10 heads from the patching top-10 list simultaneously: L1H20, L5H5, L4H28, L5H17, L3H17, L5H4, L5H19, L5H24, L4H5, L3H0.

| Condition | Sycophancy Rate | Sycophantic Count | MMLU | GSM8k (N=1319) |
|-----------|----------------|-------------------|------|----------------|
| **Baseline** | **28.0%** | 420/1500 | 62.6% | **11.3%** |
| **All 10 (zero)** | **28.5%** | 427/1500 | 62.2% | **10.6%** |
| **Change** | **+0.5 pp** | +7 | −0.4 pp | **−0.7 pp** |

**Per-source breakdown:**

| Source | Baseline | Ablated | Change |
|--------|----------|---------|--------|
| `anthropic_opinion` | 82.4% | 82.8% | +0.4 pp |
| `truthfulqa_factual` | 1.6% | 2.6% | +1.0 pp |
| `gsm8k_reasoning` | 0.0% | 0.0% | 0.0 pp |

**GSM8k (full test set):** With the complete GSM8k test set (N=1319), the baseline is 11.3% (149/1319) and ablated is 10.6% (140/1319) — a retention rate of **94.0%**. This resolves the noisy 12.5% → 8.0% drop (64% retained) observed in the initial N=200 run as small-sample noise.

**Key finding:** Even ablating all 10 patching-identified heads simultaneously produces **no meaningful sycophancy reduction** (+0.5 pp, within sampling noise). MMLU is preserved (62.2%, 99.4% retained). GSM8k shows a small, non-significant decrease (11.3% → 10.6%, 94.0% retained on full N=1319). This confirms that the sycophantic behavior is **redundantly distributed** across the network — the patching-identified circuit captures activation patterns correlated with sycophancy, but the behavior is not causally dependent on these specific heads.

---

### 5.8 Representation Steering

*Results pending (Job 12).*

<!-- NULL-RESULT NARRATIVE (activate or delete when results arrive):

Representation steering at the patching-identified layers produces no meaningful sycophancy reduction across the full alpha sweep (0.5–50.0), mirroring the null ablation result from Sections 5.6–5.7. At moderate alpha values (1.0–5.0), sycophancy rates remain within ±1 pp of baseline across all three domains. At extreme alpha values (20.0–50.0), general capabilities degrade (MMLU drops to X%, GSM8k to Y%) without corresponding sycophancy reduction — the intervention damages the model's general reasoning before it affects the sycophantic computation.

This convergent null result across two independent intervention methods — head ablation and residual-stream steering — constitutes strong evidence that opinion-domain sycophancy in Llama-3-8B-Instruct is not localized to any identifiable circuit subset. The behavior appears to be a distributed property of the network's learned representations, implemented redundantly across many components such that no inference-time intervention on a tractable subset of the computation can selectively suppress it. This finding has direct implications for the alignment community: **effective sycophancy mitigation likely requires training-time intervention** — such as RLHF with anti-sycophancy preference data (Wei et al., 2023), DPO with synthetic disagreement examples, or constitutional AI approaches — rather than post-hoc activation manipulation.
-->

---

## 6. Discussion

**Interpretation status:** This discussion reflects the current best reading of preliminary artifacts and should be treated as hypothesis-level until corrected reruns finalize the quantitative values.

### Belief Corruption vs. Social Compliance: A Layer-Dependent Picture

The original probe analysis (Section 5.3) suggested near-unanimous belief corruption (>99%). The probe control experiment (Section 5.5) substantially qualifies this conclusion. The picture that emerges is **layer-dependent**:

- **Early layers (0–10):** At these layers, 56.7% of samples show robust truth tracking, 26.9% show social compliance (probe retains correct answer, model outputs sycophantic response), and only 1.2% show belief corruption. The model *retains* correct representations in early layers even under biased prompts.

- **Deep layers (13+):** Probes achieve higher absolute accuracy on neutral data (94–95%) but collapse on biased data (50–59%). The probe at these layers is largely classifying prompt format rather than tracking a format-invariant truth direction. The original 99.5% accuracy and 99.8% belief corruption classification from mixed-format training was inflated by this confound.

This suggests a more nuanced mechanism than either hypothesis alone predicts: **the model maintains accurate early representations (consistent with social compliance) but these representations are not faithfully propagated to the output**. The sycophantic behavior may arise from how mid-to-late layers *use* early representations rather than from corruption of the representations themselves.

### Sufficiency vs. Necessity: What Activation Patching Actually Measures

The most striking finding is the **complete dissociation between patching importance and causal necessity**. The top 3 heads (L1H20, L5H5, L4H28) show the highest activation patching recovery scores (0.51–0.57), yet ablating them — individually or together — has zero effect on sycophancy. Extending to the top 10 heads still produces no reduction.

This dissociation reflects a fundamental distinction: **activation patching measures sufficiency, not necessity**. When patching restores a head's activation from the biased run to the clean (neutral) value and sycophancy decreases, this shows the head *can carry* the sycophantic signal — it is a sufficient channel. But it does not show the head *must carry* it. If multiple parallel pathways implement the same computation, ablating any one pathway allows the remaining ones to compensate. The null ablation result demonstrates exactly this: sycophancy in Llama-3-8B-Instruct is computed via a **degenerate circuit** — multiple redundant pathways encode the same behavioral transformation, and no tractable subset is causally necessary.

This phenomenon is directly analogous to the well-known dissociation between fMRI and lesion studies in neuroscience. fMRI identifies brain regions active during a task (sufficient carriers of the computation), but lesioning those regions may not impair task performance when redundant neural pathways exist. Our finding is the computational analog: activation patching is the fMRI of mechanistic interpretability — it identifies *where* a computation is expressed, but not whether it is *uniquely* expressed there.

This result carries implications for the broader circuit discovery paradigm. Activation patching may systematically overstate the causal importance of identified components in models where behaviors are redundantly implemented. **Validation via ablation or other necessity tests is essential** before drawing causal conclusions from patching results alone. This motivates our representation steering experiment (Section 5.8), which intervenes on the distributed residual-stream direction rather than individual heads — targeting the representation itself rather than its carriers.

### Methodological Implications for Linear Probing

The probe control result is a cautionary finding for the mechanistic interpretability community. Linear probes trained on format-mixed data can achieve near-perfect accuracy while primarily learning superficial distributional cues (e.g., the presence of "I think..." preambles in biased prompts) rather than the target concept. **Training probes on one condition and testing on another is essential** for validating that probes track genuine internal representations rather than input features.

### RLHF Does Not Introduce Sycophancy

The base model (36.7% sycophancy) actually exceeds the instruct model (28.0%) in overall rate. However, the distribution shifts dramatically: the base model is sycophantic broadly (opinion 50.4%, factual 37.8%, reasoning 21.8%), while the instruct model concentrates sycophancy almost entirely in opinion domains (82.4%) while nearly eliminating it on factual and reasoning tasks.

This suggests RLHF teaches the model *when* to be sycophantic (social/opinion contexts) rather than *whether* to be sycophantic. Addressing opinion-domain sycophancy may require fine-tuning data that explicitly penalizes agreement with false user opinions in ambiguous social contexts.

### Limitations

1. **Single model**: All results are from Llama-3-8B-Instruct. Generalization to other architectures, model sizes, and training regimes is an open question. However, deep mechanistic analysis of a single model — spanning 12 experiments across probing, patching, ablation, and steering — yields richer insight than shallow analysis across many models. Single-model mechanistic studies are the established norm in this subfield: Wang et al. (2022) conducted their foundational circuit discovery work on GPT-2 Small exclusively, Burns et al. (2023) focused on a single model family, and Li et al. (2023) demonstrated inference-time intervention on one architecture. The experimental pipeline developed here (contrastive datasets, activation patching, probe control, ablation, steering) is model-agnostic and can be applied to any TransformerLens-compatible architecture.
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
| `data/processed/master_sycophancy.jsonl` | Full 1,500-sample dataset |
| `data/processed/master_sycophancy_balanced.jsonl` | Balanced dataset with randomized answer positions |
| `data/processed/control_groups/` | Filtered control subsets |

---

## 8. Engineering Notes

### Issues Encountered and Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Probe conclusions depended on mixed-format training | Probe labels confounded with prompt condition in mixed data | Added explicit `--analysis-mode` with claim-bearing `neutral_transfer` default and diagnostic `mixed_diagnostic` |
| Probe fold leakage risk in mixed mode | Paired neutral/biased samples could cross folds without grouping | Added deterministic `sample_id` and GroupKFold grouping by `sample_id` |
| Dataset class-balance artifact in synthetic domains | TruthfulQA/GSM8k answer positions were fixed | Added `--randomize-positions` and recorded per-sample randomization metadata |
| Confidence metric length bias | Raw sequence log-probability penalized longer targets | Switched to length-normalized confidence metrics (per-token avg log-prob), with confidence-filtered stats reported as secondary |
| Binomial test API drift across SciPy versions | `binomtest` vs `binom_test` incompatibility | Added compatibility wrapper with modern-first fallback |
| Long steering sweeps lost progress on interruption | No checkpoint persistence | Added per-condition checkpoint JSON + `--resume-from-checkpoint` support |
| SLURM resource directives were inconsistent | Dynamic/embedded directives were brittle across scripts | Removed dynamic `#SBATCH` patterns; centralized resources in `slurm/submit_all.sh` |
| Jobs could succeed with missing outputs | No artifact contract checks | Added non-zero artifact checks across SLURM jobs, including steering final+checkpoint JSON checks |
| `KeyError: 'blocks.11.attn.hook_result'` | `use_attn_result=False` by default in TransformerLens for Llama-3 | Set `model.cfg.use_attn_result = True; model.setup()` after loading in `sycophancy_model.py` |
| `TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'use_attn_result'` | Kwarg passed directly to `from_pretrained`, leaked to HuggingFace constructor | Moved to post-load config mutation |
| `RuntimeError: expanded size (118) must match existing size (116)` | Head patch hook copied full sequence dimension; neutral/biased sequences differ in length | Changed to `n = min(act.shape[1], clean_act.shape[1]); activation[0, :n, h, :] = clean_act[0, :n, h, :]` |

### Cluster Configuration

| Setting | Value |
|---------|-------|
| Cluster | Unity HPC (UMass) |
| Partition | `gpu` |
| Account | `pi_larsonj_wit_edu` |
| GPU | A100 (`--gres=gpu:a100:1`) |
| Conda module | `conda/latest` (miniforge3-24.7.1) |
| Environment | `sycophancy-lab` |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |

---

## 9. Conclusion

Current evidence suggests the following, pending full corrected rerun confirmation:

1. **Domain dependence is likely real**, with opinion-style prompts showing substantially higher sycophancy than factual/reasoning prompts in preliminary runs.
2. **Probe interpretation depends on training regime**: mixed-mode probes appear format-sensitive, while neutral-transfer probes are the claim-bearing test for social-compliance vs belief-corruption interpretation.
3. **Patching-vs-ablation dissociation is a key mechanism hypothesis**: components with strong patching recovery may still be non-necessary under ablation, consistent with redundancy.
4. **Base-model comparison remains important** for testing whether instruction tuning introduces sycophancy or mainly redistributes it across domains.

Final numerical claims in this paper should be interpreted as provisional until the rerun matrix artifacts are complete and validated by `results/full_rerun_manifest.json`.
