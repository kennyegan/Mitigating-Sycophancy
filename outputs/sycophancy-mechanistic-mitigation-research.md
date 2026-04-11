# Peer Review Evidence Brief: "Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation"

**Reviewer:** Automated Evidence Audit
**Paper:** `paper.md` (Kenneth Egan, Wentworth Institute of Technology, April 7, 2026)
**Date of Audit:** 2026-04-10
**Archive snapshot:** `results_archive/results_20260303T225104Z/` (March 3, 2026)

---

## Executive Summary

This paper makes six major claims about sycophancy in Llama-3-8B-Instruct and Mistral-7B-Instruct. The strongest contributions — patching-to-ablation dissociation and the DPO mechanistic probe re-analysis — are genuine and well-motivated. However, the audit found **three critical data-provenance issues**, **one clear mischaracterization of a cited paper**, and **several numerical discrepancies** between the archived results and the numbers in the paper. The DPO novelty claim rests entirely on result files that are absent from the project repository. Full reproducibility cannot be confirmed from locally available artifacts.

---

## 1. Citation Verification

### 1.1 Chen et al. (2024) — ICML, "Pinpoint Tuning"

**Status: EXISTS, but characterization is inaccurate.**

- **Paper claims:** "Chen et al. (2024) use **path patching** on Llama-2-Chat to identify sycophancy-related heads and apply targeted fine-tuning ('pinpoint tuning'), achieving substantial sycophancy reduction through **head-level knockout**."
- **Actual paper:** Chen et al. (ICML 2024, pp. 6950–6972, https://proceedings.mlr.press/v235/chen24u.html) propose **Supervised Pinpoint Tuning (SPT)**, a method that identifies a small percentage (<5%) of model modules that affect sycophancy using gradient/activation analysis, then **fine-tunes only those modules** while freezing the rest. The paper never mentions "path patching" (which is an edge-level causal patching technique), and "head-level knockout" (ablation) is not what SPT does — SPT fine-tunes, not ablates. The sycophancy studied is challenge-induced (model reverses correct answers under pushback), not assertion-based.
- **⚠️ Flag:** This is a meaningful mischaracterization on two counts: (a) attributing path patching as the identification method when Chen et al. use activation/gradient-based module selection, and (b) calling the intervention "head-level knockout" when it is targeted fine-tuning. The Discussion correctly notes that the *type* of sycophancy (challenge-induced vs. assertion-based) differs, but the method description in the Related Work section is inaccurate. Reviewers familiar with Chen et al. will notice.

### 1.2 Li et al. (2025) — "When Truth Is Overridden" (arxiv 2508.02087)

**Status: EXISTS, citation accurate in spirit, year is borderline.**

- The paper is by Keyu Wang*, Jin Li*, et al. (equal contribution), first submitted to arxiv August 4, 2025, and published at AAAI 2026. Citing it as "Li et al. (2025)" uses the preprint date but is conventionally acceptable. The first alphabetical author is Jin Li, making "Li et al." the standard citation form.
- **Paper claims:** Li et al. find "sycophantic output crystallizes in late layers (16–23)" of Llama-3.1-8B-Instruct. The actual abstract describes "a **two-stage** emergence: (1) a **late-layer output preference shift** and (2) deeper representational divergence." The specific "layers 16–23" framing is an interpretation of their logit-lens results; the abstract does not specify these layer numbers. The claim is plausible but not directly verifiable from the abstract alone.
- **Assessment:** Characterization is broadly accurate and the conceptual mapping (logit-lens → late-layer crystallization; probes → early-layer social compliance) is a genuine methodological contribution.

### 1.3 O'Brien et al. (2026) — arxiv 2601.18939, NeurIPS 2025 Workshop

**Status: EXISTS, characterization accurate.**

- Confirmed: Claire O'Brien et al., "A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy," submitted Jan 26, 2026. Uses SAEs and linear probes to isolate **~3% of MLP neurons** most predictive of sycophancy in **Gemma-2-2B and 9B** models, then fine-tunes only those neurons via gradient masking.
- The paper correctly characterizes this as "sparse autoencoders on Gemma-2" and "gradient-masked fine-tuning." The model sizes (2B/9B) vs. this paper's 7–8B models are noted. The Discussion's reconciliation (SAE feature-level vs. head-level redundancy) is well-reasoned.
- **Assessment:** Accurate.

### 1.4 Sharma et al. (2024) — ICLR 2024

**Status: EXISTS, characterization accurate.**

- Confirmed at proceedings.iclr.cc (2024). Comprehensive behavioral characterization showing sycophancy scales with model capability, is amplified by RLHF, and appears across opinion/factual/reasoning domains.
- **Assessment:** Accurate.

### 1.5 Heimersheim & Nanda (2024) — arxiv 2404.15255

**Status: EXISTS, characterization accurate and well-used.**

- Confirmed: "How to use and interpret activation patching" (Apr 2024). The paper explicitly defines denoising (clean → corrupt patching) as testing **sufficiency** and noising (corrupt → clean patching) as testing **necessity**, with the AND/OR gate analogy directly anticipating exactly the dissociation this paper documents for sycophancy.
- Relevant direct quote from the paper: "An important and underrated point is that these two directions can be very different, and are not just symmetric mirrors of each other."
- **Assessment:** The citation is apt and well-deployed. The paper's empirical validation of this theoretical warning is one of its stronger contributions.

### 1.6 McGrath et al. (2023) and Rushing & Nanda (2024)

**Status: BOTH EXIST, characterizations accurate.**

- **McGrath et al. (2023):** "The Hydra Effect: Emergent Self-repair in Language Model Computations," arxiv 2307.15771. Demonstrates adaptive computation and self-repair in LLMs when components are ablated.
- **Rushing & Nanda (2024):** "Explorations of Self-Repair in Language Models," ICML 2024 (PMLR 235:42836–42855). Demonstrates self-repair across model families and sizes when ablating individual attention heads.
- **Assessment:** Both citations are accurate. The self-repair framing provides the theoretical context for why ablation fails here.

### 1.7 Novelty of "First Mechanistic Evidence" DPO Claim

**Status: Claim is plausible but needs qualification.**

The paper claims "the first mechanistic decomposition of how DPO resolves *sycophancy*." The relevant comparative work is:

1. **Lee et al. (2024, ICML):** "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and **Toxicity**" — mechanistic analysis of DPO for a different behavior (toxicity, not sycophancy).
2. **Yang et al. (2025, EMNLP):** "How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis" — DPO mechanics for toxicity across Llama-3.1-8B, Gemma-2, Mistral-7B.

Neither of these papers studies sycophancy. The specific claim for *sycophancy* — framed as social-compliance-to-robust-tracking conversion — appears novel. **However:** the claim is only as strong as the probe re-analysis, and those results (DPO probe decomposition numbers) are absent from the local artifact archive (see Section 2 below). The figures are consistent with the claimed numbers but the underlying data files do not exist in the repository.

---

## 2. Numerical Verification Against Artifacts

### 2.1 Repository State — Critical Finding

**⚠️ CRITICAL: The canonical `results/` directory does not exist.**

The paper states "All values are sourced from confirmed artifacts validated by `results/full_rerun_manifest.json` (`missing_count: 0`)." However:

- `/results/` — **does not exist**
- `/results/full_rerun_manifest.json` — **does not exist**
- The only available artifacts are in `results_archive/results_20260303T225104Z/` (a snapshot from March 3, 2026)
- The paper was finalized **April 7, 2026** — 35 days after the archive snapshot
- All result files cited in the paper (steering_results.json, dpo_eval_results.json, dpo_training_metrics.json, corrected_ablation_results.json, and all `results/mistral/` files) are **absent entirely** — not even in the archive

The paper claims reproducibility with "missing_count: 0" from a manifest that cannot be verified. The archive covers approximately the first half of the experimental pipeline (through top-10 ablation), but all Mistral replication, steering, corrected ablation, and DPO results are unverifiable from local files.

### 2.2 Baseline Sycophancy Rates

**Source:** `results_archive/.../baseline_llama3_summary.json` (timestamp: Feb 27, 2026)

| Metric | Paper Claims | Archive JSON | Status |
|--------|-------------|--------------|--------|
| Overall sycophancy rate | 28.0% | 28.0% | ✓ Match |
| Overall 95% CI | [25.8%, 30.3%] | [25.8%, 30.3%] | ✓ Match |
| Opinion sycophancy | 82.4% | 82.4% | ✓ Match |
| Opinion 95% CI | [78.8%, 85.5%] | [78.8%, 85.5%] | ✓ Match |
| Factual sycophancy | 1.6% | 1.6% | ✓ Match |
| Reasoning sycophancy | 0.0% | 0.0% | ✓ Match |
| Mean compliance gap (overall) | −0.0435 | −0.0434 | ✓ ~Match (rounding) |
| Samples skipped | **7** | **0** | ⚠️ **Discrepancy** |

**Note on samples skipped:** The paper says "1,493 (7 skipped due to tokenization)" but the archive JSON shows `samples_skipped: 0, samples_evaluated: 1500`. This is minor but inconsistent.

**Note on effect sizes:** The archive stores `effect_sizes_cohens_d` while the paper reports Cohen's h. The JSON values (d = 0.183, 0.779, 0.834) measure compliance gap distributions; the paper's h values (2.276, 2.022) measure proportion differences. Both are technically correct but measure different constructs. The paper only reports h without noting that Cohen's d on the underlying continuous measure is much smaller. This presentation choice may inflate the perceived effect size.

### 2.3 Probe Decomposition — Layer 1 (Balanced)

**Source:** `results_archive/.../probe_control_balanced_results.json` (timestamp: March 2, 2026)

| Metric | Paper Claims (§5.5) | Archive JSON (Layer 1) | Status |
|--------|---------------------|------------------------|--------|
| Best neutral CV accuracy | 89.0% | 89.0% | ✓ Match |
| Best biased CV accuracy | 77.9% | 77.9% | ✓ Match |
| Accuracy drop | 11.1 pp | 11.1 pp | ✓ Match |
| Social compliance rate | **18.0%** [16.0%, 19.9%] | **22.5%** (338/1500) | ⚠️ **Discrepancy** |
| Belief corruption rate | **10.1%** [8.6%, 11.7%] | **5.5%** (83/1500) | ⚠️ **Discrepancy** |
| Robust tracking rate | **59.9%** [57.4%, 62.3%] | **63.3%** (950/1500) | ⚠️ **Discrepancy** |
| "Other" rate | 12.1% | **8.6%** (129/1500) | ⚠️ **Discrepancy** |

**Analysis:** The biased CV accuracy and accuracy drop match, but the four-way SC/BC/robust/other decomposition does not. The archive uses `biased_accuracy_full_probe = 0.859` (85.9%) for classification, while the paper reports `biased_cv_accuracy = 0.779` (77.9%) as the headline figure. The decomposition counts in the archive were generated by the full-probe predictions on all 1500 biased samples, not by the CV predictions. This distinction is not explained in the paper.

The discrepancy in the SC/BC ratio is substantive: archive gives SC/BC = 22.5%/5.5% = **4.1:1**, while the paper claims 18.0%/10.1% = **1.8:1**. Both show SC dominance, but the paper's stated ratio understates the dominance relative to the archive.

The paper's claimed numbers (18.0% SC, 10.1% BC, 59.9% robust) match the DPO comparison figure (Figure 6) exactly, indicating an internally consistent final version of the probe analysis that is not reflected in the archived file. The likely explanation is a later rerun (Job 10 rerun, after March 3) producing different output.

**⚠️ The core SC/BC ratio claim — "1.8:1 in favor of social compliance" — cannot be verified from available artifacts.**

### 2.4 Head Importance (Patching Top-3)

**Source:** `results_archive/.../head_importance.json` (timestamp: Feb 28, 2026)

| Rank | Paper Claims (§5.4, "validated") | Archive JSON | Status |
|------|----------------------------------|--------------|--------|
| 1 | L4H28 (0.4428) | **L1H20 (0.569)** | ⚠️ **Discrepancy** |
| 2 | L4H5 (0.3020) | **L5H5 (0.567)** | ⚠️ **Discrepancy** |
| 3 | L5H31 (0.2564) | **L4H28 (0.506)** | ⚠️ Partial match |
| 4 | L2H5 (0.2445) | L5H17 (0.270) | ⚠️ Discrepancy |

The paper acknowledges two separate patching runs: an "earlier run" (L1H20, L5H5, L4H28 as top-3) and a "validated run" (L4H28, L4H5, L5H31 as top-3). The archive contains **the earlier run**, not the validated run. The "validated" run results exist only in the missing `results/` directory.

**High variance concern:** The archive head recovery standard deviations far exceed the means for top heads:
- L1H20: mean 0.569, **std 1.211** (std/mean = 2.1)
- L5H5: mean 0.567, **std 0.695** (std/mean = 1.2)
- L4H28: mean 0.506, **std 0.672** (std/mean = 1.3)

Standard deviations exceeding the mean indicate highly right-skewed, noisy recovery distributions. This level of variance across N=99 samples means the top-K ranking could easily change between runs or with different random samples. The paper does not report confidence intervals for individual head recovery scores, nor discuss this instability. The fact that two runs of the same patching pipeline produced substantially different top-3 orderings (with L5H31 going from recovery −0.171 to +0.256 between runs) validates this concern.

**⚠️ Head recovery rankings are highly unstable across runs (N=99 samples, std > mean). The "validated" top-3 cannot be confirmed from available artifacts.**

### 2.5 Top-10 Ablation (GSM8k Capability) — Critical Discrepancy

**Source:** `results_archive/.../top10_ablation_full_gsm8k.json` (timestamp: March 2, 2026)

| Metric | Paper Claims (§5.7) | Archive JSON | Status |
|--------|---------------------|--------------|--------|
| Sycophancy baseline | 28.0% (420/1500) | 28.0% (420/1500) | ✓ Match |
| Sycophancy ablated | 28.5% (427/1500) | 28.47% (427/1500) | ✓ Match |
| Sycophancy change | +0.5 pp | +0.47 pp | ✓ Match |
| MMLU baseline | 62.0% | **62.6%** (311/497) | Minor discrepancy |
| MMLU ablated | 63.4% | **62.2%** (309/497) | Minor discrepancy |
| **GSM8k baseline** | **33.2% (438/1319)** | **11.3% (149/1319)** | ⚠️ **CRITICAL DISCREPANCY** |
| **GSM8k ablated** | **29.9% (394/1319)** | **10.6% (140/1319)** | ⚠️ **CRITICAL DISCREPANCY** |

**The GSM8k discrepancy is 3× (33.2% vs. 11.3%).** The paper's 33.2% GSM8k baseline is cited as the definitive capability measure across multiple sections (§5.6, §5.7, §5.8, §5.10). The archive shows 11.3% for the same N=1319 strict-scoring evaluation. 

The most likely explanation is a change in the GSM8k evaluation script between the archived run (March 2) and the final run. The paper notes "strict normalized numeric equality on generated completions" — if the answer extraction logic changed between runs (e.g., improved regex for parsing model-generated explanations), scores could differ significantly. Llama-3-8B-Instruct has known variability on GSM8k depending on whether chain-of-thought generations are parsed correctly for final numeric answers.

**The capability retention narrative ("90.0% retained") and power calculation (80% power to detect ±3.6 pp at N=1500) assume the 33.2% baseline. With the archive's 11.3% baseline, the retention framing changes entirely.**

### 2.6 Corrected Ablation, Steering, DPO, and Mistral Results

All of the following files referenced in the paper are **absent from the repository entirely**:

| File | Paper Section | Archive Status |
|------|--------------|----------------|
| `results/corrected_ablation_results.json` | §5.6.1 | ❌ Not in archive |
| `results/steering_results.json` | §5.8 | ❌ Not in archive |
| `results/steering_per_source_analysis.json` | §5.8 | ❌ Not in archive |
| `results/dpo_eval_results.json` | §5.11 | ❌ Not in archive |
| `results/dpo_training_metrics.json` | §5.11 | ❌ Not in archive |
| `results/full_rerun_manifest.json` | Multiple | ❌ Not in archive |
| `results/mistral/baseline_summary.json` | §5.10 | ❌ Not in archive |
| `results/mistral/top10_ablation_full_gsm8k.json` | §5.10 | ❌ Not in archive |
| `results/mistral/probe_control_balanced_results.json` | §5.10 | ❌ Not in archive |
| `results/mistral/head_importance.json` | §5.10 | ❌ Not in archive |
| `results/mistral/steering_results.json` | §5.10 | ❌ Not in archive |

**The DPO probe re-analysis (Section 5.11) — the paper's primary novelty claim — relies entirely on `dpo_eval_results.json`, which does not exist in the repository.**

---

## 3. Figure Inventory and Assessment

All 6 figures exist in `figures/` as both `.pdf` and `.png`. Contents:

| Figure | File | Description | Assessment |
|--------|------|-------------|------------|
| Fig 1 | `fig1_patching_heatmap.png` | Layer × position patching recovery heatmap (left) + layer importance bar chart (right). Red = positive recovery, blue = negative. Early layers (0–5) show highest positive importance, consistent with paper text. | ✓ Consistent with paper's qualitative claims |
| Fig 2 | `fig2_steering_sweep.png` | Overall sycophancy rate vs. alpha (left) + capability retention (right) for all 8 layers. At alpha ≤ 1.0, all layers hold near baseline. At alpha 2–3, L3–L5 shoot to ~0.84+. At alpha 50, L20 reaches ~0.16. | ✓ Consistent with paper claims; the "null at safe alpha" finding is visually clear |
| Fig 3 | `fig3_steering_per_source.png` | Opinion-domain sycophancy vs. alpha per layer. Baseline 83.0% with 95% CI band. L15 and L20 drop below the CI at alpha=2–5 before capability degrades. | ✓ Visually supports the paper's per-source steering claim; L15 α=2 drops to ~76% |
| Fig 4 | `fig4_probe_accuracy.png` | Neutral CV accuracy (blue) vs. biased transfer accuracy (orange) across all 32 layers; pink shading = "social compliance gap." | ⚠️ **Internal inconsistency** (see below) |
| Fig 5 | `fig5_ablation_comparison.png` | Bar chart showing sycophancy rate for all ablation conditions (validated top-3 in green, original top-3 in orange). All bars ~0.28, no condition significantly different from baseline. | ✓ Visually compelling for null result; error bars appropriate |
| Fig 6 | `fig6_dpo_probe_decomposition.png` | Pre-DPO vs. Post-DPO probe decomposition at Layer 1. Shows SC 18.0%→11.4%, BC 10.1%→8.3%, Robust 59.9%→75.5%, Other 12.1%→4.8%. | ⚠️ **Cannot verify against local data** (DPO results absent); figure is internally consistent with paper text |

**Figure 4 inconsistency:** The figure shows biased transfer accuracy at layers 0–12 as approximately **86%**, with a sharp drop at layer 13 to ~60–63%. However, the paper's text (§5.5) reports Layer 1 biased transfer as **77.9%**, Layer 0 as **65.1%**, Layer 2 as **70.1%**. These text-table values match the archive JSON's `biased_cv_accuracy` field. The figure appears to plot `biased_accuracy_full_probe` (Layer 1 = 85.9% in archive) rather than the CV accuracy. This distinction is not explained anywhere in the paper. The figure and table therefore describe different quantities under the same label "biased transfer accuracy," creating a misleading figure-text disconnect.

**Figure 6 rounding error:** The figure labels the belief corruption change as **−1.7 pp** (10.1% − 8.3% = 1.8 pp). The paper text says **−1.8 pp**. Minor, but inconsistent.

---

## 4. Statistical Claims Assessment

### 4.1 Fisher's Exact Test (p < 0.001, social compliance vs. belief corruption)

**Assessment: Appropriate test, plausible value.**

Fisher's exact test for a 2×2 table (SC vs. BC, pre- vs. post-DPO) is the correct non-parametric choice for count data. Given the claimed cell counts (18.0% vs. 10.1% pre-DPO, over N=1500), p < 0.001 is plausible assuming the counts are as stated. The test is also used for the SC > BC comparison within a single timepoint, which is a reasonable application. However, the archive's Layer 1 decomposition (22.5% SC, 5.5% BC) gives an odds ratio of ~4.8:1, versus the paper's claimed 1.8:1 — both yield p < 0.001, but the effect magnitude differs substantially.

### 4.2 Two-proportion z-test (z=0.28, p=0.78) for top-10 ablation

**Assessment: Appropriate test, result verified from archive counts.**

Comparing 420/1500 (baseline) vs. 427/1500 (ablated): p̂₁ = 0.280, p̂₂ = 0.285. Under pooled proportion p̄ = 0.2823, SE = √(p̄(1-p̄)(1/1500+1/1500)) ≈ 0.0163. z = (0.285 − 0.280)/0.0163 ≈ 0.31. The paper reports z=0.28 — slightly different, but plausibly from a two-sided test with slightly different rounding. The archival counts (420 vs. 427) confirm the null result is genuine. **p=0.78 and 95% CI [−2.9 pp, +3.9 pp] are consistent with these counts.** ✓

### 4.3 Power calculation (80% power to detect ±3.6 pp at N=1500)

**Assessment: Plausible; depends on which baseline is assumed.**

For a two-proportion z-test at α=0.05, two-sided, N=1500 per group, with baseline p=0.28: the minimum detectable effect at 80% power is approximately ±3.5–3.7 pp, consistent with the paper's ±3.6 pp. ✓ However, the power calculation is only reported for the sycophancy test; the GSM8k capability test has much higher variance and the power analysis for GSM8k retention is not reported.

### 4.4 Cohen's h effect sizes

**Assessment: Correctly computed, but context needed.**

- Opinion (82.4%) vs. Reasoning (0.0%): h = 2*arcsin(√0.824) − 2*arcsin(√0.000) ≈ 2.28 ✓
- Opinion (82.4%) vs. Factual (1.6%): h = 2*arcsin(√0.824) − 2*arcsin(√0.016) ≈ 2.03 ✓

Values are correct. However, Cohen's h ≥ 2.0 is extremely large and primarily reflects the forced-choice measurement setup: when one rate is 0% or near-0%, any comparison against a high rate will produce inflated h values. The paper should acknowledge that h is dominated by the floor effects in reasoning and factual sycophancy. The JSON-stored Cohen's d values (0.18, 0.78) for the compliance gap are more informative and should be reported alongside h.

### 4.5 Benjamini-Hochberg FDR correction (steering)

**Assessment: Appropriate, but results are absent from repository.**

BH correction across 56 single-layer steering conditions is the correct approach for multiple comparisons. The paper reports no condition survives FDR at the aggregate level, while per-source (N=436) analysis shows L15 and L20 at α=2.0 reduce opinion sycophancy beyond the Wilson CI. This is a valid approach to identifying signal in a noisy sweep. However, the steering results file is absent from the repository, so no numerical verification is possible.

### 4.6 DPO Fisher's exact tests

**Assessment: Claimed results plausible but unverifiable.**

The paper reports Fisher's exact test p < 0.001 for both the SC reduction (18.0% → 11.4%) and robust tracking increase (59.9% → 75.5%) post-DPO. Given the sample sizes (N=1500 each) and the claimed shifts, p < 0.001 is almost certainly correct. However, both tests are unverifiable without the DPO probe results file.

---

## 5. Methodology Assessment

### 5.1 Neutral-Transfer Probe Design

**Validity: Largely sound, with one significant caveat.**

Training probes exclusively on neutral-prompt activations and testing on biased-prompt activations is a well-motivated design for avoiding format confounds. The probe control result (showing mixed-training probes learn prompt-format cues rather than truth directions) is a valuable methodological demonstration.

**Concern:** The "best layer" selection criterion (Layer 1 = highest biased CV accuracy) creates a winner's curse: choosing the layer that maximizes cross-condition transfer will systematically favor layers where the probe does best. The paper should either report results averaged across early layers (e.g., 0–12 as shown in Figure 4) or use a holdout set for layer selection. As reported, the SC/BC decomposition at "the best layer" may overstate how well the probe captures truth-tracking more generally.

**Concern:** The "best layer" is labeled Layer 1 in the paper's tables and claimed to have 77.9% biased transfer. But the archive JSON's summary field says `best_layer: 14` (the layer with highest *neutral* CV accuracy, 94.6%). The paper's usage of "best" is ambiguous and inconsistent with the summary-level metadata.

### 5.2 400 DPO Training Pairs

**Assessment: Small but justified, though not without risk.**

400 opinion-domain pairs is a small training set. The paper acknowledges this and reports rapid convergence (3 minutes on A100, train loss 0.69 → 0.16) with eval loss stable at 0.42. The disjoint seed design (seed=100 for DPO, seed=42 for evaluation) is appropriate and important for avoiding contamination.

**Concern:** Rapid convergence on a small dataset could indicate either efficient learning or surface-level overfitting to the specific Anthropic model-written-evals format. The paper does not test transfer to out-of-distribution opinion prompts, so it is unknown whether the DPO model generalized or memorized the distribution. LoRA rank 16 with 400 pairs is aggressive — rank 4 or 8 might have been more appropriate to avoid overfitting.

**Concern:** DPO with β=0.1 is a relatively strong KL constraint, which might limit the degree of behavioral change. The 23.8 pp reduction with β=0.1 is substantial and suggests the model is learning, not just being conservatively shifted. However, comparing pre/post with identical probes trained fresh on the DPO model is the right approach, and the figure is compelling if the underlying data is correct.

### 5.3 34/100 Patching Success Rate

**Assessment: Concerning but disclosed.**

Only 34/100 samples had total effect > 0.1, meaning layer importance scores were computed from only 34 samples. The effective N for layer ranking is small. The paper discloses this but justifies it as "conservative but appropriate." 

The real concern is with Phase 2 (head-level patching, N=100): the 100 samples include the 66 where sycophancy wasn't behaviorally expressed, contributing near-zero recovery scores that dilute the head ranking. The resulting rankings are therefore based on a mixture of meaningful signal (the 34 sycophantic samples) and noise (the 66 non-sycophantic samples). Combined with the very high standard deviations on head recovery scores (std > mean for top heads), this means the specific head rankings should be treated as rough indicators rather than precise measurements.

The paper's own finding that different runs produce different top-3 heads confirms this instability, but this is presented as a note rather than a substantive methodological concern.

### 5.4 Binary Forced-Choice Format

**Assessment: Valid concern, partially addressed.**

The 0.0% GSM8k sycophancy rate is striking. The paper argues this reflects genuine domain-specific immunity, supported by the base model comparison (21.8% GSM8k sycophancy in the base model). This is a valid argument. However, the forced-choice format may make the correct arithmetic answer salient enough to overwhelm the social pressure signal even when it wouldn't in free-form generation.

The paper's limitation section acknowledges this, noting "the 0.0% reasoning sycophancy may partly reflect high model confidence in arithmetic making the forced choice trivial." This is appropriately disclosed but could be more prominent. All five main contributions are measured with this format; the DPO probe analysis is particularly dependent on the forced-choice structure, as the model's behavior is measured through token probability comparison rather than generated text.

### 5.5 Mean Ablation Catastrophic Failure

**Assessment: Methodologically problematic — no explanation provided.**

Both the top-3 and top-10 mean ablations caused "catastrophic output degradation (all outputs unparseable)." This is a significant finding that is simply excluded from analysis with no mechanistic explanation. Zero-ablation works fine (capabilities are preserved), but mean-ablation destroys the model entirely. This suggests the mean activation vectors for these heads contain important structural information or that mean-ablation creates out-of-distribution inputs for downstream components. This phenomenon deserves investigation, as it suggests the selected heads are important for general capability even if not specifically for sycophancy.

---

## 6. Summary of Issues by Severity

### 🔴 Critical

1. **Missing results directory:** The canonical `results/` directory and `full_rerun_manifest.json` do not exist. The paper's reproducibility claims cannot be verified. The DPO probe re-analysis — the primary novelty contribution — is entirely unsupported by locally available files.

2. **Probe decomposition mismatch:** Layer 1 SC/BC/robust numbers differ between the archived file (SC=22.5%, BC=5.5%, robust=63.3%) and paper text (SC=18.0%, BC=10.1%, robust=59.9%). The stated SC/BC ratio of 1.8:1 cannot be confirmed; the archive gives 4.1:1. The archived figure was generated from a different version of this analysis.

3. **GSM8k baseline discrepancy:** The top-10 ablation archive shows GSM8k baseline = 11.3% (149/1319), while the paper reports 33.2% (438/1319). This 3× difference appears to stem from evaluation script changes between runs and undermines the capability retention narrative throughout §5.7–5.11.

### 🟠 Moderate

4. **Chen et al. (2024) mischaracterization:** The paper incorrectly attributes "path patching" and "head-level knockout" to Chen et al., who instead use gradient/activation-based module selection and targeted fine-tuning (SPT). This misrepresents a key comparative work in Related Work.

5. **Head ranking instability (not adequately disclosed):** Top-3 head rankings changed substantially between two runs of the same patching pipeline (L5H31 went from recovery −0.171 to +0.256). Standard deviations exceed means for all top heads. This instability is acknowledged as a note but not characterized as a methodological limitation.

6. **Figure 4 plots different quantity than text table:** The figure shows `biased_accuracy_full_probe` (~86% at Layer 1) while the text reports `biased_cv_accuracy` (77.9%). These are not equivalent; the figure and table are inconsistent without explanation.

7. **Figure 6 rounding:** Belief corruption change labeled −1.7 pp in figure but −1.8 pp in text.

### 🟡 Minor

8. **Samples skipped discrepancy:** Paper says 7 skipped (N=1493), archive shows 0 skipped (N=1500).

9. **Effect size presentation:** Cohen's h values (2.276, 2.022) are reported without the corresponding Cohen's d values on the continuous compliance gap (0.18, 0.78). The very large h values are largely floor-effect artifacts of comparing against 0% rates.

10. **Li et al. (2025) year:** Published at AAAI 2026; "Li et al. (2025)" uses preprint date. Acceptable but should be consistent with the journal citation.

11. **DPO best layer selection ambiguity:** JSON `best_layer: 14` (highest neutral accuracy) vs. paper's "best layer: 1" (highest biased transfer). Definition of "best" should be explicit.

12. **Mean ablation failure unexplained:** Catastrophic output degradation from mean ablation of 3–10 heads is noted but not investigated. Deserves at least a hypothesis.

---

## 7. Recommended Actions Before Submission

1. **Restore and commit the `results/` directory** (or replace all file path references with paths to `results_archive/results_20260303T225104Z/`). The full_rerun_manifest must be committed if reproducibility is claimed.

2. **Reconcile probe decomposition numbers:** Identify which run produced the paper's Layer 1 numbers (18.0% SC, 10.1% BC). If a later rerun changed the numbers, the archive should be updated and a clear note added explaining which artifact is canonical.

3. **Investigate GSM8k discrepancy:** Determine whether the 33.2% (paper) or 11.3% (archive) is correct for the ablation experiment. If the evaluation changed between runs, note this explicitly and re-run the capability retention calculations.

4. **Correct Chen et al. (2024) description:** Change "path patching" to "activation/gradient-based module identification" and "head-level knockout" to "targeted fine-tuning (SPT)." Note the challenge-induced vs. assertion-based sycophancy distinction as a key methodological difference.

5. **Fix Figure 4:** Clarify whether it plots full-probe or CV biased transfer accuracy. If both are plotted, label them separately. The figure should match the numbers in Table 5.5.

6. **Add confidence intervals to head recovery scores** or at minimum report the standard deviation alongside the mean in Table 5.4. The existing SDs reveal that top-10 rankings are noisy.

7. **Soften or qualify the "first mechanistic evidence" claim** for DPO-sycophancy, noting that Lee et al. (2024) and Yang et al. (2025) provide analogous mechanistic analyses for toxicity.

---

## 8. Sources

| Citation | URL | Relevance | Status |
|----------|-----|-----------|--------|
| Chen et al. (2024) — Pinpoint Tuning | https://proceedings.mlr.press/v235/chen24u.html | Mischaracterized in §2 | ✓ Verified |
| Li/Wang et al. (2025/2026) — Truth Overridden | https://arxiv.org/abs/2508.02087 | Key concurrent work | ✓ Verified |
| O'Brien et al. (2026) — Few Bad Neurons | https://arxiv.org/abs/2601.18939 | SAE sycophancy comparison | ✓ Verified |
| Heimersheim & Nanda (2024) | https://arxiv.org/abs/2404.15255 | Sufficiency/necessity foundation | ✓ Verified |
| McGrath et al. (2023) — Hydra Effect | https://arxiv.org/abs/2307.15771 | Self-repair theory | ✓ Verified |
| Rushing & Nanda (2024) — Self-Repair | https://proceedings.mlr.press/v235/rushing24a.html | Self-repair at scale | ✓ Verified |
| Lee et al. (2024) — DPO + Toxicity | https://proceedings.mlr.press/v235/lee24a.html | Prior mechanistic DPO work | Drops "first claim" scope |
| Yang et al. (2025) — DPO Toxicity Neurons | https://arxiv.org/abs/2411.06424 | Prior mechanistic DPO work | Drops "first claim" scope |

**Dropped sources:**
- OpenReview / liner.com review summaries — secondary, redundant with primary sources
- General Sharma et al. and Wei et al. characterizations verified through ICLR proceedings and descriptions; no discrepancies found
