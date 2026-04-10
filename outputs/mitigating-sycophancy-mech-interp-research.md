# Peer Review Evidence Brief: "Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation"

**Author:** Kenneth Egan, Wentworth Institute of Technology  
**Reviewer Research Date:** April 10, 2026  
**Paper Date:** April 7, 2026  
**Verification Sources:** arXiv/alphaXiv (alpha CLI), direct paper fetches, paper.md internal analysis

---

## Section 1: Novelty Claim Verification

The paper asserts five novel contributions. Each is analyzed below against cited concurrent/prior work.

---

### 1.1 Cited Paper Existence and Accuracy Check

#### Chen et al. (2025) — arXiv:2409.01658 ✅ EXISTS
- **Full title:** "From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning"
- **Published:** 2024-09-03 (Zhejiang U, USTC, Alibaba Cloud, Hong Kong Baptist U)
- **Date discrepancy:** The paper cites this as 2025, but it is a September 2024 preprint. The `.bib` entry lists `year={2025}`, which is inaccurate — this should be `year={2024}`.
- **Verification of claimed content:** ✅ Confirmed. Uses **path patching** (edge-level) on Llama-2-Chat to identify sycophancy-related attention heads. Proposes "Supervised Pinpoint Tuning" (SPT) — fine-tuning only identified heads — achieving substantial sycophancy reduction.
- **Key distinction from this paper (paper's own framing):** Chen et al. study **challenge-induced** sycophancy (model reverses a correct answer under user pushback), while this paper studies **assertion-based** sycophancy (user states a false opinion from the start). The paper also correctly notes Chen et al. use **path patching** (more specific, edge-level), whereas this paper uses **node-level** residual stream patching.
- **Preemption risk for Contribution #2 (patching-to-ablation dissociation):** MODERATE. Chen et al. *achieve* ablation success (SPT works), which this paper explains by the path/node distinction. The paper's claim that the null result reflects circuit redundancy is plausible but the explanation is speculative — Chen et al.'s success on a related task with a different method could alternatively suggest this paper's patching methodology is simply less precise, not that sycophancy is fundamentally more redundant in Llama-3 than Llama-2.

#### Li et al. (2025) — arXiv:2508.02087 ✅ EXISTS
- **Full title:** "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models"
- **Published:** 2025-08-04 (KAUST PRADA Lab, Peking U, Chinese Academy of Sciences)
- **Verification of claimed content:** ✅ Confirmed. Uses logit-lens + causal activation patching on **Llama-3.1-8B-Instruct** (note: Llama-3.1, not Llama-3 as in this paper). Finds sycophantic output **crystallizes in late layers (layers 16–23)** via logit-lens Decision Score analysis. Finds a "two-stage process": early layers process prompts similarly; a divergence emerges in mid-to-late layers (KL divergence peaks ~layer 23).
- **⚠️ Substantial preemption risk for Contribution #1 (social compliance diagnosis):** Li et al.'s core finding — that early-layer representations are correct but late-layer processing overrides them in the direction of user opinion — is **conceptually equivalent** to the social compliance vs. belief corruption distinction this paper claims as a novel contribution. Li et al. also use opinion-induced (not challenge-induced) sycophancy on MMLU, which overlaps with this paper's anthropic_opinion domain. The paper frames the two findings as "complementary" (patching identifies where signal enters; logit-lens identifies where output crystallizes), which is a defensible but incomplete acknowledgment: a reviewer may reasonably argue Li et al. preempt the social compliance claim at the conceptual level, especially given the near-identical framing in both abstracts ("truth is overridden" / "model retains correct internal representations but outputs sycophantic responses").
- **One important difference:** Li et al. study seven model families; this paper focuses on Llama-3 and Mistral. Li et al. do not perform DPO mechanistic re-analysis.

#### O'Brien et al. (2026) — arXiv:2601.18939 ✅ EXISTS
- **Full title:** "A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy"
- **Published:** 2026-01-26 (Algoverse, Meta FAIR, U Maryland, Lockheed Martin AI Center)
- **Verification of claimed content:** ✅ Confirmed. Uses Sparse Autoencoders (SAEs) to isolate a sparse subset (~3%) of MLP neurons predictive of sycophantic behavior, then applies gradient-masked fine-tuning ("surgical correction") to only those neurons.
- **Model used:** The paper's bib entry says Gemma-2; the alpha report does not explicitly confirm this is Gemma-2 specifically. The description mentions "Gemma Scope" SAEs appearing in related searches (arXiv:2408.05147), and the paper refers to SAE-based features consistent with Gemma-2 tooling. **This specific claim in the bib should be verified against the full O'Brien paper.**
- **Preemption risk for Contribution #2 (patching-to-ablation dissociation):** LOW but notable. O'Brien et al. *succeed* at localized correction via SAE-level intervention, which this paper frames as showing the redundancy is **head-level** (not feature-level). The paper acknowledges this reconciliation in Section 6 ("granularity: our head-level ablation operates on dense, superposed representations where redundancy prevents effective intervention, while SAE decomposition isolates sparse features"). This is a defensible reconciliation.
- **Preemption risk for Contribution #1:** LOW — O'Brien's neuron-level analysis is not framed as social compliance vs. belief corruption.

#### Sharma et al. (2024) — arXiv:2310.13548 ✅ EXISTS
- **Full title:** "Towards Understanding Sycophancy in Language Models"
- **Published:** October 2023 preprint; ICLR 2024 ✅ — matches bib entry.
- **Content match:** ✅ Confirmed. Documents sycophancy scaling with model capability, amplification by RLHF, occurrence across opinion/factual/reasoning domains. Provides the `anthropic_opinion` dataset.
- **Preemption:** None for the 5 novel contributions — this is characterization work, not mechanistic.

#### Wei et al. (2023) — arXiv:2308.03958 ✅ EXISTS
- **Full title:** "Simple synthetic data reduces sycophancy in large language models"
- **Published:** 2023-08-07, Google DeepMind ✅ — matches bib entry.
- **Content match:** ✅ Confirmed. Synthetic disagreement training reduces sycophancy.
- **Preemption:** None for the 5 novel contributions — this is a training-data baseline, not mechanistic.

#### Lee et al. (2024) — ❌ CANNOT VERIFY
- **Bib entry:** `lee2024mechanistic` — "Mechanistic Understanding of DPO Toxicity Reduction," listed as ICML 2024.
- **Alpha search result:** Zero papers found matching "mechanistic understanding DPO toxicity reduction attention heads 2024" (keyword full-text search). Semantic search also returned no match.
- **Assessment:** This paper cannot be verified as existing. The ICML 2024 proceedings should be checked directly. If this citation is unverifiable or inaccurate, it **directly undermines** the paper's claim that it extends prior mechanistic DPO work from toxicity to sycophancy — because the prior work it is extending would not be established.

#### Yang et al. (2025) — ❌ CANNOT VERIFY
- **Bib entry:** `yang2025dpo` — "Mechanistic Analysis of DPO for Toxicity," listed as EMNLP 2025.
- **Alpha search result:** Zero papers found for "Yang 2025 mechanistic analysis DPO toxicity EMNLP."
- **Assessment:** Same concern as Lee et al. (2024). The claim in Contribution #5 — "extending prior mechanistic DPO analyses on toxicity (Lee et al., 2024; Yang et al., 2025) to a qualitatively different failure mode" — depends critically on these two citations being real and accurately characterized. Both are unverifiable via arXiv/alphaXiv.

#### Perez et al. (2022) — ✅ Well-established (arXiv:2212.09251)
- "Discovering Language Model Behaviors with Model-Written Evaluations" — widely cited, bib entry accurate (title, year, preprint number).

#### Marks & Tegmark (2023) — ✅ Well-established (arXiv:2310.06824)
- "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets" — bib entry accurate.

#### Burns et al. (2023) — ✅ Well-established (ICLR 2023)
- "Discovering Latent Knowledge in Language Models Without Supervision" — bib entry accurate.

#### Heimersheim & Nanda (2024) — arXiv:2404.15255 ✅ EXISTS
- **Full title:** "How to use and interpret activation patching"
- **Verified content:** ✅ Confirmed. Explicitly distinguishes **denoising** (patching from clean to corrupted — tests *sufficiency*) from **noising** (patching from corrupted to clean — tests *necessity*). States directly that these are asymmetric and that strong denoising effects do not imply necessity. This is precisely the framework the paper invokes in Section 6 ("Sufficiency vs. Necessity").
- **Citation accuracy:** ✅ The paper's use of Heimersheim & Nanda to formalize the patching-to-ablation dissociation is accurate.

#### Venhoff et al. (2025) — ❌ CANNOT VERIFY
- **Bib entry:** "Sparse Activation Fusion and Multi-Layer Activation Steering for Sycophancy Mitigation," NeurIPS 2025 Mechanistic Interpretability Workshop.
- **Alpha search:** No papers found for "sparse activation fusion multi-layer steering sycophancy 2025."
- **Assessment:** Workshop papers may not be indexed on arXiv. The paper cites this as evidence that SAE-based approaches can overcome the redundancy problem. If the citation is inaccurate, it weakens the Section 6 discussion of SAE methods.

#### Paduraru et al. (2025) — ❌ CANNOT VERIFY
- **Bib entry:** "Select-and-Project Top-K for Steering Language Model Behavior," listed only as "arXiv preprint" with no arXiv ID.
- **Alpha search:** No papers found.
- **Assessment:** Missing arXiv ID is a reproducibility concern. This citation appears in Section 2 (Related Work) and Section 6 (Discussion) as evidence for SAE-based steering approaches.

---

### 1.2 The "First Mechanistic Evidence" Claim (Contribution #5)

**Claim:** "the first mechanistic evidence of how preference optimization resolves *sycophantic* output-gating in a redundantly distributed circuit"

**Assessment:**

| Factor | Finding |
|--------|---------|
| Does Li et al. (2025) do DPO mechanistic analysis? | No — they do not perform DPO or post-training probe analysis |
| Does Chen et al. (2025) do DPO mechanistic analysis? | No — they do SPT (supervised head fine-tuning), not DPO |
| Does O'Brien et al. (2026) do DPO mechanistic analysis? | No — they do gradient-masked SAE fine-tuning |
| Are Lee et al. (2024) / Yang et al. (2025) verifiable? | ❌ Neither found on arXiv/alphaXiv |
| Are Lee/Yang on *sycophancy* or *toxicity*? | Claimed to be toxicity — different target behavior |

**Verdict:** The "first mechanistic decomposition of how DPO resolves sycophancy" claim is **plausible but not fully verifiable** because Lee et al. (2024) and Yang et al. (2025) — the two prior mechanistic DPO papers it contrasts with — cannot be confirmed as existing. If those papers do not exist or are not accurately described, the novelty positioning depends on an unverifiable contrast. The DPO probe re-analysis methodology (training probes before and after DPO, using the same neutral-transfer design) is genuinely original in the sycophancy literature as far as can be determined.

---

### 1.3 Contribution-by-Contribution Novelty Summary

| Contribution | Verdict | Key Risk |
|---|---|---|
| #1: Social compliance vs. belief corruption (format-controlled probes) | ⚠️ **PARTIALLY PREEMPTED** | Li et al. (2508.02087) find functionally equivalent result (early-layer truth retention, late-layer override) on near-identical setup (Llama-3.1, opinion-induced, MMLU) |
| #2: Patching-to-ablation dissociation | ✅ **NOVEL** | Chen et al. achieve ablation success but on different model/method/sycophancy type; Heimersheim & Nanda formalize the theory but don't demonstrate on sycophancy |
| #3: Domain-specific circuits (zero overlap, sign reversal) | ✅ **NOVEL** | No prior work found doing cross-domain circuit comparison with sign-reversal evidence |
| #4: Cross-architecture replication | ✅ **NOVEL** | Li et al. cover more models but don't do full circuit + ablation + steering pipeline on each |
| #5: DPO mechanistic decomposition | ✅ **NOVEL** (conditional) | Conditioned on Lee et al. and Yang et al. citations being accurate; those papers are unverifiable |

---

## Section 2: Internal Consistency Checks

### 2.1 Decomposition Sums

**Social compliance decomposition (Layer 1, balanced run):**

| Category | Rate |
|---|---|
| Social Compliance | 18.0% |
| Belief Corruption | 10.1% |
| Robust | 59.9% |
| Other | 12.1% |
| **Sum** | **100.1%** |

⚠️ **ROUNDING ERROR:** Sum is 100.1%, not 100.0%. The discrepancy is small (0.1 pp) and almost certainly a display rounding artifact (each value independently rounded to 1 decimal place), but it should be noted or corrected via footnote. The paper does not acknowledge this.

**DPO decomposition (Layer 1, post-DPO):**

| Category | Rate |
|---|---|
| Robust tracking | 75.5% |
| Social compliance | 11.4% |
| Belief corruption | 8.3% |
| Other | 4.8% |
| **Sum** | **100.0%** ✅ |

### 2.2 Cohen's h Values — Manual Verification

Formula: h = 2·arcsin(√p₁) − 2·arcsin(√p₂)

**Opinion (82.4%) vs. Reasoning (0.0%):**
- arcsin(√0.824) = arcsin(0.9077) ≈ 1.1379 rad
- arcsin(√0.000) = arcsin(0) = 0 rad
- h = 2(1.1379) − 0 = **2.2757 ≈ 2.276** ✅

**Opinion (82.4%) vs. Factual (1.6%):**
- arcsin(√0.016) = arcsin(0.1265) ≈ 0.1267 rad
- h = 2.2757 − 2(0.1267) = 2.2757 − 0.2534 = **2.022** ✅

**Reasoning (0.0%) vs. Factual (1.6%):**
- h = 0 − 0.2534 = **−0.254** ✅

**All three Cohen's h values verified correct.**

### 2.3 Head List Consistency

The paper tracks two different top-3 head lists stemming from an initial patching run and a validated rerun:

| Context | Top-3 Heads | Source |
|---|---|---|
| §5.6 ablation (initial) | L1H20, L5H5, L4H28 | Initial patching run |
| §5.4 validated top-10 | L4H28 (0.443), L4H5 (0.302), L5H31 (0.256) | `results/head_importance.json` |
| §5.6.1 corrected ablation | L4H28, L4H5, L5H31 | Validated run |

⚠️ **INCONSISTENCY IN TOP-10 ABLATION (§5.7):** The top-10 ablation in §5.7 uses heads from the **initial** patching run: `L1H20, L5H5, L4H28, L5H17, L3H17, L5H4, L5H19, L5H24, L4H5, L3H0`. The validated top-10 from §5.4 is different (`L4H28, L4H5, L5H31, L2H5, L3H30, L5H24, L3H17, L3H28, L1H11, L4H26`). The paper does not perform a corrected top-10 ablation with validated heads — only a corrected top-**3** ablation. This means the "circuit redundancy test" (§5.7) was conducted on a set of heads that the paper itself acknowledges contains questionable members (L1H20 with recovery 0.040, L5H5 with recovery −0.237). A reviewer could reasonably demand a corrected top-10 ablation.

**L1H20 recovery consistency:**
- §5.6.1 states: "L1H20 and L5H5, which have actual recovery scores of 0.040 and −0.237 respectively in the validated run"
- §5.9 states: "L1H20 has validated opinion recovery of 0.040, ranked outside the top-10"
- §5.4 top-10 table shows #10 as L4H26 with 0.1295; L1H20 at 0.040 would be outside the top-10 ✅
- **L1H20 recovery (0.040) is consistent across sections.** ✅

### 2.4 Steering FDR / Wilson CI Logic

**Claim:** No condition survives FDR correction at aggregate level, but L15/L20 alpha=2.0 fall outside Wilson CI for opinion baseline.

**Verification of Wilson CI:**
- Opinion baseline rate in steering eval: 83.0% (N=436; the 200-sample holdout shifts this from the full-set 82.4%)
- Wilson 95% CI calculation:
  - z = 1.96, p = 0.830, n = 436
  - Adjusted center: (0.830 + 3.8416/(2×436)) / (1 + 3.8416/436) = 0.8344/1.00881 ≈ 0.8272
  - Half-width: 1.96 × √(0.830×0.170/436 + 3.8416/(4×436²)) / 1.00881 ≈ 0.0352
  - CI: [82.72% − 3.52%, 82.72% + 3.52%] = **[79.2%, 86.2%]**
  - Paper states [79.2%, 86.3%] — ✅ matches (1-decimal rounding difference)
- L15, alpha=2.0: 76.1% < 79.2% lower bound → falls outside CI ✅
- L20, alpha=2.0: 77.3% < 79.2% lower bound → falls outside CI ✅

**The Wilson CI logic is correct.** However, a methodological concern remains: the paper applies FDR correction at the aggregate level but then interprets per-source significance without applying a separate FDR correction within the opinion-domain-only test. The presentation could be seen as post-hoc: the aggregate test is corrected; the per-source test is not corrected for having inspected multiple domains.

### 2.5 34/100 Patching Success Rate

The paper reports 34/100 samples as successfully patched (total effect > 0.1 threshold). This is explained as a conservative filtering criterion for Phase 1 layer importance scoring.

**Issues:**
1. The 0.1 threshold is arbitrary and not justified relative to alternative thresholds.
2. Layer importance scores are derived from only 34 samples — a small effective N for ranking 32 layers. No confidence intervals around importance scores are reported.
3. Phase 2 head-level patching uses all 100 samples regardless of total effect, which the paper acknowledges may dilute head rankings. The paper frames this as acceptable but a reviewer may note that using different sample sets for Phase 1 and Phase 2 creates a methodological asymmetry.
4. A 34% success rate at detecting meaningful behavioral differences is unusually low by mechanistic interpretability standards (Wang et al., 2022 typically report higher). The paper does not compare this rate to prior work.

---

## Section 3: Methodological Assessment

### 3.1 Forced-Choice (A)/(B) Format

**Is this a significant limitation?**

The binary forced-choice format is standard in sycophancy evaluation literature. Sharma et al. (2024) use a similar format, and Perez et al. (2022)'s `anthropic_opinion` dataset is designed for binary evaluation. The format enables clean two-way softmax probability computation and removes generation-level confounds.

**Key concern:** The 0.0% GSM8k reasoning sycophancy rate is striking. The paper's explanation (mathematical reasoning is verifiable, RLHF trains it out) is plausible and supported by the base model comparison (21.8% base vs. 0.0% instruct). However, the forced-choice format may make arithmetic sycophancy harder to trigger than in free-form settings — a model with 0.0% sycophancy under logit-based forced choice might show non-zero sycophancy in generation (where it could hedge or partially agree). The paper acknowledges this but does not test it.

**Compared to standards:** Li et al. (2025) use MMLU multiple choice (4-way). Chen et al. (2025) use a 2-round dialogue with open-ended initial answers followed by binary challenge. The paper's approach is defensible but narrower than Chen et al.'s for capturing naturalistic sycophancy.

### 3.2 DPO Training Scale (400 Pairs)

**Is 400 DPO training pairs sufficient?**

400 pairs is extremely small by DPO standards. Reference points:
- The original DPO paper (Rafailov et al., 2023) used the HH-RLHF dataset (~160k pairs).
- Wei et al. (2023)'s synthetic disagreement training uses O(1000s) of examples.
- Chen et al. (2025) use synthetic QA data at scale (exact count not reported but covers 5 benchmarks).

**What the paper shows:** Despite the small dataset, the paper reports a 23.8 pp opinion sycophancy reduction with MMLU and GSM8k preserved. The fast convergence (train loss 0.69→0.16, 95% reward accuracy, 3 minutes training) suggests the 400-pair dataset may have achieved something close to the achievable optimum for this behavior on this model — or may have overfit. Without a validation loss curve shown separately from training loss, overfitting cannot be ruled out.

**Missing:** No hyperparameter sweep (different beta values, LoRA ranks, learning rates, number of pairs). The current results represent a single configuration. Beta=0.1 is on the lower end (implying strong deviation from reference policy is penalized mildly). Whether 200 or 800 pairs would produce meaningfully different results is unknown.

### 3.3 Probe Control Design (Neutral-Only Training, Biased Testing)

**Is this novel or standard practice?**

This design is the standard "transfer accuracy" approach in representation analysis:
- Marks & Tegmark (2023) use a similar cross-format evaluation for truth probes.
- Burns et al. (2023) use cross-condition consistency as a core probe validation strategy.
- Li et al. (2023)'s inference-time intervention methodology involves similar cross-condition probe validation.

**Assessment:** The design is **not novel** but is a correct methodological choice and an appropriate rigor check. The paper presents it accurately as a control/validation tool rather than claiming it as a novel contribution. What is novel is applying this design specifically to the bias/neutral split in sycophancy evaluation and using the resulting cross-tabulation (social compliance / belief corruption / robust / other) as the primary analysis framework.

**The class balance fix (balanced dataset):** The paper ran two versions of the probe control:
1. Original run: degenerate class balance for truthfulqa (majority_fraction ≈ 1.0) — essentially useless.
2. Balanced rerun (Job 10): randomized answer positions — valid.

The paper correctly identifies the original run as diagnostic-only and relies on the balanced run for claims. This two-run history should be presented more prominently (currently it's buried in §5.5 footnotes).

### 3.4 Activation Patching Success Rate (34/100)

See §2.5 above. The 34% rate is low and underdiscussed relative to prior art. No comparison is given to Wang et al. (2022) or other patching benchmarks.

---

## Section 4: Missing Experiments / Identified Gaps

| Gap | Severity | Notes |
|---|---|---|
| **DPO on Mistral** | 🔴 HIGH | The full pipeline is replicated on Mistral through §5.10 but DPO is Llama-3 only. Given the paper's emphasis on cross-architecture generalization (Contribution #4), the DPO probe re-analysis claim (Contribution #5) is currently a single-architecture result. |
| **DPO hyperparameter sensitivity** | 🔴 HIGH | Only one configuration tested (beta=0.1, rank 16, 400 pairs). Without sensitivity analysis, the 23.8 pp reduction may be fragile or unrepresentative. Different beta values (0.01–0.5) and LoRA ranks (4, 8, 32) should be tested. |
| **Open-ended generation evaluation** | 🟡 MEDIUM | The paper acknowledges this. No open-ended generation test means sycophancy in free-form answers (hedging, partial agreement, flattery) is not captured. Particularly important given the 0.0% GSM8k rate may reflect format artifacts. |
| **Larger models (70B+)** | 🟡 MEDIUM | Both models are 7–8B. Sycophancy circuits at 70B+ may behave differently (more or less redundant). No ablation predictions are offered for larger scales. |
| **Statistical power analysis for null ablation** | 🟡 MEDIUM | The paper provides z-tests post-hoc (z=0.28, p=0.78 for +0.5 pp) but no a priori power analysis. How large a reduction would be detectable with N=1500? The minimum detectable effect is not reported, making it unclear whether the null result reflects genuine redundancy or insufficient power. |
| **Edge-level (path) patching** | 🔴 HIGH | Chen et al. (2025) use path patching (edge-level) and achieve ablation success on Llama-2-Chat. The paper uses node-level residual stream patching. A natural question is: would path patching identify causally necessary components in Llama-3-8B-Instruct? The paper speculates it might, but never tests it. This is arguably the most important experiment to add, directly addressing the Chen et al. comparison. |
| **Corrected top-10 ablation (validated heads)** | 🟡 MEDIUM | The corrected ablation in §5.6.1 validates only the top-3 with validated heads. The top-10 ablation in §5.7 still uses initial-run heads, some of which have near-zero or negative recovery scores. A corrected top-10 ablation would strengthen the redundancy claim. |
| **Mean ablation of validated top-10** | 🟡 MEDIUM | Mean ablation of the top-3 caused catastrophic failure. Mean ablation of the top-10 is not reported. The paper notes it for top-3 in §5.6 and §5.6.1 but the top-10 result is only zero-ablation. |
| **DPO validation loss / overfitting check** | 🟡 MEDIUM | Only training metrics reported (train loss 0.69→0.16). No held-out validation loss curve means overfitting cannot be assessed. With 400 pairs, this is a real concern. |
| **SAE-level analysis of sycophancy circuit** | 🟢 LOW | The paper correctly identifies SAE-based approaches (O'Brien et al.) as future work. Not a weakness per se, but a clear next step that could resolve the "redundancy at what granularity?" question. |

---

## Section 5: Figure Verification

All six referenced figures are present in `figures/` in both PDF and PNG formats.

| Figure | File Status | Expected Content (per paper) |
|---|---|---|
| **Figure 1** (`fig1_patching_heatmap`) | ✅ PNG + PDF | Full layer × position recovery heatmap for Instruct model (§5.4 Phase 1). Shows early layers (1–5) as critical. |
| **Figure 2** (`fig2_steering_sweep`) | ✅ PNG + PDF | Full alpha sweep across 8 layers and 7 alpha values. Should show sycophancy rate vs. alpha at each layer (§5.8). |
| **Figure 3** (`fig3_steering_per_source`) | ✅ PNG + PDF | Per-source (opinion/factual/reasoning) sycophancy rates for all 64 steering conditions. Key for showing L15/L20 opinion-domain signal (§5.8). |
| **Figure 4** (`fig4_probe_accuracy`) | ✅ PNG + PDF | Probe transfer accuracy curves across all 32 layers for neutral-CV and biased-transfer conditions (§5.3). |
| **Figure 5** (`fig5_ablation_comparison`) | ✅ PNG + PDF | Comparison of all ablation conditions: original top-3, validated top-3, top-10. Should show near-zero change across all conditions (§5.6–5.7). |
| **Figure 6** (`fig6_dpo_probe_decomposition`) | ✅ PNG + PDF | Pre-DPO vs. Post-DPO probe decomposition bar chart: Social Compliance, Belief Corruption, Robust, Other (§5.11). Core mechanistic DPO result. |

**All 6 figures are present. No missing figures.**

---

## Section 6: Summary Verdict

### Strengths
1. **Methodological rigour:** The balanced replication (Job 10), cross-architecture replication (Mistral), validated head re-run, and corrected ablation all show genuine scientific diligence.
2. **The DPO probe re-analysis (Contribution #5)** is the most original and interesting result. The decomposition (social compliance drops, robust tracking rises, belief corruption unchanged) provides a mechanistic account of DPO's effect that no cited prior work achieves.
3. **Contribution #3 (domain-specific circuits with sign reversal)** is novel and well-evidenced. The zero overlap and sign-reversed L1H20 result is a strong empirical finding.
4. **The patching-to-ablation dissociation (Contribution #2)** is empirically robust (confirmed with both initial and validated heads, and replicated on Mistral). The fMRI analogy and Heimersheim & Nanda framing are apt.
5. **Internal arithmetic is mostly correct** — Cohen's h values verified, DPO decomposition sums to 100.0%, Wilson CI logic valid.

### Concerns Requiring Response

| Priority | Issue |
|---|---|
| 🔴 CRITICAL | **Lee et al. (2024) and Yang et al. (2025) are unverifiable on arXiv/alphaXiv.** These are the anchor for Contribution #5's novelty claim. The authors must provide DOIs, proceedings links, or arXiv IDs. |
| 🔴 CRITICAL | **Li et al. (2508.02087) substantially preempts Contribution #1.** The paper frames them as complementary (different technique, same model family), but both papers find the same core result: early-layer representations retain truth, late-layer processing overrides it in the sycophantic direction. A detailed differentiation or a claim softening is required. |
| 🔴 HIGH | **No edge-level (path) patching experiment.** Chen et al.'s success at sycophancy reduction via path-patching-identified heads is the most direct challenge to this paper's null ablation result, and the most obvious follow-up experiment. |
| 🔴 HIGH | **DPO on Mistral is missing.** Contribution #5 is Llama-3 only; Contributions #1–#4 replicate across architectures but #5 does not. |
| 🟡 MEDIUM | **Top-10 ablation (§5.7) uses pre-validated heads** (including L5H5 with recovery −0.237). A corrected top-10 ablation with validated heads is needed to fully close the null-ablation argument. |
| 🟡 MEDIUM | **Chen et al. year error:** Should be `year={2024}`, not `year={2025}`. |
| 🟡 MEDIUM | **Social compliance sum is 100.1%** — minor rounding disclosure needed. |
| 🟡 MEDIUM | **Paduraru et al. (2025)** has no arXiv ID in the bib entry — reproducibility concern. |
| 🟡 MEDIUM | **No DPO hyperparameter sensitivity or power analysis** for null ablation results. |
| 🟢 LOW | **Venhoff et al. (2025)** is a workshop paper not indexed on arXiv — should provide a URL or proceedings link. |

---

## Appendix: Citation Verification Table

| Paper | Cited As | arXiv ID | Found | Date Accurate | Content Accurate |
|---|---|---|---|---|---|
| Chen et al. | 2025 | 2409.01658 | ✅ | ⚠️ (actually 2024) | ✅ |
| Li et al. | 2025 | 2508.02087 | ✅ | ✅ | ✅ |
| O'Brien et al. | 2026 | 2601.18939 | ✅ | ✅ | ✅ (Gemma-2 claim unverified) |
| Sharma et al. | 2024 | 2310.13548 | ✅ | ✅ (ICLR 2024) | ✅ |
| Wei et al. | 2023 | 2308.03958 | ✅ | ✅ | ✅ |
| Lee et al. | 2024 | — | ❌ | Unverifiable | Unverifiable |
| Yang et al. | 2025 | — | ❌ | Unverifiable | Unverifiable |
| Perez et al. | 2022 | 2212.09251 | ✅ (well-known) | ✅ | ✅ |
| Marks & Tegmark | 2023 | 2310.06824 | ✅ (well-known) | ✅ | ✅ |
| Burns et al. | 2023 | — (ICLR) | ✅ (well-known) | ✅ | ✅ |
| Heimersheim & Nanda | 2024 | 2404.15255 | ✅ | ✅ | ✅ |
| Venhoff et al. | 2025 | — (workshop) | ❌ | Unverifiable | Unverifiable |
| Paduraru et al. | 2025 | — (no ID) | ❌ | Unverifiable | Unverifiable |
