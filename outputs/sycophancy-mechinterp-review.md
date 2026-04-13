# Peer Review: Mitigating Sycophancy in Large Language Models — A Mechanistic Investigation

**Paper:** Egan (2026), "Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation"
**Reviewer:** Anonymous
**Date:** April 13, 2026
**Venue format:** Standard ML conference review (NeurIPS/ICML style)

---

## 1. Summary

This paper applies a suite of mechanistic interpretability techniques — linear probing, causal activation patching, head ablation, representation steering, and DPO fine-tuning — to sycophancy in Llama-3-8B-Instruct and Mistral-7B-Instruct. Through format-controlled probes (trained on neutral prompts, tested on biased prompts), the authors establish that sycophancy is primarily *social compliance* (the model retains correct internal representations but outputs sycophantic answers) rather than *belief corruption*. Activation patching identifies attention heads carrying the sycophantic signal, but ablating the top 10 heads produces no sycophancy reduction — a "patching-to-ablation dissociation" demonstrating that these heads are sufficient carriers but not causally necessary. Control experiments on fictional entities reveal domain-specific circuits with zero head overlap and sign-reversed roles. All core findings replicate on Mistral despite entirely different circuits and sycophancy profiles. DPO fine-tuning with 360 opinion-domain preference pairs reduces opinion sycophancy by 23.8 pp (82.4% → 58.6%) while preserving capabilities; probe re-analysis reveals DPO converts social compliance into robust truth-tracking without altering internal truth representations. The paper claims five contributions: (1) format-controlled probes confirming social compliance, (2) the patching-to-ablation dissociation, (3) domain-specific circuits, (4) cross-architecture replication, and (5) the first mechanistic decomposition of how DPO resolves sycophancy.

---

## 2. Strengths

1. **Exceptional methodological transparency.** The paper reports negative results prominently (ablation null, steering null, mean-ablation catastrophe), acknowledges head ranking instability (std > mean for top heads), flags its own probe control confound (format-mixed probes learn preamble cues), and enumerates seven specific limitations. This level of self-critical reporting exceeds the norm for ML publications and significantly increases trust in the positive findings.

2. **Neutral-transfer probe design is a genuine methodological contribution.** Training probes exclusively on neutral-condition activations and testing on biased-condition activations cleanly separates truth-tracking from prompt-format classification. The probe control experiment (§5.5) demonstrating that mixed-format probes learn superficial cues is a valuable cautionary result for the broader mechanistic interpretability community, and the balanced-dataset replication with randomized answer positions eliminates position confounds. This is solid experimental design.

3. **The patching-to-ablation dissociation, empirically demonstrated at scale and replicated cross-architecture, is the paper's most impactful finding.** While the theoretical distinction between sufficiency and necessity in activation patching is known (Heimersheim & Nanda 2024), the paper provides the first clear empirical demonstration on a complex, safety-relevant behavior with both corrected and original head sets, extended to top-10 ablation, and replicated on Mistral. The neuroscience analogy (fMRI vs. lesion) is apt and makes the finding accessible.

4. **Cross-architecture replication substantially strengthens all claims.** Replicating the social compliance finding (6.4:1 SC/BC ratio on Mistral vs. 1.8:1 on Llama-3), the ablation null (+1.0 pp Mistral, +0.5 pp Llama-3), and the steering null on a model with an inverted sycophancy profile (Mistral: 99.8% factual, 50.8% opinion) transforms single-model observations into architecture-general properties.

5. **The DPO probe re-analysis is well-conceived and the novelty claim is defensible.** Applying the same neutral-transfer probe methodology pre- and post-DPO to decompose the behavioral change into representational components is a clean experimental design. The finding that DPO eliminates social compliance (−6.6 pp) while leaving belief corruption unchanged (−1.7 pp) provides a compelling mechanistic account. Literature search confirms no prior work applies probe decomposition to DPO-trained models for sycophancy specifically — extending the Lee et al. (2024) and Yang et al. (2025) toxicity analyses is a legitimate novel contribution.

6. **Domain-specific circuits with sign-reversed head roles (§5.9) is a striking result.** The zero overlap in top-5 heads between opinion and fictional-entity circuits, combined with L1H20's sign reversal, provides strong evidence against a universal sycophancy mechanism and has direct implications for intervention design.

7. **Rigorous artifact verification.** The `full_rerun_manifest.json` with `missing_count: 0`, the corrected ablation addressing the head-selection concern, and the explicit GSM8k extraction pipeline fix demonstrate mature experimental practice.

---

## 3. Weaknesses

### W1. Cohen's d labels reversed in Section 5.1
**Issue:** The paper states "d = 0.18 (opinion vs. factual), d = 0.78 (opinion vs. reasoning)" but the stored artifact (`baseline_llama3_summary.json`) records `anthropic_opinion_vs_gsm8k_reasoning: 0.1836` and `anthropic_opinion_vs_truthfulqa_factual: 0.7782`. The labels are swapped — d = 0.18 is opinion-vs-reasoning and d = 0.78 is opinion-vs-factual.
**Evidence:** Direct comparison of paper text (§5.1, paragraph after Cohen's h table) against JSON artifact keys.
**Severity:** MINOR — This is a presentation error in a supplementary effect-size metric. The Cohen's h values in the preceding sentence are correctly labeled, and no downstream analysis depends on the d values. However, the reversal could confuse readers comparing the two effect-size measures.
**Recommendation:** Swap the labels: "d = 0.18 (opinion vs. reasoning), d = 0.78 (opinion vs. factual)."

### W2. DPO train loss claim contradicts stored artifact
**Issue:** The paper states "train loss 0.69 → 0.16" but `dpo_training_metrics.json` records `train_loss: 0.356`. No step-level training logs are provided to support the 0.16 endpoint.
**Evidence:** `results/dpo_training_metrics.json` field `train_loss: 0.356` vs. paper §5.11 text.
**Severity:** MAJOR — The reported final train loss is a verifiable claim that does not match the stored artifact. If 0.16 was a single-step reading while 0.356 is the epoch average, this must be stated explicitly. As written, a reader attempting to reproduce the result would find a 2.2× discrepancy in loss convergence. This does not invalidate the DPO behavioral results (which are independently verified), but it undermines confidence in the training description.
**Recommendation:** (a) Clarify what "0.69 → 0.16" refers to — if it is step-level loss, state this and report the epoch-averaged 0.356 alongside. (b) Include the per-epoch training curve or at minimum epoch-end loss values. (c) If step-level logs are not available, report only the artifact-supported value (0.356).

### W3. "400 DPO training pairs" is imprecise
**Issue:** The paper states "We generated 400 opinion-domain DPO training pairs" and describes fine-tuning on them. The artifact shows `n_train_pairs: 360, n_eval_pairs: 40` — only 360 pairs were used for training.
**Evidence:** `results/dpo_training_metrics.json` fields.
**Severity:** MINOR — The phrasing is technically accurate ("generated 400") but contextually misleading when immediately followed by "fine-tuned ... using LoRA ... 3 epochs." A reader would reasonably interpret this as 400 training pairs.
**Recommendation:** State "We generated 400 opinion-domain DPO pairs, split into 360 training and 40 validation pairs."

### W4. Post-DPO best probe layer undisclosed
**Issue:** The DPO probe re-analysis (§5.11) reports decomposition at Layer 1 for comparability with the pre-DPO analysis, but the post-DPO model's best probe layer is actually Layer 4 (88.3% transfer accuracy vs. Layer 1's value). This methodological choice is defensible but not disclosed.
**Evidence:** `results/dpo_eval_results.json` field `best_layer: 4`.
**Severity:** MINOR — The choice to hold layer fixed for fair comparison is reasonable, and reporting at the post-DPO best layer would show even stronger effects (robust tracking +21.9 pp at Layer 4). The omission is not cherry-picking — it actually understates the result. But transparency requires disclosure.
**Recommendation:** Add a sentence: "Post-DPO, the best-transfer layer shifts from 1 to 4 (88.3% transfer accuracy); we report Layer 1 for direct comparability. Layer 4 results show qualitatively identical but larger effects (robust tracking +21.9 pp)."

### W5. Power calculation MDE is understated
**Issue:** Section 5.7 claims "80% power to detect effects of approximately ±3.6 pp" at N=1500. A correct two-proportion z-test (both groups estimated) gives MDE ≈ 4.5 pp, not 3.6 pp. The 3.6 pp figure appears to use a single-proportion SE formula treating baseline as fixed.
**Evidence:** Independent power calculation: SE = √(2 × 0.28 × 0.72 / 1500) = 0.01640; MDE = (1.96 + 0.842) × SE = 4.6 pp (two-sided).
**Severity:** MINOR — The qualitative conclusion ("the null is informative") remains valid. A 4.5 pp MDE still rules out any practically meaningful sycophancy reduction from top-10 ablation.
**Recommendation:** Recalculate using the two-proportion formula and update to ~4.5 pp. Add the formula explicitly in a footnote or methods supplement.

### W6. All DPO evaluation is in-distribution
**Issue:** Both training (360 pairs, seed=100) and evaluation (500 opinion samples, seed=42) draw from the same data-generating distribution (Anthropic model-written-evals). Despite different seeds, they share identical prompt format, question types, and domain distribution. The 23.8 pp reduction is an in-distribution result.
**Evidence:** Paper §5.11 acknowledges this in a "generalization caveat" and Limitation #7, but the abstract and conclusion present the result without qualification ("DPO fine-tuning reduces opinion sycophancy by 23.8 percentage points").
**Severity:** MAJOR — For a paper contributing to the alignment literature, the inability to assess out-of-distribution generalization is a significant gap. The 58.6% post-DPO rate — still a sycophantic majority — combined with only 360 training pairs from a single template format raises the concern that the model learned to recognize Anthropic-format opinion prompts rather than developing general sycophancy resistance. The mechanistic probe evidence partially mitigates this (social compliance → robust tracking is an internal shift, not just surface behavior), but the external validity of the behavioral claim remains untested.
**Recommendation:** (a) Add OOD evaluation on at least one other opinion-sycophancy dataset (e.g., Sharma et al. 2024's opinion benchmarks, which use different prompt templates). (b) If OOD evaluation is not feasible, qualify the abstract and conclusion claims explicitly: "reduces in-distribution opinion sycophancy by 23.8 pp."

### W7. 34/100 patching success rate creates selection bias
**Issue:** Only 34 of 100 patching samples produced sufficient total effect (>0.1) for layer importance scoring. The circuit identification is based on an effective N=34 for Phase 1, with Phase 2 using all 100 but with 66 near-zero-contribution samples diluting the signal. No confidence intervals are reported for individual head recovery scores.
**Evidence:** §5.4 reports "Samples successfully patched: 34/100" and std > mean for all top heads.
**Severity:** MINOR — The paper acknowledges head ranking instability and validates the ablation null with both head sets, which renders the exact ranking moot for the paper's main claim (the dissociation). However, the 34% rate raises the question of whether the identified circuit is representative of typical sycophancy or only extreme cases.
**Recommendation:** (a) Report bootstrap CIs for individual head recovery scores. (b) Characterize the 34 high-total-effect samples — are they disproportionately from one domain? Do they have distinctive features? (c) Discuss whether the circuit identified from these 34 samples would generalize to the remaining 66.

### W8. Post-DPO opinion sycophancy remains majority (58.6%)
**Issue:** The paper frames the 23.8 pp reduction as a success, but 58.6% opinion sycophancy means the model is *still sycophantic on the majority of opinion questions*. The abstract says DPO "reduces opinion sycophancy" without contextualizing that the post-intervention rate is well above 50%.
**Evidence:** §5.11 behavioral results table.
**Severity:** MINOR — The paper does report the absolute rate, and the mechanistic contribution (the probe decomposition) is independent of the magnitude of behavioral change. But the framing overstates the practical significance of the mitigation.
**Recommendation:** Contextualize the result: "DPO reduces opinion sycophancy from 82.4% to 58.6% — a substantial reduction but still above the 50% threshold, indicating that the majority of opinion questions remain sycophantic. Scaling the training data or combining with other methods may be necessary for full mitigation."

### W9. GSM8k evaluation asymmetry (N=200 vs N=1319)
**Issue:** Post-DPO GSM8k is evaluated on N=200 while the baseline uses N=1319. The 95% CI for N=200 spans [32.0%, 45.4%] — a 13.4 pp width — making the comparison with the 33.2% baseline essentially uninformative. This is disclosed in a footnote but the main table presents 33.2% → 38.5% without annotation.
**Evidence:** §5.11 table and footnote; `dpo_eval_results.json`.
**Severity:** MINOR — The core claim (capabilities preserved) is supported by MMLU (+0.8 pp at N=500). GSM8k is secondary. But the table presentation is misleading to a skimming reader.
**Recommendation:** Add "(N=200)" directly in the table cell for post-DPO GSM8k, not just in a footnote. Consider re-running GSM8k evaluation at N=1319 for the DPO model — it requires minimal additional compute.

### W10. Manifest does not cover Mistral artifacts
**Issue:** `full_rerun_manifest.json` tracks 17 Llama-3 + DPO artifacts but excludes all Mistral-specific artifacts. The paper's claim that "all findings replicate across architectures" rests partly on Mistral results not validated by the manifest.
**Evidence:** Research brief §1.11; manifest contains no entries from `results/mistral/`.
**Severity:** MINOR — The Mistral baseline metrics were independently spot-checked and verified. But the selective manifest coverage weakens the reproducibility claim.
**Recommendation:** Extend the manifest to cover Mistral artifacts, or add a separate `results/mistral/manifest.json`.

---

## 4. Detailed Comments

### Section 5.1 (Baseline)
- The 0% GSM8k sycophancy result is well-explained and the domain-verifiability hypothesis is compelling. The base model comparison (21.8% GSM8k sycophancy in base → 0% in instruct) provides strong supporting evidence.
- Cohen's h values > 2.0 are acknowledged as floor-effect-dominated — good practice.
- **Fix required:** Cohen's d label swap (W1).

### Section 5.3–5.5 (Probes)
- The neutral-transfer probe design is the paper's strongest methodological contribution. The progression from mixed-format (artifact: >99% "belief corruption") to neutral-transfer (18% social compliance dominant) clearly demonstrates the confound.
- The balanced-dataset replication (randomized answer positions) is a welcome robustness check.
- **Observation:** Transfer accuracy shows a bimodal profile across layers (peaks at L1 and L8-9) that is not discussed. The Layer 8–9 recovery (75–76% transfer accuracy) suggests a second window of truth-tracking activity that may have distinct computational significance.
- Fisher's exact test for mutually exclusive categories is technically suboptimal (see W5 in research brief). A chi-square goodness-of-fit on the 4-category distribution or a binomial test among sycophantic-only samples would be more appropriate. The conclusion is robust regardless — the effect size (1.8:1) is large enough to survive any reasonable test.

### Section 5.4 (Patching)
- The disclosure of head ranking instability (different top-3 across runs) is commendable and unusual.
- The 34/100 success rate deserves more analysis (W7). What is the domain composition of the 34 samples? Given that opinion sycophancy is 82.4%, factual 1.6%, and reasoning 0.0%, the 34 high-effect samples are almost certainly dominated by opinion questions. This should be stated.

### Section 5.6–5.7 (Ablation)
- The corrected ablation (§5.6.1) directly addressing the head-selection concern is excellent experimental practice and substantially strengthens the null result.
- The mean-ablation catastrophe (all outputs unparseable) is an interesting finding that deserves brief discussion — the observation that "the network is more robust to missing information than to misleading information" has implications for other ablation studies.

### Section 5.8 (Steering)
- The per-source analysis revealing a masked opinion-domain signal (−5.7 to −6.9 pp at L15/L20, α=2.0) is a valuable finding that would be invisible without domain decomposition. Good analytical depth.
- The BH-corrected finding (no condition survives at aggregate level, but opinion-domain L15/L20 fall outside Wilson CIs) is correctly presented — neither overclaiming nor dismissing.

### Section 5.9 (Fictional Entities)
- The sign reversal of L1H20 across circuits is compelling evidence for domain-specificity. This is well-presented.
- **Question:** Was the fictional-entity ablation performed? If L1H10 and L0H2 (the fictional-circuit top heads) are ablated, does fictional-entity sycophancy also show the same null as opinion-circuit ablation? This would strengthen the redundancy claim.

### Section 5.10 (Mistral Replication)
- Mistral's 99.8% factual sycophancy vs. Llama-3's 1.6% provides a striking contrast that makes the cross-architecture replication more informative than if the two models had similar profiles.
- The Mistral GSM8k baseline of 9.3% is low enough that capability preservation claims for Mistral are less meaningful (floor effect on the capability metric).

### Section 5.11 (DPO)
- The probe re-analysis design is clean — same methodology, same layer, pre vs. post comparison.
- The "Other" category decrease (12.1% → 4.8%, −7.3 pp) is noted in a footnote but deserves more attention. This suggests DPO improves internal coherence beyond the sycophancy-specific pathway.
- **Fix required:** Train loss discrepancy (W2).
- **Fix required:** Training pair count clarification (W3).
- **Fix required:** Best-layer disclosure (W4).

### Section 6 (Discussion)
- The comparison with Chen et al. (2024) is nuanced and fair — identifying four specific factors that may explain the divergence (module selection granularity, fine-tuning vs. ablation, challenge-induced vs. assertion-based sycophancy, Llama-2 vs. Llama-3 RLHF differences).
- The reconciliation with O'Brien et al. (2026) via granularity (head-level vs. feature-level redundancy) is a reasonable hypothesis but should be clearly flagged as speculative.
- The self-critical limitation section is thorough. Limitation #6 (missing path patching) correctly identifies the most important absent experiment.

### Section 9 (Conclusion)
- Contribution #5 ("Fine-tuning with 400 DPO preference pairs") should say 360 training pairs.
- The concluding sentence ("the model always knew the truth; DPO teaches it to say it") is memorable and well-earned by the evidence, but should be qualified with the in-distribution caveat.
- Yang et al. citation year: the conclusion uses "2025" while the abstract uses "2024" for the same paper. Pick one consistently (recommend 2025 for EMNLP publication year).

---

## 5. Questions for Authors

1. **What is the domain composition of the 34 high-total-effect patching samples?** If they are predominantly opinion-domain, the identified circuit is specifically an *opinion* sycophancy circuit, not a general sycophancy circuit. This has implications for how the patching results relate to the domain-specificity finding in §5.9.

2. **Can you provide per-epoch eval loss for DPO training?** The claim of "no overfitting" based on a single final eval loss (0.42) is insufficient — convergence to a good eval loss does not preclude overfitting in later steps followed by recovery. Epoch-level granularity would resolve this.

3. **What explains the discrepancy between "train loss 0.69 → 0.16" and the stored `train_loss: 0.356`?** Is 0.16 a single-step minimum? A different loss component? Or is 0.356 an aggregate over all three epochs including the high initial loss?

4. **Was the fictional-entity circuit also tested with ablation?** If ablating L1H10 and L0H2 also produces a null, this would establish redundancy as a universal property across domain circuits. If ablation *does* reduce fictional-entity sycophancy, this would suggest domain-specific circuits have different redundancy properties — a finding worth reporting either way.

5. **Why was DPO GSM8k evaluated on N=200 instead of N=1319?** Given that ablation experiments use N=1319 and the DPO evaluation already runs the model on 1500 sycophancy samples, the marginal cost of full GSM8k evaluation appears small.

6. **At Layer 1, how does the social compliance rate break down by domain?** Is the 18.0% concentrated in opinion questions (where sycophancy is highest) or distributed across domains? If concentrated, the probe-identified "social compliance" may be measuring the same phenomenon as the behavioral sycophancy rate.

7. **Have you examined the 58.6% post-DPO opinion sycophancy for patterns?** Are the remaining sycophantic responses concentrated on specific question types, topics, or prompt structures? This could inform whether the remaining sycophancy reflects DPO training coverage or a fundamentally harder subpopulation.

---

## 6. Missing Experiments

### Critical (would substantially change conclusions)

1. **Edge-level path patching on Llama-3-8B-Instruct.** The paper's central negative result — the patching-to-ablation dissociation — could reflect either genuine circuit redundancy or insufficient patching granularity. Path patching through specific attention head Q/K/V pathways would disambiguate these interpretations and address the divergence with Chen et al. (2024)'s successful targeted intervention. This is acknowledged in Limitation #6 and is the single most important missing experiment.

2. **Out-of-distribution DPO evaluation.** At minimum, evaluate the DPO model on: (a) opinion sycophancy prompts from Sharma et al. (2024) using different prompt templates, (b) a small set of manually constructed multi-turn sycophancy scenarios, (c) a free-form generation evaluation on a subset of opinion questions. Without OOD evaluation, the practical significance of the 23.8 pp reduction is unclear.

### Important (would strengthen claims)

3. **DPO replication on Mistral-7B-Instruct.** The paper's most novel contribution (Contribution #5: mechanistic DPO decomposition) is demonstrated on only one model. Given Mistral's inverted sycophancy profile (99.8% factual), DPO targeting factual-domain sycophancy on Mistral would test whether the social-compliance-to-robust-tracking mechanism generalizes across architectures and domains. Acknowledged in Limitation #6.

4. **Full GSM8k evaluation for DPO model (N=1319).** This is cheap and would resolve the uninformative N=200 comparison.

5. **Ablation of fictional-entity circuit heads.** Testing whether the fictional-entity circuit also shows the redundancy null would generalize the dissociation finding across domain-specific circuits.

### Desirable (would add depth)

6. **Characterization of the 34 high-effect patching samples.** Domain breakdown, question features, and comparison with the 66 low-effect samples.

7. **Post-DPO patching.** Run activation patching on the DPO model. If the same heads show reduced recovery scores, this would connect the behavioral and circuit-level changes.

8. **Scaling law for DPO training pairs.** Training with 100, 200, 400, 800 pairs and measuring opinion sycophancy reduction would indicate whether the 58.6% plateau reflects a training scale limitation or a harder subpopulation.

---

## 7. Overall Assessment

**Recommendation: Revise and Resubmit (Major Revision)**

### Justification

This is a well-executed mechanistic interpretability study with several genuine contributions — the neutral-transfer probe methodology, the empirically demonstrated patching-to-ablation dissociation, and the DPO probe re-analysis are all valuable additions to the alignment literature. The cross-architecture replication and transparent reporting of negative results exceed community norms. The paper tackles an important problem and makes real progress.

However, two categories of issues prevent acceptance in the current form:

**Presentation errors that must be corrected:**
- The Cohen's d label reversal (W1) and DPO train loss discrepancy (W2) are factual errors in the paper text that contradict stored artifacts. While neither invalidates conclusions, they signal insufficient proofreading of quantitative claims against source data. The train loss issue (W2) is particularly concerning because the claimed convergence (0.69 → 0.16) implies much tighter optimization than the stored 0.356 supports, potentially misleading readers about DPO training dynamics.

**Methodological gaps that weaken central claims:**
- The absence of OOD evaluation for DPO (W6) is a material limitation for a paper contributing to alignment. The 23.8 pp reduction is an in-distribution result on a single prompt template; without OOD testing, the practical significance is unknown. The mechanistic evidence (probe decomposition) partially compensates, but does not fully substitute for behavioral generalization evidence.
- The power calculation error (W5) is minor in isolation but contributes to a pattern of imprecise statistical reporting (Fisher's exact test on mutually exclusive categories, asymmetric N for GSM8k).

**What would be needed for acceptance:**
1. Correct all identified factual discrepancies (W1, W2, W3, W4).
2. Add at least one OOD evaluation for DPO (W6).
3. Run GSM8k at N=1319 for the DPO model (W9).
4. Recalculate the power analysis (W5).
5. Add the layer-4 post-DPO disclosure (W4).
6. Extend manifest to Mistral artifacts (W10).

The core findings — social compliance dominance, patching-to-ablation dissociation, domain-specific circuits, cross-architecture replication — are well-supported and would make a solid contribution after these revisions. The DPO mechanistic decomposition is novel and the claim is defensible, but needs OOD validation to reach its full impact.

### Confidence: 4/5
I have carefully reviewed the paper, verified key claims against stored JSON artifacts, checked all cited concurrent works, and performed independent statistical calculations. The two areas where my confidence is lower are: (a) the steering results file was only partially spot-checked due to size, and (b) the Mistral probe results were not independently verified against their JSON artifact.

---

## 8. Sources

All sources inspected during this review:

### JSON Artifacts (Verified)
- `results/baseline_llama3_summary.json` — Baseline sycophancy rates, effect sizes
- `results/probe_control_balanced_results.json` — Balanced neutral-transfer probe decomposition
- `results/head_importance.json` — Head-level patching recovery scores
- `results/top10_ablation_full_gsm8k.json` — Top-10 ablation with full GSM8k
- `results/corrected_ablation_results.json` — Corrected top-3 ablation
- `results/dpo_eval_results.json` — DPO behavioral + probe re-analysis
- `results/dpo_training_metrics.json` — DPO training hyperparameters and loss
- `results/mistral/baseline_summary.json` — Mistral baseline metrics
- `results/mistral/top10_ablation_full_gsm8k.json` — Mistral ablation
- `results/steering_results.json` — Steering sweep (partial verification)
- `results/full_rerun_manifest.json` — Artifact manifest

### Cited Works (Verified)
- Li et al. (2025), "When Truth Is Overridden" — arXiv:2508.02087
- Chen et al. (2024), "From Yes-Men to Truth-Tellers" — ICML 2024, proceedings.mlr.press/v235/chen24u.html
- O'Brien et al. (2026), "A Few Bad Neurons" — arXiv:2601.18939, NeurIPS 2025 Workshop
- Lee et al. (2024), "A Mechanistic Understanding of Alignment Algorithms" — ICML 2024, arXiv:2401.01967
- Yang et al. (2025), "How Does DPO Reduce Toxicity?" — EMNLP 2025, arXiv:2411.06424
- Sharma et al. (2024), "Towards Understanding Sycophancy" — ICLR 2024, arXiv:2310.13548
- Heimersheim & Nanda (2024), "How to use and interpret activation patching" — arXiv:2404.15255
- Wei et al. (2023), "Simple synthetic data reduces sycophancy" — arXiv:2308.03958
