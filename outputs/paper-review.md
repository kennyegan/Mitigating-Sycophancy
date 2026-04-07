# Peer Review: Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation

**Venue:** NeurIPS 2026  
**Reviewer:** Feynman (simulated)  
**Date:** 2026-04-07  
**Recommendation:** Borderline Accept → Accept with revisions  
**Confidence:** 4/5

---

## Overall Assessment

This paper applies mechanistic interpretability to sycophancy — a safety-relevant alignment failure — in Llama-3-8B-Instruct and Mistral-7B-Instruct. It uses linear probes, causal activation patching, head ablation, and representation steering to investigate the internal mechanism. The paper's main contribution is a set of well-controlled negative results: patching-identified heads are not causally necessary, inference-time interventions fail, and sycophancy is redundantly distributed. The cross-architecture replication is a genuine strength. The work is clearly written, carefully executed, and addresses a real problem.

However, the paper has structural and methodological issues that need addressing before it is NeurIPS-ready. The most important are: (1) a significant internal inconsistency in the Discussion section, (2) the forced-choice evaluation format limits the ecological validity of the findings, (3) the DPO intervention — the proposed solution to the negative results — is not yet executed, leaving the paper without a positive mitigation result, and (4) the related work should engage more with the self-repair and circuit faithfulness literatures that directly explain the core finding.

**Strengths outweigh weaknesses.** The patching-to-ablation dissociation on a safety-relevant behavior, the domain-specific circuits finding, and the cross-architecture replication are all novel and publishable. But the paper needs tightening.

---

## Summary of Contributions

1. **Social compliance > belief corruption** (1.8:1) via format-controlled neutral-transfer probes
2. **Patching-to-ablation dissociation**: top-3 and top-10 head ablation produces no sycophancy reduction despite high patching recovery scores
3. **Domain-specific circuits**: opinion and fictional-entity sycophancy use entirely different heads with zero overlap and sign-reversed roles
4. **Cross-architecture replication**: all findings hold on Mistral-7B despite inverted sycophancy profiles and different circuits
5. **Methodological contribution**: probe control experiment showing format-mixed probes learn superficial cues

---

## Strengths

### S1. The right question on the right behavior
Sycophancy is genuinely safety-relevant, and the mechanistic interpretability community has largely studied narrow toy tasks (IOI, greater-than, factual recall). Applying circuit-discovery tools to a complex, alignment-relevant behavior and rigorously documenting where they fail is exactly the kind of work the field needs.

### S2. Cross-architecture replication is unusually rigorous
Replicating all core findings on Mistral-7B — with entirely different circuits, inverted sycophancy profiles, and different RLHF training — transforms single-model observations into cross-architecture claims. This is rare in mech interp papers and substantially strengthens the conclusions. The fact that Mistral shows 99.8% factual sycophancy vs. Llama-3's 1.6% provides a natural "inverted control."

### S3. The probe control experiment is a genuine methodological contribution
The demonstration that probes trained on format-mixed data achieve near-perfect accuracy by learning prompt-format cues — and that the neutral-transfer design eliminates this confound — is broadly useful beyond this paper. The claim that social compliance dominates is well-supported by the balanced design with randomized answer positions.

### S4. Domain-specific circuits with sign-reversed head roles
The fictional-entity control showing zero circuit overlap and L1H20 sign reversal is a compelling result. It rules out a universal "social agreement" circuit and shows sycophancy is a family of domain-dependent behaviors. This is novel.

### S5. Honest reporting of negative results
The paper does not oversell the modest opinion-domain steering effect (−5.7pp at L20, α=2) and clearly documents the capability costs. The overall narrative — "we looked hard and the intervention doesn't work" — is refreshingly honest.

---

## Weaknesses

### W1. [Major] Discussion contains stale numbers from pre-correction run

**Section 6 Discussion states:** "The top 3 heads (L1H20, L5H5, L4H28) show the highest activation patching recovery scores (0.51–0.57)."

**Problem:** The recovery scores 0.51–0.57 do not appear anywhere in the Results section. The validated patching run (§5.4) reports L4H28 (0.443), L4H5 (0.302), L5H31 (0.256). The heads L1H20 and L5H5 have validated scores of 0.040 and −0.237, respectively.

The paper explicitly acknowledges in §5.4 and §5.6.1 that the original patching run was superseded by a validated rerun. The Discussion was not updated to reflect this. A reviewer or reader comparing the Discussion to the Results will immediately notice the discrepancy. **This must be fixed.**

Additionally, §5.7 ablates the pre-correction top-10 heads without clearly labeling them as such. The validated top-10 (§5.4) shares only 4 of 10 heads with the ablated set. While §5.6.1 demonstrates the dissociation holds for validated heads too, the top-10 ablation (§5.7) technically targets the wrong heads. Add explicit labeling.

### W2. [Major] Forced-choice (A)/(B) evaluation limits ecological validity

The entire paper uses binary forced-choice evaluation. This is a reasonable experimental design for mechanistic analysis (it produces clean logit-based measurements), but it substantially limits the generalizability of the findings to real-world sycophancy, which occurs in free-form generation.

Specific concerns:
- The 0.0% GSM8k sycophancy may be an artifact of the model having high confidence in the correct arithmetic answer, making the forced choice trivial. In free-form generation, the model might still include sycophantic framing ("That's a great approach! Let me help refine it...") around the correct answer.
- The 82.4% opinion sycophancy rate depends entirely on which answer is coded as "sycophantic" in opinion questions that have no objective answer. The paper uses lexicographic ordering of answer options as labels, which is appropriate for probe training but makes the raw sycophancy rate hard to interpret substantively.
- The compliance gap metric (difference in P(sycophantic answer) between biased and neutral prompts) is more meaningful than the raw rate, but it is less prominently featured.

**Recommendation:** Add a paragraph to Limitations explicitly discussing this. Consider adding a small free-form generation evaluation (even qualitative) to demonstrate the behavior transfers beyond forced-choice.

### W3. [Major] The paper lacks a positive mitigation result

The paper's narrative arc is: "We investigated → we found it's redundantly distributed → inference-time interventions don't work → training-time intervention is needed." But the DPO intervention (Milestone 8 per PROJECT_OVERVIEW.md) is not yet executed. This leaves the paper as purely a negative-result + characterization paper.

NeurIPS reviewers will likely ask: "So what do we do about it?" The paper gestures toward DPO but doesn't deliver. This is the single biggest weakness for acceptance.

**Options:**
- Complete the DPO experiment before submission (highest impact)
- Reframe the paper explicitly as a methodological contribution about the limits of circuit discovery, rather than about sycophancy mitigation
- Add a concrete, runnable DPO experimental design to an appendix as a "proposed experiment"

### W4. [Moderate] Related work misses the self-repair and circuit faithfulness literatures

The paper draws a compelling fMRI/lesion analogy but does not cite the two primary papers that establish the mechanism for the observed dissociation:

1. **McGrath et al. (2023), "The Hydra Effect"** — Directly demonstrates that downstream attention layers compensate for ablated upstream layers, explaining ~70% of the sufficiency-necessity gap. Your finding is a direct instance of this in a new domain.
2. **Rushing & Nanda (2024), "Explorations of Self-Repair"** — Extends the Hydra effect to individual head ablations on the full pretraining distribution, identifying LayerNorm rescaling (~30% of self-repair) and sparse MLP anti-erasure as concrete mechanisms.
3. **Miller et al. (2024), "Transformer Circuit Faithfulness Metrics Are Not Robust"** — Demonstrates that the optimal circuit depends on the ablation method, which is directly relevant to your methodology.
4. **Joshi et al. (2026), "Causality is Key for Interpretability Claims to Generalise"** — Provides the formal causal framework (Pearl's ladder) for your sufficiency ≠ necessity argument, and shows 53.5% of MI claims have a "rung mismatch."

Without these citations, the paper risks appearing less well-situated in the literature than it actually is. The patching-to-ablation dissociation is not your discovery — the self-repair literature predicts it. What *is* novel is demonstrating it on a safety-relevant behavior with cross-architecture replication and domain-specific circuits. The framing should reflect this.

### W5. [Moderate] Statistical testing is underspecified

The paper reports bootstrap CIs and Cohen's h but does not perform formal hypothesis tests. For the key claims:
- "Top-10 ablation produces no meaningful sycophancy reduction" (+0.5pp): What is the CI on this difference? Is it a pre-registered null hypothesis test, or a post-hoc assessment?
- The probe cross-tab (18.0% SC vs. 10.1% BC): Is the difference statistically significant? A χ² or Fisher's exact test would be appropriate.
- The opinion-domain steering reduction (−5.7pp to −6.9pp): CIs are not reported for these per-source numbers.

For a NeurIPS paper, formal tests (or Bayesian equivalents) should accompany the key claims, especially the nulls.

### W6. [Minor] The 2×2 probe table is mentioned as "still needed" but not present

The paper references a 2×2 probe table (social compliance vs. belief corruption × probe correct vs. incorrect) but the current draft presents the results in text and a simple table. A clear 2×2 contingency table would substantially aid reader comprehension.

### W7. [Minor] Multiple testing corrections not applied

The paper tests multiple heads, multiple layers, multiple alpha values, and multiple domains without adjusting for multiplicity. While the key findings are nulls (where multiple testing actually *strengthens* the conclusion), the modest positive results (e.g., −5.7pp opinion steering) should be evaluated against a Bonferroni or FDR-corrected threshold given the full sweep of 56+ conditions tested.

### W8. [Minor] Section 5.9 L1H20 "Rank 4" label is inaccurate

In the §5.9 comparison table, L1H20 is labeled as rank 4 in the opinion circuit with recovery 0.040. But the validated opinion circuit top-10 (§5.4) shows L2H5 at rank 4 with recovery 0.2445. L1H20 is not in the validated top-10 at all. The table includes L1H20 to demonstrate the sign reversal, which is fine — but the rank labeling is misleading. Relabel or add a footnote.

---

## Questions for the Authors

**Q1.** The corrected ablation (§5.6.1) targets L4H28, L4H5, L5H31 and produces −0.3pp. The original ablation (§5.6) targets L1H20, L5H5, L4H28 and produces +0.1pp. But §5.7 ablates the *original* pre-correction top-10. Have you run the top-10 ablation with the *validated* top-10 heads? If not, this is the most important missing experiment.

**Q2.** The probe control uses logistic regression. Have you tried nonlinear probes (e.g., 2-layer MLP)? If linear probes show social compliance at 1.8:1, but nonlinear probes show belief corruption, this would change the interpretation substantially.

**Q3.** You note that later-layer steering (L15, L20) achieves a modest opinion-domain reduction (−5.7pp). Does this partial success suggest that the sycophancy computation is *partially* localizable at the residual-stream level (just not at the head level)? If so, could a higher-dimensional steering approach (e.g., multi-direction, concept-level) be more effective than the single mean-difference vector you used?

**Q4.** The fictional-entity sycophancy rate (93.0%) is based on N=100. This is small for a control experiment supporting a core claim (domain-specific circuits). What is the CI? Have you tested sensitivity to the specific fictional entities chosen?

**Q5.** Mean ablation of top-3 heads causes "catastrophic output degradation" (§5.6, §5.6.1). This is mentioned but not analyzed. Could the catastrophic failure itself be informative about the circuit structure? Mean ablation shifts the residual stream in a specific direction — does the catastrophic response correlate with the steering vector?

---

## Minor Issues

1. **§5.1:** Cohen's h values are off by ~0.01 (Opinion vs. Reasoning: reported 2.276, calculated 2.285). Trivial but fix for final version.
2. **§5.6.1 vs. §5.6:** MMLU baseline is 62.2% vs. 62.0%. Add a footnote explaining the variation across independent evaluation runs.
3. **§5.3 and §5.5** present the same balanced probe results from different perspectives. Add a sentence in one section cross-referencing the other to avoid reader confusion.
4. **§5.8:** The per-source steering baseline (opinion 83.0%) differs from the main baseline (82.4%) due to the held-out split. The overall rate difference (28.4% vs. 28.0%) is explained, but the per-source difference is not. Add one sentence.
5. **Abstract:** "format-controlled probes reveal that sycophancy is primarily social compliance" — the abstract doesn't specify the ratio (1.8:1). Consider adding it; it's your headline number.
6. **Figures:** 5 figures exist (patching heatmap, steering sweep, per-source steering, probe accuracy, ablation comparison). The paper references them but they are not embedded in the markdown. Ensure they are properly embedded and captioned in the submission version.
7. **Missing: Table of all notation.** The paper uses compliance gap, recovery score, sycophancy rate, and several probe-specific terms without a unified notation table. Add one.

---

## Detailed Scoring

| Criterion | Score (1–10) | Notes |
|-----------|:---:|-------|
| **Novelty** | 7 | Patching-to-ablation dissociation is known from self-repair literature; novel application to sycophancy + domain-specificity + cross-arch replication |
| **Significance** | 7 | Important negative result for circuit-intervention safety approaches; would be 8–9 with DPO results |
| **Soundness** | 6 | Generally strong experimental design; undermined by stale Discussion numbers, missing statistical tests, and pre-correction head list in §5.7 |
| **Clarity** | 7 | Well-written and honest; some structural confusion between §5.3/§5.5; the head-identity inconsistencies will confuse careful readers |
| **Reproducibility** | 8 | Full code, SLURM scripts, seed=42, manifest validation. Strong. |
| **Completeness** | 5 | DPO experiment not run; the paper promises a mitigation but doesn't deliver one |

**Overall: 6.5/10** — Above the acceptance threshold if the inconsistencies are fixed and the framing is tightened. With DPO results: 7.5–8.

---

## Verdict and Path to Acceptance

### Must-fix before submission:
1. **Fix the Discussion recovery scores** (0.51–0.57 → validated values)
2. **Label §5.7 as using pre-correction heads** and either re-run with validated top-10 or explain why the pre-correction run is sufficient
3. **Fix §5.9 L1H20 rank label**
4. **Add formal statistical tests** for the key null results and the SC/BC difference
5. **Cite McGrath 2023, Rushing & Nanda 2024, Joshi et al. 2026** — position the patching-to-ablation dissociation as an instance of known self-repair, with your novelty being the domain, scale of null, and cross-architecture replication

### Strongly recommended:
6. **Complete the DPO experiment** — this transforms the paper from "negative result + characterization" to "diagnosis + treatment," which is much more compelling for NeurIPS
7. **Add multiple testing corrections** for the steering sweep
8. **Expand Limitations** to discuss forced-choice ecological validity

### Nice to have:
9. Unified notation table
10. 2×2 probe contingency table
11. Appendix with qualitative free-form generation examples

---

## Sources Referenced in This Review

| Paper | Year | URL |
|-------|------|-----|
| McGrath et al., *The Hydra Effect* | 2023 | [arXiv:2307.15771](https://arxiv.org/abs/2307.15771) |
| Rushing & Nanda, *Explorations of Self-Repair* | 2024 | [arXiv:2402.15390](https://arxiv.org/abs/2402.15390) |
| Miller et al., *Faithfulness Metrics Not Robust* | 2024 | [arXiv:2407.08734](https://arxiv.org/abs/2407.08734) |
| Joshi et al., *Causality is Key* | 2026 | [arXiv:2602.16698](https://arxiv.org/abs/2602.16698) |
| Heimersheim & Nanda, *How to Use Activation Patching* | 2024 | [arXiv:2404.15255](https://arxiv.org/abs/2404.15255) |
| uit de Bos & Garriga-Alonso, *Adversarial Circuit Evaluation* | 2024 | [arXiv:2407.15166](https://arxiv.org/abs/2407.15166) |
| Hanna et al., *Have Faith in Faithfulness* | 2024 | [arXiv:2403.17806](https://arxiv.org/abs/2403.17806) |
| Nainani et al., *Adaptive Circuit Behavior* | 2024 | [arXiv:2411.16105](https://arxiv.org/abs/2411.16105) |
