# Peer Review: Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation

## Summary

This paper presents a mechanistic interpretability study of sycophancy in Llama-3-8B-Instruct and Mistral-7B-Instruct. The authors apply a suite of techniques — linear probes with a neutral-transfer design, causal activation patching, head ablation, representation steering, and DPO fine-tuning — to characterize, localize, and mitigate sycophantic behavior. The central finding is that sycophancy is primarily "social compliance" (the model retains correct internal representations but suppresses them in output) rather than "belief corruption" (the model's internal representations are shifted). A secondary finding is a "patching-to-ablation dissociation": attention heads identified as sufficient carriers of the sycophantic signal via activation patching are not causally necessary, as ablating even the top 10 produces no sycophancy reduction. All findings replicate across architectures (Llama-3, Mistral-7B).

The paper's most novel contribution is a mechanistic decomposition of DPO's effect on sycophancy. DPO fine-tuning with 400 preference pairs reduces opinion sycophancy by 23.8 percentage points (82.4% → 58.6%) while preserving capabilities. Probe re-analysis of the DPO-adapted model reveals that DPO converts social compliance into robust truth-tracking (+15.6 pp) without altering internal truth representations, providing the first evidence of how preference optimization resolves sycophantic output-gating specifically. The paper also identifies domain-specific circuits (zero overlap between opinion and fictional-entity sycophancy heads, with sign-reversed roles for shared components) and replicates across two architectures.

## Overall Assessment

**Recommendation:** Weak Accept  
**Confidence:** High

This is a thorough and methodologically careful study that makes genuine contributions to our understanding of sycophancy mechanisms. The DPO mechanistic decomposition (Contribution #5) and domain-specific circuit evidence (Contribution #3) are novel and well-supported. The patching-to-ablation dissociation (Contribution #2) is an important empirical validation of theoretical distinctions formalized by Heimersheim & Nanda (2024). However, two critical issues temper enthusiasm: (1) two anchor citations (Lee et al. 2024, Yang et al. 2025) for the DPO novelty claim cannot be verified as existing publications, undermining the paper's novelty positioning; and (2) the social compliance finding (Contribution #1) is substantially preempted by Li et al. (2025, arXiv:2508.02087), who find a functionally equivalent result on a near-identical model. The missing edge-level path patching experiment — which would directly address the divergence with Chen et al. (2025) — is a significant gap. If the citation issues are resolved and the novelty framing for Contribution #1 is softened, this paper merits acceptance.

## Strengths

1. **Exceptional methodological rigor.** The authors demonstrate unusual scientific diligence through multiple layers of self-correction: the balanced probe replication (Job 10) fixing degenerate class balance, the corrected ablation using validated heads (§5.6.1), the full GSM8k rerun expanding N from 200 to 1,319, and the careful documentation of discrepancies between initial and validated patching runs. Few mechanistic interpretability papers are this transparent about their iterative refinement process.

2. **DPO mechanistic decomposition is genuinely novel and insightful (Contribution #5).** The probe re-analysis pre/post DPO — showing social compliance drops (18.0% → 11.4%), robust tracking rises (59.9% → 75.5%), and belief corruption barely moves (−1.8 pp) — provides a clean mechanistic account of what DPO does to sycophancy. This goes beyond behavioral evaluation to show *why* DPO works: it reconnects internal truth representations to output behavior. No verified concurrent work achieves this decomposition.

3. **Domain-specific circuits with sign reversal (Contribution #3).** The zero overlap between opinion and fictional-entity circuits (L4H28/L4H5/L5H31 vs. L1H10/L0H2/L0H0) and the sign reversal of L1H20 (+0.040 opinion, −0.115 fictional) is a strong empirical result that rules out a universal sycophancy mechanism. This has direct implications for mitigation strategies.

4. **Cross-architecture replication strengthens all claims (Contribution #4).** Replicating the full pipeline on Mistral-7B — with its inverted sycophancy profile (near-total factual sycophancy vs. Llama-3's near-zero) — and finding the same structural properties (social compliance dominance at 6.4:1, null ablation, null steering) despite entirely different circuits elevates these from single-model observations to general claims about RLHF-trained models.

5. **The patching-to-ablation dissociation is empirically robust (Contribution #2).** Confirmed with both initial and validated top-3 heads, extended to top-10, and replicated on Mistral. The fMRI/lesion analogy is apt and effectively communicates the methodological implication. The Heimersheim & Nanda (2024) framing (sufficiency vs. necessity) is correctly applied.

6. **Capability preservation under DPO is noteworthy.** MMLU +0.8 pp and GSM8k +5.3 pp post-DPO, with only 400 training pairs and 3 minutes of training, is a practically relevant result. The slight GSM8k improvement is unexpected and worth further investigation.

7. **Comprehensive artifact documentation.** The full rerun manifest (`missing_count: 0`), explicit git hashes, SLURM job matrix, and detailed output file table make this one of the more reproducible mechanistic interpretability papers.

## Weaknesses

### FATAL

**W1 (FATAL): Two anchor citations cannot be verified.** Lee et al. (2024) ("Mechanistic Understanding of DPO Toxicity Reduction," ICML 2024) and Yang et al. (2025) ("Mechanistic Analysis of DPO for Toxicity," EMNLP 2025) are unverifiable on arXiv, alphaXiv, and proceedings search. These are the two papers the authors contrast with in Contribution #5 — the claim of extending "prior mechanistic DPO analyses on toxicity" to sycophancy depends entirely on these references establishing a prior art baseline. If these papers do not exist or are inaccurately described, the novelty framing of the DPO result must be reformulated. **Fix:** Provide DOIs, arXiv IDs, or proceedings URLs for both citations. If they cannot be verified, remove the "extending prior work" framing and present the DPO mechanistic decomposition as novel without the toxicity contrast.

### MAJOR

**W2 (MAJOR): Contribution #1 is substantially preempted by Li et al. (2025).** Li et al. (arXiv:2508.02087) find that early-layer representations retain truth while late-layer processing overrides them in the sycophantic direction on Llama-3.1-8B-Instruct — conceptually equivalent to the social compliance vs. belief corruption distinction claimed as novel here. Both papers study opinion-induced sycophancy on similar model families. The paper frames these as "complementary" (§6, patching vs. logit-lens), which is defensible but insufficient: a reviewer will note that both abstracts use near-identical framing ("truth is overridden" / "model retains correct internal representations but outputs sycophantic responses"). **Fix:** Soften Contribution #1 from "novel" to "independently confirmed with a different methodology." Explicitly acknowledge Li et al.'s conceptual priority on the social compliance finding. Emphasize what is genuinely additive: the balanced neutral-transfer probe design, the quantitative decomposition (18.0% SC / 10.1% BC / 59.9% robust), and the DPO probe re-analysis as the unique extension.

**W3 (MAJOR): No edge-level (path) patching experiment.** Chen et al. (2025, arXiv:2409.01658) use path patching (edge-level) on Llama-2-Chat and *succeed* at ablation-based sycophancy reduction, while this paper uses node-level residual stream patching and finds a null. The paper speculates (§6) that path patching "may identify more causally necessary components" but never tests this hypothesis on its own models. This is the most direct experiment that could distinguish between two interpretations: (a) sycophancy in Llama-3 is genuinely more redundant than in Llama-2, or (b) node-level patching is simply too coarse to identify the relevant components. **Fix:** Run path patching on Llama-3-8B-Instruct for the opinion-domain subset. If path patching identifies causally necessary edges that ablation can target, the paper's narrative shifts from "sycophancy is fundamentally redundant" to "node-level analysis is insufficient." Either result would strengthen the paper.

**W4 (MAJOR): DPO result (Contribution #5) lacks cross-architecture replication.** Contributions #1–#4 are all replicated on Mistral, but the DPO training and probe re-analysis are Llama-3 only. Given the paper's emphasis on cross-architecture generalization, the most novel contribution is the least validated. The Mistral model has a strikingly different sycophancy profile (near-total factual sycophancy), making it an especially interesting DPO target. **Fix:** Run DPO on Mistral-7B-Instruct with the same methodology and perform probe re-analysis. This would cost approximately 2 additional GPU-hours based on the Llama-3 DPO pipeline and would substantially strengthen the paper's strongest contribution.

**W5 (MAJOR): Top-10 ablation (§5.7) uses pre-validated heads.** The top-10 ablation targets heads from the initial patching run, including L5H5 (recovery score −0.237) and L1H20 (recovery score 0.040) — heads the paper itself identifies as having near-zero or negative recovery in the validated run (§5.6.1). The corrected ablation only addresses the top-3. A corrected top-10 ablation using the validated head list (L4H28, L4H5, L5H31, L2H5, L3H30, L5H24, L3H17, L3H28, L1H11, L4H26 from §5.4) would properly close the redundancy argument. **Fix:** Run top-10 ablation with the validated head list. Given the corrected top-3 null and the original top-10 null, the result is very likely to be null as well, but this closes a methodological gap that a reviewer could exploit.

### MINOR

**W6 (MINOR): Chen et al. citation year is incorrect.** The paper cites Chen et al. as 2025 (bib entry `year={2025}`), but the paper was published on arXiv on September 3, 2024 (arXiv:2409.01658). **Fix:** Correct to `year={2024}`.

**W7 (MINOR): No DPO hyperparameter sensitivity analysis.** The DPO result uses a single configuration (beta=0.1, LoRA rank 16, alpha 32, 400 pairs, 3 epochs). Without varying beta (0.01–0.5), LoRA rank (4, 8, 32), or dataset size (100–800), it is unclear whether the 23.8 pp reduction is robust to hyperparameter choices. The fast convergence (3 minutes, 95% reward accuracy) suggests possible overfitting; no validation loss curve is shown. **Fix:** At minimum, show the DPO training and validation loss curves side by side. A small sweep over beta and dataset size would substantially strengthen confidence in the result.

**W8 (MINOR): Statistical power analysis missing for null ablation.** The paper reports post-hoc z-tests (z=0.28, p=0.78 for the +0.5 pp top-10 change) but no a priori power analysis. With N=1,500, what is the minimum detectable effect size at 80% power? This is important for distinguishing "no effect" from "effect too small to detect." **Fix:** Compute the minimum detectable effect (MDE) for the two-proportion z-test at N=1,500 with 80% power and α=0.05. Report this alongside the null results.

**W9 (MINOR): Probe decomposition sum is 100.1% (§5.5).** The pre-DPO Layer 1 decomposition sums to 100.1% (18.0 + 10.1 + 59.9 + 12.1). This is a display rounding artifact but should be acknowledged. **Fix:** Add a footnote or adjust one value by 0.1 pp.

**W10 (MINOR): Two additional cited references are unverifiable.** Venhoff et al. (2025) and Paduraru et al. (2025) cannot be found on arXiv. Paduraru et al. has no arXiv ID in the bib entry. These appear in §2 and §6 as evidence for SAE-based steering approaches. **Fix:** Provide URLs/proceedings links or remove if unverifiable.

**W11 (MINOR): 34/100 activation patching success rate is underdiscussed.** Only 34% of samples show total effect > 0.1, meaning Phase 1 layer importance is derived from 34 samples. No comparison to typical patching success rates in prior work (e.g., Wang et al. 2022) is provided, and no confidence intervals on layer importance scores are given. **Fix:** Discuss this rate in context of prior work. Provide bootstrap CIs on layer importance scores from the 34-sample subset.

## Detailed Comments

### Novelty Positioning

**Contribution #1 (Social compliance vs. belief corruption):** ⚠️ **Partially preempted.** Li et al. (2025, arXiv:2508.02087) find functionally equivalent results — early-layer truth retention with late-layer override — on Llama-3.1-8B-Instruct using logit-lens and activation patching. Both papers study opinion-induced sycophancy on the same model family. The paper's "complementary" framing (§6) is technically accurate but undersells the conceptual overlap. What remains genuinely additive is: (a) the quantitative four-way decomposition (SC/BC/robust/other), (b) the balanced neutral-transfer probe design as a methodological contribution, and (c) the DPO probe re-analysis extension.

**Contribution #2 (Patching-to-ablation dissociation):** ✅ **Novel.** Heimersheim & Nanda (2024) formalize the theory; this paper provides the first empirical demonstration on a complex, safety-relevant behavior. Chen et al.'s ablation *success* on Llama-2-Chat does not preempt this — it uses a different method (path patching), model, and sycophancy type (challenge-induced vs. assertion-based). The dissociation is confirmed with both initial and validated heads and replicated on Mistral.

**Contribution #3 (Domain-specific circuits):** ✅ **Novel.** No prior work performs cross-domain circuit comparison with sign-reversal evidence. The zero overlap in top-5 heads between opinion and fictional-entity circuits is a strong finding. The L1H20 sign reversal (+0.040 → −0.115) is particularly compelling.

**Contribution #4 (Cross-architecture replication):** ✅ **Novel in scope.** Li et al. (2025) study seven model families but do not run the full circuit discovery + ablation + steering pipeline on each. The Mistral replication adds the ablation null and steering null, which Li et al. do not test.

**Contribution #5 (DPO mechanistic decomposition):** ✅ **Novel, conditionally.** The probe re-analysis methodology is genuinely original in the sycophancy literature. However, the novelty positioning depends on two unverifiable citations (Lee et al. 2024, Yang et al. 2025). If these do not exist, the "extending prior toxicity work" framing collapses — though the DPO result itself remains novel regardless.

### Empirical Rigor

**Statistical methodology is generally sound.** The paper reports 95% CIs for sycophancy rates (Wilson intervals), uses Fisher's exact test for the SC/BC comparison, applies two-proportion z-tests for ablation nulls, computes Cohen's h for effect sizes (manually verified: all three values correct), and applies Benjamini-Hochberg FDR correction for the steering sweep. The Wilson CI calculation for the opinion steering baseline ([79.2%, 86.3%]) is verified correct.

**Gaps:**
- No a priori power analysis for the ablation null (W8). The MDE for N=1,500 at 80% power is approximately ±3.6 pp — the paper should state this explicitly to show the null is meaningful.
- The steering FDR correction is applied at the aggregate level but not within the per-source opinion-domain analysis. The L15/L20 results that "fall outside the Wilson CI" are identified post-hoc after the aggregate test fails. This is a form of subgroup analysis without correction. The paper should either apply FDR correction within the opinion domain or frame these results more cautiously.
- Phase 1 patching uses N=34 effective samples for layer ranking with no CIs on importance scores. This is a small sample for ranking 32 layers.

### Missing Experiments

In order of priority:

1. **Edge-level (path) patching on Llama-3-8B-Instruct** — directly tests whether the null ablation reflects genuine redundancy or insufficient patching granularity; addresses the key Chen et al. divergence. (HIGH)
2. **DPO on Mistral-7B-Instruct** with probe re-analysis — extends the most novel contribution to a second architecture. (HIGH)
3. **DPO hyperparameter sensitivity** — at minimum beta ∈ {0.01, 0.1, 0.5} and dataset size ∈ {100, 200, 400, 800}. (MEDIUM)
4. **Corrected top-10 ablation with validated heads** — closes the methodological gap in §5.7. (MEDIUM)
5. **Open-ended generation evaluation** — validates that sycophancy reduction transfers beyond forced-choice to naturalistic settings. (MEDIUM)
6. **Statistical power analysis** for the ablation null — state the MDE explicitly. (MEDIUM)
7. **DPO validation loss curve** — rules out overfitting with 400 pairs. (MEDIUM)
8. **Scaling to 70B+** — tests whether circuit redundancy holds at larger scales. (LOW, acknowledged in limitations)

### Internal Consistency

1. **Probe decomposition sum (§5.5):** 18.0 + 10.1 + 59.9 + 12.1 = 100.1%. Rounding artifact, not substantive, but should be noted. Post-DPO sum (75.5 + 11.4 + 8.3 + 4.8 = 100.0%) is correct.
2. **Cohen's h values (§5.1):** All three verified correct (h=2.276, 2.022, −0.254).
3. **Head list consistency:** The paper carefully documents two head lists (initial vs. validated) and explains the discrepancy. However, the top-10 ablation (§5.7) uses initial-run heads while the corrected ablation (§5.6.1) uses validated heads — this inconsistency weakens the top-10 result. L1H20 recovery (0.040) is consistent across §5.6.1 and §5.9.
4. **Steering baseline discrepancy:** The paper correctly explains the 28.4% vs. 28.0% difference as arising from the 200-sample holdout for steering vector computation.
5. **MMLU baseline variation (62.0% vs. 62.2%):** Explained as different random subsamples; acknowledged in a footnote. Acceptable.
6. **Wilson CI for steering opinion baseline:** Verified correct ([79.2%, 86.3%] matches manual calculation).

### Reproducibility

**Strong.** Fixed seeds (42 for evaluation, 100 for DPO), explicit SLURM job matrix, full artifact manifest with git hashes, TransformerLens configuration notes (`use_attn_result=True`), ~80 A100 GPU-hours total. The detailed output file table (§7) and engineering notes reference strengthen confidence. The balanced dataset creation process and answer position randomization are documented. Missing: the project repository URL is not provided (stated as "available in the project repository" without a link). A public code release is expected for camera-ready.

### Presentation

**Writing quality is high.** The paper is well-organized, clearly written, and appropriately caveated. The progressive narrowing from behavioral characterization (§5.1–5.2) through failed interventions (§5.6–5.8) to successful mitigation (§5.11) creates a compelling narrative arc.

**Concerns:**
- The paper is very long for a venue submission — 662 lines of markdown with extensive tables. The core contributions could be communicated more concisely. §5.6 (initial top-3 ablation) and §5.6.1 (corrected top-3 ablation) could be merged, with the initial run moved to an appendix.
- Six figures are referenced but not inline in the markdown — the reader must cross-reference `figures/` directory. In the LaTeX version (`paper.tex`), verify all figures are properly placed near their first reference.
- The probe control evolution (unbalanced → balanced) is buried across §5.3, §5.5, and Discussion. A cleaner presentation would describe only the balanced run in the main text and relegate the unbalanced run to an appendix as a methodological lesson.
- The "Other" category in the probe decomposition (12.1% pre-DPO) is defined only in a footnote. Given it drops 7.3 pp post-DPO — a notable change — it deserves more prominent discussion.

## Questions for Authors

1. **Can you provide DOIs, arXiv IDs, or proceedings URLs for Lee et al. (2024) and Yang et al. (2025)?** If these publications cannot be verified, how would you reframe Contribution #5's novelty positioning?

2. **Have you considered running path patching (edge-level) on Llama-3-8B-Instruct?** Given Chen et al.'s success with this method, would you predict that path patching identifies causally necessary edges, or that the null persists even at edge-level granularity?

3. **What is the minimum detectable effect (MDE) for your ablation test at N=1,500 and 80% power?** Is the null result informative — i.e., can you rule out effects larger than, say, 3 pp?

4. **Why was DPO not replicated on Mistral?** Given that Mistral has near-total factual sycophancy (99.8%), DPO targeting factual sycophancy on Mistral would test whether the social-compliance-to-robust-tracking mechanism generalizes to a qualitatively different sycophancy profile.

5. **The 34/100 patching success rate (total effect > 0.1) is notably low. How does this compare to patching success rates in Wang et al. (2022) or other circuit discovery work?** Does the low rate suggest that most samples are not strongly sycophantic under the forced-choice format, or that the total effect threshold is conservative?

6. **DPO training converged in 3 minutes with 95% reward accuracy from only 400 pairs. Have you checked for overfitting?** Specifically, can you show a held-out validation loss curve? The 23.8 pp reduction is impressive, but rapid convergence on a small dataset raises overfitting concerns.

7. **The "Other" category drops from 12.1% to 4.8% post-DPO (−7.3 pp). What does this mean mechanistically?** You describe it as "DPO improves internal coherence beyond the sycophancy-specific pathway" — can you elaborate? Does this represent a genuine improvement in internal representation quality, or a probe recalibration artifact?

8. **How do you reconcile the steering FDR result (no condition survives at aggregate level) with the per-source opinion-domain analysis (L15/L20 fall outside Wilson CI)?** Is the per-source analysis corrected for having inspected three domains?

## Minor Issues

- §5.5: Probe decomposition sums to 100.1% — adjust or add rounding footnote.
- §5.6/§5.6.1: Could be merged into a single section with the initial ablation as a subsection or appendix note.
- Chen et al. bib entry: `year={2025}` should be `year={2024}` (arXiv:2409.01658, published Sept 2024).
- Paduraru et al. (2025) bib entry: missing arXiv ID — add or note as non-indexed.
- Venhoff et al. (2025): provide workshop proceedings URL if not arXiv-indexed.
- §5.1 footnote on MMLU (62.0% vs 62.2%): this appears mid-table in markdown — verify rendering in LaTeX.
- The project repository URL is mentioned but not provided. Include for reproducibility.
- "TransformerLens 2.x" — specify exact version for reproducibility (2.0? 2.1? 2.7?).

## Sources

All external sources consulted during this review, with direct URLs where available:

| Source | URL |
|--------|-----|
| Chen et al. (2024), "From Yes-Men to Truth-Tellers" | https://arxiv.org/abs/2409.01658 |
| Li et al. (2025), "When Truth Is Overridden" | https://arxiv.org/abs/2508.02087 |
| O'Brien et al. (2026), "A Few Bad Neurons" | https://arxiv.org/abs/2601.18939 |
| Sharma et al. (2024), "Towards Understanding Sycophancy" | https://arxiv.org/abs/2310.13548 |
| Wei et al. (2023), "Simple synthetic data reduces sycophancy" | https://arxiv.org/abs/2308.03958 |
| Heimersheim & Nanda (2024), "How to use and interpret activation patching" | https://arxiv.org/abs/2404.15255 |
| Perez et al. (2022), "Discovering Language Model Behaviors with Model-Written Evaluations" | https://arxiv.org/abs/2212.09251 |
| Marks & Tegmark (2023), "The Geometry of Truth" | https://arxiv.org/abs/2310.06824 |
| Rafailov et al. (2023), "Direct Preference Optimization" | https://arxiv.org/abs/2305.18290 |
| Wang et al. (2022), "Interpretability in the Wild" (IOI circuit) | https://arxiv.org/abs/2211.00593 |
| Lee et al. (2024), "Mechanistic Understanding of DPO Toxicity Reduction" | Not found on arXiv/alphaXiv/ICML proceedings |
| Yang et al. (2025), "Mechanistic Analysis of DPO for Toxicity" | Not found on arXiv/alphaXiv/EMNLP proceedings |
| Venhoff et al. (2025), "Sparse Activation Fusion..." | Not found on arXiv (workshop paper) |
| Paduraru et al. (2025), "Select-and-Project Top-K..." | Not found on arXiv (no ID provided) |
