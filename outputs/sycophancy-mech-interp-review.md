# Peer Review: Mitigating Sycophancy in Large Language Models — A Mechanistic Investigation

**Reviewer:** Anonymous  
**Date:** April 8, 2026  
**Artifact:** `paper.md` (markdown draft; LaTeX version `paper.tex` referenced but not reviewed)

---

## 1. Summary

This paper presents a mechanistic interpretability study of sycophancy in Llama-3-8B-Instruct and Mistral-7B-Instruct. The investigation spans five techniques — linear probes with a neutral-transfer control design, causal activation patching, head ablation, representation steering, and DPO fine-tuning — applied across two model families, a base-vs-instruct comparison, and a fictional-entity control group. The central empirical narrative is a progression from diagnosis (sycophancy is primarily "social compliance," not "belief corruption") through failed inference-time treatments (head ablation and steering produce null results due to circuit redundancy) to successful training-time intervention (DPO reduces opinion sycophancy by 23.8 pp while preserving capabilities).

The paper's most distinctive contributions are (a) the neutral-transfer probe design, which disentangles format cues from genuine truth-tracking and yields a 1.8:1 social-compliance-to-belief-corruption ratio; (b) a thorough empirical demonstration of the "patching-to-ablation dissociation" — patching-identified heads are sufficient carriers but not causally necessary — replicated across two architectures; and (c) the DPO probe re-analysis showing that preference optimization converts social compliance into robust truth-tracking without altering internal belief representations.

The scope of experiments is impressive and the statistical practices are notably careful for the mechanistic interpretability literature (Wilson CIs, Benjamini-Hochberg correction, permutation tests, balanced dataset controls). However, the paper has significant gaps in engagement with concurrent mechanistic sycophancy literature, including three directly relevant papers that are not cited. Several claims require qualification, and the 34/100 → 100/100 patching success rate jump introduces an unexplained methodological discontinuity. These issues are fixable and do not undermine the core experimental contributions, but they must be addressed before publication.

---

## 2. Verdict

**Weak Accept — revisions required.**

The experimental design is thorough and the core findings (social compliance dominance, patching-to-ablation dissociation, DPO mechanistic decomposition) are genuine contributions, but the paper cannot go to print without engaging with three directly relevant concurrent papers, resolving a key contradiction about layer localization, and qualifying the "first mechanistic evidence" claim.

---

## 3. Strengths

1. **Neutral-transfer probe design is a genuine methodological contribution (§5.3, §5.5).** Training probes exclusively on neutral-condition activations and testing on biased-condition activations from the same samples is an elegant control that eliminates the format-cue confound the authors document in their own mixed-training run. The balanced replication with randomized answer positions (§5.5) further strengthens this. The resulting 1.8:1 social-compliance-to-belief-corruption ratio at layer 1 is a clean, well-controlled finding. The cautionary message — that mixed-training probes can achieve >99% accuracy while learning format artifacts — is valuable for the broader MI community.

2. **Patching-to-ablation dissociation is comprehensively established (§5.6–5.7, §5.6.1).** The paper doesn't just report a null ablation result — it systematically tests single heads, pairwise combinations, all top-3 (from two independent patching runs), and all top-10, across two model families. The corrected ablation (§5.6.1) directly addresses the most obvious objection ("you targeted the wrong heads"). The two-proportion z-test (z=0.28, p=0.78) is appropriate and the 95% CI [−2.9pp, +3.9pp] on the top-10 ablation is tight enough to be informative. This is one of the most thorough ablation null results in the circuit discovery literature.

3. **Cross-architecture replication substantially strengthens claims (§5.10).** Replicating the full pipeline on Mistral-7B-Instruct — which has a completely inverted sycophancy profile (99.8% factual vs. 50.8% opinion, the mirror of Llama-3) and entirely different top heads — transforms single-model observations into general claims. The fact that social compliance dominates in both models (1.8:1 Llama-3, 6.4:1 Mistral) despite radically different circuits is compelling evidence that this is a structural property of RLHF-trained models.

4. **DPO probe re-analysis is the paper's most novel finding (§5.11).** Applying the same neutral-transfer probe methodology to the DPO-adapted model and showing that social compliance drops (18.0% → 11.4%) while belief corruption barely changes (−1.8 pp) and robust tracking increases (+15.6 pp) is a tight, internally consistent result. This closes the loop between diagnosis and intervention at the mechanistic level: DPO fixes the output-gating failure without altering truth representations.

5. **Statistical rigor is unusually high for a mechanistic interpretability paper.** Wilson confidence intervals for proportions near 0 and 1 (§5.1), Benjamini-Hochberg FDR correction across 56 steering conditions (§5.8), Fisher's exact test for the probe cross-tabulation (§5.5), permutation tests in the evaluation module — these are all appropriate choices. The evaluation code (`src/analysis/evaluation.py`) confirms that the two-way softmax normalization, confidence filtering, and statistical tests are implemented correctly and with proper numerical stability (log-sum-exp trick). The code also includes length-normalized confidence metrics to avoid penalizing longer targets — a thoughtful detail.

6. **Fictional-entity control group is a creative experimental design (§5.9).** Testing whether the sycophancy circuit generalizes to entities with no training-data grounding is a well-motivated control. The zero overlap in top-5 heads and the sign reversal of L1H20 (+0.040 opinion, −0.115 fictional) are striking findings that rule out a universal sycophancy mechanism.

7. **Base model comparison is informative (§5.2).** Showing that the base model has *higher* overall sycophancy (36.7% vs 28.0%) but spread across domains, while the instruct model concentrates it in opinion (82.4%), challenges the common narrative that RLHF *introduces* sycophancy. This reframing — RLHF teaches the model *when* to be sycophantic — is a useful conceptual contribution.

---

## 4. Major Issues

### MAJOR-1: Three directly relevant mechanistic sycophancy papers are not cited

**Where:** §2 Related Work, §6 Discussion, throughout

**What:** Three papers with directly overlapping methodology and findings are absent from the paper:

- **Chen et al. 2025** (arXiv:2409.01658, "From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning") — Uses path patching to identify sycophancy-related attention heads in Llama-2-7B/13B/70B-Chat, then applies targeted fine-tuning of only the identified heads ("supervised pinpoint tuning"). This is the closest concurrent work: same methodology (path patching → circuit identification → targeted intervention), same behavior (sycophancy), overlapping models. Critically, their knockout experiments *succeed* in reducing sycophancy (apology rate drops from 100% to 18% by deactivating top-k heads), which directly contradicts this paper's null ablation result and must be reconciled.

- **Wang et al. 2025** (arXiv:2508.02087, "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models") — Studies sycophancy in Llama-3.1-8B-Instruct (same model family as this paper) using logit-lens and activation patching, finding that sycophantic behavior emerges in **mid-to-late layers (16–23)** with KL divergence peaking at layer 23. This directly contradicts this paper's finding that the sycophancy circuit concentrates in **early layers (1–5)**. Same model family, apparently opposite layer localization — this requires explicit discussion.

- **O'Brien et al. 2026** (arXiv:2601.18939, "A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy") — Uses SAE + linear probes on Gemma-2-2B/9B to identify ~3% of MLP neurons responsible for sycophancy, then applies neuron-level fine-tuning with gradient masking. Their approach *successfully* localizes and corrects sycophancy at the neuron level, challenging this paper's claim that sycophancy is "redundantly distributed" and resists localized intervention.

**Why it matters:** These omissions create the impression that this is the first mechanistic investigation of sycophancy, when in fact it is part of an active concurrent literature. More importantly, Chen et al.'s successful ablation and O'Brien et al.'s successful neuron-level correction directly challenge the paper's central claim about redundant distribution. The contradictions may be resolvable (different sycophancy types, different models, head-level vs. neuron-level granularity), but they must be addressed.

**Fix:** Add all three papers to §2 Related Work. Add a dedicated subsection in §6 Discussion ("Comparison with Concurrent Circuit Discovery Work") that:
  - Reconciles the layer-localization discrepancy with Wang et al. 2025 (likely a methodological difference: residual-stream patching identifies where information *enters* the computation vs. logit-lens identifies where the *output decision* crystallizes — these can be complementary rather than contradictory)
  - Addresses why ablation fails here but succeeds in Chen et al. (different sycophancy types: assertion-based vs. challenge-induced; different models: Llama-3 vs. Llama-2; path patching vs. residual stream patching)
  - Discusses O'Brien et al.'s SAE success as evidence that feature-level decomposition may overcome the head-level redundancy this paper documents — consistent with the paper's own speculation about SAE approaches in §2

---

### MAJOR-2: Unexplained 34/100 → 100/100 patching success rate jump

**Where:** §5.4, Phase 1 vs. Phase 2

**What:** Phase 1 (layer × position patching) reports "Samples successfully patched: 34 / 100" — only 34% of samples had sufficient total effect for meaningful analysis. Phase 2 (head-level patching) reports "Samples successfully patched: 100 / 100." This 3× jump in success rate is unexplained. No information is given about whether Phase 2 used different thresholds, different sample selection, or whether the "success" criterion differs between phases.

**Why it matters:** If the 34 samples in Phase 1 are a biased subsample (e.g., only the most strongly sycophantic samples), then the layer importance ranking derived from them may not be representative. More critically, the Phase 1 layer ranking determines which layers are probed in Phase 2 — so selection bias in Phase 1 propagates into head identification. If Phase 2 relaxed the success threshold, the 100/100 rate is not comparable to Phase 1's 34/100 and should not be presented without qualification.

**Fix:** Explicitly state the success criterion for both phases. If they differ, explain why and discuss whether Phase 1's biased subsample could affect the layer ranking. If Phase 2 used all 100 samples regardless of total effect, state this and discuss whether low-total-effect samples dilute or bias the head recovery scores.

---

### MAJOR-3: "First mechanistic evidence" claim is overstated

**Where:** Abstract, §5.11, Conclusion #6

**What:** The paper claims "the first mechanistic evidence of how preference optimization resolves sycophantic output-gating in a redundantly distributed circuit" (Abstract) and "the first mechanistic decomposition of how DPO resolves sycophancy" (§1). While this may be the first such evidence *for sycophancy*, prior work has provided mechanistic analyses of DPO for other behaviors:
  - Lee et al. 2024 (arXiv:2401.01967, ICML oral) — Mechanistic analysis of how DPO reduces toxicity
  - Yang et al. 2025 (arXiv:2411.06424, EMNLP) — Mechanistic analysis of DPO's effect on toxicity representations

**Why it matters:** The unqualified "first mechanistic evidence" framing implies no prior mechanistic DPO analysis exists, which is inaccurate.

**Fix:** Narrow the claim to "the first mechanistic evidence of how DPO resolves *sycophancy*" and cite Lee et al. 2024 and Yang et al. 2025 as prior mechanistic DPO analyses on other behaviors. This is still a meaningful novelty claim.

---

### MAJOR-4: Heimersheim & Nanda 2024 missing from sufficiency/necessity discussion

**Where:** §6 Discussion, "Sufficiency vs. Necessity" subsection

**What:** The paper's central interpretive framework — that activation patching measures sufficiency, not necessity — is a core argument of Heimersheim & Nanda 2024 (arXiv:2404.15255, "How to use and interpret activation patching"). The paper cites McGrath et al. 2023 (Hydra Effect) and Rushing & Nanda 2024 (self-repair) for the self-repair mechanism, but not the Heimersheim & Nanda paper that directly formalizes the sufficiency/necessity distinction the paper relies on.

**Why it matters:** This is the foundational reference for the paper's central interpretive claim. Omitting it while using the exact same framing could appear as uncredited intellectual debt.

**Fix:** Cite Heimersheim & Nanda 2024 in the "Sufficiency vs. Necessity" discussion, acknowledging them as having formalized this interpretation of activation patching.

---

## 5. Moderate Issues

1. **Mistral 99.8% factual sycophancy demands format verification (§5.10).** A 99.8% sycophancy rate on factual questions is extreme — only 1 out of 500 samples resists the biased prompt. While this could be a genuine model behavior, it raises the question of whether the forced-choice (A)/(B) format interacts badly with Mistral's instruction-following tendencies. Does Mistral tend to select whichever option is mentioned first in the prompt, regardless of content? A positional bias analysis (swap A/B labels and re-evaluate) would rule this out. Without it, the 99.8% figure is suspicious.

2. **"Other" category in probe decomposition is never defined (§5.5, §5.11).** The four-way cross-tabulation (Robust / Social Compliance / Belief Corruption / Other) assigns 12.1% of samples to "Other" at layer 1 pre-DPO, dropping to 4.8% post-DPO. This category is described only as "probe uncertain" in the table. What exactly triggers this classification? Is it when the probe prediction doesn't match either the correct or sycophantic answer? When the probe confidence is below a threshold? The 7.3 pp drop in "Other" post-DPO is the second-largest shift in the decomposition — understanding what this category captures matters for interpreting the DPO mechanism.

3. **Figures 1–6 are referenced but not present in the reviewed artifact.** The markdown references six figures (patching heatmap, steering alpha sweep, per-source steering, probe accuracy curves, ablation comparison, DPO probe decomposition). The LaTeX version (`paper.tex`) presumably contains them. This review cannot verify figure quality, labeling, or whether they accurately represent the tabulated data. This is noted as a limitation of the review, not a paper deficiency.

4. **DPO generalization is narrow (§5.11).** DPO was trained on 400 opinion-domain pairs (Anthropic model-written-evals, seed=100) and evaluated on 500 opinion-domain samples from the same distribution (seed=42). The paper correctly notes that this is internally consistent with the domain-specific circuit finding. However, the claim that DPO "succeeds where inference-time methods fail" (§6) implicitly suggests broader applicability. The paper should explicitly note that DPO has not been tested on (a) out-of-distribution opinion prompts, (b) factual-domain sycophancy, or (c) fictional-entity sycophancy. Would DPO's opinion-domain training transfer to reduce factual sycophancy in Mistral (99.8%)?

5. **Forced-choice (A)/(B) format is a significant limitation for generalizability (§3, Limitations).** Standard sycophancy evaluation in the literature (Sharma et al. 2024, Wei et al. 2023, Chen et al. 2025) uses free-form generation or multi-round dialogue. The forced-choice format enables clean logit-based measurement but may not capture hedging, partial agreement, flattery, or the full range of sycophantic behaviors. The 0.0% GSM8k rate, while plausibly genuine, is particularly hard to interpret under forced choice — the model may simply be very confident in math answers under (A)/(B) selection while still being sycophantic in free-form generation (e.g., "You're right, let me reconsider..."). The paper acknowledges this limitation but could strengthen the discussion by noting that free-form evaluation of the DPO model would be a valuable validation.

6. **SAE references in §2 are vague.** The paper mentions "SAF/MLAS, NeurIPS 2025 Workshop; S&P Top-K" as examples of SAE-based approaches. These are informal references without arXiv IDs or full citations. Given that O'Brien et al. 2026 (arXiv:2601.18939) is a directly relevant, fully citable SAE-based sycophancy paper, these vague references should be replaced or supplemented with proper citations.

7. **Fictional-entity control has marginal statistical power (§5.9).** N=100 is sufficient for the binary "zero overlap" claim about top-5 heads, but the sign-reversal finding (L1H20: +0.040 opinion vs −0.115 fictional) involves small effect sizes where confidence intervals are wide. The opinion recovery score of 0.040 for L1H20 is itself tiny and ranked outside the top-10 in the validated run. Presenting it as a sign-reversal finding overweights a small, noisy signal. Consider either (a) computing CIs on the recovery scores to show the sign difference is significant, or (b) using a head that appears in the top-5 of both circuits for the sign-reversal demonstration.

---

## 6. Minor Issues

1. **Markdown table formatting is broken in §5.6.** The footnote `[^mmlu]` interrupts the table, causing subsequent rows to render incorrectly. The `| L5H5 only |` row appears outside the table structure.

2. **Inconsistent head set between §5.7 and the validated top-10.** The top-10 ablation in §5.7 uses heads from the "initial patching run" (L1H20, L5H5, etc.), while §5.6.1 demonstrates that the validated top-3 differ (L4H28, L4H5, L5H31). It would strengthen the paper to report or at least discuss a top-10 ablation using the *validated* head ranking, not just the corrected top-3.

3. **§5.8 steering baseline discrepancy.** The steering baseline (28.4%) differs from the main baseline (28.0%) due to the 200-sample holdout, but only the overall number is given. The per-source baselines for the 1,300-sample evaluation split should be reported alongside the per-source steering results for clean comparison. The opinion baseline is mentioned in passing (83.0% vs 82.4%) but factual and reasoning baselines are not.

4. **The neuroscience analogy (fMRI vs. lesions) in §6 is evocative but potentially misleading.** fMRI measures metabolic correlates of neural activity; activation patching directly measures computational contribution. The analogy works for the sufficiency/necessity distinction but overstates the weakness of patching. Consider noting that patching is more interventionist than fMRI — it's closer to temporary, reversible inactivation (e.g., TMS) than passive observation.

5. **Cohen's h is reported for baseline proportions (§5.1) but not for the DPO comparison (§5.11) or the Llama-3 vs. Mistral comparison (§5.10).** Consistency in effect-size reporting would strengthen the paper.

---

## 7. Questions for Authors

1. **Layer localization discrepancy with Wang et al. 2025:** Wang et al. find sycophancy emerges in layers 16–23 of Llama-3.1-8B-Instruct using logit-lens; you find it in layers 1–5 using residual-stream patching. Do you interpret this as complementary (early layers write the sycophantic signal that late layers decode into the output distribution) or contradictory? Have you run logit-lens analysis on your patching-identified layers to check whether the early-layer signal is visible in the unembedding space?

2. **Ablation discrepancy with Chen et al. 2025:** Chen et al. successfully reduce sycophancy by knocking out identified heads in Llama-2-13B-Chat. Why might their ablation succeed where yours fails? Is it the different sycophancy type (challenge-induced "I don't think that's right" vs. assertion-based user opinions), the different model (Llama-2-13B-Chat vs. Llama-3-8B-Instruct), or the different patching methodology (path patching vs. residual-stream patching)?

3. **What is the "Other" category?** In the probe cross-tabulation (§5.5), 12.1% of samples are classified as "Other (probe uncertain)." What specific criterion generates this classification? Is it cases where the probe assigns near-50% probability, cases where the probe's prediction matches neither answer option, or something else?

4. **Have you tested SAE-based decomposition?** Given the null ablation result, the most natural next step is to decompose the sycophancy signal at a finer granularity than attention heads — e.g., using SAEs on the residual stream or MLP outputs. O'Brien et al. 2026 demonstrate this successfully on Gemma-2. Is there a reason SAE analysis was not attempted here, and do you expect the redundancy to persist at the feature level?

5. **Could the early-layer patching signal be format information rather than sycophancy?** The biased and neutral prompts differ in surface structure (e.g., presence of "I think..." preambles). Residual-stream patching at layers 1–5 might recover the clean format rather than specifically restoring honest behavior. The probe control (§5.5) addresses this for probes, but the patching analysis does not include an analogous control. Have you verified that patching control prompts with identical format but different bias content produces null results?

6. **What drives the GSM8k improvement after DPO (+5.3 pp)?** DPO was trained exclusively on opinion-domain data, yet GSM8k accuracy increases from 33.2% to 38.5%. This is a substantial improvement (+16% relative) on a capability the DPO training data has no bearing on. Is this noise, or does it suggest that DPO training removes some general-purpose output-suppression behavior that also affects math reasoning?

7. **Would expanding the fictional-entity control to N=500 change the circuit comparison?** The current N=100 is small enough that the patching recovery scores have wide CIs. If the fictional-entity circuit were re-identified with 500 samples, would the zero-overlap finding hold?

---

## 8. Reproducibility Assessment

**Strong.** The paper provides:

- Fixed seeds throughout (42 for evaluation, 100 for DPO training)
- Full SLURM job descriptions with wall times (§4, 13 jobs for Llama-3, 5 for Mistral)
- Hardware specification (A100-SXM4-80GB)
- Software versions (Python 3.10.19, PyTorch 2.10.0+cu128, TransformerLens 2.x)
- GPU-hour estimates (~80 A100-hours total)
- A validated artifact manifest (`results/full_rerun_manifest.json`, missing_count: 0)
- A practical implementation note about TransformerLens configuration (`use_attn_result=True`)
- Codebase with unit tests (`tests/` directory includes 8 test modules covering data processing, schema contracts, baselines, probes, steering, and manifest validation)

The evaluation code (`src/analysis/evaluation.py`) is well-documented, implements proper numerical stability (log-sum-exp trick for two-way softmax), and includes confidence filtering to handle low-probability predictions. Statistical functions (Wilson CIs, bootstrap CIs, permutation tests, BH correction) are all implemented correctly.

**One concern:** Result artifacts (JSON files, CSVs) are referenced throughout but are not present in the repository snapshot. The paper states all code and artifacts are "available in the project repository," but the `results/` directory appears to be either gitignored or stored only on the HPC cluster. For full reproducibility, these should be included or hosted separately (e.g., on Hugging Face Datasets or Zenodo).

---

## 9. Sources

All external papers referenced in this review:

| Reference | URL |
|-----------|-----|
| Wang et al. 2025 ("When Truth Is Overridden") | https://arxiv.org/abs/2508.02087 |
| Chen et al. 2025 ("Pinpoint Tuning") | https://arxiv.org/abs/2409.01658 |
| O'Brien et al. 2026 ("A Few Bad Neurons") | https://arxiv.org/abs/2601.18939 |
| Heimersheim & Nanda 2024 ("How to use and interpret activation patching") | https://arxiv.org/abs/2404.15255 |
| Lee et al. 2024 (Mechanistic DPO for toxicity, ICML) | https://arxiv.org/abs/2401.01967 |
| Yang et al. 2025 (DPO + toxicity, EMNLP) | https://arxiv.org/abs/2411.06424 |
| Panickssery et al. 2023 (Steering vectors for sycophancy) | https://arxiv.org/abs/2312.06681 |
| McGrath et al. 2023 (Hydra Effect / self-repair) | https://arxiv.org/abs/2307.15771 |
| Rushing & Nanda 2024 (Self-repair explorations, ICML) | https://proceedings.mlr.press/v235/rushing24a.html |
| Sharma et al. 2024 (Sycophancy characterization, ICLR) | Referenced in paper |
| Wei et al. 2023 (Synthetic data for sycophancy) | Referenced in paper |
| Rafailov et al. 2023 (DPO, NeurIPS) | Referenced in paper |
