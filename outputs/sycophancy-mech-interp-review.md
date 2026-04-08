# Peer Review (Revision 2): Mitigating Sycophancy in Large Language Models — A Mechanistic Investigation

**Reviewer:** Anonymous
**Date:** April 8, 2026
**Artifact:** `paper.md` (revised draft, post-first-review)
**Review type:** Verification of revisions against 4 prior MAJOR issues + remaining items

---

## 1. Summary

This paper presents a comprehensive mechanistic interpretability study of sycophancy in Llama-3-8B-Instruct and Mistral-7B-Instruct. Using linear probes, causal activation patching, head ablation, representation steering, and DPO fine-tuning, it characterizes sycophancy as primarily "social compliance" (the model retains correct internal representations but outputs sycophantic responses), demonstrates a "patching-to-ablation dissociation" (identified circuit heads are sufficient carriers but not causally necessary due to redundancy), identifies domain-specific circuits, replicates all findings across two architectures, and shows that DPO resolves sycophancy by converting social compliance into robust truth-tracking.

The revised paper now engages substantively with three concurrent mechanistic sycophancy papers (Chen et al. 2025, Li et al. 2025, O'Brien et al. 2026), properly attributes the sufficiency/necessity framing to Heimersheim & Nanda (2024), qualifies the "first mechanistic evidence" claim to sycophancy specifically, and explains the patching success rate discontinuity. These revisions significantly strengthen the paper's scholarly positioning and intellectual honesty. The core experimental contributions — the neutral-transfer probe design, the systematic ablation null replicated across architectures, and the DPO probe re-analysis — remain the paper's strongest assets.

---

## 2. Verdict

**Accept — minor revisions recommended.**

The four MAJOR issues from the first review are resolved. The paper now situates itself properly within the concurrent literature, acknowledges limitations of its redundancy claim, and makes appropriately scoped novelty claims. Remaining issues are moderate to minor and do not affect core claims.

---

## 3. Resolution Status

### MAJOR-1: Three missing mechanistic sycophancy papers ✅ Resolved

**Evidence:**

- **Chen et al. (2025)** — Cited in §2 ("Concurrent work has begun applying circuit discovery directly to sycophancy. Chen et al. (2025) use path patching on Llama-2-Chat to identify sycophancy-related heads and apply targeted fine-tuning...") and discussed in §6 under "Ablation success vs. failure" with three specific reconciliation factors: (a) edge-level vs. node-level patching, (b) challenge-induced vs. assertion-based sycophancy, (c) Llama-2 vs. Llama-3 RLHF differences. Present in `references.bib` as `chen2025pinpoint`.

- **Li et al. (2025)** — Cited in §2 ("Li et al. (2025) apply logit-lens and activation patching to Llama-3.1-8B-Instruct, finding that sycophantic output crystallizes in late layers (16–23)") and discussed in §6 under "Layer localization" with a clear complementary interpretation: residual-stream patching identifies where information *enters* computation vs. logit-lens identifies where the *output decision* crystallizes. Present in `references.bib` as `li2025truth`.

- **O'Brien et al. (2026)** — Cited in both §2 (SAE-based approaches paragraph) and discussed in §6 under "Feature-level vs. head-level redundancy." The reconciliation — head-level redundancy doesn't imply feature-level redundancy; SAE decomposition isolates sparse features that may not be redundantly encoded — is well-reasoned and appropriately scopes the "redundantly distributed" claim. Present in `references.bib` as `obrien2026fewbad`.

**Assessment:** All three papers are cited in Related Work, discussed substantively in a dedicated Discussion subsection ("Comparison with Concurrent Circuit Discovery Work"), and the contradictions are reconciled with specific mechanistic arguments. The reconciliation is intellectually honest — it acknowledges that Chen et al.'s ablation *succeeds* and O'Brien et al.'s SAE approach *succeeds*, and offers principled explanations for the divergence rather than dismissing them.

---

### MAJOR-2: Unexplained 34/100 → 100/100 patching success rate ✅ Resolved

**Evidence:** §5.4 now includes an explicit paragraph after the Phase 1 table:

> "The 34/100 success rate reflects the total-effect threshold: only samples where biased and neutral prompts produced meaningfully different outputs (total effect > 0.1) were included in layer importance scoring. Phase 2 (head-level patching) patches all 100 samples regardless of total effect, reporting recovery scores for each; samples with near-zero total effect contribute near-zero recovery scores, diluting but not biasing the head ranking."

**Assessment:** The explanation is clear and the argument that low-total-effect samples *dilute* rather than *bias* head rankings is logically sound. The characterization of Phase 1's filtering as "conservative but appropriate" is defensible — deriving layer importance only from samples that actually express sycophancy is methodologically reasonable. One could argue the paper should note that this means the top-layer ranking applies specifically to the subpopulation of "strongly sycophantic" samples, but this is a minor caveat, not a blocking issue.

---

### MAJOR-3: "First mechanistic evidence" claim overstated ✅ Resolved

**Evidence:** The claim is now appropriately qualified in all three locations:

1. **Abstract:** "the first mechanistic evidence of how preference optimization resolves *sycophantic* output-gating" (emphasis on *sycophantic* added)
2. **Introduction (§1):** "the first mechanistic decomposition of how DPO resolves *sycophancy*... Prior work has analyzed DPO mechanistically for toxicity (Lee et al., 2024; Yang et al., 2025); we extend this to sycophancy and show a qualitatively different mechanism (output-gating elimination rather than representation suppression)."
3. **Conclusion (#6):** "the first mechanistic evidence of how preference optimization resolves sycophantic behavior specifically — extending prior mechanistic DPO analyses on toxicity (Lee et al., 2024; Yang et al., 2025) to a qualitatively different failure mode"

Both Lee et al. (2024) and Yang et al. (2025) are in `references.bib`.

**Assessment:** The claim is now precisely scoped. The addition of the "qualitatively different mechanism" framing (output-gating elimination vs. representation suppression) is a nice touch that positions the contribution clearly relative to prior DPO mechanistic work.

---

### MAJOR-4: Heimersheim & Nanda 2024 missing ✅ Resolved

**Evidence:** §6, "Sufficiency vs. Necessity" subsection now reads:

> "Heimersheim & Nanda (2024) formalize this distinction explicitly, defining denoising patching as a test of sufficiency and noising patching as a test of necessity, and warning against conflating the two — a caution our results empirically validate for a complex, safety-relevant behavior."

Present in `references.bib` as `heimersheim2024patching`.

**Assessment:** Properly attributed with the specific conceptual debt (denoising = sufficiency, noising = necessity) acknowledged. The paper positions its contribution as *empirical validation* of their theoretical framework on sycophancy, which is appropriate.

---

## 4. Strengths

1. **Neutral-transfer probe design is a genuine methodological contribution (§5.3, §5.5).** Training on neutral-condition activations and testing on biased-condition activations from matched samples cleanly disentangles format cues from truth-representation tracking. The balanced replication with randomized answer positions eliminates the remaining positional confound. The cautionary finding — that mixed-training probes learn format artifacts to >99% accuracy — is independently valuable.

2. **Patching-to-ablation dissociation is the most thoroughly documented null result in the circuit discovery literature (§5.6–5.7, §5.6.1).** Three tiers of ablation (top-3 original, top-3 validated, top-10), two models, both zero and mean ablation, with proper statistical testing (z=0.28, p=0.78). The corrected ablation targeting the validated head set directly preempts the most obvious objection.

3. **Cross-architecture replication transforms single-model findings into general claims (§5.10).** Llama-3 and Mistral have inverted sycophancy profiles and entirely different top heads, yet both show social compliance dominance, null ablation, and null steering. This is compelling evidence that these are structural properties of RLHF-trained models.

4. **DPO probe re-analysis closes the diagnostic-to-intervention loop (§5.11).** Showing that social compliance drops 6.6 pp while belief corruption barely moves (−1.8 pp) and robust tracking increases 15.6 pp is a tight result. The consistency across all tested layers (0–5) strengthens the finding.

5. **Honest engagement with contradictory concurrent work (§6, new subsection).** The "Comparison with Concurrent Circuit Discovery Work" subsection is among the best parts of the revision. It doesn't dismiss Chen et al.'s successful ablation or O'Brien et al.'s successful SAE correction — it offers principled explanations (sycophancy type, granularity level, model differences) and appropriately scopes the redundancy claim to head-level and residual-stream-level interventions.

6. **Statistical rigor remains unusually high for the MI literature.** Wilson CIs for extreme proportions, Benjamini-Hochberg FDR correction across 56 steering conditions, Fisher's exact test for probe cross-tabulation, proper effect-size reporting (Cohen's h). The evaluation code includes log-sum-exp numerical stability.

7. **The "Other" category is now defined (§5.5 footnote).** The footnote clarifying "probe incorrect, model correct" and the observation that this drops 7.3 pp post-DPO (suggesting improved internal coherence) adds interpretive value.

---

## 5. Remaining Issues

### MODERATE-1: §5.6 table formatting is still broken

The `[^mmlu]` footnote at line 297 interrupts the markdown table mid-row, causing `| L5H5 only |` and subsequent rows to render outside the table structure. This is a rendering bug that will confuse readers of the markdown version. The footnote should be placed after the table closes, or reformatted as an inline note.

### MODERATE-2: Mistral 99.8% factual sycophancy still lacks format verification

The 99.8% factual sycophancy rate for Mistral (§5.10) remains unexplained beyond the observation that "sycophancy profiles are shaped by model-specific RLHF procedures." This is an extreme result — only 1 of 500 factual samples resists the biased prompt. A positional bias check (swapping A/B labels) would rule out the possibility that Mistral simply selects whichever option is mentioned first/last in the prompt regardless of content. Without this, the figure is suspicious and weakens the Mistral replication's interpretive value for factual-domain claims specifically. The opinion and reasoning domain findings are unaffected.

### MODERATE-3: DPO generalization scope remains underspecified

The paper's core argument — that DPO succeeds where inference-time methods fail — is based on DPO trained on 400 opinion-domain pairs and evaluated on 500 in-distribution opinion-domain samples. The Limitations section acknowledges binary forced choice (item 2) but does not explicitly note that DPO has not been tested on: (a) out-of-distribution opinion prompts, (b) other sycophancy domains (factual, reasoning, fictional-entity), or (c) the Mistral model. This matters because the paper's Discussion claims "training-time preference optimization is the appropriate intervention level" as a general principle — but the evidence base is a single model on a single domain with in-distribution evaluation. A sentence in Limitations explicitly bounding the DPO generalization claim would suffice.

### MODERATE-4: Venhoff et al. (2025) and Paduraru et al. (2025) SAE references remain unverifiable

The vague "SAF/MLAS, NeurIPS 2025 Workshop" and "S&P Top-K" references from the first review have been replaced with named citations (Venhoff et al., 2025; Paduraru et al., 2025), but the `references.bib` entries lack arXiv IDs or DOIs (`arXiv preprint` with no number for Paduraru; `NeurIPS 2025 Mechanistic Interpretability Workshop` with no proceedings link for Venhoff). If these are real papers, they should have full citation details. If they are placeholder references, they should be removed.

### MINOR-1: Layer reconciliation argument could be sharper

The "Comparison with Concurrent Circuit Discovery Work" reconciles the early-layer (this paper) vs. late-layer (Li et al. 2025) contradiction by arguing residual-stream patching measures where information "enters" computation while logit-lens measures where the "output decision crystallizes." This is plausible but could be made more rigorous. Specifically: if early-layer patching recovers honest behavior by replacing the signal at layers 1–5, and logit-lens shows the sycophantic output only crystallizes at layers 16–23, then the information must be *carried* through the residual stream across ~15 layers without being visible in the unembedding space until late. This is consistent with the residual stream functioning as a "communication bus" (Elhage et al., 2021), but the paper could strengthen the argument by noting this explicitly and acknowledging it as a testable prediction rather than a demonstrated fact.

### MINOR-2: Cohen's h still not reported for DPO or cross-architecture comparisons

Cohen's h is reported for the baseline domain comparisons (§5.1) but not for the DPO reduction (82.4% → 58.6%, which would be h ≈ 0.53, a medium effect) or the Llama-3 vs. Mistral opinion comparison (82.4% vs. 50.8%). Consistent effect-size reporting would strengthen quantitative claims.

### MINOR-3: The neuroscience analogy (fMRI vs. lesions) persists without qualification

§6 still compares activation patching to fMRI and ablation to lesion studies. As noted in the first review, this overstates the passivity of patching — activation patching is an interventionist technique (replacing activations), closer to temporary inactivation methods like TMS or optogenetics than to the purely observational nature of fMRI. A brief qualifying clause would make the analogy more precise.

### MINOR-4: `references.bib` entries for new citations use placeholder author fields

Several new bib entries use `author={Li, others}`, `author={O'Brien, others}`, `author={Lee, others}`, `author={Yang, others}`. These should be filled with complete author lists before submission.

---

## 6. Questions for Authors

1. **Layer complementarity test:** Have you run logit-lens analysis on the activations at layers 1–5 (your patching-identified critical layers) to verify that the sycophantic signal is *not* yet visible in the unembedding space at these early layers? This would directly test the "complementary rather than contradictory" interpretation of the discrepancy with Li et al. (2025).

2. **Edge-level patching:** Given that Chen et al. (2025) achieve successful ablation using path patching (edge-level) rather than node-level patching, have you considered running path patching on your dataset? This would directly test whether edge-level analysis identifies more causally necessary components for assertion-based sycophancy, or whether the redundancy persists at the edge level.

3. **Mistral positional bias:** What happens if you swap the A/B option positions for the 500 TruthfulQA factual samples on Mistral? If the 99.8% sycophancy rate drops substantially, it may reflect positional bias rather than content-based sycophancy. This is a quick experiment that would significantly strengthen the Mistral replication.

4. **GSM8k improvement under DPO (+5.3 pp):** DPO was trained exclusively on opinion-domain data, yet GSM8k accuracy improves from 33.2% to 38.5% — a 16% relative increase on an unrelated capability. Is this noise, or do you have a mechanistic hypothesis? Could DPO be removing a general output-suppression tendency that also slightly impairs math reasoning?

5. **SAE analysis as next step:** Given that O'Brien et al. (2026) demonstrate successful SAE-based localization on Gemma-2, and your paper explicitly notes that feature-level decomposition may overcome head-level redundancy — have you attempted SAE analysis on Llama-3 for sycophancy? Even a preliminary result (positive or negative) would substantially strengthen the Discussion.

6. **Fictional-entity circuit ablation:** You identified a distinct fictional-entity circuit (§5.9) but did not report ablation results for those heads. Would ablating L1H10, L0H2, L0H0 reduce fictional-entity sycophancy, or would the same redundancy pattern appear? This would test whether redundancy is a general property or specific to opinion-domain circuits.

---

## 7. Reproducibility Assessment

**Strong.** The paper provides:

- Fixed seeds (42 evaluation, 100 DPO training) with explicit justification for the split
- Complete SLURM job matrix (13 Llama-3 jobs + 5 Mistral jobs + DPO pipeline) with wall times
- Hardware specification (A100-SXM4-80GB, Unity HPC)
- Software versions (Python 3.10.19, PyTorch 2.10.0+cu128, TransformerLens 2.x)
- GPU-hour estimates (~80 A100-hours total)
- Validated artifact manifest (`results/full_rerun_manifest.json`, missing_count: 0)
- Practical implementation notes (TransformerLens `use_attn_result` configuration)
- Unit tests in `tests/` covering data processing, schema contracts, baselines, probes, steering, and manifests
- Git commit hashes for key artifacts (e.g., `0ad8f02`, `e292645`, `326a8b5a`)

**One concern persists:** Result JSON/CSV artifacts are referenced throughout but are not present in the repository snapshot (the `results/` directory appears gitignored or HPC-resident). For full reproducibility, these should be hosted externally (e.g., Zenodo, Hugging Face Datasets) with a persistent DOI or link in the Reproducibility Statement.

---

## 8. Sources

| Reference | URL |
|-----------|-----|
| Chen et al. 2025 ("Pinpoint Tuning") | https://arxiv.org/abs/2409.01658 |
| Li et al. / Wang et al. 2025 ("When Truth Is Overridden") | https://arxiv.org/abs/2508.02087 |
| O'Brien et al. 2026 ("A Few Bad Neurons") | https://arxiv.org/abs/2601.18939 |
| Heimersheim & Nanda 2024 ("How to use and interpret activation patching") | https://arxiv.org/abs/2404.15255 |
| Lee et al. 2024 (Mechanistic DPO for toxicity, ICML) | https://arxiv.org/abs/2401.01967 |
| Yang et al. 2025 (DPO + toxicity, EMNLP) | https://arxiv.org/abs/2411.06424 |
| Panickssery et al. 2023 (Steering vectors) | https://arxiv.org/abs/2312.06681 |
| McGrath et al. 2023 (Hydra Effect / self-repair) | https://arxiv.org/abs/2307.15771 |
| Rushing & Nanda 2024 (Self-repair, ICML) | https://proceedings.mlr.press/v235/rushing24a.html |
| Sharma et al. 2024 (Sycophancy characterization, ICLR) | Referenced in paper |
| Wei et al. 2023 (Synthetic data for sycophancy) | Referenced in paper |
| Rafailov et al. 2023 (DPO, NeurIPS) | Referenced in paper |
| Wang et al. 2022 (IOI circuit, activation patching) | Referenced in paper |
