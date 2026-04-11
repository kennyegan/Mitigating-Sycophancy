# Peer Review: Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation

## Summary

This paper applies mechanistic interpretability techniques — linear probes, causal activation patching, head ablation, representation steering, and DPO fine-tuning — to study sycophancy in Llama-3-8B-Instruct and Mistral-7B-Instruct. The core narrative proceeds in three acts: (1) format-controlled probes reveal that sycophancy is primarily "social compliance" (the model retains correct internal representations but outputs sycophantic responses), not "belief corruption"; (2) activation patching identifies attention heads carrying sycophantic signal, but ablating them — individually, in groups of 3, or all top 10 — produces zero sycophancy reduction, revealing a "patching-to-ablation dissociation" attributed to redundant circuit distribution; (3) DPO fine-tuning on 400 preference pairs reduces opinion sycophancy by 23.8 pp (82.4% → 58.6%) while preserving capabilities, and probe re-analysis shows this works by converting social compliance into robust truth-tracking.

The paper claims five novel contributions: (1) a neutral-transfer probe methodology with quantitative four-way decomposition; (2) empirical demonstration of patching-to-ablation dissociation on a safety-relevant behavior; (3) evidence of domain-specific sycophancy circuits with zero overlap; (4) cross-architecture replication on Mistral-7B; and (5) the first mechanistic decomposition of how DPO resolves sycophancy specifically. The experimental scope is ambitious — 13 SLURM jobs, two model families, multiple control experiments — and the narrative arc from diagnosis to failed inference-time intervention to successful training-time mitigation is compelling if the data holds up.

However, a detailed evidence audit reveals **critical data provenance failures** that undermine the paper's central claims: the canonical results directory does not exist, key numerical claims cannot be verified against archived artifacts, and a 3× discrepancy in a core capability metric goes unexplained.

## Overall Assessment

**Recommendation: Weak Reject**
**Confidence: High**

The paper presents a well-motivated and ambitious research program with several genuinely interesting findings — particularly the patching-to-ablation dissociation and the domain-specific circuit results. The narrative structure is exemplary for a mechanistic interpretability paper: build the methodology, discover the phenomenon, fail to intervene surgically, succeed with training-time methods, and explain why mechanistically. However, the paper cannot currently meet the evidentiary bar required for a top venue. Three critical issues block acceptance: (1) the canonical `results/` directory does not exist and the claimed validation manifest (`full_rerun_manifest.json`, `missing_count: 0`) cannot be verified — this is a fundamental reproducibility failure; (2) key numerical claims in the paper (probe decomposition ratios, GSM8k capability baselines) contradict the only available archived artifacts by factors of 1.5–3×; and (3) the paper's most novel contribution (DPO mechanistic probe re-analysis) rests entirely on result files absent from the repository. Until these provenance issues are resolved and the artifact record is complete, the paper's claims — however plausible — cannot be accepted at face value.

## Strengths

1. **Compelling experimental design.** The neutral-transfer probe methodology (train on neutral, test on biased) is an elegant control that cleanly separates format cues from genuine truth-tracking. The probe control experiment (§5.5) demonstrating that mixed-training probes learn format classification rather than truth directions is a valuable standalone methodological contribution.

2. **Important negative result.** The patching-to-ablation dissociation is the paper's most robust finding — confirmed from archived data (sycophancy counts 420 vs 427 out of 1500), replicated across two ablation batches (original top-3 and validated top-3), scaled to top-10, and cross-replicated on Mistral. The connection to Heimersheim & Nanda (2024)'s sufficiency/necessity framework is well-deployed. This finding has genuine implications for the circuit discovery community.

3. **Thorough cross-architecture replication.** Running the full pipeline on Mistral-7B with its qualitatively different sycophancy profile (near-total factual sycophancy, moderate opinion sycophancy — the inverse of Llama-3) strengthens the generality claims considerably.

4. **Well-designed control experiments.** The fictional-entity control (§5.9) revealing zero circuit overlap and sign-reversed head roles is a strong result. The base model comparison (§5.2) convincingly shows that instruction tuning reshapes rather than introduces sycophancy.

5. **Appropriate statistical analysis where verifiable.** The two-proportion z-test for the ablation null (z=0.28, p=0.78, 80% power to detect ±3.6 pp) is correctly specified and verified against archived counts. The power analysis transforms a null result into an informative bound.

6. **Honest limitations section.** The paper acknowledges the forced-choice format concern, DPO generalization scope, and missing experiments (path patching, Mistral DPO replication) that would strengthen the claims.

## Weaknesses

### FATAL

1. **FATAL: Canonical results directory does not exist; DPO claims unverifiable.** The paper states "All values are sourced from confirmed artifacts validated by `results/full_rerun_manifest.json` (`missing_count: 0`)" **[§5, header note]**, and the Reproducibility Statement reiterates this claim. However, `results/` does not exist in the repository. The only available artifacts are in `results_archive/results_20260303T225104Z/`, a snapshot from March 3, 2026 — 35 days before the paper's April 7 date. The DPO probe re-analysis **[§5.11, Table: Pre-DPO vs Post-DPO Probe Decomposition]** — the paper's headline novelty contribution and Contribution #5 — relies entirely on `results/dpo_eval_results.json`, which is absent. All Mistral results **[§5.10]**, corrected ablation results **[§5.6.1]**, steering results **[§5.8]**, and the manifest itself are missing. A paper claiming reproducibility with "missing_count: 0" from a nonexistent manifest is a serious integrity concern, even if the likely explanation is mundane (e.g., results generated on an HPC cluster but never committed to the repository).

2. **FATAL: GSM8k baseline 3× discrepancy.** The paper reports GSM8k baseline accuracy of **33.2% (438/1319)** throughout **[§5.7, Table; §5.8; §5.10; §5.11]**. The archived `top10_ablation_full_gsm8k.json` (same experiment, same N=1319, same seed=42) records **11.3% (149/1319)** — a factor of 3× difference. This is not a rounding issue. Either the evaluation methodology changed between the archived run (March 2) and the final run (undated, absent), or one of the numbers is wrong. This discrepancy propagates to every capability claim in the paper: the ablation GSM8k retention narrative ("90.0% retained" at 394/438 vs. 140/149), the steering capability baselines, and the DPO "+5.3 pp" improvement claim. The paper does not acknowledge any evaluation methodology change.

3. **FATAL: Probe decomposition numbers contradict archived data.** The paper claims Layer 1 social compliance = 18.0%, belief corruption = 10.1%, robust = 59.9%, yielding an SC/BC ratio of 1.8:1 **[§5.3, §5.5, §9]**. The archived `probe_control_balanced_results.json` for Layer 1 shows SC = 22.5% (338/1500), BC = 5.5% (83/1500), robust = 63.3% (950/1500), yielding SC/BC = 4.1:1. Both support social compliance dominance, but the stated ratio understates it by more than 2×, and the absolute percentages differ by 4.5–5 pp. The paper's decomposition numbers match Figure 6 exactly — suggesting an internally consistent later analysis — but the only verifiable artifact tells a different quantitative story. This is particularly problematic because the 1.8:1 ratio is a headline claim repeated in the abstract, conclusion, and discussion.

### MAJOR

4. **MAJOR: Chen et al. (2024) mischaracterization.** **[§2, ¶3]** The paper states Chen et al. "use **path patching** on Llama-2-Chat to identify sycophancy-related heads and apply targeted fine-tuning ('pinpoint tuning'), achieving substantial sycophancy reduction through **head-level knockout**." Chen et al. (ICML 2024) actually use gradient/activation-based module selection (not path patching) and their intervention is Supervised Pinpoint Tuning — targeted fine-tuning of identified modules, not ablation ("knockout"). This mischaracterization matters because the paper's Discussion **[§6, "Ablation success vs. failure"]** explicitly contrasts its null ablation result against Chen et al.'s supposed ablation success. If Chen et al. didn't perform ablation, this comparison is misleading.

5. **MAJOR: Head ranking instability inadequately disclosed.** **[§5.4, Table]** The archived head recovery scores show standard deviations exceeding means for all top-ranked heads (L1H20: mean=0.569, std=1.211; L5H5: mean=0.567, std=0.695; L4H28: mean=0.506, std=0.672). The paper's own methodology produced two different top-3 rankings from the same pipeline, with L5H31 going from recovery −0.171 to +0.256 between runs. This instability is acknowledged only as a "Note" in §5.4 rather than as a substantive methodological limitation. For a paper that builds two full ablation experiments (§5.6, §5.6.1) on head rankings, this variance should be prominently disclosed, and confidence intervals on individual head recovery scores should be reported.

6. **MAJOR: Figure 4 plots a different quantity than the text table.** **[§5.3, §5.5, Figure 4]** Figure 4 shows biased transfer accuracy at layers 0–12 as approximately 86%, with a sharp drop at layer 13. But the paper's text tables report Layer 1 biased transfer as 77.9%, Layer 0 as 65.1%, Layer 2 as 70.1%. The archive confirms the figure uses `biased_accuracy_full_probe` (85.9% at Layer 1) while the text reports `biased_cv_accuracy` (77.9%). These are methodologically distinct quantities — full-probe accuracy on all training data vs. cross-validated held-out accuracy — and the 8 pp gap is not cosmetic. The figure and text disagree without explanation, which undermines the reader's ability to interpret the social compliance gap visualized in the figure.

7. **MAJOR: "First mechanistic evidence" claim needs qualification.** **[Abstract, §1, §9]** The paper claims "the first mechanistic decomposition of how DPO resolves *sycophancy*." Lee et al. (ICML 2024) and Yang et al. (EMNLP 2025) provide analogous mechanistic analyses of how DPO resolves *toxicity* — the same class of analysis applied to a different behavior. The claim should be explicitly scoped as "the first for sycophancy specifically" and the toxicity precedents cited.

### MINOR

8. **MINOR: Samples skipped inconsistency.** **[§5.1]** The paper claims "1,493 (7 skipped due to tokenization)" but the archived baseline JSON shows `samples_skipped: 0, samples_evaluated: 1500`.

9. **MINOR: Effect size presentation inflates perceived magnitude.** **[§5.1]** Cohen's h values of 2.276 and 2.022 are reported for comparisons against 0% sycophancy rates. These are technically correct but dominated by floor effects (any comparison against 0% yields extreme h). The archived Cohen's d values on the continuous compliance gap (d = 0.18, 0.78) are more informative and should be reported alongside h.

10. **MINOR: Figure 6 rounding error.** **[§5.11, Figure 6]** The figure labels belief corruption change as −1.7 pp; the text states −1.8 pp. Minor but symptomatic of insufficient cross-checking between figures and text.

11. **MINOR: Mean ablation catastrophic failure unexplained.** **[§5.6, §5.6.1]** Mean-ablation of 3 heads (both original and validated sets) caused "catastrophic output degradation (all outputs unparseable)." This striking result is excluded from analysis without mechanistic investigation. If these heads carry structural information important for coherent generation (even if not sycophancy-specific), this constrains the space of viable inference-time interventions and deserves at least a hypothesis.

12. **MINOR: DPO training set size and generalization.** **[§5.11]** 400 training pairs with LoRA rank 16 is aggressive. Train loss drops from 0.69 to 0.16 in 3 minutes, suggesting the model may be memorizing the Anthropic model-written-evals format rather than learning a generalizable anti-sycophancy direction. The paper's own Limitations section acknowledges no OOD evaluation was performed, but this should be flagged more prominently as a threat to the DPO claims.

## Detailed Comments

### Novelty

**Contribution 1 (Neutral-transfer probes, four-way decomposition):** Genuinely novel methodology. The neutral-transfer design is a clean improvement over mixed-training probes. Li et al. (2025) reach a conceptually equivalent conclusion via logit-lens; this paper's probe approach and quantitative decomposition add value. **However**, the specific decomposition numbers cannot be verified (see Weakness #3). If the archived 4.1:1 ratio is correct, the novelty framing (emphasizing the more moderate 1.8:1 ratio) is misleading.

**Contribution 2 (Patching-to-ablation dissociation):** The strongest contribution. The null ablation result is verified from archived data. The connection to Heimersheim & Nanda (2024) is well-drawn. The cross-architecture replication strengthens this considerably. However, the paper should note that self-repair/redundancy has been documented for other behaviors (McGrath et al., 2023; Rushing & Nanda, 2024) — the novelty is in demonstrating it specifically for sycophancy and at this scale.

**Contribution 3 (Domain-specific circuits):** Novel and well-supported. The fictional-entity control with zero circuit overlap and sign-reversed head roles is compelling. The 93% fictional-entity sycophancy rate as a "default agreement heuristic" is an interesting interpretation.

**Contribution 4 (Cross-architecture replication):** Valuable but incremental. Running the pipeline on Mistral confirms generality but does not introduce new methodology.

**Contribution 5 (DPO mechanistic decomposition):** Potentially the most impactful contribution, but **entirely unverifiable** from available artifacts. The DPO eval results, training metrics, and model adapter are all absent from the repository. Lee et al. (2024) and Yang et al. (2025) perform analogous mechanistic DPO analyses for toxicity; the "first" claim must be scoped accordingly.

### Empirical Rigor

- **Statistical tests are generally appropriate where verifiable.** The two-proportion z-test with power analysis (§5.7) is well-specified. Fisher's exact tests for the probe decomposition are reasonable for count data.
- **Sample sizes vary without justification.** Head-level patching uses N=100, layer patching uses 34/100, ablation GSM8k uses N=200 (§5.6) or N=1319 (§5.7), steering uses a 200/1300 train/eval split. The rationale for each is unclear.
- **34/100 patching success rate (§5.4)** means layer importance is computed from only 34 samples. Phase 2 head-level patching includes the 66 non-sycophantic samples, diluting signal with noise. This is disclosed but understated.
- **Confidence intervals are missing for head recovery scores,** which have std > mean (see Weakness #5). The specific top-K ranking is essentially arbitrary given this variance.
- **Benjamini-Hochberg correction for steering (§5.8)** is appropriate, and the honest reporting that no aggregate condition survives correction is commendable. The per-source analysis showing L15/L20 effects outside the Wilson CI is a reasonable secondary analysis.

### Reproducibility

This is the paper's most serious shortcoming. The paper makes strong reproducibility claims — fixed seeds, ~80 A100 GPU-hours, validated manifest with `missing_count: 0` — but:

- The `results/` directory does not exist.
- The `full_rerun_manifest.json` does not exist.
- The only available archive (March 3) predates the paper by 35 days.
- 11+ critical result files (covering DPO, Mistral, corrected ablation, steering) are entirely absent.
- Two verified numerical discrepancies exist between the archive and paper text.

The code and SLURM scripts appear to be present, but without result artifacts, a reviewer cannot verify any claim beyond the baseline sycophancy rates and the ablation null.

### Figures and Tables

- **Figure 1** (patching heatmap): Consistent with text; early-layer concentration is visually clear.
- **Figure 2** (steering sweep): Consistent with the aggregate null finding.
- **Figure 3** (per-source steering): Supports the opinion-domain reduction claim; L15/L20 dips visible.
- **Figure 4** (probe accuracy): **Inconsistent with text tables** — plots full-probe accuracy (~86%) while text reports CV accuracy (77.9%). This is a substantive discrepancy that creates a misleading visual impression of probe transfer quality.
- **Figure 5** (ablation comparison): Effective visualization of the null result. Error bars present.
- **Figure 6** (DPO probe decomposition): Internally consistent with paper text but **cannot be verified against source data**. Minor rounding inconsistency (−1.7 pp in figure vs. −1.8 pp in text).

### Writing and Presentation

The paper is well-written with a clear narrative arc. The Discussion section is particularly strong in contextualizing results against concurrent work and drawing the fMRI/lesion analogy. However:

- At ~8,000 words of main text plus extensive tables, the paper is long for a conference submission and would benefit from tightening §5.6 and §5.6.1 (which can be consolidated since both show the same null).
- The notation "pp" (percentage points) is used throughout without definition.
- Section numbering inconsistencies: §5.9 follows §5.8 but the intervening §5.8 subsection on Benjamini-Hochberg reads as a continuation of §5.8 rather than a standalone section.
- The "Note on TransformerLens configuration" in §4 is valuable engineering context but belongs in supplementary material.

## Questions for Authors

1. **Where is the `results/` directory?** The paper claims all artifacts are validated by `results/full_rerun_manifest.json`. This file does not exist in the repository. Were results generated on the HPC cluster but never committed? Can you provide the complete artifact archive?

2. **Which probe decomposition numbers are correct?** The archived `probe_control_balanced_results.json` (Layer 1) shows SC=22.5%, BC=5.5% (ratio 4.1:1). The paper claims SC=18.0%, BC=10.1% (ratio 1.8:1). These appear to come from different runs. Which is canonical, and what changed between runs?

3. **What explains the GSM8k 3× discrepancy?** The archived `top10_ablation_full_gsm8k.json` shows GSM8k baseline accuracy of 11.3% (149/1319). The paper reports 33.2% (438/1319) for the same experiment. Did the answer extraction logic change? If so, which methodology is correct, and do all GSM8k-dependent claims (retention, DPO improvement, steering baselines) use the same evaluation?

4. **Can you provide the DPO result artifacts?** `results/dpo_eval_results.json`, `results/dpo_training_metrics.json`, and the LoRA adapter (`results/dpo_model/`) are all absent. Contribution #5 cannot be evaluated without these.

5. **What is Figure 4 plotting?** The y-axis values (~86% at Layer 1) do not match the text-table values (77.9% at Layer 1). The archive suggests the figure shows `biased_accuracy_full_probe` while the text reports `biased_cv_accuracy`. Which is the intended metric, and can you provide a corrected figure or clarifying caption?

6. **Why does the head ranking change between patching runs?** L5H31 goes from recovery −0.171 to +0.256 between the archived and "validated" runs. Can you characterize the source of this instability? Have you computed bootstrap confidence intervals on head recovery scores?

7. **Can you correct the Chen et al. (2024) characterization?** Their method is gradient/activation-based module selection with targeted fine-tuning (SPT), not "path patching" with "head-level knockout." The Discussion's comparison of your null ablation against their supposed ablation success needs revision since they did not perform ablation.

8. **Does mean ablation failure generalize?** Both the original and validated top-3 mean ablations caused complete output degradation. Does this occur for arbitrary sets of 3 heads, or is it specific to the patching-identified heads? This could reveal something important about these heads' functional roles.

9. **Has DPO transfer been tested on out-of-distribution opinion prompts?** The 400 training pairs and 500 evaluation samples both come from Anthropic model-written-evals. A held-out opinion benchmark (e.g., manually constructed prompts, or a different opinion dataset) would distinguish genuine anti-sycophancy learning from format memorization.

10. **Can you clarify the "7 skipped" claim?** The paper says 1,493 evaluated (7 skipped), but the archived baseline JSON shows 1,500 evaluated and 0 skipped. Is this from a different run?

## Verdict

This paper tackles an important problem with an ambitious and well-designed experimental program. The patching-to-ablation dissociation is a genuinely valuable finding that the mechanistic interpretability community should know about, and the domain-specific circuit results add real insight. However, **three fatal data provenance issues** — a missing results directory that invalidates all reproducibility claims, a 3× GSM8k discrepancy between archive and paper, and probe decomposition numbers that contradict the only available artifact — prevent acceptance in the current form. The paper's most novel contribution (DPO mechanistic decomposition) cannot be evaluated at all without the absent result files. I would encourage the authors to restore the complete artifact archive, reconcile all numerical discrepancies with clear audit trails, correct the Chen et al. characterization, and resubmit.

## Sources

- Chen et al. (2024), "Supervised Pinpoint Tuning" — https://proceedings.mlr.press/v235/chen24u.html
- Li/Wang et al. (2025), "When Truth Is Overridden" — https://arxiv.org/abs/2508.02087
- O'Brien et al. (2026), "A Few Bad Neurons" — https://arxiv.org/abs/2601.18939
- Heimersheim & Nanda (2024), "How to Use and Interpret Activation Patching" — https://arxiv.org/abs/2404.15255
- McGrath et al. (2023), "The Hydra Effect" — https://arxiv.org/abs/2307.15771
- Rushing & Nanda (2024), "Explorations of Self-Repair" — https://proceedings.mlr.press/v235/rushing24a.html
- Lee et al. (2024), "A Mechanistic Understanding of Alignment Algorithms (DPO + Toxicity)" — https://proceedings.mlr.press/v235/lee24a.html
- Yang et al. (2025), "How Does DPO Reduce Toxicity?" — https://arxiv.org/abs/2411.06424
- Sharma et al. (2024), "Towards Understanding Sycophancy in Language Models" — ICLR 2024
- Evidence brief: `outputs/sycophancy-mechanistic-mitigation-research.md` (automated audit, April 10, 2026)
- Archived artifacts: `results_archive/results_20260303T225104Z/` (verified probe_control_balanced_results.json, top10_ablation_full_gsm8k.json, baseline_llama3_summary.json)
