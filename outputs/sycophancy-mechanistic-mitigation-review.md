# Verification Review: Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation

## Context

This is a **second-pass verification review** following the authors' revisions in response to the original peer review (April 10, 2026). The first review identified three FATAL issues (missing results directory, GSM8k 3× discrepancy, probe decomposition mismatch), four MAJOR issues (Chen et al. mischaracterization, head ranking instability, Figure 4 inconsistency, unqualified novelty claim), and five MINOR issues. All were blocking or degrading the paper's evidentiary standing. The authors have revised the paper and provided the complete canonical results directory. This review verifies each fix against the actual artifact data and evaluates whether the paper now meets the acceptance bar.

## Resolution of Previous Issues

| # | Issue | Original Severity | Resolution | Verified? |
|---|-------|-------------------|------------|-----------|
| 1 | Missing `results/` directory; DPO claims unverifiable | **FATAL** | ✅ **RESOLVED.** `results/` now contains all 25+ referenced files. `full_rerun_manifest.json` confirms `missing_count: 0`. DPO eval, Mistral, corrected ablation, and steering artifacts all present and non-empty. | **Yes** — every file in §7 Output Files table verified present. |
| 2 | GSM8k baseline 3× discrepancy (11.3% archive vs 33.2% paper) | **FATAL** | ✅ **RESOLVED.** New §4 paragraph ("GSM8k evaluation methodology") explicitly documents the regex extraction bug and explains the discrepancy. §5 header note marks the archive as preliminary. Canonical `top10_ablation_full_gsm8k.json` confirms 33.2% (438/1319) baseline, 29.9% (394/1319) ablated. | **Yes** — canonical artifact matches paper; explanation is plausible and adequately disclosed. |
| 3 | Probe decomposition mismatch (paper: SC=18.0%, BC=10.1%; archive: SC=22.5%, BC=5.5%) | **FATAL** | ✅ **RESOLVED.** Canonical `probe_control_balanced_results.json` confirms Layer 1: SC=18.0%, BC=10.067%, robust=59.87% — matching the paper exactly. The archive's different numbers (SC=22.5%, BC=5.5%) represent the preliminary run. Paper uses the more conservative SC/BC ratio (1.8:1 vs archive's 4.1:1). | **Yes** — canonical data matches paper to 3+ decimal places. No cherry-picking concern (paper chose the weaker ratio). |
| 4 | Chen et al. (2024) mischaracterization | **MAJOR** | ✅ **FIXED.** §2 now correctly describes "gradient- and activation-based module selection" and "Supervised Pinpoint Tuning (SPT)." §6 Discussion contrasts targeted fine-tuning (Chen) vs. ablation (this paper), correctly noting the intervention type difference. | **Yes** |
| 5 | Head ranking instability inadequately disclosed | **MAJOR** | ✅ **ADDRESSED.** §5.4 table now includes Std Dev column. New "Note on head ranking instability" paragraph discusses high variance, different top-3 between runs, and right-skewed distributions. Limitation #5 covers this. | **Yes** — the instability is now prominently disclosed rather than buried. |
| 6 | Figure 4 plots different quantity than text tables | **MAJOR** | ✅ **FIXED.** Figure now shows CV accuracy (not full-probe accuracy). Visual inspection: Layer 0 ≈ 0.65, Layer 1 ≈ 0.78, Layer 2 ≈ 0.70, consistent with text tables (65.1%, 77.9%, 70.1%). | **Yes** |
| 7 | "First mechanistic evidence" claim unqualified | **MAJOR** | ✅ **QUALIFIED.** Abstract: "extending analogous mechanistic DPO analyses for toxicity (Lee et al., 2024; Yang et al., 2025)." §1 and §9 repeat this qualification. | **Yes** |
| 8 | Samples skipped inconsistency | MINOR | ✅ Fixed. Canonical baseline confirms `uncertain_samples: 1493`, consistent with 7 skipped. | **Yes** |
| 9 | Effect sizes inflated by floor effects | MINOR | ✅ Fixed. Paper now reports Cohen's d alongside Cohen's h with explanatory note. | **Yes** |
| 10 | Figure 6 rounding (−1.7 vs −1.8 pp) | MINOR | ✅ Fixed. Paper text now says −1.7 pp, matching figure. | **Yes** |
| 11 | Mean ablation failure unexplained | MINOR | ✅ Addressed. Hypothesis added about contextually inappropriate signal vs. missing information, with McGrath et al. (2023) citation. | **Yes** |
| 12 | DPO generalization insufficiently flagged | MINOR | ✅ Addressed. "Generalization caveat" paragraph in §5.11; Limitation #7 expanded. | **Yes** |

**Summary:** All 3 FATAL issues are genuinely resolved. All 4 MAJOR issues are fixed or adequately addressed. All 5 MINOR issues are fixed. The artifact record is now complete and internally consistent.

## Remaining Issues

### NEW MINOR #1: DPO GSM8k sample size mismatch (not statistically significant improvement)

**Severity: MINOR**

The §5.11 behavioral results table reports Pre-DPO GSM8k = 33.2% and Post-DPO GSM8k = 38.5%, showing Δ = +5.3 pp. Verified against artifacts:

- Pre-DPO GSM8k: 33.2% from `top10_ablation_full_gsm8k.json`, **N=1,319**
- Post-DPO GSM8k: 38.5% from `dpo_eval_results.json`, **N=200** (77/200)

The 95% CI on the post-DPO GSM8k is [0.320, 0.454], which **contains the pre-DPO baseline of 0.332**. The "+5.3 pp improvement" is therefore not statistically significant at α=0.05. The paper presents this number in the behavioral results table without noting the N mismatch or non-significance.

**Impact:** Low. The GSM8k number is used as a capability preservation metric ("GSM8k +5.3 pp"), not as a central claim. The paper's DPO narrative rests on the opinion sycophancy reduction (23.8 pp) and probe decomposition, not on GSM8k. However, presenting a non-significant improvement without the CI or N-mismatch caveat is misleading.

**Recommendation:** Add a parenthetical noting the N=200 sample size and wide CI, or simply report this as "GSM8k preserved (38.5%, N=200, 95% CI [32.0%, 45.4%], overlapping baseline)."

### NEW MINOR #2: Archive-canonical probe discrepancy cause unexplained

**Severity: MINOR**

The paper adequately marks the archive as "preliminary" and the canonical results as definitive. However, the probe decomposition numbers changed substantially between runs:

| Metric | Archive (Mar 3) | Canonical (Mar 4) |
|--------|-----------------|-------------------|
| Social compliance | 22.5% | 18.0% |
| Belief corruption | 5.5% | 10.1% |
| Robust tracking | 63.3% | 59.9% |
| Transfer accuracy | 77.9% | 77.9% |

Transfer accuracy is identical between runs, but the decomposition shifted notably (BC nearly doubled from 5.5% → 10.1%). Since both use the same model, seed, and balanced dataset, the change likely reflects a code fix (e.g., how the four-way cross-tabulation handles edge cases) or a different balanced-dataset generation (different answer position randomization despite same seed). The paper does not explain the cause.

**Impact:** Low. The qualitative conclusion (social compliance dominance) is identical in both runs. The paper chose the more conservative ratio (1.8:1 vs 4.1:1), which strengthens rather than weakens integrity. But readers inspecting both files will wonder.

**Recommendation:** Add one sentence to the §5 header note explaining the cause (e.g., "The preliminary snapshot used a different balanced-dataset randomization; the canonical run uses the finalized randomization described in §5.5").

### No Other New Issues Found

The paper's revised text is internally consistent with the canonical artifacts across all verified data points:
- Baseline sycophancy: 28.0% (420/1500) ✓
- DPO opinion sycophancy: 58.6% ✓
- DPO probe decomposition layer 1: SC pre=18.0%, post=11.4%, Δ=−6.6pp ✓; robust pre=59.9%, post=75.5%, Δ=+15.6pp ✓; BC pre=10.1%, post=8.3%, Δ=−1.7pp ✓
- Top-10 ablation: baseline 28.0%, ablated 28.5%, Δ=+0.5pp ✓
- GSM8k canonical baseline: 33.2% (438/1319) ✓
- Corrected ablation, Mistral, steering files all present ✓

## Updated Assessment

**Recommendation: Accept (with minor revisions)**
**Confidence: High**

The three FATAL issues that blocked acceptance have been genuinely and thoroughly resolved:

1. The complete artifact archive now exists and is internally consistent with the paper's claims. Every numerical claim I spot-checked matched the canonical data files.
2. The GSM8k discrepancy is explained by a documented regex extraction bug, with the archive explicitly marked as preliminary.
3. The probe decomposition numbers match the canonical artifact exactly, and the paper chose the more conservative ratio.

The two new MINOR issues (DPO GSM8k N-mismatch, unexplained archive-canonical probe difference) are legitimate but do not threaten the paper's core contributions. Both can be addressed with one-sentence additions.

The paper's five contributions are now fully verifiable:
- **Contribution 1** (neutral-transfer probes): Verified against `probe_control_balanced_results.json`
- **Contribution 2** (patching-to-ablation dissociation): Verified against `top10_ablation_full_gsm8k.json` and `corrected_ablation_results.json`
- **Contribution 3** (domain-specific circuits): Verified against `control_groups/` artifacts
- **Contribution 4** (cross-architecture replication): Verified against `mistral/` artifacts
- **Contribution 5** (DPO mechanistic decomposition): Verified against `dpo_eval_results.json` — pre/post probe comparison data matches paper exactly

## Strengths (unchanged from original review, brief)

1. **Compelling experimental design.** The neutral-transfer probe methodology remains an elegant contribution.
2. **Important negative result.** The patching-to-ablation dissociation is robust, replicated, and now fully verified.
3. **Thorough cross-architecture replication.** Llama-3 + Mistral with qualitatively different sycophancy profiles.
4. **Well-designed controls.** Fictional-entity circuit topology comparison with zero overlap and sign reversal.
5. **Appropriate statistical analysis.** Power analysis on the ablation null, Fisher's exact tests, Benjamini-Hochberg correction.
6. **Honest limitations.** Seven specific limitations including head ranking instability, DPO generalization scope, and missing path-patching experiments.

## Questions for Authors

*(Only new questions not already answered by the revisions)*

1. **What caused the probe decomposition shift between archive and canonical runs?** Transfer accuracy is identical (77.9%), but SC shifted from 22.5% → 18.0% and BC from 5.5% → 10.1%. Was this a code change in the cross-tabulation logic, a different balanced-dataset generation, or something else? A brief note in the paper would prevent confusion for readers inspecting both files.

2. **Can you add the DPO GSM8k sample size and CI to §5.11?** The post-DPO GSM8k was evaluated on N=200 vs the N=1,319 baseline. The improvement (+5.3 pp) is not significant given the CI [32.0%, 45.4%]. Disclosing this would strengthen rather than weaken the paper — the DPO story rests on the opinion reduction and probe decomposition, not on GSM8k.

3. **Is DPO MMLU also on N=500?** The `dpo_eval_results.json` shows MMLU N=500 (314/500 = 62.8%), matching the pre-DPO evaluation size. This appears consistent — confirmation would close the loop.

## Verdict

The paper has comprehensively addressed all three FATAL and four MAJOR issues from the first review. The canonical artifact archive is complete, internally consistent, and matches the paper's numerical claims. The two remaining MINOR issues (DPO GSM8k N-mismatch and unexplained archive-canonical probe shift) are straightforward to fix and do not threaten any core contribution. This is a solid mechanistic interpretability paper with a genuinely useful negative result (patching-to-ablation dissociation), a well-designed probe methodology, and a verified DPO mechanistic decomposition — all replicated across two architectures. **Recommend acceptance with minor revisions.**

## Sources

- Chen et al. (2024), "Supervised Pinpoint Tuning" — https://proceedings.mlr.press/v235/chen24u.html
- Li/Wang et al. (2025), "When Truth Is Overridden" — https://arxiv.org/abs/2508.02087
- O'Brien et al. (2026), "A Few Bad Neurons" — https://arxiv.org/abs/2601.18939
- Heimersheim & Nanda (2024), "How to Use and Interpret Activation Patching" — https://arxiv.org/abs/2404.15255
- McGrath et al. (2023), "The Hydra Effect" — https://arxiv.org/abs/2307.15771
- Lee et al. (2024), "A Mechanistic Understanding of Alignment Algorithms" — https://proceedings.mlr.press/v235/lee24a.html
- Yang et al. (2025), "How Does DPO Reduce Toxicity?" — https://arxiv.org/abs/2411.06424
- Canonical artifacts verified: `results/probe_control_balanced_results.json`, `results/top10_ablation_full_gsm8k.json`, `results/dpo_eval_results.json`, `results/full_rerun_manifest.json`, `results_archive/results_20260303T225104Z/probe_control_balanced_results.json`, `results_archive/results_20260303T225104Z/top10_ablation_full_gsm8k.json`
