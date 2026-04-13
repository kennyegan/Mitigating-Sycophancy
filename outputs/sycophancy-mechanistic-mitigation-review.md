# Verification Review (Pass 3): Mitigating Sycophancy in Large Language Models: A Mechanistic Investigation

## Context

This is a **third-pass verification review** confirming that the two remaining MINOR issues from the second review have been addressed. The first review (April 10) identified 3 FATAL, 4 MAJOR, and 5 MINOR issues. The second review (same day) verified all FATAL and MAJOR issues were resolved and identified 2 new MINOR issues. This pass checks the final fixes.

## Full Resolution History

| # | Issue | Original Severity | Resolved In | Status |
|---|-------|-------------------|-------------|--------|
| 1 | Missing `results/` directory | **FATAL** | Pass 2 | ✅ All files present, manifest `missing_count: 0` |
| 2 | GSM8k 3× discrepancy (11.3% vs 33.2%) | **FATAL** | Pass 2 | ✅ §4 documents regex bug; canonical confirms 33.2% |
| 3 | Probe decomposition mismatch (1.8:1 vs 4.1:1) | **FATAL** | Pass 2 | ✅ Canonical matches paper; archive is preliminary |
| 4 | Chen et al. (2024) mischaracterization | **MAJOR** | Pass 2 | ✅ Now correctly describes SPT |
| 5 | Head ranking instability inadequately disclosed | **MAJOR** | Pass 2 | ✅ Std dev column, note, limitation #5 |
| 6 | Figure 4 inconsistency (full-probe vs CV accuracy) | **MAJOR** | Pass 2 | ✅ Figure now shows CV accuracy |
| 7 | "First mechanistic evidence" unqualified | **MAJOR** | Pass 2 | ✅ Lee & Yang cited as precedents |
| 8 | Samples skipped inconsistency | MINOR | Pass 2 | ✅ Canonical confirms 1,493 |
| 9 | Effect sizes inflated by floor effects | MINOR | Pass 2 | ✅ Cohen's d reported alongside h |
| 10 | Figure 6 rounding (−1.7 vs −1.8 pp) | MINOR | Pass 2 | ✅ Paper now says −1.7 pp |
| 11 | Mean ablation failure unexplained | MINOR | Pass 2 | ✅ Hypothesis added |
| 12 | DPO generalization insufficiently flagged | MINOR | Pass 2 | ✅ Generalization caveat added |
| 13 | DPO GSM8k N-mismatch (N=200 vs N=1,319) | MINOR | **Pass 3** | ✅ Table footnote added; Abstract/Conclusion changed to "preserved" |
| 14 | Archive-canonical probe discrepancy unexplained | MINOR | **Pass 3** | ✅ Reproducibility Statement explains different balanced-dataset randomization |

## Verification of Pass 3 Fixes

### MINOR #13: DPO GSM8k N-mismatch — ✅ FIXED

**What changed:**
- §5.11 table: GSM8k row now has `†` footnote: "Post-DPO GSM8k evaluated on N=200 vs. baseline N=1,319. 95% CI [32.0%, 45.4%] overlaps baseline; improvement not statistically significant."
- §5.11 text: "GSM8k is preserved (38.5%, N=200, 95% CI [32.0%, 45.4%] overlapping the N=1,319 baseline of 33.2%)"
- Abstract: Changed from "GSM8k +5.3 pp" → "GSM8k preserved"
- §9 Conclusion #5: Changed to "GSM8k preserved (38.5%, N=200, 95% CI overlapping baseline)"

**Verified against artifact:** `results/dpo_eval_results.json` confirms GSM8k accuracy=0.385, n_samples=200, CI=[0.320, 0.454]. Baseline 0.332 falls within CI. The fix is accurate and honest.

### MINOR #14: Archive-canonical probe discrepancy — ✅ FIXED

**What changed:** Reproducibility Statement now includes: "A preliminary archive (`results_archive/`) is retained for provenance but is superseded by the canonical `results/` directory. The preliminary snapshot (March 3) used a different balanced-dataset randomization for probe evaluation; the canonical run uses the finalized balanced-dataset generation described in Section 5.5, which accounts for the shift in probe decomposition fractions (e.g., social compliance 22.5%→18.0%) while transfer accuracy remains identical (77.9%)."

**Assessment:** This explanation is plausible — different answer position randomizations in the balanced dataset would change which samples the model gets right/wrong, shifting the four-way decomposition while leaving the probe's underlying accuracy unchanged. The fact that transfer accuracy is identical (77.9%) in both runs supports this explanation.

## One New Issue Found

### TRIVIAL: Citation year inconsistency for Yang et al.

The Yang et al. paper (arxiv 2411.06424, EMNLP 2025) is cited as **"Yang et al., 2024"** in two places (Abstract line and §5.11 post-probe paragraph) but correctly as **"Yang et al. (2025)"** in two other places (§1 and §9). The correct year is **2025** (EMNLP publication date).

Occurrences to fix:
- Abstract: "Lee et al., 2024; Yang et al., **2024**" → should be **2025**
- §5.11 post-DPO paragraph: "Lee et al., 2024; Yang et al., **2024**" → should be **2025**

## Updated Assessment

**Recommendation: Accept**
**Confidence: High**

All 14 issues across three review passes have been resolved. The single remaining item is a trivial citation year typo (Yang et al. 2024→2025 in two locations) that can be fixed in camera-ready. The paper's artifact record is complete, internally consistent, and all numerical claims verified against canonical data files.

## Strengths (brief)

1. **Neutral-transfer probe methodology** — elegant format-invariant design with valuable probe control demonstration
2. **Patching-to-ablation dissociation** — the most robust contribution; verified, replicated across architectures, with informative power analysis
3. **Domain-specific circuits** — zero overlap and sign-reversed head roles between opinion and fictional-entity sycophancy
4. **Cross-architecture replication** — Llama-3 + Mistral with inverted sycophancy profiles
5. **DPO mechanistic decomposition** — verified social compliance → robust truth-tracking conversion, honestly scoped relative to toxicity precedents
6. **Thorough self-criticism** — 7 explicit limitations, honest null results, conservative statistical claims

## Verdict

The paper has addressed all issues raised across two prior review passes. Every numerical claim matches the canonical artifact data. The two citation year typos (Yang et al. 2024→2025) are trivially fixable. This is a solid mechanistic interpretability paper with a genuinely useful negative result, a well-designed probe methodology, and a verified DPO mechanistic decomposition — all replicated across two architectures. **Accept.**

## Sources

- Chen et al. (2024), "Supervised Pinpoint Tuning" — https://proceedings.mlr.press/v235/chen24u.html
- Li/Wang et al. (2025), "When Truth Is Overridden" — https://arxiv.org/abs/2508.02087
- O'Brien et al. (2026), "A Few Bad Neurons" — https://arxiv.org/abs/2601.18939
- Heimersheim & Nanda (2024), "How to Use and Interpret Activation Patching" — https://arxiv.org/abs/2404.15255
- McGrath et al. (2023), "The Hydra Effect" — https://arxiv.org/abs/2307.15771
- Lee et al. (2024), "A Mechanistic Understanding of Alignment Algorithms" — https://proceedings.mlr.press/v235/lee24a.html
- Yang et al. (2025), "How Does DPO Reduce Toxicity?" — https://arxiv.org/abs/2411.06424
- Canonical artifacts verified: `results/probe_control_balanced_results.json`, `results/top10_ablation_full_gsm8k.json`, `results/dpo_eval_results.json`, `results/head_importance.json`, `results/full_rerun_manifest.json`, `results/corrected_ablation_results.json`, `results/baseline_llama3_summary.json`
