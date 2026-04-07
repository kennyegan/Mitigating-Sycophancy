# Numerical Consistency Audit — paper.md

**Date:** 2026-04-07  
**Scope:** Internal cross-reference of all reported numbers across Sections 5.1–5.10, Discussion, and Conclusion.

---

## 1. Baseline Numbers (28.0%, 82.4%, 1.6%, 0.0%)

**CONSISTENT ✓**

| Location | Overall | Opinion | Factual | Reasoning |
|----------|---------|---------|---------|-----------|
| §5.1 (primary) | 28.0% | 82.4% | 1.6% | 0.0% |
| §5.2 (Instruct col) | 28.0% | 82.4% | 1.6% | 0.0% |
| §5.6 (Baseline row) | 28.0% | — | — | — |
| §5.6.1 (Baseline row) | 28.0% | 82.4% | — | — |
| §5.7 (Baseline row) | 28.0% | 82.4% | 1.6% | 0.0% |
| §5.10 (Llama-3 col) | 28.0% | 82.4% | 1.6% | 0.0% |
| §9 Conclusion | 82.4% opinion stated | — | — | — |

All references are consistent. Count check: 420/1500 = 28.0% ✓ (§5.7 sycophantic count matches rate).

---

## 2. Ablation Results: §5.6 vs §5.6.1 vs §5.7

### Consistent items ✓
- All three sections report the same **0.0% GSM8k reasoning sycophancy** at baseline (§5.7).
- Baseline overall sycophancy = 28.0% in all three sections ✓
- "All 3 (zero)" ablation in §5.6.1 yields −0.3 pp; "All 10 (zero)" in §5.7 yields +0.5 pp — directional consistency makes sense (corrected heads, larger set), no internal contradiction.

### MINOR INCONSISTENCY ⚠️ — MMLU Baseline Varies
| Section | Baseline MMLU |
|---------|--------------|
| §5.6 | 62.0% |
| §5.6.1 | **62.2%** |
| §5.7 | 62.0% |

Section 5.6.1 reports 62.2% while §5.6 and §5.7 both report 62.0%. These likely reflect independent MMLU evaluation subsets across experimental runs, but no explanation is provided. Readers may notice the inconsistency. **Recommend adding a footnote** clarifying that MMLU varies slightly across independent evaluation runs.

---

## 3. Mistral Replication Numbers (§5.10)

**CONSISTENT ✓**

| Value | §5.10 | §9 Conclusion | Discussion |
|-------|-------|--------------|------------|
| Overall sycophancy | 50.3% | — | 50.3% ✓ |
| Opinion sycophancy | 50.8% | 50.8% ✓ | 50.8% ✓ |
| Factual sycophancy | 99.8% | 99.8% ✓ | — |
| Reasoning sycophancy | 0.2% | — | — |
| SC/BC ratio | 6.4:1 | 6.4:1 ✓ | 6.4:1 ✓ |
| Top-10 ablation change | +1.0 pp | +1.0 pp ✓ | +1.0 pp ✓ |

Mistral baseline MMLU (50.6%) and GSM8k (9.3%) appear only in §5.10 and are not cross-referenced elsewhere, so no inconsistency possible.

---

## 4. Probe Results (1.8:1 ratio, 18.0% SC, 10.1% BC, 59.9% robust)

**CONSISTENT ✓**

All four values appear identically across §5.3, §5.5, §6 Discussion, and §9 Conclusion.

Arithmetic check on ratio: 18.0 / 10.1 = 1.782 → rounds to ~1.8:1 ✓

Arithmetic check on cross-tab totals: 18.0 + 10.1 + 59.9 + 12.1 = 100.1% — negligible rounding, acceptable ✓

---

## 5. Head Identifiers in Inconsistent Contexts

### SIGNIFICANT INCONSISTENCY ❌ — Discussion Recovery Scores (0.51–0.57) Do Not Match Any Results Table

**Location:** §6 Discussion, "Sufficiency vs. Necessity" subsection

**Text says:**
> "The top 3 heads (L1H20, L5H5, L4H28) show the highest activation patching recovery scores **(0.51–0.57)**"

**What the paper actually reports:**
| Source | L4H28 | L1H20 | L5H5 |
|--------|-------|-------|------|
| §5.4 validated run (top-10 table) | 0.4428 | *not in top 10* | *not in top 10* |
| §5.6.1 note (validated scores) | — | **0.040** | **−0.237** |

The recovery scores 0.51–0.57 do not appear anywhere in the paper. They appear to come from an **earlier, pre-correction patching run** that was superseded (the same run that produced the original Section 5.6 ablation targets). The Discussion text was not updated after the validated rerun replaced those numbers.

**Effect:** A reader comparing the Discussion to the Results section will find the cited recovery scores unattributable. This is especially problematic because the paper explicitly acknowledges in §5.4 and §5.6.1 that the original patching run was superseded — yet the Discussion continues to reference its scores as if they are current.

**Recommended fix:** Update the Discussion to reference the **validated scores**: L4H28 (0.443), L4H5 (0.302), L5H31 (0.256), and note that the original Section 5.6 ablation targeted heads from an earlier run. Alternatively, explicitly state that 0.51–0.57 refers to the pre-correction run.

---

### MINOR INCONSISTENCY ⚠️ — Section 5.7 Top-10 Head List Uses Earlier (Unvalidated) Run

**§5.7 ablated heads:** L1H20, L5H5, L4H28, L5H17, L3H17, L5H4, L5H19, L5H24, L4H5, L3H0

**§5.4 validated top-10:** L4H28, L4H5, L5H31, L2H5, L3H30, L5H24, L3H17, L3H28, L1H11, L4H26

Overlap: only 4 heads (L4H28, L3H17, L5H24, L4H5). Section 5.7 does not explicitly state it is using the earlier run's top-10 list. A reader will naturally compare the two lists and notice they diverge substantially. Section 5.4 says "a corrected ablation targeting the validated top-3 also showed no sycophancy reduction" (in the note), but Section 5.7 itself is not labeled as using the pre-correction top-10.

**Recommended fix:** Add a parenthetical in §5.7 clarifying: "heads from the pre-correction patching run; see §5.4 note and §5.6.1 for the validated top-3 corrected ablation."

---

### MINOR ISSUE ⚠️ — Section 5.9 "Rank 4" of Opinion Circuit is Misleading

The §5.9 comparison table shows:

| Rank | Opinion Circuit | Recovery |
|------|-----------------|----------|
| 4 | L1H20 | 0.040 |

But the actual rank-4 head in the validated opinion circuit (§5.4) is **L2H5 (0.2445)**, not L1H20. L1H20 has a validated recovery score of 0.040 and is not in the opinion circuit top-10 at all.

L1H20 is included in the §5.9 table specifically to demonstrate the sign reversal between domains, which is a legitimate and interesting finding. However, labeling it as "Rank 4" in the opinion circuit comparison is factually incorrect based on the validated results in §5.4.

**Recommended fix:** Add a footnote or rename the column header. For example: "L1H20 (shown for sign-reversal contrast; actual opinion rank > 10 in validated run, recovery = 0.040)"

---

## 6. Effect Sizes and CI Ranges

### Cohen's h — Minor Rounding Discrepancy
| Comparison | Reported | Calculated |
|------------|---------|------------|
| Opinion vs. Reasoning | h = 2.276 | h ≈ 2.285 (diff: 0.009) |
| Opinion vs. Factual | h = 2.022 | h ≈ 2.032 (diff: 0.010) |
| Reasoning vs. Factual | h = −0.254 | h = −0.254 ✓ |

The opinion comparisons are off by ~0.01, likely due to intermediate rounding. Not a material issue, but could be flagged in revision.

### GSM8k Count/Retention Arithmetic ✓
- 394/1319 = 29.87% ≈ 29.9% ✓
- 438/1319 = 33.21% ≈ 33.2% ✓
- 394/438 = 89.95% ≈ 90.0% ✓

### Ablation Count ✓
- 427/1500 = 28.47% ≈ 28.5% ✓

---

## 7. Other Observations

### Steering Baseline Discrepancy — Explained ✓
§5.8 acknowledges that the steering evaluation baseline is 28.4% (vs. main 28.0%) due to the 200-sample held-out split. This is correctly explained.

The per-source steering baseline (opinion 83.0%) vs. main baseline (82.4%) is a 0.6pp difference from the same split. This is not explicitly explained for the per-source numbers, only for the overall rate. Consider adding one sentence.

### §5.3 and §5.5 Describe the Same Experiment ✓
Both sections report the balanced probe results. This is not an error (§5.3 presents results; §5.5 presents methodology/validation), but the structure may confuse readers who encounter the same numbers in two places. The paper would benefit from a clear sentence in one of the sections noting this.

---

## Summary Table

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | Discussion recovery scores (0.51–0.57) not present in any results table | **SIGNIFICANT** | §6 Discussion |
| 2 | MMLU baseline 62.2% in §5.6.1 vs. 62.0% in §5.6 and §5.7 | Minor | §5.6, §5.6.1, §5.7 |
| 3 | §5.7 top-10 ablation list uses pre-correction patching run without labeling | Minor | §5.7 |
| 4 | L1H20 labeled "Rank 4" in §5.9 opinion circuit table, but is rank >10 in validated run | Minor | §5.9 |
| 5 | Cohen's h for opinion comparisons off by ~0.01 | Cosmetic | §5.1 |
| 6 | Per-source steering baseline (83.0%) vs. main baseline (82.4%) difference unexplained | Cosmetic | §5.8 |
