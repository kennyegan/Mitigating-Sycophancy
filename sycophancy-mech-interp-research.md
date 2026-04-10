# Peer Review Evidence — Mitigating Sycophancy (Egan, 2026)

**Prepared:** 2026-04-08
**Scope:** Numerical verification, citation verification, literature freshness, code-paper alignment, DPO evaluation design

---

## 1. Numerical Consistency Audit

All cross-checked. **No errors found.**

| Check | Computed | Paper | Status |
|-------|---------|-------|--------|
| Cohen's h (Opinion vs Reasoning) | 2.276 | 2.276 | ✅ |
| Cohen's h (Opinion vs Factual) | 2.022 | 2.022 | ✅ |
| Cohen's h (Reasoning vs Factual) | −0.254 | −0.254 | ✅ |
| DPO opinion delta | 82.4 − 58.6 = 23.8 pp | 23.8 pp | ✅ |
| DPO overall delta | 28.0 − 19.6 = 8.4 pp | 8.4 pp | ✅ |
| Probe decomposition sum (pre-DPO) | 59.9+18.0+10.1+12.1 = 100.1% | ~100% | ✅ (rounding) |
| Probe decomposition sum (post-DPO) | 75.5+11.4+8.3+4.8 = 100.0% | ~100% | ✅ |
| Robust tracking delta | 75.5−59.9 = 15.6 pp | +15.6 pp | ✅ |
| Social compliance delta | 11.4−18.0 = −6.6 pp | −6.6 pp | ✅ |
| Belief corruption delta | 8.3−10.1 = −1.8 pp | −1.8 pp | ✅ |
| Other delta | 4.8−12.1 = −7.3 pp | −7.3 pp | ✅ |
| Baseline 28.0% consistency | Appears in §5.1, §5.2, §5.6, §5.6.1, §5.7, §5.8, §5.10, §5.11, §9 | Consistent | ✅ |
| Opinion 82.4% consistency | Appears in §5.1, §5.6.1, §5.7, §5.8, §5.9, §5.10, §5.11, §6, §9 | Consistent | ✅ |

**DPO Cohen's h (not reported):** h = 0.532 for 82.4%→58.6%. This is a medium effect — worth reporting.

---

## 2. Table Formatting Issue

**§5.6 table is still broken.** The `[^mmlu]` footnote at line ~297 sits between the Baseline and L5H5 rows, breaking the markdown table:

```
| **Baseline** | **28.0%** | — | 62.0%[^mmlu] | 34.0% |
| L1H20 only | 27.9% | −0.1 | 62.8% | 29.5% |

[^mmlu]: The 0.2pp MMLU variation...
| L5H5 only | 28.5% | +0.5 | 61.6% | 31.5% |
```

The footnote definition mid-table causes rows after it to render outside the table structure. Must move the footnote after the table.

---

## 3. Citation Verification (6 concurrent papers)

### 3.1 Chen et al. 2025 — arXiv:2409.01658 ✅ VERIFIED
- Paper exists, year correct (submitted Sep 2024, v3 Feb 2025)
- **Uses path patching** on Mistral Instruct and Llama-2-Chat (Llama-2-13B Chat) — confirmed via alpha_ask_paper
- "Challenge-induced sycophancy" characterization is accurate
- Paper's description is accurate

### 3.2 Li et al. 2025 — arXiv:2508.02087 ✅ VERIFIED (with caveat)
- Paper exists, year correct (submitted Aug 2025)
- Title: "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in LLMs"
- Late-layer finding confirmed
- **Caveat:** Li et al.'s abstract mentions "structural override of learned knowledge in deeper layers" — this implies representational changes (partial belief corruption) at depth, not just output crystallization. The Egan paper frames Li et al. as finding only late-layer output crystallization. The tension is real but manageable.

### 3.3 O'Brien et al. 2026 — arXiv:2601.18939 ✅ VERIFIED
- Paper exists, year correct (Jan 26, 2026)
- **First author IS Claire O'Brien** (Algoverse). Ryan Lagasse was the arXiv submitter (5th author).
- Full author list: Claire O'Brien, Jessica Seto, Dristi Roy, Aditya Dwivedi, Sunishchal Dev, Kevin Zhu, Sean O'Brien, Ashwinee Panda, Ryan Lagasse
- SAE + gradient-masked fine-tuning on Gemma-2-2B and 9B — accurately described
- ~3% of MLP neurons — confirmed

### 3.4 Heimersheim & Nanda 2024 — arXiv:2404.15255 ✅ VERIFIED
- Fully verified. Sufficiency/necessity formalization accurately cited.

### 3.5 Lee et al. 2024 — arXiv:2401.01967 ✅ VERIFIED (with caveat)
- Paper exists, ICML 2024 confirmed
- Title: "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity"
- **Caveat:** Lee et al.'s abstract says toxic capabilities are "not removed, but rather *bypassed*." The Egan paper contrasts its finding with "representation suppression" — but "bypassed" is structurally closer to "output-gating" (Egan's own framing) than to "suppression." This weakens the claimed qualitative difference.

### 3.6 Yang et al. 2025 — arXiv:2411.06424 ✅ VERIFIED
- EMNLP 2025, fully verified. No issues.

---

## 4. Unverifiable References

### 4.1 Venhoff et al. 2025 — ⚠️ AUTHOR NAME LIKELY WRONG
- The paper IS real: "Mitigating Sycophancy in Language Models via Sparse Activation Fusion and Multi-Layer Activation Steering"
- Located at: https://openreview.net/forum?id=BCS7HHInC2
- NeurIPS 2025 MI Workshop — confirmed
- **But actual authors are: Pyae Phoo Min, Avigya Paudel, Naufal Adityo, Arthur Zhu, Andrew Rufail** — no "Venhoff" found
- Content description in the paper is accurate

### 4.2 Paduraru et al. 2025 — 🚨 CANNOT BE VERIFIED
- No paper matching "Select-and-Project Top-K" with author "Paduraru" found anywhere
- No arXiv ID provided in references.bib
- **Must be resolved: provide arXiv ID/DOI, or remove the citation**

---

## 5. Missing Literature

### 5.1 Vennemeyer et al. — "Sycophancy Is Not One Thing" (ICLR 2026 under review)
- OpenReview: https://openreview.net/forum?id=d24zTCznJu
- **Directly parallels** §5.9's domain-specific-circuit finding using causal separation methods
- This is the closest concurrent paper and is NOT cited
- A reviewer will likely raise this

### 5.2 Other potentially relevant papers
- Goldowsky-Dill et al. — "Detecting Strategic Deception with Linear Probes" (ICML 2025) — extends probe methodology to safety behaviors
- Anthropic — "Circuit Tracing: Revealing Computational Graphs in Language Models" (2025) — new circuit discovery method
- Both are lower priority but worth noting

---

## 6. references.bib Issues

### 6.1 Placeholder author fields
Multiple entries use `author={Li, others}`, `author={O'Brien, others}`, `author={Lee, others}`, `author={Yang, others}`, `author={Venhoff, others}`, `author={Paduraru, others}`. These must be filled with complete author lists.

### 6.2 Missing arXiv IDs / DOIs
- `paduraru2025sp`: `journal={arXiv preprint}` with NO arXiv number
- `venhoff2025saf`: `journal={NeurIPS 2025 Mechanistic Interpretability Workshop}` with no proceedings link
- `lee2024mechanistic`: Missing arXiv ID (should be 2401.01967)
- `yang2025dpo`: Missing arXiv ID (should be 2411.06424)

---

## 7. Code-Paper Alignment

### 7.1 Test Coverage
| Test File | Coverage | Status |
|-----------|----------|--------|
| `test_baseline.py` | `two_way_softmax`, CIs | ✅ Core eval logic |
| `test_probe_pipeline.py` | Labels, GroupKFold leakage | ✅ Critical path |
| `test_data_setup_ids.py` | Deterministic sample_id | ✅ |
| `test_evaluation_math.py` | Stats utils, binomial test | ✅ |
| `test_cli_contracts.py` | Flag presence in scripts | ✅ |
| `test_schema_contracts.py` | Schema fields | ✅ |
| `test_manifest_smoke.py` | Manifest collection | ✅ |
| `test_steering_resume_smoke.py` | Checkpoint/resume | ✅ Strong test |

### 7.2 Coverage Gaps
- **No unit test for activation patching logic** — schema contract only
- **No unit test for head ablation logic** — schema contract only
- **No test for DPO training (`06_dpo_training.py`)** or eval (`07_dpo_eval.py`)
- **No test for train/eval disjointness** between DPO pairs and benchmark

### 7.3 No Red Flags
- No hardcoded results found
- Tests exercise actual functions with synthetic inputs
- DPO script warns (but doesn't block) if seed=42 is used

---

## 8. DPO Evaluation Design

### 8.1 Train/eval disjointness
- Training: `AnthropicOpinionDataset(seed=100)`, 400 pairs generated
- Evaluation: `master_sycophancy.jsonl`, 500 opinion samples (seed=42)
- **No explicit intersection check in code** — disjointness relies on seed difference
- Plausible if dataset pool is large (>900 unique samples), but unverified

### 8.2 In-distribution evaluation
- Both training and evaluation come from the same source: `anthropic_opinion` (Perez et al.)
- The 23.8 pp reduction is measured in-distribution
- Paper does not claim OOD generalization — framing is appropriate
- **But:** Discussion makes general claims ("training-time preference optimization is the appropriate intervention level") from single-model, single-domain, in-distribution evidence

### 8.3 400 vs 360 training pairs
- `06_dpo_training.py` holds out 10% (`eval_split=0.1`) for DPO internal validation
- Effective training set: ~360 pairs, not 400
- Paper says "400 DPO training pairs" — refers to generated pool, not effective training set
- Minor but should be disclosed

### 8.4 Sample size adequacy
- N=500 for 82.4%→58.6% (delta=23.8 pp) gives z≈8.2, p<<0.001
- Statistical significance is not in question

---

## 9. Summary of Actionable Findings

### 🚨 Must-Fix
1. **Paduraru et al. 2025**: Cannot be verified. Provide arXiv ID or remove.
2. **Venhoff et al. 2025 author name**: Actual authors are Pyae Phoo Min et al., not Venhoff.
3. **Placeholder author fields in references.bib**: Multiple entries need complete author lists.

### ⚠️ Should Address
4. **Missing: Vennemeyer et al. ICLR 2026** — closest concurrent paper to domain-specific circuit finding
5. **§5.6 table formatting**: Footnote breaks table rendering
6. **Lee et al. "bypassed" vs "suppression"**: Mischaracterization weakens novelty contrast
7. **Li et al. "structural override" tension**: Social compliance claim needs more careful scoping vs. deep-layer findings
8. **400 vs 360 DPO pairs**: Disclose eval split
9. **DPO generalization scope**: Bound the general claim to in-distribution evidence

### ✅ Clean
- All numerical values are internally consistent
- Cohen's h calculations are correct
- DPO arithmetic is correct
- Probe decompositions sum to ~100%
- Code structure matches claimed methodology
- Test suite covers critical paths (probes, steering, baselines)
- O'Brien et al. citation is correct (first author IS Claire O'Brien)
- Chen et al. uses path patching on Llama-2-Chat — confirmed

---

## Sources

| Reference | URL | Status |
|-----------|-----|--------|
| Chen et al. 2025 | https://arxiv.org/abs/2409.01658 | ✅ Verified |
| Li et al. 2025 | https://arxiv.org/abs/2508.02087 | ✅ Verified |
| O'Brien et al. 2026 | https://arxiv.org/abs/2601.18939 | ✅ Verified (author confirmed) |
| Heimersheim & Nanda 2024 | https://arxiv.org/abs/2404.15255 | ✅ Verified |
| Lee et al. 2024 (ICML) | https://arxiv.org/abs/2401.01967 | ✅ Verified |
| Yang et al. 2025 (EMNLP) | https://arxiv.org/abs/2411.06424 | ✅ Verified |
| Venhoff/Min et al. 2025 (NeurIPS MI Workshop) | https://openreview.net/forum?id=BCS7HHInC2 | ⚠️ Wrong author name |
| Paduraru et al. 2025 | — | 🚨 Not found |
| Vennemeyer et al. 2026 (ICLR under review) | https://openreview.net/forum?id=d24zTCznJu | ⚠️ Missing citation |
| Panickssery et al. 2023 | https://arxiv.org/abs/2312.06681 | ✅ |
| McGrath et al. 2023 | https://arxiv.org/abs/2307.15771 | ✅ |
| Rushing & Nanda 2024 (ICML) | https://proceedings.mlr.press/v235/rushing24a.html | ✅ |
