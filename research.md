# Peer Review Evidence Gathering — Mitigating Sycophancy (Egan, 2026)

**Prepared:** 2026-04-08  
**Scope:** Literature freshness, citation verification, unverifiable references, code-paper alignment, DPO evaluation design.

---

## 1. Literature Freshness Check

### Papers Found (2025–2026) That Are Missing From the Manuscript

| Paper | Venue / arXiv | Relevance | Priority |
|-------|--------------|-----------|----------|
| Vennemeyer et al. — "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs" | ICLR 2026 under review ([OpenReview](https://openreview.net/forum?id=d24zTCznJu)) | Directly parallels the paper's domain-specific-circuit finding using causal separation methods on LLMs. **This is the closest concurrent paper and is not cited.** | ⚠️ Critical |
| Cheng et al. — "Verbalizing LLMs' Assumptions to Explain and Control Sycophancy" | arXiv:2604.03058, Apr 2026 | Post-dates the submission (paper date April 7, 2026); absent citation is forgivable. | Low |
| Goldowsky-Dill et al. — "Detecting Strategic Deception with Linear Probes" | ICML 2025 ([link](https://icml.cc/virtual/2025/poster/46082)) | Extends linear probe methodology to safety-relevant behaviors. Relevant to §5.3–5.5 probe control discussion. | Moderate |
| "Truth as a Trajectory: What Internal Representations Reveal About Large Language Model Reasoning" | arXiv:2603.01326, Mar 2026 | Directly addresses the question of how truth is encoded in LLM internal representations across layers — germane to the social-compliance interpretation. | Moderate |
| Lagasse et al. — "A Few Bad Neurons…" (O'Brien et al. in manuscript) | arXiv:2601.18939 | **Is cited**, but with a possible author name error (see §2). | — |
| Anthropic — "Circuit Tracing: Revealing Computational Graphs in Language Models" | transformer-circuits.pub, 2025 | New circuit discovery methodology using attribution graphs; relevant to the activation patching § and its limitations. Not cited. | Moderate |
| Lad et al. — "The Remarkable Robustness of LLMs: Stages of Inference?" | MIT preprint 2024 | Demonstrates that deleting/swapping adjacent transformer layers barely affects behavior — directly relevant to the circuit redundancy / self-repair discussion (§5.7, Discussion). Not cited. | Low |

### Coverage Assessment

**Well-covered areas:**
- DPO mechanistic work: Lee et al. (2024, ICML) and Yang et al. (2025, EMNLP) both verified and correctly cited.
- Self-repair: McGrath et al. (2023) and Rushing & Nanda (2024, ICML) both verified and cited.
- Linear probing foundations: Burns et al. (2023) and Marks & Tegmark (2023) both cited; foundational coverage is solid.

**Gaps:**
- **Most significant gap**: "Sycophancy Is Not One Thing" (Vennemeyer et al., ICLR 2026) directly claims causal separation of sycophantic behaviors — exactly the domain-specific-circuit contribution the Egan paper claims in §5.9 and Conclusion point 3. A reviewer will likely raise this; the Egan paper should acknowledge it or distinguish from it.
- Anthropic's Circuit Tracing paper (2025) is now the canonical reference for circuit attribution in large models; citing it in the methods section would strengthen credibility.
- Goldowsky-Dill et al. (ICML 2025) is citable as a contemporary use of linear probes for safety-critical behaviors.

---

## 2. Verification of Six Cited Concurrent Papers

### 2.1 Chen et al. 2025 — arXiv:2409.01658

| Check | Status |
|-------|--------|
| Paper exists at arXiv:2409.01658 | ✅ Confirmed |
| Year correct | ✅ Submitted Sep 2024, v3 Feb 2025 — citing as 2025 is valid |
| Title | "From Yes-Men to Truth-Tellers: Addressing Sycophancy in LLMs with Pinpoint Tuning" |
| "Pinpoint tuning" characterization | ✅ Accurate — "supervised pinpoint tuning (SPT)" is the core contribution |
| "Path patching on Llama-2-Chat" | ⚠️ **Unverified from abstract.** The abstract uses "reveals and verifies" but does not explicitly name the method as "path patching" or Llama-2-Chat as the target model. The GitHub repo (linked in paper) could verify; the full text is needed to confirm "path patching (edge-level)" and the Llama-2-Chat claim in the manuscript's Discussion. |
| "Challenge-induced sycophancy" | ✅ Accurate — abstract explicitly states "When challenged by users, LLMs tend to admit mistakes" |
| "Substantial sycophancy reduction through head-level knockout" | ✅ Accurate — SPT fine-tunes <5% of modules while freezing the rest |

**Verdict:** Mostly accurate. The "path patching" and "Llama-2-Chat" specifics should be verified against the full paper. If inaccurate, the Discussion comparison (path patching vs. node-level patching as explanation for divergent ablation results) is undermined.

---

### 2.2 Li et al. 2025 — arXiv:2508.02087

| Check | Status |
|-------|--------|
| Paper exists at arXiv:2508.02087 | ✅ Confirmed |
| Year correct | ✅ Submitted Aug 2025, v4 Nov 2025 |
| Title | "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in LLMs" |
| Model is Llama-3.1-8B-Instruct | ⚠️ Abstract says "different model families" — specific model needs full-paper check |
| "Logit-lens late-layer sycophancy (16–23)" | ✅ Consistent with abstract: "late-layer output preference shift" confirmed |
| **Deeper characterization concern** | ⚠️ **The abstract says "structural override of learned knowledge in deeper layers" and "deeper representational divergence"** — this implies sycophancy involves *representation-level* changes in deep layers, consistent with the belief-corruption hypothesis. The Egan paper frames Li et al. as finding only late-layer *output crystallization*, but the abstract language suggests Li et al. also find representational corruption at depth. This tension is real: it potentially challenges the Egan paper's claim that social compliance (not belief corruption) is dominant, or at minimum requires more careful scoping of layer depth in the comparison. |

**Verdict:** Citation exists and year is correct. **The characterization is partially accurate but glosses over a substantive tension:** Li et al.'s "structural override" finding in deeper layers arguably supports partial belief corruption at late layers, while Egan's social compliance finding is strongest at early layers (layer 1). These may be complementary (early layers: social compliance; late layers: structural override) — the Egan paper hints at this in Discussion but does not fully engage with it.

---

### 2.3 O'Brien et al. 2026 — arXiv:2601.18939

| Check | Status |
|-------|--------|
| Paper exists at arXiv:2601.18939 | ✅ Confirmed |
| Year correct | ✅ Submitted Jan 26, 2026 |
| Title | "A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy" |
| Submitting author | ⚠️ **The arXiv submission shows "From: Ryan Lagasse [view email]" — NOT "O'Brien."** "O'Brien" does not appear in the submission history metadata. This is a **possible first-author attribution error**. Full author list from the PDF is needed to confirm whether any author named O'Brien is listed. If the first author is Lagasse (or another name), the citation should be "Lagasse et al. 2026." |
| SAE on Gemma-2, ~3% of MLP neurons | ✅ Accurate — abstract: "isolate the 3% of MLP neurons most predictive of a target behavior… using Gemma-2-2B and 9B models" |
| Gradient-masked fine-tuning | ✅ Accurate — abstract: "fine-tune only those neurons using gradient masking" |
| "Successful neuron-level correction" | ✅ Accurate — matches or exceeds SOTA on four benchmarks |

**Verdict:** The paper is real and accurately described, but the author name "O'Brien" may be incorrect. This is the most actionable citation error found. **Must verify author list from PDF.**

---

### 2.4 Heimersheim & Nanda 2024 — arXiv:2404.15255

| Check | Status |
|-------|--------|
| Paper exists at arXiv:2404.15255 | ✅ Confirmed |
| Year correct | ✅ Submitted Apr 23, 2024 |
| Authors | ✅ "Stefan Heimersheim" confirmed as submitting author; "Nanda" = Neel Nanda (Google DeepMind) |
| Title | "How to use and interpret activation patching" |
| "Sufficiency vs. necessity" formalization | ✅ Fully consistent — abstract: "what evidence patching experiments provide about circuits… choice of metric and associated pitfalls" |
| "Denoising = sufficiency, noising = necessity" | ✅ This is the central technical distinction in this paper |

**Verdict:** Fully verified. Characterization is accurate and well-used.

---

### 2.5 Lee et al. 2024 — arXiv:2401.01967 (ICML)

| Check | Status |
|-------|--------|
| Paper exists at arXiv:2401.01967 | ✅ Confirmed |
| Year correct | ✅ Submitted Jan 3, 2024; published ICML 2024 |
| Title | "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity" |
| ICML venue | ✅ Confirmed at proceedings.mlr.press/v235/lee24a.html |
| Model is GPT2-medium | ✅ Abstract confirms GPT2-medium |
| Characterization as "representation suppression" | ⚠️ **Potential inaccuracy.** Abstract says capabilities "are not removed, but rather *bypassed*." The Egan paper uses the term "representation suppression" to contrast with its own "output-gating elimination" finding, but Lee et al.'s actual language is "bypassed" — implying the toxic capability is preserved internally and merely routed around, not suppressed. This is a subtle but meaningful distinction. "Bypassing" is arguably closer to Egan's own "output-gating" framing than to "suppression." |

**Verdict:** Paper verified, year and venue correct. The contrast drawn in the paper ("representation suppression rather than output-gating elimination") may mischaracterize Lee et al. and inadvertently weaken the paper's own novelty claim, since "bypassing" is structurally similar to "output-gating."

---

### 2.6 Yang et al. 2025 — arXiv:2411.06424 (EMNLP)

| Check | Status |
|-------|--------|
| Paper exists at arXiv:2411.06424 | ✅ Confirmed |
| Year correct | ✅ arXiv Nov 2024; EMNLP 2025 — citing as 2025 is correct |
| Title | "How Does DPO Reduce Toxicity? A Mechanistic Neuron-Level Analysis" |
| EMNLP venue | ✅ Confirmed at aclanthology.org/2025.emnlp-main.1501.pdf |
| Authors (partial) | ✅ Yushi Yang, Filip Sondej, Harry Mayne, Andrew Lee, Adam Mahdi — Oxford/Harvard |
| Characterization as neuron-level analysis | ✅ Accurate |

**Verdict:** Fully verified. No issues.

---

## 3. Unverifiable References Investigation

### 3.1 Venhoff et al. 2025 — "Sparse Activation Fusion and Multi-Layer Activation Steering"

**Finding:** The paper **IS real** and **IS at the NeurIPS 2025 MI Workshop**, but the author attribution appears to be **incorrect**.

- **Located at:** https://openreview.net/forum?id=BCS7HHInC2
- **Actual title:** "Mitigating Sycophancy in Language Models via Sparse Activation Fusion and Multi-Layer Activation Steering"
- **Venue:** NeurIPS 2025 Mechanistic Interpretability Workshop (confirmed on mechinterpworkshop.com/neurips2025/posters-2/)
- **Content match:** Perfect match — SAF reduces sycophancy 63%→39%, MLAS reduces false admits 78%→0%, GitHub repo linked.
- **Author mismatch:** The OpenReview submission lists authors: **Pyae Phoo Min, Avigya Paudel, Naufal Adityo, Arthur Zhu, Andrew Rufail** — **no author named "Venhoff" appears.** The Egan paper's citation "Venhoff et al., 2025" does not match the actual author list.

**Possible explanation:** The submission was anonymous at the time the Egan paper was written (NeurIPS workshop papers often use anonymous submissions), and "Venhoff" may have been a placeholder or error. Alternatively, Venhoff is an author not shown in the OpenReview listing.

**Status:** ⚠️ **Citation content is real and correctly described; author name is likely wrong.** Recommend replacing "Venhoff et al." with "Pyae Phoo Min et al." (or the correct first-author name once confirmed from the OpenReview PDF).

---

### 3.2 Paduraru et al. 2025 — "Select-and-Project Top-K"

**Finding:** This reference **cannot be verified.**

- No arXiv paper matching the title "Select-and-Project Top-K" with author "Paduraru" was found.
- The closest arXiv paper is 2601.16651 "Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations" (Jan 2026) — but this is about training data explanations, not activation steering; and the author is not Paduraru.
- No conference proceedings or workshop paper matching this title/author pair was found in any search.

**Status:** 🚨 **This reference cannot be confirmed to exist.** It is either (a) a preprint that was withdrawn or not yet indexed, (b) a citation error (wrong author name), (c) a fabricated or hallucinated reference. **This must be resolved before submission.** The manuscript should either supply an arXiv ID / DOI, or remove the citation.

---

## 4. Code-Paper Alignment

### 4.1 Test Coverage Overview

| Test File | Coverage Target | Verdict |
|-----------|----------------|---------|
| `test_baseline.py` | `two_way_softmax`, CI computation | ✅ Covers core baseline evaluation logic |
| `test_probe_pipeline.py` | Label assignment, GroupKFold leakage-safe splits | ✅ Directly validates the neutral-transfer probe methodology |
| `test_data_setup_ids.py` | Deterministic `sample_id` for reproducibility | ✅ Validates content-based hashing, not position-based IDs |
| `test_evaluation_math.py` | Statistical utils, binomial test fallback | ✅ Good; tests both modern and legacy scipy API |
| `test_cli_contracts.py` | Flag presence in scripts | ✅ Guards `--randomize-positions`, `--analysis-mode`, `--checkpoint-resume` flags |
| `test_schema_contracts.py` | Schema fields in all result files | ✅ Enforces `schema_version`, `analysis_mode`, `split_definition` across scripts |
| `test_manifest_smoke.py` | `99_collect_result_manifest.py` | ✅ Integration smoke test |
| `test_steering_resume_smoke.py` | Checkpoint/resume idempotency | ✅ Strong test: asserts no re-computation on resume |

### 4.2 Script-to-Paper Mapping

The script list (00–07 + analysis helpers) maps cleanly onto the paper's SLURM job matrix (13 jobs). Key mappings:

| Paper SLURM Job | Script | Covered by Tests? |
|----------------|--------|------------------|
| Job 3: Probes (`neutral_transfer` + `mixed_diagnostic`) | `02_train_probes.py` + `02b_probe_control.py` | ✅ Pipeline + schema tests |
| Job 4: Activation patching | `03_activation_patching.py` | ⚠️ Schema contract only, no unit test for patching logic |
| Job 7/9/11: Head ablation | `04_head_ablation.py` | ⚠️ Schema contract only |
| Job 12: Steering | `05_representation_steering.py` | ✅ Full checkpoint/resume integration test |
| DPO training | `06_dpo_training.py` | ⚠️ No test at all |
| DPO eval | `07_dpo_eval.py` | ⚠️ No test at all |

### 4.3 Red Flags Assessment

**No hardcoded results found.** All test files import functions from the actual scripts and exercise them with synthetic inputs or mock data. The schema contract tests verify field names exist in script source code, not in result JSON files — an appropriate and lightweight approach.

**Potential code smell in `06_dpo_training.py`:**
```python
if seed == 42:
    print("WARNING: seed=42 is the benchmark test seed. ...")
    print("Proceeding anyway as requested...")
```
The script *warns* but *does not prevent* running with seed=42. This is a soft guardrail only.

**No test for train/eval disjointness** (see §5 below).

**`src/models/sycophancy_model.py`** is present but has no corresponding test file covering its behavior.

**Overall:** The test suite is well-structured for a research codebase and covers the most methodologically critical paths (probe label purity, leakage-safe splits, deterministic IDs, steering idempotency). The coverage gap in patching and ablation scripts is expected at this research stage but worth flagging.

---

## 5. DPO Evaluation Design Check

### 5.1 Train/Eval Disjointness

**Design (from `06_dpo_training.py` docstring and code):**
- Training pairs: `AnthropicOpinionDataset(seed=100)`, 400 pairs requested
- Benchmark evaluation: `master_sycophancy.jsonl`, 500 opinion samples built with seed=42
- The code explicitly documents the disjointness intent; the script warns (but does not block) if seed=42 is accidentally passed

**Actual enforcement:**
```python
# No intersection test between training pairs and evaluation dataset is implemented.
```

The disjointness relies entirely on the assumption that `AnthropicOpinionDataset(seed=100)` and `AnthropicOpinionDataset(seed=42)` produce non-overlapping samples. This is plausible if the dataset has sufficient size and the seed controls shuffling of a large pool, but **no code verifies it**. If the Anthropic dataset is small (<900 unique samples), there could be overlap between the 400 training and 500 evaluation pairs.

**Reviewer risk:** A reviewer may ask for an explicit overlap check or count of shared `sample_id`s between the DPO training pair file (`results/dpo_model/dpo_training_pairs.json`) and `master_sycophancy.jsonl`. This artifact exists in the repo — it is straightforward to verify post-hoc.

### 5.2 In-Distribution vs. Out-of-Distribution

**Assessment: The DPO evaluation is in-distribution.**

Both the DPO training pairs and the benchmark evaluation samples come from the same source: `anthropic_opinion` (Perez et al.'s model-written evaluations). The only difference is the random seed. This means the 23.8 pp sycophancy reduction is measured on the same type of opinion prompts used for training.

**Implications for the paper's claims:**
- The paper does not claim out-of-distribution generalization for the DPO result; it frames DPO as a demonstration that training-time intervention succeeds where inference-time fails. The in-distribution framing is appropriate for that claim.
- **However,** the paper does not test whether DPO generalizes to the `truthfulqa_factual` or `gsm8k_reasoning` domains. The near-zero factual/reasoning sycophancy (1.6%/0.0%) makes this hard to measure, but it means the DPO model's behavior on novel opinion-domain prompts (not from `anthropic_opinion`) is unknown.
- A reviewer may note this as a limitation of the generalization claim.

### 5.3 Effective Sample Size

| Claim | Numerics |
|-------|---------|
| Pre-DPO opinion sycophancy | 82.4% (N=500, ~412/500 sycophantic) |
| Post-DPO opinion sycophancy | 58.6% (N=500, ~293/500 sycophantic) |
| Two-proportion z-test | z ≈ 8.2, p << 0.001 |
| Wilson CI on 23.8 pp reduction | Highly significant |

N=500 is adequate for the effect size claimed. Statistical significance is not in doubt.

**Minor discrepancy:** The paper states "400 DPO training pairs" but `06_dpo_training.py` holds out 10% (`eval_split=0.1`) for DPO internal validation, yielding **360 actual training pairs and 40 held-out pairs.** The paper does not disclose this 90/10 split. The claim "400 DPO training pairs" refers to the generated pool, not the effective training set. This is a minor methodological detail worth clarifying in the text.

**Additional note:** `07_dpo_eval.py` evaluates the DPO model on the full 1,500-sample `master_sycophancy.jsonl` — the same dataset used for all other evaluations. This is correct methodology.

---

## 6. Summary of Actionable Findings

### 🚨 Must-Fix Before Submission
1. **"Paduraru et al. 2025" (Select-and-Project Top-K):** Reference cannot be verified. Requires an arXiv ID, DOI, or removal.
2. **"O'Brien et al. 2026" (arXiv:2601.18939):** The arXiv submitter is "Ryan Lagasse," not O'Brien. Verify the actual first author from the PDF and correct the citation.

### ⚠️ Should Address
3. **"Venhoff et al. 2025" author name:** The OpenReview submission lists Pyae Phoo Min et al. as authors — no "Venhoff" found. Correct the author attribution.
4. **Missing concurrent paper:** "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs" (Vennemeyer et al., ICLR 2026 under review) directly parallels the domain-specific-circuit finding. Should be cited and differentiated.
5. **Li et al. (2025) tension:** Li et al.'s abstract reports "structural override of learned knowledge in deeper layers" (suggesting belief corruption at depth) — this is partially in tension with the social compliance claim and should be more directly addressed in Discussion.
6. **DPO train/eval overlap:** Add an explicit overlap check between `dpo_training_pairs.json` and `master_sycophancy.jsonl`; report the result.
7. **400 vs. 360 DPO training pairs:** Clarify that 360 pairs are used for training and 40 are held out for DPO internal eval.

### 📝 Minor / Stylistic
8. **Lee et al. (2024) "bypassing" vs. "suppression":** The paper contrasts its finding with "representation suppression" in Lee et al., but Lee et al.'s own language is "bypassed" — structurally more similar to the Egan paper's "output-gating" framing. Adjust the contrast or acknowledge the similarity.
9. **Chen et al. (2025) model and method:** Verify "path patching" and "Llama-2-Chat" claims against the full paper (not just abstract).
10. **DPO in-distribution caveat:** Add a sentence in Limitations acknowledging that the 23.8 pp reduction is measured on same-distribution opinion prompts (`anthropic_opinion`, seed=42); out-of-distribution generalization is untested.

---

## Sources

**Kept:**
- arXiv:2409.01658 (Chen et al.) — verified paper
- arXiv:2508.02087 (Li et al.) — verified paper
- arXiv:2601.18939 (Lagasse et al., cited as O'Brien) — verified paper, author issue
- arXiv:2404.15255 (Heimersheim & Nanda) — verified paper
- arXiv:2401.01967 (Lee et al.) — verified paper, ICML confirmed
- arxiv:2411.06424 (Yang et al.) — verified paper, EMNLP confirmed
- openreview.net/forum?id=BCS7HHInC2 (NeurIPS 2025 MI Workshop) — confirmed real paper, author mismatch
- openreview.net/forum?id=d24zTCznJu (Vennemeyer et al. ICLR 2026) — important missing citation
- proceedings.mlr.press/v235/rushing24a.html (Rushing & Nanda 2024) — confirms citation
- arxiv:2603.01326 — truth tracking, moderately relevant new paper
- aclanthology.org/2025.emnlp-main.1501.pdf — confirms Yang et al.

**Dropped:**
- "Paduraru et al. 2025 Select-and-Project Top-K" — not found anywhere, likely fabricated or misattributed
- arXiv:2601.16651 "Select or Project?" — wrong topic (training data explanations)

---

## Gaps

- Full author list for arXiv:2601.18939 needs PDF inspection to confirm/deny "O'Brien"
- Full text of Chen et al. (arXiv:2409.01658) needed to verify "path patching" and "Llama-2-Chat" claims
- Whether `AnthropicOpinionDataset(seed=100)` and `AnthropicOpinionDataset(seed=42)` produce truly disjoint samples cannot be confirmed without running the data pipeline; requires checking dataset size and shuffle mechanics
- `src/models/sycophancy_model.py` contents were not inspected; may warrant a brief read for any patching-specific logic
