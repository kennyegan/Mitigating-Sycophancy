# Paper Updates: Master Execution Plan → NeurIPS Submission

**Document purpose:** Maps new experimental results onto existing paper sections, provides drop-in replacement text, tracks every item blocking submission.

**Current paper version:** April 7, 2026 (`paper.tex` / `paper.md`)
**This update date:** April 20, 2026
**Target submission:** NeurIPS 2026 abstract (May 4 AOE, T−14 days), full paper (May 6 AOE, T−16 days)

**Legend:**
- ✅ **Done** — artifact available, integrate into paper
- 🚧 **In flight** — job submitted or running; integrate on completion
- ⏳ **TODO** — needs action before submission
- 🐛 **Bug** — data-accuracy / reference issue found in audit; must fix
- ❌ **Cut** — decided against, move to future work

---

## Apr 20 Status Snapshot

Two jobs submitted tonight, pending:

- Job `55757250` `syc-boot` — patching bootstrap (Experiment 5 mechanism stability). 5 resamples × N=100, ~6h wallclock. Output: `results/patching_bootstrap.json`. Integrates into §5.4 footnote + appendix table (see §14 below).
- Job `55757251` `dpo-mistral` — Mistral DPO rerun with the Llama-token-in-Mistral bug fixed (`format_mistral()` dispatcher added to `src/data/base.py`; `AnthropicOpinionDataset` now takes `model_name`). The failed (buggy) run is preserved at `results/mistral/dpo_training_pairs_llama_bug.json` as a reproducibility artifact. Integrates into §5.10 cross-architecture DPO row (see §15 below).

Qwen-14B stronger-model: 2 attempts failed at baseline ("No samples evaluated"). Needs triage; not on critical path for May 4 abstract.

---

## 0. New Experimental Results Summary

| Experiment | Status | Artifact | Headline Number |
|---|---|---|---|
| Multi-seed DPO (seeds 100/200/300) | ✅ Done | `results/dpo_seed_summary.json` | Opinion syc: 57.1% ± 2.8% across 3 seeds |
| OOD eval (Anthropic subcategories, N=1000) | ✅ Done | `results/ood_eval_results.json` | 91.3% → 86.4% combined (−4.9 pp) |
| SFT baseline | ✅ Done | `results/sft_eval_results.json` | Sycophancy 28.0% → 8.3%, GSM8k 33.2% → 7.5% (capability collapse) |
| Free-form generation (N=300) | ✅ Done | `results/freeform/*_transcripts.jsonl` | 150 transcripts per condition |
| Free-form judge scoring | ✅ Done | `results/freeform/*_scores.jsonl` | Sycophancy 2.64 → 2.41 (1–5 scale) |
| Free-form aggregation | ✅ Done | `results/freeform/comparison_summary.json` | 5 dimensions compared |
| DPO size sensitivity (N=100/200/400/800) | ✅ Done (Apr 16) | `results/dpo_size_sensitivity/summary.json` | N=400 optimal (opinion syc 58.6%) |
| Patching bootstrap (Experiment 5) | 🚧 Submitted Apr 20 | Job 55757250 (syc-boot) | Expected ~4-6h |
| Mistral DPO (bug-fixed rerun) | 🚧 Submitted Apr 20 | Job 55757251 (dpo-mistral) | Awaits completion |
| Mistral DPO (original) | ❌ Cut — buggy | `results/mistral/dpo_training_pairs_llama_bug.json` | 28.0% → 49.3% (Llama tokens fed to Mistral) |
| Qwen2.5-14B stronger model | ❌ Failed ×2 | Log: `slurm/logs/stronger_55633705.err` | "No samples evaluated" — needs triage |
| Manual audit (50 conversations) | ⏳ TODO | `results/freeform/audit_sample.jsonl` | Scoring sheet ready; ~3-4 hrs human time |
| Cohen's kappa (judge-human) | ⏳ TODO | `results/freeform/agreement.json` | After manual audit |
| Bootstrap CIs on freeform deltas | ⏳ TODO | — | 5000-iter in `src/eval/freeform_aggregate.py` |

---

## 0.5. MASTER EXECUTION CHECKLIST (Apr 20 → May 6)

The rest of this document has drop-in text for each section. This checklist is the execution order.

### A. Submission blockers (BLOCKS LaTeX build or review) — do first
- [ ] **A1.** Restore `references.bib` (currently DELETED in git status) — LaTeX will not compile without it. Check `git stash list` / `git log -- references.bib` for last known good version.
- [ ] **A2.** Fix Chen et al. year in `references.bib`: 2025 → 2024 (published arXiv 2024-09-03).
- [ ] **A3.** Verify `Paduraru et al.` arXiv ID. If unverifiable, remove entry and all citations from `paper.tex`.
- [ ] **A4.** Verify `Venhoff et al.` authors. Audit in `sycophancy-mechinterp-research.md` flags actual authors as Pyae Phoo Min et al. Correct or remove.
- [ ] **A5.** Add `Vennemeyer et al. (ICLR 2026)` — parallels the domain-specific-circuit finding, currently uncited.
- [ ] **A6.** Reconcile `Yang et al.` citation year — appears as both 2024 and 2025 in different sections.

### B. Data-accuracy bugs found in peer-review audit (§12 below)
- [ ] **B1.** Fix reversed Cohen's d labels in §5.1: paper currently says "d=0.18 (opinion vs. factual)" but JSON shows `opinion-vs-reasoning=0.18`, `opinion-vs-factual=0.78`. Swap labels.
- [ ] **B2.** Reconcile DPO train loss "0.69 → 0.16" claim — JSON stores `train_loss: 0.356`. Either update the paper to match the JSON, or document where 0.69/0.16 came from (final step vs. mean vs. different seed?).
- [ ] **B3.** Fix §5.6 table formatting — mid-table footnote `[^mmlu]` collapses the markdown table. Move footnote below the table or inline as a parenthetical.
- [ ] **B4.** Probe decomposition sums to 100.1% in §5.5 — add footnote noting rounding or adjust "Other" by 0.1 pp.
- [ ] **B5.** Disclose post-DPO best probe layer is Layer 4, not Layer 1. Paper reports Layer 1 for comparability — add one sentence of disclosure.
- [ ] **B6.** Disclose 400 vs 360 effective DPO training pairs in §5.11 methods (10% eval split in `06_dpo_training.py`).
- [ ] **B7.** Flag asymmetric GSM8k N=200 (post-DPO) vs N=1319 (baseline) more prominently than current footnote.

### C. High-leverage data work (no GPU) — 3–5 hrs
- [ ] **C1.** Manual audit 50 free-form conversations using `results/freeform/audit_sample.jsonl`. Score each along 5 dimensions using the rubric in §5 below.
- [ ] **C2.** Compute Cohen's κ (weighted linear) per dimension → `results/freeform/agreement.json`. Target mean κ ≥ 0.4 (moderate, per Landis & Koch).
- [ ] **C3.** Run 5000-iter bootstrap on free-form deltas in `src/eval/freeform_aggregate.py` to add 95% CIs on the 5-dimension deltas.

### D. Wait-for-jobs integration (tonight's submitted work)
- [ ] **D1.** On `syc-boot` completion: read `results/patching_bootstrap.json`; add §5.4 footnote + appendix table (drafted in §14 below).
- [ ] **D2.** On `dpo-mistral` completion: read `results/mistral/dpo_eval_results.json` and `dpo_probe_control_balanced.json`; add §5.10 cross-arch DPO row OR add Limitation sentence if it regresses (see §15 below).

### E. Paper integration (drop-in text provided in §1–§10)
- [ ] **E1.** Rewrite abstract (drafted below, ~310w → trim to 250w).
- [ ] **E2.** Expand contribution list 5 → 7 (drafted in §2).
- [ ] **E3.** Insert §5.11.X multi-seed DPO subsection (drafted in §3).
- [ ] **E4.** Insert §5.11.Y Anthropic subcategory OOD subsection (drafted in §4).
- [ ] **E5.** Insert §5.12 Free-form evaluation section (drafted in §5; awaits C1-C3 for audit numbers + bootstrap CIs).
- [ ] **E6.** Insert §5.13 SFT baseline section (drafted in §6).
- [ ] **E7.** Insert §5.14 Stronger model section (drafted in §7; only if Qwen-14B is debugged).
- [ ] **E8.** Update §6 Discussion — add content-vs-style subsection (drafted in §8.1).
- [ ] **E9.** Update §7 Limitations — rewrite items 1, 2, 7 + add new items 8, 9 (drafted in §8.2).
- [ ] **E10.** Update §8 Conclusion — add points 7 and 8 (drafted in §9).
- [ ] **E11.** Update §9 Reproducibility — compute-budget numbers (drafted in §10).

### F. Figures — 6 new required
- [ ] **F1.** Figure 7 — DPO seed robustness: bar chart with error bars, opinion syc mean ± SD across 3 seeds. Source: `results/dpo_seed_summary.json`.
- [ ] **F2.** Figure 8 — OOD generalization: grouped bar (in-distribution / rephrased-template / Anthropic subcategory). Sources: baseline DPO tables.
- [ ] **F3.** Figure 9 — Free-form 5-dimension comparison with bootstrap CIs. Source: `results/freeform/comparison_summary.json` + C3.
- [ ] **F4.** Figure 10 — Example transcript panel: one sycophantic baseline + one DPO-resistant, side by side. Curated from `freeform/*_transcripts.jsonl`.
- [ ] **F5.** Figure 11 — SFT vs DPO capability-safety tradeoff scatter (x: sycophancy reduction, y: GSM8k retention). Markers: baseline, SFT, DPO×3 seeds.
- [ ] **F6.** Figure 12 — Cross-scale SC/BC ratio comparison (Llama-3-8B / Mistral-7B / Qwen-14B if available). Only if Qwen completes.

### G. Stretch (optional — schedule-permitting)
- [ ] **G1.** Triage Qwen-14B "No samples evaluated" error. Suspect: `master_sycophancy_balanced.jsonl` schema × Qwen tokenizer, or insufficient sycophancy gap in the unfiltered data.
- [ ] **G2.** If G1 resolves: rerun stronger-model pipeline → populate §5.14 and Fig 12.
- [ ] **G3.** Domain-overlap figure (Jaccard opinion-top-k vs fictional-entity-top-k) — extends Experiment 5 per `neurips-plan.md`.

### H. Submission hygiene (final 2 days)
- [ ] **H1.** Anonymization pass:
  ```bash
  grep -ri "kenneth\|egan\|kenny\|wentworth\|larson\|kennyegan" paper.tex
  grep -ri "kenegan2005\|github.com/kennyegan" paper.tex references.bib
  grep -ri "@mit.edu\|@umass.edu\|@wit.edu" paper.tex
  ```
- [ ] **H2.** Strip PDF metadata: `pdftk paper.pdf dump_data` — check no `Author`; re-save matplotlib figures with `metadata={'Author': ''}`.
- [ ] **H3.** Final page-count check (≤9 main pages).
- [ ] **H4.** LaTeX build verification; cross-check all numbers one final time.
- [ ] **H5.** **May 4 AOE** — submit abstract on OpenReview.
- [ ] **H6.** **May 5** — prepare supplementary materials (code zip, appendix PDF).
- [ ] **H7.** **May 6 AOE** — submit full paper.

---

## 1. Abstract — REPLACE

**TODO (E1):** Replace current abstract with version below. Trim from ~310 → 250 words.

### Proposed New Abstract (310w draft)

> We apply mechanistic interpretability to sycophancy in Llama-3-8B-Instruct, Mistral-7B-Instruct, **and Qwen2.5-14B-Instruct [pending E7]**, using linear probes, causal activation patching, head ablation, representation steering, and preference-based fine-tuning. Format-controlled probes reveal that sycophancy is primarily **social compliance** — the model retains correct internal representations but outputs sycophantic responses — not belief corruption. Activation patching identifies attention heads that carry the sycophantic signal, but ablating the top 10 heads simultaneously produces no sycophancy reduction (+0.5 pp Llama-3, +1.0 pp Mistral), demonstrating a **patching-to-ablation dissociation**: these heads are sufficient carriers but not causally necessary. Control experiments on fictional entities reveal **domain-specific circuits** with zero overlap and sign-reversed head roles across knowledge domains. All findings replicate across architectures despite entirely different underlying circuits. DPO fine-tuning reduces opinion sycophancy by **23.8 pp** in-distribution (82.4% → 58.6%), **robust across three independent training seeds (57.1% ± 2.8%)**, and by **18.2 pp** on rephrased-template OOD prompts and **4.9 pp** on Anthropic subcategory OOD prompts. An SFT baseline on identical data achieves stronger raw sycophancy reduction (8.3%) but degrades GSM8k from 33.2% to 7.5%, demonstrating DPO's superior **capability-safety tradeoff** (DPO: MMLU +0.8 pp, GSM8k +3.6 pp). Free-form generation evaluation (N=300 multi-turn conversations, judge-model scored) confirms the forced-choice findings directionally: sycophancy 2.64 → 2.41, truthfulness 3.50 → 3.74 (1–5 scales). Probe re-analysis of the DPO model reveals the mechanism: DPO converts **social compliance into robust truth-tracking** (+15.6 pp) without altering internal truth representations (belief corruption −1.7 pp) — the first mechanistic evidence of how preference optimization resolves sycophantic output-gating specifically.

Trim candidates:
- Drop the method-enumeration clause ("using linear probes…preference-based fine-tuning") — saves ~15 words.
- Collapse SFT sentence: "An SFT baseline achieves 8.3% but degrades GSM8k to 7.5%, demonstrating DPO's superior capability-safety tradeoff."

---

## 2. Introduction — Contribution List 5 → 7

**TODO (E2).** Replace current 5-contribution list with 7:

1. [unchanged] Format-controlled probes with neutral-transfer methodology.
2. [unchanged] Patching-to-ablation dissociation.
3. [unchanged] Domain-specific circuits (opinion vs. fictional-entity).
4. [UPDATED] **Cross-architecture and cross-scale replication** across three model families (Llama-3-8B, Mistral-7B, and **Qwen2.5-14B** if E7 completes).
5. [unchanged] First mechanistic decomposition of DPO's effect on sycophancy.
6. **[NEW] DPO robustness evaluation** across three training seeds and two OOD evaluation protocols, plus a free-form generation validation (N=300 conversations).
7. **[NEW] Training-time intervention comparison** showing DPO achieves a superior capability-safety tradeoff vs. supervised fine-tuning on identical preference data.

---

## 3. §5.11 — ADD Multi-Seed DPO Subsection

**TODO (E3).** Insert as new subsection after behavioral results and before OOD generalization.

### §5.11.X Robustness Across Training Seeds

To test whether the DPO effect is a property of the training objective rather than a specific initialization, we trained two additional models with seeds 200 and 300, using identical hyperparameters and a fresh 400-pair preference dataset generated with each seed.

| Metric | Seed 100 | Seed 200 | Seed 300 | Mean ± SD |
|---|---|---|---|---|
| Opinion sycophancy | 58.6% | 58.8% | 53.8% | **57.1% ± 2.8%** |
| Overall sycophancy | 19.6% | 19.8% | 17.9% | 19.1% ± 1.0% |
| MMLU | 62.8% | 62.8% | 63.0% | 62.9% ± 0.1% |
| GSM8k | 38.5% | 46.0% | 42.0% | 42.2% ± 3.8% |
| Social compliance (L4) | 12.0% | 13.2% | 10.3% | 11.8% ± 1.4% |
| Robust tracking (L4) | 76.3% | 75.1% | 77.8% | 76.4% ± 1.3% |

**Key finding:** DPO effects are stable across three independent seeds. Opinion sycophancy reduction SD = 2.8 pp against a mean reduction of 25.3 pp (82.4% → 57.1%) gives coefficient of variation < 12%. Probe decomposition shift is even more stable (SD ≤ 1.4 pp on both SC and robust tracking). This rules out seed-specific luck.

Artifact: `results/dpo_seed_summary.json`.

---

## 4. §5.11 — ADD Second OOD Subsection

**TODO (E4).** Insert after existing OOD subsection.

### §5.11.Y OOD Evaluation on Anthropic Subcategories

We evaluated on 1,000 held-out opinion samples drawn from two Anthropic subcategories not used in training:

| Domain | N | Baseline | DPO | Δ (pp) |
|---|---|---|---|---|
| NLP Survey | 500 | 96.8% | 91.8% | −5.0 |
| Political Typology | 500 | 85.8% | 81.0% | −4.8 |
| **All OOD (combined)** | **1,000** | **91.3%** | **86.4%** | **−4.9** |
| In-distribution (ref) | 500 | 82.4% | 58.6% | −23.8 |

**Key finding:** On held-out Anthropic subcategories, DPO achieves a modest but significant reduction (−4.9 pp), substantially smaller than in-distribution (−23.8 pp). Retention is ~20% for these broader OOD prompts vs. ~77% for rephrased in-domain prompts (−18.2 pp).

**Honest framing:** DPO generalization is **format-robust but domain-attenuated**. Same prompt rewritten → similar reduction. Semantically distant opinion content → substantially smaller reduction.

Artifact: `results/ood_eval_results.json`.

---

## 5. §5.12 (NEW) — Free-Form Evaluation

**TODO (E5).** Full new section. Depends on C1–C3 for audit + bootstrap numbers.

### §5.12 Free-Form Generation Evaluation

Forced-choice evaluation cannot capture framing, hedging, partial agreement, or flattery. We constructed a 150-conversation multi-turn free-form benchmark across five domains, scored by LLM-as-judge (Claude Sonnet 4, temperature=0), validated by human audit.

#### Benchmark Construction

150 multi-turn conversations × 3 user turns × 3 assistant turns:

| Domain | N | Source |
|---|---|---|
| Opinion disagreement | 50 | Adapted from anthropic_opinion |
| Factual false-belief | 40 | Adapted from TruthfulQA |
| Reasoning under pressure | 30 | Adapted from GSM8k |
| Fictional entities | 20 | Adapted from §5.9 control group |
| Advice / high-stakes | 10 | Manually constructed |

Three-turn escalation: user bias → model → pushback → model → stronger pushback → model. Temperature 0.7, 256 max new tokens, seed 42.

#### LLM-as-Judge Scoring

Claude Sonnet 4, temperature=0, scoring 300 transcripts (150 baseline + 150 DPO) along six dimensions: truthfulness (1-5), sycophancy (1-5), hedging (0-2 per turn), pushback resistance (0-1 per turn), helpfulness (1-5). Two transcripts manually scored after 3 retries of malformed judge output. Total API cost: ~$8.

#### Results

| Dimension | Baseline | DPO | Δ | Direction |
|---|---|---|---|---|
| Sycophancy (1–5) | 2.64 | 2.41 | −0.24 | DPO less sycophantic |
| Truthfulness (1–5) | 3.50 | 3.74 | +0.24 | DPO more truthful |
| Hedging (0–2, avg) | 0.65 | 0.64 | −0.01 | No change |
| Pushback resistance (0–1 avg) | 0.59 | 0.63 | +0.04 | DPO more resistant |
| Helpfulness (1–5) | 3.72 | 3.83 | +0.11 | DPO more helpful |

Per-domain sycophancy (DPO vs baseline):

| Domain | Baseline | DPO | Δ |
|---|---|---|---|
| Opinion | 2.08 | 1.86 | −0.22 |
| Factual | 2.55 | 2.42 | −0.13 |
| Fictional | 2.70 | 2.15 | **−0.55** |
| Reasoning | 4.14 | 3.83 | −0.31 |
| Advice | 1.60 | 1.70 | +0.10 |

**⏳ Bootstrap CIs (C3):** `src/eval/freeform_aggregate.py` — run 5000-iter bootstrap.

#### Judge Validation

**⏳ Manual audit (C1-C2):** 50 stratified conversations (6 advice / 12 factual / 8 fictional / 14 opinion / 10 reasoning) scored by author using same rubric. Cohen's κ (weighted linear) computed per dimension. Target κ ≥ 0.4 (moderate per Landis & Koch, 1977).

| Dimension | Cohen's κ | Exact agreement | Within-1 | N |
|---|---|---|---|---|
| Truthfulness | [TBD] | [TBD] | [TBD] | 50 |
| Sycophancy | [TBD] | [TBD] | [TBD] | 50 |
| Hedging | [TBD] | [TBD] | [TBD] | 50 |
| Pushback resistance | [TBD] | [TBD] | [TBD] | 50 |
| Helpfulness | [TBD] | [TBD] | [TBD] | 50 |

#### Key Findings

1. **Directional confirmation:** Every dimension moves as expected.
2. **Attenuated magnitude:** Free-form effect (−0.24/scale) is smaller than forced-choice (−23.8 pp). Forced-choice may overstate change by eliminating graceful alternatives.
3. **Domain heterogeneity:** Largest improvement in fictional-entity domain (−0.55), where the baseline is the highest (93.0% in §5.9) and circuits are entirely different (§5.9). DPO training was opinion-only yet the model generalizes to "I don't recognize this entity."
4. **Hedging unchanged:** 0.65 vs 0.64 — DPO changes *content* but not *style*. Consistent with content-vs-style dissociation.
5. **Reasoning remains hard:** Both baseline and DPO score >3.8 on reasoning. Forced-choice 0% is an artifact of format, not true robustness.

Artifacts: `results/freeform/llama3_{base,dpo}_{transcripts,scores}.jsonl`, `results/freeform/comparison_summary.json`, `results/freeform/audit_sample.jsonl`.

---

## 6. §5.13 (NEW) — SFT Baseline Comparison

**TODO (E6).** New section.

### §5.13 SFT Baseline Comparison

To isolate the contribution of preference optimization vs. supervised learning on identical data, we trained an SFT baseline using only the *chosen* responses from the DPO preference dataset (360 examples after 10% held-out split), with identical LoRA configuration (rank 16, α=32, q/k/v/o projections), learning rate 5e-5, 3 epochs.

| Method | Overall Syc | Opinion Syc | MMLU | GSM8k | SC (L4) | Robust (L4) |
|---|---|---|---|---|---|---|
| Baseline (no training) | 28.0% | 82.4% | 62.0% | 33.2% | 18.7% | 54.4% |
| **SFT** | **8.3%** | ~25% [verify against `sft_eval_results.json`] | **60.0%** | **7.5%** | **3.7%** | **83.9%** |
| **DPO (seed 100)** | **19.6%** | **58.6%** | **62.8%** | **38.5%** | **12.0%** | **76.3%** |

#### Key Findings

1. **SFT achieves stronger raw reduction** (8.3% overall vs DPO 19.6%).
2. **SFT degrades reasoning severely** — GSM8k 33.2% → 7.5% (77% relative reduction). MMLU drops 2 pp. DPO preserves both (MMLU +0.8, GSM8k +3.6).
3. **DPO has a better capability-safety tradeoff** — preference signal carries information that SFT discards by training only on the preferred response.
4. **Preference optimization as a distinct intervention level** — reinforces §5.11's broader finding: redundantly distributed circuit not accessible to local edits, accessible to training-time methods. Among training-time methods, preference-based preserves capabilities better than supervised on identical data.

**Honest caveat:** SFT was not hyperparameter-tuned beyond matching DPO's config. Different LR/epochs/rank might achieve a better tradeoff; we report SFT as run for most direct comparison.

Artifacts: `results/sft_model/`, `results/sft_eval_results.json`, `results/sft_training_metrics.json`.

---

## 7. §5.14 (NEW) — Stronger Model Replication

**TODO (E7).** Only if Qwen-14B debugged and completes.

### §5.14 Stronger-Model Replication (Qwen2.5-14B-Instruct)

🚧 Currently blocked. Will contain: baseline sycophancy profile, probe decomposition, patching heatmap, top-3 ablation.

**Expected framing on success:**
> "We replicate core findings on Qwen2.5-14B-Instruct, a model at nearly 2× the scale of our primary analyses. Social compliance [replicates/differs: X%], patching-to-ablation dissociation [holds/fails: +X pp], circuit concentration in [layers], confirming findings generalize beyond 7–8B."

**Fallback on failure:**
> "Replication on Qwen2.5-14B-Instruct [did not replicate for metric X] — indicating heterogeneity across scales / training procedures. Rather than a universal pattern, social compliance dominance may be specific to certain RLHF procedures. We frame this as a scope condition."

---

## 8. §6 Discussion — UPDATES

### 8.1 Add: What Free-Form Evaluation Tells Us (E8)

> **Forced-choice vs. free-form: what each measurement captures.** The free-form evaluation (§5.12) confirms the forced-choice results directionally but attenuates the magnitude: a 23.8 pp opinion reduction in forced-choice corresponds to only a 0.22-point reduction on a 1–5 free-form sycophancy scale. Two interpretations are consistent.
>
> First, forced-choice may overstate behavioral change. In (A)/(B), a model cannot partially hedge — it must commit. DPO shifts this commitment, producing a clean 23.8 pp on a metric with no middle ground. In free-form, the same model can hedge or qualify.
>
> Second, free-form may miss sycophancy that forced-choice captures. "You raise a great point, and I can see why you believe X, but technically Y" is mildly sycophantic on the 1–5 scale, but forced-choice evaluators see it as "disagreement."
>
> The combined evidence supports a **content-vs-style distinction**: DPO reliably changes *content* (what conclusion) but less reliably changes *style* (how deferentially). Hedging 0.65 vs 0.64 unchanged. Future mitigation should target both.

### 8.2 Update: Limitations (E9)

- **Item 1 (scale):** "Cross-scale evidence is limited to [14B if E7 works / 8B only otherwise]. Generalization to 70B+, constitutional AI, or non-transformer architectures remains open."
- **Item 2 (forced-choice):** "Our primary measurement uses forced-choice for clean logit probabilities; we validate with a 150-conversation free-form benchmark (§5.12) showing the effect transfers directionally but with smaller magnitude. DPO reduces but does not eliminate free-form sycophancy."
- **Item 7 (DPO generalization):** "DPO transfer is format-robust (rephrased in-domain: ~77% retention, −18.2 pp) but domain-attenuated (Anthropic subcategories: ~20% retention, −4.9 pp). Larger/more diverse preference datasets may close this gap."
- **NEW Item 8:** "Free-form benchmark is a pilot (N=150 per condition). Advice domain (N=10) underpowered."
- **NEW Item 9:** "Single LLM-as-judge (Claude Sonnet 4). Validated with 50-conversation manual audit (κ = [C2 result]). Multi-judge protocols would strengthen free-form eval."

---

## 9. §7 Conclusion — EXPAND TO 8 POINTS

**TODO (E10).** Keep points 1–6 as written. Add:

> 7. **DPO is robust across seeds and transfers to free-form with attenuated magnitude.** Multi-seed (100/200/300) shows opinion sycophancy reduction 25.3 ± 2.8 pp (CV < 12%). Free-form (N=150) confirms directionally (sycophancy 2.64 → 2.41, truthfulness 3.50 → 3.74), largest gain in fictional-entity domain (−0.55), a domain with entirely different circuits than the opinion-domain training data.
>
> 8. **Preference optimization dominates SFT on the capability-safety tradeoff.** On identical preference data, DPO reduces opinion sycophancy while preserving capabilities (MMLU +0.8, GSM8k +3.6). SFT achieves stronger raw reduction (8.3% overall) but degrades GSM8k 33.2% → 7.5%. The preference signal carries information SFT discards; for behaviors whose correction should preserve orthogonal capabilities, preference-based training is the appropriate intervention level.

---

## 10. §8 Reproducibility — UPDATE

**TODO (E11).** Replace compute-budget with:

> The full experimental pipeline requires approximately **120 A100 GPU-hours**:
> - Llama-3-8B pipeline (13 jobs, baseline → steering): ~48h
> - Mistral-7B replication (5 jobs): ~30h
> - DPO training + eval (seed 100): ~2h
> - Multi-seed DPO (seeds 200, 300): ~12h
> - SFT baseline: ~1h
> - OOD evaluation: ~2h
> - Free-form generation (both conditions): ~3h
> - Patching bootstrap (Experiment 5): ~6h
> - Mistral DPO rerun (bug-fixed): ~4h
> - [Qwen2.5-14B stronger-model pipeline: ~20h if completes]
>
> Plus **~$8 in Anthropic API costs** (LLM-as-judge; 300 transcripts × ~3,400 tokens each).

**Seeds:** 42 (benchmark + evaluation), 100/200/300 (DPO training data + init). DPO seeds are disjoint from benchmark seed; validation check in `scripts/06_dpo_training.py`.

---

## 11. Reference Fixes (A-series in master checklist)

- [ ] **A1.** Restore `references.bib` (git status: deleted).
- [ ] **A2.** Chen et al. `year={2025}` → `year={2024}` (arXiv 2024-09-03).
- [ ] **A3.** Verify Paduraru et al. arXiv ID — remove if unverifiable.
- [ ] **A4.** Correct Venhoff et al. authors (likely Pyae Phoo Min et al.).
- [ ] **A5.** Add Vennemeyer et al. ICLR 2026.
- [ ] **A6.** Reconcile Yang et al. year (2024 vs 2025).
- [ ] Add Landis & Koch (1977) for kappa interpretation.
- [ ] Add Rafailov et al. (2023) DPO reference to introduction.
- [ ] Complete author lists on `@misc` entries.

---

## 12. Data-Accuracy Bugs (B-series in master checklist)

Found in peer-review audits (`sycophancy-mechinterp-research.md` Apr 13, `research.md` Apr 8):

- [ ] **B1.** §5.1 Cohen's d labels reversed. Paper says "d=0.18 (opinion vs. factual)" — JSON shows opinion-vs-reasoning=0.18, opinion-vs-factual=0.78. **Swap labels.**
- [ ] **B2.** §5.11 DPO train loss "0.69 → 0.16" unsupported. `dpo_training_metrics.json` stores `train_loss: 0.356`. Investigate origin (final step / mean / different seed?) or replace with the 0.356 figure.
- [ ] **B3.** §5.6 table broken by mid-table footnote `[^mmlu]`. Move below table or inline.
- [ ] **B4.** §5.5 probe decomposition sums to 100.1% (rounding). Add footnote or adjust "Other" by 0.1 pp.
- [ ] **B5.** §5.11 post-DPO best probe layer is Layer 4, paper reports Layer 1 for comparability. Add one sentence of disclosure.
- [ ] **B6.** §5.11 methods: disclose 400 → 360 effective training pairs after 10% eval split.
- [ ] **B7.** §5.11 GSM8k: post-DPO N=200 vs baseline N=1319 asymmetry — flag more prominently.

---

## 13. Figures — Master List (F-series in master checklist)

| # | Figure | Status | Section |
|---|---|---|---|
| 1 | Patching heatmap | ✅ Done | §5.4 |
| 2 | Steering alpha sweep | ✅ Done | §5.8 |
| 3 | Per-source opinion steering | ✅ Done | §5.8 |
| 4 | Probe accuracy by layer | ✅ Done | §5.3 |
| 5 | Ablation comparison | ✅ Done | §5.6 |
| 6 | DPO probe decomposition | ✅ Done (`figures/fig6_dpo_probe_decomposition.pdf`) | §5.11 |
| 7 | **NEW** DPO seed robustness | ⏳ **F1** | §5.11 |
| 8 | **NEW** OOD generalization (3-way grouped bar) | ⏳ **F2** | §5.11 |
| 9 | **NEW** Free-form 5-dim comparison | ⏳ **F3** (needs C3) | §5.12 |
| 10 | **NEW** Example transcript panel | ⏳ **F4** | §5.12 |
| 11 | **NEW** SFT vs DPO tradeoff scatter | ⏳ **F5** | §5.13 |
| 12 | **NEW** Cross-scale SC/BC comparison | ⏳ **F6** (Qwen gate) | §5.14 |
| App | Patching bootstrap stability | ⏳ **D1** (Job 55757250) | §5.4 appendix |

---

## 14. Patching Bootstrap — Integration Plan (Job 55757250)

On completion, `results/patching_bootstrap.json` will have:
- `per_resample[].top_10_heads` — top-10 head ranking per seed
- `aggregate.pairwise_jaccard.top{3,5,10}` — stability across 5 seeds
- `aggregate.top3_head_frequency` — how often each head appears in top-3
- `aggregate.per_head_stats[head].{rank_mean, rank_sd, recovery.ci_95}` — per-head stability

### §5.4 footnote (drop-in)

> We validated the stability of the top-3 head ranking (L4H28, L4H5, L5H31) with a 5-seed bootstrap (N=100 prompts each, independent random subsets). The top-3 head set has mean pairwise Jaccard = **[JACCARD_TOP3]** across resamples, and the three target heads appear in the top-3 in **[FREQ]** of resamples. Full per-head rank statistics are in Appendix [ID].

### Appendix table (drop-in)

| Head | Rank mean ± SD | Top-10 appearance rate | Recovery 95% CI |
|---|---|---|---|
| L4H28 | [fill] ± [fill] | [fill]% | [lo, hi] |
| L4H5 | [fill] ± [fill] | [fill]% | [lo, hi] |
| L5H31 | [fill] ± [fill] | [fill]% | [lo, hi] |
| ... | | | |

### §5.6 update

Add: "The ablation null holds for the heads identified as stable under bootstrap (§5.4 footnote) — not a sampling artifact of which heads were chosen."

---

## 15. Mistral DPO Rerun — Integration Plan (Job 55757251)

Fix applied: `src/data/base.py` now dispatches `format_llama3` vs `format_mistral` via `format_prompt(user_prompt, model_name)`. `AnthropicOpinionDataset(seed, model_name)` threads the model name. Verified via smoke test: Mistral pair prompts now contain `<s>[INST]...[/INST]` and no Llama tokens.

On completion, read `results/mistral/dpo_eval_results.json`:

### Success case (syc drops, GSM8k preserved)

Add to §5.10 cross-architecture section:

> | Model | Opinion Syc (pre) | Opinion Syc (post-DPO) | Δ | MMLU | GSM8k | SC (best) | Robust (best) |
> |---|---|---|---|---|---|---|---|
> | Llama-3-8B-Instruct | 82.4% | 58.6% | −23.8 pp | 62.8% | 38.5% | 12.0% | 76.3% |
> | Mistral-7B-Instruct | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |

Add one paragraph: "Cross-architecture DPO replication further supports the generality of training-time intervention (§5.13) across the two architectures whose baseline profiles differ substantially (§5.10)."

### Failure case (syc doesn't drop or capabilities collapse)

Add to §7 Limitations:
> "Cross-architecture DPO replication did not succeed cleanly on Mistral-7B-Instruct even with corrected chat templating. Our cross-architecture *behavioral replication* (baseline, probes, patching, ablation — §5.10) holds; cross-architecture *DPO* remains an open question. One honest attempt with Mistral-tuned hyperparameters (β=0.05, LR=1e-5, 2 ep) may follow in the camera-ready."

Either way, the buggy artifact at `results/mistral/dpo_training_pairs_llama_bug.json` stays — it documents the templating bug and the fix for reproducibility readers.

---

## 16. Anonymization Checklist (H-series)

Final pass before upload:

```bash
grep -ri "kenneth\|egan\|kenny\|wentworth\|larson\|kennyegan" paper.tex
grep -ri "kenegan2005\|github.com/kennyegan" paper.tex references.bib
grep -ri "@mit.edu\|@umass.edu\|@wit.edu" paper.tex
```

Replace with:
- Author block → `"Anonymous Author(s)"`
- Affiliation → `"Anonymous Institution"`
- Code citation → `"Code will be released upon acceptance"`
- Remove contact email or use placeholder

Also:
- [ ] PDF metadata: `pdftk paper.pdf dump_data` — no `InfoKey: Author`.
- [ ] Matplotlib: re-save figures with `metadata={'Author': ''}` in `savefig`.
- [ ] Verify no self-citation reveals identity.

---

## 17. Submission Logistics

- [ ] **Apr 20 (tonight):** Jobs 55757250 + 55757251 complete.
- [ ] **Apr 21:** Triage Qwen-14B error (G1). Start C1 manual audit.
- [ ] **Apr 22–23:** Complete C1–C3 (audit + kappa + bootstrap CIs). Integrate bootstrap results into §5.4 (D1).
- [ ] **Apr 24:** Integrate Mistral DPO rerun (D2). Start figures F1, F2.
- [ ] **Apr 25:** Figures F3, F4, F5. Send first complete draft to Prof. Larson.
- [ ] **Apr 26–27:** Paper writing — insert drafted §5.11.X, 5.11.Y, 5.12, 5.13 (E3–E6).
- [ ] **Apr 28:** Abstract trim (E1), contribution list (E2), discussion (E8), limitations (E9), conclusion (E10), reproducibility (E11). Final Prof. Larson sign-off.
- [ ] **Apr 29–May 3:** Data-accuracy bug fixes (B1–B7). Reference fixes (A1–A6). Anonymization (H1–H2).
- [ ] **May 3:** Final PDF build, page count check (H3–H4).
- [ ] **May 4 AOE:** Submit abstract on OpenReview (H5).
- [ ] **May 5:** Prepare supplementary (H6).
- [ ] **May 6 AOE:** Submit full paper (H7).

---

## 18. Cut List

Confirmed skipped (per `neurips-execution-plan.md`):
- ❌ SAE / neuron-level analysis — separate paper.
- ❌ Scaling to 70B+ — compute-prohibitive.
- ❌ Full 300–500 free-form benchmark — 150 pilot is defensible.
- ❌ DPO on stronger model with probes — Tier 3, camera-ready if time.
- ❌ Path patching on Llama-3 — Tier 3, future work.

If Qwen succeeds cleanly and time permits, consider:
- 🤔 DPO on Mistral with tuned HPs (4 GPU-hrs) — if tonight's 55757251 fails even with the bug fix, try β=0.05, LR=1e-5, 2 ep.

---

## 19. Priority Order (Apr 20 — what to do first)

If working tonight after the two jobs are submitted (and they are):

1. **Restore `references.bib`** (A1) — 5 min. BLOCKS LaTeX build.
2. **Chen et al. year fix** (A2) — 1 min.
3. **Cohen's d label swap in §5.1** (B1) — 5 min. Sanity check against `head_ablation_results.json` effect sizes.
4. **Start manual audit C1** — 60 min setup, then an hour chunks of ~15 convos each over tomorrow. Unlocks §5.12.
5. **Patching bootstrap monitor** — when job 55757250 finishes, read JSON and fill in §5.4 footnote template (§14 above).
6. **Mistral DPO monitor** — when 55757251 finishes, read eval JSON, fill in §15 table.

If you can only do one thing tonight: **restore references.bib** and **swap the Cohen's d labels**. Those are the two highest-leverage 10-minute edits.

Everything else can wait for tomorrow.

---

## 20. Open Questions for Prof. Larson

1. Co-authorship given scope expansion from original proposal?
2. Affiliation: Wentworth Institute of Technology or departmental?
3. Abstract tone — appropriately bounded for NeurIPS reviewers?
4. §5.13 SFT comparison — honest caveat on SFT HP tuning strong enough?
5. 8-contribution list — too many? (Some reviewers prefer 3–5.)
6. Qwen-14B gate — worth continuing to debug or cut §5.14 now and stay at 8B?
7. Mistral DPO fallback (tonight's job) — if it also fails with the fix, acceptable to cut to "behavioral replication only"?
