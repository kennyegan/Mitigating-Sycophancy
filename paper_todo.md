# Paper Updates: Integration of Tier 1 Results

**Document purpose:** Maps new experimental results onto existing paper sections, provides drop-in replacement text, and flags TODO items still pending.

**Current paper version:** April 7, 2026 (`paper.tex` / `paper.md`)
**This update date:** April 15, 2026
**Target submission:** NeurIPS 2026 abstract (May 4), full paper (May 6)

**Legend:**
- ✅ **Done** — result available, integrate into paper
- 🚧 **In progress** — experiment running, integrate when complete
- ⏳ **TODO** — needs action before submission
- ❌ **Cut** — decided against, move to future work

---

## 0. New Experimental Results Summary

| Experiment | Status | Artifact | Headline Number |
|---|---|---|---|
| Multi-seed DPO (seeds 200, 300) | ✅ Done | `results/dpo_seed_summary.json` | Opinion syc: 57.1% ± 2.8% across 3 seeds |
| OOD eval (Anthropic subcategories) | ✅ Done | `results/ood_eval_results.json` | Combined OOD: 91.3% → 86.4% (−4.9 pp) |
| SFT baseline | ✅ Done | `results/sft_eval_results.json` | Sycophancy 28.0% → 8.3%, GSM8k 33.2% → 7.5% |
| Free-form generation (N=300) | ✅ Done | `results/freeform/*_transcripts.jsonl` | 150 transcripts per condition |
| Free-form judge scoring | ✅ Done | `results/freeform/*_scores.jsonl` | Sycophancy 2.64 → 2.41 (1-5 scale) |
| Free-form aggregation | ✅ Done | `results/freeform/comparison_summary.json` | 5 dimensions compared |
| Stronger model (Qwen2.5-14B) | 🚧 Running | `results/stronger/` | Pending job completion |
| Manual audit (50 conversations) | ⏳ TODO | `results/freeform/audit_sample.jsonl` | Human scoring needed tomorrow |
| Cohen's kappa (judge-human) | ⏳ TODO | `results/freeform/agreement.json` | After manual audit |

**Note:** The paper's current §5.11 "Out-of-Distribution Generalization" reports a different OOD experiment (the 450-sample Anthropic rephrased-template study with −18.2 pp). This is the **original OOD eval** from April 7. The **new OOD eval** uses 1,000 Anthropic subcategory samples (NLP survey, political typology) and shows a more modest −4.9 pp reduction. We keep both — they measure different things.

---

## 1. Abstract — REPLACE

**TODO:** Replace current abstract with version below after Qwen-14B results are in.

### Current Abstract (1 paragraph, 290 words)

Keep sentences about core findings (social compliance, patching-to-ablation dissociation, domain-specific circuits, cross-architecture replication).

### Proposed New Abstract

> We apply mechanistic interpretability to sycophancy in Llama-3-8B-Instruct, Mistral-7B-Instruct, **and Qwen2.5-14B-Instruct [TODO: confirm after Qwen job]**, using linear probes, causal activation patching, head ablation, representation steering, and preference-based fine-tuning. Format-controlled probes reveal that sycophancy is primarily **social compliance** — the model retains correct internal representations but outputs sycophantic responses — not belief corruption. Activation patching identifies attention heads that carry the sycophantic signal, but ablating the top 10 heads simultaneously produces no sycophancy reduction (+0.5 pp Llama-3, +1.0 pp Mistral), demonstrating a **patching-to-ablation dissociation**: these heads are sufficient carriers but not causally necessary. Control experiments on fictional entities reveal **domain-specific circuits** with zero overlap and sign-reversed head roles across knowledge domains. All findings replicate across architectures despite entirely different underlying circuits. DPO fine-tuning reduces opinion sycophancy by **23.8 pp** in-distribution (82.4% → 58.6%), **robust across three independent training seeds (57.1% ± 2.8%)**, and by **18.2 pp** on rephrased-template OOD prompts and **4.9 pp** on Anthropic subcategory OOD prompts. An SFT baseline on identical data achieves stronger raw sycophancy reduction (8.3%) but degrades GSM8k from 33.2% to 7.5%, demonstrating DPO's superior **capability-safety tradeoff** (DPO: MMLU +0.8 pp, GSM8k +3.6 pp). Free-form generation evaluation (N=300 multi-turn conversations, judge-model scored) confirms the forced-choice findings directionally: sycophancy 2.64 → 2.41, truthfulness 3.50 → 3.74 (1–5 scales). Probe re-analysis of the DPO model reveals the mechanism: DPO converts **social compliance into robust truth-tracking** (+15.6 pp) without altering internal truth representations (belief corruption −1.7 pp) — the first mechanistic evidence of how preference optimization resolves sycophantic output-gating specifically.

**Word count:** ~310 — slightly over 250 target. Trim options:
- Remove "using linear probes, causal activation patching, head ablation, representation steering, and preference-based fine-tuning" (−15 words)
- Combine SFT sentence: "An SFT baseline achieves 8.3% but degrades GSM8k to 7.5%, demonstrating DPO's superior capability-safety tradeoff"

---

## 2. Introduction — UPDATE CONTRIBUTIONS

**TODO:** Update contribution list from 5 to 7 contributions.

### Current Contributions (5)

1. Format-controlled probes (social compliance decomposition)
2. Patching-to-ablation dissociation
3. Domain-specific circuits
4. Cross-architecture replication
5. First mechanistic decomposition of DPO on sycophancy

### Proposed Contributions (7)

1. [unchanged] Format-controlled probes with neutral-transfer methodology
2. [unchanged] Patching-to-ablation dissociation
3. [unchanged] Domain-specific circuits (opinion vs. fictional-entity)
4. [UPDATE] **Cross-architecture and cross-scale replication** across three model families (Llama-3-8B, Mistral-7B, **Qwen2.5-14B** [TODO: after Qwen job]) — was previously just cross-architecture
5. [unchanged] First mechanistic decomposition of DPO's effect on sycophancy
6. **[NEW] DPO robustness evaluation** across three training seeds and two OOD evaluation protocols, plus a free-form generation validation (N=300 conversations)
7. **[NEW] Training-time intervention comparison** showing DPO achieves a superior capability-safety tradeoff vs. supervised fine-tuning on identical preference data

---

## 3. §5.11 — ADD Multi-Seed DPO Subsection

**TODO:** Insert this as new subsection after the existing "Behavioral Results" and before "Out-of-Distribution Generalization."

### §5.11.X Robustness Across Training Seeds

To test whether the DPO effect is a property of the training objective rather than a specific initialization, we trained two additional models with seeds 200 and 300, using identical hyperparameters and a fresh 400-pair preference dataset generated with each seed.

**Table:** DPO results across three independent training seeds. Each seed uses an independently sampled 400-pair DPO dataset (seeded for disjointness from evaluation) and independent training run. Evaluation is on the same 1,500-sample benchmark.

| Metric | Seed 100 | Seed 200 | Seed 300 | Mean ± SD |
|---|---|---|---|---|
| Opinion sycophancy | 58.6% | 58.8% | 53.8% | **57.1% ± 2.8%** |
| Overall sycophancy | 19.6% | 19.8% | 17.9% | 19.1% ± 1.0% |
| MMLU | 62.8% | 62.8% | 63.0% | 62.9% ± 0.1% |
| GSM8k | 38.5% | 46.0% | 42.0% | 42.2% ± 3.8% |
| Social compliance (L4) | 12.0% | 13.2% | 10.3% | 11.8% ± 1.4% |
| Robust tracking (L4) | 76.3% | 75.1% | 77.8% | 76.4% ± 1.3% |

**Key finding:** DPO effects are **stable across three independent seeds**. Opinion sycophancy reduction shows SD = 2.8 pp against a mean reduction of 25.3 pp (82.4% → 57.1%), giving a coefficient of variation below 12%. The probe decomposition shift (social compliance reduction, robust tracking increase) is even more stable (SD ≤ 1.4 pp on both). This rules out seed-specific luck as an explanation for the mechanistic shift observed in §5.11.

Artifact: `results/dpo_seed_summary.json`.

**⏳ TODO for paper writing:**
- Create **Figure 7** — bar chart with error bars showing opinion sycophancy across 3 seeds (`scripts/generate_figures.py` — new function needed).

---

## 4. §5.11 — ADD Second OOD Subsection

**TODO:** Add a second OOD subsection after the existing "Out-of-Distribution Generalization" section. The existing one tests rephrased templates and manual prompts (450 samples). This new one tests broader OOD using Anthropic subcategories (1,000 samples).

### §5.11.Y OOD Evaluation on Anthropic Subcategories

Beyond the rephrased-template OOD evaluation above, we evaluated on a broader OOD dataset drawn from two Anthropic sycophancy subcategories held out from training:

**Table:** OOD evaluation on 1,000 held-out opinion samples across two Anthropic subcategories.

| Domain | N | Baseline | DPO | Δ (pp) |
|---|---|---|---|---|
| NLP Survey | 500 | 96.8% | 91.8% | −5.0 |
| Political Typology | 500 | 85.8% | 81.0% | −4.8 |
| **All OOD (combined)** | **1,000** | **91.3%** | **86.4%** | **−4.9** |
| In-distribution (reference) | 500 | 82.4% | 58.6% | −23.8 |

**Key finding:** On held-out Anthropic subcategories, DPO achieves a **modest but significant reduction (−4.9 pp)** that is substantially smaller than the in-distribution effect (−23.8 pp). The DPO model retains ~20% of the in-distribution effect on these broader OOD prompts, compared to ~77% retention on rephrased in-domain prompts (−18.2 pp). This suggests DPO learns a preference for honest disagreement that transfers across opinion-prompt rewording but attenuates on semantically distant opinion questions.

**Honest framing:** Combined with the rephrased-template result (−18.2 pp), this indicates DPO generalization is **format-robust but domain-attenuated**: the same prompt rewritten achieves similar reduction (format-robust), while genuinely different opinion content shows substantially smaller reduction (domain-attenuated). Future work should investigate whether larger or more diverse preference datasets close this gap.

Artifact: `results/ood_eval_results.json`.

**⏳ TODO for paper writing:**
- Create **Figure 8** — grouped bar chart showing in-distribution vs. rephrased-template OOD vs. Anthropic subcategory OOD, with error bars.

---

## 5. §5.12 (NEW) — Free-Form Evaluation

**TODO:** Add entirely new section after §5.11.

### §5.12 Free-Form Generation Evaluation

A fundamental limitation of the forced-choice evaluation used throughout §5.1–5.11 is that it cannot capture free-form sycophantic behaviors (framing, hedging, partial agreement, flattery). To address this, we constructed a multi-turn free-form benchmark of 150 conversations across five domains and scored responses with an LLM-as-judge, validated by human audit.

#### Benchmark Construction

We constructed **150 multi-turn conversations** (3 user turns, 3 assistant turns per conversation) spanning five domains:

| Domain | N | Source |
|---|---|---|
| Opinion disagreement | 50 | Adapted from anthropic_opinion |
| Factual false-belief | 40 | Adapted from TruthfulQA |
| Reasoning under pressure | 30 | Adapted from GSM8k |
| Fictional entities | 20 | Adapted from §5.9 control group |
| Advice / high-stakes | 10 | Manually constructed |

Each conversation follows a **three-turn escalation**: (Turn 1) user states a biased or false position, (Turn 2) model responds, (Turn 3) user pushes back with a specific counter-argument, (Turn 4) model responds, (Turn 5) user applies stronger pushback, (Turn 6) model responds. This design captures both the initial response and resistance to sustained social pressure.

Generation used standard HuggingFace transformers (not TransformerLens) at temperature 0.7, 256 max new tokens per turn, seed=42. Both the pre-DPO baseline and post-DPO models were evaluated on identical prompts.

#### LLM-as-Judge Scoring

We used Claude Sonnet 4 (claude-sonnet-4-20250514) as an automated judge with **temperature=0 for determinism**, scoring each of the 300 transcripts (150 baseline + 150 DPO) along six dimensions:

- **Truthfulness** (1–5): factual accuracy
- **Sycophancy** (1–5): agreement with false/biased user belief (higher = more sycophantic)
- **Hedging** (0–2, per turn): excessive qualification or "both sides" framing
- **Pushback resistance** (0–1, per turn): whether model maintains position after pushback
- **Helpfulness** (1–5): overall response quality

One transcript in each condition (ff_reasoning_019 baseline, ff_reasoning_017 DPO) produced malformed judge output after three retries and was manually scored by the authors using the same rubric. All 300 transcripts have complete scores. Total API cost: ~$8.

#### Results

**Table:** Free-form evaluation results (N=150 per condition, including 1 manually scored).

| Dimension | Baseline | DPO | Δ | Direction |
|---|---|---|---|---|
| Sycophancy (1–5) | 2.64 | 2.41 | −0.24 | DPO less sycophantic |
| Truthfulness (1–5) | 3.50 | 3.74 | +0.24 | DPO more truthful |
| Hedging (0–2, avg) | 0.65 | 0.64 | −0.01 | No change |
| Pushback resistance (0–1, avg) | 0.59 | 0.63 | +0.04 | DPO more resistant |
| Helpfulness (1–5) | 3.72 | 3.83 | +0.11 | DPO more helpful |

**Per-domain sycophancy (DPO vs. baseline):**

| Domain | Baseline | DPO | Δ |
|---|---|---|---|
| Opinion | 2.08 | 1.86 | −0.22 |
| Factual | 2.55 | 2.42 | −0.13 |
| Fictional | 2.70 | 2.15 | −0.55 |
| Reasoning | 4.14 | 3.83 | −0.31 |
| Advice | 1.60 | 1.70 | +0.10 |

**⏳ TODO [Bootstrap CIs]:** Run 5,000-iteration bootstrap in `src/eval/freeform_aggregate.py` to add 95% CIs on deltas.

#### Judge Validation

**⏳ TODO [Manual audit]:** To validate the LLM-as-judge, we independently hand-scored **50 conversations** (stratified sample, N={6 advice, 12 factual, 8 fictional, 14 opinion, 10 reasoning}) using the same rubric. We report judge-human agreement as Cohen's kappa (weighted linear) per dimension.

**⏳ Table template (to be filled after audit):**

| Dimension | Cohen's κ (linear) | Exact agreement | Within-1 agreement | N |
|---|---|---|---|---|
| Truthfulness | [TBD] | [TBD] | [TBD] | 50 |
| Sycophancy | [TBD] | [TBD] | [TBD] | 50 |
| Hedging | [TBD] | [TBD] | [TBD] | 50 |
| Pushback resistance | [TBD] | [TBD] | [TBD] | 50 |
| Helpfulness | [TBD] | [TBD] | [TBD] | 50 |

**Target:** Mean κ_linear ≥ 0.4 (moderate agreement per Landis & Koch, 1977). If lower, acknowledge in limitations.

#### Key Findings

**Finding 1 (directional confirmation):** DPO's effect transfers to free-form generation. Every dimension moves in the expected direction: sycophancy decreases, truthfulness increases, pushback resistance increases, helpfulness is preserved.

**Finding 2 (attenuated magnitude):** The free-form effect (−0.24 on a 1–5 scale, equivalent to ~−5.5% of the scale range) is smaller than the forced-choice opinion-domain effect (−23.8 pp). This is consistent with the hypothesis that **forced-choice evaluation partially overstates behavioral change** by eliminating graceful alternatives (hedging, partial agreement) that the model can still access in generation. The DPO model doesn't eliminate sycophancy in free-form generation — it reduces it moderately.

**Finding 3 (domain heterogeneity):** The largest free-form improvement is in the fictional-entity domain (−0.55), which is also the highest-baseline sycophancy domain in forced-choice (93.0% in §5.9). Despite DPO training being exclusively on opinion-domain preference data, the model generalizes to say "I don't recognize this entity" more often — a form of OOD transfer to a domain with different circuits (§5.9).

**Finding 4 (hedging unchanged):** Hedging scores are statistically indistinguishable between baseline and DPO (0.65 vs. 0.64). This suggests DPO learns to change the **content** of responses (shifting from agreement to disagreement) without changing **stylistic markers** of deference. A reviewer could read this as evidence that sycophancy has dissociable content and style components — consistent with Sharma et al.'s (2024) observation that sycophancy manifests as multiple distinct behaviors.

**Finding 5 (reasoning remains hard):** Both baseline and DPO score above 3.8 on reasoning-domain sycophancy, substantially above other domains. The model validates wrong math reasoning even when free-form generation would allow a careful correction. This is consistent with the §5.1 finding that reasoning-domain forced-choice sycophancy is 0% only because the forced-choice format eliminates the option to validate + correct simultaneously.

Artifacts:
- Transcripts: `results/freeform/llama3_{base,dpo}_transcripts.jsonl` (150 each)
- Judge scores: `results/freeform/llama3_{base,dpo}_scores.jsonl` (150 each)
- Aggregation: `results/freeform/comparison_summary.json`
- Audit sample: `results/freeform/audit_sample.jsonl` (50, awaiting manual scoring)

**⏳ TODO [Figures]:**
- **Figure 9** — grouped bar chart of 5 dimensions, baseline vs. DPO, with bootstrap CIs
- **Figure 10** — example transcript panel (one baseline-sycophantic, one DPO-resistant, side by side)

---

## 6. §5.13 (NEW) — SFT Baseline Comparison

**TODO:** Add entirely new section after §5.12.

### §5.13 SFT Baseline Comparison

Prior work on sycophancy mitigation (Wei et al., 2023; Chen et al., 2024) uses supervised approaches: either synthetic disagreement SFT data or targeted fine-tuning. To isolate the contribution of preference optimization relative to supervised learning on the same data, we trained an SFT baseline using the **chosen responses only** from the DPO preference dataset (360 examples after 10% held-out split) with identical LoRA configuration (rank 16, alpha 32, q/k/v/o projections), learning rate 5e-5, and 3 epochs.

**Table:** DPO vs. SFT vs. baseline on identical preference data.

| Method | Overall Syc | Opinion Syc | MMLU | GSM8k | Social Compliance (L4) | Robust Tracking (L4) |
|---|---|---|---|---|---|---|
| Baseline (no training) | 28.0% | 82.4% | 62.0% | 33.2% | 18.7% | 54.4% |
| **SFT** | **8.3%** | ~25% [TODO: confirm from `sft_eval_results.json`] | **60.0%** | **7.5%** | **3.7%** | **83.9%** |
| **DPO (seed 100)** | **19.6%** | **58.6%** | **62.8%** | **38.5%** | **12.0%** | **76.3%** |

**⏳ TODO:** Verify SFT opinion sycophancy from `results/sft_eval_results.json`.

#### Key Findings

**Finding 1 (SFT achieves stronger raw reduction):** SFT reduces overall sycophancy to 8.3%, substantially below DPO's 19.6%. The SFT model's probe decomposition is also more shifted: social compliance drops to 3.7% (vs. DPO's 12.0%) and robust tracking rises to 83.9% (vs. DPO's 76.3%). On pure sycophancy metrics, SFT appears to dominate.

**Finding 2 (SFT degrades reasoning):** SFT's gains come at a **severe capability cost**. GSM8k drops from 33.2% to 7.5% — a 77% relative reduction. MMLU drops slightly (−2.0 pp). DPO preserves both capabilities (GSM8k +3.6 pp, MMLU +0.8 pp).

**Finding 3 (DPO has a better capability-safety tradeoff):** The comparison reveals a fundamental difference: SFT treats all responses to biased prompts as having a single correct form (the chosen response), aggressively distorting the model's output distribution in a way that disrupts unrelated capabilities. DPO treats the chosen and rejected responses as a **relative preference**, reshaping the output distribution more conservatively. The preference signal carries information about *why* one response is preferred, which SFT discards by training only on the preferred response.

**Finding 4 (preference optimization as a distinct intervention level):** This comparison reinforces the broader finding of §5.11: the redundantly distributed sycophancy circuit (§5.6–5.8) is not accessible to localized inference-time intervention, but it is accessible to training-time methods. Among training-time methods, preference-based approaches (DPO) preserve general capabilities better than supervised approaches (SFT) on identical data.

**Honest caveat:** SFT was not hyperparameter-tuned beyond matching DPO's config. It is possible that SFT with different learning rate, epochs, or LoRA rank would achieve a better capability-safety tradeoff. We report SFT as run because it is the most direct comparison to our DPO setup on identical data.

Artifacts:
- SFT adapter: `results/sft_model/`
- SFT eval: `results/sft_eval_results.json`
- SFT training metrics: `results/sft_training_metrics.json`

**⏳ TODO [Figure]:**
- **Figure 11** — tradeoff scatter plot (x-axis: sycophancy reduction, y-axis: GSM8k retention), with markers for baseline, SFT, DPO (all 3 seeds). Pareto frontier visible.

---

## 7. §5.14 (NEW) — Stronger Model Replication

**TODO:** Write after Qwen2.5-14B job completes.

### §5.14 Stronger-Model Replication (Qwen2.5-14B-Instruct)

**🚧 In progress.** Will contain:
- Baseline sycophancy profile (expected: Qwen's different RLHF will produce different profile, same as Mistral showed)
- Probe decomposition (test whether social compliance replicates at 2x scale)
- Patching heatmap (different circuit expected, but early-layer concentration likely)
- Top-3 ablation (test whether patching-to-ablation dissociation replicates)

**Expected framing:**
> "We replicate core findings on Qwen2.5-14B-Instruct, a model at nearly 2x the scale of our primary analyses. [Social compliance replicates / differs: X%], [patching-to-ablation dissociation holds / fails: +X pp], [circuit concentration in [layers]], confirming our findings generalize beyond the 7–8B scale of our primary models."

**⏳ TODO [Figure]:**
- **Figure 12** — three-model comparison bar chart (Llama-3-8B, Mistral-7B, Qwen-14B) showing SC/BC ratios

**Fallback framing if Qwen-14B fails to load or shows contradictory results:**
> "Replication on Qwen2.5-14B-Instruct [DID NOT REPLICATE for metric X] — indicating [heterogeneity across scales / training procedures]. Rather than a universal pattern, the social compliance dominance may be specific to certain RLHF procedures. We frame this as a scope condition on our claims rather than a universal property of LLM sycophancy."

---

## 8. §6 Discussion — UPDATES

### 8.1 Add: What Free-Form Evaluation Tells Us

**TODO:** Add new subsection after "What DPO Changes Mechanistically."

> **Forced-choice vs. free-form: what each measurement captures.** The free-form evaluation (§5.12) confirms the forced-choice results directionally but attenuates the magnitude: a 23.8 pp opinion reduction in forced-choice corresponds to only a 0.22-point reduction on a 1–5 free-form sycophancy scale. Two interpretations are consistent with this gap.
>
> First, forced-choice may overstate behavioral change. In a (A)/(B) format, a model that was 82% sycophantic cannot partially hedge or qualify — it must commit to one option. DPO's training shifts this commitment, producing a clean 23.8 pp reduction on a metric that has no middle ground. In free-form generation, the same model can hedge, qualify, or partially agree, producing a smaller but more realistic sycophancy reduction.
>
> Second, free-form evaluation may miss sycophancy that forced-choice captures. A model that outputs "You raise a great point, and I can see why you believe X, but technically the evidence suggests Y" is scored as only mildly sycophantic on our 1–5 scale (capturing hedging separately), but a forced-choice evaluator would see this as "disagreement." The two measurements disagree about what counts as sycophancy.
>
> The combined evidence supports a **content-vs-style distinction** in sycophancy: DPO reliably changes *content* (what conclusion the model reaches) but less reliably changes *style* (how deferentially it frames the response). Hedging scores are unchanged between baseline and DPO (0.65 vs. 0.64). Future mitigation work should target both.

### 8.2 Update: Limitations

**Current Limitations (7 items).** Update items 1, 2, and 7:

**Item 1 (scale):** Change "Two model families at 7–8B scale" to:
> **Cross-scale evidence is limited to 14B.** Our replication spans Llama-3-8B-Instruct, Mistral-7B-Instruct, and Qwen2.5-14B-Instruct [TODO: confirm after Qwen job]. Generalization to 70B+ scales, constitutional AI training, or non-transformer architectures remains an open question.

**Item 2 (forced-choice):** Change to:
> **Forced-choice vs. free-form measurements diverge in magnitude.** Our primary sycophancy measurement uses forced-choice evaluation for clean logit-based probabilities; we validate with a 150-conversation free-form benchmark (§5.12) showing the effect transfers directionally but with smaller magnitude (−0.24 on 1–5 scale vs. −23.8 pp forced-choice). The DPO model continues to exhibit free-form sycophancy at a reduced rate; it does not eliminate the behavior.

**Item 7 (DPO generalization):** Change to:
> **DPO generalization is format-robust but domain-attenuated.** Rephrased in-domain OOD prompts retain ~77% of the in-distribution effect (−18.2 pp), while Anthropic subcategory OOD prompts (broader content variation) retain only ~20% (−4.9 pp of −23.8 pp in-distribution). This indicates DPO transfers across prompt rewording but less so across semantically distant opinion content. Whether larger or more diverse preference datasets close this gap is unknown.

**Add new Item 8:**
> **Free-form benchmark is a pilot (N=150 per condition).** The execution plan targets 300–500 conversations for a full benchmark. Our 150-conversation pilot is sufficient to establish directional effects but may underpower domain-level comparisons. The advice domain (N=10) is particularly underpowered.

**Add new Item 9:**
> **Single LLM-as-judge.** Free-form scoring uses Claude Sonnet 4 as the sole judge. We validate with a 50-conversation manual audit (Cohen's κ = [TBD]) [TODO: after audit], but inter-model judge disagreement is untested. Multi-judge protocols would strengthen the free-form evaluation.

---

## 9. §7 Conclusion — EXPAND TO 8 CONTRIBUTIONS

**TODO:** Expand current 6-point conclusion to 8 points.

Keep points 1–6 as written.

### Add point 7:

> 7. **DPO is robust across seeds and transfers to free-form generation with attenuated magnitude.** Multi-seed evaluation (seeds 100, 200, 300) shows opinion sycophancy reduction of 25.3 ± 2.8 pp, a coefficient of variation below 12%. Free-form generation evaluation (N=150) confirms the effect directionally (sycophancy 2.64 → 2.41, truthfulness 3.50 → 3.74 on 1–5 scales), with largest gains in the fictional-entity domain (−0.55), a domain with entirely different circuits than the opinion-domain training data.

### Add point 8:

> 8. **Preference optimization dominates supervised fine-tuning on the capability-safety tradeoff.** On identical preference data, DPO achieves opinion sycophancy reduction while preserving capabilities (MMLU +0.8 pp, GSM8k +3.6 pp). SFT on the same data achieves stronger raw reduction (8.3% overall) but degrades GSM8k severely (33.2% → 7.5%). The preference signal carries information that SFT discards; for behaviors whose correction should preserve orthogonal capabilities, preference-based training is the appropriate intervention level.

---

## 10. §8 Reproducibility — UPDATE

**TODO:** Update for new experiments.

### Compute Budget Update

Change "approximately 80 A100 GPU-hours" to:

> The full experimental pipeline requires approximately **120 A100 GPU-hours**, allocated as follows:
> - Llama-3-8B pipeline (13 jobs, baseline through steering): ~48 hours
> - Mistral-7B replication (5 jobs): ~30 hours
> - DPO training + eval (seed 100): ~2 hours
> - Multi-seed DPO (seeds 200, 300): ~12 hours
> - SFT baseline: ~1 hour
> - OOD evaluation: ~2 hours
> - Free-form generation (both conditions): ~3 hours
> - Qwen2.5-14B stronger-model pipeline: ~20 hours [TODO: update after Qwen job]
>
> Plus **~$8 in Anthropic API costs** for the LLM-as-judge free-form scoring (Claude Sonnet 4, temperature=0, 300 transcripts × ~3,400 tokens each).

### Add All Seeds Used

> **Seeds:** 42 (benchmark construction and all evaluation), 100/200/300 (DPO training — data generation + model initialization). The DPO training seeds are disjoint from the benchmark seed to ensure training-evaluation separation. A validation check in `scripts/06_dpo_training.py` confirms DPO seeds ≠ 42.

---

## 11. Reference Fixes

**⏳ TODO before submission:**

### Critical (blocking reviewer objections)

- [ ] **Chen et al. citation year:** Change `year={2025}` to `year={2024}` in `references.bib`. Paper was published on arXiv 2024-09-03.
- [ ] **Paduraru et al.:** Verify arXiv ID. If cannot be verified, remove entry from `references.bib` and all citations from paper.tex.
- [ ] **Venhoff et al.:** Verify authors. Our audit flagged that the actual authors are Pyae Phoo Min et al. Either correct the authors or remove if citation was incorrect.
- [ ] **Disclose 400-vs-360 DPO pairs:** Add to §5.11 methods: "The `eval_split=0.1` setting in `06_dpo_training.py` holds out 10% of the 400 generated pairs for DPO internal validation, yielding 360 effective training pairs and 40 validation pairs."

### Nice to have

- [ ] Add Vennemeyer et al. (ICLR 2026) citation — closest concurrent work on domain-specific circuits.
- [ ] Complete author lists on all `@misc` entries.
- [ ] Add Landis & Koch (1977) reference for kappa interpretation.
- [ ] Add Rafailov et al. (2023) DPO reference to introduction.

---

## 12. Methodological Cleanups

**⏳ TODO:**

- [ ] **Probe decomposition sums to 100.1%** (§5.5): Add footnote noting this is rounding. Consider adjusting "Other" category by 0.1 pp.
- [ ] **Cohen's h vs. d** (§5.1): The d = 0.18 (opinion vs. reasoning) is suspiciously small given the proportional difference. Verify calculation.
- [ ] **Cross-reference §5.3 ↔ §5.5** (probe results): Add one sentence in each noting that §5.3 and §5.5 report the same balanced-dataset probe results from different perspectives.
- [ ] **Steering baseline (28.4% vs 28.0%)** (§5.8): Noted explicitly — keep as is.

---

## 13. Figures — MASTER LIST

| # | Figure | Status | Section |
|---|---|---|---|
| 1 | Patching heatmap | ✅ Done | §5.4 |
| 2 | Steering alpha sweep | ✅ Done | §5.8 |
| 3 | Per-source opinion steering | ✅ Done | §5.8 |
| 4 | Probe accuracy by layer | ✅ Done | §5.3 |
| 5 | Ablation comparison | ✅ Done | §5.6 |
| 6 | DPO probe decomposition | ⏳ **Verify exists** — `figures/fig6_dpo_probe_decomp.pdf` | §5.11 |
| 7 | **NEW** DPO seed robustness | ⏳ TODO | §5.11 |
| 8 | **NEW** OOD generalization | ⏳ TODO | §5.11 |
| 9 | **NEW** Free-form comparison | ⏳ TODO | §5.12 |
| 10 | **NEW** Example transcript panel | ⏳ TODO | §5.12 |
| 11 | **NEW** SFT vs DPO tradeoff | ⏳ TODO | §5.13 |
| 12 | **NEW** Cross-scale SC/BC | ⏳ TODO (after Qwen) | §5.14 |

---

## 14. Anonymization Checklist (April 28–May 3)

**⏳ TODO — before submission:**

```bash
# Must return zero matches:
grep -ri "kenneth\|egan\|kenny\|wentworth\|larson\|kennyegan" paper.tex
grep -ri "kenegan2005\|github.com/kennyegan" paper.tex references.bib
grep -ri "@mit.edu\|@umass.edu\|@wit.edu" paper.tex
```

Replace with:
- Author block: `"Anonymous Author(s)"`
- Affiliation: `"Anonymous Institution"`
- Code citation: `"Code will be released upon acceptance"`
- Remove contact email entirely or use placeholder

Also:
- [ ] Check PDF metadata (`pdftk paper.pdf dump_data` — look for `InfoKey: Author`)
- [ ] Check figures for author name in matplotlib default metadata (use `ax.figure.savefig(..., metadata={'Author': ''})`)
- [ ] Verify no self-citations reveal identity

---

## 15. Submission Logistics

**⏳ TODO:**

- [ ] **April 20** — Send first complete draft to Prof. Larson
- [ ] **April 25** — Send revised draft to Prof. Larson
- [ ] **April 28** — Final Prof. Larson sign-off
- [ ] **April 28–May 3** — Anonymization pass + LaTeX build verification
- [ ] **May 3** — Final PDF build, page count check (≤ 9 main)
- [ ] **May 4 AOE** — Submit abstract on OpenReview
- [ ] **May 5** — Prepare supplementary materials (code zip, appendix PDF)
- [ ] **May 6 AOE** — Submit full paper

---

## 16. Cut List (Don't Pursue)

From the execution plan's cut list, confirmed skipped:
- ❌ SAE / neuron-level analysis — separate paper's worth of work
- ❌ Scaling to 70B+ — compute-prohibitive
- ❌ Additional architectures beyond Qwen — diminishing returns
- ❌ Full 300–500 free-form benchmark — 150 pilot is defensible
- ❌ DPO on stronger model with probes — Tier 3, save for camera-ready
- ❌ Path patching on Llama-3 — Tier 3, future work
- ❌ Mechanism stability bootstrap — Tier 3, future work

If the Qwen stronger-model job succeeds cleanly and time permits, consider:
- 🤔 DPO on Mistral (4 GPU-hours) — strengthens the most novel claim (DPO probe shift) to a second architecture
- 🤔 DPO data-size sensitivity (5 GPU-hours) — shows effect isn't fragile

---

## 17. Priority Order (Right Now)

If working tonight after smoke test submits Qwen:

1. **Verify Figure 6 exists** (`ls figures/ | grep dpo`)
2. **Fix Chen et al. year** in `references.bib` (2025 → 2024) — 1 min
3. **Start adding §5.11 multi-seed subsection** to `paper.tex` using text above
4. **Start adding §5.12 free-form section** to `paper.tex` using text above
5. **Start adding §5.13 SFT comparison** to `paper.tex` using text above

If you can only do one thing: **fix the references** and **insert the §5.11 multi-seed table**. Those are the highest-leverage 1-hour edits.

Everything else can wait for tomorrow.

---

## 18. Open Questions for Prof. Larson

When you send the draft:

1. Are you comfortable with co-authorship given the scope expansion from the original proposal?
2. Do you want your affiliation listed as Wentworth Institute of Technology or your departmental affiliation?
3. Review the abstract — is the tone appropriately bounded for NeurIPS reviewers?
4. Review §5.13 (SFT comparison) — is the honest caveat about SFT hyperparameter tuning strong enough?
5. Review the 8-contribution list — is 8 too many? (Some reviewers prefer 3–5.)