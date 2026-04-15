# NeurIPS Readiness Execution Plan

## Sycophancy Mechanistic Paper

## Executive summary

This document is the final execution plan for turning the current sycophancy paper from a strong small-model mechanistic case study into a credible NeurIPS main-track submission.

There is no guarantee of acceptance. The goal is to maximize the probability that reviewers see a rigorous, well-scoped, field-relevant paper with realistic evaluation, stronger generalization evidence, robust intervention results, and disciplined mechanistic claims.

The final paper should support this bounded thesis:

> Sycophancy is often best understood as a truth-preserving but output-distorting behavior under social pressure, and because this behavior is distributed and redundant, training-time interventions are more reliable than localized inference-time edits.

That is the paper we are trying to build.

---

## 1. Final paper target

### Paper transformation

Transform the project from:

**"A mechanistic sycophancy study on two small instruct models"**

into:

**"A rigorous multi-setting study showing that sycophancy is often truth-preserving but output-distorting, that localized circuit edits are limited by distributed implementation, and that training-time optimization reduces the behavior more robustly across settings."**

### Final contribution structure

The final paper should revolve around four contributions:

1. **Better evaluation:** a sycophancy framework spanning forced-choice and free-form / multi-turn conversational settings.
2. **Bounded mechanism:** evidence that sycophancy often reflects preserved truth-relevant internal information paired with socially compliant outputs.
3. **Intervention insight:** evidence that patching-identified local targets do not reliably yield robust control, consistent with distributed implementation.
4. **Robust mitigation:** a training-time mitigation result replicated across seeds, evaluated OOD, and compared against at least one meaningful baseline.

If the paper does not clearly deliver these four, it is not ready.

---

## 2. Hard choices locked now

These are the default execution choices unless a concrete compute or infrastructure issue blocks them.

### Stronger model choice

**Primary stronger model:** Qwen2.5-32B-Instruct
Reason: materially stronger than 7B–8B class, widely used, modern, and a credible scale jump.

**Backup stronger model:** Llama-3.1-70B-Instruct or a smaller but still clearly stronger instruct model that is practically runnable in available infrastructure.
Reason: if 32B/70B infrastructure becomes infeasible, use the strongest model that still creates a credible “not just small-model” story.

### Baseline mitigation choice

**Primary baseline:** supervised fine-tuning on correction / anti-sycophancy pairs.
Reason: simplest serious training-time baseline, easy to explain, easy to compare against DPO.

**Backup baseline:** synthetic disagreement-data tuning.
Reason: directly connected to prior sycophancy mitigation literature.

### Judge model choice

**Primary judge:** one strong frontier model used consistently across the full free-form evaluation.
**Validation:** manual audit on a fixed subset plus agreement analysis.

### Manual audit size

**Target:** 150–200 scored free-form conversations total across domains and intervention conditions.
This is large enough to detect systematic judge failures without becoming impossible.

### Minimum free-form benchmark size

**Target:** 300–500 prompt instances total, with multi-turn continuations.
Below this, the benchmark risks looking toy-sized.

---

## 3. What is required versus optional

## Required for submission

These are the minimum conditions for calling the paper NeurIPS-ready:

1. A free-form or multi-turn sycophancy evaluation component.
2. At least one clearly stronger model.
3. DPO run across at least 3 seeds.
4. At least one non-DPO training-time baseline.
5. OOD evaluation that is broader than template-local variation.
6. Mechanistic claims explicitly bounded around sufficiency vs. necessity.
7. A paper rewrite that removes overclaiming.

## Optional but high-value

1. Feature- or neuron-level follow-up.
2. Additional architectures beyond the stronger model.
3. Extended domain transfer into advice-heavy or high-stakes settings.
4. More than one baseline mitigation.

## First things to cut if time slips

If schedule pressure hits, cut in this order:

1. Extra mechanistic variants beyond the core stability checks.
2. Additional architectures beyond one stronger model.
3. Extended high-stakes domain expansions.
4. Stretch-goal neuron/feature analysis.

Do **not** cut:

* free-form evaluation,
* stronger-model replication,
* multi-seed DPO,
* the baseline mitigation comparison.

---

## 4. Exact experiment slate

This is the actual experiment order.

## Experiment 1: Free-form baseline evaluation on current models

### Goal

Test whether the current central story survives outside forced-choice evaluation.

### Models

* Llama-3-8B-Instruct
* Mistral-7B-Instruct

### Dataset slices

* Opinion disagreement
* Factual false-belief
* Reasoning under social pressure
* Fictional entity / fabricated knowledge

### Metrics

Each conversation receives scores for:

* **Truthfulness**: 1–5 ordinal
* **Agreement with false or biased user belief**: 1–5 ordinal
* **Hedging / partial agreement**: 0–2 categorical
* **Calibration / confidence appropriateness**: 1–5 ordinal
* **Resistance under repeated pushback**: binary at each turn plus aggregate fraction
* **Helpfulness / coherence**: 1–5 ordinal

### Validation

* Judge-model full scoring
* Manual audit on 50–75 conversations in the first pass
* If judge/human mismatch is high, fix rubric before scaling

### Output

* Free-form baseline table
* Domain breakdown figure
* Example transcript figure
* Correlation with forced-choice benchmark

### Decision gate

If the story falls apart here, the thesis must be revised before investing further.

---

## Experiment 2: Stronger-model baseline replication

### Goal

Remove the small-model artifact objection.

### Model

* Primary: Qwen2.5-32B-Instruct
* Backup: strongest practical alternative

### Required analyses

* Forced-choice sycophancy profile
* Probe decomposition: social compliance vs. belief corruption
* One local intervention gap test: patching vs. ablation or closest equivalent
* Limited free-form evaluation subset

### Output

* Cross-model comparison table
* SC vs. BC comparison figure
* Replication summary paragraph for abstract and intro

### Decision gate

If the stronger model contradicts the central story, reframe as a bounded heterogeneity result.

---

## Experiment 3: DPO robustness study

### Goal

Turn DPO from a promising result into a robust result.

### Required grid

* Seeds: 3
* Training sizes: 2 minimum
* Same eval suite for every run

### Required evaluations

* Forced-choice evaluation
* Free-form evaluation subset
* OOD prompt templates
* OOD topic categories
* Capability retention

### Required outputs

* DPO robustness table
* Seed variance plot
* Training-size sensitivity plot
* OOD generalization figure

### Mechanistic follow-up

Run probe re-analysis on at least 2 of the DPO seeds.

### Decision gate

If DPO is unstable, the paper shifts from “DPO works robustly” to “training-time methods outperform local edits, with DPO as one example.”

---

## Experiment 4: Baseline mitigation comparison

### Goal

Show whether DPO is actually better than a realistic training-time alternative.

### Baseline

* Primary: SFT on correction / anti-sycophancy pairs
* Backup: synthetic disagreement-data tuning

### Required comparison axes

* Sycophancy reduction
* Capability retention
* Free-form behavior
* OOD behavior
* Mechanistic shift where feasible

### Output

* Baseline vs. DPO comparison table
* Tradeoff figure
* Discussion of why one objective works better than another

### Decision gate

If SFT matches or beats DPO, the paper should emphasize the broader training-time lesson rather than over-centering DPO.

---

## Experiment 5: Mechanism stability checks

### Goal

Make the mechanism section reviewer-resistant.

### Required checks

* Patching target stability across seeds or sample subsets
* Domain-overlap quantification
* Variance reporting on top components
* Clean sufficiency-vs.-necessity framing

### Output

* Stability figure
* Domain-overlap figure
* Sufficiency vs. necessity figure

### Decision gate

If targets remain unstable, lean harder into the methodological caution rather than component-level claims.

---

## 5. Free-form benchmark design

This section must be concrete because it is the highest-leverage upgrade.

### Benchmark size

* **Initial pilot:** 120–150 conversations
* **Final target:** 300–500 total prompt instances with multi-turn continuations

### Domain allocation target

* Opinion disagreement: 30%
* Factual false-belief: 25%
* Reasoning under pressure: 20%
* Fictional entities / fabricated knowledge: 15%
* Advice/high-stakes optional expansion: 10%

### Conversation design

Each prompt family should support:

* initial biased or false user position,
* one turn of model response,
* one or two rounds of user pushback,
* optional escalation turn for a subset.

### Scoring protocol

For each conversation, score:

* **Truthfulness (1–5)**
* **Agreement with false/biasing user belief (1–5)**
* **Hedging / soft agreement (0=no hedging, 1=partial, 2=strong hedging)**
* **Calibration appropriateness (1–5)**
* **Pushback resistance (binary per turn, plus overall average)**
* **Helpfulness/coherence (1–5)**

### Judge validation

* Audit 50 conversations during pilot phase
* Audit 100–150 total across final benchmark
* Record judge-human agreement and major failure modes

### Required benchmark outputs

* Benchmark construction description
* Rubric appendix
* Judge validation table
* Example transcript figure
* Forced-choice vs. free-form comparison figure

---

## 6. Stronger-model execution details

### Why the stronger model matters

Without a stronger model, the paper remains too easy to dismiss as a 7B–8B interpretability project.

### Minimum analysis on stronger model

If compute is limited, these are the irreducible must-runs:

1. Forced-choice baseline
2. Probe decomposition
3. Limited free-form evaluation subset
4. One mitigation run or one intervention comparison

### If compute allows

Also run:

* local intervention gap analysis,
* OOD evaluation,
* full mitigation comparison.

### Reporting rule

Even if the stronger model result is partial, it must appear in the main paper, not hidden in appendix.

---

## 7. Paper figure lock

The paper should not be drafted seriously until these figures are planned.

### Must-have main-paper figures

1. **Forced-choice vs. free-form sycophancy overview**
2. **SC vs. BC decomposition across models**
3. **Sufficiency vs. necessity / patching-to-ablation figure**
4. **DPO robustness figure across seeds**
5. **Baseline mitigation vs. DPO tradeoff figure**
6. **OOD generalization figure**
7. **Representative transcript panel**

If these figures do not exist, the paper story is not complete.

---

## 8. Week-by-week execution order

## Week 1

* Finalize free-form taxonomy and prompt families
* Lock scoring rubric
* Build judge pipeline
* Run pilot free-form benchmark on current models

## Week 2

* Audit pilot results manually
* Revise rubric if needed
* Launch full free-form evaluation on current models
* Start stronger-model baseline runs

## Week 3

* Finish stronger-model baseline analyses
* Launch DPO multi-seed runs
* Prepare baseline SFT pipeline

## Week 4

* Finish DPO runs
* Run baseline mitigation comparison
* Start mechanism stability analyses

## Week 5

* Finish OOD evaluations
* Generate all core figures
* Decide final thesis wording based on outcomes

## Week 6

* Rewrite abstract, intro, results, discussion
* Tighten claims
* Build appendix and reproducibility sections

If the timeline compresses, keep the experiment order the same.

---

## 9. Risk management

### Risk 1: Free-form results weaken the current story

Response: revise the thesis to explicitly distinguish constrained vs. conversational sycophancy.

### Risk 2: Stronger model does not replicate cleanly

Response: frame the result as heterogeneity across architectures/scales, while preserving any stable common pattern.

### Risk 3: DPO is unstable across seeds

Response: shift the claim upward to the broader training-time-versus-local-edit lesson.

### Risk 4: SFT matches or beats DPO

Response: present the comparison honestly and center the paper on training-time objectives rather than DPO branding.

### Risk 5: Mechanistic targets remain unstable

Response: focus the contribution on the sufficiency-vs.-necessity gap and the methodological limits of local circuit intervention.

---

## 10. Paper positioning guidance

## Framing to emphasize

* Sycophancy is not just a benchmark quirk; it is a failure mode involving the relationship between internal information and socially compliant outputs.
* Forced-choice evaluation alone is insufficient.
* Mechanistic interpretability can help separate representation from output behavior, but local circuit discovery does not guarantee robust control.
* Training-time interventions are more promising than localized edits when the implementation is distributed.

## Framing to avoid

Do not write claims like:

* “This proves sycophancy is social compliance.”
* “We discovered the sycophancy circuit.”
* “Inference-time interventions cannot work.”
* “This is the universal mechanism of RLHF sycophancy.”
* “DPO solves sycophancy.”

These are review bait.

---

## 11. Submission checklist

Do not call the paper NeurIPS-ready unless most of these are true:

* The paper contains a realistic free-form or multi-turn evaluation component.
* It includes at least one stronger instruct model.
* The main mitigation result is replicated across multiple seeds.
* At least one non-DPO training-time baseline is included.
* OOD evaluation is meaningful.
* Claims are explicitly bounded.
* Mechanistic claims distinguish sufficiency from necessity.
* Figures show robustness, not isolated best-case outcomes.
* The discussion explains what the field learns, not just what this project found.

---

## 12. Final recommendation

The highest-leverage path is still:

1. Build the free-form / multi-turn benchmark.
2. Add one stronger model.
3. Run DPO across multiple seeds and data scales.
4. Compare against one real training-time baseline.
5. Rewrite the paper with tighter claims and stronger field positioning.

Everything else is secondary.

If these five are executed well, the paper has a real shot to move from “interesting case study” to “credible main-track contender.”

That is the bar.
