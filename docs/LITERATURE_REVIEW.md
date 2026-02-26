# Literature Review: Sycophancy is Social Compliance, Not Belief Corruption

**Last Updated:** 2026-02-20

Papers are organized to mirror the structure of the research proposal. The proposal's Related Work section (Section 2) is the canonical reference list — these are the papers you'll cite.

---

## Reading Priority

| Priority | Category | Papers |
|----------|----------|--------|
| **Week 1** | Sycophancy in LLMs | Papers 1-4 |
| **Week 2** | Mechanistic Interpretability | Papers 5-8 |
| **Week 3** | Inference-Time Intervention | Papers 9-11 |
| **Week 4** | Secondary / background | Papers 12+ |

---

## Group 1: Sycophancy in LLMs

These four papers collectively define the problem space and position our work.

---

#### 1. Perez et al. (2022) — "Discovering Language Model Behaviors with Model-Written Evaluations"
**arXiv:** https://arxiv.org/abs/2212.09251

Introduces model-written evaluations and documents sycophancy as a named behavior in RLHF-trained models. The Anthropic/model-written-evals dataset we use for our opinion sycophancy domain comes directly from this work. Provides baseline definitions and the behavioral characterization of sycophancy as agreement with user opinions regardless of factual accuracy.

**Relevance:** Source of our `anthropic_opinion` dataset. Definition of sycophancy we adopt. First citation in your intro.

---

#### 2. Sharma et al. (2024) — "Towards Understanding Sycophancy in Language Models"
**ICLR 2024** · https://arxiv.org/abs/2310.13548

The most comprehensive behavioral characterization of sycophancy to date. Shows sycophancy scales with model capability, is dramatically amplified by RLHF training, and occurs across opinion, factual, and reasoning domains. Key finding: models change correct answers when users express doubt, and validate opinions even when factually wrong.

**Relevance:** Primary reference for sycophancy measurement. Directly motivates our hypothesis that RLHF introduces (not just amplifies) the sycophancy circuit. The paper to read first.

---

#### 3. Wei et al. (2023) — "Simple synthetic data reduces sycophancy in large language models"
**arXiv:** https://arxiv.org/abs/2308.03958

Shows sycophancy can be reduced by training on synthetic examples where the model disagrees with incorrect user opinions. Establishes a training-data mitigation baseline. The implication: sycophancy is a learned behavior, not a fundamental capability limitation.

**Relevance:** The training-based mitigation approach our inference-time method improves upon. Supports the Social Compliance hypothesis (if training data can fix it, it's not deep corruption).

---

#### 4. Turpin et al. (2024) — "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting"
**NeurIPS 2024** · https://arxiv.org/abs/2305.04388

Demonstrates that CoT explanations can be post-hoc rationalizations rather than faithful descriptions of actual model computation. Models produce plausible-looking reasoning chains that don't reflect the true causal path. Indirect evidence for Social Compliance: if CoT is unfaithful, the internal computation may differ from the stated reasoning.

**Relevance:** Indirect prior evidence for our Social Compliance hypothesis. Cited in our Related Work as motivation for mechanistic rather than behavioral analysis.

---

## Group 2: Mechanistic Interpretability

The technical foundation for our circuit discovery methodology.

---

#### 5. Wang et al. (2022) — "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
**arXiv:** https://arxiv.org/abs/2211.00593

The canonical circuit discovery paper. Identifies specific attention heads responsible for the Indirect Object Identification (IOI) task via activation patching, demonstrating the full pipeline from behavioral characterization to causal circuit verification. The methodology we directly follow in Sections 4.2–4.3.

**Relevance:** Template for how to present sycophancy circuit discovery. The patching methodology in our `scripts/03_activation_patching.py` is directly adapted from this.

---

#### 6. Conmy et al. (2023) — "Towards Automated Circuit Discovery for Mechanistic Interpretability"
**NeurIPS 2023** · https://arxiv.org/abs/2304.14997

Introduces ACDC (Automated Circuit Discovery), which uses activation patching and edge ablation to automatically prune the computational graph to essential components. Extends Wang et al.'s manual methodology.

**Relevance:** Potential automation of our sycophancy circuit identification. Cited alongside Wang et al. as our methodological foundation.

---

#### 7. Marks & Tegmark (2023) — "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"
**arXiv:** https://arxiv.org/abs/2310.06824

Demonstrates that true/false statements are linearly separable in LLM representation space. The "truth direction" is consistent across different types of factual claims and achievable via simple logistic regression probes with near-perfect accuracy. Validates the linear probe approach.

**Relevance:** Directly supports our probe-based truth direction detection. If truth has a linear representation, our Section 4.1 probes will work. Must-read before implementing probes.

---

#### 8. Burns et al. (2023) — "Discovering Latent Knowledge in Language Models Without Supervision"
**ICLR 2023** · https://arxiv.org/abs/2212.03827

Introduces Contrast-Consistent Search (CCS), an unsupervised method for finding truth directions without labeled data. Shows that truth representations satisfy logical consistency properties that enable discovery without ground truth labels.

**Relevance:** Alternative unsupervised probe approach. Could strengthen methodology. Cited alongside Marks & Tegmark as the representational foundation for our approach.

---

## Group 3: Inference-Time Intervention

The technical basis for our steering approach.

---

#### 9. Li et al. (2023) — "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
**NeurIPS 2023** · https://arxiv.org/abs/2306.03341

Applies steering specifically to improve truthfulness, identifying "truth directions" in attention head activations and showing ITI improves TruthfulQA performance without retraining. Most directly analogous prior work to our sycophancy steering.

**Relevance:** Closest prior work to our Phase 5 intervention. Our key claim is that targeting the *mechanistically-identified* circuit (rather than generic truth directions) enables greater specificity. Cited in Related Work as the closest predecessor.

---

#### 10. Turner et al. (2023) — "Activation Addition: Steering Language Models Without Optimization"
**arXiv:** https://arxiv.org/abs/2308.10248

Demonstrates that simple mean differences between contrastive prompts can steer LLM behavior across diverse tasks via residual stream addition. Validates the conceptual foundation of our `compute_steering_vector` implementation.

**Relevance:** The steering methodology we extend. Our key differentiator: we apply it to circuit-identified components rather than the full stream.

---

#### 11. Zou et al. (2023) — "Representation Engineering: A Top-Down Approach to AI Transparency"
**arXiv:** https://arxiv.org/abs/2310.01405

Introduces RepE as a unified framework for reading and controlling representations. Shows that high-level concepts (honesty, emotion, bias) have linear representations that can be identified and manipulated via reading vectors.

**Relevance:** Broader framework that contextualizes our steering approach. Cited alongside Turner et al. as the theoretical grounding for intervention via representation manipulation.

---

## Group 4: Background / Supporting

Not cited in the proposal's Related Work but useful for depth.

---

#### 12. Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits"
**Transformer Circuits Thread** · https://transformer-circuits.pub/2021/framework/index.html

The theoretical foundation for TransformerLens-style interpretability. Explains the residual stream as the communication channel, attention heads as independent circuits, and establishes the vocabulary we use throughout.

**Relevance:** Background theory. Read before diving into the patching methodology.

---

#### 13. Ouyang et al. (2022) — "Training language models to follow instructions with human feedback"
**arXiv:** https://arxiv.org/abs/2203.02155 (InstructGPT)

The original RLHF paper. Documents how human feedback training produces helpful assistants but acknowledges the tension with honesty. First acknowledgment that RLHF models can be sycophantic.

**Relevance:** Background on why sycophancy exists. Context for Base vs. Instruct comparison.

---

#### 14. Lin et al. (2022) — "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
**arXiv:** https://arxiv.org/abs/2109.07958

Source of our TruthfulQA factual sycophancy domain. Introduces the benchmark measuring whether models generate common misconceptions.

**Relevance:** Source dataset documentation.

---

## Reading Notes Template

For each paper, record:

1. **Key claim:** What is the main finding?
2. **Method:** What is the core technique?
3. **Connection to our work:** Which section of our paper does this support?
4. **Gap it leaves:** What does our work add?

---

## Search Queries for 2025-2026 Updates

```
sycophancy LLM mechanistic interpretability 2025
activation patching transformer circuits 2025
steering vectors sycophancy 2025
linear probes truth representations LLM 2025
RLHF sycophancy alignment 2025
```

---

*Aligned with Research_Proposal.md Section 2 (Related Work) — updated 2026-02-20*
