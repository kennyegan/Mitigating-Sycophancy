# Mitigating-Sycophancy

# Sycophancy Alignment Initiative

> Building the first open-source benchmark and mitigation toolkit for detecting sycophancy in large language models.

## 🧠 Overview

**Sycophancy** is a critical failure mode in language models, where the model prioritizes agreement with the user over factual accuracy or objective reasoning. This behavior undermines trustworthiness, especially in high-stakes applications like education, finance, and healthcare.

This initiative is the first open-source research effort to systematically **quantify, analyze, and mitigate sycophancy** across different architectures, prompt styles, and model sizes. We aim to make sycophancy as measurable and actionable as truthfulness or bias.

## 🎯 Goals

1. **Define Sycophancy Operationally**  
   Formalize what constitutes sycophantic behavior using reproducible prompt templates and response criteria.

2. **Build a Benchmark Suite**  
   Release a growing set of evaluation prompts, datasets, and sycophancy-sensitive tasks (multiple-choice, open-ended, contradiction pairs).

3. **Develop Measurement Tools**  
   Use **path patching**, **task vectors**, and **causal tracing** to isolate circuits and attention heads responsible for sycophantic behavior.

4. **Compare Across Models**  
   Provide sycophancy metrics for models like GPT-2, LLaMA, Mistral, and GPT-3.5/4 via OpenAI APIs or open checkpoints.

5. **Explore Mitigation Techniques**  
   Test methods such as reinforcement learning, adversarial prompting, or architecture-level tweaks to reduce sycophantic output tendencies.

6. **Create a Community Hub**  
   Publish results on arXiv, host community discussions, and invite contributions from alignment researchers, model evaluators, and LLM developers.

## 🔬 Current Status

- ✅ **Phase 1**: GPT-2 tracing experiment complete and submitted to arXiv.
- 🔄 **Phase 2**: Benchmark prompt templates under development (truth-vs-flattery, contradiction resilience, perspective disagreement).
- 🔬 **Phase 3**: Expansion to multi-model evaluation and cross-lingual comparison.

## 🛠 Repository Structure

```bash
sycophancy-align/
├── data/
│   └── prompt_templates/       # Standardized prompts for probing sycophancy
├── analysis/
│   └── causal_tracing/         # Code for path patching and neuron attribution
├── models/
│   └── eval/                   # Model inference and sycophancy scoring scripts
├── results/
│   └── figures/                # Visualizations of activation patterns, scores
├── docs/
│   └── roadmap.md              # Multi-phase development plan
└── README.md
