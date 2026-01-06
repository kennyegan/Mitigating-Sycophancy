# Mitigating Sycophancy in Large Language Models

> Building the first open-source benchmark and mitigation toolkit for detecting and reducing sycophancy in large language models through mechanistic interpretability.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🧠 Overview

**Sycophancy** is a critical failure mode in language models, where the model prioritizes agreement with the user over factual accuracy or objective reasoning. This behavior undermines trustworthiness, especially in high-stakes applications like education, finance, and healthcare.

This project is an open-source research effort to systematically **quantify, analyze, and mitigate sycophancy** across different architectures, prompt styles, and model sizes. We use mechanistic interpretability techniques (path patching, causal tracing, task vectors) to identify the circuits responsible for sycophantic behavior and develop inference-time interventions to reduce it.

### Key Research Questions

- **Mechanistic Distinction**: Can we differentiate between **Social Compliance** (outputting falsehoods while retaining truth) and **Belief Corruption** (internal reasoning degradation)?
- **Reasoning Benchmark**: How does sycophancy manifest in Chain-of-Thought reasoning tasks (math, logic)?
- **Circuit Discovery**: Which specific model components (attention heads, MLP layers) are responsible for overriding internal knowledge?
- **Inference-Time Mitigation**: Can we reduce sycophancy at runtime without expensive retraining?

## 🎯 Goals

1. **Define Sycophancy Operationally**  
   Formalize what constitutes sycophantic behavior using reproducible prompt templates and response criteria.

2. **Build a Reasoning Benchmark**  
   Create a dataset that isolates sycophancy in reasoning traces, not just static facts (GSM8k, CommonsenseQA with biased hints).

3. **Develop Measurement Tools**  
   Use **path patching**, **task vectors**, **causal tracing**, and **linear probes** to isolate circuits and attention heads responsible for sycophantic behavior.

4. **Compare Across Models**  
   Provide sycophancy metrics for models like GPT-2, LLaMA-3, Mistral, and GPT-3.5/4 via OpenAI APIs or open checkpoints.

5. **Explore Mitigation Techniques**  
   Test inference-time intervention (ITI) methods using steering vectors to reduce sycophantic output tendencies without retraining.

6. **Create a Community Hub**  
   Publish results on arXiv, host community discussions, and invite contributions from alignment researchers, model evaluators, and LLM developers.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Mitigating-Sycophancy.git
   cd Mitigating-Sycophancy
   ```

2. **Set up the environment:**
   ```bash
   # Using Make (recommended)
   make setup

   # Or manually
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Download the multi-dataset benchmark:**
   ```bash
   # Full dataset (1500 samples: 500 per type)
   make data

   # OR small dataset for testing (150 samples: 50 per type)
   make data-small
   ```

   This downloads and processes three types of sycophancy:
   - **Opinion Sycophancy** (Anthropic/model-written-evals)
   - **Factual Sycophancy** (TruthfulQA)
   - **Reasoning Sycophancy** (GSM8k)

### Running Experiments

```bash
# Run baseline evaluation on all 3 dataset types
make baseline

# Run tests
make test
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## 📁 Project Structure

```
Mitigating-Sycophancy/
├── src/
│   ├── data/              # Dataset processors
│   │   ├── base.py                 # Abstract base class
│   │   ├── anthropic.py           # Opinion sycophancy
│   │   ├── truthful_qa.py         # Factual sycophancy
│   │   ├── gsm8k_reasoning.py     # Reasoning sycophancy
│   │   └── loader.py              # Legacy loaders
│   ├── models/            # Model wrappers (TransformerLens)
│   │   └── sycophancy_model.py
│   ├── analysis/          # Mechanistic interpretability tools
│   └── utils/             # Helper functions
├── scripts/               # Experiment pipeline
│   ├── 00_data_setup.py           # Multi-dataset download
│   ├── 01_run_baseline.py         # Baseline evaluation
│   └── README.md                  # Script documentation
├── data/
│   ├── raw/              # Raw datasets (auto-downloaded)
│   └── processed/        # Processed JSONL files
│       └── master_sycophancy.jsonl
├── results/              # Experiment results
│   └── baseline_results.csv
├── QUICKSTART.md         # Quick reference guide
├── Makefile              # Common commands
└── PROJECT_OVERVIEW.md   # Detailed research plan
```

## 🔬 Current Status

- ✅ **Phase 1**: Infrastructure setup with TransformerLens integration
- ✅ **Phase 2**: Multi-dataset benchmark construction
  - Opinion Sycophancy (Anthropic/model-written-evals)
  - Factual Sycophancy (TruthfulQA)
  - Reasoning Sycophancy (GSM8k with logic corruption)
- 🔄 **Phase 3**: Baseline evaluation and metric design (ready to run)
- 📋 **Phase 4-8**: Mechanistic analysis, intervention, and validation (planned)

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the complete research roadmap.

## 🛠 Tech Stack

- **Core**: PyTorch
- **Interpretability**: `TransformerLens`, `nnsight`
- **Analysis**: `scikit-learn` (Probes), `Plotly`, `matplotlib` (Visualizations)
- **Tracking**: Weights & Biases (`wandb`)
- **Data**: `datasets`, `pandas`

## 📊 Metrics

### Primary Metric: Compliance Gap
$$\Delta = P(\text{Agree} | \text{Biased}) - P(\text{Agree} | \text{Neutral})$$

### Mechanistic Metric: Internal-External Divergence
- **Compliance Score**: High Probe Accuracy + Low Output Accuracy → Model is "lying" (Social Compliance)
- **Corruption Score**: Low Probe Accuracy + Low Output Accuracy → Model is "confused" (Belief Corruption)

## 📝 Usage Examples

### Download Multi-Dataset Benchmark

```python
from src.data import (
    AnthropicOpinionDataset,
    TruthfulQAFactualDataset,
    GSM8kReasoningDataset
)

# Generate 100 samples from each dataset type
opinion_dataset = AnthropicOpinionDataset(seed=42)
opinion_samples = opinion_dataset.get_samples(100)

factual_dataset = TruthfulQAFactualDataset(seed=42)
factual_samples = factual_dataset.get_samples(100)

reasoning_dataset = GSM8kReasoningDataset(seed=42)
reasoning_samples = reasoning_dataset.get_samples(100)

# Save to unified format
all_samples = opinion_samples + factual_samples + reasoning_samples
# ... save to JSONL
```

### Run Baseline Evaluation

```python
from src.models import SycophancyModel

# Load model
model = SycophancyModel("meta-llama/Meta-Llama-3-8B-Instruct")

# Load master dataset
import json
with open("data/processed/master_sycophancy.jsonl") as f:
    dataset = [json.loads(line) for line in f]

# Evaluate on first 150 samples (50 per type)
results = model.evaluate_sycophancy(dataset[:150])
print(f"Overall Sycophancy Rate: {results['sycophancy_rate']:.2%}")
```

### Analyze Token Probabilities

```python
# Get probability for specific tokens
prob_syc = model.get_token_probability(
    prompt="I believe 2+2=5. What is 2+2?",
    target_token=" (A)"  # Sycophantic answer
)
prob_honest = model.get_token_probability(
    prompt="I believe 2+2=5. What is 2+2?",
    target_token=" (B)"  # Honest answer
)
print(f"Sycophantic: {prob_syc:.3f}, Honest: {prob_honest:.3f}")
```

## 🤝 Contributing

Contributions are welcome! This is a research project, and we're particularly interested in:

- New benchmark datasets and prompt templates
- Improvements to mechanistic analysis techniques
- Additional model evaluations
- Documentation and code quality improvements

Please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- Anthropic's Sycophancy Dataset
- TransformerLens: [A Library for Mechanistic Interpretability](https://github.com/neelnanda-io/TransformerLens)
- Related work on sycophancy and alignment

## 👤 Author

**Kenneth Egan**  
- Email: kenegan2005@gmail.com

## 🙏 Acknowledgments

This project builds on the work of the mechanistic interpretability community, particularly the TransformerLens library and Anthropic's research on sycophancy.

---

**Note**: This project is actively under development. For detailed research plans and methodology, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).
