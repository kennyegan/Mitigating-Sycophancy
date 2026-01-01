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

3. **Download the data:**
   ```bash
   make data
   # Or: python src/data/download_anthropic.py
   ```

### Running Experiments

```bash
# Run baseline sycophancy evaluation
make baseline
# Or: python scripts/01_check_baseline.py

# Run tests
make test
```

## 📁 Project Structure

```
Mitigating-Sycophancy/
├── src/
│   ├── data/              # Data loading and downloading utilities
│   │   ├── download_anthropic.py
│   │   └── loader.py
│   ├── models/            # Model wrappers and evaluation scripts
│   ├── analysis/          # Mechanistic interpretability analysis
│   └── utils/             # Helper functions
├── scripts/               # Main experiment scripts
│   └── 01_check_baseline.py
├── notebooks/             # Jupyter notebooks for exploration
├── data/
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── results/              # Experiment results and figures
├── configs/              # Configuration files
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── Makefile              # Common commands
└── PROJECT_OVERVIEW.md   # Detailed research plan
```

## 🔬 Current Status

- ✅ **Phase 1**: Infrastructure setup with TransformerLens integration
- 🔄 **Phase 2**: Benchmark construction (Reasoning-Sycophancy dataset)
- 🔬 **Phase 3**: Baseline evaluation and metric design
- 📋 **Phase 4-8**: Mechanistic analysis, intervention, and validation (in progress)

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

### Basic Sycophancy Evaluation

```python
from src.models import SycophancyModel
from src.data.loader import load_anthropic_dataset

# Load model
model = SycophancyModel("meta-llama/Meta-Llama-3-8B-Instruct")

# Load dataset
dataset = load_anthropic_dataset("data/processed/anthropic_sycophancy.jsonl")

# Evaluate sycophancy rate
sycophancy_rate = model.evaluate_sycophancy(dataset)
print(f"Sycophancy Rate: {sycophancy_rate:.2%}")
```

### Causal Tracing Analysis

```python
from src.analysis.causal_tracing import trace_sycophancy_circuit

# Identify which layers/heads are responsible for sycophancy
circuit = trace_sycophancy_circuit(model, dataset)
print(f"Sycophancy Circuit: {circuit}")
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
