# Mitigating Sycophancy in Large Language Models via Task-Vector Path-Patching

A comprehensive research pipeline for studying and mitigating sycophantic behavior in large language models using task vectors and path patching techniques.

## 🎯 Overview

This pipeline enables researchers to:
- **Quantify sycophancy** in open-weight transformer models
- **Generate task vectors** from truthful vs sycophantic completions  
- **Apply path patching** to steer model behavior using TransformerLens
- **Evaluate effectiveness** with comprehensive metrics and visualizations

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup directories and environment
python setup.py

# Download datasets
bash download_data.sh
```

### 2. Generate Task Vector
```bash
python vectors/generate_task_vector.py \
    --dataset anthropic \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --output vectors/task_vector.pt \
    --max_samples 100
```

### 3. Run Evaluation
```bash
python eval/eval_pipeline.py \
    --config configs/base.yaml \
    --task_vector vectors/task_vector.pt \
    --max_samples 50 \
    --wandb
```

### 4. Analyze Results
```bash
jupyter notebook notebooks/01_inspect_task_vector.ipynb
```

## 📁 Architecture

### Core Components

- **🤖 Model Wrapper** (`models/model_wrapper.py`): Unified interface for HuggingFace models with TransformerLens hooks
- **📊 Dataset Loaders** (`datasets/`): Process Anthropic Sycophancy and TruthfulQA datasets
- **🧠 Task Vector Generation** (`vectors/`): Extract and compute task vectors from model activations
- **🔧 Path Patching** (`patching/`): Apply interventions at specific model layers
- **📈 Evaluation Pipeline** (`eval/`): Comprehensive evaluation with metrics and comparison
- **📓 Analysis Notebooks** (`notebooks/`): Interactive visualization and analysis tools

### Pipeline Flow

```
[Load Dataset] → [Generate Task Vector] → [Apply Patching] → [Evaluate Results]
      ↓                     ↓                    ↓               ↓
[Contrast Pairs] → [Activation Extraction] → [Layer Intervention] → [Metrics & Plots]
```

## 🛠️ Configuration

The pipeline uses a central configuration system in `configs/base.yaml`:

```yaml
# Model Configuration
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
  device: "cuda"
  load_in_8bit: true

# Dataset Configuration  
dataset:
  name: "anthropic"
  max_samples: 1000

# Vector Configuration
vector:
  layer: 20
  position: "resid_mid"
  method: "mean_diff"

# Patching Configuration
patching:
  layers: [15, 20, 25]
  coefficients: [0.5, 1.0, 1.5, 2.0]
  ablation_type: "addition"
```
## 📖 Usage Examples
### Generate Completions
```bash
python scripts/generate_completions.py \
    --dataset anthropic \
    --max_samples 50 \
    --save_logits \
    --save_hidden_states
```

### Custom Task Vector
```bash
python vectors/generate_task_vector.py \
    --dataset truthfulqa \
    --layer 25 \
    --method pca \
    --output vectors/truthful_pca_vector.pt
```

### Ablation Study
```bash
python eval/eval_pipeline.py \
    --config configs/base.yaml \
    --task_vector vectors/task_vector.pt \
    --max_samples 100
```

## 📊 Evaluation Metrics

The pipeline computes comprehensive metrics:

- **Likelihood Analysis**: Truthful vs sycophantic completion preferences
- **Preference Rates**: How often model prefers truthful responses
- **Activation Analysis**: Norm changes and cosine similarities
- **Patching Effectiveness**: Improvement over baseline performance
- **Ablation Results**: Performance across layers and coefficients

## 🎨 Visualization Tools

The analysis notebook provides:
- Task vector property visualization
- Before/after completion comparison
- Ablation study heatmaps
- Projection analysis plots
- Interactive widgets for exploration

## 🔧 API Reference

### Model Wrapper
```python
from models.model_wrapper import ModelWrapper

model = ModelWrapper(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    device="cuda",
    load_in_8bit=True
)

# Generate without patching
completion = model.generate_unpatched(prompt)

# Generate with vector injection
completion = model.generate_with_vector_injection(
    prompt, vector, layer=20, coefficient=1.0
)
```

### Dataset Loading
```python
from datasets.load_anthropic import load_anthropic_dataset

dataset = load_anthropic_dataset(max_samples=100)
contrast_pairs = dataset.create_contrast_pairs()
```

### Task Vector Generation
```python
from vectors.generate_task_vector import TaskVectorGenerator

generator = TaskVectorGenerator(model, layer=20, method="mean_diff")
task_vector = generator.compute_task_vector_from_dataset(dataset)
```

### Path Patching
```python
from patching.patch_utils import PathPatcher, PatchConfig

patcher = PathPatcher(model.model)
patch_config = PatchConfig(
    layer=20,
    position="resid_mid", 
    vector=task_vector,
    coefficient=1.0
)

generated_text, results = patcher.patch_and_generate(
    [patch_config], prompt
)
```

## 📋 Requirements

### Hardware
- GPU with 16GB+ VRAM (recommended)
- 32GB+ RAM for large datasets
- 50GB+ disk space for models and data

### Software
- Python 3.10+
- PyTorch 2.0+
- TransformerLens 1.6+
- HuggingFace Transformers 4.40+

See `requirements.txt` for complete dependencies.

## 🧪 Supported Models

- **Mistral-7B-Instruct** (default)
- **Llama-3-8B-Instruct**
- **Mixtral-8x7B-Instruct**
- Any HuggingFace model compatible with TransformerLens

## 📚 Datasets

- **Anthropic Sycophancy**: Sycophantic vs truthful responses
- **TruthfulQA**: Truthful vs false completions
- **FEVER**: Fact verification (for generalization testing)

## 🔬 Research Applications

This pipeline enables research on:
- **Sycophancy detection** in language models
- **Intervention techniques** for behavior modification
- **Activation analysis** and interpretability
- **Task vector effectiveness** across model architectures
- **Robustness evaluation** of mitigation techniques

## 🤝 Contributing

We welcome contributions! Please see our development guidelines:

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for model interventions
- [Steering Vectors](https://github.com/steering-vectors/steering-vectors) for vector utilities
- [Anthropic's Model-Written Evals](https://github.com/anthropics/evals) for sycophancy data
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA) for truthfulness evaluation

## 📞 Contact

For questions or collaboration opportunities, please open an issue or reach out to the research team.

---

**Happy researching! 🧠🤖**
