# Compute Requirements

This document records what is required for the two modes of verification this
supplement supports.

## A. Fast documentation verification (this supplement)

- **Hardware:** any laptop or desktop. CPU-only is fine.
- **GPU:** not required.
- **Model downloads:** not required.
- **Code execution:** not required.
- **Network:** not required (all files are local Markdown and JSON).
- **Expected time:** minutes — open the Markdown files and the seven result
  JSONs in any reader.

There is nothing to install for this mode.

## B. Full experimental rerun

This mode requires the future code release.

### Compute envelope (paper-reported)

- **Total GPU compute:** approximately **120 A100 GPU-hours** for the full
  Llama-3 pipeline (baseline evaluation, probe sweep, activation patching,
  head ablation, DPO across three seeds, SFT, OOD evaluation, free-form
  generation, plus the cross-architecture replications on Mistral-7B and
  Qwen-14B).
- **Judge-API cost:** approximately **\$8** for the free-form judge scoring of
  300 transcripts (150 baseline + 150 DPO) at the configuration described in
  `prompts/JUDGE_SYSTEM_PROMPT.md` (temperature 0, max_tokens 1000, single
  pass per transcript).

### Models

| Model | Role |
|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | primary model (probes, patching, ablation, DPO/SFT, OOD, free-form) |
| `mistralai/Mistral-7B-Instruct-v0.1` | cross-architecture replication |
| `Qwen/Qwen2.5-14B-Instruct` | cross-scale replication |

### Software

- TransformerLens with `dtype=float16` for the probe / patching / ablation
  experiments (paper-reported).
- Standard PyTorch + Hugging Face `transformers` + `peft` for DPO and SFT
  training.
- Standard Anthropic SDK for the free-form judge.

### Public datasets

- `Anthropic/model-written-evals` (sycophancy split: generic + nlp_survey +
  political_typology subcategories).
- `TruthfulQA` (factual sycophancy domain).
- `GSM8K` (reasoning sycophancy domain and capability check).
- `MMLU` (capability check).

### What the full rerun covers

- Baseline forced-choice evaluation on the three models.
- Layer-and-head probe training (5-fold CV under the neutral-transfer design;
  randomized answer positions) at 32 residual-stream layers.
- Activation patching (layer-and-position scan, then head-level patching
  within the top-5 layers).
- Zero-ablation top-10 head intervention with MMLU and GSM8K capability
  retention checks.
- A 5-resample patching bootstrap (N=100 each, seeds 42/123/456/789/1011).
- DPO training across seeds 100/200/300 with regenerated preference pairs
  per seed.
- SFT training on identical preference data (for the DPO-vs-SFT comparison).
- OOD Protocol A (rephrased templates + 50 hand-crafted questions, N=450) and
  Protocol B (held-out Anthropic subcategories, N=1,000).
- Free-form generation and 5-dimension judge scoring of 150 baseline and 150
  DPO transcripts, plus a 5,000-iteration BCa bootstrap on baseline-vs-DPO
  deltas.

### Resource items intentionally not estimated here

- Per-stage VRAM, RAM, and disk breakdowns: **not measured** as separate line
  items in the working repository's manifest. We do not guess; we recommend
  reviewers consult the upstream model cards and the standard memory profile
  for `dtype=float16` inference at these scales as a starting point.
- Wall-clock time per stage on hardware other than A100: **not available in
  this supplement.**

### Anonymity-preserving notes

The paper's reproducibility statement and the working repository's full-rerun
manifest record the random seeds, the N values, and the exact public dataset
versions used. The trained DPO adapters and the cached per-prompt evaluation
outputs are not included in this anonymous supplement; they will be released
after acceptance.
