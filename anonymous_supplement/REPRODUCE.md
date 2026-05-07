# Reproduction Notes

This package supports **inspection-only verification** of the paper's reported
numbers. There are three distinct levels of verification, listed below in order
of effort.

## 1. Documentation-only verification (this supplement, no code)

What you can do entirely from this directory, with no code execution and no GPU:

- Open `README.md` and `MANIFEST.md` to see how each paper claim maps onto a
  result file.
- Open the seven result JSONs in `results/` to read the aggregate values that
  back each claim.
- Open `prompts/` to inspect the free-form benchmark prompts (150 multi-turn
  conversation seeds across five domains), the judge system prompt, and the OOD
  template summary.
- Open `rubrics/freeform_judge_rubric.json` to inspect the five-dimension
  scoring rubric used by the LLM judge.

This is the intended verification mode for the supplementary review. Expected
time: minutes. No special environment is required beyond a JSON-aware viewer.

## 2. Cached aggregate verification (requires future code release)

After the code is released, reviewers can re-run the aggregation scripts on the
cached per-prompt outputs to confirm that the aggregate numbers in this
supplement were produced by the same pipeline. This step does not require GPU
inference or model weights — only the cached evaluation outputs and the
aggregation scripts that produce summaries from them.

What it would verify:

- That the Llama-3 baseline 28.0% / 82.4% / 1.6% / 0.0% rates aggregate from a
  cached per-prompt output file with `samples_evaluated = 1493`.
- That the multi-seed DPO summary statistics (`57.1 +/- 2.8%` opinion sycophancy,
  `40.2 +/- 2.9%` GSM8k, `62.9 +/- 0.1%` MMLU) aggregate over seeds
  100/200/300.
- That the 5,000-iteration free-form bootstrap CIs reproduce when the same
  judge scores are passed through the aggregation script.

Expected time after code release: minutes to a few hours, CPU-only.

## 3. Full experimental rerun (requires future code release + significant compute)

A full rerun reproduces the entire pipeline end-to-end, starting from raw model
weights and unprocessed data:

- Baseline forced-choice evaluation on Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.1,
  Qwen2.5-14B-Instruct.
- Layer-and-head probe training (5-fold CV; neutral-transfer design with
  randomized answer positions) at 32 residual-stream layers.
- Activation patching (layer-and-position scan, then head-level patching within
  the top-5 layers) with TransformerLens at `dtype=float16`.
- Zero-ablation top-10 head intervention with capability-retention checks
  (MMLU, GSM8k).
- 5-resample patching bootstrap (seeds 42/123/456/789/1011, N=100 each).
- DPO training across three seeds (100, 200, 300) with regenerated preference
  data per seed.
- SFT training on identical preference data for the comparison.
- OOD evaluation under Protocol A (rephrased templates + 50 hand-crafted
  questions, N=450) and Protocol B (held-out Anthropic subcategories, N=1,000).
- Free-form generation and judge scoring of 150 baseline and 150 DPO transcripts.

Expected requirements (see `COMPUTE.md` for detail):

- Approximately **120 A100 GPU-hours** for the full pipeline.
- Approximately **\$8** in judge-API costs for the free-form scoring.
- Models: `meta-llama/Meta-Llama-3-8B-Instruct`,
  `mistralai/Mistral-7B-Instruct-v0.1`, `Qwen/Qwen2.5-14B-Instruct`.
- Public datasets: `Anthropic/model-written-evals` (sycophancy split),
  `TruthfulQA`, `GSM8K`, `MMLU`.

The training and evaluation code, the data preparation pipeline, and the
trained DPO adapters will be released after acceptance.

## Distinction summary

| Mode | Compute | Code release needed | What it verifies |
|---|---|---|---|
| 1. Documentation-only | None | No | Aggregate values in this supplement match the paper |
| 2. Cached aggregate | CPU-only | Yes | Aggregation scripts produce the supplement's numbers from cached per-prompt outputs |
| 3. Full rerun | ~120 A100-hours, ~\$8 API | Yes | End-to-end reproduction from raw weights + public data |

For the supplementary review, mode 1 is the intended use of this package.
