# Engineering Notes

## Issues Encountered and Fixed

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Probe conclusions depended on mixed-format training | Probe labels confounded with prompt condition in mixed data | Added explicit `--analysis-mode` with claim-bearing `neutral_transfer` default and diagnostic `mixed_diagnostic` |
| Probe fold leakage risk in mixed mode | Paired neutral/biased samples could cross folds without grouping | Added deterministic `sample_id` and GroupKFold grouping by `sample_id` |
| Dataset class-balance artifact in synthetic domains | TruthfulQA/GSM8k answer positions were fixed | Added `--randomize-positions` and recorded per-sample randomization metadata |
| Confidence metric length bias | Raw sequence log-probability penalized longer targets | Switched to length-normalized confidence metrics (per-token avg log-prob), with confidence-filtered stats reported as secondary |
| Binomial test API drift across SciPy versions | `binomtest` vs `binom_test` incompatibility | Added compatibility wrapper with modern-first fallback |
| Long steering sweeps lost progress on interruption | No checkpoint persistence | Added per-condition checkpoint JSON + `--resume-from-checkpoint` support |
| SLURM resource directives were inconsistent | Dynamic/embedded directives were brittle across scripts | Removed dynamic `#SBATCH` patterns; centralized resources in `slurm/submit_all.sh` |
| Jobs could succeed with missing outputs | No artifact contract checks | Added non-zero artifact checks across SLURM jobs, including steering final+checkpoint JSON checks |
| `KeyError: 'blocks.11.attn.hook_result'` | `use_attn_result=False` by default in TransformerLens for Llama-3 | Set `model.cfg.use_attn_result = True; model.setup()` after loading in `sycophancy_model.py` |
| `TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'use_attn_result'` | Kwarg passed directly to `from_pretrained`, leaked to HuggingFace constructor | Moved to post-load config mutation |
| `RuntimeError: expanded size (118) must match existing size (116)` | Head patch hook copied full sequence dimension; neutral/biased sequences differ in length | Changed to `n = min(act.shape[1], clean_act.shape[1]); activation[0, :n, h, :] = clean_act[0, :n, h, :]` |

## Cluster Configuration

| Setting | Value |
|---------|-------|
| Cluster | Unity HPC (UMass) |
| Partition | `gpu` |
| Account | `pi_larsonj_wit_edu` |
| GPU | A100 (`--gres=gpu:a100:1`) |
| Conda module | `conda/latest` (miniforge3-24.7.1) |
| Environment | `sycophancy-lab` |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
