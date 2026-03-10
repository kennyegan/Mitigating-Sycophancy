# Rerun Status Log (March 9, 2026)

All runs complete. Manifest validated: `missing_count: 0`.

| Experiment Track | Status | Artifact |
|---|---|---|
| Data regeneration (`--randomize-positions`) | **Complete** | `data/processed/master_sycophancy_balanced.jsonl` |
| Baseline rerun (instruct/base) | **Complete** (Mar 4) | `results/baseline_llama3_summary.json`, `results/baseline_llama3_base_summary.json` |
| Probes (`neutral_transfer`, `mixed_diagnostic`) | **Complete** (Mar 4) | `results/probe_results_neutral_transfer.json`, `results/probe_results_mixed_diagnostic.json` |
| Probe-control balanced rerun | **Complete** (Mar 4) | `results/probe_control_balanced_results.json` |
| Patching/head ranking rerun | **Complete** (Mar 4) | `results/patching_heatmap.json`, `results/head_importance.json` |
| Ablation rerun (top-3 heads) | **Complete** (Mar 6) | `results/head_ablation_results.json` |
| Ablation rerun (top-10, full GSM8k N=1319) | **Complete** (Mar 9) | `results/top10_ablation_full_gsm8k.json` |
| Steering rerun (checkpoint/resume) | **Complete** (Mar 7) | `results/steering_results.json` + checkpoint JSON |
| Consolidated manifest | **Complete** (Mar 9) | `results/full_rerun_manifest.json` |
