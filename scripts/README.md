# Script Reference

## Pipeline Order

1. `00_data_setup.py`
2. `01_run_baseline.py`
3. `02_train_probes.py` / `02b_probe_control.py`
4. `03_activation_patching.py`
5. `04_head_ablation.py`
6. `05_representation_steering.py`
7. `99_collect_result_manifest.py`

## Key Interface Changes

### `00_data_setup.py`

- Added `--randomize-positions`
- Writes deterministic `sample_id` on each record
- Stores randomization metadata per sample

### `02_train_probes.py`

- Added `--analysis-mode {neutral_transfer,mixed_diagnostic}`
- Default is `neutral_transfer` (claim-bearing)
- Outputs schema fields:
  - `schema_version`
  - `analysis_mode`
  - `split_definition`
  - CI fields for transfer/pattern rates

### `02b_probe_control.py`

- Aligned to `02_train_probes.py` neutral-transfer semantics and schema
- Keeps dedicated probe-control entrypoint for SLURM workflows

### `05_representation_steering.py`

- Added checkpoint/resume controls:
  - `--checkpoint-path`
  - `--resume-from-checkpoint`
  - `--save-every-condition` / `--no-save-every-condition`
- Saves partial JSON progress after each condition by default
- Uses strict capability scoring + CI reporting

## Consolidated Manifest

`99_collect_result_manifest.py` validates expected artifacts and writes:

- `results/full_rerun_manifest.json`

Usage:

```bash
python scripts/99_collect_result_manifest.py --output results/full_rerun_manifest.json
```
