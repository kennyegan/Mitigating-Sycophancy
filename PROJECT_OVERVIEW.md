# Project Overview

## Objective

Upgrade the repository from working research code to senior-research standard:

- stronger methodology
- leakage-safe evaluation
- checkpointable long runs
- explicit artifact contracts
- conservative claim posture until corrected reruns are complete

## Current Upgrade State (March 3, 2026)

### Milestone 1: Methodology Hardening

- Completed: shared evaluation math (`src/analysis/evaluation.py`) wired into baseline script
- Completed: length-normalized confidence metrics and SciPy binomial fallback
- Completed: deterministic `sample_id` and `--randomize-positions` path

### Milestone 2: Probe Pipeline Redesign

- Completed: `02_train_probes.py` with:
  - `neutral_transfer` (default, claim-bearing)
  - `mixed_diagnostic` (diagnostic)
- Completed: GroupKFold leakage-safe grouping by `sample_id`
- Completed: transfer/pattern CIs
- Completed: `02b_probe_control.py` aligned to unified schema

### Milestone 3: Reproducibility + SLURM Reliability

- Completed: steering checkpoint/resume and per-condition persistence
- Completed: explicit run manifests in baseline/probe/patching/ablation/steering scripts
- Completed: SLURM normalization to submit-time resource configuration (`slurm/submit_all.sh`)
- Completed: artifact existence checks in SLURM jobs
- Completed: steering checkpoint-aware job script behavior

### Milestone 4: Capability Evaluation Upgrades

- Completed: MMLU robust choice scoring (tokenization variants)
- Completed: GSM8k strict normalized numeric extraction from generation
- Completed: capability CIs and retention CIs in ablation/steering

### Milestone 5: Test Expansion

- Added: evaluation math + statistical fallback tests
- Added: probe label/leakage tests
- Added: CLI contract checks for new flags
- Added: schema-contract regression checks
- Added: steering checkpoint/resume smoke test
- Added: deterministic ID and manifest smoke tests

### Milestone 6: Full Corrected Rerun Matrix

- Configured: SLURM Jobs 1–13 include reruns + consolidated manifest
- Pending execution: full cluster reruns and artifact completion

### Milestone 7: Paper + Docs Synchronization

- Updated: `paper.md` now marks numerical claims as provisional until reruns complete
- Updated: `README.md`, `QUICKSTART.md`, `scripts/README.md` to new semantics
- Updated: this overview to milestone/gate framing

## Completion Gates

- Gate A: core refactors + tests in place (code complete; test execution pending environment)
- Gate B: small dry reruns + schema-valid outputs (pending execution)
- Gate C: full SLURM reruns + manifests (pending execution)
- Gate D: finalize paper numbers after manifest-backed confirmation (pending)
