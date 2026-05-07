#!/usr/bin/env python3
"""
Collect and validate full-pipeline artifacts into one manifest JSON.
"""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple


DEFAULT_EXPECTED: List[Tuple[str, str]] = [
    ("dataset_balanced", "data/processed/master_sycophancy_balanced.jsonl"),
    ("baseline_instruct", "results/baseline_llama3_summary.json"),
    ("baseline_base", "results/baseline_llama3_base_summary.json"),
    ("probes_neutral_transfer", "results/probe_results_neutral_transfer.json"),
    ("probes_mixed_diagnostic", "results/probe_results_mixed_diagnostic.json"),
    ("probe_control_balanced", "results/probe_control_balanced_results.json"),
    ("patching_heatmap", "results/patching_heatmap.json"),
    ("head_importance", "results/head_importance.json"),
    ("head_ablation", "results/head_ablation_results.json"),
    ("corrected_ablation", "results/corrected_ablation_results.json"),
    ("top10_ablation", "results/top10_ablation_results.json"),
    ("top10_ablation_full_gsm8k", "results/top10_ablation_full_gsm8k.json"),
    ("steering", "results/steering_results.json"),
    ("steering_per_source", "results/steering_per_source_analysis.json"),
    ("dpo_model", "results/dpo_model/adapter_config.json"),
    ("dpo_training_metrics", "results/dpo_training_metrics.json"),
    ("dpo_eval", "results/dpo_eval_results.json"),
]


def get_git_hash(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rerun artifact manifest.")
    parser.add_argument("--output", type=str, default="results/full_rerun_manifest.json",
                        help="Output manifest path")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Do not fail if expected artifacts are missing")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _first_present(obj: Dict[str, Any], dotted_paths: List[str], default: Any = None) -> Any:
    for dotted in dotted_paths:
        current = obj
        ok = True
        for part in dotted.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok:
            return current
    return default


def collect_key_metrics(root: Path, label: str, path: Path) -> Dict[str, Any]:
    if path.suffix != ".json":
        return {}
    data = _read_json(path)
    if not data:
        return {}

    metrics = {}
    if "baseline" in label:
        metrics["sycophancy_rate"] = _first_present(
            data,
            [
                "summary.overall.sycophancy_rate",
                "overall.sycophancy_rate",
                "overall_sycophancy_rate",
            ],
        )
    if "probe" in label:
        metrics["analysis_mode"] = _first_present(data, ["analysis_mode", "metadata.analysis_mode"])
        metrics["best_layer"] = _first_present(
            data,
            [
                "per_position_summary.final.best_layer",
                "summary.best_layer",
            ],
        )
    if "ablation" in label:
        metrics["baseline_sycophancy"] = _first_present(
            data,
            ["conditions.baseline.sycophancy.overall_sycophancy_rate"],
        )
    if "steering" in label:
        metrics["baseline_sycophancy"] = _first_present(
            data,
            ["conditions.baseline.sycophancy.overall_sycophancy_rate"],
        )
        metrics["schema_version"] = _first_present(data, ["schema_version"])
    return {k: v for k, v in metrics.items() if v is not None}


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    artifacts = []
    missing = []
    for label, rel_path in DEFAULT_EXPECTED:
        path = root / rel_path
        exists = path.exists()
        non_empty = exists and path.stat().st_size > 0
        status = "ok" if non_empty else "missing"
        if status != "ok":
            missing.append(rel_path)
        artifact = {
            "label": label,
            "path": rel_path,
            "exists": exists,
            "non_empty": non_empty,
            "status": status,
        }
        artifact["key_metrics"] = collect_key_metrics(root, label, path) if non_empty else {}
        artifacts.append(artifact)

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": get_git_hash(root),
        "artifact_count": len(artifacts),
        "missing_count": len(missing),
        "artifacts": artifacts,
    }

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest: {output}")
    if missing:
        print("Missing artifacts:")
        for rel in missing:
            print(f"  - {rel}")

    if missing and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
