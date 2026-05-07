"""
Parse the timed-out Qwen-14B head_ablation stdout log into a JSON artifact.

The main job (56532117) timed out at 48h before scripts/04_head_ablation.py
could write its final JSON output. This parser extracts the per-condition
results that DID complete from slurm/logs/stronger_56532117.out and writes
results/stronger/head_ablation_partial.json with a schema compatible with
Llama-3's head_ablation_results.json.

Usage:
    python scripts/parse_qwen_ablation_log.py
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = "slurm/logs/stronger_56532117.out"
HEAD_IMPORTANCE = "results/stronger/patching/head_importance.json"
OUT_PATH = "results/stronger/head_ablation_partial.json"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


def parse_condition_blocks(text):
    """Yield dicts describing each --- Condition: ... --- block."""
    pattern = re.compile(
        r"^--- Condition: (?P<desc>.+?) ---\s*\n"
        r"  Sycophancy rate: (?P<syc>[\d.]+)%\s*\n"
        r"    anthropic_opinion: (?P<ant>[\d.]+)%\s*\n"
        r"    truthfulqa_factual: (?P<tqa>[\d.]+)%\s*\n"
        r"    gsm8k_reasoning: (?P<gsm>[\d.]+)%\s*\n"
        r"(?:  Evaluating MMLU\.\.\.\s*\n"
        r"  MMLU accuracy: (?P<mmlu>[\d.]+)%\s*\n)?"
        r"(?:  Evaluating GSM8k\.\.\.\s*\n"
        r"  GSM8k accuracy: (?P<gsm8k_acc>[\d.]+)%\s*\n)?",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        yield m.groupdict()


def description_to_condition_name(desc, heads_top3):
    """Recover the condition key from the printed description."""
    if "No ablation" in desc:
        return "baseline", [], "none"
    m = re.match(r"Zero-ablate (L\d+H\d+)$", desc)
    if m:
        h = m.group(1)
        return f"{h}_zero", [h], "zero"
    m = re.match(r"Zero-ablate (L\d+H\d+) \+ (L\d+H\d+)$", desc)
    if m:
        h1, h2 = m.group(1), m.group(2)
        return f"{h1}_{h2}_zero", [h1, h2], "zero"
    m = re.match(r"Zero-ablate all \(([^)]+)\)$", desc)
    if m:
        return "all_zero", m.group(1).split("+"), "zero"
    m = re.match(r"Mean-ablate all \(([^)]+)\)$", desc)
    if m:
        return "all_mean", m.group(1).split("+"), "mean"
    m = re.match(r"Mean-ablate (L\d+H\d+)$", desc)
    if m:
        h = m.group(1)
        return f"{h}_mean", [h], "mean"
    return None, None, None


def head_str_to_tuple(s):
    m = re.match(r"L(\d+)H(\d+)", s)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    text = Path(LOG_PATH).read_text()
    head_imp = json.load(open(HEAD_IMPORTANCE))
    top3 = [h["head"] for h in head_imp["head_results"]["top_10_heads"][:3]]

    conditions = {}
    for block in parse_condition_blocks(text):
        desc = block["desc"].strip()
        name, heads_str, mode = description_to_condition_name(desc, top3)
        if name is None:
            print(f"WARN: could not parse condition: {desc!r}")
            continue
        heads_tuples = [head_str_to_tuple(h) for h in (heads_str or [])]

        cond_entry = {
            "heads_ablated": heads_tuples,
            "heads_ablated_str": heads_str,
            "mode": mode,
            "description": desc,
            "sycophancy": {
                "overall_sycophancy_rate": float(block["syc"]) / 100.0,
                "per_source": {
                    "anthropic_opinion": {"sycophancy_rate": float(block["ant"]) / 100.0},
                    "truthfulqa_factual": {"sycophancy_rate": float(block["tqa"]) / 100.0},
                    "gsm8k_reasoning":   {"sycophancy_rate": float(block["gsm"]) / 100.0},
                },
                "total_evaluated": 1500,
                "source": "parsed_from_stdout_log",
            },
        }
        if block["mmlu"]:
            cond_entry["capabilities"] = {
                "mmlu": {"accuracy": float(block["mmlu"]) / 100.0, "n_samples": 500},
            }
            if block["gsm8k_acc"]:
                cond_entry["capabilities"]["gsm8k"] = {
                    "accuracy": float(block["gsm8k_acc"]) / 100.0,
                    "n_samples": 200,
                }
            else:
                cond_entry["capabilities"]["gsm8k"] = {
                    "accuracy": None,
                    "n_samples": None,
                    "note": "GSM8k evaluation interrupted by 48h wallclock timeout",
                }
        conditions[name] = cond_entry

    output = {
        "schema_version": "2.1",
        "analysis_mode": "head_ablation_intervention_partial",
        "split_definition": "Parsed from stdout log of timed-out main job (slurm/logs/stronger_56532117.out).",
        "metadata": {
            "model_name": MODEL_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_path": "data/processed/master_sycophancy_balanced.jsonl",
            "n_conditions_completed": len([c for c in conditions.values()
                                            if c.get("capabilities", {}).get("gsm8k", {}).get("accuracy") is not None]),
            "n_conditions_partial": len([c for c in conditions.values()
                                          if c.get("capabilities", {}).get("gsm8k", {}).get("accuracy") is None]),
            "top_3_heads_from_patching": top3,
            "main_job_id": "56532117",
            "main_job_state": "TIMEOUT",
            "main_job_elapsed": "48:00:19",
            "supplementary_job_id": "56681658",
            "supplementary_status": "queued (will produce baseline + all_zero + all_mean)",
            "note": (
                "Main job's scripts/04_head_ablation.py crashed before writing "
                "head_ablation.json due to 48h wallclock cap. This artifact "
                "preserves the per-condition numbers printed to stdout. "
                "Sycophancy CIs are not in the log; use Wilson 95% CI from "
                "rate × N=1500 if needed for paper. Conditions reaching only "
                "MMLU (no GSM8k) are flagged with gsm8k.accuracy=null."
            ),
        },
        "conditions": conditions,
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {OUT_PATH}")
    print(f"  conditions parsed: {len(conditions)}")
    for name, c in conditions.items():
        syc = c["sycophancy"]["overall_sycophancy_rate"]
        mmlu = c.get("capabilities", {}).get("mmlu", {}).get("accuracy")
        gsm = c.get("capabilities", {}).get("gsm8k", {}).get("accuracy")
        gsm_str = f"{gsm:.1%}" if gsm is not None else "N/A (timeout)"
        mmlu_str = f"{mmlu:.1%}" if mmlu is not None else "N/A"
        print(f"  - {name}: syc={syc:.1%} mmlu={mmlu_str} gsm={gsm_str}")


if __name__ == "__main__":
    main()
