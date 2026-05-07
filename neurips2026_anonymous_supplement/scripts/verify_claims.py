#!/usr/bin/env python3
"""Verify paper numbers against cached result files.

CPU-only, stdlib-only.

This script never hard-codes paper numbers as observed values.  For every
claim it loads the cited cached file, extracts the relevant field, and
compares to the paper's reported value within a tolerance.  The expected
(paper) value is documented per claim so that mismatches surface clearly.

Status legend:
  PASS   — observed matches expected within tolerance
  WARN   — observed within a looser tolerance; documented in MANIFEST.md
  FAIL   — observed differs by more than the loose tolerance (review needed)
  OBS    — informational row (no expected value); observed printed
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def load(p: Path) -> dict:
    return json.load(p.open())


def get(d: dict, path: str):
    cur = d
    for key in path.split("/"):
        if not key:
            continue
        if isinstance(cur, list):
            cur = cur[int(key)]
        else:
            cur = cur[key]
    return cur


CLAIMS: list[dict] = []
_FIELD_LOOKUP = object()  # sentinel: "use the field path, not a transform"


def add(name, source, field, expected, tol=0.005, transform=_FIELD_LOOKUP, note=""):
    CLAIMS.append(dict(
        name=name, source=source, field=field, expected=expected,
        tol=tol, transform=transform, note=note,
    ))


# Section 1 — Llama baseline
add("Llama baseline overall", "baseline_llama3_summary.json",
    "overall/sycophancy_rate", 0.280, tol=0.005)
add("Llama baseline opinion", "baseline_llama3_summary.json",
    "per_source/anthropic_opinion/sycophancy_rate", 0.824, tol=0.01)
add("Llama baseline factual", "baseline_llama3_summary.json",
    "per_source/truthfulqa_factual/sycophancy_rate", 0.016, tol=0.005)
add("Llama baseline reasoning", "baseline_llama3_summary.json",
    "per_source/gsm8k_reasoning/sycophancy_rate", 0.000, tol=0.005)

# Section 2 — Probe decomposition (Layer 1)
add("Probe best-transfer layer", "probe_control_balanced_results.json",
    "per_position_summary/final/best_layer", 1, tol=0)
add("Probe robust tracking", "probe_control_balanced_results.json",
    "per_position_summary/final/best_robust_rate", 0.599, tol=0.005)
add("Probe social compliance", "probe_control_balanced_results.json",
    "per_position_summary/final/best_social_compliance_rate", 0.180, tol=0.005)
add("Probe belief corruption", "probe_control_balanced_results.json",
    "per_position_summary/final/best_belief_corruption_rate", 0.101, tol=0.005)


def _other(d):
    s = d["per_position_summary"]["final"]
    return 1.0 - s["best_robust_rate"] - s["best_social_compliance_rate"] - s["best_belief_corruption_rate"]


CLAIMS.append(dict(
    name="Probe other (1 - robust - social - belief)",
    source="probe_control_balanced_results.json",
    field=None, expected=0.121, tol=0.005, transform=_other, note="derived",
))
CLAIMS.append(dict(
    name="Probe: social > belief",
    source="probe_control_balanced_results.json",
    field=None,
    expected=True, tol=0,
    transform=lambda d: d["per_position_summary"]["final"]["best_social_compliance_rate"]
                     >  d["per_position_summary"]["final"]["best_belief_corruption_rate"],
    note="qualitative",
))

# Section 3 — Patching/ablation
add("Top-3 patching Jaccard", "patching_bootstrap.json",
    "aggregate/pairwise_jaccard/top3/mean", 0.09, tol=0.01)
CLAIMS.append(dict(
    name="Llama top-10 zero-ablation Δ (pp)",
    source="top10_ablation_full_gsm8k.json",
    field=None, expected=0.5, tol=0.5,
    transform=lambda d: 100.0 * (
        d["conditions"]["all_zero"]["sycophancy"]["overall_sycophancy_rate"]
        - d["conditions"]["baseline"]["sycophancy"]["overall_sycophancy_rate"]
    ),
    note="paper says no reduction; observed Δ should be small",
))
CLAIMS.append(dict(
    name="Mistral top-10 zero-ablation Δ (pp)",
    source="mistral/top10_ablation_full_gsm8k.json",
    field=None, expected=1.0, tol=0.5,
    transform=lambda d: 100.0 * (
        d["conditions"]["all_zero"]["sycophancy"]["overall_sycophancy_rate"]
        - d["conditions"]["baseline"]["sycophancy"]["overall_sycophancy_rate"]
    ),
    note="paper: Δ +1.0 pp",
))

# Section 4 — DPO vs SFT
add("DPO opinion (3-seed mean)", "dpo_seed_summary.json",
    "summary/opinion_sycophancy/mean", 0.571, tol=0.01)
add("DPO opinion (3-seed sd)", "dpo_seed_summary.json",
    "summary/opinion_sycophancy/sd", 0.028, tol=0.005)


def _dpo_gsm_mean(_=None):
    files = ["dpo_gsm8k_full_results.json", "dpo_gsm8k_full_seed200.json", "dpo_gsm8k_full_seed300.json"]
    vals = [load(R / f)["capabilities"]["gsm8k"]["accuracy"] for f in files]
    return mean(vals)


def _dpo_gsm_sd(_=None):
    files = ["dpo_gsm8k_full_results.json", "dpo_gsm8k_full_seed200.json", "dpo_gsm8k_full_seed300.json"]
    vals = [load(R / f)["capabilities"]["gsm8k"]["accuracy"] for f in files]
    return (pstdev(vals) * (len(vals) / (len(vals) - 1)) ** 0.5)  # sample sd


CLAIMS.append(dict(
    name="DPO GSM8k full (3-seed mean)",
    source="dpo_gsm8k_full_{seed100,seed200,seed300}.json",
    field=None, expected=0.402, tol=0.01, transform=_dpo_gsm_mean,
    note="full 1319-sample GSM8k",
))
CLAIMS.append(dict(
    name="DPO GSM8k full (3-seed sd)",
    source="dpo_gsm8k_full_{…}.json",
    field=None, expected=0.029, tol=0.01, transform=_dpo_gsm_sd,
    note="sample sd",
))
add("Baseline GSM8k accuracy", "top10_ablation_full_gsm8k.json",
    "conditions/baseline/gsm8k/accuracy", 0.332, tol=0.01)
add("SFT GSM8k full accuracy", "sft_gsm8k_full.json",
    "capabilities/gsm8k/accuracy", 0.058, tol=0.01)

# Section 5 — Cross-model
add("Mistral DPO factual sycophancy", "mistral/dpo_eval_results.json",
    "behavioral/per_source/truthfulqa_factual/sycophancy_rate", 1.000, tol=0.005)
add("Mistral DPO GSM8k accuracy", "mistral/dpo_eval_results.json",
    "capabilities/gsm8k/accuracy", 0.000, tol=0.005)
add("Qwen baseline opinion sycophancy", "stronger/baseline_summary.json",
    "per_source/anthropic_opinion/sycophancy_rate", 0.753, tol=0.01)
add("Qwen baseline opinion mean compliance gap", "stronger/baseline_summary.json",
    "per_source/anthropic_opinion/mean_compliance_gap", 0.006, tol=0.005)
CLAIMS.append(dict(
    name="Qwen top-3 zero-ablation Δ (pp)",
    source="stronger/head_ablation_supplementary.json",
    field=None, expected=20.3, tol=2.0,
    transform=lambda d: 100.0 * (
        d["conditions"]["all_zero"]["sycophancy"]["overall_sycophancy_rate"]
        - d["conditions"]["baseline"]["sycophancy"]["overall_sycophancy_rate"]
    ),
    note="paper: +20.3 pp",
))

# Section 6 — Free-form / OOD
add("Free-form overall baseline sycophancy",
    "freeform/comparison_summary.json",
    "domains/overall/baseline/sycophancy/mean", 2.66, tol=0.05)
add("Free-form overall DPO sycophancy",
    "freeform/comparison_summary.json",
    "domains/overall/dpo/sycophancy/mean", 2.43, tol=0.05)


def _ood_retention(d, condition):
    in_dist = d["comparison"]["in_distribution_reference"]["delta_pp"]
    cond = d["comparison"][condition]["delta_pp"]
    return cond / in_dist


CLAIMS.append(dict(
    name="OOD retention condition_1 (new Anthropic, fixed template)",
    source="ood_opinion_eval_results.json",
    field=None, expected=None, tol=None,
    transform=lambda d: _ood_retention(d, "condition_1"),
    note="OBS — paper labels Protocol A/B, file labels condition_1/2/3",
))
CLAIMS.append(dict(
    name="OOD retention condition_2 (rephrased templates)",
    source="ood_opinion_eval_results.json",
    field=None, expected=None, tol=None,
    transform=lambda d: _ood_retention(d, "condition_2"),
    note="OBS",
))
CLAIMS.append(dict(
    name="OOD retention condition_3 (manual diverse)",
    source="ood_opinion_eval_results.json",
    field=None, expected=None, tol=None,
    transform=lambda d: _ood_retention(d, "condition_3"),
    note="OBS — closest match for Protocol B",
))


def status_for(observed, expected, tol):
    if expected is None:
        return "OBS"
    if isinstance(expected, bool):
        return "PASS" if bool(observed) == expected else "FAIL"
    try:
        diff = abs(float(observed) - float(expected))
    except Exception:
        return "FAIL"
    if diff <= tol:
        return "PASS"
    if diff <= tol * 3:
        return "WARN"
    return "FAIL"


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if abs(x) < 1:
            return f"{x:.4f}"
        return f"{x:.3f}"
    return str(x)


def main() -> int:
    rows = []
    for c in CLAIMS:
        path = R / c["source"].split("{")[0].rstrip(".json,/_") if "{" in c["source"] else R / c["source"]
        d = None
        if path.exists():
            try:
                d = load(path)
            except Exception:
                d = None
        if c["transform"] is _FIELD_LOOKUP:
            try:
                obs = get(d, c["field"]) if d is not None and c["field"] else None
            except Exception as e:
                obs = f"ERROR: {e}"
        else:
            try:
                obs = c["transform"](d)
            except Exception as e:
                obs = f"ERROR: {e}"
        st = status_for(obs, c["expected"], c["tol"]) if not isinstance(obs, str) or not obs.startswith("ERROR") else "FAIL"
        rows.append((c["name"], c["expected"], obs, c["tol"], st, c["source"], c["note"]))

    # Print as a table
    headers = ("claim", "expected", "observed", "tol", "status", "source", "note")
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    fmt_row = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt_row.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        nm, exp, obs, tol, st, src, note = r
        print(fmt_row.format(nm, fmt(exp), fmt(obs), fmt(tol), st, src, note))

    n_pass = sum(1 for r in rows if r[4] == "PASS")
    n_warn = sum(1 for r in rows if r[4] == "WARN")
    n_fail = sum(1 for r in rows if r[4] == "FAIL")
    n_obs = sum(1 for r in rows if r[4] == "OBS")
    print()
    print(f"Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL, {n_obs} OBS  (of {len(rows)} claims)")
    if n_fail:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
