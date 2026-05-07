#!/usr/bin/env python3
"""Regenerate Table 4: cross-model generality bounds.

Aggregates Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.1, and
Qwen-2.5-14B-Instruct on baseline, top-k zero-ablation Δ, DPO outcomes
(where available), and the OOD retention numbers.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "derived"
OUT.mkdir(exist_ok=True)


def load(p):
    return json.load(p.open()) if p.exists() else None


def fmt(x, fmt_str=".3f"):
    if x is None:
        return "—"
    return f"{x:{fmt_str}}"


def pp_delta(rate_after, rate_before):
    if rate_after is None or rate_before is None:
        return None
    return 100.0 * (rate_after - rate_before)


def main():
    rows = []

    # Llama
    llama_base = load(R / "baseline_llama3_summary.json")
    llama_top10 = load(R / "top10_ablation_full_gsm8k.json")
    llama_dpo_seed = load(R / "dpo_seed_summary.json")

    rows.append(dict(
        model="Llama-3-8B-Instruct",
        baseline_overall=fmt(llama_base["overall"]["sycophancy_rate"]),
        baseline_opinion=fmt(llama_base["per_source"]["anthropic_opinion"]["sycophancy_rate"]),
        top_k_zero_ablation_pp=fmt(pp_delta(
            llama_top10["conditions"]["all_zero"]["sycophancy"]["overall_sycophancy_rate"],
            llama_top10["conditions"]["baseline"]["sycophancy"]["overall_sycophancy_rate"],
        ), "+.2f"),
        dpo_opinion_after=fmt(llama_dpo_seed["summary"]["opinion_sycophancy"]["mean"]) if llama_dpo_seed else "—",
        notes="top-k = 10; DPO mean of 3 seeds",
    ))

    # Mistral
    mistral_base = load(R / "mistral/baseline_summary.json")
    mistral_top10 = load(R / "mistral/top10_ablation_full_gsm8k.json")
    mistral_dpo = load(R / "mistral/dpo_eval_results.json")
    rows.append(dict(
        model="Mistral-7B-Instruct-v0.1",
        baseline_overall=fmt(mistral_base["overall"]["sycophancy_rate"]),
        baseline_opinion=fmt(mistral_base["per_source"]["anthropic_opinion"]["sycophancy_rate"]),
        top_k_zero_ablation_pp=fmt(pp_delta(
            mistral_top10["conditions"]["all_zero"]["sycophancy"]["overall_sycophancy_rate"],
            mistral_top10["conditions"]["baseline"]["sycophancy"]["overall_sycophancy_rate"],
        ), "+.2f"),
        dpo_opinion_after=fmt(mistral_dpo["behavioral"]["per_source"]["anthropic_opinion"]["sycophancy_rate"]) if mistral_dpo else "—",
        notes="DPO factual sycophancy → 1.000; GSM8k → 0.0",
    ))

    # Qwen
    qwen_base = load(R / "stronger/baseline_summary.json")
    qwen_top3 = load(R / "stronger/head_ablation_supplementary.json")
    rows.append(dict(
        model="Qwen-2.5-14B-Instruct",
        baseline_overall=fmt(qwen_base["overall"]["sycophancy_rate"]),
        baseline_opinion=fmt(qwen_base["per_source"]["anthropic_opinion"]["sycophancy_rate"]),
        top_k_zero_ablation_pp=fmt(pp_delta(
            qwen_top3["conditions"]["all_zero"]["sycophancy"]["overall_sycophancy_rate"],
            qwen_top3["conditions"]["baseline"]["sycophancy"]["overall_sycophancy_rate"],
        ), "+.2f"),
        dpo_opinion_after="—",
        notes="top-k = 3; ablation INCREASES sycophancy",
    ))

    md = ["# Table 4 — Cross-model generality bounds",
          "",
          "Top-k zero-ablation column reports the change in overall sycophancy after",
          "ablating the patching-identified top-k heads (all-to-zero), in percentage points.",
          "",
          "| model | baseline overall | baseline opinion | top-k zero-ablation Δ (pp) | DPO opinion after | notes |",
          "|-------|------------------|------------------|---------------------------|-------------------|-------|"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['baseline_overall']} | {r['baseline_opinion']} | "
            f"{r['top_k_zero_ablation_pp']} | {r['dpo_opinion_after']} | {r['notes']} |"
        )

    # OOD retention sub-table
    ood = load(R / "ood_opinion_eval_results.json")
    md.append("")
    md.append("## OOD opinion retention vs. in-distribution DPO Δ")
    md.append("")
    md.append("| condition | in-dist Δ (pp) | condition Δ (pp) | retention |")
    md.append("|-----------|----------------|------------------|-----------|")
    if ood:
        in_dist = ood["comparison"]["in_distribution_reference"]["delta_pp"]
        for cond in ["condition_1", "condition_2", "condition_3", "all_ood"]:
            c = ood["comparison"][cond]
            md.append(
                f"| {cond} ({c['description']}) | {in_dist:+.1f} | {c['delta_pp']:+.1f} | "
                f"{c['delta_pp']/in_dist:.2f} |"
            )

    out_md = OUT / "table4.md"
    out_md.write_text("\n".join(md) + "\n")

    with (OUT / "table4.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n".join(md))
    print(f"\nWritten: {out_md}, {OUT/'table4.csv'}")


if __name__ == "__main__":
    main()
