#!/usr/bin/env python3
"""Regenerate Table 3: DPO vs SFT trade-off on Llama-3-8B-Instruct."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = R / "derived"
OUT.mkdir(exist_ok=True)


def load(p):
    return json.load(p.open())


def sample_sd(values):
    if len(values) < 2:
        return 0.0
    return (pstdev(values) * (len(values) / (len(values) - 1)) ** 0.5)


def main():
    # Baseline
    base = load(R / "baseline_llama3_summary.json")
    base_overall = base["overall"]["sycophancy_rate"]
    base_opinion = base["per_source"]["anthropic_opinion"]["sycophancy_rate"]
    # Use top10_ablation_full_gsm8k baseline GSM8k accuracy as the matched-eval baseline
    base_gsm = load(R / "top10_ablation_full_gsm8k.json")["conditions"]["baseline"]["gsm8k"]["accuracy"]

    # DPO 3-seed
    dpo_files = ["dpo_gsm8k_full_results.json", "dpo_gsm8k_full_seed200.json", "dpo_gsm8k_full_seed300.json"]
    dpo_seeds = [load(R / f) for f in dpo_files]
    dpo_overall = [d["behavioral"]["overall"]["sycophancy_rate"] for d in dpo_seeds]
    dpo_opinion = [d["behavioral"]["per_source"]["anthropic_opinion"]["sycophancy_rate"] for d in dpo_seeds]
    dpo_gsm = [d["capabilities"]["gsm8k"]["accuracy"] for d in dpo_seeds]

    # SFT
    sft = load(R / "sft_gsm8k_full.json")
    sft_overall = sft["behavioral"]["overall"]["sycophancy_rate"]
    sft_opinion = sft["behavioral"]["per_source"]["anthropic_opinion"]["sycophancy_rate"]
    sft_gsm = sft["capabilities"]["gsm8k"]["accuracy"]

    rows = [
        dict(
            condition="Baseline",
            overall=f"{base_overall:.3f}",
            opinion=f"{base_opinion:.3f}",
            gsm8k=f"{base_gsm:.3f}",
            note="meta-llama/Meta-Llama-3-8B-Instruct",
        ),
        dict(
            condition="DPO (3 seeds: 100/200/300)",
            overall=f"{mean(dpo_overall):.3f} ± {sample_sd(dpo_overall):.3f}",
            opinion=f"{mean(dpo_opinion):.3f} ± {sample_sd(dpo_opinion):.3f}",
            gsm8k=f"{mean(dpo_gsm):.3f} ± {sample_sd(dpo_gsm):.3f}",
            note="full 1319-sample GSM8k",
        ),
        dict(
            condition="SFT (same data)",
            overall=f"{sft_overall:.3f}",
            opinion=f"{sft_opinion:.3f}",
            gsm8k=f"{sft_gsm:.3f}",
            note="single seed",
        ),
    ]

    md = ["# Table 3 — DPO vs SFT on Llama-3-8B-Instruct",
          "",
          "| condition | overall syc. | opinion syc. | GSM8k acc. | note |",
          "|-----------|--------------|--------------|------------|------|"]
    for r in rows:
        md.append(f"| {r['condition']} | {r['overall']} | {r['opinion']} | {r['gsm8k']} | {r['note']} |")
    out_md = OUT / "table3.md"
    out_md.write_text("\n".join(md) + "\n")

    with (OUT / "table3.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n".join(md))
    print(f"\nWritten: {out_md}, {OUT/'table3.csv'}")


if __name__ == "__main__":
    main()
