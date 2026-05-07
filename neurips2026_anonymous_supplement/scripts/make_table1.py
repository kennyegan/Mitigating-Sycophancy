#!/usr/bin/env python3
"""Regenerate Table 1: per-source baseline sycophancy for Llama-3-8B-Instruct
and the corresponding base (non-instruct) model.  Output: Markdown + CSV
under results/derived/.
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
    return json.load(p.open())


def main():
    rows = []
    files = [
        ("Llama-3-8B-Instruct", "baseline_llama3_summary.json"),
        ("Llama-3-8B (base)", "baseline_llama3_base_summary.json"),
        ("Mistral-7B-Instruct-v0.1", "mistral/baseline_summary.json"),
        ("Qwen-2.5-14B-Instruct", "stronger/baseline_summary.json"),
    ]
    for label, rel in files:
        p = R / rel
        if not p.exists():
            continue
        d = load(p)
        rows.append({
            "model": label,
            "overall": d["overall"]["sycophancy_rate"],
            "opinion": d["per_source"]["anthropic_opinion"]["sycophancy_rate"],
            "factual": d["per_source"]["truthfulqa_factual"]["sycophancy_rate"],
            "reasoning": d["per_source"]["gsm8k_reasoning"]["sycophancy_rate"],
            "n": d["metadata"]["samples_evaluated"],
            "source": rel,
        })

    md = ["# Table 1 — Baseline sycophancy by source",
          "",
          "| model | overall | opinion | factual | reasoning | n | source |",
          "|-------|---------|---------|---------|-----------|---|--------|"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['overall']:.3f} | {r['opinion']:.3f} | "
            f"{r['factual']:.3f} | {r['reasoning']:.3f} | {r['n']} | "
            f"`{r['source']}` |"
        )
    (OUT / "table1.md").write_text("\n".join(md) + "\n")

    with (OUT / "table1.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n".join(md))
    print(f"\nWritten: {OUT/'table1.md'}, {OUT/'table1.csv'}")


if __name__ == "__main__":
    main()
