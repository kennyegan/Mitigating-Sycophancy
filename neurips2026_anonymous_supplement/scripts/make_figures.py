#!/usr/bin/env python3
"""Regenerate figures from cached JSON files where a regenerator is
available.  For figures whose original data is not in this supplement we
print a note pointing to the bundled PDF/PNG copy in `figures/`.

CPU-only.  Requires `matplotlib`; will gracefully degrade if it's not
installed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
F = ROOT / "figures"
DERIVED = F / "regenerated"
DERIVED.mkdir(exist_ok=True)


def load(p):
    return json.load(p.open()) if p.exists() else None


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib/numpy not installed — skipping regeneration. "
          "Original PDFs/PNGs remain in figures/.")
    sys.exit(0)


def fig7_dpo_seed_robustness():
    d = load(R / "dpo_seed_summary.json")
    if d is None:
        return
    metrics = ["opinion_sycophancy", "overall_sycophancy", "gsm8k_accuracy", "mmlu_accuracy"]
    seeds = [s["seed"] for s in d["per_seed"]]
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(metrics))
    width = 0.25
    for i, seed_record in enumerate(d["per_seed"]):
        vals = [seed_record[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=f"seed {seed_record['seed']}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("rate / accuracy")
    ax.set_title("Fig 7 — DPO seed robustness (seeds 100/200/300)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = DERIVED / "fig7_dpo_seed_robustness.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig8_ood_generalization():
    d = load(R / "ood_opinion_eval_results.json")
    if d is None:
        return
    conds = [("condition_1", "C1: new Anthropic"),
             ("condition_2", "C2: rephrased"),
             ("condition_3", "C3: manual diverse"),
             ("all_ood", "all OOD")]
    in_dist = d["comparison"]["in_distribution_reference"]["delta_pp"]
    deltas = [d["comparison"][c[0]]["delta_pp"] for c in conds]
    retention = [c / in_dist for c in deltas]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([c[1] for c in conds], retention)
    ax.axhline(1.0, ls="--", c="grey", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("retention vs in-distribution DPO Δ")
    ax.set_title("Fig 8 — OOD opinion retention")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    out = DERIVED / "fig8_ood_generalization.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig9_freeform_5dim():
    d = load(R / "freeform" / "comparison_summary.json")
    if d is None:
        return
    overall = d["domains"]["overall"]
    dims = ["truthfulness", "sycophancy", "helpfulness", "hedging", "pushback_resistance"]
    base_means = [overall["baseline"][k]["mean"] for k in dims]
    dpo_means = [overall["dpo"][k]["mean"] for k in dims]
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(dims))
    ax.bar(x - 0.2, base_means, 0.4, label="baseline")
    ax.bar(x + 0.2, dpo_means, 0.4, label="DPO")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=20, ha="right")
    ax.legend()
    ax.set_title("Fig 9 — Free-form 5-dimensional comparison")
    fig.tight_layout()
    out = DERIVED / "fig9_freeform_5dim.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig11_sft_dpo_tradeoff():
    dpo = load(R / "dpo_seed_summary.json")
    sft = load(R / "sft_gsm8k_full.json")
    if dpo is None or sft is None:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    base_gsm = load(R / "top10_ablation_full_gsm8k.json")["conditions"]["baseline"]["gsm8k"]["accuracy"]
    base_op = load(R / "baseline_llama3_summary.json")["per_source"]["anthropic_opinion"]["sycophancy_rate"]
    points = [
        ("Baseline", base_op, base_gsm, "grey"),
        ("DPO mean", dpo["summary"]["opinion_sycophancy"]["mean"],
                     # use full-GSM8k 3-seed mean
                     0.4023, "tab:blue"),
        ("SFT", sft["behavioral"]["per_source"]["anthropic_opinion"]["sycophancy_rate"],
                sft["capabilities"]["gsm8k"]["accuracy"], "tab:red"),
    ]
    for label, x, y, c in points:
        ax.scatter(x, y, c=c, s=80, label=label)
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("opinion sycophancy ↓")
    ax.set_ylabel("GSM8k accuracy ↑")
    ax.set_title("Fig 11 — SFT/DPO trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DERIVED / "fig11_sft_dpo_tradeoff.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig12_cross_scale():
    files = [
        ("Llama-3-8B-Instruct", R / "baseline_llama3_summary.json", 8),
        ("Mistral-7B-Instruct-v0.1", R / "mistral/baseline_summary.json", 7),
        ("Qwen-2.5-14B-Instruct", R / "stronger/baseline_summary.json", 14),
    ]
    fig, ax = plt.subplots(figsize=(5, 3))
    xs, ys, names = [], [], []
    for name, p, scale in files:
        d = load(p)
        if d is None:
            continue
        xs.append(scale)
        ys.append(d["per_source"]["anthropic_opinion"]["sycophancy_rate"])
        names.append(name)
    ax.scatter(xs, ys)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n.split("-")[0], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("model size (B params)")
    ax.set_ylabel("opinion sycophancy")
    ax.set_title("Fig 12 — Opinion sycophancy across models / scales")
    fig.tight_layout()
    out = DERIVED / "fig12_cross_scale.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


REGENERATORS = [
    ("fig7_dpo_seed_robustness", fig7_dpo_seed_robustness),
    ("fig8_ood_generalization", fig8_ood_generalization),
    ("fig9_freeform_5dim", fig9_freeform_5dim),
    ("fig11_sft_dpo_tradeoff", fig11_sft_dpo_tradeoff),
    ("fig12_cross_scale", fig12_cross_scale),
]


def main():
    print("Regenerating figures from cached JSON …")
    for name, fn in REGENERATORS:
        try:
            fn()
        except Exception as e:
            print(f"  {name}: SKIP ({e})")
    not_regen = ["fig1_patching_heatmap", "fig2_steering_sweep", "fig3_steering_per_source",
                 "fig4_probe_accuracy", "fig5_ablation_comparison",
                 "fig6_dpo_probe_decomposition", "fig10_transcript_panel"]
    print()
    print("Figures NOT regenerated by this script (PDF/PNG copies in figures/ remain canonical):")
    for n in not_regen:
        if (F / f"{n}.pdf").exists() or (F / f"{n}.png").exists():
            print(f"  figures/{n}.{{pdf,png}}")
    print("\nDone.")


if __name__ == "__main__":
    main()
