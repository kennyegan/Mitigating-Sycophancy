#!/usr/bin/env python3
"""Generate the 6 new NeurIPS figures (F7-F12) referenced in paper.tex.

Reads result JSON files and produces PDF + PNG figures with consistent style.

Usage:
    python scripts/generate_figures_neurips.py
    python scripts/generate_figures_neurips.py --figures 7 9 11
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------- Style (mirrors generate_figures.py for consistency) ----------
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FONT_SIZE_TICK,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
FIGURES = PROJECT / "figures"
FIGURES.mkdir(exist_ok=True)

COLOR_BASELINE = "#7f7f7f"
COLOR_DPO = "#1f77b4"
COLOR_SFT = "#d62728"
COLOR_LLAMA = "#1f77b4"
COLOR_MISTRAL = "#ff7f0e"
COLOR_QWEN = "#2ca02c"


def save_fig(fig, name):
    pdf = FIGURES / f"{name}.pdf"
    png = FIGURES / f"{name}.png"
    fig.savefig(pdf, dpi=150, metadata={"Author": ""})
    fig.savefig(png, dpi=150, metadata={"Author": ""})
    plt.close(fig)
    print(f"  wrote {png.name} + {pdf.name}")


# ---------- F7: DPO seed robustness ----------
def fig7_dpo_seed_robustness():
    data = json.load(open(RESULTS / "dpo_seed_summary.json"))
    # Override GSM8k with the full-N=1,319 numbers (the multi-seed summary used N=200)
    full_gsm = {
        100: json.load(open(RESULTS / "dpo_gsm8k_full_results.json"))["capabilities"]["gsm8k"]["accuracy"],
        200: json.load(open(RESULTS / "dpo_gsm8k_full_seed200.json"))["capabilities"]["gsm8k"]["accuracy"],
        300: json.load(open(RESULTS / "dpo_gsm8k_full_seed300.json"))["capabilities"]["gsm8k"]["accuracy"],
    }
    for s in data["per_seed"]:
        s["gsm8k_accuracy"] = full_gsm[s["seed"]]
    full_vals = [full_gsm[s] for s in [100, 200, 300]]
    data["summary"]["gsm8k_accuracy"] = {"mean": float(np.mean(full_vals)), "sd": float(np.std(full_vals, ddof=1))}

    seeds = [s["seed"] for s in data["per_seed"]]
    summary = data["summary"]

    metrics = [
        ("opinion_sycophancy", "Opinion Sycophancy", 100, "Lower is better"),
        ("gsm8k_accuracy", "GSM8k Accuracy (full N=1,319)", 100, "Higher is better"),
        ("social_compliance", "Social Compliance (best layer)", 100, "Lower is better"),
        ("robust_tracking", "Robust Tracking (best layer)", 100, "Higher is better"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (key, title, scale, sub) in zip(axes, metrics):
        per_seed_vals = [s[key] * scale for s in data["per_seed"]]
        mean_v = summary[key]["mean"] * scale
        sd_v = summary[key]["sd"] * scale
        x = np.arange(len(seeds))
        bars = ax.bar(x, per_seed_vals, color=COLOR_DPO, alpha=0.7, width=0.7)
        ax.axhline(mean_v, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.fill_between(
            [-0.5, len(seeds) - 0.5], mean_v - sd_v, mean_v + sd_v,
            color="black", alpha=0.08, label=f"Mean $\\pm$ SD: {mean_v:.1f}±{sd_v:.1f}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in seeds])
        ax.set_xlim(-0.5, len(seeds) - 0.5)
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title, fontsize=FONT_SIZE_TITLE - 1)
        ax.legend(loc="best", fontsize=FONT_SIZE_LEGEND - 1, framealpha=0.9)
        for i, v in enumerate(per_seed_vals):
            ax.text(i, v + (0.5 if "Higher" in sub else -0.5), f"{v:.1f}",
                    ha="center", va="bottom" if "Higher" in sub else "top",
                    fontsize=FONT_SIZE_TICK - 1)

    fig.suptitle(
        "DPO Robustness Across 3 Independent Training Seeds (100, 200, 300)",
        fontsize=FONT_SIZE_TITLE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "fig7_dpo_seed_robustness")


# ---------- F8: OOD generalization (in-dist + Protocol A + Protocol B) ----------
def fig8_ood_generalization():
    proto_b = json.load(open(RESULTS / "ood_eval_results.json"))["comparison"]
    proto_a = json.load(open(RESULTS / "ood_opinion_eval_results.json"))["comparison"]

    in_dist = proto_b["in_distribution_reference"]

    groups = [
        ("In-Distribution\n(Anthropic seed=42)", in_dist["baseline_rate"], in_dist["dpo_rate"]),
        ("Protocol A:\nNew Anthropic Seed\n(N=200)", proto_a["condition_1"]["baseline_rate"], proto_a["condition_1"]["dpo_rate"]),
        ("Protocol A:\nRephrased Templates\n(N=200)", proto_a["condition_2"]["baseline_rate"], proto_a["condition_2"]["dpo_rate"]),
        ("Protocol A:\nManual Diverse\n(N=50)", proto_a["condition_3"]["baseline_rate"], proto_a["condition_3"]["dpo_rate"]),
        ("Protocol B:\nNLP Survey\n(N=500)", proto_b["nlp_survey"]["baseline_rate"], proto_b["nlp_survey"]["dpo_rate"]),
        ("Protocol B:\nPolitical Typology\n(N=500)", proto_b["political_typology"]["baseline_rate"], proto_b["political_typology"]["dpo_rate"]),
    ]

    labels = [g[0] for g in groups]
    base = [g[1] * 100 for g in groups]
    post = [g[2] * 100 for g in groups]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(groups))
    w = 0.4
    ax.bar(x - w / 2, base, w, color=COLOR_BASELINE, label="Baseline (pre-DPO)")
    ax.bar(x + w / 2, post, w, color=COLOR_DPO, label="Post-DPO")

    for i, (b, p) in enumerate(zip(base, post)):
        delta = p - b
        ax.text(i, max(b, p) + 1.5, f"$\\Delta$={delta:+.1f}", ha="center",
                fontsize=FONT_SIZE_TICK, color="darkred" if delta < -10 else "black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK - 1)
    ax.set_ylabel("Sycophancy Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title(
        "OOD Generalization: Format-Robust (~77% retention)\nbut Domain-Attenuated (~20% retention)",
        fontsize=FONT_SIZE_TITLE,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig8_ood_generalization")


# ---------- F9: Free-form 5-dimension comparison with bootstrap CIs ----------
def fig9_freeform_5dim():
    data = json.load(open(RESULTS / "freeform" / "comparison_summary.json"))
    overall = data["domains"]["overall"]
    cmp = overall["comparison"]

    # Order matches paper Table: truthfulness, sycophancy, helpfulness, hedging, pushback
    dims = [
        ("truthfulness", "Truthfulness", "Higher", 1, 5),
        ("sycophancy", "Sycophancy", "Lower", 1, 5),
        ("helpfulness", "Helpfulness", "Higher", 1, 5),
        ("hedging", "Hedging", "Mixed", 0, 2),
        ("pushback_resistance", "Pushback Resistance", "Higher", 0, 1),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(dims))
    w = 0.38

    base_means, dpo_means, lower_errs, upper_errs = [], [], [], []
    for key, _label, _direction, _lo, _hi in dims:
        if key in cmp:
            c = cmp[key]
            base_means.append(c["baseline_mean"])
            dpo_means.append(c["dpo_mean"])
            lo, hi = c["ci_95"]
            # CI is on the delta (DPO - baseline). Plot as error bar around DPO mean.
            delta = c["dpo_mean"] - c["baseline_mean"]
            lower_errs.append(abs(delta - lo))
            upper_errs.append(abs(hi - delta))
        else:
            # pushback_resistance might be keyed differently
            base_means.append(overall["baseline"][key]["mean"])
            dpo_means.append(overall["dpo"][key]["mean"])
            lower_errs.append(0)
            upper_errs.append(0)

    ax.bar(x - w / 2, base_means, w, color=COLOR_BASELINE, label="Baseline (pre-DPO)")
    ax.bar(x + w / 2, dpo_means, w, color=COLOR_DPO, label="Post-DPO",
           yerr=[lower_errs, upper_errs], ecolor="black", capsize=3, error_kw={"linewidth": 1.0})

    for i, (b, d, lo_e, up_e) in enumerate(zip(base_means, dpo_means, lower_errs, upper_errs)):
        delta = d - b
        ci_crosses_zero = lo_e > abs(delta) or up_e > abs(delta)
        marker = " (CI crosses 0)" if ci_crosses_zero else ""
        ax.text(i, max(b, d) + 0.15, f"$\\Delta$={delta:+.2f}{marker}", ha="center",
                fontsize=FONT_SIZE_TICK - 1)

    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in dims])
    ax.set_ylabel("Score (per-dimension scale; see caption)")
    ax.set_title(
        "Free-Form Generation Evaluation: 5-Dimension Comparison ($N{=}150$ per condition)\n"
        "Bootstrap CIs (5,000-iter BCa) on $\\Delta$; all overall deltas directional, CIs cross 0 at $N{=}150$",
        fontsize=FONT_SIZE_TITLE - 1,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 5.5)
    fig.tight_layout()
    save_fig(fig, "fig9_freeform_5dim")


# ---------- F10: Example transcript panel ----------
def fig10_transcript_panel():
    import textwrap

    base_records = [json.loads(l) for l in open(RESULTS / "freeform" / "llama3_base_transcripts.jsonl")]
    dpo_records = [json.loads(l) for l in open(RESULTS / "freeform" / "llama3_dpo_transcripts.jsonl")]
    base_scores = [json.loads(l) for l in open(RESULTS / "freeform" / "llama3_base_scores.jsonl")]
    dpo_scores = [json.loads(l) for l in open(RESULTS / "freeform" / "llama3_dpo_scores.jsonl")]

    by_id_base = {r["prompt_id"]: r for r in base_records}
    by_id_dpo = {r["prompt_id"]: r for r in dpo_records}
    by_id_base_score = {s["prompt_id"]: s for s in base_scores}
    by_id_dpo_score = {s["prompt_id"]: s for s in dpo_scores}

    # Find a prompt where baseline sycophancy >= 4 and DPO sycophancy is at least 1.5 lower
    best_pick = None
    best_delta = 0
    for sid in by_id_base:
        if sid not in by_id_dpo or sid not in by_id_base_score or sid not in by_id_dpo_score:
            continue
        base_syc = by_id_base_score[sid].get("scores", {}).get("sycophancy")
        dpo_syc = by_id_dpo_score[sid].get("scores", {}).get("sycophancy")
        if base_syc is None or dpo_syc is None:
            continue
        delta = base_syc - dpo_syc
        if base_syc >= 4 and delta > best_delta:
            best_delta = delta
            best_pick = sid

    if best_pick is None:
        # Fallback: any opinion-domain pair
        for sid, r in by_id_base.items():
            if r["domain"] == "opinion" and sid in by_id_dpo:
                best_pick = sid
                break

    base_t = by_id_base[best_pick]
    dpo_t = by_id_dpo[best_pick]
    base_score = by_id_base_score[best_pick].get("scores", {})
    dpo_score = by_id_dpo_score[best_pick].get("scores", {})

    def format_transcript(record, max_chars_per_turn=400, max_turns=4):
        lines = []
        for turn in record["turns"][:max_turns]:
            role = turn["role"].upper()
            content = str(turn.get("content", "")).strip()
            if len(content) > max_chars_per_turn:
                content = content[:max_chars_per_turn - 3] + "..."
            wrapped = textwrap.fill(content, width=70, initial_indent="    ", subsequent_indent="    ")
            lines.append(f"[{role}]\n{wrapped}")
        return "\n\n".join(lines)

    base_text = format_transcript(base_t)
    dpo_text = format_transcript(dpo_t)

    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f"Example Free-Form Transcript: Same Prompt ({best_pick}, domain={base_t['domain']})  -  Baseline vs. DPO Response",
        fontsize=FONT_SIZE_TITLE + 1, y=0.98,
    )

    score_summary = (
        f"Sycophancy: baseline {base_score.get('sycophancy', '?')}/5  $\\to$  DPO {dpo_score.get('sycophancy', '?')}/5"
        f"     |     Truthfulness: baseline {base_score.get('truthfulness', '?')}/5  $\\to$  DPO {dpo_score.get('truthfulness', '?')}/5"
    )
    fig.text(0.5, 0.94, score_summary, ha="center", fontsize=FONT_SIZE_LABEL, style="italic")

    for ax_idx, (text, title, color) in enumerate([
        (base_text, "Baseline (Llama-3-8B-Instruct)", COLOR_BASELINE),
        (dpo_text, "Post-DPO (seed 100)", COLOR_DPO),
    ]):
        ax = fig.add_axes([0.04 + ax_idx * 0.48, 0.05, 0.44, 0.85])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.99, title, ha="center", va="top",
                fontsize=FONT_SIZE_LABEL + 1, weight="bold", color=color,
                transform=ax.transAxes)
        # Render transcript text with monospace for clean wrapping
        ax.text(0.02, 0.94, text, ha="left", va="top",
                fontsize=10, family="monospace",
                transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor=color, linewidth=1.5,
                          boxstyle="round,pad=0.5", alpha=0.95))

    save_fig(fig, "fig10_transcript_panel")


# ---------- F11: SFT vs DPO capability-safety tradeoff scatter ----------
def fig11_sft_dpo_tradeoff():
    sft = json.load(open(RESULTS / "sft_eval_results.json"))
    sft_gsm = json.load(open(RESULTS / "sft_gsm8k_full.json"))
    dpo_summary = json.load(open(RESULTS / "dpo_seed_summary.json"))
    dpo_per_seed = dpo_summary["per_seed"]
    dpo_full_gsm = {
        100: json.load(open(RESULTS / "dpo_gsm8k_full_results.json"))["capabilities"]["gsm8k"]["accuracy"],
        200: json.load(open(RESULTS / "dpo_gsm8k_full_seed200.json"))["capabilities"]["gsm8k"]["accuracy"],
        300: json.load(open(RESULTS / "dpo_gsm8k_full_seed300.json"))["capabilities"]["gsm8k"]["accuracy"],
    }

    baseline_syc = 0.28  # Llama-3 baseline overall syc
    baseline_gsm = 0.332  # full N=1,319 baseline

    points = []
    points.append({"name": "Baseline\n(no fine-tune)", "syc_red_pp": 0.0, "gsm": baseline_gsm * 100, "color": COLOR_BASELINE, "marker": "s", "size": 240})
    points.append({"name": "SFT (chosen-only)", "syc_red_pp": (baseline_syc - sft["behavioral"]["overall"]["sycophancy_rate"]) * 100,
                   "gsm": sft_gsm["capabilities"]["gsm8k"]["accuracy"] * 100, "color": COLOR_SFT, "marker": "X", "size": 220})
    for seed_data in dpo_per_seed:
        seed = seed_data["seed"]
        points.append({"name": f"DPO seed {seed}", "syc_red_pp": (baseline_syc - seed_data["overall_sycophancy"]) * 100,
                       "gsm": dpo_full_gsm[seed] * 100, "color": COLOR_DPO, "marker": "o", "size": 160})

    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Per-point label offsets to avoid overlap
    label_offsets = {
        "Baseline\n(no fine-tune)": (0, -3.0, "center", "top"),
        "SFT (chosen-only)": (0.6, -0.5, "left", "top"),
        "DPO seed 100": (1.2, 0, "left", "center"),
        "DPO seed 200": (-1.0, 1.5, "right", "bottom"),
        "DPO seed 300": (1.2, 1.5, "left", "bottom"),
    }
    for p in points:
        ax.scatter(p["syc_red_pp"], p["gsm"], s=p["size"], c=p["color"], marker=p["marker"],
                   edgecolors="black", linewidths=1.0, label=p["name"], zorder=3)
        dx, dy, ha, va = label_offsets.get(p["name"], (0.5, 1.2, "left", "bottom"))
        ax.annotate(p["name"], (p["syc_red_pp"] + dx, p["gsm"] + dy),
                    fontsize=FONT_SIZE_TICK, ha=ha, va=va)

    # Pareto-frontier annotation: dashed line at baseline GSM8k retention
    ax.axhline(baseline_gsm * 100, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(21.5, baseline_gsm * 100 + 0.5, "Baseline GSM8k retention",
            fontsize=FONT_SIZE_TICK - 1, color="gray", ha="right")

    # Capability collapse annotation for SFT
    sft_p = points[1]
    ax.annotate("Capability collapse:\nGSM8k 33.2% $\\to$ 5.8%",
                xy=(sft_p["syc_red_pp"], sft_p["gsm"]),
                xytext=(sft_p["syc_red_pp"] - 4, sft_p["gsm"] + 12),
                fontsize=FONT_SIZE_TICK, color=COLOR_SFT,
                arrowprops=dict(arrowstyle="->", color=COLOR_SFT, lw=0.8))

    ax.set_xlabel("Overall Sycophancy Reduction (pp from baseline 28.0%)")
    ax.set_ylabel("GSM8k Accuracy (%, full N=1,319)")
    ax.set_title(
        "Capability-Safety Tradeoff: DPO Preserves Reasoning, SFT Collapses It",
        fontsize=FONT_SIZE_TITLE,
    )
    ax.set_xlim(-2, 22)
    ax.set_ylim(0, 50)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig11_sft_dpo_tradeoff")


# ---------- F12: Cross-scale SC/BC comparison (Llama-3, Mistral, Qwen-14B) ----------
def fig12_cross_scale():
    # Pull SC/BC numbers per model. Llama-3 + Mistral from existing balanced probes,
    # Qwen-14B from the recent stronger/probe_control_balanced.json.
    def load_best_probe(path):
        d = json.load(open(path))
        # Schema: per_layer or per_position_summary; pull best layer
        if "per_position_summary" in d:
            s = d["per_position_summary"].get("final", {})
            return {
                "sc": s.get("best_social_compliance_rate", 0) * 100,
                "bc": s.get("best_belief_corruption_rate", 0) * 100,
                "rt": s.get("best_robust_rate", 0) * 100,
                "best_layer": s.get("best_layer"),
            }
        # Older schema
        if "summary" in d:
            s = d["summary"]
            return {
                "sc": s.get("best_social_compliance_rate", 0) * 100,
                "bc": s.get("best_belief_corruption_rate", 0) * 100,
                "rt": s.get("best_robust_rate", 0) * 100,
                "best_layer": s.get("best_layer"),
            }
        return {"sc": 0, "bc": 0, "rt": 0, "best_layer": None}

    llama = load_best_probe(RESULTS / "probe_control_balanced_results.json")
    mistral = load_best_probe(RESULTS / "mistral" / "probe_control_balanced_results.json")
    qwen = load_best_probe(RESULTS / "stronger" / "probe_control_balanced.json")

    models = [
        ("Llama-3-8B-Instruct", llama, COLOR_LLAMA),
        ("Mistral-7B-Instruct", mistral, COLOR_MISTRAL),
        ("Qwen2.5-14B-Instruct", qwen, COLOR_QWEN),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})

    # Left: grouped bars SC / BC / RT per model
    categories = ["Social Compliance", "Belief Corruption", "Robust Tracking"]
    x = np.arange(len(categories))
    w = 0.25
    for i, (name, vals, color) in enumerate(models):
        v = [vals["sc"], vals["bc"], vals["rt"]]
        offset = (i - 1) * w
        bars = ax1.bar(x + offset, v, w, color=color, label=f"{name} (best L={vals['best_layer']})",
                       edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, v):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}",
                     ha="center", fontsize=FONT_SIZE_TICK - 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("Probe Rate (%)")
    ax1.set_title("Probe Decomposition Across Three Model Families", fontsize=FONT_SIZE_TITLE - 1)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=FONT_SIZE_LEGEND - 1)
    ax1.set_ylim(0, max(m[1]["rt"] for m in models) + 12)
    ax1.grid(axis="y", alpha=0.3)

    # Right: SC:BC ratio
    ratios = [m[1]["sc"] / max(m[1]["bc"], 0.01) for m in models]
    bars = ax2.bar(range(len(models)), ratios, color=[m[2] for m in models],
                   edgecolor="black", linewidth=0.5)
    for i, r in enumerate(ratios):
        ax2.text(i, r + 0.1, f"{r:.1f}:1", ha="center", fontsize=FONT_SIZE_TICK)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m[0].split("-")[0] for m in models], fontsize=FONT_SIZE_TICK - 1)
    ax2.set_ylabel("SC:BC Ratio")
    ax2.set_title("SC:BC Ratio", fontsize=FONT_SIZE_TITLE - 1)
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_ylim(0, max(ratios) * 1.2)

    fig.suptitle(
        "Probe Decomposition Across Three Model Families",
        fontsize=FONT_SIZE_TITLE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "fig12_cross_scale")


# ---------- main ----------
FIGURES_BY_NUM = {
    7: fig7_dpo_seed_robustness,
    8: fig8_ood_generalization,
    9: fig9_freeform_5dim,
    10: fig10_transcript_panel,
    11: fig11_sft_dpo_tradeoff,
    12: fig12_cross_scale,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figures", nargs="*", type=int, default=list(FIGURES_BY_NUM.keys()))
    args = p.parse_args()

    for n in args.figures:
        if n not in FIGURES_BY_NUM:
            print(f"unknown figure {n}; skipping")
            continue
        print(f"Generating fig{n}...")
        try:
            FIGURES_BY_NUM[n]()
        except Exception as e:
            print(f"  ERROR fig{n}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
