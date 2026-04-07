#!/usr/bin/env python3
"""Generate publication-quality figures for the sycophancy mitigation paper.

Reads result JSON files and produces PDF + PNG figures suitable for NeurIPS.

Usage:
    python scripts/generate_figures.py                       # all figures
    python scripts/generate_figures.py --figures 1 3 5       # selected figures
    python scripts/generate_figures.py --output-dir plots/   # custom output dir
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8

def setup_style():
    """Configure matplotlib for clean, publication-ready plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,     # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


def get_layer_palette(layers):
    """Return a colour map from layer index to colour."""
    if HAS_SEABORN:
        colors = sns.color_palette("viridis", n_colors=len(layers))
    else:
        cmap = plt.cm.viridis
        colors = [cmap(i / max(1, len(layers) - 1)) for i in range(len(layers))]
    return {l: c for l, c in zip(layers, colors)}


def _load_json(path):
    """Load a JSON file, returning None if it does not exist."""
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {p} not found")
        return None
    with open(p) as f:
        return json.load(f)


def _save(fig, output_dir, name):
    """Save figure as both PDF and PNG."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(out)
        print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Activation-patching heatmap
# ---------------------------------------------------------------------------

def figure_patching_heatmap(data_dir, output_dir):
    """Patching recovery heatmap with side-panel layer importance."""
    data = _load_json(os.path.join(data_dir, "patching_heatmap.json"))
    if data is None:
        return
    lr = data["layer_results"]
    heatmap = np.array(lr["mean_recovery_heatmap"])  # (layers, positions)
    importance = lr["layer_importance"]               # dict str->float
    layers = data["metadata"]["layers"]
    n_layers, n_pos = heatmap.shape

    # Clip extreme values for better colour range
    vmax = np.percentile(np.abs(heatmap), 97)
    heatmap_clipped = np.clip(heatmap, -vmax, vmax)

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)

    # --- Heatmap ---
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(
        heatmap_clipped,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax0.set_xlabel("Token position")
    ax0.set_ylabel("Layer")
    ax0.set_title("Activation-patching recovery score")

    # Thin out y-ticks
    ytick_step = max(1, n_layers // 8)
    ax0.set_yticks(range(0, n_layers, ytick_step))
    ax0.set_yticklabels([str(layers[i]) for i in range(0, n_layers, ytick_step)])

    xtick_step = max(1, n_pos // 10)
    ax0.set_xticks(range(0, n_pos, xtick_step))

    cb = fig.colorbar(im, ax=ax0, fraction=0.03, pad=0.02)
    cb.set_label("Recovery score", fontsize=FONT_SIZE_TICK)
    cb.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)

    # --- Side panel: layer importance ---
    ax1 = fig.add_subplot(gs[1])
    imp_vals = [importance[str(l)] for l in layers]
    ax1.barh(range(n_layers), imp_vals, color="#4C72B0", height=0.8)
    ax1.set_ylim(ax0.get_ylim())
    ax1.set_yticks([])
    ax1.set_xlabel("Importance")
    ax1.set_title("Layer importance")
    ax1.invert_yaxis()

    _save(fig, output_dir, "fig1_patching_heatmap")


# ---------------------------------------------------------------------------
# Figure 2: Steering alpha sweep
# ---------------------------------------------------------------------------

def figure_steering_sweep(data_dir, output_dir):
    """Sycophancy rate vs alpha per layer, with capability retention subplot."""
    data = _load_json(os.path.join(data_dir, "steering_results.json"))
    if data is None:
        return

    target_layers = [1, 2, 3, 4, 5, 10, 15, 20]
    alphas_tested = sorted(data["metadata"]["alphas_tested"])
    conditions = data["conditions"]
    baseline_rate = conditions["baseline"]["sycophancy"]["overall_sycophancy_rate"]

    palette = get_layer_palette(target_layers)

    fig, (ax_syc, ax_cap) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Collect data per layer
    for layer in target_layers:
        alphas_plot = []
        syc_rates = []
        mmlu_ret = []
        gsm8k_ret = []
        for alpha in alphas_tested:
            key = f"layer{layer}_alpha{alpha}"
            if key not in conditions:
                continue
            cond = conditions[key]
            alphas_plot.append(alpha)
            syc_rates.append(cond["sycophancy"]["overall_sycophancy_rate"])
            if "mmlu" in cond:
                mmlu_ret.append(cond.get("mmlu_retained", None))
                gsm8k_ret.append(cond.get("gsm8k_retained", None))
            else:
                mmlu_ret.append(None)
                gsm8k_ret.append(None)

        color = palette[layer]
        ax_syc.plot(alphas_plot, syc_rates, "o-", color=color, label=f"L{layer}",
                    markersize=4, linewidth=1.5)

        # Capability subplot: only points where we have data
        mmlu_valid = [(a, m) for a, m in zip(alphas_plot, mmlu_ret) if m is not None]
        gsm8k_valid = [(a, g) for a, g in zip(alphas_plot, gsm8k_ret) if g is not None]
        if mmlu_valid:
            a_m, v_m = zip(*mmlu_valid)
            ax_cap.plot(a_m, v_m, "o-", color=color, markersize=3, linewidth=1.2,
                        alpha=0.85)
        if gsm8k_valid:
            a_g, v_g = zip(*gsm8k_valid)
            ax_cap.plot(a_g, v_g, "s--", color=color, markersize=3, linewidth=1.0,
                        alpha=0.55)

    # Baseline dashed line
    ax_syc.axhline(baseline_rate, ls="--", color="grey", linewidth=1, label="Baseline")
    ax_syc.set_xscale("log")
    ax_syc.set_xlabel(r"Steering strength $\alpha$")
    ax_syc.set_ylabel("Overall sycophancy rate")
    ax_syc.set_title("Sycophancy rate vs. steering strength")
    ax_syc.legend(ncol=3, loc="upper left", frameon=False)

    ax_cap.axhline(1.0, ls="--", color="grey", linewidth=1)
    ax_cap.set_xscale("log")
    ax_cap.set_xlabel(r"Steering strength $\alpha$")
    ax_cap.set_ylabel("Capability retention (fraction of baseline)")
    ax_cap.set_title("MMLU (circles) / GSM8K (squares) retention")
    ax_cap.set_ylim(-0.05, 1.15)

    fig.tight_layout()
    _save(fig, output_dir, "fig2_steering_sweep")


# ---------------------------------------------------------------------------
# Figure 3: Steering per-source (opinion) analysis
# ---------------------------------------------------------------------------

def figure_steering_per_source(data_dir, output_dir):
    """Opinion sycophancy rate vs alpha per layer with baseline CI band."""
    data = _load_json(os.path.join(data_dir, "steering_per_source_analysis.json"))
    if data is None:
        return

    baseline = data["baseline"]
    bl_rate = baseline["opinion_sycophancy_rate"]
    bl_ci = baseline["opinion_wilson_ci_95"]
    conditions = data["conditions_sorted_by_opinion_reduction"]

    # Parse conditions into per-layer series
    target_layers = [1, 2, 3, 4, 5, 10, 15, 20]
    palette = get_layer_palette(target_layers)
    layer_data = {l: {"alpha": [], "rate": []} for l in target_layers}

    for cond in conditions:
        m = re.match(r"layer(\d+)_alpha([\d.]+)", cond["condition"])
        if not m:
            continue
        layer = int(m.group(1))
        alpha = float(m.group(2))
        if layer not in layer_data:
            continue
        layer_data[layer]["alpha"].append(alpha)
        layer_data[layer]["rate"].append(cond["opinion_syc_rate"])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Baseline CI band
    ax.axhspan(bl_ci[0], bl_ci[1], color="grey", alpha=0.18, label="Baseline 95% CI")
    ax.axhline(bl_rate, ls="--", color="grey", linewidth=1, label=f"Baseline ({bl_rate:.1%})")

    for layer in target_layers:
        ld = layer_data[layer]
        if not ld["alpha"]:
            continue
        # Sort by alpha
        order = np.argsort(ld["alpha"])
        alphas = np.array(ld["alpha"])[order]
        rates = np.array(ld["rate"])[order]
        ax.plot(alphas, rates, "o-", color=palette[layer], label=f"L{layer}",
                markersize=4, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Steering strength $\alpha$")
    ax.set_ylabel("Opinion sycophancy rate")
    ax.set_title("Opinion sycophancy vs. steering strength (per layer)")
    ax.legend(ncol=3, loc="lower left", frameon=False)
    ax.set_ylim(0.35, 0.95)

    fig.tight_layout()
    _save(fig, output_dir, "fig3_steering_per_source")


# ---------------------------------------------------------------------------
# Figure 4: Probe accuracy by layer
# ---------------------------------------------------------------------------

def figure_probe_accuracy(data_dir, output_dir):
    """Neutral CV accuracy and biased transfer accuracy by layer, with gap shading."""
    data = _load_json(os.path.join(data_dir, "probe_results_neutral_transfer.json"))
    if data is None:
        return

    # Use "final" position
    per_layer = data["per_layer"]["final"]
    layer_ids = sorted(per_layer.keys(), key=int)
    layers_int = [int(l) for l in layer_ids]

    neutral_acc = [per_layer[l]["neutral_cv_accuracy"] for l in layer_ids]
    biased_acc = [per_layer[l]["biased_transfer_accuracy"] for l in layer_ids]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(layers_int, neutral_acc, "o-", color="#4C72B0", label="Neutral CV accuracy",
            markersize=4, linewidth=1.5)
    ax.plot(layers_int, biased_acc, "s-", color="#DD8452", label="Biased transfer accuracy",
            markersize=4, linewidth=1.5)

    # Shade the gap (social compliance zone)
    ax.fill_between(layers_int, biased_acc, neutral_acc,
                     where=[n > b for n, b in zip(neutral_acc, biased_acc)],
                     alpha=0.20, color="#C44E52", label="Social compliance gap")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe accuracy: neutral vs. biased transfer")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(layers_int[0], layers_int[-1])
    ax.set_ylim(0.45, 1.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    fig.tight_layout()
    _save(fig, output_dir, "fig4_probe_accuracy")


# ---------------------------------------------------------------------------
# Figure 5: Ablation comparison (corrected + original head sets)
# ---------------------------------------------------------------------------

def figure_ablation_comparison(data_dir, output_dir):
    """Bar chart comparing sycophancy rates across ablation conditions with Wilson CIs."""
    corrected = _load_json(os.path.join(data_dir, "corrected_ablation_results.json"))
    original = _load_json(os.path.join(data_dir, "head_ablation_results.json"))
    if corrected is None and original is None:
        return

    # Build lists: (label, rate, ci_lo, ci_hi, group)
    entries = []

    def _add_conditions(src, group_label, label_prefix=""):
        if src is None:
            return
        conds = src["conditions"]
        for key, cond in conds.items():
            syc = cond["sycophancy"]
            rate = syc["overall_sycophancy_rate"]
            ci = syc["overall_sycophancy_rate_ci"]
            n_eval = syc["total_evaluated"]
            # Skip degenerate conditions (all skipped)
            if n_eval == 0:
                continue
            heads_str = cond.get("heads_ablated_str", [])
            if key == "baseline":
                label = "Baseline"
            elif key.startswith("all_"):
                mode = cond.get("mode", "zero")
                label = f"All ({mode})"
            else:
                label = " + ".join(heads_str)
            entries.append({
                "label": f"{label_prefix}{label}",
                "rate": rate,
                "ci_lo": ci[0],
                "ci_hi": ci[1],
                "group": group_label,
                "key": key,
            })

    _add_conditions(corrected, "Validated top-3\n(L4H28, L4H5, L5H31)", "")
    _add_conditions(original, "Original top-3\n(L1H20, L5H5, L4H28)", "")

    if not entries:
        return

    # De-duplicate baselines: keep only one
    seen_baseline = False
    deduped = []
    for e in entries:
        if e["key"] == "baseline":
            if seen_baseline:
                continue
            seen_baseline = True
        deduped.append(e)
    entries = deduped

    # Separate by group
    groups = []
    seen_groups = []
    for e in entries:
        if e["group"] not in seen_groups:
            seen_groups.append(e["group"])
    groups = seen_groups

    fig, ax = plt.subplots(figsize=(10, 5))

    if HAS_SEABORN:
        group_colors = sns.color_palette("Set2", n_colors=len(groups))
    else:
        group_colors = [plt.cm.Set2(i / max(1, len(groups) - 1)) for i in range(len(groups))]
    gcolor = {g: c for g, c in zip(groups, group_colors)}

    # Sort: baseline first, then by group, then by number of heads
    def _sort_key(e):
        if e["key"] == "baseline":
            return (0, 0, "")
        gidx = groups.index(e["group"])
        n_heads = e["label"].count("+") + 1
        return (1, gidx, n_heads, e["label"])

    entries.sort(key=_sort_key)

    x = np.arange(len(entries))
    labels = [e["label"] for e in entries]
    rates = [e["rate"] for e in entries]
    yerr_lo = [e["rate"] - e["ci_lo"] for e in entries]
    yerr_hi = [e["ci_hi"] - e["rate"] for e in entries]
    colors = []
    for e in entries:
        if e["key"] == "baseline":
            colors.append("0.6")
        else:
            colors.append(gcolor[e["group"]])

    ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, rates, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.3",
                capsize=3, linewidth=1)

    # Baseline reference line
    bl_rate = entries[0]["rate"]
    ax.axhline(bl_rate, ls="--", color="grey", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=FONT_SIZE_TICK - 1)
    ax.set_ylabel("Overall sycophancy rate")
    ax.set_title("Head ablation: sycophancy rate across conditions")

    # Legend for groups
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="0.6", label="Baseline")]
    for g in groups:
        handles.append(Patch(facecolor=gcolor[g], label=g))
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=FONT_SIZE_LEGEND)

    fig.tight_layout()
    _save(fig, output_dir, "fig5_ablation_comparison")


# ---------------------------------------------------------------------------
# Figure 6: DPO probe decomposition (pre vs post)
# ---------------------------------------------------------------------------

def figure_dpo_probe(data_dir, output_dir):
    """Pre-DPO vs Post-DPO probe decomposition grouped bar chart."""
    data = _load_json(os.path.join(data_dir, "dpo_eval_results.json"))
    if data is None:
        return

    # Extract pre-DPO and post-DPO probe numbers at best layer (layer 1)
    probes = data.get("probes", {})
    comparison = data.get("comparison", {})

    # Try to get layer 1 data from probes section
    per_layer = probes.get("per_layer", {})
    layer_key = "1"
    if layer_key not in per_layer:
        # Fall back to first available layer
        if per_layer:
            layer_key = sorted(per_layer.keys(), key=lambda x: int(x))[0]
        else:
            print("  [SKIP] No per-layer probe data found")
            return

    post = per_layer[layer_key]

    # Get pre-DPO numbers from comparison section (keyed as "layer_N")
    layer_comp = comparison.get("probes", {}).get(f"layer_{layer_key}", {})

    categories = ["Robust\ntracking", "Social\ncompliance", "Belief\ncorruption", "Other"]

    # Pre-DPO values from comparison (stored as fractions)
    pre_robust = layer_comp.get("robust_tracking", {}).get("pre_dpo", 0.599) * 100
    pre_social = layer_comp.get("social_compliance", {}).get("pre_dpo", 0.180) * 100
    pre_belief = layer_comp.get("belief_corruption", {}).get("pre_dpo", 0.101) * 100
    pre_other = 100.0 - pre_robust - pre_social - pre_belief

    # Post-DPO values from per-layer probes (stored as fractions)
    post_robust = post.get("robust_rate", 0) * 100
    post_social = post.get("social_compliance_rate", 0) * 100
    post_belief = post.get("belief_corruption_rate", 0) * 100
    post_other = 100.0 - post_robust - post_social - post_belief

    pre_vals = [pre_robust, pre_social, pre_belief, pre_other]
    post_vals = [post_robust, post_social, post_belief, post_other]

    x = np.arange(len(categories))
    width = 0.35

    if HAS_SEABORN:
        c_pre, c_post = sns.color_palette("Set2", 2)
    else:
        c_pre = plt.cm.Set2(0.0)
        c_post = plt.cm.Set2(0.15)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_pre = ax.bar(x - width / 2, pre_vals, width, label="Pre-DPO", color=c_pre, edgecolor="white")
    bars_post = ax.bar(x + width / 2, post_vals, width, label="Post-DPO", color=c_post, edgecolor="white")

    # Add value labels on bars
    for bar_group in [bars_pre, bars_post]:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FONT_SIZE_TICK)

    # Delta annotations
    for i, (pre, post_v) in enumerate(zip(pre_vals, post_vals)):
        delta = post_v - pre
        sign = "+" if delta >= 0 else ""
        color = "#2ca02c" if (i == 0 and delta > 0) or (i > 0 and delta < 0) else "#d62728"
        ax.annotate(f"{sign}{delta:.1f}pp", xy=(x[i], max(pre, post_v) + 4),
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICK, fontweight="bold",
                    color=color)

    ax.set_ylabel("Percentage of samples")
    ax.set_title("Probe decomposition: Pre-DPO vs Post-DPO (Layer 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(max(pre_vals), max(post_vals)) + 12)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    _save(fig, output_dir, "fig6_dpo_probe_decomposition")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURE_MAP = {
    "1": ("Patching heatmap", figure_patching_heatmap),
    "2": ("Steering alpha sweep", figure_steering_sweep),
    "3": ("Steering per-source (opinion)", figure_steering_per_source),
    "4": ("Probe accuracy by layer", figure_probe_accuracy),
    "5": ("Ablation comparison", figure_ablation_comparison),
    "6": ("DPO probe decomposition", figure_dpo_probe),
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--figures", nargs="*", default=None,
        help="Which figures to generate (e.g. 1 3 5). Default: all.",
    )
    parser.add_argument(
        "--output-dir", default="figures/",
        help="Directory to save figures (default: figures/).",
    )
    parser.add_argument(
        "--data-dir", default="results/",
        help="Directory containing result JSON files (default: results/).",
    )
    args = parser.parse_args()

    setup_style()

    figs = list(FIGURE_MAP.keys()) if args.figures is None else args.figures

    for fig_id in figs:
        if fig_id not in FIGURE_MAP:
            print(f"Unknown figure id '{fig_id}'. Available: {list(FIGURE_MAP.keys())}")
            continue
        name, func = FIGURE_MAP[fig_id]
        print(f"Generating Figure {fig_id}: {name}")
        try:
            func(args.data_dir, args.output_dir)
        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
