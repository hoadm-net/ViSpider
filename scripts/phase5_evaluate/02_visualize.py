#!/usr/bin/env python3
"""
Phase 5 – Step 2: Visualization

Generate plots from the JSON produced by 01_labse_eval.py.

Dataset-quality mode produces:
  01_score_distribution.png  – histogram + KDE + threshold line
  02_cdf.png                 – cumulative distribution function
  03_boxplot_hardness.png    – box plot grouped by hardness
  04_boxplot_source.png      – box plot grouped by source
  05_passrate_bars.png       – pass-rate bar chart by hardness & source

Model-prediction mode (--pred-file) additionally produces:
  06_scatter_en_vs_ref.png   – EN↔PRED vs REF↔PRED scatter
  07_delta_histogram.png     – per-sample delta (EN↔PRED − REF↔PRED)

Usage
-----
# Visualize dev-split dataset-quality run
python3 scripts/phase5_evaluate/02_visualize.py

# Specify a different input file
python3 scripts/phase5_evaluate/02_visualize.py \\
    --input results/phase5_evaluate/vispider_test_labse.json

# Visualize a model-prediction run
python3 scripts/phase5_evaluate/02_visualize.py \\
    --input results/phase5_evaluate/model_eval_dev_labse.json
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results/phase5_evaluate"
DEFAULT_IN   = RESULTS_DIR / "vispider_dev_labse.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize LaBSE evaluation results")
    p.add_argument(
        "--input", "-i",
        default=str(DEFAULT_IN),
        help="JSON output from 01_labse_eval.py (default: vispider_dev_labse.json)",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to save PNG files (default: results/phase5_evaluate/plots/<stem>/)",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="Plot resolution in DPI (default: 150)",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() – useful in headless environments",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colour palette (colourblind-friendly)
# ---------------------------------------------------------------------------
PALETTE = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B3",
    "brown":  "#937860",
    "pink":   "#DA8BC3",
    "gray":   "#8C8C8C",
}
HARDNESS_ORDER = ["easy", "medium", "hard", "extra_hard", "unknown"]
SOURCE_ORDER   = ["manual", "gpt", "unknown"]

HARDNESS_COLORS = [PALETTE["blue"], PALETTE["orange"],
                   PALETTE["green"], PALETTE["red"], PALETTE["gray"]]
SOURCE_COLORS   = [PALETTE["blue"], PALETTE["orange"], PALETTE["gray"]]


def _color_seq(labels: list, order: list, colors: list) -> list:
    """Return each label's colour, falling back to gray for unknowns."""
    result = []
    for lbl in labels:
        if lbl in order:
            result.append(colors[order.index(lbl)])
        else:
            result.append(PALETTE["gray"])
    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def fig_score_distribution(scores: list, threshold: float, title: str, out: Path, dpi: int):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)

    n, _, patches = ax.hist(scores, bins=bins, color=PALETTE["blue"],
                            alpha=0.7, edgecolor="white", linewidth=0.6, label="Samples")

    # Colour bars below threshold red
    for patch, left in zip(patches, bins):
        if left < threshold:
            patch.set_facecolor(PALETTE["red"])
            patch.set_alpha(0.65)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(0, 1, 300)
        kde   = gaussian_kde(scores, bw_method=0.08)
        ax2   = ax.twinx()
        ax2.plot(kde_x, kde(kde_x), color="black", linewidth=1.5, label="KDE")
        ax2.set_ylabel("Density", fontsize=10)
        ax2.tick_params(labelsize=9)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize=9)
    except ImportError:
        pass

    ax.axvline(threshold, color=PALETTE["red"], linewidth=1.8,
               linestyle="--", label=f"Threshold {threshold}")
    ax.axvline(float(sum(scores) / len(scores)), color="black", linewidth=1.4,
               linestyle=":", label=f"Mean {sum(scores)/len(scores):.3f}")

    above = sum(1 for s in scores if s >= threshold)
    ax.set_xlabel("LaBSE Cosine Similarity", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{title}\n"
                 f"n={len(scores)}  mean={sum(scores)/len(scores):.4f}  "
                 f"≥{threshold}: {above}/{len(scores)} ({100*above/len(scores):.1f}%)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def fig_cdf(scores: list, threshold: float, title: str, out: Path, dpi: int):
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_s = sorted(scores)
    cdf      = [(i + 1) / len(sorted_s) for i in range(len(sorted_s))]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sorted_s, cdf, color=PALETTE["blue"], linewidth=2, label="CDF")
    ax.axvline(threshold, color=PALETTE["red"], linewidth=1.8,
               linestyle="--", label=f"Threshold {threshold}")

    # Mark the threshold quantile
    idx     = next((i for i, s in enumerate(sorted_s) if s >= threshold), len(sorted_s))
    q_val   = idx / len(sorted_s)
    ax.annotate(
        f"{100*(1-q_val):.1f}% ≥ {threshold}",
        xy=(threshold, q_val),
        xytext=(threshold - 0.28, q_val + 0.07),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )

    ax.set_xlabel("LaBSE Cosine Similarity", fontsize=11)
    ax.set_ylabel("Cumulative Fraction", fontsize=11)
    ax.set_title(f"CDF – {title}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def fig_boxplot(groups: dict[str, list], order: list, colors: list,
                xlabel: str, title: str, threshold: float, out: Path, dpi: int):
    import matplotlib.pyplot as plt

    labels  = [l for l in order if l in groups] + [l for l in groups if l not in order]
    data    = [groups[l] for l in labels]
    clrs    = _color_seq(labels, order, colors)
    n_items = [len(groups[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], clrs):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.axhline(threshold, color=PALETTE["red"], linewidth=1.4,
               linestyle="--", alpha=0.8, label=f"Threshold {threshold}")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(
        [f"{l}\n(n={n})" for l, n in zip(labels, n_items)],
        fontsize=9,
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("LaBSE Cosine Similarity", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def fig_passrate_bars(by_hardness: dict, by_source: dict, threshold: float,
                      title: str, out: Path, dpi: int):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def _draw(ax, groups, order, colors, group_label):
        labels  = [l for l in order if l in groups] + [l for l in groups if l not in order]
        pcts    = [groups[l].get(f"above_{threshold}_pct", 0) for l in labels]
        ns      = [groups[l].get("n", 0)                      for l in labels]
        clrs    = _color_seq(labels, order, colors)
        bars    = ax.bar(labels, pcts, color=clrs, alpha=0.75, edgecolor="white")
        ax.axhline(100 * (sum(pcts) / len(pcts)) if pcts else 0,
                   color="black", linewidth=1.2, linestyle=":", alpha=0.6,
                   label=f"Avg {sum(pcts)/len(pcts):.1f}%")
        for bar, pct, n in zip(bars, pcts, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(pct + 2, 97), f"{pct:.0f}%\nn={n}",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0, 110)
        ax.set_xlabel(group_label, fontsize=10)
        ax.set_ylabel(f"% samples ≥ {threshold}", fontsize=10)
        ax.set_title(f"Pass rate by {group_label}", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=9)

    _draw(ax1, by_hardness, HARDNESS_ORDER, HARDNESS_COLORS, "Hardness")
    _draw(ax2, by_source,   SOURCE_ORDER,   SOURCE_COLORS,   "Source")

    fig.suptitle(f"Pass Rate (≥ {threshold}) – {title}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def fig_scatter_en_vs_ref(records: list, threshold: float, title: str, out: Path, dpi: int):
    """EN↔PRED vs REF↔PRED scatter with histograms."""
    import matplotlib.pyplot as plt
    import numpy as np

    x = [r["en_pred_sim"]  for r in records]
    y = [r["ref_pred_sim"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, alpha=0.45, s=18, c=PALETTE["blue"], edgecolors="none")

    # Diagonal guide
    lim_min = min(min(x), min(y)) - 0.02
    lim_max = max(max(x), max(y)) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="gray", linewidth=1, linestyle="--", alpha=0.6, label="EN=REF line")

    ax.axvline(threshold, color=PALETTE["red"], linewidth=1.2, linestyle=":", alpha=0.7)
    ax.axhline(threshold, color=PALETTE["red"], linewidth=1.2, linestyle=":", alpha=0.7,
               label=f"Threshold {threshold}")

    corr = float(np.corrcoef(x, y)[0, 1])
    ax.set_xlabel("EN ↔ PRED similarity", fontsize=11)
    ax.set_ylabel("REF ↔ PRED similarity", fontsize=11)
    ax.set_title(f"{title}\nPearson r = {corr:.3f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ {out.name}")


def fig_delta_histogram(records: list, threshold: float, title: str, out: Path, dpi: int):
    """Histogram of delta = EN↔PRED − REF↔PRED."""
    import matplotlib.pyplot as plt
    import numpy as np

    deltas = [r["en_pred_sim"] - r["ref_pred_sim"] for r in records]
    mean_d = sum(deltas) / len(deltas)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-0.5, 0.5, 51)
    ax.hist(deltas, bins=bins, color=PALETTE["blue"], alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0,      color="black",        linewidth=1.5, linestyle="-",  label="Zero")
    ax.axvline(mean_d, color=PALETTE["red"], linewidth=1.5, linestyle="--",
               label=f"Mean delta {mean_d:+.3f}")

    ax.set_xlabel("EN↔PRED − REF↔PRED", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Per-sample delta – {title}\n"
                 f"Positive = model closer to EN than reference",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    in_path  = Path(args.input)

    if not in_path.exists():
        print(f"❌ File not found: {in_path}")
        print("   Run 01_labse_eval.py first to generate the input file.")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else (
        RESULTS_DIR / "plots" / in_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    stats   = data["stats"]
    records = data["records"]
    mode    = stats["mode"]
    thresh  = stats["threshold"]
    stem    = in_path.stem.replace("_labse", "")

    print(f"\n{'='*72}")
    print(f"VISUALIZING  →  {in_path.name}")
    print(f"Output dir   →  {out_dir}")
    print(f"{'='*72}\n")

    # ── Dataset quality mode ─────────────────────────────────────────────
    if mode == "dataset_quality":
        scores = [r["en_vi_sim"] for r in records]
        title  = f"ViSpider – {stem}"

        # Group records by hardness and source
        from collections import defaultdict
        by_hardness: dict[str, list] = defaultdict(list)
        by_source:   dict[str, list] = defaultdict(list)
        for r in records:
            by_hardness[r.get("hardness", "unknown")].append(r["en_vi_sim"])
            by_source  [r.get("source",   "unknown")].append(r["en_vi_sim"])

        fig_score_distribution(scores, thresh, title,
                               out_dir / "01_score_distribution.png", args.dpi)
        fig_cdf(scores, thresh, title,
                out_dir / "02_cdf.png", args.dpi)
        fig_boxplot(by_hardness, HARDNESS_ORDER, HARDNESS_COLORS,
                    "Hardness", f"Score Distribution by Hardness – {title}",
                    thresh, out_dir / "03_boxplot_hardness.png", args.dpi)
        fig_boxplot(by_source, SOURCE_ORDER, SOURCE_COLORS,
                    "Source", f"Score Distribution by Source – {title}",
                    thresh, out_dir / "04_boxplot_source.png", args.dpi)
        fig_passrate_bars(stats["by_hardness"], stats["by_source"], thresh, stem,
                          out_dir / "05_passrate_bars.png", args.dpi)

    # ── Model prediction mode ────────────────────────────────────────────
    elif mode == "model_prediction":
        title = f"Model eval – {stem}"

        # EN↔PRED distribution
        en_pred_scores = [r["en_pred_sim"] for r in records]
        fig_score_distribution(en_pred_scores, thresh, f"EN↔PRED – {stem}",
                               out_dir / "01_en_pred_distribution.png", args.dpi)

        # REF↔PRED distribution
        ref_pred_scores = [r["ref_pred_sim"] for r in records]
        fig_score_distribution(ref_pred_scores, thresh, f"REF↔PRED – {stem}",
                               out_dir / "02_ref_pred_distribution.png", args.dpi)

        # CDF for both
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, scores, color in [
            ("EN↔PRED",  en_pred_scores,  PALETTE["blue"]),
            ("REF↔PRED", ref_pred_scores, PALETTE["orange"]),
        ]:
            s = sorted(scores)
            c = [(i + 1) / len(s) for i in range(len(s))]
            ax.plot(s, c, color=color, linewidth=2, label=label)
        ax.axvline(thresh, color=PALETTE["red"], linewidth=1.6,
                   linestyle="--", label=f"Threshold {thresh}")
        ax.set_xlabel("LaBSE Cosine Similarity", fontsize=11)
        ax.set_ylabel("Cumulative Fraction", fontsize=11)
        ax.set_title(f"CDF comparison – {title}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(axis="both", linestyle="--", alpha=0.4)
        fig.tight_layout()
        cdf_out = out_dir / "03_cdf_comparison.png"
        fig.savefig(cdf_out, dpi=args.dpi); plt.close(fig)
        print(f"  ✓ {cdf_out.name}")

        # Boxplot by hardness – EN↔PRED
        from collections import defaultdict
        by_h_en: dict[str, list] = defaultdict(list)
        by_h_ref: dict[str, list] = defaultdict(list)
        for r in records:
            h = r.get("hardness", "unknown")
            by_h_en [h].append(r["en_pred_sim"])
            by_h_ref[h].append(r["ref_pred_sim"])
        fig_boxplot(by_h_en, HARDNESS_ORDER, HARDNESS_COLORS,
                    "Hardness", f"EN↔PRED by Hardness – {title}",
                    thresh, out_dir / "04_boxplot_hardness_en_pred.png", args.dpi)
        fig_boxplot(by_h_ref, HARDNESS_ORDER, HARDNESS_COLORS,
                    "Hardness", f"REF↔PRED by Hardness – {title}",
                    thresh, out_dir / "05_boxplot_hardness_ref_pred.png", args.dpi)

        # Pass-rate bars
        fig_passrate_bars(
            stats["en_pred"]["by_hardness"], stats["en_pred"]["by_source"], thresh,
            f"EN↔PRED – {stem}", out_dir / "06_passrate_en_pred.png", args.dpi,
        )

        # Scatter & delta
        fig_scatter_en_vs_ref(records, thresh, title,
                              out_dir / "07_scatter_en_vs_ref.png", args.dpi)
        fig_delta_histogram(records, thresh, title,
                            out_dir / "08_delta_histogram.png", args.dpi)

    else:
        print(f"❌ Unknown mode '{mode}' in stats. Cannot visualize.")
        sys.exit(1)

    print(f"\n✅ All plots saved to: {out_dir}")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
