#!/usr/bin/env python3
"""
Phase 5 – Step 7: Compare EN vs VI Execution Accuracy

Side-by-side comparison of the two language variants, with a breakdown by
hardness level and optional visualization.

Usage
-----
# Compare EN and VI results (test split)
python3 scripts/phase5_evaluate/07_compare_en_vi.py \\
    --en results/phase5_evaluate/predictions_en_test_ex.json \\
    --vi results/phase5_evaluate/predictions_vi_test_ex.json

# Also save comparison plots
python3 scripts/phase5_evaluate/07_compare_en_vi.py \\
    --en results/phase5_evaluate/predictions_en_test_ex.json \\
    --vi results/phase5_evaluate/predictions_vi_test_ex.json \\
    --plot
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_BASE = PROJECT_ROOT / "results/phase5_evaluate"

HARDNESS_ORDER = ["easy", "medium", "hard", "extra_hard"]


def parse_args():
    p = argparse.ArgumentParser(description="Compare EN vs VI Execution Accuracy")
    p.add_argument("--en",     required=True,
                   help="EX result JSON for English (output of 06_execution_accuracy.py)")
    p.add_argument("--vi",     required=True,
                   help="EX result JSON for Vietnamese")
    p.add_argument("--output", default=None,
                   help="Save comparison JSON (default: results/phase5_evaluate/comparison_en_vi.json)")
    p.add_argument("--plot",   action="store_true",
                   help="Generate comparison bar charts")
    p.add_argument("--dpi",    type=int, default=150)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def load_ex(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["stats"]


def print_table(rows: list[tuple], headers: list[str], col_widths: list[int]):
    """Simple fixed-width table printer."""
    def fmt(val, w):
        return str(val).ljust(w)[:w]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "|" + "|".join(f" {fmt(h, w)} " for h, w in zip(headers, col_widths)) + "|"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "|" + "|".join(f" {fmt(v, w)} " for v, w in zip(row, col_widths)) + "|"
        print(line)
    print(sep)


def make_plots(en_stats: dict, vi_stats: dict, out_dir: Path, dpi: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PALETTE = {"blue": "#4C72B0", "orange": "#DD8452", "red": "#C44E52", "gray": "#8C8C8C"}
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Overall bar ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    langs = ["English (EN)", "Vietnamese (VI)"]
    vals  = [en_stats["execution_accuracy_pct"], vi_stats["execution_accuracy_pct"]]
    bars  = ax.bar(langs, vals, color=[PALETTE["blue"], PALETTE["orange"]],
                   alpha=0.8, edgecolor="white", width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    gap = vals[0] - vals[1]
    ax.set_title(f"Execution Accuracy\nCross-lingual gap: {gap:+.1f}%", fontsize=12)
    ax.set_ylabel("Execution Accuracy (%)", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = out_dir / "01_overall_ex.png"
    fig.savefig(out, dpi=dpi); plt.close(fig)
    print(f"  ✓ {out.name}")

    # ── By hardness grouped bar ──────────────────────────────────────────
    hardnesses = [h for h in HARDNESS_ORDER if h in en_stats["by_hardness"] or h in vi_stats["by_hardness"]]
    en_vals  = [en_stats["by_hardness"].get(h, {}).get("ex_pct", 0) for h in hardnesses]
    vi_vals  = [vi_stats["by_hardness"].get(h, {}).get("ex_pct", 0) for h in hardnesses]
    x        = np.arange(len(hardnesses))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_en = ax.bar(x - width/2, en_vals, width, label="EN", color=PALETTE["blue"],   alpha=0.8)
    bars_vi = ax.bar(x + width/2, vi_vals, width, label="VI", color=PALETTE["orange"], alpha=0.8)

    for bar, v in list(zip(bars_en, en_vals)) + list(zip(bars_vi, vi_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(hardnesses, fontsize=10)
    ax.set_ylabel("Execution Accuracy (%)", fontsize=11)
    ax.set_title("Execution Accuracy by Hardness: EN vs VI", fontsize=12)
    ax.set_ylim(0, max(max(en_vals, default=0), max(vi_vals, default=0)) * 1.25)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = out_dir / "02_ex_by_hardness.png"
    fig.savefig(out, dpi=dpi); plt.close(fig)
    print(f"  ✓ {out.name}")

    # ── Gap by hardness ─────────────────────────────────────────────────
    gaps    = [e - v for e, v in zip(en_vals, vi_vals)]
    colors  = [PALETTE["red"] if g > 0 else PALETTE["blue"] for g in gaps]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hardnesses, gaps, color=colors, alpha=0.75, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    for i, (h, g) in enumerate(zip(hardnesses, gaps)):
        ax.text(i, g + (0.3 if g >= 0 else -1.2),
                f"{g:+.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("EN − VI gap (pp)", fontsize=11)
    ax.set_title("Cross-lingual Gap (EN − VI) by Hardness", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = out_dir / "03_gap_by_hardness.png"
    fig.savefig(out, dpi=dpi); plt.close(fig)
    print(f"  ✓ {out.name}")


def main():
    args = parse_args()

    en_stats = load_ex(args.en)
    vi_stats = load_ex(args.vi)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EN vs VI EXECUTION ACCURACY COMPARISON")
    print("=" * 72)

    gap = en_stats["execution_accuracy_pct"] - vi_stats["execution_accuracy_pct"]
    rows = [
        ("Overall EX",
         f"{en_stats['correct']}/{en_stats['evaluated']} ({en_stats['execution_accuracy_pct']}%)",
         f"{vi_stats['correct']}/{vi_stats['evaluated']} ({vi_stats['execution_accuracy_pct']}%)",
         f"{gap:+.1f} pp"),
    ]
    print_table([("", "EN", "VI", "Gap (EN−VI)")], ["Metric", "EN", "VI", "Gap"], [14, 22, 22, 12])
    print_table(rows, ["Metric", "EN", "VI", "Gap (EN−VI)"], [14, 22, 22, 12])

    # By hardness
    print("\n  By hardness:")
    h_rows = []
    all_hardness = [h for h in HARDNESS_ORDER
                    if h in en_stats["by_hardness"] or h in vi_stats["by_hardness"]]
    for h in all_hardness:
        e = en_stats["by_hardness"].get(h, {})
        v = vi_stats["by_hardness"].get(h, {})
        e_str = f"{e.get('correct',0)}/{e.get('n',0)} ({e.get('ex_pct',0):.1f}%)"
        v_str = f"{v.get('correct',0)}/{v.get('n',0)} ({v.get('ex_pct',0):.1f}%)"
        g_val = e.get('ex_pct', 0) - v.get('ex_pct', 0)
        h_rows.append((h, e_str, v_str, f"{g_val:+.1f} pp"))
    print_table(h_rows, ["Hardness", "EN", "VI", "Gap"], [12, 22, 22, 10])

    # ── Save comparison JSON ───────────────────────────────────────────────
    comparison = {
        "en": en_stats,
        "vi": vi_stats,
        "gap_overall_pp":    round(gap, 2),
        "gap_by_hardness_pp": {
            h: round(
                en_stats["by_hardness"].get(h, {}).get("ex_pct", 0)
                - vi_stats["by_hardness"].get(h, {}).get("ex_pct", 0),
                2
            )
            for h in all_hardness
        },
    }
    out_path = Path(args.output) if args.output else (RESULTS_BASE / "comparison_en_vi.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved → {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if args.plot:
        plot_dir = out_path.parent / "plots" / "comparison_en_vi"
        print(f"\n  Generating plots → {plot_dir}")
        make_plots(en_stats, vi_stats, plot_dir, args.dpi)

    if not args.no_show and args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
