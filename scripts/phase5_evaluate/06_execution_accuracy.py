#!/usr/bin/env python3
"""
Phase 5 – Step 6: Execution Accuracy (EX)

Executes both gold and predicted SQL against Spider's SQLite databases and
computes Execution Accuracy: fraction where result sets match exactly.

Databases are read from  data/raw/database/{db_id}/{db_id}.sqlite
(same layout as the original Spider release).

Usage
-----
# Evaluate predictions for English fine-tuned model on test split
python3 scripts/phase5_evaluate/06_execution_accuracy.py \\
    --pred-file results/phase5_evaluate/predictions_en_test.json

# Evaluate VI predictions
python3 scripts/phase5_evaluate/06_execution_accuracy.py \\
    --pred-file results/phase5_evaluate/predictions_vi_test.json

# Specify custom database directory
python3 scripts/phase5_evaluate/06_execution_accuracy.py \\
    --pred-file results/phase5_evaluate/predictions_en_test.json \\
    --db-dir data/raw/database
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_DB_DIR   = PROJECT_ROOT / "data/raw/database"
RESULTS_BASE     = PROJECT_ROOT / "results/phase5_evaluate"


def parse_args():
    p = argparse.ArgumentParser(description="Compute Execution Accuracy for predicted SQL")
    p.add_argument("--pred-file", required=True,
                   help="JSON file produced by 05_predict_sql.py")
    p.add_argument("--db-dir",    default=str(DEFAULT_DB_DIR),
                   help="Root directory containing {db_id}/{db_id}.sqlite")
    p.add_argument("--output",    default=None,
                   help="Output path (default: next to pred-file with _ex suffix)")
    p.add_argument("--timeout",   type=float, default=30.0,
                   help="Per-query SQLite timeout in seconds (default: 30)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# SQL execution helpers
# ---------------------------------------------------------------------------

def execute_sql(db_path: Path, sql: str, timeout: float) -> tuple[bool, list | str]:
    """
    Execute SQL against a SQLite database.
    Returns (success, result_rows_or_error_message).
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        # Normalise: sort rows, lower-case strings for case-insensitive compare
        normalised = frozenset(
            tuple(_norm_val(v) for v in row)
            for row in rows
        )
        return True, normalised
    except Exception as e:
        return False, str(e)


def _norm_val(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def results_match(gold_result, pred_result) -> bool:
    """True when both result sets are equal (order-insensitive)."""
    return gold_result == pred_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    pred_path = Path(args.pred_file)
    db_dir    = Path(args.db_dir)

    if not pred_path.exists():
        print(f"❌ Prediction file not found: {pred_path}")
        sys.exit(1)

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    lang  = predictions[0].get("lang",  "?") if predictions else "?"
    split = predictions[0].get("split", "?") if predictions else "?"
    mode  = predictions[0].get("mode",  "?") if predictions else "?"

    print("=" * 72)
    print(f"EXECUTION ACCURACY  [{lang.upper()}  {split}  {mode}]")
    print(f"  Predictions : {len(predictions)}")
    print(f"  DB dir      : {db_dir}")
    print("=" * 72 + "\n")

    correct        = 0
    total          = 0
    gold_errors    = []
    pred_errors    = []
    results_detail = []

    by_hardness: dict[str, list] = defaultdict(list)
    by_source:   dict[str, list] = defaultdict(list)

    for pred in predictions:
        db_id    = pred["db_id"]
        db_path  = db_dir / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"  ⚠  DB not found: {db_path} – skipping {pred['id']}")
            continue

        total += 1
        gold_ok, gold_result = execute_sql(db_path, pred["gold_sql"], args.timeout)
        pred_ok, pred_result = execute_sql(db_path, pred["pred_sql"], args.timeout)

        if not gold_ok:
            gold_errors.append({"id": pred["id"], "sql": pred["gold_sql"], "error": gold_result})
            # Can't evaluate this sample
            results_detail.append({**pred, "ex": None, "gold_error": gold_result})
            continue

        if not pred_ok:
            pred_errors.append({"id": pred["id"], "sql": pred["pred_sql"], "error": pred_result})
            match = False
        else:
            match = results_match(gold_result, pred_result)

        if match:
            correct += 1

        ex_val = 1 if match else 0
        by_hardness[pred.get("hardness", "unknown")].append(ex_val)
        by_source  [pred.get("source",   "unknown")].append(ex_val)

        results_detail.append({
            **pred,
            "ex":         ex_val,
            "pred_error": None if pred_ok else pred_result,
        })

        if total % 50 == 0:
            print(f"  [{total}/{len(predictions)}]  EX so far: {correct}/{total} "
                  f"({100*correct/total:.1f}%)")

    # ── Aggregate stats ────────────────────────────────────────────────────
    evaluated = sum(1 for r in results_detail if r.get("ex") is not None)
    ex_score  = correct / evaluated if evaluated else 0.0

    HARDNESS_ORDER = ["easy", "medium", "hard", "extra_hard"]
    by_hardness_stats = {}
    for h in HARDNESS_ORDER + [k for k in by_hardness if k not in HARDNESS_ORDER]:
        v = by_hardness.get(h)
        if v:
            by_hardness_stats[h] = {
                "n":       len(v),
                "correct": sum(v),
                "ex_pct":  round(100 * sum(v) / len(v), 1),
            }

    by_source_stats = {
        s: {"n": len(v), "correct": sum(v), "ex_pct": round(100 * sum(v) / len(v), 1)}
        for s, v in sorted(by_source.items())
    }

    stats = {
        "lang":             lang,
        "split":            split,
        "mode":             mode,
        "total_samples":    total,
        "evaluated":        evaluated,
        "correct":          correct,
        "execution_accuracy": round(ex_score, 4),
        "execution_accuracy_pct": round(100 * ex_score, 1),
        "gold_errors":      len(gold_errors),
        "pred_errors":      len(pred_errors),
        "by_hardness":      by_hardness_stats,
        "by_source":        by_source_stats,
    }

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"\n  Execution Accuracy : {stats['correct']}/{stats['evaluated']} "
          f"= {stats['execution_accuracy_pct']}%")
    if gold_errors:
        print(f"  Gold SQL errors    : {len(gold_errors)} (excluded from EX)")
    if pred_errors:
        print(f"  Pred SQL errors    : {len(pred_errors)} (counted as wrong)")

    print(f"\n  By hardness:")
    for h, v in by_hardness_stats.items():
        print(f"    {h:<12}  {v['correct']}/{v['n']} = {v['ex_pct']}%")

    print(f"\n  By source:")
    for s, v in by_source_stats.items():
        print(f"    {s:<12}  {v['correct']}/{v['n']} = {v['ex_pct']}%")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else (
        pred_path.parent / (pred_path.stem + "_ex.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats":         stats,
            "gold_errors":   gold_errors,
            "pred_errors":   pred_errors[:50],   # cap to keep file manageable
            "results":       results_detail,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved → {out_path}")


if __name__ == "__main__":
    main()
