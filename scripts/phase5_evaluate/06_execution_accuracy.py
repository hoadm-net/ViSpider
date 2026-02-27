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

# Add spider_eval to path so we can import process_sql and evaluation
SPIDER_EVAL_DIR = PROJECT_ROOT / "scripts/utils/spider_eval"
sys.path.insert(0, str(SPIDER_EVAL_DIR))

DEFAULT_DB_DIR   = PROJECT_ROOT / "data/raw/database"
RESULTS_BASE     = PROJECT_ROOT / "results/phase5_evaluate"

# Try to import Spider's official eval functions
try:
    from process_sql import Schema, get_schema, get_sql  # type: ignore
    from evaluation import eval_exec_match               # type: ignore
    SPIDER_EVAL_AVAILABLE = True
except Exception as _e:
    SPIDER_EVAL_AVAILABLE = False
    print(f"⚠  Spider eval not available ({_e}), falling back to frozenset comparison")

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

def _norm_val(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def _exec_raw(db_path: Path, sql: str, timeout: float):
    """Execute SQL, return (success, rows_or_error)."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows
    except Exception as e:
        return False, str(e)


def compare_results(db_path: Path, p_str: str, g_str: str,
                    timeout: float) -> tuple[bool | None, str | None, str | None]:
    """
    Compare execution results using Spider's official eval_exec_match when
    available, falling back to normalised frozenset comparison.

    Returns (match: bool|None, pred_error: str|None, gold_error: str|None).
    match is None when gold SQL itself fails.
    """
    db_str = str(db_path)

    # ── Spider official path ────────────────────────────────────────────────
    if SPIDER_EVAL_AVAILABLE:
        try:
            schema = Schema(get_schema(db_str))
            p_sql  = get_sql(schema, p_str)
            g_sql  = get_sql(schema, g_str)
            # eval_exec_match opens its own connection; gold error will raise
            match = eval_exec_match(db_str, p_str, g_str, p_sql, g_sql)
            return match, None, None
        except Exception:
            pass  # fall through to frozenset path

    # ── Frozenset fallback ──────────────────────────────────────────────────
    gold_ok, gold_rows = _exec_raw(db_path, g_str, timeout)
    if not gold_ok:
        return None, None, gold_rows   # gold error

    pred_ok, pred_rows = _exec_raw(db_path, p_str, timeout)
    if not pred_ok:
        return False, pred_rows, None  # pred error

    gold_set = frozenset(tuple(_norm_val(v) for v in row) for row in gold_rows)
    pred_set = frozenset(tuple(_norm_val(v) for v in row) for row in pred_rows)
    return gold_set == pred_set, None, None


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
        match, pred_err, gold_err = compare_results(
            db_path, pred["pred_sql"], pred["gold_sql"], args.timeout
        )

        if gold_err is not None:
            gold_errors.append({"id": pred["id"], "sql": pred["gold_sql"], "error": gold_err})
            results_detail.append({**pred, "ex": None, "gold_error": gold_err})
            continue

        if pred_err is not None:
            pred_errors.append({"id": pred["id"], "sql": pred["pred_sql"], "error": pred_err})
            match = False

        if match:
            correct += 1

        ex_val = 1 if match else 0
        by_hardness[pred.get("hardness", "unknown")].append(ex_val)
        by_source  [pred.get("source",   "unknown")].append(ex_val)

        results_detail.append({
            **pred,
            "ex":         ex_val,
            "pred_error": pred_err,
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
