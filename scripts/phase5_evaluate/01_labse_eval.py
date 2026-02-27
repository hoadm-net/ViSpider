#!/usr/bin/env python3
"""
Phase 5 – Step 1: LaBSE Evaluation

Evaluate translation quality using cross-lingual LaBSE cosine similarity.

Two modes
---------
1. Dataset quality  (default)
   Input: any ViSpider JSON file with `question` + `vi_question` fields.
   Metric: EN ↔ VI cosine similarity (measures semantic alignment between the
           English source and the Vietnamese translation).

2. Model prediction comparison  (--pred-file)
   Input: model-eval JSON produced by phase4_finetune/04_evaluate.py.
   Metrics computed for each sample:
     • EN ↔ PRED   – how semantically aligned is the model output with the source
     • REF ↔ PRED  – how close is the model output to the human reference

Usage
-----
# Evaluate dev split (dataset quality)
python3 scripts/phase5_evaluate/01_labse_eval.py

# Evaluate a specific split or file
python3 scripts/phase5_evaluate/01_labse_eval.py --input data/merged/vispider_test.json

# Compare model predictions against reference
python3 scripts/phase5_evaluate/01_labse_eval.py \\
    --pred-file results/quality_analysis/model_eval_dev.json

# Evaluate all three splits in one run
python3 scripts/phase5_evaluate/01_labse_eval.py --all-splits
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/utils"))

OUTPUT_DIR   = PROJECT_ROOT / "results/phase5_evaluate"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LaBSE evaluation of ViSpider translations")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--input", "-i",
        default=str(PROJECT_ROOT / "data/merged/vispider_dev.json"),
        help="ViSpider JSON file with `question` and `vi_question` fields "
             "(default: data/merged/vispider_dev.json)",
    )
    src.add_argument(
        "--all-splits",
        action="store_true",
        help="Evaluate all three splits (dev, test, train) sequentially",
    )
    p.add_argument(
        "--pred-file",
        default=None,
        help="Model-eval JSON (from phase4/04_evaluate.py) with `vi_predicted` "
             "and `vi_reference` fields. When provided, also compute REF↔PRED.",
    )
    p.add_argument(
        "--batch", type=int, default=64,
        help="LaBSE encoding batch size (default: 64)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.75,
        help="Similarity threshold for pass/fail metric (default: 0.75)",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: results/phase5_evaluate/<input_stem>_labse.json)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_stats(scores: list, threshold: float) -> dict:
    """Return summary statistics for a list of similarity scores."""
    n = len(scores)
    if n == 0:
        return {}
    above = sum(1 for s in scores if s >= threshold)
    return {
        "n":                n,
        "mean":             round(statistics.mean(scores), 4),
        "median":           round(statistics.median(scores), 4),
        "stdev":            round(statistics.stdev(scores) if n > 1 else 0.0, 4),
        "min":              round(min(scores), 4),
        "max":              round(max(scores), 4),
        "p25":              round(sorted(scores)[int(n * 0.25)], 4),
        "p75":              round(sorted(scores)[int(n * 0.75)], 4),
        f"above_{threshold}":       above,
        f"above_{threshold}_pct":   round(100 * above / n, 1),
    }


def breakdown(records: list, score_key: str, group_key: str, threshold: float) -> dict:
    """Return per-group stats for a given score and grouping key."""
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        g = r.get(group_key) or "unknown"
        groups[g].append(r[score_key])
    return {g: compute_stats(v, threshold) for g, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Dataset quality mode
# ---------------------------------------------------------------------------

def evaluate_dataset(input_path: Path, batch_size: int, threshold: float) -> tuple[list, dict]:
    """Compute EN ↔ VI similarity for each sample in a ViSpider dataset file."""
    from embeddings_utils import compute_embeddings

    print(f"\n{'='*72}")
    print(f"DATASET QUALITY  →  {input_path.name}")
    print(f"{'='*72}")

    with open(input_path, encoding="utf-8") as f:
        samples = json.load(f)

    print(f"  Loaded {len(samples)} samples")

    en_texts = [s["question"]    for s in samples]
    vi_texts = [s["vi_question"] for s in samples]

    # Encode both sides in batches
    print("\n  Encoding English questions …")
    en_embs = _batch_encode(en_texts, batch_size)
    print("  Encoding Vietnamese translations …")
    vi_embs = _batch_encode(vi_texts, batch_size)

    import numpy as np
    sims = (en_embs * vi_embs).sum(axis=1).tolist()

    records = []
    for s, sim in zip(samples, sims):
        records.append({
            "id":          s["id"],
            "question":    s["question"],
            "vi_question": s["vi_question"],
            "source":      s.get("source", "unknown"),
            "hardness":    s.get("hardness", "unknown"),
            "en_vi_sim":   round(float(sim), 4),
        })

    stats = {
        "file":               input_path.name,
        "mode":               "dataset_quality",
        "metric":             "EN↔VI cosine similarity",
        "threshold":          threshold,
        "overall":            compute_stats(sims, threshold),
        "by_source":          breakdown(records, "en_vi_sim", "source",   threshold),
        "by_hardness":        breakdown(records, "en_vi_sim", "hardness", threshold),
    }

    _print_summary(stats)
    return records, stats


# ---------------------------------------------------------------------------
# Model prediction mode
# ---------------------------------------------------------------------------

def evaluate_predictions(pred_path: Path, batch_size: int, threshold: float) -> tuple[list, dict]:
    """Compute EN↔PRED and REF↔PRED similarities for model evaluation output."""
    from embeddings_utils import compute_embeddings

    print(f"\n{'='*72}")
    print(f"MODEL PREDICTIONS  →  {pred_path.name}")
    print(f"{'='*72}")

    with open(pred_path, encoding="utf-8") as f:
        data = json.load(f)

    # Support two formats: plain list, or {"stats": ..., "predictions": [...]}
    if isinstance(data, list):
        preds = data
    elif isinstance(data, dict) and "predictions" in data:
        preds = data["predictions"]
    else:
        print("❌ Unrecognised prediction-file format. Expected a list or {predictions: [...]}")
        sys.exit(1)

    print(f"  Loaded {len(preds)} predictions")

    en_texts   = [p["question"]     for p in preds]
    ref_texts  = [p["vi_reference"] for p in preds]
    pred_texts = [p["vi_predicted"] for p in preds]

    print("\n  Encoding English questions …")
    en_embs   = _batch_encode(en_texts,   batch_size)
    print("  Encoding reference translations …")
    ref_embs  = _batch_encode(ref_texts,  batch_size)
    print("  Encoding model predictions …")
    pred_embs = _batch_encode(pred_texts, batch_size)

    import numpy as np
    en_pred_sims  = (en_embs   * pred_embs).sum(axis=1).tolist()
    ref_pred_sims = (ref_embs  * pred_embs).sum(axis=1).tolist()

    records = []
    for p, en_pred, ref_pred in zip(preds, en_pred_sims, ref_pred_sims):
        records.append({
            "id":           p.get("id",           "?"),
            "question":     p["question"],
            "vi_reference": p["vi_reference"],
            "vi_predicted": p["vi_predicted"],
            "source":       p.get("source",   "unknown"),
            "hardness":     p.get("hardness", "unknown"),
            "en_pred_sim":  round(float(en_pred),  4),
            "ref_pred_sim": round(float(ref_pred), 4),
        })

    stats = {
        "file":               pred_path.name,
        "mode":               "model_prediction",
        "threshold":          threshold,
        "en_pred": {
            "metric":    "EN↔PRED cosine similarity",
            "overall":   compute_stats(en_pred_sims,  threshold),
            "by_source":    breakdown(records, "en_pred_sim",  "source",   threshold),
            "by_hardness":  breakdown(records, "en_pred_sim",  "hardness", threshold),
        },
        "ref_pred": {
            "metric":    "REF↔PRED cosine similarity",
            "overall":   compute_stats(ref_pred_sims, threshold),
            "by_source":    breakdown(records, "ref_pred_sim", "source",   threshold),
            "by_hardness":  breakdown(records, "ref_pred_sim", "hardness", threshold),
        },
    }

    _print_summary(stats)
    return records, stats


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _batch_encode(texts: list, batch_size: int):
    """Encode a list of texts using LaBSE in batches, return L2-normalised numpy array."""
    import numpy as np
    from embeddings_utils import get_labse_model

    model  = get_labse_model()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs  = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        all_embs.append(embs / norms)
        done = min(i + batch_size, len(texts))
        print(f"    [{done}/{len(texts)}]", end="\r")
    print()
    return np.vstack(all_embs)


def _print_summary(stats: dict):
    threshold = stats["threshold"]

    if stats["mode"] == "dataset_quality":
        o = stats["overall"]
        print(f"\n  EN↔VI cosine similarity:")
        print(f"    mean   = {o['mean']:.4f}   median = {o['median']:.4f}   stdev = {o['stdev']:.4f}")
        print(f"    min    = {o['min']:.4f}   max    = {o['max']:.4f}")
        print(f"    ≥ {threshold}:  {o[f'above_{threshold}']}/{o['n']} ({o[f'above_{threshold}_pct']}%)")
        print(f"\n  By source:")
        for g, v in stats["by_source"].items():
            print(f"    {g:<12}  mean={v['mean']:.4f}  ≥{threshold}: {v[f'above_{threshold}_pct']}%  (n={v['n']})")
        print(f"\n  By hardness:")
        for g, v in stats["by_hardness"].items():
            print(f"    {g:<12}  mean={v['mean']:.4f}  ≥{threshold}: {v[f'above_{threshold}_pct']}%  (n={v['n']})")

    else:  # model_prediction
        for key, label in [("en_pred", "EN↔PRED"), ("ref_pred", "REF↔PRED")]:
            o = stats[key]["overall"]
            print(f"\n  {label} cosine similarity:")
            print(f"    mean   = {o['mean']:.4f}   median = {o['median']:.4f}   stdev = {o['stdev']:.4f}")
            print(f"    min    = {o['min']:.4f}   max    = {o['max']:.4f}")
            print(f"    ≥ {threshold}:  {o[f'above_{threshold}']}/{o['n']} ({o[f'above_{threshold}_pct']}%)")
            print(f"    By hardness:")
            for g, v in stats[key]["by_hardness"].items():
                print(f"      {g:<12}  mean={v['mean']:.4f}  ≥{threshold}: {v[f'above_{threshold}_pct']}%")


def save_results(records: list, stats: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stats": stats, "records": records}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.pred_file:
        # ── Model prediction comparison mode ──────────────────────────────
        pred_path   = Path(args.pred_file)
        output_path = Path(args.output) if args.output else (
            OUTPUT_DIR / f"{pred_path.stem}_labse.json"
        )
        records, stats = evaluate_predictions(pred_path, args.batch, args.threshold)
        save_results(records, stats, output_path)

    elif args.all_splits:
        # ── Evaluate all splits ───────────────────────────────────────────
        for split in ("dev", "test", "train"):
            input_path  = PROJECT_ROOT / f"data/merged/vispider_{split}.json"
            output_path = OUTPUT_DIR / f"vispider_{split}_labse.json"
            if not input_path.exists():
                print(f"  ⚠  {input_path.name} not found – skipping")
                continue
            records, stats = evaluate_dataset(input_path, args.batch, args.threshold)
            save_results(records, stats, output_path)

    else:
        # ── Single dataset file ───────────────────────────────────────────
        input_path  = Path(args.input)
        output_path = Path(args.output) if args.output else (
            OUTPUT_DIR / f"{input_path.stem}_labse.json"
        )
        records, stats = evaluate_dataset(input_path, args.batch, args.threshold)
        save_results(records, stats, output_path)


if __name__ == "__main__":
    main()
