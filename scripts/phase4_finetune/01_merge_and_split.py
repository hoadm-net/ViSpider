#!/usr/bin/env python3
"""
Merge Phase 1 (manual) and Phase 2 (GPT) translations into a unified dataset,
then split into train/dev/test sets with stratified sampling.

Input:
  - data/manual_translations/vispider_train_2000.json   (Phase 1 gold seed)
  - data/chatgpt_translations/gpt_translations_final.json  (Phase 2 GPT)

Output (data/merged/):
  - vispider_all.json       Full merged dataset (canonical format)
  - vispider_train.json     Train split
  - vispider_dev.json       Dev split
  - vispider_test.json      Test split
  - split_report.json       Statistics per split

Canonical sample format:
{
  "id":          str   -- original ID (train-XXXX)
  "db_id":       str   -- database name
  "question":    str   -- English question
  "vi_question": str   -- Vietnamese translation
  "query":       str   -- SQL query
  "hardness":    str   -- easy / medium / hard / extra_hard / unknown
  "sql_patterns": list -- SQL pattern tags
  "source":      str   -- "manual" | "gpt"
}

Split strategy:
  - Stratify by (source × hardness) to preserve distribution
  - Default ratio: 80% train / 10% dev / 10% test
  - Shuffle with fixed random seed for reproducibility
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

PHASE1_FILE = PROJECT_ROOT / 'data/manual_translations/vispider_train_2000.json'
PHASE2_FILE = PROJECT_ROOT / 'data/chatgpt_translations/gpt_translations_final.json'
OUTPUT_DIR  = PROJECT_ROOT / 'data/merged'

TRAIN_RATIO = 0.80
DEV_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

assert abs(TRAIN_RATIO + DEV_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1"


# ── Normalise ────────────────────────────────────────────────────────────────

def normalise_phase1(sample: dict) -> dict:
    """Convert Phase 1 record to canonical format."""
    return {
        'id':           sample['id'],
        'db_id':        sample['db_id'],
        'question':     sample['question'],
        'vi_question':  sample['vi_question'],
        'query':        sample['query'],
        'hardness':     sample.get('hardness', 'unknown'),
        'sql_patterns': sample.get('sql_patterns', sample.get('patterns', [])),
        'source':       'manual',
    }


def normalise_phase2(sample: dict) -> dict:
    """Convert Phase 2 record to canonical format."""
    return {
        'id':           sample['id'],
        'db_id':        sample['db_id'],
        'question':     sample['question'],
        'vi_question':  sample['vi_question'],
        'query':        sample['query'],
        'hardness':     sample.get('hardness', 'unknown'),
        'sql_patterns': sample.get('sql_patterns', []),
        'source':       'gpt',
    }


# ── Stratified split ─────────────────────────────────────────────────────────

def stratified_split(
    samples: list,
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple:
    """
    Split samples into train/dev/test while preserving
    (source × hardness) distribution as much as possible.
    """
    rng = random.Random(seed)
    test_ratio = 1.0 - train_ratio - dev_ratio  # derive instead of using global constant

    # Group by stratum
    strata: dict[str, list] = defaultdict(list)
    for s in samples:
        key = f"{s['source']}|{s['hardness']}"
        strata[key].append(s)

    train, dev, test = [], [], []

    for key, group in strata.items():
        rng.shuffle(group)
        n = len(group)
        n_dev  = max(1, round(n * dev_ratio))
        n_test = max(1, round(n * test_ratio))
        n_train = n - n_dev - n_test

        if n_train <= 0:
            # Too small to split — put everything in train
            train.extend(group)
            continue

        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        test.extend(group[n_train + n_dev:])

    # Final shuffle within each split
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    return train, dev, test


# ── Report ───────────────────────────────────────────────────────────────────

def split_stats(name: str, samples: list) -> dict:
    source_counts   = defaultdict(int)
    hardness_counts = defaultdict(int)
    for s in samples:
        source_counts[s['source']] += 1
        hardness_counts[s['hardness']] += 1
    return {
        'split': name,
        'total': len(samples),
        'by_source': dict(sorted(source_counts.items())),
        'by_hardness': dict(sorted(hardness_counts.items())),
    }


def print_stats(stats: dict):
    print(f"\n  {'Split':<8} {stats['split']}  ({stats['total']} samples)")
    print(f"  {'Source:':<12}", end='')
    for k, v in stats['by_source'].items():
        print(f"  {k}={v}", end='')
    print()
    print(f"  {'Hardness:':<12}", end='')
    for k, v in stats['by_hardness'].items():
        print(f"  {k}={v}", end='')
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("DATASET MERGE & SPLIT")
    print("=" * 72)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\nLoading Phase 1 (manual)...")
    with open(PHASE1_FILE, encoding='utf-8') as f:
        p1_raw = json.load(f)
    p1 = [normalise_phase1(s) for s in p1_raw]
    print(f"  ✓ {len(p1)} samples")

    print("Loading Phase 2 (GPT)...")
    with open(PHASE2_FILE, encoding='utf-8') as f:
        p2_raw = json.load(f)
    p2 = [normalise_phase2(s) for s in p2_raw]
    print(f"  ✓ {len(p2)} samples")

    # ── Deduplicate ────────────────────────────────────────────────────────
    seen_ids: set[str] = set()
    merged = []
    for s in p1 + p2:
        if s['id'] not in seen_ids:
            seen_ids.add(s['id'])
            merged.append(s)
        else:
            print(f"  ⚠️  Duplicate ID skipped: {s['id']} (source={s['source']})")

    print(f"\n  Merged total: {len(merged)} samples")
    print(f"  (Phase 1: {len(p1)}  |  Phase 2: {len(p2)})")

    # ── Split ──────────────────────────────────────────────────────────────
    print(f"\nStratified split  (train={TRAIN_RATIO:.0%} / dev={DEV_RATIO:.0%} / test={TEST_RATIO:.0%})")
    print(f"  Random seed: {RANDOM_SEED}")

    train, dev, test = stratified_split(merged, TRAIN_RATIO, DEV_RATIO, RANDOM_SEED)

    all_stats = [
        split_stats('train', train),
        split_stats('dev',   dev),
        split_stats('test',  test),
    ]
    for st in all_stats:
        print_stats(st)

    # ── Save ───────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        'vispider_all.json':   merged,
        'vispider_train.json': train,
        'vispider_dev.json':   dev,
        'vispider_test.json':  test,
    }
    for fname, data in files.items():
        out = OUTPUT_DIR / fname
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ {fname}  ({len(data)} samples)")

    report_file = OUTPUT_DIR / 'split_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'train_ratio': TRAIN_RATIO,
                'dev_ratio':   DEV_RATIO,
                'test_ratio':  TEST_RATIO,
                'random_seed': RANDOM_SEED,
            },
            'splits': all_stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"  ✓ split_report.json")

    print("\n" + "=" * 72)
    print("✅ DONE")
    print("=" * 72)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
