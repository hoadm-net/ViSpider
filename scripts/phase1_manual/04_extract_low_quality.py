#!/usr/bin/env python3
"""
Extract samples with similarity < 0.75 for manual review.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# Get project root (2 levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

print("\n" + "="*80)
print("EXTRACTING LOW-QUALITY SAMPLES (< 0.75)")
print("="*80 + "\n")

# Load data
print("Loading data...")
with open(PROJECT_ROOT / 'data/manual_translations/vispider_train_2000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load similarity scores
with open(PROJECT_ROOT / 'results/quality_analysis/similarity_analysis.json', 'r', encoding='utf-8') as f:
    analysis = json.load(f)

similarities = analysis['all_similarities']
print(f"✓ Loaded {len(data)} samples with similarity scores\n")

# Filter samples with similarity < 0.75
low_quality = []
for i, (sample, sim) in enumerate(zip(data, similarities)):
    if sim < 0.75:
        sample_with_sim = sample.copy()
        sample_with_sim['labse_similarity'] = round(sim, 4)
        sample_with_sim['index'] = i
        low_quality.append(sample_with_sim)

print(f"Found {len(low_quality)} samples with similarity < 0.75\n")

# Sort by similarity (lowest first)
low_quality.sort(key=lambda x: x['labse_similarity'])

# Statistics
from collections import Counter

similarities_low = [s['labse_similarity'] for s in low_quality]
print("Statistics:")
print(f"  Count:      {len(low_quality)}")
print(f"  Percentage: {len(low_quality)/len(data)*100:.1f}%")
print(f"  Mean:       {sum(similarities_low)/len(similarities_low):.4f}")
print(f"  Median:     {sorted(similarities_low)[len(similarities_low)//2]:.4f}")
print(f"  Min:        {min(similarities_low):.4f}")
print(f"  Max:        {max(similarities_low):.4f}")
print()

# By severity
severe = [s for s in low_quality if s['labse_similarity'] < 0.50]
moderate = [s for s in low_quality if 0.50 <= s['labse_similarity'] < 0.60]
mild = [s for s in low_quality if 0.60 <= s['labse_similarity'] < 0.75]

print("By severity:")
print(f"  Severe    (< 0.50):     {len(severe):3d} ({len(severe)/len(low_quality)*100:5.1f}%)")
print(f"  Moderate  (0.50-0.60):  {len(moderate):3d} ({len(moderate)/len(low_quality)*100:5.1f}%)")
print(f"  Mild      (0.60-0.75):  {len(mild):3d} ({len(mild)/len(low_quality)*100:5.1f}%)")
print()

# By difficulty
diff_count = Counter(s['hardness'] for s in low_quality)
print("By difficulty:")
for diff in ['easy', 'medium', 'hard', 'extra_hard']:
    if diff in diff_count:
        count = diff_count[diff]
        pct = count / len(low_quality) * 100
        print(f"  {diff:<12s}: {count:3d} ({pct:5.1f}%)")
print()

# Save to file
output_file = PROJECT_ROOT / 'results/quality_analysis/vispider_low_quality_samples.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(low_quality, f, ensure_ascii=False, indent=2)

print(f"✓ Saved {len(low_quality)} samples to: {output_file}")

# Also create summary structure
summary = {
    "summary": {
        "total_samples": len(low_quality),
        "percentage_of_dataset": round(len(low_quality)/len(data)*100, 2),
        "mean_similarity": round(sum(similarities_low)/len(similarities_low), 4),
        "median_similarity": round(sorted(similarities_low)[len(similarities_low)//2], 4),
        "min_similarity": round(min(similarities_low), 4),
        "max_similarity": round(max(similarities_low), 4)
    },
    "by_severity": {
        "severe_lt_0.50": len(severe),
        "moderate_0.50_0.60": len(moderate),
        "mild_0.60_0.75": len(mild)
    },
    "by_difficulty": dict(diff_count),
    "samples": low_quality
}

summary_file = PROJECT_ROOT / 'results/quality_analysis/vispider_low_quality_summary.json'
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"✓ Saved summary to: {summary_file}")

# Export IDs only to text file
ids_file = PROJECT_ROOT / 'results/quality_analysis/vispider_low_quality_ids.txt'
with open(ids_file, 'w', encoding='utf-8') as f:
    for sample in low_quality:
        f.write(f"{sample['id']}\n")

print(f"✓ Saved IDs list to: {ids_file}")

print()
print("="*80)
print("✅ EXTRACTION COMPLETE")
print("="*80)
print()
print("Files created:")
print(f"  1. {output_file} - Clean list of samples")
print(f"  2. {summary_file} - With statistics summary")
print(f"  3. {ids_file} - IDs only (one per line)")
print()
print("Sample preview (top 5 lowest):")
for i, s in enumerate(low_quality[:5], 1):
    print(f"\n{i}. [{s['id']}] Similarity: {s['labse_similarity']:.4f} | {s['hardness']}")
    print(f"   EN: {s['question'][:70]}...")
    print(f"   VI: {s['vi_question'][:70]}...")
print()
