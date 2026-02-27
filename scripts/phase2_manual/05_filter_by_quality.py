#!/usr/bin/env python3
"""
Generate filtered high-quality dataset based on LaBSE similarity threshold.
"""

import json
import sys
from typing import List, Dict
from pathlib import Path

# Get project root (2 levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

def filter_dataset(input_file, output_file, min_similarity=0.75):
    """
    Filter dataset to keep only high-quality translations.
    
    Args:
        input_file: vispider_train_2000.json
        output_file: filtered output file
        min_similarity: minimum similarity threshold
    """
    print(f"\n{'='*80}")
    print(f"FILTERING VISPIDER DATASET")
    print(f"{'='*80}\n")
    
    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples\n")
    
    # Load similarity analysis
    print("Loading similarity scores...")
    with open(PROJECT_ROOT / 'results/quality_analysis/similarity_analysis.json', 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    similarities = analysis['all_similarities']
    print(f"✓ Loaded {len(similarities)} similarity scores\n")
    
    # Filter
    print(f"Filtering samples with similarity >= {min_similarity}...")
    filtered_data = []
    for i, (sample, sim) in enumerate(zip(data, similarities)):
        if sim >= min_similarity:
            # Add similarity score to the sample for reference
            sample_with_sim = sample.copy()
            sample_with_sim['labse_similarity'] = round(sim, 4)
            filtered_data.append(sample_with_sim)
    
    print(f"✓ Kept {len(filtered_data)} samples ({len(filtered_data)/len(data)*100:.1f}%)\n")
    
    # Statistics
    kept_similarities = [s for s in similarities if s >= min_similarity]
    print(f"Filtered dataset statistics:")
    print(f"  Count:  {len(filtered_data)}")
    print(f"  Mean:   {sum(kept_similarities)/len(kept_similarities):.4f}")
    print(f"  Median: {sorted(kept_similarities)[len(kept_similarities)//2]:.4f}")
    print(f"  Min:    {min(kept_similarities):.4f}")
    print(f"  Max:    {max(kept_similarities):.4f}")
    print()
    
    # Difficulty distribution
    from collections import Counter
    diff_count = Counter(s['hardness'] for s in filtered_data)
    print(f"Difficulty distribution:")
    for diff in ['easy', 'medium', 'hard', 'extra_hard']:
        if diff in diff_count:
            count = diff_count[diff]
            pct = count / len(filtered_data) * 100
            print(f"  {diff:<12s}: {count:4d} ({pct:5.1f}%)")
    print()
    
    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(filtered_data)} high-quality samples\n")
    
    print(f"{'='*80}")
    print(f"✅ FILTERING COMPLETE")
    print(f"{'='*80}\n")
    
    return filtered_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 filter_high_quality.py <threshold>")
        print()
        print("Examples:")
        print("  python3 filter_high_quality.py 0.75  # Good quality (recommended)")
        print("  python3 filter_high_quality.py 0.80  # Very good quality")
        print("  python3 filter_high_quality.py 0.85  # Excellent quality")
        print()
        threshold = 0.75
        print(f"Using default threshold: {threshold}")
    else:
        threshold = float(sys.argv[1])
    
    input_file = PROJECT_ROOT / 'data/manual_translations/vispider_train_2000.json'
    output_file = PROJECT_ROOT / f'results/quality_analysis/vispider_train_filtered_{int(threshold*100)}.json'
    
    filter_dataset(input_file, output_file, threshold)
    
    print("Filtered dataset ready for use!")
    print(f"File: {output_file}")


if __name__ == '__main__':
    main()
