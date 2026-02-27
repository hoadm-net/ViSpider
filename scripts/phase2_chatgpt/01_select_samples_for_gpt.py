#!/usr/bin/env python3
"""
Select 3,000 samples for GPT translation with pattern coverage optimization.

Strategy:
1. Exclude manual translation IDs
2. Stratified sampling by SQL patterns
3. Ensure coverage of underrepresented patterns
4. Balance difficulty levels
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set


# Get project root (2 levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Set random seed for reproducibility
random.seed(42)


def load_manual_ids() -> Set[str]:
    """Load IDs from manual translations to exclude."""
    manual_file = PROJECT_ROOT / 'data/manual_translations/vispider_train_2000.json'
    
    print(f"Loading manual translation IDs from: {manual_file}")
    with open(manual_file, 'r', encoding='utf-8') as f:
        manual_data = json.load(f)
    
    manual_ids = {sample['id'] for sample in manual_data}
    print(f"✓ Loaded {len(manual_ids)} manual translation IDs to exclude\n")
    
    return manual_ids


def load_extracted_data(manual_ids: Set[str]) -> List[Dict]:
    """Load extracted Spider data and filter out manual IDs."""
    extracted_file = PROJECT_ROOT / 'data/extracted/train.json'
    
    # Fallback: if extracted doesn't exist, use raw data
    if not extracted_file.exists():
        print(f"⚠️  Extracted data not found: {extracted_file}")
        print(f"Attempting to load from raw data...")
        
        raw_spider = PROJECT_ROOT / 'data/raw/train_spider.json'
        raw_others = PROJECT_ROOT / 'data/raw/train_others.json'
        
        if not raw_spider.exists():
            print(f"❌ ERROR: Neither extracted nor raw data found!")
            print(f"Please run: python3 scripts/phase0_prepare/00_extract_spider_data.py")
            print(f"Or ensure Git LFS data is pulled: git lfs pull")
            raise FileNotFoundError("No Spider data available")
        
        # Load raw data and reconstruct with proper IDs (1-indexed, matching Phase 0/1)
        with open(raw_spider, 'r', encoding='utf-8') as f:
            spider_data = json.load(f)
        with open(raw_others, 'r', encoding='utf-8') as f:
            others_data = json.load(f)
        
        train_combined = spider_data + others_data
        
        all_data = []
        for idx, sample in enumerate(train_combined, start=1):
            all_data.append({
                'id': f'train-{idx:04d}',
                'db_id': sample['db_id'],
                'question': sample['question'],
                'query': sample['query'],
                'hardness': sample.get('hardness', 'unknown'),
            })
        
        print(f"✓ Loaded {len(all_data)} samples from raw data")
        
    else:
        print(f"Loading extracted Spider data from: {extracted_file}")
        with open(extracted_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        print(f"✓ Loaded {len(all_data)} total samples")
    
    # Filter out manual translation IDs
    filtered_data = [s for s in all_data if s['id'] not in manual_ids]
    
    print(f"✓ After excluding manual IDs: {len(filtered_data)} samples available")
    print(f"  (Excluded {len(all_data) - len(filtered_data)} samples)\n")
    
    return filtered_data


def analyze_pattern_distribution(manual_patterns: Dict) -> Dict:
    """Analyze pattern distribution from manual translations."""
    print("="*80)
    print("PATTERN COVERAGE ANALYSIS")
    print("="*80)
    print()
    
    pattern_counts = manual_patterns['pattern_distribution']
    total_manual = manual_patterns['total_samples']
    
    # Top patterns from manual translations
    print("Top 15 patterns in manual translations:")
    print(f"{'Pattern':<20} {'Count':<8} {'Percentage':<10}")
    print("-"*80)
    
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    for pattern, count in sorted_patterns[:15]:
        pct = count / total_manual * 100
        print(f"{pattern:<20} {count:<8} {pct:>6.1f}%")
    
    print()
    
    return pattern_counts


def stratified_pattern_sampling(available_samples: List[Dict], 
                                 manual_pattern_dist: Dict,
                                 target_count: int = 3000) -> List[Dict]:
    """
    Stratified sampling based on SQL patterns.
    
    Strategy:
    1. Group samples by primary pattern
    2. Sample proportionally to pattern frequency in manual set
    3. Prioritize underrepresented patterns
    """
    print("="*80)
    print("STRATIFIED SAMPLING")
    print("="*80)
    print()
    
    # Add simple patterns for samples without pattern annotation
    # (from extracted data, not yet analyzed)
    for sample in available_samples:
        if 'sql_patterns' not in sample:
            # Simple heuristic: extract basic patterns from query
            query_upper = sample['query'].upper()
            patterns = []
            
            if 'COUNT(' in query_upper:
                patterns.append('COUNT')
            if 'JOIN' in query_upper:
                patterns.append('JOIN')
            if 'GROUP BY' in query_upper:
                patterns.append('GROUP_BY')
            if 'WHERE' in query_upper:
                patterns.append('WHERE')
            
            # Default patterns
            if not patterns:
                patterns = ['SELECT', 'FROM']
            
            sample['sql_patterns'] = patterns
    
    # Group by primary pattern
    pattern_groups = {}
    for sample in available_samples:
        patterns = sample.get('sql_patterns', ['SELECT'])
        primary = patterns[0] if patterns else 'SELECT'
        
        if primary not in pattern_groups:
            pattern_groups[primary] = []
        pattern_groups[primary].append(sample)
    
    print(f"Grouped samples into {len(pattern_groups)} pattern categories")
    print()
    
    # Calculate target samples per pattern based on manual distribution
    total_manual = sum(manual_pattern_dist.values())
    target_per_pattern = {}
    
    for pattern, count in manual_pattern_dist.items():
        proportion = count / total_manual
        target = int(target_count * proportion)
        
        if pattern in pattern_groups:
            available = len(pattern_groups[pattern])
            # Don't request more than available
            target_per_pattern[pattern] = min(target, available)
    
    # Sample from each pattern group
    selected_samples = []
    
    print("Sampling per pattern:")
    print(f"{'Pattern':<20} {'Target':<8} {'Available':<10} {'Selected':<10}")
    print("-"*80)
    
    for pattern, target in sorted(target_per_pattern.items(), key=lambda x: x[1], reverse=True)[:15]:
        if pattern not in pattern_groups:
            continue
        
        available = len(pattern_groups[pattern])
        to_select = min(target, available)
        
        # Random sample from this pattern group
        selected = random.sample(pattern_groups[pattern], to_select)
        selected_samples.extend(selected)
        
        print(f"{pattern:<20} {target:<8} {available:<10} {to_select:<10}")
    
    print()
    
    # If we haven't reached target, fill with remaining samples
    if len(selected_samples) < target_count:
        remaining_needed = target_count - len(selected_samples)
        
        # Get all samples not yet selected
        selected_ids = {s['id'] for s in selected_samples}
        remaining_samples = [s for s in available_samples if s['id'] not in selected_ids]
        
        # Sample randomly
        additional = random.sample(remaining_samples, min(remaining_needed, len(remaining_samples)))
        selected_samples.extend(additional)
        
        print(f"Added {len(additional)} additional samples to reach target")
        print()
    
    # Trim if we have too many
    if len(selected_samples) > target_count:
        selected_samples = random.sample(selected_samples, target_count)
    
    return selected_samples


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Select samples for GPT translation')
    parser.add_argument('-n', '--count', type=int, default=3000,
                        help='Number of samples to select (default: 3000)')
    args = parser.parse_args()
    target_count = args.count

    print("="*80)
    print("GPT SAMPLE SELECTION")
    print("="*80)
    print()
    
    # Load manual translation IDs to exclude
    manual_ids = load_manual_ids()
    
    # Load extracted Spider data (excluding manual IDs)
    available_samples = load_extracted_data(manual_ids)
    
    # Load pattern analysis from manual translations
    pattern_analysis_file = PROJECT_ROOT / 'results/quality_analysis/sql_pattern_analysis.json'
    
    print(f"Loading pattern analysis from: {pattern_analysis_file}")
    with open(pattern_analysis_file, 'r', encoding='utf-8') as f:
        pattern_analysis = json.load(f)
    
    print(f"✓ Loaded pattern analysis\n")
    
    # Analyze manual patterns
    manual_pattern_dist = analyze_pattern_distribution(pattern_analysis)
    
    # Stratified sampling
    selected_samples = stratified_pattern_sampling(
        available_samples, 
        manual_pattern_dist, 
        target_count
    )
    
    print("="*80)
    print("SELECTION RESULTS")
    print("="*80)
    print()
    
    print(f"Total selected: {len(selected_samples)} samples")
    print()
    
    # Analyze selected distribution
    selected_patterns = []
    for sample in selected_samples:
        selected_patterns.extend(sample.get('sql_patterns', ['SELECT']))
    
    pattern_counter = Counter(selected_patterns)
    
    print("Top 10 patterns in selected samples:")
    print(f"{'Pattern':<20} {'Count':<8}")
    print("-"*80)
    
    for pattern, count in pattern_counter.most_common(10):
        print(f"{pattern:<20} {count:<8}")
    
    print()
    
    # Save selected samples
    output_file = PROJECT_ROOT / 'data/chatgpt_translations/gpt_target_samples.json'
    
    print(f"Saving {len(selected_samples)} samples to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(selected_samples)} samples")
    print()
    
    print("="*80)
    print("✅ SAMPLE SELECTION COMPLETE")
    print("="*80)
    print()
    
    print("Next step:")
    print("  python3 scripts/phase2_chatgpt/02_translate_with_validation.py")


if __name__ == "__main__":
    main()
