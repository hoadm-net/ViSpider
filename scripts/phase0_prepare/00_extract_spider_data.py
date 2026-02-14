#!/usr/bin/env python3
"""
Extract simplified data from Spider dataset.
Creates train.json, dev.json, test.json with format:
{
    "id": "train-0001",
    "db_id": "concert_singer",
    "question": "How many singers do we have?",
    "query": "SELECT count(*) FROM singer"
}
"""

import json
import os
from pathlib import Path


def load_json(filepath):
    """Load JSON file"""
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_data(data, prefix, start_id=1):
    """
    Extract simplified data from Spider format.
    
    Args:
        data: List of examples from Spider dataset
        prefix: ID prefix (e.g., 'train', 'dev', 'test')
        start_id: Starting ID number
    
    Returns:
        List of simplified examples
    """
    extracted = []
    
    for idx, example in enumerate(data, start=start_id):
        simplified = {
            "id": f"{prefix}-{idx:04d}",
            "db_id": example["db_id"],
            "question": example["question"],
            "query": example["query"]
        }
        extracted.append(simplified)
    
    return extracted


def save_json(data, filepath):
    """Save data to JSON file"""
    # Create output directory if not exists
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(data)} examples to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {filepath}")


def print_statistics(data, split_name):
    """Print statistics for a split"""
    db_ids = set(example["db_id"] for example in data)
    avg_question_len = sum(len(ex["question"].split()) for ex in data) / len(data)
    avg_query_len = sum(len(ex["query"].split()) for ex in data) / len(data)
    
    print(f"\n📊 {split_name.upper()} Statistics:")
    print(f"  Total examples: {len(data)}")
    print(f"  Unique databases: {len(db_ids)}")
    print(f"  Avg question length: {avg_question_len:.1f} words")
    print(f"  Avg query length: {avg_query_len:.1f} words")


def main():
    # Get project root (2 levels up from this script)
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    
    # Define paths
    spider_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "extracted"
    
    print("="*60)
    print("Spider Dataset Extraction")
    print("="*60)
    
    # ==================== TRAIN ====================
    print("\n[1/3] Processing TRAIN data...")
    
    # Load train_spider.json (main training data)
    train_spider = load_json(spider_dir / "train_spider.json")
    
    # Load train_others.json (additional training data)
    train_others = load_json(spider_dir / "train_others.json")
    
    # Combine train data
    print(f"  train_spider.json: {len(train_spider)} examples")
    print(f"  train_others.json: {len(train_others)} examples")
    
    train_combined = train_spider + train_others
    print(f"  Combined: {len(train_combined)} examples")
    
    # Extract simplified format
    train_data = extract_data(train_combined, prefix="train")
    
    # Save train data
    save_json(train_data, output_dir / "train.json")
    print_statistics(train_data, "train")
    
    # ==================== DEV ====================
    print("\n[2/3] Processing DEV data...")
    
    # Load dev.json
    dev_raw = load_json(spider_dir / "dev.json")
    print(f"  dev.json: {len(dev_raw)} examples")
    
    # Extract simplified format
    dev_data = extract_data(dev_raw, prefix="dev")
    
    # Save dev data
    save_json(dev_data, output_dir / "dev.json")
    print_statistics(dev_data, "dev")
    
    # ==================== TEST ====================
    print("\n[3/3] Processing TEST data...")
    
    # Load test.json
    test_raw = load_json(spider_dir / "test.json")
    print(f"  test.json: {len(test_raw)} examples")
    
    # Extract simplified format
    test_data = extract_data(test_raw, prefix="test")
    
    # Save test data
    save_json(test_data, output_dir / "test.json")
    print_statistics(test_data, "test")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*60)
    print("✓ Extraction Complete!")
    print("="*60)
    
    total_examples = len(train_data) + len(dev_data) + len(test_data)
    total_dbs = len(set(
        ex["db_id"] for split in [train_data, dev_data, test_data] 
        for ex in split
    ))
    
    print(f"\n📈 Overall Summary:")
    print(f"  Total examples: {total_examples}")
    print(f"  Total databases: {total_dbs}")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Dev: {len(dev_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"  - train.json")
    print(f"  - dev.json")
    print(f"  - test.json")
    
    # Show sample examples
    print("\n" + "="*60)
    print("Sample Examples:")
    print("="*60)
    
    for split_name, split_data in [("TRAIN", train_data), ("DEV", dev_data), ("TEST", test_data)]:
        print(f"\n[{split_name}] First example:")
        example = split_data[0]
        print(f"  ID: {example['id']}")
        print(f"  DB: {example['db_id']}")
        print(f"  Q:  {example['question']}")
        print(f"  SQL: {example['query']}")


if __name__ == "__main__":
    main()
