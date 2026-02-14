#!/usr/bin/env python3
"""
Parse Label Studio export file and extract important data.
Converts Label Studio format to clean Spider-like format with Vietnamese translations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_label_studio_item(item: Dict) -> Optional[Dict]:
    """
    Parse a single Label Studio item and extract important fields.
    
    Args:
        item: Label Studio task item
        
    Returns:
        Parsed data dict or None if invalid
    """
    # Extract original data
    data = item.get('data', {})
    
    # Extract annotation (translation)
    annotations = item.get('annotations', [])
    if not annotations:
        return None
    
    # Get the first (and should be only) annotation
    annotation = annotations[0]
    result = annotation.get('result', [])
    
    if not result:
        return None
    
    # Extract Vietnamese translation
    vi_question = result[0].get('value', {}).get('text', [''])[0].strip()
    
    if not vi_question:
        return None
    
    # Build cleaned data structure
    parsed = {
        'id': data.get('id', ''),
        'db_id': data.get('db_id', ''),
        'question': data.get('question', ''),
        'vi_question': vi_question,
        'query': data.get('query', ''),
        'hardness': data.get('hardness', ''),
        'patterns': data.get('patterns', []),
    }
    
    return parsed


def parse_label_studio_file(input_path: str, output_path: str = None) -> List[Dict]:
    """
    Parse entire Label Studio export file.
    
    Args:
        input_path: Path to Label Studio JSON file
        output_path: Optional path to save parsed data
        
    Returns:
        List of parsed items
    """
    print(f"Loading data from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total items: {len(data)}")
    
    # Parse all items
    parsed_items = []
    skipped = 0
    
    for item in data:
        parsed = parse_label_studio_item(item)
        if parsed:
            parsed_items.append(parsed)
        else:
            skipped += 1
    
    print(f"Successfully parsed: {len(parsed_items)}")
    print(f"Skipped (no translation): {skipped}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  - Unique databases: {len(set(item['db_id'] for item in parsed_items))}")
    
    hardness_counts = {}
    for item in parsed_items:
        h = item['hardness']
        hardness_counts[h] = hardness_counts.get(h, 0) + 1
    
    print(f"  - Hardness distribution:")
    for hardness, count in sorted(hardness_counts.items()):
        print(f"      {hardness}: {count}")
    
    # Sample data
    print(f"\nSample item:")
    if parsed_items:
        sample = parsed_items[0]
        print(f"  ID: {sample['id']}")
        print(f"  DB: {sample['db_id']}")
        print(f"  EN: {sample['question']}")
        print(f"  VI: {sample['vi_question']}")
        print(f"  SQL: {sample['query']}")
        print(f"  Hardness: {sample['hardness']}")
        print(f"  Patterns: {sample['patterns']}")
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving parsed data to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(parsed_items)} items")
    
    return parsed_items


def main():
    """Main entry point."""
    input_file = 'data/manual_translations/label_studio_2000_samples.json'
    output_file = 'data/manual_translations/vispider_train_2000.json'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Parse and save
    parsed_items = parse_label_studio_file(input_file, output_file)
    
    print(f"\n✓ Done! Parsed {len(parsed_items)} items")
    print(f"  Output file: {output_file}")


if __name__ == '__main__':
    main()
