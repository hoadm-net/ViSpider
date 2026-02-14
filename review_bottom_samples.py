#!/usr/bin/env python3
"""
Manual Review Helper for Bottom Similarity Samples
Helps navigate and review low-similarity translations.
"""

import json
from typing import Dict, List


def print_sample(sample: Dict, rank: int):
    """Pretty print a sample for review."""
    print(f"\n{'='*80}")
    print(f"RANK #{rank} | Similarity: {sample['similarity']:.4f} | Difficulty: {sample['hardness']}")
    print(f"{'='*80}")
    print(f"ID: {sample['id']}")
    print(f"Database: {sample['db_id']}")
    print(f"\n📝 ENGLISH:")
    print(f"  {sample['question']}")
    print(f"\n🇻🇳 VIETNAMESE:")
    print(f"  {sample['vi_question']}")
    print(f"\n💾 SQL:")
    print(f"  {sample['query']}")
    print(f"\n{'-'*80}")


def interactive_review(samples: List[Dict]):
    """Interactive review interface."""
    print(f"\n{'='*80}")
    print("INTERACTIVE REVIEW MODE")
    print(f"{'='*80}")
    print(f"\nTotal samples to review: {len(samples)}")
    print(f"\nCommands:")
    print(f"  n/next     - Next sample")
    print(f"  p/prev     - Previous sample")
    print(f"  j <num>    - Jump to sample number")
    print(f"  s          - Show summary")
    print(f"  q/quit     - Quit")
    print(f"\n{'='*80}")
    
    current_idx = 0
    
    while True:
        if 0 <= current_idx < len(samples):
            print_sample(samples[current_idx], current_idx + 1)
        
        try:
            cmd = input(f"\n[{current_idx+1}/{len(samples)}] Enter command: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break
        
        if cmd in ['q', 'quit', 'exit']:
            break
        elif cmd in ['n', 'next', '']:
            current_idx = min(current_idx + 1, len(samples) - 1)
        elif cmd in ['p', 'prev']:
            current_idx = max(current_idx - 1, 0)
        elif cmd.startswith('j '):
            try:
                target = int(cmd.split()[1]) - 1
                if 0 <= target < len(samples):
                    current_idx = target
                else:
                    print(f"Invalid index. Must be between 1 and {len(samples)}")
            except (ValueError, IndexError):
                print("Invalid jump command. Use: j <number>")
        elif cmd == 's':
            print_summary(samples)
        else:
            print("Unknown command. Use: n/next, p/prev, j <num>, s, q/quit")


def print_summary(samples: List[Dict]):
    """Print summary statistics of the samples."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    similarities = [s['similarity'] for s in samples]
    
    print(f"Total samples: {len(samples)}")
    print(f"Min similarity: {min(similarities):.4f}")
    print(f"Max similarity: {max(similarities):.4f}")
    print(f"Avg similarity: {sum(similarities)/len(similarities):.4f}")
    
    # By difficulty
    from collections import Counter
    hardness_count = Counter(s['hardness'] for s in samples)
    
    print(f"\nBy difficulty:")
    for hardness in ['easy', 'medium', 'hard', 'extra_hard']:
        count = hardness_count.get(hardness, 0)
        pct = count / len(samples) * 100
        print(f"  {hardness:12s}: {count:3d} ({pct:5.1f}%)")
    
    # By database
    db_count = Counter(s['db_id'] for s in samples)
    print(f"\nTop 10 databases:")
    for db_id, count in db_count.most_common(10):
        print(f"  {db_id:20s}: {count:3d}")


def export_markdown(samples: List[Dict], output_file: str = "review_report.md"):
    """Export samples to markdown for easy reading."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Low Similarity Translation Review\n\n")
        f.write(f"Total samples: {len(samples)}\n\n")
        
        for i, sample in enumerate(samples, 1):
            f.write(f"## Sample #{i}\n\n")
            f.write(f"- **Similarity**: {sample['similarity']:.4f}\n")
            f.write(f"- **Difficulty**: {sample['hardness']}\n")
            f.write(f"- **Database**: {sample['db_id']}\n")
            f.write(f"- **ID**: {sample['id']}\n\n")
            
            f.write(f"**English**:\n")
            f.write(f"> {sample['question']}\n\n")
            
            f.write(f"**Vietnamese**:\n")
            f.write(f"> {sample['vi_question']}\n\n")
            
            f.write(f"**SQL**:\n")
            f.write(f"```sql\n{sample['query']}\n```\n\n")
            
            f.write(f"**Review Comments**:\n")
            f.write(f"- [ ] Translation is accurate\n")
            f.write(f"- [ ] Translation has semantic drift\n")
            f.write(f"- [ ] Low similarity due to embedding model limitation\n")
            f.write(f"- [ ] Needs revision\n\n")
            f.write(f"**Notes**: \n\n")
            f.write(f"---\n\n")
    
    print(f"\n✓ Markdown report exported to: {output_file}")


def main():
    """Main execution."""
    input_file = 'bottom_100_for_review.json'
    
    print(f"\n{'='*80}")
    print("MANUAL REVIEW HELPER")
    print(f"{'='*80}\n")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        print(f"Loaded {len(samples)} samples from {input_file}")
        
        while True:
            print(f"\n{'='*80}")
            print("Options:")
            print("  1. Interactive review")
            print("  2. Show summary")
            print("  3. Export to markdown")
            print("  4. View specific sample")
            print("  5. Exit")
            print(f"{'='*80}")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                interactive_review(samples)
            elif choice == '2':
                print_summary(samples)
            elif choice == '3':
                export_markdown(samples)
            elif choice == '4':
                try:
                    num = int(input(f"Enter sample number (1-{len(samples)}): "))
                    if 1 <= num <= len(samples):
                        print_sample(samples[num-1], num)
                    else:
                        print(f"Invalid number. Must be between 1 and {len(samples)}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please select 1-5.")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found.")
        print("Please run semantic_alignment_analysis.py first.")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{input_file}'")


if __name__ == '__main__':
    main()
