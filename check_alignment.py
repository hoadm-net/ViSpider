#!/usr/bin/env python3
"""Check if EN and VI questions are properly aligned."""

import json

# Load data
print("Loading data...")
with open('data/manual_translations/vispider_train_2000.json') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}\n")

# Check first 10 samples
print("=" * 80)
print("Checking first 10 samples for EN-VI alignment:")
print("=" * 80)

for i in range(min(10, len(data))):
    item = data[i]
    print(f"\n[{i}] ID: {item['id']}, DB: {item['db_id']}")
    print(f"EN: {item['question'][:80]}")
    print(f"VI: {item['vi_question'][:80]}")
    print(f"SQL: {item['query'][:60]}")

# Load embeddings
print("\n" + "=" * 80)
print("Checking embeddings alignment:")
print("=" * 80)

with open('data/manual_translations/vispider_embeddings.json') as f:
    emb_data = json.load(f)

embeddings = emb_data['embeddings']

# Check if indices match
print(f"\nData samples: {len(data)}")
print(f"Embedding entries: {len(embeddings)}")

# Check a few with low similarity
print("\n" + "=" * 80)
print("Samples with LOW similarity (< 0.4):")
print("=" * 80)

low_sim_samples = [e for e in embeddings if e.get('similarity', 1) < 0.4][:5]
for e in low_sim_samples:
    idx = e['index']
    sim = e.get('similarity', 0)
    
    if idx < len(data):
        item = data[idx]
        print(f"\n[{idx}] Similarity: {sim:.4f}")
        print(f"  EN: {item['question']}")
        print(f"  VI: {item['vi_question']}")
        print(f"  Query: {item['query'][:50]}")

# Check a few with high similarity
print("\n" + "=" * 80)
print("Samples with HIGH similarity (> 0.7):")
print("=" * 80)

high_sim_samples = sorted([e for e in embeddings if e.get('similarity', 0) > 0.7], 
                         key=lambda x: x.get('similarity', 0), reverse=True)[:5]
for e in high_sim_samples:
    idx = e['index']
    sim = e.get('similarity', 0)
    
    if idx < len(data):
        item = data[idx]
        print(f"\n[{idx}] Similarity: {sim:.4f}")
        print(f"  EN: {item['question']}")
        print(f"  VI: {item['vi_question']}")
        print(f"  Hardness: {item['hardness']}")
