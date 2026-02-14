#!/usr/bin/env python3
"""Quick check of similarity values in embeddings file."""

import json
import numpy as np

# Load embeddings
with open('data/manual_translations/vispider_embeddings.json') as f:
    data = json.load(f)

embeddings = data['embeddings']
print(f'Total samples: {len(embeddings)}')

# Count different states
has_both = 0
has_en_only = 0
has_vi_only = 0
has_none = 0
non_zero_sim = 0
zero_vectors = 0

for e in embeddings:
    en = e.get('en_embedding')
    vi = e.get('vi_embedding')
    sim = e.get('similarity', 0)
    
    if en and vi:
        has_both += 1
        # Check if zero vector
        if all(x == 0 for x in en[:10]) or all(x == 0 for x in vi[:10]):
            zero_vectors += 1
        if sim > 0.01:
            non_zero_sim += 1
    elif en:
        has_en_only += 1
    elif vi:
        has_vi_only += 1
    else:
        has_none += 1

print(f'\n📊 Embeddings status:')
print(f'  Both EN+VI: {has_both}')
print(f'  EN only: {has_en_only}')
print(f'  VI only: {has_vi_only}')
print(f'  None: {has_none}')
print(f'  Zero vectors: {zero_vectors}')
print(f'  Non-zero similarity: {non_zero_sim}')

# Check similarities that are stored
stored_sims = [e.get('similarity', 0) for e in embeddings]
valid_sims = [s for s in stored_sims if s > 0.001]

print(f'\n📈 Stored similarity values:')
print(f'  Total: {len(stored_sims)}')
print(f'  Valid (>0.001): {len(valid_sims)}')
if valid_sims:
    print(f'  Mean: {np.mean(valid_sims):.4f}')
    print(f'  Min: {np.min(valid_sims):.4f}')
    print(f'  Max: {np.max(valid_sims):.4f}')
    print(f'  Median: {np.median(valid_sims):.4f}')

# Calculate fresh similarities for samples with both embeddings
print(f'\n🔍 Recalculating similarities for {has_both} samples with both embeddings:')
fresh_sims = []
for e in embeddings:
    en = e.get('en_embedding')
    vi = e.get('vi_embedding')
    
    if en and vi:
        # Check for zero vectors
        if all(x == 0 for x in en[:10]) or all(x == 0 for x in vi[:10]):
            continue
            
        # Calculate cosine similarity
        en_arr = np.array(en)
        vi_arr = np.array(vi)
        
        dot_product = np.dot(en_arr, vi_arr)
        norm_en = np.linalg.norm(en_arr)
        norm_vi = np.linalg.norm(vi_arr)
        
        if norm_en > 0 and norm_vi > 0:
            sim = dot_product / (norm_en * norm_vi)
            fresh_sims.append(sim)

if fresh_sims:
    print(f'  Valid recalculations: {len(fresh_sims)}')
    print(f'  Mean: {np.mean(fresh_sims):.4f}')
    print(f'  Min: {np.min(fresh_sims):.4f}')
    print(f'  Max: {np.max(fresh_sims):.4f}')
    print(f'  Median: {np.median(fresh_sims):.4f}')
    print(f'  Std: {np.std(fresh_sims):.4f}')
    
    # Show distribution
    print(f'\n📊 Distribution:')
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for s in fresh_sims if bins[i] <= s < bins[i+1])
        pct = count * 100 / len(fresh_sims)
        print(f'  {bins[i]:.1f}-{bins[i+1]:.1f}: {count} ({pct:.1f}%)')

# Check a few examples
print(f'\n🔎 Sample examples (index, stored_sim, recalc_sim):')
shown = 0
for i, e in enumerate(embeddings[:20]):
    en = e.get('en_embedding')
    vi = e.get('vi_embedding')
    stored_sim = e.get('similarity', 0)
    
    if en and vi and not all(x == 0 for x in en[:10]):
        en_arr = np.array(en)
        vi_arr = np.array(vi)
        recalc_sim = np.dot(en_arr, vi_arr) / (np.linalg.norm(en_arr) * np.linalg.norm(vi_arr))
        print(f'  [{i}] stored: {stored_sim:.4f}, recalc: {recalc_sim:.4f}')
        shown += 1
        if shown >= 5:
            break
