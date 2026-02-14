#!/usr/bin/env python3
"""Test LaBSE model for cross-lingual similarity on CPU."""

import time
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers installed")
except ImportError:
    print("❌ sentence-transformers not installed")
    print("Run: pip install sentence-transformers")
    exit(1)

# Test samples from your data
test_pairs = [
    ("List the creation year, name and budget of each department.",
     "Liệt kê năm thành lập, tên và ngân sách của mỗi phòng ban."),
    
    ("What are the names of the heads who are born outside the California state?",
     "Tên của những trưởng phòng mà được sinh ra ngoài bang California là gì?"),
    
    ("Return the themes of farm competitions, sorted by year ascending.",
     "Trả về các chủ đề của các hội chợ nông sản, sắp xếp tăng dần theo năm"),
]

print("\nLoading LaBSE model...")
start = time.time()
model = SentenceTransformer('sentence-transformers/LaBSE')
load_time = time.time() - start
print(f"✓ Model loaded in {load_time:.2f}s")

print(f"\n{'='*80}")
print("Testing cross-lingual semantic similarity:")
print(f"{'='*80}\n")

start = time.time()
for i, (en, vi) in enumerate(test_pairs, 1):
    # Encode both sentences
    en_emb = model.encode(en, convert_to_numpy=True)
    vi_emb = model.encode(vi, convert_to_numpy=True)
    
    # Calculate cosine similarity
    similarity = np.dot(en_emb, vi_emb) / (np.linalg.norm(en_emb) * np.linalg.norm(vi_emb))
    
    print(f"[{i}] Similarity: {similarity:.4f}")
    print(f"  EN: {en[:70]}")
    print(f"  VI: {vi[:70]}")
    print()

encode_time = time.time() - start
print(f"Encoding time: {encode_time:.2f}s for {len(test_pairs)*2} sentences")
print(f"Speed: {encode_time/(len(test_pairs)*2):.2f}s per sentence")

# Estimate for full dataset
total_samples = 1996
total_sentences = total_samples * 2  # EN + VI
estimated_time = (encode_time / (len(test_pairs)*2)) * total_sentences
print(f"\nEstimated time for {total_samples} samples: {estimated_time:.1f}s (~{estimated_time/60:.1f} minutes)")
