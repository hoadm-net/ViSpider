#!/usr/bin/env python3
"""
Test LaBSE setup and verify it works correctly for ViSpider.
Run this before running the full semantic_alignment_analysis.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

def test_labse_basic():
    """Test basic LaBSE functionality."""
    print("\n" + "="*80)
    print("Testing LaBSE Setup for ViSpider")
    print("="*80 + "\n")
    
    # Load model
    print("1. Loading LaBSE model...")
    try:
        model = SentenceTransformer('sentence-transformers/LaBSE')
        print("   ✓ Model loaded successfully\n")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        print("\n   Please install sentence-transformers:")
        print("   pip install sentence-transformers")
        return False
    
    # Test samples from ViSpider
    print("2. Testing with real ViSpider samples...\n")
    
    test_pairs = [
        {
            "en": "List the creation year, name and budget of each department.",
            "vi": "Liệt kê năm thành lập, tên và ngân sách của mỗi phòng ban.",
        },
        {
            "en": "What are the names of the heads who are born outside the California state?",
            "vi": "Tên của những trưởng phòng mà được sinh ra ngoài bang California là gì?",
        },
        {
            "en": "How many singers do we have?",
            "vi": "Chúng ta có bao nhiêu ca sĩ?",
        },
    ]
    
    for i, pair in enumerate(test_pairs, 1):
        # Encode
        en_emb = model.encode(pair["en"], convert_to_numpy=True)
        vi_emb = model.encode(pair["vi"], convert_to_numpy=True)
        
        # Calculate similarity
        similarity = np.dot(en_emb, vi_emb) / (np.linalg.norm(en_emb) * np.linalg.norm(vi_emb))
        
        print(f"   Sample {i}:")
        print(f"   EN: {pair['en'][:70]}...")
        print(f"   VI: {pair['vi'][:70]}...")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Embedding dimension: {len(en_emb)}")
        print()
    
    # Test batch encoding
    print("3. Testing batch encoding...")
    try:
        en_texts = [p["en"] for p in test_pairs]
        vi_texts = [p["vi"] for p in test_pairs]
        
        en_embeddings = model.encode(en_texts, convert_to_numpy=True, batch_size=32)
        vi_embeddings = model.encode(vi_texts, convert_to_numpy=True, batch_size=32)
        
        print(f"   ✓ Batch encoding successful")
        print(f"   EN embeddings shape: {en_embeddings.shape}")
        print(f"   VI embeddings shape: {vi_embeddings.shape}")
        print()
    except Exception as e:
        print(f"   ✗ Batch encoding failed: {e}\n")
        return False
    
    # Load actual data file to verify
    print("4. Checking ViSpider data file...")
    data_file = 'data/manual_translations/vispider_train_2000.json'
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   ✓ Found {len(data)} samples in {data_file}")
        
        # Test encoding first sample
        if data:
            sample = data[0]
            en_emb = model.encode(sample['question'], convert_to_numpy=True)
            vi_emb = model.encode(sample['vi_question'], convert_to_numpy=True)
            sim = np.dot(en_emb, vi_emb) / (np.linalg.norm(en_emb) * np.linalg.norm(vi_emb))
            
            print(f"\n   First sample test:")
            print(f"   EN: {sample['question'][:60]}...")
            print(f"   VI: {sample['vi_question'][:60]}...")
            print(f"   Similarity: {sim:.4f}")
        print()
    except FileNotFoundError:
        print(f"   ✗ Data file not found: {data_file}")
        print(f"   Make sure you're in the project root directory\n")
        return False
    except Exception as e:
        print(f"   ✗ Error loading data: {e}\n")
        return False
    
    print("="*80)
    print("✅ All tests passed! LaBSE is ready for semantic alignment analysis.")
    print("="*80)
    print("\nNext step:")
    print("  python3 semantic_alignment_analysis.py")
    print()
    
    return True


if __name__ == '__main__':
    success = test_labse_basic()
    exit(0 if success else 1)
