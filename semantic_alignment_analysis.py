#!/usr/bin/env python3
"""
Cross-lingual Semantic Alignment Analysis
Evaluates translation quality using OpenAI embeddings and cosine similarity.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # Process in batches to avoid rate limits
CACHE_FILE = "embeddings_cache.json"
MAX_TOKENS = 8000  # Safe limit, model supports 8192
ERROR_LOG_FILE = "embedding_errors.json"
EMBEDDINGS_FILE = "data/manual_translations/vispider_embeddings.json"  # Main embeddings file


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Truncate text to fit within token limit.
    Rough estimate: 1 token ≈ 4 characters for English, ≈ 1-2 chars for Vietnamese
    """
    # Conservative estimate: 1 token = 2 characters
    max_chars = max_tokens * 2
    if len(text) > max_chars:
        print(f"⚠️  Warning: Text truncated from {len(text)} to {max_chars} chars")
        return text[:max_chars]
    return text


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
    """
    Get embedding for a single text.
    Returns None if error occurs.
    """
    try:
        # Truncate if too long
        text = truncate_text(text)
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error getting embedding: {str(e)[:100]}")
        return None


def get_embeddings_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Get embeddings for a batch of texts."""
    # Clean texts
    texts = [text.replace("\n", " ") for text in texts]
    
    # Call API
    response = client.embeddings.create(input=texts, model=model)
    
    # Extract embeddings in order
    return [item.embedding for item in response.data]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)


def load_embeddings_cache() -> Dict:
    """Load cached embeddings if available."""
    if Path(CACHE_FILE).exists():
        print(f"Loading cached embeddings from {CACHE_FILE}")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_embeddings_cache(cache: Dict):
    """Save embeddings to cache."""
    print(f"Saving embeddings to {CACHE_FILE}")
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f)


def load_existing_embeddings(data: List[Dict]) -> Tuple[List[Optional[List[float]]], List[Optional[List[float]]]]:
    """
    Load existing embeddings from file if available.
    Returns two lists with None for missing embeddings.
    """
    if not Path(EMBEDDINGS_FILE).exists():
        print(f"No existing embeddings found at {EMBEDDINGS_FILE}")
        return [None] * len(data), [None] * len(data)
    
    print(f"Loading existing embeddings from {EMBEDDINGS_FILE}...")
    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    # Initialize with None
    en_embeddings = [None] * len(data)
    vi_embeddings = [None] * len(data)
    
    # Fill in saved embeddings
    for item in saved_data.get('embeddings', []):
        idx = item['index']
        if idx < len(data):
            en_embeddings[idx] = item['en_embedding']
            vi_embeddings[idx] = item['vi_embedding']
    
    loaded_count = sum(1 for e in en_embeddings if e is not None)
    print(f"✓ Loaded {loaded_count}/{len(data)} existing embeddings")
    
    return en_embeddings, vi_embeddings


def save_embeddings_incremental(data: List[Dict], 
                                en_embeddings: List[List[float]], 
                                vi_embeddings: List[List[float]],
                                similarities: Optional[List[float]] = None):
    """
    Save embeddings incrementally to file.
    This allows resuming if the process crashes.
    """
    print(f"💾 Saving embeddings to {EMBEDDINGS_FILE}...")
    
    # Calculate similarities if not provided
    if similarities is None:
        similarities = []
        for en_emb, vi_emb in zip(en_embeddings, vi_embeddings):
            if en_emb and vi_emb:
                sim = cosine_similarity(en_emb, vi_emb)
                similarities.append(sim)
            else:
                similarities.append(0.0)
    
    embeddings_data = {
        'model': EMBEDDING_MODEL,
        'total_samples': len(data),
        'embedding_dimension': 1536,
        'embeddings': [
            {
                'index': i,
                'en_embedding': en_emb,
                'vi_embedding': vi_emb,
                'similarity': float(similarities[i]) if i < len(similarities) else 0.0
            }
            for i, (en_emb, vi_emb) in enumerate(zip(en_embeddings, vi_embeddings))
        ]
    }
    
    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)


def compute_embeddings(data: List[Dict], use_cache: bool = True) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute embeddings for all English and Vietnamese questions.
    Loads existing embeddings and only computes missing ones.
    Saves after each batch to allow resuming if process crashes.
    
    Returns:
        Tuple of (en_embeddings, vi_embeddings)
    """
    total = len(data)
    
    print(f"\n{'='*80}")
    print(f"Computing embeddings for {total} samples")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max tokens per text: {MAX_TOKENS}")
    print(f"{'='*80}\n")
    
    # Load existing embeddings
    en_embeddings, vi_embeddings = load_existing_embeddings(data)
    
    # Count missing embeddings
    missing_indices = [i for i in range(len(data)) if en_embeddings[i] is None or vi_embeddings[i] is None]
    
    if not missing_indices:
        print(f"\n✅ All {total} embeddings already exist!")
        return en_embeddings, vi_embeddings
    
    print(f"\n📝 Need to compute {len(missing_indices)} missing embeddings")
    print(f"   (Skipping {total - len(missing_indices)} existing embeddings)\n")
    
    errors = []
    
    # Process missing embeddings with progress bar
    with tqdm(total=len(missing_indices), desc="Computing embeddings", unit="sample") as pbar:
        for batch_start in range(0, len(missing_indices), BATCH_SIZE):
            batch_indices = missing_indices[batch_start:batch_start + BATCH_SIZE]
            
            for idx in batch_indices:
                # Process English if missing
                if en_embeddings[idx] is None:
                    text = data[idx]['question']
                    emb = get_embedding(text)
                    if emb is None:
                        errors.append({
                            'index': idx,
                            'lang': 'en',
                            'text_preview': text[:100],
                            'text_length': len(text)
                        })
                        emb = [0.0] * 1536
                        print(f"  ⚠️  Using zero vector for EN sample {idx}")
                    en_embeddings[idx] = emb
                    time.sleep(0.05)
                
                # Process Vietnamese if missing
                if vi_embeddings[idx] is None:
                    text = data[idx]['vi_question']
                    emb = get_embedding(text)
                    if emb is None:
                        errors.append({
                            'index': idx,
                            'lang': 'vi',
                            'text_preview': text[:100],
                            'text_length': len(text)
                        })
                        emb = [0.0] * 1536
                        print(f"  ⚠️  Using zero vector for VI sample {idx}")
                    vi_embeddings[idx] = emb
                    time.sleep(0.05)
                
                pbar.update(1)
            
            # Save after each batch
            pbar.set_postfix({"status": "Saving checkpoint"})
            save_embeddings_incremental(data, en_embeddings, vi_embeddings)
            pbar.set_postfix({"status": "Ready"})
    
    print(f"\n✅ All embeddings computed and saved!")
    
    # Save error log if any errors occurred
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred during processing")
        with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"   Error details saved to {ERROR_LOG_FILE}")
    
    return en_embeddings, vi_embeddings


def analyze_similarities(data: List[Dict], similarities: List[float]) -> Dict:
    """Perform comprehensive similarity analysis."""
    
    print(f"\n{'='*80}")
    print("STEP 1: OVERALL SIMILARITY STATISTICS")
    print(f"{'='*80}\n")
    
    similarities_array = np.array(similarities)
    
    # Overall statistics
    stats = {
        'mean': float(np.mean(similarities_array)),
        'std': float(np.std(similarities_array)),
        'min': float(np.min(similarities_array)),
        'max': float(np.max(similarities_array)),
        'percentile_10': float(np.percentile(similarities_array, 10)),
        'percentile_25': float(np.percentile(similarities_array, 25)),
        'percentile_50': float(np.percentile(similarities_array, 50)),
        'percentile_75': float(np.percentile(similarities_array, 75)),
        'percentile_90': float(np.percentile(similarities_array, 90)),
    }
    
    print(f"Mean:             {stats['mean']:.4f}")
    print(f"Std Dev:          {stats['std']:.4f}")
    print(f"Min:              {stats['min']:.4f}")
    print(f"Max:              {stats['max']:.4f}")
    print(f"10th percentile:  {stats['percentile_10']:.4f}")
    print(f"25th percentile:  {stats['percentile_25']:.4f}")
    print(f"Median:           {stats['percentile_50']:.4f}")
    print(f"75th percentile:  {stats['percentile_75']:.4f}")
    print(f"90th percentile:  {stats['percentile_90']:.4f}")
    
    # Step 2: By difficulty
    print(f"\n{'='*80}")
    print("STEP 2: SIMILARITY BY DIFFICULTY LEVEL")
    print(f"{'='*80}\n")
    
    difficulty_stats = {}
    difficulty_order = ['easy', 'medium', 'hard', 'extra_hard']
    
    print(f"{'Difficulty':<15} {'Count':<8} {'Mean Similarity':<20} {'Std Dev':<10}")
    print(f"{'-'*15} {'-'*8} {'-'*20} {'-'*10}")
    
    for difficulty in difficulty_order:
        indices = [i for i, item in enumerate(data) if item['hardness'] == difficulty]
        if indices:
            diff_similarities = similarities_array[indices]
            difficulty_stats[difficulty] = {
                'count': len(indices),
                'mean': float(np.mean(diff_similarities)),
                'std': float(np.std(diff_similarities)),
                'min': float(np.min(diff_similarities)),
                'max': float(np.max(diff_similarities)),
            }
            print(f"{difficulty:<15} {len(indices):<8} {difficulty_stats[difficulty]['mean']:.4f}               {difficulty_stats[difficulty]['std']:.4f}")
    
    # Check for drops
    print(f"\nAnalysis:")
    if 'hard' in difficulty_stats and 'medium' in difficulty_stats:
        drop_hard = difficulty_stats['medium']['mean'] - difficulty_stats['hard']['mean']
        print(f"  Drop from Medium to Hard: {drop_hard:.4f} ({drop_hard*100:.2f}%)")
        if abs(drop_hard) < 0.02:
            print(f"  ✅ No significant drop at Hard level")
        else:
            print(f"  ⚠️  Notable drop at Hard level")
    
    if 'extra_hard' in difficulty_stats:
        if difficulty_stats['extra_hard']['mean'] >= 0.80:
            print(f"  ✅ Extra Hard mean ({difficulty_stats['extra_hard']['mean']:.4f}) >= 0.80")
        else:
            print(f"  ⚠️  Extra Hard mean ({difficulty_stats['extra_hard']['mean']:.4f}) < 0.80")
    
    # Step 3: Bottom samples
    print(f"\n{'='*80}")
    print("STEP 3: BOTTOM SAMPLES ANALYSIS")
    print(f"{'='*80}\n")
    
    # Get bottom 10%
    bottom_threshold = np.percentile(similarities_array, 10)
    bottom_indices = np.where(similarities_array <= bottom_threshold)[0]
    
    print(f"Bottom 10% threshold: {bottom_threshold:.4f}")
    print(f"Number of samples in bottom 10%: {len(bottom_indices)}")
    
    # Get bottom 100 for manual review
    sorted_indices = np.argsort(similarities_array)
    bottom_100_indices = sorted_indices[:min(100, len(sorted_indices))]
    
    bottom_100 = []
    for idx in bottom_100_indices:
        bottom_100.append({
            'index': int(idx),
            'id': data[idx]['id'],
            'db_id': data[idx]['db_id'],
            'hardness': data[idx]['hardness'],
            'similarity': float(similarities[idx]),
            'question': data[idx]['question'],
            'vi_question': data[idx]['vi_question'],
            'query': data[idx]['query'],
        })
    
    # Show top 10 lowest
    print(f"\nTop 10 lowest similarity samples:")
    print(f"{'Rank':<6} {'Similarity':<12} {'Difficulty':<12} {'ID':<15}")
    print(f"{'-'*6} {'-'*12} {'-'*12} {'-'*15}")
    for i, sample in enumerate(bottom_100[:10]):
        print(f"{i+1:<6} {sample['similarity']:.4f}       {sample['hardness']:<12} {sample['id']:<15}")
    
    return {
        'overall_stats': stats,
        'difficulty_stats': difficulty_stats,
        'bottom_100': bottom_100,
        'bottom_threshold': float(bottom_threshold),
    }


def save_results(results: Dict, similarities: List[float], 
                en_embeddings: List[List[float]], vi_embeddings: List[List[float]],
                output_file: str = "similarity_analysis.json"):
    """Save analysis results to file."""
    results['all_similarities'] = [float(s) for s in similarities]
    
    print(f"\n{'='*80}")
    print(f"Saving results to {output_file}")
    print(f"{'='*80}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Also save bottom 100 separately for easy review
    bottom_file = "bottom_100_for_review.json"
    with open(bottom_file, 'w', encoding='utf-8') as f:
        json.dump(results['bottom_100'], f, ensure_ascii=False, indent=2)
    
    # Update embeddings file with final similarities
    # (embeddings were already saved incrementally during compute_embeddings)
    print(f"Updating {EMBEDDINGS_FILE} with final similarities...")
    if Path(EMBEDDINGS_FILE).exists():
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
        
        # Update similarities
        for item in embeddings_data['embeddings']:
            idx = item['index']
            if idx < len(similarities):
                item['similarity'] = float(similarities[idx])
        
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Full results saved to: {output_file}")
    print(f"✓ Bottom 100 saved to: {bottom_file}")
    print(f"✓ Embeddings updated in: {EMBEDDINGS_FILE}")


def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print("CROSS-LINGUAL SEMANTIC ALIGNMENT ANALYSIS")
    print("ViSpider Translation Quality Evaluation")
    print(f"{'='*80}\n")
    
    # Load data
    data_file = 'data/manual_translations/vispider_train_2000.json'
    data_file_abs = Path(data_file).resolve()
    
    if not Path(data_file).exists():
        print(f"❌ ERROR: Data file not found: {data_file}")
        print(f"Expected location: {data_file_abs}")
        print("\nPlease ensure you have:")
        print("  1. Run parse_label_studio.py first")
        print("  2. Running from project root directory")
        return
    
    print(f"Loading data from {data_file}...")
    print(f"Full path: {data_file_abs}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples\n")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your API key (see .env.example)")
        return
    
    # Compute embeddings
    en_embeddings, vi_embeddings = compute_embeddings(data, use_cache=True)
    
    # Calculate similarities
    print(f"\n{'='*80}")
    print("Calculating cosine similarities...")
    print(f"{'='*80}\n")
    
    similarities = []
    for i, (en_emb, vi_emb) in enumerate(zip(en_embeddings, vi_embeddings)):
        sim = cosine_similarity(en_emb, vi_emb)
        similarities.append(sim)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(data)} similarities")
    
    print(f"  Completed {len(similarities)}/{len(data)} similarities\n")
    
    # Analyze
    results = analyze_similarities(data, similarities)
    
    # Save results with embeddings
    save_results(results, similarities, en_embeddings, vi_embeddings)
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    print("Summary:")
    print(f"  Mean similarity: {results['overall_stats']['mean']:.4f}")
    print(f"  Samples analyzed: {len(data)}")
    print(f"  Bottom 10% threshold: {results['bottom_threshold']:.4f}")
    print(f"\nNext step: Manually review bottom_100_for_review.json")


if __name__ == '__main__':
    main()
