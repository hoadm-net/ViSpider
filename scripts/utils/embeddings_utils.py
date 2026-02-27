#!/usr/bin/env python3
"""
Embeddings Utilities
Helper functions to load and work with saved embeddings,
and compute new embeddings using LaBSE model.
"""

import json
import numpy as np
from typing import Dict, List, Tuple

# LaBSE model - lazy loaded on first use
_labse_model = None
LABSE_MODEL_NAME = 'sentence-transformers/LaBSE'


def get_labse_model():
    """Lazy-load LaBSE model (singleton)."""
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading LaBSE model: {LABSE_MODEL_NAME}")
        _labse_model = SentenceTransformer(LABSE_MODEL_NAME)
        print("✓ LaBSE model loaded")
    return _labse_model


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """
    Compute LaBSE embeddings for a list of texts.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_labse_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using LaBSE.
    
    Returns:
        float in [-1, 1], higher is more similar
    """
    embeddings = compute_embeddings([text1, text2])
    return float(np.dot(embeddings[0], embeddings[1]))


def load_embeddings(embeddings_file: str = "vispider_embeddings.json") -> Dict:
    """
    Load saved embeddings from file.
    
    Returns:
        Dict with model info and embeddings list
    """
    print(f"Loading embeddings from {embeddings_file}...")
    
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {data['total_samples']} embeddings")
    print(f"  Model: {data['model']}")
    print(f"  Dimension: {data['embedding_dimension']}")
    
    return data


def get_embedding_by_index(embeddings_data: Dict, index: int) -> Tuple[List[float], List[float], float]:
    """
    Get embeddings for a specific sample by index.
    
    Returns:
        Tuple of (en_embedding, vi_embedding, similarity)
    """
    emb = embeddings_data['embeddings'][index]
    return emb['en_embedding'], emb['vi_embedding'], emb['similarity']


def get_all_en_embeddings(embeddings_data: Dict) -> np.ndarray:
    """Get all English embeddings as numpy array."""
    return np.array([item['en_embedding'] for item in embeddings_data['embeddings']])


def get_all_vi_embeddings(embeddings_data: Dict) -> np.ndarray:
    """Get all Vietnamese embeddings as numpy array."""
    return np.array([item['vi_embedding'] for item in embeddings_data['embeddings']])


def get_all_similarities(embeddings_data: Dict) -> np.ndarray:
    """Get all similarity scores as numpy array."""
    return np.array([item['similarity'] for item in embeddings_data['embeddings']])


def find_similar_questions(embeddings_data: Dict, data: List[Dict], 
                          target_index: int, top_k: int = 5) -> List[Dict]:
    """
    Find similar questions to a target question based on embeddings.
    
    Args:
        embeddings_data: Loaded embeddings
        data: Original vispider data
        target_index: Index of target question
        top_k: Number of similar questions to return
        
    Returns:
        List of similar questions with similarity scores
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get target embedding (use Vietnamese embedding)
    target_emb = embeddings_data['embeddings'][target_index]['vi_embedding']
    target_emb = np.array(target_emb).reshape(1, -1)
    
    # Get all Vietnamese embeddings
    all_vi_embs = get_all_vi_embeddings(embeddings_data)
    
    # Calculate similarities
    similarities = cosine_similarity(target_emb, all_vi_embs)[0]
    
    # Get top-k (excluding self)
    sorted_indices = np.argsort(similarities)[::-1]
    similar_indices = [idx for idx in sorted_indices if idx != target_index][:top_k]
    
    # Build results
    results = []
    for idx in similar_indices:
        results.append({
            'index': int(idx),
            'similarity': float(similarities[idx]),
            'id': data[idx]['id'],
            'db_id': data[idx]['db_id'],
            'question': data[idx]['question'],
            'vi_question': data[idx]['vi_question'],
            'query': data[idx]['query'],
        })
    
    return results


def export_embeddings_for_model(embeddings_data: Dict, data: List[Dict], 
                                output_file: str = "embeddings_for_training.npz"):
    """
    Export embeddings in numpy format for ML model training.
    
    Saves:
        - en_embeddings: English embeddings
        - vi_embeddings: Vietnamese embeddings
        - similarities: Similarity scores
        - ids: Sample IDs
        - db_ids: Database IDs
    """
    print(f"Exporting embeddings to {output_file}...")
    
    en_embs = get_all_en_embeddings(embeddings_data)
    vi_embs = get_all_vi_embeddings(embeddings_data)
    similarities = get_all_similarities(embeddings_data)
    
    ids = np.array([item['id'] for item in data])
    db_ids = np.array([item['db_id'] for item in data])
    
    np.savez_compressed(
        output_file,
        en_embeddings=en_embs,
        vi_embeddings=vi_embs,
        similarities=similarities,
        ids=ids,
        db_ids=db_ids,
        model=embeddings_data['model'],
        dimension=embeddings_data['embedding_dimension']
    )
    
    print(f"✓ Exported to {output_file}")
    print(f"  Shape: {en_embs.shape}")
    
    # Print loading instructions
    print(f"\nTo load:")
    print(f"  data = np.load('{output_file}')")
    print(f"  en_embeddings = data['en_embeddings']")
    print(f"  vi_embeddings = data['vi_embeddings']")


def create_embedding_index(embeddings_data: Dict, data: List[Dict], 
                          output_file: str = "vispider_with_embeddings.json"):
    """
    Create a merged file with data + embeddings for each sample.
    
    Args:
        embeddings_data: Loaded embeddings
        data: Original vispider data
        output_file: Output file path
    """
    print(f"Creating merged data+embeddings file...")
    
    merged_data = []
    
    for i, item in enumerate(data):
        emb = embeddings_data['embeddings'][i]
        
        merged_item = {
            **item,  # Original data
            'en_embedding': emb['en_embedding'],
            'vi_embedding': emb['vi_embedding'],
            'similarity': emb['similarity']
        }
        
        merged_data.append(merged_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved merged data to {output_file}")
    print(f"  Total samples: {len(merged_data)}")


def main():
    """Demo usage of embeddings utilities."""
    print("\n" + "="*80)
    print("EMBEDDINGS UTILITIES - DEMO")
    print("="*80 + "\n")
    
    # Load embeddings
    try:
        embeddings_data = load_embeddings()
    except FileNotFoundError:
        print("❌ Error: data/manual_translations/vispider_embeddings.json not found")
        print("Please run semantic_alignment_analysis.py first")
        return
    
    # Load original data
    data_file = 'data/manual_translations/vispider_train_2000.json'
    print(f"\nLoading data from {data_file}...")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} samples")
    except FileNotFoundError:
        from pathlib import Path
        abs_path = Path(data_file).resolve()
        print(f"❌ Error: Data file not found: {data_file}")
        print(f"Expected location: {abs_path}")
        return
    
    print("\n" + "="*80)
    print("Available Operations:")
    print("="*80)
    print("1. Export embeddings for ML (NumPy format)")
    print("2. Create merged data+embeddings file")
    print("3. Find similar questions")
    print("4. Get statistics")
    print("="*80 + "\n")
    
    choice = input("Select operation (1-4): ").strip()
    
    if choice == '1':
        export_embeddings_for_model(embeddings_data, data)
        
    elif choice == '2':
        create_embedding_index(embeddings_data, data)
        
    elif choice == '3':
        try:
            idx = int(input("Enter sample index (0-1995): "))
            if 0 <= idx < len(data):
                print(f"\nTarget question:")
                print(f"  EN: {data[idx]['question']}")
                print(f"  VI: {data[idx]['vi_question']}")
                
                print(f"\nFinding similar questions...")
                similar = find_similar_questions(embeddings_data, data, idx, top_k=5)
                
                print(f"\nTop 5 similar questions:")
                for i, item in enumerate(similar, 1):
                    print(f"\n{i}. Similarity: {item['similarity']:.4f}")
                    print(f"   EN: {item['question']}")
                    print(f"   VI: {item['vi_question']}")
            else:
                print("Invalid index")
        except ValueError:
            print("Invalid input")
            
    elif choice == '4':
        similarities = get_all_similarities(embeddings_data)
        print(f"\nStatistics:")
        print(f"  Total samples: {len(similarities)}")
        print(f"  Mean similarity: {np.mean(similarities):.4f}")
        print(f"  Std deviation: {np.std(similarities):.4f}")
        print(f"  Min: {np.min(similarities):.4f}")
        print(f"  Max: {np.max(similarities):.4f}")
        
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
