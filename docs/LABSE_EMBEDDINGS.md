# LaBSE Embeddings for Quality Assessment

## What is LaBSE?

**LaBSE** (Language-agnostic BERT Sentence Embeddings) is a multilingual sentence encoder developed by Google Research. It produces semantically meaningful sentence embeddings that work across 109+ languages.

**Model**: `sentence-transformers/LaBSE`
**Dimensions**: 768
**Library**: sentence-transformers

## Why LaBSE for Translation Quality?

### Cross-Lingual Semantic Similarity
LaBSE is specifically designed to embed sentences from different languages into the same vector space. This means:
- Similar sentences in different languages have high cosine similarity
- Semantic meaning is preserved across languages
- Ideal for measuring translation quality

### Advantages Over General Embeddings
- **Multilingual**: Trained on 109 languages including Vietnamese and English
- **Aligned**: English and Vietnamese embeddings are comparable
- **Proven**: Used in production translation systems at Google

### Quality Metric: Cosine Similarity
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

**Interpretation**:
- `>= 0.85`: Excellent translation, semantic equivalence preserved
- `0.75 - 0.85`: Good translation, minor nuance differences
- `0.65 - 0.75`: Acceptable translation, may need review
- `< 0.65`: Poor translation, significant semantic drift

## How It Works

### 1. Encode Questions
```
EN: "How many singers do we have?"
     ↓ LaBSE
EN_embedding: [768-dimensional vector]

VI: "Chúng ta có bao nhiêu ca sĩ?"
     ↓ LaBSE
VI_embedding: [768-dimensional vector]
```

### 2. Compute Similarity
```
similarity = cosine(EN_embedding, VI_embedding)
```

### 3. Quality Assessment
Higher similarity = better translation quality
- Semantic meaning preserved
- Context maintained
- Intent captured correctly

## Embeddings Storage Format

### JSON Structure
```json
{
  "model": "sentence-transformers/LaBSE",
  "embedding_dimension": 768,
  "embeddings": [
    {
      "index": 0,
      "en_embedding": [768 floats],
      "vi_embedding": [768 floats],
      "similarity": 0.8756
    }
  ]
}
```

### File Size
- Raw embeddings: ~25-30 MB per 2,000 samples
- Compressed: Can be reduced with numpy binary format

## Usage in ViSpider

1. **Compute embeddings** for EN/VI question pairs
2. **Calculate similarity** using cosine distance
3. **Filter low-quality** translations (< 0.75)
4. **Manual review** problematic samples
5. **Re-translate** and re-evaluate

## References

- Paper: "Language-agnostic BERT Sentence Embedding" (Feng et al., 2020)
- Model: https://huggingface.co/sentence-transformers/LaBSE
- Library: https://www.sbert.net/
