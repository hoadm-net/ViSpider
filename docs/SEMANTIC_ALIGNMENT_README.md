# Cross-lingual Semantic Alignment Analysis

## Overview

This analysis evaluates the quality of Vietnamese translations using **OpenAI text embeddings** (text-embedding-3-small) and **cosine similarity** to measure semantic alignment between English and Vietnamese questions.

## Setup

### 1. Create .env file

Copy the example and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### 2. Install dependencies

```bash
source venv/bin/activate  # If not already activated
pip install openai python-dotenv numpy scikit-learn
```

Or install all:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run Semantic Alignment Analysis

**Important**: Run from project root directory (`/Users/hoadinh/Desktop/Code/ViSpider`)

```bash
cd /Users/hoadinh/Desktop/Code/ViSpider
python semantic_alignment_analysis.py
```

This will:
1. Load the 2000 translated samples from `data/vispider_train_2000.json`
2. Compute OpenAI embeddings for English and Vietnamese questions
3. Calculate cosine similarities
4. Generate statistics and reports

**Outputs**:
- `similarity_analysis.json` - Full analysis results
- `bottom_100_for_review.json` - 100 lowest similarity samples
- `embeddings_cache.json` - Cached embeddings (saves API calls)
- `vispider_embeddings.json` - **Saved embeddings for all samples** (2000 × 2 embeddings)

**Expected Runtime**: 
- First run: ~5-10 minutes (with caching)
- Subsequent runs: ~30 seconds (using cache)

### Step 2: Review Bottom Samples

```bash
python review_bottom_samples.py
```

Interactive tool to:
- Browse low-similarity samples
- Export markdown review report
- Check if low similarity = bad translation or embedding limitation

## Analysis Steps

### ✅ Step 1: Overall Statistics

Reports:
- **Mean** similarity across all 2000 samples
- **Standard deviation**
- **Min** and **Max**
- **10th percentile** (bottom 10% threshold)

**Expected**: Mean should be > 0.85 for good translations

### ✅ Step 2: By Difficulty

Table showing mean similarity for:
- Easy
- Medium
- Hard  
- Extra Hard

**Quality Checks**:
- Is there a major drop at Hard level?
- Is Extra Hard mean still ≥ 0.80?

**Expected**: Fairly stable across difficulties (no drop > 0.03)

### ✅ Step 3: Bottom 100 Manual Review

Reviews the 100 lowest-similarity samples to determine:
- ❌ Real semantic drift (bad translation)
- ✅ Embedding model limitation (translation is actually fine)

**Expected**: Most should be embedding limitations, not translation errors

## Interpretation Guide

### Similarity Score Ranges

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 - 1.00 | Excellent semantic alignment |
| 0.85 - 0.89 | Good alignment |
| 0.80 - 0.84 | Acceptable alignment |
| 0.75 - 0.79 | Review recommended |
| < 0.75 | Manual review required |

### Common Causes of Lower Similarity

1. **Legitimate variations** (not errors):
   - Different word order (Vietnamese vs English grammar)
   - More/less verbose translations
   - Cultural adaptations
   
2. **Embedding model limitations**:
   - Model may not capture all cross-lingual semantics
   - Technical terms handled differently
   
3. **Actual translation issues**:
   - Missed meanings
   - Incorrect interpretations
   - Omitted information

## Files

### Scripts
- `semantic_alignment_analysis.py` - Main analysis script
- `review_bottom_samples.py` - Interactive review tool
- `embeddings_utils.py` - **Utilities to work with saved embeddings**

### Configuration
- `.env` - API keys (not in git)
- `.env.example` - Template for .env
- `requirements.txt` - Python dependencies

### Outputs (gitignored)
- `similarity_analysis.json` - Full results
- `bottom_100_for_review.json` - Low-similari (by text)
- `vispider_embeddings.json` - **Saved embeddings for all 2000 samples**
- `embeddings_for_training.npz` - NumPy format for ML models
- `Working with Saved Embeddings

After running the analysis, embeddings are saved in `vispider_embeddings.json`. You can reuse them:

### Load Embeddings

```python
from embeddings_utils import load_embeddings

# Load saved embeddings
embeddings_data = load_embeddings()

# Access embeddings
en_embedding = embeddings_data['embeddings'][0]['en_embedding']
vi_embedding = embeddings_data['embeddings'][0]['vi_embedding']
similarity = embeddings_data['embeddings'][0]['similarity']
```

### Export for ML Training

```bash
python embeddings_utils.py
# Select option 1 to export NumPy format
```

This creates `embeddings_for_training.npz` with:
- English embeddings (1996 × 1536)
- Vietnamese embeddings (1996 × 1536)
- Similarity scores
- Sample IDs

### Find Similar Questions

```python
from embeddings_utils import find_similar_questions

# Find 5 most similar questions to sample 100
similar = find_similar_questions(embeddings_data, data, target_index=100, top_k=5)

for item in similar:
    print(f"Similarity: {item['similarity']:.4f}")
    print(f"Question: {item['vi_question']}")
```

### Create Merged Dataset

```bash
python embeddings_utils.py
# Select option 2
```

Creates `vispider_with_embeddings.json` with embeddings included in each sample.

## vispider_with_embeddings.json` - Merged data + embeddingsty samples
- `embeddings_cache.json` - Cached embeddings
- `review_report.md` - Markdown review template

## Tips

### Reduce API Costs
- Embeddings are cached automatically
- Rerun analysis uses cache (no new API calls)
- Delete `embeddings_cache.json` to force refresh

### Batch Processing
- Script processes in batches with rate limiting
- Adjust `BATCH_SIZE` in script if needed

### Review Efficiency
- Use `review_bottom_samples.py` for structured review
- Export to markdown for team collaboration
- Focus on bottom 10% (systematic issues show up here)

## Expected Results

Based on the data analysis, we expect:

- **Overall mean**: 0.87 - 0.92
- **Standard deviation**: 0.03 - 0.06
- **10th percentile**: > 0.82
- **Difficulty variance**: < 0.03 between levels
- **Bottom 100**: Mostly false positives (embedding limitations)

## Troubleshooting

### Error: OPENAI_API_KEY not found
- Create `.env` file with your API key
- Ensure file is in project root

### Rate limit errors
- Script includes rate limiting (0.05s delay)
- If hit limit, wait and rerun (cache will save progress)

### Out of memory
- Process uses ~2GB RAM for embeddings
- Reduce batch size if needed

## Next Steps

After analysis:

1. Review bottom 100 samples
2. Identify patterns in low-similarity cases
3. Update translation guidelines if needed
4. Re-translate problematic samples
5. Rerun analysis to verify improvements

---

**Last Updated**: February 14, 2026
