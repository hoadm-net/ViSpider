# Quick Reference Guide

## 🚦 Commands Cheatsheet

### Setup
```bash
# Navigate to project
cd /Users/hoadinh/Desktop/Code/ViSpider

# Verify setup
python3 verify_setup.py

# Activate venv (if needed)
source venv/bin/activate
```

### Run Analysis
```bash
# Full automated run
./run_analysis.sh

# Manual step-by-step
python3 verify_setup.py                    # Check setup
python3 semantic_alignment_analysis.py     # Run analysis
python3 review_bottom_samples.py           # Review results
python3 embeddings_utils.py                # Work with embeddings
python3 test_embeddings.py                 # Test integrity
```

### Parse Data
```bash
# Parse Label Studio export
python3 parse_label_studio.py

# Analyze translations
python3 analyze_vispider.py
```

## 📁 Important File Paths

### Input Data
```
data/vispider_train_2000.json              ← Main data file (1996 samples)
data/label_studio_2000_samples.json        ← Raw Label Studio export
data/raw/train_spider.json                 ← Original Spider train
data/raw/tables.json                       ← Database schemas
```

### Output Files
```
vispider_embeddings.json                   ← Embeddings (~55MB) [gitignored]
similarity_analysis.json                   ← Analysis results [gitignored]
bottom_100_for_review.json                 ← Low-similarity samples [gitignored]
embeddings_cache.json                      ← API cache [gitignored]
embeddings_for_training.npz                ← NumPy format [gitignored]
```

### Configuration
```
.env                                       ← API keys (local only)
.env.example                               ← Template for .env
requirements.txt                           ← Python dependencies
```

## 🔑 Environment Variables

```bash
# .env file format
OPENAI_API_KEY=sk-your-actual-key-here     # Required
```

## 📊 Expected Outputs

### Semantic Alignment Analysis
```
Mean similarity:         > 0.87
Std deviation:           < 0.06
10th percentile:         > 0.82
Difficulty variance:     < 0.03
Bottom 100 review:       Mostly false positives
```

### File Sizes
```
vispider_embeddings.json:          ~55 MB
embeddings_for_training.npz:       ~24 MB
vispider_with_embeddings.json:     ~75 MB
embeddings_cache.json:             ~10 MB
```

## 🐛 Troubleshooting

### "File not found" errors
```bash
# Solution 1: Check working directory
pwd
cd /Users/hoadinh/Desktop/Code/ViSpider

# Solution 2: Verify files exist
python3 verify_setup.py
```

### "OPENAI_API_KEY not found"
```bash
# Solution: Create and configure .env
cp .env.example .env
nano .env  # or vim, code, etc.
# Add: OPENAI_API_KEY=sk-your-key
```

### Rate limit errors
```bash
# Wait a few minutes and rerun
# Embeddings are cached, so you won't lose progress
python3 semantic_alignment_analysis.py
```

### Out of memory
```bash
# Reduce batch size in semantic_alignment_analysis.py
# Change: BATCH_SIZE = 50  (default is 100)
```

## 📈 Workflow

### First Time Setup
```
1. Navigate to project directory
2. Create virtual environment
3. Install dependencies
4. Configure .env with API key
5. Verify setup
```

### Analysis Workflow
```
1. Run semantic_alignment_analysis.py
   → Generates embeddings and similarity scores
   
2. Review similarity_analysis.json
   → Check overall statistics
   
3. Run review_bottom_samples.py
   → Manual review of low-similarity samples
   
4. Export findings
   → Create review report
```

### Working with Embeddings
```
1. Load embeddings
   → embeddings_utils.py option 4
   
2. Export for ML
   → embeddings_utils.py option 1
   
3. Find similar questions
   → embeddings_utils.py option 3
   
4. Verify integrity
   → python3 test_embeddings.py
```

## 💡 Tips

### Speed up analysis
- Embeddings are cached - rerunning is fast
- Use `embeddings_cache.json` to avoid re-calling API
- Process smaller batches if hitting rate limits

### Save API costs
- Cache is persistent across runs
- Only new questions call the API
- Use `vispider_embeddings.json` instead of re-computing

### Review efficiency
- Use interactive mode in `review_bottom_samples.py`
- Export to markdown for team collaboration
- Focus on bottom 10% (systematic issues)

## 🔄 Common Tasks

### Re-run analysis without new API calls
```bash
# Embeddings are saved, just re-analyze
python3 semantic_alignment_analysis.py
```

### Export embeddings for ML model
```bash
python3 embeddings_utils.py
# Select option 1
```

### Find similar questions to a sample
```bash
python3 embeddings_utils.py
# Select option 3
# Enter sample index (0-1995)
```

### Check if embeddings are valid
```bash
python3 test_embeddings.py
```

### Generate review report
```bash
python3 review_bottom_samples.py
# Select option 3 - Export to markdown
```

## 📦 Package Versions

```
openai>=1.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## 🌐 Resources

- [SPIDER_OVERVIEW.md](SPIDER_OVERVIEW.md) - Dataset info
- [VISPIDER_2000_TRANSLATIONS.md](VISPIDER_2000_TRANSLATIONS.md) - Translation details
- [SEMANTIC_ALIGNMENT_README.md](SEMANTIC_ALIGNMENT_README.md) - Full analysis guide
- [EMBEDDINGS_FORMAT.md](EMBEDDINGS_FORMAT.md) - Embeddings documentation

---

**Quick Start**: `python3 verify_setup.py && ./run_analysis.sh`
