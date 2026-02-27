# ViSpider Quick Reference

## Project Structure

```
ViSpider/
├── data/
│   ├── raw/                          # Original Spider dataset
│   ├── manual_translations/          # Phase 1: Manual translations
│   ├── chatgpt_translations/         # Phase 2: GPT translations
│   └── model_translations/           # Phase 3: Fine-tuned model (planned)
│
├── scripts/
│   ├── phase0_prepare/               # Phase 0: Data extraction
│   ├── phase1_manual/                # Phase 1: Manual translation pipeline
│   ├── phase2_chatgpt/               # Phase 2: GPT translation
│   ├── phase3_finetune/              # Phase 3: Model training (planned)
│   └── utils/                        # Shared utilities
│
├── results/                          # Analysis outputs (gitignored)
│   ├── embeddings/
│   ├── quality_analysis/
│   └── comparisons/
│
├── experiments/                      # Experimental code (gitignored)
│
└── docs/                             # Documentation
```

## Phase 0 Scripts

Data extraction from raw Spider format:

| Script | Purpose |
|--------|---------|  
| `00_extract_spider_data.py` | Extract and simplify Spider data (raw → extracted) |

## Phase 1 Scripts

Run from `scripts/phase1_manual/` directory:

| Script | Purpose |
|--------|---------|

| `01_parse_label_studio.py` | Parse Label Studio export to ViSpider format |
| `02_compute_embeddings.py` | Generate LaBSE embeddings and similarities |
| `02b_extract_sql_patterns.py` | Extract SQL patterns (rule-based validation) |
| `03_analyze_quality.py` | Analyze quality distribution |
| `04_extract_low_quality.py` | Extract samples needing review (< 0.75) |
| `05_filter_by_quality.py` | Create filtered high-quality dataset |
| `06_review_samples.py` | Display samples for manual review |

## Phase 2 Scripts

Run from `scripts/phase2_chatgpt/` directory:

| Script | Purpose |
|--------|---------|  
| `01_select_samples_for_gpt.py` | Select N samples with pattern coverage (`-n` flag, default 3000) |
| `02_translate_with_validation.py` | Translate with GPT + real-time validation |

## Common Commands

### Run Phase 2 Pipeline
```bash
cd /home/hoadm/ViSpider
source venv/bin/activate

# Step 1: Select samples (adjust -n as needed)
python3 scripts/phase2_chatgpt/01_select_samples_for_gpt.py -n 3000

# Step 2: Translate with validation (~7-8 sec/sample)
python3 scripts/phase2_chatgpt/02_translate_with_validation.py
```

### Run Full Phase 1 Pipeline
```bash
cd scripts/phase1_manual
python3 01_parse_label_studio.py
python3 02_compute_embeddings.py
python3 02b_extract_sql_patterns.py
python3 03_analyze_quality.py
python3 04_extract_low_quality.py
```

### Custom Thresholds
```bash
# Filter with custom threshold
python3 05_filter_by_quality.py --threshold 0.80

# Extract low-quality with custom cutoff
python3 04_extract_low_quality.py --threshold 0.70
```

### Review Specific Samples
```bash
# Review bottom N samples
python3 06_review_samples.py --count 50

# Review specific difficulty
python3 06_review_samples.py --difficulty hard
```

## Data Files

### Input Files
- `data/raw/train_spider.json` - Original Spider training data
- `data/raw/dev.json` - Spider dev set (for future translation)
- `data/raw/test.json` - Spider test set (for future translation)
- `data/raw/tables.json` - Database schemas

### Phase 1 Output Files
- `vispider_train_2000.json` - Parsed manual translations
- `vispider_embeddings.json` - LaBSE embeddings (~25 MB)
- `similarity_analysis.json` - Quality analysis results
- `vispider_low_quality_samples.json` - Flagged samples
- `vispider_train_filtered_75.json` - High-quality filtered set

### Phase 2 Output Files
- `gpt_target_samples.json` - Selected samples for translation
- `gpt_translations_final.json` - Successfully translated samples
- `gpt_translations_checkpoint_*.json` - Progress checkpoints
- `results/quality_analysis/gpt_validation_report.json` - Quality metrics
- `results/quality_analysis/gpt_failed_samples.json` - Failed samples

## Quality Interpretation

| Similarity Range | Quality Level | Action |
|-----------------|---------------|--------|
| >= 0.85 | Excellent | Accept |
| 0.75 - 0.85 | Good | Accept |
| 0.65 - 0.75 | Acceptable | Review |
| 0.50 - 0.65 | Poor | Re-translate |
| < 0.50 | Very Poor | Re-translate |

## Environment Setup

### Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Required Packages
- `sentence-transformers` - LaBSE embeddings
- `numpy` - Numerical operations
- `scipy` - Statistical functions
- `matplotlib` - Visualization
- `tqdm` - Progress bars

### GPU vs CPU
- LaBSE works on both CPU and GPU
- CPU: ~4 minutes for 2,000 samples
- GPU: ~30 seconds for 2,000 samples

## Troubleshooting

### Import Errors
```bash
# From scripts/phase1_manual/
# All paths are relative: ../../data/, ../../results/
```

### Memory Issues
```bash
# Reduce batch size in 02_compute_embeddings.py
BATCH_SIZE = 16  # Default is 32
```

### Corrupted Embeddings
```bash
# Re-generate embeddings
rm ../../data/manual_translations/vispider_embeddings.json
python3 02_compute_embeddings.py
```

## Next Steps

After Phase 1:
1. **Phase 2**: Use ChatGPT to translate 3,000 more samples
2. **Phase 3**: Fine-tune 3-7B parameter model for remaining translations
3. **Validation**: Run quality analysis on all phases
4. **Publication**: Release complete ViSpider dataset
