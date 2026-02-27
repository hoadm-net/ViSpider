# ViSpider Quick Reference

## Project Structure

```
ViSpider/
├── data/
│   ├── raw/                          # Original Spider dataset
│   ├── extracted/                    # Simplified extraction of raw data
│   ├── manual_translations/          # Phase 1: Manual translations
│   ├── chatgpt_translations/         # Phase 2: GPT translations
│   ├── merged/                       # Phase 3.1: Train/dev/test split
│   └── model_translations/           # Phase 3+: Fine-tuned model output
│
├── scripts/
│   ├── phase0_prepare/               # Phase 0: Data extraction
│   ├── phase1_manual/                # Phase 1: Manual translation pipeline
│   ├── phase2_chatgpt/               # Phase 2: GPT translation
│   ├── phase3_finetune/              # Phase 3: Dataset assembly & model training
│   └── utils/                        # Shared utilities
│
├── models/                           # Trained model weights (gitignored)
├── results/                          # Analysis outputs (gitignored)
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
| `04_extract_low_quality.py` | Extract samples below quality threshold for review |
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
cd ViSpider
source venv/bin/activate

# Step 1: Select samples
python3 scripts/phase2_chatgpt/01_select_samples_for_gpt.py

# Step 2: Translate with validation (auto-loads gpt_target_samples.json)
python3 scripts/phase2_chatgpt/02_translate_with_validation.py
```

### Run Phase 3 Pipeline
```bash
cd ViSpider
source venv/bin/activate

# Step 1: Merge & split
python3 scripts/phase3_finetune/01_merge_and_split.py

# Step 2: Fine-tune (requires GPU)
python3 scripts/phase3_finetune/02_finetune.py

# Step 3: Merge adapter into standalone model
python3 scripts/phase3_finetune/03_merge_adapter.py

# Step 4: Evaluate on dev/test set
python3 scripts/phase3_finetune/04_evaluate.py --split dev
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
python3 05_filter_by_quality.py --threshold <value>

# Extract low-quality with custom cutoff
python3 04_extract_low_quality.py --threshold <value>
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
- `vispider_embeddings.json` - LaBSE embeddings
- `similarity_analysis.json` - Quality analysis results
- `vispider_low_quality_samples.json` - Flagged samples
- `vispider_train_filtered_75.json` - High-quality filtered set

### Phase 2 Output Files
- `gpt_target_samples.json` - Selected samples for translation
- `gpt_translations_final.json` - Successfully translated samples
- `gpt_translations_checkpoint_*.json` - Progress checkpoints
- `results/quality_analysis/gpt_validation_report.json` - Quality metrics
- `results/quality_analysis/gpt_failed_samples.json` - Failed samples

### Phase 3 Output Files
- `data/merged/vispider_all.json` - All merged samples
- `data/merged/vispider_train.json` - Training split
- `data/merged/vispider_dev.json` - Dev split
- `data/merged/vispider_test.json` - Test split
- `data/merged/split_report.json` - Split statistics by source and hardness
- `models/qwen25_vispider/final/` - Trained LoRA adapter (after fine-tuning)
- `models/qwen25_vispider_merged/` - Merged standalone model (after 03_merge_adapter.py)
- `results/quality_analysis/model_eval_dev.json` - Dev set evaluation results

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

## Troubleshooting

### Import Errors
```bash
# From scripts/phase1_manual/
# All paths are relative: ../../data/, ../../results/
```

### Memory Issues
```bash
# Reduce batch size in 02_compute_embeddings.py
# Edit BATCH_SIZE variable near the top of the script
```

### Corrupted Embeddings
```bash
# Re-generate embeddings
rm data/manual_translations/vispider_embeddings.json
python3 scripts/phase1_manual/02_compute_embeddings.py
```
