# ViSpider: Vietnamese Text-to-SQL Dataset

ViSpider is a Vietnamese translation of the [Spider dataset](https://yale-lily.github.io/spider), a large-scale text-to-SQL benchmark for semantic parsing. This project creates a high-quality Vietnamese text-to-SQL dataset through a systematic 7-step methodology combining human translation, GPT expansion, and fine-tuned model scaling.

## Project Goal

Translate the entire Spider dataset from English to Vietnamese while maintaining:
- Semantic accuracy via LaBSE embeddings
- SQL operator consistency via rule-based validation
- Query logic preservation across difficulty levels

## Methodology Overview

### Step 1: Gold Seed Construction

Manually translate a representative subset of Spider questions into Vietnamese via Label Studio. Translations are validated using LaBSE semantic similarity and rule-based SQL operator checks. Low-quality samples are flagged for review and re-translation.

**Scripts**: `scripts/phase1_manual/`  
**Output**: `data/manual_translations/vispider_train_2000.json`

### Step 2: GPT Expansion ✅
**Status**: Complete

Expand the dataset using GPT with few-shot prompting. Diverse examples with the same SQL pattern are selected from the gold seed as few-shot context. Each translation is validated in real time against LaBSE similarity and SQL operator consistency; failed samples are automatically retried with a different prompt. Progress is saved to checkpoints and automatically resumed on restart.

**Scripts**: `scripts/phase2_chatgpt/`  
**Output**: `data/chatgpt_translations/gpt_translations_final.json`

### Step 3: Dataset Assembly ✅
**Status**: Complete

Merge gold seed and GPT-translated data, deduplicate by sample ID (gold seed takes priority), then split into train/dev/test sets with stratified sampling by source and difficulty level.

**Script**: `scripts/phase3_finetune/01_merge_and_split.py`  
**Output**: `data/merged/` — unified train/dev/test split

### Step 4: Fine-tune Translation Model ✅
**Status**: Complete

Fine-tune Qwen2.5-7B-Instruct with QLoRA on the merged dataset for the task `(EN question + SQL + db_id) → VI question`. After training, merge the LoRA adapter into a standalone model.

**Scripts**: `scripts/phase3_finetune/02_finetune.py`, `03_merge_adapter.py`  
**Output**: `models/qwen25_vispider_merged/`

### Step 5: Pre-Scaling Evaluation 🔄
**Status**: Planned

Evaluate the fine-tuned model on the held-out dev and test sets using LaBSE similarity. This acts as a quality gate before full-scale translation.

**Script**: `scripts/phase3_finetune/04_evaluate.py`  
**Output**: `results/quality_analysis/`

### Step 6: Full Dataset Translation (Hybrid Scaling) 🔄
**Status**: Planned

Translate the remaining Spider samples using the fine-tuned model. Outputs that do not meet the quality threshold are automatically re-routed to GPT as a fallback.

**Scripts**: `scripts/phase3_finetune/` *(in development)*  
**Output**: `data/model_translations/`

### Step 7: Final Quality Control 🔄
**Status**: Planned

Validate the complete dataset: LaBSE similarity distribution, difficulty-level balance, and spot checks. Generate the final quality report.

**Output**: Production-ready ViSpider dataset

## Project Structure

```
ViSpider/
├── data/
│   ├── raw/                      # Original Spider dataset
│   ├── extracted/                # Simplified extraction of raw data
│   ├── manual_translations/      # Step 1: Gold seed (human)
│   ├── chatgpt_translations/     # Step 2: GPT expansion
│   ├── merged/                   # Step 3: Train/dev/test split
│   └── model_translations/       # Step 6: Hybrid scaling output
│
├── scripts/
│   ├── phase0_prepare/           # Step 0: Data extraction
│   ├── phase1_manual/            # Step 1: Manual translation pipeline
│   ├── phase2_chatgpt/           # Step 2: GPT translation
│   ├── phase3_finetune/          # Steps 3–7: Assembly, training & scaling
│   └── utils/                    # Shared utilities (LaBSE, embeddings)
│
├── models/                       # Trained model weights (gitignored)
├── results/                      # Analysis outputs (gitignored)
├── logs/                         # Training logs (gitignored)
└── docs/                         # Documentation
```

## Data Pipeline Flow

```
Step 1: Manual → Gold Seed
Step 2: GPT + Few-shot → Synthetic Data
Step 3: Gold + Synthetic → Training Dataset
Step 4: Train → Translation Model
Step 5: Evaluate → [GATE: Quality Check]
Step 6: Model + GPT Fallback → Full Dataset
Step 7: QC → Final ViSpider
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/hoadm-net/ViSpider.git
cd ViSpider

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Manual Translation Pipeline

```bash
# Phase 1: Parse and process manual translations
cd scripts/phase1_manual
python3 01_parse_label_studio.py
python3 02_compute_embeddings.py
python3 02b_extract_sql_patterns.py
python3 03_analyze_quality.py
python3 04_extract_low_quality.py
python3 05_filter_by_quality.py
python3 06_review_samples.py

# Phase 2: GPT expansion
cd ../phase2_chatgpt
python3 01_select_samples_for_gpt.py
python3 02_translate_with_validation.py

# Phase 3: Fine-tune and evaluate
cd ../phase3_finetune
python3 01_merge_and_split.py
python3 02_finetune.py
python3 03_merge_adapter.py
python3 04_evaluate.py
```

## Documentation

- [Quick Reference](docs/QUICK_REFERENCE.md) - All commands and troubleshooting
- [Phase 1 Manual Translation](docs/PHASE1_MANUAL.md) - Manual translation workflow
- [Phase 2 GPT Expansion](docs/PHASE2_CHATGPT.md) - GPT translation workflow
- [Phase 3 Fine-tuning](docs/PHASE3_FINETUNE.md) - Model training and evaluation
- [LaBSE Embeddings](docs/LABSE_EMBEDDINGS.md) - Quality assessment methodology
- [Spider Dataset Overview](docs/SPIDER_OVERVIEW.md) - Original dataset details

## Requirements

- Python 3.7+
- sentence-transformers
- numpy, scipy
- matplotlib
- tqdm

## Dataset Format

Each sample in ViSpider contains:

```json
{
  "id": "train-0",
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "vi_question": "Chúng ta có bao nhiêu ca sĩ?",
  "query": "SELECT count(*) FROM singer",
  "hardness": "easy",
  "sql_patterns": ["SELECT", "FROM", "COUNT"],
  "sql_complexity": "basic"
}
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any step of the pipeline.

## License

This project follows the original [Spider dataset license](https://yale-lily.github.io/spider). The Vietnamese translations are provided for research purposes.

## Citation

If you use ViSpider in your research, please cite:

```bibtex
@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3911--3921},
  year={2018}
}
```

## Contact

For questions or collaboration opportunities, please open an issue or submit a pull request.
