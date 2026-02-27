# ViSpider: Vietnamese Text-to-SQL Dataset

ViSpider is a Vietnamese translation of the [Spider dataset](https://yale-lily.github.io/spider), a large-scale text-to-SQL benchmark for semantic parsing. The project builds a high-quality Vietnamese text-to-SQL dataset through a systematic pipeline combining human translation, GPT expansion, model fine-tuning, and downstream evaluation.

## Project Goal

Translate the entire Spider dataset from English to Vietnamese while maintaining:
- Semantic accuracy via LaBSE embeddings
- SQL operator consistency via rule-based validation
- Query logic preservation across difficulty levels

## Methodology

### Phase 1: Data Preparation

Extract and simplify the raw Spider dataset into a clean format for downstream translation phases.

**Script**: `scripts/phase1_prepare/01_extract_spider_data.py`  
**Output**: `data/extracted/` — simplified train/dev/test JSON files

---

### Phase 2: Gold Seed Construction (Manual Translation)

Manually translate a representative subset of Spider questions into Vietnamese via Label Studio. Translations are validated using LaBSE semantic similarity and rule-based SQL operator checks. Low-quality samples are flagged for review and re-translation.

**Scripts**: `scripts/phase2_manual/`  
**Output**: `data/manual_translations/vispider_train_2000.json`

---

### Phase 3: GPT Expansion

Expand the dataset using GPT with few-shot prompting. Diverse examples with the same SQL pattern are selected from the gold seed as few-shot context. Each translation is validated in real time against LaBSE similarity and SQL operator consistency; failed samples are automatically retried with a different prompt. Progress is saved to checkpoints and automatically resumed on restart.

**Scripts**: `scripts/phase3_chatgpt/`  
**Output**: `data/chatgpt_translations/gpt_translations_final.json`

---

### Phase 4: Dataset Assembly & Translation Model Fine-tuning

Merge gold seed and GPT-translated data, deduplicate by sample ID (gold seed takes priority), then split into train/dev/test sets with stratified sampling by source and difficulty level. Fine-tune Qwen2.5-7B-Instruct with QLoRA on the merged dataset for the task `(EN question + SQL + db_id) → VI question`.

**Scripts**: `scripts/phase4_finetune/`  
**Outputs**:
- `data/merged/` — unified train/dev/test split
- `models/qwen25_vispider/final/` — LoRA adapter
- `models/qwen25_vispider_merged/` — standalone merged model

---

### Phase 5: EN vs VI Text-to-SQL Evaluation

Fine-tune Qwen2.5-Coder-7B-Instruct separately on English and Vietnamese ViSpider data. Compare downstream text-to-SQL performance (Execution Accuracy) between the two language variants, with breakdown by difficulty level. Also includes LaBSE-based translation quality analysis.

**Scripts**: `scripts/phase5_evaluate/`  
**Outputs**: `results/phase5_evaluate/` — LaBSE scores, predictions, EX metrics, comparison plots

## Project Structure

```
ViSpider/
├── data/
│   ├── raw/                          # Original Spider dataset + SQLite databases
│   ├── extracted/                    # Phase 1: Simplified extraction
│   ├── manual_translations/          # Phase 2: Gold seed (human)
│   ├── chatgpt_translations/         # Phase 3: GPT expansion
│   ├── merged/                       # Phase 4: Train/dev/test split
│   └── finetune_coder/               # Phase 5: Instruction-tuning JSONL (generated)
│
├── scripts/
│   ├── phase1_prepare/               # Phase 1: Data extraction
│   ├── phase2_manual/                # Phase 2: Manual translation pipeline
│   ├── phase3_chatgpt/               # Phase 3: GPT translation
│   ├── phase4_finetune/              # Phase 4: Dataset assembly & model training
│   ├── phase5_evaluate/              # Phase 5: LaBSE eval + EN/VI text-to-SQL
│   └── utils/                        # Shared utilities (schema, LaBSE, Spider eval)
│
├── models/                           # Trained model weights (gitignored)
├── results/                          # Analysis outputs (gitignored)
└── docs/                             # Documentation
```

## Quick Start

### Installation

```bash
git clone https://github.com/hoadm-net/ViSpider.git
cd ViSpider
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Phase 1: Extract Spider Data

```bash
python3 scripts/phase1_prepare/01_extract_spider_data.py
```

### Phase 2: Manual Translation Pipeline

```bash
python3 scripts/phase2_manual/01_parse_label_studio.py
python3 scripts/phase2_manual/02_compute_embeddings.py
python3 scripts/phase2_manual/02b_extract_sql_patterns.py
python3 scripts/phase2_manual/03_analyze_quality.py
python3 scripts/phase2_manual/04_extract_low_quality.py
python3 scripts/phase2_manual/05_filter_by_quality.py
python3 scripts/phase2_manual/06_review_samples.py
```

### Phase 3: GPT Expansion

```bash
python3 scripts/phase3_chatgpt/01_select_samples_for_gpt.py
python3 scripts/phase3_chatgpt/02_translate_with_validation.py
```

### Phase 4: Dataset Assembly & Fine-tuning

```bash
python3 scripts/phase4_finetune/01_merge_and_split.py
python3 scripts/phase4_finetune/02_finetune.py
python3 scripts/phase4_finetune/03_merge_adapter.py
python3 scripts/phase4_finetune/04_evaluate.py --split dev
```

### Phase 5: EN vs VI Evaluation

```bash
# Prepare instruction-tuning JSONL
python3 scripts/phase5_evaluate/03_prepare_data.py

# Fine-tune Qwen2.5-Coder separately for EN and VI
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang en --epochs 5
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang vi --epochs 5

# Predict SQL on test split
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en --split test
python3 scripts/phase5_evaluate/05_predict_sql.py --lang vi --split test

# Compute Execution Accuracy
python3 scripts/phase5_evaluate/06_execution_accuracy.py \
    --pred-file results/phase5_evaluate/predictions_en_test.json
python3 scripts/phase5_evaluate/06_execution_accuracy.py \
    --pred-file results/phase5_evaluate/predictions_vi_test.json

# Compare and generate plots
python3 scripts/phase5_evaluate/07_compare_en_vi.py \
    --en results/phase5_evaluate/predictions_en_test_ex.json \
    --vi results/phase5_evaluate/predictions_vi_test_ex.json \
    --plot
```

## Documentation

| Document | Description |
|----------|-------------|
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | All commands and troubleshooting |
| [PHASE1_PREPARE.md](docs/PHASE1_PREPARE.md) | Spider data extraction |
| [PHASE2_MANUAL.md](docs/PHASE2_MANUAL.md) | Manual translation workflow |
| [PHASE3_CHATGPT.md](docs/PHASE3_CHATGPT.md) | GPT translation workflow |
| [PHASE4_FINETUNE.md](docs/PHASE4_FINETUNE.md) | Dataset assembly & model training |
| [PHASE5_EVALUATE.md](docs/PHASE5_EVALUATE.md) | EN vs VI text-to-SQL evaluation |
| [LABSE_EMBEDDINGS.md](docs/LABSE_EMBEDDINGS.md) | LaBSE quality assessment methodology |
| [SPIDER_OVERVIEW.md](docs/SPIDER_OVERVIEW.md) | Original Spider dataset details |

## Dataset Format

Each sample in ViSpider contains:

```json
{
  "id": "train-0001",
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "vi_question": "Chúng ta có bao nhiêu ca sĩ?",
  "query": "SELECT count(*) FROM singer",
  "hardness": "easy",
  "sql_patterns": ["SELECT", "FROM", "COUNT"],
  "source": "manual"
}
```

The `source` field is either `"manual"` (Phase 2) or `"gpt"` (Phase 3).

## Requirements

- Python 3.10+
- GPU with CUDA (Phases 4 and 5)
- See `requirements.txt` for full dependency list

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project follows the original [Spider dataset license](https://yale-lily.github.io/spider). The Vietnamese translations are provided for research purposes.

