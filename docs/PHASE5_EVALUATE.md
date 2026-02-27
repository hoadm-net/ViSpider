# Phase 5: EN vs VI Text-to-SQL Evaluation

## Overview

Phase 5 measures whether fine-tuning Qwen2.5-Coder-7B-Instruct on **Vietnamese** questions produces competitive text-to-SQL performance compared to fine-tuning on the original **English** questions.

The pipeline has two independent tracks:

1. **Translation quality** — LaBSE-based semantic similarity between source and translated questions (Steps 1–2).
2. **Text-to-SQL accuracy** — Fine-tune → Predict → Execution Accuracy (EX) comparison for EN and VI (Steps 3–7).

---

## Prerequisites

```bash
pip install -r requirements.txt   # includes nltk, peft, trl, bitsandbytes
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

A CUDA-capable GPU is required for Steps 4 and 5.  
Spider's SQLite databases must be present at `data/raw/database/`.

---

## Workflow Steps

### Step 1: LaBSE Evaluation
**Script**: `scripts/phase5_evaluate/01_labse_eval.py`

**Purpose**: Evaluate translation quality using cross-lingual LaBSE cosine similarity in two modes:
- **Dataset quality** — measures EN ↔ VI semantic alignment across the merged dataset.
- **Model prediction** — measures EN ↔ PRED and REF ↔ PRED for a fine-tuned translation model's outputs.

**Inputs**:
- `data/merged/vispider_dev.json` (default)
- Any ViSpider JSON with `question` + `vi_question` fields

**Outputs** (`results/phase5_evaluate/`):
- `vispider_{split}_labse.json` — per-sample scores + aggregate statistics

**Run**:
```bash
# Evaluate dev split (dataset quality)
python3 scripts/phase5_evaluate/01_labse_eval.py

# Specific file
python3 scripts/phase5_evaluate/01_labse_eval.py \
    --input data/merged/vispider_test.json

# All three splits at once
python3 scripts/phase5_evaluate/01_labse_eval.py --all-splits

# Compare model predictions vs. reference
python3 scripts/phase5_evaluate/01_labse_eval.py \
    --pred-file results/quality_analysis/model_eval_dev.json
```

---

### Step 2: Visualize LaBSE Results
**Script**: `scripts/phase5_evaluate/02_visualize.py`

**Purpose**: Generate publication-quality plots from the JSON produced by Step 1.

Dataset-quality mode produces:
| File | Description |
|------|-------------|
| `01_score_distribution.png` | Histogram + KDE + threshold line |
| `02_cdf.png` | Cumulative distribution function |
| `03_boxplot_hardness.png` | Box plot grouped by hardness |
| `04_boxplot_source.png` | Box plot grouped by source |
| `05_passrate_bars.png` | Pass-rate bar chart by hardness & source |

Model-prediction mode additionally produces:
| File | Description |
|------|-------------|
| `06_scatter_en_vs_ref.png` | EN↔PRED vs REF↔PRED scatter |
| `07_delta_histogram.png` | Per-sample delta (EN↔PRED − REF↔PRED) |

**Run**:
```bash
# Visualize default dev-split LaBSE results
python3 scripts/phase5_evaluate/02_visualize.py

# Specify input file
python3 scripts/phase5_evaluate/02_visualize.py \
    --input results/phase5_evaluate/vispider_test_labse.json
```

---

### Step 3: Prepare Fine-tuning Data
**Script**: `scripts/phase5_evaluate/03_prepare_data.py`

**Purpose**: Convert ViSpider splits into instruction-tuning JSONL for Qwen2.5-Coder in both language variants. Each sample is formatted as a prompt containing the database schema (CREATE TABLE) and the question, with the gold SQL as the completion.

**Inputs**:
- `data/merged/vispider_{train,dev,test}.json`
- `data/raw/tables.json` (database schemas)

**Outputs**:
```
data/finetune_coder/
├── en/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── vi/
    ├── train.jsonl
    ├── dev.jsonl
    └── test.jsonl
```

Each line:
```json
{
  "id": "train-0001",
  "db_id": "concert_singer",
  "hardness": "easy",
  "source": "manual",
  "prompt": "### Database Schema\nCREATE TABLE ...\n\n### Question\nHow many singers are there?",
  "completion": "SELECT count(*) FROM singer"
}
```

**Run**:
```bash
# Prepare both EN and VI (default: full dataset)
python3 scripts/phase5_evaluate/03_prepare_data.py

# English only
python3 scripts/phase5_evaluate/03_prepare_data.py --lang en

# Smoke test: first 50 samples
python3 scripts/phase5_evaluate/03_prepare_data.py --max-samples 50

# Compact schema format
python3 scripts/phase5_evaluate/03_prepare_data.py --schema compact
```

---

### Step 4: Fine-tune Qwen2.5-Coder
**Script**: `scripts/phase5_evaluate/04_finetune_coder.py`

**Purpose**: Fine-tune `Qwen2.5-Coder-7B-Instruct` with **QLoRA** (4-bit nf4, LoRA r=16, α=32) on either the English or Vietnamese JSONL data from Step 3. Training uses trl's `SFTTrainer`.

**Inputs**:
- `data/finetune_coder/{lang}/train.jsonl`
- `data/finetune_coder/{lang}/dev.jsonl`

**Outputs**:
- `models/qwen25coder_vispider_{lang}/final/` — LoRA adapter weights
- `models/qwen25coder_vispider_{lang}/training_summary.json` — loss curve + hyperparameters

**Run**:
```bash
# Fine-tune English (recommended: --epochs 5 for full run)
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang en --epochs 5

# Fine-tune Vietnamese
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang vi --epochs 5

# Smoke test (fast, 5 steps)
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang en --max-steps 5

# Custom hyperparameters
python3 scripts/phase5_evaluate/04_finetune_coder.py \
    --lang vi --epochs 3 --lr 2e-4 --batch-size 4

# Resume from checkpoint
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang en --resume
```

**GPU requirement**: ~5 GB VRAM with 4-bit quantization (default). Pass `--no-quantize` for full bf16 (requires ~16 GB).

---

### Step 5: Predict SQL
**Script**: `scripts/phase5_evaluate/05_predict_sql.py`

**Purpose**: Run inference with a fine-tuned (or base zero-shot) Qwen2.5-Coder model and save the predicted SQL for each sample.

**Inputs**:
- `data/finetune_coder/{lang}/{split}.jsonl`
- `models/qwen25coder_vispider_{lang}/final/` — LoRA adapter (unless `--zero-shot`)

**Outputs** (`results/phase5_evaluate/`):
- `predictions_{lang}_{split}.json` — per-sample predictions with gold SQL, predicted SQL, metadata

**Run**:
```bash
# Predict with fine-tuned EN model on test split (default)
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en

# Vietnamese model, dev split
python3 scripts/phase5_evaluate/05_predict_sql.py --lang vi --split dev

# Zero-shot baseline (no fine-tuning, base model only)
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en --zero-shot

# Quick smoke test: first 20 samples
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en --n 20
```

> **Note**: The model is loaded in 4-bit (BitsAndBytesConfig) to fit within GPU memory. The LoRA adapter is applied on top of the quantized base model.

---

### Step 6: Execution Accuracy (EX)
**Script**: `scripts/phase5_evaluate/06_execution_accuracy.py`

**Purpose**: Execute both gold and predicted SQL against Spider's SQLite databases and compute **Execution Accuracy** — the fraction of predictions whose result set matches the gold result set exactly.

Uses Spider's official `eval_exec_match()` function (column-mapping comparison, not simple frozenset), with a normalised frozenset fallback for edge cases.

**Inputs**:
- `results/phase5_evaluate/predictions_{lang}_{split}.json`
- `data/raw/database/{db_id}/{db_id}.sqlite`

**Outputs** (`results/phase5_evaluate/`):
- `predictions_{lang}_{split}_ex.json` — per-sample EX score + aggregate breakdown by hardness and source

**Run**:
```bash
# English test split
python3 scripts/phase5_evaluate/06_execution_accuracy.py \
    --pred-file results/phase5_evaluate/predictions_en_test.json

# Vietnamese test split
python3 scripts/phase5_evaluate/06_execution_accuracy.py \
    --pred-file results/phase5_evaluate/predictions_vi_test.json

# Custom database directory
python3 scripts/phase5_evaluate/06_execution_accuracy.py \
    --pred-file results/phase5_evaluate/predictions_en_test.json \
    --db-dir data/raw/database
```

> **EX metric**: A prediction counts as correct if its result set is semantically equivalent to the gold result set (column-aware). Gold SQL errors are excluded from the denominator; prediction errors count as wrong.

---

### Step 7: Compare EN vs VI
**Script**: `scripts/phase5_evaluate/07_compare_en_vi.py`

**Purpose**: Side-by-side comparison of English and Vietnamese EX scores, with breakdown by hardness level and source. Optionally generates comparison plots.

**Inputs**:
- `results/phase5_evaluate/predictions_en_{split}_ex.json`
- `results/phase5_evaluate/predictions_vi_{split}_ex.json`

**Outputs**:
- `results/phase5_evaluate/comparison_en_vi.json` — structured comparison table
- `results/phase5_evaluate/plots/comparison_en_vi/` — 3 PNG plots (if `--plot`)

| Plot | Description |
|------|-------------|
| `01_overall_ex.png` | EN vs VI overall EX bar chart |
| `02_ex_by_hardness.png` | Grouped bars by hardness level |
| `03_gap_by_hardness.png` | EN−VI gap by hardness |

**Run**:
```bash
# Compare test split results
python3 scripts/phase5_evaluate/07_compare_en_vi.py \
    --en results/phase5_evaluate/predictions_en_test_ex.json \
    --vi results/phase5_evaluate/predictions_vi_test_ex.json

# With plots
python3 scripts/phase5_evaluate/07_compare_en_vi.py \
    --en results/phase5_evaluate/predictions_en_test_ex.json \
    --vi results/phase5_evaluate/predictions_vi_test_ex.json \
    --plot
```

---

## Running the Full Pipeline

```bash
# 1. Prepare fine-tuning data (both languages)
python3 scripts/phase5_evaluate/03_prepare_data.py

# 2. Fine-tune EN and VI models
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang en --epochs 5
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang vi --epochs 5

# 3. Predict on test split
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en --split test
python3 scripts/phase5_evaluate/05_predict_sql.py --lang vi --split test

# 4. Compute EX for both
python3 scripts/phase5_evaluate/06_execution_accuracy.py \
    --pred-file results/phase5_evaluate/predictions_en_test.json
python3 scripts/phase5_evaluate/06_execution_accuracy.py \
    --pred-file results/phase5_evaluate/predictions_vi_test.json

# 5. Compare and visualize
python3 scripts/phase5_evaluate/07_compare_en_vi.py \
    --en results/phase5_evaluate/predictions_en_test_ex.json \
    --vi results/phase5_evaluate/predictions_vi_test_ex.json \
    --plot
```

---

## Shared Utilities

### `scripts/utils/schema_utils.py`
Builds `CREATE TABLE` strings from Spider's `tables.json` for use in prompts.

```python
from schema_utils import load_tables, build_schema, build_schema_compact

tables = load_tables("data/raw/tables.json")
schema_str = build_schema("concert_singer", tables)        # full CREATE TABLE + FK
compact    = build_schema_compact("concert_singer", tables) # one-liner per table
```

### `scripts/utils/spider_eval/`
Spider's official evaluation code (`evaluation.py`, `process_sql.py`) vendored from [taoyds/spider](https://github.com/taoyds/spider). Used by `06_execution_accuracy.py` for `eval_exec_match()`.

> **Note on official Spider metric**: Since November 2020, the official Spider leaderboard metric is **Test Suite Accuracy** ([taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)), which evaluates SQL programs against multiple database instances. This project reports the simpler **Execution Accuracy (EX)** which remains widely used in the literature.

---

## Output File Formats

### LaBSE results (`vispider_{split}_labse.json`)
```json
{
  "mode": "dataset_quality",
  "stats": {
    "n": 203,
    "mean": 0.7814,
    "median": 0.8021,
    "pass_rate_75": 0.700
  },
  "results": [
    {
      "id": "dev-0001",
      "db_id": "concert_singer",
      "question": "...",
      "vi_question": "...",
      "sim": 0.832
    }
  ]
}
```

### Prediction file (`predictions_{lang}_{split}.json`)
```json
[
  {
    "id": "test-0001",
    "db_id": "concert_singer",
    "lang": "en",
    "split": "test",
    "mode": "fine-tuned",
    "question": "How many singers are there?",
    "gold_sql": "SELECT count(*) FROM singer",
    "pred_sql": "SELECT count(*) FROM singer",
    "hardness": "easy",
    "source": "manual"
  }
]
```

### EX results (`predictions_{lang}_{split}_ex.json`)
```json
{
  "stats": {
    "lang": "en",
    "split": "test",
    "total_samples": 203,
    "evaluated": 203,
    "correct": 142,
    "execution_accuracy": 0.6995,
    "by_hardness": {
      "easy":       {"n": 60, "correct": 52, "ex_pct": 86.7},
      "medium":     {"n": 80, "correct": 55, "ex_pct": 68.8},
      "hard":       {"n": 40, "correct": 25, "ex_pct": 62.5},
      "extra_hard": {"n": 23, "correct":  9, "ex_pct": 39.1}
    }
  }
}
```
