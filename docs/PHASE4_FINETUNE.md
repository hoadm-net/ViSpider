# Phase 3: Fine-tuning Workflow

## Overview

Phase 3 merges the gold seed and GPT-translated data into a unified dataset, fine-tunes a translation model on it, and evaluates translation quality.

## Workflow Steps

### Step 1: Merge & Split Dataset
**Script**: `scripts/phase4_finetune/01_merge_and_split.py`

**Purpose**: Merge Phase 1 (manual) and Phase 2 (GPT) translations into a single dataset, then split into train/dev/test with stratified sampling by source and difficulty level. In case of duplicate IDs, the manual translation takes priority.

**Inputs**:
- `data/manual_translations/vispider_train_2000.json`
- `data/chatgpt_translations/gpt_translations_final.json`

**Outputs** (`data/merged/`):
- `vispider_all.json` — full merged dataset
- `vispider_train.json` — training split
- `vispider_dev.json` — development split
- `vispider_test.json` — test split
- `split_report.json` — split statistics by source and hardness

**Run**:
```bash
python3 scripts/phase4_finetune/01_merge_and_split.py
```

---

### Step 2: Fine-tune Translation Model
**Script**: `scripts/phase4_finetune/02_finetune.py`

**Purpose**: Fine-tune Qwen2.5-7B-Instruct with QLoRA for the task `(EN question + SQL query + db_id) → VI question`. Uses 4-bit quantization (QLoRA) by default to fit within GPU memory constraints.

**Inputs**:
- `data/merged/vispider_train.json`
- `data/merged/vispider_dev.json`

**Output**: `models/qwen25_vispider/final/` — LoRA adapter weights

**Run**:
```bash
# QLoRA 4-bit (default, recommended)
python3 scripts/phase4_finetune/02_finetune.py

# Full precision (higher VRAM requirement)
python3 scripts/phase4_finetune/02_finetune.py --no-quantize

# Custom hyperparameters
python3 scripts/phase4_finetune/02_finetune.py --epochs 3 --lr 2e-4 --batch-size 4

# Resume from checkpoint
python3 scripts/phase4_finetune/02_finetune.py --resume

# Smoke test (limited steps)
python3 scripts/phase4_finetune/02_finetune.py --max-steps 10
```

**Requirements**: GPU with sufficient VRAM. Install extra dependencies first:
```bash
pip install peft trl bitsandbytes accelerate
```

---

### Step 3: Merge Adapter into Standalone Model
**Script**: `scripts/phase4_finetune/03_merge_adapter.py`

**Purpose**: Merge the LoRA adapter weights into the base model to produce a standalone model that can be loaded without PEFT.

**Input**: `models/qwen25_vispider/final/` — adapter directory

**Output**: `models/qwen25_vispider_merged/` — standalone model

**Run**:
```bash
# Default paths (reads adapter from models/qwen25_vispider/final/)
python3 scripts/phase4_finetune/03_merge_adapter.py

# Custom paths
python3 scripts/phase4_finetune/03_merge_adapter.py \
  --adapter models/qwen25_vispider/final \
  --output models/qwen25_vispider_merged
```

---

### Step 4: Evaluate Translation Quality
**Script**: `scripts/phase4_finetune/04_evaluate.py`

**Purpose**: Run inference on the dev or test split and compute LaBSE similarity scores between model outputs and reference Vietnamese translations.

**Inputs**:
- `data/merged/vispider_dev.json` or `data/merged/vispider_test.json`
- Trained model (adapter or merged)

**Output**: `results/quality_analysis/model_eval_{split}.json`

**Run**:
```bash
# Evaluate on dev set (default, uses adapter)
python3 scripts/phase4_finetune/04_evaluate.py

# Evaluate on test set
python3 scripts/phase4_finetune/04_evaluate.py --split test

# Use merged standalone model
python3 scripts/phase4_finetune/04_evaluate.py --model models/qwen25_vispider_merged

# Quick smoke-test on first N samples
python3 scripts/phase4_finetune/04_evaluate.py --n 50
```

---

## Running the Full Pipeline

```bash
cd ViSpider
source venv/bin/activate

python3 scripts/phase4_finetune/01_merge_and_split.py
python3 scripts/phase4_finetune/02_finetune.py
python3 scripts/phase4_finetune/03_merge_adapter.py
python3 scripts/phase4_finetune/04_evaluate.py --split dev
```

## Dataset Format

Each sample in the merged dataset:
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

The `source` field is either `"manual"` (Phase 1) or `"gpt"` (Phase 2).
