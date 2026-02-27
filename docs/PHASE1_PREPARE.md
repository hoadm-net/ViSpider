# Phase 1: Data Preparation

## Overview

Phase 1 extracts the raw Spider dataset into a simplified format suitable for translation workflows. This is the prerequisite step before any translation phase.

## Workflow Steps

### Step 1: Extract Spider Data
**Script**: `scripts/phase1_prepare/01_extract_spider_data.py`

**Purpose**: Convert raw Spider JSON format into a simplified 4-field format, removing metadata fields not needed for translation.

**Input Files**:
- `data/raw/train_spider.json` - Spider in-domain training examples
- `data/raw/train_others.json` - Spider cross-domain training examples
- `data/raw/dev.json` - Spider development set
- `data/raw/test.json` - Spider test set

**Output Format**:
```json
{
  "id": "train-0001",
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "query": "SELECT count(*) FROM singer"
}
```

**Output Files**:
- `data/extracted/train.json` - Combined training set (train_spider + train_others)
- `data/extracted/dev.json` - Development set
- `data/extracted/test.json` - Test set

## Running the Script

```bash
cd ViSpider
source venv/bin/activate
python3 scripts/phase1_prepare/01_extract_spider_data.py
```

## Output

The extracted files in `data/extracted/` serve as input for all downstream translation phases:

- **Phase 2**: Manual translation workflow samples are drawn from the extracted training set
- **Phase 3**: GPT few-shot prompting uses extracted samples for context
- **Phase 4**: Merge and split script reads the extracted dev/test splits

## Notes

- The training set combines `train_spider.json` (in-domain) and `train_others.json` (cross-domain)
- Field `id` is synthesized as `{split}-{index}` since Spider's original format uses positional indexing
- Raw Spider fields such as `sql`, `query_toks`, `query_toks_no_value`, `question_toks` are discarded
