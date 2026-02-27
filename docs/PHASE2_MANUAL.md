# Phase 2: Manual Translation Workflow

## Overview

Phase 2 focuses on manually translating a representative subset of the Spider dataset into Vietnamese. This creates the gold seed for subsequent automated translation phases.

## Workflow Steps

### Step 1: Manual Translation
**Tool**: Label Studio (web-based annotation platform)

**Process**:
1. Load English questions from Spider train set
2. Display database schema for context
3. Translate question into Vietnamese
4. Review and validate translation
5. Export completed annotations

**Translation Guidelines**:
- Preserve semantic meaning
- Keep technical terms (table/column names) in English
- Consider database schema context
- Use natural Vietnamese phrasing
- Maintain query intent

### Step 2: Parse Label Studio Export
**Script**: `scripts/phase2_manual/01_parse_label_studio.py`

**Purpose**: Convert Label Studio JSON format to standard ViSpider format

**Output Format**:
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

### Step 3: Compute Embeddings
**Script**: `scripts/phase2_manual/02_compute_embeddings.py`

**Purpose**: Generate LaBSE embeddings for English and Vietnamese question pairs and compute cosine similarity for each.

### Step 3b: Extract SQL Patterns (Rule-Based Validation)
**Script**: `scripts/phase2_manual/02b_extract_sql_patterns.py`

**Purpose**: Extract SQL operators and patterns from queries for rule-based validation. Categorizes query complexity and validates SQL structure preservation.

### Step 4: Quality Analysis
**Script**: `scripts/phase2_manual/03_analyze_quality.py`

**Purpose**: Analyze translation quality distribution

**Outputs**:
- Similarity statistics and percentiles
- Quality bucket distribution
- Difficulty level breakdown
- Visualization plots

### Step 5: Extract Low-Quality Samples
**Script**: `scripts/phase2_manual/04_extract_low_quality.py`

**Purpose**: Identify samples below the quality threshold that need manual review.

### Step 6: Manual Review & Re-translation
**Script**: `scripts/phase2_manual/06_review_samples.py`

**Purpose**: Display problematic samples for manual correction

**Process**:
1. Review flagged samples
2. Correct translations in Label Studio
3. Re-export and re-run pipeline
4. Verify improved quality scores

### Step 7: Filter High-Quality Dataset
**Script**: `scripts/phase2_manual/05_filter_by_quality.py`

**Purpose**: Create filtered dataset with only high-quality translations

**Threshold**: Configurable similarity cutoff (see script defaults)

## File Locations

### Input
- `data/raw/train_spider.json` - Original Spider data
- `data/manual_translations/label_studio_2000_samples.json` - Label Studio export

### Output
- `data/manual_translations/vispider_train_2000.json` - Parsed ViSpider format
- `data/manual_translations/vispider_embeddings.json` - LaBSE embeddings
- `results/quality_analysis/similarity_analysis.json` - Quality metrics
- `results/quality_analysis/vispider_low_quality_samples.json` - Flagged samples

## Running the Pipeline

```bash
cd scripts/phase2_manual

# 1. Parse Label Studio export
python3 01_parse_label_studio.py

# 2. Compute embeddings and similarities
python3 02_compute_embeddings.py

# 3. Analyze quality distribution
python3 03_analyze_quality.py

# 4. Extract low-quality samples
python3 04_extract_low_quality.py

# 5. Review and correct (manual)
python3 06_review_samples.py

# 6. Filter high-quality dataset
python3 05_filter_by_quality.py
```

## Quality Criteria

**Acceptable Translation**:
- Preserves semantic intent
- Natural Vietnamese phrasing
- Correct understanding of schema context

**Common Issues**:
- Literal word-by-word translation (loses meaning)
- Missing context from database schema
- Translator notes instead of translation
- Copy-paste errors from ChatGPT conversations
