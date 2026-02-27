# Phase 2: GPT Expansion Workflow

## Overview

Phase 2 expands the dataset using GPT with few-shot prompting. This phase translates additional samples using diverse few-shot examples from the gold seed (Phase 1).

**Goal**: Expand the dataset using GPT with few-shot prompting, producing validated Vietnamese translations of Spider samples.

## Key Features

### 1. Pattern-Based Sampling
- Stratified sampling ensures coverage of all SQL patterns
- Excludes samples already in manual translations
- Maintains pattern distribution similar to gold seed

### 2. Intra-Class Diversity Selection
For each target sample:
- Select diverse few-shot examples with **same SQL pattern**
- Maximize diversity using a greedy algorithm based on LaBSE cosine distance

### 3. Automatic Tiered Retry
Each translation goes through up to 2 attempts automatically:
- **Attempt 1**: Standard prompt with diverse few-shot examples
- **Attempt 2**: Different examples + a hint reminding the model to map SQL operators to Vietnamese keywords
- If both attempts fail validation → log to `gpt_failed_samples.json` and continue to next sample

### 4. Checkpoint & Recovery
- Saves progress every 100 samples
- Resume from checkpoint if interrupted
- Detailed logging for debugging

## Workflow Steps

### Step 1: Sample Selection
**Script**: `scripts/phase3_chatgpt/01_select_samples_for_gpt.py`

**Purpose**: Select samples for GPT translation using stratified sampling by SQL pattern distribution, excluding samples already covered by manual translations.

**Output**: `data/chatgpt_translations/gpt_target_samples.json`

### Step 2: Translation with Validation
**Script**: `scripts/phase3_chatgpt/02_translate_with_validation.py`

**Purpose**: Translate samples with GPT and validate quality in real time.

**Process**:
1. For each target sample:
   - **Attempt 1**: Select diverse few-shot examples (same pattern), build prompt, call GPT, validate
   - If valid → save result and move on
   - If invalid → **Attempt 2**: select new examples (excluding attempt 1's), add operator hint, call GPT, validate
   - If still invalid → log to failed samples and move on
2. Save checkpoint periodically with automatic resume support
3. Generate validation report

**Outputs**:
- `data/chatgpt_translations/gpt_translations_final.json` - Successful translations
- `data/chatgpt_translations/gpt_translations_checkpoint_*.json` - Progress checkpoints
- `results/quality_analysis/gpt_validation_report.json` - Quality metrics
- `results/quality_analysis/gpt_failed_samples.json` - Samples that failed validation

## Configuration

### Environment Variables (.env)

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
GPT_MODEL=gpt-5-mini  # Reasoning model with low effort setting
```

### Models

Set the model via the `GPT_MODEL` environment variable in `.env`. The script automatically routes to the correct API based on whether the model is a reasoning model (`o1`, `o3`, `o4-mini`, etc.) or a standard chat model.

**Recommended**: Use a capable chat model (e.g., `gpt-4o-mini` for cost, `gpt-4o` for quality).

## Translation Prompt Structure

```
System: You are an expert Vietnamese translator.

User:
You are a professional translator specializing in technical text.
Translate English questions about SQL databases into natural Vietnamese.

RULES:
1. Preserve meaning and intent
2. Keep technical terms in English
3. Use natural Vietnamese phrasing
4. Maintain SQL operator semantics:
   - "how many" → "bao nhiêu" or "số lượng"
   - "maximum" → "lớn nhất" or "cao nhất"
   ...

Examples (3 few-shot examples with same SQL pattern, full context):
Database: [db_name]
English: [example English question]
SQL: [example SQL query]
Vietnamese: [example Vietnamese translation]

...

Now translate the following question:
Database: [target db]
English: [target question]
SQL: [target query]

Vietnamese:
```

## Validation Criteria

### LaBSE Similarity
- **Threshold**: Configurable minimum cosine similarity between English and Vietnamese embeddings
- Short questions use a lower threshold as cross-lingual embeddings are inherently less similar with limited context

### Operator Validation
SQL operators must have corresponding Vietnamese keywords — but only when the operator is semantically explicit in the English question:

- `COUNT` → bao nhiêu, số lượng, mấy, đếm
  - **Note**: Only checked when the English question explicitly asks "how many / count / number of". Questions like "which X is most common" use COUNT in SQL implicitly but do not require a count keyword in Vietnamese.
- `MAX` → lớn nhất, cao nhất, nhiều nhất, tối đa
- `MIN` → nhỏ nhất, thấp nhất, ít nhất, tối thiểu
- `AVG` → trung bình, bình quân
- `SUM` → tổng, tổng cộng
- `GREATER_THAN` → lớn hơn, cao hơn, nhiều hơn, trên
- `LESS_THAN` → nhỏ hơn, thấp hơn, ít hơn, dưới

## Usage

### Run Full Pipeline

```bash
cd ViSpider
source venv/bin/activate

# Step 1: Select samples
python3 scripts/phase3_chatgpt/01_select_samples_for_gpt.py

# Step 2: Translate with validation (auto-resumes from checkpoint if interrupted)
python3 scripts/phase3_chatgpt/02_translate_with_validation.py
```

### Resume from Checkpoint

Checkpoint recovery is automatic. If the script is interrupted, re-run it and it will scan for the latest checkpoint file in `data/chatgpt_translations/` and skip already-translated samples.

### Monitor Progress

The script outputs per-sample status: translation preview, LaBSE score, operator validation result, and which attempt succeeded or failed. Checkpoints are saved periodically.

## Troubleshooting

### API Key Error
```
❌ ERROR: OPENAI_API_KEY not found in .env file
```
**Solution**: Add your OpenAI API key to `.env` file

### Rate Limit Error
```
❌ GPT API error: Rate limit exceeded
```
**Solution**: 
- Wait a few minutes
- Increase `time.sleep()` delay in script
- Upgrade to higher API tier

### Low Success Rate
If many samples fail validation:
1. Check pattern: Are certain patterns failing more?
2. Review prompt: May need to adjust rules
3. Try different model: gpt-4o may perform better

### Checkpoint Recovery
Checkpoint recovery is handled automatically on restart. No manual intervention needed.
