# Phase 2: GPT Expansion Workflow

## Overview

Phase 2 expands the dataset using GPT with few-shot prompting. This phase translates additional samples using diverse few-shot examples from the gold seed (Phase 1).

**Goal**: Produce GPT-translated samples with sufficient LaBSE semantic similarity and valid SQL operator keywords.

## Key Features

### 1. Pattern-Based Sampling
- Stratified sampling ensures coverage of all SQL patterns
- Excludes samples already in manual translations
- Maintains pattern distribution similar to gold seed

### 2. Intra-Class Diversity Selection
For each target sample:
- Select N diverse few-shot examples with **same SQL pattern** (N depends on attempt)
- Maximize diversity using greedy algorithm: each new example maximizes minimum cosine distance to all already-selected examples
- Distance metric: Cosine distance on LaBSE embeddings

### 3. Automatic Tiered Retry
Each translation goes through up to 2 attempts automatically:
- **Attempt 1**: 3 diverse examples, standard prompt
- **Attempt 2**: 5 completely new examples (the 3 from attempt 1 are excluded) + a literal-translation hint reminding the model to map SQL operators to Vietnamese keywords
- If both attempts fail validation → log to `gpt_failed_samples.json` and continue to next sample

### 4. Checkpoint & Recovery
- Saves progress every 100 samples
- Resume from checkpoint if interrupted
- Detailed logging for debugging

## Workflow Steps

### Step 1: Sample Selection
**Script**: `scripts/phase2_chatgpt/01_select_samples_for_gpt.py`

**Purpose**: Select 3,000 samples for GPT translation

**Strategy**:
1. Load extracted Spider data (~8,659 samples)
2. Exclude manual translation IDs (1,996 samples)
3. Stratified sampling by SQL pattern distribution
4. Ensure coverage of underrepresented patterns

**Output**: `data/chatgpt_translations/gpt_target_samples.json`

### Step 2: Translation with Validation
**Script**: `scripts/phase2_chatgpt/02_translate_with_validation.py`

**Purpose**: Translate samples with GPT and validate quality

**Process**:
1. For each target sample:
   - **Attempt 1**: Select 3 diverse few-shot examples (same pattern), build standard prompt, call GPT, validate
   - If valid → save result and move on
   - If invalid → **Attempt 2**: select 5 completely new examples (excluding attempt 1's examples), add operator hint to prompt, call GPT, validate
   - If still invalid → log to failed samples and move on
2. Save checkpoint every 100 samples
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

### API Configuration

**New Responses API** (for reasoning models like gpt-5-mini):
- Uses `client.responses.create()` instead of `chat.completions.create()`
- Supports `reasoning.effort` parameter: `"low"`, `"medium"`, `"high"`
- Setting `effort: "low"` reduces reasoning token usage
- Format: `input` array with `input_text` type

**Benefits**:
- Reduced reasoning tokens (less cost)
- Better performance for translation tasks
- Consistent quality with lower effort setting

### Models Comparison

| Model | Speed | Cost | Quality | Reasoning Tokens |
|-------|-------|------|---------|------------------|
| gpt-5-mini (effort: low) | Fast | Low | Excellent | Minimal |
| gpt-4o-mini | Fast | Low | Good | None |
| gpt-4o | Medium | Medium | Excellent | None |
| gpt-4 | Slow | High | Excellent | None |

**Recommended**: `gpt-5-mini` with `reasoning.effort: "low"` for best quality/cost balance

### Rate Limits

Script includes rate limiting (0.5s delay between requests = 2 req/sec):
- **Tier 1**: 500 RPM, 200K TPM
- **Tier 2**: 5,000 RPM, 2M TPM
- **Tier 3**: 10,000 RPM, 10M TPM

Adjust `time.sleep()` in script if you have higher tier.

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
- **Threshold**: >= 0.75 for normal questions; >= 0.70 for short questions (≤ 7 words)
  - Short questions contain less context, making cross-lingual embeddings inherently less similar
- **Metric**: Cosine similarity between English and Vietnamese embeddings
- **Purpose**: Ensures semantic preservation

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
cd /home/hoadm/ViSpider

# Activate virtual environment
source venv/bin/activate

# Step 1: Select samples (use -n to control count, default 3000)
python3 scripts/phase2_chatgpt/01_select_samples_for_gpt.py -n 3000

# Step 2: Translate with validation (auto-loads gpt_target_samples.json)
python3 scripts/phase2_chatgpt/02_translate_with_validation.py
```

### Resume from Checkpoint

If script is interrupted:
1. Check latest checkpoint: `data/chatgpt_translations/gpt_translations_checkpoint_*.json`
2. Modify script to skip already translated samples
3. Re-run script

### Monitor Progress

Script outputs real-time per-sample:
- Translation preview
- LaBSE score and operator validation status
- Which attempt succeeded or failed
- Progress stats and ETA every 100 samples (checkpoints)

Example output:
```
[1/3000] train-0001
  Translation: Có bao nhiêu ca sĩ trong cơ sở dữ liệu?
  LaBSE: 0.8234 | Operators: ✓ | Attempt 1 (3 examples)
  ✅ Success (attempt 1)

[5/3000] train-0082
  Translation: ...
  LaBSE: 0.6812 | Operators: ✓ | Attempt 1 (3 examples)
  ⚠️  Attempt 1: Validation failed (LaBSE=0.6812, operators=ok)
  Translation: ...
  LaBSE: 0.7901 | Operators: ✓ | Attempt 2 (5 examples, +hint)
  ✅ Success (attempt 2)

[100/3000] train-0245
  📊 Progress: 100/3000 (3.3%)
  ⏱️  Rate: 0.13 samples/sec | ETA: 107.4 minutes
  💾 Checkpoint saved: gpt_translations_checkpoint_0001.json
```

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
To resume from checkpoint:
```python
# In 02_translate_with_validation.py
# Load checkpoint and filter already translated
checkpoint_file = 'gpt_translations_checkpoint_0020.json'
with open(checkpoint_file) as f:
    done_ids = {s['id'] for s in json.load(f)}

target_samples = [s for s in target_samples if s['id'] not in done_ids]
```

## Next Steps

After Phase 2 completion:
1. **Merge datasets**: Combine gold seed (1,996) + GPT translations (3,000) = 4,996 samples
2. **Phase 3**: Fine-tune translation model using merged dataset
3. **Scaling**: Use fine-tuned model to translate remaining ~3,663 samples
