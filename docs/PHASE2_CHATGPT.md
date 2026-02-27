# Phase 2: GPT Expansion Workflow

## Overview

Phase 2 expands the dataset using GPT with few-shot prompting. This phase translates 3,000 additional samples using diverse few-shot examples from the gold seed (Phase 1).

**Goal**: Produce 3,000 GPT-translated samples with LaBSE similarity >= 0.75 and valid operator keywords

## Key Features

### 1. Pattern-Based Sampling
- Stratified sampling ensures coverage of all SQL patterns
- Excludes samples already in manual translations
- Maintains pattern distribution similar to gold seed

### 2. Intra-Class Diversity Selection
For each target sample:
- Select 3 few-shot examples with **same SQL pattern**
- Maximize diversity using greedy algorithm:
  - Example 1: Random from pattern pool
  - Example 2: Max distance from Example 1
  - Example 3: Max min-distance from {Example 1, Example 2}
- Distance metric: Cosine distance on LaBSE embeddings

### 3. Real-Time Validation
Each translation is validated immediately:
- **LaBSE similarity** >= 0.75
- **Operator validation**: Vietnamese keywords match SQL operators
- Retry with different examples if validation fails
- Max 3 attempts per sample

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
   - Select 3 diverse few-shot examples (same pattern, max diversity)
   - Create prompt with examples + rules
   - Call GPT API for translation
   - Validate: LaBSE + operator consistency
   - If valid → save result
   - If invalid → retry with different examples (max 3 attempts)
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
- **Threshold**: >= 0.75
- **Metric**: Cosine similarity between English and Vietnamese embeddings
- **Purpose**: Ensures semantic preservation

### Operator Validation
Critical operators must have Vietnamese keywords:
- `COUNT` → bao nhiêu, số lượng, mấy, đếm
- `MAX` → lớn nhất, cao nhất, nhiều nhất
- `MIN` → nhỏ nhất, thấp nhất, ít nhất
- `AVG` → trung bình, bình quân
- `SUM` → tổng, tổng cộng
- `GREATER_THAN` → lớn hơn, cao hơn, nhiều hơn
- `LESS_THAN` → nhỏ hơn, thấp hơn, ít hơn

## Usage

### Run Full Pipeline

```bash
cd /home/hoadm/ViSpider

# Activate virtual environment
source venv/bin/activate

# Step 1: Select 3K samples (use -n to control count)
python3 scripts/phase2_chatgpt/01_select_samples_for_gpt.py -n 3000

# Step 2: Translate with validation (auto-loads gpt_target_samples.json)
# Estimated ~7-8 sec/sample → ~6-7 hours for 3K samples
python3 scripts/phase2_chatgpt/02_translate_with_validation.py
```

### Resume from Checkpoint

If script is interrupted:
1. Check latest checkpoint: `data/chatgpt_translations/gpt_translations_checkpoint_*.json`
2. Modify script to skip already translated samples
3. Re-run script

### Monitor Progress

Script outputs:
- Real-time translation results
- LaBSE scores and operator validation status
- Progress stats every 100 samples
- ETA based on current rate

Example output:
```
[1/3000] train-0001
  Translation: Có bao nhiêu ca sĩ trong cơ sở dữ liệu?
  LaBSE: 0.8234 | Operators: ✓
  ✅ Success (attempt 1)

[100/3000] train-0245
  📊 Progress: 100/3000 (3.3%)
  ⏱️  Rate: 0.45 samples/sec | ETA: 107.4 minutes
  💾 Checkpoint saved: gpt_translations_checkpoint_0001.json
```

## Quality Expectations

Based on Phase 1 manual translations and Phase 2 test run (n=20):
- **Observed success rate**: 90% (18/20)
- **Observed LaBSE mean**: 0.8499
- **Observed operator validation**: 100%
- **Average time per sample**: ~7.7 sec

**Target for production run**:
- Success rate >= 90%
- LaBSE mean >= 0.80
- Operator validation >= 90%

If success rate is low:
1. Check API key and model configuration
2. Review failed samples for patterns
3. Adjust prompt templates if needed
4. Consider using higher-tier model (gpt-4o)

## Cost Estimation

For 3,000 samples with `gpt-5-mini` (reasoning effort: low):
- Average tokens per request: ~600-800 (prompt + completion + reasoning)
- Estimated total cost: **~$1-2 USD**

For `gpt-4o-mini` (no reasoning):
- Average tokens per request: ~500
- Cost: ~$0.23 USD (input) + ~$0.90 USD (output) = **~$1.13 USD**

For `gpt-4o`:
- Cost: ~$7.50 USD (5x more expensive)

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
