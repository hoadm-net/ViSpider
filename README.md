# ViSpider: Vietnamese Text-to-SQL Dataset

ViSpider is a Vietnamese translation of the [Spider dataset](https://yale-lily.github.io/spider), a large-scale text-to-SQL benchmark for semantic parsing. This project creates a high-quality Vietnamese text-to-SQL dataset through a systematic 7-step methodology combining human translation, GPT expansion, and fine-tuned model scaling.

## Project Goal

Translate the entire Spider dataset from English to Vietnamese while maintaining:
- Semantic accuracy via LaBSE embeddings
- SQL operator consistency via rule-based validation
- Query logic preservation across difficulty levels

## Methodology Overview

### Step 1: Gold Seed Construction ✅
**Status**: In Progress

Build high-quality seed dataset through manual translation:
- Translate samples covering all difficulty levels (Easy/Medium/Hard/Extra Hard)
- Validate semantic alignment using LaBSE (cosine similarity ≥ 0.75)
- Apply rule-based checks for SQL operators (COUNT, MAX/MIN, NOT, comparisons, GROUP BY, etc.)
- Review and correct samples below quality threshold

**Output**: Gold standard human-translated dataset

### Step 2: GPT Expansion 🔄
**Status**: Planned

Expand dataset using GPT with few-shot prompting:
- Use gold seed as few-shot examples for context
- Prompt constraints: preserve SQL logic, maintain literals, avoid paraphrasing
- Quality gates: LaBSE similarity ≥ 0.75 + operator consistency checks
- Manual review of sampled outputs

**Output**: GPT-translated synthetic dataset (filtered)

### Step 3: Dataset Assembly 🔄
**Status**: Planned

Merge human and synthetic data:
- Combine gold seed + GPT expansion
- Balance difficulty distribution across train/dev splits
- Reserve validation set for model evaluation

**Output**: Unified training dataset with validation split

### Step 4: Fine-tune Translation Model 🔄
**Status**: Planned

Train specialized translation model:
- Architecture: 3-7B parameter model with LoRA/QLoRA
- Task format: `(EN question, SQL) → VI question`
- Training: Early stopping on validation loss
- Optimization: Efficient fine-tuning for resource constraints

**Output**: Vietnamese question translation model

### Step 5: Pre-Scaling Evaluation 🔄
**Status**: Planned

Validate model quality before full-scale deployment:
- Benchmark on held-out samples
- Compare model outputs vs GPT baselines
- Metrics: LaBSE similarity, operator consistency, performance on Hard subset
- Decision gate: Proceed to scaling only if quality thresholds met

**Output**: Model quality report and scaling decision

### Step 6: Full Dataset Translation (Hybrid Scaling) 🔄
**Status**: Planned

Translate remaining Spider dataset using hybrid approach:
- Primary: Fine-tuned model translates all remaining samples
- Quality filter: LaBSE checks each output
- Fallback: Low-confidence samples routed to GPT for re-translation
- Final validation: Rule-based operator consistency across all samples

**Output**: Complete Vietnamese Spider dataset

### Step 7: Final Quality Control 🔄
**Status**: Planned

Comprehensive dataset validation:
- Compute LaBSE similarity distribution across full dataset
- Verify difficulty level distribution matches original Spider
- Spot-check representative samples
- Generate quality report and dataset statistics

**Output**: Production-ready ViSpider dataset

## Project Structure

```
ViSpider/
├── data/
│   ├── raw/                      # Original Spider dataset
│   ├── manual_translations/      # Step 1: Gold seed (human)
│   ├── chatgpt_translations/     # Step 2: GPT expansion
│   ├── merged_train/             # Step 3: Assembly
│   └── model_translations/       # Step 6: Hybrid scaling output
│
├── scripts/
│   ├── phase1_manual/            # Step 1: Manual translation pipeline
│   ├── phase2_chatgpt/           # Step 2: GPT translation
│   ├── phase3_finetune/          # Steps 4-7: Model training & scaling
│   └── utils/                    # Shared utilities
│
├── results/                      # Analysis outputs (gitignored)
├── experiments/                  # Experimental code (gitignored)
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
cd scripts/phase1_manual

# 1. Parse Label Studio annotations
python3 01_parse_label_studio.py

# 2. Compute embeddings and quality scores
python3 02_compute_embeddings.py

# 3. Analyze translation quality distribution
python3 03_analyze_quality.py

# 4. Extract low-quality samples for review
python3 04_extract_low_quality.py

# 5. Filter high-quality dataset
python3 05_filter_by_quality.py

# 6. Review samples interactively
python3 06_review_samples.py
```

## Quality Validation Methods

### Semantic Validation (LaBSE)

Translation quality measured using **LaBSE** (Language-agnostic BERT Sentence Embeddings):
- **Metric**: Cosine similarity between EN/VI embeddings
- **Threshold**: ≥ 0.75 for acceptable quality
- **Model**: `sentence-transformers/LaBSE`
- **Purpose**: Ensures semantic equivalence across languages

**Quality Bands**:
- **≥ 0.85**: Excellent - semantic equivalence preserved
- **0.75-0.85**: Good - minor nuance differences
- **0.65-0.75**: Acceptable - requires review
- **< 0.65**: Poor - requires re-translation

### Rule-Based Validation

SQL operator consistency checks:
- **Aggregations**: COUNT, SUM, AVG, MIN, MAX
- **Comparisons**: >, <, >=, <=, =, !=
- **Logic**: AND, OR, NOT
- **Grouping**: GROUP BY, HAVING
- **Set Operations**: UNION, INTERSECT, EXCEPT
- **Subqueries**: Nested query structure

**Purpose**: Ensures SQL semantics preserved in translation

## Documentation

- [Spider Dataset Overview](docs/SPIDER_OVERVIEW.md) - Original dataset details
- [LaBSE Embeddings](docs/LABSE_EMBEDDINGS.md) - Quality assessment methodology
- [Phase 1 Manual Translation](docs/PHASE1_MANUAL.md) - Manual translation workflow
- [Quick Reference](docs/QUICK_REFERENCE.md) - Commands and troubleshooting

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
  "patterns": ["basic_select", "count"]
}
```

## Contributing

Contributions are welcome across all pipeline steps:

**Step 1 - Manual Translation**:
- Review and correct low-quality translations
- Translate additional samples in Label Studio
- Validate semantic alignment

**Step 2 - GPT Expansion**:
- Improve few-shot prompt engineering
- Implement batch translation pipeline
- Optimize GPT API usage

**Steps 4-6 - Model Fine-tuning & Scaling**:
- Experiment with model architectures
- Optimize LoRA/QLoRA configurations
- Implement hybrid translation fallback logic

**Step 7 - Quality Control**:
- Develop additional validation rules
- Create visualization tools
- Write quality analysis reports

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
