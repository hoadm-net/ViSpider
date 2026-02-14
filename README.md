# ViSpider - Vietnamese Spider Dataset Translation

Vietnamese translation of the Spider text-to-SQL dataset with quality assurance using semantic alignment analysis.

## 📁 Project Structure

```
ViSpider/
├── data/
│   ├── manual_translations/              # 2000 manually translated samples
│   │   ├── README.md                     # Translation details
│   │   ├── vispider_train_2000.json      # Clean format (1,996 samples)
│   │   ├── label_studio_2000_samples.json # Raw Label Studio export
│   │   └── vispider_embeddings.json      # Saved embeddings (~55MB, gitignored)
│   └── raw/                              # Original Spider dataset
│       ├── train_spider.json
│       ├── dev.json
│       ├── tables.json
│       └── database/                     # SQLite databases (200 databases)
│
├── docs/                                 # Documentation
│   ├── SPIDER_OVERVIEW.md               # Spider dataset info
│   ├── VISPIDER_2000_TRANSLATIONS.md    # Translation statistics
│   ├── SEMANTIC_ALIGNMENT_README.md     # Quality analysis guide
│   ├── EMBEDDINGS_FORMAT.md             # Embeddings documentation
│   └── QUICK_REFERENCE.md               # Commands cheatsheet
│
├── Core Scripts/
│   ├── parse_label_studio.py            # Parse Label Studio export
│   ├── semantic_alignment_analysis.py   # Quality analysis (OpenAI embeddings)
│   ├── review_bottom_samples.py         # Interactive review tool
│   └── embeddings_utils.py              # Work with saved embeddings
│
├── Configuration/
│   ├── .env                             # API keys (local only, gitignored)
│   ├── .env.example                     # Template for .env
│   ├── .gitignore                       # Git ignore rules
│   ├── requirements.txt                 # Python dependencies
│   └── run_analysis.sh                  # Automated analysis script
│
└── Generated Outputs/ (root, gitignored)
    ├── similarity_analysis.json         # Analysis results
    ├── bottom_100_for_review.json       # Low-similarity samples
    └── embedding_errors.json            # Error log (if any)
```

## 🚀 Quick Start

### 0. Navigate to Project Root

```bash
cd /Users/hoadinh/Desktop/Code/ViSpider
```

**Important**: All scripts must be run from the project root directory.

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Run Semantic Alignment Analysis

```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis (with progress bar)
python3 semantic_alignment_analysis.py
```

**Features:**
- ✅ Incremental save every 100 samples
- ✅ Resume from checkpoint if interrupted
- ✅ Progress bar with tqdm
- ✅ Error handling for long texts

### 4. Review Results

```bash
python review_bottom_samples.py
```

## 📊 Dataset Overview

### Spider Original
- **11,840 questions** across 200 databases
- Complex, cross-domain SQL queries
- Train/dev/test splits with different schemas

### ViSpider Translation (Current)
- **1,996 samples** translated from train set
- Professional human translation via Label Studio
- **99.8% success rate** (4 skipped)
- Coverage: 146 databases, all difficulty levels

### Distribution

| Difficulty | Count | Percentage |
|-----------|-------|------------|
| Easy      | 539   | 27.0%      |
| Medium    | 700   | 35.1%      |
| Hard      | 444   | 22.2%      |
| Extra Hard| 313   | 15.7%      |

## 🔍 Quality Assurance

### Semantic Alignment Analysis

Uses OpenAI embeddings to measure translation quality:

```bash
# Run analysis (saves embeddings)
python semantic_alignment_analysis.py

# Review low-similarity samples
python review_bottom_samples.py

# Work with embeddings
python embeddings_utils.py
```

**Key Metrics:**
- ✅ Mean similarity: Expected > 0.87
- ✅ Cross-difficulty consistency: Variance < 0.03
- ✅ Bottom 10% review: Identify false positives

See [docs/SEMANTIC_ALIGNMENT_README.md](docs/SEMANTIC_ALIGNMENT_README.md) for details.

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [SPIDER_OVERVIEW.md](SPIDER_OVERVIEW.md) | Original Spider dataset info |
| [VISPIDER_2000_TRANSLATIONS.md](VISPIDER_2000_TRANSLATIONS.md) | Translation statistics & samples |
| [SEMANTIC_ALIGNMENT_README.md](SEMANTIC_ALIGNMENT_README.md) | Quality analysis guide |
| [EMBEDDINGS_FORMAT.md](EMBEDDINGS_FORMAT.md) | Embeddings structure & usage |

## 🛠️ Scripts Usage

### Parse Label Studio Export

```bash
python parse_label_studio.py [input_file] [output_file]

# Default: 
# Input:  data/label_studio_2000_samples.json
# Output: data/vispider_train_2000.json
```

### Basic Statistics

```bash
python analyze_vispider.py

# Shows:
# - Dataset size and distribution
# - Top databases
# - SQL pattern coverage
# - Translation length metrics
```

### Semantic Alignment

```bash
python semantic_alignment_analysis.py

# Outputs:
# - similarity_analysis.json (full results)
# - bottom_100_for_review.json (review list)
# - vispider_embeddings.json (all embeddings, ~55MB)
# - embeddings_cache.json (API cache)
```

### Review Tool

```bash
python review_bottom_samples.py

# Interactive menu:
# 1. Browse samples interactively
# 2. Show summary statistics
# 3. Export markdown report
# 4. View specific sample
```

### Embeddings Utilities

```bash
python embeddings_utils.py

# Operations:
# 1. Export NumPy format for ML
# 2. Create merged data+embeddings file
# 3. Find similar questions
# 4. Show statistics
```

## 💾 Data Files

### Input Files

- `data/label_studio_2000_samples.json` - Raw Label Studio export (3.1 MB)
- `data/raw/*` - Original Spider dataset

### Output Files

- `data/vispider_train_2000.json` - Cleaned translations (500 KB)
- `vispider_embeddings.json` - All embeddings (~55 MB) **[gitignored]**
- `similarity_analysis.json` - Analysis results (2 MB) **[gitignored]**

### Working with Large Files

Embeddings are gitignored. To share:

```bash
# Compress
tar -czf vispider_embeddings.tar.gz vispider_embeddings.json

# Upload to cloud storage
# aws s3 cp vispider_embeddings.tar.gz s3://your-bucket/

# Download
# wget https://your-storage.com/vispider_embeddings.tar.gz
# tar -xzf vispider_embeddings.tar.gz
```

## 🔬 Research Use Cases

### 1. Vietnamese Text-to-SQL Models

```python
import json

# Load data
with open('data/vispider_train_2000.json') as f:
    data = json.load(f)

# Use for training
for sample in data:
    input_text = sample['vi_question']
    database_schema = sample['db_id']
    target_sql = sample['query']
    # Train your model...
```

### 2. Cross-lingual Transfer Learning

```python
import numpy as np

# Load embeddings
data = np.load('embeddings_for_training.npz')
en_embeddings = data['en_embeddings']
vi_embeddings = data['vi_embeddings']

# Alignment learning
# Train model to map EN -> VI space
```

### 3. Semantic Search

```python
from embeddings_utils import load_embeddings, find_similar_questions

embeddings = load_embeddings()
similar = find_similar_questions(embeddings, data, target_index=0, top_k=5)
```

## 📈 Next Steps

### Expansion
- [ ] Translate remaining ~5,000 training samples
- [ ] Translate dev set (1,034 samples)
- [ ] Translate test set (2,147 samples)

### Quality
- [ ] Second-pass review of bottom 10%
- [ ] Inter-annotator agreement study
- [ ] Schema translation (table/column names)

### Infrastructure
- [ ] Setup cloud storage for embeddings
- [ ] Create train/dev split for ViSpider
- [ ] Develop evaluation scripts

## 🤝 Contributing

### Translation Guidelines

Follow principles in [VISPIDER_2000_TRANSLATIONS.md](VISPIDER_2000_TRANSLATIONS.md):

1. **Accuracy**: Preserve exact SQL semantics
2. **Naturalness**: Use natural Vietnamese
3. **Consistency**: Maintain terminology
4. **Context**: Keep domain-specific terms

### Quality Checks

Before submitting translations:
1. Run `semantic_alignment_analysis.py`
2. Review samples with similarity < 0.80
3. Verify SQL patterns are maintained

## 📝 Citation

If you use this dataset:

```bibtex
@misc{vispider2026,
  title     = {ViSpider: Vietnamese Translation of Spider Text-to-SQL Dataset},
  author    = {[Your Team]},
  year      = 2026,
  note      = {Vietnamese translation of 2,000 Spider training samples}
}

@inproceedings{Yu&al.18c,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev},
  booktitle = "EMNLP",
  year      = 2018
}
```

## 📄 License

CC BY-SA 4.0 (following Spider's license)

## 📧 Contact

[Your contact information]

---

**Last Updated**: February 14, 2026  
**Version**: 0.1 (2,000 samples)  
**Status**: Active Development
