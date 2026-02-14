# ViSpider: Vietnamese Text-to-SQL Dataset

ViSpider is a Vietnamese translation of the [Spider dataset](https://yale-lily.github.io/spider), a large-scale text-to-SQL benchmark for semantic parsing. This project aims to create a high-quality Vietnamese text-to-SQL dataset through a three-phase translation approach.

## Project Goal

Translate the entire Spider dataset (~10,000 samples) from English to Vietnamese while maintaining semantic accuracy and SQL query correctness.

## Translation Strategy

### Phase 1: Manual Translation ✅ 
**Status**: Completed (~2,000 samples)
- High-quality manual translations using Label Studio
- LaBSE-based quality assessment
- Target similarity: >= 0.75

### Phase 2: ChatGPT Translation 🔄
**Status**: Planned (~3,000 samples)
- Automated translation using ChatGPT API
- Prompt engineering with schema context
- Quality validation with LaBSE

### Phase 3: Fine-tuned Model Translation 🔄
**Status**: Planned (remaining ~5,000 samples)
- Fine-tune 3-7B parameter model on Phases 1+2
- Translate remaining samples
- Final quality check and dataset release

## Project Structure

```
ViSpider/
├── data/
│   ├── raw/                      # Original Spider dataset
│   ├── manual_translations/      # Phase 1 outputs
│   ├── chatgpt_translations/     # Phase 2 outputs (planned)
│   └── model_translations/       # Phase 3 outputs (planned)
│
├── scripts/
│   ├── phase1_manual/            # Manual translation pipeline
│   ├── phase2_chatgpt/           # ChatGPT translation (planned)
│   ├── phase3_finetune/          # Model fine-tuning (planned)
│   └── utils/                    # Shared utilities
│
├── results/                      # Analysis outputs (gitignored)
├── experiments/                  # Experimental code (gitignored)
└── docs/                         # Documentation
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ViSpider

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Phase 1 Pipeline

```bash
cd scripts/phase1_manual

# 1. Parse Label Studio annotations
python3 01_parse_label_studio.py

# 2. Compute embeddings and quality scores
python3 02_compute_embeddings.py

# 3. Analyze translation quality
python3 03_analyze_quality.py

# 4. Extract low-quality samples for review
python3 04_extract_low_quality.py

# 5. Filter high-quality dataset
python3 05_filter_by_quality.py
```

## Quality Assessment

Translation quality is measured using **LaBSE** (Language-agnostic BERT Sentence Embeddings):

- **Metric**: Cosine similarity between English and Vietnamese embeddings
- **Threshold**: >= 0.75 for acceptable quality
- **Model**: `sentence-transformers/LaBSE`

### Quality Levels
- **>= 0.85**: Excellent - semantic equivalence preserved
- **0.75-0.85**: Good - minor nuance differences
- **0.65-0.75**: Acceptable - may need review
- **< 0.65**: Poor - requires re-translation

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

Contributions are welcome! Areas for contribution:
- Reviewing and correcting low-quality translations
- Implementing Phase 2 ChatGPT translation pipeline
- Fine-tuning models for Phase 3
- Adding Vietnamese schema descriptions

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
