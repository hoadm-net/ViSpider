# LaBSE Embeddings for Quality Assessment

## What is LaBSE?

**LaBSE** (Language-agnostic BERT Sentence Embeddings) is a multilingual sentence encoder developed by Google Research. It produces semantically meaningful sentence embeddings that are comparable across languages.

**Model**: `sentence-transformers/LaBSE`  
**Library**: `sentence-transformers`

## Why LaBSE for Translation Quality?

LaBSE is designed to embed sentences from different languages into the same vector space, so semantically equivalent sentences receive high cosine similarity regardless of language. This makes it well-suited for automated translation quality assessment without requiring reference translations.

## Usage in ViSpider

LaBSE is used in two places in the pipeline:

1. **Phase 2** (`scripts/phase2_manual/02_compute_embeddings.py`): Compute embeddings and cosine similarity for all manually translated EN/VI question pairs. Results are stored in `data/manual_translations/vispider_embeddings.json`.

2. **Phase 3** (`scripts/phase3_chatgpt/02_translate_with_validation.py`): Validate each GPT translation in real time before accepting it. Translations below the similarity threshold trigger an automatic retry.

Low-similarity samples are flagged for manual review or re-translation via the Phase 2 quality analysis scripts.

## References

- Paper: "Language-agnostic BERT Sentence Embedding" (Feng et al., 2020)
- Model: https://huggingface.co/sentence-transformers/LaBSE
- Library: https://www.sbert.net/
