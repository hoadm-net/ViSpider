#!/usr/bin/env python3
"""
Phase 5 – Step 3: Prepare fine-tuning data for Qwen2.5-Coder

Converts ViSpider splits into instruction-tuning JSONL files in two language
variants (EN and VI), ready to feed into 04_finetune_coder.py.

Output files (per language):
  data/finetune_coder/{en|vi}/train.jsonl
  data/finetune_coder/{en|vi}/dev.jsonl
  data/finetune_coder/{en|vi}/test.jsonl

Each line:
  {"id": "...", "db_id": "...", "prompt": "...", "completion": "SELECT ..."}

Usage
-----
# Prepare both EN and VI (default)
python3 scripts/phase5_evaluate/03_prepare_data.py

# Only English
python3 scripts/phase5_evaluate/03_prepare_data.py --lang en

# Smoke test: first 50 samples, compact schema
python3 scripts/phase5_evaluate/03_prepare_data.py --max-samples 50 --schema compact
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/utils"))

DATA_DIR    = PROJECT_ROOT / "data"
SPLITS_DIR  = DATA_DIR / "merged"
TABLES_FILE = DATA_DIR / "raw" / "tables.json"
OUT_BASE    = DATA_DIR / "finetune_coder"

SYSTEM_PROMPT_EN = (
    "You are an expert SQL generator. "
    "Given a database schema and an English question, write the SQL query that answers it. "
    "Output only the SQL query, no explanation."
)

SYSTEM_PROMPT_VI = (
    "Bạn là chuyên gia tạo câu lệnh SQL. "
    "Dựa trên lược đồ cơ sở dữ liệu và câu hỏi bằng tiếng Việt, hãy viết truy vấn SQL tương ứng. "
    "Chỉ xuất câu lệnh SQL, không giải thích."
)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare EN/VI fine-tuning data for Qwen2.5-Coder")
    p.add_argument("--lang", choices=["en", "vi", "both"], default="both",
                   help="Language variant to prepare (default: both)")
    p.add_argument("--schema", choices=["create_table", "compact"], default="create_table",
                   help="Schema format in prompt (default: create_table)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples per split (for smoke testing)")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                   help="Which splits to prepare (default: train dev test)")
    return p.parse_args()


def load_split(split: str, max_samples: int | None) -> list:
    path = SPLITS_DIR / f"vispider_{split}.json"
    if not path.exists():
        print(f"  ⚠  {path.name} not found – skipping")
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    return data


def make_prompt(question: str, schema: str, lang: str) -> str:
    """Build the user-turn text."""
    if lang == "en":
        return (
            f"### Database Schema\n{schema}\n\n"
            f"### Question\n{question}\n\n"
            f"### SQL"
        )
    else:
        return (
            f"### Lược đồ cơ sở dữ liệu\n{schema}\n\n"
            f"### Câu hỏi\n{question}\n\n"
            f"### SQL"
        )


def prepare_lang(
    lang: str,
    splits: list[str],
    tables_lookup: dict,
    schema_mode: str,
    max_samples: int | None,
):
    from schema_utils import build_schema, build_schema_compact
    schema_fn = build_schema if schema_mode == "create_table" else build_schema_compact
    system    = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_VI
    q_field   = "question" if lang == "en" else "vi_question"

    out_dir = OUT_BASE / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        samples = load_split(split, max_samples)
        if not samples:
            continue

        skipped = 0
        records  = []
        for s in samples:
            question = s.get(q_field, "").strip()
            if not question:          # translation not yet available
                skipped += 1
                continue
            schema = schema_fn(s["db_id"], tables_lookup)
            records.append({
                "id":         s["id"],
                "db_id":      s["db_id"],
                "hardness":   s.get("hardness", "unknown"),
                "source":     s.get("source",   "unknown"),
                "prompt":     make_prompt(question, schema, lang),
                "system":     system,
                "completion": s["query"],
            })

        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"  [{lang.upper()}] {split:5s}  {len(records):5d} samples"
              + (f"  ({skipped} skipped – no VI translation)" if skipped else "")
              + f"  → {out_path.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()

    print("=" * 72)
    print("PREPARE FINE-TUNING DATA")
    print(f"  Language : {args.lang}")
    print(f"  Schema   : {args.schema}")
    print(f"  Splits   : {args.splits}")
    if args.max_samples:
        print(f"  Max/split: {args.max_samples}  ← smoke-test mode")
    print("=" * 72 + "\n")

    from schema_utils import load_tables
    tables_lookup = load_tables(TABLES_FILE)
    print(f"  Loaded schema for {len(tables_lookup)} databases\n")

    langs = ["en", "vi"] if args.lang == "both" else [args.lang]
    for lang in langs:
        prepare_lang(lang, args.splits, tables_lookup, args.schema, args.max_samples)

    print("\n✅ Done. Output in:", OUT_BASE)


if __name__ == "__main__":
    main()
