#!/usr/bin/env python3
"""
Phase 5 – Step 5: Predict SQL

Run inference with a fine-tuned (or base) Qwen2.5-Coder model on the test
split and save predicted SQL queries.

Usage
-----
# Predict with fine-tuned EN model
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en

# Predict with fine-tuned VI model
python3 scripts/phase5_evaluate/05_predict_sql.py --lang vi

# Zero-shot with base model (no fine-tuning)
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en --zero-shot

# Evaluate dev instead of test split
python3 scripts/phase5_evaluate/05_predict_sql.py --lang vi --split dev

# Quick smoke test: first 20 samples
python3 scripts/phase5_evaluate/05_predict_sql.py --lang en --n 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/utils"))

DATA_BASE    = PROJECT_ROOT / "data/finetune_coder"
MODELS_BASE  = PROJECT_ROOT / "models"
RESULTS_BASE = PROJECT_ROOT / "results/phase5_evaluate"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="Generate SQL predictions with Qwen2.5-Coder")
    p.add_argument("--lang",          required=True, choices=["en", "vi"])
    p.add_argument("--split",         default="test", choices=["dev", "test"])
    p.add_argument("--model",         default=None,
                   help="Adapter or merged model path. "
                        "Default: models/qwen25coder_vispider_{lang}/final")
    p.add_argument("--base",          default=DEFAULT_BASE_MODEL,
                   help="Base model (needed when loading adapter)")
    p.add_argument("--zero-shot",     action="store_true",
                   help="Use base model directly without any adapter")
    p.add_argument("--n",             type=int, default=None,
                   help="Only predict first N samples (smoke test)")
    p.add_argument("--batch",         type=int, default=4)
    p.add_argument("--max-new-tokens",type=int, default=256)
    p.add_argument("--output",        default=None,
                   help="Output JSON path (default: results/phase5_evaluate/predictions_{lang}_{split}.json)")
    return p.parse_args()


def load_model(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if args.zero_shot:
        model_path = args.base
        print(f"Loading base model (zero-shot): {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    else:
        model_path = Path(args.model) if args.model else (
            MODELS_BASE / f"qwen25coder_vispider_{args.lang}" / "final"
        )
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            print(f"   Run 04_finetune_coder.py --lang {args.lang} first, or use --zero-shot")
            sys.exit(1)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        is_adapter = (model_path / "adapter_config.json").exists()
        if is_adapter:
            from peft import PeftModel
            print(f"Loading base: {args.base}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            print(f"Loading adapter: {model_path}")
            model = PeftModel.from_pretrained(model, str(model_path))
        else:
            print(f"Loading merged model: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )

    model.eval()
    return model, tokenizer


def batch_generate(model, tokenizer, prompts: list, max_new_tokens: int) -> list[str]:
    import torch
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Extract only the SQL (stop at first blank line or comment)
        sql = _extract_sql(text)
        results.append(sql)
    return results


def _extract_sql(text: str) -> str:
    """Clean generated text to keep only the SQL query."""
    # Strip markdown code fences  ```sql ... ```
    if "```" in text:
        lines = text.split("\n")
        in_block = False
        sql_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                sql_lines.append(line)
        if sql_lines:
            return " ".join(sql_lines).strip()
    # Otherwise take first non-empty paragraph
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            return para
    return text.strip()


def build_inference_prompt(record: dict, tokenizer, system: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": record["prompt"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main():
    args = parse_args()

    # Load test data
    data_file = DATA_BASE / args.lang / f"{args.split}.jsonl"
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("   Run 03_prepare_data.py first.")
        sys.exit(1)

    records = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.n:
        records = records[:args.n]

    mode = "zero-shot" if args.zero_shot else "fine-tuned"
    print("=" * 72)
    print(f"PREDICT SQL  [{args.lang.upper()}  {args.split}  {mode}]")
    print(f"  Samples: {len(records)}")
    print("=" * 72 + "\n")

    model, tokenizer = load_model(args)

    # Read system prompt from first record
    system = records[0]["system"] if records else ""

    # Inference
    print("Running inference…")
    start = time.time()
    predictions = []

    for i in range(0, len(records), args.batch):
        batch = records[i:i + args.batch]
        prompts = [build_inference_prompt(r, tokenizer, system) for r in batch]
        sqls    = batch_generate(model, tokenizer, prompts, args.max_new_tokens)
        for rec, pred_sql in zip(batch, sqls):
            predictions.append({
                "id":         rec["id"],
                "db_id":      rec["db_id"],
                "hardness":   rec.get("hardness", "unknown"),
                "source":     rec.get("source",   "unknown"),
                "gold_sql":   rec["completion"],
                "pred_sql":   pred_sql,
                "lang":       args.lang,
                "split":      args.split,
                "mode":       mode,
            })

        done    = min(i + args.batch, len(records))
        elapsed = time.time() - start
        rate    = done / elapsed
        eta     = (len(records) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(records)}]  {rate:.1f} samples/s  ETA {eta:.0f}s")

    # Save
    out_path = Path(args.output) if args.output else (
        RESULTS_BASE / f"predictions_{args.lang}_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(predictions)} predictions → {out_path}")


if __name__ == "__main__":
    main()
