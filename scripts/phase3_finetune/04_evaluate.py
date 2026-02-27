#!/usr/bin/env python3
"""
Run inference with the fine-tuned model on the dev/test set and compute
LaBSE similarity scores to evaluate translation quality.

Usage:
  # Evaluate on dev set (default)
  python3 04_evaluate.py

  # Evaluate on test set
  python3 04_evaluate.py --split test

  # Use merged standalone model
  python3 04_evaluate.py --model models/qwen25_vispider_merged

  # Quick smoke-test on first N samples
  python3 04_evaluate.py --n 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/utils"))

DEFAULT_ADAPTER = PROJECT_ROOT / "models/qwen25_vispider/final"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned ViSpider model")
    p.add_argument("--model",  default=str(DEFAULT_ADAPTER),
                   help="Adapter directory or merged model path")
    p.add_argument("--base",   default="Qwen/Qwen2.5-7B-Instruct",
                   help="Base model ID (only needed when loading adapter, not merged model)")
    p.add_argument("--split",  default="dev", choices=["dev", "test"],
                   help="Which split to evaluate")
    p.add_argument("--n",      type=int, default=None,
                   help="Evaluate only first N samples (default: all)")
    p.add_argument("--batch",  type=int, default=8,
                   help="Inference batch size")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output", default=None,
                   help="Path to save evaluation results JSON (default: results/quality_analysis/model_eval_{split}.json)")
    return p.parse_args()


SYSTEM_PROMPT = (
    "You are a professional translator specializing in technical text-to-SQL questions. "
    "Translate the English question into natural Vietnamese. "
    "Keep ALL technical terms, table names, and column names in English. "
    "Output only the Vietnamese translation, no explanation."
)


def build_prompt(sample: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Database: {sample['db_id']}\n"
                f"English:  {sample['question']}\n"
                f"SQL:      {sample['query']}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model)

    # Detect if this is an adapter directory (has adapter_config.json) or merged model
    is_adapter = (model_path / "adapter_config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Left-padding is required for correct batch inference with decoder-only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        from peft import PeftModel
        print(f"Loading base model: {args.base}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading adapter: {model_path}")
        model = PeftModel.from_pretrained(model, str(model_path))
    else:
        print(f"Loading merged model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def batch_generate(model, tokenizer, prompts: list, max_new_tokens: int) -> list:
    import torch
    # padding_side must be "left" (set in load_model) for decoder-only batch inference
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (not the prompt)
    results = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated = output[input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(text)
    return results


def main():
    args = parse_args()

    print("=" * 72)
    print(f"EVALUATION: fine-tuned model on {args.split.upper()} set")
    print("=" * 72)

    # Load data
    data_file = PROJECT_ROOT / f"data/merged/vispider_{args.split}.json"
    with open(data_file, encoding="utf-8") as f:
        samples = json.load(f)
    if args.n:
        samples = samples[:args.n]
    print(f"\nEvaluating {len(samples)} samples from: {data_file.name}")

    # Load model
    try:
        model, tokenizer = load_model(args)
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install: pip install peft trl bitsandbytes accelerate")
        raise SystemExit(1)

    # Load LaBSE
    from embeddings_utils import compute_similarity

    # Inference
    print("\nRunning inference...")
    start = time.time()
    predictions = []

    for i in range(0, len(samples), args.batch):
        batch = samples[i:i + args.batch]
        prompts = [build_prompt(s, tokenizer) for s in batch]
        translations = batch_generate(model, tokenizer, prompts, args.max_new_tokens)
        for sample, pred in zip(batch, translations):
            predictions.append({
                "id":           sample["id"],
                "question":     sample["question"],
                "vi_reference": sample["vi_question"],
                "vi_predicted": pred,
                "source":       sample.get("source", "?"),
                "hardness":     sample.get("hardness", "?"),
            })

        done = min(i + args.batch, len(samples))
        elapsed = time.time() - start
        rate = done / elapsed
        eta  = (len(samples) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(samples)}]  {rate:.1f} samples/sec  ETA {eta:.0f}s")

    # Compute LaBSE scores
    print("\nComputing LaBSE similarity scores...")
    scores = []
    for p in predictions:
        sim = compute_similarity(p["question"], p["vi_predicted"])
        p["labse_similarity"] = round(sim, 4)
        scores.append(sim)

    # Aggregate
    import statistics
    above_75 = sum(1 for s in scores if s >= 0.75)
    stats = {
        "split":            args.split,
        "n_samples":        len(scores),
        "labse_mean":       round(statistics.mean(scores), 4),
        "labse_median":     round(statistics.median(scores), 4),
        "labse_min":        round(min(scores), 4),
        "labse_max":        round(max(scores), 4),
        "labse_stdev":      round(statistics.stdev(scores), 4),
        "above_0.75":       above_75,
        "above_0.75_pct":   round(100 * above_75 / len(scores), 1),
    }

    # Per-hardness breakdown
    from collections import defaultdict
    by_hardness: dict[str, list] = defaultdict(list)
    for p in predictions:
        by_hardness[p["hardness"]].append(p["labse_similarity"])
    stats["by_hardness"] = {
        h: {
            "n": len(v),
            "mean": round(statistics.mean(v), 4),
            "above_0.75_pct": round(100 * sum(1 for s in v if s >= 0.75) / len(v), 1),
        }
        for h, v in sorted(by_hardness.items())
    }

    # Print summary
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    print(f"\n  LaBSE mean:    {stats['labse_mean']:.4f}")
    print(f"  LaBSE median:  {stats['labse_median']:.4f}")
    print(f"  >= 0.75:       {stats['above_0.75']}/{stats['n_samples']} ({stats['above_0.75_pct']}%)")
    print(f"\n  By hardness:")
    for h, v in stats["by_hardness"].items():
        print(f"    {h:<12}  mean={v['mean']:.4f}  >=0.75: {v['above_0.75_pct']}%  (n={v['n']})")

    # Print a few examples
    print("\n  Sample predictions:")
    for p in predictions[:5]:
        print(f"\n  [{p['id']}] {p['hardness']} | LaBSE={p['labse_similarity']}")
        print(f"    EN:  {p['question'][:80]}")
        print(f"    REF: {p['vi_reference'][:80]}")
        print(f"    → :  {p['vi_predicted'][:80]}")

    # Save
    out_path = args.output or str(
        PROJECT_ROOT / f"results/quality_analysis/model_eval_{args.split}.json"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "predictions": predictions}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
