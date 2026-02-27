#!/usr/bin/env python3
"""
Merge LoRA adapter weights into the base model to produce a standalone model.

After fine-tuning, the output directory contains only the LoRA adapter weights
(~100 MB), not the full model. This script merges them with the base model to
produce a standalone model that can be loaded without PEFT.

Usage:
  python3 03_merge_adapter.py
  python3 03_merge_adapter.py --adapter models/qwen25_vispider/final --output models/qwen25_vispider_merged
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_ADAPTER = PROJECT_ROOT / 'models/qwen25_vispider/final'
DEFAULT_OUTPUT  = PROJECT_ROOT / 'models/qwen25_vispider_merged'


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--adapter", default=str(DEFAULT_ADAPTER),
                   help="Path to fine-tuned adapter directory")
    p.add_argument("--output",  default=str(DEFAULT_OUTPUT),
                   help="Path to save merged model")
    p.add_argument("--dtype",   default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Output model dtype")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        raise SystemExit(1)

    adapter_path = Path(args.adapter)
    output_path  = Path(args.output)

    print("=" * 72)
    print("MERGE LORA ADAPTER → STANDALONE MODEL")
    print("=" * 72)
    print(f"\nAdapter: {adapter_path}")
    print(f"Output:  {output_path}")

    # Read base model name from training config
    config_file = adapter_path / "training_config.json"
    if config_file.exists():
        with open(config_file) as f:
            train_cfg = json.load(f)
        base_model_id = train_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")
    else:
        base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Base:    {base_model_id}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load base model (full precision for clean merge)
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu",        # merge on CPU to avoid OOM
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # Load and merge adapter
    print("Loading adapter and merging...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    # Save
    print(f"\nSaving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("\n✅ Done — merged model saved.")
    print(f"   Load with: AutoModelForCausalLM.from_pretrained('{output_path}')")


if __name__ == "__main__":
    main()
