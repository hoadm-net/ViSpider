#!/usr/bin/env python3
"""
Phase 5 – Step 4: Fine-tune Qwen2.5-Coder for text-to-SQL

Fine-tunes Qwen2.5-Coder-7B-Instruct with QLoRA on either the English or
Vietnamese ViSpider dataset produced by 03_prepare_data.py.

Usage
-----
# Fine-tune on English data (full run)
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang en

# Fine-tune on Vietnamese data
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang vi

# Smoke test: 10 steps, no quantization
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang en --max-steps 10 --no-quantize

# Custom hyperparameters
python3 scripts/phase5_evaluate/04_finetune_coder.py --lang vi --epochs 3 --lr 2e-4 --batch-size 4
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts/utils"))

DATA_BASE    = PROJECT_ROOT / "data/finetune_coder"
MODELS_BASE  = PROJECT_ROOT / "models"
RESULTS_BASE = PROJECT_ROOT / "results/phase5_evaluate"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder for text-to-SQL")
    p.add_argument("--lang",         required=True, choices=["en", "vi"],
                   help="Language variant to fine-tune on")
    p.add_argument("--base",         default=DEFAULT_BASE_MODEL,
                   help=f"Base model ID (default: {DEFAULT_BASE_MODEL})")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--grad-accum",   type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    p.add_argument("--max-steps",    type=int,   default=-1,
                   help="Override epochs: stop after N steps (-1 = use epochs)")
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--no-quantize",  action="store_true",
                   help="Disable 4-bit quantization (for smoke test on CPU)")
    p.add_argument("--resume",       action="store_true",
                   help="Resume from last checkpoint if it exists")
    p.add_argument("--output-dir",   default=None,
                   help="Override output directory")
    return p.parse_args()


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_to_hf_dataset(records: list, tokenizer):
    """Convert JSONL records to HuggingFace Dataset with chat-template text."""
    from datasets import Dataset

    def format_record(r: dict) -> dict:
        messages = [
            {"role": "system",    "content": r["system"]},
            {"role": "user",      "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text, "id": r["id"]}

    rows = [format_record(r) for r in records]
    return Dataset.from_list(rows)


def main():
    args = parse_args()

    data_dir  = DATA_BASE / args.lang
    train_file = data_dir / "train.jsonl"
    dev_file   = data_dir / "dev.jsonl"

    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        print("   Run 03_prepare_data.py first.")
        sys.exit(1)

    out_name  = f"qwen25coder_vispider_{args.lang}"
    out_dir   = Path(args.output_dir) if args.output_dir else MODELS_BASE / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"FINE-TUNE Qwen2.5-Coder  [{args.lang.upper()}]")
    print(f"  Base model : {args.base}")
    print(f"  Output dir : {out_dir}")
    print(f"  Epochs     : {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}×{args.grad_accum}")
    if args.max_steps > 0:
        print(f"  Max steps  : {args.max_steps}  ← smoke-test mode")
    print("=" * 72 + "\n")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data…")
    train_records = load_jsonl(train_file)
    dev_records   = load_jsonl(dev_file) if dev_file.exists() else []
    print(f"  Train: {len(train_records)}   Dev: {len(dev_records)}")

    train_ds = records_to_hf_dataset(train_records, tokenizer)
    eval_ds  = records_to_hf_dataset(dev_records,   tokenizer) if dev_records else None

    # ── Model ──────────────────────────────────────────────────────────────
    print("Loading model…")
    if args.no_quantize:
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Training config ────────────────────────────────────────────────────
    checkpoint_dir = out_dir / "checkpoints"
    last_checkpoint = None
    if args.resume and checkpoint_dir.exists():
        ckpts = sorted(checkpoint_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            last_checkpoint = str(ckpts[-1])
            print(f"\n▶  Resuming from: {last_checkpoint}")

    eval_steps = 50
    save_steps = 50
    if args.max_steps > 0 and args.max_steps < eval_steps:
        eval_steps = save_steps = max(1, args.max_steps // 2)

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=min(50, max(1, len(train_ds) // (args.batch_size * args.grad_accum * 10))),
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss",
        report_to="none",
        max_length=1024,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print("\nStarting training…\n")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── Save final adapter ────────────────────────────────────────────────
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n✅ Adapter saved → {final_dir}")

    # Save training summary
    summary = {
        "lang":       args.lang,
        "base_model": args.base,
        "train_size": len(train_records),
        "dev_size":   len(dev_records),
        "epochs":     args.epochs,
        "max_steps":  args.max_steps,
        "lr":         args.lr,
        "lora_r":     args.lora_r,
        "adapter_dir": str(final_dir),
    }
    import json as _json
    with open(out_dir / "training_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"   Summary   → {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
