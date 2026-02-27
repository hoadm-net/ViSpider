#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B-Instruct for EN→VI question translation using QLoRA.

Task:  (English question + SQL query + db_id)  →  Vietnamese question

Training data: data/merged/vispider_train.json
Validation:    data/merged/vispider_dev.json

Prompt format (Qwen2.5 ChatML):
  <|im_start|>system
  You are a professional translator ...
  <|im_end|>
  <|im_start|>user
  Database: {db_id}
  English:  {question}
  SQL:      {query}
  <|im_end|>
  <|im_start|>assistant
  {vi_question}<|im_end|>

Usage:
  # QLoRA 4-bit (recommended for 16-24 GB VRAM)
  python3 02_finetune.py

  # Full precision (requires ~28 GB VRAM)
  python3 02_finetune.py --no-quantize

  # Custom hyperparams
  python3 02_finetune.py --epochs 5 --lr 2e-4 --batch-size 4

  # Resume from checkpoint
  python3 02_finetune.py --resume

Requirements:
  pip install peft trl bitsandbytes accelerate
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset


# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

TRAIN_FILE  = PROJECT_ROOT / 'data/merged/vispider_train.json'
DEV_FILE    = PROJECT_ROOT / 'data/merged/vispider_dev.json'
OUTPUT_DIR  = PROJECT_ROOT / 'models/qwen25_vispider'

# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a professional translator specializing in technical text-to-SQL questions. "
    "Translate the English question into natural Vietnamese. "
    "Keep ALL technical terms, table names, and column names in English. "
    "Output only the Vietnamese translation, no explanation."
)


def build_prompt(sample: dict, tokenizer, include_response: bool = True) -> str:
    """
    Build a ChatML prompt for a single sample.
    If include_response=False, returns the prompt-only string (for inference).
    """
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

    if include_response:
        messages.append({"role": "assistant", "content": sample["vi_question"]})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not include_response,
    )


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_dataset(path: Path, tokenizer) -> Dataset:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts = [build_prompt(r, tokenizer) for r in records]
    return Dataset.from_dict({"text": texts})


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B-Instruct for ViSpider")

    # Model
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model ID or local path")
    p.add_argument("--no-quantize", action="store_true",
                   help="Disable 4-bit quantization (needs ~28 GB VRAM)")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank (higher = more params, better quality)")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA alpha (scaling = alpha / r)")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                   help="Which linear layers to apply LoRA to")

    # Training
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=2,
                   help="Per-device train batch size")
    p.add_argument("--grad-accum",  type=int,   default=8,
                   help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max-length",  type=int,   default=512,
                   help="Maximum token length per sample")
    p.add_argument("--warmup-ratio",type=float, default=0.05)
    p.add_argument("--lr-scheduler",default="cosine",
                   choices=["cosine", "linear", "constant"])
    p.add_argument("--weight-decay",type=float, default=0.01)
    p.add_argument("--save-steps",  type=int,   default=100)
    p.add_argument("--eval-steps",  type=int,   default=100)
    p.add_argument("--logging-steps",type=int,  default=10)
    p.add_argument("--max-steps",   type=int,   default=-1,
                   help="Max training steps. -1 = use epochs. Set e.g. 10 for smoke test")

    # I/O
    p.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    p.add_argument("--resume",      action="store_true",
                   help="Resume training from latest checkpoint in output-dir")
    p.add_argument("--seed",        type=int, default=42)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Late imports (requires peft/trl/bitsandbytes) ──────────────────────
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install with:")
        print("  pip install peft trl bitsandbytes accelerate")
        raise SystemExit(1)

    print("=" * 72)
    print("FINE-TUNING: Qwen2.5-7B-Instruct → ViSpider EN→VI")
    print("=" * 72)
    print(f"\nModel:       {args.model}")
    print(f"Quantize:    {'4-bit QLoRA' if not args.no_quantize else 'None (full precision)'}")
    print(f"LoRA rank:   {args.lora_r}  alpha={args.lora_alpha}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size} × {args.grad_accum} grad_accum "
          f"= {args.batch_size * args.grad_accum} effective")
    print(f"LR:          {args.lr}  scheduler={args.lr_scheduler}")
    print(f"Output:      {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"   # required for SFTTrainer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_dataset = load_dataset(TRAIN_FILE, tokenizer)
    eval_dataset  = load_dataset(DEV_FILE,   tokenizer)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Dev:   {len(eval_dataset)} samples")

    # ── Model ──────────────────────────────────────────────────────────────
    print("\nLoading model...")
    bnb_config = None
    if not args.no_quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,     # nested quantization: saves ~0.4 bits/param
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.no_quantize else None,
    )

    if not args.no_quantize:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"\n  Trainable params: {trainable:,}  ({100 * trainable / total:.2f}% of {total:,})")

    # ── Training args ──────────────────────────────────────────────────────
    # Align save_steps to be a round multiple of eval_steps
    # (required by load_best_model_at_end)
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    if save_steps % eval_steps != 0:
        save_steps = max(eval_steps, ((save_steps + eval_steps - 1) // eval_steps) * eval_steps)

    # Warmup: convert ratio → steps (rough estimate using train size / batch)
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = steps_per_epoch * args.epochs if args.max_steps < 0 else args.max_steps
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    training_args = SFTConfig(
        output_dir=str(output_dir),

        # Epochs & steps
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,          # -1 = disabled (use epochs); >0 overrides epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,      # trade compute for VRAM

        # Optimiser
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        optim="paged_adamw_8bit" if not args.no_quantize else "adamw_torch",

        # Precision (GPU only)
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),

        # Logging / saving
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # SFT-specific  (trl ≥ 0.26: max_seq_length → max_length)
        max_length=args.max_length,
        dataset_text_field="text",
        packing=False,                    # don't pack samples; each sample is independent

        # Misc
        seed=args.seed,
        report_to="none",                 # change to "wandb" or "tensorboard" if desired
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TRAINING")
    print("=" * 72)

    resume_from = True if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SAVING")
    print("=" * 72)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training config alongside the model
    config_out = final_dir / "training_config.json"
    with open(config_out, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n✅ Model saved to: {final_dir}")
    print(f"   Adapter weights: {final_dir}/adapter_model.safetensors")
    print(f"   Load with:  AutoModelForCausalLM.from_pretrained('{final_dir}')")
    print(f"    or merge:   python3 03_merge_adapter.py")


if __name__ == "__main__":
    main()
