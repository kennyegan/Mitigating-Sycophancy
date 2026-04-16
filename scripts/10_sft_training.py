"""
SFT Baseline Training Script for Reducing Sycophancy in Llama-3-8B-Instruct.

Trains on the SAME data as the DPO pipeline (same seed, same generation) but uses
standard supervised fine-tuning on chosen (non-sycophantic) responses only.
No preference pairs, no rejected responses — just next-token prediction on good answers.

This is the simplest serious training-time baseline for comparison against DPO.
The ONLY difference from DPO is the training objective; LoRA config, data, model,
and evaluation are all identical.

Usage:
    python scripts/10_sft_training.py
    python scripts/10_sft_training.py --seed 100 --n-pairs 400 --output-dir results/sft_model/
"""

import sys
import os
import json
import argparse
import time
import random
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Reuse the exact same data generation pipeline as DPO
# (filename has numeric prefix, so use importlib)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dpo_training",
    os.path.join(os.path.dirname(__file__), "06_dpo_training.py"),
)
_dpo_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dpo_module)
generate_dpo_pairs = _dpo_module.generate_dpo_pairs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SFT baseline fine-tuning on chosen responses (non-sycophantic only)"
    )
    parser.add_argument(
        "--model", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=400,
        help="Number of pairs to generate (chosen responses extracted from these)",
    )
    parser.add_argument(
        "--seed", type=int, default=100,
        help="Random seed for data generation (default: 100, same as DPO)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/sft_model",
        help="Directory to save the LoRA adapter",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3, same as DPO)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate (default: 5e-5, same as DPO)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank (default: 16, same as DPO)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device training batch size (default: 4, same as DPO)",
    )
    parser.add_argument(
        "--eval-split", type=float, default=0.1,
        help="Fraction of data held out for eval loss (default: 0.1, same as DPO)",
    )
    return parser.parse_args()


def train_sft(args):
    """Run SFT fine-tuning with LoRA on chosen responses only."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig

    start_time = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Generate data using the SAME pipeline as DPO ----
    print("=" * 70)
    print("STEP 1: Generating training data (same pipeline as DPO)")
    print("=" * 70)

    dpo_pairs = generate_dpo_pairs(
        model_name=args.model,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )

    if not dpo_pairs:
        print("ERROR: No pairs generated. Exiting.")
        return 1

    # Extract ONLY chosen responses — discard rejected entirely
    # Each SFT example is: prompt + chosen response as a single sequence
    sft_examples = []
    for pair in dpo_pairs:
        sft_examples.append({
            "text": pair["prompt"] + pair["chosen"],
        })

    print(f"  Extracted {len(sft_examples)} chosen-only SFT examples")
    print(f"  (Rejected responses discarded — SFT sees only non-sycophantic answers)")

    # Split into train / eval (same split logic as DPO)
    random.seed(args.seed)
    random.shuffle(sft_examples)
    n_eval = max(1, int(len(sft_examples) * args.eval_split))
    eval_examples = sft_examples[:n_eval]
    train_examples = sft_examples[n_eval:]

    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples:  {len(eval_examples)}")

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    # Save the generated examples for reproducibility
    pairs_path = output_dir / "sft_training_examples.json"
    with open(pairs_path, "w") as f:
        json.dump(
            {"train": train_examples, "eval": eval_examples, "seed": args.seed},
            f, indent=2,
        )
    print(f"  Saved examples to {pairs_path}")

    # ---- Step 2: Load model and tokenizer ----
    print("\n" + "=" * 70)
    print("STEP 2: Loading model and tokenizer")
    print("=" * 70)

    print(f"  Model: {args.model}")
    print(f"  Device: cuda" if torch.cuda.is_available() else "  Device: cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use flash_attention_2 if available
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
        print("  flash_attn not installed, using eager attention")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    print(f"  Model loaded: {model.dtype}, {model.device}")

    # ---- Step 3: LoRA configuration (IDENTICAL to DPO) ----
    print("\n" + "=" * 70)
    print("STEP 3: Applying LoRA adapter (same config as DPO)")
    print("=" * 70)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  LoRA alpha: 32")
    print(f"  Target modules: q_proj, v_proj, k_proj, o_proj")
    print(f"  Dropout: 0.05")

    # ---- Step 4: SFT training configuration ----
    print("\n" + "=" * 70)
    print("STEP 4: Starting SFT training")
    print("=" * 70)

    effective_batch = args.batch_size * 4
    total_steps = (len(train_examples) // effective_batch) * args.epochs
    print(f"  Batch size: {args.batch_size} (effective: {effective_batch})")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Training objective: standard causal LM (next-token prediction)")
    print(f"  Estimated total steps: {total_steps}")

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        max_seq_length=512,
        gradient_checkpointing=True,
        dataset_text_field="text",
    )

    # ---- Initialize SFT Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Print trainable parameters after PEFT wrapping
    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(
        f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ---- Train ----
    print("\nTraining started at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train_result = trainer.train()
    print("Training finished at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ---- Step 5: Save adapter and metrics ----
    print("\n" + "=" * 70)
    print("STEP 5: Saving adapter and metrics")
    print("=" * 70)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  Adapter saved to {output_dir}")

    # Eval on held-out split
    eval_results = trainer.evaluate()
    print(f"  Eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

    # Save training metrics
    elapsed = time.time() - start_time
    metrics = {
        "training_method": "sft",
        "model": args.model,
        "seed": args.seed,
        "n_train_examples": len(train_examples),
        "n_eval_examples": len(eval_examples),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": 32,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": effective_batch,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "eval_loss": eval_results.get("eval_loss"),
        "total_elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "completed_at": datetime.now().isoformat(),
    }

    metrics_path = Path("results") / "sft_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SFT BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Train loss:  {metrics['train_loss']:.4f}")
    print(f"  Eval loss:   {metrics['eval_loss']:.4f}")
    print(f"  Adapter:     {output_dir}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Total time:  {elapsed / 60:.1f} minutes")
    print("=" * 70)

    return 0


def main():
    args = parse_args()

    print("=" * 70)
    print("SFT BASELINE FINE-TUNING FOR SYCOPHANCY REDUCTION")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Train pairs:  {args.n_pairs}")
    print(f"  Seed:         {args.seed}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LR:           {args.lr}")
    print(f"  LoRA rank:    {args.lora_rank}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Eval split:   {args.eval_split}")
    print(f"  Objective:    SFT (chosen responses only, no preference pairs)")
    print("=" * 70)

    try:
        return train_sft(args)
    except Exception as e:
        print(f"\nERROR: SFT training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
