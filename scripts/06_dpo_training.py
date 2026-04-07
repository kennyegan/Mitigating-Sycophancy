"""
DPO Fine-Tuning Script for Reducing Sycophancy in Llama-3-8B-Instruct.

This script generates DPO (Direct Preference Optimization) training pairs from
the Anthropic opinion sycophancy dataset using a SEPARATE seed from the benchmark,
then fine-tunes the model with LoRA to prefer honest responses over sycophantic ones.

Key design decisions:
    - Uses seed=100 (not seed=42) to generate NEW opinion pairs disjoint from the
      500-sample benchmark test set.
    - Chosen = honest answer (disagrees with user's wrong opinion)
    - Rejected = sycophantic answer (agrees with user's wrong opinion)
    - LoRA adapters only, saved to results/dpo_model/

Usage:
    python scripts/06_dpo_training.py
    python scripts/06_dpo_training.py --n-train-pairs 400 --epochs 3
    python scripts/06_dpo_training.py --seed 100 --lr 5e-5 --beta 0.1
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DPO fine-tuning to reduce sycophancy in Llama-3-8B-Instruct"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model identifier (default: meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--n-train-pairs",
        type=int,
        default=400,
        help="Number of DPO training pairs to generate (default: 400)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for training data generation (default: 100, NOT 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dpo_model",
        help="Directory to save the LoRA adapter (default: results/dpo_model)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (default: 0.1)",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of DPO data held out for eval loss (default: 0.1)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_dpo_pairs(
    model_name: str,
    n_pairs: int,
    seed: int,
    tokenizer=None,
) -> List[Dict[str, str]]:
    """
    Generate DPO training pairs from the Anthropic opinion sycophancy dataset.

    Uses a SEPARATE seed from the benchmark (seed=42) so that the training
    pairs are disjoint from the 500-sample test set.

    Each pair:
        prompt  = biased prompt (contains user opinion + question + choices)
        chosen  = honest response token (correct answer despite user opinion)
        rejected = sycophantic response token (agrees with user's wrong opinion)

    The prompt and responses are formatted with the Llama-3 chat template so
    that TRL's DPOTrainer receives ready-to-tokenize strings.

    Args:
        model_name: HuggingFace model name (used for tokenizer if not provided)
        n_pairs: Number of training pairs to generate
        seed: Random seed (should NOT be 42)
        tokenizer: Optional pre-loaded tokenizer

    Returns:
        List of {"prompt": str, "chosen": str, "rejected": str} dicts
    """
    from src.data.anthropic import AnthropicOpinionDataset
    from src.data.base import SycophancyDataset

    if seed == 42:
        print(
            "WARNING: seed=42 is the benchmark test seed. "
            "Using it for training will contaminate the evaluation. "
            "Proceeding anyway as requested, but consider using --seed 100."
        )

    print(f"Generating {n_pairs} DPO pairs with seed={seed} ...")
    print(f"  (Benchmark test set uses seed=42 with 500 samples — these are disjoint)")

    # Load opinion dataset with training seed
    opinion_dataset = AnthropicOpinionDataset(seed=seed)
    # Request more samples than needed to account for any filtering losses
    raw_samples = opinion_dataset.get_samples(int(n_pairs * 1.3))

    if len(raw_samples) < n_pairs:
        print(
            f"WARNING: Only got {len(raw_samples)} valid samples, "
            f"requested {n_pairs}. Using all available."
        )
        n_pairs = len(raw_samples)

    # Take exactly n_pairs
    samples = raw_samples[:n_pairs]

    dpo_pairs = []
    for sample in samples:
        # The biased_prompt already has the Llama-3 chat template applied
        # (from AnthropicOpinionDataset.get_samples -> format_llama3)
        # It ends with "<|start_header_id|>assistant<|end_header_id|>\n\n"
        #
        # For DPO, we need:
        #   prompt  = the full biased prompt (up to where the assistant responds)
        #   chosen  = the honest (non-sycophantic) answer token
        #   rejected = the sycophantic answer token
        prompt = sample.biased_prompt
        chosen = sample.honest_target       # e.g., " (B)" — the correct answer
        rejected = sample.sycophantic_target  # e.g., " (A)" — agrees with user

        dpo_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    print(f"Generated {len(dpo_pairs)} DPO training pairs")
    return dpo_pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dpo(args):
    """Run DPO fine-tuning with LoRA."""
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig

    start_time = time.time()

    # ---- Output directories ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate DPO training data ----
    print("=" * 70)
    print("STEP 1: Generating DPO training pairs")
    print("=" * 70)

    dpo_pairs = generate_dpo_pairs(
        model_name=args.model,
        n_pairs=args.n_train_pairs,
        seed=args.seed,
    )

    if not dpo_pairs:
        print("ERROR: No DPO pairs generated. Exiting.")
        return 1

    # Split into train / eval
    import random
    random.seed(args.seed)
    random.shuffle(dpo_pairs)
    n_eval = max(1, int(len(dpo_pairs) * args.eval_split))
    eval_pairs = dpo_pairs[:n_eval]
    train_pairs = dpo_pairs[n_eval:]

    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Eval pairs:  {len(eval_pairs)}")

    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset = Dataset.from_list(eval_pairs)

    # Save the generated pairs for reproducibility
    pairs_path = output_dir / "dpo_training_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(
            {"train": train_pairs, "eval": eval_pairs, "seed": args.seed},
            f,
            indent=2,
        )
    print(f"  Saved pairs to {pairs_path}")

    # ---- Load model and tokenizer ----
    print("\n" + "=" * 70)
    print("STEP 2: Loading model and tokenizer")
    print("=" * 70)

    print(f"  Model: {args.model}")
    print(f"  Device: cuda" if torch.cuda.is_available() else "  Device: cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use flash_attention_2 if available, fall back to eager
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

    # ---- LoRA configuration ----
    print("\n" + "=" * 70)
    print("STEP 3: Applying LoRA adapter")
    print("=" * 70)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  LoRA alpha: 32")
    print(f"  Target modules: q_proj, v_proj, k_proj, o_proj")
    print(f"  Dropout: 0.05")

    # ---- DPO training configuration ----
    print("\n" + "=" * 70)
    print("STEP 4: Starting DPO training")
    print("=" * 70)

    effective_batch = args.batch_size * 4
    total_steps = (len(train_pairs) // effective_batch) * args.epochs
    print(f"  Batch size: {args.batch_size} (effective: {effective_batch})")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  DPO beta: {args.beta}")
    print(f"  Estimated total steps: {total_steps}")

    training_args = DPOConfig(
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
        remove_unused_columns=False,
        seed=args.seed,
        max_length=512,
        beta=args.beta,
        gradient_checkpointing=True,
    )

    # ---- Initialize DPO Trainer ----
    trainer = DPOTrainer(
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

    # ---- Save the final LoRA adapter ----
    print("\n" + "=" * 70)
    print("STEP 5: Saving adapter and metrics")
    print("=" * 70)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  Adapter saved to {output_dir}")

    # ---- Eval on held-out split ----
    eval_results = trainer.evaluate()
    print(f"  Eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

    # ---- Save training metrics ----
    elapsed = time.time() - start_time
    metrics = {
        "model": args.model,
        "seed": args.seed,
        "n_train_pairs": len(train_pairs),
        "n_eval_pairs": len(eval_pairs),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "dpo_beta": args.beta,
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

    metrics_path = Path("results") / "dpo_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("DPO TRAINING COMPLETE")
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
    print("DPO FINE-TUNING FOR SYCOPHANCY REDUCTION")
    print("=" * 70)
    print(f"  Model:        {args.model}")
    print(f"  Train pairs:  {args.n_train_pairs}")
    print(f"  Seed:         {args.seed}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LR:           {args.lr}")
    print(f"  LoRA rank:    {args.lora_rank}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  DPO beta:     {args.beta}")
    print(f"  Eval split:   {args.eval_split}")
    print("=" * 70)

    try:
        return train_dpo(args)
    except Exception as e:
        print(f"\nERROR: DPO training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
