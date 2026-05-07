#!/usr/bin/env python3
"""
Free-Form Sycophancy Transcript Generator.

Runs multi-turn conversations on the free-form sycophancy benchmark (150 prompts
across 5 JSONL files in data/freeform/). For each prompt, the script conducts a
3-turn exchange:

  Turn 1: initial_user_message   -> model response
  Turn 2: user_pushback_1        -> model response
  Turn 3: user_pushback_2        -> model response

The conversation history is built incrementally so each model response conditions
on all prior turns. Transcripts are written one JSONL line at a time for
incremental saving and resume support.

Usage:
    python src/eval/freeform_generate.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --prompts data/freeform/ \
        --output results/freeform/llama3_base_transcripts.jsonl

    python src/eval/freeform_generate.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --prompts data/freeform/ \
        --output results/freeform/llama3_dpo_transcripts.jsonl \
        --adapter-path results/dpo_model/

Output (one JSONL line per prompt):
    {
      "prompt_id": "ff_opinion_001",
      "domain": "opinion",
      "model": "meta-llama/Meta-Llama-3-8B-Instruct",
      "adapter_path": null,
      "turns": [ {role, content}, ... ],   # 6 messages (3 user + 3 assistant)
      "ground_truth_assessment": "...",
      "sycophancy_indicators": [...]
    }
"""

import argparse
import json
import os
import random
import sys
from glob import glob
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_SEED = 42


# =============================================================================
# Reproducibility
# =============================================================================

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_name: str, adapter_path: str | None, device: str):
    """Load base model with optional LoRA adapter merged in."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA merge complete.")

    model.eval()
    return model, tokenizer


# =============================================================================
# Data Loading
# =============================================================================

def load_prompts(prompts_dir: str) -> list[dict]:
    """Read all .jsonl files from the prompts directory."""
    pattern = os.path.join(prompts_dir, "*.jsonl")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found in {prompts_dir}. "
            "Check that the directory contains the freeform benchmark data."
        )

    prompts = []
    for filepath in files:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Validate required fields
                required = [
                    "prompt_id", "domain", "initial_user_message",
                    "user_pushback_1", "user_pushback_2",
                ]
                missing = [k for k in required if k not in item]
                if missing:
                    print(f"  WARNING: Skipping {item.get('prompt_id', '?')} "
                          f"— missing fields: {missing}")
                    continue
                prompts.append(item)
        print(f"  Loaded {filepath} ({sum(1 for p in prompts if True)}+ prompts)")

    print(f"Total prompts loaded: {len(prompts)}")
    return prompts


def load_completed_ids(output_path: str) -> set[str]:
    """Read already-completed prompt_ids from the output file for resume."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pid = record.get("prompt_id")
                if pid:
                    completed.add(pid)
            except json.JSONDecodeError:
                continue
    return completed


# =============================================================================
# Generation
# =============================================================================

def generate_response(model, tokenizer, messages: list[dict], device: str) -> str:
    """Generate a single assistant response given the conversation so far."""
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def run_conversation(model, tokenizer, prompt: dict, device: str) -> list[dict]:
    """Run the 3-turn conversation for a single prompt, returning 6 messages."""
    user_messages = [
        prompt["initial_user_message"],
        prompt["user_pushback_1"],
        prompt["user_pushback_2"],
    ]

    turns = []
    messages = []  # running conversation for the chat template

    for user_text in user_messages:
        messages.append({"role": "user", "content": user_text})
        turns.append({"role": "user", "content": user_text})

        response = generate_response(model, tokenizer, messages, device)

        messages.append({"role": "assistant", "content": response})
        turns.append({"role": "assistant", "content": response})

    return turns


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-turn free-form sycophancy evaluation transcripts.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Base model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompts", required=True,
                        help="Directory containing freeform .jsonl prompt files")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path for transcripts")
    parser.add_argument("--adapter-path", default=None,
                        help="Optional path to LoRA adapter directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("FREE-FORM SYCOPHANCY TRANSCRIPT GENERATION")
    print("=" * 70)
    print(f"Model:        {args.model}")
    print(f"Adapter:      {args.adapter_path or '(none)'}")
    print(f"Prompts dir:  {args.prompts}")
    print(f"Output:       {args.output}")
    print(f"Seed:         {args.seed}")
    print(f"Device:       {device}")
    print()

    # Load prompts
    prompts = load_prompts(args.prompts)

    # Resume support
    completed_ids = load_completed_ids(args.output)
    remaining = [p for p in prompts if p["prompt_id"] not in completed_ids]
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} already done, "
              f"{len(remaining)} remaining.")

    if not remaining:
        print("All prompts already processed. Nothing to do.")
        return 0

    # Load model
    model, tokenizer = load_model(args.model, args.adapter_path, device)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Generate transcripts
    response_lengths = []

    for prompt in tqdm(remaining, desc="Generating transcripts"):
        turns = run_conversation(model, tokenizer, prompt, device)

        record = {
            "prompt_id": prompt["prompt_id"],
            "domain": prompt["domain"],
            "model": args.model,
            "adapter_path": args.adapter_path,
            "turns": turns,
            "ground_truth_assessment": prompt.get("ground_truth_assessment", ""),
            "sycophancy_indicators": prompt.get("sycophancy_indicators", []),
        }

        # Incremental save — append one line
        with open(args.output, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Track assistant response lengths
        for turn in turns:
            if turn["role"] == "assistant":
                response_lengths.append(len(turn["content"]))

    # Summary
    total = len(completed_ids) + len(remaining)
    avg_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total prompts processed: {total}")
    print(f"  Previously completed:  {len(completed_ids)}")
    print(f"  Generated this run:    {len(remaining)}")
    print(f"Avg response length:     {avg_len:.0f} chars")
    print(f"Output saved to:         {args.output}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
