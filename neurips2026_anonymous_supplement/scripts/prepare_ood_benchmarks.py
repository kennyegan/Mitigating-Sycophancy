#!/usr/bin/env python3
"""
Prepare Out-of-Distribution (OOD) Sycophancy Benchmarks.

Downloads and adapts three sycophancy subcategory files from the Anthropic
model-written-evals HuggingFace repository into the project's evaluation format.

These subcategories are distinct from the generic "sycophancy" split used for
DPO training (seed=100) and in-distribution evaluation (seed=42). They cover
different topic domains:

  1. NLP Survey         — NLP researcher opinion questions
  2. Political Typology — Pew political typology quiz questions

Note: The upstream repo lists a third file (sycophancy_on_philpapers2020.jsonl)
but it is a duplicate of the NLP survey file (identical blob hash), so only two
distinct subcategories are available.

Source: Anthropic/model-written-evals (Perez et al., 2022)
  https://huggingface.co/datasets/Anthropic/model-written-evals

Usage:
    python scripts/prepare_ood_benchmarks.py
    python scripts/prepare_ood_benchmarks.py --n-samples 500 --seed 42
    python scripts/prepare_ood_benchmarks.py --output-dir data/ood_prompts

Output:
    data/ood_prompts/nlp_survey.jsonl
    data/ood_prompts/political_typology.jsonl
"""

import sys
import os
import json
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.base import (
    SycophancyDataset,
    validate_single_token,
    build_deterministic_sample_id,
)


# =============================================================================
# Configuration
# =============================================================================

REPO_ID = "Anthropic/model-written-evals"

# Sycophancy subcategory files in the HuggingFace repo
# NOTE: The upstream repo lists three sycophancy files, but philpapers2020.jsonl is
# a duplicate of nlp_survey.jsonl (identical blob hash). Only two distinct datasets
# are available. See data/ood_prompts/README.md for details.
SYCOPHANCY_FILES = {
    "nlp_survey": "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "political_typology": "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
}

DOMAIN_DESCRIPTIONS = {
    "nlp_survey": "NLP researcher opinion questions (Perez et al., 2022)",
    "political_typology": "Pew political typology quiz questions (Perez et al., 2022)",
}

DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "data/ood_prompts"


# =============================================================================
# Download
# =============================================================================

def download_sycophancy_files(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Download sycophancy subcategory JSONL files from HuggingFace.

    Returns dict mapping domain name -> local file path.
    """
    from huggingface_hub import hf_hub_download

    local_paths = {}
    for domain, repo_path in SYCOPHANCY_FILES.items():
        print(f"  Downloading {repo_path}...")
        kwargs = dict(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type="dataset",
        )
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        local_path = hf_hub_download(**kwargs)
        local_paths[domain] = local_path
        print(f"    -> {local_path}")

    return local_paths


# =============================================================================
# Neutral Prompt Extraction
# =============================================================================

def extract_neutral_prompt(question_text: str) -> Optional[str]:
    """
    Extract the question + choices WITHOUT the user biography/opinion.

    Handles three Anthropic sycophancy subcategory formats:

    Format A (nlp_survey, philpapers2020):
        "Hello, my name is [name]. [bio]... Do you agree with [claim]?

         Choices:
         (A) Agree
         (B) Disagree"

    Format B (political_typology):
        "Hello, my name is [name]. [bio]... If you had to choose, would you rather have...
         (A) Choice A
         (B) Choice B"

    Returns the question stem + choice lines, or None if extraction fails.
    """
    lines = question_text.strip().split("\n")

    # Find lines containing choice markers
    choice_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\([A-Z]\)\s", stripped):
            choice_indices.append(i)

    if not choice_indices:
        return None

    first_choice_idx = choice_indices[0]

    # Collect choice lines (and any label like "Choices:")
    choice_lines = []
    for i in range(first_choice_idx, len(lines)):
        stripped = lines[i].strip()
        if stripped and stripped not in ("Answer:", ""):
            choice_lines.append(stripped)

    # Also grab "Choices:" label if present just before
    if first_choice_idx > 0:
        prev = lines[first_choice_idx - 1].strip()
        if prev.lower() in ("choices:", "options:"):
            pass  # don't include label, just use question + choice lines

    # Find the question stem
    # Strategy: look for the question text before choices.
    # It may be on its own line (Format A) or embedded at the end of the bio (Format B).
    question_stem = None

    # First, try to find a standalone question line (not containing "Hello, my name is")
    for i in range(first_choice_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped or stripped.lower() in ("choices:", "options:", "answer:"):
            continue
        if "Hello, my name is" not in stripped:
            # This is a standalone question line (Format A)
            question_stem = stripped
            break
        else:
            # Biography line contains the question (Format B)
            # Extract question from end of biography using sentence-level heuristics
            question_stem = _extract_question_from_bio(stripped)
            break

    if question_stem is None:
        return None

    # Build neutral prompt
    neutral = question_stem + "\n" + "\n".join(choice_lines)

    # Final sanity check
    if "Hello, my name is" in neutral:
        return None

    return neutral


# Common question-start patterns for political typology and similar formats
_QUESTION_PATTERNS = [
    r"(If you had to choose.*)",
    r"(Which of the following\b.*)",
    r"(How much\b.*)",
    r"(How do you\b.*)",
    r"(In general\b.*)",
    r"(Do you (?:think|agree|believe|support|favor)\b.*)",
    r"(Would you (?:say|rather|favor|support)\b.*)",
    r"(Overall\b.*)",
    r"(In the future\b.*)",
    r"(Has .*\?)",
    r"(Is .*\?)",
    r"(Are .*\?)",
    r"(Should .*\?)",
    r"(Can .*\?)",
]


def _extract_question_from_bio(bio_line: str) -> Optional[str]:
    """
    Extract the question stem from a biography line where the question is
    embedded at the end (e.g., political typology format).

    Example:
        "Hello, my name is Jane. ... If you had to choose, would you rather have..."
        -> "If you had to choose, would you rather have..."
    """
    # Try each pattern — they're ordered by specificity
    for pattern in _QUESTION_PATTERNS:
        m = re.search(pattern, bio_line, re.IGNORECASE)
        if m:
            return m.group(1)

    # Fallback: split by sentence boundaries and take the last sentence
    # that looks like a question (contains '?' or ends with '...')
    sentences = re.split(r'(?<=[.!?])\s+', bio_line)
    for sent in reversed(sentences):
        if sent.endswith("?") or sent.endswith("..."):
            if "Hello, my name is" not in sent:
                return sent

    return None


# =============================================================================
# Processing
# =============================================================================

def load_raw_jsonl(filepath: str) -> List[Dict]:
    """Load all JSON lines from a file."""
    items = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def process_domain(
    raw_path: str,
    domain_name: str,
    n_samples: int,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    """
    Process a single sycophancy subcategory file into evaluation format.

    Returns (processed_items, stats_dict).
    """
    raw_items = load_raw_jsonl(raw_path)
    total_raw = len(raw_items)

    stats = {
        "domain": domain_name,
        "raw_count": total_raw,
        "skipped_missing_fields": 0,
        "skipped_list_answer": 0,
        "skipped_extraction": 0,
        "skipped_validation": 0,
        "skipped_biography_in_neutral": 0,
        "valid_count": 0,
    }

    # Filter for required fields
    valid_raw = []
    for item in raw_items:
        if not all(k in item for k in ["question", "answer_matching_behavior", "answer_not_matching_behavior"]):
            stats["skipped_missing_fields"] += 1
            continue
        valid_raw.append(item)

    # Process each item
    candidates = []
    for item in valid_raw:
        question = item["question"]
        answer_matching = item["answer_matching_behavior"]
        answer_not_matching = item["answer_not_matching_behavior"]

        # Handle list-valued answer_not_matching_behavior (common in philpapers2020)
        if isinstance(answer_not_matching, list):
            if len(answer_not_matching) == 0:
                stats["skipped_list_answer"] += 1
                continue
            answer_not_matching = answer_not_matching[0]
            stats["skipped_list_answer"] += 0  # tracked but not skipped

        # Ensure leading spaces on targets
        sycophantic_target = SycophancyDataset.ensure_leading_space(
            answer_matching.strip()
        )
        honest_target = SycophancyDataset.ensure_leading_space(
            answer_not_matching.strip()
        )

        # Validate single-token targets
        if not validate_single_token(sycophantic_target):
            stats["skipped_validation"] += 1
            continue
        if not validate_single_token(honest_target):
            stats["skipped_validation"] += 1
            continue

        # Extract neutral prompt (strip biography)
        neutral_text = extract_neutral_prompt(question)
        if neutral_text is None:
            stats["skipped_extraction"] += 1
            continue

        # Biased prompt is the full question text
        biased_text = question

        # Double-check neutral prompt doesn't contain biography
        if "Hello, my name is" in neutral_text:
            stats["skipped_biography_in_neutral"] += 1
            continue

        # Format with Llama-3 chat template
        neutral_prompt = SycophancyDataset.format_llama3(neutral_text)
        biased_prompt = SycophancyDataset.format_llama3(biased_text)

        candidates.append({
            "neutral_prompt": neutral_prompt,
            "biased_prompt": biased_prompt,
            "sycophantic_target": sycophantic_target,
            "honest_target": honest_target,
            "metadata": {
                "source": f"anthropic_{domain_name}",
                "category": domain_name,
                "dataset": REPO_ID,
                "ood_type": "ood_same_format",
            },
        })

    stats["valid_count"] = len(candidates)

    # Sample deterministically
    rng = random.Random(seed)
    if len(candidates) > n_samples:
        selected = rng.sample(candidates, n_samples)
    else:
        selected = candidates
        rng.shuffle(selected)

    # Assign deterministic sample IDs
    for item in selected:
        item["sample_id"] = build_deterministic_sample_id(
            item, prefix=domain_name
        )

    return selected, stats


def save_domain(items: List[Dict], output_path: str) -> None:
    """Save processed items to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare OOD sycophancy benchmarks from Anthropic model-written-evals"
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per domain (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSONL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace cache directory (default: uses HF_HOME or ~/.cache/huggingface)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("OOD SYCOPHANCY BENCHMARK PREPARATION")
    print("=" * 70)
    print(f"Source:     {REPO_ID}")
    print(f"Samples:    {args.n_samples} per domain")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output_dir}")
    print()

    # -------------------------------------------------------------------------
    # Step 1: Download files
    # -------------------------------------------------------------------------
    print("Step 1: Downloading sycophancy subcategory files...")
    local_paths = download_sycophancy_files(cache_dir=args.cache_dir)
    print()

    # -------------------------------------------------------------------------
    # Step 2: Process each domain
    # -------------------------------------------------------------------------
    print("Step 2: Processing domains...")
    all_stats = []
    total_saved = 0

    for domain_name in SYCOPHANCY_FILES:
        print(f"\n  --- {domain_name} ---")
        print(f"  Description: {DOMAIN_DESCRIPTIONS[domain_name]}")

        items, stats = process_domain(
            raw_path=local_paths[domain_name],
            domain_name=domain_name,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        all_stats.append(stats)

        # Save
        output_path = os.path.join(args.output_dir, f"{domain_name}.jsonl")
        save_domain(items, output_path)
        total_saved += len(items)

        print(f"  Raw samples:     {stats['raw_count']:>6d}")
        print(f"  Valid candidates: {stats['valid_count']:>6d}")
        print(f"  Saved:           {len(items):>6d}")
        if stats['skipped_missing_fields'] > 0:
            print(f"  Skipped (missing fields): {stats['skipped_missing_fields']}")
        if stats['skipped_validation'] > 0:
            print(f"  Skipped (token validation): {stats['skipped_validation']}")
        if stats['skipped_extraction'] > 0:
            print(f"  Skipped (neutral extraction): {stats['skipped_extraction']}")
        if stats['skipped_biography_in_neutral'] > 0:
            print(f"  Skipped (biography in neutral): {stats['skipped_biography_in_neutral']}")

        # Print example
        if items:
            ex = items[0]
            # Show just the user-facing text (strip Llama-3 template for readability)
            biased_raw = ex["biased_prompt"].split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
            neutral_raw = ex["neutral_prompt"].split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
            print(f"\n  Example biased prompt (first 200 chars):")
            print(f"    {biased_raw[:200]}...")
            print(f"  Example neutral prompt (first 200 chars):")
            print(f"    {neutral_raw[:200]}...")
            print(f"  Sycophantic target: {repr(ex['sycophantic_target'])}")
            print(f"  Honest target:      {repr(ex['honest_target'])}")

    # -------------------------------------------------------------------------
    # Step 3: Save metadata
    # -------------------------------------------------------------------------
    metadata = {
        "source": REPO_ID,
        "citation": "Perez et al. (2022), Discovering Language Model Behaviors with Model-Written Evaluations",
        "n_samples_per_domain": args.n_samples,
        "seed": args.seed,
        "domains": {s["domain"]: s for s in all_stats},
        "total_samples": total_saved,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Domain':<25s} {'Raw':>8s} {'Valid':>8s} {'Saved':>8s}")
    print("-" * 55)
    for stats in all_stats:
        print(f"  {stats['domain']:<23s} {stats['raw_count']:>8d} "
              f"{stats['valid_count']:>8d} {min(args.n_samples, stats['valid_count']):>8d}")
    print("-" * 55)
    print(f"  {'TOTAL':<23s} {'':>8s} {'':>8s} {total_saved:>8d}")
    print()
    print(f"Output files:")
    for domain_name in SYCOPHANCY_FILES:
        p = os.path.join(args.output_dir, f"{domain_name}.jsonl")
        print(f"  {p}")
    print(f"  {meta_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
