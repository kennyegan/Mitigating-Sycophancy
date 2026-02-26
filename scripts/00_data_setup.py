"""
Data Setup Orchestrator for Multi-Dataset Sycophancy Pipeline.

This script downloads and processes samples from three distinct sycophancy
evaluation sources, creating a unified master dataset for research.

Datasets:
    1. Opinion Sycophancy (Anthropic): Tests agreement with user beliefs
    2. Factual Sycophancy (TruthfulQA): Tests agreement with misconceptions
    3. Reasoning Sycophancy (GSM8k): Tests agreement with flawed logic

Output:
    data/processed/master_sycophancy.jsonl - Unified dataset

Usage:
    python scripts/data_setup.py
    python scripts/data_setup.py --samples 100  # Override sample count
    python scripts/data_setup.py --output data/custom_path.jsonl
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.base import SycophancySample
from src.data.anthropic import AnthropicOpinionDataset
from src.data.truthful_qa import TruthfulQAFactualDataset
from src.data.gsm8k_reasoning import GSM8kReasoningDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and process multi-dataset sycophancy benchmark"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples per dataset (default: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/master_sycophancy.jsonl",
        help="Output path for master dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Also save individual dataset files"
    )
    parser.add_argument(
        "--control-groups",
        action="store_true",
        help="Generate control group datasets (requires a model for filtering)"
    )
    parser.add_argument(
        "--control-model",
        type=str,
        default="gpt2-medium",
        help="Model for control group filtering (default: gpt2-medium)"
    )
    parser.add_argument(
        "--control-device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device for control group model (default: auto-detect)"
    )
    return parser.parse_args()


def create_dataset_summary(
    samples: List[SycophancySample],
    dataset_name: str
) -> Dict:
    """
    Create a summary of dataset statistics.

    Args:
        samples: List of samples
        dataset_name: Name of the dataset

    Returns:
        Summary dictionary
    """
    if not samples:
        return {"dataset": dataset_name, "count": 0}

    categories = {}
    for sample in samples:
        cat = sample.metadata.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "dataset": dataset_name,
        "count": len(samples),
        "categories": categories,
        "sample_fields": list(samples[0].to_dict().keys()) if samples else []
    }


def save_samples(
    samples: List[SycophancySample],
    output_path: str
) -> None:
    """
    Save samples to JSONL file.

    Args:
        samples: List of SycophancySample objects
        output_path: Path to output file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")

    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 70)
    print("MULTI-DATASET SYCOPHANCY PIPELINE")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Samples per dataset: {args.samples}")
    print(f"  Output path: {args.output}")
    print(f"  Random seed: {args.seed}")
    print("=" * 70)

    all_samples: List[SycophancySample] = []
    summaries = []

    # 1. Opinion Sycophancy (Anthropic)
    print("\n[1/3] Processing Opinion Sycophancy (Anthropic)...")
    print("-" * 50)
    try:
        opinion_dataset = AnthropicOpinionDataset(seed=args.seed)
        opinion_samples = opinion_dataset.get_samples(args.samples)
        all_samples.extend(opinion_samples)
        summaries.append(create_dataset_summary(opinion_samples, "anthropic_opinion"))

        if args.save_individual:
            save_samples(
                opinion_samples,
                "data/processed/opinion_sycophancy.jsonl"
            )
    except Exception as e:
        print(f"Error processing Opinion dataset: {e}")
        summaries.append({"dataset": "anthropic_opinion", "count": 0, "error": str(e)})

    # 2. Factual Sycophancy (TruthfulQA)
    print("\n[2/3] Processing Factual Sycophancy (TruthfulQA)...")
    print("-" * 50)
    try:
        factual_dataset = TruthfulQAFactualDataset(seed=args.seed)
        factual_samples = factual_dataset.get_samples(args.samples)
        all_samples.extend(factual_samples)
        summaries.append(create_dataset_summary(factual_samples, "truthfulqa_factual"))

        if args.save_individual:
            save_samples(
                factual_samples,
                "data/processed/factual_sycophancy.jsonl"
            )
    except Exception as e:
        print(f"Error processing Factual dataset: {e}")
        summaries.append({"dataset": "truthfulqa_factual", "count": 0, "error": str(e)})

    # 3. Reasoning Sycophancy (GSM8k)
    print("\n[3/3] Processing Reasoning Sycophancy (GSM8k)...")
    print("-" * 50)
    try:
        reasoning_dataset = GSM8kReasoningDataset(seed=args.seed)
        reasoning_samples = reasoning_dataset.get_samples(args.samples)
        all_samples.extend(reasoning_samples)
        summaries.append(create_dataset_summary(reasoning_samples, "gsm8k_reasoning"))

        if args.save_individual:
            save_samples(
                reasoning_samples,
                "data/processed/reasoning_sycophancy.jsonl"
            )
    except Exception as e:
        print(f"Error processing Reasoning dataset: {e}")
        summaries.append({"dataset": "gsm8k_reasoning", "count": 0, "error": str(e)})

    # Control groups
    if args.control_groups and all_samples:
        print("\n" + "=" * 70)
        print("GENERATING CONTROL GROUPS")
        print("=" * 70)

        from src.data.control_groups import (
            filter_uncertain_knowledge,
            generate_fictional_entities,
            filter_adversarially_true,
        )
        from src.models import SycophancyModel

        control_dir = Path("data/processed/control_groups")
        control_dir.mkdir(parents=True, exist_ok=True)

        # Load all_samples as dicts for filtering
        all_dicts = [s.to_dict() for s in all_samples]

        # Initialize model for filtering
        print(f"\nLoading model for control group filtering: {args.control_model}")
        ctrl_model = SycophancyModel(args.control_model, device=args.control_device)

        # 1. Fictional Entities (no model needed)
        print("\n[Control 1/3] Fictional Entities...")
        fictional_samples = generate_fictional_entities(num_samples=100, seed=args.seed)
        save_samples(fictional_samples, str(control_dir / "fictional_entities.jsonl"))

        # 2. Uncertain Knowledge
        print("\n[Control 2/3] Uncertain Knowledge...")
        uncertain, confident = filter_uncertain_knowledge(all_dicts, ctrl_model, threshold=0.60)
        with open(control_dir / "uncertain_knowledge.jsonl", 'w') as f:
            for item in uncertain:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(uncertain)} uncertain samples")

        # 3. Adversarially-True Hints
        print("\n[Control 3/3] Adversarially-True Hints...")
        adversarial, clean = filter_adversarially_true(all_dicts, ctrl_model, confidence_threshold=0.70)
        with open(control_dir / "adversarially_true.jsonl", 'w') as f:
            for item in adversarial:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(adversarial)} adversarially-true samples")

        # Save control group metadata
        ctrl_metadata = {
            "created_at": datetime.now().isoformat(),
            "model": args.control_model,
            "seed": args.seed,
            "fictional_entities": len(fictional_samples),
            "uncertain_knowledge": len(uncertain),
            "adversarially_true": len(adversarial),
            "clean_after_filters": len(clean),
        }
        with open(control_dir / "control_groups_metadata.json", 'w') as f:
            json.dump(ctrl_metadata, f, indent=2)

        print(f"\nControl group files saved to {control_dir}/")

    # Save master dataset
    print("\n" + "=" * 70)
    print("SAVING MASTER DATASET")
    print("=" * 70)

    if all_samples:
        save_samples(all_samples, args.output)

        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(all_samples),
            "seed": args.seed,
            "datasets": summaries
        }

        metadata_path = args.output.replace(".jsonl", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    total = 0
    for summary in summaries:
        count = summary.get("count", 0)
        total += count
        status = "OK" if count > 0 else "FAILED"
        print(f"  {summary['dataset']:25s}: {count:5d} samples [{status}]")

    print("-" * 50)
    print(f"  {'TOTAL':25s}: {total:5d} samples")
    print("=" * 70)

    # Print sample entry
    if all_samples:
        print("\nSample entry from master dataset:")
        print("-" * 50)
        sample_dict = all_samples[0].to_dict()
        # Truncate long strings for display
        for key in ["neutral_prompt", "biased_prompt"]:
            if key in sample_dict and len(sample_dict[key]) > 100:
                sample_dict[key] = sample_dict[key][:100] + "..."
        print(json.dumps(sample_dict, indent=2))

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
