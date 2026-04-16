#!/usr/bin/env python3
"""
Free-form sycophancy evaluation: stratified audit sampling.

Samples conversations for manual human audit with stratified sampling
proportional to domain sizes, ensuring each domain has at least 5 samples.

Outputs a clean format with:
  - Full conversation text
  - Judge scores (for comparison after manual scoring)
  - Blank manual scoring fields

Usage:
    python src/eval/sample_audit.py \
        --transcripts results/freeform/llama3_base_transcripts.jsonl \
        --scores results/freeform/llama3_base_scores.jsonl \
        --n 50 --seed 42 \
        --output results/freeform/audit_sample.jsonl
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DIMENSIONS = [
    "truthfulness",
    "sycophancy",
    "hedging",
    "pushback_resistance_turn2",
    "pushback_resistance_turn3",
    "helpfulness",
]


def load_jsonl(path: str) -> dict[str, dict]:
    """Load a JSONL file into a dict keyed by prompt_id."""
    items = {}
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                pid = entry.get("prompt_id")
                if pid:
                    items[pid] = entry
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d in %s: %s", i, path, e)
    return items


def format_conversation_text(turns: list[dict]) -> str:
    """Format conversation turns into a readable string for human auditors."""
    lines = []
    for i, turn in enumerate(turns, 1):
        role = "USER" if turn["role"] == "user" else "ASSISTANT"
        lines.append(f"--- Turn {i} ({role}) ---")
        lines.append(turn["content"])
        lines.append("")
    return "\n".join(lines)


def stratified_sample(
    items_by_domain: dict[str, list[str]],
    n: int,
    min_per_domain: int = 5,
    seed: int = 42,
) -> list[str]:
    """Stratified sampling proportional to domain sizes.

    Guarantees at least min_per_domain from each domain (if available).
    Returns a list of prompt_ids.
    """
    import random

    rng = random.Random(seed)

    domains = sorted(items_by_domain.keys())
    total_available = sum(len(v) for v in items_by_domain.values())

    if n > total_available:
        logger.warning(
            "Requested %d samples but only %d available. Returning all.",
            n, total_available,
        )
        n = total_available

    # Phase 1: guarantee minimum per domain
    selected: dict[str, list[str]] = {}
    remaining_budget = n

    for domain in domains:
        pool = list(items_by_domain[domain])
        rng.shuffle(pool)
        take = min(min_per_domain, len(pool), remaining_budget)
        selected[domain] = pool[:take]
        remaining_budget -= take

    # Phase 2: distribute remaining budget proportionally
    if remaining_budget > 0:
        # Available items not yet selected
        leftover_pools = {}
        for domain in domains:
            already = set(selected[domain])
            leftover = [pid for pid in items_by_domain[domain] if pid not in already]
            rng.shuffle(leftover)
            leftover_pools[domain] = leftover

        total_leftover = sum(len(v) for v in leftover_pools.values())

        if total_leftover > 0:
            for domain in domains:
                pool = leftover_pools[domain]
                # Proportional allocation
                proportion = len(pool) / total_leftover
                extra = min(
                    math.floor(proportion * remaining_budget),
                    len(pool),
                )
                selected[domain].extend(pool[:extra])

            # Handle rounding remainder
            current_total = sum(len(v) for v in selected.values())
            still_needed = n - current_total
            if still_needed > 0:
                all_remaining = []
                for domain in domains:
                    already = set(selected[domain])
                    for pid in leftover_pools[domain]:
                        if pid not in already:
                            all_remaining.append(pid)
                rng.shuffle(all_remaining)
                # Add to the domain they belong to (but we just need the IDs)
                for pid in all_remaining[:still_needed]:
                    # Find which domain
                    for domain in domains:
                        if pid in items_by_domain[domain]:
                            selected[domain].append(pid)
                            break

    # Flatten
    result = []
    for domain in domains:
        result.extend(selected[domain])

    return result[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Sample conversations for manual audit with stratified sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transcripts", required=True,
        help="Path to JSONL file with conversation transcripts",
    )
    parser.add_argument(
        "--scores", required=True,
        help="Path to JSONL file with judge scores",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of conversations to sample for audit",
    )
    parser.add_argument(
        "--min-per-domain", type=int, default=5,
        help="Minimum samples per domain",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL file for audit samples",
    )

    args = parser.parse_args()

    # Load data
    transcripts = load_jsonl(args.transcripts)
    scores = load_jsonl(args.scores)

    if not transcripts:
        logger.error("No transcripts found")
        sys.exit(1)

    # Only include items that have both transcripts and valid scores
    valid_ids = set(transcripts.keys()) & set(scores.keys())
    valid_ids = {
        pid for pid in valid_ids
        if scores[pid].get("scores") is not None
    }
    logger.info(
        "%d transcripts, %d scores, %d with both valid",
        len(transcripts), len(scores), len(valid_ids),
    )

    # Group by domain
    items_by_domain: dict[str, list[str]] = defaultdict(list)
    for pid in valid_ids:
        domain = transcripts[pid].get("domain", "unknown")
        items_by_domain[domain].append(pid)

    logger.info("Domain distribution: %s", {
        d: len(v) for d, v in sorted(items_by_domain.items())
    })

    # Sample
    sampled_ids = stratified_sample(
        items_by_domain, args.n,
        min_per_domain=args.min_per_domain,
        seed=args.seed,
    )

    # Build audit entries
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as fout:
        for pid in sampled_ids:
            transcript = transcripts[pid]
            score_entry = scores[pid]

            audit_entry = {
                "prompt_id": pid,
                "domain": transcript.get("domain", "unknown"),
                "conversation_text": format_conversation_text(
                    transcript.get("turns", [])
                ),
                "judge_scores": score_entry.get("scores"),
                "manual_scores": {
                    dim: {"score": None, "explanation": ""}
                    for dim in DIMENSIONS
                },
                "auditor_notes": "",
                "judge_human_agreement": None,  # filled after manual scoring
            }

            fout.write(json.dumps(audit_entry) + "\n")

    logger.info(
        "Wrote %d audit samples to %s", len(sampled_ids), args.output
    )

    # Print domain breakdown
    domain_counts: dict[str, int] = defaultdict(int)
    for pid in sampled_ids:
        domain_counts[transcripts[pid].get("domain", "unknown")] += 1
    print("\nAudit sample domain breakdown:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
