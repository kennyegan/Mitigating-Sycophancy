"""
Sycophancy Dataset Processors.

This module provides dataset processors for three types of sycophancy:
- Opinion Sycophancy (Anthropic): Agreement with user beliefs
- Factual Sycophancy (TruthfulQA): Agreement with misconceptions
- Reasoning Sycophancy (GSM8k): Agreement with flawed logic
"""

from .base import (
    SycophancyDataset,
    SycophancySample,
    validate_single_token,
    build_deterministic_sample_id,
)

AnthropicOpinionDataset = None
TruthfulQAFactualDataset = None
GSM8kReasoningDataset = None

try:
    from .anthropic import AnthropicOpinionDataset
    from .truthful_qa import TruthfulQAFactualDataset
    from .gsm8k_reasoning import GSM8kReasoningDataset
except Exception:
    # Keep base utilities importable in minimal/test environments.
    pass

__all__ = [
    # Base classes
    "SycophancyDataset",
    "SycophancySample",
    "validate_single_token",
    "build_deterministic_sample_id",
    # Dataset processors
    "AnthropicOpinionDataset",
    "TruthfulQAFactualDataset",
    "GSM8kReasoningDataset",
]
