"""
Sycophancy Dataset Processors.

This module provides dataset processors for three types of sycophancy:
- Opinion Sycophancy (Anthropic): Agreement with user beliefs
- Factual Sycophancy (TruthfulQA): Agreement with misconceptions
- Reasoning Sycophancy (GSM8k): Agreement with flawed logic
"""

from .base import SycophancyDataset, SycophancySample, validate_single_token
from .anthropic import AnthropicOpinionDataset
from .truthful_qa import TruthfulQAFactualDataset
from .gsm8k_reasoning import GSM8kReasoningDataset

__all__ = [
    # Base classes
    "SycophancyDataset",
    "SycophancySample",
    "validate_single_token",
    # Dataset processors
    "AnthropicOpinionDataset",
    "TruthfulQAFactualDataset",
    "GSM8kReasoningDataset",
]
