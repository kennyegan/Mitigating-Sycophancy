"""
Base classes for sycophancy dataset processing.

This module provides the abstract foundation for all sycophancy dataset
processors, ensuring consistent output format and Llama-3 chat templating.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json


@dataclass
class SycophancySample:
    """
    Standardized format for a single sycophancy evaluation sample.

    Attributes:
        neutral_prompt: The question/task without any user opinion bias
        biased_prompt: The question/task with user opinion that may induce sycophancy
        sycophantic_target: Token the model outputs if being sycophantic (agreeing with wrong view)
        honest_target: Token the model outputs if being honest (correct answer)
        metadata: Additional information about the sample (source, category, etc.)
    """
    neutral_prompt: str
    biased_prompt: str
    sycophantic_target: str
    honest_target: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class SycophancyDataset(ABC):
    """
    Abstract base class for sycophancy dataset processors.

    All dataset processors should inherit from this class and implement
    the `get_samples()` method. This ensures consistent output format
    across different data sources.

    Usage:
        class MyDataset(SycophancyDataset):
            def get_samples(self, num_samples: int) -> List[SycophancySample]:
                # Implementation here
                pass

        dataset = MyDataset(seed=42)
        samples = dataset.get_samples(500)
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the dataset processor.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        import random
        random.seed(seed)

    @abstractmethod
    def get_samples(self, num_samples: int) -> List[SycophancySample]:
        """
        Retrieve and process samples from the dataset.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            List of SycophancySample objects
        """
        pass

    @staticmethod
    def format_llama3(
        user_prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format a prompt using the Llama-3 Instruct chat template.

        This is the official template for Llama-3-Instruct models.
        Using the correct template is critical for proper model behavior.

        Args:
            user_prompt: The user's message/question
            system_prompt: Optional system message (not commonly used for evals)

        Returns:
            Formatted prompt string ready for model input
        """
        if system_prompt:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            return (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

    @staticmethod
    def ensure_leading_space(token: str) -> str:
        """
        Ensure a token has a leading space for proper Llama tokenization.

        Llama-3 tokenizes "A" and " A" differently. For MCQ answers,
        we typically want " (A)" not "(A)" to match how the model generates.

        Args:
            token: The token string

        Returns:
            Token with leading space if not already present
        """
        if not token.startswith(" "):
            return " " + token
        return token

    def save_samples(
        self,
        samples: List[SycophancySample],
        output_path: str
    ) -> None:
        """
        Save samples to a JSONL file.

        Args:
            samples: List of SycophancySample objects
            output_path: Path to output file
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(sample.to_json() + "\n")

        print(f"Saved {len(samples)} samples to {output_path}")


def validate_single_token(
    token: str,
    tokenizer_check_fn: Optional[callable] = None
) -> bool:
    """
    Check if a string is likely to be a single token.

    This is a heuristic check. For production use, pass a tokenizer
    function that actually checks tokenization.

    Args:
        token: The token string to validate
        tokenizer_check_fn: Optional function that returns True if single token

    Returns:
        True if likely single token, False otherwise
    """
    if tokenizer_check_fn:
        try:
            return tokenizer_check_fn(token)
        except Exception:
            return False

    # Heuristic: Single tokens are typically short
    # Common single tokens: " (A)", " (B)", " Yes", " No", " A", " B"
    stripped = token.strip()
    if len(stripped) <= 4:
        return True
    # Parenthetical choices are usually single tokens
    if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 4:
        return True
    return False


def validate_sample(item: Dict[str, Any]) -> bool:
    """
    Validate that a sample dictionary has all required fields with non-empty values.

    Args:
        item: Dictionary to validate

    Returns:
        True if all required fields are present and non-empty, False otherwise
    """
    required_fields = ["neutral_prompt", "biased_prompt", "sycophantic_target", "honest_target"]
    for field in required_fields:
        if field not in item:
            return False
        if not item[field]:
            return False
    return True
