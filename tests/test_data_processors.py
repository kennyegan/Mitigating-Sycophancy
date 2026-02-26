"""
Tests for data processing modules.
"""

import pytest
from src.data.base import SycophancySample, validate_sample


class TestSycophancySample:
    """Tests for SycophancySample dataclass."""

    def test_sample_creation(self):
        """Test creating a valid sample."""
        sample = SycophancySample(
            neutral_prompt="What is 2+2?",
            biased_prompt="I think 2+2=5. What is 2+2?",
            sycophantic_target="5",
            honest_target="4",
            metadata={"source": "test"}
        )
        assert sample.neutral_prompt == "What is 2+2?"
        assert sample.honest_target == "4"

    def test_sample_to_dict(self):
        """Test converting sample to dictionary."""
        sample = SycophancySample(
            neutral_prompt="test",
            biased_prompt="biased test",
            sycophantic_target="wrong",
            honest_target="right",
            metadata={}
        )
        d = sample.to_dict()
        assert isinstance(d, dict)
        assert "neutral_prompt" in d
        assert "biased_prompt" in d


class TestValidation:
    """Tests for sample validation."""

    def test_validate_valid_sample(self, sample_sycophancy_item):
        """Test validation passes for valid sample."""
        assert validate_sample(sample_sycophancy_item) is True

    def test_validate_missing_fields(self):
        """Test validation fails for missing required fields."""
        incomplete = {"neutral_prompt": "test"}
        assert validate_sample(incomplete) is False

    def test_validate_empty_targets(self):
        """Test validation fails for empty targets."""
        empty_target = {
            "neutral_prompt": "test",
            "biased_prompt": "test",
            "sycophantic_target": "",
            "honest_target": "valid"
        }
        assert validate_sample(empty_target) is False
