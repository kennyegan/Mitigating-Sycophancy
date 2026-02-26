"""
Tests for baseline evaluation utilities.
"""

import pytest
import math
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestTwoWaySoftmax:
    """Tests for probability normalization."""

    def test_equal_log_probs(self):
        """Equal log probs should give 50/50 split."""
        # Import the function from the baseline script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_baseline",
            Path(__file__).parent.parent / "scripts" / "01_run_baseline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        two_way_softmax = module.two_way_softmax

        p_a, p_b = two_way_softmax(-1.0, -1.0)
        assert abs(p_a - 0.5) < 1e-6
        assert abs(p_b - 0.5) < 1e-6

    def test_probabilities_sum_to_one(self):
        """Probabilities should always sum to 1."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_baseline",
            Path(__file__).parent.parent / "scripts" / "01_run_baseline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        two_way_softmax = module.two_way_softmax

        test_cases = [
            (-0.5, -1.5),
            (-10.0, -0.1),
            (-100.0, -100.0),
        ]
        for log_a, log_b in test_cases:
            p_a, p_b = two_way_softmax(log_a, log_b)
            assert abs(p_a + p_b - 1.0) < 1e-6


class TestConfidenceInterval:
    """Tests for statistical utilities."""

    def test_ci_contains_mean(self):
        """CI should contain the mean."""
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location(
            "run_baseline",
            Path(__file__).parent.parent / "scripts" / "01_run_baseline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compute_confidence_interval = module.compute_confidence_interval

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = compute_confidence_interval(data)
        mean = np.mean(data)
        assert low <= mean <= high
