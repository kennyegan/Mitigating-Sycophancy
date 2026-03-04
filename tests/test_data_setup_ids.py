"""
Tests for deterministic sample_id assignment in data setup.
"""

import importlib.util
from pathlib import Path

from src.data.base import SycophancySample


def load_data_setup_module():
    script_path = Path(__file__).parent.parent / "scripts" / "00_data_setup.py"
    spec = importlib.util.spec_from_file_location("data_setup_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_ids_are_content_stable_across_ordering():
    module = load_data_setup_module()

    sample_a = SycophancySample(
        neutral_prompt="N1",
        biased_prompt="B1",
        sycophantic_target=" (A)",
        honest_target=" (B)",
        metadata={"source": "truthfulqa_factual"},
    )
    sample_b = SycophancySample(
        neutral_prompt="N2",
        biased_prompt="B2",
        sycophantic_target=" (B)",
        honest_target=" (A)",
        metadata={"source": "gsm8k_reasoning"},
    )

    forward = module.assign_deterministic_sample_ids([sample_a, sample_b])

    # Recreate in swapped order and ensure IDs are content-based, not position-based.
    sample_a2 = SycophancySample(
        neutral_prompt="N1",
        biased_prompt="B1",
        sycophantic_target=" (A)",
        honest_target=" (B)",
        metadata={"source": "truthfulqa_factual"},
    )
    sample_b2 = SycophancySample(
        neutral_prompt="N2",
        biased_prompt="B2",
        sycophantic_target=" (B)",
        honest_target=" (A)",
        metadata={"source": "gsm8k_reasoning"},
    )
    swapped = module.assign_deterministic_sample_ids([sample_b2, sample_a2])

    id_forward = {s.neutral_prompt: s.sample_id for s in forward}
    id_swapped = {s.neutral_prompt: s.sample_id for s in swapped}
    assert id_forward["N1"] == id_swapped["N1"]
    assert id_forward["N2"] == id_swapped["N2"]
