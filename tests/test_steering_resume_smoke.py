"""
Smoke integration for steering checkpoint/resume flow.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


def load_steering_module():
    pytest.importorskip("torch")
    script_path = Path(__file__).parent.parent / "scripts" / "05_representation_steering.py"
    spec = importlib.util.spec_from_file_location("steering_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyModel:
    def __init__(self):
        self.model_name = "dummy-model"


def test_steering_checkpoint_resume_without_duplication(tmp_path, monkeypatch):
    module = load_steering_module()

    dataset = [
        {
            "neutral_prompt": "neutral",
            "biased_prompt": "biased",
            "sycophantic_target": " (A)",
            "honest_target": " (B)",
            "metadata": {"source": "test"},
        }
        for _ in range(8)
    ]

    monkeypatch.setattr(
        module,
        "compute_steering_vectors",
        lambda model, data, layers, n_samples=200: ({l: torch.zeros(4) for l in layers}, {l: 0.0 for l in layers}),
    )
    monkeypatch.setattr(
        module,
        "evaluate_sycophancy_with_steering",
        lambda *args, **kwargs: {
            "overall_sycophancy_rate": 0.25,
            "overall_sycophancy_rate_ci": [0.1, 0.4],
            "total_sycophantic": 2,
            "total_evaluated": 8,
            "skipped": 0,
            "per_source": {"test": {"sycophancy_rate": 0.25, "sycophancy_rate_ci": [0.1, 0.4], "sycophantic_count": 2, "total": 8}},
        },
    )
    monkeypatch.setattr(
        module,
        "evaluate_mmlu_with_steering",
        lambda *args, **kwargs: {
            "accuracy": 0.5,
            "accuracy_ci": [0.2, 0.8],
            "correct": 1,
            "n_evaluated": 2,
            "per_example": [{"id": 0, "correct": 1}, {"id": 1, "correct": 0}],
        },
    )
    monkeypatch.setattr(
        module,
        "evaluate_gsm8k_with_steering",
        lambda *args, **kwargs: {
            "accuracy": 0.5,
            "accuracy_ci": [0.2, 0.8],
            "correct": 1,
            "n_evaluated": 2,
            "per_example": [{"id": 0, "correct": 1}, {"id": 1, "correct": 0}],
        },
    )

    checkpoint = tmp_path / "steering.checkpoint.json"
    model = DummyModel()
    conditions_1, _, _ = module.run_steering_pipeline(
        model=model,
        dataset=dataset,
        layers=[1, 2],
        alphas=[0.5],
        n_steering_samples=2,
        eval_capabilities=True,
        checkpoint_path=str(checkpoint),
        save_every_condition=True,
        resume_from_checkpoint=False,
    )
    assert checkpoint.exists()
    first_count = len(conditions_1)

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("Sycophancy condition was recomputed despite checkpoint resume.")

    monkeypatch.setattr(module, "evaluate_sycophancy_with_steering", fail_if_recomputed)
    conditions_2, _, _ = module.run_steering_pipeline(
        model=model,
        dataset=dataset,
        layers=[1, 2],
        alphas=[0.5],
        n_steering_samples=2,
        eval_capabilities=False,
        checkpoint_path=str(checkpoint),
        save_every_condition=True,
        resume_from_checkpoint=True,
    )
    assert len(conditions_2) == first_count
