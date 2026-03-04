"""
Tests for probe labeling and leakage-safe split assumptions.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold


def load_probe_script_module():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    script_path = Path(__file__).parent.parent / "scripts" / "02_train_probes.py"
    spec = importlib.util.spec_from_file_location("probe_train_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_answer_identity_labels_use_option_metadata():
    module = load_probe_script_module()
    valid_samples = [
        {
            "honest_target": " (A)",
            "sycophantic_target": " (B)",
            "metadata": {"source": "truthfulqa_factual", "honest_option": "A", "sycophantic_option": "B"},
        },
        {
            "honest_target": " (B)",
            "sycophantic_target": " (A)",
            "metadata": {"source": "gsm8k_reasoning", "honest_option": "B", "sycophantic_option": "A"},
        },
        {
            "honest_target": "no",
            "sycophantic_target": "yes",
            "metadata": {"source": "legacy_source"},
        },
    ]

    labels, stats = module.create_answer_identity_labels(valid_samples)
    assert labels.tolist() == [0, 1, 1]
    assert stats["overall"]["n_total"] == 3
    assert stats["per_source"]["truthfulqa_factual"]["n_total"] == 1


def test_group_kfold_keeps_paired_sample_ids_together():
    sample_ids = np.array(["s1", "s2", "s3", "s4"])
    groups = np.array(list(sample_ids) + list(sample_ids))
    X = np.zeros((len(groups), 1))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    splitter = GroupKFold(n_splits=2)
    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        assert train_groups.isdisjoint(val_groups)
