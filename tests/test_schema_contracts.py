"""
Regression checks for result-schema fields in experiment scripts.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _script_text(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text()


def test_baseline_schema_fields_present():
    text = _script_text("01_run_baseline.py")
    assert "schema_version" in text
    assert "analysis_mode" in text
    assert "split_definition" in text


def test_probe_schema_fields_present():
    text = _script_text("02_train_probes.py")
    assert "schema_version" in text
    assert "analysis_mode" in text
    assert "split_definition" in text
    assert "biased_transfer_accuracy_ci" in text


def test_probe_control_schema_fields_present():
    text = _script_text("02b_probe_control.py")
    assert "analysis_mode" in text
    assert "schema_version" in text


def test_patching_schema_fields_present():
    text = _script_text("03_activation_patching.py")
    assert "schema_version" in text
    assert "analysis_mode" in text
    assert "split_definition" in text


def test_ablation_schema_and_ci_fields_present():
    text = _script_text("04_head_ablation.py")
    assert "schema_version" in text
    assert "analysis_mode" in text
    assert "split_definition" in text
    assert "mmlu_retained_ci" in text
    assert "gsm8k_retained_ci" in text


def test_steering_schema_and_ci_fields_present():
    text = _script_text("05_representation_steering.py")
    assert "schema_version" in text
    assert "analysis_mode" in text
    assert "split_definition" in text
    assert "mmlu_retained_ci" in text
    assert "gsm8k_retained_ci" in text
