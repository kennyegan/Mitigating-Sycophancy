"""
CLI contract checks via source inspection (dependency-light).
"""

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _script_text(name: str) -> str:
    return (REPO_ROOT / "scripts" / name).read_text()


def test_data_setup_has_randomize_positions_flag():
    text = _script_text("00_data_setup.py")
    assert "--randomize-positions" in text


def test_probe_train_has_analysis_mode_flag():
    text = _script_text("02_train_probes.py")
    assert "--analysis-mode" in text
    assert "neutral_transfer" in text
    assert "mixed_diagnostic" in text


def test_steering_has_checkpoint_resume_flags():
    text = _script_text("05_representation_steering.py")
    assert "--checkpoint-path" in text
    assert "--resume-from-checkpoint" in text
    assert "--save-every-condition" in text
