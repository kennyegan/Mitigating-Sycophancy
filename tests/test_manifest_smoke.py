"""
Smoke test for consolidated result-manifest generator.
"""

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def test_collect_result_manifest_runs_with_allow_missing(tmp_path):
    output = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "99_collect_result_manifest.py"),
            "--allow-missing",
            "--output",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    assert output.exists()
    data = json.loads(output.read_text())
    assert "artifacts" in data
    assert isinstance(data["artifacts"], list)
