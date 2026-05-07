#!/usr/bin/env bash
# Build the anonymous-supplement zip after running every check.
#
# Usage: bash release/make_release_zip.sh
#
# Aborts and does not produce a zip if any of the three checks fails.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

OUT_ZIP="$HERE/../neurips2026_anonymous_supplement.zip"

echo "==> Running anonymization check"
python tools/check_anonymization.py

echo "==> Running manifest check"
python scripts/check_manifest.py

echo "==> Running claim verification"
python scripts/verify_claims.py

echo "==> All checks passed; building zip $OUT_ZIP"

# Excludes: caches, model weights, transcripts, git, hidden metadata.
# Note: relative to $HERE which is the supplement folder root.
zip -r "$OUT_ZIP" . \
    -x ".git/*" \
       "*/.git/*" \
       "*/__pycache__/*" \
       "__pycache__/*" \
       ".pytest_cache/*" \
       "*/.DS_Store" \
       "*.pyc" \
       "*.pyo" \
       "*.so" \
       "*.safetensors" \
       "*/checkpoints/*" \
       "*/dpo_model*/*" \
       "*/sft_model/*" \
       "wandb/*" \
       ".venv/*" \
       "*.log" \
       "release/*.zip"

# Summary
n_files=$(unzip -l "$OUT_ZIP" | tail -1 | awk '{print $2}')
size=$(du -h "$OUT_ZIP" | awk '{print $1}')

echo ""
echo "==> Release zip ready"
echo "    file:  $OUT_ZIP"
echo "    size:  $size"
echo "    files: $n_files"
