#!/usr/bin/env bash
# Publish trained model files as GitHub release assets (never commit weights to git).
# Usage: scripts/release_models.sh v0.1.0 [DATA_DIR]
set -euo pipefail
TAG="${1:?tag like v0.1.0}"
DATA_DIR="${2:-${PTAA_DATA_DIR:-./data}}"
command -v gh >/dev/null || { echo "install GitHub CLI: https://cli.github.com"; exit 1; }
FILES=$(find "$DATA_DIR/models" -type f \( -name "*.txt" -o -name "*.onnx" -o -name "*.json" \) | sort)
[ -n "$FILES" ] || { echo "no model files under $DATA_DIR/models"; exit 1; }
gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" --title "$TAG models" --notes "Model files for $TAG. Verify with: ptaa models verify <id>. Paper trading only; not investment advice."
# shellcheck disable=SC2086
gh release upload "$TAG" $FILES --clobber
echo "uploaded:"; echo "$FILES"
echo "on the laptop: gh release download $TAG --dir \$PTAA_DATA_DIR/models && ptaa models list"
