#!/usr/bin/env bash
# Download model files from a GitHub release into the data dir and verify them.
set -euo pipefail
TAG="${1:?tag like v0.1.0}"
DATA_DIR="${2:-${PTAA_DATA_DIR:-./data}}"
mkdir -p "$DATA_DIR/models"
gh release download "$TAG" --dir "$DATA_DIR/models" --clobber
# files are flat in the release; move into their kind folders expected by the registry
cd "$DATA_DIR/models"
for f in gbm-*.txt gbm-*.bg.json; do [ -e "$f" ] && mkdir -p gbm && mv -f "$f" gbm/; done
for f in seq_lstm-*.onnx seq_lstm-*.meta.json; do [ -e "$f" ] && mkdir -p seq && mv -f "$f" seq/; done
for f in ppo-*.onnx; do [ -e "$f" ] && mkdir -p ppo && mv -f "$f" ppo/; done
ptaa models list
