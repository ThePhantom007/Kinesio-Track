#!/usr/bin/env bash
# scripts/download_models.sh
#
# Downloads the MediaPipe PoseLandmarker model weights into pose_engine/models/.
# Run automatically during Docker build (see docker/Dockerfile.api).
# Safe to run manually — idempotent, skips download if the file already exists.
#
# Usage:
#   bash scripts/download_models.sh               # full model (default)
#   bash scripts/download_models.sh lite          # lite model
#   bash scripts/download_models.sh heavy         # heavy model
#
# Model variants:
#   full  (~10 MB) — 33 landmarks, best accuracy for physiotherapy.  DEFAULT.
#   lite  (~3 MB)  — faster, slightly lower accuracy.
#   heavy (~25 MB) — highest accuracy, slowest.

set -euo pipefail

VARIANT="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_DIR="${PROJECT_ROOT}/pose_engine/models"
MODEL_FILE="${MODELS_DIR}/pose_landmarker.task"

# Google CDN base URL for MediaPipe models
BASE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker"

case "$VARIANT" in
  full)
    URL="${BASE_URL}/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ;;
  lite)
    URL="${BASE_URL}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ;;
  heavy)
    URL="${BASE_URL}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ;;
  *)
    echo "ERROR: Unknown variant '${VARIANT}'. Choose: full | lite | heavy" >&2
    exit 1
    ;;
esac

echo "→ Project root: ${PROJECT_ROOT}"
echo "→ Models dir:   ${MODELS_DIR}"

mkdir -p "$MODELS_DIR"

if [ -f "$MODEL_FILE" ]; then
  SIZE=$(wc -c < "$MODEL_FILE")
  echo "✓ Model already present at ${MODEL_FILE} (${SIZE} bytes). Skipping download."
  exit 0
fi

echo "→ Downloading MediaPipe PoseLandmarker (${VARIANT}) from Google CDN..."
echo "  URL:  ${URL}"
echo "  Dest: ${MODEL_FILE}"

if command -v curl &>/dev/null; then
  curl -fsSL --progress-bar -o "$MODEL_FILE" "$URL"
elif command -v wget &>/dev/null; then
  wget -q --show-progress -O "$MODEL_FILE" "$URL"
else
  echo "ERROR: Neither curl nor wget found. Install one and retry." >&2
  exit 1
fi

SIZE=$(wc -c < "$MODEL_FILE")
echo "✓ Download complete: ${MODEL_FILE} (${SIZE} bytes)"