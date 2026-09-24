#!/usr/bin/env bash
# Скачивает боевые веса из GitHub Release в weights/ (в git они не лежат: 7 файлов, ~360 МБ).
#   scripts/download_weights.sh            # тег по умолчанию — см. TAG
#   TAG=v0.2.0 scripts/download_weights.sh
# Пока репозиторий приватный, нужен авторизованный `gh`; после публикации хватит обычного curl.
set -euo pipefail
REPO="YODA02041994/lct2026-dxa-quality"
TAG="${TAG:-v0.2.0}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
FILES=(landmarks_spine.pt landmarks_hip.pt cnn_spine_artifact_r18_512.pt cnn_hip_positioning_rotation_eff_320.pt cnn_hip_any_r18_320.pt cnn_hip_positioning_rotation_lt100.pt cnn_hip_positioning_rotation_lt100_e40.pt cnn_hip_positioning_rotation_isch100.pt cnn_hip_positioning_rotation_prox170.pt cnn_hip_any_any_lt100.pt criteria.json)
mkdir -p "$DIR"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --dir "$DIR" --clobber
else
  for f in "${FILES[@]}"; do
    echo "→ $f"
    curl -fL --retry 3 -o "$DIR/$f" "https://github.com/$REPO/releases/download/$TAG/$f"
  done
fi
ls -la "$DIR"
echo "готово: $(ls "$DIR" | wc -l | tr -d ' ') файлов в $DIR"
