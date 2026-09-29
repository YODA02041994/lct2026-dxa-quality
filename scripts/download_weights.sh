#!/usr/bin/env bash
# Скачивает веса модели из GitHub Release в weights/ (в git они не лежат: 18 файлов, 0,9 ГБ).
#   scripts/download_weights.sh            # тег по умолчанию — см. TAG
#   TAG=… scripts/download_weights.sh      # другой релиз; список файлов ниже относится к v1.0.0
# Пока репозиторий приватный, нужен авторизованный `gh`; после публикации хватит обычного curl.
set -euo pipefail
REPO="YODA02041994/lct2026-dxa-quality"
TAG="${TAG:-v1.0.0}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/weights"
FILES=(landmarks_spine.pt landmarks_hip.pt cnn_spine_artifact_r18_320_ocxrF.pt cnn_spine_artifact_r18_320_ocxrF_s1.pt cnn_spine_artifact_r18_320_ocxrF_s2.pt cnn_hip_positioning_rotation_eff_320.pt cnn_hip_any_r18_320.pt cnn_hip_positioning_rotation_lt100.pt cnn_hip_positioning_rotation_lt100_e40.pt cnn_hip_positioning_rotation_isch100.pt cnn_hip_positioning_rotation_prox170.pt cnn_hip_any_any_lt100.pt cnn_hip_positioning_rotation_lt100_xrv.pt cnn_hip_positioning_rotation_isch100_xrv.pt cnn_hip_positioning_rotation_prox170_xrv.pt pretrain_objectcxr_full320_r18.pt objmap_objectcxr_r18fpn.pt criteria.json)
mkdir -p "$DIR"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  for f in "${FILES[@]}"; do                      # только файлы модели: в релизе могут лежать и другие материалы
    echo "→ $f"
    gh release download "$TAG" --repo "$REPO" --dir "$DIR" --clobber --pattern "$f"
  done
else
  for f in "${FILES[@]}"; do
    echo "→ $f"
    curl -fL --retry 3 -o "$DIR/$f" "https://github.com/$REPO/releases/download/$TAG/$f"
  done
fi
ls -la "$DIR"
echo "готово: $(ls "$DIR" | wc -l | tr -d ' ') файлов в $DIR"
