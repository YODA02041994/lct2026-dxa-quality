#!/usr/bin/env bash
# Скачивает веса модели из GitHub Release в weights/ (в git они не лежат: 18 файлов, 0,9 ГБ + FleXray 0,4 ГБ).
#   scripts/download_weights.sh            # тег по умолчанию — см. TAG
#   TAG=… scripts/download_weights.sh      # другой релиз; список файлов ниже относится к v1.0.0
# Пока репозиторий приватный, нужен авторизованный `gh`; после публикации хватит обычного curl.
set -euo pipefail
REPO="YODA02041994/lct2026-dxa-quality"
TAG="${TAG:-v1.1.0}"
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
# FleXray (MIT CSAIL, CC-BY-NC-4.0) — необязательный модуль «сколиоз» и голос за область; в релизе файлы лежат с префиксом flexray_
mkdir -p "$DIR/flexray"
for f in model.safetensors config.yml label_schema.json preprocessing.json; do
  echo "→ flexray/$f"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --dir "$DIR/flexray" --clobber --pattern "flexray_$f" && mv -f "$DIR/flexray/flexray_$f" "$DIR/flexray/$f"
  else
    curl -fL --retry 3 -o "$DIR/flexray/$f" "https://github.com/$REPO/releases/download/$TAG/flexray_$f"
  fi
done
ls -la "$DIR"
echo "готово: $(ls "$DIR" | wc -l | tr -d ' ') файлов в $DIR"
