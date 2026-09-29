#!/usr/bin/env bash
# Сборка и запуск контейнера одним скриптом (Linux, macOS; нужен Docker 24+).
#   ./run.sh build                          скачать веса, если их нет, и собрать образ dxaqc
#   ./run.sh batch <вход> <выход> [флаги]   пакетная обработка: <вход> — папка с DICOM или zip-архив
#                                           флаги: --overlays --sr --mode sensitive --format xlsx|csv|both
#   ./run.sh serve [порт]                   веб-страница и API: http://localhost:<порт> (по умолчанию 8000)
#   ./run.sh save [файл]                    выгрузить образ в архив для машины без интернета (docker load < файл)
# Образ собирается сам при первом вызове batch или serve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${DXAQC_IMAGE:-dxaqc}"
# Проверенная платформа — linux/amd64. На ARM-машине: DXAQC_PLATFORM=linux/amd64 ./run.sh build (образ работает через эмуляцию).
PLAT=(); [ -n "${DXAQC_PLATFORM:-}" ] && PLAT=(--platform "$DXAQC_PLATFORM")

build() {
  if [ ! -f "$ROOT/weights/criteria.json" ] || [ ! -f "$ROOT/weights/landmarks_hip.pt" ]; then
    "$ROOT/scripts/download_weights.sh"
  fi
  docker build ${PLAT[@]+"${PLAT[@]}"} -t "$IMAGE" "$ROOT"
}

ensure_image() { docker image inspect "$IMAGE" >/dev/null 2>&1 || build; }

abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }

cmd="${1:-help}"; shift || true
case "$cmd" in
  build) build ;;
  batch)
    [ $# -ge 2 ] || { echo "нужно: ./run.sh batch <вход> <выход> [флаги]" >&2; exit 2; }
    in="$(abspath "$1")"; mkdir -p "$2"; out="$(abspath "$2")"; shift 2
    [ -e "$in" ] || { echo "нет входа: $in" >&2; exit 2; }
    ensure_image
    if [ -d "$in" ]; then
      docker run --rm ${PLAT[@]+"${PLAT[@]}"} --network none -v "$in:/in:ro" -v "$out:/out" "$IMAGE" run -i /in -o /out "$@"
    else
      docker run --rm ${PLAT[@]+"${PLAT[@]}"} --network none -v "$(dirname "$in"):/in:ro" -v "$out:/out" "$IMAGE" run -i "/in/$(basename "$in")" -o /out "$@"
    fi
    echo "результаты: $out" ;;
  serve)
    ensure_image
    port="${1:-8000}"
    echo "страница: http://localhost:$port   Swagger: http://localhost:$port/docs"
    docker run --rm ${PLAT[@]+"${PLAT[@]}"} -p "$port:8000" "$IMAGE" serve ;;
  save)
    ensure_image
    file="${1:-dxaqc.tar.gz}"
    case "$IMAGE" in *:*) ref="$IMAGE" ;; *) ref="$IMAGE:latest" ;; esac      # один тег, иначе в архив попадут все теги образа
    docker save "$ref" | gzip > "$file"
    echo "образ сохранён: $file" ;;
  *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
