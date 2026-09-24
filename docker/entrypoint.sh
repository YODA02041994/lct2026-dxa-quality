#!/bin/sh
# dxaqc run -i /in -o /out [--format xlsx|csv|both]   — пакетная обработка
# dxaqc serve [--port 8000]                            — HTTP-сервис (страница + Swagger)
set -e
cmd="${1:-serve}"; shift || true
case "$cmd" in
  run)   exec python -m dxaqc run "$@" ;;
  serve) port=8000; while [ $# -gt 0 ]; do case "$1" in --port) port="$2"; shift;; esac; shift; done
         exec uvicorn dxaqc.api:app --host 0.0.0.0 --port "$port" ;;
  *)     exec "$cmd" "$@" ;;
esac
