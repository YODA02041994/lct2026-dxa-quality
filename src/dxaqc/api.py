"""HTTP-сервис поверх того же конвейера, что и CLI: загрузил DICOM/zip → таблица официального формата + объяснение.

    uvicorn dxaqc.api:app --host 0.0.0.0 --port 8000        (PYTHONPATH=src)
Swagger: /docs. Страница для человека: /. Модель грузится один раз при старте.

POST /api/analyze  (multipart: files=... ×N — .dcm или .zip)  → {run_id, summary, rows[]}
GET  /api/runs/{run_id}/results.xlsx | results.csv           → файлы официального формата
GET  /api/runs/{run_id}/overlay/{n}.png                       → снимок n с ориентирами и вердиктом
GET  /health                                                  → {"status": "ok", "model": ..., "version": ...}
Прогоны живут в RUNS_DIR (по умолчанию временная папка) не дольше RUN_TTL_MIN минут.
За обратным прокси с префиксом (nginx: location /dxa/ → 127.0.0.1:8878/) запускать `uvicorn ... --root-path /dxa`:
страница ходит по относительным ссылкам, Swagger берёт префикс из root_path.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid

import pydicom
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import __version__
from .infer import OFFICIAL, _extract_zip, _pixels, run
from .predict import load_default_predictor
from .viz import draw_overlay, encode_png

log = logging.getLogger("dxaqc.api")
RUNS_DIR = os.environ.get("DXAQC_RUNS", os.path.join(tempfile.gettempdir(), "dxaqc_runs"))
RUN_TTL_MIN = int(os.environ.get("DXAQC_RUN_TTL_MIN", "180"))
MAX_UPLOAD_MB = int(os.environ.get("DXAQC_MAX_UPLOAD_MB", "500"))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FILES_DIR = os.environ.get("DXAQC_FILES", "")            # необязательная папка с материалами (презентация, демо-запись)

app = FastAPI(title="DXA quality control", version=__version__,
              description="Контроль качества денситометрии: область → нарушение → тип (ЛЦТ-2026, задача №4)")
_predictor = None
_lock = threading.Lock()


def predictor():
    global _predictor
    if _predictor is None:
        _predictor = load_default_predictor()
        log.info("модель загружена: %s", _predictor.name)
    return _predictor


def _cleanup():
    now = time.time()
    if not os.path.isdir(RUNS_DIR):
        return
    for d in os.listdir(RUNS_DIR):
        p = os.path.join(RUNS_DIR, d)
        try:
            if now - os.path.getmtime(p) > RUN_TTL_MIN * 60:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


@app.on_event("startup")
def _startup():
    os.makedirs(RUNS_DIR, exist_ok=True)
    threading.Thread(target=predictor, daemon=True).start()     # прогрев весов, не блокируя старт


@app.get("/health")
def health():
    p = predictor()
    return {"status": "ok", "model": p.name, "version": __version__, "weights_loaded": p.name != "null"}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/files/{name}")
def files(name: str):
    """Материалы решения (презентация, демо-запись), если задана папка DXAQC_FILES."""
    path = os.path.join(FILES_DIR, os.path.basename(name))
    if not FILES_DIR or not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path, filename=os.path.basename(name))


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    _cleanup()
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    rdir = os.path.join(RUNS_DIR, run_id)
    inp, out = os.path.join(rdir, "input"), os.path.join(rdir, "output")
    os.makedirs(inp)
    total = 0
    for uf in files:
        data = await uf.read()
        total += len(data)
        if total > MAX_UPLOAD_MB * 1024 * 1024:
            shutil.rmtree(rdir, ignore_errors=True)
            raise HTTPException(413, f"суммарный размер больше {MAX_UPLOAD_MB} МБ")
        name = os.path.basename(uf.filename or "file")
        path = os.path.join(inp, name)
        with open(path, "wb") as f:
            f.write(data)
        if name.lower().endswith(".zip"):
            sub = os.path.join(inp, name[:-4])
            os.makedirs(sub, exist_ok=True)
            try:
                _extract_zip(path, sub)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"архив {name} не распакован: {exc}") from exc
            os.remove(path)
    with _lock:                                                   # модель одна, MPS/CUDA не любят гонки
        rows, s = run(inp, out, predictor(), fmt="both")
    for i, r in enumerate(rows):
        r["n"] = i
        r["input_path"] = os.path.join(inp, r["path_to_study"])
    json.dump(rows, open(os.path.join(rdir, "rows.json"), "w"), ensure_ascii=False)
    public = [{k: r[k] for k in OFFICIAL + ["side", "duplicate_of", "error_message", "n"]} | {"details": json.loads(r["details"]) if r["details"] else None} for r in rows]
    return {"run_id": run_id, "summary": s.__dict__ | {"xlsx": f"api/runs/{run_id}/results.xlsx", "csv": f"api/runs/{run_id}/results.csv"}, "rows": public}


def _rows(run_id: str) -> list[dict]:
    p = os.path.join(RUNS_DIR, run_id, "rows.json")
    if not os.path.isfile(p) or "/" in run_id or ".." in run_id:
        raise HTTPException(404, "прогон не найден или истёк")
    return json.load(open(p, encoding="utf-8"))


@app.get("/api/runs/{run_id}/results.{ext}")
def results(run_id: str, ext: str):
    if ext not in ("xlsx", "csv"):
        raise HTTPException(404)
    _rows(run_id)
    p = os.path.join(RUNS_DIR, run_id, "output", f"results.{ext}")
    return FileResponse(p, filename=f"results.{ext}")


@app.get("/api/runs/{run_id}/series.zip")
def series_zip(run_id: str):
    """Zip с дополнительными сериями прогона: DICOM-серия с разметкой (Secondary Capture), DICOM SR, PNG."""
    from .infer import pack_series, write_overlays, write_sr
    rows = _rows(run_id)
    rdir = os.path.join(RUNS_DIR, run_id)
    out = os.path.join(rdir, "output")
    path = os.path.join(out, "additional_series.zip")
    if not os.path.isfile(path):
        with _lock:
            write_overlays(rows, os.path.join(rdir, "input"), out, predictor().thresholds)
            write_sr(rows, os.path.join(rdir, "input"), out)
            pack_series(out)
    return FileResponse(path, filename="additional_series.zip")


@app.get("/api/runs/{run_id}/overlay/{n}.png")
def overlay(run_id: str, n: int):
    rows = _rows(run_id)
    if n < 0 or n >= len(rows):
        raise HTTPException(404)
    r = rows[n]
    if r["processing_status"] != "Success":
        raise HTTPException(422, r.get("error_message") or "Failure")
    src = r if r["duplicate_of"] == "" else next((x for x in rows if x["path_to_study"] == r["duplicate_of"]), r)
    details = json.loads(src["details"]) if src["details"] else {}
    img = _pixels(pydicom.dcmread(r["input_path"]))
    bgr = draw_overlay(img, r["anatomical_region"], details,
                       {"quality_class": r["quality_class"], "violation_type": r["violation_type"], "quality_prob": r["quality_prob"]},
                       predictor().thresholds)
    return Response(encode_png(bgr), media_type="image/png")


@app.exception_handler(Exception)
async def _any_error(_, exc: Exception):
    log.exception("api error")
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
