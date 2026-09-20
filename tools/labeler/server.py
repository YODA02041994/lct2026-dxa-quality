#!/usr/bin/env python3
"""Сервер разметки ориентиров: общая очередь для команды, автосохранение, бронь снимка за разметчиком.

Логика очереди:
  • 10 «общих» снимков (stage=overlap) размечает КАЖДЫЙ — по ним считаем согласованность врачей;
  • остальные — общая очередь: снимок, помеченный «Готово» кем угодно, исчезает у всех;
  • выданный снимок бронируется за человеком на LEASE_MIN минут, чтобы двое не делали одно и то же.
Разметка: data/work/labeler/ann/<id>__<разметчик>.json (в git не попадает).

Запуск локально:   python tools/labeler/server.py            → http://127.0.0.1:8877
С защитой ссылкой: LABEL_TOKEN=секрет python tools/labeler/server.py   → открывать как /?t=секрет
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("LABEL_DATA", os.path.join(ROOT, "data", "work", "labeler"))
ANN = os.path.join(DATA, "ann")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TOKEN = os.environ.get("LABEL_TOKEN", "")
LEASE_MIN = 15
os.makedirs(ANN, exist_ok=True)

app = FastAPI(title="DXA labeler")
_lock = threading.Lock()
_leases: dict[str, tuple[str, float]] = {}       # image_id → (разметчик, истекает)


def _images() -> list[dict]:
    with open(os.path.join(DATA, "images.json"), encoding="utf-8") as f:
        return json.load(f)


def _clean(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_\- ]", "", (name or "").strip())[:40].strip()
    if not name:
        raise HTTPException(400, "укажите имя разметчика")
    return name.replace(" ", "_")


def _ann_path(iid: str, who: str) -> str:
    if not re.fullmatch(r"\d{3}_(spine|hipR|hipL)", iid):
        raise HTTPException(400, "неверный id снимка")
    return os.path.join(ANN, f"{iid}__{who}.json")


def _done_map() -> dict[str, set[str]]:
    """image_id → кто пометил «Готово»."""
    out: dict[str, set[str]] = {}
    for fn in os.listdir(ANN):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(ANN, fn), encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if d.get("done"):
            out.setdefault(d["image_id"], set()).add(d["annotator"])
    return out


@app.middleware("http")
async def _auth(request: Request, call_next):
    if TOKEN and request.url.path.startswith(("/api", "/img")):
        if request.query_params.get("t") != TOKEN and request.headers.get("x-label-token") != TOKEN:
            return JSONResponse({"detail": "нужна ссылка с ключом доступа"}, status_code=401)
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/img/{iid}.png")
def img(iid: str):
    p = os.path.join(DATA, "png", iid + ".png")
    if not re.fullmatch(r"\d{3}_(spine|hipR|hipL)", iid) or not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


def _state(who: str) -> dict:
    imgs, done = _images(), _done_map()
    now = time.time()
    queue = [i for i in imgs if i["stage"] != "overlap"]
    overlap = [i for i in imgs if i["stage"] == "overlap"]
    people: dict[str, int] = {}
    for s in done.values():
        for p in s:
            people[p] = people.get(p, 0) + 1
    busy = {k: v[0] for k, v in _leases.items() if v[1] > now and v[0] != who and k not in done}
    return {
        "total": len(imgs), "queue_total": len(queue),
        "queue_done": sum(1 for i in queue if i["id"] in done),
        "overlap_total": len(overlap),
        "overlap_mine": sum(1 for i in overlap if who in done.get(i["id"], ())),
        "mine": people.get(who, 0), "people": people, "busy": busy,
    }


@app.get("/api/state")
def state(annotator: str = Query(...)):
    return _state(_clean(annotator))


@app.get("/api/next")
def next_image(annotator: str = Query(...), kind: str = Query("any")):
    who = _clean(annotator)
    with _lock:
        imgs, done, now = _images(), _done_map(), time.time()
        cand = None
        for i in imgs:                                   # 1) общие снимки — каждому свои
            if i["stage"] == "overlap" and who not in done.get(i["id"], ()) and kind in ("any", i["kind"]):
                cand = i
                break
        if cand is None:                                 # 2) общая очередь
            for i in imgs:
                if i["stage"] == "overlap" or i["id"] in done or kind not in ("any", i["kind"]):
                    continue
                lease = _leases.get(i["id"])
                if lease and lease[1] > now and lease[0] != who:
                    continue
                cand = i
                break
        if cand is None:
            return {"image": None, "state": _state(who)}
        _leases[cand["id"]] = (who, now + LEASE_MIN * 60)
        prev = None
        p = _ann_path(cand["id"], who)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                prev = json.load(f)
        return {"image": cand, "annotation": prev, "state": _state(who)}


@app.get("/api/mine")
def mine(annotator: str = Query(...)):
    who = _clean(annotator)
    ids = sorted(i for i, s in _done_map().items() if who in s)
    return {"ids": ids}


@app.get("/api/open")
def open_image(annotator: str = Query(...), id: str = Query(...)):
    who = _clean(annotator)
    im = next((i for i in _images() if i["id"] == id), None)
    if not im:
        raise HTTPException(404)
    p, prev = _ann_path(id, who), None
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            prev = json.load(f)
    return {"image": im, "annotation": prev, "state": _state(who)}


@app.post("/api/save")
def save(payload: dict = Body(...)):
    who = _clean(payload.get("annotator", ""))
    iid = str(payload.get("image_id", ""))
    path = _ann_path(iid, who)
    rec = {
        "image_id": iid, "annotator": who, "kind": payload.get("kind"),
        "points": payload.get("points", []), "boxes": payload.get("boxes", []),
        "absent": payload.get("absent", []), "unsure": bool(payload.get("unsure")),
        "comment": str(payload.get("comment", ""))[:500], "done": bool(payload.get("done")),
        "seconds": payload.get("seconds"), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": "labeler-1",
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)                                 # атомарно: недописанных файлов не бывает
    with _lock:
        if rec["done"]:
            _leases.pop(iid, None)
        else:
            _leases[iid] = (who, time.time() + LEASE_MIN * 60)
    return {"ok": True, "state": _state(who)}


@app.post("/api/release")
def release(payload: dict = Body(...)):
    who = _clean(payload.get("annotator", ""))
    with _lock:
        lease = _leases.get(str(payload.get("image_id", "")))
        if lease and lease[0] == who:
            _leases.pop(str(payload.get("image_id")), None)
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("LABEL_HOST", "127.0.0.1"), port=int(os.environ.get("LABEL_PORT", "8877")),
                log_level="warning")
