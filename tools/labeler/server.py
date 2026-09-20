#!/usr/bin/env python3
"""Сервер разметки ориентиров: общая очередь для команды, автосохранение, бронь снимка, вход по паролю.

Логика очереди:
  • 10 «общих» снимков (stage=overlap) размечает КАЖДЫЙ — по ним считаем согласованность врачей;
  • остальные — общая очередь: снимок, помеченный «Готово» кем угодно, исчезает у всех;
  • выданный снимок бронируется за человеком на LEASE_MIN минут, чтобы двое не делали одно и то же.

Настройки (переменные окружения):
  LABEL_DATA      папка с png/ и images.json (только чтение)      по умолчанию data/work/labeler
  LABEL_ANN       папка для разметки (запись)                      по умолчанию <LABEL_DATA>/ann
  LABEL_PASSWORD  пароль входа; не задан → вход свободный (локальная работа)
  LABEL_SECRET    секрет подписи cookie; не задан → создаётся и хранится в LABEL_ANN/.secret
  LABEL_NAMES     имена разметчиков через запятую (кнопки выбора)
  LABEL_HOST / LABEL_PORT   по умолчанию 127.0.0.1:8877

Все адреса на странице относительные — сервис работает под любым префиксом (например /razmetka/ за nginx).
Запуск локально:  python tools/labeler/server.py  →  http://127.0.0.1:8877
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from urllib.parse import parse_qs

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("LABEL_DATA", os.path.join(ROOT, "data", "work", "labeler"))
ANN = os.environ.get("LABEL_ANN", os.path.join(DATA, "ann"))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PASSWORD = os.environ.get("LABEL_PASSWORD", "")
NAMES = [n.strip() for n in os.environ.get("LABEL_NAMES", "Андрей,Александр,Саак").split(",") if n.strip()]
LEASE_MIN = 15
COOKIE = "dxa_auth"
COOKIE_DAYS = 30
os.makedirs(ANN, exist_ok=True)


def _secret() -> bytes:
    env = os.environ.get("LABEL_SECRET")
    if env:
        return env.encode()
    p = os.path.join(ANN, ".secret")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(secrets.token_hex(32))
        os.chmod(p, 0o600)
    return open(p).read().strip().encode()


SECRET = _secret()
app = FastAPI(title="DXA labeler", docs_url=None, redoc_url=None, openapi_url=None)
_lock = threading.Lock()
_leases: dict[str, tuple[str, float]] = {}       # image_id → (разметчик, истекает)
_fails: dict[str, list[float]] = {}               # ip → времена неудачных входов


# ---------------------------------------------------------------- вход по паролю
def _sign(exp: int) -> str:
    return f"{exp}.{hmac.new(SECRET, str(exp).encode(), hashlib.sha256).hexdigest()}"


def _authed(request: Request) -> bool:
    if not PASSWORD:
        return True
    try:
        exp, sig = request.cookies.get(COOKIE, "").split(".", 1)
        return int(exp) > time.time() and hmac.compare_digest(_sign(int(exp)).split(".", 1)[1], sig)
    except (ValueError, AttributeError):
        return False


def _ip(request: Request) -> str:
    return (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()


@app.middleware("http")
async def _guard(request: Request, call_next):
    path = request.url.path
    if path.startswith(("/api", "/img")) and not _authed(request):
        return JSONResponse({"detail": "нужен вход"}, status_code=401)
    resp = await call_next(request)
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if not path.startswith("/img"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index(request: Request):
    return FileResponse(os.path.join(STATIC, "index.html" if _authed(request) else "login.html"))


@app.post("/login")
async def login(request: Request):
    ip, now = _ip(request), time.time()
    recent = [t for t in _fails.get(ip, []) if now - t < 600]
    if len(recent) >= 8:
        return RedirectResponse("./?e=wait", status_code=303)
    form = parse_qs((await request.body()).decode("utf-8", "ignore"))
    given = (form.get("password") or [""])[0]
    if PASSWORD and hmac.compare_digest(given.encode(), PASSWORD.encode()):
        _fails.pop(ip, None)
        exp = int(now + COOKIE_DAYS * 86400)
        resp = RedirectResponse("./", status_code=303)
        resp.set_cookie(COOKIE, _sign(exp), max_age=COOKIE_DAYS * 86400, httponly=True, samesite="lax",
                        secure=request.headers.get("x-forwarded-proto") == "https")
        return resp
    _fails[ip] = recent + [now]
    time.sleep(1.0)
    return RedirectResponse("./?e=1", status_code=303)


@app.post("/logout")
def logout():
    resp = RedirectResponse("./", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp


# ---------------------------------------------------------------- данные
def _images() -> list[dict]:
    with open(os.path.join(DATA, "images.json"), encoding="utf-8") as f:
        return json.load(f)


def _clean(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_\- ]", "", (name or "").strip())[:40].strip()
    if not name:
        raise HTTPException(400, "не выбран разметчик")
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


@app.get("/img/{iid}.png")
def img(iid: str):
    p = os.path.join(DATA, "png", iid + ".png")
    if not re.fullmatch(r"\d{3}_(spine|hipR|hipL)", iid) or not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


@app.get("/imgv/{variant}/{iid}.jpg")
def img_view(variant: str, iid: str):
    """Увеличенные версии для показа: sharp — чёткая, contrast — с локальным контрастом."""
    p = os.path.join(DATA, "view", variant, iid + ".jpg")
    if variant not in ("sharp", "contrast") or not re.fullmatch(r"\d{3}_(spine|hipR|hipL)", iid) or not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/config")
def config():
    return {"names": NAMES, "lease_min": LEASE_MIN, "password": bool(PASSWORD)}


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
    return {"ids": sorted(i for i, s in _done_map().items() if who in s)}


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
        "points": payload.get("points", [])[:60], "boxes": payload.get("boxes", [])[:30],
        "absent": payload.get("absent", [])[:20], "fuzzy": payload.get("fuzzy", [])[:20], "unsure": bool(payload.get("unsure")),
        "comment": str(payload.get("comment", ""))[:500], "done": bool(payload.get("done")),
        "seconds": payload.get("seconds"), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": "labeler-2",
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
                log_level="warning", proxy_headers=True, forwarded_allow_ips="127.0.0.1")
