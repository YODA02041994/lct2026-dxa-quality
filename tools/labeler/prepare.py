#!/usr/bin/env python3
"""Подготовка к разметке ориентиров: DICOM → PNG + список снимков с порядком показа.

Порядок: сначала 10 «общих» снимков (их размечает КАЖДЫЙ — по ним меряем согласованность врачей),
затем остальная стартовая партия (seed, ~50 снимков — на ней обучается сеть-подсказчик), затем все прочие.
В стартовую партию взяты все редкие нарушения и разнообразные нормы.
    PYTHONPATH=src python tools/labeler/prepare.py
"""
import csv, json, os, random, sys
import cv2
import pydicom
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "work", "labeler")
SP = "Поясничный отдел позвоночника"

rows = list(csv.DictReader(open(os.path.join(ROOT, "data/work/manifest.csv"), encoding="utf-8-sig")))
os.makedirs(os.path.join(OUT, "png"), exist_ok=True)
for v in ("sharp", "contrast"):
    os.makedirs(os.path.join(OUT, "view", v), exist_ok=True)
UPSCALE = 4
items = []
for r in rows:
    kind = "spine" if r["region"] == SP else "hip"
    iid = f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"
    a = pydicom.dcmread(os.path.join(ROOT, r["path"]), force=True).pixel_array
    Image.fromarray(a).save(os.path.join(OUT, "png", iid + ".png"))
    # Версии для показа разметчику: ×4 Lanczos + лёгкое подчёркивание краёв. Координаты разметки остаются
    # в пикселях ИСХОДНОГО снимка — увеличение влияет только на то, как человек видит картинку.
    big = (a.shape[1] * UPSCALE, a.shape[0] * UPSCALE)
    sharp = ImageFilter.UnsharpMask(radius=2.5, percent=90, threshold=2)
    Image.fromarray(a).resize(big, Image.LANCZOS).filter(sharp).save(os.path.join(OUT, "view", "sharp", iid + ".jpg"), quality=93)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(a)          # локальный контраст: лучше видны вертелы и контуры
    Image.fromarray(eq).resize(big, Image.LANCZOS).filter(sharp).save(os.path.join(OUT, "view", "contrast", iid + ".jpg"), quality=93)
    rare = any(r[c] == "1" for c in ("spine_positioning", "spine_axis_tilt", "hip_roi_field"))
    items.append({"id": iid, "kind": kind, "side": r["side"], "w": int(r["cols"]), "h": int(r["rows"]),
                  "_rare": rare, "_viol": r["truth"] == "1"})

rng = random.Random(20260920)
def pick(kind, n_rare, n_viol, n_norm):
    pool = [i for i in items if i["kind"] == kind]; rng.shuffle(pool)
    rare = [i for i in pool if i["_rare"]][:n_rare]
    viol = [i for i in pool if i["_viol"] and i not in rare][:n_viol]
    norm = [i for i in pool if not i["_viol"]][:n_norm]
    return rare + viol + norm
seed = pick("spine", 12, 4, 6) + pick("hip", 7, 11, 12)          # ~22 позвоночника + ~30 бёдер
rng.shuffle(seed)
overlap = [i for i in seed if i["kind"] == "spine"][:5] + [i for i in seed if i["kind"] == "hip"][:5]
for i in items:
    i["stage"] = "overlap" if i in overlap else ("seed" if i in seed else "rest")
order = {"overlap": 0, "seed": 1, "rest": 2}
items.sort(key=lambda i: (order[i["stage"]], i["id"]))
for i in items:
    i.pop("_rare"); i.pop("_viol")          # разметчик не должен видеть экспертную метку качества
json.dump(items, open(os.path.join(OUT, "images.json"), "w"), ensure_ascii=False, indent=0)
print("снимков:", len(items), {s: sum(i["stage"] == s for i in items) for s in order},
      "| позвоночник", sum(i["kind"] == "spine" for i in items), "| бедро", sum(i["kind"] == "hip" for i in items))
