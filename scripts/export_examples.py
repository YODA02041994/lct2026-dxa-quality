#!/usr/bin/env python3
"""Папка наглядных примеров для человека: снимок + вердикт по-русски + точки/оси/рамки + протокол и совет.
Вердикт — из честных OOF-вероятностей (снимок вне обучения), рядом метка эксперта. Три подпапки:
подтверждённые нарушения, подтверждённая норма, ошибки модели (пропуски и ложные тревоги).
    PYTHONPATH=src python scripts/export_examples.py [папка]
"""
import csv, json, os, sys, textwrap
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dxaqc.labels import VIOLATION_TEXT
from dxaqc.criteria import explain
from dxaqc.cnn import ObjMapScorer
from dxaqc.predict import aggregate
from train_criteria import manifest, features_for
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/lct_task4/примеры_результатов")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"; FONTB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f_h, f_b, f_s = ImageFont.truetype(FONTB, 26), ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 15)
REGION_RU = {"spine": "Поясничный отдел позвоночника", "hip": "Проксимальный отдел бедра"}
SP = ["spine_positioning", "spine_axis_tilt", "spine_artifact"]; HP = ["hip_positioning_rotation", "hip_roi_field"]
man = manifest(); oof = json.load(open(os.path.join(WORK, "criteria_oof.json")))
cj = json.load(open(os.path.join(ROOT, "weights", "criteria.json"))); thr = {c: m["threshold"] for c, m in cj["models"].items()}
lms = {k: json.load(open(os.path.join(WORK, f"landmarks_oof_{k}.json"))) for k in ("spine", "hip")}
feats = {k: features_for(k, lms[k], man) for k in ("spine", "hip")}
objmap = ObjMapScorer(os.path.join(ROOT, "weights", "objmap_objectcxr_r18fpn.pt"))
SCALE = 3

def render(iid, subdir, tag):
    m = man[iid]; kind = m["kind"]; flags = SP if kind == "spine" else HP
    img = cv2.imread(os.path.join(PNG, iid + ".png"), 0); h, w = img.shape
    probs = {c: oof[iid][c] for c in flags if c in oof[iid]}; v = aggregate(probs, thr)
    truth = {c: m[c] for c in flags if m[c] is not None}; truth_any = int(any(truth.values()))
    f = feats[kind][iid]; comment, advice = explain(REGION_RU[kind], f, probs, thr)
    lm = lms[kind][iid]
    boxes = objmap.features(img)[1] if kind == "spine" else []
    # --- снимок с разметкой (OpenCV, латиница) ---
    can = cv2.cvtColor(cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    P = lambda k: (int(lm[k]["x"] * SCALE), int(lm[k]["y"] * SCALE)) if lm.get(k, {}).get("present") else None
    if kind == "spine":
        pts = [P(k) for k in ("L1", "L2", "L3", "L4", "L5") if P(k)]
        if len(pts) >= 2:
            xs, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float); k_, b_ = np.polyfit(ys, xs, 1)
            cv2.line(can, (int(b_), 0), (int(k_ * h * SCALE + b_), h * SCALE), (255, 200, 80), 2, cv2.LINE_AA)
            cv2.line(can, (int(xs.mean()), 0), (int(xs.mean()), h * SCALE), (150, 150, 150), 1, cv2.LINE_AA)
    else:
        if P("shaft_top") and P("shaft_bot"): cv2.line(can, P("shaft_top"), P("shaft_bot"), (255, 200, 80), 2, cv2.LINE_AA)
    for b in boxes:
        x0, y0 = b["x"] * SCALE, b["y"] * SCALE; cv2.rectangle(can, (x0, y0), (x0 + b["w"] * SCALE, y0 + b["h"] * SCALE), (0, 0, 255), 2)
    for k, val in lm.items():
        if not val.get("present"): continue
        p = (int(val["x"] * SCALE), int(val["y"] * SCALE)); cv2.circle(can, p, 4, (60, 220, 255), -1, cv2.LINE_AA)
    pil_img = Image.fromarray(cv2.cvtColor(can, cv2.COLOR_BGR2RGB))
    # --- рамка с русским текстом (PIL) ---
    W = max(pil_img.width, 760); top, bottom = 118, 170
    page = Image.new("RGB", (W, top + pil_img.height + bottom), (250, 247, 242)); d = ImageDraw.Draw(page)
    bad = v.quality_class == 1
    d.rectangle([0, 0, W, top], fill=(201, 56, 42) if bad else (46, 139, 87))
    title = ("НАРУШЕНИЕ: " + (v.violation_type or "")) if bad else "НОРМА — укладка корректна"
    d.text((14, 12), title, font=f_h, fill="white")
    d.text((14, 50), f"{REGION_RU[kind]}{' (' + m['side'] + ')' if m['side'] else ''} · снимок {iid} · вероятность нарушения {v.quality_prob:.2f}", font=f_b, fill="white")
    exp = "нарушение: " + "; ".join(VIOLATION_TEXT[c] for c in flags if truth.get(c) == 1) if truth_any else "норма"
    d.text((14, 78), f"Оценка эксперта организатора — {exp}   |   {tag}", font=f_b, fill="white")
    page.paste(pil_img, ((W - pil_img.width) // 2, top))
    y = top + pil_img.height + 10
    for line in textwrap.wrap("Протокол измерений: " + comment, 115)[:2]:
        d.text((14, y), line, font=f_s, fill=(31, 29, 26)); y += 19
    y += 3
    for c, p in probs.items():
        d.text((14, y), f"  {VIOLATION_TEXT[c]}: {p:.2f} (порог {thr[c]:.2f}){'  ← сработал' if p >= thr[c] else ''}", font=f_s, fill=(201, 56, 42) if p >= thr[c] else (107, 102, 95)); y += 19
    if advice:
        for line in textwrap.wrap("Совет лаборанту: " + " ".join(advice), 110)[:3]:
            d.text((14, y), line, font=f_s, fill=(199, 85, 61)); y += 19
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    path = os.path.join(OUT, subdir, f"{iid}.png"); page.save(path); return path

# --- отбор: подтверждённые нарушения по типам, норма, ошибки ---
picked = []
def verdict(iid):
    m = man[iid]; flags = SP if m["kind"] == "spine" else HP
    probs = {c: oof[iid][c] for c in flags if c in oof[iid]}; v = aggregate(probs, thr)
    truth_any = int(any(m[c] == 1 for c in flags if m[c] is not None)); return v, truth_any, probs, flags
tp = {c: [] for c in SP + HP}; tn = {"spine": [], "hip": []}; fn, fp = [], []
for iid in sorted(oof):
    if iid not in man or not man[iid]["labeled"]: continue
    v, ta, probs, flags = verdict(iid)
    if ta and v.quality_class:
        for c in flags:
            if man[iid][c] == 1 and probs.get(c, 0) >= thr[c]: tp[c].append(iid)
    elif not ta and not v.quality_class: tn[man[iid]["kind"]].append(iid)
    elif ta and not v.quality_class: fn.append(iid)
    else: fp.append(iid)
n = 0
for c in SP + HP:
    for iid in tp[c][:3]:
        render(iid, "1_нарушение_подтверждено", "модель права"); n += 1
for kind in ("spine", "hip"):
    for iid in tn[kind][:4]:
        render(iid, "2_норма_подтверждена", "модель права"); n += 1
for iid in fn[:3]: render(iid, "3_ошибки_модели", "ПРОПУСК модели"); n += 1
for iid in fp[:3]: render(iid, "3_ошибки_модели", "ЛОЖНАЯ ТРЕВОГА модели"); n += 1
open(os.path.join(OUT, "ЧИТАТЬ.txt"), "w").write(
"""Примеры результатов сервиса контроля качества DXA (ЛЦТ-2026, задача 4).

Каждая картинка: сверху вердикт модели (красная плашка — нарушение с официальным типом, зелёная — норма),
оценка эксперта организатора для того же снимка, ниже снимок с точками-ориентирами, осью (голубая линия),
рамками найденных посторонних предметов (красные), внизу протокол измерений, вероятности по критериям
и совет лаборанту.

Вердикты честные: вероятности взяты для снимков, которых модель не видела при обучении (кросс-валидация).

1_нарушение_подтверждено — модель нашла нарушение, эксперт согласен (по 3 примера на каждый тип).
2_норма_подтверждена — модель: норма, эксперт: норма.
3_ошибки_модели — пропуски (эксперт видит нарушение, модель нет) и ложные тревоги.
results_обучающий_набор.xlsx — таблица официального формата по всем 499 файлам обучающего набора
(лист 1 — по файлам, лист 2 — по исследованиям: что переснять).
""")
print(f"сохранено {n} примеров → {OUT}")
