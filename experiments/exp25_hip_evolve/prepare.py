#!/usr/bin/env python3
"""Эксп. 25: подготовка данных для эволюции признаков укладки/ротации бедра.

Кэш: снимки бедра с меткой, ориентиры локализатора ВНЕ обучения (landmarks_oof_hip.json), базовые признаки боевой
регрессии (shaft_img_deg, cnn_hip_pos — OOF), разбиение исследований на поиск (60 %) и отложенную часть (40 %).
Отложенная часть открывается только для финальной проверки кандидатов, прошедших поиск.
    PYTHONPATH=src python experiments/exp25_hip_evolve/prepare.py
"""
import json, os, random, sys
import numpy as np
import cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_criteria import manifest, cnn_oof                     # noqa: E402
from dxaqc.criteria import shaft_axis_from_image                  # noqa: E402
WORK = os.path.join(ROOT, "data", "work"); OUT = os.path.join(WORK, "exp25"); os.makedirs(OUT, exist_ok=True)
man = manifest(); lm_all = json.load(open(os.path.join(WORK, "landmarks_oof_hip.json"))); cnn = cnn_oof("hip")
rows = []
for iid, lm in sorted(lm_all.items()):
    m = man.get(iid)
    if not m or m.get("hip_positioning_rotation") is None or iid not in cnn:
        continue
    img = cv2.imread(os.path.join(WORK, "labeler", "png", iid + ".png"), cv2.IMREAD_GRAYSCALE)
    sb = lm.get("shaft_bot", {})
    shaft = shaft_axis_from_image(img, sb.get("x", 0.0)) if sb.get("conf", 0) > 0 else 0.0
    rows.append({"id": iid, "study": m["study"], "side": m["side"], "h": m["h"], "w": m["w"], "y": int(m["hip_positioning_rotation"]),
                 "shaft_img_deg": float(shaft), "cnn_hip_pos": float(cnn[iid]["cnn_hip_pos"]),
                 "lm": {k: [round(v["x"], 2), round(v["y"], 2), round(v.get("conf", 0.0), 3)] for k, v in lm.items()}})
studies = sorted({r["study"] for r in rows}); random.Random(25).shuffle(studies)
search = set(studies[: int(round(len(studies) * 0.6))])
for r in rows:
    r["part"] = "search" if r["study"] in search else "holdout"
json.dump(rows, open(os.path.join(OUT, "cache.json"), "w"), ensure_ascii=False)
for part in ("search", "holdout"):
    rs = [r for r in rows if r["part"] == part]
    print(f"{part}: снимков {len(rs)}, нарушений {sum(r['y'] for r in rs)}, исследований {len({r['study'] for r in rs})}")
