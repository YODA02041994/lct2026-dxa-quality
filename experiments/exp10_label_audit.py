#!/usr/bin/env python3
"""Эксп. 10: аудит меток (идея confident learning / cleanlab). Снимки, где модель вне обучения уверенно не согласна
с меткой эксперта, — кандидаты на ошибку разметки или на спорный случай. Список — на сверку с экспертами.
    PYTHONPATH=src python experiments/exp10_label_audit.py  → docs/09_Спорные_метки.md
"""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "data", "work")
oof = json.load(open(os.path.join(WORK, "criteria_oof.json")))
cj = json.load(open(os.path.join(ROOT, "weights", "criteria.json"))); thr = {c: m["threshold"] for c, m in cj["models"].items()}
NAME = {"spine_positioning": "укладка позвоночника", "spine_axis_tilt": "ось позвоночника", "spine_artifact": "посторонние предметы",
        "hip_positioning_rotation": "укладка бедра", "hip_roi_field": "поле ROI бедра"}
man = {}
for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
    k = "spine" if r["region"].startswith("Пояс") else "hip"; iid = f"{int(r['num']):03d}_{'spine' if k == 'spine' else 'hip' + r['side']}"
    if r["labeled"] == "1": man[iid] = r
rows = []
for iid, probs in oof.items():
    r = man.get(iid)
    if not r: continue
    for c, p in probs.items():
        if r[c] == "": continue
        y = int(r[c]); t = thr[c]
        # «уверенность несогласия»: насколько далеко вероятность от порога в сторону, противоположную метке
        gap = (p - t) if y == 0 else (t - p)
        if gap > 0.15:
            rows.append({"id": iid, "crit": c, "label": y, "p": p, "gap": gap, "comment": r["comment"].strip()})
rows.sort(key=lambda x: -x["gap"])
lines = ["# Спорные метки — кандидаты на сверку с экспертами", "",
         "Снимки, где модель **вне обучения** уверенно расходится с меткой (вероятность дальше 0,15 от порога в противоположную сторону).",
         "Это не «модель права»: это список, который стоит показать экспертам. Комментарии — из исходной разметки организатора.", "",
         f"Всего кандидатов: {len(rows)} (метка 1, модель «норма»: {sum(1 for x in rows if x['label'] == 1)}; метка 0, модель «нарушение»: {sum(1 for x in rows if x['label'] == 0)}).", "",
         "| снимок | критерий | метка | модель p | порог | комментарий эксперта |", "|---|---|---|---|---|---|"]
for x in rows:
    lines.append(f"| {x['id']} | {NAME[x['crit']]} | {x['label']} | {x['p']:.2f} | {thr[x['crit']]:.2f} | {x['comment']} |")
lines += ["", "## Что с этим делать", "",
          "1. Сколиозы с меткой «ось = 0» при наклоне 6–8° — уточнить у экспертов, считается ли наклон при сколиозе нарушением укладки (сейчас: нет).",
          "2. Снимок 011 — эксперт сам написал «не верная разметка».",
          "3. «Требует внимание» (010, 019) — оба бедра помечены нарушением, но модель не видит; посмотреть глазами.",
          "4. Метки, подтверждённые экспертами как верные, оставить; подтверждённые ошибки — исправить в `data/work/manifest.csv` и переобучить (`scripts/train_criteria.py`)."]
open(os.path.join(ROOT, "docs", "09_Спорные_метки.md"), "w").write("\n".join(lines))
print("\n".join(lines[:8])); print("..."); print(f"→ docs/09_Спорные_метки.md ({len(rows)} строк)")
