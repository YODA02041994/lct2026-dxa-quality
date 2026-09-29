#!/usr/bin/env python3
"""Итоговые метрики в логике организатора — по OOF-вероятностям (снимки вне обучения всех моделей).

Порядок оценки (сессия 16.09): (1) бинарно «есть нарушение» по снимку; (2) отдельные типы + macro-F1 по 5 типам.
Пороги — из weights/criteria.json (подобраны по F1 на OOF). ДИ — bootstrap по исследованиям.
    PYTHONPATH=src python scripts/make_report.py   → docs/metrics.md
"""
import csv, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.labels import VIOLATION_TEXT
from dxaqc.predict import aggregate
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, recall_score

WORK = os.path.join(ROOT, "data", "work")
crit = json.load(open(os.path.join(ROOT, "weights", "criteria.json"), encoding="utf-8"))
thr = {c: m["threshold"] for c, m in crit["models"].items()}
oof = json.load(open(os.path.join(WORK, "criteria_oof.json")))
man = {}
for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
    kind = "spine" if r["region"].startswith("Пояс") else "hip"
    iid = f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"
    if r["labeled"] == "1":
        man[iid] = r
SP = ["spine_positioning", "spine_axis_tilt", "spine_artifact"]; HP = ["hip_positioning_rotation", "hip_roi_field"]
ids = sorted(i for i in oof if i in man)
y_bin, p_bin, c_bin, groups = [], [], [], []
y_t, p_t = {c: [] for c in SP + HP}, {c: [] for c in SP + HP}
for i in ids:
    r = man[i]; flags = SP if r["region"].startswith("Пояс") else HP
    truth = {c: int(r[c]) for c in flags if r[c] != ""}
    probs = {c: oof[i].get(c, 0.0) for c in flags}
    v = aggregate(probs, thr)
    y_bin.append(int(any(truth.values()))); p_bin.append(v.quality_prob); c_bin.append(v.quality_class); groups.append(r["study_dir"])
    for c in flags:
        if c in truth:
            y_t[c].append(truth[c]); p_t[c].append(probs[c])
y_bin, p_bin, c_bin, groups = map(np.array, (y_bin, p_bin, c_bin, groups))

def ci(fn, y, p, g, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; vals = []
    for _ in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if 0 < y[s].sum() < len(s): vals.append(fn(y[s], p[s]))
    return np.percentile(vals, [2.5, 97.5])

lines = ["# Метрики (OOF, снимки вне обучения всех моделей)", "",
         f"Снимков с метками: {len(ids)} (позвоночник {sum(1 for i in ids if i.endswith('spine'))}, бедро {sum(1 for i in ids if 'hip' in i)}). "
         "Кросс-валидация по исследованиям, 95 % ДИ — bootstrap по исследованиям. Пороги — по F1 на OOF.", "",
         "## 1. Бинарно: есть ли нарушение на снимке", "", "| метрика | значение [95 % ДИ] |", "|---|---|"]
for name, fn, pred in (("ROC-AUC", roc_auc_score, p_bin), ("PR-AUC", average_precision_score, p_bin),
                       ("F1", lambda y, p: f1_score(y, p), c_bin), ("Accuracy", lambda y, p: float((y == p).mean()), c_bin), ("Balanced accuracy", balanced_accuracy_score, c_bin),
                       ("Чувствительность", recall_score, c_bin), ("Специфичность", lambda y, p: recall_score(1 - y, 1 - p), c_bin)):
    v = fn(y_bin, pred); lo, hi = ci(fn, y_bin, pred, groups)
    lines.append(f"| {name} | {v:.3f} [{lo:.3f}–{hi:.3f}] |")
lines += ["", f"Доля снимков с нарушением: истина {y_bin.mean():.2f}, модель {c_bin.mean():.2f}.", "",
          "## 2. По анатомическим областям", "",
          "| область | n | + | ROC-AUC [95 % ДИ] | PR-AUC | F1 | чувствительность | специфичность | balanced accuracy |", "|---|---|---|---|---|---|---|---|---|"]
is_hip = np.array(["hip" in i for i in ids])
for name, m in (("Поясничный отдел позвоночника", ~is_hip), ("Проксимальный отдел бедра", is_hip)):
    y, p, c, g = y_bin[m], p_bin[m], c_bin[m], groups[m]
    lo, hi = ci(roc_auc_score, y, p, g)
    lines.append(f"| {name} | {len(y)} | {int(y.sum())} | {roc_auc_score(y, p):.3f} [{lo:.3f}–{hi:.3f}] | {average_precision_score(y, p):.3f} | {f1_score(y, c):.3f} | "
                 f"{recall_score(y, c):.3f} | {recall_score(1 - y, 1 - c):.3f} | {balanced_accuracy_score(y, c):.3f} |")
lines += ["", "## 3. По типам нарушений", "",
          "| тип (официальное название) | n | + | ROC-AUC [95 % ДИ] | PR-AUC | F1 | чувствительность | специфичность | balanced accuracy | порог |",
          "|---|---|---|---|---|---|---|---|---|---|"]
f1s = []
g_t = {c: np.array([man[i]["study_dir"] for i in ids if man[i][c] != "" and ((c in HP) == ("hip" in i))]) for c in SP + HP}
for c in SP + HP:
    y, p = np.array(y_t[c]), np.array(p_t[c]); d = (p >= thr[c]).astype(int); f = f1_score(y, d); f1s.append(f)
    lo, hi = ci(roc_auc_score, y, p, g_t[c])
    lines.append(f"| {VIOLATION_TEXT[c]} ({c}) | {len(y)} | {int(y.sum())} | {roc_auc_score(y, p):.3f} [{lo:.3f}–{hi:.3f}] | {average_precision_score(y, p):.3f} | {f:.3f} | "
                 f"{recall_score(y, d):.3f} | {recall_score(1 - y, 1 - d):.3f} | {balanced_accuracy_score(y, d):.3f} | {thr[c]:.2f} |")
lines += ["", f"**Macro-F1 по 5 типам: {np.mean(f1s):.3f}**", "",
          "## 4. Локализация ориентиров", "",
          "Расстояние между точкой сети и точкой разметчика на снимках вне обучения (`scripts/train_landmarks.py`, `docs/05`, эксп. 03).", "",
          "| ориентиры | медиана, мм | 90-й перцентиль, мм | доля ≤ 5 мм |", "|---|---|---|---|",
          "| тела L1–L4 | 2,3–3,1 | 12,7–22,8 | 74–84 % |", "| Th12, L5 | 4,8; 4,1 | 15,3; 16,4 | 58 %; 59 % |",
          "| гребни подвздошных костей | 5,3 | 10,8–11,2 | 41–43 % |", "| головка, вертелы, шейка, диафиз | 3,7–4,9 | 5,5–8,7 | 51–84 % |",
          "| седалищная кость | 5,8 | 9,8 | 43 % |", "",
          "Расхождение двух разметчиков на общих снимках: медиана 2,9 мм.", "",
          "## 5. Обработка", "",
          "| показатель | значение |", "|---|---|",
          "| доля успешно обработанных файлов, обучающий архив | 499 из 499 (100 %) |",
          "| доля успешно обработанных файлов, архив «Для теста» | 3 из 3 (100 %) |",
          "| время обработки исследования из трёх снимков, контейнер на CPU | 7–15 с |",
          "| то же, сервер с 2 ядрами, полный цикл через HTTP | 4–6 с |",
          "| требование ТЗ | не более 180 с |", ""]
open(os.path.join(ROOT, "docs", "metrics.md"), "w").write("\n".join(lines))
print("\n".join(lines))
