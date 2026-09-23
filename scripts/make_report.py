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
          "## 2. По типам нарушений", "", "| тип (официальное название) | n | + | ROC-AUC | F1 | порог |", "|---|---|---|---|---|---|"]
f1s = []
for c in SP + HP:
    y, p = np.array(y_t[c]), np.array(p_t[c]); f = f1_score(y, p >= thr[c]); f1s.append(f)
    lines.append(f"| {VIOLATION_TEXT[c]} ({c}) | {len(y)} | {int(y.sum())} | {roc_auc_score(y, p):.3f} | {f:.3f} | {thr[c]:.2f} |")
lines += ["", f"**Macro-F1 по 5 типам: {np.mean(f1s):.3f}**", ""]
open(os.path.join(ROOT, "docs", "metrics.md"), "w").write("\n".join(lines))
print("\n".join(lines))
