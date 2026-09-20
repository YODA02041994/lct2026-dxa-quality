#!/usr/bin/env python3
"""Эксперимент 01: замороженный DINOv2 + логистическая регрессия (linear probe).

Вопрос: сколько качества дают ГОТОВЫЕ признаки универсальной сети без дообучения на наших 249 снимках?
Сеть не обучается вообще — учится только линейный слой поверх её признаков, поэтому переобучиться почти нечем.
Проверка: StratifiedGroupKFold по исследованиям (снимки одного пациента не попадают в обе части),
5 фолдов × 5 повторов; 95 % ДИ — bootstrap по исследованиям.
    PYTHONPATH=src python experiments/exp01_dinov2_probe.py
"""
import csv, os, sys, warnings, json
import numpy as np, torch, pydicom
from PIL import Image
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov2-small"
SIZE = 448
dev = "cpu"   # MPS не поддерживает bicubic-интерполяцию позиционных эмбеддингов DINOv2 при 448 px
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "data/work/manifest.csv"), encoding="utf-8-sig")) if r["labeled"] == "1"]

def load(path):
    a = pydicom.dcmread(os.path.join(ROOT, path), force=True).pixel_array
    h, w = a.shape; s = max(h, w)
    canvas = np.zeros((s, s), dtype=np.uint8); canvas[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = a
    return Image.fromarray(canvas).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)

cache = os.path.join(ROOT, "data/work/feat_dinov2s_448.npy")
if os.path.exists(cache):
    X = np.load(cache)
else:
    proc = AutoImageProcessor.from_pretrained(MODEL); net = AutoModel.from_pretrained(MODEL).to(dev).eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            ims = [load(r["path"]) for r in rows[i:i+16]]
            x = proc(images=ims, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to(dev)
            out = net(pixel_values=x).last_hidden_state            # [B, 1+N, C]
            feats.append(torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1).cpu().numpy())   # CLS + среднее по патчам
    X = np.concatenate(feats); np.save(cache, X)
print("признаки:", X.shape, "| устройство:", dev)

groups = np.array([r["study_dir"] for r in rows]); region = np.array([r["region"] for r in rows])
SP, HP = "Поясничный отдел позвоночника", "Проксимальный отдел бедра"
def col(name): return np.array([int(r[name]) if r[name] != "" else 0 for r in rows])
tasks = [("Позвоночник: есть нарушение", SP, col("truth")), ("  ось > 5°", SP, col("spine_axis_tilt")),
         ("  посторонние предметы", SP, col("spine_artifact")), ("  укладка", SP, col("spine_positioning")),
         ("Бедро: есть нарушение", HP, col("truth")), ("  укладка/ротация", HP, col("hip_positioning_rotation")),
         ("  поле ROI", HP, col("hip_roi_field"))]

def oof(Xr, y, g, repeats=5):
    P = np.zeros((repeats, len(y)))
    for rep in range(repeats):
        k = min(5, int(y.sum()))
        for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rep).split(Xr, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000))
            clf.fit(Xr[tr], y[tr]); P[rep, te] = clf.predict_proba(Xr[te])[:, 1]
    return P.mean(0)

def ci(metric, y, p, g, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; vals = []
    for _ in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if 0 < y[s].sum() < len(s): vals.append(metric(y[s], p[s]))
    return np.percentile(vals, [2.5, 97.5])

res = []
print(f"\n{'задача':34} {'n':>4} {'+':>3}   ROC-AUC [95% ДИ]        PR-AUC   F1@0.5")
for name, reg, y_all in tasks:
    m = region == reg; y = y_all[m]; p = oof(X[m], y, groups[m])
    auc = roc_auc_score(y, p); lo, hi = ci(roc_auc_score, y, p, groups[m]); ap = average_precision_score(y, p); f1 = f1_score(y, p >= 0.5)
    print(f"{name:34} {m.sum():>4} {int(y.sum()):>3}   {auc:.3f} [{lo:.3f}–{hi:.3f}]   {ap:.3f}    {f1:.3f}")
    res.append(dict(task=name.strip(), n=int(m.sum()), pos=int(y.sum()), auc=round(auc, 3), ci=[round(lo, 3), round(hi, 3)], pr_auc=round(ap, 3), f1=round(f1, 3)))
json.dump(res, open(os.path.join(ROOT, "data/work/exp01_results.json"), "w"), ensure_ascii=False, indent=1)
