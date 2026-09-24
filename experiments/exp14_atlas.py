#!/usr/bin/env python3
"""Эксп. 14 (клиницист): атлас нормального бедра. Все бёдра выравниваются на средний контур подобием (сдвиг+поворот+масштаб)
по 5 надёжным ориентирам сети (головка, верх/латеральный край большого вертела, верх/низ диафиза); по 109 нормам строится
средний снимок и разброс; у каждого снимка — z-отклонение от нормы в зонах малого вертела, седалищной кости и всего кадра.
Оценка честная: атлас в каждом фолде строится только по нормам обучающей части."""
import csv, json, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "src"))
from train_criteria import oof_probs, best_f1
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png")
lms = json.load(open(os.path.join(WORK, "landmarks_oof_hip.json")))
BASE = ["head", "gt_top", "gt_lat", "shaft_top", "shaft_bot"]; S = 256
items = []
for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
    if not r["region"].startswith("Прокс") or r["labeled"] != "1" or r["hip_positioning_rotation"] == "": continue
    iid = f"{int(r['num']):03d}_hip{r['side']}"; img = cv2.imread(os.path.join(PNG, iid + ".png"), 0); h, w = img.shape
    lm = lms[iid]; mir = r["side"] == "L"
    if mir: img = np.ascontiguousarray(img[:, ::-1])
    pts = np.array([[(w - 1 - lm[k]["x"]) if mir else lm[k]["x"], lm[k]["y"]] for k in BASE], np.float32)
    lt = np.array([(w - 1 - lm["lt_tip"]["x"]) if mir else lm["lt_tip"]["x"], lm["lt_tip"]["y"]], np.float32)
    isch = np.array([(w - 1 - lm["ischium"]["x"]) if mir else lm["ischium"]["x"], lm["ischium"]["y"]], np.float32)
    items.append({"id": iid, "img": img.astype(np.float32) / 255, "pts": pts, "lt": lt, "isch": isch, "y": int(r["hip_positioning_rotation"]), "g": r["study_dir"]})
# средняя форма (в общих координатах): просто среднее по всем — только для системы координат, метки не используются
mean_shape = np.mean([it["pts"] for it in items], 0)
c = mean_shape.mean(0); sc = S * 0.32 / np.linalg.norm(mean_shape - c, axis=1).mean()
target = (mean_shape - c) * sc + S / 2
def warp(it):
    M, _ = cv2.estimateAffinePartial2D(it["pts"], target, method=cv2.LMEDS)
    if M is None: return None, None, None
    w = cv2.warpAffine(it["img"], M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
    T = lambda p: (M[:, :2] @ p + M[:, 2])
    return w, T(it["lt"]), T(it["isch"])
for it in items: it["w"], it["ltw"], it["ischw"] = warp(it)
items = [it for it in items if it["w"] is not None]
y = np.array([it["y"] for it in items]); g = np.array([it["g"] for it in items])
mean_lt = np.mean([it["ltw"] for it in items], 0); mean_isch = np.mean([it["ischw"] for it in items], 0)
def roi(center, r=22):
    x0, y0 = int(center[0]) - r, int(center[1]) - r; return slice(max(0, y0), y0 + 2 * r), slice(max(0, x0), x0 + 2 * r)
feats = np.zeros((len(items), 6))
for tr, te in GroupKFold(5).split(items, y, groups=g):
    normals = [items[i]["w"] for i in tr if items[i]["y"] == 0]
    mu = np.mean(normals, 0); sd = np.std(normals, 0) + 0.03
    for i in te:
        z = (items[i]["w"] - mu) / sd; valid = items[i]["w"] > 0
        rl, ri = roi(mean_lt), roi(mean_isch)
        feats[i] = [np.abs(z[valid]).mean(), np.abs(z[rl]).mean(), z[rl].mean(), np.abs(z[ri]).mean(), z[ri].mean(), (np.abs(z) > 2).mean()]
names = ["|z| весь кадр", "|z| зона вертела", "z зона вертела (ярче нормы)", "|z| зона седалищной", "z зона седалищной", "доля |z|>2"]
print(f"бёдер {len(items)}, нарушений {y.sum()}")
for j, n in enumerate(names): print(f"  {n:30} AUC {roc_auc_score(y, feats[:, j]):.3f}")
p = oof_probs(feats, y, g); print(f"  LR по 6 признакам атласа: AUC {roc_auc_score(y, p):.3f} F1 {best_f1(y, p)[0]:.3f}")
json.dump({it["id"]: feats[i].tolist() for i, it in enumerate(items)}, open(os.path.join(WORK, "exp14_atlas.json"), "w"))
# картинка: средняя норма, средний «нарушенный», разница
mu_all = np.mean([it["w"] for it in items if it["y"] == 0], 0); mu_bad = np.mean([it["w"] for it in items if it["y"] == 1], 0)
diff = np.clip((mu_bad - mu_all) * 4 + 0.5, 0, 1)
vis = np.hstack([mu_all, mu_bad, diff]); cv2.imwrite(os.path.join(ROOT, "docs", "img", "atlas_hip.jpg"), (vis * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("→ docs/img/atlas_hip.jpg (норма | нарушение | разница)")
