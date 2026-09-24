#!/usr/bin/env python3
"""Эксп. 08: RAD-DINO (Microsoft, DINOv2 ViT-B/14, обучен на рентгене грудной клетки) как замороженный извлекатель
признаков + логрегрессия. Сравнение с эксп. 01 (DINOv2 ImageNet: позвоночник «любое» 0,67, бедро 0,60).
    PYTHONPATH=src python experiments/exp08_raddino_probe.py
"""
import csv, os, sys, time, json
import cv2, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_criteria import oof_probs, best_f1
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModel
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png")
FLAGS = {"spine": ["spine_positioning", "spine_axis_tilt", "spine_artifact"], "hip": ["hip_positioning_rotation", "hip_roi_field"]}
name = "microsoft/rad-dino"
proc = AutoImageProcessor.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
print("модель:", name, "| размер входа:", proc.crop_size if hasattr(proc, "crop_size") else proc.size)
feats, meta = {}, {}
t0 = time.time()
for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
    if r["labeled"] != "1": continue
    kind = "spine" if r["region"].startswith("Пояс") else "hip"
    iid = f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"
    img = cv2.imread(os.path.join(PNG, iid + ".png"), 0)
    if kind == "hip" and r["side"] == "L": img = np.ascontiguousarray(img[:, ::-1])
    rgb = np.stack([img] * 3, -1)
    with torch.no_grad():
        out = model(**proc(images=rgb, return_tensors="pt"))
    h = out.last_hidden_state[0]
    feats[iid] = torch.cat([h[0], h[1:].mean(0)]).numpy()
    meta[iid] = {"kind": kind, "study": r["study_dir"], "any": int(any(r[c] == "1" for c in FLAGS[kind])), **{c: (int(r[c]) if r[c] != "" else None) for c in FLAGS[kind]}}
print(f"признаки для {len(feats)} снимков за {time.time() - t0:.0f} с, размерность {len(next(iter(feats.values())))}")
np.save(os.path.join(WORK, "feat_raddino.npy"), {"feats": feats, "meta": meta}, allow_pickle=True)
from sklearn.decomposition import PCA
for kind in ("spine", "hip"):
    ids = sorted(i for i in feats if meta[i]["kind"] == kind)
    X = np.array([feats[i] for i in ids]); g = np.array([meta[i]["study"] for i in ids])
    Xp = PCA(n_components=32, random_state=0).fit_transform(X)          # 1536 → 32, чтобы LR не переобучалась
    print(f"\n=== {kind}: {len(ids)} снимков")
    for c in ["any"] + FLAGS[kind]:
        y = np.array([meta[i][c] for i in ids]); ok = np.array([v is not None for v in y]); y = y[ok].astype(int)
        p = oof_probs(Xp[ok], y, g[ok]); p_full = oof_probs(X[ok], y, g[ok])
        print(f"  {c:26} +{y.sum():3d}  PCA32: AUC {roc_auc_score(y, p):.3f} F1 {best_f1(y, p)[0]:.3f} | 1536-d: AUC {roc_auc_score(y, p_full):.3f}")
