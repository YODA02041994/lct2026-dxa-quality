"""Ротация бедра: CNN на вырезке вокруг малого вертела и шейки (по точкам сети, как в инференсе), а не по всему кадру."""
import json, os, sys
import cv2, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_cnn_criterion import load, train, predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
WORK = os.path.join(ROOT, "data", "work")
net = json.load(open(os.path.join(WORK, "landmarks_oof_hip.json")))
kind, items = load("hip_positioning_rotation")
half = int(sys.argv[1]) if len(sys.argv) > 1 else 60
out_items = []
for it in items:
    lm = net[it["id"]]; img = it["img"]; h, w = img.shape
    side = "L" if it["id"].endswith("L") else "R"
    def X(p): return (w - 1 - p["x"]) if side == "L" else p["x"]     # изображение уже отражено для L
    pts = [lm[k] for k in ("lt_tip", "neck_inf", "shaft_top", "lt_up", "lt_down")]
    cx, cy = np.mean([X(p) for p in pts]), np.mean([p["y"] for p in pts])
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    pad = np.pad(img, half * 2, constant_values=0)
    crop = pad[y0 + half * 2:y0 + half * 4, x0 + half * 2:x0 + half * 4]
    out_items.append({**it, "img": np.ascontiguousarray(crop)})
y = np.array([it["y"] for it in out_items]); groups = np.array([it["study"] for it in out_items])
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"вырезка {half*2}×{half*2} px вокруг вертела/шейки, вход 224: снимков {len(y)}, +{y.sum()}")
oof = {}
for f, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(out_items, y, groups)):
    n = train(out_items, tr, 25, device, seed=f, rotate=True, size=224, masks=False)
    oof.update(predict(n, out_items, te, device, size=224))
p = np.array([oof[it["id"]] for it in out_items])
print(f"=== OOF LT-patch{half*2}: ROC-AUC {roc_auc_score(y, p):.3f} | PR-AUC {average_precision_score(y, p):.3f}")
json.dump(oof, open(os.path.join(WORK, f"cnn_oof_hip_ltpatch{half*2}.json"), "w"))
