#!/usr/bin/env python3
"""Эксп. 07: аномалия без учителя (PatchCore-lite). Банк «нормальных» патчей (ResNet18 ImageNet, layer2+layer3),
оценка снимка = максимум по его патчам расстояния до ближайшего нормального патча. Ни одной метки нарушения при
обучении не используется: банк — только снимки без нарушений из обучающих фолдов (GroupKFold по исследованиям).
Идея: посторонние предметы и грубые укладки — это то, чего «не бывает на нормальном снимке».
    PYTHONPATH=src python experiments/exp07_patchcore.py
"""
import csv, json, os, sys, time
import cv2, numpy as np, torch, torch.nn.functional as F, torchvision
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.cnn import to_input
from dxaqc.landmarks import letterbox_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png")
SIZE = 320
dev = "mps" if torch.backends.mps.is_available() else "cpu"
r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval().to(dev)

@torch.no_grad()
def patches(img):
    h, w = img.shape
    x = torch.from_numpy(to_input(img, letterbox_matrix(h, w, SIZE)))[None].to(dev)
    x = r.maxpool(r.relu(r.bn1(r.conv1(x)))); l1 = r.layer1(x); l2 = r.layer2(l1); l3 = r.layer3(l2)
    l3u = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3u], 1)                                   # 384 × 40 × 40
    f = F.avg_pool2d(f, 3, 1, 1)                                  # локальное сглаживание, как в PatchCore
    return f[0].permute(1, 2, 0).reshape(-1, f.shape[1])          # 1600 × 384

def load(kind):
    items = []
    for row in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        k = "spine" if row["region"].startswith("Пояс") else "hip"
        if k != kind or row["labeled"] != "1": continue
        iid = f"{int(row['num']):03d}_{'spine' if kind == 'spine' else 'hip' + row['side']}"
        img = cv2.imread(os.path.join(PNG, iid + ".png"), 0)
        if kind == "hip" and row["side"] == "L": img = np.ascontiguousarray(img[:, ::-1])
        flags = ["spine_positioning", "spine_axis_tilt", "spine_artifact"] if kind == "spine" else ["hip_positioning_rotation", "hip_roi_field"]
        lab = {c: (int(row[c]) if row[c] != "" else None) for c in flags}
        lab["any"] = int(any(v == 1 for v in lab.values()))
        items.append({"id": iid, "img": img, "study": row["study_dir"], **lab})
    return items

out = {}
for kind in ("spine", "hip"):
    items = load(kind); t0 = time.time()
    feats = [patches(it["img"]) for it in items]
    groups = np.array([it["study"] for it in items]); anyv = np.array([it["any"] for it in items])
    scores = np.zeros(len(items)); scores_mean = np.zeros(len(items))
    for tr, te in GroupKFold(5).split(items, anyv, groups):
        bank = torch.cat([feats[i] for i in tr if anyv[i] == 0])          # только норма
        idx = torch.randperm(len(bank))[:60000]; bank = bank[idx]
        for i in te:
            d = torch.cdist(feats[i], bank).min(1).values                # ближайший нормальный патч
            scores[i] = float(d.max()); scores_mean[i] = float(d.topk(20).values.mean())
    flags = ["spine_positioning", "spine_axis_tilt", "spine_artifact"] if kind == "spine" else ["hip_positioning_rotation", "hip_roi_field"]
    print(f"\n=== {kind}: {len(items)} снимков, норма {int((anyv == 0).sum())}, {time.time() - t0:.0f} с")
    for name, sc in (("max", scores), ("top20", scores_mean)):
        line = f"  аномалия-{name:6} any {roc_auc_score(anyv, sc):.3f}"
        for c in flags:
            y = np.array([it[c] for it in items]); ok = np.array([v is not None for v in y])
            line += f" | {c} {roc_auc_score(y[ok].astype(int), sc[ok]):.3f}"
        print(line)
    out.update({it["id"]: {"anom_max": float(scores[i]), "anom_top20": float(scores_mean[i])} for i, it in enumerate(items)})
json.dump(out, open(os.path.join(WORK, "anomaly_oof.json"), "w"))
print("→ data/work/anomaly_oof.json")
