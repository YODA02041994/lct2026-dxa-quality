#!/usr/bin/env python3
"""Эксп. 12: предобучение «есть посторонний предмет / нет» на object-CXR (рентген грудной клетки, CC BY-NC 4.0),
затем дообучение на наших 99 снимках позвоночника через train_cnn_criterion --init.
Вход: папка с уменьшенными JPEG (см. tools/objectcxr_prepare.py на vps) и csv object-CXR (image_name, annotation).
    PYTHONPATH=src python experiments/exp12_objectcxr_pretrain.py --data ~/Downloads/lct_task4/object_cxr/dev320 --csv dev.csv --epochs 6
"""
import argparse, csv, os, sys, time
import cv2, numpy as np, torch, torch.nn as nn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.cnn import make_net, to_input
from dxaqc.landmarks import letterbox_matrix
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--csv", required=True)
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--size", type=int, default=320)
ap.add_argument("--arch", default="resnet18"); ap.add_argument("--out", default=os.path.join(ROOT, "weights", "pretrain_objectcxr_r18.pt"))
ap.add_argument("--val-frac", type=float, default=0.1)
a = ap.parse_args()
device = "mps" if torch.backends.mps.is_available() else "cpu"
items = []
for d, c in zip(a.data.split(","), a.csv.split(",")):                  # несколько папок: train320,dev320 + train.csv,dev.csv
    for r in csv.DictReader(open(os.path.join(d, c))):
        p = os.path.join(d, os.path.splitext(r["image_name"])[0] + ".jpg")
        if os.path.exists(p):
            items.append((p, int(bool(r.get("annotation", "").strip()))))
rng = np.random.default_rng(0); rng.shuffle(items)
nv = int(len(items) * a.val_frac); val, tr = items[:nv], items[nv:]
print(f"object-CXR: {len(items)} снимков ({sum(y for _, y in items)} с предметами) | train {len(tr)} / val {len(val)} | {device} | {a.arch} @ {a.size}")
net = make_net(arch=a.arch).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
steps = a.epochs * int(np.ceil(len(tr) / 32))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.15)
cache = {}
def load(p):
    if p not in cache:
        cache[p] = cv2.imread(p, 0)
    return cache[p]
def batch(chunk, aug):
    xs = []
    for p, _ in chunk:
        img = load(p); h, w = img.shape
        M = letterbox_matrix(h, w, a.size)
        if aug:
            s = rng.uniform(0.9, 1.1); M = M.copy(); M[0, 0] *= s; M[1, 1] *= s
            M[0, 2] += rng.uniform(-0.05, 0.05) * a.size; M[1, 2] += rng.uniform(-0.05, 0.05) * a.size
            if rng.random() < 0.5: M[0, 0] *= -1; M[0, 2] = a.size - M[0, 2]     # отражение: предметы симметричны
        xs.append(to_input(img, M, rng if aug else None, size=a.size, masks=False))
    return torch.from_numpy(np.stack(xs)).to(device)
t0 = time.time()
for ep in range(a.epochs):
    net.train(); order = rng.permutation(len(tr)); tot = 0.0
    for i in range(0, len(order), 32):
        chunk = [tr[j] for j in order[i:i + 32]]
        x = batch(chunk, True); y = torch.tensor([c[1] for c in chunk], dtype=torch.float32, device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += float(loss) * len(chunk)
    net.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(val), 64):
            ps.extend(torch.sigmoid(net(batch(val[i:i + 64], False))[:, 0]).cpu().numpy().tolist())
    auc = roc_auc_score([y for _, y in val], ps) if val else float("nan")
    print(f"  эпоха {ep + 1}/{a.epochs}: loss {tot / len(tr):.3f} | val AUC {auc:.3f} | {time.time() - t0:.0f} с", flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({"state_dict": net.state_dict(), "arch": a.arch, "size": a.size, "source": "object-CXR", "val_auc": auc, "n_train": len(tr)}, a.out)
print("сохранено:", a.out)
