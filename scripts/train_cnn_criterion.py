#!/usr/bin/env python3
"""CNN-второе мнение по одному критерию: ResNet18 (ImageNet) → вероятность нарушения по всему снимку.

Зачем: «посторонние предметы» по ориентирам не определяются (мелкие яркие объекты у краёв), а для укладки
бедра CNN может подхватить то, чего нет в наших признаках. Выход — OOF-вероятности (GroupKFold по исследованиям)
в data/work/cnn_oof_<criterion>.json (их подмешиваем признаком в классификатор критерия) и финальные веса
weights/cnn_<criterion>.pt (fp16, ансамбль фолдов + финал).

    PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion spine_artifact
    PYTHONPATH=src python scripts/train_cnn_criterion.py --criterion hip_positioning_rotation --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.landmarks import MEAN, STD, letterbox_matrix, random_augment  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

WORK = os.path.join(ROOT, "data", "work")
PNG = os.path.join(WORK, "labeler", "png")
SIZE = 320


def load(criterion: str):
    kind = "spine" if criterion.startswith("spine") else "hip"
    items = []
    for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        k = "spine" if r["region"].startswith("Пояс") else "hip"
        if k != kind or r["labeled"] != "1" or r[criterion] == "":
            continue
        iid = f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"
        img = cv2.imread(os.path.join(PNG, iid + ".png"), cv2.IMREAD_GRAYSCALE)
        if kind == "hip" and r["side"] == "L":
            img = np.ascontiguousarray(img[:, ::-1])          # одна ориентация, как у локализатора
        items.append({"id": iid, "img": img, "y": int(r[criterion]), "study": r["study_dir"]})
    return kind, items


def to_input(img, M, rng=None):
    x = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255.0
    if rng is not None:
        x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
        x = np.clip(x * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1)
        if rng.random() < 0.5:
            for _ in range(rng.integers(1, 3)):
                rw, rh = int(rng.uniform(0.08, 0.3) * SIZE), int(rng.uniform(0.15, 0.6) * SIZE)
                x0 = 0 if rng.random() < 0.5 else SIZE - rw
                y0 = int(rng.uniform(0, SIZE - rh))
                x[y0:y0 + rh, x0:x0 + rw] = 0
    x3 = np.stack([x, x, x])
    return (x3 - MEAN[:, None, None]) / STD[:, None, None]


def make_net():
    r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    r.fc = nn.Linear(512, 1)
    return r


def train(items, idx, epochs, device, seed, rotate):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net().to(device)
    pos = sum(items[i]["y"] for i in idx)
    pw = torch.tensor([(len(idx) - pos) / max(1, pos)], device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    steps = epochs * max(1, int(np.ceil(len(idx) / 16)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.2)
    net.train()
    for ep in range(epochs):
        order = rng.permutation(idx)
        for i in range(0, len(order), 16):
            chunk = [items[j] for j in order[i:i + 16]]
            xs = []
            for it in chunk:
                h, w = it["img"].shape
                M = letterbox_matrix(h, w, SIZE)
                if rotate:
                    A = random_augment(h, w, rng)
                    M = (np.vstack([M, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32)
                else:                                              # ось: без поворотов — только сдвиг/масштаб
                    s = rng.uniform(0.94, 1.06)
                    M = M.copy(); M[0, 0] *= s; M[1, 1] *= s; M[0, 2] += rng.uniform(-0.04, 0.04) * SIZE; M[1, 2] += rng.uniform(-0.04, 0.04) * SIZE
                xs.append(to_input(it["img"], M, rng))
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.tensor([it["y"] for it in chunk], dtype=torch.float32, device=device)
            loss = nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net


@torch.no_grad()
def predict(net, items, idx, device):
    out = {}
    for i in idx:
        h, w = items[i]["img"].shape
        x = torch.from_numpy(to_input(items[i]["img"], letterbox_matrix(h, w, SIZE)))[None].to(device)
        out[items[i]["id"]] = float(torch.sigmoid(net(x))[0, 0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    kind, items = load(a.criterion)
    y = np.array([it["y"] for it in items])
    groups = np.array([it["study"] for it in items])
    rotate = a.criterion != "spine_axis_tilt"
    print(f"{a.criterion}: снимков {len(items)}, нарушений {int(y.sum())} | {device} | повороты в аугментации: {'да' if rotate else 'нет'}")
    oof, sds, t0 = {}, [], time.time()
    for f, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=0).split(items, y, groups)):
        net = train(items, tr, a.epochs, device, seed=f, rotate=rotate)
        oof.update(predict(net, items, te, device))
        sds.append({k: v.detach().cpu().half() for k, v in net.state_dict().items()})
        yt = y[te]
        pt = np.array([oof[items[i]["id"]] for i in te])
        print(f"  фолд {f + 1}: обучение {len(tr)}, проверка {len(te)} (+{int(yt.sum())}) | AUC {roc_auc_score(yt, pt):.3f} | {time.time() - t0:.0f} с", flush=True)
    p = np.array([oof[it["id"]] for it in items])
    auc, ap_ = roc_auc_score(y, p), average_precision_score(y, p)
    print(f"=== OOF {a.criterion}: ROC-AUC {auc:.3f} | PR-AUC {ap_:.3f} (доля позитивов {y.mean():.2f})")
    json.dump(oof, open(os.path.join(WORK, f"cnn_oof_{a.criterion}.json"), "w"))
    net = train(items, np.arange(len(items)), a.epochs, device, seed=42, rotate=rotate)
    final = {k: v.detach().cpu().half() for k, v in net.state_dict().items()}
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    path = os.path.join(ROOT, "weights", f"cnn_{a.criterion}.pt")
    torch.save({"state_dict": final, "state_dicts": [final] + sds, "criterion": a.criterion, "kind": kind, "size": SIZE,
                "oof_auc": auc, "trained_on": len(items)}, path)
    print(f"сохранено: {path} ({os.path.getsize(path) // 1048576} МБ), OOF → data/work/cnn_oof_{a.criterion}.json")


if __name__ == "__main__":
    main()
