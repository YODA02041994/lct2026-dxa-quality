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
from dxaqc.cnn import ARCHS, CROP_LM, crop_around, make_net, to_input  # noqa: E402
from dxaqc.landmarks import letterbox_matrix, random_augment  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

WORK = os.path.join(ROOT, "data", "work")
PNG = os.path.join(WORK, "labeler", "png")
SIZE = 320
FLAGS = {"spine": ["spine_positioning", "spine_axis_tilt", "spine_artifact"], "hip": ["hip_positioning_rotation", "hip_roi_field"]}


def truth_of(r: dict, criterion: str):
    """Метка снимка: конкретный критерий или «любое нарушение области» (<kind>_any)."""
    kind = "spine" if criterion.startswith("spine") else "hip"
    if criterion.endswith("_any"):
        vals = [r[c] for c in FLAGS[kind] if r[c] != ""]
        return int(any(v == "1" for v in vals)) if vals else None
    return int(r[criterion]) if r[criterion] != "" else None


def load(criterion: str, crop: str | None = None, half: int = 50):
    """Снимки с метками; crop задан → вырезка вокруг ориентиров ПО ТОЧКАМ СЕТИ вне обучения (как будет в инференсе)."""
    kind = "spine" if criterion.startswith("spine") else "hip"
    lms = json.load(open(os.path.join(WORK, f"landmarks_oof_{kind}.json"))) if crop else {}
    items = []
    for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        k = "spine" if r["region"].startswith("Пояс") else "hip"
        if k != kind or r["labeled"] != "1" or truth_of(r, criterion) is None:
            continue
        iid = f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"
        img = cv2.imread(os.path.join(PNG, iid + ".png"), cv2.IMREAD_GRAYSCALE)
        if kind == "hip" and r["side"] == "L":
            img = np.ascontiguousarray(img[:, ::-1])          # одна ориентация, как у локализатора
        if crop:
            img = crop_around(img, lms[iid], CROP_LM[crop], half, mirrored=(kind == "hip" and r["side"] == "L"))
        items.append({"id": iid, "img": img, "y": truth_of(r, criterion), "study": r["study_dir"]})
    return kind, items




def train(items, idx, epochs, device, seed, rotate, arch="resnet18", size=SIZE, masks=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net(arch=arch).to(device)
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
                M = letterbox_matrix(h, w, size)
                if rotate:
                    A = random_augment(h, w, rng)
                    M = (np.vstack([M, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32)
                else:                                              # ось: без поворотов — только сдвиг/масштаб
                    s = rng.uniform(0.94, 1.06)
                    M = M.copy(); M[0, 0] *= s; M[1, 1] *= s; M[0, 2] += rng.uniform(-0.04, 0.04) * size; M[1, 2] += rng.uniform(-0.04, 0.04) * size
                xs.append(to_input(it["img"], M, rng, size=size, masks=masks))
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
def predict(net, items, idx, device, size=SIZE):
    out = {}
    for i in idx:
        h, w = items[i]["img"].shape
        x = torch.from_numpy(to_input(items[i]["img"], letterbox_matrix(h, w, size), size=size))[None].to(device)
        out[items[i]["id"]] = float(torch.sigmoid(net(x))[0, 0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--arch", default="resnet18", choices=ARCHS)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--tag", default="", help="суффикс имён выходных файлов (перебор вариантов, чтобы не затирать боевые веса)")
    ap.add_argument("--crop", default=None, choices=list(CROP_LM), help="вырезка вокруг ориентиров вместо всего кадра")
    ap.add_argument("--half", type=int, default=50, help="полуширина вырезки, px")
    a = ap.parse_args()
    name = a.criterion + (f"_{a.tag}" if a.tag else "")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    kind, items = load(a.criterion, a.crop, a.half)
    y = np.array([it["y"] for it in items])
    groups = np.array([it["study"] for it in items])
    rotate = a.criterion != "spine_axis_tilt"
    print(f"{a.criterion} [{a.arch} @ {a.size}px, {a.epochs} эп.]: снимков {len(items)}, нарушений {int(y.sum())} | {device} | повороты в аугментации: {'да' if rotate else 'нет'}")
    oof, sds, t0 = {}, [], time.time()
    for f, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=0).split(items, y, groups)):
        net = train(items, tr, a.epochs, device, seed=f, rotate=rotate, arch=a.arch, size=a.size, masks=(kind == "hip" and not a.crop))
        oof.update(predict(net, items, te, device, size=a.size))
        sds.append({k: v.detach().cpu().half() for k, v in net.state_dict().items()})
        yt = y[te]
        pt = np.array([oof[items[i]["id"]] for i in te])
        print(f"  фолд {f + 1}: обучение {len(tr)}, проверка {len(te)} (+{int(yt.sum())}) | AUC {roc_auc_score(yt, pt):.3f} | {time.time() - t0:.0f} с", flush=True)
    p = np.array([oof[it["id"]] for it in items])
    auc, ap_ = roc_auc_score(y, p), average_precision_score(y, p)
    print(f"=== OOF {a.criterion}: ROC-AUC {auc:.3f} | PR-AUC {ap_:.3f} (доля позитивов {y.mean():.2f})")
    json.dump(oof, open(os.path.join(WORK, f"cnn_oof_{name}.json"), "w"))
    net = train(items, np.arange(len(items)), a.epochs, device, seed=42, rotate=rotate, arch=a.arch, size=a.size, masks=(kind == "hip" and not a.crop))
    final = {k: v.detach().cpu().half() for k, v in net.state_dict().items()}
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    path = os.path.join(ROOT, "weights", f"cnn_{name}.pt")
    torch.save({"state_dict": final, "state_dicts": [final] + sds[:2], "criterion": a.criterion, "kind": kind, "size": a.size, "arch": a.arch, "crop": a.crop, "half": a.half,
                "oof_auc": auc, "trained_on": len(items)}, path)
    print(f"сохранено: {path} ({os.path.getsize(path) // 1048576} МБ), OOF → data/work/cnn_oof_{name}.json")


if __name__ == "__main__":
    main()
