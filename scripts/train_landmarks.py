#!/usr/bin/env python3
"""Обучение локализатора ориентиров: кросс-валидация по исследованиям + финальная модель.

    PYTHONPATH=src python scripts/train_landmarks.py --kind hip            # 5 фолдов + финал → weights/landmarks_hip.pt
    PYTHONPATH=src python scripts/train_landmarks.py --kind spine --epochs 40
    PYTHONPATH=src python scripts/train_landmarks.py --kind hip --quick     # 1 фолд, 8 эпох — проверка, что всё крутится

Метрики CV: ошибка в мм по каждому ориентиру (медиана, 90-й перцентиль), доля точек в пределах 3 и 5 мм,
качество распознавания «нет в кадре» (AUC по уверенности). Пишутся в data/work/landmarks_cv_<kind>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.landmarks import (LM, SIZE, HeatmapNet, Sample, apply_affine, decode, decode_spine_levels, heatmap_loss,  # noqa: E402
                             invert_affine, letterbox_matrix, load_samples, make_heatmaps, make_input, random_augment)
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

MM = {300: 0.60, 280: 0.65, 248: 0.61}
ANN = os.path.join(ROOT, "data", "work", "labeler", "ann")
PNG = os.path.join(ROOT, "data", "work", "labeler", "png")
MANIFEST = os.path.join(ROOT, "data", "work", "manifest.csv")


def batches(samples: list[Sample], idx: np.ndarray, bs: int, rng: np.random.Generator | None):
    order = rng.permutation(idx) if rng is not None else idx
    for i in range(0, len(order), bs):
        chunk = [samples[j] for j in order[i:i + bs]]
        xs, hms, ws = [], [], []
        for s in chunk:
            h, w = s.img.shape
            M = letterbox_matrix(h, w)
            if rng is not None:
                A = random_augment(h, w, rng)
                M = (np.vstack([M, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32)
            xs.append(make_input(s.img, M, rng=rng))
            hm, wg = make_heatmaps(apply_affine(s.pts, M), s.state)
            hms.append(hm)
            ws.append(wg)
        yield (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(hms)), torch.from_numpy(np.stack(ws)))


def train(samples, tr_idx, epochs, device, seed=0, log_every=10):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = HeatmapNet(len(LM[samples[0].kind])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=1e-4)
    steps = epochs * max(1, int(np.ceil(len(tr_idx) / 16)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=steps, pct_start=0.15)
    net.train()
    t0, it = time.time(), 0
    for ep in range(epochs):
        tot = 0.0
        for x, hm, wg in batches(samples, tr_idx, 16, rng):
            x, hm, wg = x.to(device), hm.to(device), wg.to(device)
            loss = heatmap_loss(net(x), hm, wg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            it += 1
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"    эпоха {ep + 1:3d}/{epochs}  loss {tot / max(1, int(np.ceil(len(tr_idx) / 16))):.4f}  {time.time() - t0:.0f} с", flush=True)
    net.eval()
    return net


@torch.no_grad()
def predict(net, samples, idx, device):
    out = {}
    for i in idx:
        s = samples[i]
        h, w = s.img.shape
        M = letterbox_matrix(h, w)
        x = torch.from_numpy(make_input(s.img, M))[None].to(device)
        logits = net(x)
        xy, conf = decode(logits)
        if s.kind == "spine":
            xy, conf = decode_spine_levels(logits, LM["spine"], mm_per_px=0.6 / M[0, 0], xy=xy, conf=conf)
        out[i] = (apply_affine(xy[0], invert_affine(M)), conf[0])
    return out


def evaluate(samples, preds, kind):
    names = LM[kind]
    err = {k: [] for k in names}
    conf_present, conf_absent = {k: [] for k in names}, {k: [] for k in names}
    for i, (pts, conf) in preds.items():
        s = samples[i]
        mm = MM.get(s.img.shape[1], 0.6)
        for j, k in enumerate(names):
            if s.state[j] == 1:
                err[k].append(float(np.hypot(*(pts[j] - s.pts[j])) * mm))
                conf_present[k].append(float(conf[j]))
            elif s.state[j] == 0:
                conf_absent[k].append(float(conf[j]))
    rows = {}
    for k in names:
        e = np.array(err[k])
        cp, ca = conf_present[k], conf_absent[k]
        auc = None
        if len(cp) > 1 and len(ca) > 1:
            auc = float(roc_auc_score([1] * len(cp) + [0] * len(ca), cp + ca))
        rows[k] = {"n": int(len(e)), "median_mm": float(np.median(e)) if len(e) else None,
                   "p90_mm": float(np.percentile(e, 90)) if len(e) else None,
                   "within3": float((e <= 3).mean()) if len(e) else None, "within5": float((e <= 5).mean()) if len(e) else None,
                   "n_absent": len(ca), "absent_auc": auc, "conf_present_med": float(np.median(cp)) if cp else None,
                   "conf_absent_med": float(np.median(ca)) if ca else None}
    return rows


def print_table(rows):
    print(f"    {'ориентир':11}{'n':>5}{'медиана,мм':>12}{'p90,мм':>9}{'≤3мм':>7}{'≤5мм':>7}{'нет,n':>7}{'AUC нет':>9}{'conf есть':>11}{'conf нет':>10}")
    for k, r in rows.items():
        f = lambda v, p=1: "—" if v is None else (f"{v:.{p}f}")
        print(f"    {k:11}{r['n']:>5}{f(r['median_mm']):>12}{f(r['p90_mm']):>9}{f(r['within3'], 2):>7}{f(r['within5'], 2):>7}{r['n_absent']:>7}{f(r['absent_auc'], 3):>9}{f(r['conf_present_med'], 2):>11}{f(r['conf_absent_med'], 2):>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["spine", "hip"], required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(ANN, PNG, MANIFEST, a.kind)
    st = np.stack([s.state for s in samples])
    print(f"{a.kind}: снимков {len(samples)} | устройство {device} | точек: есть {(st == 1).sum()}, нет {(st == 0).sum()}, неизвестно {(st == -1).sum()}")
    groups = np.array([s.study for s in samples])
    epochs, folds = (8, 1) if a.quick else (a.epochs, a.folds)
    all_preds, fold_sds = {}, []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=max(2, a.folds)).split(samples, groups=groups)):
        if f >= folds:
            break
        print(f"  фолд {f + 1}/{folds}: обучение {len(tr)}, проверка {len(te)}")
        net = train(samples, tr, epochs, device, seed=f)
        all_preds.update(predict(net, samples, te, device))
        fold_sds.append({k: v.detach().cpu().half() for k, v in net.state_dict().items()})
    rows = evaluate(samples, all_preds, a.kind)
    print(f"\n=== CV, {a.kind}, {len(all_preds)} снимков вне обучения:")
    print_table(rows)
    # порог «нет в кадре»: по всем ориентирам вместе — максимум balanced accuracy
    cp = [c for k in rows for c in []]
    confs, labels = [], []
    for i, (pts, conf) in all_preds.items():
        s = samples[i]
        for j in range(len(conf)):
            if s.state[j] in (0, 1):
                confs.append(float(conf[j]))
                labels.append(int(s.state[j]))
    confs, labels = np.array(confs), np.array(labels)
    best_thr, best_ba = 0.3, 0
    for thr in np.linspace(0.05, 0.9, 35):
        pred = confs >= thr
        tpr = (pred & (labels == 1)).sum() / max(1, (labels == 1).sum())
        tnr = (~pred & (labels == 0)).sum() / max(1, (labels == 0).sum())
        if (tpr + tnr) / 2 > best_ba:
            best_ba, best_thr = (tpr + tnr) / 2, float(thr)
    print(f"    порог «нет в кадре»: conf < {best_thr:.2f} (balanced accuracy {best_ba:.3f}; точек есть/нет = {(labels == 1).sum()}/{(labels == 0).sum()})")
    os.makedirs(os.path.join(ROOT, "data", "work"), exist_ok=True)
    json.dump({"rows": rows, "absent_thr": best_thr, "n": len(all_preds), "epochs": epochs},
              open(os.path.join(ROOT, "data", "work", f"landmarks_cv_{a.kind}.json"), "w"), ensure_ascii=False, indent=1)
    # предсказания вне обучения — для честной оценки всего конвейера «сеть → критерии» (координаты в пикселях
    # исходника; для левого бедра — уже отражены обратно в исходную ориентацию)
    oof = {}
    for i, (pts, conf) in all_preds.items():
        s = samples[i]
        w = s.img.shape[1]
        d = {}
        for j, k in enumerate(LM[a.kind]):
            x = float(pts[j, 0]) if not (s.kind == "hip" and s.side == "L") else float((w - 1) - pts[j, 0])
            d[k] = {"x": x, "y": float(pts[j, 1]), "conf": float(conf[j]), "present": bool(conf[j] >= best_thr)}
        oof[s.image_id] = d
    json.dump(oof, open(os.path.join(ROOT, "data", "work", f"landmarks_oof_{a.kind}.json"), "w"), ensure_ascii=False)
    print(f"    предсказания вне обучения → data/work/landmarks_oof_{a.kind}.json")
    if a.no_final or a.quick:
        return
    print(f"\n  финальная модель на всех {len(samples)} снимках…")
    net = train(samples, np.arange(len(samples)), epochs, device, seed=42)
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    path = os.path.join(ROOT, "weights", f"landmarks_{a.kind}.pt")
    final_sd = {k: v.detach().cpu().half() for k, v in net.state_dict().items()}
    torch.save({"state_dict": final_sd, "state_dicts": [final_sd] + fold_sds, "kind": a.kind, "names": LM[a.kind], "size": SIZE,
                "absent_thr": best_thr, "cv": rows, "trained_on": len(samples), "ensemble": 1 + len(fold_sds)}, path)
    print(f"  сохранено: {path} ({os.path.getsize(path) // 1048576} МБ)")


if __name__ == "__main__":
    main()
