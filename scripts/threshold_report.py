#!/usr/bin/env python3
"""Порог: чувствительность, специфичность, F1 и доля помеченных снимков при разных порогах — по снимкам вне обучения.
Пороги критериев умножаются на общий множитель; отдельно — режим sensitive и зона «проверить» (±0,10 от порога).
    PYTHONPATH=src python scripts/threshold_report.py [путь к criteria_oof.json] [путь к criteria.json]
"""
import csv, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.predict import aggregate
from sklearn.metrics import f1_score

WORK = os.path.join(ROOT, "data", "work")
SP = ["spine_positioning", "spine_axis_tilt", "spine_artifact"]; HP = ["hip_positioning_rotation", "hip_roi_field"]


def load(oof_path=None, crit_path=None):
    oof = json.load(open(oof_path or os.path.join(WORK, "criteria_oof.json")))
    cj = json.load(open(crit_path or os.path.join(ROOT, "weights", "criteria.json"), encoding="utf-8"))
    rows = {}
    for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        kind = "spine" if r["region"].startswith("Пояс") else "hip"
        iid = f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"
        if r["labeled"] == "1" and iid in oof:
            rows[iid] = r
    return oof, cj, rows


def evaluate(oof, rows, thr):
    yb, cb, rev = [], [], []
    yt = {c: [] for c in SP + HP}; pt = {c: [] for c in SP + HP}
    for i in sorted(rows):
        r = rows[i]; flags = SP if r["region"].startswith("Пояс") else HP
        truth = {c: int(r[c]) for c in flags if r[c] != ""}
        probs = {c: oof[i].get(c, 0.0) for c in flags}
        v = aggregate(probs, thr)
        yb.append(int(any(truth.values()))); cb.append(v.quality_class); rev.append(v.needs_review)
        for c in flags:
            if c in truth:
                yt[c].append(truth[c]); pt[c].append(probs[c])
    yb, cb, rev = np.array(yb), np.array(cb), np.array(rev)
    tp = int(((cb == 1) & (yb == 1)).sum()); fp = int(((cb == 1) & (yb == 0)).sum())
    fn = int(((cb == 0) & (yb == 1)).sum()); tn = int(((cb == 0) & (yb == 0)).sum())
    macro = float(np.mean([f1_score(np.array(yt[c]), np.array(pt[c]) >= thr[c], zero_division=0) for c in SP + HP]))
    return {"sens": tp / max(1, tp + fn), "spec": tn / max(1, tn + fp), "ppv": tp / max(1, tp + fp), "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "acc": (tp + tn) / len(yb), "flagged": float(cb.mean()), "macro": macro, "review": float(rev.mean()),
            "acc_out": float((yb == cb)[~rev].mean()), "acc_in": float((yb == cb)[rev].mean()) if rev.any() else float("nan")}


def main():
    oof, cj, rows = load(*(sys.argv[1:3] + [None, None])[:2])
    base = {c: float(m["threshold"]) for c, m in cj["models"].items()}
    print("| режим | чувств. | спец. | PPV | F1 | accuracy | помечено | macro-F1 |\n|---|---|---|---|---|---|---|---|")
    for k in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5):
        m = evaluate(oof, rows, {c: min(0.99, t * k) for c, t in base.items()})
        print(f"| ×{k} | {m['sens']:.2f} | {m['spec']:.2f} | {m['ppv']:.2f} | {m['f1']:.3f} | {m['acc']:.3f} | {m['flagged']:.2f} | {m['macro']:.3f} |")
    if "modes" in cj:
        m = evaluate(oof, rows, {c: float(t) for c, t in cj["modes"]["sensitive"].items()})
        print(f"| sensitive | {m['sens']:.2f} | {m['spec']:.2f} | {m['ppv']:.2f} | {m['f1']:.3f} | {m['acc']:.3f} | {m['flagged']:.2f} | {m['macro']:.3f} |")
    m = evaluate(oof, rows, base)
    print(f"\nзона «проверить» (±0,10 от порога): {m['review']:.2f} снимков; accuracy вне зоны {m['acc_out']:.3f}, в зоне {m['acc_in']:.3f}")


if __name__ == "__main__":
    main()
