#!/usr/bin/env python3
"""Классификаторы критериев ТЗ поверх признаков из ориентиров. Обучение, честная оценка, экспорт.

Две оценки по каждому критерию (GroupKFold по исследованиям, 95 % ДИ bootstrap по исследованиям):
  • «врач»  — признаки по точкам разметчиков (верхняя граница: сколько даёт сама геометрия);
  • «сеть»  — признаки по точкам локализатора на снимках вне его обучения (data/work/landmarks_oof_*.json) —
              это и есть ожидаемое качество на закрытом тесте.
Финальная модель учится на признаках сети (обучаемся на том, что увидим в тесте) → weights/criteria.json.

    PYTHONPATH=src python scripts/train_criteria.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.criteria import CRITERION_FEATURES, HIP_CRITERIA, HIP_FEATURES, SPINE_CRITERIA, SPINE_FEATURES, CriteriaModel, hip_features, spine_features  # noqa: E402
from dxaqc.landmarks import LM, _pick_annotation  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

WORK = os.path.join(ROOT, "data", "work")
ANN, PNG = os.path.join(WORK, "labeler", "ann"), os.path.join(WORK, "labeler", "png")
LABEL_COL = {"spine_positioning": "spine_positioning", "spine_axis_tilt": "spine_axis_tilt", "spine_artifact": "spine_artifact",
             "hip_positioning_rotation": "hip_positioning_rotation", "hip_roi_field": "hip_roi_field"}


def manifest():
    out = {}
    for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        kind = "spine" if r["region"].startswith("Пояс") else "hip"
        iid = f"{int(r['num']):03d}_{'spine' if kind == 'spine' else 'hip' + r['side']}"
        out[iid] = {"kind": kind, "side": r["side"] or None, "study": r["study_dir"], "w": int(r["cols"]), "h": int(r["rows"]),
                    "labeled": r["labeled"] == "1", **{c: (int(r[c]) if r[c] != "" else None) for c in LABEL_COL}}
    return out


def human_landmarks(kind: str) -> dict[str, dict]:
    """Точки врача в исходной ориентации: {image_id: {name: {x, y, present}}}."""
    by: dict[str, list] = {}
    for p in glob.glob(os.path.join(ANN, "*.json")):
        try:
            a = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if a.get("kind") == kind:
            by.setdefault(a["image_id"], []).append(a)
    out = {}
    for iid, files in by.items():
        a = _pick_annotation(files)
        if not a:
            continue
        got = {}
        for p in a["points"]:
            got.setdefault(p["label"], (p["x"], p["y"]))
        if kind == "spine" and any(p["label"] == "vb" for p in a["points"]):
            vb = sorted([(p["x"], p["y"]) for p in a["points"] if p["label"] == "vb"], key=lambda t: t[1])
            if len(vb) == 5 and "th12" in got:
                for k, v in zip(["L1", "L2", "L3", "L4", "L5"], vb):
                    got[k] = v
        lm = {}
        for k in LM[kind]:
            if k in got:
                lm[k] = {"x": got[k][0], "y": got[k][1], "present": True, "conf": 1.0}
            else:
                lm[k] = {"x": 0.0, "y": 0.0, "present": False, "conf": 0.0}
        out[iid] = lm
    return out


def features_for(kind: str, lms: dict[str, dict], man: dict) -> dict[str, dict]:
    out = {}
    for iid, lm in lms.items():
        m = man.get(iid)
        if not m:
            continue
        img = cv2.imread(os.path.join(PNG, iid + ".png"), cv2.IMREAD_GRAYSCALE)
        out[iid] = spine_features(lm, m["h"], m["w"], img) if kind == "spine" else hip_features(lm, m["h"], m["w"], m["side"], img)
    return out


def fit_lr(X, y):
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)
    clf.fit((X - mean) / std, y)
    return {"coef": clf.coef_[0], "intercept": float(clf.intercept_[0]), "mean": mean, "std": std}


def oof_probs(X, y, groups, X_apply=None, n_splits=5):
    """OOF-вероятности; если X_apply задан — модель учится на X, применяется к X_apply тех же снимков."""
    X_apply = X if X_apply is None else X_apply
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = fit_lr(X[tr], y[tr])
        z = ((X_apply[te] - m["mean"]) / m["std"]) @ m["coef"] + m["intercept"]
        p[te] = 1 / (1 + np.exp(-z))
    return p


def ci(metric, y, p, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {u: np.where(groups == u)[0] for u in ug}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if 0 < y[s].sum() < len(s):
            vals.append(metric(y[s], p[s]))
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def best_f1(y, p):
    best = (0.0, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, p >= t, zero_division=0)
        if f > best[0]:
            best = (f, float(t))
    return best


def main():
    man = manifest()
    models, report = {}, []
    for kind, crits, feat_names in (("spine", SPINE_CRITERIA, SPINE_FEATURES), ("hip", HIP_CRITERIA, HIP_FEATURES)):
        hum = features_for(kind, human_landmarks(kind), man)
        oof_path = os.path.join(WORK, f"landmarks_oof_{kind}.json")
        net = features_for(kind, json.load(open(oof_path)), man) if os.path.exists(oof_path) else {}
        ids = sorted(i for i in hum if man[i]["labeled"])
        groups = np.array([man[i]["study"] for i in ids])
        Xh = np.array([[hum[i][f] for f in feat_names] for i in ids], float)
        Xn = np.array([[net[i][f] for f in feat_names] for i in ids], float) if net and all(i in net for i in ids) else None
        print(f"\n=== {kind}: снимков с метками {len(ids)}, признаков {len(feat_names)}, точки сети: {'есть' if Xn is not None else 'НЕТ (сначала train_landmarks.py)'}")
        print(f"  {'критерий':26}{'+':>4}   {'AUC врач [95% ДИ]':22} {'AUC сеть←врач':>14} {'AUC сеть←сеть [95% ДИ]':24} {'F1 сеть (порог)':>16}")
        # диагностика: насколько признаки по сети совпадают с признаками по врачу
        if Xn is not None:
            print("  признак: корреляция сеть↔врач | доля совпадений флагов")
            for j, fn in enumerate(feat_names):
                a, b = Xh[:, j], Xn[:, j]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                agree = float(((a > 0.5) == (b > 0.5)).mean()) if set(np.unique(a)) <= {0.0, 1.0, 2.0} else float("nan")
                print(f"    {fn:16} r={corr:5.2f}" + (f"  совпадение {agree:.2f}" if agree == agree else ""))
        for c in crits:
            y = np.array([man[i][c] for i in ids])
            ok = np.array([v is not None for v in y])
            y = y[ok].astype(int)
            g = groups[ok]
            sub = [feat_names.index(f) for f in CRITERION_FEATURES.get(c, feat_names)]
            # сырые AUC по каждому признаку набора — без обучения, для прозрачности
            raw = []
            for j in sub:
                for src, X in (("врач", Xh), ("сеть", Xn)):
                    if X is None or X[ok][:, j].std() < 1e-9:
                        continue
                    a = roc_auc_score(y, X[ok][:, j]); raw.append(f"{feat_names[j]}:{src} {max(a, 1 - a):.2f}{'↓' if a < 0.5 else ''}")
            print(f"  · {c}: " + ", ".join(raw))
            Xh_c = Xh[:, sub]
            Xn_c = Xn[:, sub] if Xn is not None else None
            ph = oof_probs(Xh_c[ok], y, g)
            auc_h = roc_auc_score(y, ph)
            lo, hi = ci(roc_auc_score, y, ph, g)
            row = {"kind": kind, "criterion": c, "n": int(ok.sum()), "pos": int(y.sum()), "auc_human": round(auc_h, 3), "ci_human": [round(lo, 3), round(hi, 3)]}
            line = f"  {c:26}{int(y.sum()):>4}   {auc_h:.3f} [{lo:.3f}–{hi:.3f}]     "
            if Xn is not None:
                p_hn = oof_probs(Xh_c[ok], y, g, X_apply=Xn_c[ok])   # учили на враче, применили к сети
                p_nn = oof_probs(Xn_c[ok], y, g)                      # учили на сети, применили к сети
                auc_hn, auc_nn = roc_auc_score(y, p_hn), roc_auc_score(y, p_nn)
                lo2, hi2 = ci(roc_auc_score, y, p_nn, g)
                f1, thr = best_f1(y, p_nn)
                row.update({"auc_net_from_human": round(auc_hn, 3), "auc_net": round(auc_nn, 3), "ci_net": [round(lo2, 3), round(hi2, 3)], "f1_net": round(f1, 3), "thr": thr})
                line += f"{auc_hn:>10.3f}     {auc_nn:.3f} [{lo2:.3f}–{hi2:.3f}]      {f1:.3f} ({thr:.2f})"
                m = fit_lr(Xn_c[ok], y)                                # финал: на признаках сети
            else:
                m = fit_lr(Xh_c[ok], y)
            m["features"] = [feat_names[j] for j in sub]
            m["threshold"] = row.get("thr", 0.5)
            models[c] = m
            report.append(row)
            print(line)
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    cm = CriteriaModel(models)
    out = cm.to_json()
    for c, m in models.items():
        out[c]["threshold"] = m["threshold"]
    json.dump({"models": out, "report": report}, open(os.path.join(ROOT, "weights", "criteria.json"), "w"), ensure_ascii=False, indent=1)
    print("\nсохранено: weights/criteria.json")


if __name__ == "__main__":
    main()
