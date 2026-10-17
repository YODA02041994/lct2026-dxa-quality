#!/usr/bin/env python3
"""Доверенная фитнес-функция эксп. 25. Кандидат — функция extra(lm, img, side, h, w) -> {имя: число} (не больше 4 признаков).

Признаки кандидата добавляются к боевым (shaft_img_deg, cnn_hip_pos) в ту же логистическую регрессию, что в бою (train_criteria.fit_lr).
  --part search  : GroupKFold(3) по исследованиям внутри поисковой части → OOF ROC-AUC кандидата и базы
  --part holdout : обучение на всей поисковой части, оценка на отложенной; ΔAUC и 95 % ДИ парным бутстрепом по исследованиям
Код кандидата исполняется в отдельном процессе с урезанными builtins; запрещённые конструкции отсекает evolve.py.
    python fitness.py --candidate cand.py --part search
"""
import argparse, json, math, os, sys
import numpy as np
import cv2
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png")
SAFE = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in
        ("abs", "min", "max", "len", "range", "sum", "float", "int", "round", "sorted", "enumerate", "zip", "list", "dict", "tuple",
         "bool", "any", "all", "map", "filter", "isinstance", "reversed", "set", "ValueError", "ZeroDivisionError", "Exception")}
BASE = ["shaft_img_deg", "cnn_hip_pos"]


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p, float); pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit((X - mean) / std, y)
    return lambda Z: 1 / (1 + np.exp(-(((Z - mean) / std) @ clf.coef_[0] + clf.intercept_[0])))


def load_rows():
    return json.load(open(os.path.join(WORK, "exp25", "cache.json")))


def normalized(r):
    """Снимок и точки приводятся к виду правого бедра (таз справа, большой вертел слева)."""
    img = cv2.imread(os.path.join(PNG, r["id"] + ".png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    lm = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in r["lm"].items() if v[2] >= 0.3}
    if r["side"] == "L":
        img = img[:, ::-1].copy(); lm = {k: (r["w"] - 1 - x, y, c) for k, (x, y, c) in lm.items()}
    return lm, img


def candidate_features(code, rows):
    g = {"__builtins__": SAFE, "np": np, "math": math}
    exec(compile(code, "candidate", "exec"), g)                   # noqa: S102 — код проверен evolve.py (FORBIDDEN) и изолирован процессом
    fn = g["extra"]
    feats, names = [], None
    for r in rows:
        lm, img = normalized(r)
        try:
            d = fn(lm, img, "R", r["h"], r["w"]) or {}
        except Exception:                                          # noqa: BLE001
            d = {}
        d = {str(k): float(v) for k, v in list(d.items())[:4] if isinstance(v, (int, float)) and math.isfinite(float(v))}
        feats.append(d)
        names = names or sorted(d)
    names = sorted({k for d in feats for k in d})[:4]
    X = np.array([[d.get(k, 0.0) for k in names] for d in feats], dtype=float) if names else np.zeros((len(rows), 0))
    return X, names


def base_matrix(rows):
    return np.array([[r[k] for k in BASE] for r in rows], dtype=float)


def oof_auc(X, y, groups, k=3):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        p[te] = fit_lr(X[tr], y[tr])(X[te])
    return auc(y, p), p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", required=True); ap.add_argument("--part", default="search")
    ap.add_argument("--boot", type=int, default=1000); a = ap.parse_args()
    code = open(a.candidate, encoding="utf-8").read()
    rows = load_rows()
    S = [r for r in rows if r["part"] == "search"]; H = [r for r in rows if r["part"] == "holdout"]
    if a.part == "search":
        Xc, names = candidate_features(code, S)
        y = np.array([r["y"] for r in S]); g = np.array([r["study"] for r in S]); B = base_matrix(S)
        base, _ = oof_auc(B, y, g); cand, _ = oof_auc(np.hstack([B, Xc]), y, g)
        # отдельная сила признаков без сети — чтобы видеть, несут ли они новую информацию
        alone = oof_auc(np.hstack([B[:, :1], Xc]), y, g)[0] if Xc.shape[1] else float("nan")
        print(json.dumps({"part": "search", "n": len(S), "pos": int(y.sum()), "base_auc": round(base, 4), "auc": round(cand, 4),
                          "geom_alone_auc": round(alone, 4), "features": names}))
        return
    Xs, names = candidate_features(code, S); Xh, _ = candidate_features(code, H)
    ys = np.array([r["y"] for r in S]); yh = np.array([r["y"] for r in H]); gh = np.array([r["study"] for r in H])
    pb = fit_lr(base_matrix(S), ys)(base_matrix(H)); pc = fit_lr(np.hstack([base_matrix(S), Xs]), ys)(np.hstack([base_matrix(H), Xh]))
    d0 = auc(yh, pc) - auc(yh, pb)
    rng = np.random.default_rng(0); ug = np.unique(gh); idx = {u: np.where(gh == u)[0] for u in ug}; ds = []
    for _ in range(a.boot):
        sel = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if yh[sel].min() == yh[sel].max():
            continue
        ds.append(auc(yh[sel], pc[sel]) - auc(yh[sel], pb[sel]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(json.dumps({"part": "holdout", "n": len(H), "pos": int(yh.sum()), "base_auc": round(auc(yh, pb), 4), "auc": round(auc(yh, pc), 4),
                      "delta": round(d0, 4), "ci": [round(float(lo), 4), round(float(hi), 4)], "features": names}))


if __name__ == "__main__":
    main()
