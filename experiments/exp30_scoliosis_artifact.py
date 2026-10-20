#!/usr/bin/env python3
"""Эксп. 30 (20.10): ложные «посторонние предметы» на сколиотических позвоночниках.

Наблюдение (владелец, 20.10): на снимках со сколиозом система показывает «посторонние предметы».
Разметка эксперта тут ни при чём: метка «предметы» у позвоночников с деформацией ≥ 10° стоит в 4 из 20 (20 %),
у остальных — в 13 из 72 (18 %). А система ошибается: из 5 её ложных «предметов» 3 — на сколиозах (037, 049, 063).

Механизм: признак «тонкие яркие линии» (artifact_features, эксп. 05) ищет предметы ВНЕ полосы позвоночника,
а полоса — прямая вертикаль ±20 % ширины кадра вокруг самого яркого столбца. При сколиозе позвоночник выходит
из прямой полосы, и края тел и отростков засчитываются как «предметы».

Вариант (один, задан до запуска): центр полосы в каждой строке — по центрам тел Th12, L1–L5, найденным сетью
(точки с present = True, линейная интерполяция по высоте, за крайними точками — значение крайней точки);
ширина полосы та же (±20 % ширины кадра). Меньше 3 точек — прежняя прямая полоса.

Условие приёмки (записано до запуска):
  1) ложных «предметов» на позвоночниках с деформацией ≥ 10° по нашей мере становится меньше;
  2) верно найденных «предметов» из 17 — не меньше;
  3) ROC-AUC критерия: парная разница по исследованиям, нижняя граница 95 % ДИ ≥ −0,02 (не хуже);
  4) итоговые ROC-AUC и F1 по снимку (scripts/make_report.py) не ниже 0,861 и 0,778 — проверяется после переобучения.
Всё — на снимках вне обучения (GroupKFold по исследованиям), точки — сети вне её обучения (landmarks_oof_spine.json).

    PYTHONPATH=src python experiments/exp30_scoliosis_artifact.py   → experiments/exp30_scoliosis_artifact.json
"""
import csv, importlib.util, json, os, sys
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc import criteria as C                      # noqa: E402
from dxaqc.scoliosis import measure                  # noqa: E402

spec = importlib.util.spec_from_file_location("tc", os.path.join(ROOT, "scripts", "train_criteria.py"))
tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)

FEATS = ["th_frac40_log", "cnn_artifact", "om_max_all", "om_area"]
LEVELS = ("th12", "L1", "L2", "L3", "L4", "L5")


def centers_of(lm):
    pts = [C._pt(lm, k) for k in LEVELS]
    return [p for p in pts if p]


def main():
    man = tc.manifest()
    lms = json.load(open(os.path.join(tc.WORK, "landmarks_oof_spine.json")))
    cnn = tc.cnn_oof("spine")
    comments = {}
    for r in csv.DictReader(open(os.path.join(tc.WORK, "manifest.csv"), encoding="utf-8-sig")):
        if r["region"].startswith("Пояс"):
            comments[f"{int(r['num']):03d}_spine"] = (r.get("comment") or "").lower()
    ids = sorted(i for i in lms if i in man and man[i]["labeled"] and man[i].get("spine_artifact") is not None)
    rows = []
    for i in ids:
        img = cv2.imread(os.path.join(tc.PNG, i + ".png"), cv2.IMREAD_GRAYSCALE)
        base = C.spine_features(lms[i], man[i]["h"], man[i]["w"], img, cnn.get(i))
        old = C.artifact_features(img)
        new = C.artifact_features(img, centers=centers_of(lms[i]))
        bend = measure(lms[i]).get("bend_deg") or 0.0
        rows.append(dict(id=i, y=int(man[i]["spine_artifact"]), g=man[i]["study"], bend=bend,
                         scol_com="сколиоз" in comments.get(i, ""), n_centers=len(centers_of(lms[i])),
                         X_old=[old["th_frac40_log"] if f == "th_frac40_log" else base[f] for f in FEATS],
                         X_new=[new["th_frac40_log"] if f == "th_frac40_log" else base[f] for f in FEATS],
                         th_old=old["th_frac40"], th_new=new["th_frac40"]))
    y = np.array([r["y"] for r in rows]); g = np.array([r["g"] for r in rows])
    out = {"n": len(rows), "pos": int(y.sum())}
    probs = {}
    for v in ("old", "new"):
        X = np.array([r["X_" + v] for r in rows], float)
        p = tc.oof_probs(X, y, g)
        f1, thr = tc.best_f1(y, p)
        pred = p >= thr
        scol = np.array([r["bend"] >= 10 for r in rows]); com = np.array([r["scol_com"] for r in rows])
        out[v] = {"auc": round(roc_auc_score(y, p), 4), "f1": round(f1, 4), "thr": round(thr, 2),
                  "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
                  "fp_deform10": int((pred & (y == 0) & scol).sum()), "neg_deform10": int(((y == 0) & scol).sum()),
                  "fp_comment": int((pred & (y == 0) & com).sum()), "neg_comment": int(((y == 0) & com).sum()),
                  "fp_ids": [r["id"] for r, q in zip(rows, pred) if q and r["y"] == 0]}
        probs[v] = p
    # парная разница ROC-AUC по исследованиям
    rng = np.random.default_rng(0); ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; d = []
    for _ in range(2000):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if 0 < y[s].sum() < len(s):
            d.append(roc_auc_score(y[s], probs["new"][s]) - roc_auc_score(y[s], probs["old"][s]))
    out["delta_auc"] = [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    o, n = out["old"], out["new"]
    out["gate"] = {"1_fp_deform_down": n["fp_deform10"] < o["fp_deform10"], "2_tp_not_lower": n["tp"] >= o["tp"],
                   "3_auc_noninferior": out["delta_auc"][1] >= -0.02}
    out["per_image"] = [{"id": r["id"], "y": r["y"], "bend": round(r["bend"], 1), "centers": r["n_centers"],
                         "th_old": round(r["th_old"], 4), "th_new": round(r["th_new"], 4),
                         "p_old": round(float(po), 3), "p_new": round(float(pn), 3)}
                        for r, po, pn in zip(rows, probs["old"], probs["new"])]
    json.dump(out, open(os.path.join(ROOT, "experiments", "exp30_scoliosis_artifact.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
