#!/usr/bin/env python3
"""Эксп. 28: синтетические нарушения из нормальных снимков для редких критериев — с воротами, рецепт зафиксирован ДО запуска.

Рецепт (не меняется по результату; одна синтетическая копия на каждый подходящий нормальный снимок, зерно 28):
  • укладка позвоночника (6 настоящих нарушений): снимки с меткой 0, оба гребня найдены (уверенность ≥ 0,5) →
    обрезка низа кадра на 8 px выше верхнего из гребней (гребни уходят из кадра) → метка 1;
  • ось позвоночника (10): метка 0, искривление по центрам тел < 10°, наклон по массе кости < 3° → поворот содержимого
    на ±7…12° (знак и угол случайные), края — отражением (BORDER_REFLECT_101), размер кадра прежний → метка 1;
  • поле сканирования бедра (7): метка 0, нижний ориентир (lt_down / lt_tip / shaft_top) найден → обрезка низа так, чтобы
    до него осталось 0,3…1,2 см; ориентиры ниже среза становятся «не найдены» (преобразуются те же ориентиры ВНЕ обучения,
    локализатор на синтетике не запускается) → метка 1.
  Предметы и укладку бедра не трогаем: их признаки — выходы сетей, веса сетей по блокам не сохранены, признак синтетики
  от боевой сети был бы получен моделью, видевшей проверочный блок (утечка); к тому же эксп. 25 показал, что геометрия
  поверх ансамбля бедра ничего не даёт.

Проверка (ворота):
  этап 1 — синтетические нарушения должны попадать туда же, куда настоящие, в пространстве признаков регрессии (медианы);
  этап 2 — GroupKFold(5) по исследованиям; синтетика из исследования попадает только в обучающие блоки (assert),
           оценка — ТОЛЬКО на настоящих снимках вне обучения; ΔROC-AUC и ΔF1 (порог по максимуму F1 у каждой ветки),
           95 % ДИ — парный бутстреп по исследованиям, 1000 повторов.
  В модель — только если нижняя граница ДИ ΔROC-AUC > 0 и ΔF1 ≥ 0. Иначе результат записывается как отрицательный.
    PYTHONPATH=src python scripts/exp28_synthetic.py   → data/work/exp28.json
"""
import csv, json, math, os, random, sys
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dxaqc.criteria import CRITERION_FEATURES, hip_features, spine_features, spine_axis_from_image  # noqa: E402
from dxaqc.labels import pixel_mm  # noqa: E402
from dxaqc.scoliosis import measure  # noqa: E402
WORK = os.path.join(ROOT, "data", "work"); PNG = os.path.join(WORK, "labeler", "png")


def fit_lr(X, y):
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).fit((X - mean) / std, y)
    return lambda Z: 1 / (1 + np.exp(-(((Z - mean) / std) @ clf.coef_[0] + clf.intercept_[0])))


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p, float); pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def best_f1(y, p):
    best = 0.0
    for t in np.unique(np.round(p, 3)):
        yh = p >= t; tp = (yh & (y == 1)).sum(); fp = (yh & (y == 0)).sum(); fn = (~yh & (y == 1)).sum()
        best = max(best, 2 * tp / max(1, 2 * tp + fp + fn))
    return float(best)


def load():
    man = {}
    for r in csv.DictReader(open(os.path.join(WORK, "manifest.csv"), encoding="utf-8-sig")):
        kind = "spine" if r["region"].startswith("Пояс") else "hip" + r["side"]
        man[f"{int(r['num']):03d}_{kind}"] = r
    lms = {**json.load(open(os.path.join(WORK, "landmarks_oof_spine.json"))), **json.load(open(os.path.join(WORK, "landmarks_oof_hip.json")))}
    return man, lms


def feats(crit, iid, img, lm, side):
    h, w = img.shape
    f = spine_features(lm, h, w, img) if crit.startswith("spine") else hip_features(lm, h, w, side, img)
    return [float(f[k]) for k in CRITERION_FEATURES[crit]]


def synth(crit, img, lm, rng):
    """→ (картинка, ориентиры) синтетического нарушения или None, если снимок не подходит по рецепту."""
    h, w = img.shape
    if crit == "spine_positioning":
        cr = [lm[k] for k in ("crest_l", "crest_r") if k in lm and lm[k].get("conf", 0) >= 0.5]
        if len(cr) < 2:
            return None
        cut = int(min(c["y"] for c in cr)) - 8
        if cut < h * 0.5:
            return None
        return img[:cut].copy(), {k: v for k, v in lm.items() if v["y"] < cut}
    if crit == "spine_axis_tilt":
        if (measure(lm)["bend_deg"] or 0) >= 10 or spine_axis_from_image(img) >= 3:
            return None
        ang = rng.uniform(7, 12) * rng.choice([-1, 1])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), {}
    if crit == "hip_roi_field":
        low = next((lm[k] for k in ("lt_down", "lt_tip", "shaft_top") if k in lm and lm[k].get("conf", 0) >= 0.5), None)
        if not low:
            return None
        mm = pixel_mm(w)
        cut = int(round(low["y"] + rng.uniform(0.3, 1.2) * 10 / mm))
        if cut >= h - 2 or cut < h * 0.5:
            return None
        lm2 = {k: (dict(v) if v["y"] < cut else {**v, "conf": 0.0, "present": False}) for k, v in lm.items()}
        return img[:cut].copy(), lm2
    return None


def main():
    man, lms = load(); rng = random.Random(28); out = {}
    for crit in ("spine_positioning", "spine_axis_tilt", "hip_roi_field"):
        region = "spine" if crit.startswith("spine") else "hip"
        ids = sorted(i for i in man if (i.endswith("spine") if region == "spine" else "hip" in i) and man[i].get(crit, "") != "" and i in lms)
        X, y, g, SX, Sg = [], [], [], [], []
        for i in ids:
            img = cv2.imread(os.path.join(PNG, i + ".png"), cv2.IMREAD_GRAYSCALE)
            lm = {k: v for k, v in lms[i].items() if v.get("conf", 0) >= 0.3}
            side = man[i]["side"] or None
            X.append(feats(crit, i, img, lm, side)); y.append(int(man[i][crit])); g.append(man[i]["study_dir"])
            if int(man[i][crit]) == 0:
                s = synth(crit, img, lm, rng)
                if s is not None:
                    SX.append(feats(crit, i, s[0], s[1], side)); Sg.append(man[i]["study_dir"])
        X, y, g, SX, Sg = np.array(X), np.array(y), np.array(g), np.array(SX), np.array(Sg)
        names = CRITERION_FEATURES[crit]
        # этап 1: где лежат синтетические нарушения
        med = lambda A: [round(float(v), 3) for v in np.median(A, 0)]  # noqa: E731
        stage1 = {"features": names, "real_pos": med(X[y == 1]), "real_neg": med(X[y == 0]), "synthetic": med(SX), "n_synth": int(len(SX))}
        # этап 2: OOF на настоящих снимках
        pb, ps = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            test_studies = set(g[te])
            keep = np.array([s not in test_studies for s in Sg]) if len(Sg) else np.zeros(0, bool)
            assert not any(s in test_studies for s in Sg[keep]), "синтетика из проверочного исследования"
            pb[te] = fit_lr(X[tr], y[tr])(X[te])
            Xa = np.vstack([X[tr], SX[keep]]) if keep.any() else X[tr]
            ya = np.concatenate([y[tr], np.ones(int(keep.sum()), int)])
            ps[te] = fit_lr(Xa, ya)(X[te])
        d_auc, d_f1 = auc(y, ps) - auc(y, pb), best_f1(y, ps) - best_f1(y, pb)
        ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; bt = []; brng = np.random.default_rng(0)
        for _ in range(1000):
            sel = np.concatenate([idx[u] for u in brng.choice(ug, len(ug))])
            if y[sel].min() == y[sel].max():
                continue
            bt.append(auc(y[sel], ps[sel]) - auc(y[sel], pb[sel]))
        lo, hi = np.percentile(bt, [2.5, 97.5])
        passed = bool(lo > 0 and d_f1 >= 0)
        out[crit] = {"n": int(len(y)), "pos": int(y.sum()), "stage1": stage1, "auc_base": round(auc(y, pb), 3), "auc_synth": round(auc(y, ps), 3),
                     "delta_auc": round(d_auc, 3), "ci": [round(float(lo), 3), round(float(hi), 3)], "f1_base": round(best_f1(y, pb), 3),
                     "f1_synth": round(best_f1(y, ps), 3), "passed": passed}
        o = out[crit]
        print(f"\n== {crit}: {o['n']} снимков, {o['pos']} нарушений, синтетических {stage1['n_synth']}")
        print(f"  этап 1, медианы {names}: настоящие нарушения {stage1['real_pos']} | норма {stage1['real_neg']} | синтетика {stage1['synthetic']}")
        print(f"  этап 2: ROC-AUC {o['auc_base']} → {o['auc_synth']} (Δ {o['delta_auc']} [{o['ci'][0]}; {o['ci'][1]}]), F1 {o['f1_base']} → {o['f1_synth']} → {'ВОРОТА ПРОЙДЕНЫ' if passed else 'отклонено'}")
    json.dump(out, open(os.path.join(WORK, "exp28.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
